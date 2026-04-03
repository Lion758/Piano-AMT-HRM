"""
TRM-style recursive encoder for AMT.

This replaces the older HRM two-state hierarchy with a single shared recursive
state that is refined multiple times by the same transformer block.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore[import]
except ImportError:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore[import]
    except ImportError:
        _flash_attn_func = None

from model.Layers import FixedEmbed, LayerNorm
from model.layers.hrm_layers import (
    Attention as HRMAttention,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
    rms_norm,
)


def _warn_once(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _mask_mean(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return hidden.mean(dim=1)

    mask_float = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)


def _predict_steps_from_logits(
    halt_logits: List[torch.Tensor],
    halt_threshold: float,
    min_recursions: int,
) -> torch.Tensor:
    if not halt_logits:
        raise ValueError("halt_logits must be non-empty")

    device = halt_logits[0].device
    batch_size = halt_logits[0].size(0)
    num_steps = len(halt_logits)

    probs = torch.sigmoid(torch.stack([logit.detach() for logit in halt_logits], dim=1))
    step_ids = torch.arange(1, num_steps + 1, device=device, dtype=torch.long).unsqueeze(0)
    eligible = step_ids >= min_recursions
    halt_mask = (probs >= halt_threshold) & eligible

    any_halt = halt_mask.any(dim=1)
    first_halt = halt_mask.float().argmax(dim=1).to(torch.long) + 1
    fallback = torch.full((batch_size,), num_steps, device=device, dtype=torch.long)
    return torch.where(any_halt, first_halt, fallback)


def _scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    batch_size, seq_len, num_heads, head_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_bias = None
    if attention_mask is not None:
        key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        attn_bias = torch.zeros(
            batch_size,
            1,
            seq_len,
            seq_len,
            device=q.device,
            dtype=q.dtype,
        )
        attn_bias = attn_bias.masked_fill(~key_mask, float("-inf"))

    if window_size is not None:
        positions = torch.arange(seq_len, device=q.device)
        local_mask = (positions[None, :] - positions[:, None]).abs() <= (window_size // 2)
        local_bias = torch.zeros(
            1,
            1,
            seq_len,
            seq_len,
            device=q.device,
            dtype=q.dtype,
        )
        local_bias = local_bias.masked_fill(~local_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn_bias = local_bias if attn_bias is None else attn_bias + local_bias

    out = F.scaled_dot_product_attention(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        attn_mask=attn_bias,
        dropout_p=0.0,
        is_causal=False,
    )
    return out.transpose(1, 2).to(q.dtype).reshape(batch_size, seq_len, num_heads * head_dim)


class TRMTransformerLayer(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        expansion: float,
        dropout_rate: float,
        rms_norm_eps: float,
        window_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size

        self.attn = HRMAttention(
            hidden_size=emb_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False,
        )
        self.drop1 = nn.Dropout(dropout_rate)
        self.mlp = SwiGLU(hidden_size=emb_dim, expansion=expansion)
        self.drop2 = nn.Dropout(dropout_rate)
        self.norm_eps = rms_norm_eps

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
        deterministic: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        orig_dtype = x.dtype
        fa_dtype = torch.float16

        qkv = self.attn.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, self.num_heads * 3, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads * 2]
        v = qkv[:, :, self.num_heads * 2:]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if _flash_attn_func is None:
            raw = _scaled_attention(
                q,
                k,
                v,
                attention_mask=attention_mask,
                window_size=self.window_size,
            )
        elif self.window_size is None:
            raw = _flash_attn_func(
                q.to(fa_dtype),
                k.to(fa_dtype),
                v.to(fa_dtype),
                causal=False,
            )
            if isinstance(raw, tuple):
                raw = raw[0]
            raw = raw.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        else:
            half_window = self.window_size // 2
            raw = _flash_attn_func(
                q.to(fa_dtype),
                k.to(fa_dtype),
                v.to(fa_dtype),
                window_size=(half_window, half_window),
                causal=False,
            )
            if isinstance(raw, tuple):
                raw = raw[0]
            raw = raw.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        attn_out = self.attn.o_proj(raw).to(orig_dtype)

        if self.training and not deterministic:
            attn_out = self.drop1(attn_out)
        x = rms_norm(x + attn_out, variance_epsilon=self.norm_eps)

        mlp_out = self.mlp(x)
        if self.training and not deterministic:
            mlp_out = self.drop2(mlp_out)
        return rms_norm(x + mlp_out, variance_epsilon=self.norm_eps)


class TRMRecursiveBlock(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers

    def forward(
        self,
        hidden: torch.Tensor,
        injection: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
        deterministic: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(
                hidden,
                cos_sin=cos_sin,
                deterministic=deterministic,
                attention_mask=attention_mask,
            )
        return hidden


class TrmEncoder(nn.Module):
    _IGNORED_HRM_KEYS = (
        "hrm_H_cycles",
        "hrm_L_cycles",
        "hrm_content_init",
        "hrm_injection_gate",
        "hrm_pooled_qhead",
        "hrm_output_adapter",
        "hrm_q_soft_k",
        "hrm_halt_explore_prob",
    )

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        emb_dim = config.emb_dim
        num_heads = config.num_heads
        head_dim = config.head_dim
        dropout = config.dropout_rate
        expansion = getattr(config, "hrm_expansion", 4.0)
        rms_norm_eps = getattr(config, "hrm_rms_norm_eps", 1e-5)

        self.trm_layers = self._resolve_layers(config)
        self.recursions = self._resolve_recursions(config)
        self.local_window = self._resolve_local_window(config)
        self.min_recursions = int(getattr(config, "trm_min_recursions", 2))
        self.halt_threshold = float(getattr(config, "trm_halt_threshold", 0.5))
        self.halt_target_sharpness = float(
            getattr(config, "trm_halt_target_sharpness", 4.0)
        )

        if self.min_recursions > self.recursions:
            _warn_once(
                "trm_min_recursions exceeds trm_recursions; clamping to the recursion count."
            )
            self.min_recursions = self.recursions

        self.use_rope = self._resolve_model_flag(
            config,
            canonical_key="trm_use_rope",
            legacy_key="hrm_use_rope",
            default=True,
        )
        self.use_fixed_embed = not self.use_rope or self._resolve_model_flag(
            config,
            canonical_key="trm_force_fixed_embed",
            legacy_key="hrm_force_fixed_embed",
            default=False,
        )

        if self.use_fixed_embed:
            self.embed = FixedEmbed(features=emb_dim)

        if self.use_rope:
            rope_max_len = self._resolve_model_value(
                config,
                canonical_key="trm_rope_max_len",
                legacy_key="hrm_rope_max_len",
                default=2048,
            )
            rope_theta = self._resolve_model_value(
                config,
                canonical_key="trm_rope_theta",
                legacy_key="hrm_rope_theta",
                default=10000.0,
            )
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=int(rope_max_len),
                base=float(rope_theta),
            )

        self.dense = nn.Linear(config.encoder_input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = LayerNorm(emb_dim)

        recursive_layers = nn.ModuleList()
        for layer_idx in range(self.trm_layers):
            window_size = self.local_window if layer_idx == 0 else None
            recursive_layers.append(
                TRMTransformerLayer(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    expansion=expansion,
                    dropout_rate=dropout,
                    rms_norm_eps=rms_norm_eps,
                    window_size=window_size,
                )
            )
        self.recursive_block = TRMRecursiveBlock(recursive_layers)

        self.halt_head = nn.Linear(emb_dim, 1, bias=True)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, -2.0)

        self.use_recording_id_embedding = getattr(
            config, "use_recording_id_embedding", True
        )
        self.num_recordings = getattr(config, "num_recordings", 0)
        rec_dim = getattr(config, "recording_emb_dim", 64)
        self.has_recording_embedding = (
            self.use_recording_id_embedding and self.num_recordings > 0
        )
        if self.has_recording_embedding:
            self.recording_embed = nn.Embedding(self.num_recordings, rec_dim)
            nn.init.zeros_(self.recording_embed.weight)
            if rec_dim != emb_dim:
                self.recording_proj = nn.Linear(rec_dim, emb_dim, bias=False)
                nn.init.zeros_(self.recording_proj.weight)
            else:
                self.recording_proj = nn.Identity()

        self.layer_norm = LayerNorm(emb_dim)
        self._warn_on_ignored_legacy_keys()

    @staticmethod
    def _resolve_model_flag(config, canonical_key: str, legacy_key: str, default: bool) -> bool:
        if hasattr(config, canonical_key):
            return bool(getattr(config, canonical_key))
        if hasattr(config, legacy_key):
            _warn_once(f"{legacy_key} is deprecated; use {canonical_key} instead.")
            return bool(getattr(config, legacy_key))
        return default

    @staticmethod
    def _resolve_model_value(config, canonical_key: str, legacy_key: str, default):
        if hasattr(config, canonical_key):
            return getattr(config, canonical_key)
        if hasattr(config, legacy_key):
            _warn_once(f"{legacy_key} is deprecated; use {canonical_key} instead.")
            return getattr(config, legacy_key)
        return default

    @staticmethod
    def _resolve_layers(config) -> int:
        if hasattr(config, "trm_layers"):
            return max(1, int(getattr(config, "trm_layers")))

        legacy_values = []
        for key in ("hrm_H_layers", "hrm_L_layers"):
            if hasattr(config, key):
                legacy_values.append((key, int(getattr(config, key))))
        if legacy_values:
            mapped_value = max(value for _, value in legacy_values)
            _warn_once(
                "Legacy HRM layer knobs are deprecated; mapping "
                + ", ".join(f"{key}={value}" for key, value in legacy_values)
                + f" to trm_layers={mapped_value}."
            )
            return max(1, mapped_value)

        return 2

    @staticmethod
    def _resolve_recursions(config) -> int:
        if hasattr(config, "trm_recursions"):
            return max(1, int(getattr(config, "trm_recursions")))
        if hasattr(config, "hrm_max_steps"):
            mapped_value = int(getattr(config, "hrm_max_steps"))
            _warn_once(
                f"hrm_max_steps is deprecated; mapping hrm_max_steps={mapped_value} to trm_recursions."
            )
            return max(1, mapped_value)
        return 6

    @staticmethod
    def _resolve_local_window(config) -> int:
        if hasattr(config, "trm_local_window_size"):
            return max(1, int(getattr(config, "trm_local_window_size")))
        if hasattr(config, "hrm_L_window_size"):
            mapped_value = int(getattr(config, "hrm_L_window_size"))
            _warn_once(
                f"hrm_L_window_size is deprecated; mapping it to trm_local_window_size={mapped_value}."
            )
            return max(1, mapped_value)
        return 64

    def _warn_on_ignored_legacy_keys(self) -> None:
        ignored = [key for key in self._IGNORED_HRM_KEYS if hasattr(self.config, key)]
        if ignored:
            _warn_once(
                "Ignoring deprecated HRM-only knobs in TrmEncoder: " + ", ".join(ignored)
            )

    def _compute_cos_sin(self, seq_len: int) -> Optional[CosSin]:
        if not self.use_rope or not hasattr(self, "rotary_emb"):
            return None
        cos_full, sin_full = self.rotary_emb()
        return cos_full[:seq_len], sin_full[:seq_len]

    @staticmethod
    def _prepare_attention_mask(
        encoder_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if encoder_mask is None:
            return None
        if encoder_mask.dim() == 4:
            return encoder_mask[:, 0, 0, :].bool()
        if encoder_mask.dim() == 2:
            return encoder_mask.bool()
        return encoder_mask.bool()

    def _compute_halt_logit(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pooled = _mask_mean(hidden, attention_mask)
        return self.halt_head(pooled).squeeze(-1).to(torch.float32)

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        encoder_mask=None,
        deterministic: bool = False,
        recording_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        batch_size, seq_len, _ = encoder_input_tokens.shape
        device = encoder_input_tokens.device
        deterministic_mode = bool(deterministic) or (not self.training)

        attn_mask = self._prepare_attention_mask(encoder_mask)
        cos_sin = self._compute_cos_sin(seq_len)

        x = self.dense(encoder_input_tokens)
        if self.use_fixed_embed:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.embed(positions)

        x = self.input_norm(
            F.dropout(
                x,
                p=self.dropout.p,
                training=self.training and (not deterministic_mode),
            )
        )

        if self.has_recording_embedding and recording_ids is not None:
            rec = self.recording_proj(self.recording_embed(recording_ids))
            x = x + rec.unsqueeze(1)

        z = x
        trm_aux = None

        if deterministic_mode:
            halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for step in range(self.recursions):
                z_candidate = self.recursive_block(
                    z,
                    x,
                    cos_sin=cos_sin,
                    deterministic=True,
                    attention_mask=attn_mask,
                )
                halt_logit = self._compute_halt_logit(z_candidate, attn_mask)
                halted_before = halted
                if (step + 1) >= self.min_recursions:
                    halted = halted | (torch.sigmoid(halt_logit) >= self.halt_threshold)
                z = torch.where(halted_before.view(-1, 1, 1), z, z_candidate)
                if halted.all():
                    break
        else:
            halt_logits: List[torch.Tensor] = []
            for step in range(self.recursions):
                z = self.recursive_block(
                    z,
                    x,
                    cos_sin=cos_sin,
                    deterministic=False,
                    attention_mask=attn_mask,
                )
                halt_logits.append(self._compute_halt_logit(z, attn_mask))
                if step < self.recursions - 1:
                    z = z.detach()

            trm_aux = {
                "halt_logits": halt_logits,
                "halt_logits_last": halt_logits[-1].detach() if halt_logits else None,
                "steps_per_example": _predict_steps_from_logits(
                    halt_logits,
                    halt_threshold=self.halt_threshold,
                    min_recursions=self.min_recursions,
                ),
                "recursions": self.recursions,
                "halt_target_sharpness": self.halt_target_sharpness,
                "halt_threshold": self.halt_threshold,
                "min_recursions": self.min_recursions,
            }

        encoded = self.layer_norm(
            F.dropout(
                z,
                p=self.dropout.p,
                training=self.training and (not deterministic_mode),
            )
        )

        if hasattr(self.config, "dtype") and self.config.dtype is not None:
            encoded = encoded.to(self.config.dtype)

        return encoded, trm_aux


def compute_trm_halt_loss(
    trm_aux: dict,
    decoder_loss_per_example: torch.Tensor,
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    halt_logits = trm_aux["halt_logits"]
    if not halt_logits:
        loss = torch.tensor(0.0, device=decoder_loss_per_example.device)
        if return_components:
            return loss, {
                "halt_term": loss.detach(),
                "target_step_mean": loss.detach(),
            }
        return loss

    num_steps = len(halt_logits)
    device = halt_logits[0].device
    losses = decoder_loss_per_example.detach().to(device=device, dtype=torch.float32)

    if losses.numel() == 1:
        percentiles = torch.full_like(losses, 0.5)
    else:
        sorted_indices = torch.argsort(losses)
        ranks = torch.empty_like(sorted_indices, dtype=torch.float32)
        ranks[sorted_indices] = torch.arange(
            losses.numel(),
            device=device,
            dtype=torch.float32,
        )
        percentiles = ranks / float(max(losses.numel() - 1, 1))

    target_step = 1.0 + percentiles * float(max(num_steps - 1, 0))
    sharpness = float(trm_aux.get("halt_target_sharpness", 4.0))

    halt_term = torch.tensor(0.0, device=device)
    for step_idx, halt_logit in enumerate(halt_logits, start=1):
        step_value = torch.full_like(target_step, float(step_idx))
        target = torch.sigmoid(sharpness * (step_value - target_step))
        halt_term = halt_term + F.binary_cross_entropy_with_logits(
            halt_logit.to(torch.float32),
            target,
            reduction="mean",
        )

    halt_term = halt_term / max(num_steps, 1)

    if not return_components:
        return halt_term

    return halt_term, {
        "halt_term": halt_term.detach(),
        "target_step_mean": target_step.mean().detach(),
    }
