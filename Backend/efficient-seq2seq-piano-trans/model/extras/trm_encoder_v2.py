"""
TRM v2 encoder for AMT -- faithful two-state recursive architecture.

Implements the Tiny Recursive Model (TRM) from arXiv:2510.04871 with:
  - Two states: z_H (answer / current best encoding) and z_L (latent reasoning)
  - Single shared transformer block applied for both z_L and z_H updates
  - Structured recursion: T-1 cycles without grad, 1 cycle with full backprop
  - Deep supervision via carry state reuse across N_sup steps
  - Simplified ACT halt mechanism (toggleable Q-learning)
  - EMA support
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from model.Layers import FixedEmbed, LayerNorm
from model.layers.hrm_layers import (
    Attention as HRMAttention,
    CosSin,
    RotaryEmbedding,
    SwiGLU,
    apply_rotary_pos_emb,
    rms_norm,
)
from model.layers.hrm_common import trunc_normal_init_


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _warn_once(message: str) -> None:
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def _mask_mean(hidden: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean-pool over the sequence dimension, respecting an optional mask."""
    if mask is None:
        return hidden.mean(dim=1)
    mask_float = mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)


def _scaled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    window_size: Optional[int] = None,
) -> torch.Tensor:
    """Fallback SDPA with optional local-window masking."""
    batch_size, seq_len, num_heads, head_dim = q.shape
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    attn_bias = None
    if attention_mask is not None:
        key_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        attn_bias = torch.zeros(batch_size, 1, seq_len, seq_len, device=q.device, dtype=q.dtype)
        attn_bias = attn_bias.masked_fill(~key_mask, float("-inf"))

    if window_size is not None:
        positions = torch.arange(seq_len, device=q.device)
        local_mask = (positions[None, :] - positions[:, None]).abs() <= (window_size // 2)
        local_bias = torch.zeros(1, 1, seq_len, seq_len, device=q.device, dtype=q.dtype)
        local_bias = local_bias.masked_fill(~local_mask.view(1, 1, seq_len, seq_len), float("-inf"))
        attn_bias = local_bias if attn_bias is None else attn_bias + local_bias

    out = F.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32),
        attn_mask=attn_bias, dropout_p=0.0, is_causal=False,
    )
    return out.transpose(1, 2).to(q.dtype).reshape(batch_size, seq_len, num_heads * head_dim)


try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func
except ImportError:
    try:
        from flash_attn import flash_attn_func as _flash_attn_func
    except ImportError:
        _flash_attn_func = None


# ---------------------------------------------------------------------------
# Stable-max cross-entropy (ported from reference)
# ---------------------------------------------------------------------------

def _stablemax_s(x: torch.Tensor, epsilon: float = 1e-30) -> torch.Tensor:
    return torch.where(x < 0, 1.0 / (1.0 - x + epsilon), x + 1.0)


def stablemax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Numerically stable alternative to softmax cross-entropy."""
    s_x = _stablemax_s(logits.to(torch.float64))
    logprobs = torch.log(s_x / s_x.sum(dim=-1, keepdim=True))
    valid_mask = labels != ignore_index
    safe_labels = torch.where(valid_mask, labels, 0).long()
    gathered = torch.gather(logprobs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return -torch.where(valid_mask, gathered, torch.zeros_like(gathered)).to(torch.float32)


# ---------------------------------------------------------------------------
# Transformer layer (same as v1 but re-exported for clarity)
# ---------------------------------------------------------------------------

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
        fa_dtype = torch.float16

        qkv = self.attn.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, self.num_heads * 3, self.head_dim)
        q = qkv[:, :, : self.num_heads]
        k = qkv[:, :, self.num_heads : self.num_heads * 2]
        v = qkv[:, :, self.num_heads * 2 :]

        if cos_sin is not None:
            cos, sin = cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if _flash_attn_func is None:
            raw = _scaled_attention(q, k, v, attention_mask=attention_mask, window_size=self.window_size)
        elif self.window_size is None:
            raw = _flash_attn_func(q.to(fa_dtype), k.to(fa_dtype), v.to(fa_dtype), causal=False)
            if isinstance(raw, tuple):
                raw = raw[0]
            raw = raw.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        else:
            half_window = self.window_size // 2
            raw = _flash_attn_func(
                q.to(fa_dtype), k.to(fa_dtype), v.to(fa_dtype),
                window_size=(half_window, half_window), causal=False,
            )
            if isinstance(raw, tuple):
                raw = raw[0]
            raw = raw.reshape(batch_size, seq_len, self.num_heads * self.head_dim)

        attn_out = self.attn.o_proj(raw).to(x.dtype)
        if self.training and not deterministic:
            attn_out = self.drop1(attn_out)
        x = rms_norm(x + attn_out, variance_epsilon=self.norm_eps)

        mlp_out = self.mlp(x)
        if self.training and not deterministic:
            mlp_out = self.drop2(mlp_out)
        return rms_norm(x + mlp_out, variance_epsilon=self.norm_eps)


# ---------------------------------------------------------------------------
# Recursive block -- single shared block applied to both z_L and z_H
# ---------------------------------------------------------------------------

class TRMRecursiveBlock(nn.Module):
    """Shared transformer block: hidden = block(hidden + injection)."""

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
            hidden = layer(hidden, cos_sin=cos_sin, deterministic=deterministic, attention_mask=attention_mask)
        return hidden


# ---------------------------------------------------------------------------
# TrmEncoderV2 -- faithful two-state TRM
# ---------------------------------------------------------------------------

class TrmEncoderV2(nn.Module):
    """
    Two-state TRM encoder faithful to arXiv:2510.04871.

    States:
        z_H  -- the "answer" / current best encoding (output to decoder)
        z_L  -- the latent reasoning state

    Recursion (per supervision step):
        T-1 full cycles under torch.no_grad(), then 1 full cycle with grad.
        Each full cycle = L_cycles z_L-updates + 1 z_H-update.
        z_L update injection: z_H + x  (includes input)
        z_H update injection: z_L      (no input)

    Deep supervision:
        Up to N_sup steps, reusing detached carry (z_H, z_L) across steps.
        Driven by the encode() loop in T5.py.

    ACT (toggleable):
        Q-head produces a halt logit from pooled z_H.
        Training target: BCE on whether aux head prediction is correct.
        Controlled by `trm_use_q_halt` config flag.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        emb_dim: int = config.emb_dim
        num_heads: int = config.num_heads
        head_dim: int = config.head_dim
        dropout: float = config.dropout_rate
        expansion: float = getattr(config, "hrm_expansion", 4.0)
        rms_norm_eps: float = getattr(config, "hrm_rms_norm_eps", 1e-5)

        # --- Recursion structure ---
        self.trm_layers = self._get(config, "trm_layers", 2)
        self.H_cycles = self._get(config, "trm_H_cycles", 3)        # T
        self.L_cycles = self._get(config, "trm_L_cycles", 6)        # n
        self.n_sup = self._get(config, "trm_n_sup", 4)              # deep supervision steps
        self.use_gradient_checkpointing = bool(self._get(config, "trm_gradient_checkpointing", False))

        # --- Halt / ACT (toggleable) ---
        self.use_q_halt = bool(self._get(config, "trm_use_q_halt", True))
        self.halt_exploration_prob = float(self._get(config, "trm_halt_exploration_prob", 0.1))

        # --- Position encoding ---
        self.use_rope = bool(self._get(config, "trm_use_rope", True))
        self.use_fixed_embed = not self.use_rope or bool(self._get(config, "trm_force_fixed_embed", False))

        if self.use_fixed_embed:
            self.embed = FixedEmbed(features=emb_dim)

        if self.use_rope:
            rope_max_len = int(self._get(config, "trm_rope_max_len", 2048))
            rope_theta = float(self._get(config, "trm_rope_theta", 10000.0))
            self.rotary_emb = RotaryEmbedding(dim=head_dim, max_position_embeddings=rope_max_len, base=rope_theta)

        local_window = int(self._get(config, "trm_local_window_size", 64))

        # --- Input projection ---
        self.dense = nn.Linear(config.encoder_input_dim, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.input_norm = LayerNorm(emb_dim)

        # --- Recording embedding (optional) ---
        self.use_recording_id_embedding = getattr(config, "use_recording_id_embedding", True)
        self.num_recordings = getattr(config, "num_recordings", 0)
        rec_dim = getattr(config, "recording_emb_dim", 64)
        self.has_recording_embedding = self.use_recording_id_embedding and self.num_recordings > 0
        if self.has_recording_embedding:
            self.recording_embed = nn.Embedding(self.num_recordings, rec_dim)
            nn.init.zeros_(self.recording_embed.weight)
            if rec_dim != emb_dim:
                self.recording_proj = nn.Linear(rec_dim, emb_dim, bias=False)
                nn.init.zeros_(self.recording_proj.weight)
            else:
                self.recording_proj = nn.Identity()

        # --- Shared recursive block (single network, 2 layers) ---
        recursive_layers = nn.ModuleList()
        for layer_idx in range(self.trm_layers):
            win = local_window if layer_idx == 0 else None
            recursive_layers.append(
                TRMTransformerLayer(
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    expansion=expansion,
                    dropout_rate=dropout,
                    rms_norm_eps=rms_norm_eps,
                    window_size=win,
                )
            )
        self.recursive_block = TRMRecursiveBlock(recursive_layers)

        # --- Learned initial states (from paper: H_init, L_init) ---
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(emb_dim), std=1.0))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(emb_dim), std=1.0))

        # --- Q-head for halt decision (toggleable) ---
        if self.use_q_halt:
            self.q_head = nn.Linear(emb_dim, 1, bias=True)
            nn.init.zeros_(self.q_head.weight)
            nn.init.constant_(self.q_head.bias, -5.0)

        # --- Auxiliary head for encoder-level deep supervision ---
        aux_vocab_size = int(self._get(config, "trm_aux_vocab_size", getattr(config, "vocab_size", 0)))
        self.has_aux_head = aux_vocab_size > 0
        if self.has_aux_head:
            self.aux_head = nn.Linear(emb_dim, aux_vocab_size, bias=False)

        # --- Output norm ---
        self.layer_norm = LayerNorm(emb_dim)

    # --- Config helpers ---
    @staticmethod
    def _get(config, key: str, default):
        return getattr(config, key, default)

    # --- Positional encoding ---
    def _compute_cos_sin(self, seq_len: int) -> Optional[CosSin]:
        if not self.use_rope or not hasattr(self, "rotary_emb"):
            return None
        cos_full, sin_full = self.rotary_emb()
        return cos_full[:seq_len], sin_full[:seq_len]

    @staticmethod
    def _prepare_attention_mask(encoder_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if encoder_mask is None:
            return None
        if encoder_mask.dim() == 4:
            return encoder_mask[:, 0, 0, :].bool()
        return encoder_mask.bool()

    # --- Core recursion ---
    def _recurse_once(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        x: torch.Tensor,
        cos_sin: Optional[CosSin],
        attn_mask: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One full recursion cycle (paper's inner loop):
            n times: z_L = block(z_L, z_H + x)   -- latent reasoning, injection includes input
            1 time:  z_H = block(z_H, z_L)        -- answer update, injection is z_L only (no input!)
        """
        for _ in range(self.L_cycles):
            if self.use_gradient_checkpointing and not deterministic and torch.is_grad_enabled():
                z_L = grad_checkpoint(
                    self.recursive_block, z_L, z_H + x,
                    cos_sin, deterministic, attn_mask,
                    use_reentrant=False,
                )
            else:
                z_L = self.recursive_block(z_L, z_H + x, cos_sin=cos_sin, deterministic=deterministic, attention_mask=attn_mask)

        # z_H update: injection is z_L only, no input x
        if self.use_gradient_checkpointing and not deterministic and torch.is_grad_enabled():
            z_H = grad_checkpoint(
                self.recursive_block, z_H, z_L,
                cos_sin, deterministic, attn_mask,
                use_reentrant=False,
            )
        else:
            z_H = self.recursive_block(z_H, z_L, cos_sin=cos_sin, deterministic=deterministic, attention_mask=attn_mask)

        return z_H, z_L

    def _run_recursion(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        x: torch.Tensor,
        cos_sin: Optional[CosSin],
        attn_mask: Optional[torch.Tensor],
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full recursion: H_cycles-1 cycles without grad, then 1 cycle with grad.
        This is the core TRM innovation from the paper (Section 4.1).
        """
        # T-1 cycles without gradient (cheap, no activation storage)
        if self.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.H_cycles - 1):
                    z_H, z_L = self._recurse_once(z_H, z_L, x, cos_sin, attn_mask, deterministic=True)

        # 1 cycle with full gradient (backprop through all n+1 block applications)
        z_H, z_L = self._recurse_once(z_H, z_L, x, cos_sin, attn_mask, deterministic=deterministic)

        return z_H, z_L

    # --- Halt logit ---
    def _compute_halt_logit(
        self,
        z_H: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pooled = _mask_mean(z_H, attn_mask)
        return self.q_head(pooled).squeeze(-1).to(torch.float32)

    # --- Forward ---
    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        encoder_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        recording_ids: Optional[torch.Tensor] = None,
        carry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            encoder_input_tokens: [B, T, input_dim] mel spectrogram features
            encoder_mask: optional attention mask
            deterministic: disable dropout (inference mode)
            recording_ids: optional per-recording IDs
            carry: optional (z_H, z_L) from previous supervision step

        Returns:
            encoded: [B, T, emb_dim] -- z_H output for decoder cross-attention
            trm_aux: dict with carry, halt info, aux predictions
        """
        batch_size, seq_len, _ = encoder_input_tokens.shape
        device = encoder_input_tokens.device
        deterministic_mode = bool(deterministic) or (not self.training)

        attn_mask = self._prepare_attention_mask(encoder_mask)
        cos_sin = self._compute_cos_sin(seq_len)

        # --- Input projection ---
        x = self.dense(encoder_input_tokens)
        if self.use_fixed_embed:
            positions = torch.arange(seq_len, device=device).unsqueeze(0)
            x = x + self.embed(positions)
        x = self.input_norm(
            F.dropout(x, p=self.dropout_layer.p, training=self.training and not deterministic_mode)
        )
        if self.has_recording_embedding and recording_ids is not None:
            rec = self.recording_proj(self.recording_embed(recording_ids))
            x = x + rec.unsqueeze(1)

        # --- Initialize or reuse carry states ---
        if carry is not None:
            z_H, z_L = carry
        else:
            # Learned initial states broadcast to [B, T, D]
            z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # --- Run structured recursion ---
        z_H, z_L = self._run_recursion(z_H, z_L, x, cos_sin, attn_mask, deterministic=deterministic_mode)

        # --- Halt logit (Q-learning, toggleable) ---
        halt_logit = None
        if self.use_q_halt:
            halt_logit = self._compute_halt_logit(z_H, attn_mask)

        # --- Auxiliary head prediction ---
        aux_logits = None
        if self.has_aux_head:
            aux_logits = self.aux_head(z_H)  # [B, T, vocab_size]

        # --- Output ---
        encoded = self.layer_norm(
            F.dropout(z_H, p=self.dropout_layer.p, training=self.training and not deterministic_mode)
        )

        if hasattr(self.config, "dtype") and self.config.dtype is not None:
            encoded = encoded.to(self.config.dtype)

        trm_aux = {
            # Carry for next supervision step (always detached)
            "carry_z_H": z_H.detach(),
            "carry_z_L": z_L.detach(),
            # Halt
            "halt_logit": halt_logit,
            # Aux predictions
            "aux_logits": aux_logits,
            # Metadata
            "H_cycles": self.H_cycles,
            "L_cycles": self.L_cycles,
            "n_sup": self.n_sup,
            "use_q_halt": self.use_q_halt,
        }

        return encoded, trm_aux


# ---------------------------------------------------------------------------
# Loss helpers for TrmEncoderV2
# ---------------------------------------------------------------------------

def compute_trm_v2_halt_loss(
    halt_logits: List[Optional[torch.Tensor]],
    aux_correct: List[torch.Tensor],
) -> torch.Tensor:
    """
    Simplified ACT halt loss (paper Section 4.6).
    BCE on whether the auxiliary head prediction is correct at each step.

    Args:
        halt_logits: list of [B] tensors (one per supervision step), may contain None
        aux_correct: list of [B] bool tensors indicating correctness
    Returns:
        scalar loss
    """
    loss = torch.tensor(0.0, device=halt_logits[0].device if halt_logits and halt_logits[0] is not None else "cpu")
    count = 0
    for h_logit, correct in zip(halt_logits, aux_correct):
        if h_logit is None:
            continue
        target = correct.to(h_logit.dtype)
        loss = loss + F.binary_cross_entropy_with_logits(h_logit, target, reduction="mean")
        count += 1
    return loss / max(count, 1)


def compute_trm_v2_aux_loss(
    aux_logits_list: List[Optional[torch.Tensor]],
    targets: torch.Tensor,
    targets_mask: torch.Tensor,
    decoder_targets_frame_index: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute auxiliary head loss for deep supervision.

    The aux head predicts at encoder frame level. We use decoder targets aligned
    to encoder frames via decoder_targets_frame_index for supervision.

    If frame index alignment is not available, falls back to a simple
    frame-level cross-entropy using the first token per frame.

    Args:
        aux_logits_list: list of [B, T_enc, vocab] tensors per supervision step
        targets: [B, T_dec] decoder target token IDs
        targets_mask: [B, T_dec] mask for valid targets
        decoder_targets_frame_index: [B, T_dec] mapping decoder positions to encoder frames

    Returns:
        total_aux_loss: scalar
        per_step_correct: list of [B] bool tensors for halt correctness signal
    """
    total_loss = torch.tensor(0.0, device=targets.device)
    per_step_correct = []
    count = 0

    for aux_logits in aux_logits_list:
        if aux_logits is None:
            per_step_correct.append(torch.zeros(targets.size(0), dtype=torch.bool, device=targets.device))
            continue

        batch_size, T_enc, vocab_size = aux_logits.shape

        if decoder_targets_frame_index is not None:
            # Scatter decoder targets to encoder frames via frame index
            # For each encoder frame, predict the token(s) that map to it
            # Use the last token mapping to each frame as the target
            frame_idx = decoder_targets_frame_index.clamp(0, T_enc - 1).long()  # [B, T_dec]

            # Build per-frame target: for each frame, use the last decoder token that maps to it
            frame_targets = torch.full((batch_size, T_enc), -100, dtype=torch.long, device=targets.device)
            frame_mask = torch.zeros(batch_size, T_enc, dtype=torch.bool, device=targets.device)

            for b in range(batch_size):
                valid = targets_mask[b].bool()
                valid_frames = frame_idx[b][valid]
                valid_tokens = targets[b][valid]
                if valid_frames.numel() > 0:
                    frame_targets[b].scatter_(0, valid_frames, valid_tokens)
                    frame_mask[b].scatter_(0, valid_frames, torch.ones_like(valid_frames, dtype=torch.bool))

            # Cross-entropy loss on frames with targets
            if frame_mask.any():
                loss_per_token = F.cross_entropy(
                    aux_logits.reshape(-1, vocab_size),
                    frame_targets.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(batch_size, T_enc)

                valid_counts = frame_mask.sum(dim=1).clamp_min(1).float()
                per_example_loss = (loss_per_token * frame_mask.float()).sum(dim=1) / valid_counts
                step_loss = per_example_loss.mean()

                # Correctness: argmax matches target on valid frames
                preds = aux_logits.argmax(dim=-1)  # [B, T_enc]
                correct_per_frame = (preds == frame_targets) & frame_mask
                correct = correct_per_frame.sum(dim=1).float() / valid_counts
                per_step_correct.append(correct > 0.5)  # majority correct
            else:
                step_loss = torch.tensor(0.0, device=targets.device)
                per_step_correct.append(torch.zeros(batch_size, dtype=torch.bool, device=targets.device))
        else:
            # Fallback: no frame alignment available, skip aux loss
            step_loss = torch.tensor(0.0, device=targets.device)
            per_step_correct.append(torch.zeros(batch_size, dtype=torch.bool, device=targets.device))

        total_loss = total_loss + step_loss
        count += 1

    return total_loss / max(count, 1), per_step_correct
