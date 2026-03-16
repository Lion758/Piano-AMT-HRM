"""
HRM Encoder for AMT (Automatic Music Transcription)


"""

import math
from typing import Optional, Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# FA3 / FA2 compatibility — same pattern as hrm_layers.py
try:
    from flash_attn_interface import flash_attn_func as _flash_attn_func  # type: ignore[import]
except ImportError:
    from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore[import]

from model.Layers import LayerNorm, FixedEmbed
from model.layers.hrm_layers import (
    rms_norm,
    SwiGLU,
    Attention as HRMAttention,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    CosSin,
)


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-Aware Input Stem  (Change 1)
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyAwareStem(nn.Module):
    """
    Processes mel-spectrogram along the frequency axis before temporal attention.

    For each time frame, applies a small 1D conv stack across the mel bins to
    detect local spectral peaks and harmonic patterns, then projects to emb_dim.

    A gated residual from the original dense path ensures backward compatibility:
        output = gate * conv_path + (1 - gate) * dense_path
    Gate is initialized to 0.0 so the module starts as the plain dense projection.

    Input:  [B, T, input_dim]  (mel bins)
    Output: [B, T, emb_dim]
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        stem_channels: int = 64,
    ):
        super().__init__()
        # Learned frequency positional embedding — tells the model which mel bin is which
        self.freq_pos_emb = nn.Parameter(torch.zeros(1, 1, input_dim))

        # Depthwise-separable 1D conv stack along frequency axis
        # Kernel sizes chosen to capture local peaks (5), ~semitone neighborhoods (13), refinement (3)
        self.conv_stack = nn.Sequential(
            # Layer 1: expand to stem_channels
            nn.Conv1d(1, stem_channels, kernel_size=5, padding=2),
            nn.GELU(),
            # Layer 2: capture wider harmonic relationships (depthwise + pointwise)
            nn.Conv1d(stem_channels, stem_channels, kernel_size=13, padding=6, groups=stem_channels),
            nn.Conv1d(stem_channels, stem_channels, kernel_size=1),
            nn.GELU(),
            # Layer 3: refine (depthwise + pointwise)
            nn.Conv1d(stem_channels, stem_channels, kernel_size=3, padding=1, groups=stem_channels),
            nn.Conv1d(stem_channels, stem_channels, kernel_size=1),
            nn.GELU(),
        )
        # Reduce channels back to 1 to get [B*T, 1, input_dim], then project
        self.channel_reduce = nn.Conv1d(stem_channels, 1, kernel_size=1)
        self.conv_proj = nn.Linear(input_dim, emb_dim)

        # Dense path (original projection)
        self.dense = nn.Linear(input_dim, emb_dim)

        # Gated residual: sigmoid(-5) ≈ 0.007 → conv path nearly off at init
        self.gate_alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim]"""
        B, T, D = x.shape

        # Dense path (original behavior)
        dense_out = self.dense(x)  # [B, T, emb_dim]

        # Conv path: process frequency axis per frame
        x_freq = x + self.freq_pos_emb  # [B, T, D]
        # Reshape to [B*T, 1, D] for 1D conv along frequency
        x_freq = x_freq.reshape(B * T, 1, D)
        conv_out = self.conv_stack(x_freq)    # [B*T, stem_channels, D]
        conv_out = self.channel_reduce(conv_out)  # [B*T, 1, D]
        conv_out = conv_out.squeeze(1)        # [B*T, D]
        conv_out = conv_out.reshape(B, T, D)  # [B, T, D]
        conv_out = self.conv_proj(conv_out)   # [B, T, emb_dim]

        # Gated combination
        gate = torch.sigmoid(self.gate_alpha)
        return (1.0 - gate) * dense_out + gate * conv_out


def trunc_normal_init_(
    tensor: torch.Tensor,
    std: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> torch.Tensor:
    """
    Truncated normal initialisation matching JAX's default (used in HRM).
    PyTorch's built-in trunc_normal_ has a known bug: the actual std of the
    initialised tensor is not equal to the std argument. This fixes it.
    """
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z
                - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor



class HRMTransformerLayer(nn.Module):
    """
    Single HRM-style transformer block: post-norm RMSNorm + SwiGLU.

    Uses hrm_layers::Attention (correct separate Q/K/V projections via a combined
    qkv_proj that is split, NOT a shared projection) with optional RoPE.

    Two attention paths dispatched by self.window_size:
      - window_size=None  → H-level global attention via HRMAttention.forward directly
      - window_size=int   → L-level local window attention via manual QKV split +
                            flash_attn_func(window_size=...), applying RoPE manually
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        expansion: float,
        dropout_rate: float,
        rms_norm_eps: float,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size

        # Correct attention: qkv_proj outputs 3×dim, split into independent Q, K, V
        self.attn = HRMAttention(
            hidden_size=emb_dim,
            head_dim=head_dim,
            num_heads=num_heads,
            num_key_value_heads=num_heads,   # standard MHA, no GQA reduction
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
        B, T, _ = x.shape

        # flash_attn requires fp16 or bf16; cast regardless of training dtype.
        # We use float16 as the safe default; bf16 would also work.
        orig_dtype = x.dtype
        fa_dtype = torch.float16

        if self.window_size is None:
            # ── H-level: global attention ─────────────────────────────────────
            # hrm_layers::Attention calls flash_attn_func internally without
            # casting, so we cast x here and restore after.
            x_fa = x.to(fa_dtype)
            attn_out = self.attn(cos_sin, x_fa, attention_mask).to(orig_dtype)
        else:
            # ── L-level: local sliding-window attention ────────────────────────
            # HRMAttention.forward does not accept window_size, so we use its
            # qkv_proj/o_proj directly and call flash_attn_func with window_size.
            qkv = self.attn.qkv_proj(x)
            # qkv: [B, T, (num_heads + 2*num_kv_heads) * head_dim]
            # With num_key_value_heads == num_heads: [B, T, 3*H*D]
            qkv = qkv.view(B, T, self.num_heads * 3, self.head_dim)
            q = qkv[:, :, :self.num_heads]                              # [B, T, H, D]
            k = qkv[:, :, self.num_heads: self.num_heads * 2]           # [B, T, H, D]
            v = qkv[:, :, self.num_heads * 2:]                          # [B, T, H, D]

            if cos_sin is not None:
                cos, sin = cos_sin
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            w = self.window_size // 2
            q = q.to(fa_dtype)
            k = k.to(fa_dtype)
            v = v.to(fa_dtype)
            raw = _flash_attn_func(q, k, v, window_size=(w, w), causal=False)
            if isinstance(raw, tuple):
                raw = raw[0]   # FA3 returns (out, softmax_lse, ...)
            attn_out = self.attn.o_proj(raw.reshape(B, T, self.num_heads * self.head_dim)).to(orig_dtype)

        if self.training and not deterministic:
            attn_out = self.drop1(attn_out)
        x = rms_norm(x + attn_out, variance_epsilon=self.norm_eps)
        mlp_out = self.mlp(x)
        if self.training and not deterministic:
            mlp_out = self.drop2(mlp_out)
        x = rms_norm(x + mlp_out, variance_epsilon=self.norm_eps)
        return x


class HRMReasoningStack(nn.Module):
    """
    Stack of transformer layers with additive input injection on entry.

        hidden = hidden + injection   (grounds the stack in the input signal)
        for layer in layers:
            hidden = layer(hidden, cos_sin=cos_sin, ...)

    This is exactly HRM's ReasoningModule pattern (hrm_act_v1.py line ~80).
    The injection for L-level is (z_H + audio_features), keeping L perpetually
    grounded in the raw audio regardless of how many refinement steps have run.
    """

    def __init__(self, layers: nn.ModuleList):
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


# ─────────────────────────────────────────────────────────────────────────────
# HRM Encoder
# ─────────────────────────────────────────────────────────────────────────────

class HRMEncoder(nn.Module):
    """
    Drop-in replacement for model/Encoder.py::Encoder.

    Public signature (identical to Encoder.forward):
        forward(encoder_input_tokens, encoder_mask=None, deterministic=False,
                recording_ids=None)
        → (encoded [B, T, emb_dim], hrm_aux dict | None)

    hrm_aux is None during inference (deterministic=True) and contains
    Q-logits + bootstrap targets during training for the Q-learning loss.

    Config fields (add to T5.yaml or experiment yaml):
        use_hrm_encoder:          bool  (default False)
        hrm_H_layers:             int   (default 3)
        hrm_L_layers:             int   (default 3)
        hrm_H_cycles:             int   (default 2)
        hrm_L_cycles:             int   (default 2)
        hrm_max_steps:            int   (default 4)
        hrm_L_window_size:        int   (default 64)   — local attn window, L-level only
        hrm_halt_explore_prob:    float (default 0.1)
        use_recording_id_embedding: bool (default True) — hard switch to enable/disable
        num_recordings:           int   (default 0)    — 0 disables recording embeddings
        recording_emb_dim:        int   (default 64)

        hrm_use_rope:             bool  (default True) — use RoPE instead of FixedEmbed
        hrm_rope_max_len:         int   (default 2048) — upper bound for cos/sin cache
        hrm_rope_theta:           float (default 10000.0) — RoPE base frequency
        hrm_force_fixed_embed:    bool  (default False) — force sinusoidal even with RoPE
        hrm_q_soft_k:             float (default 5.0)  — sharpness of soft is_easy target

        # ── Pitch accuracy improvements (all default False for backward compat) ──
        hrm_use_freq_stem:        bool  (default False) — frequency-aware input stem
        hrm_freq_stem_channels:   int   (default 64)    — conv channels in freq stem
        hrm_content_init:         bool  (default False) — content-conditioned carry init
        hrm_injection_gate:       bool  (default False) — gated audio injection
        hrm_pooled_qhead:         bool  (default False) — sequence-wide Q-head pooling
        hrm_output_adapter:       bool  (default False) — output bottleneck adapter
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        emb_dim   = config.emb_dim        # 512
        num_heads = config.num_heads       # 8
        head_dim  = config.head_dim        # 64
        dropout   = config.dropout_rate    # 0.1
        expansion = getattr(config, "hrm_expansion", 4.0)
        rms_norm_eps = getattr(config, "hrm_rms_norm_eps", 1e-5)

        H_layers = getattr(config, "hrm_H_layers", 3)
        L_layers = getattr(config, "hrm_L_layers", 3)
        L_window = getattr(config, "hrm_L_window_size", 64)

        self.H_cycles  = getattr(config, "hrm_H_cycles", 2)
        self.L_cycles  = getattr(config, "hrm_L_cycles", 2)
        self.max_steps = getattr(config, "hrm_max_steps", 4)
        self.explore_p = getattr(config, "hrm_halt_explore_prob", 0.1)

        # ── Positional encoding choice ────────────────────────────────────────
        # hrm_use_rope=True (default): use RoPE inside attention; skip FixedEmbed
        # hrm_use_rope=False or hrm_force_fixed_embed=True: use sinusoidal FixedEmbed
        self.use_rope = getattr(config, "hrm_use_rope", True)
        self.use_fixed_embed = (
            (not self.use_rope)
            or getattr(config, "hrm_force_fixed_embed", False)
        )

        if self.use_fixed_embed:
            self.embed = FixedEmbed(features=emb_dim)

        if self.use_rope:
            rope_max_len = getattr(config, "hrm_rope_max_len", 2048)
            rope_theta   = getattr(config, "hrm_rope_theta", 10000.0)
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=rope_max_len,
                base=rope_theta,
            )

        # ── Input projection (matches Encoder.dense + Encoder.embed) ─────────
        # Change 1: Frequency-Aware Input Stem
        self.use_freq_stem = getattr(config, "hrm_use_freq_stem", False)
        if self.use_freq_stem:
            stem_channels = getattr(config, "hrm_freq_stem_channels", 64)
            self.freq_stem = FrequencyAwareStem(
                input_dim=config.encoder_input_dim,
                emb_dim=emb_dim,
                stem_channels=stem_channels,
            )
        else:
            self.dense = nn.Linear(config.encoder_input_dim, emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.input_norm = LayerNorm(emb_dim)

        # ── L-level: audio-grounded, local window attention ──────────────────
        self.L_level = HRMReasoningStack(
            nn.ModuleList([
                HRMTransformerLayer(
                    emb_dim, num_heads, head_dim, expansion, dropout, rms_norm_eps,
                    window_size=L_window,
                )
                for _ in range(L_layers)
            ])
        )

        # ── H-level: abstract planner, global attention ──────────────────────
        self.H_level = HRMReasoningStack(
            nn.ModuleList([
                HRMTransformerLayer(
                    emb_dim, num_heads, head_dim, expansion, dropout, rms_norm_eps,
                    window_size=None,
                )
                for _ in range(H_layers)
            ])
        )

        # ── Initial carry states ─────────────────────────────────────────────
        # Change 2: Content-conditioned carry initialization
        self.use_content_init = getattr(config, "hrm_content_init", False)
        if self.use_content_init:
            self.H_init_proj = nn.Linear(emb_dim, emb_dim, bias=True)
            self.L_init_proj = nn.Linear(emb_dim, emb_dim, bias=True)
            # Identity init so initial behavior matches the old static path
            nn.init.eye_(self.H_init_proj.weight)
            nn.init.zeros_(self.H_init_proj.bias)
            nn.init.eye_(self.L_init_proj.weight)
            nn.init.zeros_(self.L_init_proj.bias)
        # Static fallback (always kept for backward compat / when content_init=False)
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(emb_dim), std=1.0),
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(emb_dim), std=1.0),
        )

        # ── Q-head ───────────────────────────────────────────────────────────
        # Change 4: Sequence-wide Q-head (pooled)
        self.use_pooled_qhead = getattr(config, "hrm_pooled_qhead", False)
        self.q_head = nn.Linear(emb_dim, 2, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

        # ── Change 3: Gated audio injection ──────────────────────────────────
        self.use_injection_gate = getattr(config, "hrm_injection_gate", False)
        if self.use_injection_gate:
            # sigmoid(3.0) ≈ 0.95 → near pass-through at init
            self.injection_alpha = nn.Parameter(torch.tensor(3.0))

        # ── Per-recording embedding ──────────────────────────────────────────
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

        # ── Output norm (matches Encoder.layer_norm) ─────────────────────────
        self.layer_norm = LayerNorm(emb_dim)

        # ── Change 5: Output adapter ─────────────────────────────────────────
        self.use_output_adapter = getattr(config, "hrm_output_adapter", False)
        if self.use_output_adapter:
            adapter_dim = emb_dim // 4  # 128 for emb_dim=512
            self.output_adapter = nn.Sequential(
                nn.Linear(emb_dim, adapter_dim),
                nn.GELU(),
                nn.Linear(adapter_dim, emb_dim),
            )
            # Zero-init output projection → identity residual at start
            nn.init.zeros_(self.output_adapter[-1].weight)
            nn.init.zeros_(self.output_adapter[-1].bias)

    # ─────────────────────────────────────────────────────────────────────────
    # Carry initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_carry(
        self,
        B: int,
        T: int,
        device: torch.device,
        audio_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Change 2: content-conditioned carry initialization
        if self.use_content_init and audio_features is not None:
            z_H = self.H_init_proj(audio_features)  # [B, T, emb_dim]
            z_L = self.L_init_proj(audio_features)   # [B, T, emb_dim]
        else:
            z_H = self.H_init.view(1, 1, -1).expand(B, T, -1).clone()
            z_L = self.L_init.view(1, 1, -1).expand(B, T, -1).clone()
        return z_H, z_L

    # ─────────────────────────────────────────────────────────────────────────
    # Single ACT step
    # ─────────────────────────────────────────────────────────────────────────

    def _act_step(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        audio_features: torch.Tensor,
        cos_sin: Optional[CosSin] = None,
        deterministic: bool = False,
        encoder_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs H_cycles × L_cycles transformer passes with truncated BPTT.

        Structure (H_cycles=2, L_cycles=2 example, total 4 passes):
            [no_grad] L pass  (H=0, L=0)
            [no_grad] L pass  (H=0, L=1)
            [no_grad] H pass  (H=0)
            [no_grad] L pass  (H=1, L=0)
            [with_grad] L pass (H=1, L=1) ← gradient window starts here
            [with_grad] H pass (H=1)       ← gradient window ends here

        Returns:
            z_H  [B, T, emb_dim]  with gradient (from final H pass)
            z_L  [B, T, emb_dim]  with gradient (from final L pass)
            q_halt    [B]         halt logit
            q_continue [B]        continue logit

        IMPORTANT: z_H and z_L returned here HAVE gradients.
        The caller MUST detach them before passing to the next ACT step.
        """
        # Change 3: gated audio injection
        if self.use_injection_gate:
            gate = torch.sigmoid(self.injection_alpha)
            l_injection = z_H + gate * audio_features
        else:
            l_injection = z_H + audio_features

        # Warm-up passes — no gradients (truncated BPTT window = 1 step)
        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    is_last = (
                        h == self.H_cycles - 1 and l == self.L_cycles - 1
                    )
                    if not is_last:
                        z_L = self.L_level(
                            z_L,
                            l_injection,
                            cos_sin=cos_sin,
                            deterministic=deterministic,
                            attention_mask=encoder_mask,
                        )

                if h < self.H_cycles - 1:
                    z_H = self.H_level(
                        z_H, z_L,
                        cos_sin=cos_sin,
                        deterministic=deterministic,
                        attention_mask=encoder_mask,
                    )
                    # Recompute injection after H update
                    if self.use_injection_gate:
                        l_injection = z_H + gate * audio_features
                    else:
                        l_injection = z_H + audio_features

        # Recompute injection for final pass (z_H may have changed in warm-up)
        if self.use_injection_gate:
            l_injection = z_H + gate * audio_features
        else:
            l_injection = z_H + audio_features

        # Final L and H pass — gradient flows here
        z_L = self.L_level(
            z_L,
            l_injection,
            cos_sin=cos_sin,
            deterministic=deterministic,
            attention_mask=encoder_mask,
        )
        z_H = self.H_level(
            z_H, z_L,
            cos_sin=cos_sin,
            deterministic=deterministic,
            attention_mask=encoder_mask,
        )

        # Change 4: sequence-wide Q-head with pooling
        if self.use_pooled_qhead:
            if encoder_mask is not None:
                mask_float = encoder_mask.float().unsqueeze(-1)  # [B, T, 1]
                pooled = (z_H * mask_float).sum(1) / mask_float.sum(1).clamp_min(1)
            else:
                pooled = z_H.mean(dim=1)  # [B, emb_dim]
            q = self.q_head(pooled)       # [B, 2]
        else:
            q = self.q_head(z_H[:, 0])    # [B, 2]
        q_halt    = q[:, 0]               # [B]
        q_continue = q[:, 1]              # [B]

        return z_H, z_L, q_halt, q_continue

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,
        encoder_mask=None,
        deterministic: bool = False,
        recording_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Returns:
            encoded   [B, T, emb_dim]  — final z_H, layer-normed, ready for
                                          cross-attention in the decoder
            hrm_aux   dict | None      — Q-logits and bootstrap targets for
                                          compute_hrm_q_loss() in train.py
                                          (None when deterministic=True)
        """
        B, T, _ = encoder_input_tokens.shape
        device = encoder_input_tokens.device
        deterministic_mode = bool(deterministic) or (not self.training)

        # ── Convert encoder_mask to key-validity [B, T] bool if needed ───────
        # T5.encode() currently passes encoder_mask=None for HRMEncoder, so
        # this conversion is a no-op in practice but makes the code correct
        # for variable-length batches with padding.
        attn_mask: Optional[torch.Tensor] = None
        if encoder_mask is not None:
            if encoder_mask.dim() == 4:
                # [B, 1, T, T] → [B, T] key-validity
                attn_mask = encoder_mask[:, 0, 0, :].bool()
            elif encoder_mask.dim() == 2:
                attn_mask = encoder_mask.bool()

        # ── 1. Compute RoPE cos/sin once per forward ─────────────────────────
        cos_sin: Optional[CosSin] = None
        if self.use_rope and hasattr(self, "rotary_emb"):
            cos_full, sin_full = self.rotary_emb()    # [max_len, head_dim]
            cos_sin = (cos_full[:T], sin_full[:T])     # slice to actual seq len T

        # ── 2. Input projection + optional positional encoding ───────────────
        # Change 1: frequency-aware stem or plain dense
        if self.use_freq_stem:
            x = self.freq_stem(encoder_input_tokens)   # [B, T, emb_dim]
        else:
            x = self.dense(encoder_input_tokens)       # [B, T, emb_dim]

        if self.use_fixed_embed:
            positions = torch.arange(T, device=device).unsqueeze(0)
            x = x + self.embed(positions)

        x = self.input_norm(
            F.dropout(
                x,
                p=self.dropout.p,
                training=self.training and (not deterministic_mode),
            )
        )

        # ── 3. Per-recording embedding ────────────────────────────────────────
        if self.has_recording_embedding and recording_ids is not None:
            rec = self.recording_proj(
                self.recording_embed(recording_ids)  # [B, rec_dim]
            )                                         # [B, emb_dim]
            x = x + rec.unsqueeze(1)

        audio_features = x   # Fixed anchor — identical on every ACT step

        # ── 4. Initialise carry ───────────────────────────────────────────────
        # Change 2: pass audio_features for content-conditioned init
        z_H, z_L = self._init_carry(B, T, device, audio_features=audio_features)

        # ── 5. ACT loop ───────────────────────────────────────────────────────
        hrm_aux = None

        if deterministic_mode:
            # ── Inference path ────────────────────────────────────────────────
            for _ in range(self.max_steps):
                z_H, z_L, _, _ = self._act_step(
                    z_H, z_L, audio_features,
                    cos_sin=cos_sin,
                    deterministic=True,
                    encoder_mask=attn_mask,
                )
                z_H = z_H.detach()
                z_L = z_L.detach()

        else:
            # ── Training path ─────────────────────────────────────────────────
            # Phase 1: run all steps, collect Q values — no bootstrap peek yet.
            q_halt_list: List[torch.Tensor] = []
            q_cont_list: List[torch.Tensor] = []
            steps_run = 0

            for step in range(self.max_steps):
                is_last = step == self.max_steps - 1
                steps_run = step + 1

                z_H, z_L, q_halt, q_cont = self._act_step(
                    z_H, z_L, audio_features,
                    cos_sin=cos_sin,
                    deterministic=False,
                    encoder_mask=attn_mask,
                )
                q_halt_list.append(q_halt)
                q_cont_list.append(q_cont)

                if not is_last:
                    # Exploration: randomly require a minimum number of steps
                    min_steps = torch.where(
                        torch.rand(B, device=device) < self.explore_p,
                        torch.randint(2, self.max_steps + 1, (B,), device=device),
                        torch.zeros(B, dtype=torch.long, device=device),
                    )
                    should_halt = (q_halt > q_cont) & ((step + 1) >= min_steps)
                    if should_halt.all():
                        break
                    # Detach carry between steps (truncated BPTT).
                    # Do NOT detach after the final step — keep gradient to encoded.
                    z_H = z_H.detach()
                    z_L = z_L.detach()

            # Phase 2: build bootstrap targets post-loop (no extra forward pass
            # for intermediate steps — reduces computation from 2×max_steps to
            # max_steps+1 forward passes per training iteration).
            n_steps = len(q_halt_list)
            # One peek-ahead for the last step only
            with torch.no_grad():
                _, _, nq_halt_peek, _ = self._act_step(
                    z_H.detach(), z_L.detach(), audio_features,
                    cos_sin=cos_sin,
                    deterministic=True,
                    encoder_mask=attn_mask,
                )
                bootstrap_last = torch.sigmoid(nq_halt_peek)

            q_cont_pairs: List[tuple] = []
            for i in range(n_steps):
                if i < n_steps - 1:
                    # Next-step Q values are already in lists — no extra call needed
                    nq_halt = q_halt_list[i + 1].detach()
                    nq_cont = q_cont_list[i + 1].detach()
                    bootstrap = torch.sigmoid(torch.maximum(nq_halt, nq_cont))
                else:
                    bootstrap = bootstrap_last
                q_cont_pairs.append((q_cont_list[i], bootstrap))

            hrm_aux = {
                "q_halt_list":  q_halt_list,
                "q_cont_pairs": q_cont_pairs,
                "steps_per_example": torch.full(
                    (B,), steps_run, device=device, dtype=torch.long,
                ),
                "q_halt_last": q_halt_list[-1].detach() if q_halt_list else None,
                "q_continue_last": q_cont_list[-1].detach() if q_cont_list else None,
                # For Q-loss normalisation and soft target sharpness
                "max_steps": self.max_steps,
                "soft_k": float(getattr(self.config, "hrm_q_soft_k", 5.0)),
            }

        # z_H here has gradient from its last _act_step call (training)
        # or is detached (inference). Either way, layer_norm keeps the gradient.
        encoded = self.layer_norm(
            F.dropout(
                z_H,
                p=self.dropout.p,
                training=self.training and (not deterministic_mode),
            )
        )

        # Change 5: output adapter — zero-init bottleneck for decoder alignment
        if self.use_output_adapter:
            encoded = encoded + self.output_adapter(encoded)

        # Cast to config.dtype (e.g. bfloat16) to match what the decoder expects.
        # Mirrors Encoder.forward: x = x.type(cfg.dtype)
        if hasattr(self.config, "dtype") and self.config.dtype is not None:
            encoded = encoded.to(self.config.dtype)

        return encoded, hrm_aux


def compute_hrm_q_loss(
    hrm_aux: dict,
    decoder_loss_per_example: torch.Tensor,  # [B] per-example CE loss, detached
    return_components: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
    """
    Q-learning losses for HRM encoder halt/continue heads.

    Matching HRM losses.py:
        total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

    halt target:    soft_is_easy = sigmoid(-soft_k * (per_example_loss - median))
                    soft_k=5.0 by default; at large k this approaches the hard binary
                    (loss < median) target from the original HRM. The soft version
                    provides gradient signal for boundary examples.
    continue target: bootstrapped from next step's Q values (deferred in forward).

    Normalisation: divided by max_steps (not n_steps actually run), so early halting
    does not inflate gradient magnitude.

    reduction='mean': consistent per-step gradient magnitude across batch sizes and
    step counts (original used 'sum' which coupled magnitude to batch size).

    Args:
        hrm_aux: dict from HRMEncoder containing
            'q_halt_list':  List[Tensor[B]]
            'q_cont_pairs': List[(Tensor[B], Tensor[B])]  — (logit, target)
            'max_steps':    int
            'soft_k':       float
        decoder_loss_per_example: [B] per-example cross-entropy (detached)

    Returns:
        scalar: 0.5 * (q_halt_term + q_cont_term)
    """
    # Soft is_easy target: sigmoid(-k * (loss - median))
    # Higher loss → lower is_easy (closer to 0 → continue)
    # Lower loss  → higher is_easy (closer to 1 → halt)
    soft_k = float(hrm_aux.get("soft_k", 5.0))
    median_loss = decoder_loss_per_example.median().detach()
    is_easy = torch.sigmoid(
        -soft_k * (decoder_loss_per_example.detach() - median_loss)
    )

    device = is_easy.device
    q_halt_loss = torch.tensor(0.0, device=device)
    q_cont_loss = torch.tensor(0.0, device=device)
    n_steps = len(hrm_aux["q_halt_list"])
    # Normalise by max_steps so early halting does not inflate gradient magnitude
    normalizer = max(int(hrm_aux.get("max_steps", n_steps)), 1)

    for q_halt in hrm_aux["q_halt_list"]:
        # reduction='mean': gradient magnitude is independent of batch size
        q_halt_loss = q_halt_loss + F.binary_cross_entropy_with_logits(
            q_halt, is_easy, reduction="mean"
        )

    for q_cont, bootstrap in hrm_aux["q_cont_pairs"]:
        q_cont_loss = q_cont_loss + F.binary_cross_entropy_with_logits(
            q_cont, bootstrap, reduction="mean"
        )

    q_halt_term = 0.5 * q_halt_loss / normalizer
    q_cont_term = 0.5 * q_cont_loss / normalizer
    q_loss_total = q_halt_term + q_cont_term

    if not return_components:
        return q_loss_total

    components = {
        "q_halt_term": q_halt_term.detach(),
        "q_cont_term": q_cont_term.detach(),
        "q_steps": torch.tensor(float(n_steps), device=device),
    }
    return q_loss_total, components
