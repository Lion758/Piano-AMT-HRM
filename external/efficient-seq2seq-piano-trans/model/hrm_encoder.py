"""
HRM Encoder for AMT (Automatic Music Transcription)

Ported from: external/HRM/models/hrm/hrm_act_v1.py
Fixes vs first draft:
  - BUG: z_H/z_L now detached between ACT steps (truncated BPTT was broken)
  - BUG: target_q_continue computed correctly for last ACT step
  - INIT: trunc_normal_init_ ported from HRM common.py (PyTorch's version is wrong)
  - INIT: H_init/L_init std=1 (matching HRM) not 0.02
  - INIT: H_init/L_init are nn.Buffer (non-trainable) matching HRM exactly
  - LOSS: Q-loss weight 0.5 per HRM losses.py, reduction='sum' not 'mean'

Intentional divergences from HRM (AMT-appropriate):
  - Pre-norm (your existing style) rather than HRM's post-norm
  - Sinusoidal FixedEmbed rather than RoPE (matches your existing Encoder)
  - ReLU MLP rather than SwiGLU (matches your existing MlpBlock)
  - float32 rather than bfloat16
  - Per-example CE loss proxy for seq_is_correct (AMT has no exact-match signal)
  - No puzzle_emb_len offset on output (no prepended puzzle tokens in AMT)

Optional additions from HRM (see bottom of file):
  - trunc_normal_init_ available as a utility
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.Layers import LayerNorm, MlpBlock, FixedEmbed
from model.Attention import Multi_Head_Attention


# ─────────────────────────────────────────────────────────────────────────────
# Utility: correct truncated normal init
# Ported directly from external/HRM/models/common.py
# PyTorch's nn.init.trunc_normal_ is NOT mathematically correct —
# the std of the initialised tensor does not equal the std argument.
# This version matches JAX's default truncated normal (used throughout HRM).
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class HRMTransformerLayer(nn.Module):
    """
    Single pre-norm transformer block (your existing style).

    Note on norm style:
        HRM original uses POST-norm:
            hidden = rms_norm(hidden + attn(hidden))
            hidden = rms_norm(hidden + mlp(hidden))
        This implementation uses PRE-norm (your existing EncoderLayer style):
            hidden = hidden + attn(norm(hidden))
            hidden = hidden + mlp(norm(hidden))
        Pre-norm is more stable for training and matches your codebase.
        To switch to post-norm, invert the order of norm/residual below.
    """

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int,
        dropout_rate: float,
        window_size: Optional[int] = None,
    ):
        super().__init__()
        self.norm1 = LayerNorm(emb_dim)
        self.attn = Multi_Head_Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            window_size=window_size,  # None → global; int → flash-attn sliding window
            is_causal=False,
        )
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(emb_dim)
        self.mlp = MlpBlock(
            emb_dim=emb_dim,
            intermediate_dim=mlp_dim,
            activations="relu",
            intermediate_dropout_rate=dropout_rate,
        )
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm1(x)
        x = self.attn(x, x)
        x = self.drop1(x) + r
        r = x
        x = self.norm2(x)
        x = self.mlp(x)
        return self.drop2(x) + r


class HRMReasoningStack(nn.Module):
    """
    Stack of transformer layers with additive input injection on entry.

        hidden = hidden + injection   (grounds the stack in the input signal)
        for layer in layers:
            hidden = layer(hidden)

    This is exactly HRM's ReasoningModule pattern (hrm_act_v1.py line ~80).
    The injection for L-level is (z_H + audio_features), keeping L perpetually
    grounded in the raw audio regardless of how many refinement steps have run.
    """

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(
        self, hidden: torch.Tensor, injection: torch.Tensor
    ) -> torch.Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(hidden)
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
        use_hrm_encoder:       bool  (default False)
        hrm_H_layers:          int   (default 3)
        hrm_L_layers:          int   (default 3)
        hrm_H_cycles:          int   (default 2)
        hrm_L_cycles:          int   (default 2)
        hrm_max_steps:         int   (default 4)
        hrm_L_window_size:     int   (default 64)   — local attn window, L-level only
        hrm_halt_explore_prob: float (default 0.1)
        num_recordings:        int   (default 0)    — 0 disables recording embeddings
        recording_emb_dim:     int   (default 64)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        emb_dim   = config.emb_dim        # 512
        num_heads = config.num_heads       # 8
        head_dim  = config.head_dim        # 64
        mlp_dim   = config.mlp_dim         # 1024
        dropout   = config.dropout_rate    # 0.1

        H_layers = getattr(config, "hrm_H_layers", 3)
        L_layers = getattr(config, "hrm_L_layers", 3)
        L_window = getattr(config, "hrm_L_window_size", 64)

        self.H_cycles  = getattr(config, "hrm_H_cycles", 2)
        self.L_cycles  = getattr(config, "hrm_L_cycles", 2)
        self.max_steps = getattr(config, "hrm_max_steps", 4)
        self.explore_p = getattr(config, "hrm_halt_explore_prob", 0.1)

        # ── Input projection (matches Encoder.dense + Encoder.embed) ────────
        self.dense    = nn.Linear(config.encoder_input_dim, emb_dim)
        self.embed    = FixedEmbed(features=emb_dim)
        self.dropout  = nn.Dropout(dropout)
        self.input_norm = LayerNorm(emb_dim)

        # ── L-level: audio-grounded, local window attention ──────────────────
        # window_size=L_window → flash_attn sliding window path in your
        # Multi_Head_Attention (same as V1 encoder_window_size=64)
        self.L_level = HRMReasoningStack(
            nn.ModuleList([
                HRMTransformerLayer(
                    emb_dim, num_heads, head_dim, mlp_dim, dropout,
                    window_size=L_window,
                )
                for _ in range(L_layers)
            ])
        )

        # ── H-level: abstract planner, global attention ──────────────────────
        # window_size=None → F.scaled_dot_product_attention (full attention)
        self.H_level = HRMReasoningStack(
            nn.ModuleList([
                HRMTransformerLayer(
                    emb_dim, num_heads, head_dim, mlp_dim, dropout,
                    window_size=None,
                )
                for _ in range(H_layers)
            ])
        )

        # ── Initial carry states ─────────────────────────────────────────────
        # MATCHING HRM: nn.Buffer (non-trainable), std=1 (not 0.02),
        # and using the correct truncated normal (not PyTorch's buggy version).
        #
        # Why non-trainable (Buffer, not Parameter)?
        #   HRM treats these as fixed initial conditions, not learned.
        #   The model learns to refine FROM these states, not to optimise them.
        #   Making them trainable would conflate "good starting point" with
        #   "what the model should compute", which can cause instability.
        #
        # Why std=1?
        #   Broad initial state gives the model diverse directions to refine
        #   from on step 1. std=0.02 would start near-zero (essentially the
        #   zero vector) and may slow convergence.
        self.register_buffer(
            "H_init",
            trunc_normal_init_(torch.empty(emb_dim), std=1.0),
        )
        self.register_buffer(
            "L_init",
            trunc_normal_init_(torch.empty(emb_dim), std=1.0),
        )

        # ── Q-head ───────────────────────────────────────────────────────────
        # Reads z_H[:, 0] — position 0 acts as a CLS / global summary token.
        # Output: [q_halt_logit, q_continue_logit] per example.
        # Zero-weight + bias=-5 init: starts near-zero → forces exploration early.
        # Matching HRM exactly.
        self.q_head = nn.Linear(emb_dim, 2, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)

        # ── Per-recording embedding ──────────────────────────────────────────
        # Directly analogous to HRM's puzzle_emb (CastedSparseEmbedding).
        # audio_ids is already in your batch from dataset_Audio2Midi.py.
        # Zero-init: model starts from unbiased baseline and learns offsets.
        self.num_recordings = getattr(config, "num_recordings", 0)
        rec_dim = getattr(config, "recording_emb_dim", 64)
        if self.num_recordings > 0:
            self.recording_embed = nn.Embedding(self.num_recordings, rec_dim)
            nn.init.zeros_(self.recording_embed.weight)
            # Project rec_dim → emb_dim only if they differ
            if rec_dim != emb_dim:
                self.recording_proj = nn.Linear(rec_dim, emb_dim, bias=False)
                nn.init.zeros_(self.recording_proj.weight)
            else:
                self.recording_proj = nn.Identity()

        # ── Output norm (matches Encoder.layer_norm) ─────────────────────────
        self.layer_norm = LayerNorm(emb_dim)

    # ─────────────────────────────────────────────────────────────────────────
    # Carry initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_carry(
        self, B: int, T: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (z_H, z_L) each [B, T, emb_dim], broadcast from H_init/L_init.
        Equivalent to HRM's reset_carry() when halted=True for all examples.
        """
        z_H = self.H_init.view(1, 1, -1).expand(B, T, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, T, -1).clone()
        return z_H, z_L

    # ─────────────────────────────────────────────────────────────────────────
    # Single ACT step
    # ─────────────────────────────────────────────────────────────────────────

    def _act_step(
        self,
        z_H: torch.Tensor,          # [B, T, emb_dim] — detached on entry
        z_L: torch.Tensor,          # [B, T, emb_dim] — detached on entry
        audio_features: torch.Tensor,  # [B, T, emb_dim] — fixed, never changes
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

        This exactly matches hrm_act_v1.py _Inner.forward() lines ~110-135.

        Returns:
            z_H  [B, T, emb_dim]  with gradient (from final H pass)
            z_L  [B, T, emb_dim]  with gradient (from final L pass)
            q_halt    [B]         halt logit (positive = confident, should stop)
            q_continue [B]        continue logit (positive = uncertain, keep going)

        IMPORTANT: z_H and z_L returned here HAVE gradients.
        The caller MUST detach them before passing to the next ACT step.
        See forward() for the detach pattern.
        """
        # Warm-up passes — no gradients (truncated BPTT window = 1 step)
        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    is_last = (
                        h == self.H_cycles - 1 and l == self.L_cycles - 1
                    )
                    if not is_last:
                        # L receives z_H + raw audio (always grounded in input)
                        z_L = self.L_level(z_L, z_H + audio_features)

                if h < self.H_cycles - 1:
                    # H reads L's updated summary
                    z_H = self.H_level(z_H, z_L)

        # Final L and H pass — gradient flows here
        z_L = self.L_level(z_L, z_H + audio_features)
        z_H = self.H_level(z_H, z_L)

        # Q-head: z_H[:, 0] as global summary token (same as HRM)
        q = self.q_head(z_H[:, 0])   # [B, 2]
        q_halt    = q[:, 0]           # [B]
        q_continue = q[:, 1]          # [B]

        # NOTE: returns z_H/z_L WITH gradients.
        # Caller must .detach() before feeding to next step.
        return z_H, z_L, q_halt, q_continue

    # ─────────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,       # [B, T, encoder_input_dim]
        encoder_mask=None,                         # accepted, unused (HRM uses no mask)
        deterministic: bool = False,
        recording_ids: Optional[torch.Tensor] = None,  # [B] long — audio_ids_contiguous
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

        # ── 1. Input projection + positional encoding ────────────────────────
        x = self.dense(encoder_input_tokens)          # [B, T, emb_dim]
        positions = torch.arange(T, device=device).unsqueeze(0)
        x = x + self.embed(positions)
        x = self.input_norm(self.dropout(x))

        # ── 2. Per-recording embedding ───────────────────────────────────────
        # Analogous to HRM's puzzle_emb. Zero-init means recordings start
        # identical and diverge as the model learns recording-specific biases.
        if self.num_recordings > 0 and recording_ids is not None:
            rec = self.recording_proj(
                self.recording_embed(recording_ids)  # [B, rec_dim]
            )                                         # [B, emb_dim]
            x = x + rec.unsqueeze(1)                 # broadcast over T

        audio_features = x   # Fixed anchor — identical on every ACT step

        # ── 3. Initialise carry ──────────────────────────────────────────────
        # Both z_H and z_L start from their registered buffer values.
        # No gradients yet (buffers are non-trainable; clone() gives a fresh tensor).
        z_H, z_L = self._init_carry(B, T, device)

        # ── 4. ACT loop ──────────────────────────────────────────────────────
        hrm_aux = None

        if deterministic or not self.training:
            # ── Inference path ───────────────────────────────────────────────
            # Always run exactly max_steps (no early halting, for batch uniformity).
            # Matching HRM comment: "During evaluation, always use max steps,
            # this is to guarantee the same halting steps inside a batch."
            for _ in range(self.max_steps):
                z_H, z_L, _, _ = self._act_step(z_H, z_L, audio_features)
                # FIX: detach carry between steps.
                # z_H/z_L from _act_step have gradients; detach before next step.
                # (Moot in inference since torch.no_grad() wraps this, but
                # kept explicit for clarity and safety.)
                z_H = z_H.detach()
                z_L = z_L.detach()

        else:
            # ── Training path ─────────────────────────────────────────────────
            q_halt_list:  List[torch.Tensor] = []
            q_cont_pairs: List[tuple]        = []

            for step in range(self.max_steps):
                is_last = step == self.max_steps - 1

                z_H, z_L, q_halt, q_cont = self._act_step(z_H, z_L, audio_features)
                q_halt_list.append(q_halt)

                # ── Bootstrap target for Q_continue ──────────────────────────
                # Peek at next step's Q values to get the bootstrap target.
                # MATCHING HRM losses.py and hrm_act_v1.py:
                #   - non-last step: target = sigmoid(max(next_q_halt, next_q_cont))
                #   - last step:     target = sigmoid(next_q_halt)   ← NOT None
                # Both cases compute a target; only the formula differs.
                with torch.no_grad():
                    _, _, nq_halt, nq_cont = self._act_step(
                        z_H.detach(), z_L.detach(), audio_features
                    )
                    if is_last:
                        # Last step: can only halt next, so target uses q_halt only
                        bootstrap = torch.sigmoid(nq_halt)
                    else:
                        # Non-last: take the best available action next step
                        bootstrap = torch.sigmoid(torch.maximum(nq_halt, nq_cont))

                q_cont_pairs.append((q_cont, bootstrap))

                # ── Halting decision ──────────────────────────────────────────
                if not is_last:
                    # Exploration: randomly require a minimum number of steps
                    # before allowing halting. This prevents the model from
                    # always halting at step 1 before learning anything.
                    min_steps = torch.where(
                        torch.rand(B, device=device) < self.explore_p,
                        torch.randint(2, self.max_steps + 1, (B,), device=device),
                        torch.zeros(B, dtype=torch.long, device=device),
                    )
                    should_halt = (q_halt > q_cont) & ((step + 1) >= min_steps)
                    if should_halt.all():
                        # All examples in batch are confident — stop early
                        break

                # FIX (CRITICAL): Detach z_H and z_L before next ACT step.
                # This implements truncated BPTT with window=1 (matching HRM).
                # Without this, gradients flow across ACT steps, which:
                #   (a) is not what HRM does
                #   (b) causes memory to grow linearly with max_steps
                #   (c) can cause gradient instability
                # The 1-step gradient window in _act_step only applies WITHIN
                # a single call. Between calls, we always detach.
                z_H = z_H.detach()
                z_L = z_L.detach()

            hrm_aux = {
                "q_halt_list":  q_halt_list,   # List[Tensor[B]]
                "q_cont_pairs": q_cont_pairs,  # List[(q_cont [B], bootstrap [B])]
            }

        # z_H here has gradient from its last _act_step call (training)
        # or is detached (inference). Either way, layer_norm keeps the gradient.
        encoded = self.layer_norm(self.dropout(z_H))
        return encoded, hrm_aux


# ─────────────────────────────────────────────────────────────────────────────
# Q-loss function (put in train.py before MT3Trainer class)
# ─────────────────────────────────────────────────────────────────────────────

def compute_hrm_q_loss(
    hrm_aux: dict,
    decoder_loss_per_example: torch.Tensor,  # [B] per-example CE loss, detached
) -> torch.Tensor:
    """
    Q-learning losses for HRM encoder halt/continue heads.

    Matching HRM losses.py:
        total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)

    The 0.5 weight and reduction='sum' are from the original.
    reduction='sum' is consistent with how lm_loss is summed over the batch.

    halt target:    is_easy = (per_example_loss < batch_median)
                    This is the AMT proxy for HRM's binary seq_is_correct.
    continue target: bootstrap from next step's Q values (computed in forward).

    Args:
        hrm_aux: dict from HRMEncoder containing
            'q_halt_list':  List[Tensor[B]]          — one per ACT step run
            'q_cont_pairs': List[(Tensor[B], Tensor[B])] — (logit, target)
        decoder_loss_per_example: [B] per-example cross-entropy (detached)

    Returns:
        scalar: 0.5 * (q_halt_loss + q_continue_loss)
                (to be added to main CE loss: total = ce + q_loss)
    """
    # "is_easy": below-median loss → encoder should be confident → halt
    # Detached: Q-loss does not backprop through the decoder loss computation.
    median_loss = decoder_loss_per_example.median().detach()
    is_easy = (decoder_loss_per_example.detach() < median_loss).float()

    device = is_easy.device
    q_halt_loss = torch.tensor(0.0, device=device)
    q_cont_loss = torch.tensor(0.0, device=device)
    n_steps = len(hrm_aux["q_halt_list"])

    for q_halt in hrm_aux["q_halt_list"]:
        # Matching HRM: reduction='sum' (consistent with lm_loss summing)
        q_halt_loss = q_halt_loss + F.binary_cross_entropy_with_logits(
            q_halt, is_easy, reduction="sum"
        )

    for q_cont, bootstrap in hrm_aux["q_cont_pairs"]:
        # bootstrap is always a valid tensor (never None — see forward())
        q_cont_loss = q_cont_loss + F.binary_cross_entropy_with_logits(
            q_cont, bootstrap, reduction="sum"
        )

    # Average over steps, then weight at 0.5 (matching HRM losses.py)
    return 0.5 * (q_halt_loss + q_cont_loss) / max(n_steps, 1)
