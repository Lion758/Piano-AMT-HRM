# HRM Integration into Your T5 AMT System
## Codebase-Anchored Engineering Guide

---

## What You Are Adding and Where It Lives

| HRM Component | File | Mechanism |
|---|---|---|
| Two-level hierarchy (L + H) | `model/hrm_encoder.py` (new) | L-level = local-attn encoder, H-level = global-attn encoder |
| Recurrent carry (z_H, z_L) | `model/hrm_encoder.py` (new) | Iterative refinement loop inside a single `forward()` call |
| Learned halting (Q-head) | `model/hrm_encoder.py` (new) | Q-head on z_H[:, 0]; loss computed in `train.py` |
| Per-recording embedding | `model/hrm_encoder.py` (new) | nn.Embedding over `audio_ids` already in your batch |
| Wiring | `model/T5.py` | Swap `Encoder` → `HRMEncoder` in `__init__` and `encode()` |
| Loss | `train.py` | Add Q-loss in `forward_step()`, add recording_ids forwarding |
| Config | `config/experiment_T5_V5_HRM.yaml` (new) | Inherits from V4, adds HRM fields |

The decoder (`model/Decoder.py`) is **completely untouched**. The carry is internal to `HRMEncoder.forward()` — it runs N passes on the same audio clip within a single call, then discards. You do not carry state between training steps.

---

## Step 1 — Create `model/hrm_encoder.py`

This file uses only classes already in your codebase:
- `model.Layers`: `LayerNorm`, `MlpBlock`, `FixedEmbed`
- `model.Attention`: `Multi_Head_Attention` (already has `window_size` / flash-attn support)

```python
# model/hrm_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from model.Layers import LayerNorm, MlpBlock, FixedEmbed
from model.Attention import Multi_Head_Attention


# ─── Building block ──────────────────────────────────────────────────────────

class HRMTransformerLayer(nn.Module):
    """
    Single pre-norm transformer block.
    Identical structure to your existing EncoderLayer in model/Encoder.py,
    but re-stated here so H and L can have independent weight sets.
    """
    def __init__(self, emb_dim: int, num_heads: int, head_dim: int,
                 mlp_dim: int, dropout_rate: float,
                 window_size: Optional[int] = None):
        super().__init__()
        self.norm1 = LayerNorm(emb_dim)
        self.attn = Multi_Head_Attention(
            num_heads=num_heads,
            head_dim=head_dim,
            dropout_rate=dropout_rate,
            window_size=window_size,   # None → global; int → flash-attn sliding window
            is_causal=False
        )
        self.drop1 = nn.Dropout(dropout_rate)
        self.norm2 = LayerNorm(emb_dim)
        self.mlp = MlpBlock(
            emb_dim=emb_dim,
            intermediate_dim=mlp_dim,
            activations='relu',
            intermediate_dropout_rate=dropout_rate
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
        x = self.drop2(x) + r
        return x


class HRMReasoningStack(nn.Module):
    """
    A stack of transformer layers with additive injection on entry.
    hidden = hidden + injection  →  layers(hidden)
    This is exactly HRM's ReasoningModule pattern.
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, hidden: torch.Tensor,
                injection: torch.Tensor) -> torch.Tensor:
        hidden = hidden + injection
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


# ─── Main HRM Encoder ────────────────────────────────────────────────────────

class HRMEncoder(nn.Module):
    """
    Drop-in replacement for model/Encoder.py::Encoder.

    Public interface (identical to Encoder):
        forward(encoder_input_tokens, encoder_mask=None, deterministic=False)
        → (encoded [B, T, emb_dim], hrm_aux dict | None)

    Internal mechanism:
        1. Project input + add sinusoidal position embeddings (same as Encoder)
        2. (Optional) Add per-recording embedding
        3. Run max_steps ACT iterations:
              for each step:
                  [no_grad warm-up] H_cycles-1 × L_cycles-1 passes
                  [with_grad] final L pass then final H pass
                  q_head(z_H[:, 0]) → halt/continue logits
        4. Return layer_norm(z_H) and Q-logits for loss

    Config fields read (add to T5.yaml or experiment yaml):
        hrm_H_layers:         int   (default 3)   — H-level transformer blocks
        hrm_L_layers:         int   (default 3)   — L-level transformer blocks
        hrm_H_cycles:         int   (default 2)   — inner H passes per ACT step
        hrm_L_cycles:         int   (default 2)   — inner L passes per ACT step
        hrm_max_steps:        int   (default 4)   — ACT refinement iterations
        hrm_L_window_size:    int   (default 64)  — local attn window for L-level
                                                     matches your V1 encoder_window_size=64
        hrm_halt_explore_prob: float (default 0.1)
        num_recordings:       int   (default 0)   — 0 = disabled
                                                     set to number of unique audio files
        recording_emb_dim:    int   (default 64)  — small offset, not full 512
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        emb_dim   = config.emb_dim        # 512
        num_heads = config.num_heads       # 8
        head_dim  = config.head_dim        # 64
        mlp_dim   = config.mlp_dim         # 1024
        dropout   = config.dropout_rate    # 0.1

        H_layers  = getattr(config, 'hrm_H_layers', 3)
        L_layers  = getattr(config, 'hrm_L_layers', 3)
        L_window  = getattr(config, 'hrm_L_window_size', 64)

        self.H_cycles  = getattr(config, 'hrm_H_cycles', 2)
        self.L_cycles  = getattr(config, 'hrm_L_cycles', 2)
        self.max_steps = getattr(config, 'hrm_max_steps', 4)
        self.explore_p = getattr(config, 'hrm_halt_explore_prob', 0.1)

        # ── Input projection (identical to Encoder.dense + Encoder.embed) ──
        self.dense  = nn.Linear(config.encoder_input_dim, emb_dim)
        self.embed  = FixedEmbed(features=emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = LayerNorm(emb_dim)

        # ── L-level: audio-grounded, local window attention ─────────────────
        # Uses flash-attn sliding window — same path as your V1 encoder_window_size
        self.L_level = HRMReasoningStack(nn.ModuleList([
            HRMTransformerLayer(emb_dim, num_heads, head_dim, mlp_dim,
                                dropout, window_size=L_window)
            for _ in range(L_layers)
        ]))

        # ── H-level: abstract planner, global attention ──────────────────────
        # No window_size → uses F.scaled_dot_product_attention (your existing path)
        self.H_level = HRMReasoningStack(nn.ModuleList([
            HRMTransformerLayer(emb_dim, num_heads, head_dim, mlp_dim,
                                dropout, window_size=None)
            for _ in range(H_layers)
        ]))

        # ── Learned initial carry states ─────────────────────────────────────
        # These are broadcast over (B, T) at the start of each forward
        self.H_init = nn.Parameter(torch.zeros(emb_dim))
        self.L_init = nn.Parameter(torch.zeros(emb_dim))
        nn.init.trunc_normal_(self.H_init, std=0.02)
        nn.init.trunc_normal_(self.L_init, std=0.02)

        # ── Q-head: halt (confident) vs continue (uncertain) ─────────────────
        # Reads from position 0 of z_H as a "CLS" summary token
        self.q_head = nn.Linear(emb_dim, 2, bias=True)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.0)  # near-zero → explore first

        # ── Per-recording embedding (optional) ───────────────────────────────
        # audio_ids is already in your batch (dataset_Audio2Midi.py line:
        #   "audio_ids": torch.tensor(int(self.audio_idx), dtype=torch.long)
        # You need to build a contiguous id map (see Step 5).
        self.num_recordings = getattr(config, 'num_recordings', 0)
        rec_dim = getattr(config, 'recording_emb_dim', 64)
        if self.num_recordings > 0:
            # Small embedding (rec_dim << emb_dim). Acts as a learned bias.
            self.recording_embed = nn.Embedding(self.num_recordings, rec_dim)
            nn.init.zeros_(self.recording_embed.weight)   # zero-init: baseline unchanged
            self.recording_proj = nn.Linear(rec_dim, emb_dim, bias=False)
            nn.init.zeros_(self.recording_proj.weight)

        # ── Output norm (identical to Encoder.layer_norm) ────────────────────
        self.layer_norm = LayerNorm(emb_dim)

    # ── Carry initialization ─────────────────────────────────────────────────

    def _init_carry(self, B: int, T: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_H, z_L) each shaped [B, T, emb_dim]."""
        z_H = self.H_init.view(1, 1, -1).expand(B, T, -1).clone()
        z_L = self.L_init.view(1, 1, -1).expand(B, T, -1).clone()
        return z_H, z_L

    # ── Single ACT step ──────────────────────────────────────────────────────

    def _act_step(
        self,
        z_H: torch.Tensor,
        z_L: torch.Tensor,
        audio_features: torch.Tensor,   # [B, T, emb_dim] — fixed across all steps
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs H_cycles × L_cycles transformer passes.
        All but the last H and L pass run under torch.no_grad (truncated BPTT).
        Returns: (new_z_H, new_z_L, q_halt [B], q_continue [B])
        """
        # Warm-up passes — no gradients (mirrors HRM's inner loop)
        with torch.no_grad():
            for h in range(self.H_cycles):
                for l in range(self.L_cycles):
                    is_last = (h == self.H_cycles - 1) and (l == self.L_cycles - 1)
                    if not is_last:
                        # L always receives raw audio injection (grounded in input)
                        z_L = self.L_level(z_L, z_H + audio_features)
                if h < self.H_cycles - 1:
                    # H reads L's summarised state
                    z_H = self.H_level(z_H, z_L)

        # Final pass receives gradients
        z_L = self.L_level(z_L, z_H + audio_features)
        z_H = self.H_level(z_H, z_L)

        # Q-head: reads z_H position 0 as global summary
        q = self.q_head(z_H[:, 0])   # [B, 2]
        return z_H, z_L, q[:, 0], q[:, 1]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        encoder_input_tokens: torch.Tensor,     # [B, T, encoder_input_dim]
        encoder_mask=None,                       # accepted but not used in HRM path
        deterministic: bool = False,
        recording_ids: Optional[torch.Tensor] = None,  # [B] long — audio_ids
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Returns:
            encoded [B, T, emb_dim]  — final z_H after layer_norm
            hrm_aux dict | None      — Q-logits and bootstrap targets for loss
                                        (None during deterministic/inference)
        """
        B, T, _ = encoder_input_tokens.shape
        device = encoder_input_tokens.device

        # 1. Project + positional encoding (same as Encoder)
        x = self.dense(encoder_input_tokens)
        positions = torch.arange(T, device=device).unsqueeze(0)   # [1, T]
        x = x + self.embed(positions)
        x = self.input_norm(self.dropout(x))

        # 2. Per-recording embedding (zero at init, learns recording-specific bias)
        if self.num_recordings > 0 and recording_ids is not None:
            rec = self.recording_proj(
                self.recording_embed(recording_ids)  # [B, rec_dim]
            )                                         # [B, emb_dim]
            x = x + rec.unsqueeze(1)                 # broadcast over T

        audio_features = x   # fixed anchor — never changes across ACT steps

        # 3. Initialise carry
        z_H, z_L = self._init_carry(B, T, device)

        # 4. ACT loop
        hrm_aux = None

        if deterministic or not self.training:
            # Inference: always run max_steps, no halting logic needed
            for _ in range(self.max_steps):
                z_H, z_L, _, _ = self._act_step(z_H, z_L, audio_features)
        else:
            # Training: run with Q-learning bookkeeping
            q_halt_list:   List[torch.Tensor] = []
            q_cont_pairs:  List[tuple]        = []

            for step in range(self.max_steps):
                is_last = (step == self.max_steps - 1)
                z_H, z_L, q_halt, q_cont = self._act_step(z_H, z_L, audio_features)
                q_halt_list.append(q_halt)

                if not is_last:
                    # Bootstrap target for Q_continue: peek at next step's Q_halt
                    with torch.no_grad():
                        _, _, nq_halt, nq_cont = self._act_step(
                            z_H.detach(), z_L.detach(), audio_features
                        )
                        bootstrap = torch.sigmoid(torch.maximum(nq_halt, nq_cont))
                    q_cont_pairs.append((q_cont, bootstrap))
                else:
                    q_cont_pairs.append((q_cont, None))  # last step has no target

                # Exploration: force minimum steps before halting
                if not is_last:
                    min_steps = torch.where(
                        torch.rand(B, device=device) < self.explore_p,
                        torch.randint(2, self.max_steps + 1, (B,), device=device),
                        torch.zeros(B, dtype=torch.long, device=device)
                    )
                    should_halt = (q_halt > q_cont) & ((step + 1) >= min_steps)
                    if should_halt.all():
                        break

            hrm_aux = {
                'q_halt_list':  q_halt_list,   # List[Tensor[B]], length = n_steps run
                'q_cont_pairs': q_cont_pairs,  # List[(q_cont, bootstrap | None)]
            }

        encoded = self.layer_norm(self.dropout(z_H))
        return encoded, hrm_aux
```

---

## Step 2 — Modify `model/T5.py`

Three targeted changes only. Everything else stays exactly as-is.

### 2a — Import HRMEncoder at the top

```python
# model/T5.py — add after existing imports
from model.hrm_encoder import HRMEncoder
```

### 2b — Add HRM branch in `Transformer.__init__`

The existing block starts at line ~29:
```python
if config.encoder_name == "TransformerEncoder":
    self.encoder = Encoder(config)
```

Add one branch:
```python
if getattr(config, 'use_hrm_encoder', False):
    self.encoder = HRMEncoder(config)
elif config.encoder_name == "TransformerEncoder":
    self.encoder = Encoder(config)
elif config.encoder_name == "CNNEncoder":
    ...   # rest unchanged
```

### 2c — Update `Transformer.encode()` to handle HRM's two-tuple return

Replace the current `encode()` method:

```python
def encode(self, encoder_input_tokens, encoder_segment_ids=None,
           enable_dropout=True, recording_ids=None):
    """
    Returns:
        encoder_outputs [B, T, emb_dim]
        hrm_aux dict | None
    """
    assert encoder_input_tokens.ndim == 3

    if isinstance(self.encoder, HRMEncoder):
        encoder_outputs, hrm_aux = self.encoder(
            encoder_input_tokens,
            deterministic=not enable_dropout,
            recording_ids=recording_ids,
        )
        return encoder_outputs, hrm_aux

    # ── Original path (all non-HRM encoders) ─────────────────────────────
    encoder_mask = make_attention_mask(
        torch.ones(encoder_input_tokens.shape[:-1]),
        torch.ones(encoder_input_tokens.shape[:-1]),
        dtype=self.config.dtype
    )
    if encoder_segment_ids is not None:
        encoder_mask = combine_masks(
            encoder_mask,
            make_attention_mask(encoder_segment_ids, encoder_segment_ids,
                                torch.equal, dtype=self.config.dtype)
        )
    encoder_mask = encoder_mask.to(encoder_input_tokens.device)

    if self.config.froze_encoder:
        with torch.no_grad():
            encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask,
                                           deterministic=not enable_dropout)
            encoder_outputs = encoder_outputs.detach()
    else:
        encoder_outputs = self.encoder(encoder_input_tokens, encoder_mask,
                                       deterministic=not enable_dropout)
    return encoder_outputs, None
```

### 2d — Update `Transformer.forward()` to pass recording_ids and collect hrm_aux

Add `recording_ids=None` to the signature, then change the encode call:

```python
def forward(self, encoder_input_tokens=None, decoder_target_tokens=None,
            decoder_input_tokens=None, encoder_segment_ids=None,
            decoder_segment_ids=None, encoder_positions=None,
            decoder_positions=None, enable_dropout=True, decode=False,
            dur_inputs=None, dur_targets=None,
            decoder_targets_frame_index=None, encoder_decoder_mask=None,
            recording_ids=None):   # ← ADD THIS

    res_dict = {}
    if decoder_input_tokens is None:
        decoder_input_tokens = self._shift_right(decoder_target_tokens, shift_step=1)

    # ── encode ────────────────────────────────────────────────────────────
    encoder_outputs, hrm_aux = self.encode(
        encoder_input_tokens,
        encoder_segment_ids=encoder_segment_ids,
        enable_dropout=enable_dropout,
        recording_ids=recording_ids,    # ← ADD
    )
    if hrm_aux is not None:
        res_dict['hrm_aux'] = hrm_aux   # ← carry through for loss

    # ── decode (UNCHANGED) ────────────────────────────────────────────────
    decoder_output_dict = self.decode(
        encoder_outputs, encoder_outputs,
        decoder_input_tokens, decoder_target_tokens,
        encoder_segment_ids=encoder_segment_ids,
        decoder_segment_ids=decoder_segment_ids,
        decoder_positions=decoder_positions,
        enable_dropout=enable_dropout,
        decode=decode,
        decoder_targets_frame_index=decoder_targets_frame_index,
        encoder_decoder_mask=encoder_decoder_mask
    )
    res_dict.update(decoder_output_dict)
    return res_dict
```

Also update `generate()` — it calls `self.encode()`. Change its call to:
```python
encoded, _ = self.encode(encoder_inputs, enable_dropout=False)
```
(The `_` discards hrm_aux since inference runs deterministic and returns None.)

---

## Step 3 — Modify `train.py`

Three targeted changes.

### 3a — Add the Q-loss function (module-level, before the class)

Place this after the imports, before `class MT3Trainer`:

```python
def compute_hrm_q_loss(hrm_aux: dict,
                       decoder_loss_per_example: torch.Tensor) -> torch.Tensor:
    """
    Q-learning loss for HRM encoder halt/continue heads.

    halt target:    is this example "easy" (below-median decoder loss)?
    continue target: bootstrap from next step's best Q value.

    Args:
        hrm_aux: dict from HRMEncoder with 'q_halt_list' and 'q_cont_pairs'
        decoder_loss_per_example: [B] per-example cross-entropy (detached)

    Returns:
        scalar Q-loss
    """
    # "is_easy" replaces HRM's binary seq_is_correct.
    # Below-median loss → easy → encoder should halt.
    median_loss = decoder_loss_per_example.median().detach()
    is_easy = (decoder_loss_per_example.detach() < median_loss).float()

    halt_loss = torch.tensor(0.0, device=is_easy.device)
    cont_loss = torch.tensor(0.0, device=is_easy.device)
    n = len(hrm_aux['q_halt_list'])

    for q_halt in hrm_aux['q_halt_list']:
        halt_loss += F.binary_cross_entropy_with_logits(
            q_halt, is_easy, reduction='mean'
        )

    for q_cont, bootstrap in hrm_aux['q_cont_pairs']:
        if bootstrap is not None:
            cont_loss += F.binary_cross_entropy_with_logits(
                q_cont, bootstrap, reduction='mean'
            )

    return (halt_loss + cont_loss) / max(n, 1)
```

### 3b — Pass `recording_ids` into `forward()` inside `forward_step()`

In `MT3Trainer.forward_step()`, the line that calls `self.forward(...)`:

```python
# BEFORE:
outputs_dict = self.forward(
    encoder_input_tokens=inputs,
    decoder_target_tokens=decoder_target_tokens,
    decode=False,
    decoder_input_tokens=decoder_inputs,
    decoder_targets_frame_index=decoder_targets_frame_index,
    encoder_decoder_mask=encoder_decoder_mask
)

# AFTER — add recording_ids:
outputs_dict = self.forward(
    encoder_input_tokens=inputs,
    decoder_target_tokens=decoder_target_tokens,
    decode=False,
    decoder_input_tokens=decoder_inputs,
    decoder_targets_frame_index=decoder_targets_frame_index,
    encoder_decoder_mask=encoder_decoder_mask,
    recording_ids=batch.get('audio_ids_contiguous', None),  # see Step 5
)
```

Also update `MT3Trainer.forward()`:
```python
def forward(self, encoder_input_tokens, decoder_target_tokens, decode,
            decoder_input_tokens=None, decoder_positions=None,
            decoder_targets_frame_index=None, encoder_decoder_mask=None,
            recording_ids=None):  # ← ADD
    return self.model.forward(
        encoder_input_tokens, decoder_target_tokens,
        decode=decode,
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        decoder_targets_frame_index=decoder_targets_frame_index,
        encoder_decoder_mask=encoder_decoder_mask,
        recording_ids=recording_ids,   # ← ADD
    )
```

### 3c — Add Q-loss computation in `forward_step()` after the existing loss block

Find the block that ends with `loss_dict["loss"] = total_loss`. After it, add:

```python
# HRM Q-loss (only when using HRMEncoder)
if 'hrm_aux' in outputs_dict and outputs_dict['hrm_aux'] is not None:
    # Compute per-example decoder loss as the Q-learning reward signal.
    # Using the cross-entropy loss over non-PAD tokens as a proxy for
    # "how hard was this example for the decoder".
    with torch.no_grad():
        decoder_outputs_for_q = outputs_dict['decoder_outputs'].detach()  # [B, T, vocab]
        targets_for_q = targets.clone()
        targets_for_q[targets_mask == 0] = TOKEN_PAD
        # [B]
        per_example_loss = Functional.cross_entropy(
            decoder_outputs_for_q.transpose(1, 2).float(),
            targets_for_q,
            ignore_index=TOKEN_PAD,
            reduction='none'
        ).mean(dim=1)

    q_loss = compute_hrm_q_loss(outputs_dict['hrm_aux'], per_example_loss)

    # Start with weight 0.05 to avoid disrupting the main CE loss early.
    # Increase to 0.1-0.2 after ~20k steps once the encoder is stable.
    q_loss_weight = getattr(self.config.training, 'hrm_q_loss_weight', 0.05)
    loss_dict['hrm_q_loss'] = q_loss
    loss_dict['loss'] = loss_dict['loss'] + q_loss_weight * q_loss
```

---

## Step 4 — Handle `audio_ids` Contiguous Mapping

Your `audio_ids` in the batch are **raw CSV row indices** from the MAESTRO CSV (e.g., 0, 3, 7, 11...). These are not contiguous, so you cannot use them directly as embedding indices. You need a `{raw_idx → 0..N-1}` mapping.

### 4a — Build the mapping in `Audio2Midi_Dataset.__init__`

In `data/dataset_Audio2Midi.py`, inside `Audio2Midi_Dataset.__init__()`, add after `subset_data` is determined:

```python
# Build contiguous ID map for per-recording embedding
# raw audio_idx (CSV row index) → contiguous integer 0..N-1
self.audio_id_to_contiguous = {
    int(idx): i for i, (idx, _) in enumerate(subset_data)
}
self.num_recordings = len(subset_data)
```

### 4b — Emit contiguous id from `SingleWavDataset.__getitem__`

At the bottom of `SingleWavDataset.__getitem__()`, the `row.update({...})` block already has:
```python
"audio_ids": torch.tensor(int(self.audio_idx), dtype=torch.long),
```

You need the parent dataset's mapping available in `SingleWavDataset`. The simplest approach — pass the map at construction time:

In `Audio2Midi_Dataset.__init__()`, where it constructs `SingleWavDataset`:
```python
self.dataset_list = [
    SingleWavDataset(
        config, dataset_dir, dataset_index, i, path, audio_h5_path,
        random_clip=random_clip,
        audio_id_contiguous=self.audio_id_to_contiguous.get(int(i), 0)
    )
    for i, path in tqdm(zip(idx_list, mid_paths), total=len(mid_paths))
]
```

In `SingleWavDataset.__init__()`, add parameter and store it:
```python
def __init__(self, config, dataset_dir, dataset_index, audio_idx, midi_path,
             audio_h5_path, random_clip=True,
             audio_id_contiguous: int = 0):   # ← ADD
    ...
    self.audio_id_contiguous = audio_id_contiguous
```

In `SingleWavDataset.__getitem__()`, add to `row.update(...)`:
```python
"audio_ids_contiguous": torch.tensor(self.audio_id_contiguous, dtype=torch.long),
```

### 4c — Pass `num_recordings` to config

The total number of unique recordings across **train+validation+test** must be set in the config as `num_recordings`. For MAESTRO v3, this is 1276 total recordings. Set this in your experiment config file (see Step 5).

---

## Step 5 — Create `config/experiment_T5_V5_HRM.yaml`

```yaml
# config/experiment_T5_V5_HRM.yaml
defaults:
  - experiment_T5_V4_HierarchyPool   # inherit everything from V4

model:
  # ── Replace TransformerEncoder with HRMEncoder ──────────────────────────
  use_hrm_encoder: true

  # Two-level hierarchy: L (local, grounded) + H (global, abstract)
  hrm_L_layers: 3
  hrm_H_layers: 3

  # Inner cycles per ACT step (truncated BPTT window = 1)
  hrm_L_cycles: 2
  hrm_H_cycles: 2

  # ACT refinement depth. Start conservative — increase after validation.
  hrm_max_steps: 4          # Phase 1-2: 4 steps; Phase 3+: increase to 8

  # L-level local attention window — matches your V1 encoder_window_size: 64
  hrm_L_window_size: 64     # H-level always uses global attention

  # Halting exploration probability during training
  hrm_halt_explore_prob: 0.1

  # Per-recording embeddings.
  # MAESTRO v3 total: 1276 recordings across all splits.
  # Set to 0 to disable during initial debugging (Phase 1).
  num_recordings: 1276
  recording_emb_dim: 64     # small — acts as a bias offset, not full 512-dim

training:
  notes: "V5_HRM_encoder"
  hrm_q_loss_weight: 0.05   # increase to 0.1 after ~20k steps

  # Keep other training settings from V4 (batch, lr, etc.)
```

Update `config/main_config.yaml`:
```yaml
defaults:
  - experiment_T5_V5_HRM
```

---

## Step 6 — Update `configure_optimizers` in `train.py`

The recording embedding benefits from a **higher learning rate** than the rest of the model (analogous to HRM's 100× higher puzzle_emb_lr). The sparse embedding only updates rows seen in the current batch.

```python
def configure_optimizers(self):
    # Separate recording embedding parameters for higher LR
    rec_emb_params = []
    main_params = []
    for name, param in self.model.named_parameters():
        if 'recording_embed' in name or 'recording_proj' in name:
            rec_emb_params.append(param)
        else:
            main_params.append(param)

    optimizer_groups = [
        {'params': main_params,
         'lr': self.config.training.learning_rate},    # e.g. 1e-4
        {'params': rec_emb_params,
         'lr': self.config.training.learning_rate * 100,  # e.g. 1e-2
         'weight_decay': 0.01},
    ]
    optimizer = AdamW(optimizer_groups)
    return optimizer
```

The standard `AdamW` handles this cleanly — no custom sparse optimizer is needed since your batch sizes ensure all `rec_emb_params` rows receive gradients regularly during training.

---

## Rollout Sequence

Do not enable everything at once. Each phase validates the previous one.

| Phase | What is active | Validation signal |
|---|---|---|
| 1 | `use_hrm_encoder: true`, `hrm_max_steps: 1`, `num_recordings: 0`, Q-loss weight = 0 | Verify training loss matches V4 baseline — sanity check the wiring |
| 2 | `hrm_max_steps: 4`, Q-loss weight = 0 | Does validation `note_f1` improve over V4 at same steps? |
| 3 | Enable Q-loss at weight 0.05 | Does `hrm_q_loss` decrease? Does halting correlate with decoder confidence? |
| 4 | `num_recordings: 1276`, recording LR = 1e-2 | Does per-recording adaptation improve F1 on hard recordings? |
| 5 | `hrm_max_steps: 8` | Diminishing returns check — is step 8 better than step 4? |

Phase 1 is the critical gate: if training is unstable or slower without benefit, debug the wiring before adding complexity.

---

## The Call Graph After Integration

```
MT3Trainer.training_step(batch)
  └── forward_step(batch)
        ├── features_extracter(audio) → inputs [B, T, 512]
        ├── self.forward(inputs, targets, ...,
        │               recording_ids=batch['audio_ids_contiguous'])
        │     └── Transformer.forward()
        │           ├── encode(inputs, recording_ids=ids)
        │           │     └── HRMEncoder.forward()
        │           │           ├── dense(inputs) + embed(positions) + recording_embed(ids)
        │           │           │   → audio_features [B, T, 512]  — FIXED
        │           │           ├── z_H, z_L = H_init, L_init  broadcast to [B, T, 512]
        │           │           ├── for step in range(max_steps):
        │           │           │     ├── [no_grad] H_cycles-1 × L_cycles-1 warm-up passes
        │           │           │     │   z_L = L_level(z_L, z_H + audio_features)
        │           │           │     │   z_H = H_level(z_H, z_L)
        │           │           │     ├── [with_grad] final L pass → z_L (gradient here)
        │           │           │     ├── [with_grad] final H pass → z_H (gradient here)
        │           │           │     └── q_head(z_H[:, 0]) → q_halt [B], q_cont [B]
        │           │           └── returns (layer_norm(z_H), hrm_aux)
        │           └── decode(z_H, ...) → decoder_outputs  [UNCHANGED]
        ├── CrossEntropyLoss(decoder_outputs, targets) → ce_loss [B]
        ├── compute_hrm_q_loss(hrm_aux, ce_loss_per_example) → q_loss
        └── total_loss = ce_loss + 0.05 * q_loss
```

---

## What Each Component Does for Piano Transcription

**L-level (local window = 64 frames = 640ms):** Stays perpetually grounded in the raw CQT features via the injection `z_L = L(z_L, z_H + audio_features)`. Captures fine-grained onset transients, pitch-specific energy, local harmonic intervals. The local window keeps compute tractable (flash-attn, same as your V1 config).

**H-level (global attention, no window):** Reads L's refined summary. Attends across the full 512-frame clip. Captures phrase-level dynamics, key context, recurring rhythmic patterns, and global intensity curves. Informs L on the next step about what "should" be present given the broader context.

**Recurrent carry:** After step 1, z_H and z_L encode a first-pass estimate. On step 2, L refines its detail view using H's updated global context. On step 3, H updates its global model using L's improved details. By step 4, the encoder has iterated this dialogue four times — equivalent in representational depth to a much deeper fixed encoder but without the parameter cost.

**Q-head:** Learns to predict whether this audio clip's decoder output is above or below the batch median difficulty. Easy clips (sparse notes, clear onsets) should halt early. Dense, pedal-heavy passages should use more steps. This is the adaptive compute mechanism.

**Recording embedding:** Learns a small bias vector (64-dim, zero-initialized) per recording. After training it captures recording-specific characteristics: instrument brightness, microphone response, reverb character. Because it is zero-initialized, the model starts from the non-adapted baseline and only diverges when data is consistent within a recording.

---

*All class names, method signatures, config keys, and batch field names in this guide are drawn directly from your codebase as uploaded.*
