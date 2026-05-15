# Efficient Transformer With Pedal Heads: Architecture And Training Breakdown

This document explains the model in `Experiment/Efficient-Transformer-with-pedals`,
with special attention to the three auxiliary pedal heads added on top of the
encoder.

The short version:

```text
audio clip
  -> mel frames
  -> 6-layer Transformer encoder
      -> decoder cross-attention -> autoregressive MIDI-like event tokens -> CE loss
      -> pedal frame head         -> pedal-down frame logits             -> BCE loss
      -> pedal onset head         -> PedalOn boundary frame logits       -> BCE or focal loss
      -> pedal offset head        -> PedalOff boundary frame logits      -> BCE or focal loss
```

The decoder and the three auxiliary pedal heads use the same encoder input stream:
the same mel-frame sequence from the same audio clip. They do not use the exact
same target tensor. The decoder trains against a token sequence derived from the
MIDI/TSV events; the pedal heads train against three frame-level vectors derived
from the pedal events in the same MIDI/TSV.

## 1. Where The Relevant Pieces Live

Main files:

| File | Role |
| --- | --- |
| `model/T5.py` | Top-level encoder-decoder model and the three auxiliary pedal heads. |
| `model/Encoder.py` | Audio-frame Transformer encoder. |
| `model/Decoder.py` | Autoregressive Transformer decoder and cross-attention logic. |
| `data/dataset_Audio2Midi.py` | Builds audio clips, decoder token targets, cross-attention masks, and pedal-frame targets. |
| `data/symbolic_music_tokenizer.py` | Converts MIDI/TSV note and pedal events to event tokens and back. |
| `loss/CrossEntropyLoss.py` | Decoder token cross entropy. |
| `loss/PedalFrameBCELoss.py` | BCE-with-logits loss for pedal frame-style heads. |
| `loss/PedalBoundaryFocalLoss.py` | Optional quality focal loss for sparse soft pedal onset/offset boundary heads. |
| `train.py` | Lightning training loop, loss aggregation, validation, and test-time generation. |
| `metrics/transcription_metrics.py` | Token, event, pedal-frame, and frame-head-to-event metrics. |

Key source locations:

- The auxiliary heads are defined in `model/T5.py`, lines 55-59.
- The encoder returns a final sequence tensor in `model/Encoder.py`, lines 75-91.
- The pedal heads are applied to that final encoder tensor in `model/T5.py`,
  lines 219-224.
- The total loss is assembled from config-defined criteria in `train.py`,
  lines 350-366.
- The active loss list is in `config/base_config.yaml`, lines 69-73.
- The decoder and pedal target tensors are built in `data/dataset_Audio2Midi.py`,
  lines 383-509.

## 2. Active Model Shape

The experiment config chain is:

```text
main_config
  -> experiment_T5_V4_HierarchyPool
  -> experiment_T5_V3_HybridGlobalLocalCrossAttn
  -> experiment_T5_V2_LocalEncDecAttn
  -> experiment_T5_V1_LocalEncAttn
  -> experiment_T5_V0
  -> base_config + model/T5 + data/data_config + training/training_config
```

The important settings are:

| Setting | Value / meaning |
| --- | --- |
| Model family | T5/MT3-style Transformer encoder-decoder. |
| Encoder | 6 Transformer encoder layers. |
| Decoder | 6 Transformer decoder layers. |
| Embedding width | `emb_dim = 512`. |
| Attention | 8 heads, `head_dim = 64`. |
| Feed-forward | `mlp_dim = 1024`, ReLU in the listed checkpoint configs. |
| Input feature | Mel spectrogram. |
| Main V4 frame count | `n_frames = 512` from `experiment_T5_V0.yaml`. |
| Hop | `hop_length = 320` at `sample_rate = 16000`, so one encoder frame is 20 ms. |
| Token length | `max_token_length = 1353` in the V4 chain. |
| Token timing grid | `ONSET_SEC_UP_SAMPLING = 20`, so event tokens quantize time to 50 ms. |
| Local encoder attention | `encoder_window_size = 64`. |
| Local decoder self-attention | `decoder_window_size = 64`. |
| Local encoder-decoder mask | `encoder_decoder_slide_window_size = 64`. |
| Hierarchical cross-attention pooling | Decoder layers use pooled encoder sequences with pooling sizes `[4, 4, 2, 2, 1, 1]`. |
| Pedal tokens in decoder target | `emit_pedal_tokens = true` in the pedal-aware configs. |
| Note-off target semantics | `use_truth_offsets = true` in the pedalheads evaluation config, meaning note-off tokens use physical key release rather than pedal-extended offsets. |

The model's `config/model/T5.yaml` sets `encoder_input_dim = 512`, `emb_dim = 512`,
`num_encoder_layers = 6`, `num_decoder_layers = 6`, and `vocab_size = 1024`.
The tokenizer's valid tokens occupy a smaller range inside that output vocabulary,
but the decoder output projection still produces `1024` logits per token position.

## 3. Input Pipeline

Each dataset item is a fixed-length audio chunk:

```text
raw waveform samples [n_frames * hop_length]
  -> MelSpectrogram
  -> transpose
  -> encoder input tensor [B, T, F]
```

For the V4 experiment, this is normally:

```text
B = batch size
T = 512 frames
F = 512 mel bins
```

In `train.py`, the feature extractor is run under `torch.no_grad()`:

```python
inputs = batch["inputs"]
inputs = self.features_extracter(inputs[:, :-1]).transpose(-1, -2)
inputs = inputs.detach()
```

That means the mel frontend is not learned. Training starts, for gradient
purposes, at the Transformer encoder. If SpecAugment is enabled, it is applied to
the mel features after extraction. In the V4 hierarchy-pooling config, data
augmentation is disabled.

## 4. Encoder Architecture

The encoder receives a sequence of mel frames:

```text
X: [B, T, F]
```

Then:

1. A dense layer projects each frame from `F` to `emb_dim`.
2. Fixed positional embeddings are added.
3. Six Transformer encoder layers run over the frame sequence.
4. A final layer norm and dropout are applied.
5. The encoder returns a sequence, not a single vector:

```text
Z: [B, T, 512]
```

This `Z` is the "encoder final output state." It is better to think of it as
one contextual embedding per audio frame:

```text
Z[b, t] = final 512-dimensional representation of audio frame t in example b
```

The encoder does not return all intermediate layer outputs. Conceptually:

```text
H0 = dense(mel) + position
Z1 = encoder_layer_1(H0)
Z2 = encoder_layer_2(Z1)
Z3 = encoder_layer_3(Z2)
Z4 = encoder_layer_4(Z3)
Z5 = encoder_layer_5(Z4)
Z6 = encoder_layer_6(Z5)
Z  = layer_norm(Z6)
```

The rest of the model receives `Z`, the final encoder sequence. `Z1` through
`Z6` are not identical; they are successive transformations. The auxiliary heads
are not attached separately to `Z1`, `Z2`, ..., `Z6`. They all read the same final
`Z`.

## 5. Decoder Architecture

The decoder is autoregressive and token-based.

During training it uses teacher forcing:

```text
decoder_input_tokens = shift_right(decoder_target_tokens)
```

At each decoder position it receives:

- previous target tokens, embedded into 512-dimensional vectors;
- positional embeddings;
- causal self-attention over previous decoder positions;
- cross-attention into the encoder output sequence `Z`;
- a final vocabulary projection that produces logits.

The decoder output is:

```text
decoder_outputs: [B, L, vocab_size]
```

where `L` is the padded decoder target length.

The decoder has three token-position output projections when
`hybrid_global_local_cross_attn` is enabled:

| Token position | Expected token family | Projection |
| --- | --- | --- |
| `i % 3 == 0` | Onset token or EOS | `dense` |
| `i % 3 == 1` | Pitch, NoteOff, PedalOn, or PedalOff | `dense_pitch` |
| `i % 3 == 2` | Velocity or BLANK | `dense_velocity` |

These are not the three auxiliary pedal heads. They are decoder token-family
heads. PedalOn and PedalOff are emitted by the decoder at the `i % 3 == 1`
positions, alongside pitch and note-off tokens.

The decoder's efficient cross-attention behavior comes from two mechanisms:

- For training, `encoder_decoder_mask` lets onset-token positions attend broadly,
  while the following event-detail positions attend to a local 64-frame window
  around the event's frame.
- With hierarchy pooling, decoder layers attend to pooled versions of the final
  encoder sequence: the first two decoder layers see frames pooled by 4, the next
  two by 2, and the last two see the unpooled final encoder sequence.

So the decoder does not get intermediate encoder layer outputs either. It gets
the same final encoder representation, sometimes averaged over neighboring
frames for particular decoder layers.

## 6. What The Decoder Predicts

The symbolic target is a flat token stream made of 3-token event records:

```text
[Onset, event-kind, value-or-BLANK]
```

Examples:

```text
Note on:   [Onset, Pitch,    Velocity]
Note off:  [Onset, NoteOff,  BLANK]
Pedal on:  [Onset, PedalOn,  BLANK]
Pedal off: [Onset, PedalOff, BLANK]
EOS:       [EOS,   PAD,      PAD]
```

The decoder is trained to reproduce this entire sequence. It is not a piano-roll
or frame-regression decoder. Final MIDI-like notes and pedals are obtained by
detokenizing the generated token stream.

## 7. The Three Auxiliary Pedal Heads

The model adds three independent dense heads on top of the encoder output:

```python
self.pedal_frame_head = nn.Linear(config.emb_dim, 1)
self.pedal_onset_head = nn.Linear(config.emb_dim, 1)
self.pedal_offset_head = nn.Linear(config.emb_dim, 1)
```

In the forward pass:

```python
encoder_outputs = self.encode(...)

pedal_frame_logits  = pedal_frame_head(encoder_outputs).squeeze(-1)
pedal_onset_logits  = pedal_onset_head(encoder_outputs).squeeze(-1)
pedal_offset_logits = pedal_offset_head(encoder_outputs).squeeze(-1)
```

Each head receives the same tensor:

```text
Z: [B, T, 512]
```

Each head outputs one scalar logit per encoder frame:

```text
pedal_frame_logits:  [B, T]
pedal_onset_logits:  [B, T]
pedal_offset_logits: [B, T]
```

They are simple linear probes:

```text
pedal_frame_logits[b, t]  = dot(W_state,  Z[b, t]) + b_state
pedal_onset_logits[b, t]  = dot(W_onset,  Z[b, t]) + b_onset
pedal_offset_logits[b, t] = dot(W_offset, Z[b, t]) + b_offset
```

The heads do not see the decoder tokens, decoder hidden states, generated MIDI,
or the target MIDI directly. They only see the encoder's final per-frame
representations.

## 8. Are The Heads Getting Copies Of `Z6`?

Mostly yes, if by `Z6` you mean the final encoder sequence after the sixth
encoder layer and final normalization.

More precise answer:

- `Z1` through `Z6` are not identical.
- The code does not keep and expose all six intermediate tensors.
- The encoder returns only the final `Z`.
- All three pedal heads receive that same final `Z`.
- The decoder also receives that same final `Z`.
- With hierarchy pooling, decoder layers may receive pooled versions of `Z`, but
  those pooled tensors are still computed from the final encoder output, not from
  earlier encoder layers.

From a backpropagation perspective, it is as if several branches read the same
shared tensor:

```text
                         -> decoder -> CE
mel -> encoder -> Z ----> pedal_frame_head  -> BCE_state
                         -> pedal_onset_head -> BCE_onset
                         -> pedal_offset_head -> BCE_offset
```

The gradients from those branches add together at `Z`.

## 9. Where The Decoder Cross-Entropy Targets Come From

The decoder CE targets are built in `data/dataset_Audio2Midi.py`.

For each audio/MIDI item:

1. The dataset locates the cached MIDI path and corresponding `audio.h5` entry.
2. It reads a cached `*.midi-notes.tsv` next to the MIDI:

   ```python
   tsv_path = os.path.splitext(self.midi_path)[0] + ".midi-notes.tsv"
   source_dataframe_midi = pd.read_csv(tsv_path, sep="\t")
   ```

3. The TSV dataframe is converted into MIDI-like events:

   ```python
   dataframe_midi = sm_tokenizer.notes_to_midi_events(
       source_dataframe_midi,
       use_truth_offsets=...,
       emit_pedal_tokens=...,
   )
   ```

4. `notes_to_midi_events` creates:

   - one `NoteOn` event per note onset;
   - one `NoteOff` event per note offset;
   - `PedalOn` and `PedalOff` events if `emit_pedal_tokens = true`.

5. For a given training clip, the dataset filters those events to the clip time:

   ```python
   df_midi = dataframe_midi[
       (onset_sec >= begin_sec) & (onset_sec <= end_sec)
   ]
   ```

6. Event times are shifted so the clip starts at `0.0`.
7. The tokenizer converts the dataframe into the flat token stream.
8. EOS is appended.
9. The result is padded to `max_token_length`.
10. `decoder_targets_mask` is `1` for non-PAD tokens and `0` for PAD tokens.

So the decoder target is:

```text
decoder_targets:      [L] token ids
decoder_targets_mask: [L] 1 for real target tokens, 0 for PAD
```

When batching:

```text
decoder_targets:      [B, L]
decoder_targets_mask: [B, L]
```

The cross entropy loss masks padded positions by replacing them with `TOKEN_PAD`
and uses `ignore_index=TOKEN_PAD`.

Important detail: when `use_truth_offsets = true`, `NoteOff` tokens are generated
from `offset_sec_truth`, the physical key-release time. Pedal targets are not
changed by this setting.

## 10. Where The Three BCE Targets Come From

The three BCE targets are built from the pedal rows in the same MIDI/TSV source.

When the dataset loads the cache, it separately extracts pedal rows:

```python
self.dataframe_pedal_events = self._get_pedal_rows(source_dataframe_midi)
self._prepare_pedal_cache()
```

`_get_pedal_rows` keeps rows whose type is:

```text
PedalOn
PedalOff
```

It also sorts same-time repedaling so `PedalOff` is processed before `PedalOn`.
That closes the old span before opening the next one.

For each clip, the dataset calls:

```python
pedal_frame_target, pedal_onset_target, pedal_offset_target =
    self._build_pedal_targets_fast(begin_sec, end_sec, second_per_frame)
```

The output tensors are:

```text
pedal_frame_target:  [T]
pedal_onset_target:  [T]
pedal_offset_target: [T]
```

with `T = n_frames`.

### 10.1 `pedal_frame_target`

This is the sustain pedal state at each encoder frame:

```text
0 = pedal up
1 = pedal down
```

The starting state at the beginning of the clip is determined by the most recent
pedal event before `begin_sec`. Then events inside the clip update the state.

If there are no pedal events inside the clip, the whole vector is the carried-in
state. That means a clip entirely inside a long pedal hold can still have all
ones even if no `PedalOn` event occurs in the clip.

### 10.2 `pedal_onset_target`

This is a soft boundary target around `PedalOn` events. It is mostly zero. By
default, around each PedalOn frame, the dataset writes a small triangular kernel:

```text
event frame:      1.00
one frame away:   0.50
two frames away:  0.25
```

At a 20 ms frame hop, this supervises roughly +/- 40 ms around the boundary.
The kernel can also be configured as a Gaussian for boundary-head experiments,
for example radius 5 and sigma 3 frames.

### 10.3 `pedal_offset_target`

This is the same configurable soft boundary idea, but around `PedalOff` events.
The default triangular kernel is:

```text
event frame:      1.00
one frame away:   0.50
two frames away:  0.25
```

### 10.4 Masks

The dataset sets:

```python
pedal_target_mask = torch.ones(self.n_frames, dtype=torch.long)
```

and emits separate masks:

```text
pedal_frame_target_mask
pedal_onset_target_mask
pedal_offset_target_mask
```

During test-time metrics, final chunks are trimmed to the valid number of frames
so padding beyond the real audio is not counted.

## 11. The Four Training Losses

The baseline loss config is:

```yaml
losses:
  loss_decoder:      [decoder_outputs, decoder_targets, loss.CrossEntropyLoss]
  loss_pedal_state:  [pedal_frame_logits,  pedal_frame_target,  loss.PedalFrameBCELoss, 0.3]
  loss_pedal_onset:  [pedal_onset_logits,  pedal_onset_target,  loss.PedalFrameBCELoss, 0.3]
  loss_pedal_offset: [pedal_offset_logits, pedal_offset_target, loss.PedalFrameBCELoss, 0.3]
```

The total loss is:

```text
total_loss =
    CE(decoder_outputs, decoder_targets)
  + 0.3 * BCEWithLogits(pedal_frame_logits,  pedal_frame_target)
  + 0.3 * BCEWithLogits(pedal_onset_logits,  pedal_onset_target)
  + 0.3 * BCEWithLogits(pedal_offset_logits, pedal_offset_target)
```

The decoder CE is a multiclass loss over token IDs:

```text
decoder_outputs[b, i, :] -> target token decoder_targets[b, i]
```

The pedal BCE losses are independent binary losses over frames:

```text
pedal_head_logits[b, t] -> target value in [0, 1]
```

`BCEWithLogitsLoss` accepts the soft onset/offset targets directly, so the 0.5
and 0.25 neighboring labels are meaningful training targets, not just hard labels.

Gaussian-boundary experiments may keep BCE for all three heads, or use quality
focal loss only for the sparse onset/offset heads:

```yaml
loss_pedal_state:  [pedal_frame_logits,  pedal_frame_target,  loss.PedalFrameBCELoss, 0.3]
loss_pedal_onset:  [pedal_onset_logits,  pedal_onset_target,  loss.PedalBoundaryFocalLoss, 0.3]
loss_pedal_offset: [pedal_offset_logits, pedal_offset_target, loss.PedalBoundaryFocalLoss, 0.3]
```

The state head remains BCE because the positive/negative balance is much less
extreme than for boundary frames.

## 12. Are The Decoder And Pedal Heads Training Against The Same Targets?

They train against different target tensors derived from the same source
annotation.

Same source:

```text
cached MIDI / *.midi-notes.tsv
```

Different derived targets:

```text
decoder CE target:
  tokenized note and pedal event sequence

pedal state BCE target:
  frame-wise sustain-pedal down/up state

pedal onset BCE target:
  frame-wise soft PedalOn boundary map

pedal offset BCE target:
  frame-wise soft PedalOff boundary map
```

In active pedal-aware configs, both the decoder target and the BCE targets include
pedal information:

- The decoder sees PedalOn/PedalOff as event tokens if `emit_pedal_tokens=true`.
- The auxiliary heads see the same underlying pedal events as frame supervision.

But these are not the same representation. A single long pedal span might be:

```text
decoder target:
  [Onset, PedalOn, BLANK] ... [Onset, PedalOff, BLANK]

pedal_frame_target:
  0 0 0 1 1 1 1 1 1 ... 1 1 0 0 0

pedal_onset_target:
  0 0 .25 .5 1 .5 .25 0 ...

pedal_offset_target:
  ... 0 .25 .5 1 .5 .25 0
```

So yes, they come from the same MIDI performance; no, they are not literally the
same target sequence.

One useful edge case: if `emit_pedal_tokens=false`, the decoder token target can
omit pedal tokens, while the auxiliary frame targets can still be built from the
raw pedal rows. In that setup the heads can learn pedals, but the decoder would
not be trained to emit PedalOn/PedalOff tokens unless some other objective or
evaluation path uses the frame head.

## 13. Why Auxiliary Pedal Heads Can Improve Decoder Pedal Tokens

The heads are not fed into the decoder. There is no line like:

```text
decoder_input = concat(decoder_input, pedal_head_output)
```

So the improvement is not a direct inference-time connection. The improvement is
from shared representation learning.

During backpropagation, the encoder receives gradients from both branches:

```text
dL/dZ =
    dL_decoder_CE/dZ
  + 0.3 * dL_pedal_state/dZ
  + 0.3 * dL_pedal_onset/dZ
  + 0.3 * dL_pedal_offset/dZ
```

Those gradients update the same encoder parameters. The decoder then cross-attends
to the improved encoder states.

This helps for several reasons:

1. The decoder pedal-token objective is sparse.

   PedalOn and PedalOff tokens occur only at event boundaries. Many decoder
   positions are note events, velocities, blanks, EOS, or padding. The frame
   state loss gives the encoder dense pedal supervision at every audio frame.

2. The auxiliary targets tell the encoder exactly what pedal evidence should be
   linearly recoverable.

   A single `Linear(512, 1)` head can only succeed if `Z[b, t]` contains pedal
   state or boundary information in an accessible form. That pressures the
   encoder to preserve pedal cues in the hidden sequence.

3. The decoder cross-attends to `Z`.

   When the decoder is trying to emit a PedalOn/PedalOff token, its cross-attention
   reads the same final encoder sequence supervised by the pedal heads. Better
   pedal information in `Z` gives the decoder cleaner evidence for pedal tokens.

4. It regularizes timing.

   The onset and offset heads supervise boundary localization on the 20 ms encoder
   frame grid. The decoder token grid is coarser, 50 ms. Even though the final
   decoder tokens still quantize to the token grid, the encoder can learn sharper
   temporal cues before the decoder turns them into events.

5. It reduces ambiguity between note sustain and pedal sustain.

   Piano audio can contain lingering energy from key hold, room/reverb, and
   sustain pedal. Dense pedal-state supervision encourages the encoder to separate
   "pedal is down" from "sound is still ringing," which can improve both pedal
   event tokens and pedal-extended note metrics.

This is ordinary multi-task learning: an auxiliary task changes the shared
features used by the main task.

## 14. What The Heads Do And Do Not Do At Inference

There are two distinct inference/evaluation paths:

### Decoder-token path

The usual seq2seq output is:

```text
mel -> encoder -> decoder.generate(...) -> output_tokens -> detokenize -> notes + pedal events
```

In this path, auxiliary pedal logits are not used to create the decoder tokens.
They only helped shape the encoder during training.

### Frame-head path

The test code can also compute:

```text
mel -> encoder -> pedal_frame_head -> sigmoid probabilities
```

Then `metrics/transcription_metrics.py` can convert frame-head probabilities into
PedalOn/PedalOff events. The default extractor uses pedal-state hysteresis. An
optional extractor verifies PedalOn events with an upward state trend and can end
PedalOff events from either the offset head or a state drop. This is controlled by:

```yaml
evaluation:
  pedal_event_source: decoder   # or frame_head
  midi_pedal_event_source: decoder   # or frame_head
  frame_head_event_extractor: state_hysteresis   # or trend_dual_trigger
```

The base config defaults to the decoder event source. The
`evaluate_maestro_pedalheads_200k.yaml` file currently sets:

```yaml
pedal_event_source: frame_head
midi_pedal_event_source: frame_head
```

So when reading metrics, check which event source was used. A decoder-token pedal
F1 improvement means the auxiliary heads improved the shared encoder representation
used by the decoder. A frame-head pedal F1 result may be measuring a different
output path altogether.

## 15. Pedal Frame Metrics Versus Decoder Pedal Metrics

The frame metrics compare per-frame head probabilities with per-frame targets:

```text
pedal_frame_output[t] > 0.5
vs
pedal_frame_target[t] > 0.5
```

For onset and offset frame metrics, the code thresholds the soft target at
`> 0.99`, so only the center `1.0` boundary frame is counted as positive for the
metric. This keeps the diagnostic comparable when Gaussian kernels make nearby
frames larger than `0.5`. Neighboring soft labels still influence training, but
they are not positive labels for the central boundary frame metric.

Decoder pedal event metrics compare detokenized PedalOn/PedalOff events or spans
against reference pedal events. They answer a different question:

```text
Did the final event stream put pedal changes at the right times?
```

Frame metrics answer:

```text
Was the auxiliary frame classifier correct at each time frame?
```

Those two can move together, but they are not the same metric.

## 16. End-To-End Training Flow

For one batch, training does this:

1. The dataloader returns raw audio plus several target tensors:

   ```text
   inputs
   decoder_targets
   decoder_targets_mask
   decoder_targets_frame_index
   encoder_decoder_mask
   pedal_frame_target
   pedal_frame_target_mask
   pedal_onset_target
   pedal_onset_target_mask
   pedal_offset_target
   pedal_offset_target_mask
   ```

2. `train.py` converts raw audio to mel features:

   ```text
   inputs -> [B, T, F]
   ```

3. `model.forward` shifts decoder targets right for teacher forcing.

4. The encoder produces:

   ```text
   Z = encoder(inputs)  # [B, T, 512]
   ```

5. The pedal heads produce:

   ```text
   pedal_frame_logits  # [B, T]
   pedal_onset_logits  # [B, T]
   pedal_offset_logits # [B, T]
   ```

6. The decoder produces:

   ```text
   decoder_outputs     # [B, L, vocab_size]
   ```

7. The trainer loops over the configured losses.

8. The total weighted loss is backpropagated through the decoder, the pedal
   heads, and the shared encoder.

9. AdamW updates all trainable model parameters. The active config has
   `froze_encoder: false`, so the auxiliary losses can update the encoder.

## 17. Common Confusions Resolved

### "Are both components using the same input stream?"

Yes. The decoder and the three auxiliary heads both use the same mel sequence
from the same audio clip. The decoder consumes `Z` through cross-attention; the
pedal heads consume `Z` directly frame by frame.

### "Are they training against the same targets?"

They train against targets from the same MIDI/TSV source, but in different
representations:

- decoder: event-token sequence;
- heads: frame-state and boundary vectors.

### "Does the auxiliary head output go into the decoder?"

No. The decoder does not consume `pedal_frame_logits`, `pedal_onset_logits`, or
`pedal_offset_logits`.

### "Then how can the heads improve token pedal F1?"

Because the heads change the shared encoder representation during training. The
decoder is trained by CE to emit pedal tokens, and it cross-attends to that same
encoder representation. Better pedal information in the encoder can make the
decoder better at emitting PedalOn/PedalOff tokens.

### "Are the three heads three layers of the encoder?"

No. They are three separate `Linear(512, 1)` modules attached after the encoder.
They all read the same final encoder output sequence.

### "Do the heads replace the decoder pedal tokens?"

Not during normal decoder-token generation. They are auxiliary training heads.
However, the evaluation code can optionally use the pedal-frame head as a separate
pedal event source by thresholding its frame probabilities.

### "Is the pedal BCE target extracted from the decoder target tokens?"

No. It is extracted from the source pedal rows in the same MIDI/TSV dataframe.
The two target families share the same source annotations, but the frame targets
are built independently of the tokenized decoder target.

## 18. Mental Model

Think of the encoder as learning a shared audio representation:

```text
Z[t] should contain:
  - note-onset evidence,
  - pitch evidence,
  - note-release evidence,
  - velocity/percussive evidence,
  - sustain-pedal state evidence,
  - sustain-pedal boundary evidence.
```

The decoder asks questions of that representation through cross-attention:

```text
"What event token should I emit next?"
```

The auxiliary heads ask simpler frame-wise questions:

```text
"At this frame, is the pedal down?"
"Is this frame near a pedal press?"
"Is this frame near a pedal release?"
```

Those simpler questions make the encoder more pedal-aware. The decoder can then
use that improved representation to produce better pedal events, even when the
heads are not directly used in the output pipeline.
