# Transcription Model and Frame Metrics

This note explains the transcription model in
[`Experiment/Efficient-Transformer-with-pedals`](Efficient-Transformer-with-pedals),
what the frame metrics in the evaluation CSV mean, and how this approach differs
from the reference implementation in
[`reference/piano_transcription`](../reference/piano_transcription).

## Short Answer

The Experiment model is an audio-to-symbolic-event Transformer. It takes mel
spectrogram frames from piano audio and autoregressively decodes a token sequence
representing note onsets, note offsets, velocities, and sustain-pedal on/off
events. The active evaluation config,
[`evaluate_maestro_pedalheads_200k.yaml`](Efficient-Transformer-with-pedals/config/evaluate_maestro_pedalheads_200k.yaml),
uses the `T5_V4_HierarchyPool` Transformer checkpoint trained for 200k steps with
extra pedal supervision.

The `pedal_frame_*` metrics in
[`!test_metrics_summary.csv`](Efficient-Transformer-with-pedals/evaluations/MAESTRO_PedalHeads_200k_update/!test_metrics_summary.csv)
measure the auxiliary encoder pedal-state head only: at each audio frame, did the
model think the sustain pedal was down? They do not directly measure the final
MIDI pedal events emitted by the decoder. The final pedal MIDI quality is measured
by `pedal_*`, `diagnostic_pedal_*`, and pedal-extended note metrics.

Compared with the reference `piano_transcription` system, the Experiment model is
not a CRNN frame-regression model. The reference predicts note/pedal frame rolls
plus regression targets for precise onsets and offsets, then post-processes those
frame outputs into MIDI. The Experiment model predicts MIDI-like event tokens
directly and uses frame heads only as auxiliary pedal supervision.

## Where The Model Is Defined

Main files:

| File | Role |
| --- | --- |
| [`train.py`](Efficient-Transformer-with-pedals/train.py) | Lightning module, losses, test-time generation, CSV metric aggregation. |
| [`model/T5.py`](Efficient-Transformer-with-pedals/model/T5.py) | Transformer wrapper and the auxiliary pedal heads. |
| [`model/Encoder.py`](Efficient-Transformer-with-pedals/model/Encoder.py) | Audio-frame Transformer encoder. |
| [`model/Decoder.py`](Efficient-Transformer-with-pedals/model/Decoder.py) | Autoregressive Transformer decoder and hybrid/global-local cross attention. |
| [`data/symbolic_music_tokenizer.py`](Efficient-Transformer-with-pedals/data/symbolic_music_tokenizer.py) | Converts notes and pedals to/from token sequences. |
| [`metrics/transcription_metrics.py`](Efficient-Transformer-with-pedals/metrics/transcription_metrics.py) | Note, pedal, pedal-frame, and pedal-extended metrics. |

The active evaluation config inherits:

```text
evaluate_maestro_pedalheads_200k
  -> experiment_T5_V4_HierarchyPool
  -> experiment_T5_V3_HybridGlobalLocalCrossAttn
  -> experiment_T5_V2_LocalEncDecAttn
  -> experiment_T5_V1_LocalEncAttn
  -> experiment_T5_V0
  -> base_config + model/T5 + data/data_config + training/training_config
```

Important active settings:

| Setting | Meaning |
| --- | --- |
| `model_name: Transformer-T5` | Encoder-decoder Transformer, MT3/T5-like. |
| `num_encoder_layers: 6`, `num_decoder_layers: 6` | Six encoder and six decoder layers. |
| `emb_dim: 512`, `num_heads: 8`, `head_dim: 64` | 512-dimensional model with 8 attention heads. |
| `mlp_dim: 1024`, `mlp_activations: relu` | Feed-forward block shape. |
| `n_frames: 512`, `hop_length: 320`, `sample_rate: 16000` | 512 frames per chunk, 20 ms per frame, about 10.24 s per chunk. |
| `features: mel`, `num_mel_bins: 512` | Mel features are fed to the encoder. |
| `encoder_window_size: 64` | Local encoder self-attention window. |
| `decoder_window_size: 64` | Local decoder self-attention window. |
| `encoder_decoder_slide_window_size: 64` | Local cross-attention mask for most token positions. |
| `hybrid_global_local_cross_attn: true` | Decoder positions are handled with token-role-aware cross attention. |
| `cross_attention_hierarchy_pooling: true` | Decoder layers attend to pooled encoder representations with pooling sizes `[4, 4, 2, 2, 1, 1]`. |
| `use_truth_offsets: true` | Note-off tokens are trained/evaluated against raw key-release offsets, not pedal-extended offsets. |
| `emit_pedal_tokens: true` | The target sequence includes `PedalOn` and `PedalOff` tokens. |

## What The Model Predicts

The model predicts a flat token stream. In the tokenizer, each musical event is
represented as a 3-token compound word:

```text
[Onset, Pitch or NoteOff or PedalOn or PedalOff, Velocity or BLANK]
```

Examples:

```text
Note on:   [Onset, Pitch,    Velocity]
Note off:  [Onset, NoteOff,  BLANK]
Pedal on:  [Onset, PedalOn,  BLANK]
Pedal off: [Onset, PedalOff, BLANK]
```

The decoder therefore emits symbolic events rather than a piano-roll matrix. The
tokenizer detokenizes those events back into note and pedal lists, then optionally
saves them as MIDI.

Timing caveat: event onset tokens are quantized by
`ONSET_SEC_UP_SAMPLING = 20`, so decoded event times are on a 50 ms grid inside
each chunk. The encoder frame grid is finer, 20 ms per frame, but final decoded
events still use the token time grid.

## Pedal Heads

This branch adds three dense heads on top of the encoder output in
[`model/T5.py`](Efficient-Transformer-with-pedals/model/T5.py):

```text
pedal_frame_head  -> pedal down/up state for every encoder frame
pedal_onset_head  -> soft target around PedalOn boundaries
pedal_offset_head -> soft target around PedalOff boundaries
```

These heads are trained with
[`PedalFrameBCELoss`](Efficient-Transformer-with-pedals/loss/PedalFrameBCELoss.py)
and each auxiliary pedal loss is weighted by `0.3` in
[`base_config.yaml`](Efficient-Transformer-with-pedals/config/base_config.yaml).

The final MIDI is still produced by decoder tokens. The frame heads help train the
encoder to represent pedal information, and the `pedal_frame_head` is also used
for diagnostic frame metrics, but the test code does not convert the frame-head
output into pedal events for MIDI.

## What Frame Metrics Mean

Frame metrics treat transcription as a binary classification problem at each time
frame.

For a pedal frame metric:

```text
positive frame = sustain pedal is down
negative frame = sustain pedal is up
```

The Experiment code thresholds the pedal frame probability at `0.5`, flattens all
valid frames for a track, and computes:

| Metric | Meaning |
| --- | --- |
| `pedal_frame_precision` | Of all frames predicted as pedal-down, how many were truly pedal-down? Low precision means too many false pedal-down frames. |
| `pedal_frame_recall` | Of all true pedal-down frames, how many did the model catch? Low recall means the model misses sustained pedal regions. |
| `pedal_frame_f1` | Harmonic mean of precision and recall. A compact summary of pedal-down state quality. |

Frame metrics answer "was the state correct at this moment?" They do not directly
answer "was the event boundary exactly correct?" A pedal release that is late by
one frame only affects a small number of frame labels, while a missed long pedal
span can affect hundreds of frames. That is why frame F1 can be high even when
event F1 is lower, and why it is possible for a model to have a much better
`pedal_frame_f1` without an equally large improvement in decoded `PedalOn` and
`PedalOff` event metrics.

This is especially important in this Experiment repo:

```text
pedal_frame_*       = auxiliary encoder pedal-state head
pedal_*             = decoded pedal token events compared to reference pedals
note_*              = decoded note token events compared to target notes
note+offset_*       = decoded notes with both onset and offset criteria
pedal_extended_*    = notes re-evaluated after applying pedal sustain semantics
```

The first row of each `!test_metrics_summary.csv` is a simple mean over tracks.
Values are stored as decimals from `0` to `1`; this document displays them as
percentages.

## Main Experiment Results

These are the average rows from:

- [`MAESTRO_PedalHeads_200k_update`](Efficient-Transformer-with-pedals/evaluations/MAESTRO_PedalHeads_200k_update/!test_metrics_summary.csv)
- [`MAESTRO_OldS_200k_update`](Efficient-Transformer-with-pedals/evaluations/MAESTRO_OldS_200k_update/!test_metrics_summary.csv)

| Metric | PedalHeads 200k | OldS 200k | Delta | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `pedal_frame_precision` | 87.58% | 54.56% | +33.02 pp | Much fewer false pedal-down frames. |
| `pedal_frame_recall` | 93.74% | 57.01% | +36.73 pp | Much fewer missed pedal-down frames. |
| `pedal_frame_f1` | 90.21% | 52.12% | +38.09 pp | The new pedal heads learned pedal state far better. |
| `pedal_f1` | 78.52% | 75.37% | +3.15 pp | Decoded pedal events improved, but much less than the frame head. |
| `note_f1` | 94.63% | 95.11% | -0.48 pp | Onset-only note event quality is slightly lower. |
| `note+offset_f1` | 67.27% | 67.96% | -0.69 pp | Raw note offset event quality is slightly lower. |
| `note+offset+velocity_f1` | 66.33% | 67.13% | -0.81 pp | Adding velocity keeps the same slight regression. |
| `note+offset_f1_pedal_extended` | 72.71% | 72.48% | +0.23 pp | After pedal sustain semantics, note-offset quality is slightly better. |
| `note+offset+velocity_f1_pedal_extended` | 71.78% | 71.65% | +0.13 pp | Same trend with velocity included. |

The big story is not "all metrics improved." The big story is:

1. The PedalHeads checkpoint strongly improves the direct pedal-state frame head.
2. That only modestly improves decoded pedal events.
3. Raw note metrics are slightly worse.
4. Pedal-extended note metrics are slightly better, which suggests the better
   pedal representation is helping where sustain semantics matter.

The PedalHeads `pedal_frame_recall` is higher than `pedal_frame_precision`
(`93.74%` vs `87.58%`). That means it catches most true pedal-down regions, but
it still tends to add some extra pedal-down frames.

## Note And Pedal Event Metrics

The event metrics are computed after detokenization in
[`train.py`](Efficient-Transformer-with-pedals/train.py) and
[`metrics/transcription_metrics.py`](Efficient-Transformer-with-pedals/metrics/transcription_metrics.py).

| Metric family | What counts as correct |
| --- | --- |
| `note_precision`, `note_recall`, `note_f1` | Pitch and onset match. Offset is ignored with `offset_ratio=None`. |
| `note+offset_*` | Pitch, onset, and offset match using `mir_eval` note matching. |
| `note+offset+velocity_*` | Same as `note+offset_*`, with velocity agreement required. |
| `pedal_precision`, `pedal_recall`, `pedal_f1` | Reference-style pedal interval metric against reference pedal events. In this repo, the headline reference pedal metric uses a 200 ms onset tolerance and offset-ratio criterion. |
| `diagnostic_pedal_*` | Extra pedal diagnostics, including stricter onset-only and onset+offset variants. |
| `pedal_on_*`, `pedal_off_*` | Individual pedal boundary event matching. |
| `pedal_frame_*` | Auxiliary pedal-down frame classification, not decoded MIDI events. |
| `*_pedal_extended` | Note metrics after applying sustain-pedal extension to note offsets. |
| `diagnostic_*_uncapped` | Pedal-extended diagnostics using uncapped cached-TSV semantics. Useful for debugging, not the clean headline number. |

Because the active config uses `use_truth_offsets: true`, the raw note-off target
is the physical key-release time. The pedal-extended metrics are therefore the
better place to ask "does the transcription sound sustained correctly?"

## Comparison With `reference/piano_transcription`

The reference implementation is the high-resolution piano transcription system
from the local paper PDF:
[`High-resolution Piano Transcription with Pedals by Regressing Precise Onsets and Offsets Times_v0.2.pdf`](<../reference/piano_transcription/paper/High-resolution Piano Transcription with Pedals by Regressing Precise Onsets and Offsets Times_v0.2.pdf>).
Its main code is in
[`reference/piano_transcription/pytorch/models.py`](../reference/piano_transcription/pytorch/models.py)
and
[`reference/piano_transcription/pytorch/calculate_score_for_paper.py`](../reference/piano_transcription/pytorch/calculate_score_for_paper.py).

| Aspect | Experiment Transformer | Reference `piano_transcription` |
| --- | --- | --- |
| Model family | T5/MT3-style Transformer encoder-decoder. | CRNN frame/regression system. |
| Main output | Autoregressive symbolic event tokens. | Frame-wise note and pedal probabilities plus onset/offset regression maps. |
| Note representation | `NoteOn` and `NoteOff` tokens with quantized onset-time tokens. | 88-way frame, onset-regression, offset-regression, and velocity outputs. |
| Pedal representation | Decoder emits `PedalOn` and `PedalOff` tokens; encoder also has auxiliary pedal frame/onset/offset heads. | Separate pedal CRNN predicts pedal frame, pedal onset regression, and pedal offset regression. |
| Timing resolution | Encoder frames are 20 ms; decoded onset tokens are 50 ms. | 100 frames/s, 10 ms frame grid, then analytical regression post-processing for sub-frame timing. |
| Post-processing | Detokenize generated tokens into note and pedal events. | Detect local maxima and use regression outputs to refine precise event times. |
| Frame metrics | Only pedal frame metrics are reported in current CSVs. | Note frame and pedal frame metrics are native model outputs. |
| Sustain semantics | Active config trains raw key-release note offsets plus separate pedal tokens; evaluation can also report pedal-extended note metrics. | Target generation commonly uses `extend_pedal=True`, so note offsets can be extended by sustain pedal during reference evaluation. |

The reference paper reports these MAESTRO test-set headline numbers:

| Reference result | F1 |
| --- | ---: |
| Note frame | 89.62% |
| Note onset only | 96.72% |
| Note with offset | 82.47% |
| Note with offset and velocity | 80.92% |
| Pedal frame | 94.25% |
| Pedal event onset | 91.86% |
| Pedal event with offset | 86.58% |

Those reference numbers are useful orientation, but they are not a perfectly
controlled apples-to-apples comparison with the Experiment CSVs. The repos differ
in model family, token/event representation, target offset semantics, MAESTRO
version/configuration, frame rate, timing resolution, post-processing, and
averaging details.

## Practical Reading Of The Current CSV

Use this order when interpreting the current Experiment results:

1. Check `note_f1` for "did it find the right pitches at the right onsets?"
2. Check `note+offset_f1` for "did it also stop notes at the right raw key-release time?"
3. Check `note+offset_f1_pedal_extended` for "does the note duration make sense after sustain pedal behavior?"
4. Check `pedal_f1` and `diagnostic_pedal+offset_f1` for "did it emit reasonable pedal events?"
5. Check `pedal_frame_f1` for "does the encoder know when the pedal is down?"

For the PedalHeads checkpoint, the frame result is a strong positive signal about
pedal-state learning. The modest gain in decoded pedal events says the decoder
and token timing are now the likely bottlenecks if the goal is better MIDI pedal
events. The small regression in raw note event metrics says the pedal auxiliary
loss is not free; it shifts the model slightly toward pedal-aware behavior.

## Bottom Line

The Experiment transcription model is best understood as a pedal-aware
audio-to-event Transformer, not as the same kind of high-resolution frame
regression model used in `reference/piano_transcription`.

The frame metrics tell us whether the model's per-frame pedal state is right.
They are valuable because sustain pedal is a long-duration state, but they should
not be treated as final transcription quality by themselves. Final transcription
quality is still the event-level note and pedal metrics after the generated token
sequence has been converted back into notes and MIDI pedal events.
