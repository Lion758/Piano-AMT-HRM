# Evaluation CSV Metric Label Key

This key explains the labels in `!test_metrics_summary.csv` and
`threshold_sweep_*.csv` for the Efficient Transformer pedal evaluations.

Values are decimals from `0` to `1`; multiply by `100` for percentages. In
`!test_metrics_summary.csv`, the first data row is the mean over tracks, and
later rows are per-track values.

## Basic Terms

| Label part | Meaning |
| --- | --- |
| `precision` | Of the model's predicted positives/events, how many were correct. |
| `recall` | Of the reference positives/events, how many the model found. |
| `f1` | Harmonic mean of precision and recall. |
| `+offset` | Event/interval match also checks the pedal release time, not only the press/onset time. |
| `+velocity` | Note metric also checks velocity agreement. This applies to notes, not pedals. |
| `pedal_on` | Pedal press boundary only: `PedalOn` event times. |
| `pedal_off` | Pedal release boundary only: `PedalOff` event times. |

## Which Pedal Source Is Being Scored

| Column/prefix | Meaning |
| --- | --- |
| `pedal_event_source_used` | The source used for headline `pedal_*` metrics. With the conv-head configs this should usually be `frame_head`. |
| `midi_pedal_event_source_used` | The source used when saving output MIDI pedals. |
| `decoder_token_*` | Pedal events emitted directly by the autoregressive decoder tokens. |
| `frame_head_*` | Pedal events produced by thresholding the auxiliary pedal state head, then converting state spans into `PedalOn`/`PedalOff` events. |
| unprefixed `pedal_*` | The headline pedal metric for the configured `evaluation.pedal_event_source`. If `pedal_event_source: frame_head`, these are scoring frame-head-derived pedal events. |

Important distinction: `frame_head_*` does **not** mean raw per-frame accuracy. It
means event metrics after converting the pedal state head into pedal events.

## Headline vs Diagnostic Pedal Metrics

| Label | Meaning |
| --- | --- |
| `pedal_precision`, `pedal_recall`, `pedal_f1` | Headline reference-style pedal interval metric for the selected pedal source. Uses a looser pedal-event convention: 200 ms onset tolerance plus offset-ratio matching. This is the main `pedal_f1`. |
| `diagnostic_pedal_precision`, `diagnostic_pedal_recall`, `diagnostic_pedal_f1` | Diagnostic onset-only interval metric for the selected pedal source. Uses a stricter 50 ms onset tolerance and ignores release timing. |
| `diagnostic_pedal+offset_*` | Diagnostic interval metric that checks both pedal press and release. |
| `diagnostic_pedal_on_*` | Diagnostic `PedalOn` boundary matching only, within the event tolerance. |
| `diagnostic_pedal_off_*` | Diagnostic `PedalOff` boundary matching only, within the event tolerance. |
| `*_reference_pedal_*` | The same reference-style metric as headline `pedal_*`, but reported for a named source such as `decoder_token` or `frame_head`. |

If `pedal_event_source: frame_head`, then headline `pedal_f1` and
`frame_head_reference_pedal_f1` should represent the same metric family for the
same source. The `diagnostic_*` columns are there to show stricter boundary
behavior and should not be mixed directly with the headline score.

## Raw Auxiliary Frame-Head Metrics

These columns score the raw auxiliary heads frame by frame, before MIDI/event
post-processing:

| Label | Meaning |
| --- | --- |
| `pedal_frame_precision`, `pedal_frame_recall`, `pedal_frame_f1` | Binary per-frame pedal-down state classification from `pedal_frame_head`. Positive means "pedal is down" at that frame. |
| `pedal_onset_frame_*` | Binary per-frame classification for the auxiliary onset head target. Positive means "near a `PedalOn` boundary." |
| `pedal_offset_frame_*` | Binary per-frame classification for the auxiliary offset head target. Positive means "near a `PedalOff` boundary." |

The current frame-head MIDI/event conversion uses `pedal_frame_output` only. The
`pedal_onset_frame_*` and `pedal_offset_frame_*` columns are auxiliary-head
diagnostics; they do not by themselves mean that final pedal events improved.

## Frame-Head Event Metrics

These score pedal events after converting the pedal state head into spans:

| Label | Meaning |
| --- | --- |
| `frame_head_reference_pedal_*` | Reference-style interval metric for frame-head-derived pedal events. Comparable to headline `pedal_*` when the configured source is `frame_head`. |
| `frame_head_pedal_*` | Diagnostic onset-only metric for frame-head-derived pedal events. |
| `frame_head_pedal+offset_*` | Diagnostic press-and-release interval metric for frame-head-derived pedal events. Useful for threshold sweeps. |
| `frame_head_pedal_on_*` | `PedalOn` boundary metric for frame-head-derived events. |
| `frame_head_pedal_off_*` | `PedalOff` boundary metric for frame-head-derived events. |

## Note Metrics

| Label | Meaning |
| --- | --- |
| `note_*` | Decoder note events scored by pitch and onset. Offset is ignored. |
| `note+offset_*` | Decoder note events scored by pitch, onset, and offset. |
| `note+offset+velocity_*` | Same as `note+offset_*`, also requiring velocity agreement. |
| `*_pedal_extended` | Note metrics after applying sustain-pedal extension semantics. These are often more meaningful for audible sustain quality. |
| `diagnostic_*_pedal_extended_uncapped` | Pedal-extended note diagnostics using uncapped extension behavior. Useful for analysis, not the clean headline note score. |

## Threshold Sweep CSV Columns

`threshold_sweep_*.csv` rescoring starts from saved validation/test JSON, so it
does not rerun the model. It changes only the post-processing of
`pedal_frame_output`.

| Label | Meaning |
| --- | --- |
| `frame_head_threshold_on` | State-head probability threshold used to start a pedal-down span. |
| `frame_head_threshold_off` | State-head probability threshold used to end a pedal-down span. Usually lower than `threshold_on` for hysteresis. |
| `frame_head_min_down_frames` | Minimum accepted pedal-down span length. Shorter predicted spans are filtered. |
| `frame_head_min_up_frames` | Minimum accepted gap between pedal-down spans. Shorter gaps can be merged/ignored by post-processing. |
| `track_count` | Number of tracks read from saved JSON files. |
| `scored_track_count` | Number of tracks with reference pedal events available for scoring. |
| `mean_predicted_pedal_span_count` | Average number of predicted pedal spans after thresholding/post-processing. |
| `mean_reference_pedal_span_count` | Average number of reference pedal spans. |
| `default_distance_steps` | Tie-breaker distance from the default settings: on `0.50`, off `0.40`, min-down `3`, min-up `2`. Lower means closer to defaults. |

For threshold selection, prefer comparing `frame_head_pedal+offset_f1`,
`frame_head_pedal_f1`, and `frame_head_reference_pedal_f1` together. A threshold
that improves onset-only F1 while hurting `+offset` may be making releases worse.
