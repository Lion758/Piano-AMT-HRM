from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

librosa_module = types.ModuleType("librosa")
librosa_display_module = types.ModuleType("librosa.display")
librosa_module.display = librosa_display_module
sys.modules["librosa"] = librosa_module
sys.modules["librosa.display"] = librosa_display_module

sys.modules["music21"] = types.ModuleType("music21")
sys.modules["pretty_midi"] = types.ModuleType("pretty_midi")
sys.modules["torchaudio"] = types.ModuleType("torchaudio")
sys.modules["h5py"] = types.ModuleType("h5py")

matplotlib_module = types.ModuleType("matplotlib")
matplotlib_pyplot_module = types.ModuleType("matplotlib.pyplot")
matplotlib_module.pyplot = matplotlib_pyplot_module
sys.modules["matplotlib"] = matplotlib_module
sys.modules["matplotlib.pyplot"] = matplotlib_pyplot_module

omegaconf_module = types.ModuleType("omegaconf")
omegaconf_module.OmegaConf = type("OmegaConf", (), {})
sys.modules["omegaconf"] = omegaconf_module

hydra_module = types.ModuleType("hydra")
hydra_module.main = lambda *args, **kwargs: (lambda fn: fn)
sys.modules["hydra"] = hydra_module

symusic_module = types.ModuleType("symusic")
symusic_module.Score = type("Score", (), {})
symusic_module.TimeUnit = types.SimpleNamespace(second="second")
sys.modules["symusic"] = symusic_module

utils_module = types.ModuleType("utils")
utils_module.__path__ = []
pianoroll_parser_module = types.ModuleType("utils.pianoroll_parser")
pianoroll_parser_module.get_notes_with_pedal = lambda midi_path: (None, None)
sys.modules["utils"] = utils_module
sys.modules["utils.pianoroll_parser"] = pianoroll_parser_module

from data.dataset_Audio2Midi import SingleWavDataset
from data.constants import sm_tokenizer


def _make_dataset(events, n_frames=8):
    dataset = object.__new__(SingleWavDataset)
    dataset.n_frames = n_frames
    rows = [
        {
            "type": event_type,
            "onset_sec": float(onset_sec),
            "pitch": -1,
            "velocity": 0,
        }
        for event_type, onset_sec in events
    ]
    dataset.dataframe_midi = pd.DataFrame(rows, columns=["type", "onset_sec", "pitch", "velocity"])
    dataset._prepare_pedal_cache()
    return dataset


def _assert_fast_matches_slow(events, begin_sec, end_sec, second_per_frame=0.02, n_frames=8):
    dataset = _make_dataset(events, n_frames=n_frames)
    slow_targets = dataset._build_pedal_targets_slow(begin_sec, end_sec, second_per_frame)
    fast_targets = dataset._build_pedal_targets_fast(begin_sec, end_sec, second_per_frame)

    for slow, fast in zip(slow_targets, fast_targets):
        assert fast.shape == (n_frames,)
        assert fast.dtype == torch.float32
        assert torch.equal(fast, slow)


def test_fast_pedal_targets_match_slow_with_no_pedal_events():
    _assert_fast_matches_slow([], begin_sec=0.0, end_sec=0.16)


def test_fast_pedal_targets_match_slow_when_pedal_starts_before_clip():
    _assert_fast_matches_slow(
        [
            ("PedalOn", 0.00),
            ("PedalOff", 0.08),
        ],
        begin_sec=0.04,
        end_sec=0.18,
    )


def test_fast_pedal_targets_match_slow_when_pedal_starts_and_ends_inside_clip():
    _assert_fast_matches_slow(
        [
            ("PedalOn", 0.04),
            ("PedalOff", 0.10),
        ],
        begin_sec=0.00,
        end_sec=0.16,
    )


def test_fast_pedal_targets_match_slow_for_same_time_off_then_on():
    _assert_fast_matches_slow(
        [
            ("PedalOn", 0.00),
            ("PedalOff", 0.06),
            ("PedalOn", 0.06),
            ("PedalOff", 0.12),
        ],
        begin_sec=0.02,
        end_sec=0.16,
    )


def test_fast_pedal_targets_canonicalize_same_time_on_before_off_repedal():
    dataset = _make_dataset(
        [
            ("PedalOn", 0.00),
            ("PedalOn", 0.06),
            ("PedalOff", 0.06),
        ],
        n_frames=6,
    )

    frame_target, onset_target, offset_target = dataset._build_pedal_targets_fast(
        begin_sec=0.02,
        end_sec=0.14,
        second_per_frame=0.02,
    )

    assert torch.equal(frame_target, torch.ones(6, dtype=torch.float32))
    assert onset_target[2] == 1.0
    assert offset_target[2] == 1.0


def test_pedal_targets_do_not_depend_on_decoder_pedal_tokens():
    dataset = object.__new__(SingleWavDataset)
    dataset.n_frames = 6
    dataset.dataframe_midi = pd.DataFrame(
        [
            {
                "type": "NoteOn",
                "onset_sec": 0.02,
                "pitch": 60,
                "velocity": 80,
            },
            {
                "type": "NoteOff",
                "onset_sec": 0.10,
                "pitch": 60,
                "velocity": 0,
            },
        ]
    )
    dataset.dataframe_pedal_events = pd.DataFrame(
        [
            {
                "type": "PedalOn",
                "onset_sec": 0.02,
                "pitch": -1,
                "velocity": 0,
            },
            {
                "type": "PedalOff",
                "onset_sec": 0.08,
                "pitch": -1,
                "velocity": 0,
            },
        ]
    )
    dataset._prepare_pedal_cache()

    frame_target, onset_target, offset_target = dataset._build_pedal_targets_fast(
        begin_sec=0.00,
        end_sec=0.12,
        second_per_frame=0.02,
    )

    assert not dataset.dataframe_midi["type"].isin(["PedalOn", "PedalOff"]).any()
    assert torch.equal(
        frame_target,
        torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    )
    assert onset_target[1] == 1.0
    assert offset_target[4] == 1.0


def test_tokenizer_orders_same_time_pedal_off_before_on():
    df = pd.DataFrame(
        [
            {
                "type": "PedalOn",
                "onset_sec": 0.50,
                "offset_sec": 0.50,
                "pitch": -1,
                "velocity": 0,
            },
            {
                "type": "PedalOff",
                "onset_sec": 0.50,
                "offset_sec": 0.50,
                "pitch": -1,
                "velocity": 0,
            },
        ]
    )

    event_df = sm_tokenizer.notes_to_midi_events(
        df,
        use_truth_offsets=False,
        emit_pedal_tokens=True,
    )

    assert event_df["type"].tolist() == ["PedalOff", "PedalOn"]


def test_fast_pedal_targets_match_slow_at_clip_boundaries_and_rounding_points():
    _assert_fast_matches_slow(
        [
            ("PedalOn", 0.02),
            ("PedalOff", 0.05),
            ("PedalOn", 0.10),
            ("PedalOff", 0.12),
        ],
        begin_sec=0.02,
        end_sec=0.10,
    )


def test_fast_pedal_targets_keep_training_batch_contract():
    dataset = _make_dataset([("PedalOn", 0.02), ("PedalOff", 0.08)], n_frames=6)
    targets = dataset._build_pedal_targets_fast(0.00, 0.12, 0.02)
    pedal_target_mask = torch.ones(dataset.n_frames, dtype=torch.long)
    row = {
        "pedal_frame_target": targets[0],
        "pedal_frame_target_mask": pedal_target_mask,
        "pedal_onset_target": targets[1],
        "pedal_onset_target_mask": pedal_target_mask,
        "pedal_offset_target": targets[2],
        "pedal_offset_target_mask": pedal_target_mask,
    }

    for key in ("pedal_frame_target", "pedal_onset_target", "pedal_offset_target"):
        assert row[key].shape == (dataset.n_frames,)
        assert row[key].dtype == torch.float32
    for key in ("pedal_frame_target_mask", "pedal_onset_target_mask", "pedal_offset_target_mask"):
        assert row[key].shape == (dataset.n_frames,)
        assert row[key].dtype == torch.long
