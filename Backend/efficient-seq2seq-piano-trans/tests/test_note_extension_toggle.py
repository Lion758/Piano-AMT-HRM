from pathlib import Path
from types import SimpleNamespace
import sys
import types

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("music21", types.ModuleType("music21"))
sys.modules.setdefault("pretty_midi", types.ModuleType("pretty_midi"))

symusic_module = sys.modules.setdefault("symusic", types.ModuleType("symusic"))
if not hasattr(symusic_module, "Score"):
    symusic_module.Score = type("Score", (), {})
if not hasattr(symusic_module, "TimeUnit"):
    symusic_module.TimeUnit = types.SimpleNamespace(second="second")

utils_module = sys.modules.setdefault("utils", types.ModuleType("utils"))
if not hasattr(utils_module, "__path__"):
    utils_module.__path__ = []
pianoroll_parser_module = types.ModuleType("utils.pianoroll_parser")
pianoroll_parser_module.get_notes_with_pedal = lambda midi_path: (None, None)
sys.modules.setdefault("utils.pianoroll_parser", pianoroll_parser_module)

from data.offset_utils import resolve_use_truth_offsets
from data.symbolic_music_tokenizer import SymbolicMusicTokenizer

TOKENIZER = SymbolicMusicTokenizer()


def _make_config(**data_kwargs):
    return SimpleNamespace(data=SimpleNamespace(**data_kwargs))


def _make_dual_offset_dataframe():
    return pd.DataFrame(
        {
            "type": ["note"],
            "pitch": [60],
            "onset_sec": [1.0],
            "offset_sec": [3.0],
            "duration_sec": [2.0],
            "offset_sec_truth": [2.0],
            "duration_sec_truth": [1.0],
            "velocity": [80],
        }
    )


def test_resolve_use_truth_offsets_prefers_use_note_extensions_false():
    config = _make_config(use_note_extensions=False, use_truth_offsets=False)

    assert resolve_use_truth_offsets(config) is True


def test_resolve_use_truth_offsets_prefers_use_note_extensions_true():
    config = _make_config(use_note_extensions=True, use_truth_offsets=True)

    assert resolve_use_truth_offsets(config) is False


def test_resolve_use_truth_offsets_falls_back_to_legacy_flag():
    config = _make_config(use_truth_offsets=True)

    assert resolve_use_truth_offsets(config) is True


def test_notes_to_midi_events_switches_note_off_column_with_toggle():
    df = _make_dual_offset_dataframe()

    extended_events = TOKENIZER.notes_to_midi_events(df, use_truth_offsets=False)
    raw_events = TOKENIZER.notes_to_midi_events(df, use_truth_offsets=True)

    extended_note_off = extended_events[extended_events["type"] == "NoteOff"]["onset_sec"].tolist()
    raw_note_off = raw_events[raw_events["type"] == "NoteOff"]["onset_sec"].tolist()

    assert extended_note_off == [3.0]
    assert raw_note_off == [2.0]


def test_notes_to_midi_events_requires_raw_offset_columns_in_raw_mode():
    df = pd.DataFrame(
        {
            "type": ["note"],
            "pitch": [60],
            "onset_sec": [1.0],
            "offset_sec": [3.0],
            "duration_sec": [2.0],
            "velocity": [80],
        }
    )

    with pytest.raises(ValueError, match="data.use_note_extensions=false"):
        TOKENIZER.notes_to_midi_events(df, use_truth_offsets=True)
