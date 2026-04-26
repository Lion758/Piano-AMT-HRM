from pathlib import Path
from types import SimpleNamespace
import importlib
import sys
import types

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import wavfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _real_pretty_midi():
    module = sys.modules.get("pretty_midi")
    if module is not None and not hasattr(module, "PrettyMIDI"):
        del sys.modules["pretty_midi"]
    return importlib.import_module("pretty_midi")


pretty_midi = _real_pretty_midi()


def _install_torchaudio_stub():
    torchaudio_module = types.ModuleType("torchaudio")

    def load(path):
        sample_rate, audio = wavfile.read(path)
        audio = audio.astype(np.float32)
        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.T
        max_abs = max(float(np.abs(audio).max()), 1.0)
        return torch.from_numpy(audio / max_abs), sample_rate

    class Resample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, audio):
            if self.orig_freq != self.new_freq:
                raise NotImplementedError("Test torchaudio stub only supports matching sample rates.")
            return audio

    torchaudio_module.load = load
    torchaudio_module.transforms = types.SimpleNamespace(Resample=Resample)
    transforms_module = types.ModuleType("torchaudio.transforms")
    transforms_module.Resample = Resample
    sys.modules["torchaudio"] = torchaudio_module
    sys.modules["torchaudio.transforms"] = transforms_module


def _install_lightweight_dataset_import_stubs():
    librosa_module = types.ModuleType("librosa")
    librosa_module.__path__ = []
    librosa_display_module = types.ModuleType("librosa.display")
    librosa_module.display = librosa_display_module
    sys.modules.setdefault("librosa", librosa_module)
    sys.modules.setdefault("librosa.display", librosa_display_module)

    music21_module = sys.modules.setdefault("music21", types.ModuleType("music21"))
    music21_module.meter = types.SimpleNamespace(
        TimeSignature=lambda value: types.SimpleNamespace(barDuration=types.SimpleNamespace(quarterLength=4.0))
    )
    hydra_module = sys.modules.setdefault("hydra", types.ModuleType("hydra"))
    hydra_module.main = lambda *args, **kwargs: (lambda func: func)
    omegaconf_module = sys.modules.setdefault("omegaconf", types.ModuleType("omegaconf"))
    omegaconf_module.OmegaConf = type("OmegaConf", (), {})

    symusic_module = sys.modules.setdefault("symusic", types.ModuleType("symusic"))
    symusic_module.Score = type("Score", (), {})
    symusic_module.TimeUnit = types.SimpleNamespace(second="second")

    utils_module = sys.modules.setdefault("utils", types.ModuleType("utils"))
    utils_module.__path__ = [str(PROJECT_ROOT / "utils")]
    pianoroll_parser_module = types.ModuleType("utils.pianoroll_parser")
    pianoroll_parser_module.get_notes_with_pedal = lambda midi_path: (None, None)
    sys.modules.setdefault("utils.pianoroll_parser", pianoroll_parser_module)

    matplotlib_module = sys.modules.setdefault("matplotlib", types.ModuleType("matplotlib"))
    matplotlib_module.__path__ = []
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    pyplot_module.subplots = lambda *args, **kwargs: (None, types.SimpleNamespace())
    pyplot_module.imsave = lambda *args, **kwargs: None
    sys.modules.setdefault("matplotlib.pyplot", pyplot_module)


def _install_metric_import_stubs():
    mir_eval_module = sys.modules.setdefault("mir_eval", types.ModuleType("mir_eval"))
    mir_eval_util = types.ModuleType("mir_eval.util")
    mir_eval_util.midi_to_hz = lambda values: values
    mir_eval_multipitch = types.ModuleType("mir_eval.multipitch")
    mir_eval_multipitch.evaluate = lambda *args, **kwargs: {}
    mir_eval_transcription = types.ModuleType("mir_eval.transcription")
    mir_eval_transcription.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    mir_eval_velocity = types.ModuleType("mir_eval.transcription_velocity")
    mir_eval_velocity.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
    mir_eval_module.util = mir_eval_util
    sys.modules.setdefault("mir_eval.util", mir_eval_util)
    sys.modules.setdefault("mir_eval.multipitch", mir_eval_multipitch)
    sys.modules.setdefault("mir_eval.transcription", mir_eval_transcription)
    sys.modules.setdefault("mir_eval.transcription_velocity", mir_eval_velocity)


def _write_test_wav(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 16000, np.zeros(16000, dtype=np.float32))


def _write_test_maps_tsv(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# onset,offset,note,velocity\n"
        "0.100000\t0.400000\t60.000000\t80.000000\n"
        "0.500000\t0.800000\t64.000000\t72.000000\n",
        encoding="utf-8",
    )


def _write_test_midi(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="piano")
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.10, end=0.40))
    instrument.control_changes.append(pretty_midi.ControlChange(number=64, value=127, time=0.20))
    instrument.control_changes.append(pretty_midi.ControlChange(number=64, value=0, time=0.60))
    midi.instruments.append(instrument)
    midi.write(str(path))


def test_prepare_maps_writes_expected_oneshot_artifacts(tmp_path):
    _install_torchaudio_stub()
    from data.prepare_maps import prepare_maps_dataset

    maps_root = tmp_path / "maps"
    audio_path = maps_root / "raw" / "ENSTDkAm" / "MUS" / "MAPS_MUS-test_ENSTDkAm.wav"
    midi_path = audio_path.with_suffix(".mid")
    _write_test_wav(audio_path)
    _write_test_midi(midi_path)

    items = prepare_maps_dataset(maps_root=maps_root, settings=("ENSTDkAm",), category="MUS", jobs=1)

    assert len(items) == 1
    assert (maps_root / "subset_train.tsv").read_text(encoding="utf-8") == ""
    assert (maps_root / "subset_validation.tsv").read_text(encoding="utf-8") == ""
    assert (maps_root / "subset_test.tsv").read_text(encoding="utf-8").strip().endswith(
        "ENSTDkAm/MUS/MAPS_MUS-test_ENSTDkAm.wav"
    )
    assert (maps_root / "audio.h5").exists()
    assert (maps_root / "maps_metadata.csv").exists()

    cache_midi = maps_root / "cache" / "ENSTDkAm>MUS>MAPS_MUS-test_ENSTDkAm.mid"
    notes_tsv = maps_root / "cache" / "ENSTDkAm>MUS>MAPS_MUS-test_ENSTDkAm.midi-notes.tsv"
    assert cache_midi.exists()
    assert notes_tsv.exists()

    notes_df = pd.read_csv(notes_tsv, sep="\t")
    assert {"note", "PedalOn", "PedalOff"}.issubset(set(notes_df["type"]))
    note_row = notes_df[notes_df["type"] == "note"].iloc[0]
    assert note_row["offset_sec_truth"] == pytest.approx(0.40)
    assert note_row["offset_sec_pedal_extended"] == pytest.approx(0.60)

    with h5py.File(maps_root / "audio.h5") as h5:
        assert "0" in h5
        assert h5["0"]["sample_rate"][()] == 16000


def test_prepare_maps_writes_flattened_note_only_artifacts(tmp_path):
    _install_torchaudio_stub()
    from data.prepare_maps import prepare_maps_dataset

    maps_root = tmp_path / "maps"
    audio_path = maps_root / "flac" / "MAPS_MUS-test_ENSTDkAm.flac"
    tsv_path = maps_root / "tsv" / "matched" / "MAPS_MUS-test_ENSTDkAm.tsv"
    _write_test_wav(audio_path)
    _write_test_maps_tsv(tsv_path)

    items = prepare_maps_dataset(maps_root=maps_root, settings=("ENSTDkAm",), category="MUS", jobs=1)

    assert len(items) == 1
    assert items[0].source_format == "flattened-tsv"
    assert (maps_root / "subset_test.tsv").read_text(encoding="utf-8").strip() == (
        "0\tflac/MAPS_MUS-test_ENSTDkAm.flac"
    )

    cache_midi = maps_root / "cache" / "flac>MAPS_MUS-test_ENSTDkAm.mid"
    notes_tsv = maps_root / "cache" / "flac>MAPS_MUS-test_ENSTDkAm.midi-notes.tsv"
    assert cache_midi.exists()
    assert notes_tsv.exists()

    notes_df = pd.read_csv(notes_tsv, sep="\t")
    assert set(notes_df["type"]) == {"note"}
    assert notes_df.iloc[0]["offset_sec_truth"] == pytest.approx(0.40)
    assert notes_df.iloc[0]["offset_sec_pedal_extended"] == pytest.approx(0.40)

    midi = pretty_midi.PrettyMIDI(str(cache_midi))
    assert len(midi.instruments) == 1
    assert len(midi.instruments[0].notes) == 2
    assert midi.instruments[0].control_changes == []


def test_maps_fixture_loads_through_audio2midi_dataset(tmp_path):
    _install_torchaudio_stub()
    _install_lightweight_dataset_import_stubs()
    from data.prepare_maps import prepare_maps_dataset
    from data.dataset_Audio2Midi import Audio2Midi_Dataset

    maps_root = tmp_path / "maps"
    audio_path = maps_root / "raw" / "ENSTDkAm" / "MUS" / "MAPS_MUS-test_ENSTDkAm.wav"
    _write_test_wav(audio_path)
    _write_test_midi(audio_path.with_suffix(".mid"))
    prepare_maps_dataset(maps_root=maps_root, settings=("ENSTDkAm",), category="MUS", jobs=1)

    config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_sequence_types=["performance-sequence"],
            n_frames=10,
            max_token_length=128,
            hop_length=320,
            cache_dir_name="cache",
            include_pedal_events=True,
            use_note_extensions=False,
        ),
        model=SimpleNamespace(),
    )
    dataset = Audio2Midi_Dataset(config, str(maps_root), dataset_index=0, subset="test", random_clip=False)

    row = dataset[0]

    assert row["inputs"].shape[0] == 3200
    assert row["decoder_targets_len"].item() > 0


def test_flattened_maps_fixture_loads_through_audio2midi_dataset(tmp_path):
    _install_torchaudio_stub()
    _install_lightweight_dataset_import_stubs()
    from data.prepare_maps import prepare_maps_dataset
    from data.dataset_Audio2Midi import Audio2Midi_Dataset

    maps_root = tmp_path / "maps"
    audio_path = maps_root / "flac" / "MAPS_MUS-test_ENSTDkAm.flac"
    tsv_path = maps_root / "tsv" / "matched" / "MAPS_MUS-test_ENSTDkAm.tsv"
    _write_test_wav(audio_path)
    _write_test_maps_tsv(tsv_path)
    prepare_maps_dataset(maps_root=maps_root, settings=("ENSTDkAm",), category="MUS", jobs=1)

    config = SimpleNamespace(
        data=SimpleNamespace(
            dataset_sequence_types=["performance-sequence"],
            n_frames=10,
            max_token_length=128,
            hop_length=320,
            cache_dir_name="cache",
            include_pedal_events=False,
            use_note_extensions=False,
        ),
        model=SimpleNamespace(),
    )
    dataset = Audio2Midi_Dataset(config, str(maps_root), dataset_index=0, subset="test", random_clip=False)

    row = dataset[0]

    assert row["inputs"].shape[0] == 3200
    assert row["decoder_targets_len"].item() > 0


def test_pedal_tokens_round_trip_and_save_to_cc64(tmp_path):
    _install_lightweight_dataset_import_stubs()
    import data.symbolic_music_tokenizer as tokenizer_module
    from data.constants import sm_tokenizer

    tokenizer_module.pretty_midi = pretty_midi
    df = pd.DataFrame(
        {
            "type": ["PedalOn", "PedalOff"],
            "type_id": [0, 0],
            "pitch": [-1, -1],
            "onset_sec": [0.25, 0.75],
            "offset_sec": [0.75, 0.75],
            "duration_sec": [0.50, 0.0],
            "offset_sec_truth": [0.75, 0.75],
            "duration_sec_truth": [0.50, 0.0],
            "offset_sec_pedal_extended": [0.75, 0.75],
            "duration_sec_pedal_extended": [0.50, 0.0],
            "velocity": [0, 0],
        }
    )

    events = sm_tokenizer.notes_to_midi_events(df, include_pedal_events=True)
    tokens, _ = sm_tokenizer.tokenize_dataframe(events)
    decoded_events = sm_tokenizer.detokenize(tokens, offsets_sec=0.0)
    pedal_events = sm_tokenizer.midi_events_to_pedals(decoded_events)

    assert pedal_events == [
        {"time": 0.25, "type": "PedalOn", "value": 127},
        {"time": 0.75, "type": "PedalOff", "value": 0},
    ]

    midi_path = tmp_path / "pedal.mid"
    sm_tokenizer.save_midi(
        [{"pitch": 60, "velocity": 80, "onset": 0.10, "duration": 0.20, "staff": 0}],
        str(midi_path),
        pedal_event_list=pedal_events,
    )
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    controls = midi.instruments[0].control_changes

    assert [(cc.number, cc.value, cc.time) for cc in controls] == [(64, 127, 0.25), (64, 0, 0.75)]


def test_pedal_metric_matching_uses_50ms_tolerance():
    _install_lightweight_dataset_import_stubs()
    _install_metric_import_stubs()
    from metrics.transcription_metrics import cal_pedal_event_metrics

    reference = [
        {"type": "PedalOn", "time": 1.00},
        {"type": "PedalOff", "time": 2.00},
    ]
    estimated = [
        {"type": "PedalOn", "time": 1.04},
        {"type": "PedalOff", "time": 2.06},
    ]

    metrics = cal_pedal_event_metrics(estimated, reference, tolerance=0.05)

    assert metrics["pedal_on_precision"] == pytest.approx(1.0)
    assert metrics["pedal_on_recall"] == pytest.approx(1.0)
    assert metrics["pedal_on_f1"] == pytest.approx(1.0)
    assert metrics["pedal_off_precision"] == pytest.approx(0.0)
    assert metrics["pedal_off_recall"] == pytest.approx(0.0)
    assert metrics["pedal_off_f1"] == pytest.approx(0.0)
