import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pretty_midi
import torch
import torchaudio
from torchaudio.transforms import Resample
from tqdm import tqdm


DEFAULT_SETTINGS = ("ENSTDkAm", "ENSTDkCl")
ALL_SETTINGS = (
    "AkPnBcht",
    "AkPnBsdf",
    "AkPnCGdD",
    "AkPnStgb",
    "ENSTDkAm",
    "ENSTDkCl",
    "SptkBGAm",
    "SptkBGCl",
    "StbgTGd2",
)
DEFAULT_CATEGORY = "MUS"
DEFAULT_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class MapsItem:
    audio_path: Path
    relative_audio_path: str
    setting: str
    category: str
    audio_id: int
    midi_path: Path | None = None
    tsv_path: Path | None = None
    source_format: str = "raw-midi"


def _find_source_root(maps_root: Path, raw_root: Path | None, settings: tuple[str, ...]) -> Path:
    if raw_root is not None:
        return raw_root
    if any((maps_root / setting).is_dir() for setting in settings):
        return maps_root
    raw_candidate = maps_root / "raw"
    if any((raw_candidate / setting).is_dir() for setting in settings):
        return raw_candidate
    return raw_candidate


def _find_midi_for_audio(audio_path: Path) -> Path | None:
    for suffix in (".mid", ".midi"):
        candidate = audio_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _parse_flattened_maps_name(audio_path: Path) -> tuple[str, str]:
    stem = audio_path.stem
    setting = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    category = DEFAULT_CATEGORY
    if stem.startswith("MAPS_") and "-" in stem:
        category = stem.split("-", 1)[0].replace("MAPS_", "", 1)
    return setting, category


def _discover_raw_midi_items(
    maps_root: Path,
    raw_root: Path | None = None,
    settings: tuple[str, ...] = DEFAULT_SETTINGS,
    category: str = DEFAULT_CATEGORY,
) -> list[MapsItem]:
    source_root = _find_source_root(maps_root, raw_root, settings)
    items = []
    for setting in settings:
        category_dir = source_root / setting / category
        if not category_dir.is_dir():
            continue
        for audio_path in sorted(category_dir.glob("*.wav")):
            midi_path = _find_midi_for_audio(audio_path)
            if midi_path is None:
                continue
            relative_audio_path = str(Path(setting) / category / audio_path.name)
            items.append(
                MapsItem(
                    audio_path=audio_path,
                    relative_audio_path=relative_audio_path,
                    setting=setting,
                    category=category,
                    audio_id=len(items),
                    midi_path=midi_path,
                )
            )
    return items


def _discover_flattened_tsv_items(
    maps_root: Path,
    raw_root: Path | None = None,
    settings: tuple[str, ...] = ALL_SETTINGS,
    category: str = DEFAULT_CATEGORY,
) -> list[MapsItem]:
    source_root = raw_root if raw_root is not None else maps_root
    flac_dir = source_root / "flac"
    tsv_dir = source_root / "tsv" / "matched"
    if not flac_dir.is_dir() or not tsv_dir.is_dir():
        return []

    selected_settings = set(settings)
    items = []
    for audio_path in sorted(flac_dir.glob("*.flac")):
        setting, item_category = _parse_flattened_maps_name(audio_path)
        if selected_settings and setting not in selected_settings:
            continue
        if category and item_category != category:
            continue
        tsv_path = tsv_dir / f"{audio_path.stem}.tsv"
        if not tsv_path.exists():
            continue
        relative_audio_path = str(Path("flac") / audio_path.name)
        items.append(
            MapsItem(
                audio_path=audio_path,
                relative_audio_path=relative_audio_path,
                setting=setting,
                category=item_category,
                audio_id=len(items),
                tsv_path=tsv_path,
                source_format="flattened-tsv",
            )
        )
    return items


def discover_maps_items(
    maps_root: Path,
    raw_root: Path | None = None,
    settings: tuple[str, ...] = ALL_SETTINGS,
    category: str = DEFAULT_CATEGORY,
) -> list[MapsItem]:
    raw_items = _discover_raw_midi_items(maps_root, raw_root=raw_root, settings=settings, category=category)
    if raw_items:
        return raw_items
    return _discover_flattened_tsv_items(maps_root, raw_root=raw_root, settings=settings, category=category)


def cache_midi_path(maps_root: Path, relative_audio_path: str) -> Path:
    cache_name = Path(relative_audio_path).with_suffix("").as_posix().replace("/", ">") + ".mid"
    return maps_root / "cache" / cache_name


def cache_notes_path(maps_root: Path, relative_audio_path: str) -> Path:
    return cache_midi_path(maps_root, relative_audio_path).with_suffix(".midi-notes.tsv")


def _pedal_segments_from_controls(instrument: pretty_midi.Instrument, end_time: float) -> list[tuple[float, float]]:
    controls = sorted(
        [cc for cc in instrument.control_changes if cc.number == 64],
        key=lambda cc: cc.time,
    )
    segments = []
    pedal_on_time = None
    for control in controls:
        is_on = control.value >= 64
        if is_on and pedal_on_time is None:
            pedal_on_time = float(control.time)
        elif not is_on and pedal_on_time is not None:
            off_time = float(control.time)
            if off_time >= pedal_on_time:
                segments.append((pedal_on_time, off_time))
            pedal_on_time = None
    if pedal_on_time is not None and end_time >= pedal_on_time:
        segments.append((pedal_on_time, end_time))
    return segments


def compute_pedal_extended_offsets(raw_offsets: np.ndarray, pedals: list[tuple[float, float]]) -> np.ndarray:
    extended_offsets = np.copy(raw_offsets)
    for pedal_start, pedal_end in pedals:
        mask = (raw_offsets >= pedal_start) & (raw_offsets <= pedal_end)
        extended_offsets[mask] = pedal_end
    return extended_offsets


def midi_to_notes_dataframe(midi_path: Path) -> pd.DataFrame:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    rows = []
    pedal_segments = []
    end_time = midi.get_end_time()

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        instrument_pedals = _pedal_segments_from_controls(instrument, end_time)
        pedal_segments.extend(instrument_pedals)

        offsets_truth = np.array([note.end for note in instrument.notes], dtype=float)
        offsets_extended = compute_pedal_extended_offsets(offsets_truth, instrument_pedals)
        for note, offset_truth, offset_extended in zip(instrument.notes, offsets_truth, offsets_extended):
            rows.append(
                {
                    "type": "note",
                    "type_id": 1,
                    "onset_sec": float(note.start),
                    "duration_sec": float(offset_extended - note.start),
                    "offset_sec": float(offset_extended),
                    "duration_sec_truth": float(offset_truth - note.start),
                    "offset_sec_truth": float(offset_truth),
                    "duration_sec_pedal_extended": float(offset_extended - note.start),
                    "offset_sec_pedal_extended": float(offset_extended),
                    "pitch": int(note.pitch),
                    "velocity": int(note.velocity),
                }
            )

    for pedal_start, pedal_end in pedal_segments:
        rows.append(
            {
                "type": "PedalOn",
                "type_id": 0,
                "onset_sec": float(pedal_start),
                "duration_sec": float(pedal_end - pedal_start),
                "offset_sec": float(pedal_end),
                "duration_sec_truth": float(pedal_end - pedal_start),
                "offset_sec_truth": float(pedal_end),
                "duration_sec_pedal_extended": float(pedal_end - pedal_start),
                "offset_sec_pedal_extended": float(pedal_end),
                "pitch": -1,
                "velocity": 0,
            }
        )
        rows.append(
            {
                "type": "PedalOff",
                "type_id": 0,
                "onset_sec": float(pedal_end),
                "duration_sec": 0.0,
                "offset_sec": float(pedal_end),
                "duration_sec_truth": 0.0,
                "offset_sec_truth": float(pedal_end),
                "duration_sec_pedal_extended": 0.0,
                "offset_sec_pedal_extended": float(pedal_end),
                "pitch": -1,
                "velocity": 0,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    df.sort_values(by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True], inplace=True)
    return df.reset_index(drop=True)


def flattened_tsv_to_notes_dataframe(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep=r"\s+",
        comment="#",
        header=None,
        names=["onset_sec", "offset_sec", "pitch", "velocity"],
        engine="python",
    )
    if len(df) == 0:
        return pd.DataFrame(
            columns=[
                "type",
                "type_id",
                "onset_sec",
                "duration_sec",
                "offset_sec",
                "duration_sec_truth",
                "offset_sec_truth",
                "duration_sec_pedal_extended",
                "offset_sec_pedal_extended",
                "pitch",
                "velocity",
            ]
        )

    df = df.astype(
        {
            "onset_sec": float,
            "offset_sec": float,
            "pitch": float,
            "velocity": float,
        }
    )
    durations = np.maximum(df["offset_sec"].to_numpy(dtype=float) - df["onset_sec"].to_numpy(dtype=float), 0.0)
    notes_df = pd.DataFrame(
        {
            "type": "note",
            "type_id": 1,
            "onset_sec": df["onset_sec"].astype(float),
            "duration_sec": durations,
            "offset_sec": df["offset_sec"].astype(float),
            "duration_sec_truth": durations,
            "offset_sec_truth": df["offset_sec"].astype(float),
            "duration_sec_pedal_extended": durations,
            "offset_sec_pedal_extended": df["offset_sec"].astype(float),
            "pitch": df["pitch"].round().astype(int),
            "velocity": df["velocity"].round().clip(0, 127).astype(int),
        }
    )
    notes_df.sort_values(by=["onset_sec", "type_id", "pitch"], ascending=[True, True, True], inplace=True)
    return notes_df.reset_index(drop=True)


def write_note_only_midi(notes_df: pd.DataFrame, midi_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="piano")
    for row in notes_df[notes_df["type"] == "note"].itertuples(index=False):
        start = float(row.onset_sec)
        end = max(float(row.offset_sec_truth), start + 1e-3)
        velocity = int(np.clip(int(row.velocity), 1, 127))
        pitch = int(np.clip(int(row.pitch), 0, 127))
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(midi_path))


def _prepare_cache_one(args):
    item, maps_root = args
    midi_out = cache_midi_path(maps_root, item.relative_audio_path)
    notes_out = cache_notes_path(maps_root, item.relative_audio_path)
    midi_out.parent.mkdir(parents=True, exist_ok=True)
    if item.midi_path is not None:
        shutil.copyfile(item.midi_path, midi_out)
        df = midi_to_notes_dataframe(item.midi_path)
    elif item.tsv_path is not None:
        df = flattened_tsv_to_notes_dataframe(item.tsv_path)
        write_note_only_midi(df, midi_out)
    else:
        raise ValueError(f"MAPS item {item.relative_audio_path} has neither MIDI nor TSV annotations.")
    df.to_csv(notes_out, sep="\t", index=False)
    return item.audio_id


def write_subset_lists(maps_root: Path, items: list[MapsItem]) -> None:
    for subset in ("train", "validation"):
        (maps_root / f"subset_{subset}.tsv").write_text("", encoding="utf-8")
    with (maps_root / "subset_test.tsv").open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item.audio_id}\t{item.relative_audio_path}\n")


def write_metadata(maps_root: Path, items: list[MapsItem]) -> None:
    rows = [
        {
            "audio_id": item.audio_id,
            "split": "test",
            "setting": item.setting,
            "category": item.category,
            "audio_filename": item.relative_audio_path,
            "source_audio_path": str(item.audio_path),
            "source_midi_path": str(item.midi_path) if item.midi_path is not None else "",
            "source_tsv_path": str(item.tsv_path) if item.tsv_path is not None else "",
            "source_format": item.source_format,
            "has_pedals": bool(item.midi_path is not None),
            "cache_midi_path": str(cache_midi_path(maps_root, item.relative_audio_path).relative_to(maps_root)),
            "notes_tsv_path": str(cache_notes_path(maps_root, item.relative_audio_path).relative_to(maps_root)),
        }
        for item in items
    ]
    pd.DataFrame(rows).to_csv(maps_root / "maps_metadata.csv", index=False)


def write_audio_h5(maps_root: Path, items: list[MapsItem]) -> None:
    h5_path = maps_root / "audio.h5"
    with h5py.File(h5_path, "w") as h5:
        for item in tqdm(items, desc="Writing audio.h5"):
            audio, sr = torchaudio.load(str(item.audio_path))
            audio = audio.mean(dim=0)
            if sr != DEFAULT_SAMPLE_RATE:
                audio = Resample(orig_freq=sr, new_freq=DEFAULT_SAMPLE_RATE)(audio.unsqueeze(0)).squeeze(0)
            audio_np = audio.detach().cpu().numpy().astype(np.float32)
            group = h5.create_group(str(item.audio_id))
            group["audio"] = audio_np
            group["path"] = item.relative_audio_path
            group["source_path"] = str(item.audio_path)
            group["sample_rate"] = DEFAULT_SAMPLE_RATE
            group["duration"] = len(audio_np) / DEFAULT_SAMPLE_RATE


def prepare_maps_dataset(
    maps_root: Path,
    raw_root: Path | None = None,
    settings: tuple[str, ...] = ALL_SETTINGS,
    category: str = DEFAULT_CATEGORY,
    jobs: int = 16,
) -> list[MapsItem]:
    maps_root.mkdir(parents=True, exist_ok=True)
    items = discover_maps_items(maps_root, raw_root=raw_root, settings=settings, category=category)
    if not items:
        raise FileNotFoundError(
            f"No MAPS wav/midi pairs found for settings={settings} category={category} under {raw_root or maps_root}."
        )

    write_subset_lists(maps_root, items)
    write_metadata(maps_root, items)
    write_audio_h5(maps_root, items)

    cache_args = [(item, maps_root) for item in items]
    if jobs <= 1:
        for args in tqdm(cache_args, desc="Writing MIDI caches"):
            _prepare_cache_one(args)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            list(tqdm(executor.map(_prepare_cache_one, cache_args), total=len(cache_args), desc="Writing MIDI caches"))
    return items


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare MAPS for Audio2Midi evaluation.")
    parser.add_argument("--maps-root", default="../dataset/MAPS", help="Normalized MAPS dataset output root.")
    parser.add_argument(
        "--raw-root",
        default=None,
        help="Optional raw MAPS root; defaults to <maps-root>/raw, <maps-root>, or the flattened <maps-root>/flac layout.",
    )
    parser.add_argument("--settings", nargs="+", default=list(ALL_SETTINGS), help="MAPS settings to include.")
    parser.add_argument("--category", default=DEFAULT_CATEGORY, help="MAPS category to include, e.g. MUS.")
    parser.add_argument("--jobs", type=int, default=16, help="Parallel MIDI cache workers.")
    return parser.parse_args()


def main():
    args = parse_args()
    maps_root = Path(args.maps_root).expanduser().resolve()
    raw_root = Path(args.raw_root).expanduser().resolve() if args.raw_root else None
    items = prepare_maps_dataset(
        maps_root=maps_root,
        raw_root=raw_root,
        settings=tuple(args.settings),
        category=args.category,
        jobs=args.jobs,
    )
    print(f"Prepared {len(items)} MAPS recordings at {maps_root}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
