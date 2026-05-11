from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _get_separator():
    try:
        from spleeter.separator import Separator
    except ImportError as exc:
        raise RuntimeError(
            "Spleeter is not installed. Install requirements/separation.txt or build the backend image "
            "with INSTALL_SPLEETER=true to enable stem separation."
        ) from exc

    # Use 5 stems so piano.wav can be produced.
    return Separator("spleeter:5stems")


def run_spleeter(input_audio_path: str, output_root: str = "separated") -> dict:
    input_path = Path(input_audio_path)
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spleeter creates a subfolder named after the input file stem.
    _get_separator().separate_to_file(str(input_path), str(output_dir))

    song_folder = output_dir / input_path.stem

    stems = {
        "vocals": song_folder / "vocals.wav",
        "drums": song_folder / "drums.wav",
        "bass": song_folder / "bass.wav",
        "piano": song_folder / "piano.wav",
        "other": song_folder / "other.wav",
    }

    existing_stems = {name: str(path) for name, path in stems.items() if path.exists()}
    return existing_stems
