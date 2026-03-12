"""
paper_time_alignment.py

Paper-based note-level alignment using the ASAP/parangonar stack.

Default model:
- parangonar.AutomaticNoteMatcher
  (the paper's best fully automatic model, hDTW+sym)

This module is intentionally API-compatible with analyzer.py's
run_alignment() expectation.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import partitura as pt  # type: ignore
except Exception:
    pt = None

try:
    import parangonar as pa  # type: ignore
except Exception:
    pa = None


class PaperBestTimeAlignment:
    """
    Note-level alignment based on parangonar.

    For MIDI-to-MIDI input we build:
    - pseudo-score note array from reference performance MIDI
    - performance note array from performed MIDI
    """

    def __init__(
        self,
        reference_midi_path: str,
        performance_midi_path: str,
        *,
        model: str = "automatic_hdtw_sym",
    ) -> None:
        self.reference_midi_path = reference_midi_path
        self.performance_midi_path = performance_midi_path
        self.model = model

        self.aligned_pairs: List[Dict[str, Any]] = []
        self._score_na: Optional[np.ndarray] = None
        self._ref_perf_na: Optional[np.ndarray] = None
        self._perf_na: Optional[np.ndarray] = None
        self._score_index_by_id: Dict[str, int] = {}
        self._perf_index_by_id: Dict[str, int] = {}
        self.quality: Dict[str, Any] = {}

    # -----------------------------
    # Public API
    # -----------------------------

    def run_alignment(self) -> Dict[str, Any]:
        self._require_dependencies()
        score_na, perf_na = self._build_matcher_inputs()
        matcher, model_name = self._build_matcher()

        raw_alignment = list(matcher(score_na, perf_na))
        self.aligned_pairs = self._convert_alignment(raw_alignment)

        report = {
            "alignment_type": "paper_parangonar_automatic_note_matching",
            "quality": {
                "library": "parangonar",
                "model_requested": self.model,
                "model_used": model_name,
                "reference_notes": int(len(score_na)),
                "performance_notes": int(len(perf_na)),
                "raw_alignment_items": int(len(raw_alignment)),
            },
            "selected_hypothesis": model_name,
            "statistics": self.get_alignment_statistics(),
            "aligned_pairs": self.aligned_pairs,
            "repeat_jumps": {},
            "global_warping_path_sample": None,
        }
        return report

    def export_alignment_report(self, output_file: str) -> Dict[str, Any]:
        import json

        report = self.run_alignment()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        return report

    def get_alignment_statistics(self) -> Dict[str, Any]:
        pairs = self.aligned_pairs or []
        aligned = [p for p in pairs if p.get("error_type") == "none"]
        missing = [p for p in pairs if p.get("error_type") == "missing_note"]
        extra = [p for p in pairs if p.get("error_type") in ("extra_note", "ornament_insertion")]

        n_ref = len(self._score_na) if self._score_na is not None else 0
        n_perf = len(self._perf_na) if self._perf_na is not None else 0
        min_reliable_aligned = max(10, int(0.05 * max(1, n_ref)))
        is_reliable = len(aligned) >= min_reliable_aligned

        stats: Dict[str, Any] = {
            "total_reference_notes": int(n_ref),
            "total_performance_notes": int(n_perf),
            "aligned_notes": int(len(aligned)),
            "missing_notes": int(len(missing)),
            "extra_notes": int(len(extra)),
            "alignment_rate": float(len(aligned) / max(1, n_ref)),
            "reliability": {
                "is_reliable": bool(is_reliable),
                "insufficient_alignment": bool(not is_reliable),
                "aligned_note_count": int(len(aligned)),
                "minimum_required_aligned_notes": int(min_reliable_aligned),
                "reason": (
                    None
                    if is_reliable
                    else "too_few_aligned_pairs_for_stable_timing_and_pitch_metrics"
                ),
            },
        }
        if aligned:
            td = np.array([abs(float(p["time_difference"])) for p in aligned], dtype=float)
            pd = np.array([abs(int(p["pitch_difference"])) for p in aligned], dtype=float)
            cf = np.array([float(p.get("alignment_confidence", 0.0)) for p in aligned], dtype=float)
            stats["timing_accuracy"] = {
                "mean_abs_error": float(np.mean(td)),
                "std_abs_error": float(np.std(td)),
                "max_abs_error": float(np.max(td)),
                "rushing_tendency": float(np.mean([p["time_difference"] < -0.1 for p in aligned])),
                "dragging_tendency": float(np.mean([p["time_difference"] > 0.1 for p in aligned])),
            }
            stats["pitch_accuracy"] = {
                "notes_with_pitch_error": int(np.sum(pd > 0)),
                "mean_abs_pitch_error": float(np.mean(pd)),
            }
            stats["confidence"] = {"mean": float(np.mean(cf)), "min": float(np.min(cf))}
        return stats

    # -----------------------------
    # Data preparation
    # -----------------------------

    def _require_dependencies(self) -> None:
        if pt is None or pa is None:
            raise ImportError(
                "Paper backend requires 'partitura' and 'parangonar'. "
                "Install with: pip install partitura parangonar"
            )

    def _build_matcher_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        assert pt is not None

        ref_perf = pt.load_performance_midi(self.reference_midi_path)
        perf_perf = pt.load_performance_midi(self.performance_midi_path)

        ref_na = self._normalize_performance_note_array(ref_perf.note_array(), prefix="r")
        perf_na = self._normalize_performance_note_array(perf_perf.note_array(), prefix="p")
        score_na = self._pseudo_score_from_reference(ref_na)

        self._score_na = score_na
        self._ref_perf_na = ref_na
        self._perf_na = perf_na
        self._score_index_by_id = {str(score_na[i]["id"]): i for i in range(len(score_na))}
        self._perf_index_by_id = {str(perf_na[i]["id"]): i for i in range(len(perf_na))}

        return score_na, perf_na

    def _normalize_performance_note_array(self, na: np.ndarray, *, prefix: str) -> np.ndarray:
        size = len(na)
        dtype = [
            ("onset_sec", "f8"),
            ("duration_sec", "f8"),
            ("pitch", "i4"),
            ("velocity", "i4"),
            ("track", "i4"),
            ("channel", "i4"),
            ("id", "U256"),
        ]
        out = np.zeros(size, dtype=dtype)
        out["onset_sec"] = self._field_or_default(na, "onset_sec", 0.0)
        out["duration_sec"] = self._field_or_default(na, "duration_sec", 0.0)
        out["pitch"] = self._field_or_default(na, "pitch", 0).astype(int)
        out["velocity"] = self._field_or_default(na, "velocity", 64).astype(int)
        out["track"] = self._field_or_default(na, "track", -1).astype(int)
        out["channel"] = self._field_or_default(na, "channel", -1).astype(int)
        out["id"] = self._extract_ids(na, prefix=prefix)
        return out

    def _pseudo_score_from_reference(self, ref_perf_na: np.ndarray) -> np.ndarray:
        dtype = [
            ("onset_beat", "f8"),
            ("duration_beat", "f8"),
            ("pitch", "i4"),
            ("id", "U256"),
        ]
        out = np.zeros(len(ref_perf_na), dtype=dtype)
        # For MIDI-to-MIDI matching we reuse seconds as the score time axis.
        out["onset_beat"] = ref_perf_na["onset_sec"]
        out["duration_beat"] = ref_perf_na["duration_sec"]
        out["pitch"] = ref_perf_na["pitch"]
        out["id"] = ref_perf_na["id"]
        return out

    @staticmethod
    def _field_or_default(arr: np.ndarray, field: str, default: Any) -> np.ndarray:
        if field in arr.dtype.names:
            return np.asarray(arr[field])
        return np.full(len(arr), default)

    def _extract_ids(self, arr: np.ndarray, *, prefix: str) -> np.ndarray:
        if "id" in arr.dtype.names:
            ids = [self._to_id_str(v, fallback=f"{prefix}{i}") for i, v in enumerate(arr["id"])]
            return np.asarray(ids, dtype="U256")
        return np.asarray([f"{prefix}{i}" for i in range(len(arr))], dtype="U256")

    @staticmethod
    def _to_id_str(value: Any, fallback: str) -> str:
        if value is None:
            return fallback
        if isinstance(value, bytes):
            try:
                return value.decode("utf-8")
            except Exception:
                return fallback
        txt = str(value)
        if not txt or txt.lower() == "none":
            return fallback
        return txt

    # -----------------------------
    # Matcher selection
    # -----------------------------

    def _build_matcher(self) -> Tuple[Any, str]:
        assert pa is not None

        key = self.model.strip().lower()
        if key in {"automatic_hdtw_sym", "paper_best", "automatic", "hdtw+sym"}:
            cls = getattr(pa, "AutomaticNoteMatcher", None)
            if cls is None:
                raise AttributeError("parangonar.AutomaticNoteMatcher is not available in this version.")
            return cls(), "AutomaticNoteMatcher(hDTW+sym)"

        if key in {"dualdtw", "dual_dtw", "dualdtwnotematcher"}:
            cls = getattr(pa, "DualDTWNoteMatcher", None)
            if cls is None:
                raise AttributeError("parangonar.DualDTWNoteMatcher is not available in this version.")
            return cls(), "DualDTWNoteMatcher"

        if key in {"theglue", "glue", "thegluenotematcher"}:
            cls = getattr(pa, "TheGlueNoteMatcher", None)
            if cls is None:
                raise AttributeError("parangonar.TheGlueNoteMatcher is not available in this version.")
            return cls(), "TheGlueNoteMatcher"

        if key in {"anchorpoint", "anchor_point", "anchorpointnotematcher"}:
            cls = getattr(pa, "AnchorPointNoteMatcher", None)
            if cls is None:
                raise AttributeError("parangonar.AnchorPointNoteMatcher is not available in this version.")
            return cls(), "AnchorPointNoteMatcher"

        raise ValueError(f"Unsupported paper alignment model: {self.model}")

    # -----------------------------
    # Output conversion
    # -----------------------------

    def _convert_alignment(self, raw_alignment: Iterable[Any]) -> List[Dict[str, Any]]:
        assert self._score_na is not None and self._ref_perf_na is not None and self._perf_na is not None

        out: List[Dict[str, Any]] = []
        used_score_idx: set[int] = set()
        used_perf_idx: set[int] = set()

        for item in raw_alignment:
            label, score_id, perf_id, score_idx, perf_idx = self._parse_alignment_item(item)

            s_idx = self._resolve_score_index(score_id, score_idx)
            p_idx = self._resolve_perf_index(perf_id, perf_idx)
            if s_idx is not None:
                used_score_idx.add(s_idx)
            if p_idx is not None:
                used_perf_idx.add(p_idx)

            ref_note = self._reference_note_dict(s_idx)
            perf_note = self._performance_note_dict(p_idx)

            if label == "match":
                if ref_note is not None and perf_note is not None:
                    out.append(self._emit_match(ref_note, perf_note))
                elif ref_note is not None:
                    out.append(self._emit_missing(ref_note, "paper_match_missing_perf"))
                elif perf_note is not None:
                    out.append(self._emit_extra(perf_note, "paper_match_missing_ref"))
                continue

            if label == "deletion":
                if ref_note is not None:
                    out.append(self._emit_missing(ref_note, "paper_deletion"))
                continue

            if label == "insertion":
                if perf_note is not None:
                    out.append(self._emit_extra(perf_note, "paper_insertion"))
                continue

            if label == "ornament":
                if perf_note is not None:
                    out.append(self._emit_extra(perf_note, "paper_ornament", error_type="ornament_insertion"))
                continue

            # Fallback for unknown labels
            if ref_note is not None and perf_note is not None:
                out.append(self._emit_match(ref_note, perf_note))
            elif ref_note is not None:
                out.append(self._emit_missing(ref_note, f"paper_unknown_label:{label}"))
            elif perf_note is not None:
                out.append(self._emit_extra(perf_note, f"paper_unknown_label:{label}"))

        for idx in range(len(self._score_na)):
            if idx in used_score_idx:
                continue
            ref_note = self._reference_note_dict(idx)
            if ref_note is not None:
                out.append(self._emit_missing(ref_note, "paper_uncovered_score_note"))

        for idx in range(len(self._perf_na)):
            if idx in used_perf_idx:
                continue
            perf_note = self._performance_note_dict(idx)
            if perf_note is not None:
                out.append(self._emit_extra(perf_note, "paper_uncovered_performance_note"))

        return out

    @staticmethod
    def _parse_alignment_item(item: Any) -> Tuple[str, Optional[str], Optional[str], Optional[int], Optional[int]]:
        if isinstance(item, dict):
            label = str(item.get("label", "match")).lower()
            score_id = item.get("score_id")
            perf_id = item.get("performance_id")
            score_idx = item.get("score_idx", item.get("score_index"))
            perf_idx = item.get("performance_idx", item.get("performance_index"))
            return (
                label,
                str(score_id) if score_id is not None else None,
                str(perf_id) if perf_id is not None else None,
                int(score_idx) if isinstance(score_idx, (int, np.integer)) else None,
                int(perf_idx) if isinstance(perf_idx, (int, np.integer)) else None,
            )
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            a, b = item[0], item[1]
            ai = int(a) if isinstance(a, (int, np.integer)) else None
            bi = int(b) if isinstance(b, (int, np.integer)) else None
            return "match", None, None, ai, bi
        return "unknown", None, None, None, None

    def _resolve_score_index(self, score_id: Optional[str], score_idx: Optional[int]) -> Optional[int]:
        if score_id is not None and score_id in self._score_index_by_id:
            return self._score_index_by_id[score_id]
        if score_idx is not None and self._score_na is not None and 0 <= score_idx < len(self._score_na):
            return score_idx
        return None

    def _resolve_perf_index(self, perf_id: Optional[str], perf_idx: Optional[int]) -> Optional[int]:
        if perf_id is not None and perf_id in self._perf_index_by_id:
            return self._perf_index_by_id[perf_id]
        if perf_idx is not None and self._perf_na is not None and 0 <= perf_idx < len(self._perf_na):
            return perf_idx
        return None

    def _reference_note_dict(self, score_idx: Optional[int]) -> Optional[Dict[str, Any]]:
        if score_idx is None or self._score_na is None or self._ref_perf_na is None:
            return None
        s = self._score_na[score_idx]
        r = self._ref_perf_na[score_idx]
        onset = float(s["onset_beat"])
        dur = max(0.0, float(s["duration_beat"]))
        return {
            "pitch": int(s["pitch"]),
            "start": onset,
            "end": onset + dur,
            "velocity": int(r["velocity"]),
            "duration": dur,
            "track_id": int(r["track"]) if int(r["track"]) >= 0 else None,
            "instrument": str(int(r["channel"])) if int(r["channel"]) >= 0 else None,
        }

    def _performance_note_dict(self, perf_idx: Optional[int]) -> Optional[Dict[str, Any]]:
        if perf_idx is None or self._perf_na is None:
            return None
        p = self._perf_na[perf_idx]
        onset = float(p["onset_sec"])
        dur = max(0.0, float(p["duration_sec"]))
        return {
            "pitch": int(p["pitch"]),
            "start": onset,
            "end": onset + dur,
            "velocity": int(p["velocity"]),
            "duration": dur,
            "track_id": int(p["track"]) if int(p["track"]) >= 0 else None,
            "instrument": str(int(p["channel"])) if int(p["channel"]) >= 0 else None,
        }

    @staticmethod
    def _emit_match(ref_note: Dict[str, Any], perf_note: Dict[str, Any]) -> Dict[str, Any]:
        onset_diff = float(perf_note["start"]) - float(ref_note["start"])
        pitch_diff = int(perf_note["pitch"]) - int(ref_note["pitch"])
        velocity_diff = int(perf_note["velocity"]) - int(ref_note["velocity"])
        cost = abs(onset_diff) * 3.0 + abs(pitch_diff) / 12.0 * 2.0
        return {
            "reference_note": ref_note,
            "performance_note": perf_note,
            "time_difference": onset_diff,
            "pitch_difference": pitch_diff,
            "velocity_difference": velocity_diff,
            "alignment_confidence": float(1.0 / (1.0 + cost)),
            "error_type": "none",
            "match_level": "paper_best_model",
        }

    @staticmethod
    def _emit_missing(ref_note: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "reference_note": ref_note,
            "performance_note": None,
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": "missing_note",
            "reason": reason,
        }

    @staticmethod
    def _emit_extra(perf_note: Dict[str, Any], reason: str, error_type: str = "extra_note") -> Dict[str, Any]:
        return {
            "reference_note": None,
            "performance_note": perf_note,
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": error_type,
            "reason": reason,
        }


def align_midi_files_paper_best(
    reference_path: str,
    performance_path: str,
    output_file: Optional[str] = None,
    *,
    model: str = "automatic_hdtw_sym",
) -> Dict[str, Any]:
    aligner = PaperBestTimeAlignment(reference_path, performance_path, model=model)
    report = aligner.run_alignment()
    if output_file:
        import json

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    return report
