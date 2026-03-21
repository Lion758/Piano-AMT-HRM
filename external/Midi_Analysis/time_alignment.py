"""
time_alignment.py (MIDI-only, repeat-inferred)

Implements a hierarchical, music-focused alignment pipeline WITHOUT explicit score repeat directives.

Key idea for "repeat awareness" when you only have MIDI:
- Segment reference into phrases (gaps / rests)
- Compute phrase fingerprints (pitch-class histogram + energy/density stats)
- Detect repeated phrases/sections via self-similarity
- Allow backward jumps ONLY at phrase boundaries via a small beam of timeline hypotheses
- For each phrase under each hypothesis: compute a local time map (frame DTW)
- Within phrase: DP align event clusters, then Hungarian within chord clusters
- Post-process: monotonicity, reject absurd, crossing repair, ornament insertion labeling

This is a pragmatic way to approximate written repeats using only MIDI content.

Optional deps:
- fastdtw (speed; falls back to classic DTW)
- scipy.optimize.linear_sum_assignment (Hungarian; falls back to greedy)
- pretty_midi (if you pass PrettyMIDI objects or use align_midi_files wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import numpy as np

try:
    from fastdtw import fastdtw  # type: ignore
except Exception:
    fastdtw = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None

try:
    import pretty_midi  # type: ignore
except Exception:
    pretty_midi = None


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class NoteEvent:
    onset: float
    offset: float
    duration: float
    pitch: int
    velocity: int
    track_id: Optional[int] = None
    instrument: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "NoteEvent":
        onset = float(d.get("start", 0.0))
        offset = float(d.get("end", onset))
        dur = float(d.get("duration", max(0.0, offset - onset)))
        pitch = int(d.get("pitch", 0))
        vel = int(d.get("velocity", 64))
        return NoteEvent(onset, offset, dur, pitch, vel, d.get("track_id"), d.get("instrument"))

    @staticmethod
    def from_pretty_midi(pm: Any) -> List["NoteEvent"]:
        notes: List[NoteEvent] = []
        for tid, inst in enumerate(pm.instruments):
            inst_id = str(getattr(inst, "program", ""))
            for n in inst.notes:
                onset = float(n.start)
                offset = float(n.end)
                dur = float(max(0.0, offset - onset))
                pitch = int(n.pitch)
                vel = int(getattr(n, "velocity", 64))
                notes.append(NoteEvent(onset, offset, dur, pitch, vel, tid, inst_id))
        notes.sort(key=lambda x: (x.onset, x.pitch))
        return notes


@dataclass
class EventCluster:
    onset: float
    notes: List[NoteEvent]

    @property
    def density(self) -> int:
        return len(self.notes)

    @property
    def pitch_classes(self) -> List[int]:
        return sorted({n.pitch % 12 for n in self.notes})

    @property
    def pitch_centroid(self) -> float:
        return float(np.mean([n.pitch for n in self.notes])) if self.notes else 0.0

    @property
    def velocity_mean(self) -> float:
        return float(np.mean([n.velocity for n in self.notes])) if self.notes else 0.0


@dataclass
class TimeMap:
    """Piecewise linear map t_perf ~= f(t_ref)."""
    t_ref: np.ndarray
    t_perf: np.ndarray

    def __post_init__(self) -> None:
        assert len(self.t_ref) == len(self.t_perf) >= 2
        order = np.argsort(self.t_ref)
        self.t_ref = self.t_ref[order]
        self.t_perf = self.t_perf[order]
        for i in range(1, len(self.t_ref)):
            if self.t_ref[i] <= self.t_ref[i - 1]:
                self.t_ref[i] = self.t_ref[i - 1] + 1e-6

    def __call__(self, t: float) -> float:
        if t <= self.t_ref[0]:
            return float(self.t_perf[0] + (t - self.t_ref[0]) * self._slope0())
        if t >= self.t_ref[-1]:
            return float(self.t_perf[-1] + (t - self.t_ref[-1]) * self._slopen())
        return float(np.interp(t, self.t_ref, self.t_perf))

    def _slope0(self) -> float:
        dt = self.t_ref[1] - self.t_ref[0]
        return float((self.t_perf[1] - self.t_perf[0]) / dt) if abs(dt) > 1e-9 else 1.0

    def _slopen(self) -> float:
        dt = self.t_ref[-1] - self.t_ref[-2]
        return float((self.t_perf[-1] - self.t_perf[-2]) / dt) if abs(dt) > 1e-9 else 1.0


# -----------------------------
# Repeat-inferred timeline hypotheses
# -----------------------------

@dataclass
class TimelineHypothesis:
    """
    A hypothesis is a sequence of phrase indices in reference order that we 'traverse' in time.
    With MIDI-only repeats, we allow backward jumps ONLY at phrase boundaries based on repeat similarity.
    """
    phrase_order: List[int]
    cost: float = 0.0
    alive: bool = True
    name: str = "H"


MidiLike = Union[Dict[str, Any], Any]


class TimeAlignment:
    def __init__(
        self,
        reference: MidiLike,
        performance: MidiLike,
        *,
        onset_cluster_ms: float = 50.0,
        frame_hop: float = 0.05,
        gap_phrase_s: float = 0.8,
        gap_section_s: float = 2.0,
        soft_first_anchor_max_s: float = 0.5,
        max_hypotheses: int = 6,
        max_repeat_jumps_per_phrase: int = 2,
    ) -> None:
        self.reference = reference
        self.performance = performance

        self.onset_cluster_s = onset_cluster_ms / 1000.0
        self.frame_hop = frame_hop
        self.gap_phrase_s = gap_phrase_s
        self.gap_section_s = gap_section_s
        self.soft_first_anchor_max_s = soft_first_anchor_max_s

        self.max_hypotheses = max_hypotheses
        self.max_repeat_jumps_per_phrase = max_repeat_jumps_per_phrase

        # Cache
        self.ref_notes: List[NoteEvent] = []
        self.perf_notes: List[NoteEvent] = []
        self.ref_events: List[EventCluster] = []
        self.perf_events: List[EventCluster] = []

        # Outputs
        self.global_time_map: Optional[TimeMap] = None
        self.global_path: Optional[List[Tuple[int, int]]] = None
        self.quality: Dict[str, Any] = {}

        self.phrases: List[Tuple[float, float]] = []                 # list of (t0,t1) in reference time
        self.phrase_fingerprints: np.ndarray = np.zeros((0, 0))      # (P, D)
        #self.repeat_jumps: Dict[int, List[int]] = {}                 # phrase i -> list of earlier phrase indices

        self.hypotheses: List[TimelineHypothesis] = []
        self.selected_hypothesis: Optional[TimelineHypothesis] = None

        self.phrase_time_maps: Dict[Tuple[int, int], TimeMap] = {}   # (phrase_idx in order, perf_window_id) -> fk
        self.phrase_choice: Dict[int, str] = {}                      # phrase index -> hypothesis name used

        self.aligned_pairs: List[Dict[str, Any]] = []

    # -----------------------------
    # Public API
    # -----------------------------

    def run_alignment(self) -> Dict[str, Any]:
        # Stage 0
        self.ref_notes = self._get_notes(self.reference)
        self.perf_notes = self._get_notes(self.performance)
        self.ref_events = self._cluster_events(self.ref_notes)
        self.perf_events = self._cluster_events(self.perf_notes)

        # Deterministic fast path: if reference/performance are effectively the same note stream,
        # emit exact 1:1 matches and skip heuristic alignment stages.
        if self._notes_are_identical(self.ref_notes, self.perf_notes):
            self.global_time_map = self._identity_time_map()
            self.global_path = [(i, i) for i in range(min(len(self.ref_notes), len(self.perf_notes)))]
            self.quality["global"] = {"method": "identity_fast_path", "exact_match": True}
            self.quality["anchors"] = {"count": 0}
            self.quality["hypothesis_selection"] = {"num_hypotheses": 1, "best": "H_identity"}
            self.selected_hypothesis = TimelineHypothesis(phrase_order=[0], cost=0.0, alive=True, name="H_identity")
            self.hypotheses = [self.selected_hypothesis]
            self.phrases = self._segment_phrases([])
            self.repeat_jumps = {}
            self.aligned_pairs = self._build_exact_aligned_pairs(self.ref_notes, self.perf_notes)
            report = {
                "alignment_type": "midi_only_repeat_inferred_hierarchical",
                "quality": self.quality,
                "selected_hypothesis": self.selected_hypothesis.name,
                "hypotheses": [{"name": self.selected_hypothesis.name, "cost": 0.0, "phrase_order": [0]}],
                "statistics": self.get_alignment_statistics(),
                "aligned_pairs": self.aligned_pairs,
                "global_warping_path_sample": self._sample_path(self.global_path),
                "repeat_jumps": {},
            }
            return report

        # Level 1 Stage 1: global map from frame DTW
        self.global_time_map, self.global_path, qg = self._compute_global_time_map()
        self.quality["global"] = qg

        # Stage 2: soft first anchor
        anchors = self._compute_soft_anchors()

        # Stage 3: sections + structural anchors
        sections, structural_anchors = self._detect_sections_and_structural_anchors()
        anchors.extend(structural_anchors)
        self.quality["anchors"] = {"count": len(anchors)}

        # Level 2 Stage 4: phrase segmentation (on reference)
        self.phrases = self._segment_phrases(sections)

        # NEW: MIDI-only repeat inference from phrase fingerprints
        self.phrase_fingerprints = self._compute_phrase_fingerprints()
        self.repeat_jumps = self._infer_repeat_jumps(self.phrase_fingerprints)

        # Stage 3.5 replacement: timeline hypotheses over phrase indices
        self.hypotheses = self._build_timeline_hypotheses()
        self._beam_select_hypothesis(anchors)

        # Stage 5: compute phrase maps and do matching under selected hypothesis
        self.aligned_pairs = self._match_under_selected_hypothesis(anchors)

        # Level 4: post-processing
        self.aligned_pairs = self._post_process(self.aligned_pairs)

        report = {
            "alignment_type": "midi_only_repeat_inferred_hierarchical",
            "quality": self.quality,
            "selected_hypothesis": self.selected_hypothesis.name if self.selected_hypothesis else None,
            "hypotheses": [{"name": h.name, "cost": float(h.cost), "phrase_order": h.phrase_order} for h in self.hypotheses],
            "statistics": self.get_alignment_statistics(),
            "aligned_pairs": self.aligned_pairs,
            "global_warping_path_sample": self._sample_path(self.global_path),
            "repeat_jumps": {int(k): [int(x) for x in v] for k, v in self.repeat_jumps.items()},
        }
        return report

    def export_alignment_report(self, output_file: str) -> Dict[str, Any]:
        import json
        report = self.run_alignment()
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return report

    def get_alignment_statistics(self) -> Dict[str, Any]:
        pairs = self.aligned_pairs or []
        aligned = [p for p in pairs if p.get("error_type") == "none"]
        missing = [p for p in pairs if p.get("error_type") == "missing_note"]
        extra = [p for p in pairs if p.get("error_type") in ("extra_note", "ornament_insertion")]

        min_reliable_aligned = max(10, int(0.05 * max(1, len(self.ref_notes))))
        is_reliable = len(aligned) >= min_reliable_aligned

        stats: Dict[str, Any] = {
            "total_reference_notes": int(len(self.ref_notes)),
            "total_performance_notes": int(len(self.perf_notes)),
            "aligned_notes": int(len(aligned)),
            "missing_notes": int(len(missing)),
            "extra_notes": int(len(extra)),
            "alignment_rate": float(len(aligned) / max(1, len(self.ref_notes))),
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
    # Stage 0: parse + cluster
    # -----------------------------

    def _get_notes(self, midi_like: MidiLike) -> List[NoteEvent]:
        if isinstance(midi_like, dict) and "notes" in midi_like:
            notes = [NoteEvent.from_dict(n) for n in midi_like.get("notes", [])]
            notes.sort(key=lambda x: (x.onset, x.pitch))
            return notes
        if pretty_midi is not None and hasattr(midi_like, "instruments"):
            return NoteEvent.from_pretty_midi(midi_like)
        raise ValueError("Unsupported input type; pass parsed dict with 'notes' or PrettyMIDI.")

    def _cluster_events(self, notes: List[NoteEvent]) -> List[EventCluster]:
        if not notes:
            return []
        clusters: List[EventCluster] = []
        cur: List[NoteEvent] = [notes[0]]
        cur_t = notes[0].onset
        for n in notes[1:]:
            if (n.onset - cur_t) <= self.onset_cluster_s:
                cur.append(n)
            else:
                clusters.append(EventCluster(cur_t, cur))
                cur = [n]
                cur_t = n.onset
        clusters.append(EventCluster(cur_t, cur))
        return clusters

    # -----------------------------
    # Frame features + DTW
    # -----------------------------

    def _frame_features(self, notes: List[NoteEvent], hop: float) -> Tuple[np.ndarray, np.ndarray]:
        if not notes:
            return np.array([0.0]), np.zeros((1, 16), dtype=float)

        t_end = max(n.offset for n in notes)
        T = max(1, int(math.ceil(t_end / hop)) + 1)
        times = np.arange(T, dtype=float) * hop

        onset = np.array([n.onset for n in notes], dtype=float)
        offset = np.array([n.offset for n in notes], dtype=float)
        pitch = np.array([n.pitch for n in notes], dtype=float)
        vel = np.array([n.velocity for n in notes], dtype=float)

        activity = np.zeros(T, dtype=float)
        onset_count = np.zeros(T, dtype=float)
        pitch_centroid = np.zeros(T, dtype=float)
        vel_energy = np.zeros(T, dtype=float)
        pc_hist = np.zeros((T, 12), dtype=float)

        for i, t in enumerate(times):
            t0, t1 = t, t + hop
            active = (onset <= t) & (offset > t)
            starts = (onset >= t0) & (onset < t1)
            a = np.where(active)[0]
            s = np.where(starts)[0]

            activity[i] = float(len(a))
            onset_count[i] = float(len(s))
            if len(a) > 0:
                pitch_centroid[i] = float(np.mean(pitch[a]))
                vel_energy[i] = float(np.mean(vel[a]))
                pcs = (pitch[a].astype(int) % 12)
                for pc in pcs:
                    pc_hist[i, pc] += 1.0
                sm = pc_hist[i].sum()
                if sm > 0:
                    pc_hist[i] /= sm

        feat = np.column_stack([
            np.log1p(activity),
            np.log1p(onset_count),
            pitch_centroid / 127.0,
            vel_energy / 127.0,
            pc_hist
        ])
        return times, feat

    def _dtw(self, X: np.ndarray, Y: np.ndarray, use_fast: bool = True, radius: int = 2) -> Tuple[float, List[Tuple[int, int]]]:
        def euclid(a: np.ndarray, b: np.ndarray) -> float:
            d = a - b
            return float(np.sqrt(np.dot(d, d)))

        if fastdtw is not None and use_fast:
            dist, path = fastdtw(X, Y, radius=radius, dist=euclid)  # type: ignore
            return float(dist), [(int(i), int(j)) for i, j in path]

        N, M = len(X), len(Y)
        D = np.full((N + 1, M + 1), np.inf, dtype=float)
        D[0, 0] = 0.0
        bp = np.zeros((N, M), dtype=np.uint8)  # 0 diag, 1 up, 2 left

        for i in range(N):
            for j in range(M):
                c = euclid(X[i], Y[j])
                diag = D[i, j]
                up = D[i, j + 1]
                left = D[i + 1, j]
                if diag <= up and diag <= left:
                    D[i + 1, j + 1] = c + diag
                    bp[i, j] = 0
                elif up <= left:
                    D[i + 1, j + 1] = c + up
                    bp[i, j] = 1
                else:
                    D[i + 1, j + 1] = c + left
                    bp[i, j] = 2

        i, j = N - 1, M - 1
        path = [(i, j)]
        while i > 0 or j > 0:
            mv = bp[i, j]
            if mv == 0 and i > 0 and j > 0:
                i -= 1; j -= 1
            elif mv == 1 and i > 0:
                i -= 1
            elif mv == 2 and j > 0:
                j -= 1
            else:
                if i > 0: i -= 1
                elif j > 0: j -= 1
            path.append((i, j))
        path.reverse()
        return float(D[N, M]), path

    def _refine_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not path:
            return []
        out = [path[0]]
        pi, pj = path[0]
        for i, j in path[1:]:
            if i >= pi and j >= pj and (i > pi or j > pj):
                out.append((i, j))
                pi, pj = i, j
        return out

    def _compute_global_time_map(self) -> Tuple[TimeMap, List[Tuple[int, int]], Dict[str, Any]]:
        tR, XR = self._frame_features(self.ref_notes, self.frame_hop)
        tP, YP = self._frame_features(self.perf_notes, self.frame_hop)

        dist, path = self._dtw(XR, YP, use_fast=True)
        path = self._refine_path(path)

        ref_k = np.array([tR[i] for i, _ in path], dtype=float)
        perf_k = np.array([tP[j] for _, j in path], dtype=float)

        step = max(1, len(ref_k) // 250)
        ref_k = ref_k[::step]
        perf_k = perf_k[::step]
        if len(ref_k) < 2:
            ref_k = np.array([tR[0], tR[-1]], dtype=float)
            perf_k = np.array([tP[0], tP[-1]], dtype=float)

        tmap = TimeMap(ref_k, perf_k)

        sample = path[::max(1, len(path)//400)]
        local = []
        for i, j in sample:
            d = XR[i] - YP[j]
            local.append(float(np.sqrt(np.dot(d, d))))
        local = np.array(local, dtype=float) if local else np.array([0.0])

        q = {
            "dtw_distance": float(dist),
            "path_length": int(len(path)),
            "mean_local_cost": float(np.mean(local)),
            "max_local_cost": float(np.max(local)),
            "frame_hop": float(self.frame_hop),
        }
        return tmap, path, q

    # -----------------------------
    # Anchors / sections / phrases
    # -----------------------------

    def _compute_soft_anchors(self) -> List[Tuple[float, float, str]]:
        anchors: List[Tuple[float, float, str]] = []
        if not self.global_time_map:
            return anchors
        if self.ref_events and self.perf_events:
            t0 = self.ref_events[0].onset
            pred = self.global_time_map(t0)
            early = self.perf_events[:min(25, len(self.perf_events))]
            nearest = min(early, key=lambda e: abs(e.onset - pred))
            if abs(nearest.onset - pred) <= self.soft_first_anchor_max_s:
                anchors.append((t0, nearest.onset, "start_soft"))
        return anchors

    def _largest_gaps(self, events: List[EventCluster], threshold: float, topk: int = 10) -> List[Tuple[float, float]]:
        gaps = []
        for a, b in zip(events, events[1:]):
            g = b.onset - a.onset
            if g >= threshold:
                gaps.append((a.onset, g))
        gaps.sort(key=lambda x: x[1], reverse=True)
        return gaps[:topk]

    def _detect_sections_and_structural_anchors(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, str]]]:
        if not self.ref_events or not self.global_time_map:
            return [(0.0, max([n.offset for n in self.ref_notes], default=0.0))], []

        anchors: List[Tuple[float, float, str]] = []
        onsets = [e.onset for e in self.ref_events]
        boundaries = [onsets[0]]

        # Rest-gap anchors
        for t_gap, _g in self._largest_gaps(self.ref_events, self.gap_section_s, topk=12):
            idx = next((i for i, e in enumerate(self.ref_events) if e.onset > t_gap), None)
            if idx is None:
                continue
            boundaries.append(self.ref_events[idx].onset)
            anchors.append((t_gap, self.global_time_map(t_gap), "rest_gap_start"))
            anchors.append((self.ref_events[idx].onset, self.global_time_map(self.ref_events[idx].onset), "rest_gap_reentry"))

        boundaries.append(onsets[-1] + 1e-3)
        boundaries = sorted(set(boundaries))
        sections = [(a, b) for a, b in zip(boundaries, boundaries[1:]) if b > a]

        # Energy/density peaks anchors (coarse)
        t, feat = self._frame_features(self.ref_notes, hop=0.25)
        energy = feat[:, 0] + feat[:, 1] + feat[:, 3]  # activity+onset+vel
        if len(energy) > 20:
            idx = np.argsort(energy)[-8:]
            for i in idx:
                tr = float(t[i])
                anchors.append((tr, self.global_time_map(tr), "energy_peak"))

        return sections, anchors

    def _segment_phrases(self, sections: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not self.ref_events:
            return [(0.0, 0.0)]
        phrases: List[Tuple[float, float]] = []
        for s0, s1 in sections:
            idx = [i for i, e in enumerate(self.ref_events) if s0 <= e.onset < s1]
            if not idx:
                continue
            starts = [self.ref_events[idx[0]].onset]
            for i in idx[:-1]:
                gap = self.ref_events[i + 1].onset - self.ref_events[i].onset
                if gap >= self.gap_phrase_s:
                    starts.append(self.ref_events[i + 1].onset)
            starts.append(self.ref_events[idx[-1]].onset + 1e-3)
            starts = sorted(set(starts))
            for a, b in zip(starts, starts[1:]):
                if b > a:
                    phrases.append((a, b))
        return phrases

    # -----------------------------
    # MIDI-only repeat inference (self similarity)
    # -----------------------------

    def _compute_phrase_fingerprints(self) -> np.ndarray:
        """
        Fingerprint each phrase using reference notes in that time range:
          - pitch-class histogram (12)
          - density stats (#notes / sec)
          - velocity mean
          - register (pitch centroid)
        Returns (P, D).
        """
        P = len(self.phrases)
        if P == 0:
            return np.zeros((0, 16), dtype=float)

        fp_list = []
        for (t0, t1) in self.phrases:
            notes = [n for n in self.ref_notes if t0 <= n.onset < t1]
            dur = max(1e-6, t1 - t0)
            if not notes:
                fp = np.zeros(16, dtype=float)
                fp_list.append(fp)
                continue

            pitches = np.array([n.pitch for n in notes], dtype=int)
            vels = np.array([n.velocity for n in notes], dtype=float)

            pc = pitches % 12
            hist = np.zeros(12, dtype=float)
            for c in pc:
                hist[c] += 1.0
            if hist.sum() > 0:
                hist /= hist.sum()

            density = float(len(notes) / dur)
            vel_mean = float(np.mean(vels) / 127.0)
            pitch_centroid = float(np.mean(pitches) / 127.0)

            # additional compact stats
            pitch_std = float(np.std(pitches) / 127.0)
            vel_std = float(np.std(vels) / 127.0)

            fp = np.concatenate([
                hist,
                np.array([math.log1p(density), vel_mean, pitch_centroid, pitch_std], dtype=float)
            ])
            fp_list.append(fp)

        X = np.stack(fp_list, axis=0)
        # L2 normalize for cosine similarity
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return Xn

    def _infer_repeat_jumps(self, fp: np.ndarray, *, sim_threshold: float = 0.90) -> Dict[int, List[int]]:
        """
        Infer repeat jump candidates from phrase fingerprints.
        For each phrase i, find earlier phrases j<i with high cosine similarity.
        Only keep a few strongest to avoid explosion.
        """
        P = fp.shape[0]
        jumps: Dict[int, List[int]] = {}
        if P == 0:
            return jumps

        # cosine sim = dot for normalized vectors
        S = fp @ fp.T
        for i in range(P):
            cand = []
            for j in range(i):
                if S[i, j] >= sim_threshold:
                    cand.append((j, float(S[i, j])))
            cand.sort(key=lambda x: x[1], reverse=True)
            if cand:
                jumps[i] = [j for j, _ in cand[: self.max_repeat_jumps_per_phrase]]
        self.quality["repeat_inference"] = {
            "phrases": int(P),
            "threshold": float(sim_threshold),
            "num_phrases_with_jumps": int(len(jumps)),
        }
        return jumps

    def _build_timeline_hypotheses(self) -> List[TimelineHypothesis]:
        """
        Build initial hypotheses over phrase indices.
        Baseline: straight 0..P-1.
        Add branches where phrase i may jump back to j (repeat) ONCE, producing alternative order.
        Keep small set.
        """
        P = len(self.phrases)
        if P == 0:
            return [TimelineHypothesis(phrase_order=[], name="H0", cost=0.0)]

        base = TimelineHypothesis(list(range(P)), name="H0", cost=0.0)
        hyps = [base]

        # Create a few branched hypotheses: at phrase i, jump to earlier j once.
        # Simple but useful: this captures "A B A" structures.
        counter = 1
        for i in range(P):
            if i not in self.repeat_jumps:
                continue
            for j in self.repeat_jumps[i]:
                if counter >= self.max_hypotheses:
                    break
                # phrase order: 0..i then j..P-1 (re-entry at j)
                order = list(range(0, i + 1)) + list(range(j, P))
                hyps.append(TimelineHypothesis(order, name=f"H{counter}:jump{i}->{j}", cost=0.0))
                counter += 1
            if counter >= self.max_hypotheses:
                break

        return hyps[: self.max_hypotheses]

    def _beam_select_hypothesis(self, anchors: List[Tuple[float, float, str]]) -> None:
        """
        Prune hypotheses using anchor consistency under global map and repeat jump penalty.
        (Without score, we approximate: hypotheses that imply big discontinuities are penalized.)
        """
        assert self.global_time_map is not None

        # Anchor residual independent of hypothesis coverage (since phrases cover whole piece),
        # but we add a penalty proportional to number of backward jumps implied by order.
        def jump_penalty(order: List[int]) -> float:
            # count discontinuities where next phrase index < prev phrase index
            pen = 0.0
            for a, b in zip(order, order[1:]):
                if b < a:
                    pen += 50.0
            return pen

        def anchor_residual() -> float:
            res = 0.0
            for tr, tp, _ in anchors:
                res += abs(tp - self.global_time_map(tr))
            return res

        base_anchor = anchor_residual()
        for h in self.hypotheses:
            h.cost = base_anchor + jump_penalty(h.phrase_order)

        self.hypotheses.sort(key=lambda x: x.cost)
        self.selected_hypothesis = self.hypotheses[0] if self.hypotheses else None
        self.quality["hypothesis_selection"] = {
            "num_hypotheses": int(len(self.hypotheses)),
            "best": self.selected_hypothesis.name if self.selected_hypothesis else None,
        }

    # -----------------------------
    # Phrase maps + matching under selected hypothesis
    # -----------------------------

    def _compute_phrase_time_map(self, t0: float, t1: float, anchors: List[Tuple[float, float, str]]) -> Tuple[TimeMap, Dict[str, Any]]:
        """
        Local DTW on phrase frames to build fk. Uses global_map to pick perf window.
        """
        assert self.global_time_map is not None
        pad = 0.5
        ref_win = (max(0.0, t0 - pad), t1 + pad)
        p0 = self.global_time_map(ref_win[0]) - 1.0
        p1 = self.global_time_map(ref_win[1]) + 1.0

        ref_slice = [n for n in self.ref_notes if ref_win[0] <= n.onset <= ref_win[1]]
        perf_slice = [n for n in self.perf_notes if p0 <= n.onset <= p1]

        if len(ref_slice) < 2 or len(perf_slice) < 2:
            fk = TimeMap(np.array([t0, t1], float), np.array([self.global_time_map(t0), self.global_time_map(t1)], float))
            return fk, {"method": "fallback_linear", "resync_needed": True, "cost": 1e4}

        tR, XR = self._frame_features(ref_slice, self.frame_hop)
        tP, YP = self._frame_features(perf_slice, self.frame_hop)
        dist, path = self._dtw(XR, YP, use_fast=True)
        path = self._refine_path(path)

        ref_k = np.array([tR[i] for i, _ in path], dtype=float)
        perf_k = np.array([tP[j] for _, j in path], dtype=float)

        step = max(1, len(ref_k) // 80)
        ref_k = np.concatenate([[t0], ref_k[::step], [t1]]).astype(float)
        perf_k = np.concatenate([[self.global_time_map(t0)], perf_k[::step], [self.global_time_map(t1)]]).astype(float)

        fk = TimeMap(ref_k, perf_k)

        # phrase cost: DTW distance + anchor penalty within phrase
        anc_cost = 0.0
        for tr, tp, _k in anchors:
            if t0 <= tr < t1:
                anc_cost += abs(tp - self.global_time_map(tr))

        info = {"method": "local_frame_dtw", "dtw_distance": float(dist), "anchor_cost": float(anc_cost)}
        info["cost"] = float(dist + 2.0 * anc_cost)
        info["resync_needed"] = bool(info["cost"] > 2000.0)  # heuristic
        return fk, info

    def _match_under_selected_hypothesis(self, anchors: List[Tuple[float, float, str]]) -> List[Dict[str, Any]]:
        """
        Traverse phrases according to selected hypothesis order.
        For each phrase:
          - compute fk
          - DP align event clusters in phrase window, gated by fk
          - Hungarian within events
        """
        assert self.selected_hypothesis is not None
        assert self.global_time_map is not None

        out: List[Dict[str, Any]] = []
        order = self.selected_hypothesis.phrase_order
        matched_perf_keys: set = set()
        emitted_extra_perf_keys: set = set()

        # Build event indices per phrase on reference
        phrase_events_ref: List[List[int]] = []
        for (t0, t1) in self.phrases:
            idx = [i for i, e in enumerate(self.ref_events) if t0 <= e.onset < t1]
            phrase_events_ref.append(idx)

        # Iterate in hypothesis order
        for k, phrase_idx in enumerate(order):
            t0, t1 = self.phrases[phrase_idx]
            fk, info = self._compute_phrase_time_map(t0, t1, anchors)
            self.quality.setdefault("phrase_maps", []).append({"phrase_idx": int(phrase_idx), "t0": t0, "t1": t1, **info})

            # Perf event window based on fk
            p0 = fk(t0) - 1.0
            p1 = fk(t1) + 1.0
            perf_idx = [j for j, e in enumerate(self.perf_events) if p0 <= e.onset <= p1]

            ref_idx = phrase_events_ref[phrase_idx]
            if not ref_idx:
                continue

            if not perf_idx:
                # all missing
                for i in ref_idx:
                    for r in self.ref_events[i].notes:
                        out.append(self._emit_missing(r, "phrase_no_perf", phrase_idx))
                continue

            ref_seq = [self.ref_events[i] for i in ref_idx]
            perf_seq = [self.perf_events[j] for j in perf_idx]

            pairs, unr, unp = self._dp_align_events(ref_seq, perf_seq, fk)

            # Matched events -> Hungarian within clusters
            for i_local, j_local in pairs:
                r_ev = ref_seq[i_local]
                p_ev = perf_seq[j_local]
                matches, m_unr, m_unp = self._hungarian_match(r_ev.notes, p_ev.notes)

                for a, b in matches:
                    r = r_ev.notes[a]
                    p = p_ev.notes[b]
                    p_key = self._note_key(p)
                    cost = self._note_cost(r, p)
                    matched_perf_keys.add(p_key)
                    out.append({
                        "reference_note": self._note_to_dict(r),
                        "performance_note": self._note_to_dict(p),
                        "time_difference": float(p.onset - r.onset),
                        "pitch_difference": int(p.pitch - r.pitch),
                        "velocity_difference": int(p.velocity - r.velocity),
                        "alignment_confidence": float(1.0 / (1.0 + cost)),
                        "error_type": "none",
                        "phrase_index": int(phrase_idx),
                        "match_level": "note_in_event",
                    })
                for a in m_unr:
                    out.append(self._emit_missing(r_ev.notes[a], "hungarian_unmatched_ref", phrase_idx))
                for b in m_unp:
                    p = p_ev.notes[b]
                    p_key = self._note_key(p)
                    if p_key in matched_perf_keys or p_key in emitted_extra_perf_keys:
                        continue
                    emitted_extra_perf_keys.add(p_key)
                    out.append(self._emit_extra(p, "hungarian_unmatched_perf", phrase_idx))

            # Unmatched events -> missing/extra
            for i_local in unr:
                for r in ref_seq[i_local].notes:
                    out.append(self._emit_missing(r, "dp_unmatched_ref_event", phrase_idx))
            for j_local in unp:
                for p in perf_seq[j_local].notes:
                    p_key = self._note_key(p)
                    if p_key in matched_perf_keys or p_key in emitted_extra_perf_keys:
                        continue
                    emitted_extra_perf_keys.add(p_key)
                    out.append(self._emit_extra(p, "dp_unmatched_perf_event", phrase_idx))

        return out

    # -----------------------------
    # Level 3: DP + Hungarian (note matching)
    # -----------------------------

    def _passage_type(self, ev: EventCluster) -> str:
        if ev.density >= 3:
            return "chordal"
        d = ev.notes[0].duration if ev.notes else 0.2
        if ev.density == 1 and d < 0.12:
            return "run"
        return "melodic"

    def _event_cost(self, r: EventCluster, p: EventCluster, fk: TimeMap) -> float:
        onset_err = abs(p.onset - fk(r.onset))

        A = set(r.pitch_classes)
        B = set(p.pitch_classes)
        inter = len(A & B)
        union = max(1, len(A | B))
        pc_dist = 1.0 - (inter / union)

        reg_dist = abs(r.pitch_centroid - p.pitch_centroid) / 12.0
        dens_dist = abs(r.density - p.density) / max(1.0, float(r.density))

        pt = self._passage_type(r)
        if pt == "chordal":
            w_t, w_pc, w_reg, w_den = 2.0, 4.0, 1.0, 2.0
            window = 0.35
        elif pt == "run":
            w_t, w_pc, w_reg, w_den = 4.0, 1.0, 0.5, 0.5
            window = 1.0
        else:
            w_t, w_pc, w_reg, w_den = 3.0, 2.0, 1.0, 1.0
            window = 0.7

        return w_t * onset_err + w_pc * pc_dist + w_reg * reg_dist + w_den * dens_dist

    def _dp_align_events(
        self,
        ref_seq: List[EventCluster],
        perf_seq: List[EventCluster],
        fk: TimeMap,
        *,
        ins_cost: float = 1.2,
        del_cost: float = 1.2,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n, m = len(ref_seq), len(perf_seq)
        DP = np.full((n + 1, m + 1), np.inf, dtype=float)
        BT = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 0 match, 1 del, 2 ins

        DP[0, 0] = 0.0
        for i in range(1, n + 1):
            DP[i, 0] = DP[i - 1, 0] + del_cost
            BT[i, 0] = 1
        for j in range(1, m + 1):
            DP[0, j] = DP[0, j - 1] + ins_cost
            BT[0, j] = 2

        for i in range(1, n + 1):
            r = ref_seq[i - 1]
            pt = self._passage_type(r)
            window = 0.35 if pt == "chordal" else (1.0 if pt == "run" else 0.7)
            pred = fk(r.onset)

            for j in range(1, m + 1):
                p = perf_seq[j - 1]

                best = DP[i - 1, j] + del_cost
                bt = 1
                alt = DP[i, j - 1] + ins_cost
                if alt < best:
                    best = alt
                    bt = 2

                if abs(p.onset - pred) <= window:
                    mc = self._event_cost(r, p, fk)
                    alt = DP[i - 1, j - 1] + mc
                    if alt < best:
                        best = alt
                        bt = 0

                DP[i, j] = best
                BT[i, j] = bt

        i, j = n, m
        pairs: List[Tuple[int, int]] = []
        used_r = set()
        used_p = set()
        while i > 0 or j > 0:
            bt = BT[i, j]
            if bt == 0 and i > 0 and j > 0:
                pairs.append((i - 1, j - 1))
                used_r.add(i - 1)
                used_p.add(j - 1)
                i -= 1; j -= 1
            elif bt == 1 and i > 0:
                i -= 1
            elif bt == 2 and j > 0:
                j -= 1
            else:
                if i > 0: i -= 1
                elif j > 0: j -= 1

        pairs.reverse()
        unr = [ii for ii in range(n) if ii not in used_r]
        unp = [jj for jj in range(m) if jj not in used_p]
        return pairs, unr, unp

    def _note_cost(self, r: NoteEvent, p: NoteEvent) -> float:
        onset_c = abs(p.onset - r.onset)
        pitch_c = abs(p.pitch - r.pitch) / 12.0
        dur_ratio = (p.duration + 1e-6) / (r.duration + 1e-6)
        dur_c = abs(math.log(dur_ratio))
        return 3.0 * onset_c + 2.0 * pitch_c + 0.5 * dur_c

    def _hungarian_match(
        self, ref_notes: List[NoteEvent], perf_notes: List[NoteEvent]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        n, m = len(ref_notes), len(perf_notes)
        if n == 0 and m == 0:
            return [], [], []
        if n == 0:
            return [], [], list(range(m))
        if m == 0:
            return [], list(range(n)), []

        if linear_sum_assignment is None:
            used_p = set()
            matches = []
            for i, r in enumerate(ref_notes):
                best_j, best = None, None
                for j, p in enumerate(perf_notes):
                    if j in used_p:
                        continue
                    c = self._note_cost(r, p)
                    if best is None or c < best:
                        best = c
                        best_j = j
                if best_j is not None:
                    matches.append((i, best_j))
                    used_p.add(best_j)
            unr = [i for i in range(n) if i not in {a for a, _ in matches}]
            unp = [j for j in range(m) if j not in used_p]
            return matches, unr, unp

        size = max(n, m)
        C = np.full((size, size), 5.0, dtype=float)
        for i in range(n):
            for j in range(m):
                C[i, j] = self._note_cost(ref_notes[i], perf_notes[j])

        r_idx, p_idx = linear_sum_assignment(C)
        matches = []
        used_r, used_p = set(), set()
        for i, j in zip(r_idx, p_idx):
            if i < n and j < m and C[i, j] < 3.5:
                matches.append((i, j))
                used_r.add(i)
                used_p.add(j)

        unr = [i for i in range(n) if i not in used_r]
        unp = [j for j in range(m) if j not in used_p]
        return matches, unr, unp

    # -----------------------------
    # Level 4: post-processing
    # -----------------------------

    def _post_process(self, pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Reject absurd matches
        cleaned: List[Dict[str, Any]] = []
        for p in pairs:
            if p.get("error_type") != "none":
                cleaned.append(p)
                continue
            td = abs(float(p["time_difference"]))
            pd = abs(int(p["pitch_difference"]))
            if td > 3.0 or pd > 24:
                cleaned.append(self._emit_missing_dict(p["reference_note"], "post_reject_absurd"))
                cleaned.append(self._emit_extra_dict(p["performance_note"], "post_reject_absurd"))
            else:
                cleaned.append(p)

        # Enforce monotonicity by reference onset
        aligned = [p for p in cleaned if p.get("error_type") == "none"]
        aligned.sort(key=lambda x: x["reference_note"]["start"])
        fixed: List[Dict[str, Any]] = []
        last_perf = -1e18
        for p in aligned:
            perf_t = p["performance_note"]["start"]
            if perf_t + 1e-6 >= last_perf:
                fixed.append(p)
                last_perf = perf_t
            else:
                fixed.append(self._emit_missing_dict(p["reference_note"], "post_crossing_fix"))
                fixed.append(self._emit_extra_dict(p["performance_note"], "post_crossing_fix"))

        non = [p for p in cleaned if p.get("error_type") != "none"]

        # Ornament insertion labeling (very light)
        if fixed:
            perf_aligned = [
                (p["performance_note"]["start"], p["performance_note"]["pitch"])
                for p in fixed
                if p.get("performance_note") is not None
            ]
            for p in non:
                if p.get("error_type") == "extra_note" and p.get("performance_note") is not None:
                    t = p["performance_note"]["start"]
                    pitch = p["performance_note"]["pitch"]
                    nearest = min(perf_aligned, key=lambda x: abs(x[0] - t)) if perf_aligned else None
                    if nearest and abs(nearest[0] - t) <= 0.12 and (pitch - nearest[1]) % 12 in (0, 3, 4, 7, 9):
                        p["error_type"] = "ornament_insertion"

        matched_perf_keys = {
            self._note_dict_key(p["performance_note"])
            for p in fixed
            if p.get("performance_note") is not None
        }
        dedup_non: List[Dict[str, Any]] = []
        seen_extra_perf_keys: set = set()
        for p in non:
            et = p.get("error_type")
            if et in {"extra_note", "ornament_insertion"} and p.get("performance_note") is not None:
                key = self._note_dict_key(p["performance_note"])
                if key in matched_perf_keys or key in seen_extra_perf_keys:
                    continue
                seen_extra_perf_keys.add(key)
            dedup_non.append(p)

        return fixed + dedup_non

    # -----------------------------
    # Output helpers
    # -----------------------------

    def _note_to_dict(self, n: NoteEvent) -> Dict[str, Any]:
        return {
            "pitch": int(n.pitch),
            "start": float(n.onset),
            "end": float(n.offset),
            "velocity": int(n.velocity),
            "duration": float(n.duration),
            "track_id": n.track_id,
            "instrument": n.instrument,
        }

    def _note_key(self, n: NoteEvent) -> Tuple[int, int, int, int, int, int, str]:
        return (
            int(round(n.onset * 1_000_000)),
            int(round(n.offset * 1_000_000)),
            int(round(n.duration * 1_000_000)),
            int(n.pitch),
            int(n.velocity),
            int(n.track_id) if n.track_id is not None else -1,
            str(n.instrument) if n.instrument is not None else "",
        )

    def _note_dict_key(self, d: Dict[str, Any]) -> Tuple[int, int, int, int, int, int, str]:
        return (
            int(round(float(d.get("start", 0.0)) * 1_000_000)),
            int(round(float(d.get("end", 0.0)) * 1_000_000)),
            int(round(float(d.get("duration", 0.0)) * 1_000_000)),
            int(d.get("pitch", 0)),
            int(d.get("velocity", 0)),
            int(d.get("track_id")) if d.get("track_id") is not None else -1,
            str(d.get("instrument")) if d.get("instrument") is not None else "",
        )

    def _emit_missing(self, r: NoteEvent, reason: str, phrase_idx: Optional[int] = None) -> Dict[str, Any]:
        d = {
            "reference_note": self._note_to_dict(r),
            "performance_note": None,
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": "missing_note",
            "reason": reason,
        }
        if phrase_idx is not None:
            d["phrase_index"] = int(phrase_idx)
        return d

    def _emit_extra(self, p: NoteEvent, reason: str, phrase_idx: Optional[int] = None) -> Dict[str, Any]:
        d = {
            "reference_note": None,
            "performance_note": self._note_to_dict(p),
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": "extra_note",
            "reason": reason,
        }
        if phrase_idx is not None:
            d["phrase_index"] = int(phrase_idx)
        return d

    def _emit_missing_dict(self, ref_note_dict: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "reference_note": ref_note_dict,
            "performance_note": None,
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": "missing_note",
            "reason": reason,
        }

    def _emit_extra_dict(self, perf_note_dict: Dict[str, Any], reason: str) -> Dict[str, Any]:
        return {
            "reference_note": None,
            "performance_note": perf_note_dict,
            "time_difference": None,
            "pitch_difference": None,
            "velocity_difference": None,
            "alignment_confidence": 0.0,
            "error_type": "extra_note",
            "reason": reason,
        }

    def _sample_path(self, path: Optional[List[Tuple[int, int]]], max_points: int = 200) -> Optional[List[Tuple[int, int]]]:
        if not path:
            return None
        if len(path) <= max_points:
            return path
        step = max(1, len(path) // max_points)
        return path[::step]

    def _notes_are_identical(self, ref_notes: List[NoteEvent], perf_notes: List[NoteEvent], tol: float = 1e-6) -> bool:
        if len(ref_notes) != len(perf_notes):
            return False
        for r, p in zip(ref_notes, perf_notes):
            if r.pitch != p.pitch:
                return False
            if r.velocity != p.velocity:
                return False
            if abs(r.onset - p.onset) > tol:
                return False
            if abs(r.offset - p.offset) > tol:
                return False
        return True

    def _identity_time_map(self) -> TimeMap:
        t0 = 0.0
        t1 = max([n.offset for n in self.ref_notes], default=1.0)
        if t1 <= t0:
            t1 = t0 + 1.0
        return TimeMap(np.array([t0, t1], dtype=float), np.array([t0, t1], dtype=float))

    def _build_exact_aligned_pairs(self, ref_notes: List[NoteEvent], perf_notes: List[NoteEvent]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for r, p in zip(ref_notes, perf_notes):
            out.append({
                "reference_note": self._note_to_dict(r),
                "performance_note": self._note_to_dict(p),
                "time_difference": float(p.onset - r.onset),
                "pitch_difference": int(p.pitch - r.pitch),
                "velocity_difference": int(p.velocity - r.velocity),
                "alignment_confidence": 1.0,
                "error_type": "none",
                "phrase_index": 0,
                "match_level": "exact_identity",
            })
        return out


# -----------------------------
# Convenience wrappers
# -----------------------------

def align_midi_files(reference_path: str, performance_path: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    if pretty_midi is None:
        raise ImportError("pretty_midi is required to load MIDI files by path.")
    ref = pretty_midi.PrettyMIDI(reference_path)
    perf = pretty_midi.PrettyMIDI(performance_path)
    aligner = TimeAlignment(ref, perf)
    report = aligner.run_alignment()
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def align_parsed_data(reference_data: Dict[str, Any], performance_data: Dict[str, Any], output_file: Optional[str] = None) -> Dict[str, Any]:
    aligner = TimeAlignment(reference_data, performance_data)
    report = aligner.run_alignment()
    if output_file:
        import json
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report
