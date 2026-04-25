import statistics
from typing import Any, Dict, List, Tuple


class SimplePhraseSegmenter:
    """Fast, deterministic phrase segmentation for solo MIDI analysis."""

    def __init__(self, parsed_data: Dict[str, Any]):
        self.parsed_data = parsed_data
        self.notes = sorted(
            parsed_data.get("notes", []),
            key=lambda n: (float(n.get("start", 0.0)), int(n.get("pitch", 0))),
        )
        self.total_duration = (
            parsed_data.get("total_duration")
            or parsed_data.get("metadata", {}).get("total_duration")
            or max((float(n.get("end", n.get("start", 0.0))) for n in self.notes), default=0.0)
            or 0.0
        )

    def segment(self) -> Dict[str, Any]:
        if not self.notes:
            return {
                "phrases": [],
                "phrase_count": 0,
                "segmentation_method": "simple_gap_and_time",
                "segmentation_confidence": 0.0,
                "limitations": [
                    "Simple phrase segmentation is heuristic and based on gaps/time windows, not score-aware musical analysis."
                ],
            }

        phrase_note_lists, used_fixed_windows, heuristic_phrase_count = self._segment_note_lists()
        phrases = [
            self._summarize_phrase(phrase_id=index + 1, notes=phrase_notes)
            for index, phrase_notes in enumerate(phrase_note_lists)
        ]

        if used_fixed_windows:
            confidence = 0.45
        elif heuristic_phrase_count >= 2:
            confidence = 0.8
        elif len(phrases) == 1:
            confidence = 0.65
        else:
            confidence = 0.8

        return {
            "phrases": phrases,
            "phrase_count": len(phrases),
            "segmentation_method": "simple_gap_and_time",
            "segmentation_confidence": confidence,
            "limitations": [
                "Simple phrase segmentation is heuristic and based on gaps/time windows, not score-aware musical analysis."
            ],
        }

    def _segment_note_lists(self) -> Tuple[List[List[Dict[str, Any]]], bool, int]:
        min_notes_per_phrase = 4
        max_phrase_duration = 8.0

        positive_iois = self._positive_iois(self.notes)
        non_negative_gaps = self._non_negative_gaps(self.notes)
        median_ioi = statistics.median(positive_iois) if positive_iois else 0.5
        median_gap = statistics.median(non_negative_gaps) if non_negative_gaps else 0.0
        silence_threshold = max(0.75, median_ioi * 2.5, median_gap * 3.0)

        segments: List[List[Dict[str, Any]]] = []
        current_segment = [self.notes[0]]
        current_start = float(self.notes[0].get("start", 0.0))
        current_end = float(self.notes[0].get("end", self.notes[0].get("start", 0.0)))

        for note_index in range(1, len(self.notes)):
            previous_note = self.notes[note_index - 1]
            current_note = self.notes[note_index]

            gap = float(current_note.get("start", 0.0)) - float(
                previous_note.get("end", previous_note.get("start", 0.0))
            )
            phrase_duration = max(0.0, current_end - current_start)

            boundary = (
                gap > silence_threshold
                or (phrase_duration > max_phrase_duration and len(current_segment) >= min_notes_per_phrase)
            )

            if boundary:
                segments.append(current_segment)
                current_segment = [current_note]
                current_start = float(current_note.get("start", 0.0))
                current_end = float(current_note.get("end", current_note.get("start", 0.0)))
                continue

            current_segment.append(current_note)
            current_end = max(
                current_end,
                float(current_note.get("end", current_note.get("start", 0.0))),
            )

        if current_segment:
            segments.append(current_segment)

        heuristic_phrase_count = len(segments)
        used_fixed_windows = False

        if heuristic_phrase_count <= 1 and self.total_duration > max_phrase_duration:
            segments = self._split_into_fixed_windows(window_duration=7.0)
            used_fixed_windows = len(segments) > 1

        if not used_fixed_windows:
            segments = self._merge_small_fragments(segments, min_notes_per_phrase=min_notes_per_phrase)

        if not segments:
            segments = [self.notes]

        return segments, used_fixed_windows, heuristic_phrase_count

    def _split_into_fixed_windows(self, window_duration: float) -> List[List[Dict[str, Any]]]:
        if not self.notes:
            return []

        segments: List[List[Dict[str, Any]]] = []
        current_segment = [self.notes[0]]
        window_start = float(self.notes[0].get("start", 0.0))

        for note in self.notes[1:]:
            note_start = float(note.get("start", 0.0))
            if current_segment and (note_start - window_start) > window_duration:
                segments.append(current_segment)
                current_segment = [note]
                window_start = note_start
                continue
            current_segment.append(note)

        if current_segment:
            segments.append(current_segment)

        return segments

    def _merge_small_fragments(
        self,
        segments: List[List[Dict[str, Any]]],
        min_notes_per_phrase: int,
    ) -> List[List[Dict[str, Any]]]:
        if not segments:
            return []
        if len(segments) == 1:
            return [segments[0][:]]

        working_segments = [segment[:] for segment in segments]
        merged: List[List[Dict[str, Any]]] = []
        segment_index = 0

        while segment_index < len(working_segments):
            segment = working_segments[segment_index]
            if len(segment) >= min_notes_per_phrase:
                merged.append(segment[:])
                segment_index += 1
                continue

            if segment_index == 0 and len(working_segments) > 1:
                working_segments[1] = segment + working_segments[1]
                segment_index += 1
                continue

            if segment_index == len(working_segments) - 1 and merged:
                merged[-1].extend(segment)
                segment_index += 1
                continue

            next_segment = working_segments[segment_index + 1] if segment_index + 1 < len(working_segments) else []
            previous_size = len(merged[-1]) if merged else 0
            next_size = len(next_segment)

            if merged and previous_size <= next_size:
                merged[-1].extend(segment)
            elif segment_index + 1 < len(working_segments):
                working_segments[segment_index + 1] = segment + working_segments[segment_index + 1]
            else:
                merged.append(segment[:])

            segment_index += 1

        return merged

    def _summarize_phrase(self, phrase_id: int, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        start_time = min(float(note.get("start", 0.0)) for note in notes)
        end_time = max(float(note.get("end", note.get("start", 0.0))) for note in notes)
        duration = max(0.0, end_time - start_time)
        velocity_mean, velocity_std = self._velocity_stats(notes)

        return {
            "phrase_id": phrase_id,
            "start_time": round(start_time, 6),
            "end_time": round(end_time, 6),
            "duration": round(duration, 6),
            "note_count": len(notes),
            "pitch_range": self._pitch_range(notes),
            "average_velocity": round(velocity_mean, 3),
            "velocity_std": round(velocity_std, 3),
            "note_density": round((len(notes) / duration), 3) if duration > 1e-9 else 0.0,
            "dynamic_shape": self._dynamic_shape(notes),
            "rhythmic_character": self._rhythmic_character(notes),
            "articulation_profile": self._articulation_profile(notes),
            "tempo_stability": self._tempo_stability(notes),
            "notes": notes,
        }

    def _pitch_range(self, notes: List[Dict[str, Any]]) -> Dict[str, int]:
        if not notes:
            return {"min": 0, "max": 0, "range": 0}
        pitches = [int(note.get("pitch", 0)) for note in notes]
        return {
            "min": min(pitches),
            "max": max(pitches),
            "range": max(pitches) - min(pitches),
        }

    def _velocity_stats(self, notes: List[Dict[str, Any]]) -> Tuple[float, float]:
        if not notes:
            return 0.0, 0.0
        velocities = [float(note.get("velocity", 0.0)) for note in notes]
        mean_velocity = statistics.mean(velocities)
        velocity_std = statistics.stdev(velocities) if len(velocities) > 1 else 0.0
        return float(mean_velocity), float(velocity_std)

    def _dynamic_shape(self, notes: List[Dict[str, Any]]) -> str:
        if len(notes) < 4:
            return "unclear"

        thirds = self._split_into_thirds(notes)
        averages = [statistics.mean(float(note.get("velocity", 0.0)) for note in chunk) for chunk in thirds]
        first, middle, last = averages

        if (middle - first) > 8.0 and (middle - last) > 8.0:
            return "arch"
        if (last - first) > 8.0:
            return "crescendo"
        if (first - last) > 8.0:
            return "decrescendo"
        if max(abs(middle - first), abs(middle - last), abs(last - first)) <= 6.0:
            return "steady"
        return "unclear"

    def _rhythmic_character(self, notes: List[Dict[str, Any]]) -> str:
        iois = self._positive_iois(notes)
        if len(iois) < 2:
            return "steady"
        mean_ioi = statistics.mean(iois)
        if mean_ioi <= 1e-9:
            return "steady"
        cv = (statistics.stdev(iois) / mean_ioi) if len(iois) > 1 else 0.0
        if cv < 0.15:
            return "steady"
        if cv < 0.35:
            return "moderately_varied"
        return "varied"

    def _articulation_profile(self, notes: List[Dict[str, Any]]) -> Dict[str, float]:
        ratios: List[float] = []
        for index in range(len(notes) - 1):
            current_note = notes[index]
            next_note = notes[index + 1]
            next_onset = float(next_note.get("start", 0.0))
            ioi = next_onset - float(current_note.get("start", 0.0))
            if ioi <= 1e-9:
                continue
            duration = float(current_note.get("duration", 0.0))
            ratios.append(duration / ioi)

        if not ratios:
            return {
                "detached_percent": 0.0,
                "connected_percent": 0.0,
                "overlap_percent": 0.0,
            }

        detached = sum(1 for ratio in ratios if ratio < 0.50)
        connected = sum(1 for ratio in ratios if 0.50 <= ratio <= 1.05)
        overlap = sum(1 for ratio in ratios if ratio > 1.05)
        total = float(len(ratios))
        return {
            "detached_percent": round((detached / total) * 100.0, 3),
            "connected_percent": round((connected / total) * 100.0, 3),
            "overlap_percent": round((overlap / total) * 100.0, 3),
        }

    def _tempo_stability(self, notes: List[Dict[str, Any]]) -> Dict[str, Any]:
        iois = self._positive_iois(notes)
        if len(iois) < 2:
            return {"ioi_cv": 0.0, "label": "stable"}

        mean_ioi = statistics.mean(iois)
        if mean_ioi <= 1e-9:
            return {"ioi_cv": 0.0, "label": "stable"}

        ioi_cv = (statistics.stdev(iois) / mean_ioi) if len(iois) > 1 else 0.0
        if ioi_cv < 0.15:
            label = "stable"
        elif ioi_cv < 0.30:
            label = "moderately_variable"
        else:
            label = "variable"

        return {"ioi_cv": round(float(ioi_cv), 4), "label": label}

    def _split_into_thirds(self, notes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        count = len(notes)
        if count < 3:
            return notes[:], notes[:], notes[:]

        first_end = max(1, count // 3)
        second_end = max(first_end + 1, (2 * count) // 3)
        if second_end >= count:
            second_end = count - 1

        first = notes[:first_end]
        middle = notes[first_end:second_end] or notes[first_end:first_end + 1]
        last = notes[second_end:] or notes[-1:]
        return first, middle, last

    def _positive_iois(self, notes: List[Dict[str, Any]]) -> List[float]:
        iois: List[float] = []
        for index in range(1, len(notes)):
            ioi = float(notes[index].get("start", 0.0)) - float(notes[index - 1].get("start", 0.0))
            if ioi > 1e-9:
                iois.append(ioi)
        return iois

    def _non_negative_gaps(self, notes: List[Dict[str, Any]]) -> List[float]:
        gaps: List[float] = []
        for index in range(1, len(notes)):
            previous_end = float(notes[index - 1].get("end", notes[index - 1].get("start", 0.0)))
            current_start = float(notes[index].get("start", 0.0))
            gap = current_start - previous_end
            if gap >= 0.0:
                gaps.append(gap)
        return gaps
