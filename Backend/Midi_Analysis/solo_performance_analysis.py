import math
import statistics
from typing import Any, Dict, List, Optional, Tuple


class SoloPerformanceAnalysis:
    """Observation-only MIDI analysis for solo performances without a reference."""

    def __init__(self, performance_data: Dict[str, Any], phrase_data: Optional[Dict[str, Any]] = None):
        self.performance_data = performance_data
        self.notes = sorted(
            performance_data.get("notes", []),
            key=lambda n: (float(n.get("start", 0.0)), int(n.get("pitch", 0))),
        )
        self.phrase_data = phrase_data or {}
        self.timing = performance_data.get("timing", {})
        self.metadata = performance_data.get("metadata", {})
        self.phrases = self.phrase_data.get("phrases", []) or []
        self.total_duration = float(
            performance_data.get("total_duration")
            or self.metadata.get("total_duration")
            or max((float(n.get("end", n.get("start", 0.0))) for n in self.notes), default=0.0)
            or 0.0
        )
        self._pedal_metrics_cache: Optional[Dict[str, Any]] = None
        self._register_summary_cache: Optional[Dict[str, Any]] = None

    def analyze(self) -> Dict[str, Any]:
        observations: List[Dict[str, Any]] = []
        observations.extend(self._analyze_tempo_stability())
        observations.extend(self._analyze_hesitations())
        observations.extend(self._analyze_phrase_shapes())
        observations.extend(self._analyze_dynamics())
        observations.extend(self._analyze_articulation())
        observations.extend(self._analyze_repeated_patterns_fast())
        observations.extend(self._analyze_register_balance())
        observations.extend(self._analyze_pedaling())
        observations = self._sort_observations(observations)

        summary_statistics = self._summary_statistics(observations)
        strengths = self._identify_strengths(observations, summary_statistics)
        practice_recommendations = self._make_practice_recommendations(observations)

        return {
            "mode": "solo_no_reference",
            "no_score": True,
            "observations": observations,
            "strengths": strengths,
            "practice_recommendations": practice_recommendations,
            "summary_statistics": summary_statistics,
            "limitations": [
                "No reference MIDI was provided, so note accuracy and intended rhythm cannot be judged.",
                "This analysis reports observable timing, velocity, articulation, register, repetition, and pedaling patterns only.",
                "MIDI velocity suggests attack strength but not full acoustic tone quality.",
                "Fingering, wrist tension, hand posture, and other physical technique details cannot be inferred from MIDI alone.",
                "Pedal observations depend on CC64 sustain data being present in the MIDI file.",
            ],
        }

    def _summary_statistics(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        note_count = len(self.notes)
        note_density = (note_count / self.total_duration) if self.total_duration > 1e-9 else 0.0
        pitches = [int(note.get("pitch", 0)) for note in self.notes]
        velocities = [float(note.get("velocity", 0.0)) for note in self.notes]
        positive_iois = self._positive_iois(self.notes)
        global_ioi_cv = 0.0
        if len(positive_iois) >= 2:
            mean_ioi = self._safe_mean(positive_iois)
            if mean_ioi > 1e-9:
                global_ioi_cv = self._safe_std(positive_iois) / mean_ioi

        articulation_mix = self._global_articulation_mix()
        register_summary = self._register_summary()
        pedal_summary = self._pedal_metrics()
        hesitation_count = sum(1 for obs in observations if obs.get("category") == "continuity")

        return {
            "note_count": note_count,
            "duration_seconds": round(self.total_duration, 6),
            "note_density": round(note_density, 4),
            "pitch_range": {
                "min": min(pitches) if pitches else 0,
                "max": max(pitches) if pitches else 0,
                "range": (max(pitches) - min(pitches)) if pitches else 0,
            },
            "estimated_tempo": round(float(self.timing.get("average_tempo", 0.0) or 0.0), 3),
            "global_ioi_cv": round(global_ioi_cv, 4),
            "hesitation_count": hesitation_count,
            "phrase_count": len(self.phrases),
            "dynamic_summary": {
                "mean_velocity": round(self._safe_mean(velocities), 3),
                "min_velocity": round(min(velocities), 3) if velocities else 0.0,
                "max_velocity": round(max(velocities), 3) if velocities else 0.0,
                "dynamic_range": round((max(velocities) - min(velocities)), 3) if velocities else 0.0,
                "velocity_std": round(self._safe_std(velocities), 3),
                "velocity_variety": round((len(set(int(v) for v in velocities)) / note_count), 4) if note_count else 0.0,
            },
            "articulation_mix": articulation_mix,
            "register_summary": register_summary,
            "pedal_summary": pedal_summary,
            "observation_count": len(observations),
        }

    def _analyze_tempo_stability(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        ioi_events: List[Dict[str, float]] = []
        for index in range(1, len(self.notes)):
            previous = self.notes[index - 1]
            current = self.notes[index]
            ioi = float(current.get("start", 0.0)) - float(previous.get("start", 0.0))
            if ioi <= 1e-9:
                continue
            ioi_events.append(
                {
                    "start_time": float(previous.get("start", 0.0)),
                    "end_time": float(current.get("start", 0.0)),
                    "ioi": ioi,
                }
            )

        if len(ioi_events) < 4:
            return observations

        window_size = 10 if len(ioi_events) >= 10 else min(len(ioi_events), 8)
        flagged_windows: List[Dict[str, Any]] = []
        for start_index in range(0, len(ioi_events) - window_size + 1):
            window = ioi_events[start_index:start_index + window_size]
            iois = [event["ioi"] for event in window]
            mean_ioi = self._safe_mean(iois)
            if mean_ioi <= 1e-9:
                continue
            std_ioi = self._safe_std(iois)
            cv = std_ioi / mean_ioi
            priority = None
            if cv > 0.35:
                priority = "high"
            elif cv > 0.22:
                priority = "medium"
            if not priority:
                continue

            flagged_windows.append(
                {
                    "priority": priority,
                    "start_time": float(window[0]["start_time"]),
                    "end_time": float(window[-1]["end_time"]),
                    "cv": cv,
                    "mean_ioi": mean_ioi,
                    "std_ioi": std_ioi,
                    "local_tempo": 60.0 / mean_ioi if mean_ioi > 1e-9 else 0.0,
                }
            )

        merged_windows = self._merge_flagged_windows(flagged_windows)
        for window in merged_windows:
            start_time = float(window["start_time"])
            end_time = float(window["end_time"])
            phrase_id = self._phrase_id_for_span(start_time, end_time)
            confidence = min(0.97, 0.55 + float(window["cv"]))
            observations.append(
                self._make_observation(
                    category="timing",
                    priority=window["priority"],
                    confidence=confidence,
                    start_time=start_time,
                    end_time=end_time,
                    phrase_id=phrase_id,
                    evidence={
                        "window_ioi_cv": round(float(window["cv"]), 4),
                        "mean_ioi": round(float(window["mean_ioi"]), 4),
                        "local_tempo_bpm": round(float(window["local_tempo"]), 2),
                    },
                    interpretation="The spacing between notes becomes less stable in this passage, which may indicate local tempo variability.",
                    feedback=(
                        f"The spacing between notes becomes less stable between {start_time:.1f}s and {end_time:.1f}s. "
                        "If this was not intentional, practise the passage with clear subdivision and steady pulse."
                    ),
                )
            )
        return observations

    def _analyze_hesitations(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        gaps: List[float] = []
        gap_events: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []

        for index in range(len(self.notes) - 1):
            current = self.notes[index]
            nxt = self.notes[index + 1]
            gap = float(nxt.get("start", 0.0)) - float(current.get("end", current.get("start", 0.0)))
            if gap < 0.0:
                continue
            gaps.append(gap)
            gap_events.append((current, nxt, gap))

        if not gap_events:
            return observations

        median_gap = statistics.median(gaps) if gaps else 0.0
        threshold = max(0.75, median_gap * 4.0)

        for current, nxt, gap in gap_events:
            if gap <= threshold:
                continue
            start_time = float(current.get("end", current.get("start", 0.0)))
            end_time = float(nxt.get("start", 0.0))
            priority = "high" if gap > threshold * 1.75 else "medium"
            confidence = min(0.95, 0.55 + (gap / max(threshold, 1e-6)) * 0.15)
            observations.append(
                self._make_observation(
                    category="continuity",
                    priority=priority,
                    confidence=confidence,
                    start_time=start_time,
                    end_time=end_time,
                    phrase_id=self._phrase_id_for_span(start_time, end_time),
                    evidence={
                        "gap_seconds": round(gap, 4),
                        "median_gap_seconds": round(median_gap, 4),
                        "threshold_seconds": round(threshold, 4),
                    },
                    interpretation="There is a noticeable pause before the next entry, which may indicate a local hesitation or a deliberate breath in the line.",
                    feedback=(
                        f"There is a noticeable pause around {start_time:.1f}s to {end_time:.1f}s. "
                        "If this was not intentional, practise connecting into the next entry with a steadier lead-in."
                    ),
                )
            )

        return observations

    def _analyze_phrase_shapes(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        for phrase in self.phrases:
            notes = phrase.get("notes", []) or []
            if len(notes) < 4:
                continue

            start_time = float(phrase.get("start_time", 0.0))
            end_time = float(phrase.get("end_time", start_time))
            phrase_id = phrase.get("phrase_id")
            velocity_std = float(phrase.get("velocity_std", 0.0))
            dynamic_shape = str(phrase.get("dynamic_shape", "unclear"))
            tempo_label = str(phrase.get("tempo_stability", {}).get("label", "stable"))
            note_density = float(phrase.get("note_density", 0.0))
            thirds = self._split_into_thirds(notes)
            velocity_thirds = [self._safe_mean([float(n.get("velocity", 0.0)) for n in chunk]) for chunk in thirds]
            ioi_thirds = [self._safe_mean(self._positive_iois(chunk)) for chunk in thirds]
            first_ioi = ioi_thirds[0]
            mid_ioi = ioi_thirds[1]
            last_ioi = ioi_thirds[2]
            middle_reference = self._safe_mean([value for value in [first_ioi, mid_ioi] if value > 0.0])

            if dynamic_shape == "steady" and velocity_std < 7.0 and len(notes) >= 6:
                observations.append(
                    self._make_observation(
                        category="phrasing",
                        priority="medium",
                        confidence=0.78,
                        start_time=start_time,
                        end_time=end_time,
                        phrase_id=phrase_id,
                        evidence={
                            "dynamic_shape": dynamic_shape,
                            "velocity_std": round(velocity_std, 3),
                            "note_density": round(note_density, 3),
                        },
                        interpretation="This phrase stays at a fairly even velocity level, so its dynamic direction is not strongly emphasized in the MIDI.",
                        feedback=(
                            f"The phrase from {start_time:.1f}s to {end_time:.1f}s has mostly steady velocity. "
                            "If this phrase should grow or relax musically, practise shaping it with a clearer dynamic direction."
                        ),
                    )
                )

            if middle_reference > 1e-9 and last_ioi > 1e-9 and last_ioi < middle_reference * 0.75:
                tail_start = float(thirds[2][0].get("start", start_time))
                observations.append(
                    self._make_observation(
                        category="phrasing",
                        priority="medium" if tempo_label != "variable" else "high",
                        confidence=0.8,
                        start_time=tail_start,
                        end_time=end_time,
                        phrase_id=phrase_id,
                        evidence={
                            "final_third_mean_ioi": round(last_ioi, 4),
                            "earlier_mean_ioi": round(middle_reference, 4),
                        },
                        interpretation="The phrase ending compresses compared with the earlier note spacing, which may suggest a rushed release of the line.",
                        feedback=(
                            f"The phrase ending around {tail_start:.1f}s to {end_time:.1f}s moves more quickly than the earlier part of the phrase. "
                            "If this was not intentional, practise arriving at the cadence with the same pulse you establish earlier."
                        ),
                    )
                )

            later_reference_values = [value for value in [mid_ioi, last_ioi] if value > 0.0]
            later_reference = self._safe_mean(later_reference_values)
            if first_ioi > 1e-9 and later_reference > 1e-9 and first_ioi > later_reference * 1.35:
                head_end = float(thirds[0][-1].get("end", start_time))
                observations.append(
                    self._make_observation(
                        category="phrasing",
                        priority="medium",
                        confidence=0.75,
                        start_time=start_time,
                        end_time=head_end,
                        phrase_id=phrase_id,
                        evidence={
                            "opening_mean_ioi": round(first_ioi, 4),
                            "later_mean_ioi": round(later_reference, 4),
                        },
                        interpretation="The opening of the phrase is more spaced out than what follows, which may indicate a cautious phrase start.",
                        feedback=(
                            f"The phrase opening from {start_time:.1f}s to {head_end:.1f}s feels more hesitant than the continuation. "
                            "If that was not intentional, practise the first few notes as a single gesture into the rest of the phrase."
                        ),
                    )
                )

            if dynamic_shape == "unclear" and velocity_std >= 6.0:
                observations.append(
                    self._make_observation(
                        category="phrasing",
                        priority="low",
                        confidence=0.62,
                        start_time=start_time,
                        end_time=end_time,
                        phrase_id=phrase_id,
                        evidence={
                            "dynamic_shape": dynamic_shape,
                            "velocity_std": round(velocity_std, 3),
                        },
                        interpretation="The phrase direction is less clearly outlined in the MIDI profile, so the musical arc may not be strongly projected here.",
                        feedback=(
                            f"The phrase from {start_time:.1f}s to {end_time:.1f}s has a less defined shape. "
                            "If you want a stronger musical line, experiment with a clearer rise, release, or destination point."
                        ),
                    )
                )

            if len(self.phrases) >= 2:
                next_phrase = self._next_phrase(phrase_id)
                if next_phrase is not None:
                    next_avg_velocity = float(next_phrase.get("average_velocity", 0.0))
                    current_avg_velocity = float(phrase.get("average_velocity", 0.0))
                    contrast = abs(current_avg_velocity - next_avg_velocity)
                    if contrast >= 12.0:
                        next_phrase_id = next_phrase.get("phrase_id")
                        observations.append(
                            self._make_observation(
                                category="phrasing",
                                priority="low",
                                confidence=0.72,
                                start_time=start_time,
                                end_time=float(next_phrase.get("end_time", end_time)),
                                phrase_id=phrase_id,
                                evidence={
                                    "phrase_velocity_contrast": round(contrast, 3),
                                    "current_phrase_id": phrase_id,
                                    "next_phrase_id": next_phrase_id,
                                },
                                interpretation="Adjacent phrases show a noticeable contrast in average velocity, which suggests a stronger sectional distinction here.",
                                feedback=(
                                    f"There is a noticeable contrast between phrase {phrase_id} and phrase {next_phrase_id}. "
                                    "If that contrast matches your musical idea, this contrast can help clarify the larger phrase structure."
                                ),
                            )
                        )
        return observations

    def _analyze_dynamics(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        if not self.notes:
            return observations

        velocities = [float(note.get("velocity", 0.0)) for note in self.notes]
        mean_velocity = self._safe_mean(velocities)
        min_velocity = min(velocities) if velocities else 0.0
        max_velocity = max(velocities) if velocities else 0.0
        dynamic_range = max_velocity - min_velocity
        velocity_std = self._safe_std(velocities)
        velocity_variety = (len(set(int(v) for v in velocities)) / len(velocities)) if velocities else 0.0

        if dynamic_range < 25.0:
            observations.append(
                self._make_observation(
                    category="dynamics",
                    priority="medium",
                    confidence=0.82,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "dynamic_range": round(dynamic_range, 3),
                        "velocity_std": round(velocity_std, 3),
                        "velocity_variety": round(velocity_variety, 4),
                    },
                    interpretation="The MIDI velocities suggest limited dynamic contrast across the performance.",
                    feedback=(
                        "The MIDI velocities suggest limited dynamic contrast overall. "
                        "If you want a wider expressive range, practise shaping louder and softer layers more deliberately."
                    ),
                )
            )

        if velocity_std < 8.0:
            observations.append(
                self._make_observation(
                    category="dynamics",
                    priority="medium",
                    confidence=0.76,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "velocity_std": round(velocity_std, 3),
                        "mean_velocity": round(mean_velocity, 3),
                    },
                    interpretation="The velocity profile is quite even from note to note, which may indicate a flatter dynamic surface.",
                    feedback=(
                        "The MIDI suggests a fairly even touch from note to note. "
                        "If you want more contour, practise shaping each phrase with clearer arrival points and releases."
                    ),
                )
            )

        consecutive_flat_start: Optional[int] = None
        for index in range(len(self.phrases) - 2):
            phrase_block = self.phrases[index:index + 3]
            averages = [float(phrase.get("average_velocity", 0.0)) for phrase in phrase_block]
            if (max(averages) - min(averages)) <= 5.0:
                consecutive_flat_start = index
                break
        if consecutive_flat_start is not None:
            phrase_block = self.phrases[consecutive_flat_start:consecutive_flat_start + 3]
            start_time = float(phrase_block[0].get("start_time", 0.0))
            end_time = float(phrase_block[-1].get("end_time", start_time))
            observations.append(
                self._make_observation(
                    category="dynamics",
                    priority="medium",
                    confidence=0.79,
                    start_time=start_time,
                    end_time=end_time,
                    phrase_id=phrase_block[0].get("phrase_id"),
                    evidence={
                        "phrase_average_velocities": [round(float(phrase.get("average_velocity", 0.0)), 3) for phrase in phrase_block],
                    },
                    interpretation="Several consecutive phrases stay at a similar average velocity level, so the larger-scale phrase contrast may be understated.",
                    feedback=(
                        f"From {start_time:.1f}s to {end_time:.1f}s, nearby phrases stay at a similar dynamic level. "
                        "If you want clearer phrase contrast, try giving each phrase a more distinct destination and release."
                    ),
                )
            )

        if mean_velocity > 105.0:
            observations.append(
                self._make_observation(
                    category="dynamics",
                    priority="medium",
                    confidence=0.71,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={"mean_velocity": round(mean_velocity, 3)},
                    interpretation="The MIDI velocities stay quite high overall, which may indicate a consistently forceful attack profile.",
                    feedback=(
                        "The MIDI suggests a consistently strong attack profile. "
                        "If you want more color, practise keeping the louder moments but making room for lighter ones too."
                    ),
                )
            )
        elif 0.0 < mean_velocity < 45.0:
            observations.append(
                self._make_observation(
                    category="dynamics",
                    priority="medium",
                    confidence=0.71,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={"mean_velocity": round(mean_velocity, 3)},
                    interpretation="The MIDI velocities stay relatively light overall, which may indicate an under-projected attack profile in this take.",
                    feedback=(
                        "The MIDI suggests a consistently light attack profile. "
                        "If you want the line to project more clearly, experiment with firmer arrivals in the musical high points."
                    ),
                )
            )

        return observations

    def _analyze_articulation(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        ratios = self._articulation_ratio_events(self.notes)
        if not ratios:
            return observations

        global_mix = self._global_articulation_mix()
        dense_phrases = [
            phrase for phrase in self.phrases
            if float(phrase.get("note_density", 0.0)) >= 6.0 and int(phrase.get("note_count", 0)) >= 6
        ]

        for phrase in dense_phrases:
            articulation = phrase.get("articulation_profile", {})
            detached_percent = float(articulation.get("detached_percent", 0.0))
            overlap_percent = float(articulation.get("overlap_percent", 0.0))
            start_time = float(phrase.get("start_time", 0.0))
            end_time = float(phrase.get("end_time", start_time))
            phrase_id = phrase.get("phrase_id")

            if detached_percent >= 65.0:
                observations.append(
                    self._make_observation(
                        category="articulation",
                        priority="medium",
                        confidence=0.76,
                        start_time=start_time,
                        end_time=end_time,
                        phrase_id=phrase_id,
                        evidence={
                            "detached_percent": round(detached_percent, 3),
                            "note_density": round(float(phrase.get("note_density", 0.0)), 3),
                        },
                        interpretation="This dense passage is played with a strongly detached profile, which may shorten the line more than nearby material.",
                        feedback=(
                            f"The articulation from {start_time:.1f}s to {end_time:.1f}s is noticeably detached for a dense passage. "
                            "If that contrast was not intentional, practise slightly longer connections between notes."
                        ),
                    )
                )

            if overlap_percent >= 45.0:
                observations.append(
                    self._make_observation(
                        category="articulation",
                        priority="medium",
                        confidence=0.79,
                        start_time=start_time,
                        end_time=end_time,
                        phrase_id=phrase_id,
                        evidence={
                            "overlap_percent": round(overlap_percent, 3),
                            "note_density": round(float(phrase.get("note_density", 0.0)), 3),
                        },
                        interpretation="This dense passage shows a high amount of note overlap, which may create a blurrier texture in fast material.",
                        feedback=(
                            f"The articulation from {start_time:.1f}s to {end_time:.1f}s has a lot of overlap. "
                            "If you want a clearer texture, practise releasing each note a little earlier while keeping the line connected."
                        ),
                    )
                )

        for index in range(len(self.phrases) - 1):
            current = self.phrases[index]
            nxt = self.phrases[index + 1]
            current_art = current.get("articulation_profile", {})
            next_art = nxt.get("articulation_profile", {})
            detached_diff = abs(float(current_art.get("detached_percent", 0.0)) - float(next_art.get("detached_percent", 0.0)))
            overlap_diff = abs(float(current_art.get("overlap_percent", 0.0)) - float(next_art.get("overlap_percent", 0.0)))
            if max(detached_diff, overlap_diff) < 35.0:
                continue
            start_time = float(current.get("start_time", 0.0))
            end_time = float(nxt.get("end_time", start_time))
            observations.append(
                self._make_observation(
                    category="articulation",
                    priority="medium",
                    confidence=0.74,
                    start_time=start_time,
                    end_time=end_time,
                    phrase_id=current.get("phrase_id"),
                    evidence={
                        "detached_percent_difference": round(detached_diff, 3),
                        "overlap_percent_difference": round(overlap_diff, 3),
                        "phrase_ids": [current.get("phrase_id"), nxt.get("phrase_id")],
                    },
                    interpretation="The articulation changes noticeably between adjacent phrases, which may be a deliberate contrast or may indicate inconsistency in release length.",
                    feedback=(
                        f"The articulation changes noticeably between phrase {current.get('phrase_id')} and phrase {nxt.get('phrase_id')}. "
                        "If that contrast was not intentional, practise matching the release length across both passages."
                    ),
                )
            )

        if global_mix["detached_percent"] >= 70.0 and len(self.notes) >= 24:
            observations.append(
                self._make_observation(
                    category="articulation",
                    priority="low",
                    confidence=0.68,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence=global_mix,
                    interpretation="The overall articulation profile leans strongly detached across the piece.",
                    feedback=(
                        "The overall articulation profile is quite detached. "
                        "If you want a more singing line, experiment with slightly longer note connections in the melodic passages."
                    ),
                )
            )

        return observations

    def _analyze_repeated_patterns_fast(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        reps = self._onset_representatives()
        if len(reps) < 8:
            return observations

        signature_occurrences: Dict[Tuple[Any, ...], List[int]] = {}
        for window_size in (4, 5, 6):
            if len(reps) < window_size:
                continue
            for index in range(0, len(reps) - window_size + 1):
                window = reps[index:index + window_size]
                signature = self._motif_signature(window)
                if signature is None:
                    continue
                bucket = signature_occurrences.setdefault(signature, [])
                if len(bucket) < 12:
                    bucket.append(index)

        candidates: List[Dict[str, Any]] = []
        for signature, occurrences in signature_occurrences.items():
            if len(occurrences) < 2:
                continue

            occurrence_features = []
            for start_index in occurrences:
                signature_length = int(signature[0])
                window = reps[start_index:start_index + signature_length]
                feature = self._occurrence_features(window)
                feature["start_index"] = start_index
                occurrence_features.append(feature)

            avg_iois = [feature["avg_ioi"] for feature in occurrence_features if feature["avg_ioi"] > 0.0]
            if not avg_iois:
                continue

            avg_ioi_range = (max(avg_iois) - min(avg_iois)) / max(self._safe_mean(avg_iois), 1e-6)
            velocity_range = max(feature["avg_velocity"] for feature in occurrence_features) - min(
                feature["avg_velocity"] for feature in occurrence_features
            )
            articulation_range = max(feature["articulation_ratio"] for feature in occurrence_features) - min(
                feature["articulation_ratio"] for feature in occurrence_features
            )
            final_ratio_range = max(feature["final_timing_ratio"] for feature in occurrence_features) - min(
                feature["final_timing_ratio"] for feature in occurrence_features
            )

            if not (
                avg_ioi_range > 0.22
                or velocity_range > 12.0
                or articulation_range > 0.3
                or final_ratio_range > 0.3
            ):
                continue

            dominant_metric = max(
                [
                    ("tempo", avg_ioi_range),
                    ("velocity", velocity_range / 12.0),
                    ("articulation", articulation_range / 0.3 if articulation_range > 0 else 0.0),
                    ("ending", final_ratio_range / 0.3 if final_ratio_range > 0 else 0.0),
                ],
                key=lambda item: item[1],
            )[0]

            medians = {
                "avg_ioi": statistics.median(feature["avg_ioi"] for feature in occurrence_features),
                "avg_velocity": statistics.median(feature["avg_velocity"] for feature in occurrence_features),
                "articulation_ratio": statistics.median(feature["articulation_ratio"] for feature in occurrence_features),
                "final_timing_ratio": statistics.median(feature["final_timing_ratio"] for feature in occurrence_features),
            }
            target_feature = max(
                occurrence_features,
                key=lambda feature: abs(feature["avg_ioi"] - medians["avg_ioi"])
                + abs(feature["avg_velocity"] - medians["avg_velocity"]) / 12.0
                + abs(feature["articulation_ratio"] - medians["articulation_ratio"]) / 0.3
                + abs(feature["final_timing_ratio"] - medians["final_timing_ratio"]) / 0.3,
            )

            interpretation = (
                "The same short pattern appears several times, and one occurrence differs noticeably in pacing, dynamic profile, or release shape."
            )
            feedback = (
                "A repeated figure changes noticeably from one appearance to the next. "
                "If that contrast was not intentional, practise matching the pacing, dynamic level, and release shape of each repetition."
            )

            if dominant_metric == "ending":
                interpretation = "The same short pattern appears several times, and one occurrence compresses or stretches at the end more than the others."
                feedback = (
                    "The same pattern comes back with a different ending shape. "
                    "If this was not intentional, practise the last notes of the figure separately so each repetition arrives the same way."
                )
            elif dominant_metric == "velocity":
                interpretation = "A repeated pattern returns with noticeably different velocity levels, which may reduce consistency between repetitions."
            elif dominant_metric == "articulation":
                interpretation = "A repeated pattern returns with a noticeably different articulation ratio, which may change how clearly each repetition speaks."

            candidates.append(
                {
                    "observation": self._make_observation(
                        category="repetition",
                        priority="medium",
                        confidence=min(
                            0.92,
                            0.68 + avg_ioi_range * 0.3 + (velocity_range / 127.0) + min(0.12, articulation_range * 0.2),
                        ),
                        start_time=float(target_feature["start_time"]),
                        end_time=float(target_feature["end_time"]),
                        phrase_id=self._phrase_id_for_span(
                            float(target_feature["start_time"]),
                            float(target_feature["end_time"]),
                        ),
                        evidence={
                            "signature_length": int(signature[0]),
                            "occurrence_count": len(occurrences),
                            "avg_ioi_range_ratio": round(avg_ioi_range, 4),
                            "velocity_range": round(velocity_range, 3),
                            "articulation_ratio_range": round(articulation_range, 4),
                            "final_timing_ratio_range": round(final_ratio_range, 4),
                        },
                        interpretation=interpretation,
                        feedback=feedback,
                    ),
                    "score": avg_ioi_range + (velocity_range / 12.0) + articulation_range + final_ratio_range,
                }
            )

        for candidate in sorted(candidates, key=lambda item: item["score"], reverse=True)[:5]:
            observations.append(candidate["observation"])
        return observations

    def _analyze_register_balance(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        summary = self._register_summary()
        bass_avg = float(summary["bass"]["average_velocity"])
        treble_avg = float(summary["treble"]["average_velocity"])
        middle_avg = float(summary["middle"]["average_velocity"])
        bass_count = int(summary["bass"]["note_count"])
        treble_count = int(summary["treble"]["note_count"])
        total_notes = max(len(self.notes), 1)

        if bass_count >= max(12, total_notes * 0.15) and treble_count >= max(12, total_notes * 0.15):
            velocity_gap = bass_avg - treble_avg
            if velocity_gap >= 12.0:
                observations.append(
                    self._make_observation(
                        category="register_balance",
                        priority="medium",
                        confidence=0.74,
                        start_time=0.0,
                        end_time=self.total_duration,
                        phrase_id=None,
                        evidence={
                            "bass_average_velocity": round(bass_avg, 3),
                            "treble_average_velocity": round(treble_avg, 3),
                            "velocity_difference": round(velocity_gap, 3),
                        },
                        interpretation="Lower-register notes are consistently stronger than upper-register notes in the MIDI profile.",
                        feedback=(
                            "The lower register is projected more strongly than the upper register overall. "
                            "If the melody is intended to be in the upper voice, you may want to practise bringing out the top line more clearly."
                        ),
                    )
                )
            elif (treble_avg - bass_avg) >= 12.0:
                observations.append(
                    self._make_observation(
                        category="register_balance",
                        priority="low",
                        confidence=0.68,
                        start_time=0.0,
                        end_time=self.total_duration,
                        phrase_id=None,
                        evidence={
                            "bass_average_velocity": round(bass_avg, 3),
                            "treble_average_velocity": round(treble_avg, 3),
                            "velocity_difference": round(treble_avg - bass_avg, 3),
                        },
                        interpretation="Upper-register notes are consistently emphasized more than lower-register notes in the MIDI profile.",
                        feedback=(
                            "The upper register is consistently more prominent than the lower register. "
                            "If you want more bass support, experiment with a slightly firmer foundation in the lower notes."
                        ),
                    )
                )

        multi_note_groups = summary["voicing"]["multi_note_group_count"]
        top_note_not_prominent = summary["voicing"]["top_note_not_prominent_count"]
        if multi_note_groups >= 8 and (top_note_not_prominent / max(multi_note_groups, 1)) >= 0.6:
            observations.append(
                self._make_observation(
                    category="register_balance",
                    priority="medium",
                    confidence=0.79,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "multi_note_group_count": multi_note_groups,
                        "top_note_not_prominent_count": top_note_not_prominent,
                        "top_note_not_prominent_ratio": round(top_note_not_prominent / max(multi_note_groups, 1), 4),
                    },
                    interpretation="In many chord groups, the top note is not stronger than the lower notes, which may reduce upper-voice projection if the melody is on top.",
                    feedback=(
                        "In many chord groups, the top note is not more prominent than the lower notes. "
                        "If the melody is intended to be in the upper voice, you may want to practise bringing out the top note."
                    ),
                )
            )

        if max(bass_count, int(summary["middle"]["note_count"]), treble_count) / max(total_notes, 1) >= 0.75:
            dominant_register = max(
                ("bass", bass_count),
                ("middle", int(summary["middle"]["note_count"])),
                ("treble", treble_count),
                key=lambda item: item[1],
            )[0]
            observations.append(
                self._make_observation(
                    category="register_balance",
                    priority="low",
                    confidence=0.66,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "dominant_register": dominant_register,
                        "dominant_register_note_ratio": round(
                            max(bass_count, int(summary["middle"]["note_count"]), treble_count) / max(total_notes, 1),
                            4,
                        ),
                    },
                    interpretation="Most notes lie in one register, which creates a focused register profile throughout the performance.",
                    feedback=(
                        f"The piece stays mostly in the {dominant_register} register. "
                        "If you want contrast across the keyboard, explore whether any supporting lines can be projected more clearly in the other registers."
                    ),
                )
            )

        return observations

    def _analyze_pedaling(self) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        metrics = self._pedal_metrics()
        if not metrics["pedal_analysis_available"]:
            observations.append(
                self._make_observation(
                    category="pedaling",
                    priority="low",
                    confidence=1.0,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={"pedal_event_count": 0, "pedal_segment_count": 0},
                    interpretation="No sustain pedal CC64 data is present in this MIDI file, so pedaling cannot be observed directly here.",
                    feedback="No sustain pedal CC64 data is available in this MIDI, so pedaling observations are limited for this take.",
                )
            )
            return observations

        if metrics["pedal_coverage_percent"] > 80.0:
            observations.append(
                self._make_observation(
                    category="pedaling",
                    priority="medium",
                    confidence=0.82,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "pedal_coverage_percent": round(metrics["pedal_coverage_percent"], 3),
                        "total_pedal_time": round(metrics["total_pedal_time"], 3),
                    },
                    interpretation="The sustain pedal covers a large share of the performance, which may create a more continuous sound profile.",
                    feedback=(
                        "The sustain pedal appears to cover a large share of the piece. "
                        "On an acoustic piano this may create blur, so consider clearing pedal more often near phrase or harmony changes."
                    ),
                )
            )

        if metrics["average_hold_duration"] > 4.0:
            observations.append(
                self._make_observation(
                    category="pedaling",
                    priority="medium",
                    confidence=0.79,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={
                        "average_hold_duration": round(metrics["average_hold_duration"], 3),
                        "max_hold_duration": round(metrics["max_hold_duration"], 3),
                    },
                    interpretation="The average pedal hold is relatively long, which may sustain harmonies and textures beyond each local gesture.",
                    feedback=(
                        "The average sustain hold is relatively long. "
                        "If you want a clearer texture, experiment with clearing pedal earlier at changes of harmony or phrase."
                    ),
                )
            )

        if metrics["pedal_during_dense_passages"] > 0:
            observations.append(
                self._make_observation(
                    category="pedaling",
                    priority="high",
                    confidence=0.84,
                    start_time=metrics["dense_overlap_start_time"],
                    end_time=metrics["dense_overlap_end_time"],
                    phrase_id=self._phrase_id_for_span(
                        metrics["dense_overlap_start_time"],
                        metrics["dense_overlap_end_time"],
                    ),
                    evidence={
                        "pedal_during_dense_passages": int(metrics["pedal_during_dense_passages"]),
                        "long_pedal_segments": int(metrics["long_pedal_segments"]),
                    },
                    interpretation="Long pedal spans overlap dense note passages, which may produce a blurrier texture if the resonance is not intentionally sustained.",
                    feedback=(
                        "The sustain pedal appears to be held for long spans during dense passages. "
                        "On an acoustic piano this may create blur, so consider clearing pedal near phrase or harmony changes."
                    ),
                )
            )

        if metrics["rapid_repedal_count"] >= 3:
            observations.append(
                self._make_observation(
                    category="pedaling",
                    priority="medium",
                    confidence=0.75,
                    start_time=0.0,
                    end_time=self.total_duration,
                    phrase_id=None,
                    evidence={"rapid_repedal_count": int(metrics["rapid_repedal_count"])},
                    interpretation="There are several quick pedal changes in close succession, which may indicate frequent re-pedaling gestures.",
                    feedback=(
                        "There are several quick pedal changes in close succession. "
                        "If you want the pedaling to feel steadier, practise coordinating each pedal change with a clear harmonic or phrase event."
                    ),
                )
            )

        return observations

    def _identify_strengths(
        self,
        observations: List[Dict[str, Any]],
        summary_statistics: Dict[str, Any],
    ) -> List[str]:
        strengths: List[str] = []
        high_medium_counts = {
            category: sum(
                1
                for obs in observations
                if obs.get("category") == category and obs.get("priority") in {"high", "medium"}
            )
            for category in {
                "timing",
                "continuity",
                "dynamics",
                "articulation",
                "phrasing",
                "repetition",
                "register_balance",
                "pedaling",
            }
        }

        if summary_statistics["global_ioi_cv"] > 0.0 and summary_statistics["global_ioi_cv"] < 0.18 and high_medium_counts["timing"] == 0:
            strengths.append("The overall note spacing stays fairly stable for much of the performance.")

        dynamic_summary = summary_statistics["dynamic_summary"]
        if dynamic_summary["dynamic_range"] >= 30.0 and dynamic_summary["velocity_std"] >= 10.0 and high_medium_counts["dynamics"] <= 1:
            strengths.append("The MIDI velocities show a usable amount of dynamic contrast across the performance.")

        phrase_shapes = [str(phrase.get("dynamic_shape", "unclear")) for phrase in self.phrases]
        if any(shape in {"crescendo", "decrescendo", "arch"} for shape in phrase_shapes):
            strengths.append("At least some phrases show a clearer dynamic contour rather than staying completely flat.")

        articulation_mix = summary_statistics["articulation_mix"]
        if 40.0 <= articulation_mix["connected_percent"] <= 85.0 and high_medium_counts["articulation"] == 0:
            strengths.append("The articulation profile stays reasonably connected through much of the performance.")

        register_summary = summary_statistics["register_summary"]
        if (
            register_summary["voicing"]["multi_note_group_count"] >= 6
            and register_summary["voicing"]["top_note_not_prominent_count"]
            < register_summary["voicing"]["multi_note_group_count"] * 0.5
            and high_medium_counts["register_balance"] == 0
        ):
            strengths.append("Chord balance often leaves room for the upper voice to come through clearly.")

        pedal_summary = summary_statistics["pedal_summary"]
        if (
            pedal_summary["pedal_analysis_available"]
            and 10.0 <= pedal_summary["pedal_coverage_percent"] <= 60.0
            and pedal_summary["average_hold_duration"] < 2.5
            and high_medium_counts["pedaling"] == 0
        ):
            strengths.append("The sustain pedal usage appears relatively measured rather than continuously held.")

        return strengths[:6]

    def _make_practice_recommendations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not observations:
            return []

        by_priority = {"high": [], "medium": [], "low": []}
        for observation in observations:
            by_priority.setdefault(str(observation.get("priority", "low")), []).append(observation)

        source = by_priority["high"] or by_priority["medium"] or by_priority["low"]
        recommendations: List[Dict[str, Any]] = []
        seen_focuses = set()

        for observation in source:
            focus = self._recommendation_focus(observation)
            if focus in seen_focuses:
                continue
            seen_focuses.add(focus)
            location = observation.get("location", {}) or {}
            start_time = location.get("start_time")
            end_time = location.get("end_time")
            why = str(observation.get("interpretation", "")).strip()
            if start_time is not None and end_time is not None:
                why = f"{why} This is most noticeable between {float(start_time):.1f}s and {float(end_time):.1f}s."

            recommendations.append(
                {
                    "focus": focus,
                    "why": why,
                    "exercise": self._exercise_for_observation(observation),
                    "location": {
                        "start_time": float(start_time) if start_time is not None else None,
                        "end_time": float(end_time) if end_time is not None else None,
                    },
                }
            )
            if len(recommendations) >= 6:
                break

        return recommendations

    def _sort_observations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        priority_order = {"high": 0, "medium": 1, "low": 2}
        return sorted(
            observations,
            key=lambda obs: (
                priority_order.get(str(obs.get("priority", "low")), 3),
                -float(obs.get("confidence", 0.0)),
                float((obs.get("location", {}) or {}).get("start_time", 0.0) or 0.0),
            ),
        )

    def _merge_flagged_windows(self, windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not windows:
            return []
        priority_rank = {"high": 2, "medium": 1, "low": 0}
        merged = [windows[0].copy()]
        for window in windows[1:]:
            current = merged[-1]
            if float(window["start_time"]) <= float(current["end_time"]) + 1.0:
                current["end_time"] = max(float(current["end_time"]), float(window["end_time"]))
                if priority_rank.get(str(window["priority"]), 0) > priority_rank.get(str(current["priority"]), 0):
                    current["priority"] = window["priority"]
                current["cv"] = max(float(current["cv"]), float(window["cv"]))
                current["mean_ioi"] = max(float(current["mean_ioi"]), float(window["mean_ioi"]))
                current["std_ioi"] = max(float(current["std_ioi"]), float(window["std_ioi"]))
                current["local_tempo"] = self._safe_mean([float(current["local_tempo"]), float(window["local_tempo"])])
                continue
            merged.append(window.copy())
        return merged

    def _make_observation(
        self,
        category: str,
        priority: str,
        confidence: float,
        start_time: float,
        end_time: float,
        phrase_id: Optional[int],
        evidence: Dict[str, Any],
        interpretation: str,
        feedback: str,
    ) -> Dict[str, Any]:
        location: Dict[str, Any] = {
            "start_time": round(float(start_time), 6),
            "end_time": round(float(end_time), 6),
        }
        if phrase_id is not None:
            location["phrase_id"] = int(phrase_id)
        return {
            "category": category,
            "priority": priority,
            "confidence": round(float(max(0.0, min(1.0, confidence))), 3),
            "location": location,
            "evidence": evidence,
            "interpretation": interpretation,
            "student_friendly_feedback": feedback,
        }

    def _positive_iois(self, notes: List[Dict[str, Any]]) -> List[float]:
        iois: List[float] = []
        for index in range(1, len(notes)):
            previous = notes[index - 1]
            current = notes[index]
            ioi = float(current.get("start", 0.0)) - float(previous.get("start", 0.0))
            if ioi > 1e-9:
                iois.append(ioi)
        return iois

    def _articulation_ratio_events(self, notes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for index in range(len(notes) - 1):
            current = notes[index]
            nxt = notes[index + 1]
            ioi = float(nxt.get("start", 0.0)) - float(current.get("start", 0.0))
            if ioi <= 1e-9:
                continue
            duration = float(current.get("duration", current.get("end", current.get("start", 0.0)) - current.get("start", 0.0)))
            ratio = duration / ioi
            if ratio < 0.50:
                label = "detached"
            elif ratio <= 1.05:
                label = "connected"
            else:
                label = "overlapping"
            events.append(
                {
                    "ratio": ratio,
                    "label": label,
                    "start_time": float(current.get("start", 0.0)),
                    "end_time": float(nxt.get("start", 0.0)),
                }
            )
        return events

    def _global_articulation_mix(self) -> Dict[str, float]:
        ratios = self._articulation_ratio_events(self.notes)
        total = len(ratios)
        if total == 0:
            return {
                "detached_percent": 0.0,
                "connected_percent": 0.0,
                "overlap_percent": 0.0,
            }
        detached = sum(1 for event in ratios if event["label"] == "detached")
        connected = sum(1 for event in ratios if event["label"] == "connected")
        overlap = sum(1 for event in ratios if event["label"] == "overlapping")
        return {
            "detached_percent": round(detached * 100.0 / total, 3),
            "connected_percent": round(connected * 100.0 / total, 3),
            "overlap_percent": round(overlap * 100.0 / total, 3),
        }

    def _onset_representatives(self) -> List[Dict[str, float]]:
        if not self.notes:
            return []
        groups: List[List[Dict[str, Any]]] = []
        current_group = [self.notes[0]]
        anchor = float(self.notes[0].get("start", 0.0))
        for note in self.notes[1:]:
            note_start = float(note.get("start", 0.0))
            if note_start - anchor <= 0.05:
                current_group.append(note)
                continue
            groups.append(current_group)
            current_group = [note]
            anchor = note_start
        if current_group:
            groups.append(current_group)

        representatives: List[Dict[str, float]] = []
        for group in groups:
            starts = [float(note.get("start", 0.0)) for note in group]
            ends = [float(note.get("end", note.get("start", 0.0))) for note in group]
            durations = [float(note.get("duration", note.get("end", note.get("start", 0.0)) - note.get("start", 0.0))) for note in group]
            velocities = [float(note.get("velocity", 0.0)) for note in group]
            top_pitch = max(int(note.get("pitch", 0)) for note in group)
            representatives.append(
                {
                    "start_time": min(starts),
                    "end_time": max(ends),
                    "pitch": float(top_pitch),
                    "duration": self._safe_mean(durations),
                    "velocity": self._safe_mean(velocities),
                }
            )
        return representatives

    def _motif_signature(self, window: List[Dict[str, float]]) -> Optional[Tuple[Any, ...]]:
        if len(window) < 4:
            return None
        pitches = [int(event["pitch"]) for event in window]
        intervals = tuple(pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1))
        contour = tuple("U" if interval > 0 else "D" if interval < 0 else "S" for interval in intervals)
        iois = [window[index + 1]["start_time"] - window[index]["start_time"] for index in range(len(window) - 1)]
        if any(ioi <= 1e-9 for ioi in iois):
            return None
        mean_ioi = self._safe_mean(iois)
        rhythm_shape = tuple(round(ioi / mean_ioi, 1) for ioi in iois) if mean_ioi > 1e-9 else tuple(1.0 for _ in iois)
        return (len(window), intervals, contour, rhythm_shape)

    def _occurrence_features(self, window: List[Dict[str, float]]) -> Dict[str, float]:
        iois = [window[index + 1]["start_time"] - window[index]["start_time"] for index in range(len(window) - 1)]
        iois = [ioi for ioi in iois if ioi > 1e-9]
        avg_ioi = self._safe_mean(iois)
        ioi_cv = (self._safe_std(iois) / avg_ioi) if len(iois) >= 2 and avg_ioi > 1e-9 else 0.0
        articulation_ratios = []
        for index in range(len(window) - 1):
            ioi = window[index + 1]["start_time"] - window[index]["start_time"]
            if ioi <= 1e-9:
                continue
            articulation_ratios.append(window[index]["duration"] / ioi)
        previous_iois = iois[:-1] if len(iois) > 1 else iois
        final_timing_ratio = 1.0
        if iois and previous_iois:
            denominator = self._safe_mean(previous_iois)
            if denominator > 1e-9:
                final_timing_ratio = iois[-1] / denominator
        return {
            "start_time": float(window[0]["start_time"]),
            "end_time": float(window[-1]["end_time"]),
            "avg_ioi": avg_ioi,
            "ioi_cv": ioi_cv,
            "avg_velocity": self._safe_mean([event["velocity"] for event in window]),
            "articulation_ratio": self._safe_mean(articulation_ratios),
            "final_timing_ratio": final_timing_ratio,
        }

    def _register_summary(self) -> Dict[str, Any]:
        if self._register_summary_cache is not None:
            return self._register_summary_cache

        registers = {
            "bass": [note for note in self.notes if int(note.get("pitch", 0)) < 55],
            "middle": [note for note in self.notes if 55 <= int(note.get("pitch", 0)) <= 72],
            "treble": [note for note in self.notes if int(note.get("pitch", 0)) > 72],
        }

        summary: Dict[str, Any] = {}
        for name, notes in registers.items():
            velocities = [float(note.get("velocity", 0.0)) for note in notes]
            summary[name] = {
                "note_count": len(notes),
                "average_velocity": round(self._safe_mean(velocities), 3),
            }

        groups = self._group_chord_onsets()
        multi_note_groups = [group for group in groups if len(group) >= 2]
        top_note_not_prominent_count = 0
        for group in multi_note_groups:
            sorted_group = sorted(group, key=lambda note: int(note.get("pitch", 0)))
            top_note = sorted_group[-1]
            lower_notes = sorted_group[:-1]
            lower_avg = self._safe_mean([float(note.get("velocity", 0.0)) for note in lower_notes])
            if float(top_note.get("velocity", 0.0)) <= lower_avg:
                top_note_not_prominent_count += 1

        summary["voicing"] = {
            "multi_note_group_count": len(multi_note_groups),
            "top_note_not_prominent_count": top_note_not_prominent_count,
        }
        self._register_summary_cache = summary
        return summary

    def _group_chord_onsets(self) -> List[List[Dict[str, Any]]]:
        if not self.notes:
            return []
        groups: List[List[Dict[str, Any]]] = []
        current_group = [self.notes[0]]
        anchor = float(self.notes[0].get("start", 0.0))
        for note in self.notes[1:]:
            note_start = float(note.get("start", 0.0))
            if note_start - anchor <= 0.05:
                current_group.append(note)
                continue
            groups.append(current_group)
            current_group = [note]
            anchor = note_start
        if current_group:
            groups.append(current_group)
        return groups

    def _pedal_metrics(self) -> Dict[str, Any]:
        if self._pedal_metrics_cache is not None:
            return self._pedal_metrics_cache

        pedaling = self.performance_data.get("pedaling", {}) or {}
        pedal_events = self.performance_data.get("pedals", []) or []
        pedal_segments = self.performance_data.get("pedal_segments", []) or []

        if not pedal_events and pedaling.get("events"):
            pedal_events = [
                {
                    "type": "sustain",
                    "time": float(event.get("time", 0.0)),
                    "value": int(event.get("value", 0)),
                    "state": "down" if str(event.get("event_type", "")).startswith("down") else "up",
                }
                for event in pedaling.get("events", [])
            ]

        if not pedal_segments and pedaling.get("segments"):
            pedal_segments = [
                {
                    "start_time": float(segment.get("start", 0.0)),
                    "end_time": float(segment.get("end", 0.0)),
                    "duration": float(segment.get("duration", 0.0)),
                }
                for segment in pedaling.get("segments", [])
            ]

        durations = [float(segment.get("duration", 0.0)) for segment in pedal_segments]
        total_pedal_time = sum(durations)
        coverage_percent = (total_pedal_time / self.total_duration * 100.0) if self.total_duration > 1e-9 else 0.0
        rapid_repedal_count = 0
        long_pedal_segments = 0
        pedal_during_dense_passages = 0
        dense_overlap_start_time = 0.0
        dense_overlap_end_time = 0.0

        sorted_segments = sorted(
            pedal_segments,
            key=lambda segment: (float(segment.get("start_time", 0.0)), float(segment.get("end_time", 0.0))),
        )
        for index, segment in enumerate(sorted_segments):
            duration = float(segment.get("duration", 0.0))
            if duration >= 4.0:
                long_pedal_segments += 1

            start_time = float(segment.get("start_time", 0.0))
            end_time = float(segment.get("end_time", start_time))
            span_density = self._density_in_span(start_time, end_time)
            if duration >= 2.0 and span_density >= 6.0:
                pedal_during_dense_passages += 1
                if dense_overlap_start_time == 0.0 and dense_overlap_end_time == 0.0:
                    dense_overlap_start_time = start_time
                    dense_overlap_end_time = end_time

            if index > 0:
                previous = sorted_segments[index - 1]
                gap = float(segment.get("start_time", 0.0)) - float(previous.get("end_time", previous.get("start_time", 0.0)))
                if gap <= 0.2:
                    rapid_repedal_count += 1

        metrics = {
            "pedal_analysis_available": bool(pedaling.get("available") or pedal_events or pedal_segments),
            "pedal_event_count": int(len(pedal_events)),
            "pedal_segment_count": int(len(sorted_segments)),
            "total_pedal_time": round(total_pedal_time, 6),
            "pedal_coverage_percent": round(coverage_percent, 3),
            "average_hold_duration": round(self._safe_mean(durations), 6),
            "max_hold_duration": round(max(durations), 6) if durations else 0.0,
            "rapid_repedal_count": int(rapid_repedal_count),
            "long_pedal_segments": int(long_pedal_segments),
            "pedal_during_dense_passages": int(pedal_during_dense_passages),
            "dense_overlap_start_time": round(dense_overlap_start_time, 6),
            "dense_overlap_end_time": round(dense_overlap_end_time, 6),
        }
        self._pedal_metrics_cache = metrics
        return metrics

    def _density_in_span(self, start_time: float, end_time: float) -> float:
        if end_time <= start_time:
            return 0.0
        notes = [
            note for note in self.notes
            if float(note.get("start", 0.0)) < end_time and float(note.get("end", note.get("start", 0.0))) > start_time
        ]
        return len(notes) / max(end_time - start_time, 1e-6)

    def _phrase_id_for_span(self, start_time: float, end_time: float) -> Optional[int]:
        best_phrase_id = None
        best_overlap = 0.0
        for phrase in self.phrases:
            phrase_start = float(phrase.get("start_time", 0.0))
            phrase_end = float(phrase.get("end_time", phrase_start))
            overlap = max(0.0, min(end_time, phrase_end) - max(start_time, phrase_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_phrase_id = phrase.get("phrase_id")
        return int(best_phrase_id) if best_phrase_id is not None else None

    def _next_phrase(self, phrase_id: Optional[int]) -> Optional[Dict[str, Any]]:
        if phrase_id is None:
            return None
        for index, phrase in enumerate(self.phrases):
            if int(phrase.get("phrase_id", -1)) == int(phrase_id) and index + 1 < len(self.phrases):
                return self.phrases[index + 1]
        return None

    def _split_into_thirds(self, notes: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not notes:
            return [[], [], []]
        total = len(notes)
        cut_1 = max(1, total // 3)
        cut_2 = max(cut_1 + 1, (2 * total) // 3)
        return [
            notes[:cut_1],
            notes[cut_1:cut_2] or notes[:cut_1],
            notes[cut_2:] or notes[cut_1:cut_2] or notes[:cut_1],
        ]

    def _recommendation_focus(self, observation: Dict[str, Any]) -> str:
        mapping = {
            "timing": "Timing stability",
            "continuity": "Continuity between entries",
            "dynamics": "Dynamic shaping",
            "articulation": "Articulation consistency",
            "phrasing": "Phrase direction",
            "repetition": "Repeated-pattern consistency",
            "register_balance": "Register balance and voicing",
            "pedaling": "Pedal clarity",
        }
        return mapping.get(str(observation.get("category", "")), "Performance shaping")

    def _exercise_for_observation(self, observation: Dict[str, Any]) -> str:
        category = str(observation.get("category", ""))
        if category == "timing":
            return "Loop the passage slowly with a metronome and subdivide out loud, then increase tempo only when the spacing feels even."
        if category == "continuity":
            return "Practise the lead-in note and the next entry as a two-note loop, aiming for one continuous gesture without an extra pause."
        if category == "dynamics":
            return "Play the passage in exaggerated dynamic layers first, then narrow the contrast until the shape still feels clear but natural."
        if category == "articulation":
            return "Repeat the passage using one chosen articulation profile, listening for whether each release length matches the last."
        if category == "phrasing":
            return "Sing the phrase shape away from the keyboard, then play it while aiming for one clear destination and one clear release."
        if category == "repetition":
            return "Choose one model repetition, then copy its pacing and release into each later repetition before putting the full passage back together."
        if category == "register_balance":
            return "Practise the upper voice alone, then add the lower notes back quietly so the melodic layer keeps its profile."
        if category == "pedaling":
            return "Practise the passage once without pedal, then add pedal back only at clear harmonic or phrase points so each change is intentional."
        return "Practise the passage in shorter loops and reintroduce speed only when the musical shape stays clear."

    def _safe_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _safe_std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean_value = self._safe_mean(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return float(math.sqrt(max(variance, 0.0)))
