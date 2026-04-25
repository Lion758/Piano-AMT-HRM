from datetime import datetime
from typing import Any, Dict, List


class SoloJSONSummarization:
    """Creates a GPT-ready summary for observation-only solo MIDI analysis."""

    def __init__(self, analysis_data: Dict[str, Any]):
        self.performance_data = analysis_data.get("performance_data", {})
        self.metrics = analysis_data.get("metrics", {})
        self.musical_structure = analysis_data.get("musical_structure", {})
        self.solo_analysis = analysis_data.get("solo_analysis", {})

    def create_summary(self) -> Dict[str, Any]:
        observations = self.solo_analysis.get("observations", []) or []
        strengths = self.solo_analysis.get("strengths", []) or []
        practice_recommendations = self.solo_analysis.get("practice_recommendations", []) or []
        summary_stats = self.solo_analysis.get("summary_statistics", {}) or {}
        phrases = self.musical_structure.get("phrases", []) or []

        return {
            "metadata": {
                "analysis_type": "solo_performance",
                "analysis_timestamp": datetime.now().isoformat(),
                "no_reference_provided": True,
                "no_score": True,
            },
            "performance_overview": {
                "duration_seconds": round(
                    float(
                        self.performance_data.get("total_duration")
                        or self.performance_data.get("metadata", {}).get("total_duration")
                        or self.metrics.get("total_duration")
                        or 0.0
                    ),
                    6,
                ),
                "note_count": int(self.metrics.get("note_count", len(self.performance_data.get("notes", []) or []))),
                "note_density": round(
                    float(
                        summary_stats.get("note_density")
                        or self.metrics.get("notes_per_second")
                        or 0.0
                    ),
                    4,
                ),
                "pitch_range": {
                    "min": int(
                        summary_stats.get("pitch_range", {}).get("min")
                        or self.metrics.get("pitch_range", {}).get("min", 0)
                        or 0
                    ),
                    "max": int(
                        summary_stats.get("pitch_range", {}).get("max")
                        or self.metrics.get("pitch_range", {}).get("max", 0)
                        or 0
                    ),
                    "range": int(
                        summary_stats.get("pitch_range", {}).get("range")
                        or (
                            int(self.metrics.get("pitch_range", {}).get("max", 0))
                            - int(self.metrics.get("pitch_range", {}).get("min", 0))
                        )
                    ),
                },
                "velocity_summary": {
                    "mean_velocity": round(
                        float(
                            summary_stats.get("dynamic_summary", {}).get("mean_velocity")
                            or self.metrics.get("velocity_stats", {}).get("mean", 0.0)
                            or 0.0
                        ),
                        3,
                    ),
                    "min_velocity": round(
                        float(
                            summary_stats.get("dynamic_summary", {}).get("min_velocity")
                            or self.metrics.get("velocity_stats", {}).get("min", 0.0)
                            or 0.0
                        ),
                        3,
                    ),
                    "max_velocity": round(
                        float(
                            summary_stats.get("dynamic_summary", {}).get("max_velocity")
                            or self.metrics.get("velocity_stats", {}).get("max", 0.0)
                            or 0.0
                        ),
                        3,
                    ),
                    "dynamic_range": round(
                        float(
                            summary_stats.get("dynamic_summary", {}).get("dynamic_range")
                            or self.metrics.get("velocity_stats", {}).get("dynamic_range", 0.0)
                            or 0.0
                        ),
                        3,
                    ),
                    "velocity_std": round(
                        float(summary_stats.get("dynamic_summary", {}).get("velocity_std", 0.0) or 0.0),
                        3,
                    ),
                },
                "estimated_tempo": round(
                    float(
                        summary_stats.get("estimated_tempo")
                        or self.performance_data.get("timing", {}).get("average_tempo", 0.0)
                        or 0.0
                    ),
                    3,
                ),
            },
            "musical_structure": {
                "phrase_count": int(self.musical_structure.get("phrase_count", len(phrases))),
                "segmentation_method": self.musical_structure.get("segmentation_method", "simple_gap_and_time"),
                "phrases_summary": [self._phrase_summary(phrase) for phrase in phrases],
            },
            "observations": observations,
            "strengths": strengths,
            "practice_recommendations": practice_recommendations,
            "gpt_prompt_context": {
                "role": "You are an experienced piano teacher analyzing a student's solo MIDI performance.",
                "tone": "Constructive, encouraging, specific, honest about uncertainty.",
                "critical_instruction": (
                    "Do not grade the performance. Do not claim wrong notes or incorrect rhythm because no reference MIDI was provided. "
                    "Use phrases like 'the MIDI suggests', 'this may indicate', and 'if this was not intentional'."
                ),
                "allowed_claims": [
                    "onset timing patterns",
                    "duration patterns",
                    "velocity patterns",
                    "articulation ratios",
                    "repeated pattern consistency",
                    "pedal CC64 timing if available",
                ],
                "forbidden_claims": [
                    "wrong notes",
                    "incorrect rhythm",
                    "bad fingering",
                    "wrist tension",
                    "poor hand posture",
                    "bad tone quality",
                    "definitive pedaling mistakes",
                ],
                "suggested_feedback_order": [
                    "brief overall observation",
                    "strengths",
                    "main practice priorities",
                    "specific passage-level suggestions",
                    "limitations of MIDI-only analysis",
                ],
            },
            "limitations": [
                "No reference MIDI was provided, so note accuracy and intended rhythm cannot be judged.",
                "MIDI velocity suggests attack strength but not full acoustic tone quality.",
                "Fingering, wrist tension, hand posture, and physical technique cannot be inferred from MIDI alone.",
                "Pedal analysis depends on CC64 data being present in the MIDI file.",
            ],
        }

    def _phrase_summary(self, phrase: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "phrase_id": phrase.get("phrase_id"),
            "start_time": phrase.get("start_time"),
            "end_time": phrase.get("end_time"),
            "duration": phrase.get("duration"),
            "note_count": phrase.get("note_count"),
            "pitch_range": phrase.get("pitch_range", {}),
            "average_velocity": phrase.get("average_velocity"),
            "velocity_std": phrase.get("velocity_std"),
            "note_density": phrase.get("note_density"),
            "dynamic_shape": phrase.get("dynamic_shape"),
            "rhythmic_character": phrase.get("rhythmic_character"),
            "articulation_profile": phrase.get("articulation_profile", {}),
            "tempo_stability": phrase.get("tempo_stability", {}),
        }
