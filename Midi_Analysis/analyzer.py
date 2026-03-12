"""
analyzer.py

Main orchestrator for the MIDI analysis pipeline.

Supports BOTH time_alignment APIs:
(NEW) TimeAlignment.run_alignment() -> {"aligned_pairs": ..., "statistics": ...}
(OLD) TimeAlignment.compute_dtw_alignment(); align_notes(); get_alignment_statistics(aligned_notes)

This prevents breakage when time_alignment.py is updated.

Also supports backend selection:
- native: src.time_alignment.TimeAlignment (default)
- paper_best: src.paper_time_alignment.PaperBestTimeAlignment (parangonar)
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple

from .midi_parser import MIDIParser
from .time_alignment import TimeAlignment
from .paper_time_alignment import PaperBestTimeAlignment
from .error_analysis import ErrorAnalysis
from .json_summarization import JSONSummarization


class MIDIAnalyzer:
    """
    Main orchestrator for the MIDI analysis pipeline.
    Handles both solo performance analysis and reference-based comparison.
    """

    def __init__(self):
        self.parser = MIDIParser()
        self.analysis_results: Dict[str, Any] = {}

    # -------------------------
    # Public API
    # -------------------------

    def analyze_solo_performance(self, performance_path: str) -> Dict[str, Any]:
        print(f"Analyzing solo performance: {performance_path}")

        performance_data = self.parser.parse_midi(performance_path)
        if not performance_data:
            raise ValueError(f"Failed to parse performance MIDI: {performance_path}")

        notes = performance_data.get("notes", [])

        total_duration = (
            performance_data.get("total_duration")
            or performance_data.get("metadata", {}).get("total_duration")
            or self._infer_duration_from_notes(notes)
            or 0.0
        )

        metrics = self._calculate_basic_metrics(notes, float(total_duration))

        self.analysis_results = {
            "analysis_type": "solo_performance",
            "performance_file": performance_path,
            "parsed_data": performance_data,
            "metrics": metrics,
            "practice_recommendations": self._generate_solo_recommendations(metrics),
        }
        return self.analysis_results

    def analyze_with_reference(
        self,
        reference_path: str,
        performance_path: str,
        output_dir: Optional[str] = None,
        alignment_backend: str = "native",
        alignment_model: str = "automatic_hdtw_sym",
    ) -> Dict[str, Any]:
        print("Analyzing performance against reference...")
        print(f"Reference: {reference_path}")
        print(f"Performance: {performance_path}")

        # 1) Parse both files
        print("1. Parsing MIDI files...")
        reference_parsed = self.parser.parse_midi(reference_path)
        performance_parsed = self.parser.parse_midi(performance_path)
        if not reference_parsed or not performance_parsed:
            raise ValueError("Failed to parse one or both MIDI files")

        # 2) Time alignment (supports both new/old time_alignment APIs)
        print("2. Performing time alignment...")
        print(f"   Backend: {alignment_backend}")
        aligner = self._build_time_aligner(
            reference_path,
            performance_path,
            reference_parsed,
            performance_parsed,
            alignment_backend=alignment_backend,
            alignment_model=alignment_model,
        )

        aligned_notes, alignment_stats, alignment_debug = self._run_time_alignment_compat(aligner)

        # 3) Error analysis
        print("3. Analyzing performance errors...")
        error_analyzer = ErrorAnalysis(
            {
                "reference": reference_parsed,
                "performance": performance_parsed,
                "alignment": aligned_notes,
            }
        )
        error_analysis = error_analyzer.analyze_performance()

        # 4) JSON summarization for GPT
        print("4. Generating comprehensive summary...")
        analysis_data = {
            "reference_data": reference_parsed,
            "performance_data": performance_parsed,
            "alignment": aligned_notes,
            "alignment_statistics": alignment_stats,
            "error_analysis": error_analysis,
        }
        summarizer = JSONSummarization(analysis_data)
        gpt_summary = summarizer.create_summary()

        # Compile final results
        self.analysis_results = {
            "analysis_type": "reference_comparison",
            "reference_file": reference_path,
            "performance_file": performance_path,
            "alignment_backend": alignment_backend,
            "alignment_model": alignment_model if alignment_backend == "paper_best" else None,
            "timestamp": self._get_timestamp(),
            "parsed_data": {"reference": reference_parsed, "performance": performance_parsed},
            "time_alignment": {
                "aligned_notes": aligned_notes,
                "statistics": alignment_stats,
                "debug": alignment_debug,
            },
            "performance_analysis": error_analysis,
            "gpt_ready_summary": gpt_summary,
        }

        if output_dir:
            self._save_analysis_reports(output_dir)

        print("Analysis complete!")
        return self.analysis_results

    def print_analysis_summary(self) -> None:
        if not self.analysis_results:
            print("No analysis results available. Run an analysis first.")
            return

        analysis_type = self.analysis_results.get("analysis_type", "unknown")

        print("\n" + "=" * 50)
        print("MIDI ANALYSIS SUMMARY")
        print("=" * 50)

        if analysis_type == "solo_performance":
            self._print_solo_summary()
        elif analysis_type == "reference_comparison":
            self._print_reference_summary()
        else:
            print(f"Unknown analysis type: {analysis_type}")

        print("=" * 50)

    def print_parsed_data(self, midi_path: str) -> Optional[Dict[str, Any]]:
        try:
            midi_data = self.parser.parse_midi(midi_path)
            if not midi_data:
                print("No parsed data returned.")
                return None

            notes = midi_data.get("notes", [])
            total_duration = (
                midi_data.get("total_duration")
                or midi_data.get("metadata", {}).get("total_duration")
                or self._infer_duration_from_notes(notes)
                or 0.0
            )

            print("\n" + "=" * 50)
            print("MIDI PARSED DATA")
            print("=" * 50)
            print(f"File: {midi_path}")
            print(f"Total notes: {len(notes)}")
            print(f"Total duration: {float(total_duration):.2f} seconds")
            print(f"Instruments: {midi_data.get('metadata', {}).get('instruments', [])}")

            print("\nFIRST 5 NOTES:")
            for i, note in enumerate(notes[:5]):
                print(
                    f"Note {i+1}: pitch={note.get('pitch')}, start={note.get('start', 0.0):.2f}s, "
                    f"duration={note.get('duration', 0.0):.2f}s, velocity={note.get('velocity')}"
                )

            if len(notes) > 5:
                print(f"... and {len(notes) - 5} more notes")

            return midi_data

        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return None

    # -------------------------
    # Alignment compatibility layer
    # -------------------------

    def _run_time_alignment_compat(
        self, aligner: Any
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """
        Run alignment using whichever API the TimeAlignment implementation provides.

        Returns:
            aligned_notes: list
            alignment_stats: dict
            alignment_debug: dict (extra fields, optional)
        """
        # NEW API
        if hasattr(aligner, "run_alignment") and callable(getattr(aligner, "run_alignment")):
            report = aligner.run_alignment()
            aligned_notes = report.get("aligned_pairs", []) or report.get("aligned_notes", []) or []
            alignment_stats = report.get("statistics", {}) or {}
            debug = {
                k: report.get(k)
                for k in [
                    "alignment_type",
                    "quality",
                    "selected_hypothesis",
                    "repeat_jumps",
                    "global_warping_path_sample",
                ]
                if k in report
            }
            return aligned_notes, alignment_stats, debug

        # OLD API
        if hasattr(aligner, "compute_dtw_alignment") and callable(getattr(aligner, "compute_dtw_alignment")):
            _ = aligner.compute_dtw_alignment()

            if not (hasattr(aligner, "align_notes") and callable(getattr(aligner, "align_notes"))):
                raise AttributeError("TimeAlignment is missing align_notes() in old-style API")

            aligned_notes = aligner.align_notes()

            if hasattr(aligner, "get_alignment_statistics") and callable(getattr(aligner, "get_alignment_statistics")):
                try:
                    alignment_stats = aligner.get_alignment_statistics(aligned_notes)
                except TypeError:
                    # Some older variants are get_alignment_statistics() without args
                    alignment_stats = aligner.get_alignment_statistics()
            else:
                alignment_stats = {}

            return aligned_notes, alignment_stats, {}

        raise AttributeError(
            "Unsupported TimeAlignment API. Expected run_alignment() OR compute_dtw_alignment(). "
            "Check your src/time_alignment.py methods."
        )

    def _build_time_aligner(
        self,
        reference_path: str,
        performance_path: str,
        reference_parsed: Dict[str, Any],
        performance_parsed: Dict[str, Any],
        alignment_backend: str = "native",
        alignment_model: str = "automatic_hdtw_sym",
    ) -> Any:
        """
        Create the requested alignment backend.
        """
        backend = (alignment_backend or "native").strip().lower()

        if backend == "paper_best":
            return PaperBestTimeAlignment(reference_path, performance_path, model=alignment_model)

        if backend != "native":
            raise ValueError(
                f"Unsupported alignment_backend={alignment_backend!r}. "
                "Use 'native' or 'paper_best'."
            )

        try:
            import pretty_midi

            reference_midi = pretty_midi.PrettyMIDI(reference_path)
            performance_midi = pretty_midi.PrettyMIDI(performance_path)
            return TimeAlignment(reference_midi, performance_midi)
        except Exception as e:
            print(f"Warning: PrettyMIDI load failed; using parsed data. Reason: {e}")
            return TimeAlignment(reference_parsed, performance_parsed)

    # -------------------------
    # Metrics + utility
    # -------------------------

    def _infer_duration_from_notes(self, notes: List[Dict[str, Any]]) -> float:
        if not notes:
            return 0.0
        ends = [n.get("end", 0.0) for n in notes if isinstance(n.get("end", None), (int, float))]
        return float(max(ends)) if ends else 0.0

    def _calculate_basic_metrics(self, notes: List[Dict[str, Any]], total_duration: float) -> Dict[str, Any]:
        if not notes:
            return {}

        velocities = [float(note.get("velocity", 0.0)) for note in notes]
        durations = [float(note.get("duration", 0.0)) for note in notes]
        pitches = [int(note.get("pitch", 0)) for note in notes]

        def _std(vals: List[float]) -> float:
            if not vals:
                return 0.0
            m = sum(vals) / len(vals)
            var = sum((x - m) ** 2 for x in vals) / len(vals)
            return var ** 0.5

        dur_mean = (sum(durations) / len(durations)) if durations else 0.0

        return {
            "note_count": len(notes),
            "total_duration": float(total_duration),
            "notes_per_second": (len(notes) / total_duration) if total_duration > 0 else 0.0,
            "velocity_stats": {
                "mean": (sum(velocities) / len(velocities)) if velocities else 0.0,
                "min": min(velocities) if velocities else 0.0,
                "max": max(velocities) if velocities else 0.0,
                "dynamic_range": (max(velocities) - min(velocities)) if velocities else 0.0,
            },
            "duration_stats": {
                "mean": dur_mean,
                "min": min(durations) if durations else 0.0,
                "max": max(durations) if durations else 0.0,
                "std": _std(durations),
            },
            "pitch_range": {"min": min(pitches) if pitches else 0, "max": max(pitches) if pitches else 0},
        }

    def _generate_solo_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        recommendations: List[str] = []

        velocity_stats = metrics.get("velocity_stats", {})
        dynamic_range = float(velocity_stats.get("dynamic_range", 0.0))
        if dynamic_range < 30:
            recommendations.append("Try to incorporate more dynamic variation in your playing")

        notes_per_second = float(metrics.get("notes_per_second", 0.0))
        if notes_per_second > 10:
            recommendations.append("Fast passage detected - consider using metronome for even timing")
        elif 0 < notes_per_second < 2:
            recommendations.append("Slow passage - focus on musical expression and phrasing")

        duration_stats = metrics.get("duration_stats", {})
        duration_std = float(duration_stats.get("std", 0.0))
        if duration_std > 0.5:
            recommendations.append("Work on consistent note durations for cleaner articulation")

        return recommendations

    def _save_analysis_reports(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)

        full_report_path = os.path.join(output_dir, "full_analysis.json")
        with open(full_report_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False, default=str)

        gpt_summary_path = os.path.join(output_dir, "gpt_summary.json")
        gpt_summary = self.analysis_results.get("gpt_ready_summary", {})
        with open(gpt_summary_path, "w", encoding="utf-8") as f:
            json.dump(gpt_summary, f, indent=2, ensure_ascii=False, default=str)

        alignment_path = os.path.join(output_dir, "alignment_report.json")
        alignment_data = {
            "aligned_notes": self.analysis_results.get("time_alignment", {}).get("aligned_notes", []),
            "statistics": self.analysis_results.get("time_alignment", {}).get("statistics", {}),
            "debug": self.analysis_results.get("time_alignment", {}).get("debug", {}),
        }
        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(alignment_data, f, indent=2, ensure_ascii=False, default=str)

        # Detailed error dump (kept separate so gpt_summary stays compact)
        error_details_path = os.path.join(output_dir, "error_details.json")
        perf_analysis = self.analysis_results.get("performance_analysis", {})
        error_details = {
            "reference_file": self.analysis_results.get("reference_file"),
            "performance_file": self.analysis_results.get("performance_file"),
            "timestamp": self.analysis_results.get("timestamp"),
            "metrics": perf_analysis.get("metrics", {}),
            "error_categories": perf_analysis.get("error_categories", {}),
            "detailed_errors": perf_analysis.get("detailed_errors", []),
            "performance_summary": perf_analysis.get("performance_summary", {}),
            "practice_recommendations": perf_analysis.get("practice_recommendations", []),
        }
        with open(error_details_path, "w", encoding="utf-8") as f:
            json.dump(error_details, f, indent=2, ensure_ascii=False, default=str)

        print(f"Analysis reports saved to: {output_dir}")

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _print_solo_summary(self) -> None:
        metrics = self.analysis_results.get("metrics", {})
        performance_file = self.analysis_results.get("performance_file", "Unknown")

        print(f"Performance File: {performance_file}")
        print(f"Total Notes: {metrics.get('note_count', 0)}")
        print(f"Duration: {metrics.get('total_duration', 0.0):.2f} seconds")
        print(f"Note Density: {metrics.get('notes_per_second', 0.0):.2f} notes/sec")

        print("\nPRACTICE RECOMMENDATIONS:")
        recommendations = self.analysis_results.get("practice_recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        else:
            print("  No specific recommendations available.")

    def _print_reference_summary(self) -> None:
        stats = self.analysis_results.get("time_alignment", {}).get("statistics", {})

        print(f"Reference: {self.analysis_results.get('reference_file', 'Unknown')}")
        print(f"Performance: {self.analysis_results.get('performance_file', 'Unknown')}")

        # Old schema
        alignment_summary = stats.get("alignment_summary", {})
        timing_analysis = stats.get("timing_analysis", {})

        if alignment_summary:
            aligned = alignment_summary.get("successfully_aligned", 0)
            missing = alignment_summary.get("missing_notes", 0)
            extra = alignment_summary.get("extra_notes", 0)
            rate = alignment_summary.get("alignment_rate", 0.0)
        else:
            # Newer schema
            aligned = stats.get("aligned_notes", 0)
            missing = stats.get("missing_notes", 0)
            extra = stats.get("extra_notes", 0)
            rate = stats.get("alignment_rate", 0.0)

        print("\nALIGNMENT RESULTS:")
        print(f"  Successfully Aligned: {aligned} notes")
        print(f"  Missing Notes: {missing}")
        print(f"  Extra Notes: {extra}")
        print(f"  Alignment Rate: {rate * 100:.1f}%")

        if timing_analysis:
            print("\nTIMING ANALYSIS:")
            print(f"  Avg Time Difference: {timing_analysis.get('mean_time_difference', 0.0):.3f}s")
            print(f"  Timing Consistency: +/-{timing_analysis.get('std_time_difference', 0.0):.3f}s")

        timing_accuracy = stats.get("timing_accuracy", {})
        if timing_accuracy:
            print("\nTIMING ANALYSIS:")
            print(f"  Mean |Time Error|: {timing_accuracy.get('mean_abs_error', 0.0):.3f}s")
            print(f"  Std  |Time Error|: +/-{timing_accuracy.get('std_abs_error', 0.0):.3f}s")


# -------------------------
# Convenience functions
# -------------------------

def quick_analyze(performance_path: str) -> Dict[str, Any]:
    analyzer = MIDIAnalyzer()
    return analyzer.analyze_solo_performance(performance_path)


def compare_performance(
    reference_path: str,
    performance_path: str,
    output_dir: Optional[str] = None,
    alignment_backend: str = "native",
    alignment_model: str = "automatic_hdtw_sym",
) -> Dict[str, Any]:
    analyzer = MIDIAnalyzer()
    return analyzer.analyze_with_reference(
        reference_path,
        performance_path,
        output_dir,
        alignment_backend=alignment_backend,
        alignment_model=alignment_model,
    )
