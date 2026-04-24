import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import statistics
from collections import defaultdict
from bisect import bisect_left

class JSONSummarization:
    """
    Create structured JSON summaries from MIDI analysis results.
    Designed to be easily consumable by GPT for generating feedback.
    """
    
    def __init__(self, analysis_data: Dict[str, Any]):
        """
        Initialize with analysis data from the pipeline.
        
        Args:
            analysis_data: Dictionary containing:
                - reference_data: Parsed reference MIDI
                - performance_data: Parsed performance MIDI
                - alignment: Time alignment results
                - alignment_statistics: Alignment metrics
                - phrases: Phrase segmentation results
                - error_analysis: Error analysis results
        """
        self.analysis_data = analysis_data
        self.reference_data = analysis_data.get('reference_data', {})
        self.performance_data = analysis_data.get('performance_data', {})
        self.error_analysis = analysis_data.get('error_analysis', {})
        self.alignment = analysis_data.get('alignment', [])
        self.phrases = analysis_data.get('phrases', {})
        
    def create_summary(self, include_detailed_data: bool = False) -> Dict[str, Any]:
        """
        Create comprehensive JSON summary for GPT consumption.
        
        Args:
            include_detailed_data: Whether to include raw analysis data
            
        Returns:
            Structured JSON summary optimized for GPT prompts
        """
        print("Creating JSON summary for GPT...")
        
        summary = {
            'metadata': self._create_metadata(),
            'performance_overview': self._create_performance_overview(),
            'error_analysis_summary': self._create_error_summary(),
            'pedaling': self._create_pedaling_section(),
            'practice_recommendations': self._create_practice_recommendations(),
            'musical_analysis': self._create_musical_analysis(),
            'progress_metrics': self._create_progress_metrics(),
            'gpt_prompt_context': self._create_gpt_context()
        }
        
        if include_detailed_data:
            summary['detailed_data'] = self._extract_detailed_data()
        
        return summary
    
    def _get_total_duration(self, parsed: Dict[str, Any]) -> float:
        return (
            parsed.get('total_duration')
            or parsed.get('metadata', {}).get('total_duration')
            or 0
        )

    def _create_metadata(self) -> Dict[str, Any]:
        reference_notes = self.reference_data.get('notes', [])
        performance_notes = self.performance_data.get('notes', [])
        ref_dur = self._get_total_duration(self.reference_data)
        perf_dur = self._get_total_duration(self.performance_data)
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_version': '1.0',
            'reference_statistics': {
                'total_notes': len(reference_notes),
                'duration': ref_dur,
                'instruments': self.reference_data.get('metadata', {}).get('instruments', []),
                'pitch_range': self._calculate_pitch_range(reference_notes),
                'note_density': len(reference_notes) / ref_dur if ref_dur > 0 else 0
            },
            'performance_statistics': {
                'total_notes': len(performance_notes),
                'duration': perf_dur,
                'pitch_range': self._calculate_pitch_range(performance_notes),
                'note_density': len(performance_notes) / perf_dur if perf_dur > 0 else 0
            }
        }
    
    def _create_performance_overview(self) -> Dict[str, Any]:
        """Create high-level performance overview."""
        error_metrics = self.error_analysis.get('metrics', {})
        score_data = error_metrics.get('performance_score', {})
        
        # Extract key metrics with fallbacks
        note_accuracy = error_metrics.get('note_accuracy', {}).get('accuracy_percentage', 0)
        timing_errors = error_metrics.get('timing_errors', {})
        reliability = error_metrics.get('alignment_reliability', {})
        
        return {
            'overall_assessment': {
                'grade': score_data.get('grade', 'N/A'),
                'score': score_data.get('overall_score', 0),
                'performance_level': self._determine_performance_level(score_data.get('overall_score', 0))
            },
            'key_metrics': {
                'note_accuracy': f"{note_accuracy:.1f}%",
                'timing_consistency': (
                    f"+/-{timing_errors.get('std_error_ms', 0):.1f} ms"
                    if timing_errors.get('available', True) and timing_errors.get('std_error_ms') is not None
                    else "N/A (insufficient aligned pairs)"
                ),
                'dynamic_range': error_metrics.get('dynamic_control', {}).get('dynamic_range', 0),
                'rhythmic_consistency': error_metrics.get('rhythmic_consistency', {}).get('duration_consistency_score', 0)
            },
            'analysis_reliability': reliability,
            'strengths': self.error_analysis.get('performance_summary', {}).get('strengths', []),
            'weaknesses': self.error_analysis.get('performance_summary', {}).get('weaknesses', []),
            'performance_characteristics': self._identify_performance_characteristics()
        }
    
    def _create_error_summary(self) -> Dict[str, Any]:
        """Create structured error summary."""
        error_metrics = self.error_analysis.get('metrics', {})
        error_categories = self.error_analysis.get('error_categories', {})
        
        return {
            'note_accuracy': {
                'summary': self._summarize_note_accuracy(error_metrics.get('note_accuracy', {})),
                'priority': 'high' if error_metrics.get('note_accuracy', {}).get('accuracy_percentage', 100) < 90 else 'medium'
            },
            'timing': {
                'summary': self._summarize_timing_errors(error_metrics.get('timing_errors', {})),
                'patterns': self._extract_timing_patterns(error_metrics.get('timing_errors', {})),
                'priority': self._determine_timing_priority(error_metrics.get('timing_errors', {}))
            },
            'rhythm': {
                'summary': self._summarize_rhythmic_errors(error_metrics.get('rhythmic_consistency', {})),
                'consistency_score': error_metrics.get('rhythmic_consistency', {}).get('duration_consistency_score', 0),
                'priority': 'medium'
            },
            'dynamics': {
                'summary': self._summarize_dynamic_errors(error_metrics.get('dynamic_control', {})),
                'expression_level': error_metrics.get('dynamic_control', {}).get('expression_level', 'Unknown'),
                'priority': 'low'
            },
            'articulation': {
                'summary': self._summarize_articulation_errors(error_metrics.get('articulation', {})),
                'priority': 'medium'
            },
            'pedaling': {
                'summary': self._summarize_pedaling(error_metrics.get('pedaling', {})),
                'priority': self._determine_pedaling_priority(error_metrics.get('pedaling', {}))
            },
            'error_distribution': self._calculate_error_distribution(error_categories),
            'categorized_errors': self._build_categorized_errors(
                error_metrics,
                error_categories,
                include_samples=False
            )
        }
    
    def _create_practice_recommendations(self) -> Dict[str, Any]:
        """Create structured practice recommendations."""
        recommendations = self.error_analysis.get('practice_recommendations', [])
        next_steps = self.error_analysis.get('performance_summary', {}).get('next_steps', [])
        recommendations = self._dedupe_text_list(recommendations)
        next_steps = self._dedupe_text_list(next_steps)
        
        # Categorize recommendations
        categorized = {
            'urgent': [],
            'technique': [],
            'musicality': [],
            'general': []
        }
        
        all_recs = self._dedupe_text_list(recommendations + next_steps)
        for rec in all_recs:
            rec_lower = rec.lower()
            if any(word in rec_lower for word in ['focus', 'urgent', 'critical', 'significant']):
                categorized['urgent'].append(rec)
            elif any(word in rec_lower for word in ['technique', 'finger', 'hand', 'position']):
                categorized['technique'].append(rec)
            elif any(word in rec_lower for word in ['musical', 'expression', 'phrasing', 'dynamic']):
                categorized['musicality'].append(rec)
            else:
                categorized['general'].append(rec)
        
        return {
            'immediate_focus': categorized['urgent'][:3] if categorized['urgent'] else recommendations[:3],
            'technical_development': categorized['technique'],
            'musical_development': categorized['musicality'],
            'general_practice_tips': categorized['general'],
            'practice_schedule': self._create_practice_schedule(categorized),
            'specific_exercises': self._suggest_exercises()
        }

    def _create_pedaling_section(self) -> Dict[str, Any]:
        """Expose pedal-specific results in a dedicated summary section."""
        pedaling_metrics = self.error_analysis.get('metrics', {}).get('pedaling', {})
        pedaling_categories = self.error_analysis.get('error_categories', {}).get('pedaling', {})
        pedaling_recommendations = self.error_analysis.get('pedaling_recommendations', [])

        if not isinstance(pedaling_recommendations, list) or not pedaling_recommendations:
            pedaling_recommendations = [
                rec for rec in self.error_analysis.get('practice_recommendations', [])
                if 'pedal' in str(rec).lower()
            ]

        return {
            'available': bool(pedaling_metrics.get('pedal_analysis_available', False)),
            'mode': pedaling_metrics.get('mode', self.error_analysis.get('analysis_mode', 'reference_comparison')),
            'summary': self._summarize_pedaling(pedaling_metrics),
            'metrics': pedaling_metrics,
            'error_categories': pedaling_categories,
            'practice_suggestions': self._dedupe_text_list(pedaling_recommendations),
        }
    
    def _create_musical_analysis(self) -> Dict[str, Any]:
        """Create musical analysis section."""
        reference_notes = self.reference_data.get('notes', [])
        
        return {
            'technical_difficulty': {
                'level': self._assess_technical_difficulty(reference_notes),
                'challenging_sections': self._identify_challenging_sections(),
                'fastest_passage': self._find_fastest_passage(reference_notes),
                'largest_leap': self._find_largest_interval(reference_notes)
            },
            'musical_characteristics': {
                'tempo_profile': self._analyze_tempo_profile(),
                'dynamic_contour': self._analyze_dynamic_contour(),
                'articulation_style': self._analyze_articulation_style()
            }
        }
    
    def _create_progress_metrics(self) -> Dict[str, Any]:
        """Create metrics for tracking progress."""
        error_metrics = self.error_analysis.get('metrics', {})
        score_data = error_metrics.get('performance_score', {})
        
        return {
            'current_performance': {
                'overall_score': score_data.get('overall_score', 0),
                'component_scores': score_data.get('component_scores', {}),
                'benchmarks': self._create_benchmarks()
            },
            'improvement_areas': {
                'highest_priority': self._identify_highest_priority_areas(),
                'quick_wins': self._identify_quick_wins(),
                'long_term_goals': self._identify_long_term_goals()
            },
            'progress_tracking': {
                'metrics_to_track': ['note_accuracy', 'timing_consistency', 'dynamic_range'],
                'target_scores': self._set_target_scores(score_data.get('overall_score', 0)),
                'measurement_frequency': 'weekly'
            }
        }
    
    def _create_gpt_context(self) -> Dict[str, Any]:
        """Create context specifically for GPT prompts."""
        return {
            'instruction_context': {
                'role': "You are an experienced piano teacher analyzing a student's performance.",
                'tone': "Constructive, encouraging, specific",
                'format': "Provide feedback in this order: 1. Overall assessment 2. Strengths 3. Areas for improvement 4. Specific practice suggestions",
                'detail_level': "Be specific about measures and passages"
            },
            'student_profile': {
                'assumed_level': self._infer_student_level(),
                'likely_age': self._estimate_student_age(),
                'practice_habits': self._infer_practice_habits()
            },
            'piece_context': {
                'difficulty_level': self._assess_piece_difficulty(),
                'composer_style': self._infer_composer_style(),
                'musical_period': self._infer_musical_period()
            },
            'response_formatting': {
                'max_length': "500-700 words",
                'include_examples': True,
                'use_musical_terms': "Appropriate for student level",
                'include_encouragement': True
            }
        }
    
    # Helper Methods
    
    def _calculate_pitch_range(self, notes: List[Dict]) -> Dict[str, int]:
        """Calculate pitch range statistics."""
        if not notes:
            return {'min': 0, 'max': 0, 'range': 0}
        
        pitches = [note['pitch'] for note in notes]
        return {
            'min': min(pitches),
            'max': max(pitches),
            'range': max(pitches) - min(pitches)
        }
    
    def _determine_performance_level(self, score: float) -> str:
        """Determine performance level based on score."""
        if score >= 90:
            return "Advanced"
        elif score >= 80:
            return "Intermediate-Advanced"
        elif score >= 70:
            return "Intermediate"
        elif score >= 60:
            return "Late Beginner"
        elif score >= 50:
            return "Early Beginner"
        else:
            return "Novice"
    
    def _identify_performance_characteristics(self) -> List[str]:
        """Identify unique characteristics of this performance."""
        characteristics = []
        error_metrics = self.error_analysis.get('metrics', {})
        
        # Check timing tendencies
        timing = error_metrics.get('timing_errors', {})
        rushing = timing.get('rushing_percentage', 0)
        dragging = timing.get('dragging_percentage', 0)
        
        if rushing > dragging + 10:
            characteristics.append("Energetic, forward-moving tempo")
        elif dragging > rushing + 10:
            characteristics.append("Relaxed, deliberate tempo")
        
        # Check dynamic expression
        dynamics = error_metrics.get('dynamic_control', {})
        if dynamics.get('expression_level', '') == 'Highly Expressive':
            characteristics.append("Expressive dynamic control")
        
        # Check articulation
        articulation = error_metrics.get('articulation', {})
        staccato = articulation.get('staccato_percentage', 0)
        legato = articulation.get('legato_percentage', 0)
        
        if staccato > 50:
            characteristics.append("Crisp, articulated playing")
        elif legato > 50:
            characteristics.append("Smooth, connected playing")

        pedaling = error_metrics.get('pedaling', {})
        if pedaling.get('pedal_analysis_available', False):
            if str(pedaling.get('mode', 'reference_comparison')) == 'solo':
                stability = str(pedaling.get('stability', '')).lower()
                if stability == 'stable':
                    characteristics.append("Consistent supportive pedaling")
                elif stability == 'excessive':
                    characteristics.append("Heavy sustained pedal wash")
            elif float(pedaling.get('mean_overlap_ratio', 0.0) or 0.0) >= 0.8:
                characteristics.append("Pedaling closely follows the reference plan")
        
        return characteristics if characteristics else ["Balanced musical approach"]
    
    def _summarize_note_accuracy(self, accuracy_data: Dict) -> str:
        """Create summary text for note accuracy."""
        accuracy = accuracy_data.get('accuracy_percentage', 100)
        missing = accuracy_data.get('missing_percentage', 0)
        wrong = accuracy_data.get('wrong_notes', 0)
        
        if accuracy >= 98:
            return "Excellent note accuracy with virtually no errors."
        elif accuracy >= 95:
            return f"Very good note accuracy ({accuracy:.1f}% correct). {missing:.1f}% of notes were missed."
        elif accuracy >= 90:
            return f"Good note accuracy overall ({accuracy:.1f}% correct). Focus on the {missing:.1f}% of missing notes."
        elif accuracy >= 80:
            return f"Note accuracy needs improvement ({accuracy:.1f}% correct). {missing:.1f}% of notes were missed."
        else:
            return f"Significant note accuracy issues ({accuracy:.1f}% correct). {missing:.1f}% of notes were missed."
    
    def _summarize_timing_errors(self, timing_data: Dict) -> str:
        """Create summary text for timing errors."""
        if not timing_data.get('available', True):
            return "Timing analysis unavailable (insufficient aligned notes)."

        mean_raw = timing_data.get('mean_error_ms', 0)
        std_raw = timing_data.get('std_error_ms', 0)
        rushing_raw = timing_data.get('rushing_percentage', 0)
        dragging_raw = timing_data.get('dragging_percentage', 0)

        mean_error = float(mean_raw) if isinstance(mean_raw, (int, float)) else 0.0
        std_error = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        rushing = float(rushing_raw) if isinstance(rushing_raw, (int, float)) else 0.0
        dragging = float(dragging_raw) if isinstance(dragging_raw, (int, float)) else 0.0
        
        if abs(mean_error) < 20 and std_error < 30:
            return "Excellent timing control with precise rhythm."
        elif rushing > dragging + 15:
            return f"Tendency to rush ({rushing:.1f}% of notes early by average {mean_error:.1f}ms)."
        elif dragging > rushing + 15:
            return f"Tendency to drag ({dragging:.1f}% of notes late by average {abs(mean_error):.1f}ms)."
        elif std_error > 50:
            return f"Inconsistent timing (+/-{std_error:.1f}ms variability)."
        else:
            return f"Generally good timing (+/-{std_error:.1f}ms consistency)."
    
    def _extract_timing_patterns(self, timing_data: Dict) -> List[str]:
        """Extract timing patterns for summary."""
        patterns = timing_data.get('rhythmic_patterns', [])
        pattern_descriptions = []
        
        for pattern in patterns[:3]:  # Top 3 patterns
            if pattern['type'] == 'consistent_rushing':
                pattern_descriptions.append(
                    f"Consistent rushing in measures {pattern['start_index']}-{pattern['start_index'] + pattern['length']}"
                )
            elif pattern['type'] == 'consistent_dragging':
                pattern_descriptions.append(
                    f"Consistent dragging in measures {pattern['start_index']}-{pattern['start_index'] + pattern['length']}"
                )
        
        return pattern_descriptions
    
    def _determine_timing_priority(self, timing_data: Dict) -> str:
        """Determine priority level for timing issues."""
        if not timing_data.get('available', True):
            return 'low'

        std_raw = timing_data.get('std_error_ms', 0)
        rushing_raw = timing_data.get('rushing_percentage', 0)
        dragging_raw = timing_data.get('dragging_percentage', 0)

        std_error = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        rushing = float(rushing_raw) if isinstance(rushing_raw, (int, float)) else 0.0
        dragging = float(dragging_raw) if isinstance(dragging_raw, (int, float)) else 0.0
        
        if std_error > 80 or max(rushing, dragging) > 40:
            return 'high'
        elif std_error > 50 or max(rushing, dragging) > 25:
            return 'medium'
        else:
            return 'low'
    
    def _summarize_rhythmic_errors(self, rhythm_data: Dict) -> str:
        """Create summary text for rhythmic errors."""
        consistency = rhythm_data.get('duration_consistency_score', 0)
        
        if consistency >= 0.9:
            return "Excellent rhythmic consistency."
        elif consistency >= 0.8:
            return "Good rhythmic consistency with steady pulse."
        elif consistency >= 0.7:
            return "Adequate rhythmic consistency, some variability present."
        elif consistency >= 0.6:
            return "Rhythmic consistency needs improvement."
        else:
            return "Significant rhythmic inconsistency issues."
    
    def _summarize_dynamic_errors(self, dynamics_data: Dict) -> str:
        """Create summary text for dynamic errors."""
        dynamic_range = dynamics_data.get('dynamic_range', 0)
        expression = dynamics_data.get('expression_level', 'Unknown')
        
        if dynamic_range > 70 and expression == 'Highly Expressive':
            return "Excellent dynamic control with wide expressive range."
        elif dynamic_range > 50:
            return f"Good dynamic control ({dynamic_range} range). {expression} expression."
        elif dynamic_range > 30:
            return f"Adequate dynamics ({dynamic_range} range). Could use more variety."
        else:
            return f"Limited dynamic range ({dynamic_range}). Focus on creating more contrast."
    
    def _summarize_articulation_errors(self, articulation_data: Dict) -> str:
        """Create summary text for articulation errors."""
        staccato = articulation_data.get('staccato_percentage', 0)
        legato = articulation_data.get('legato_percentage', 0)
        consistency = articulation_data.get('articulation_consistency', 0)

        # If one articulation style is overwhelmingly dominant, describe it as
        # consistent in that style (this is not inherently a problem).
        if legato >= 95:
            return f"consistently legato articulation with {staccato:.1f}% staccato and {legato:.1f}% legato notes."
        if staccato >= 95:
            return f"consistently staccato articulation with {staccato:.1f}% staccato and {legato:.1f}% legato notes."

        if consistency >= 0.8:
            consistency_text = "consistent"
        elif consistency >= 0.6:
            consistency_text = "somewhat consistent"
        else:
            # Low consistency can happen even when a single style is intended;
            # avoid overly negative wording.
            consistency_text = "variable"

        return f"{consistency_text} articulation with {staccato:.1f}% staccato and {legato:.1f}% legato notes."

    def _summarize_pedaling(self, pedaling_data: Dict) -> str:
        """Create summary text for pedaling results."""
        if not pedaling_data or not pedaling_data.get('pedal_analysis_available', False):
            return "No pedal analysis was available from the current MIDI data."

        mode = str(pedaling_data.get('mode', 'reference_comparison'))
        if mode == 'solo':
            segment_count = int(pedaling_data.get('pedal_segment_count', 0))
            avg_hold = float(pedaling_data.get('average_hold_duration', 0.0) or 0.0)
            stability = str(pedaling_data.get('stability', 'unknown')).replace('_', ' ')
            phrase_ratio = pedaling_data.get('phrase_end_release_ratio', None)
            phrase_text = ""
            if isinstance(phrase_ratio, (int, float)):
                phrase_text = f" Releases land near phrase endings {float(phrase_ratio) * 100:.0f}% of the time."
            return (
                f"The performance uses {segment_count} pedal spans with an average hold of {avg_hold:.2f}s, "
                f"and the overall pedaling profile feels {stability}.{phrase_text}"
            ).strip()

        missed = int(pedaling_data.get('missed_pedals', 0))
        extra = int(pedaling_data.get('extra_pedals', 0))
        overlap = float(pedaling_data.get('mean_overlap_ratio', 0.0) or 0.0)
        release_issues = int(pedaling_data.get('late_release_count', 0)) + int(pedaling_data.get('early_release_count', 0))
        return (
            f"Pedal alignment averages {overlap:.2f} overlap with the reference, with "
            f"{missed} missed pedals, {extra} extra pedals, and {release_issues} notable release-timing issues."
        )

    def _determine_pedaling_priority(self, pedaling_data: Dict) -> str:
        """Assign a priority level to pedaling feedback."""
        if not pedaling_data or not pedaling_data.get('pedal_analysis_available', False):
            return 'low'

        if str(pedaling_data.get('mode', 'reference_comparison')) == 'solo':
            if bool(pedaling_data.get('excessive_pedaling', False)):
                return 'medium'
            phrase_ratio = pedaling_data.get('phrase_end_release_ratio', None)
            if isinstance(phrase_ratio, (int, float)) and phrase_ratio < 0.4:
                return 'medium'
            return 'low'

        issue_count = (
            int(pedaling_data.get('missed_pedals', 0))
            + int(pedaling_data.get('extra_pedals', 0))
            + int(pedaling_data.get('harmonic_blur_count', 0))
            + int(pedaling_data.get('phrase_boundary_clearance_issues', 0))
        )
        if issue_count >= 4:
            return 'high'
        if issue_count >= 1:
            return 'medium'
        return 'low'
    
    def _calculate_error_distribution(self, error_categories: Dict) -> Dict[str, float]:
        """
        Calculate distribution of *actual errors only*.

        Fix:
        - Do NOT count non-errors like 'matched' notes or 'accurate' timing.
        - If there are 0 errors, return 0% across categories.
        """
        # Map each category to the subkeys that represent real errors
        error_keys = {
            'note_accuracy': ['missing', 'extra', 'wrong'],
            'timing': ['rushing', 'dragging'],
            'pedaling': [
                'missed',
                'extra',
                'split',
                'merged',
                'late_onset',
                'early_onset',
                'late_release',
                'early_release',
                'harmonic_blur',
                'phrase_boundary_clearance',
                'early_release_while_notes_ring',
            ],
        }

        counts = {}
        total_errors = 0

        for category, keys in error_keys.items():
            cat_obj = error_categories.get(category, {})
            cat_count = 0

            if isinstance(cat_obj, dict):
                for k in keys:
                    v = cat_obj.get(k, [])
                    if isinstance(v, list):
                        cat_count += len(v)
            elif isinstance(cat_obj, list):
                # If a category is directly a list of errors
                cat_count += len(cat_obj)

            counts[category] = cat_count
            total_errors += cat_count

        # If no errors, return zeros (stable schema)
        if total_errors == 0:
            return {k: 0.0 for k in counts.keys()}

        # Convert to percentages
        return {k: (v / total_errors) * 100 for k, v in counts.items()}

    def _build_categorized_errors(
        self,
        error_metrics: Dict,
        error_categories: Dict,
        include_samples: bool = False,
        sample_limit: int = 10,
    ) -> Dict[str, Any]:
        """Expose explicit error categories and counts for downstream consumers."""
        note_acc = error_metrics.get('note_accuracy', {}) if isinstance(error_metrics, dict) else {}
        timing = error_metrics.get('timing_errors', {}) if isinstance(error_metrics, dict) else {}
        pedaling = error_metrics.get('pedaling', {}) if isinstance(error_metrics, dict) else {}

        total_ref = max(1, int(note_acc.get('total_reference_notes', 0)))
        missing = int(note_acc.get('missing_notes', 0))
        extra = int(note_acc.get('extra_notes', 0))
        wrong = int(note_acc.get('wrong_notes', 0))

        note_cat = error_categories.get('note_accuracy', {}) if isinstance(error_categories, dict) else {}
        pedal_cat = error_categories.get('pedaling', {}) if isinstance(error_categories, dict) else {}

        def _samples(items: Any, n: int = 10) -> List[Dict[str, Any]]:
            out = []
            if not isinstance(items, list):
                return out
            for p in items[:n]:
                ref = p.get('reference_note') if isinstance(p, dict) else None
                perf = p.get('performance_note') if isinstance(p, dict) else None
                out.append({
                    'ref_pitch': ref.get('pitch') if isinstance(ref, dict) else None,
                    'ref_time': ref.get('start') if isinstance(ref, dict) else None,
                    'perf_pitch': perf.get('pitch') if isinstance(perf, dict) else None,
                    'perf_time': perf.get('start') if isinstance(perf, dict) else None,
                    'error_type': p.get('error_type') if isinstance(p, dict) else None,
                    'reason': p.get('reason') if isinstance(p, dict) else None,
                })
            return out

        by_error_type: Dict[str, int] = {}
        for pair in self.alignment:
            if not isinstance(pair, dict):
                continue
            et = str(pair.get('error_type', 'none'))
            if et == 'none':
                continue
            by_error_type[et] = by_error_type.get(et, 0) + 1

        out = {
            'note_errors': {
                'missing_notes': {
                    'count': missing,
                    'percentage_of_reference': round((missing / total_ref) * 100, 1)
                },
                'extra_notes': {
                    'count': extra
                },
                'wrong_notes': {
                    'count': wrong,
                    'percentage_of_reference': round((wrong / total_ref) * 100, 1)
                }
            },
            'timing_errors': {
                'rushing_count': int(timing.get('rushing_count', 0)),
                'dragging_count': int(timing.get('dragging_count', 0)),
                'rushing_percentage': float(timing.get('rushing_percentage', 0.0)),
                'dragging_percentage': float(timing.get('dragging_percentage', 0.0))
            },
            'pedaling_errors': {
                'missed_pedals': int(pedaling.get('missed_pedals', 0)),
                'extra_pedals': int(pedaling.get('extra_pedals', 0)),
                'split_reference_spans': int(pedaling.get('split_reference_spans', 0)),
                'merged_performance_spans': int(pedaling.get('merged_performance_spans', 0)),
                'late_release_count': int(pedaling.get('late_release_count', 0)),
                'early_release_count': int(pedaling.get('early_release_count', 0)),
                'harmonic_blur_count': int(pedaling.get('harmonic_blur_count', 0)),
                'phrase_boundary_clearance_issues': int(pedaling.get('phrase_boundary_clearance_issues', 0)),
                'mode': pedaling.get('mode', 'reference_comparison'),
            },
            'other_error_types': by_error_type
        }
        if include_samples:
            out['note_errors']['missing_notes']['samples'] = _samples(note_cat.get('missing', []), n=sample_limit)
            out['note_errors']['extra_notes']['samples'] = _samples(note_cat.get('extra', []), n=sample_limit)
            out['note_errors']['wrong_notes']['samples'] = _samples(note_cat.get('wrong', []), n=sample_limit)
            out['pedaling_errors']['samples'] = {
                'missed': pedal_cat.get('missed', [])[:sample_limit] if isinstance(pedal_cat.get('missed', []), list) else [],
                'extra': pedal_cat.get('extra', [])[:sample_limit] if isinstance(pedal_cat.get('extra', []), list) else [],
                'late_release': pedal_cat.get('late_release', [])[:sample_limit] if isinstance(pedal_cat.get('late_release', []), list) else [],
                'phrase_boundary_clearance': pedal_cat.get('phrase_boundary_clearance', [])[:sample_limit] if isinstance(pedal_cat.get('phrase_boundary_clearance', []), list) else [],
            }
        return out
    
    def _create_practice_schedule(self, categorized_recs: Dict) -> Dict[str, Any]:
        """Create a suggested practice schedule."""
        return {
            'daily_focus': {
                'warmup': "5 minutes: scales with metronome",
                'technical_work': "10 minutes: " + ("; ".join(categorized_recs['technique'][:2]) if categorized_recs['technique'] else "sight-reading"),
                'piece_work': "15 minutes: focus on " + (categorized_recs['urgent'][0] if categorized_recs['urgent'] else "musical expression"),
                'musicality': "5 minutes: " + ("; ".join(categorized_recs['musicality'][:2]) if categorized_recs['musicality'] else "dynamics practice")
            },
            'weekly_goals': [
                "Improve timing consistency by 10%",
                "Master 2 most challenging measures",
                "Work on dynamic contrast"
            ],
            'practice_duration': "35-45 minutes daily"
        }
    
    def _suggest_exercises(self) -> List[str]:
        """Suggest specific exercises based on errors."""
        error_metrics = self.error_analysis.get('metrics', {})
        exercises = []
        
        # Timing exercises
        timing = error_metrics.get('timing_errors', {})
        std_raw = timing.get('std_error_ms', 0)
        std_error = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        if std_error > 50:
            exercises.append("Practice with metronome at slow tempo (50% of performance tempo)")
        
        # Dynamic exercises
        dynamics = error_metrics.get('dynamic_control', {})
        if dynamics.get('dynamic_range', 0) < 40:
            exercises.append("Crescendo/decrescendo exercises on single notes")
        
        # Articulation exercises
        articulation = error_metrics.get('articulation', {})
        if articulation.get('articulation_consistency', 0) < 0.7:
            exercises.append("Staccato-legato contrast exercises")
        
        # Note accuracy exercises
        note_acc = error_metrics.get('note_accuracy', {})
        if note_acc.get('accuracy_percentage', 100) < 90:
            exercises.append("Hands-separate practice of difficult passages")

        pedaling = error_metrics.get('pedaling', {})
        if pedaling.get('pedal_analysis_available', False):
            if str(pedaling.get('mode', 'reference_comparison')) == 'solo':
                if bool(pedaling.get('excessive_pedaling', False)):
                    exercises.append("Half-bar pedal refresh drills to clear resonance without breaking the line")
            elif (
                int(pedaling.get('late_release_count', 0)) > 0
                or int(pedaling.get('phrase_boundary_clearance_issues', 0)) > 0
            ):
                exercises.append("Slow pedal-only practice, coordinating releases with harmonic changes and phrase ends")
        
        return exercises[:5]  # Top 5 exercises
    
    def _analyze_musical_form(self) -> str:
        """Analyze the musical form/structure."""
        phrases = self.phrases.get('phrases', [])
        if isinstance(phrases, list) and len(phrases) >= 4:
            return f"Clear phrase structure with {len(phrases)} distinct phrases"
        elif isinstance(phrases, list) and len(phrases) >= 2:
            return f"Simple phrase structure with {len(phrases)} phrases"
        else:
            return "Continuous musical line"
    
    def _assess_technical_difficulty(self, reference_notes: List[Dict]) -> str:
        """Assess technical difficulty level."""
        if not reference_notes:
            return "Unknown"
        
        note_count = len(reference_notes)
        duration = self._get_total_duration(self.reference_data) or 1.0
        notes_per_second = note_count / duration
        
        if notes_per_second > 10:
            return "Advanced (virtuosic)"
        elif notes_per_second > 6:
            return "Intermediate-Advanced"
        elif notes_per_second > 3:
            return "Intermediate"
        else:
            return "Beginner"
    
    def _identify_challenging_sections(self) -> List[Dict]:
        """
        Identify challenging measures/sections using computed signals from:
        - note-level alignment pairs
        - error categories/metrics
        - reference score measure metadata (when available)

        Returns:
            Top challenging spans with stable keys:
            `section`, `reason`, `difficulty`
        """
        pairs = [p for p in self.alignment if isinstance(p, dict)]
        if not pairs:
            return []

        reference_notes = [n for n in self.reference_data.get('notes', []) if isinstance(n, dict)]
        performance_notes = [n for n in self.performance_data.get('notes', []) if isinstance(n, dict)]

        def _as_float(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        def _as_int(v: Any, default: Optional[int] = 0) -> Optional[int]:
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, int):
                return v
            if isinstance(v, float):
                return int(v)
            return default

        def _measure_from_note(note: Any) -> Optional[int]:
            if not isinstance(note, dict):
                return None
            mp = note.get('measure_position', {})
            if isinstance(mp, dict):
                m = mp.get('measure')
                if isinstance(m, (int, float)):
                    m_int = int(m)
                    if m_int > 0:
                        return m_int
            return None

        def _note_key(note: Any) -> Optional[tuple]:
            if not isinstance(note, dict):
                return None
            pitch = _as_int(note.get('pitch'), None)
            start = _as_float(note.get('start'), None)
            duration = _as_float(note.get('duration'), None)
            if pitch is None or start is None:
                return None
            dur_ms = int(round((duration if duration is not None else 0.0) * 1000))
            return (int(pitch), int(round(start * 1000)), dur_ms)

        def _loose_note_key(note: Any) -> Optional[tuple]:
            if not isinstance(note, dict):
                return None
            pitch = _as_int(note.get('pitch'), None)
            start = _as_float(note.get('start'), None)
            if pitch is None or start is None:
                return None
            return (int(pitch), int(round(start * 1000)))

        def _build_measure_lookup(notes: List[Dict]) -> tuple:
            exact = defaultdict(list)
            loose = defaultdict(list)
            ordered_measures: List[int] = []
            for note in sorted(
                notes,
                key=lambda n: (
                    _as_float(n.get('start'), 0.0),
                    _as_int(n.get('pitch'), 0),
                    _as_float(n.get('duration'), 0.0)
                )
            ):
                measure = _measure_from_note(note)
                if measure is None:
                    continue
                k = _note_key(note)
                lk = _loose_note_key(note)
                if k is not None:
                    exact[k].append(measure)
                if lk is not None:
                    loose[lk].append(measure)
                ordered_measures.append(measure)
            return exact, loose, ordered_measures

        def _timing_abs(pair: Dict[str, Any]) -> Optional[float]:
            td = pair.get('time_difference')
            if isinstance(td, (int, float)):
                return abs(float(td))
            return None

        def _difficulty_label(score: float, p60: float, p85: float) -> str:
            if score >= p85:
                return 'high'
            if score >= p60:
                return 'medium'
            return 'low'

        def _fallback_phrase_sections() -> List[Dict]:
            phrase_stats = defaultdict(lambda: {
                'count': 0,
                'missing': 0,
                'extra': 0,
                'wrong': 0,
                'timing_bad': 0,
                'timing_samples': 0,
                'error_score': 0.0,
            })

            for pair in pairs:
                phrase_idx = pair.get('phrase_index')
                if not isinstance(phrase_idx, int):
                    continue
                s = phrase_stats[int(phrase_idx)]
                s['count'] += 1

                ref_note = pair.get('reference_note')
                perf_note = pair.get('performance_note')
                err = str(pair.get('error_type', 'none'))
                pitch_diff = _as_int(pair.get('pitch_difference'), 0) or 0

                if isinstance(ref_note, dict) and not isinstance(perf_note, dict):
                    s['missing'] += 1
                    s['error_score'] += 2.5
                elif isinstance(perf_note, dict) and not isinstance(ref_note, dict):
                    s['extra'] += 1
                    s['error_score'] += 1.2 if err == 'extra_note' else 0.8
                else:
                    if pitch_diff != 0:
                        s['wrong'] += 1
                        s['error_score'] += 2.0
                    abs_td = _timing_abs(pair)
                    if abs_td is not None:
                        s['timing_samples'] += 1
                        if abs_td > 0.08:
                            s['timing_bad'] += 1
                            s['error_score'] += 0.8
                        elif abs_td > 0.05:
                            s['error_score'] += 0.4

            if not phrase_stats:
                return []

            scored = []
            for phrase_idx, s in phrase_stats.items():
                denom = max(1, s['count'])
                score = s['error_score'] / denom
                scored.append((phrase_idx, score, s))
            scored.sort(key=lambda x: x[1], reverse=True)

            scores = [x[1] for x in scored]
            if len(scores) >= 4:
                q = statistics.quantiles(scores, n=100, method='inclusive')
                p60 = q[59]
                p85 = q[84]
            elif len(scores) >= 2:
                p60 = statistics.mean(scores)
                p85 = max(scores)
            else:
                p60 = p85 = scores[0]

            out = []
            for phrase_idx, score, s in scored[:3]:
                reasons = []
                if s['missing'] > 0:
                    reasons.append("missing notes cluster")
                if s['wrong'] > 0:
                    reasons.append("pitch mismatches")
                if s['timing_bad'] > 0:
                    reasons.append("timing instability")
                if s['extra'] > 0 and len(reasons) < 2:
                    reasons.append("extra-note insertions")
                if not reasons:
                    reasons = ["elevated combined error density"]

                out.append({
                    'section': f"Phrase {int(phrase_idx) + 1}",
                    'reason': "; ".join(reasons[:2]),
                    'difficulty': _difficulty_label(score, p60, p85),
                    'score': round(float(score), 3),
                })
            return out

        ref_exact_lookup, ref_loose_lookup, ref_ordered_measures = _build_measure_lookup(reference_notes)
        perf_exact_lookup, perf_loose_lookup, perf_ordered_measures = _build_measure_lookup(performance_notes)
        ref_exact_cursor = defaultdict(int)
        ref_loose_cursor = defaultdict(int)
        perf_exact_cursor = defaultdict(int)
        perf_loose_cursor = defaultdict(int)

        reference_measure_meta = defaultdict(lambda: {
            'note_count': 0,
            'min_pitch': 127,
            'max_pitch': 0,
        })
        for note in reference_notes:
            m = _measure_from_note(note)
            if m is None:
                continue
            pitch = _as_int(note.get('pitch'), 60) or 60
            meta = reference_measure_meta[m]
            meta['note_count'] += 1
            meta['min_pitch'] = min(meta['min_pitch'], pitch)
            meta['max_pitch'] = max(meta['max_pitch'], pitch)

        performance_measures = []
        for note in performance_notes:
            m = _measure_from_note(note)
            if m is not None:
                performance_measures.append(m)

        reference_max_measure = max(reference_measure_meta.keys()) if reference_measure_meta else 0
        performance_max_measure = max(performance_measures) if performance_measures else 0

        def _pair_anchor_time(pair: Dict[str, Any]) -> float:
            rn = pair.get('reference_note')
            pn = pair.get('performance_note')
            if isinstance(rn, dict):
                t = _as_float(rn.get('start'), None)
                if t is not None:
                    return t
            if isinstance(pn, dict):
                t = _as_float(pn.get('start'), None)
                if t is not None:
                    return t
            return 0.0

        pairs_sorted = sorted(
            pairs,
            key=lambda p: (
                _pair_anchor_time(p),
                _as_int((p.get('reference_note') or p.get('performance_note') or {}).get('pitch'), 0)
            )
        )

        total_ref_pairs = sum(1 for p in pairs_sorted if isinstance(p.get('reference_note'), dict))

        def _consume_lookup_measure(note: Dict[str, Any], is_reference: bool) -> Optional[int]:
            exact_lookup = ref_exact_lookup if is_reference else perf_exact_lookup
            loose_lookup = ref_loose_lookup if is_reference else perf_loose_lookup
            exact_cursor = ref_exact_cursor if is_reference else perf_exact_cursor
            loose_cursor = ref_loose_cursor if is_reference else perf_loose_cursor

            k = _note_key(note)
            if k is not None and k in exact_lookup:
                idx = exact_cursor[k]
                vals = exact_lookup[k]
                if idx < len(vals):
                    exact_cursor[k] += 1
                    return int(vals[idx])

            lk = _loose_note_key(note)
            if lk is not None and lk in loose_lookup:
                idx = loose_cursor[lk]
                vals = loose_lookup[lk]
                if idx < len(vals):
                    loose_cursor[lk] += 1
                    return int(vals[idx])
            return None

        def _fallback_reference_measure(ref_seen: int) -> Optional[int]:
            if not ref_ordered_measures:
                return None
            if total_ref_pairs <= 1:
                return int(ref_ordered_measures[0])
            ratio = max(0.0, min(1.0, ref_seen / max(1, total_ref_pairs - 1)))
            idx = int(round(ratio * (len(ref_ordered_measures) - 1)))
            return int(ref_ordered_measures[idx])

        measure_stats = defaultdict(lambda: {
            'reference_notes': 0,
            'matched': 0,
            'missing': 0,
            'extra': 0,
            'wrong': 0,
            'timing_moderate': 0,
            'timing_severe': 0,
            'timing_abs_sum': 0.0,
            'timing_samples': 0,
            'confidence_penalty': 0.0,
            'error_score': 0.0,
            'phrase_ids': set(),
        })

        perf_anchors: List[tuple] = []
        ref_seen = 0

        for pair in pairs_sorted:
            ref_note = pair.get('reference_note')
            perf_note = pair.get('performance_note')
            phrase_idx = pair.get('phrase_index')

            if not isinstance(ref_note, dict):
                continue

            measure = _measure_from_note(ref_note)
            if measure is None:
                measure = _consume_lookup_measure(ref_note, is_reference=True)
            if measure is None:
                measure = _fallback_reference_measure(ref_seen)
            ref_seen += 1

            if measure is None:
                continue

            s = measure_stats[measure]
            if isinstance(phrase_idx, int):
                s['phrase_ids'].add(int(phrase_idx))
            s['reference_notes'] += 1

            err = str(pair.get('error_type', 'none'))
            pitch_diff = _as_int(pair.get('pitch_difference'), 0) or 0

            if isinstance(perf_note, dict):
                s['matched'] += 1
                if pitch_diff != 0:
                    s['wrong'] += 1
                    s['error_score'] += 2.1

                abs_td = _timing_abs(pair)
                if abs_td is not None:
                    s['timing_abs_sum'] += float(abs_td)
                    s['timing_samples'] += 1
                    if abs_td > 0.10:
                        s['timing_severe'] += 1
                        s['error_score'] += 1.0
                    elif abs_td > 0.05:
                        s['timing_moderate'] += 1
                        s['error_score'] += 0.5

                conf = pair.get('alignment_confidence')
                if isinstance(conf, (int, float)):
                    c = max(0.0, min(1.0, float(conf)))
                    penalty = (1.0 - c) * 0.35
                    s['confidence_penalty'] += penalty
                    s['error_score'] += penalty

                if err not in {'none', 'missing_note', 'extra_note', 'ornament_insertion'} and pitch_diff == 0:
                    s['error_score'] += 0.6

                perf_start = _as_float(perf_note.get('start'), None)
                if perf_start is not None:
                    perf_anchors.append((float(perf_start), int(measure)))
            else:
                s['missing'] += 1
                missing_weight = 2.6
                reason = str(pair.get('reason', ''))
                if 'phrase_no_perf' in reason:
                    missing_weight += 0.4
                s['error_score'] += missing_weight

        perf_anchors.sort(key=lambda x: x[0])
        perf_anchor_times = [x[0] for x in perf_anchors]

        def _resolve_extra_measure(perf_note: Dict[str, Any]) -> Optional[int]:
            perf_start = _as_float(perf_note.get('start'), None)
            if perf_start is not None and perf_anchors:
                i = bisect_left(perf_anchor_times, perf_start)
                candidates = []
                if i < len(perf_anchors):
                    candidates.append(perf_anchors[i])
                if i > 0:
                    candidates.append(perf_anchors[i - 1])
                if candidates:
                    best = min(candidates, key=lambda x: abs(x[0] - perf_start))
                    return int(best[1])

            perf_measure = _measure_from_note(perf_note)
            if perf_measure is None:
                perf_measure = _consume_lookup_measure(perf_note, is_reference=False)

            if perf_measure is not None:
                if reference_max_measure > 0 and performance_max_measure > 0:
                    scaled = int(round((perf_measure / max(1, performance_max_measure)) * reference_max_measure))
                    return max(1, min(reference_max_measure, scaled))
                return int(perf_measure)

            if reference_max_measure > 0:
                return int(reference_max_measure)
            return None

        for pair in pairs_sorted:
            ref_note = pair.get('reference_note')
            perf_note = pair.get('performance_note')
            if isinstance(ref_note, dict) or not isinstance(perf_note, dict):
                continue

            measure = _resolve_extra_measure(perf_note)
            if measure is None:
                continue

            s = measure_stats[measure]
            phrase_idx = pair.get('phrase_index')
            if isinstance(phrase_idx, int):
                s['phrase_ids'].add(int(phrase_idx))
            s['extra'] += 1

            err = str(pair.get('error_type', 'extra_note'))
            if err == 'ornament_insertion':
                s['error_score'] += 0.9
            elif err == 'extra_note':
                s['error_score'] += 1.4
            else:
                s['error_score'] += 1.1

        if not measure_stats:
            return _fallback_phrase_sections()

        complexity_values = []
        for measure, meta in reference_measure_meta.items():
            note_count = max(1, int(meta.get('note_count', 0)))
            span = int(meta.get('max_pitch', 0)) - int(meta.get('min_pitch', 0))
            complexity_values.append((measure, note_count, span))

        median_measure_notes = 1.0
        if complexity_values:
            note_counts = [x[1] for x in complexity_values]
            median_measure_notes = statistics.median(note_counts) if note_counts else 1.0
            if median_measure_notes <= 0:
                median_measure_notes = 1.0

        measure_scores: Dict[int, float] = {}
        measure_complexity: Dict[int, float] = {}
        for measure, s in measure_stats.items():
            ref_note_count = int(
                reference_measure_meta.get(measure, {}).get('note_count', s.get('reference_notes', 0))
            )
            ref_note_count = max(1, ref_note_count)

            err_density = float(s['error_score']) / ref_note_count

            meta = reference_measure_meta.get(measure, {})
            measure_note_count = int(meta.get('note_count', ref_note_count))
            pitch_span = int(meta.get('max_pitch', 0)) - int(meta.get('min_pitch', 0)) if meta else 0
            density_factor = measure_note_count / max(1.0, median_measure_notes)
            span_factor = min(2.0, max(0.0, pitch_span / 12.0))
            complexity = (0.7 * density_factor) + (0.3 * span_factor)

            score = err_density + (0.2 * complexity)

            measure_complexity[measure] = complexity
            measure_scores[measure] = score

        if not measure_scores:
            return _fallback_phrase_sections()

        ranked_measures = sorted(measure_scores.items(), key=lambda x: x[1], reverse=True)
        score_values = [x[1] for x in ranked_measures]

        if len(score_values) >= 4:
            q = statistics.quantiles(score_values, n=100, method='inclusive')
            p60 = q[59]
            p75 = q[74]
            p85 = q[84]
        elif len(score_values) >= 2:
            p60 = statistics.mean(score_values)
            p75 = max(score_values)
            p85 = max(score_values)
        else:
            p60 = p75 = p85 = score_values[0]

        mean_score = statistics.mean(score_values)
        std_score = statistics.stdev(score_values) if len(score_values) > 1 else 0.0
        threshold = max(p75, mean_score + 0.35 * std_score)

        selected = [m for m, score in ranked_measures if score >= threshold]
        if len(selected) < 2:
            selected = [m for m, _ in ranked_measures[:min(4, len(ranked_measures))]]
        selected = sorted(set(selected))

        if not selected:
            return _fallback_phrase_sections()

        groups = []
        g_start = selected[0]
        g_end = selected[0]
        for m in selected[1:]:
            if m <= g_end + 1:
                g_end = m
            else:
                groups.append((g_start, g_end))
                g_start = m
                g_end = m
        groups.append((g_start, g_end))

        def _build_reason(agg: Dict[str, Any], ref_notes_in_span: int, complexity: float) -> str:
            reasons = []
            ref_base = max(1, ref_notes_in_span)
            matched_base = max(1, int(agg['matched']))

            if (agg['missing'] / ref_base) >= 0.12:
                reasons.append("high missing-note density")
            if (agg['wrong'] / ref_base) >= 0.06:
                reasons.append("frequent pitch mismatches")
            if (agg['timing_severe'] / matched_base) >= 0.20:
                reasons.append("unstable timing")
            elif agg['timing_samples'] > 0:
                avg_ms = (agg['timing_abs_sum'] / agg['timing_samples']) * 1000.0
                if avg_ms >= 80:
                    reasons.append("large timing deviations")
            if (agg['extra'] / ref_base) >= 0.10 and len(reasons) < 2:
                reasons.append("extra-note insertions")
            if complexity >= 1.35 and len(reasons) < 2:
                reasons.append("dense/wide-range writing")
            if not reasons:
                reasons.append("elevated combined error density")
            return "; ".join(reasons[:2])

        sections: List[Dict[str, Any]] = []
        for start_m, end_m in groups:
            span_measures = list(range(start_m, end_m + 1))
            agg = {
                'reference_notes': 0,
                'matched': 0,
                'missing': 0,
                'extra': 0,
                'wrong': 0,
                'timing_moderate': 0,
                'timing_severe': 0,
                'timing_abs_sum': 0.0,
                'timing_samples': 0,
                'phrase_ids': set(),
            }
            scores = []
            complexities = []
            for m in span_measures:
                s = measure_stats.get(m)
                if s is None:
                    continue
                agg['reference_notes'] += int(s['reference_notes'])
                agg['matched'] += int(s['matched'])
                agg['missing'] += int(s['missing'])
                agg['extra'] += int(s['extra'])
                agg['wrong'] += int(s['wrong'])
                agg['timing_moderate'] += int(s['timing_moderate'])
                agg['timing_severe'] += int(s['timing_severe'])
                agg['timing_abs_sum'] += float(s['timing_abs_sum'])
                agg['timing_samples'] += int(s['timing_samples'])
                agg['phrase_ids'].update(s['phrase_ids'])
                if m in measure_scores:
                    scores.append(float(measure_scores[m]))
                if m in measure_complexity:
                    complexities.append(float(measure_complexity[m]))

            if not scores:
                continue

            score = statistics.mean(scores)
            complexity = statistics.mean(complexities) if complexities else 0.0
            difficulty = _difficulty_label(score, p60, p85)
            reason = _build_reason(agg, agg['reference_notes'], complexity)

            if start_m == end_m:
                section_label = f"Measure {start_m}"
            else:
                section_label = f"Measures {start_m}-{end_m}"

            out_entry = {
                'section': section_label,
                'reason': reason,
                'difficulty': difficulty,
                'score': round(float(score), 3),
            }

            if agg['phrase_ids']:
                phrase_ids = sorted(int(x) for x in agg['phrase_ids'])
                if len(phrase_ids) == 1:
                    out_entry['phrase'] = f"Phrase {phrase_ids[0] + 1}"
                else:
                    out_entry['phrase'] = f"Phrases {phrase_ids[0] + 1}-{phrase_ids[-1] + 1}"

            sections.append(out_entry)

        sections.sort(key=lambda d: d.get('score', 0), reverse=True)
        if sections:
            return sections[:3]

        return _fallback_phrase_sections()

    def _find_fastest_passage(self, notes: List[Dict]) -> Dict:
        """Find the fastest local passage based on note density and local IOI."""
        if len(notes) < 4:
            return {'tempo': 'N/A', 'location': 'N/A'}

        notes_sorted = sorted(notes, key=lambda n: float(n.get('start', 0.0)))
        # Use unique onset events so block chords are not misclassified as "fast".
        starts = sorted({round(float(n.get('start', 0.0)), 6) for n in notes_sorted})
        if len(starts) < 3:
            return {'tempo': 'N/A', 'location': 'N/A'}

        window_s = 2.0
        best_i = 0
        best_j = 0
        best_nps = 0.0

        j = 0
        for i in range(len(starts)):
            while j + 1 < len(starts) and starts[j + 1] - starts[i] <= window_s:
                j += 1
            count = (j - i + 1)
            # Normalize by a fixed window to avoid divide-by-near-zero spikes.
            nps = count / window_s
            if nps > best_nps:
                best_nps = nps
                best_i = i
                best_j = j

        local = starts[best_i:best_j + 1]
        iois = [local[k] - local[k - 1] for k in range(1, len(local)) if (local[k] - local[k - 1]) > 1e-6]
        if iois:
            median_ioi = statistics.median(iois)
            bpm = 60.0 / median_ioi if median_ioi > 1e-6 else 0.0
            if bpm > 300:
                tempo_text = ">300 BPM (very fast subdivision run)"
            else:
                tempo_text = f"~{int(round(bpm))} BPM (median IOI)"
        else:
            tempo_text = "N/A"

        t0 = starts[best_i]
        t1 = starts[best_j]
        return {
            'tempo': tempo_text,
            'location': f"{t0:.2f}s - {t1:.2f}s",
            'notes_per_second': round(best_nps, 2),
            'window_seconds': round(window_s, 2),
            'note_count': int(best_j - best_i + 1)
        }

    def _find_largest_interval(self, notes: List[Dict]) -> Dict:
        """Find the largest melodic leap between consecutive note onsets."""
        if len(notes) < 2:
            return {'interval': 0, 'location': 'N/A'}

        notes_sorted = sorted(notes, key=lambda n: (float(n.get('start', 0.0)), int(n.get('pitch', 0))))
        best = None
        for i in range(1, len(notes_sorted)):
            p0 = int(notes_sorted[i - 1].get('pitch', 0))
            p1 = int(notes_sorted[i].get('pitch', 0))
            leap = abs(p1 - p0)
            if best is None or leap > best['semitones']:
                best = {
                    'semitones': leap,
                    'from_pitch': p0,
                    'to_pitch': p1,
                    'time': float(notes_sorted[i].get('start', 0.0)),
                    'direction': 'up' if p1 >= p0 else 'down'
                }

        if best is None:
            return {'interval': 0, 'location': 'N/A'}

        return {
            'interval': f"{best['semitones']} semitones",
            'location': f"{best['time']:.2f}s",
            'direction': best['direction'],
            'from_pitch': best['from_pitch'],
            'to_pitch': best['to_pitch']
        }

    def _analyze_tempo_profile(self) -> str:
        """Analyze tempo profile from timing and rhythm metrics."""
        error_metrics = self.error_analysis.get('metrics', {})
        timing = error_metrics.get('timing_errors', {}) if isinstance(error_metrics, dict) else {}
        rhythm = error_metrics.get('rhythmic_consistency', {}) if isinstance(error_metrics, dict) else {}
        perf_timing = (
            self.performance_data.get('performance_data', {}).get('timing_consistency', {})
            if isinstance(self.performance_data, dict)
            else {}
        )
        parsed_timing = self.performance_data.get('timing', {}) if isinstance(self.performance_data, dict) else {}

        def _num(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        avg_tempo = _num(parsed_timing.get('average_tempo'), None)
        tempo_anchor = f" around {int(round(avg_tempo))} BPM" if isinstance(avg_tempo, (int, float)) else ""

        # Prefer reference-comparison timing metrics when available.
        if timing.get('available', True):
            std_ms = _num(timing.get('std_error_ms'), 0.0) or 0.0
            mean_ms = _num(timing.get('mean_error_ms'), 0.0) or 0.0
            rushing = _num(timing.get('rushing_percentage'), 0.0) or 0.0
            dragging = _num(timing.get('dragging_percentage'), 0.0) or 0.0

            if std_ms < 30 and abs(mean_ms) < 20:
                stability = "Very steady tempo control"
            elif std_ms < 60:
                stability = "Generally steady tempo"
            elif std_ms < 120:
                stability = "Noticeable tempo fluctuation"
            else:
                stability = "Large tempo inconsistency"

            if rushing > dragging + 12:
                tendency = "with a forward (rushing) tendency"
            elif dragging > rushing + 12:
                tendency = "with a behind-the-beat (dragging) tendency"
            else:
                tendency = "with balanced early/late timing"

            tempo_stability = rhythm.get('tempo_stability', {}) if isinstance(rhythm, dict) else {}
            rubato_count = 0
            if isinstance(tempo_stability, dict):
                rubato = tempo_stability.get('rubato_patterns', [])
                rubato_count = len(rubato) if isinstance(rubato, list) else 0

            rubato_text = ""
            if rubato_count >= 2:
                rubato_text = ", including recurring rubato-like shaping"
            elif rubato_count == 1:
                rubato_text = ", with occasional rubato-like shaping"

            return f"{stability}{tempo_anchor} {tendency}{rubato_text}."

        # Fallback to solo timing-consistency metrics from parser output.
        if perf_timing.get('available', False):
            stability_score = _num(perf_timing.get('stability_score'), 0.5) or 0.5
            drift = _num(perf_timing.get('tempo_drift_ratio'), 0.0) or 0.0
            tendency = str(perf_timing.get('tempo_tendency', 'stable')).replace('_', ' ')

            if stability_score >= 0.85:
                stability = "Stable internal pulse"
            elif stability_score >= 0.7:
                stability = "Moderately stable pulse"
            else:
                stability = "Variable pulse control"

            drift_text = ""
            if abs(drift) >= 0.08:
                drift_text = f"; tempo drift ~{drift * 100:.1f}% ({tendency})"

            return f"{stability}{tempo_anchor}{drift_text}."

        return "Tempo profile unavailable (insufficient timing data)."
    
    def _analyze_dynamic_contour(self) -> str:
        """Analyze dynamic contour/shape from velocity and expression metrics."""
        error_metrics = self.error_analysis.get('metrics', {})
        dynamics = error_metrics.get('dynamic_control', {}) if isinstance(error_metrics, dict) else {}
        velocity_profile = (
            self.performance_data.get('performance_data', {}).get('velocity_profile', {})
            if isinstance(self.performance_data, dict)
            else {}
        )

        def _num(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        dynamic_range = _num(dynamics.get('dynamic_range'), None)
        if dynamic_range is None:
            vr = velocity_profile.get('dynamic_range', {}) if isinstance(velocity_profile, dict) else {}
            if isinstance(vr, dict):
                vmin = _num(vr.get('min'), None)
                vmax = _num(vr.get('max'), None)
                if vmin is not None and vmax is not None:
                    dynamic_range = max(0.0, vmax - vmin)
        if dynamic_range is None:
            dynamic_range = 0.0

        reference_range = _num(dynamics.get('reference_dynamic_range'), None)
        avg_vel = _num(dynamics.get('average_velocity'), None)
        if avg_vel is None:
            avg_vel = _num(velocity_profile.get('mean_velocity'), 0.0) or 0.0

        patterns = dynamics.get('dynamic_patterns', [])
        cresc = 0
        decresc = 0
        if isinstance(patterns, list):
            for p in patterns:
                if not isinstance(p, dict):
                    continue
                kind = str(p.get('type', ''))
                if kind == 'crescendo':
                    cresc += 1
                elif kind == 'decrescendo':
                    decresc += 1

        if dynamic_range >= 65:
            contour = "Wide dynamic contour"
        elif dynamic_range >= 40:
            contour = "Clear dynamic contour"
        elif dynamic_range >= 25:
            contour = "Moderate dynamic contour"
        else:
            contour = "Limited dynamic contour"

        if cresc > 0 and decresc > 0:
            shaping = "with both crescendo and decrescendo gestures"
        elif cresc > 0:
            shaping = "with mostly crescendo shaping"
        elif decresc > 0:
            shaping = "with mostly decrescendo shaping"
        else:
            shaping = "with minimal large-scale shaping"

        center = ""
        if avg_vel >= 90:
            center = " and a forte-leaning center"
        elif avg_vel <= 50:
            center = " and a piano-leaning center"

        ref_text = ""
        if isinstance(reference_range, (int, float)) and reference_range > 1:
            ratio = dynamic_range / reference_range
            if ratio < 0.65:
                ref_text = " (below reference contrast)"
            elif ratio > 1.25:
                ref_text = " (broader than reference contrast)"

        return f"{contour} {shaping}{center}{ref_text}."
    
    def _analyze_articulation_style(self) -> str:
        """Analyze articulation style from articulation metrics."""
        error_metrics = self.error_analysis.get('metrics', {})
        articulation = error_metrics.get('articulation', {}) if isinstance(error_metrics, dict) else {}
        perf_art = (
            self.performance_data.get('performance_data', {}).get('articulation_patterns', {})
            if isinstance(self.performance_data, dict)
            else {}
        )

        def _num(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        staccato = _num(articulation.get('staccato_percentage'), None)
        legato = _num(articulation.get('legato_percentage'), None)
        consistency = _num(articulation.get('articulation_consistency'), None)
        detached = _num(perf_art.get('detached_percentage'), None)

        if staccato is None:
            staccato = _num(perf_art.get('staccato_percentage'), 0.0) or 0.0
        if legato is None:
            legato = _num(perf_art.get('legato_percentage'), 0.0) or 0.0

        if legato >= 70 and staccato <= 20:
            base = "Predominantly legato articulation"
        elif staccato >= 70 and legato <= 20:
            base = "Predominantly staccato articulation"
        else:
            base = "Mixed articulation profile"

        if consistency is not None:
            if consistency >= 0.8:
                consistency_text = "with high consistency"
            elif consistency >= 0.6:
                consistency_text = "with moderate consistency"
            else:
                consistency_text = "with variable consistency"
        else:
            consistency_text = "with uncalibrated consistency"

        detached_text = ""
        if isinstance(detached, (int, float)) and detached >= 20:
            detached_text = f"; detached touch appears on {detached:.1f}% of transitions"

        return f"{base} ({staccato:.1f}% staccato, {legato:.1f}% legato) {consistency_text}{detached_text}."
    
    def _create_benchmarks(self) -> Dict[str, float]:
        """Create benchmark scores for comparison."""
        return {
            'beginner_target': 60,
            'intermediate_target': 75,
            'advanced_target': 85,
            'professional_target': 92
        }
    
    def _identify_highest_priority_areas(self) -> List[str]:
        """Identify highest priority improvement areas."""
        priorities = []
        error_metrics = self.error_analysis.get('metrics', {})
        
        note_acc = error_metrics.get('note_accuracy', {})
        if note_acc.get('accuracy_percentage', 100) < 85:
            priorities.append("Note accuracy")
        
        timing = error_metrics.get('timing_errors', {})
        std_raw = timing.get('std_error_ms', 0)
        std_error = float(std_raw) if isinstance(std_raw, (int, float)) else 0.0
        if std_error > 60:
            priorities.append("Timing consistency")
        
        return priorities[:2] if priorities else ["Musical expression"]
    
    def _identify_quick_wins(self) -> List[str]:
        """Identify areas that could improve quickly from near-threshold metrics."""
        error_metrics = self.error_analysis.get('metrics', {})
        note_acc = error_metrics.get('note_accuracy', {}) if isinstance(error_metrics, dict) else {}
        timing = error_metrics.get('timing_errors', {}) if isinstance(error_metrics, dict) else {}
        rhythm = error_metrics.get('rhythmic_consistency', {}) if isinstance(error_metrics, dict) else {}
        dynamics = error_metrics.get('dynamic_control', {}) if isinstance(error_metrics, dict) else {}
        articulation = error_metrics.get('articulation', {}) if isinstance(error_metrics, dict) else {}

        def _num(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        quick: List[str] = []

        accuracy = _num(note_acc.get('accuracy_percentage'), 0.0) or 0.0
        if 75 <= accuracy < 95:
            quick.append("Raise note accuracy with slow blocked practice on the top 1-2 weak passages")

        if timing.get('available', True):
            std_ms = _num(timing.get('std_error_ms'), 0.0) or 0.0
            rushing = _num(timing.get('rushing_percentage'), 0.0) or 0.0
            dragging = _num(timing.get('dragging_percentage'), 0.0) or 0.0

            if 35 <= std_ms <= 120:
                quick.append("Reduce timing spread by looping short fragments with metronome subdivision")
            if rushing > dragging + 10 and rushing <= 60:
                quick.append("Trim rushing tendency by delaying right-hand entries against a steady click")
            elif dragging > rushing + 10 and dragging <= 60:
                quick.append("Trim dragging tendency by practicing phrase pickups with forward motion")

        rhythm_score = _num(rhythm.get('duration_consistency_score'), 0.0) or 0.0
        if 0.6 <= rhythm_score < 0.85:
            quick.append("Improve rhythmic consistency through counted subdivision on transitions")

        dyn_range = _num(dynamics.get('dynamic_range'), 0.0) or 0.0
        ref_range = _num(dynamics.get('reference_dynamic_range'), None)
        if isinstance(ref_range, (int, float)) and ref_range > 1 and dyn_range < ref_range * 0.75:
            quick.append("Expand dynamic contrast in repeated ideas to better match the reference profile")
        elif 20 <= dyn_range < 45:
            quick.append("Add clearer dynamic contrast between phrase peaks and releases")

        art_cons = _num(articulation.get('articulation_consistency'), 0.0) or 0.0
        if 0.55 <= art_cons < 0.8:
            quick.append("Tighten articulation consistency with short staccato/legato contrast drills")

        # Tie quick wins to currently detected high-impact section if available.
        challenging = self._identify_challenging_sections()
        if challenging:
            section = challenging[0].get('section')
            if isinstance(section, str) and section.strip():
                quick.append(f"Isolate {section} in 2-4 bar loops before full-run integration")

        deduped = self._dedupe_text_list(quick)
        if len(deduped) >= 3:
            return deduped[:3]

        fillers = [
            "Stabilize one short section with metronome layering",
            "Refine one articulation pattern at slow tempo",
            "Shape one phrase with clearer dynamic contrast",
        ]
        for item in fillers:
            if len(deduped) >= 3:
                break
            if item not in deduped:
                deduped.append(item)
        return deduped[:3]
    
    def _identify_long_term_goals(self) -> List[str]:
        """Identify long-term development goals from persistent gaps."""
        error_metrics = self.error_analysis.get('metrics', {})
        note_acc = error_metrics.get('note_accuracy', {}) if isinstance(error_metrics, dict) else {}
        timing = error_metrics.get('timing_errors', {}) if isinstance(error_metrics, dict) else {}
        rhythm = error_metrics.get('rhythmic_consistency', {}) if isinstance(error_metrics, dict) else {}
        dynamics = error_metrics.get('dynamic_control', {}) if isinstance(error_metrics, dict) else {}
        articulation = error_metrics.get('articulation', {}) if isinstance(error_metrics, dict) else {}
        score_data = error_metrics.get('performance_score', {}) if isinstance(error_metrics, dict) else {}

        def _num(v: Any, default: Optional[float] = 0.0) -> Optional[float]:
            if isinstance(v, (int, float)):
                return float(v)
            return default

        goals: List[str] = []

        accuracy = _num(note_acc.get('accuracy_percentage'), 0.0) or 0.0
        if accuracy < 90:
            goals.append("Build dependable note accuracy at performance tempo through staged tempo ramps")

        std_ms = _num(timing.get('std_error_ms'), 0.0) or 0.0
        rhythm_score = _num(rhythm.get('duration_consistency_score'), 0.0) or 0.0
        if std_ms > 60 or rhythm_score < 0.75:
            goals.append("Develop durable tempo/rhythm control across full phrases, not only isolated bars")

        dyn_range = _num(dynamics.get('dynamic_range'), 0.0) or 0.0
        ref_range = _num(dynamics.get('reference_dynamic_range'), None)
        if dyn_range < 40 or (isinstance(ref_range, (int, float)) and ref_range > 1 and dyn_range < ref_range * 0.75):
            goals.append("Expand and stabilize dynamic range while preserving tone quality")

        art_cons = _num(articulation.get('articulation_consistency'), 0.0) or 0.0
        if art_cons < 0.75:
            goals.append("Improve articulation control so touch remains consistent through tempo changes")

        score = _num(score_data.get('overall_score'), 0.0) or 0.0
        if score >= 80:
            goals.append("Expand repertoire complexity while maintaining current technical reliability")
        else:
            goals.append("Establish a sustainable weekly practice cycle with measured metric targets")

        deduped = self._dedupe_text_list(goals)
        return deduped[:3]
    
    def _set_target_scores(self, current_score: float) -> Dict[str, float]:
        """
        Set realistic target scores for progress tracking.

        Fix:
        - Never produce targets below the current score.
        - Use a ceiling of 100 (since your grading can output 100.0).
        - Apply diminishing returns near the top.
        """
        MAX_SCORE = 100.0

        # If already essentially perfect, you're "maintaining", not "improving"
        if current_score >= 99.5:
            return {
                'next_week': round(current_score, 1),
                'one_month': round(current_score, 1),
                'three_months': round(current_score, 1)
            }

        # Diminishing increments as score increases
        if current_score >= 90:
            week_inc, month_inc, three_month_inc = 2.0, 4.0, 6.0
        elif current_score >= 80:
            week_inc, month_inc, three_month_inc = 4.0, 7.0, 10.0
        else:
            week_inc, month_inc, three_month_inc = 6.0, 10.0, 15.0

        next_week = min(current_score + week_inc, MAX_SCORE)
        one_month = min(current_score + month_inc, MAX_SCORE)
        three_months = min(current_score + three_month_inc, MAX_SCORE)

        # Ensure monotonic targets and never below current
        next_week = max(next_week, current_score)
        one_month = max(one_month, next_week)
        three_months = max(three_months, one_month)

        return {
            'next_week': round(next_week, 1),
            'one_month': round(one_month, 1),
            'three_months': round(three_months, 1)
        }
    
    def _infer_student_level(self) -> str:
        """Infer student level from performance."""
        score = self.error_analysis.get('metrics', {}).get('performance_score', {}).get('overall_score', 0)
        return self._determine_performance_level(score)
    
    def _estimate_student_age(self) -> str:
        """Estimate student age range."""
        # Very rough estimation based on performance characteristics
        score = self.error_analysis.get('metrics', {}).get('performance_score', {}).get('overall_score', 0)
        
        if score >= 80:
            return "Teen to Adult"
        elif score >= 70:
            return "Older Child to Teen"
        elif score >= 60:
            return "Child (8-12)"
        else:
            return "Young Beginner"
    
    def _infer_practice_habits(self) -> str:
        """Infer likely practice habits."""
        error_metrics = self.error_analysis.get('metrics', {})
        consistency = error_metrics.get('rhythmic_consistency', {}).get('duration_consistency_score', 0)
        
        if consistency > 0.8:
            return "Likely practices regularly with metronome"
        elif consistency > 0.6:
            return "Regular practice, could be more focused"
        else:
            return "Would benefit from more structured practice sessions"
    
    def _assess_piece_difficulty(self) -> str:
        """Assess the difficulty level of the piece."""
        reference_notes = self.reference_data.get('notes', [])
        return self._assess_technical_difficulty(reference_notes)
    
    def _infer_composer_style(self) -> str:
        """Infer composer style from piece characteristics."""
        # Simplified inference
        pitch_range = self._calculate_pitch_range(self.reference_data.get('notes', []))
        range_size = pitch_range.get('range', 0)
        
        if range_size > 48:  # 4 octaves
            return "Romantic/Virtuosic"
        elif range_size > 36:  # 3 octaves
            return "Classical"
        else:
            return "Baroque/Early Classical"
    
    def _infer_musical_period(self) -> str:
        """Infer musical period from piece characteristics."""
        style = self._infer_composer_style()
        
        if 'Romantic' in style:
            return "Romantic Period"
        elif 'Classical' in style:
            return "Classical Period"
        elif 'Baroque' in style:
            return "Baroque Period"
        else:
            return "Various/Modern"
    
    def _extract_detailed_data(self) -> Dict[str, Any]:
        """Extract detailed data for advanced analysis."""
        return {
            'raw_alignment': self.alignment[:100],  # First 100 aligned pairs
            'error_categories': self.error_analysis.get('error_categories', {}),
            'note_level_analysis': self._extract_note_level_data()
        }

    def _dedupe_text_list(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for item in items:
            if not isinstance(item, str):
                continue
            key = item.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(item.strip())
        return out
    
    def _extract_note_level_data(self) -> List[Dict]:
        """Extract note-level analysis data."""
        note_data = []
        aligned_pairs = self.alignment[:50]  # First 50 notes
        
        for pair in aligned_pairs:
            if pair.get('reference_note') and pair.get('performance_note'):
                note_data.append({
                    'time': pair['reference_note'].get('start', 0),
                    'pitch': pair['reference_note'].get('pitch', 0),
                    'timing_error': pair.get('time_difference', 0),
                    'velocity_error': pair.get('velocity_difference', 0),
                    'alignment_confidence': pair.get('alignment_confidence', 0)
                })
        
        return note_data
    
    def save_summary(self, filepath: str, include_detailed: bool = False):
        """
        Save JSON summary to file.
        
        Args:
            filepath: Path to save JSON file
            include_detailed: Whether to include detailed data
        """
        summary = self.create_summary(include_detailed_data=include_detailed)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"Summary saved to: {filepath}")
        return summary


# Utility functions for quick summarization
def create_gpt_summary(analysis_data: Dict[str, Any], 
                      output_file: str = None) -> Dict[str, Any]:
    """
    Create GPT-ready summary from analysis data.
    
    Args:
        analysis_data: Complete analysis data from pipeline
        output_file: Optional file to save summary
        
    Returns:
        GPT-ready summary JSON
    """
    summarizer = JSONSummarization(analysis_data)
    summary = summarizer.create_summary()
    
    if output_file:
        summarizer.save_summary(output_file)
    
    return summary


def create_minimal_summary(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create minimal summary for quick feedback.
    
    Returns:
        Minimal summary with key information
    """
    error_analysis = analysis_data.get('error_analysis', {})
    metrics = error_analysis.get('metrics', {})
    score_data = metrics.get('performance_score', {})
    timing_raw = metrics.get('timing_errors', {}).get('std_error_ms', 0)
    timing_consistency = float(timing_raw) if isinstance(timing_raw, (int, float)) else 0.0
    
    return {
        'overall_grade': score_data.get('grade', 'N/A'),
        'overall_score': score_data.get('overall_score', 0),
        'note_accuracy': metrics.get('note_accuracy', {}).get('accuracy_percentage', 0),
        'timing_consistency': timing_consistency,
        'top_recommendation': error_analysis.get('practice_recommendations', [''])[0] if error_analysis.get('practice_recommendations') else '',
        'timestamp': datetime.now().isoformat()
    }

