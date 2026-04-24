import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import statistics

class ErrorAnalysis:
    """
    Analyze performance errors by comparing reference and performance MIDI data.
    Focuses on educational aspects: timing, rhythm, dynamics, and note accuracy.
    """
    
    def __init__(self, analysis_data: Dict[str, Any]):
        """
        Initialize with analysis data containing reference, performance, and alignment.
        
        Args:
            analysis_data: Dictionary containing:
                - 'reference': Parsed reference MIDI data
                - 'performance': Parsed performance MIDI data
                - 'alignment': Aligned note pairs from time_alignment module
        """
        self.analysis_data = analysis_data
        self.reference_data = analysis_data.get('reference', {})
        self.performance_data = analysis_data.get('performance', {})
        self.aligned_notes = analysis_data.get('alignment', [])
        self.mode = self._detect_analysis_mode()
        
        # Performance metrics storage
        self.metrics = {}
        self.error_categories = {}
        self.practice_recommendations = []
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Run complete performance error analysis.
        
        Returns:
            Comprehensive error analysis results
        """
        print("Running error analysis...")
        
        # Reset metrics
        self.metrics = {}
        self.error_categories = {}
        self.practice_recommendations = []
        
        # Run all analysis modules
        self._analyze_note_accuracy()
        self._analyze_alignment_reliability()
        self._analyze_timing_errors()
        self._analyze_rhythmic_consistency()
        self._analyze_dynamic_control()
        self._analyze_articulation()
        self._analyze_phrasing_errors()
        self._analyze_pedaling_errors()
        
        # Generate overall performance score
        self._calculate_performance_score()
        
        # Generate practice recommendations
        self._generate_practice_recommendations()
        
        return {
            'metrics': self.metrics,
            'error_categories': self.error_categories,
            'practice_recommendations': self.practice_recommendations,
            'pedaling_recommendations': self._collect_pedaling_recommendations(),
            'analysis_mode': self.mode,
            'detailed_errors': self._get_detailed_error_list(),
            'performance_summary': self._get_performance_summary()
        }

    def analyze_pedaling(self) -> Dict[str, Any]:
        """Run only the pedaling stream for solo or compare workflows."""
        self.metrics = {}
        self.error_categories = {}
        self.practice_recommendations = []

        self._analyze_pedaling_errors()
        self.practice_recommendations = self._collect_pedaling_recommendations()

        return {
            'metrics': {'pedaling': self.metrics.get('pedaling', {})},
            'error_categories': {'pedaling': self.error_categories.get('pedaling', {})},
            'practice_recommendations': self.practice_recommendations,
            'pedaling_recommendations': self.practice_recommendations,
            'analysis_mode': self.mode,
        }

    def _detect_analysis_mode(self) -> str:
        """Infer whether this run is compare-mode or solo-mode."""
        if self.reference_data.get('notes'):
            return 'reference_comparison'

        reference_pedaling = self.reference_data.get('pedaling', {}) if isinstance(self.reference_data, dict) else {}
        if reference_pedaling.get('raw_events') or reference_pedaling.get('segments'):
            return 'reference_comparison'

        if self.aligned_notes:
            return 'reference_comparison'

        return 'solo'

    def _analyze_alignment_reliability(self):
        """Flag whether there are enough aligned note pairs for stable error metrics."""
        aligned_pairs = [
            p for p in self.aligned_notes
            if p.get('reference_note') and p.get('performance_note')
        ]
        reference_pairs = [p for p in self.aligned_notes if p.get('reference_note')]

        min_required = max(10, int(0.05 * max(1, len(reference_pairs))))
        is_reliable = len(aligned_pairs) >= min_required

        self.metrics['alignment_reliability'] = {
            'is_reliable': bool(is_reliable),
            'insufficient_alignment': bool(not is_reliable),
            'aligned_pair_count': int(len(aligned_pairs)),
            'minimum_required_aligned_pairs': int(min_required),
            'reason': None if is_reliable else 'too_few_aligned_pairs_for_stable_metrics',
        }
    
    def _analyze_note_accuracy(self):
        """Analyze note accuracy: missing notes, extra notes, wrong notes."""
        aligned_pairs = self.aligned_notes
        
        # Count different types of note matches
        matched_notes = []
        missing_notes = []
        extra_notes = []
        wrong_notes = []  # Notes with pitch errors
        
        for pair in aligned_pairs:
            if pair.get('reference_note') and pair.get('performance_note'):
                # Check for pitch errors
                pitch_diff = pair.get('pitch_difference', 0)
                if pitch_diff != 0:
                    wrong_notes.append(pair)
                else:
                    matched_notes.append(pair)
            elif pair.get('reference_note') and not pair.get('performance_note'):
                missing_notes.append(pair)
            elif pair.get('performance_note') and not pair.get('reference_note'):
                extra_notes.append(pair)
        
        total_reference_notes = len(missing_notes) + len(matched_notes) + len(wrong_notes)
        
        # Calculate accuracy percentages
        if total_reference_notes > 0:
            note_accuracy = (len(matched_notes) / total_reference_notes) * 100
            missing_percentage = (len(missing_notes) / total_reference_notes) * 100
        else:
            note_accuracy = 0
            missing_percentage = 0
        
        self.metrics['note_accuracy'] = {
            'total_reference_notes': total_reference_notes,
            'matched_notes': len(matched_notes),
            'missing_notes': len(missing_notes),
            'extra_notes': len(extra_notes),
            'wrong_notes': len(wrong_notes),
            'accuracy_percentage': round(note_accuracy, 1),
            'missing_percentage': round(missing_percentage, 1)
        }
        
        self.error_categories['note_accuracy'] = {
            'matched': matched_notes,
            'missing': missing_notes,
            'extra': extra_notes,
            'wrong': wrong_notes
        }
    
    def _analyze_timing_errors(self):
        """Analyze timing errors: rushing, dragging, inconsistency."""
        aligned_pairs = [p for p in self.aligned_notes 
                        if p.get('reference_note') and p.get('performance_note')]
        
        if not aligned_pairs:
            self.metrics['timing_errors'] = {
                'available': False,
                'reason': 'no_aligned_pairs',
                'mean_error_ms': None,
                'std_error_ms': None,
                'max_error_ms': None,
                'rushing_count': 0,
                'dragging_count': 0,
                'accurate_count': 0,
                'rushing_percentage': 0.0,
                'dragging_percentage': 0.0,
                'rhythmic_patterns': []
            }
            self.error_categories['timing'] = {
                'rushing': [],
                'dragging': [],
                'accurate': []
            }
            return
        
        time_differences = [pair.get('time_difference', 0) for pair in aligned_pairs]
        abs_time_differences = [abs(td) for td in time_differences]
        
        # Categorize timing errors
        rushing_notes = [pair for pair in aligned_pairs 
                        if pair.get('time_difference', 0) < -0.05]  # >50ms early
        dragging_notes = [pair for pair in aligned_pairs 
                         if pair.get('time_difference', 0) > 0.05]  # >50ms late
        accurate_notes = [pair for pair in aligned_pairs 
                         if -0.05 <= pair.get('time_difference', 0) <= 0.05]
        
        # Statistical analysis
        if time_differences:
            mean_error = statistics.mean(time_differences)
            std_error = statistics.stdev(time_differences) if len(time_differences) > 1 else 0
            max_error = max(abs_time_differences) if abs_time_differences else 0
            
            # Detect rhythmic patterns (grouping errors)
            rhythmic_patterns = self._detect_rhythmic_patterns(time_differences)
        else:
            mean_error = std_error = max_error = 0
            rhythmic_patterns = []
        
        self.metrics['timing_errors'] = {
            'available': True,
            'mean_error_ms': round(mean_error * 1000, 1),  # Convert to milliseconds
            'std_error_ms': round(std_error * 1000, 1),
            'max_error_ms': round(max_error * 1000, 1),
            'rushing_count': len(rushing_notes),
            'dragging_count': len(dragging_notes),
            'accurate_count': len(accurate_notes),
            'rushing_percentage': round((len(rushing_notes) / len(aligned_pairs)) * 100, 1) if aligned_pairs else 0,
            'dragging_percentage': round((len(dragging_notes) / len(aligned_pairs)) * 100, 1) if aligned_pairs else 0,
            'rhythmic_patterns': rhythmic_patterns
        }
        
        self.error_categories['timing'] = {
            'rushing': rushing_notes,
            'dragging': dragging_notes,
            'accurate': accurate_notes
        }
    
    def _analyze_rhythmic_consistency(self):
        """Analyze rhythmic consistency.

        NOTE (bug fix):
        In reference-comparison mode, rhythmic consistency should measure how
        well the performance matches the reference rhythm (IOIs/durations), not
        whether the performance note durations are "uniform". Using the
        coefficient of variation (CV) on raw performance durations penalizes
        pieces with naturally varied rhythms even when performed perfectly.
        """
        import math

        # Prefer aligned data for reference-vs-performance comparison
        aligned_pairs = [p for p in self.aligned_notes
                        if p.get('reference_note') and p.get('performance_note')]

        if len(aligned_pairs) >= 3:
            # Sort by reference time for stable consecutive IOI comparison
            aligned_pairs.sort(key=lambda x: x['reference_note']['start'])

            ref_starts = [p['reference_note']['start'] for p in aligned_pairs]
            perf_starts = [p['performance_note']['start'] for p in aligned_pairs]
            ref_durs = [p['reference_note']['duration'] for p in aligned_pairs]
            perf_durs = [p['performance_note']['duration'] for p in aligned_pairs]

            # Inter-onset intervals (IOIs)
            ref_ioi = [ref_starts[i] - ref_starts[i - 1] for i in range(1, len(ref_starts))]
            perf_ioi = [perf_starts[i] - perf_starts[i - 1] for i in range(1, len(perf_starts))]

            # Use log-ratio deviation so 0.5x and 2x are equally "bad"
            ioi_log_devs = []
            for r, p in zip(ref_ioi, perf_ioi):
                if r > 1e-6 and p > 1e-6:
                    ioi_log_devs.append(abs(math.log(p / r)))

            dur_log_devs = []
            for r, p in zip(ref_durs, perf_durs):
                if r > 1e-6 and p > 1e-6:
                    dur_log_devs.append(abs(math.log(p / r)))

            def _devs_to_score(devs, k_mean: float = 6.0, k_std: float = 3.0) -> float:
                if not devs:
                    return 0.5
                mean_dev = statistics.mean(devs)
                std_dev = statistics.stdev(devs) if len(devs) > 1 else 0.0
                score = 1.0 / (1.0 + k_mean * mean_dev + k_std * std_dev)
                return max(0.0, min(1.0, score))

            ioi_score = _devs_to_score(ioi_log_devs)
            dur_score = _devs_to_score(dur_log_devs)

            # IOI match is more important than absolute note length for rhythm
            rhythm_score = 0.7 * ioi_score + 0.3 * dur_score

            avg_ioi_ratio = statistics.mean([
                (p / r) for r, p in zip(ref_ioi, perf_ioi) if r > 1e-6
            ]) if ref_ioi else 0

            self.metrics['rhythmic_consistency'] = {
                'duration_consistency_score': round(rhythm_score, 2),
                'average_duration_ratio': round(statistics.mean([
                    (p / r) for r, p in zip(ref_durs, perf_durs) if r > 1e-6
                ]), 2) if ref_durs else 0,
                'average_ioi_ratio': round(avg_ioi_ratio, 3),
                'ioi_match_score': round(ioi_score, 2),
                'duration_match_score': round(dur_score, 2),
                'tempo_stability': self._analyze_tempo_stability()
            }
            return

        # Fallback (solo-style heuristic) when no alignment is available
        reference_notes = self.reference_data.get('notes', [])
        performance_notes = self.performance_data.get('notes', [])
        if not reference_notes or not performance_notes:
            return

        perf_durations = [note['duration'] for note in performance_notes]
        perf_intervals = self._calculate_note_intervals(performance_notes)
        duration_consistency = self._calculate_consistency_score(perf_durations)

        self.metrics['rhythmic_consistency'] = {
            'duration_consistency_score': round(duration_consistency, 2),
            'average_duration_ratio': 0,
            'duration_std': round(statistics.stdev(perf_durations), 3) if len(perf_durations) > 1 else 0,
            'interval_consistency': round(self._calculate_consistency_score(perf_intervals), 2) if perf_intervals else 0,
            'tempo_stability': self._analyze_tempo_stability()
        }
    
    def _analyze_dynamic_control(self):
        """Analyze dynamic (velocity) control and expression."""
        reference_notes = self.reference_data.get('notes', [])
        performance_notes = self.performance_data.get('notes', [])
        aligned_pairs = [
            p for p in self.aligned_notes
            if p.get('reference_note') and p.get('performance_note')
        ]
        
        if not reference_notes or not performance_notes:
            return
        
        ref_velocities = [note['velocity'] for note in reference_notes]
        perf_velocities = [note['velocity'] for note in performance_notes]
        
        # Calculate dynamic metrics
        dynamic_range = max(perf_velocities) - min(perf_velocities) if perf_velocities else 0
        ref_dynamic_range = max(ref_velocities) - min(ref_velocities) if ref_velocities else 0
        dynamic_variety = len(set(perf_velocities)) / len(perf_velocities) if perf_velocities else 0
        
        # Analyze crescendo/decrescendo patterns
        dynamic_patterns = self._analyze_dynamic_patterns(perf_velocities)
        
        # Compare with reference dynamics on aligned note pairs only.
        dynamic_deviation = None
        if aligned_pairs:
            diffs = [
                abs(int(p['performance_note'].get('velocity', 0)) - int(p['reference_note'].get('velocity', 0)))
                for p in aligned_pairs
            ]
            if diffs:
                dynamic_deviation = sum(diffs) / len(diffs)
        
        self.metrics['dynamic_control'] = {
            'dynamic_range': int(dynamic_range),
            'reference_dynamic_range': int(ref_dynamic_range),
            'dynamic_variety': round(dynamic_variety, 2),
            'average_velocity': round(statistics.mean(perf_velocities), 1) if perf_velocities else 0,
            'velocity_std': round(statistics.stdev(perf_velocities), 1) if len(perf_velocities) > 1 else 0,
            'dynamic_deviation': round(dynamic_deviation, 1) if dynamic_deviation is not None else None,
            'dynamic_deviation_source': 'aligned_pairs' if dynamic_deviation is not None else 'unavailable',
            'dynamic_patterns': dynamic_patterns,
            'expression_level': self._assess_expression_level(perf_velocities)
        }
    
    def _analyze_articulation(self):
        """Analyze articulation: staccato, legato, note durations."""
        performance_notes = self.performance_data.get('notes', [])
        
        if not performance_notes:
            return
        
        perf_durations = [note['duration'] for note in performance_notes]
        
        # Calculate articulation ratios per track to avoid cross-voice IOI mixing.
        by_track = defaultdict(list)
        for note in performance_notes:
            by_track[note.get('track_id', 0)].append(note)

        articulation_ratios = []
        for track_notes in by_track.values():
            track_notes.sort(key=lambda n: n['start'])
            note_intervals = self._calculate_note_intervals(track_notes)
            for i in range(len(track_notes) - 1):
                duration = track_notes[i]['duration']
                interval = note_intervals[i] if i < len(note_intervals) else 0
                if interval > 1e-6:
                    articulation_ratios.append(duration / interval)
        
        # Detect articulation patterns
        staccato_notes = [ratio for ratio in articulation_ratios if ratio < 0.5]
        legato_notes = [ratio for ratio in articulation_ratios if ratio > 0.9]
        normal_notes = [ratio for ratio in articulation_ratios if 0.5 <= ratio <= 0.9]
        
        articulation_consistency = self._calculate_consistency_score(articulation_ratios)
        
        self.metrics['articulation'] = {
            'average_duration': round(statistics.mean(perf_durations), 3) if perf_durations else 0,
            'articulation_consistency': round(articulation_consistency, 2),
            'staccato_percentage': round((len(staccato_notes) / len(articulation_ratios)) * 100, 1) if articulation_ratios else 0,
            'legato_percentage': round((len(legato_notes) / len(articulation_ratios)) * 100, 1) if articulation_ratios else 0,
            'normal_percentage': round((len(normal_notes) / len(articulation_ratios)) * 100, 1) if articulation_ratios else 0,
            'articulation_variety': self._assess_articulation_variety(articulation_ratios)
        }
    
    def _analyze_phrasing_errors(self):
        """Analyze musical phrasing errors."""
        # This assumes phrase segmentation data is available
        # For now, analyze based on timing patterns
        
        aligned_pairs = [p for p in self.aligned_notes 
                        if p.get('reference_note') and p.get('performance_note')]
        
        if len(aligned_pairs) < 10:  # Need enough data for phrasing analysis
            return
        
        # Detect phrase boundaries based on longer pauses
        time_differences = [pair.get('time_difference', 0) for pair in aligned_pairs]
        
        # Simple phrase boundary detection
        phrase_boundaries = []
        for i in range(1, len(aligned_pairs)):
            if aligned_pairs[i].get('reference_note', {}).get('start', 0) - \
               aligned_pairs[i-1].get('reference_note', {}).get('end', 0) > 1.0:  # 1-second gap
                phrase_boundaries.append(i)
        
        # Analyze consistency within phrases
        phrase_consistency = []
        start_idx = 0
        for boundary in phrase_boundaries:
            phrase_errors = time_differences[start_idx:boundary]
            if phrase_errors:
                phrase_consistency.append(statistics.stdev(phrase_errors) if len(phrase_errors) > 1 else 0)
            start_idx = boundary
        
        self.metrics['phrasing'] = {
            'detected_phrases': len(phrase_boundaries) + 1,
            'average_phrase_length': len(aligned_pairs) / (len(phrase_boundaries) + 1) if phrase_boundaries else len(aligned_pairs),
            'phrase_consistency': round(statistics.mean(phrase_consistency), 3) if phrase_consistency else 0,
            'phrasing_regularity': self._assess_phrasing_regularity(phrase_boundaries, len(aligned_pairs))
        }
    
    # Add this method to the ErrorAnalysis class in error_analysis.py
# You can add it anywhere in the class, perhaps after the _analyze_phrasing_errors method

    def _analyze_tempo_stability(self) -> Dict[str, Any]:
        """Analyze tempo stability throughout the performance."""
        performance_notes = self.performance_data.get('notes', [])
        
        if len(performance_notes) < 10:  # Need enough notes for tempo analysis
            return {
                'stability_score': 0.5,
                'tempo_variation': 'Insufficient data',
                'rubato_patterns': []
            }
        
        # Calculate inter-onset intervals (IOI)
        iois = []
        for i in range(1, len(performance_notes)):
            ioi = performance_notes[i]['start'] - performance_notes[i-1]['start']
            iois.append(ioi)
        
        # Calculate local tempo variations
        if len(iois) >= 5:
            # Use moving window to detect tempo changes
            tempo_variations = []
            window_size = 5
            
            for i in range(len(iois) - window_size + 1):
                window = iois[i:i+window_size]
                avg_ioi = sum(window) / window_size
                # Convert IOI to BPM (60 seconds / IOI in seconds)
                if avg_ioi > 0:
                    bpm = 60 / avg_ioi
                    tempo_variations.append(bpm)
            
            # Calculate tempo stability
            if tempo_variations:
                mean_tempo = statistics.mean(tempo_variations)
                if mean_tempo > 0:
                    cv = statistics.stdev(tempo_variations) / mean_tempo if len(tempo_variations) > 1 else 0
                    stability_score = 1 / (1 + cv)  # Convert to 0-1 scale
                    
                    # Detect rubato patterns (intentional tempo variations)
                    rubato_patterns = self._detect_rubato_patterns(tempo_variations)
                    
                    return {
                        'stability_score': round(stability_score, 2),
                        'tempo_variation': round(cv * 100, 1),  # as percentage
                        'average_tempo': round(mean_tempo, 1),
                        'tempo_range': {
                            'min': round(min(tempo_variations), 1),
                            'max': round(max(tempo_variations), 1)
                        },
                        'rubato_patterns': rubato_patterns
                    }
        
        return {
            'stability_score': 0.5,
            'tempo_variation': 'Normal',
            'rubato_patterns': []
        }

    def _detect_rubato_patterns(self, tempo_variations: List[float]) -> List[Dict]:
        """Detect intentional tempo variations (rubato)."""
        patterns = []
        
        if len(tempo_variations) < 10:
            return patterns
        
        # Look for patterns of slowing down and speeding up
        for i in range(len(tempo_variations) - 4):
            segment = tempo_variations[i:i+4]
            # Check if pattern goes down then up (rubato)
            if segment[0] > segment[1] and segment[1] < segment[2] and segment[2] < segment[3]:
                patterns.append({
                    'type': 'rubato_slow_then_fast',
                    'start_index': i,
                    'intensity': abs(segment[0] - segment[3]) / segment[0]
                })
            # Check if pattern goes up then down (reverse rubato)
            elif segment[0] < segment[1] and segment[1] > segment[2] and segment[2] > segment[3]:
                patterns.append({
                    'type': 'rubato_fast_then_slow',
                    'start_index': i,
                    'intensity': abs(segment[0] - segment[3]) / segment[0]
                })
        
        return patterns[:5]  # Return top 5 patterns

    def _analyze_pedaling_errors(self):
        """Analyze sustain pedal usage as a dedicated stream."""
        categories = self._empty_pedaling_categories()
        self.error_categories['pedaling'] = categories

        performance_segments = self._get_pedal_segments(self.performance_data)
        performance_raw_events = self.performance_data.get('pedaling', {}).get('raw_events', [])
        performance_summary = self.performance_data.get('pedaling', {}).get('summary', {})

        if self.mode == 'reference_comparison':
            reference_segments = self._get_pedal_segments(self.reference_data)
            reference_raw_events = self.reference_data.get('pedaling', {}).get('raw_events', [])
            reference_summary = self.reference_data.get('pedaling', {}).get('summary', {})
            self._analyze_compare_pedaling(
                reference_segments=reference_segments,
                performance_segments=performance_segments,
                reference_raw_events=reference_raw_events,
                performance_raw_events=performance_raw_events,
                reference_summary=reference_summary,
                performance_summary=performance_summary,
                categories=categories,
            )
            return

        self._analyze_solo_pedaling(
            performance_segments=performance_segments,
            performance_raw_events=performance_raw_events,
            performance_summary=performance_summary,
            categories=categories,
        )

    def _analyze_compare_pedaling(
        self,
        reference_segments: List[Dict[str, Any]],
        performance_segments: List[Dict[str, Any]],
        reference_raw_events: List[Dict[str, Any]],
        performance_raw_events: List[Dict[str, Any]],
        reference_summary: Dict[str, Any],
        performance_summary: Dict[str, Any],
        categories: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        if not reference_raw_events and not performance_raw_events:
            self.metrics['pedaling'] = {
                'available': False,
                'pedal_analysis_available': False,
                'mode': 'reference_comparison',
                'included_in_overall_score': False,
                'reference_segment_count': 0,
                'performance_segment_count': 0,
                'note': 'No CC64 sustain-pedal data was found in either MIDI.',
            }
            return

        ref_times, perf_times, mapping_source = self._build_reference_to_performance_map()
        mapped_reference_segments = [
            self._map_segment_to_performance_time(seg, ref_times, perf_times)
            for seg in reference_segments
        ]

        pairing = self._pair_pedal_segments(mapped_reference_segments, performance_segments)
        matched_records = pairing['matched_records']
        clean_matches = [m for m in matched_records if m.get('clean_match')]

        onset_errors = [m['onset_error_ms'] for m in matched_records]
        release_errors = [m['release_error_ms'] for m in matched_records]
        overlap_ratios = [m['overlap_ratio'] for m in matched_records]

        categories['missed'] = [mapped_reference_segments[idx] for idx in pairing['missed_reference_indices']]
        categories['extra'] = [performance_segments[idx] for idx in pairing['extra_performance_indices']]
        categories['split'] = [
            {
                'reference_segment': mapped_reference_segments[idx],
                'performance_segments': [performance_segments[p_idx] for p_idx in pairing['ref_to_perf'][idx]],
            }
            for idx in pairing['split_reference_indices']
        ]
        categories['merged'] = [
            {
                'performance_segment': performance_segments[idx],
                'reference_segments': [mapped_reference_segments[r_idx] for r_idx in pairing['perf_to_ref'][idx]],
            }
            for idx in pairing['merged_performance_indices']
        ]

        onset_threshold_ms = 150.0
        release_threshold_ms = 180.0
        categories['late_onset'] = [m for m in matched_records if m['onset_error_ms'] > onset_threshold_ms]
        categories['early_onset'] = [m for m in matched_records if m['onset_error_ms'] < -onset_threshold_ms]
        categories['late_release'] = [m for m in matched_records if m['release_error_ms'] > release_threshold_ms]
        categories['early_release'] = [m for m in matched_records if m['release_error_ms'] < -release_threshold_ms]

        interaction_issues = self._collect_compare_pedal_interaction_issues(
            mapped_reference_segments=mapped_reference_segments,
            performance_segments=performance_segments,
            matched_records=matched_records,
            ref_times=ref_times,
            perf_times=perf_times,
        )
        categories['harmonic_blur'] = interaction_issues['harmonic_blur']
        categories['phrase_boundary_clearance'] = interaction_issues['phrase_boundary_clearance']
        categories['early_release_while_notes_ring'] = interaction_issues['early_release_while_notes_ring']

        performance_duration = (
            self.performance_data.get('total_duration')
            or self.performance_data.get('metadata', {}).get('total_duration')
            or 0.0
        )
        mapped_reference_pedal_time = sum(float(seg.get('duration', 0.0)) for seg in mapped_reference_segments)
        performance_pedal_time = sum(float(seg.get('duration', 0.0)) for seg in performance_segments)

        self.metrics['pedaling'] = {
            'available': True,
            'pedal_analysis_available': True,
            'mode': 'reference_comparison',
            'included_in_overall_score': False,
            'mapping_source': mapping_source,
            'reference_segment_count': int(len(reference_segments)),
            'performance_segment_count': int(len(performance_segments)),
            'matched_segment_count': int(len(matched_records)),
            'clean_match_count': int(len(clean_matches)),
            'missed_pedals': int(len(categories['missed'])),
            'extra_pedals': int(len(categories['extra'])),
            'split_reference_spans': int(len(categories['split'])),
            'merged_performance_spans': int(len(categories['merged'])),
            'mean_onset_error_ms': round(float(statistics.mean(onset_errors)), 1) if onset_errors else None,
            'mean_release_error_ms': round(float(statistics.mean(release_errors)), 1) if release_errors else None,
            'mean_overlap_ratio': round(float(statistics.mean(overlap_ratios)), 3) if overlap_ratios else 0.0,
            'median_overlap_ratio': round(float(statistics.median(overlap_ratios)), 3) if overlap_ratios else 0.0,
            'late_onset_count': int(len(categories['late_onset'])),
            'early_onset_count': int(len(categories['early_onset'])),
            'late_release_count': int(len(categories['late_release'])),
            'early_release_count': int(len(categories['early_release'])),
            'harmonic_blur_count': int(len(categories['harmonic_blur'])),
            'phrase_boundary_clearance_issues': int(len(categories['phrase_boundary_clearance'])),
            'early_release_while_notes_ring_count': int(len(categories['early_release_while_notes_ring'])),
            'reference_pedal_time': round(float(mapped_reference_pedal_time), 3),
            'performance_pedal_time': round(float(performance_pedal_time), 3),
            'reference_pedal_coverage_ratio': (
                round(float(mapped_reference_pedal_time / performance_duration), 4)
                if performance_duration > 1e-9 else float(reference_summary.get('pedal_coverage_ratio', 0.0) or 0.0)
            ),
            'performance_pedal_coverage_ratio': (
                round(float(performance_pedal_time / performance_duration), 4)
                if performance_duration > 1e-9 else float(performance_summary.get('pedal_coverage_ratio', 0.0) or 0.0)
            ),
            'reference_raw_event_count': int(len(reference_raw_events)),
            'performance_raw_event_count': int(len(performance_raw_events)),
        }

    def _analyze_solo_pedaling(
        self,
        performance_segments: List[Dict[str, Any]],
        performance_raw_events: List[Dict[str, Any]],
        performance_summary: Dict[str, Any],
        categories: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        if not performance_raw_events:
            self.metrics['pedaling'] = {
                'available': False,
                'pedal_analysis_available': False,
                'mode': 'solo',
                'included_in_overall_score': False,
                'pedal_segment_count': 0,
                'note': 'No CC64 sustain-pedal data was found in the performance MIDI.',
            }
            return

        total_duration = (
            self.performance_data.get('total_duration')
            or self.performance_data.get('metadata', {}).get('total_duration')
            or 0.0
        )
        durations = [float(seg.get('duration', 0.0)) for seg in performance_segments]
        mean_duration = statistics.mean(durations) if durations else 0.0
        duration_cv = (
            (statistics.stdev(durations) / mean_duration)
            if len(durations) > 1 and mean_duration > 1e-9 else 0.0
        )
        longest_hold = max(durations) if durations else 0.0
        long_hold_threshold = max(4.0, (statistics.median(durations) * 1.75) if durations else 4.0)

        categories['long_holds'] = [
            seg for seg in performance_segments
            if float(seg.get('duration', 0.0)) >= long_hold_threshold
        ]
        phrase_data = self._collect_solo_phrase_release_data(performance_segments)
        categories['phrase_boundary_releases'] = phrase_data['near_releases']
        categories['phrase_boundary_misses'] = phrase_data['misses']
        categories['excessive_holds'] = categories['long_holds'] if (
            float(performance_summary.get('pedal_coverage_ratio', 0.0) or 0.0) > 0.75 or longest_hold >= 6.0
        ) else []

        pedals_per_minute = (
            len(performance_segments) / max(total_duration / 60.0, 1e-9)
            if total_duration > 1e-9 else 0.0
        )
        coverage_ratio = float(performance_summary.get('pedal_coverage_ratio', 0.0) or 0.0)

        if coverage_ratio > 0.75 or longest_hold >= 6.0:
            stability = 'excessive'
        elif coverage_ratio < 0.08 and len(performance_segments) <= 1:
            stability = 'light'
        elif duration_cv <= 0.45:
            stability = 'stable'
        else:
            stability = 'variable'

        phrase_ratio = phrase_data['matched_ratio']

        self.metrics['pedaling'] = {
            'available': True,
            'pedal_analysis_available': True,
            'mode': 'solo',
            'included_in_overall_score': False,
            'pedal_segment_count': int(len(performance_segments)),
            'raw_event_count': int(len(performance_raw_events)),
            'pedals_per_minute': round(float(pedals_per_minute), 2),
            'average_hold_duration': round(float(mean_duration), 3) if durations else 0.0,
            'median_hold_duration': round(float(statistics.median(durations)), 3) if durations else 0.0,
            'longest_hold_duration': round(float(longest_hold), 3) if durations else 0.0,
            'long_hold_threshold_s': round(float(long_hold_threshold), 3),
            'long_hold_count': int(len(categories['long_holds'])),
            'pedal_coverage_ratio': round(float(coverage_ratio), 4),
            'duration_cv': round(float(duration_cv), 3),
            'stability': stability,
            'excessive_pedaling': bool(stability == 'excessive'),
            'phrase_end_release_count': int(len(categories['phrase_boundary_releases'])),
            'phrase_end_release_ratio': round(float(phrase_ratio), 3) if phrase_ratio is not None else None,
        }

    def _empty_pedaling_categories(self) -> Dict[str, List[Dict[str, Any]]]:
        return {
            'missed': [],
            'extra': [],
            'split': [],
            'merged': [],
            'late_onset': [],
            'early_onset': [],
            'late_release': [],
            'early_release': [],
            'harmonic_blur': [],
            'phrase_boundary_clearance': [],
            'early_release_while_notes_ring': [],
            'long_holds': [],
            'phrase_boundary_releases': [],
            'phrase_boundary_misses': [],
            'excessive_holds': [],
        }

    def _get_pedal_segments(self, parsed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        pedaling = parsed_data.get('pedaling', {}) if isinstance(parsed_data, dict) else {}
        segments = pedaling.get('segments', []) if isinstance(pedaling, dict) else []

        normalized = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = float(seg.get('start', 0.0))
            end = float(seg.get('end', start))
            if end < start:
                end = start
            normalized.append(
                {
                    **seg,
                    'start': start,
                    'end': end,
                    'duration': float(seg.get('duration', max(0.0, end - start))),
                }
            )
        normalized.sort(key=lambda x: (float(x.get('start', 0.0)), float(x.get('end', 0.0))))
        return normalized

    def _build_reference_to_performance_map(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        aligned_pairs = [
            p for p in self.aligned_notes
            if p.get('reference_note') and p.get('performance_note')
        ]
        if len(aligned_pairs) < 2:
            return None, None, 'identity_fallback'

        points: List[Tuple[float, float]] = []
        for pair in aligned_pairs:
            ref = pair['reference_note']
            perf = pair['performance_note']
            points.append((float(ref.get('start', 0.0)), float(perf.get('start', 0.0))))
            if ref.get('end') is not None and perf.get('end') is not None:
                points.append((float(ref.get('end', 0.0)), float(perf.get('end', 0.0))))

        points.sort(key=lambda x: x[0])
        ref_points: List[float] = []
        perf_points: List[float] = []
        current_ref = None
        bucket: List[float] = []

        for ref_t, perf_t in points:
            if current_ref is None or abs(ref_t - current_ref) <= 1e-6:
                current_ref = ref_t if current_ref is None else current_ref
                bucket.append(perf_t)
                continue
            ref_points.append(float(current_ref))
            perf_points.append(float(statistics.mean(bucket)))
            current_ref = ref_t
            bucket = [perf_t]

        if current_ref is not None and bucket:
            ref_points.append(float(current_ref))
            perf_points.append(float(statistics.mean(bucket)))

        if len(ref_points) < 2:
            return None, None, 'identity_fallback'

        return np.asarray(ref_points, dtype=float), np.asarray(perf_points, dtype=float), 'aligned_notes'

    def _map_reference_time(
        self,
        time_value: float,
        ref_times: Optional[np.ndarray],
        perf_times: Optional[np.ndarray],
    ) -> float:
        t = float(time_value)
        if ref_times is None or perf_times is None or len(ref_times) < 2 or len(perf_times) < 2:
            return t

        if t <= ref_times[0]:
            dt = ref_times[1] - ref_times[0]
            slope = (perf_times[1] - perf_times[0]) / dt if abs(dt) > 1e-9 else 1.0
            return float(perf_times[0] + (t - ref_times[0]) * slope)

        if t >= ref_times[-1]:
            dt = ref_times[-1] - ref_times[-2]
            slope = (perf_times[-1] - perf_times[-2]) / dt if abs(dt) > 1e-9 else 1.0
            return float(perf_times[-1] + (t - ref_times[-1]) * slope)

        return float(np.interp(t, ref_times, perf_times))

    def _map_segment_to_performance_time(
        self,
        segment: Dict[str, Any],
        ref_times: Optional[np.ndarray],
        perf_times: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        mapped_start = self._map_reference_time(float(segment.get('start', 0.0)), ref_times, perf_times)
        mapped_end = self._map_reference_time(float(segment.get('end', mapped_start)), ref_times, perf_times)
        if mapped_end < mapped_start:
            mapped_end = mapped_start

        return {
            **segment,
            'reference_start': float(segment.get('start', 0.0)),
            'reference_end': float(segment.get('end', mapped_end)),
            'start': float(mapped_start),
            'end': float(mapped_end),
            'duration': float(max(0.0, mapped_end - mapped_start)),
            'mapped_from_reference': True,
        }

    def _pair_pedal_segments(
        self,
        reference_segments: List[Dict[str, Any]],
        performance_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        ref_to_perf = defaultdict(list)
        perf_to_ref = defaultdict(list)
        candidate_pairs = []

        for ref_idx, ref_seg in enumerate(reference_segments):
            ref_dur = max(1e-6, float(ref_seg.get('duration', 0.0)))
            for perf_idx, perf_seg in enumerate(performance_segments):
                perf_dur = max(1e-6, float(perf_seg.get('duration', 0.0)))
                overlap = self._segment_overlap(ref_seg, perf_seg)
                onset_diff = abs(float(perf_seg.get('start', 0.0)) - float(ref_seg.get('start', 0.0)))
                release_diff = abs(float(perf_seg.get('end', 0.0)) - float(ref_seg.get('end', 0.0)))
                tolerance = max(0.18, min(0.8, 0.45 * max(ref_dur, perf_dur)))

                if overlap <= 0 and onset_diff > tolerance and release_diff > tolerance:
                    continue

                union = max(
                    1e-6,
                    max(float(ref_seg.get('end', 0.0)), float(perf_seg.get('end', 0.0)))
                    - min(float(ref_seg.get('start', 0.0)), float(perf_seg.get('start', 0.0))),
                )
                overlap_ratio = overlap / union
                score = (0.75 * overlap_ratio) + (0.25 / (1.0 + onset_diff + release_diff))
                candidate_pairs.append((score, ref_idx, perf_idx, overlap_ratio))
                ref_to_perf[ref_idx].append(perf_idx)
                perf_to_ref[perf_idx].append(ref_idx)

        matched_records = []
        used_refs = set()
        used_perfs = set()

        for score, ref_idx, perf_idx, overlap_ratio in sorted(candidate_pairs, key=lambda x: x[0], reverse=True):
            if ref_idx in used_refs or perf_idx in used_perfs:
                continue
            used_refs.add(ref_idx)
            used_perfs.add(perf_idx)
            ref_seg = reference_segments[ref_idx]
            perf_seg = performance_segments[perf_idx]
            matched_records.append(
                {
                    'reference_segment': ref_seg,
                    'performance_segment': perf_seg,
                    'match_score': round(float(score), 4),
                    'overlap_ratio': round(float(overlap_ratio), 4),
                    'onset_error_ms': round((float(perf_seg.get('start', 0.0)) - float(ref_seg.get('start', 0.0))) * 1000.0, 1),
                    'release_error_ms': round((float(perf_seg.get('end', 0.0)) - float(ref_seg.get('end', 0.0))) * 1000.0, 1),
                    'clean_match': len(ref_to_perf[ref_idx]) == 1 and len(perf_to_ref[perf_idx]) == 1,
                }
            )

        return {
            'matched_records': matched_records,
            'ref_to_perf': ref_to_perf,
            'perf_to_ref': perf_to_ref,
            'missed_reference_indices': [idx for idx in range(len(reference_segments)) if not ref_to_perf[idx]],
            'extra_performance_indices': [idx for idx in range(len(performance_segments)) if not perf_to_ref[idx]],
            'split_reference_indices': [idx for idx, matches in ref_to_perf.items() if len(matches) > 1],
            'merged_performance_indices': [idx for idx, matches in perf_to_ref.items() if len(matches) > 1],
        }

    def _collect_compare_pedal_interaction_issues(
        self,
        mapped_reference_segments: List[Dict[str, Any]],
        performance_segments: List[Dict[str, Any]],
        matched_records: List[Dict[str, Any]],
        ref_times: Optional[np.ndarray],
        perf_times: Optional[np.ndarray],
    ) -> Dict[str, List[Dict[str, Any]]]:
        mapped_chords = sorted(
            self._map_reference_time(float(ch.get('start_time', 0.0)), ref_times, perf_times)
            for ch in self.reference_data.get('harmony', {}).get('chords', [])
            if isinstance(ch, dict)
        )
        phrase_boundaries = sorted(
            self._map_reference_time(float(ph.get('end_time', 0.0)), ref_times, perf_times)
            for ph in self.reference_data.get('structure', {}).get('phrases', [])[:-1]
            if isinstance(ph, dict) and ph.get('end_time') is not None
        )
        mapped_reference_notes = [
            {
                'start': self._map_reference_time(float(note.get('start', 0.0)), ref_times, perf_times),
                'end': self._map_reference_time(float(note.get('end', note.get('start', 0.0))), ref_times, perf_times),
                'pitch': int(note.get('pitch', 0)),
            }
            for note in self.reference_data.get('notes', [])
            if isinstance(note, dict)
        ]

        harmonic_blur = []
        early_release_while_notes_ring = []

        for match in matched_records:
            ref_seg = match['reference_segment']
            perf_seg = match['performance_segment']

            if match['release_error_ms'] > 180.0:
                crossed_changes = [
                    t for t in mapped_chords
                    if float(ref_seg.get('end', 0.0)) + 0.05 <= t <= float(perf_seg.get('end', 0.0)) - 0.05
                ]
                if crossed_changes:
                    harmonic_blur.append(
                        {
                            **match,
                            'crossed_harmonic_change_count': int(len(crossed_changes)),
                            'harmonic_change_times': [round(float(t), 3) for t in crossed_changes[:5]],
                        }
                    )

            if match['release_error_ms'] < -180.0:
                lingering_notes = [
                    note for note in mapped_reference_notes
                    if float(ref_seg.get('start', 0.0)) - 0.05 <= note['start'] <= float(ref_seg.get('end', 0.0)) + 0.05
                    and note['end'] >= float(perf_seg.get('end', 0.0)) + 0.12
                ]
                if lingering_notes:
                    early_release_while_notes_ring.append(
                        {
                            **match,
                            'remaining_note_count': int(len(lingering_notes)),
                            'latest_expected_ring_end': round(float(max(n['end'] for n in lingering_notes)), 3),
                        }
                    )

        phrase_boundary_clearance = []
        for boundary in phrase_boundaries:
            ref_active = self._find_active_segment(mapped_reference_segments, boundary)
            perf_active = self._find_active_segment(performance_segments, boundary)
            if ref_active is None or perf_active is None:
                continue
            if float(ref_active.get('end', 0.0)) <= boundary + 0.18 and float(perf_active.get('end', 0.0)) > boundary + 0.22:
                phrase_boundary_clearance.append(
                    {
                        'boundary_time': round(float(boundary), 3),
                        'reference_release_time': round(float(ref_active.get('end', 0.0)), 3),
                        'performance_release_time': round(float(perf_active.get('end', 0.0)), 3),
                        'overhang_ms': round((float(perf_active.get('end', 0.0)) - float(boundary)) * 1000.0, 1),
                    }
                )

        return {
            'harmonic_blur': harmonic_blur,
            'phrase_boundary_clearance': phrase_boundary_clearance,
            'early_release_while_notes_ring': early_release_while_notes_ring,
        }

    def _collect_solo_phrase_release_data(
        self,
        performance_segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        phrases = self.performance_data.get('structure', {}).get('phrases', [])
        if not phrases:
            return {'near_releases': [], 'misses': [], 'matched_ratio': None}

        near_releases = []
        misses = []
        release_times = [float(seg.get('end', 0.0)) for seg in performance_segments]

        for phrase in phrases[:-1]:
            boundary = float(phrase.get('end_time', 0.0))
            nearest = None
            nearest_dist = None
            for seg, release_time in zip(performance_segments, release_times):
                dist = abs(release_time - boundary)
                if nearest is None or dist < nearest_dist:
                    nearest = seg
                    nearest_dist = dist
            if nearest is not None and nearest_dist is not None and nearest_dist <= 0.35:
                near_releases.append(
                    {
                        'phrase_end_time': round(float(boundary), 3),
                        'release_time': round(float(nearest.get('end', 0.0)), 3),
                        'distance_ms': round(float(nearest_dist * 1000.0), 1),
                    }
                )
            else:
                misses.append({'phrase_end_time': round(float(boundary), 3)})

        total_boundaries = len(phrases[:-1])
        matched_ratio = (len(near_releases) / total_boundaries) if total_boundaries > 0 else None
        return {
            'near_releases': near_releases,
            'misses': misses,
            'matched_ratio': matched_ratio,
        }

    def _find_active_segment(
        self,
        segments: List[Dict[str, Any]],
        time_value: float,
    ) -> Optional[Dict[str, Any]]:
        t = float(time_value)
        for seg in segments:
            if float(seg.get('start', 0.0)) <= t <= float(seg.get('end', 0.0)):
                return seg
        return None

    def _segment_overlap(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
    ) -> float:
        return max(
            0.0,
            min(float(a.get('end', 0.0)), float(b.get('end', 0.0)))
            - max(float(a.get('start', 0.0)), float(b.get('start', 0.0))),
        )
    
    def _calculate_performance_score(self):
        """Calculate an overall performance score based on all metrics."""
        weights = {
            'note_accuracy': 0.30,
            'timing_errors': 0.25,
            'rhythmic_consistency': 0.15,
            'dynamic_control': 0.15,
            'articulation': 0.10,
            'phrasing': 0.05
        }
        
        component_scores = {}
        total_score = 0
        total_weight = 0
        
        # Calculate component scores
        if 'note_accuracy' in self.metrics:
            accuracy = self.metrics['note_accuracy'].get('accuracy_percentage', 0)
            component_scores['note_accuracy'] = accuracy / 100  # Convert to 0-1 scale
            total_score += (accuracy / 100) * weights['note_accuracy']
            total_weight += weights['note_accuracy']
        
        if 'timing_errors' in self.metrics and self.metrics['timing_errors'].get('available', True):
            timing_score = self._calculate_timing_score()
            component_scores['timing'] = timing_score
            total_score += timing_score * weights['timing_errors']
            total_weight += weights['timing_errors']
        
        if 'rhythmic_consistency' in self.metrics:
            rhythm_score = self.metrics['rhythmic_consistency'].get('duration_consistency_score', 0.5)
            component_scores['rhythm'] = rhythm_score
            total_score += rhythm_score * weights['rhythmic_consistency']
            total_weight += weights['rhythmic_consistency']
        
        if 'dynamic_control' in self.metrics:
            dynamic_score = self._calculate_dynamic_score()
            component_scores['dynamics'] = dynamic_score
            total_score += dynamic_score * weights['dynamic_control']
            total_weight += weights['dynamic_control']

        if 'articulation' in self.metrics:
            import math

            # Prefer reference-vs-performance articulation match on aligned note pairs.
            aligned_pairs = [
                p for p in self.aligned_notes
                if p.get('reference_note') and p.get('performance_note')
            ]

            if aligned_pairs:
                ref_durs = [p['reference_note']['duration'] for p in aligned_pairs]
                perf_durs = [p['performance_note']['duration'] for p in aligned_pairs]
                log_devs = []
                for ref_duration, perf_duration in zip(ref_durs, perf_durs):
                    if ref_duration > 1e-6 and perf_duration > 1e-6:
                        log_devs.append(abs(math.log(perf_duration / ref_duration)))

                if log_devs:
                    mean_dev = statistics.mean(log_devs)
                    articulation_score = 1.0 / (1.0 + 6.0 * mean_dev)
                else:
                    articulation_score = 1.0
            else:
                # Fallback for solo mode: use internal articulation consistency.
                articulation_score = float(
                    self.metrics['articulation'].get('articulation_consistency', 0.5)
                )

            articulation_score = max(0.0, min(1.0, articulation_score))
            component_scores['articulation'] = articulation_score
            total_score += articulation_score * weights['articulation']
            total_weight += weights['articulation']

        if 'phrasing' in self.metrics:
            phrasing_data = self.metrics.get('phrasing', {})
            phr_cons = phrasing_data.get('phrase_consistency', None)
            if isinstance(phr_cons, (int, float)):
                phrasing_score = 1.0 / (1.0 + abs(float(phr_cons)) * 10.0)
            else:
                phrasing_score = 0.5
            phrasing_score = max(0.0, min(1.0, phrasing_score))
            component_scores['phrasing'] = phrasing_score
            total_score += phrasing_score * weights['phrasing']
            total_weight += weights['phrasing']
        
        # Normalize score
        if total_weight > 0:
            overall_score = total_score / total_weight
        else:
            overall_score = 0
        
        # Grade performance
        grade = self._assign_grade(overall_score)
        
        self.metrics['performance_score'] = {
            'overall_score': round(overall_score * 100, 1),  # Convert to percentage
            'component_scores': component_scores,
            'grade': grade,
            'weights_used': weights
        }
    
    def _generate_practice_recommendations(self):
        """Generate specific practice recommendations based on analysis."""
        recommendations = []
        
        # Note accuracy recommendations
        if 'note_accuracy' in self.metrics:
            accuracy = self.metrics['note_accuracy'].get('accuracy_percentage', 100)
            missing = self.metrics['note_accuracy'].get('missing_percentage', 0)
            
            if accuracy < 90:
                recommendations.append("Focus on note accuracy: practice slowly with attention to correct pitches")
            if missing > 10:
                recommendations.append(f"Missing {missing:.1f}% of notes: isolate difficult passages")
        
        # Timing recommendations
        if 'timing_errors' in self.metrics:
            rushing = self.metrics['timing_errors'].get('rushing_percentage', 0)
            dragging = self.metrics['timing_errors'].get('dragging_percentage', 0)
            
            if rushing > 20:
                recommendations.append("You tend to rush: practice with a metronome focusing on steady tempo")
            if dragging > 20:
                recommendations.append("You tend to drag: work on maintaining forward momentum in phrases")
        
        # Dynamic recommendations
        if 'dynamic_control' in self.metrics:
            dynamic_range = self.metrics['dynamic_control'].get('dynamic_range', 0)
            if 'dynamic_control' in self.metrics:
                dyn = self.metrics['dynamic_control']
                ref_range = dyn.get('reference_dynamic_range', None)
                perf_range = dyn.get('dynamic_range', 0)
                deviation = dyn.get('dynamic_deviation', None)

                # Only recommend more range if the reference actually has more range
                # OR if deviation shows they are not matching reference dynamics
                if ref_range is not None and ref_range > 20 and perf_range < ref_range * 0.6:
                    recommendations.append("Increase dynamic range to match the reference: practice crescendos and decrescendos")
                elif isinstance(deviation, (int, float)) and deviation > 12:
                    recommendations.append("Work on matching the reference dynamics more closely (control velocity changes)")

        # Rhythmic recommendations
        if 'rhythmic_consistency' in self.metrics:
            consistency = self.metrics['rhythmic_consistency'].get('duration_consistency_score', 0)
            if consistency < 0.7:
                recommendations.append("Work on rhythmic consistency: practice with subdivision counting")

        recommendations.extend(self._collect_pedaling_recommendations())

        deduped: List[str] = []
        seen = set()
        for rec in recommendations:
            key = rec.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(rec)

        self.practice_recommendations = deduped

    def _collect_pedaling_recommendations(self) -> List[str]:
        """Generate pedal-specific recommendations from the pedaling metrics."""
        pedaling = self.metrics.get('pedaling', {})
        if not isinstance(pedaling, dict) or not pedaling.get('pedal_analysis_available', False):
            return []

        recommendations: List[str] = []

        if pedaling.get('mode') == 'reference_comparison':
            if int(pedaling.get('missed_pedals', 0)) > 0:
                recommendations.append("Add the missing pedal changes from the reference so key harmonic arrivals stay supported")
            if int(pedaling.get('extra_pedals', 0)) > 0:
                recommendations.append("Remove extra pedal taps that are not present in the reference pedaling plan")
            if int(pedaling.get('split_reference_spans', 0)) > 0:
                recommendations.append("Avoid splitting a single reference pedal span into multiple lifts unless the harmony clearly changes")
            if int(pedaling.get('merged_performance_spans', 0)) > 0:
                recommendations.append("Clear the pedal between adjacent reference pedal spans instead of merging them into one long hold")
            if int(pedaling.get('late_release_count', 0)) > 0 or int(pedaling.get('phrase_boundary_clearance_issues', 0)) > 0:
                recommendations.append("Practice releasing the pedal a little earlier at phrase ends and harmonic changes")
            if int(pedaling.get('early_release_count', 0)) > 0 or int(pedaling.get('early_release_while_notes_ring_count', 0)) > 0:
                recommendations.append("Hold the pedal slightly longer where the reference lets tones continue ringing")
            if int(pedaling.get('harmonic_blur_count', 0)) > 0:
                recommendations.append("Listen for harmonic blur and clear the pedal before the next harmony arrives")
            return recommendations

        if bool(pedaling.get('excessive_pedaling', False)):
            recommendations.append("Use shorter pedal cycles and listen for cleaner resonance instead of one continuous wash")
        if int(pedaling.get('long_hold_count', 0)) > 0:
            recommendations.append("Break up very long pedal holds with planned refreshes so the texture stays clear")

        phrase_ratio = pedaling.get('phrase_end_release_ratio', None)
        if isinstance(phrase_ratio, (int, float)) and phrase_ratio < 0.45:
            recommendations.append("Try coordinating more pedal releases with phrase endings so musical sentences can breathe")

        stability = str(pedaling.get('stability', '')).strip().lower()
        if stability == 'variable':
            recommendations.append("Keep pedal depth and release timing more consistent from one gesture to the next")
        elif stability == 'light':
            recommendations.append("Experiment with a few more supportive pedal changes in sustained passages if the sound feels dry")

        return recommendations
    
    # Helper Methods
    
    def _calculate_note_intervals(self, notes: List[Dict]) -> List[float]:
        """Calculate time intervals between consecutive note starts."""
        if len(notes) < 2:
            return []
        
        intervals = []
        for i in range(1, len(notes)):
            interval = notes[i]['start'] - notes[i-1]['start']
            intervals.append(interval)
        
        return intervals
    
    def _calculate_consistency_score(self, values: List[float]) -> float:
        """Calculate consistency score (0-1) based on coefficient of variation."""
        if len(values) < 2:
            return 0.5  # Neutral score for insufficient data
        
        mean = statistics.mean(values)
        if mean == 0:
            return 0
        
        cv = statistics.stdev(values) / mean  # Coefficient of variation
        # Convert to consistency score (lower CV = higher consistency)
        consistency = 1 / (1 + cv)
        return max(0, min(1, consistency))  # Clamp to 0-1
    
    def _detect_rhythmic_patterns(self, time_differences: List[float]) -> List[Dict]:
        """Detect rhythmic error patterns."""
        patterns = []
        
        if len(time_differences) < 4:
            return patterns
        
        # Look for consistent early/late patterns
        for window_size in [2, 3, 4]:
            for i in range(len(time_differences) - window_size + 1):
                window = time_differences[i:i+window_size]
                if all(td < -0.02 for td in window):  # Consistently early
                    patterns.append({
                        'type': 'consistent_rushing',
                        'start_index': i,
                        'length': window_size,
                        'average_error': statistics.mean(window)
                    })
                elif all(td > 0.02 for td in window):  # Consistently late
                    patterns.append({
                        'type': 'consistent_dragging',
                        'start_index': i,
                        'length': window_size,
                        'average_error': statistics.mean(window)
                    })
        
        return patterns[:5]  # Return top 5 patterns
    
    def _analyze_dynamic_patterns(self, velocities: List[int]) -> List[Dict]:
        """Analyze dynamic patterns like crescendo and decrescendo."""
        patterns = []
        
        if len(velocities) < 3:
            return patterns
        
        # Look for crescendo patterns (increasing dynamics)
        for i in range(len(velocities) - 2):
            if velocities[i] < velocities[i+1] < velocities[i+2]:
                patterns.append({
                    'type': 'crescendo',
                    'start_index': i,
                    'length': 3,
                    'increase': velocities[i+2] - velocities[i]
                })
            elif velocities[i] > velocities[i+1] > velocities[i+2]:
                patterns.append({
                    'type': 'decrescendo',
                    'start_index': i,
                    'length': 3,
                    'decrease': velocities[i] - velocities[i+2]
                })
        
        return patterns
    
    def _assess_expression_level(self, velocities: List[int]) -> str:
        """Assess the level of dynamic expression."""
        if not velocities:
            return "Unknown"
        
        dynamic_range = max(velocities) - min(velocities)
        variety = len(set(velocities)) / len(velocities)
        
        if dynamic_range > 60 and variety > 0.4:
            return "Highly Expressive"
        elif dynamic_range > 40 and variety > 0.3:
            return "Expressive"
        elif dynamic_range > 20:
            return "Moderately Expressive"
        else:
            return "Limited Expression"
    
    def _assess_articulation_variety(self, articulation_ratios: List[float]) -> str:
        """Assess variety in articulation."""
        if not articulation_ratios:
            return "Unknown"
        
        staccato_count = sum(1 for r in articulation_ratios if r < 0.5)
        legato_count = sum(1 for r in articulation_ratios if r > 0.9)
        
        total = len(articulation_ratios)
        if total == 0:
            return "Unknown"
        
        if staccato_count > total * 0.3 and legato_count > total * 0.3:
            return "Varied Articulation"
        elif staccato_count > total * 0.5:
            return "Staccato Dominant"
        elif legato_count > total * 0.5:
            return "Legato Dominant"
        else:
            return "Mixed Articulation"
    
    def _assess_phrasing_regularity(self, phrase_boundaries: List[int], total_notes: int) -> str:
        """Assess regularity of phrasing."""
        if not phrase_boundaries:
            return "Single Phrase"
        
        phrase_lengths = []
        start_idx = 0
        for boundary in phrase_boundaries:
            phrase_lengths.append(boundary - start_idx)
            start_idx = boundary
        phrase_lengths.append(total_notes - start_idx)
        
        if len(phrase_lengths) < 2:
            return "Insufficient Phrases"
        
        cv = statistics.stdev(phrase_lengths) / statistics.mean(phrase_lengths)
        
        if cv < 0.2:
            return "Very Regular"
        elif cv < 0.4:
            return "Regular"
        elif cv < 0.6:
            return "Moderately Irregular"
        else:
            return "Irregular"
    
    def _calculate_timing_score(self) -> float:
        """Calculate timing score (0-1)."""
        timing_metrics = self.metrics.get('timing_errors', {})
        if not timing_metrics.get('available', True):
            return 0.5

        mean_raw = timing_metrics.get('mean_error_ms', 0)
        std_raw = timing_metrics.get('std_error_ms', 0)

        mean_error = abs(float(mean_raw)) / 1000 if isinstance(mean_raw, (int, float)) else 0.0
        std_error = float(std_raw) / 1000 if isinstance(std_raw, (int, float)) else 0.0
        
        # Lower errors = higher score
        timing_score = 1 / (1 + mean_error * 10 + std_error * 5)
        return max(0, min(1, timing_score))
    
    def _calculate_dynamic_score(self) -> float:
        """
        Calculate dynamic control score (0-1).
        IMPORTANT FIX (reference mode):
        If performance matches the reference dynamics, do NOT penalize low dynamic range.
        Grade based on similarity to reference velocities, and only use "expressiveness"
        as a fallback when no reference is available.
        """
        dynamic_metrics = self.metrics.get('dynamic_control', {})

        # If we have reference-comparison info, score by deviation from reference
        # dynamic_deviation is avg |perf_vel - ref_vel| across matched indices
        dynamic_deviation = dynamic_metrics.get('dynamic_deviation', None)

        # Heuristic: if dynamic_deviation exists, we are in reference comparison mode
        if dynamic_deviation is not None:
            # Convert deviation to a 0-1 similarity score
            # 0 difference -> 1.0 score
            # ~10 velocity difference -> still good, ~20+ starts to drop more
            # Tune denominator as needed
            similarity = 1.0 - min(dynamic_deviation / 25.0, 1.0)

            # OPTIONAL: also compare dynamic range similarity (helps when reference has crescendos)
            ref_range = dynamic_metrics.get('reference_dynamic_range', None)
            perf_range = dynamic_metrics.get('dynamic_range', 0)

            range_match = None
            if isinstance(ref_range, (int, float)) and ref_range is not None:
                if ref_range <= 1:
                    # Reference has basically no dynamic contrast; matching it should be perfect.
                    range_match = 1.0
                else:
                    # Compare ranges proportionally
                    range_match = 1.0 - min(abs(perf_range - ref_range) / ref_range, 1.0)

            # Combine: mostly similarity of velocities; range_match adds context if available
            if range_match is None:
                return max(0.0, min(1.0, similarity))
            return max(0.0, min(1.0, 0.8 * similarity + 0.2 * range_match))

        # ---- Fallback: SOLO-style scoring (no reference available) ----
        dynamic_range = dynamic_metrics.get('dynamic_range', 0)
        expression = dynamic_metrics.get('expression_level', 'Limited Expression')
        range_score = min(dynamic_range / 60, 1)  # 60+ is excellent
        expression_score = {
            'Highly Expressive': 1.0,
            'Expressive': 0.8,
            'Moderately Expressive': 0.6,
            'Limited Expression': 0.3,
            'Unknown': 0.5
        }.get(expression, 0.5)
        return (range_score * 0.4 + expression_score * 0.6)
    
    def _assign_grade(self, score: float) -> str:
        """Assign a letter grade based on performance score."""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B (Good)"
        elif score >= 0.6:
            return "C (Fair)"
        else:
            return "F (Needs Significant Practice)"
    
    def _get_detailed_error_list(self) -> List[Dict]:
        """Get a detailed list of individual errors."""
        errors = []
        
        # Timing errors
        timing_errors = self.error_categories.get('timing', {})
        for category, note_list in timing_errors.items():
            for note_pair in note_list[:10]:  # Limit to first 10 of each type
                errors.append({
                    'type': f'timing_{category}',
                    'time': note_pair.get('reference_note', {}).get('start', 0),
                    'error_value': note_pair.get('time_difference', 0),
                    'severity': 'high' if abs(note_pair.get('time_difference', 0)) > 0.1 else 'medium'
                })
        
        return errors
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get a concise performance summary."""
        score_data = self.metrics.get('performance_score', {})
        
        return {
            'overall_grade': score_data.get('grade', 'N/A'),
            'overall_score': score_data.get('overall_score', 0),
            'strengths': self._identify_strengths(),
            'weaknesses': self._identify_weaknesses(),
            'next_steps': self.practice_recommendations[:3]  # Top 3 recommendations
        }
    
    def _identify_strengths(self) -> List[str]:
        """Identify performance strengths."""
        strengths = []
        
        if 'note_accuracy' in self.metrics:
            accuracy = self.metrics['note_accuracy'].get('accuracy_percentage', 0)
            if accuracy >= 95:
                strengths.append("Excellent note accuracy")
        
        if 'timing_errors' in self.metrics:
            timing = self.metrics['timing_errors']
            rushing = timing.get('rushing_percentage', 0)
            dragging = timing.get('dragging_percentage', 0)
            if timing.get('available', True) and rushing < 10 and dragging < 10:
                strengths.append("Good timing control")
        
        if 'dynamic_control' in self.metrics:
            expression = self.metrics['dynamic_control'].get('expression_level', '')
            if 'Expressive' in expression:
                strengths.append("Good dynamic expression")

        pedaling = self.metrics.get('pedaling', {})
        if pedaling.get('pedal_analysis_available', False):
            if pedaling.get('mode') == 'reference_comparison':
                if (
                    int(pedaling.get('missed_pedals', 0)) == 0
                    and int(pedaling.get('extra_pedals', 0)) == 0
                    and float(pedaling.get('mean_overlap_ratio', 0.0)) >= 0.75
                ):
                    strengths.append("Pedal changes track the reference well")
            else:
                if str(pedaling.get('stability', '')).lower() == 'stable':
                    strengths.append("Pedaling is generally stable and supportive")
        
        return strengths if strengths else ["Solid foundation - keep practicing!"]
    
    def _identify_weaknesses(self) -> List[str]:
        """Identify performance weaknesses."""
        weaknesses = []
        
        if 'note_accuracy' in self.metrics:
            accuracy = self.metrics['note_accuracy'].get('accuracy_percentage', 0)
            if accuracy < 80:
                weaknesses.append("Note accuracy needs improvement")
        
        if 'timing_errors' in self.metrics:
            rushing = self.metrics['timing_errors'].get('rushing_percentage', 0)
            if rushing > 30:
                weaknesses.append("Tendency to rush")

        pedaling = self.metrics.get('pedaling', {})
        if pedaling.get('pedal_analysis_available', False):
            if pedaling.get('mode') == 'reference_comparison':
                if (
                    int(pedaling.get('harmonic_blur_count', 0)) > 0
                    or int(pedaling.get('phrase_boundary_clearance_issues', 0)) > 0
                ):
                    weaknesses.append("Pedal releases are blurring harmonic or phrase boundaries")
            elif bool(pedaling.get('excessive_pedaling', False)):
                weaknesses.append("Pedal is being held too continuously")
        
        return weaknesses


# Utility function for quick analysis
def analyze_performance_errors(reference_data: Dict, performance_data: Dict, 
                             aligned_notes: List = None) -> Dict[str, Any]:
    """
    Convenience function for quick error analysis.
    
    Args:
        reference_data: Parsed reference MIDI data
        performance_data: Parsed performance MIDI data
        aligned_notes: Optional aligned note pairs
        
    Returns:
        Error analysis results
    """
    analysis_data = {
        'reference': reference_data,
        'performance': performance_data,
        'alignment': aligned_notes or []
    }
    
    analyzer = ErrorAnalysis(analysis_data)
    return analyzer.analyze_performance()
