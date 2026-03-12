import numpy as np
from typing import Dict, List, Any, Tuple
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
            'detailed_errors': self._get_detailed_error_list(),
            'performance_summary': self._get_performance_summary()
        }

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
        reference_notes = self.reference_data.get('notes', [])
        performance_notes = self.performance_data.get('notes', [])
        
        if not performance_notes:
            return
        
        perf_durations = [note['duration'] for note in performance_notes]
        
        # Categorize articulation types
        note_intervals = self._calculate_note_intervals(performance_notes)
        articulation_ratios = []
        
        for i in range(len(performance_notes) - 1):
            duration = performance_notes[i]['duration']
            interval = note_intervals[i] if i < len(note_intervals) else 0
            if interval > 0:
                ratio = duration / interval
                articulation_ratios.append(ratio)
        
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
        """Analyze sustain pedal usage (if pedal data is available)."""
        # This is a placeholder for pedaling analysis
        # In real implementation, you would parse control change messages for pedal
        
        self.metrics['pedaling'] = {
            'pedal_analysis_available': False,
            'note': 'Pedal analysis requires parsing of control change messages'
        }
    
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
            articulation_score = float(self.metrics['articulation'].get('articulation_consistency', 0.5))
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

        deduped: List[str] = []
        seen = set()
        for rec in recommendations:
            key = rec.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(rec)

        self.practice_recommendations = deduped
    
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
