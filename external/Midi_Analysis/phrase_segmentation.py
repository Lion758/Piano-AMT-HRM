import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import statistics

class PhraseSegmentation:
    """
    Detect and analyze musical phrases in MIDI data.
    Uses multiple algorithms for robust phrase detection.
    """
    
    def __init__(self, parsed_data: Dict[str, Any]):
        """
        Initialize with parsed MIDI data.
        
        Args:
            parsed_data: Output from MIDIParser.parse_midi()
        """
        self.parsed_data = parsed_data
        self.notes = parsed_data.get('notes', [])
        self.total_duration = parsed_data.get('total_duration', 0)
        
        # Store segmentation results
        self.phrases = []
        self.sections = []
        self.motifs = []
        self.segmentation_metrics = {}
    
    def segment_phrases(self, method: str = 'combined', **kwargs) -> Dict[str, Any]:
        """
        Segment music into phrases using specified method.
        
        Args:
            method: 'silence', 'structure', 'rhythm', 'combined'
            **kwargs: Additional parameters for segmentation
            
        Returns:
            Dictionary containing phrase segmentation results
        """
        print(f"Segmenting phrases using {method} method...")
        
        if not self.notes:
            print("Warning: No notes to segment")
            return self._empty_result()
        
        # Reset previous results
        self.phrases = []
        self.sections = []
        self.motifs = []
        
        # Apply selected segmentation method
        if method == 'silence':
            self.phrases = self._segment_by_silence(**kwargs)
        elif method == 'structure':
            self.phrases = self._segment_by_structural_patterns(**kwargs)
        elif method == 'rhythm':
            self.phrases = self._segment_by_rhythmic_patterns(**kwargs)
        elif method == 'combined':
            # Use multiple methods and combine results
            silence_phrases = self._segment_by_silence(**kwargs)
            structure_phrases = self._segment_by_structural_patterns(**kwargs)
            
            # Combine results (favor structure when both agree)
            self.phrases = self._combine_segmentations(silence_phrases, structure_phrases)
        else:
            raise ValueError(f"Unknown segmentation method: {method}")
        
        # Analyze detected phrases
        self._analyze_phrases()
        
        # Detect larger sections
        self.sections = self._detect_sections()
        
        # Find repeating motifs
        self.motifs = self._find_repeating_motifs()
        
        # Calculate segmentation metrics
        self.segmentation_metrics = self._calculate_segmentation_metrics()
        
        return self._create_result_dict()
    
    def _segment_by_silence(self, silence_threshold: float = 0.5, 
                           min_notes_per_phrase: int = 4) -> List[List[Dict]]:
        """
        Segment based on rests/silences between notes.
        
        Args:
            silence_threshold: Minimum gap to consider a phrase boundary (seconds)
            min_notes_per_phrase: Minimum notes required for a valid phrase
            
        Returns:
            List of phrases, each phrase is a list of note dictionaries
        """
        if len(self.notes) < min_notes_per_phrase:
            return [self.notes] if self.notes else []
        
        phrases = []
        current_phrase = []
        
        for i in range(len(self.notes)):
            current_note = self.notes[i]
            current_phrase.append(current_note)
            
            # Check if next note is far enough away to be a phrase boundary
            if i < len(self.notes) - 1:
                next_note = self.notes[i + 1]
                gap = next_note['start'] - current_note['end']
                
                # Also consider large gaps in note starts (for staccato passages)
                start_gap = next_note['start'] - current_note['start']
                
                if gap > silence_threshold or start_gap > silence_threshold * 2:
                    if len(current_phrase) >= min_notes_per_phrase:
                        phrases.append(current_phrase.copy())
                    current_phrase = []
        
        # Add the last phrase if it has enough notes
        if len(current_phrase) >= min_notes_per_phrase:
            phrases.append(current_phrase)
        
        # If no phrases detected (continuous music), split by time
        if not phrases:
            phrases = self._split_by_time_intervals()
        
        return phrases
    
    def _segment_by_structural_patterns(self, 
                                       similarity_threshold: float = 0.7,
                                       min_pattern_length: int = 4) -> List[List[Dict]]:
        """
        Segment based on repeating patterns and structural cues.
        
        Args:
            similarity_threshold: Minimum similarity score to consider patterns similar
            min_pattern_length: Minimum notes in a pattern to consider
            
        Returns:
            List of phrases based on structural patterns
        """
        if len(self.notes) < min_pattern_length * 2:
            # Not enough notes for pattern detection
            return self._segment_by_silence()
        
        # Extract features for pattern matching
        note_features = self._extract_note_features()
        
        # Find repeating patterns
        patterns = self._find_patterns(note_features, min_pattern_length, similarity_threshold)
        
        # Use patterns to identify phrase boundaries
        phrase_boundaries = self._identify_boundaries_from_patterns(patterns)
        
        # Create phrases based on boundaries
        phrases = self._create_phrases_from_boundaries(phrase_boundaries)
        
        return phrases
    
    def _segment_by_rhythmic_patterns(self, 
                                     rhythm_change_threshold: float = 0.3,
                                     min_notes_for_rhythm: int = 8) -> List[List[Dict]]:
        """
        Segment based on rhythmic patterns and changes.
        
        Args:
            rhythm_change_threshold: Threshold for detecting rhythmic changes
            min_notes_for_rhythm: Minimum notes needed to analyze rhythm
            
        Returns:
            List of phrases based on rhythmic patterns
        """
        if len(self.notes) < min_notes_for_rhythm:
            return [self.notes] if self.notes else []
        
        # Analyze rhythmic patterns
        rhythmic_features = self._extract_rhythmic_features()
        
        # Detect changes in rhythmic patterns
        change_points = self._detect_rhythmic_changes(rhythmic_features, rhythm_change_threshold)
        
        # Create phrases based on rhythmic change points
        phrases = []
        start_idx = 0
        
        for change_point in change_points:
            if change_point - start_idx >= 4:  # Minimum notes per phrase
                phrases.append(self.notes[start_idx:change_point])
                start_idx = change_point
        
        # Add final phrase
        if start_idx < len(self.notes) and len(self.notes) - start_idx >= 4:
            phrases.append(self.notes[start_idx:])
        
        return phrases if phrases else [self.notes]
    
    def _extract_note_features(self) -> List[Dict]:
        """Extract features for pattern matching."""
        features = []
        
        for i, note in enumerate(self.notes):
            # Basic features for pattern matching
            feature = {
                'pitch': note['pitch'],
                'duration': note['duration'],
                'interval_to_next': 0,
                'rhythmic_value': self._quantize_duration(note['duration']),
                'melodic_direction': 0
            }
            
            # Calculate interval to next note
            if i < len(self.notes) - 1:
                next_note = self.notes[i + 1]
                feature['interval_to_next'] = next_note['pitch'] - note['pitch']
            
            # Calculate melodic direction
            if i > 0 and i < len(self.notes) - 1:
                prev_note = self.notes[i - 1]
                next_note = self.notes[i + 1]
                feature['melodic_direction'] = self._calculate_melodic_direction(
                    prev_note['pitch'], note['pitch'], next_note['pitch']
                )
            
            features.append(feature)
        
        return features
    
    def _extract_rhythmic_features(self) -> List[Dict]:
        """Extract rhythmic features for segmentation."""
        features = []
        
        # Calculate inter-onset intervals (IOI)
        iois = []
        for i in range(1, len(self.notes)):
            ioi = self.notes[i]['start'] - self.notes[i-1]['start']
            iois.append(ioi)
        
        # Calculate duration ratios
        for i in range(len(self.notes)):
            note = self.notes[i]
            rhythmic_feature = {
                'duration': note['duration'],
                'quantized_duration': self._quantize_duration(note['duration']),
                'position_in_measure': self._estimate_measure_position(note['start']),
                'is_downbeat': self._is_downbeat_position(note['start'])
            }
            
            # Add IOI information if available
            if i < len(iois):
                rhythmic_feature['ioi'] = iois[i]
                if i > 0:
                    rhythmic_feature['ioi_ratio'] = iois[i] / iois[i-1] if iois[i-1] > 0 else 1
            
            features.append(rhythmic_feature)
        
        return features
    
    def _find_patterns(self, features: List[Dict], min_length: int, 
                      similarity_threshold: float) -> List[Dict]:
        """Find repeating patterns in note features."""
        patterns = []
        n = len(features)
        
        # Simple pattern matching (sliding window)
        for pattern_length in range(min_length, min(12, n // 2)):
            for start in range(n - pattern_length):
                pattern = features[start:start + pattern_length]
                
                # Look for this pattern elsewhere in the piece
                for compare_start in range(start + pattern_length, n - pattern_length):
                    comparison = features[compare_start:compare_start + pattern_length]
                    
                    similarity = self._calculate_pattern_similarity(pattern, comparison)
                    
                    if similarity >= similarity_threshold:
                        patterns.append({
                            'pattern': pattern,
                            'locations': [start, compare_start],
                            'length': pattern_length,
                            'similarity': similarity
                        })
        
        # Filter to unique patterns
        unique_patterns = []
        seen_patterns = set()
        
        for pattern in patterns:
            pattern_key = self._create_pattern_key(pattern['pattern'])
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _identify_boundaries_from_patterns(self, patterns: List[Dict]) -> List[int]:
        """Identify phrase boundaries based on patterns."""
        if not patterns:
            return []
        
        # Count boundary occurrences (patterns often start/end phrase boundaries)
        boundary_scores = defaultdict(int)
        
        for pattern in patterns:
            for location in pattern['locations']:
                # Pattern start might be a phrase beginning
                boundary_scores[location] += pattern['similarity'] * 2
                # Pattern end might be a phrase ending
                boundary_scores[location + pattern['length']] += pattern['similarity']
        
        # Find peaks in boundary scores
        boundaries = []
        sorted_boundaries = sorted(boundary_scores.items(), key=lambda x: x[1], reverse=True)
        
        for idx, score in sorted_boundaries[:10]:  # Top 10 potential boundaries
            if 0 < idx < len(self.notes):
                boundaries.append(idx)
        
        return sorted(boundaries)
    
    def _detect_rhythmic_changes(self, rhythmic_features: List[Dict], 
                                threshold: float) -> List[int]:
        """Detect points where rhythm changes significantly."""
        if len(rhythmic_features) < 8:
            return []
        
        change_points = []
        
        # Analyze sliding windows for rhythmic consistency
        window_size = 8
        for i in range(window_size, len(rhythmic_features) - window_size):
            prev_window = rhythmic_features[i-window_size:i]
            curr_window = rhythmic_features[i:i+window_size]
            
            rhythm_change = self._calculate_rhythmic_change(prev_window, curr_window)
            
            if rhythm_change > threshold:
                change_points.append(i)
        
        # Filter adjacent change points
        filtered_points = []
        for point in change_points:
            if not filtered_points or point - filtered_points[-1] > window_size:
                filtered_points.append(point)
        
        return filtered_points
    
    def _analyze_phrases(self):
        """Analyze detected phrases for musical characteristics."""
        if not self.phrases:
            return
        
        for i, phrase in enumerate(self.phrases):
            if not phrase:
                continue
            
            # Add analysis metadata to phrase
            phrase_analysis = {
                'phrase_id': i + 1,
                'start_time': phrase[0]['start'],
                'end_time': phrase[-1]['end'],
                'duration': phrase[-1]['end'] - phrase[0]['start'],
                'note_count': len(phrase),
                'pitch_range': self._calculate_pitch_range(phrase),
                'average_velocity': statistics.mean([n['velocity'] for n in phrase]) if phrase else 0,
                'rhythmic_character': self._analyze_rhythmic_character(phrase),
                'melodic_contour': self._analyze_melodic_contour(phrase),
                'dynamic_shape': self._analyze_dynamic_shape(phrase)
            }
            
            # Store analysis with phrase
            self.phrases[i] = {
                'notes': phrase,
                'analysis': phrase_analysis
            }
    
    def _detect_sections(self) -> List[Dict]:
        """Detect larger musical sections (A, B, A', etc.)."""
        if len(self.phrases) < 3:
            return []
        
        sections = []
        current_section = {
            'type': 'A',
            'phrases': [0],
            'start_phrase': 0,
            'similarity_score': 1.0
        }
        
        # Compare phrases to find section boundaries
        for i in range(1, len(self.phrases)):
            phrase1 = self.phrases[current_section['phrases'][0]]['notes']
            phrase2 = self.phrases[i]['notes']
            
            similarity = self._calculate_phrase_similarity(phrase1, phrase2)
            
            if similarity < 0.5:  # Significant change - new section
                sections.append(current_section.copy())
                current_section = {
                    'type': chr(ord(current_section['type']) + 1),  # A -> B -> C, etc.
                    'phrases': [i],
                    'start_phrase': i,
                    'similarity_score': 1.0
                }
            else:
                current_section['phrases'].append(i)
                current_section['similarity_score'] = min(
                    current_section['similarity_score'], similarity
                )
        
        # Add final section
        if current_section['phrases']:
            sections.append(current_section)
        
        # Analyze sections
        for section in sections:
            section_notes = []
            for phrase_idx in section['phrases']:
                if phrase_idx < len(self.phrases):
                    section_notes.extend(self.phrases[phrase_idx]['notes'])
            
            if section_notes:
                section.update({
                    'start_time': section_notes[0]['start'],
                    'end_time': section_notes[-1]['end'],
                    'duration': section_notes[-1]['end'] - section_notes[0]['start'],
                    'note_count': len(section_notes),
                    'character': self._describe_section_character(section_notes)
                })
        
        return sections
    
    def _find_repeating_motifs(self) -> List[Dict]:
        """Find short repeating motifs (2-6 notes)."""
        if len(self.notes) < 8:
            return []
        
        motifs = []
        
        # Look for short patterns (2-6 notes)
        for motif_length in range(2, 7):
            for start in range(len(self.notes) - motif_length):
                motif = self.notes[start:start + motif_length]
                motif_features = self._extract_motif_features(motif)
                
                # Search for similar motifs
                occurrences = [start]
                for compare_start in range(start + motif_length, len(self.notes) - motif_length):
                    comparison = self.notes[compare_start:compare_start + motif_length]
                    comparison_features = self._extract_motif_features(comparison)
                    
                    similarity = self._calculate_motif_similarity(motif_features, comparison_features)
                    
                    if similarity > 0.8:
                        occurrences.append(compare_start)
                
                if len(occurrences) >= 2:  # Found at least 2 occurrences
                    motifs.append({
                        'motif': motif,
                        'length': motif_length,
                        'occurrences': occurrences,
                        'first_occurrence': start,
                        'interval_pattern': self._extract_interval_pattern(motif)
                    })
        
        # Filter duplicates
        unique_motifs = []
        seen_motifs = set()
        
        for motif in motifs:
            motif_key = str(motif['interval_pattern'])
            if motif_key not in seen_motifs:
                seen_motifs.add(motif_key)
                unique_motifs.append(motif)
        
        return unique_motifs[:10]  # Return top 10 motifs
    
    def _calculate_segmentation_metrics(self) -> Dict[str, Any]:
        """Calculate metrics for segmentation quality."""
        if not self.phrases:
            return {}
        
        phrase_lengths = [len(p['notes']) for p in self.phrases if isinstance(p, dict)]
        phrase_durations = [p['analysis']['duration'] for p in self.phrases if isinstance(p, dict)]
        
        return {
            'total_phrases': len(self.phrases),
            'average_phrase_length': statistics.mean(phrase_lengths) if phrase_lengths else 0,
            'phrase_length_std': statistics.stdev(phrase_lengths) if len(phrase_lengths) > 1 else 0,
            'average_phrase_duration': statistics.mean(phrase_durations) if phrase_durations else 0,
            'phrase_density': len(self.notes) / len(self.phrases) if self.phrases else 0,
            'section_count': len(self.sections),
            'motif_count': len(self.motifs),
            'segmentation_confidence': self._calculate_segmentation_confidence()
        }
    
    def _create_result_dict(self) -> Dict[str, Any]:
        """Create the final result dictionary."""
        return {
            'phrases': self.phrases,
            'sections': self.sections,
            'motifs': self.motifs,
            'segmentation_metrics': self.segmentation_metrics,
            'total_notes': len(self.notes),
            'segmentation_method': 'combined'
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure."""
        return {
            'phrases': [],
            'sections': [],
            'motifs': [],
            'segmentation_metrics': {},
            'total_notes': 0,
            'segmentation_method': 'none'
        }
    
    # Helper Methods
    
    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest musical value."""
        common_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125, 0.0625]  # Whole to 64th notes
        
        if duration <= 0:
            return 0
        
        # Find closest common duration
        closest = min(common_durations, key=lambda x: abs(x - duration))
        return closest
    
    def _calculate_melodic_direction(self, prev_pitch: int, curr_pitch: int, next_pitch: int) -> int:
        """Calculate melodic direction: -1=down, 0=same, 1=up, 2=changing."""
        prev_to_curr = curr_pitch - prev_pitch
        curr_to_next = next_pitch - curr_pitch
        
        if prev_to_curr > 0 and curr_to_next > 0:
            return 1  # Upward
        elif prev_to_curr < 0 and curr_to_next < 0:
            return -1  # Downward
        elif prev_to_curr == 0 and curr_to_next == 0:
            return 0  # Same
        else:
            return 2  # Changing direction
    
    def _estimate_measure_position(self, time: float) -> float:
        """Estimate position within measure (0-1)."""
        # Simple implementation assuming 4/4 time
        beat_duration = 0.5  # Assume 120 BPM = 0.5s per beat
        measure_duration = beat_duration * 4  # 4 beats per measure
        
        position_in_measure = (time % measure_duration) / measure_duration
        return position_in_measure
    
    def _is_downbeat_position(self, time: float) -> bool:
        """Check if time is likely a downbeat."""
        position = self._estimate_measure_position(time)
        return position < 0.1  # First 10% of measure
    
    def _calculate_pattern_similarity(self, pattern1: List[Dict], pattern2: List[Dict]) -> float:
        """Calculate similarity between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0
        
        scores = []
        for f1, f2 in zip(pattern1, pattern2):
            # Pitch similarity
            pitch_sim = 1 - abs(f1['pitch'] - f2['pitch']) / 48  # 4 octaves
            
            # Duration similarity
            dur_sim = 1 - abs(f1['duration'] - f2['duration']) / 2.0  # Max 2s difference
            
            # Interval similarity
            int_sim = 1 - abs(f1['interval_to_next'] - f2['interval_to_next']) / 24  # 2 octaves
            
            # Average similarity
            pattern_sim = (pitch_sim + dur_sim + int_sim) / 3
            scores.append(pattern_sim)
        
        return statistics.mean(scores) if scores else 0
    
    def _create_pattern_key(self, pattern: List[Dict]) -> str:
        """Create a unique key for a pattern."""
        key_parts = []
        for feature in pattern[:5]:  # Use first 5 notes for key
            key_parts.append(f"{feature['pitch']}:{feature['duration']:.2f}")
        return "|".join(key_parts)
    
    def _calculate_rhythmic_change(self, window1: List[Dict], window2: List[Dict]) -> float:
        """Calculate rhythmic change between two windows."""
        if len(window1) != len(window2):
            return 1.0  # Maximum change
        
        # Compare durations and IOIs
        dur_diffs = []
        ioi_diffs = []
        
        for w1, w2 in zip(window1, window2):
            dur_diffs.append(abs(w1['duration'] - w2['duration']))
            
            if 'ioi' in w1 and 'ioi' in w2:
                ioi_diffs.append(abs(w1['ioi'] - w2['ioi']))
        
        # Normalize differences
        avg_dur_diff = statistics.mean(dur_diffs) if dur_diffs else 0
        avg_ioi_diff = statistics.mean(ioi_diffs) if ioi_diffs else 0
        
        # Combine metrics
        rhythmic_change = (avg_dur_diff * 0.6 + avg_ioi_diff * 0.4) / 0.5  # Normalize to ~0-1
        
        return min(rhythmic_change, 1.0)
    
    def _split_by_time_intervals(self, interval: float = 4.0) -> List[List[Dict]]:
        """Split music into phrases by fixed time intervals."""
        if not self.notes:
            return []
        
        phrases = []
        current_phrase = []
        current_window_start = self.notes[0]['start']
        
        for note in self.notes:
            if note['start'] - current_window_start >= interval:
                if current_phrase:
                    phrases.append(current_phrase.copy())
                    current_phrase = []
                current_window_start = note['start']
            
            current_phrase.append(note)
        
        if current_phrase:
            phrases.append(current_phrase)
        
        return phrases
    
    def _combine_segmentations(self, phrases1: List[List[Dict]], 
                              phrases2: List[List[Dict]]) -> List[List[Dict]]:
        """Combine results from multiple segmentation methods."""
        if not phrases1:
            return phrases2
        if not phrases2:
            return phrases1
        
        # Simple combination: use phrases1 as base, add boundaries from phrases2
        combined = []
        all_boundaries = set()
        
        # Collect boundaries from both methods
        for phrases in [phrases1, phrases2]:
            current_idx = 0
            for phrase in phrases:
                if phrase:
                    start_idx = self._find_note_index(phrase[0])
                    if start_idx is not None:
                        all_boundaries.add(start_idx)
        
        # Sort boundaries and create phrases
        sorted_boundaries = sorted(all_boundaries)
        if not sorted_boundaries:
            return phrases1
        
        start_idx = 0
        for boundary in sorted_boundaries:
            if boundary - start_idx >= 4:  # Minimum notes per phrase
                combined.append(self.notes[start_idx:boundary])
                start_idx = boundary
        
        # Add final phrase
        if start_idx < len(self.notes):
            combined.append(self.notes[start_idx:])
        
        return combined if combined else phrases1
    
    def _find_note_index(self, target_note: Dict) -> Optional[int]:
        """Find index of a note in the notes list."""
        for i, note in enumerate(self.notes):
            if (abs(note['start'] - target_note['start']) < 0.01 and
                note['pitch'] == target_note['pitch']):
                return i
        return None
    
    def _create_phrases_from_boundaries(self, boundaries: List[int]) -> List[List[Dict]]:
        """Create phrases from boundary indices."""
        if not boundaries:
            return [self.notes] if self.notes else []
        
        phrases = []
        start_idx = 0
        
        for boundary in sorted(boundaries):
            if boundary - start_idx >= 4:  # Minimum notes per phrase
                phrases.append(self.notes[start_idx:boundary])
                start_idx = boundary
        
        # Add final phrase
        if start_idx < len(self.notes) and len(self.notes) - start_idx >= 4:
            phrases.append(self.notes[start_idx:])
        
        return phrases if phrases else [self.notes]
    
    def _calculate_pitch_range(self, notes: List[Dict]) -> Dict[str, int]:
        """Calculate pitch range for a set of notes."""
        if not notes:
            return {'min': 0, 'max': 0, 'range': 0}
        
        pitches = [n['pitch'] for n in notes]
        return {
            'min': min(pitches),
            'max': max(pitches),
            'range': max(pitches) - min(pitches)
        }
    
    def _analyze_rhythmic_character(self, notes: List[Dict]) -> str:
        """Analyze rhythmic character of a phrase."""
        if len(notes) < 3:
            return "Simple"
        
        durations = [n['duration'] for n in notes]
        dur_std = statistics.stdev(durations) if len(durations) > 1 else 0
        
        if dur_std < 0.05:
            return "Rhythmically steady"
        elif dur_std < 0.15:
            return "Moderately varied rhythm"
        else:
            return "Rhythmically varied"
    
    def _analyze_melodic_contour(self, notes: List[Dict]) -> str:
        """Analyze melodic contour of a phrase."""
        if len(notes) < 3:
            return "Simple"
        
        pitches = [n['pitch'] for n in notes]
        direction_changes = 0
        
        for i in range(1, len(pitches) - 1):
            prev_dir = pitches[i] - pitches[i-1]
            curr_dir = pitches[i+1] - pitches[i]
            
            if (prev_dir > 0 and curr_dir < 0) or (prev_dir < 0 and curr_dir > 0):
                direction_changes += 1
        
        change_ratio = direction_changes / (len(pitches) - 2)
        
        if change_ratio < 0.2:
            return "Smooth contour"
        elif change_ratio < 0.4:
            return "Moderately varied"
        else:
            return "Angular contour"
    
    def _analyze_dynamic_shape(self, notes: List[Dict]) -> str:
        """Analyze dynamic shape of a phrase."""
        if len(notes) < 4:
            return "Neutral"
        
        velocities = [n['velocity'] for n in notes]
        
        # Check for crescendo/decrescendo
        first_half = velocities[:len(velocities)//2]
        second_half = velocities[len(velocities)//2:]
        
        avg_first = statistics.mean(first_half)
        avg_second = statistics.mean(second_half)
        
        if avg_second > avg_first + 10:
            return "Crescendo"
        elif avg_second < avg_first - 10:
            return "Decrescendo"
        else:
            return "Steady dynamics"
    
    def _calculate_phrase_similarity(self, phrase1: List[Dict], phrase2: List[Dict]) -> float:
        """Calculate similarity between two phrases."""
        if not phrase1 or not phrase2:
            return 0
        
        # Compare pitch contours
        pitches1 = [n['pitch'] for n in phrase1]
        pitches2 = [n['pitch'] for n in phrase2]
        
        # Compare durations
        durations1 = [n['duration'] for n in phrase1]
        durations2 = [n['duration'] for n in phrase2]
        
        # Simple similarity measure
        min_len = min(len(pitches1), len(pitches2))
        if min_len < 3:
            return 0
        
        # Compare first few notes
        pitch_sim = 1 - sum(abs(p1 - p2) for p1, p2 in zip(pitches1[:min_len], pitches2[:min_len])) / (min_len * 12)
        dur_sim = 1 - sum(abs(d1 - d2) for d1, d2 in zip(durations1[:min_len], durations2[:min_len])) / (min_len * 0.5)
        
        return (pitch_sim + dur_sim) / 2
    
    def _describe_section_character(self, notes: List[Dict]) -> str:
        """Describe the character of a section."""
        if not notes:
            return "Empty"
        
        velocities = [n['velocity'] for n in notes]
        avg_velocity = statistics.mean(velocities) if velocities else 64
        
        if avg_velocity > 100:
            return "Loud and energetic"
        elif avg_velocity > 80:
            return "Moderately loud"
        elif avg_velocity > 60:
            return "Moderate"
        elif avg_velocity > 40:
            return "Soft"
        else:
            return "Very soft"
    
    def _extract_motif_features(self, motif: List[Dict]) -> Dict[str, Any]:
        """Extract features for motif matching."""
        if not motif:
            return {}
        
        pitches = [n['pitch'] for n in motif]
        durations = [n['duration'] for n in motif]
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(pitches)):
            intervals.append(pitches[i] - pitches[i-1])
        
        return {
            'pitches': pitches,
            'durations': durations,
            'intervals': intervals,
            'contour': self._calculate_contour(pitches)
        }
    
    def _calculate_motif_similarity(self, motif1: Dict, motif2: Dict) -> float:
        """Calculate similarity between two motifs."""
        if not motif1 or not motif2:
            return 0
        
        # Compare intervals (most important for motif recognition)
        intervals1 = motif1.get('intervals', [])
        intervals2 = motif2.get('intervals', [])
        
        if len(intervals1) != len(intervals2):
            return 0
        
        interval_sim = 1 - sum(abs(i1 - i2) for i1, i2 in zip(intervals1, intervals2)) / (len(intervals1) * 6)
        
        # Compare contours
        contour1 = motif1.get('contour', [])
        contour2 = motif2.get('contour', [])
        
        if len(contour1) == len(contour2):
            contour_sim = sum(1 for c1, c2 in zip(contour1, contour2) if c1 == c2) / len(contour1)
        else:
            contour_sim = 0
        
        return (interval_sim * 0.7 + contour_sim * 0.3)
    
    def _calculate_contour(self, pitches: List[int]) -> List[str]:
        """Calculate contour representation (U=up, D=down, S=same)."""
        contour = []
        for i in range(1, len(pitches)):
            diff = pitches[i] - pitches[i-1]
            if diff > 0:
                contour.append('U')
            elif diff < 0:
                contour.append('D')
            else:
                contour.append('S')
        return contour
    
    def _extract_interval_pattern(self, motif: List[Dict]) -> List[int]:
        """Extract interval pattern from motif."""
        pattern = []
        for i in range(1, len(motif)):
            interval = motif[i]['pitch'] - motif[i-1]['pitch']
            pattern.append(interval)
        return pattern
    
    def _calculate_segmentation_confidence(self) -> float:
        """Calculate confidence score for segmentation."""
        if not self.phrases:
            return 0
        
        # Confidence based on phrase regularity and method agreement
        phrase_lengths = [len(p['notes']) for p in self.phrases if isinstance(p, dict)]
        
        if len(phrase_lengths) < 2:
            return 0.5
        
        # More regular phrase lengths = higher confidence
        length_std = statistics.stdev(phrase_lengths)
        length_mean = statistics.mean(phrase_lengths)
        
        if length_mean == 0:
            return 0
        
        cv = length_std / length_mean  # Coefficient of variation
        regularity_score = 1 / (1 + cv)
        
        # Boost confidence if we found sections or motifs
        extra_score = min(0.3, len(self.sections) * 0.1 + len(self.motifs) * 0.05)
        
        return min(regularity_score + extra_score, 1.0)


# Utility functions for easy usage
def segment_midi_phrases(parsed_data: Dict[str, Any], method: str = 'combined') -> Dict[str, Any]:
    """
    Convenience function for quick phrase segmentation.
    
    Args:
        parsed_data: Output from MIDIParser
        method: Segmentation method
        
    Returns:
        Phrase segmentation results
    """
    segmenter = PhraseSegmentation(parsed_data)
    return segmenter.segment_phrases(method=method)


def extract_phrases_only(parsed_data: Dict[str, Any]) -> List[List[Dict]]:
    """
    Extract just the phrases without additional analysis.
    
    Returns:
        List of phrases (each phrase is list of notes)
    """
    segmenter = PhraseSegmentation(parsed_data)
    result = segmenter.segment_phrases(method='silence')
    return [p['notes'] for p in result['phrases']] if 'phrases' in result else []