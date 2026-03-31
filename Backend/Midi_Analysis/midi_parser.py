import pretty_midi
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import statistics

class MIDIParser:
    def __init__(self):
        self.parsed_data = {}
        self.midi_data = None
    
    def parse_midi(self, midi_path) -> Dict[str, Any]:
        """Parse MIDI file into comprehensive structured data."""
        try:
            self.midi_data = pretty_midi.PrettyMIDI(midi_path)

            # Extract notes ONCE
            notes = self._extract_notes()

            # Compute duration from actual max note end time (fallback to get_end_time)
            if notes:
                total_duration = max(n['end'] for n in notes)
            else:
                total_duration = self.midi_data.get_end_time() if self.midi_data else 0

            metadata = self._extract_metadata_from_notes(notes, total_duration)

            return {
                'notes': notes,
                'total_duration': total_duration,          # add top-level
                'metadata': metadata,                      # metadata also has total_duration
                'timing': self._extract_timing_info(),
                'harmony': self._extract_harmonic_content(),
                'structure': self._extract_musical_structure(),
                'performance_data': self._extract_performance_patterns()
            }
            
        except Exception as e:
            print(f"Error parsing MIDI file: {e}")
            return {}
    
    def _extract_notes(self) -> List[Dict]:
        """Extract detailed note information."""
        notes = []
        for i, instrument in enumerate(self.midi_data.instruments):
            for note in instrument.notes:
                note_data = {
                    'pitch': int(note.pitch),
                    'pitch_name': pretty_midi.note_number_to_name(note.pitch),
                    'start': float(note.start),
                    'end': float(note.end),
                    'velocity': int(note.velocity),
                    'duration': float(note.end - note.start),
                    'instrument': int(instrument.program),
                    'instrument_name': instrument.name,
                    'track_id': i,
                    'beat_position': self._get_beat_position(note.start),
                    'measure_position': self._get_measure_position(note.start)
                }
                notes.append(note_data)
        
        return sorted(notes, key=lambda x: x['start'])
    
    def _extract_metadata_from_notes(self, notes: List[Dict], total_duration: float) -> Dict:
        """Extract MIDI file metadata using already-extracted notes."""
        return {
            'total_duration': total_duration,
            'instruments': [
                {
                    'name': inst.name,
                    'program': int(inst.program),
                    'is_drum': bool(inst.is_drum),
                    'note_count': int(len(inst.notes))
                } for inst in self.midi_data.instruments
            ],
            'key_signature_changes': [
                {'key': str(ks.key), 'time': ks.time}
                for ks in self.midi_data.key_signature_changes
            ],
            'lyrics': [ly.text for ly in self.midi_data.lyrics] if hasattr(self.midi_data, 'lyrics') else []
        }
    def _extract_timing_info(self) -> Dict:
        """Extract tempo and time signature information."""
        # Get beats and downbeats for rhythmic analysis
        beats = self.midi_data.get_beats()
        downbeats = self.midi_data.get_downbeats()
        
        return {
            # 'tempo_changes': [
            #     {
            #         'tempo': tc.tempo,
            #         'time': tc.time,
            #         'bpm': tc.tempo
            #     } for tc in self.midi_data.tempo_changes
            # ],
            'time_signature_changes': [
                {
                    'numerator': ts.numerator,
                    'denominator': ts.denominator,
                    'time': ts.time
                } for ts in self.midi_data.time_signature_changes
            ],
            'average_tempo': self.midi_data.estimate_tempo(),
            'beats': beats.tolist() if beats is not None else [],
            'downbeats': downbeats.tolist() if downbeats is not None else [],
            'ticks_per_beat': getattr(self.midi_data, 'ticks_per_beat', None)
        }
    
    def _extract_harmonic_content(self) -> Dict:
        """Extract chords and harmonic analysis."""
        notes = self._extract_notes()
        
        # Simple chord detection (you can enhance this)
        chords = self._detect_chords(notes)
        
        return {
            'chords': chords,
            'pitch_range': {
                'min_pitch': min(note['pitch'] for note in notes) if notes else 0,
                'max_pitch': max(note['pitch'] for note in notes) if notes else 0,
                'pitch_variety': len(set(note['pitch'] for note in notes))
            }
        }
    
    def _extract_musical_structure(self) -> Dict:
        """Extract phrases and musical sections."""
        notes = self._extract_notes()
        
        return {
            'phrases': self._detect_phrases(notes),
            'sections': self._detect_sections(notes),
            'note_density': self._calculate_note_density(notes)
        }
    
    def _extract_performance_patterns(self) -> Dict:
        """Extract patterns useful for performance analysis."""
        notes = self._extract_notes()
        
        return {
            'velocity_profile': {
                'mean_velocity': np.mean([n['velocity'] for n in notes]) if notes else 0,
                'velocity_std': np.std([n['velocity'] for n in notes]) if notes else 0,
                'dynamic_range': {
                    'min': min(n['velocity'] for n in notes) if notes else 0,
                    'max': max(n['velocity'] for n in notes) if notes else 0
                }
            },
            'timing_consistency': self._analyze_timing_consistency(notes),
            'articulation_patterns': self._analyze_articulation(notes)
        }
    
    # Helper methods would go here...
    def _get_beat_position(self, time: float) -> float:
        """Return absolute fractional beat index (0-based) at `time`."""
        if self.midi_data is None:
            return 0.0

        beats = self.midi_data.get_beats()
        if beats is None or len(beats) == 0:
            return 0.0

        beats_arr = np.asarray(beats, dtype=float)
        idx = int(np.searchsorted(beats_arr, float(time), side='right') - 1)

        if idx < 0:
            if len(beats_arr) > 1:
                beat_len = max(float(beats_arr[1] - beats_arr[0]), 1e-6)
            else:
                beat_len = 0.5
            return float((time - beats_arr[0]) / beat_len)

        if idx >= len(beats_arr) - 1:
            if len(beats_arr) > 1:
                beat_len = max(float(beats_arr[-1] - beats_arr[-2]), 1e-6)
            else:
                beat_len = 0.5
            return float((len(beats_arr) - 1) + ((time - beats_arr[-1]) / beat_len))

        beat_len = max(float(beats_arr[idx + 1] - beats_arr[idx]), 1e-6)
        return float(idx + ((time - beats_arr[idx]) / beat_len))
    
    def _get_measure_position(self, time: float) -> Dict:
        """Return measure-aware position metadata for `time`."""
        if self.midi_data is None:
            return {
                'measure': 1,
                'beat_in_measure': 1.0,
                'beats_per_measure': 4.0,
                'time_signature': '4/4'
            }

        t = float(time)
        raw_downbeats = self.midi_data.get_downbeats()
        if raw_downbeats is None:
            downbeats = np.asarray([], dtype=float)
        else:
            downbeats = np.asarray(raw_downbeats, dtype=float)

        # Active time signature at this moment (fallback 4/4).
        numerator = 4
        denominator = 4
        for ts in sorted(self.midi_data.time_signature_changes, key=lambda x: x.time):
            if float(ts.time) <= t:
                numerator = int(ts.numerator)
                denominator = int(ts.denominator)
            else:
                break

        beats_per_measure = float(numerator) * (4.0 / float(denominator))

        if downbeats.size == 0:
            abs_beat = self._get_beat_position(t)
            measure_number = int(max(0.0, abs_beat) // max(beats_per_measure, 1e-6)) + 1
            beat_in_measure = (max(0.0, abs_beat) % max(beats_per_measure, 1e-6)) + 1.0
            return {
                'measure': measure_number,
                'beat_in_measure': round(float(beat_in_measure), 3),
                'beats_per_measure': round(beats_per_measure, 3),
                'time_signature': f'{numerator}/{denominator}'
            }

        measure_idx = int(np.searchsorted(downbeats, t, side='right') - 1)
        if measure_idx < 0:
            measure_idx = 0

        measure_start = float(downbeats[measure_idx])
        beat_in_measure = (self._get_beat_position(t) - self._get_beat_position(measure_start)) + 1.0
        if beat_in_measure < 1.0:
            beat_in_measure = 1.0

        return {
            'measure': int(measure_idx + 1),
            'beat_in_measure': round(float(beat_in_measure), 3),
            'beats_per_measure': round(beats_per_measure, 3),
            'time_signature': f'{numerator}/{denominator}'
        }
    
    def _detect_chords(self, notes: List[Dict]) -> List[Dict]:
        """Simple chord detection algorithm."""
        if not notes:
            return []

        time_window = 0.05  # 50ms onset window
        min_notes_for_chord = 3
        notes_sorted = sorted(notes, key=lambda n: (float(n.get('start', 0.0)), int(n.get('pitch', 0))))

        def _pc_name(pc: int) -> str:
            names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            return names[int(pc) % 12]

        def _classify_chord(pitches: List[int]) -> Tuple[str, str]:
            unique_pcs = sorted({p % 12 for p in pitches})
            if len(unique_pcs) < 3:
                return "unknown", "unknown"

            chord_templates = [
                ("major7", {0, 4, 7, 11}),
                ("minor7", {0, 3, 7, 10}),
                ("dominant7", {0, 4, 7, 10}),
                ("half_diminished7", {0, 3, 6, 10}),
                ("diminished7", {0, 3, 6, 9}),
                ("major", {0, 4, 7}),
                ("minor", {0, 3, 7}),
                ("diminished", {0, 3, 6}),
                ("augmented", {0, 4, 8}),
                ("sus2", {0, 2, 7}),
                ("sus4", {0, 5, 7}),
            ]

            best = None
            for root in unique_pcs:
                intervals = {(pc - root) % 12 for pc in unique_pcs}
                for quality, tpl in chord_templates:
                    if tpl.issubset(intervals):
                        extra = len(intervals - tpl)
                        score = (len(tpl), -extra)
                        if best is None or score > best["score"]:
                            best = {
                                "root": root,
                                "quality": quality,
                                "score": score,
                            }
            if best is None:
                return "unknown", "unknown"

            root_name = _pc_name(best["root"])
            quality = best["quality"]
            return f"{root_name}:{quality}", quality

        clusters: List[List[Dict]] = []
        cur_cluster: List[Dict] = [notes_sorted[0]]
        cur_anchor = float(notes_sorted[0].get('start', 0.0))

        for note in notes_sorted[1:]:
            st = float(note.get('start', 0.0))
            if st - cur_anchor <= time_window:
                cur_cluster.append(note)
            else:
                clusters.append(cur_cluster)
                cur_cluster = [note]
                cur_anchor = st
        clusters.append(cur_cluster)

        out = []
        chord_id = 1
        for cluster in clusters:
            if len(cluster) < min_notes_for_chord:
                continue

            pitches = sorted(int(n.get('pitch', 0)) for n in cluster)
            unique_pitches = sorted(set(pitches))
            if len(unique_pitches) < min_notes_for_chord:
                continue

            starts = [float(n.get('start', 0.0)) for n in cluster]
            ends = [float(n.get('end', 0.0)) for n in cluster]
            velocities = [int(n.get('velocity', 0)) for n in cluster]

            chord_name, quality = _classify_chord(unique_pitches)
            first_note = min(cluster, key=lambda n: float(n.get('start', 0.0)))
            measure_pos = first_note.get('measure_position', {})
            measure = (
                int(measure_pos.get('measure'))
                if isinstance(measure_pos, dict) and isinstance(measure_pos.get('measure'), (int, float))
                else None
            )

            out.append({
                'chord_id': chord_id,
                'start_time': round(min(starts), 6),
                'end_time': round(max(ends), 6),
                'duration': round(max(0.0, max(ends) - min(starts)), 6),
                'note_count': len(cluster),
                'pitches': unique_pitches,
                'pitch_names': [pretty_midi.note_number_to_name(p) for p in unique_pitches],
                'bass_pitch': int(min(unique_pitches)),
                'top_pitch': int(max(unique_pitches)),
                'mean_velocity': round(float(statistics.mean(velocities)), 2) if velocities else 0.0,
                'measure': measure,
                'chord_name': chord_name,
                'quality': quality,
            })
            chord_id += 1

        return out
    
    def _detect_phrases(self, notes: List[Dict]) -> List[Dict]:
        """Detect musical phrases based on rests and patterns."""
        if not notes:
            return []

        notes_sorted = sorted(notes, key=lambda n: (float(n.get('start', 0.0)), int(n.get('pitch', 0))))
        if len(notes_sorted) < 2:
            n = notes_sorted[0]
            measure = n.get('measure_position', {}).get('measure') if isinstance(n.get('measure_position', {}), dict) else None
            return [{
                'phrase_id': 1,
                'start_time': float(n.get('start', 0.0)),
                'end_time': float(n.get('end', n.get('start', 0.0))),
                'duration': float(max(0.0, float(n.get('end', n.get('start', 0.0))) - float(n.get('start', 0.0)))),
                'note_count': 1,
                'start_measure': int(measure) if isinstance(measure, (int, float)) else None,
                'end_measure': int(measure) if isinstance(measure, (int, float)) else None,
                'mean_velocity': float(n.get('velocity', 0.0)),
                'pitch_range': {'min': int(n.get('pitch', 0)), 'max': int(n.get('pitch', 0)), 'range': 0},
            }]

        iois = []
        rests = []
        for i in range(1, len(notes_sorted)):
            prev = notes_sorted[i - 1]
            curr = notes_sorted[i]
            ioi = float(curr.get('start', 0.0)) - float(prev.get('start', 0.0))
            rest = float(curr.get('start', 0.0)) - float(prev.get('end', prev.get('start', 0.0)))
            if ioi > 1e-6:
                iois.append(ioi)
            rests.append(max(0.0, rest))

        median_ioi = statistics.median(iois) if iois else 0.25
        median_rest = statistics.median(rests) if rests else 0.0
        rest_threshold = max(0.45, median_ioi * 2.6, median_rest * 3.0)
        start_gap_threshold = max(0.8, median_ioi * 4.5)
        min_notes_per_phrase = 4

        boundaries = []
        for i in range(len(notes_sorted) - 1):
            a = notes_sorted[i]
            b = notes_sorted[i + 1]
            rest_gap = float(b.get('start', 0.0)) - float(a.get('end', a.get('start', 0.0)))
            start_gap = float(b.get('start', 0.0)) - float(a.get('start', 0.0))
            if rest_gap >= rest_threshold or start_gap >= start_gap_threshold:
                boundaries.append(i)

        segments = []
        seg_start = 0
        for boundary_idx in boundaries:
            segments.append(notes_sorted[seg_start:boundary_idx + 1])
            seg_start = boundary_idx + 1
        segments.append(notes_sorted[seg_start:])

        merged_segments: List[List[Dict]] = []
        for seg in segments:
            if len(seg) < min_notes_per_phrase and merged_segments:
                merged_segments[-1].extend(seg)
            else:
                merged_segments.append(seg.copy())

        if len(merged_segments) > 1 and len(merged_segments[-1]) < min_notes_per_phrase:
            merged_segments[-2].extend(merged_segments[-1])
            merged_segments = merged_segments[:-1]

        phrases = []
        for idx, seg in enumerate(merged_segments):
            if not seg:
                continue
            seg = sorted(seg, key=lambda n: (float(n.get('start', 0.0)), int(n.get('pitch', 0))))
            starts = [float(n.get('start', 0.0)) for n in seg]
            ends = [float(n.get('end', n.get('start', 0.0))) for n in seg]
            velocities = [int(n.get('velocity', 0)) for n in seg]
            pitches = [int(n.get('pitch', 0)) for n in seg]

            first_measure_obj = seg[0].get('measure_position', {})
            last_measure_obj = seg[-1].get('measure_position', {})
            start_measure = (
                int(first_measure_obj.get('measure'))
                if isinstance(first_measure_obj, dict) and isinstance(first_measure_obj.get('measure'), (int, float))
                else None
            )
            end_measure = (
                int(last_measure_obj.get('measure'))
                if isinstance(last_measure_obj, dict) and isinstance(last_measure_obj.get('measure'), (int, float))
                else None
            )

            phrases.append({
                'phrase_id': idx + 1,
                'start_time': round(min(starts), 6),
                'end_time': round(max(ends), 6),
                'duration': round(max(0.0, max(ends) - min(starts)), 6),
                'note_count': len(seg),
                'start_measure': start_measure,
                'end_measure': end_measure,
                'mean_velocity': round(float(statistics.mean(velocities)), 2) if velocities else 0.0,
                'pitch_range': {
                    'min': min(pitches) if pitches else 0,
                    'max': max(pitches) if pitches else 0,
                    'range': (max(pitches) - min(pitches)) if pitches else 0,
                },
            })

        return phrases
    
    def _detect_sections(self, notes: List[Dict]) -> List[Dict]:
        """Detect musical sections based on patterns and changes."""
        if not notes:
            return []

        notes_sorted = sorted(notes, key=lambda n: (float(n.get('start', 0.0)), int(n.get('pitch', 0))))
        phrases = self._detect_phrases(notes_sorted)
        if not phrases:
            return []

        # Section boundaries from phrase-level shifts.
        boundaries = [0]
        for i in range(1, len(phrases)):
            prev = phrases[i - 1]
            curr = phrases[i]

            prev_density = prev.get('note_count', 0) / max(1e-6, float(prev.get('duration', 0.0) or 1e-6))
            curr_density = curr.get('note_count', 0) / max(1e-6, float(curr.get('duration', 0.0) or 1e-6))
            density_ratio = (curr_density + 1e-6) / (prev_density + 1e-6)

            prev_center = (
                (prev.get('pitch_range', {}).get('min', 0) + prev.get('pitch_range', {}).get('max', 0)) / 2.0
                if isinstance(prev.get('pitch_range', {}), dict) else 0.0
            )
            curr_center = (
                (curr.get('pitch_range', {}).get('min', 0) + curr.get('pitch_range', {}).get('max', 0)) / 2.0
                if isinstance(curr.get('pitch_range', {}), dict) else 0.0
            )
            pitch_shift = abs(curr_center - prev_center)

            gap = float(curr.get('start_time', 0.0)) - float(prev.get('end_time', 0.0))

            if gap > 1.5 or density_ratio > 1.9 or density_ratio < (1.0 / 1.9) or pitch_shift >= 9.0:
                boundaries.append(i)

        boundaries = sorted(set(boundaries))
        boundaries.append(len(phrases))

        sections = []
        for sec_idx in range(len(boundaries) - 1):
            a = boundaries[sec_idx]
            b = boundaries[sec_idx + 1]
            block = phrases[a:b]
            if not block:
                continue

            start_time = float(block[0].get('start_time', 0.0))
            end_time = float(block[-1].get('end_time', start_time))
            duration = max(0.0, end_time - start_time)
            note_count = int(sum(int(p.get('note_count', 0)) for p in block))

            start_measure = block[0].get('start_measure')
            end_measure = block[-1].get('end_measure')

            density = note_count / duration if duration > 1e-6 else 0.0
            mean_velocity_vals = [float(p.get('mean_velocity', 0.0)) for p in block]
            mean_velocity = statistics.mean(mean_velocity_vals) if mean_velocity_vals else 0.0

            section_type = chr(ord('A') + sec_idx) if sec_idx < 26 else f"S{sec_idx + 1}"

            if density >= 8:
                character = "dense and active"
            elif density >= 4:
                character = "moderately active"
            else:
                character = "sparser texture"
            if mean_velocity >= 90:
                character += ", louder dynamic profile"
            elif mean_velocity <= 55:
                character += ", softer dynamic profile"

            sections.append({
                'section_id': sec_idx + 1,
                'label': section_type,
                'start_phrase': int(a + 1),
                'end_phrase': int(b),
                'start_measure': int(start_measure) if isinstance(start_measure, (int, float)) else None,
                'end_measure': int(end_measure) if isinstance(end_measure, (int, float)) else None,
                'start_time': round(start_time, 6),
                'end_time': round(end_time, 6),
                'duration': round(duration, 6),
                'note_count': note_count,
                'phrase_count': len(block),
                'note_density': round(float(density), 3),
                'mean_velocity': round(float(mean_velocity), 2),
                'character': character,
            })

        return sections
    
    def _calculate_note_density(self, notes: List[Dict]) -> List[Dict]:
        """Calculate note density over time."""
        if not notes:
            return []

        notes_sorted = sorted(notes, key=lambda n: float(n.get('start', 0.0)))
        total_duration = max(float(n.get('end', n.get('start', 0.0))) for n in notes_sorted)
        if total_duration <= 0:
            return []

        # Adaptive windowing: smaller for short files, larger for long files.
        if total_duration <= 30:
            window = 1.0
            hop = 0.5
        elif total_duration <= 180:
            window = 2.0
            hop = 1.0
        else:
            window = 4.0
            hop = 2.0

        out = []
        t = 0.0
        while t <= total_duration + 1e-6:
            t1 = min(total_duration, t + window)
            bucket = [n for n in notes_sorted if t <= float(n.get('start', 0.0)) < t1]
            count = len(bucket)
            unique_onsets = len({round(float(n.get('start', 0.0)), 4) for n in bucket})
            density = count / max(1e-6, (t1 - t))
            onset_density = unique_onsets / max(1e-6, (t1 - t))

            out.append({
                'window_start': round(float(t), 6),
                'window_end': round(float(t1), 6),
                'note_count': int(count),
                'unique_onset_count': int(unique_onsets),
                'notes_per_second': round(float(density), 3),
                'onsets_per_second': round(float(onset_density), 3),
            })
            t += hop

        return out
    
    def _analyze_timing_consistency(self, notes: List[Dict]) -> Dict:
        """Analyze timing consistency for performance evaluation."""
        if len(notes) < 3:
            return {
                'available': False,
                'reason': 'insufficient_notes',
                'stability_score': 0.5,
                'ioi_mean': 0.0,
                'ioi_std': 0.0,
                'ioi_cv': 0.0,
                'mean_beat_deviation_ms': None,
                'tempo_drift_ratio': 0.0,
            }

        notes_sorted = sorted(notes, key=lambda n: float(n.get('start', 0.0)))
        starts = [float(n.get('start', 0.0)) for n in notes_sorted]
        iois = [starts[i] - starts[i - 1] for i in range(1, len(starts)) if (starts[i] - starts[i - 1]) > 1e-6]

        if len(iois) < 2:
            return {
                'available': False,
                'reason': 'insufficient_ioi',
                'stability_score': 0.5,
                'ioi_mean': 0.0,
                'ioi_std': 0.0,
                'ioi_cv': 0.0,
                'mean_beat_deviation_ms': None,
                'tempo_drift_ratio': 0.0,
            }

        ioi_mean = float(statistics.mean(iois))
        ioi_std = float(statistics.stdev(iois)) if len(iois) > 1 else 0.0
        ioi_cv = (ioi_std / ioi_mean) if ioi_mean > 1e-9 else 0.0
        stability_score = max(0.0, min(1.0, 1.0 / (1.0 + ioi_cv)))

        q = max(2, len(iois) // 4)
        head_mean = float(statistics.mean(iois[:q]))
        tail_mean = float(statistics.mean(iois[-q:]))
        tempo_drift_ratio = ((tail_mean - head_mean) / head_mean) if head_mean > 1e-9 else 0.0

        beat_deviation_ms: Optional[float] = None
        if self.midi_data is not None:
            beats = self.midi_data.get_beats()
            if beats is not None and len(beats) > 1:
                beat_arr = np.asarray(beats, dtype=float)
                deviations = []
                for st in starts:
                    idx = int(np.searchsorted(beat_arr, st))
                    nearest = []
                    if 0 <= idx < len(beat_arr):
                        nearest.append(abs(st - float(beat_arr[idx])))
                    if idx - 1 >= 0:
                        nearest.append(abs(st - float(beat_arr[idx - 1])))
                    if nearest:
                        deviations.append(min(nearest))
                if deviations:
                    beat_deviation_ms = float(statistics.mean(deviations) * 1000.0)

        tendency = "stable"
        if tempo_drift_ratio > 0.12:
            tendency = "slowing_down"
        elif tempo_drift_ratio < -0.12:
            tendency = "speeding_up"

        return {
            'available': True,
            'stability_score': round(float(stability_score), 3),
            'ioi_mean': round(ioi_mean, 4),
            'ioi_std': round(ioi_std, 4),
            'ioi_cv': round(float(ioi_cv), 4),
            'mean_beat_deviation_ms': round(beat_deviation_ms, 2) if beat_deviation_ms is not None else None,
            'tempo_drift_ratio': round(float(tempo_drift_ratio), 4),
            'tempo_tendency': tendency,
        }
    
    def _analyze_articulation(self, notes: List[Dict]) -> Dict:
        """Analyze articulation patterns (staccato, legato)."""
        if len(notes) < 2:
            return {
                'available': False,
                'reason': 'insufficient_notes',
                'average_duration': 0.0,
                'average_ioi': 0.0,
                'articulation_ratio_mean': 0.0,
                'articulation_ratio_std': 0.0,
                'staccato_percentage': 0.0,
                'legato_percentage': 0.0,
                'detached_percentage': 0.0,
            }

        notes_sorted = sorted(notes, key=lambda n: float(n.get('start', 0.0)))
        durations = [max(0.0, float(n.get('duration', 0.0))) for n in notes_sorted]

        iois = []
        ratios = []
        for i in range(len(notes_sorted) - 1):
            cur = notes_sorted[i]
            nxt = notes_sorted[i + 1]
            ioi = float(nxt.get('start', 0.0)) - float(cur.get('start', 0.0))
            if ioi <= 1e-6:
                continue
            dur = max(0.0, float(cur.get('duration', 0.0)))
            iois.append(ioi)
            ratios.append(dur / ioi)

        if not ratios:
            return {
                'available': False,
                'reason': 'insufficient_valid_intervals',
                'average_duration': round(float(statistics.mean(durations)), 4) if durations else 0.0,
                'average_ioi': 0.0,
                'articulation_ratio_mean': 0.0,
                'articulation_ratio_std': 0.0,
                'staccato_percentage': 0.0,
                'legato_percentage': 0.0,
                'detached_percentage': 0.0,
            }

        staccato = sum(1 for r in ratios if r < 0.55)
        legato = sum(1 for r in ratios if r > 0.95)
        detached = len(ratios) - staccato - legato

        mean_ratio = float(statistics.mean(ratios))
        std_ratio = float(statistics.stdev(ratios)) if len(ratios) > 1 else 0.0

        articulation_style = "mixed"
        if legato / len(ratios) >= 0.6:
            articulation_style = "legato"
        elif staccato / len(ratios) >= 0.6:
            articulation_style = "staccato"

        return {
            'available': True,
            'average_duration': round(float(statistics.mean(durations)), 4) if durations else 0.0,
            'average_ioi': round(float(statistics.mean(iois)), 4) if iois else 0.0,
            'articulation_ratio_mean': round(mean_ratio, 4),
            'articulation_ratio_std': round(std_ratio, 4),
            'staccato_percentage': round((staccato / len(ratios)) * 100.0, 2),
            'legato_percentage': round((legato / len(ratios)) * 100.0, 2),
            'detached_percentage': round((detached / len(ratios)) * 100.0, 2),
            'articulation_style': articulation_style,
        }
    
