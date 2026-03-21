import pretty_midi
from typing import Dict, List, Any
import numpy as np

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
        # Group notes by time windows to find simultaneous notes
        chords = []
        time_window = 0.05  # 50ms window for chord detection
        
        # Implementation for chord detection
        return chords
    
    def _detect_phrases(self, notes: List[Dict]) -> List[Dict]:
        """Detect musical phrases based on rests and patterns."""
        # Implementation for phrase detection
        return []
    
    def _detect_sections(self, notes: List[Dict]) -> List[Dict]:
        """Detect musical sections based on patterns and changes."""
        return []
    
    def _calculate_note_density(self, notes: List[Dict]) -> List[Dict]:
        """Calculate note density over time."""
        return []
    
    def _analyze_timing_consistency(self, notes: List[Dict]) -> Dict:
        """Analyze timing consistency for performance evaluation."""
        return {}
    
    def _analyze_articulation(self, notes: List[Dict]) -> Dict:
        """Analyze articulation patterns (staccato, legato)."""
        return {}
    
