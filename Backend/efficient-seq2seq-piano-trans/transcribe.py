from piano_transcription_inference import PianoTranscription, sample_rate, load_audio

# Load audio
(audio, _) = load_audio('audio/Passionfruit_Katherine_Cordova.mp3', sr=sample_rate, mono=True)

# Transcriptor
transcriptor = PianoTranscription(device='cuda')  # Use 'cpu' if no GPU

# Transcribe and write out to MIDI file
transcribed_dict = transcriptor.transcribe(audio, 'KongPassionfruit.mid')

