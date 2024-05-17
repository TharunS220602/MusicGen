import streamlit as st
from model_script import load_tf_model, predict_tf_model, load_torch_model, generate_music
from midiutil import MIDIFile
import pretty_midi
from pydub import AudioSegment
import numpy as np
import os

# Define paths and configuration
TF_MODEL_PATH = 'models/sentiment_model.h5'
TORCH_MODEL_PATH = 'models/music_transformer_model.pth'
CONFIG_PATH = 'models/gpt2_config.json'  # Ensure your config file is properly placed in models directory
UPLOAD_FOLDER = 'uploads/'
MIDI_FOLDER = 'generated_midi/'
AUDIO_FOLDER = 'audio_files/'

# Ensure upload, MIDI, and audio folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MIDI_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Load models
st.title('Image to Music Generation')
st.write('Upload an image to generate music based on its sentiment')

# Upload image section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_image.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_image.getbuffer())

    # Display the uploaded image
    st.image(image_path, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying the image...")

    # Load the TensorFlow model and make prediction
    sentiment_model = load_tf_model(TF_MODEL_PATH)
    sentiment = predict_tf_model(sentiment_model, image_path)
    st.write(f"Predicted Sentiment: {sentiment}")

    # Load the GPT-2 model and generate music
    sentiment_dict = {'Happy': 0, 'Sad': 1, 'Neutral': 2}
    music_model = load_torch_model(TORCH_MODEL_PATH, CONFIG_PATH)

    st.write("Generating music based on sentiment...")
    generated_midi = generate_music(music_model, sentiment, sentiment_dict)
    st.write("Music generation completed!")

    # Define mapping for tempo and rhythm
    emotion_to_tempo = {
        'Happy': 140,
        'Sad': 60,
        'Neutral': 100
    }

    emotion_to_rhythm = {
        'Happy': 0.5,
        'Sad': 1.0,
        'Neutral': 0.75
    }
    tempo = emotion_to_tempo.get(sentiment, 100)
    rhythm = emotion_to_rhythm.get(sentiment, 1.0)

    # Create and save MIDI file
    def create_midi_file(midi_sequence, tempo, rhythm, output_file):
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        for i, note_number in enumerate(midi_sequence):
            if note_number != 0:
                start = i * rhythm
                end = start + rhythm
                note = pretty_midi.Note(
                    velocity=100, pitch=note_number, start=start, end=end
                )
                instrument.notes.append(note)

        midi.instruments.append(instrument)
        midi.write(output_file)

    output_midi_path = os.path.join(MIDI_FOLDER, 'generated_music.mid')
    create_midi_file(generated_midi, tempo, rhythm, output_midi_path)

    # Convert MIDI to WAV using pydub
    def midi_to_wav(midi_path, wav_path):
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        audio_data = midi_data.synthesize()
        audio = np.int16(audio_data / np.max(np.abs(audio_data)) * 32767)
        AudioSegment(
            audio.tobytes(),
            frame_rate=44100,
            sample_width=audio.dtype.itemsize,
            channels=1
        ).export(wav_path, format='wav')

    output_wav_path = os.path.join(AUDIO_FOLDER, 'generated_music.wav')
    midi_to_wav(output_midi_path, output_wav_path)

    # Convert MIDI to human-readable notes
    def midi_to_note_sequence(midi_sequence):
        note_to_midi = {
            'C': 60, 'C#': 61, 'Db': 61, 'D': 62, 'D#': 63, 'Eb': 63, 'E': 64, 'F': 65, 'F#': 66, 'Gb': 66,
            'G': 67, 'G#': 68, 'Ab': 68, 'A': 69, 'A#': 70, 'Bb': 70, 'B': 71,
            'Cm': 60, 'C#m': 61, 'Dm': 62, 'D#m': 63, 'Ebm': 63, 'Em': 64, 'Fm': 65, 'F#m': 66, 'Gm': 67,
            'G#m': 68, 'Am': 69, 'A#m': 70, 'Bbm': 70, 'Bm': 71  # Minor chords root notes
        }
        midi_to_note = {v: k for k, v in note_to_midi.items()}
        notes = [midi_to_note.get(midi, 'Unknown') for midi in midi_sequence]
        return ' '.join(notes)

    generated_notes = midi_to_note_sequence(generated_midi)
    st.write(f"Generated Notes: {generated_notes}")

    # Add download link for the generated MIDI file
    with open(output_midi_path, 'rb') as f:
        st.download_button(
            label="Download Generated MIDI",
            data=f,
            file_name='generated_music.mid',
            mime='audio/midi'
        )

    # Add download link for the generated WAV file
    with open(output_wav_path, 'rb') as f:
        st.download_button(
            label="Download Generated WAV",
            data=f,
            file_name='generated_music.wav',
            mime='audio/wav'
        )