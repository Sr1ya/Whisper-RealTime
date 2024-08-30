import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import torch

# Load the Whisper model
model = WhisperModel("distil-medium.en", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

def transcribe_audio_data(model, audio_data):
    # Convert audio data to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize audio
    # Perform speech recognition
    segments, _ = model.transcribe(audio_np)
    transcription = ''.join(segment.text for segment in segments)
    return transcription

def speech_to_text_live(model, output_text_file):
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the audio stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)  # Smaller buffer for less latency
    stream.start_stream()

    print("Listening... (press Ctrl+C to stop)")

    try:
        with open(output_text_file, "w", encoding="utf-8") as text_file:
            while True:
                data = stream.read(1024, exception_on_overflow=False)  # Read smaller chunks
                transcription = transcribe_audio_data(model, data)
                if transcription:
                    print(transcription)
                    text_file.write(transcription + "\n")

    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()

# Provide the output text file
output_text_file = "text.txt"

# Start the speech-to-text process
speech_to_text_live(model, output_text_file)
