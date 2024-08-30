import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import pyaudio
from faster_whisper import WhisperModel
import torch

# Load the Whisper model
model = WhisperModel("distil-medium.en", device="cuda" if torch.cuda.is_available() else "cpu")

def transcribe_audio_data(model, audio_data):
    # Convert audio data to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    # Perform speech recognition
    segments, _ = model.transcribe(audio_np, beam_size=7)
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
                    frames_per_buffer=1024)
    stream.start_stream()

    # Initialize an empty string to accumulate transcriptions
    accumulated_transcription = ""

    print("Listening... (press Ctrl+C to stop)")

    try:
        with open(output_text_file, "w", encoding="utf-8") as text_file:
            buffer = b""
            while True:
                data = stream.read(1024, exception_on_overflow=False)
                buffer += data
                
                # Process buffer when it reaches a certain size (e.g., 4 seconds worth of data)
                if len(buffer) >= 16000 * 4 * 2:  # 4 seconds of audio (16kHz sample rate * 4 seconds * 2 bytes per sample)
                    transcription = transcribe_audio_data(model, buffer)
                    if transcription:
                        print(transcription)
                        text_file.write(transcription + "\n")
                        accumulated_transcription += transcription + " "
                    buffer = b""  # Clear buffer after processing

    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        # Write the accumulated transcription to the log file
        with open(output_text_file, "a", encoding="utf-8") as text_file:
            text_file.write(accumulated_transcription + "\n")

        # Clean up
        stream.stop_stream()
        stream.close()
        p.terminate()

# Provide the output text file
output_text_file = "text.txt"

# Start the speech-to-text process
speech_to_text_live(model, output_text_file)
