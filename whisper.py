import os
import pyaudio
import numpy as np
from faster_whisper import WhisperModel
import torch

# Define constants
NEON_GREEN = '\033[32m'
RESET_COLOR = '\033[0m'

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Function for transcribing audio data
def transcribe_audio(model, audio_data):
    # Convert audio data to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    # Perform speech recognition
    segments, _ = model.transcribe(audio_np, beam_size=7)
    transcription = ''.join(segment.text for segment in segments)
    return transcription

def main2():
    """
    The main function of the program.
    """

    # Select the Whisper model
    model = WhisperModel("medium", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the recording stream
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

    # Initialize an empty string to accumulate transcriptions
    accumulated_transcription = ""

    try:
        print("Listening... Press Ctrl+C to stop.")
        while True:
            # Read audio data from the stream
            audio_data = stream.read(1024)
            
            # Transcribe the audio data
            transcription = transcribe_audio(model, audio_data)
            if transcription:
                print(NEON_GREEN + transcription + RESET_COLOR)
                # Add the transcription to the accumulated transcription
                accumulated_transcription += transcription + " "

    except KeyboardInterrupt:
        print("Stopping...")

        # Write the accumulated transcription to a log file
        with open("log.txt", "w") as log_file:
            log_file.write(accumulated_transcription)

    finally:
        print("LOG" + accumulated_transcription)
        # Close the recording stream
        stream.stop_stream()
        stream.close()

        # Stop PyAudio
        p.terminate()

if __name__ == "__main__":
    main2()
