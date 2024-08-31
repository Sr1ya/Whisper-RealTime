import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pyaudio
import numpy as np
from faster_whisper import WhisperModel

CHUNK_SIZE = 16000 * 5
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

model_size = "tiny.en"

def send_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE)

    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    try:
        while True:
            data = stream.read(CHUNK_SIZE)
            numpy_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            segments, info = model.transcribe(numpy_data, beam_size=5)

            for segment in segments:
                print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == '__main__':
    send_audio()
#Faking Bitch