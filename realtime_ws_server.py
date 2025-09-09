import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import queue
import threading
import os, re, time
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sample_rate = 16000
block_duration = 0.5  # seconds
chunk_duration = 2
channels = 1
frame_per_block = int(sample_rate * block_duration)
frame_per_chunk = int(sample_rate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
transcript_parts = []
model_load_time_sec = None
total_decode_time_sec = 0.0
total_audio_sec = 0.0
num_chunks = 0
last_info = None
REFERENCE_TEXT = os.getenv("REFERENCE_TEXT", "")

_t0 = time.perf_counter()
model = WhisperModel("base", device="cpu", compute_type="int8")
model_load_time_sec = time.perf_counter() - _t0

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def start_recorder():
    sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback, blocksize=frame_per_block).start()

async def transcriber(websocket):
    global audio_buffer, transcript_parts, total_decode_time_sec, total_audio_sec, num_chunks, last_info
    while True:
        try:
            block = audio_queue.get(timeout=1)
            audio_buffer.append(block)
            total_frames = sum(len(b) for b in audio_buffer)
            if total_frames >= frame_per_chunk:
                audio_data = np.concatenate(audio_buffer)[:frame_per_chunk]
                audio_buffer = []
                audio_data = audio_data.flatten().astype(np.float32)
                _dec_t0 = time.perf_counter()
                segments, info = model.transcribe(
                    audio_data, beam_size=1, language=None, word_timestamps=True
                )
                _dec_dt = time.perf_counter() - _dec_t0
                total_decode_time_sec += _dec_dt
                total_audio_sec += len(audio_data) / float(sample_rate)
                num_chunks += 1
                last_info = info
                chunk_texts = []
                for segment in segments:
                    chunk_texts.append(segment.text.strip())
                if chunk_texts:
                    transcript_parts.append(" ".join(chunk_texts))
                    current_text = " ".join(transcript_parts).strip()
                    await websocket.send(json.dumps({"transcript": current_text}))
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue

async def handler(websocket, path):
    threading.Thread(target=start_recorder, daemon=True).start()
    await transcriber(websocket)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765):
        print("WebSocket server running on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
