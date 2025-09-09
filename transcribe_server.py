import asyncio
import websockets
import json
from faster_whisper import WhisperModel

model_size = "large-v3"
model = WhisperModel(model_size, device="cpu", compute_type="int8")

async def transcribe_audio(websocket, path):
    async for message in websocket:
        # Expecting message to be a path to an audio file or raw audio bytes
        try:
            data = json.loads(message)
            audio_path = data.get("audio_path")
            if not audio_path:
                await websocket.send(json.dumps({"error": "Missing audio_path"}))
                continue
            segments, info = model.transcribe(audio_path)
            result = {
                "language": info.language,
                "language_probability": info.language_probability,
                "segments": [
                    {"start": seg.start, "end": seg.end, "text": seg.text}
                    for seg in segments
                ]
            }
            await websocket.send(json.dumps(result))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))

async def main():
    async with websockets.serve(transcribe_audio, "0.0.0.0", 8765):
        print("WebSocket server running on ws://0.0.0.0:8765")
        await asyncio.Future()  # run forever

asyncio.run(main())
