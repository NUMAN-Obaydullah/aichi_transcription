from faster_whisper import WhisperModel

model_size = "medium.en"

# Run on GPU with FP16 for maximum performance
model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("/Users/numan/faster-whisper/tests/data/physicsworks.wav", language="en", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))