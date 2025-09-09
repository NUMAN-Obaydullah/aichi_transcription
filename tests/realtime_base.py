# Real-time transcription using base model (multilingual)
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
# Add env, timing, metrics, saving
import os, re, time, json
from dotenv import load_dotenv

# Load env from root and tests/.env
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Ground-truth and GPT comparison
REFERENCE_TEXT = "at the doctor's conversation good morning Mr Johnson how are you feeling today good morning doctor I've been experiencing some pain in my chest and shortness of breath it's been bothering me for a few weeks now I'm glad you came in it's important not to ignore any chest related symptoms let me ask you a few questions to better understand your condition have you noticed if these symptoms occur during any specific activities or times of the day yes I've noticed that it happens mostly when I push myself physically like when I'm climbing stairs or walking fast all right do you have any history of heart disease in your family not that I'm aware of my parents and siblings don't have any heart related issues based on your symptoms and their relation to physical activity it's important to consider your heart health I'd like to order an electrocardiogram also called ECG to get a baseline assessment of your heart's electrical activity it will help us determine if any abnormalities are present I trust your judgment doctor please go ahead with the necessary tests I want to make sure everything is okay thank you for your trust Mr Johnson in the meantime I recommend making some lifestyle changes to support your heart health have you been following a balanced diet and engaging in regular exercise I must admit my diet hasn't been the healthiest lately and my exercise routine is almost non-existent due to work it's never too late to make positive changes Mr Johnson a heart-healthy diet with fruits vegetables whole grains and lean proteins along with regular physical activity can significantly improve your heart's Condition it's also essential to manage stress and get enough sleep that would be great doctor I'm ready to commit to a healthier lifestyle are there any medications I should be taking or any other treatments we should consider we'll wait for the ECG results before discussing specific medications as they will depend on the findings if necessary we may consider prescribing medications to manage your symptoms and support your heart health in the meantime I encourage you to take it easy and avoid any heavy activities I appreciate your guidance doctor I'll be cautious and prioritize my health is there anything else I should be aware of or any warning signs I should watch out for some warning signs to watch out for include severe chest pain or pressure pain radiating to your arms or jaw and sudden shortness of breath even at rest if you experience any of these symptoms it's crucial to seek immediate medical attention I'll keep a close eye on those symptoms and won't hesitate to reach out for help thank you for your time and expertise you're welcome Mr Johnson remember we're here to support you every step of the way let's focus on getting your test scheduled and making those positive Lifestyle Changes we'll discuss the results and next steps once we have more information take care and feel better thank you doctor I appreciate your care and concern I'll be in touch and follow your advice goodbye Goodbye Mr Johnson take care and stay well too hello again Mr Johnson I have the results of your ECG and I'm happy to say that your heart rhythm appears normal that's a relief to hear doctor so does that mean my symptoms are not related to any heart issues while the ECG results are reassuring we still need to investigate further to determine the exact cause of your symptoms other factors like lung function could be contributing I suggest a few more tests to get a clearer picture I understand doctor what tests do you recommend I'd like to conduct a pulmonary function test to assess your lung function and a stress test to evaluate your heart's response to physical exertion these tests will help us gather more information all right doctor all proceed with the tests I appreciate your thoroughness in finding the cause it's important for us to have a full understanding of your health Mr Johnson these tests will help us rule out any underlying conditions and guide us toward inaccurate diagnosis in the meantime I encourage you to continue with the lifestyle modifications we discussed earlier absolutely doctor I've been making conscious efforts to improve my diet and incorporate exercise into my routine it's challenging but I understand the importance of prioritizing my health that's great to hear Mr Johnson lifestyle changes can have a significant impact on overall well-being please remember to listen to your body and vote any activities that exacerbate your symptoms I'll definitely keep that in mind doctor thank you for your guidance and support throughout this process it's reassuring to know that I'm in good hands you're very welcome Mr Johnson it's my duty to provide you with the best care possible once we have the results of the additional tests we can schedule a follow-up appointment to discuss the findings and determine the next course of action I appreciate your prompt attention doctor I'll eagerly await the results and look forward to our next meeting thank you again for your professionalism and care it's my pleasure Mr Johnson remember if you have any concerns or experience any significant changes in your symptoms don't hesitate to reach out to me or the clinic we're here to support you take care and we'll be in touch soon thank you Doctor take care as well and I'll be sure to keep you updated goodbye for now Goodbye Mr Johnson wishing you the best of health"
USE_GPT_COMMENT = True
GPT_MODEL = "gpt-4o-mini"
_MIN_CHARS_FOR_GPT = 120

# Save final JSON next to this script
SAVE_FINAL_JSON = True
FINAL_JSON_PATH = os.path.join(
    os.path.dirname(__file__),
    os.path.splitext(os.path.basename(__file__))[0] + "_results.json",
)

# Simple normalization & metrics

def _normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]

def _wer(ref, hyp):
    rt = _normalize_text(ref).split()
    ht = _normalize_text(hyp).split()
    if not rt:
        return 0.0 if not ht else 1.0
    return _levenshtein(rt, ht) / len(rt)

def _cer(ref, hyp):
    rc = list(_normalize_text(ref))
    hc = list(_normalize_text(hyp))
    if not rc:
        return 0.0 if not hc else 1.0
    return _levenshtein(rc, hc) / len(rc)

# Sentence utilities for auto-exit

def _split_sentences(text: str):
    parts = re.split(r"[.?!]+", text)
    return [p.strip() for p in parts if p.strip()]

def _should_auto_exit(reference_text: str, hypothesis_text: str) -> bool:
    if not reference_text or not hypothesis_text:
        return False
    ref_s = _split_sentences(reference_text.lower())
    hyp_s = _split_sentences(hypothesis_text.lower())
    if len(ref_s) < 2 or len(hyp_s) < 2:
        return False
    return hyp_s[-2:] == ref_s[-2:]

sample_rate = 16000
block_duration = 0.5  # seconds
chunk_duration = 2
channels = 1
frame_per_block = int(sample_rate * block_duration)
frame_per_chunk = int(sample_rate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
# Accumulated hypothesis text
transcript_parts = []

# Tracking metrics
model_load_time_sec = None
total_decode_time_sec = 0.0
total_audio_sec = 0.0
num_chunks = 0
last_info = None
# Auto exit flag
auto_exit_triggered = False

# Time model load
_t0 = time.perf_counter()
model = WhisperModel("base", device="cpu", compute_type="int8")
model_load_time_sec = time.perf_counter() - _t0

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback, blocksize=frame_per_block):
        print("Recording... Press Ctrl+C to stop.")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer, transcript_parts, total_decode_time_sec, total_audio_sec, num_chunks, last_info, auto_exit_triggered
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
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                    chunk_texts.append(segment.text.strip())
                if chunk_texts:
                    transcript_parts.append(" ".join(chunk_texts))
                    current_text = " ".join(transcript_parts).strip()
                    print(f"[Transcript] {current_text}")

                if REFERENCE_TEXT:
                    hyp_text = " ".join(transcript_parts).strip()
                    w = _wer(REFERENCE_TEXT, hyp_text)
                    c = _cer(REFERENCE_TEXT, hyp_text)
                    print(f"[Metrics] WER={w:.3f}, CER={c:.3f}")
                    # GPT evaluation now runs only once at the end in _finalize_and_save to reduce cost.

                # Auto-exit when last two sentences match reference
                if REFERENCE_TEXT and not auto_exit_triggered:
                    full_hyp = " ".join(transcript_parts).strip()
                    if _should_auto_exit(REFERENCE_TEXT, full_hyp):
                        print("[AutoExit] Last two sentences match reference. Finalizing and exiting...")
                        auto_exit_triggered = True
                        _finalize_and_save()
                        raise SystemExit

                print(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        except queue.Empty:
            continue

# Finalize and save

def _finalize_and_save():
    hyp_text = " ".join(transcript_parts).strip()
    result = {
        "model": "base",
        "compute_type": "int8",
        "sample_rate": sample_rate,
        "chunk_duration_sec": chunk_duration,
        "total_audio_sec": round(total_audio_sec, 3),
        "num_chunks": num_chunks,
        "model_load_time_sec": round(model_load_time_sec or 0.0, 3),
        "total_decode_time_sec": round(total_decode_time_sec, 3),
        "real_time_factor": round((total_decode_time_sec / total_audio_sec), 3) if total_audio_sec > 0 else None,
        "language": getattr(last_info, "language", None) if last_info else None,
        "language_probability": getattr(last_info, "language_probability", None) if last_info else None,
        "reference_chars": len(REFERENCE_TEXT or ""),
        "hypothesis_chars": len(hyp_text),
        "wer": _wer(REFERENCE_TEXT, hyp_text) if REFERENCE_TEXT else None,
        "cer": _cer(REFERENCE_TEXT, hyp_text) if REFERENCE_TEXT else None,
        "hypothesis": hyp_text,
    }

    print("\n=== Final Summary ===")
    print(f"Chunks: {result['num_chunks']} | Audio: {result['total_audio_sec']}s | Load: {result['model_load_time_sec']}s | Decode: {result['total_decode_time_sec']}s | RTF: {result['real_time_factor']}")
    if REFERENCE_TEXT:
        print(f"WER: {result['wer']:.3f} | CER: {result['cer']:.3f}")

    if USE_GPT_COMMENT and REFERENCE_TEXT and os.getenv("OPENAI_API_KEY") and result["hypothesis_chars"] >= _MIN_CHARS_FOR_GPT:
        try:
            from openai import OpenAI
            client = OpenAI()
            prompt = (
                "Give a brief one-line evaluation comparing the full ASR transcript to the reference. "
                "Mention accuracy and key errors.\n\n"
                f"Reference:\n{REFERENCE_TEXT}\n\nHypothesis:\n{hyp_text}\n"
            )
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=80,
            )
            msg = resp.choices[0].message.content.strip()
            if msg:
                result["gpt_comment_final"] = msg
                print(f"[GPT Final] {msg}")
        except Exception as e:
            print(f"[GPT Final] Skipped ({e.__class__.__name__})")

    if SAVE_FINAL_JSON:
        try:
            with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Saved results to {FINAL_JSON_PATH}")
        except Exception as e:
            print(f"Failed to save results: {e}")

threading.Thread(target=recorder, daemon=True).start()
try:
    transcriber()
except KeyboardInterrupt:
    _finalize_and_save()
    print("Stopped.")
except SystemExit:
    pass
