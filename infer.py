import os
import runpod
from pyannote.audio import Pipeline

# ───────────────────────────────────────────────
# טעינת המודל פעם אחת בעת עליית השרת (Warm-up)
# ───────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("❌ Environment variable HF_TOKEN is missing!")

print("🔄 Loading diarization model (ivrit-ai/pyannote-speaker-diarization-3.1)...")
diarization_pipeline = Pipeline.from_pretrained(
    "ivrit-ai/pyannote-speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
print("✅ Diarization model loaded and ready!")

# ───────────────────────────────────────────────
# פונקציה ראשית לטיפול בבקשות
# ───────────────────────────────────────────────
def diarize_audio(job):
    """
    קלט צפוי:
    {
      "input": {
        "file_url": "https://example.com/audio.ogg"
      }
    }
    """
    file_url = job["input"].get("file_url")
    if not file_url:
        return {"error": "Missing 'file_url' in input"}

    print(f"🎧 Processing file: {file_url}")
    try:
        diarization = diarization_pipeline(file_url)
        segments = [
            {"start": float(s.start), "end": float(s.end), "speaker": s.label}
            for s in diarization.itertracks(yield_label=True)
        ]
        return {"segments": segments}
    except Exception as e:
        return {"error": str(e)}

# ───────────────────────────────────────────────
# הפעלת השרת ב-RunPod
# ───────────────────────────────────────────────
runpod.serverless.start({"handler": diarize_audio})
