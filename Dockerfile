# ───────────────────────────────────────────────
# בסיס: PyTorch עם CUDA, ריצה בלבד
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /

# ספריות GPU
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# התקנת ffmpeg לעיבוד קבצי אודיו
RUN apt update && apt install -y ffmpeg

# ───────────────────────────────────────────────
# התקנת הספריות הנדרשות בלבד
RUN pip3 install ivrit[diarization]==0.1.8 torch==2.4.1 huggingface-hub==0.36.0 runpod

# ───────────────────────────────────────────────
# טעינת מודלים דרושים בלבד מראש (Cache Warmup)
# טוען רק את מודל הדיאריזציה של Pyannote
RUN python3 -c 'import pyannote.audio; p = pyannote.audio.Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1")'

# אם אתה רוצה גם זיהוי דובר לפי קול (Voice Embeddings)
RUN python3 -c 'from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")'

# ───────────────────────────────────────────────
# העתקת קובץ הקוד שלך
ADD infer.py .

# ───────────────────────────────────────────────
# הרצת הקובץ הראשי
CMD ["python", "-u", "/infer.py"]
