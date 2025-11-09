# ───────────────────────────────────────────────
# בסיס: PyTorch עם CUDA — גרסת runtime קלה
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# הגדרת סביבת העבודה
WORKDIR /

# הגדרת ספריות CUDA כדי למנוע בעיות טעינה
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# ───────────────────────────────────────────────
# התקנת ffmpeg לעיבוד קבצי אודיו
RUN apt update && apt install -y ffmpeg && apt clean && rm -rf /var/lib/apt/lists/*

# ───────────────────────────────────────────────
# התקנת ספריות Python הנדרשות בלבד לדיאריזציה
RUN pip3 install --no-cache-dir \
    ivrit[diarization]==0.1.8 \
    torch==2.4.1 \
    huggingface-hub==0.36.0 \
    runpod

# ───────────────────────────────────────────────
# טעינה מוקדמת של מודל הדיאריזציה (cache warm-up)
RUN python3 -c 'import pyannote.audio; p = pyannote.audio.Pipeline.from_pretrained("ivrit-ai/pyannote-speaker-diarization-3.1")'

# אם אתה רוצה גם זיהוי קולות (Voice Embeddings) – השאר את השורה הבאה:
# RUN python3 -c 'from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")'

# ───────────────────────────────────────────────
# העתקת קובץ ההרצה שלך
ADD infer.py .

# ───────────────────────────────────────────────
# הפקודה שתופעל בעת הרצת הקונטיינר
CMD ["python", "-u", "/infer.py"]
