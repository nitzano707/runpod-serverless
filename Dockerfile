# ───────────────────────────────────────────────
# Base image עם תמיכה ב-CUDA
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# סביבת עבודה
WORKDIR /

# הגדרות CUDA
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# התקנת כלים חיוניים
RUN apt update && apt install -y ffmpeg git

# ───────────────────────────────────────────────
# התקנת חבילות פייתון
RUN pip install --no-cache-dir torch==2.4.1 runpod pyannote.audio==3.1.1

# ───────────────────────────────────────────────
# העתקת קובץ האינפרנס
COPY infer.py .

# ───────────────────────────────────────────────
# פקודת ההפעלה של השרת
CMD ["python", "-u", "infer.py"]
