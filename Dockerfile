# Base image עם תמיכה ב-CUDA
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# הגדרת סביבת עבודה
WORKDIR /

# ספריות CUDA
ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:/opt/conda/lib/python3.11/site-packages/nvidia/cublas/lib"

# התקנות בסיסיות
RUN apt update && apt install -y ffmpeg git

# התקנת חבילות פייתון
RUN pip install --no-cache-dir torch==2.4.1 pyannote.audio==3.1.1 runpod

# העתקת קובץ הקוד
COPY infer.py .

# הרצת הקובץ
CMD ["python", "-u", "infer.py"]
