import dataclasses
import runpod
import ivrit
import types
import logging
import os

# הגדרת גודל מקסימלי לפריטים בזרם (לשמירה על מגבלת RunPod)
MAX_RUNPOD_STREAM_ELEMENT_SIZE = 500000

# משתנה גלובלי למודל הטעון כעת
current_model = None

# פונקציה ראשית של RunPod
def transcribe(job):
    engine = job['input'].get('engine', 'faster-whisper')
    model_name = job['input'].get('model', None)
    is_streaming = job['input'].get('streaming', False)

    if engine not in ['faster-whisper', 'stable-whisper']:
        yield {"error": f"engine must be 'faster-whisper' or 'stable-whisper', got {engine}"}
        return

    if not model_name:
        yield {"error": "Model not provided."}
        return

    transcribe_args = job['input'].get('transcribe_args', None)
    if not transcribe_args:
        yield {"error": "transcribe_args field not provided."}
        return
    if not ('blob' in transcribe_args or 'url' in transcribe_args):
        yield {"error": "transcribe_args must contain either 'blob' or 'url' field."}
        return

    # הפעלת הליבה
    stream_gen = transcribe_core(engine, model_name, transcribe_args)

    if is_streaming:
        for entry in stream_gen:
            yield entry
    else:
        result = [entry for entry in stream_gen]
        yield {"result": result}


def transcribe_core(engine, model_name, transcribe_args):
    print('Starting process...')
    global current_model

    # בדיקה אם המודל כבר טעון
    different_model = (
        not current_model
        or current_model.engine != engine
        or current_model.model != model_name
    )

    if different_model:
        print(f'Loading model: {engine} / {model_name}')
        current_model = ivrit.load_model(engine=engine, model=model_name, local_files_only=True)
    else:
        print(f'Reusing existing model: {engine} / {model_name}')

    diarize = transcribe_args.get('diarize', False)
    diarize_only = transcribe_args.get('diarize_only', False)

    # ───────────────────────────────────────────────
    # דיאריזציה בלבד (ללא תמלול)
    if diarize and diarize_only:
        print("Running diarization only (no transcription)...")
        from pyannote.audio import Pipeline

        hf_token = os.getenv("HF_TOKEN", None)
        if not hf_token:
            yield {"error": "Missing HF_TOKEN environment variable for PyAnnote."}
            return

        pipeline = Pipeline.from_pretrained(
            "ivrit-ai/pyannote-speaker-diarization-3.1",
            use_auth_token=hf_token
        )

        # הורדת הקובץ מה-URL ועיבוד
        audio_url = transcribe_args.get("url")
        diarization = pipeline(audio_url)

        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker
            })

        yield results
        return
    # ───────────────────────────────────────────────

    # אחרת — תמלול רגיל (עם או בלי דיאריזציה)
    print("Running transcription (with or without diarization)...")
    transcribe_args["stream"] = True
    segs = current_model.transcribe(**transcribe_args)

    if isinstance(segs, types.GeneratorType):
        for s in segs:
            yield [dataclasses.asdict(s)]
    else:
        current_group = []
        current_size = 0
        for s in segs:
            seg_dict = dataclasses.asdict(s)
            seg_size = len(str(seg_dict))
            if current_group and (current_size + seg_size > MAX_RUNPOD_STREAM_ELEMENT_SIZE):
                yield current_group
                current_group = []
                current_size = 0
            current_group.append(seg_dict)
            current_size += seg_size
        if current_group:
            yield current_group


runpod.serverless.start({
    "handler": transcribe,
    "return_aggregate_stream": True
})
