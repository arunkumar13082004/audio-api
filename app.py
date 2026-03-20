import os
import json
import torch
import subprocess
from fastapi import FastAPI
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from huggingface_hub import login

import imageio_ffmpeg as ffmpeg_tool
import os

os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_tool.get_ffmpeg_exe()

app = FastAPI()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Login once
if HF_TOKEN:
    login(token=HF_TOKEN)

# Load models once (important for performance)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(torch.device(device))

asr_model = WhisperModel(
    "small",
    device=device,
    compute_type="float16" if device == "cuda" else "int8"
)

# ----------------------------------------
# Helper: Convert video → audio
# ----------------------------------------
def prepare_audio(url):
    audio_path = "temp_audio.wav"

    subprocess.run([
        "ffmpeg", "-y",
        "-i", url,
        "-ar", "16000",
        "-ac", "1",
        audio_path
    ])

    return audio_path

# ----------------------------------------
# API Endpoint
# ----------------------------------------
@app.get("/process")
def process(url: str):

    audio_path = prepare_audio(url)

    diarization = pipeline(audio_path)

    segments_asr, _ = asr_model.transcribe(audio_path)
    asr_segments = list(segments_asr)

    results = []

    for seg in asr_segments:
        start = seg.start
        end = seg.end
        text = seg.text.strip()

        speaker_label = "UNKNOWN"

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if (start >= turn.start and start <= turn.end) or \
               (end >= turn.start and end <= turn.end):
                speaker_label = speaker
                break

        results.append({
            "start": start,
            "end": end,
            "text": text
        })

    # Convert to DynamoDB JSON
    output = []

    for r in results:
        output.append({
            "M": {
                "start_time": {"N": str(int(r["start"]))},
                "end_time": {"N": str(int(r["end"]))},
                "text": {"S": r["text"]}
            }
        })

    return output