from fastapi import FastAPI
import os

app = FastAPI()

# ----------------------------------------
# HEALTH CHECK
# ----------------------------------------
@app.get("/")
def home():
    return {"status": "API is running"}

@app.head("/process")
def process_head():
    return {"status": "ok"}


# ----------------------------------------
# API ENDPOINT
# ----------------------------------------
@app.get("/process")
def process(url: str):

    try:
        import torch
        import subprocess
        from pyannote.audio import Pipeline
        from faster_whisper import WhisperModel
        from huggingface_hub import login
        import imageio_ffmpeg as ffmpeg_tool

        # ----------------------------------------
        # Setup FFmpeg
        # ----------------------------------------
        ffmpeg_path = ffmpeg_tool.get_ffmpeg_exe()
        os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_path

        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

        if HF_TOKEN:
            login(token=HF_TOKEN)

        # ----------------------------------------
        # Load models (IMPORTANT: inside API)
        # ----------------------------------------
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(device))

        # 🔥 Use SMALLER MODEL (VERY IMPORTANT)
        asr_model = WhisperModel(
            "tiny",   # changed from "small" → avoids crashes
            device=device,
            compute_type="float16" if device == "cuda" else "int8"
        )

        # ----------------------------------------
        # Convert video → audio
        # ----------------------------------------
        audio_path = "temp.wav"

        subprocess.run([
            ffmpeg_path, "-y",
            "-i", url,
            "-ar", "16000",
            "-ac", "1",
            audio_path
        ], check=True)

        # ----------------------------------------
        # Run diarization
        # ----------------------------------------
        diarization = pipeline(audio_path)

        # ----------------------------------------
        # Run ASR
        # ----------------------------------------
        segments_asr, _ = asr_model.transcribe(audio_path)
        asr_segments = list(segments_asr)

        # ----------------------------------------
        # Format output
        # ----------------------------------------
        output = []

        for seg in asr_segments:
            output.append({
                "M": {
                    "start_time": {"N": str(int(seg.start))},
                    "end_time": {"N": str(int(seg.end))},
                    "text": {"S": seg.text.strip()}
                }
            })

        return output

    except Exception as e:
        return {
            "error": str(e),
            "message": "Processing failed"
        }