from pyannote.audio import Pipeline
from video_input.extract_audio import extract_audio
from collections import defaultdict
import os
import torch

token = os.getenv("HF_TOKEN", "")
if not token:
    raise RuntimeError("HF_TOKEN environment variable not set")

pipeline= Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",use_auth_token=token).to(torch.device("cuda"))

def get_speaker(audio_path):
    diarization = pipeline(audio_path)
    return diarization

