import whisper
import torch

from video_input.get_speaker import get_speaker
from video_input.extract_audio import extract_audio
from video_input.get_segments import get_segments
from video_input.seperate_background import seperate_background

model = whisper.load_model("large").to(torch.device("cuda"))

def get_text(segments):
    segmented_text=[]
    for segment in segments:
        segment["text"] = model.transcribe(str(segment["temp_path"]), language="en", task="transcribe")["text"]
        segmented_text.append(segment)
    return segmented_text




