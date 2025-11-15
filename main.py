from video_input.seperate_background import seperate_background
from video_input.get_speaker import get_speaker
from video_input.extract_audio import extract_audio
from video_input.get_segments import get_segments
from video_input.get_text import get_text
from translation.pre_trained import translate
from dub.voice_clone import to_speech

from pydub import AudioSegment
import tempfile
import subprocess
import os
import glob



audio_path = extract_audio(
    "your video location",
    44100,
    2
)


vocals_path, background_path = seperate_background(audio_path)

diarization = get_speaker(vocals_path)
segments = get_segments(diarization, vocals_path)

longest_segment = max(segments, key=lambda x: x["end_time"] - x["start_time"])
start_ms = int(longest_segment["start_time"] * 1000)
end_ms = int(longest_segment["end_time"] * 1000)
sample_temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
full = AudioSegment.from_wav(vocals_path)
clean_clip = full[start_ms:end_ms]
clean_clip.export(sample_temp, format="wav")

segments = get_text(segments)
for segment in segments:
    segment["text"] = translate(segment["text"])

background = AudioSegment.from_wav(background_path)
dubbed_audio = background[:]

for segment in segments:
    if not segment["text"].strip():
        continue

    temp_wav_path = to_speech(segment["text"], sample_temp)
    tts_audio = AudioSegment.from_wav(temp_wav_path)
    start_ms = int(segment["start_time"] * 1000)

    dubbed_audio = dubbed_audio.overlay(tts_audio, position=start_ms)

final_audio_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
dubbed_audio.export(final_audio_path, format="wav")

subprocess.run([
    "ffmpeg", "-y",
    "-i", "your video location",
    "-i", final_audio_path,
    "-c:v", "copy",
    "-map", "0:v:0",
    "-map", "1:a:0",
    "desired output location"
], check=True)

for wav_file in glob.glob("/tmp/*.wav"):
    try:
        os.remove(wav_file)
    except Exception as e:
        print(f"[WARN] Failed to delete {wav_file}: {e}")
