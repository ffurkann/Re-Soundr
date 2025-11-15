import tempfile
import subprocess
from pathlib import Path
from video_input.get_speaker import get_speaker
from video_input.extract_audio import extract_audio


def get_segments(diarization, audio_path):
    segments = []

    for segment, _, speaker in diarization.itertracks(yield_label=True):
        start_time = segment.start
        duration = segment.end - segment.start


        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()


        cmd = [
            "ffmpeg", "-y",
            "-i", str(audio_path),
            "-ss", str(start_time),
            "-t", str(duration),
            "-acodec", "copy",
            str(temp_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if segment.end - segment.start > 1:
            segments.append({
                "speaker": speaker,
                "start_time": start_time,
                "end_time": segment.end,
                "temp_path": temp_path
            })
        else:
            pass

    return segments


