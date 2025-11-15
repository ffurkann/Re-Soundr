import subprocess
import tempfile
import os

def extract_audio(path, sample_rate, channels):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    result = subprocess.run([
        "ffmpeg",
        "-y",
        "-threads", "0",
        "-i", path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        temp_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    if result.returncode != 0 or os.stat(temp_path).st_size == 0:
        raise RuntimeError(f"FFmpeg failed:\n{result.stderr.decode()}")

    return temp_path


