import tempfile
import torch
from TTS.api import TTS
import torch.serialization
from torch.serialization import load as original_load

def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)

torch.serialization.load = patched_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig,
])

MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
tts = TTS(model_name=MODEL, progress_bar=False)
tts.to(torch.device("cuda"))

def to_speech(text: str, speaker_wav: str) -> str:
    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="tr",
        file_path=temp.name
    )
    return temp.name
