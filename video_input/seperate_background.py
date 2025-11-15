from demucs.pretrained import get_model
from demucs.apply import apply_model
import torchaudio
import torchaudio.transforms as T
import torch
import tempfile
import soundfile as sf
from librosa import resample

from video_input.extract_audio import extract_audio

model=get_model("htdemucs").to(torch.device("cuda")).eval()

def seperate_background(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.unsqueeze(0)

    with torch.no_grad():
        source_tensors=apply_model(model,waveform.to(torch.device("cuda")))[0]

    sources={name: tensor for name, tensor in zip(model.sources, source_tensors)}

    vocals_np=sources["vocals"]
    background_np = sum(tensor.to("cuda") for name, tensor in sources.items() if name != "vocals")

    resampler=T.Resample(sample_rate,16000).to(torch.device("cuda"))

    vocals_np=resampler(vocals_np).squeeze(0).cpu().numpy().mean(axis=0)
    background_np=resampler(background_np).squeeze(0).cpu().numpy().mean(axis=0)

    temp_vocals = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_background = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    sf.write(temp_vocals.name, vocals_np.T, 16000)
    sf.write(temp_background.name, background_np.T, 16000)

    temp_vocals.flush()
    temp_background.flush()


    return temp_vocals.name, temp_background.name