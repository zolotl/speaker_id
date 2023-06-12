import torch
import torchaudio
from denoiser import pretrained
from denoiser.dsp import convert_audio

model = pretrained.dns64().cuda()

def denoise_audio(audio_file):
    wav, sr = torchaudio.load('noise_test_AirportAnnouncements_1.wav')
    wav = convert_audio(wav.cuda(), sr, model.sample_rate, model.chin)
    with torch.no_grad():
        denoised = model(wav[None])[0].cpu()
    return denoised

