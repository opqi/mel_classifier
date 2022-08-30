import librosa  # for mel-spectrogram estimation
import soundfile  # for opening .flac audio
import numpy as np
import torch

from torch.autograd import Variable
from transforms import Utils


def audio_to_mel(aud_file: str) -> torch.tensor:
    audio, _ = soundfile.read(aud_file)

    print(audio)
    mel = librosa.feature.melspectrogram(audio,
                                         sr=16000,
                                         n_fft=1024,
                                         hop_length=256,
                                         fmin=20,
                                         fmax=8000, n_mels=80)

    return torch.Tensor(1 + np.log(1.e-12 + mel).T / 10.)


def open_mel(mel_file: str, channels: int, duration: int) -> torch.tensor:
    mel = Utils.open(mel_file)
    # mel = Utils.pad_truncate(mel, duration)
    mel = torch.unsqueeze(mel, dim=0).repeat(1, channels, 1, 1)
    mel = Variable(mel, requires_grad=False)

    return mel


def check_device(device: str) -> str:
    if device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('Cuda device not found, starting on cpu')
    return device
