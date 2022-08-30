import torch
import numpy as np

from torchaudio import transforms as T


class Utils():
    # ----------------------------
    # Open file
    # ----------------------------
    @staticmethod
    def open(file: str) -> torch.tensor:
        data = torch.tensor(np.load(file).astype('float32').T)
        return data

# ----------------------------
# Pad/Truncate - resize the mel spectrogram
# ----------------------------
    @staticmethod
    def pad_truncate(spec: torch.tensor, max_len: int) -> torch.tensor:
        num_rows, sig_len = spec.shape

        if sig_len > max_len:
            rand = np.random.randint(0, sig_len - max_len)
            spec = spec[:, rand:rand + max_len]
        elif sig_len < max_len:
            pad_begin_len = np.random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            spec = torch.cat((pad_begin, spec, pad_end), 1)
        return spec

# ----------------------------
# Augment the Spectrogram by masking out some sections of it in both the frequency
# dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
# overfitting and to help the model generalise better. The masked sections are
# replaced with the mean value.
# ----------------------------
    @staticmethod
    def augment(spec: torch.tensor, max_mask_pct: float = 0.1,
                n_freq_masks: int = 1, n_time_masks: int = 1) -> torch.tensor:
        n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            masking = T.FrequencyMasking(freq_mask_param, mask_value)
            aug_spec = masking(spec)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            masking = T.TimeMasking(time_mask_param, mask_value)
            aug_spec = masking(spec)

        return aug_spec
