import torch

from torch.utils.data import Dataset
from transforms import Utils


class Mel(Dataset):
    def __init__(self, df, duration: int, channels: int = 3):
        self.df = df
        self.duration = duration
        self.channels = channels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spec_file = self.df.path.loc[idx]

        label = self.df.label.loc[idx]

        spec = Utils.open(spec_file)
        spec = Utils.pad_truncate(spec, self.duration)
        aug_spec = Utils.augment(
            spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1)

        spec = torch.unsqueeze(aug_spec, dim=0).repeat(self.channels, 1, 1)
        return spec, label
