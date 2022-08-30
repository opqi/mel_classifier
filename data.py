import os
import pandas as pd

from torch.utils.data import DataLoader
from mel_dataset import Mel


def get_data(file_path: str, data_type: str = 'clean') -> list:
    file_paths = list()
    walk_path = os.path.join(file_path, data_type)
    for path, _, files in os.walk(walk_path):
        for name in files:
            file_paths.append(os.path.join(path, name))
    return file_paths


def get_df(path: str) -> pd.DataFrame:
    x_clean = get_data(path, 'clean')
    y_clean = [0]*len(x_clean)
    x_noisy = get_data(path, 'noisy')
    y_noisy = [1]*len(x_noisy)

    clean = dict({'path': x_clean, 'label': y_clean})
    noisy = dict({'path': x_noisy, 'label': y_noisy})

    df_clean = pd.DataFrame.from_dict(clean)
    df_noisy = pd.DataFrame.from_dict(noisy)

    return pd.concat([df_clean, df_noisy], ignore_index=True)


def data_loader(df: pd.DataFrame,
                duration: int,
                batch_size: int, channels: int) -> DataLoader:
    mel_ds = Mel(df, duration, channels)

    return DataLoader(mel_ds, batch_size, shuffle=True)
