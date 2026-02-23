# dataset_sliding.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SleepWindowDataset(Dataset):
    def __init__(self, file_list, window_size=15, stride=1):
        self.samples = []
        half_w = window_size // 2
        for file in file_list:
            data = np.load(file)
            features = data["features"]
            arousal = data["arousal"]
            osa = data["osa"]
            hypopnea = data["hypopnea"]

            length = len(features)
            for i in range(half_w, length - half_w, stride):
                window_feats = features[i - half_w : i + half_w + 1]
                self.samples.append((
                    torch.tensor(window_feats, dtype=torch.float32),
                    int(arousal[i]),
                    int(osa[i]),
                    int(hypopnea[i])
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn_window(batch):
    features, arousals, osas, hypopneas = zip(*batch)
    return (
        torch.stack(features),          # (B, W, 1024)
        torch.tensor(arousals),        # (B,)
        torch.tensor(osas),            # (B,)
        torch.tensor(hypopneas)        # (B,)
    )
