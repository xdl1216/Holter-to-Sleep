# dataset_sliding.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SleepWindowDataset(Dataset):
    def __init__(self, file_list, window_size=15, stride=1, cache_size=10):
        self.file_list = file_list
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size
        self.cache = {} 
        self.samples = self._build_samples()
        
    def _build_samples(self):
        samples = []
        for file in self.file_list:
            samples.append(file)
        return samples

    def _load_file(self, file_path):
        if file_path in self.cache:
            return self.cache[file_path]
        
        data = np.load(file_path)
        features = data["features"]
        arousal = data["arousal"]
        respiratory = data["respiratory"]
        
        windows = []
        for i in range(self.window_size // 2, len(features) - self.window_size // 2, self.stride):
            window_feats = features[i - self.window_size // 2 : i + self.window_size // 2 + 1]
            window = (torch.tensor(window_feats, dtype=torch.float32),
                      int(arousal[i]),
                      int(respiratory[i]))
            windows.append(window)
        
        if len(self.cache) < self.cache_size:
            self.cache[file_path] = windows
        
        return windows

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        windows = self._load_file(file_path)
        return windows

def collate_fn_window(batch):
    features, arousals, respiratorys = [], [], []

    for file_samples in batch:
        for sample in file_samples:
            # sample: (window_feats, arousal, respiratory)
            features.append(sample[0])
            arousals.append(sample[1])
            respiratorys.append(sample[2])

    return (
        torch.stack(features),          # (B, W, 1152)
        torch.tensor(arousals),        # (B,)
        torch.tensor(respiratorys)     # (B,)
    )