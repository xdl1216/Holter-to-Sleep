import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

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
        labels = data["labels"]
        
        windows = []
        for i in range(self.window_size // 2, len(features) - self.window_size // 2, self.stride):
            window_feats = features[i - self.window_size // 2 : i + self.window_size // 2 + 1]
            window_feats = window_feats.reshape(-1, window_feats.shape[-1])
            window = (torch.tensor(window_feats, dtype=torch.float32), 
                      torch.tensor(labels[i], dtype=torch.long))
            windows.append(window)

        if len(self.cache) < self.cache_size:
            self.cache[file_path] = windows
        
        return windows

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        windows = self._load_file(file_path)
        features = [window[0] for window in windows]  # Extract feature tensors
        labels = [window[1] for window in windows]  # Extract label tensors
        return torch.stack(features), torch.stack(labels)  # Return as tensors

def collate_fn_window(batch):
    features, labels = zip(*batch)  # Unpacking into 2 values

    lengths = [f.shape[0] for f in features]  # Get lengths of features
    max_len = max(lengths)

    padded_features = pad_sequence(features, batch_first=True)  # Pad sequences
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)  # Pad labels

    return padded_features, padded_labels

