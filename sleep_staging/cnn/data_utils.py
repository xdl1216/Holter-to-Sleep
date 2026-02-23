'''
Distributed + Handle Leakage + Faster Transmission
'''
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from typing import List, Tuple, Optional, Union

def load_paths_from_file(filename: str) -> List[str]:
    with open(filename, 'r') as f:
        return [ln.strip() for ln in f if ln.strip()]

class NPZDataset(Dataset):
    def __init__(self, file_paths: List[str], max_cached_files: int = 4, enforce_channel_dim: bool = True, dtype=np.float32):
        self.file_paths = list(file_paths)
        self.max_cached_files = max_cached_files
        self.enforce_channel_dim = enforce_channel_dim
        self.dtype = dtype

        self._sizes = []
        for fp in self.file_paths:
            with np.load(fp, allow_pickle=False) as z:
                n = len(z['x'])
            self._sizes.append(n)

        # Global idx -> (file_idx, local_idx) Fast mapping
        self._offsets = np.cumsum([0] + self._sizes)

        # file_path -> (x, f, y)
        self._cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _map_index(self, idx: int) -> Tuple[int, int, str]:
        file_idx = int(np.searchsorted(self._offsets, idx, side='right') - 1)
        local_idx = int(idx - self._offsets[file_idx])
        return file_idx, local_idx, self.file_paths[file_idx]

    def _get_arrays_from_file(self, file_path: str):
        if file_path in self._cache:
            x, f, y = self._cache.pop(file_path)
            self._cache[file_path] = (x, f, y)
            return x, f, y

        with np.load(file_path, allow_pickle=False) as z:
            x = z['x']               # (N, L) or (N, 1, L)
            f = z['freq_features']   # (N, F)
            y = z['y']               # (N,)

        if self._cache and len(self._cache) >= self.max_cached_files:
            self._cache.popitem(last=False)
        self._cache[file_path] = (x, f, y)
        return x, f, y

    def __getitem__(self, idx: int):
        _, local_idx, file_path = self._map_index(idx)
        x_arr, f_arr, y_arr = self._get_arrays_from_file(file_path)

        signal = x_arr[local_idx]
        if self.enforce_channel_dim:
            if signal.ndim == 1:
                signal = signal[None, :]
            elif signal.ndim == 2 and signal.shape[0] != 1 and signal.shape[1] == 1:
                signal = signal.T
        signal = signal.astype(self.dtype, copy=False)

        freq = f_arr[local_idx].astype(self.dtype, copy=False)
        label = np.int64(y_arr[local_idx])

        signal = torch.from_numpy(signal)
        freq = torch.from_numpy(freq)
        label = torch.tensor(label, dtype=torch.long)
        return signal, freq, label

def worker_init_fn(worker_id: int):
    import torch, numpy as np, random, os
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)
    os.environ["PYTHONHASHSEED"] = str(seed + worker_id)

def get_dataloader(
    paths_file: str,
    batch_size: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    max_cached_files: int = 4,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
):
    paths = load_paths_from_file(paths_file)
    dataset = NPZDataset(paths, max_cached_files=max_cached_files)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=True)
        shuffle_flag = False
    else:
        sampler = None
        shuffle_flag = shuffle

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        sampler=sampler,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
    return loader, sampler
