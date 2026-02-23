# data_utils.py
# -*- coding: utf-8 -*-
import bisect
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import threading

__all__ = ["NPZDataset", "load_paths_from_file"]

def load_paths_from_file(filename: str):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip()]

class FileCache:
    def __init__(self, max_files=20):
        self.max_files = max_files
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        
    def get(self, file_path):
        with self.lock:
            if file_path in self.cache:
                self.cache.move_to_end(file_path)
                return self.cache[file_path]
            return None
    
    def put(self, file_path, data):
        with self.lock:
            if file_path in self.cache:
                self.cache.move_to_end(file_path)
                return
            
            if len(self.cache) >= self.max_files:
                oldest = next(iter(self.cache))
                removed_data = self.cache.pop(oldest)
                del removed_data
            
            self.cache[file_path] = data
    
    def clear(self):
        with self.lock:
            self.cache.clear()
    
    def __len__(self):
        return len(self.cache)


class NPZDataset(Dataset):
    def __init__(self, file_paths, max_cached_files=20, verbose=True):
        self.file_paths = list(file_paths)
        self.max_cached_files = max_cached_files
        self.verbose = verbose
        
        if self.verbose:
            print(f"Loading metadata for {len(self.file_paths)} files...")
        
        self.samples_per_file = []
        for idx, fp in enumerate(self.file_paths):
            try:
                with np.load(fp) as data:
                    if "x" not in data:
                        raise KeyError(f"{fp} miss key 'x'")
                    n_samples = int(data["x"].shape[0])
                    self.samples_per_file.append(n_samples)
                    
                if self.verbose and (idx + 1) % 500 == 0:
                    print(f"  Processed {idx + 1}/{len(self.file_paths)} files")
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                raise
        
        if not self.samples_per_file:
            raise RuntimeError("miss useful .npz files")
        
        self.cum = np.cumsum(self.samples_per_file).tolist()
        
        if self.verbose:
            print(f"Total samples: {self.cum[-1]}")
            print(f"Cache size: {max_cached_files} files per worker")
        
        self._file_cache = None
        self._worker_id = None

    def _init_worker_cache(self):
        if self._file_cache is None:
            self._file_cache = FileCache(max_files=self.max_cached_files)
            self._worker_id = self._get_worker_id()

    def _get_worker_id(self):
        try:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                return worker_info.id
        except:
            pass
        return 0

    def __len__(self):
        return self.cum[-1]

    def _locate(self, idx: int):
        file_idx = bisect.bisect_right(self.cum, idx)
        start = 0 if file_idx == 0 else self.cum[file_idx - 1]
        local_idx = idx - start
        return file_idx, local_idx

    def _load_file(self, file_idx: int):
        self._init_worker_cache()
        
        file_path = self.file_paths[file_idx]
        
        cached_data = self._file_cache.get(file_path)
        if cached_data is not None:
            return cached_data
        
        try:
            with np.load(file_path) as data:
                file_data = {
                    'x': np.array(data['x'], dtype=np.float32),
                    'arousal': np.array(data['arousal'], dtype=np.int64),
                    'osa': np.array(data['osa'], dtype=np.int64),
                    'hypopnea': np.array(data['hypopnea'], dtype=np.int64),
                }
                
                if 'freq_features' in data:
                    file_data['freq_features'] = np.array(
                        data['freq_features'], dtype=np.float32
                    )
            
            self._file_cache.put(file_path, file_data)
            
            return file_data
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise

    def __getitem__(self, idx: int):
        try:
            file_idx, local_idx = self._locate(idx)
            
            file_data = self._load_file(file_idx)

            x = file_data['x'][local_idx]
            signal = torch.from_numpy(x)
            
            if signal.ndim == 1:
                signal = signal.unsqueeze(0)
            elif signal.ndim == 2 and signal.shape[0] != 1:
                signal = signal[:1, :]
            
            if 'freq_features' in file_data:
                freq = file_data['freq_features'][local_idx]
                freq_tensor = torch.from_numpy(freq).view(-1)
            else:
                freq_tensor = torch.zeros(39, dtype=torch.float32)
            
            arousal = int(file_data['arousal'][local_idx])
            osa = int(file_data['osa'][local_idx])
            hypopnea = int(file_data['hypopnea'][local_idx])

            respiratory = 1 if (osa == 1 or hypopnea == 1) else 0
            
            labels = {
                "arousal": torch.tensor(arousal, dtype=torch.long),
                "respiratory": torch.tensor(respiratory, dtype=torch.long),
            }
            
            return signal, freq_tensor, labels
            
        except Exception as e:
            print(f"Error loading sample {idx} (file {file_idx}, local {local_idx}): {e}")
            raise

    def get_cache_stats(self):
        if self._file_cache is not None:
            return {
                'cached_files': len(self._file_cache),
                'max_files': self.max_cached_files,
                'worker_id': self._worker_id
            }
        return None

class ChunkedNPZDataset(Dataset):
    def __init__(self, file_paths, chunk_size=50, verbose=True):
        self.file_paths = list(file_paths)
        self.chunk_size = chunk_size
        self.verbose = verbose
        
        if self.verbose:
            print(f"Loading metadata for {len(self.file_paths)} files...")
        
        self.samples_per_file = []
        for idx, fp in enumerate(self.file_paths):
            try:
                with np.load(fp) as data:
                    n_samples = int(data["x"].shape[0])
                    self.samples_per_file.append(n_samples)
            except Exception as e:
                print(f"Error loading {fp}: {e}")
                raise
        
        self.cum = np.cumsum(self.samples_per_file).tolist()
        
        if self.verbose:
            print(f"Total samples: {self.cum[-1]}")
            print(f"Chunk size: {chunk_size} files")
        
        self._current_chunk_idx = -1
        self._chunk_data = {}

    def __len__(self):
        return self.cum[-1]

    def _locate(self, idx: int):
        file_idx = bisect.bisect_right(self.cum, idx)
        start = 0 if file_idx == 0 else self.cum[file_idx - 1]
        local_idx = idx - start
        return file_idx, local_idx

    def _load_chunk(self, file_idx: int):
        chunk_idx = file_idx // self.chunk_size
        
        if chunk_idx == self._current_chunk_idx:
            return
        
        self._chunk_data.clear()
        
        start_file = chunk_idx * self.chunk_size
        end_file = min((chunk_idx + 1) * self.chunk_size, len(self.file_paths))
        
        if self.verbose:
            print(f"Loading chunk {chunk_idx}: files {start_file}-{end_file}")
        
        for fi in range(start_file, end_file):
            try:
                with np.load(self.file_paths[fi]) as data:
                    self._chunk_data[fi] = {
                        'x': np.array(data['x'], dtype=np.float32),
                        'arousal': np.array(data['arousal']),
                        'osa': np.array(data['osa']),
                        'hypopnea': np.array(data['hypopnea']),
                    }
                    if 'freq_features' in data:
                        self._chunk_data[fi]['freq_features'] = np.array(
                            data['freq_features'], dtype=np.float32
                        )
            except Exception as e:
                print(f"Error loading file {self.file_paths[fi]}: {e}")
                raise
        
        self._current_chunk_idx = chunk_idx

    def __getitem__(self, idx: int):
        file_idx, local_idx = self._locate(idx)
        
        self._load_chunk(file_idx)
        
        file_data = self._chunk_data[file_idx]

        x = file_data['x'][local_idx]
        signal = torch.from_numpy(x)
        
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        elif signal.ndim == 2 and signal.shape[0] != 1:
            signal = signal[:1, :]
        
        if 'freq_features' in file_data:
            freq = file_data['freq_features'][local_idx]
            freq_tensor = torch.from_numpy(freq).view(-1)
        else:
            freq_tensor = torch.zeros(39, dtype=torch.float32)
        
        arousal = int(file_data['arousal'][local_idx])
        osa = int(file_data['osa'][local_idx])
        hypopnea = int(file_data['hypopnea'][local_idx])
        respiratory = 1 if (osa == 1 or hypopnea == 1) else 0
        
        labels = {
            "arousal": torch.tensor(arousal, dtype=torch.long),
            "respiratory": torch.tensor(respiratory, dtype=torch.long),
        }
        
        return signal, freq_tensor, labels
