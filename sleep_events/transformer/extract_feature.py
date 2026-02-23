# extract_feature.py
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from net1d import Net1D

class AttentionLayer(nn.Module):
    def __init__(self, in_features, reduction=4):
        super(AttentionLayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // reduction, in_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class MultiTaskNet1D_Fusion(Net1D):
    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes_list, use_bn=True, use_do=True, verbose=False):
        super(MultiTaskNet1D_Fusion, self).__init__(in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes_list[0], use_bn, use_do, verbose)
        
        if hasattr(self, 'dense'):
            delattr(self, 'dense')

        time_feature_dim = filter_list[-1]  # 1024

        self.freq_fc = nn.Sequential(
            nn.Linear(39, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.freq_attention = AttentionLayer(128)

        self.time_fc = nn.Sequential(
            nn.Linear(time_feature_dim, 1024),
            nn.ReLU()
        )
        self.time_attention = AttentionLayer(1024)

        fusion_dim = 1024 + 128
        self.head_arousal = nn.Linear(fusion_dim, n_classes_list[0])
        self.head_respiratory = nn.Linear(fusion_dim, n_classes_list[1])
        
    def forward(self, x, x_freq):
        pass

class PersonDataset(Dataset):
    def __init__(self, file_path):
        try:
            data = np.load(file_path)
            self.signals = data['x']      
            
            if 'freq_features' in data:
                self.freq_features = data['freq_features'] 
            else:
                print(f"Warning: No 'freq_features' in {os.path.basename(file_path)}, using zeros.")
                self.freq_features = np.zeros((len(self.signals), 39))

            self.arousal = data['arousal']
            self.osa = data['osa']
            self.hypopnea = data['hypopnea']
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            self.signals = np.array([])
            self.freq_features = np.array([])
            self.arousal = np.array([])
            self.osa = np.array([])
            self.hypopnea = np.array([])

    def __len__(self):
        return len(self.arousal)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0) 
            
        freq = self.freq_features[idx]
        freq_tensor = torch.from_numpy(freq).float().view(-1) 

        return signal, freq_tensor, int(self.arousal[idx]), int(self.osa[idx]), int(self.hypopnea[idx])

def extract_features(file_path, model, device):
    dataset = PersonDataset(file_path)
    if len(dataset) == 0:
        return np.empty((0, 1152)), np.array([]), np.array([])

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    model.eval()
    all_features = []
    arousal_labels = []
    respiratory_labels = []

    with torch.no_grad():
        for x, x_freq, arousal_label, osa_label, hypopnea_label in loader:
            x = x.to(device)
            x_freq = x_freq.to(device)

            # --- A. time domain ---
            out = model.first_conv(x)
            if model.use_bn:
                out = model.first_bn(out)
            out = model.first_activation(out)
            
            for stage in model.stage_list:
                out = stage(out)
            
            out_time = out.mean(-1)
            out_time = model.time_fc(out_time)        
            att_time = model.time_attention(out_time) 
            out_time = out_time * att_time            

            # --- B. fre-time-fre domain ---
            out_freq = model.freq_fc(x_freq)          
            att_freq = model.freq_attention(out_freq) 
            out_freq = out_freq * att_freq            

            # --- C. fusion ---
            fusion_features = torch.cat([out_time, out_freq], dim=1) # (B, 1152)

            all_features.append(fusion_features.cpu())
            arousal_labels.append(arousal_label)

            resp_batch = (osa_label | hypopnea_label).long() 
            respiratory_labels.append(resp_batch)

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0).numpy()
        all_arousal = torch.cat(arousal_labels, dim=0).numpy()
        all_respiratory = torch.cat(respiratory_labels, dim=0).numpy()
    else:
        all_features = np.empty((0, 1152))
        all_arousal = np.array([])
        all_respiratory = np.array([])

    return all_features, all_arousal, all_respiratory

def process_npz_files(npz_dirs, save_dir, model_path=None):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MultiTaskNet1D_Fusion(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5],
        kernel_size=16,
        stride=2,
        groups_width=16,
        n_classes_list=[2, 2], # Arousal, Respiratory
        verbose=False,
        use_bn=True,
        use_do=True
    ).to(device)

    print(f"Loading weights from {model_path} ...")
    state_dict = torch.load(model_path, map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    model.eval()

    skipped_log_path = os.path.join(save_dir, "skipped_files.txt")
    with open(skipped_log_path, "w") as skipped_log:
        for npz_dir in npz_dirs:
            if not os.path.exists(npz_dir):
                print(f"Skipping directory: {npz_dir}")
                continue
                
            npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
            print(f"Processing {len(npz_files)} files in {npz_dir} ...")
            
            for f in tqdm(npz_files):
                full_path = os.path.join(npz_dir, f)
                
                try:
                    features, arousal, respiratory = extract_features(full_path, model, device)
                except Exception as e:
                    err_msg = f"Error processing {f}: {e}"
                    print(err_msg)
                    skipped_log.write(f"{err_msg}\n")
                    continue

                if features.shape[0] == 0:
                    continue

                save_path = os.path.join(save_dir, f.replace(".npz", "_features.npz"))
                
                np.savez_compressed(
                    save_path, 
                    features=features,
                    arousal=arousal,
                    respiratory=respiratory
                )

if __name__ == "__main__":
    input_dirs = [
        "/path/to/your/npz/files",
    ]
    
    output_dir = "/path/to/save/extracted/features"
    # step1 best.pth
    model_ckpt = "/path/to/your/model/best.pth"
    process_npz_files(input_dirs, output_dir, model_ckpt)

