import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from net1d import Net1DWithAttention
from tqdm import tqdm

class PersonDataset(Dataset):
    def __init__(self, file_path):
        data = np.load(file_path)
        self.signals = data['x']      # (n, 1, 3000)
        self.freqs = data['freq_features']  # (n, 1, 39)
        self.labels = data['y']       # (n,)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx]).float()
        freq = torch.from_numpy(self.freqs[idx]).float()
        label = int(self.labels[idx])
        return signal, freq, label

def extract_features(file_path, model, device):
    dataset = PersonDataset(file_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for x, freq, y in loader:
            x = x.to(device)
            freq = freq.to(device)

            # forward
            out = model.first_conv(x)
            out = model.first_bn(out)
            out = model.first_activation(out)
            for stage in model.stage_list:
                out = stage(out)
            out = out.mean(-1)  # shape: (B, 1024)
            out = model.time_attention(out)
            out = model.time_fc(out)  # shape: (B, 1024)

            freq_out = model.freq_fc(freq)  # shape: (B, 128)
            freq_out = model.freq_attention(freq_out)

            features = torch.cat([out, freq_out], dim=1)  # shape: (B, 1152)
            all_features.append(features.cpu())
            all_labels.append(y)

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_features, all_labels

def process_npz_files(npz_dirs, save_dir, model_path=None):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net1DWithAttention(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=True,
        use_do=True,
        n_classes=5,
    ).to(device)

    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {model_path}")
    model.eval()

    npz_files_processed = 0

    skipped_log_path = os.path.join(save_dir, "mesa_skipped_files.txt")
    with open(skipped_log_path, "w") as skipped_log:
        for npz_dir in npz_dirs:
            npz_files = [f for f in os.listdir(npz_dir) if f.endswith('.npz')]
            for f in tqdm(npz_files):
                full_path = os.path.join(npz_dir, f)
                features, labels = extract_features(full_path, model, device)

                if features.shape[0] > 1680:
                    msg = f"⚠️ Skipped: {f} | Too many segments: {features.shape[0]}"
                    print(msg)
                    skipped_log.write(f"{msg}\n")
                    continue

                save_path = os.path.join(save_dir, f.replace(".npz", "_features.npz"))
                np.savez_compressed(save_path, features=features, labels=labels)

                print(f"Saved: {save_path} | features.shape={features.shape}, labels.shape={labels.shape}")
                npz_files_processed += 1

    print(f"\nProcessed {npz_files_processed} files.")

if __name__ == "__main__":
    input_dirs = [
    "/path/to/your/npz/files"
    ]
    output_dir = "/path/to/save/extracted/features"
    os.makedirs(output_dir, exist_ok=True)
    model_ckpt = "/path/to/your/model_checkpoint.pth"
    process_npz_files(input_dirs, output_dir, model_ckpt)
