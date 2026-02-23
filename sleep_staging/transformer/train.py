import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer_window import TransformerSleepModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

CONFIG = {
    "train_list": "/path/to/your/train_list.txt",
    "val_list": "/path/to/your/val_list.txt",
    "save_path": "/path/to/save/model/checkpoint.pth",
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-4,
    "window_size": 15,
    "stride": 1,
    "seed": 42,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
    "early_stop_patience": 5,
    "num_classes": 5
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_paths_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.read().splitlines()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(loader)

def main():
    set_seed(CONFIG["seed"])

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)

    train_files = load_paths_from_txt(CONFIG["train_list"])
    val_files = load_paths_from_txt(CONFIG["val_list"])

    train_set = SleepWindowDataset(train_files, CONFIG["window_size"], CONFIG["stride"])
    val_set = SleepWindowDataset(val_files, CONFIG["window_size"], CONFIG["stride"])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    train_loader = DataLoader(
        train_set, 
        batch_size=CONFIG["batch_size"],
        sampler=train_sampler, 
        collate_fn=collate_fn_window,
        num_workers=4
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=CONFIG["batch_size"],
        sampler=val_sampler, 
        collate_fn=collate_fn_window,
        num_workers=4
    )

    input_dim = CONFIG["window_size"] * 1152
    model = TransformerSleepModel(
        input_dim=input_dim,
        hidden_dim=CONFIG["hidden_dim"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"]
    ).to(device)

    model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(CONFIG["epochs"]):
        train_sampler.set_epoch(epoch)
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            if local_rank == 0:
                torch.save(model.module.state_dict(), CONFIG["save_path"])
                print(f"✅ Best model saved at epoch {epoch+1} with val loss {val_loss:.4f}")
        else:
            patience_counter += 1
            if local_rank == 0:
                print(f"⚠️ No improvement. Patience: {patience_counter}/{CONFIG['early_stop_patience']}")

        should_stop = torch.tensor([0], device=device)
        if patience_counter >= CONFIG["early_stop_patience"]:
            should_stop.fill_(1)
        
        dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        
        if should_stop.item() == 1:
            if local_rank == 0:
                print(f"🛑 Early stopping triggered at epoch {epoch+1}!")
                print(f"Best model was at epoch {best_epoch} with val loss {best_loss:.4f}")
            dist.barrier()
            dist.destroy_process_group()
            return

        if local_rank == 0:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    if local_rank == 0:
        print(f"🏁 Training completed after {CONFIG['epochs']} epochs")
        print(f"Best model was at epoch {best_epoch} with val loss {best_loss:.4f}")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()