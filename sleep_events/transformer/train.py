# train.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer_window import TransformerSleepModel
from torch.cuda.amp import autocast, GradScaler

CONFIG = {
    "train_list": "/path/to/train_paths.txt",
    "val_list": "/path/to/val_paths.txt",
    "save_path": "./checkpoints/transformer_sleep_model.pth",
    "batch_size": 16,
    "epochs": 200,
    "lr": 3e-4,
    "window_size": 15,
    "stride": 1,
    "seed": 42,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "early_stop_patience": 6, 
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

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return local_rank, True
    else:
        print("Not using distributed mode (Single GPU/CPU mode).")
        return 0, False

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, is_distributed):
    model.train()
    local_loss_sum = 0.0
    
    is_main = (not is_distributed) or (dist.get_rank() == 0)
    iterator = tqdm(loader, desc="Training") if is_main else loader

    for x, arousals, respiratorys in iterator:
        x = x.to(device, non_blocking=True)
        arousals = arousals.to(device, non_blocking=True)
        respiratorys = respiratorys.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            arousal_logits, respiratory_logits = model(x)
            
            arousal_loss = criterion(arousal_logits, arousals)
            respiratory_loss = criterion(respiratory_logits, respiratorys)
            
            loss = arousal_loss + respiratory_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        local_loss_sum += loss.item()
    
    local_avg_loss = local_loss_sum / len(loader)
    
    if is_distributed:
        loss_tensor = torch.tensor([local_avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / dist.get_world_size()
        return global_avg_loss
    else:
        return local_avg_loss

def validate(model, loader, criterion, device, is_distributed):
    model.eval()
    local_loss_sum = 0.0
    
    is_main = (not is_distributed) or (dist.get_rank() == 0)
    iterator = tqdm(loader, desc="Validating") if is_main else loader

    with torch.no_grad():
        for x, arousals, respiratorys in iterator:
            x = x.to(device, non_blocking=True)
            arousals = arousals.to(device, non_blocking=True)
            respiratorys = respiratorys.to(device, non_blocking=True)

            with autocast():
                arousal_logits, respiratory_logits = model(x)
                
                arousal_loss = criterion(arousal_logits, arousals)
                respiratory_loss = criterion(respiratory_logits, respiratorys)
                
                loss = arousal_loss + respiratory_loss

            local_loss_sum += loss.item()

    local_avg_loss = local_loss_sum / len(loader)

    if is_distributed:
        loss_tensor = torch.tensor([local_avg_loss], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / dist.get_world_size()
        return global_avg_loss
    else:
        return local_avg_loss

def main():
    local_rank, is_distributed = init_distributed_mode()
    set_seed(CONFIG["seed"] + local_rank) 
    device = torch.device("cuda", local_rank)

    train_files = load_paths_from_txt(CONFIG["train_list"])
    val_files = load_paths_from_txt(CONFIG["val_list"])

    train_set = SleepWindowDataset(train_files, CONFIG["window_size"], CONFIG["stride"])
    val_set = SleepWindowDataset(val_files, CONFIG["window_size"], CONFIG["stride"])

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_set, batch_size=CONFIG["batch_size"], 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=0, collate_fn=collate_fn_window, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG["batch_size"], 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=0, collate_fn=collate_fn_window, pin_memory=False
    )

    model = TransformerSleepModel(
        input_dim=1152,
        hidden_dim=CONFIG["hidden_dim"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    )
    model = model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(CONFIG["epochs"]):
        if is_distributed:
            train_sampler.set_epoch(epoch)

        if local_rank == 0:
            print(f"\n--- Epoch {epoch + 1}/{CONFIG['epochs']} ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, is_distributed)
        val_loss = validate(model, val_loader, criterion, device, is_distributed)

        if local_rank == 0:
            print(f"Train Loss (Global): {train_loss:.4f} | Val Loss (Global): {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            if local_rank == 0:
                os.makedirs(os.path.dirname(CONFIG["save_path"]), exist_ok=True)
                state_dict = model.module.state_dict() if is_distributed else model.state_dict()
                torch.save(state_dict, CONFIG["save_path"])
                print("✅ Global Best model saved!")
        else:
            patience_counter += 1
            if local_rank == 0:
                print(f"⚠️ No improvement. Patience: {patience_counter}/{CONFIG['early_stop_patience']}")

        if patience_counter >= CONFIG["early_stop_patience"]:
            if local_rank == 0:
                print("🛑 Early stopping triggered (Global criteria).")
            break

    if is_distributed:
        dist.destroy_process_group()
    
    if local_rank == 0:
        print("Training completed.")

if __name__ == "__main__":
    main()

