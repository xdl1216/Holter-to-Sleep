'''
DDP + AMP + tqdm for each batch
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import warnings
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from net1d import Net1DWithAttention
from data_utils import get_dataloader

warnings.filterwarnings('ignore')

def init_distributed_mode(args):
    args.distributed = False
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.rank, args.world_size, args.gpu = 0, 1, 0

def setup(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    seed = 666 + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    return device

def get_model(device, args):
    model = Net1DWithAttention(
        in_channels=1, base_filters=64, ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,3,3,4,4,5,5],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=5, verbose=False, use_bn=True, use_do=True
    ).to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)
    return model

@torch.no_grad()
def _ddp_reduce_scalar(value: float, device):
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float32, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())

def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, is_main, train_sampler=None):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(total=len(train_loader), desc=f"Train {epoch+1}", dynamic_ncols=True, disable=not is_main)

    for inputs, freq_features, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        freq_features = freq_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs, freq_features)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.detach().item())
        num_batches += 1
        if is_main:
            pbar.update(1)

    if is_main:
        pbar.close()

    epoch_loss = running_loss / max(num_batches, 1)
    epoch_loss = _ddp_reduce_scalar(epoch_loss, device)
    return epoch_loss

@torch.no_grad()
def validate(model, val_loader, criterion, device, is_main):
    model.eval()
    running_loss, n_batches = 0.0, 0
    pbar = tqdm(total=len(val_loader), desc="Valid", dynamic_ncols=True, disable=not is_main)

    for inputs, freq_features, labels in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        freq_features = freq_features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(inputs, freq_features)
            loss = criterion(outputs, labels)

        running_loss += float(loss.item())
        n_batches += 1
        if is_main:
            pbar.update(1)

    if is_main:
        pbar.close()

    val_loss = running_loss / max(n_batches, 1)
    val_loss = _ddp_reduce_scalar(val_loss, device)
    return val_loss

def save_model(model, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    torch.save(to_save.state_dict(), path)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.0):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.best_score, self.epochs_no_improve, self.early_stop = None, 0, False
    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                if self.verbose:
                    print("Early stopping")
                self.early_stop = True

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='train_dir')
    parser.add_argument('--val_dir',   type=str, default='val_dir')
    parser.add_argument('--model_save_dir', type=str, default='model_save_dir')
    parser.add_argument('--final_model_save_dir', type=str, default='final_model_save_dir')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch', type=int, default=2)
    parser.add_argument('--cache_files', type=int, default=4)
    args = parser.parse_args()

    init_distributed_mode(args)
    device = setup(args)
    is_main = (args.rank == 0)

    train_loader, train_sampler = get_dataloader(
        args.train_dir, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, distributed=args.distributed,
        max_cached_files=args.cache_files, prefetch_factor=args.prefetch,
        pin_memory=True, persistent_workers=True
    )
    val_loader, val_sampler = get_dataloader(
        args.val_dir, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, distributed=args.distributed,
        max_cached_files=max(1, args.cache_files//2), prefetch_factor=args.prefetch,
        pin_memory=True, persistent_workers=True
    )

    model = get_model(device, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.3, mode='min')
    scaler = torch.cuda.amp.GradScaler()

    best_val = float('inf')
    early_stopping = EarlyStopping(patience=args.patience, verbose=is_main)

    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, is_main, train_sampler)
        val_loss   = validate(model, val_loader, criterion, device, is_main)
        scheduler.step(val_loss)

        if is_main:
            print(f'Epoch {epoch+1}/{args.num_epochs} | Train {train_loss:.4f} | Val {val_loss:.4f}')
            if val_loss < best_val:
                best_val = val_loss
                save_model(model, args.model_save_dir)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            if is_main:
                print("Early stopping triggered")
            break

    if is_main:
        save_model(model, args.final_model_save_dir)
        print("Training completed. Final model saved.")

if __name__ == '__main__':
    main()



