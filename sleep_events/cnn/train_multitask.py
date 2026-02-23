# train.py
# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('NCCL_IB_DISABLE', '1')
os.environ.setdefault('NCCL_SOCKET_IFNAME', '^docker0,lo')
os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')
os.environ.setdefault('NCCL_TIMEOUT', '3600')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from net1d import MultiTaskNet1D
from data_utils import NPZDataset, ChunkedNPZDataset

# ----------------- ddp -----------------
def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(args.gpu)
        dist.init_process_group(backend='nccl', init_method='env://')
        dist.barrier()
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False

def setup(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    seed = 666 + args.rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return device

def load_paths_from_file(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def simple_augment(x):
    return x + torch.randn_like(x) * 0.01

def get_dataloader(args, list_file, batch_size, num_workers, is_train=True):
    paths = load_paths_from_file(list_file)
    
    is_main = (not args.distributed or args.rank == 0)

    if args.use_chunked:
        if is_main:
            print(f"Using ChunkedNPZDataset (chunk_size={args.chunk_size})")
        dataset = ChunkedNPZDataset(
            paths, 
            chunk_size=args.chunk_size,
            verbose=is_main
        )
    else:
        if is_main:
            print(f"Using NPZDataset with file-level cache (max_cached_files={args.max_cached_files})")
        dataset = NPZDataset(
            paths,
            max_cached_files=args.max_cached_files,
            verbose=is_main
        )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=is_train,
        drop_last=is_train,
    )

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=is_train,
        num_workers=num_workers,
        pin_memory=True,
        timeout=args.timeout,
    )
    
    if num_workers > 0:
        dl_kwargs.update(dict(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor
        ))
    
    loader = DataLoader(**dl_kwargs)
    return loader, sampler

# ----------------- model -----------------
def get_model(device, args):
    model = MultiTaskNet1D(
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
        n_classes_list=[2, 2]
    ).to(device)

    if args.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
    return model

@torch.no_grad()
def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

# ----------------- train and val -----------------
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, train_sampler, scaler, args):
    model.train()
    train_sampler.set_epoch(epoch)

    is_main = (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0)
    data_iter = tqdm(train_loader, desc=f"Train {epoch+1}", leave=False) if is_main else train_loader

    running_loss = 0.0
    successful_batches = 0
    skipped_batches = 0
    
    for step, batch in enumerate(data_iter):
        try:
            inputs, freq_data, labels = batch
            
            inputs = inputs.to(device, non_blocking=True)
            freq_data = freq_data.to(device, non_blocking=True)
            
            la = labels['arousal'].to(device, non_blocking=True)
            lr = labels['respiratory'].to(device, non_blocking=True)

            inputs = simple_augment(inputs)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_a, out_r = model(inputs, freq_data)
                loss = criterion(out_a, la) + criterion(out_r, lr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss)
            successful_batches += 1
            
            if is_main and (step + 1) % 20 == 0:
                data_iter.set_postfix(
                    loss=float(loss),
                    skip=skipped_batches
                )
                
        except RuntimeError as e:
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg:
                if is_main:
                    print(f"\n⚠️  OOM at step {step}, skipping batch and clearing cache")
                torch.cuda.empty_cache()
                skipped_batches += 1
                continue
                
            elif "timeout" in error_msg or "timed out" in error_msg:
                if is_main:
                    print(f"\n⚠️  Timeout at step {step}, skipping batch (total skipped: {skipped_batches + 1})")
                skipped_batches += 1

                if skipped_batches > len(train_loader) * 0.1:
                    if is_main:
                        print(f"\n⚠️  WARNING: Too many skipped batches ({skipped_batches}/{len(train_loader)})")
                        print("    Consider: 1) Reduce num_workers, 2) Increase timeout, 3) Use chunked mode")
                continue
            else:
                print(f"\n❌ Error at step {step}: {e}")
                raise
        
        except Exception as e:
            print(f"\n❌ Unexpected error at step {step}: {e}")
            raise

    if successful_batches == 0:
        print("\n❌ ERROR: No successful batches in this epoch!")
        return float('inf')
    
    if is_main and skipped_batches > 0:
        print(f"\n  Skipped {skipped_batches}/{len(train_loader)} batches due to errors")
    
    epoch_loss = running_loss / successful_batches
    epoch_loss_t = torch.tensor([epoch_loss], device=device)
    epoch_loss_avg = reduce_mean(epoch_loss_t)[0].item()
    return epoch_loss_avg

@torch.no_grad()
def validate(model, val_loader, criterion, device, val_sampler=None):
    model.eval()
    if val_sampler is not None and hasattr(val_sampler, 'set_epoch'):
        val_sampler.set_epoch(0)

    running_loss = 0.0
    successful_batches = 0
    skipped_batches = 0
    
    for batch in val_loader:
        try:
            inputs, freq_data, labels = batch
            
            inputs = inputs.to(device, non_blocking=True)
            freq_data = freq_data.to(device, non_blocking=True)
            
            la = labels['arousal'].to(device, non_blocking=True)
            lr = labels['respiratory'].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                out_a, out_r = model(inputs, freq_data)
                loss = criterion(out_a, la) + criterion(out_r, lr)

            running_loss += float(loss)
            successful_batches += 1
            
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                skipped_batches += 1
                continue
            else:
                raise

    if successful_batches == 0:
        print("⚠️  WARNING: No successful validation batches!")
        return float('inf')
    
    val_loss = running_loss / successful_batches
    val_loss_t = torch.tensor([val_loss], device=device)
    val_loss_avg = reduce_mean(val_loss_t)[0].item()
    return val_loss_avg

def save_model(model, path="model_final.pth"):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    to_save = model.module if isinstance(model, DDP) else model
    torch.save(to_save.state_dict(), path)

# ----------------- early stopping -----------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, val):
        score = -val
        if self.best is None or score > self.best + self.delta:
            self.best = score
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return self.should_stop

# ----------------- main -----------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_dir', 
                       default='/path/to/train_paths.txt', 
                       type=str)
    parser.add_argument('--val_dir', 
                       default='/path/to/val_paths.txt', 
                       type=str)
    parser.add_argument('--model_save_dir', 
                       default='./best_model.pth', 
                       type=str)

    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_epochs', default=50, type=int)
    parser.add_argument('--patience', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--num_workers', default=2, type=int, 
                       help='Number of workers')
    parser.add_argument('--timeout', default=1800, type=int, 
                       help='DataLoader timeout')
    parser.add_argument('--prefetch_factor', default=2, type=int, 
                       help='Prefetch factor')

    parser.add_argument('--max_cached_files', default=30, type=int,
                       help='The maximum number of files that each worker can cache (recommendation: 20 - 50)')
    parser.add_argument('--use_chunked', action='store_true',
                       help='Use the chunk loading mode (suitable for extremely large datasets)')
    parser.add_argument('--chunk_size', default=100, type=int,
                       help='The number of files in each block under the chunk mode')

    args = parser.parse_args()

    init_distributed_mode(args)
    is_main = (not args.distributed or args.rank == 0)
    
    if is_main:
        print("=" * 70)
        print("Training Configuration (v2 - I/O Optimized)")
        print("=" * 70)
        print(f"  Batch size: {args.batch_size}")
        print(f"  Num workers: {args.num_workers}")
        print(f"  Timeout: {args.timeout}s ({args.timeout//60}mins)")
        print(f"  Prefetch factor: {args.prefetch_factor}")
        
        if args.use_chunked:
            print(f"  Mode: Chunked loading (chunk_size={args.chunk_size})")
        else:
            print(f"  Mode: File-level LRU cache (max_cached_files={args.max_cached_files})")
        
        if args.distributed:
            print(f"  Distributed: Yes (world_size={args.world_size}, rank={args.rank})")
        print("=" * 70)
        print()
    
    device = setup(args)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if is_main:
                print(f"Creating dataloaders (attempt {attempt + 1}/{max_retries})...")
            
            train_loader, train_sampler = get_dataloader(
                args, args.train_dir, args.batch_size, args.num_workers, is_train=True
            )
            val_loader, val_sampler = get_dataloader(
                args, args.val_dir, args.batch_size, args.num_workers, is_train=False
            )
            
            if is_main:
                print(f"✓ Dataloaders created successfully!")
                print(f"  Training batches: {len(train_loader)}")
                print(f"  Validation batches: {len(val_loader)}")
                print()
            break
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"❌ Failed to create dataloaders: {e}")
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"❌ Failed after {max_retries} attempts")
                raise

    model = get_model(device, args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.3, mode='min')
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')
    early_stopping = EarlyStopping(patience=args.patience, delta=0.0)

    if is_main:
        print("Starting training...")
        print("=" * 70)

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        
        if is_main:
            print(f"\nEpoch {epoch+1}/{args.num_epochs}")
            print("-" * 70)
        
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, 
            epoch, train_sampler, scaler, args
        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        val_loss = validate(model, val_loader, criterion, device, val_sampler)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        
        if is_main:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'\nResults:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss:   {val_loss:.6f}')
            print(f'  LR:         {current_lr:.2e}')
            print(f'  Time:       {epoch_time:.1f}s')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(model, args.model_save_dir)
                print(f'  ✓ New best model saved! (Val Loss: {val_loss:.6f})')

        stop_flag = torch.tensor([0], device=device)
        if is_main and early_stopping.step(val_loss):
            stop_flag[0] = 1
            
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stop_flag, src=0)
            
        if stop_flag.item() == 1:
            if is_main:
                print("\n" + "=" * 70)
                print("Early stopping triggered.")
                print("=" * 70)
            break

    if is_main:
        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best model saved to: {args.model_save_dir}")
        print("=" * 70)

if __name__ == '__main__':
    main()