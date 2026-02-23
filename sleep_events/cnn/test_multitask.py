# test_step1_simple.py
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

from net1d import MultiTaskNet1D
from data_utils import NPZDataset

# ================= Configuration =================
CONFIG = {
    "test_list": "data/test_files.txt",
    "model_path": "models/model.pth",
    "output_csv": "results/multitask_metrics.csv",

    "batch_size": 256,
    "num_workers": 8,

    "threshold_steps": 100,
    "metric_for_best_thresh": "f1"
}

# ================= Helper Functions =================

def load_paths(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_dataloader(config):
    paths = load_paths(config["test_list"])
    dataset = NPZDataset(paths)
    return DataLoader(
        dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=config["num_workers"], 
        pin_memory=True
    )

def load_model(config, device):
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
        n_classes_list=[2, 2]  # [Arousal, Respiratory]
    ).to(device)
    
    print(f"Loading model from {config['model_path']}...")
    state_dict = torch.load(config["model_path"], map_location=device)
    
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def run_inference(model, loader, device):
    results = {
        "arousal": {"probs": [], "labels": []},
        "respiratory": {"probs": [], "labels": []}
    }
    
    print("🚀 Running Inference...")
    with torch.no_grad():
        for inputs, freq_data, labels in tqdm(loader, desc="Testing"):
            inputs = inputs.to(device)
            freq_data = freq_data.to(device)
            
            la = labels['arousal'].to(device)
            lr = labels['respiratory'].to(device)

            logits_a, logits_r = model(inputs, freq_data)

            prob_a = torch.softmax(logits_a, dim=-1)[:, 1].cpu().numpy()
            prob_r = torch.softmax(logits_r, dim=-1)[:, 1].cpu().numpy()

            results["arousal"]["probs"].extend(prob_a)
            results["arousal"]["labels"].extend(la.cpu().numpy())
            
            results["respiratory"]["probs"].extend(prob_r)
            results["respiratory"]["labels"].extend(lr.cpu().numpy())
            
    for task in results:
        results[task]["probs"] = np.array(results[task]["probs"])
        results[task]["labels"] = np.array(results[task]["labels"])
        
    return results

def calculate_metrics(y_true, y_prob, config):
    best_f1 = -1
    best_thresh = 0.5
    thresholds = np.linspace(0.01, 0.99, config["threshold_steps"])

    for th in thresholds:
        y_pred_tmp = (y_prob >= th).astype(int)
        f1 = f1_score(y_true, y_pred_tmp, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    y_pred = (y_prob >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    try:
        auc_val = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_val = 0.0
        
    return {
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Specificity": round(spec, 4),
        "F1-Score": round(f1, 4),
        "AUC": round(auc_val, 4),
        "Best_Threshold": round(best_thresh, 3)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    loader = get_dataloader(CONFIG)
    model = load_model(CONFIG, device)

    results_data = run_inference(model, loader, device)

    final_metrics = {}
    
    print("\n" + "="*100)
    print(f"{'Task':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<12} {'F1-Score':<10} {'AUC':<10} {'Threshold':<10}")
    print("-" * 100)
    
    for task in ["arousal", "respiratory"]:
        metrics = calculate_metrics(
            results_data[task]["labels"], 
            results_data[task]["probs"],
            CONFIG
        )
        final_metrics[task] = metrics
        
        print(f"{task:<12} {metrics['Accuracy']:<10} {metrics['Precision']:<10} {metrics['Recall']:<10} "
              f"{metrics['Specificity']:<12} {metrics['F1-Score']:<10} {metrics['AUC']:<10} {metrics['Best_Threshold']:<10}")
        
    print("="*100 + "\n")

    df = pd.DataFrame.from_dict(final_metrics, orient='index')
    df.index.name = 'Task'
    cols = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC", "Best_Threshold"]
    df = df[cols]
    
    os.makedirs(os.path.dirname(CONFIG["output_csv"]), exist_ok=True)
    df.to_csv(CONFIG["output_csv"])
    print(f"💾 Metrics saved to: {CONFIG['output_csv']}")

if __name__ == "__main__":
    main()