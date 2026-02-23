import os
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix
)
from torch.utils.data import DataLoader
from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer_window import TransformerSleepModel

CONFIG = {
    "test_list": "/path/to/your/test_list.txt",
    "model_path": "/path/to/your/model_checkpoint.pth",
    "save_dir": "/path/to/save/evaluation/results",
    "batch_size": 16,
    "window_size": 15,
    "stride": 1,
    "num_classes": 5,
    "input_dim": 15 * 1152,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 3,
    "dropout": 0.1,
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

def load_paths_from_txt(txt_file):
    with open(txt_file, 'r') as f:
        return f.read().splitlines()

def plot_confusion_matrix(cm, classes, save_path, normalize_axis=1):
    plt.figure(figsize=(10, 8))
    if normalize_axis == 1:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)
    
    plt.ylabel('True Label', fontsize=18)
    plt.xlabel('Predicted Label', fontsize=18)
    plt.tick_params(axis='x', labelbottom=True, labeltop=False)
    
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pct = f"{cm_normalized[i, j]:.1%}"
        
        plt.text(j, i, pct,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black", 
                 fontsize=24)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=800)
    plt.close()

def bootstrap_ci(metric_fn, y_true, y_pred, n=1000, alpha=0.95):
    stats = []
    for _ in range(n):
        indices = np.random.choice(np.arange(len(y_true)), size=len(y_true), replace=True)
        stats.append(metric_fn(y_true[indices], y_pred[indices]))
    stats = np.sort(stats)
    lower = np.percentile(stats, (1 - alpha) / 2 * 100)
    upper = np.percentile(stats, (1 + alpha) / 2 * 100)
    return np.mean(stats), lower, upper

def evaluate_with_ci(y_true, y_pred, y_prob, labels, name):
    results = {}

    acc_fn = lambda y_t, y_p: accuracy_score(y_t, y_p)
    f1_fn = lambda y_t, y_p: f1_score(y_t, y_p, average='weighted')
    kappa_fn = lambda y_t, y_p: cohen_kappa_score(y_t, y_p)
    auc_fn = lambda y_t, y_p: roc_auc_score(np.eye(len(labels))[y_t], y_p, average='macro', multi_class='ovr')

    acc, l1, u1 = bootstrap_ci(acc_fn, y_true, y_pred)
    f1, l2, u2 = bootstrap_ci(f1_fn, y_true, y_pred)
    kappa, l3, u3 = bootstrap_ci(kappa_fn, y_true, y_pred)
    try:
        auc, l4, u4 = bootstrap_ci(auc_fn, y_true, y_prob)
    except:
        auc, l4, u4 = float('nan'), float('nan'), float('nan')

    results['Accuracy'] = f"{acc:.3f} [{l1:.3f}, {u1:.3f}]"
    results['F1 Score'] = f"{f1:.3f} [{l2:.3f}, {u2:.3f}]"
    results['Kappa'] = f"{kappa:.3f} [{l3:.3f}, {u3:.3f}]"
    results['Macro AUC'] = f"{auc:.3f} [{l4:.3f}, {u4:.3f}]"

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes=labels, 
                          save_path=os.path.join(CONFIG["save_dir"], f"cm_{name}_row_norm.png"),
                          normalize_axis=1)
    plot_confusion_matrix(cm, classes=labels, 
                          save_path=os.path.join(CONFIG["save_dir"], f"cm_{name}_col_norm.png"),
                          normalize_axis=0)

    return results

def map_labels(y, mapping):
    new_y = np.copy(y)
    for old, new in mapping.items():
        new_y[y == old] = new
    return new_y

def predict_all(model, dataloader, device):
    model.eval()
    y_true_all, y_pred_all, y_prob_all = [], [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            
            mask = y != -100
            for i in range(x.size(0)):
                valid_indices = mask[i]
                y_true_all.append(y[i, valid_indices].cpu().numpy())
                y_pred_all.append(preds[i, valid_indices].cpu().numpy())
                y_prob_all.append(probs[i, valid_indices].cpu().numpy())
    
    return np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(y_prob_all)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_files = load_paths_from_txt(CONFIG["test_list"])
    test_set = SleepWindowDataset(test_files, CONFIG["window_size"], CONFIG["stride"])
    test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"], 
                             shuffle=False, collate_fn=collate_fn_window, num_workers=4)

    model = TransformerSleepModel(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        num_classes=CONFIG["num_classes"],
        dropout=CONFIG["dropout"]
    ).to(device)
    
    state_dict = torch.load(CONFIG["model_path"], map_location=device)
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"✅ Loaded model from {CONFIG['model_path']}")

    y_true, y_pred, y_prob = predict_all(model, test_loader, device)

    results = {}

    results["5-class"] = evaluate_with_ci(
        y_true, y_pred, y_prob, 
        labels=["W", "N1", "N2", "N3", "REM"], 
        name="5class"
    )

    map_2 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}
    results["2-class"] = evaluate_with_ci(
        map_labels(y_true, map_2),
        map_labels(y_pred, map_2),
        np.stack([y_prob[:, 0], y_prob[:, 1:].sum(axis=1)], axis=1),
        labels=["W", "Sleep"],
        name="2class"
    )

    map_3 = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}
    results["3-class"] = evaluate_with_ci(
        map_labels(y_true, map_3),
        map_labels(y_pred, map_3),
        np.stack([y_prob[:, 0], y_prob[:, 1:4].sum(axis=1), y_prob[:, 4]], axis=1),
        labels=["W", "NREM", "REM"],
        name="3class"
    )

    map_4 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}
    results["4-class"] = evaluate_with_ci(
        map_labels(y_true, map_4),
        map_labels(y_pred, map_4),
        np.stack([y_prob[:, 0], y_prob[:, 1:3].sum(axis=1), y_prob[:, 3], y_prob[:, 4]], axis=1),
        labels=["W", "Light", "Deep", "REM"],
        name="4class"
    )

    print("\n📊 Evaluation Summary (with 95% CI):")
    headers = ["Accuracy", "F1 Score", "Kappa", "Macro AUC"]
    print(f"{'Category':<10}  " + "  ".join(f"{h:<20}" for h in headers))
    print("-" * 90)
    for cat, metrics in results.items():
        print(f"{cat:<10}  " + "  ".join(f"{metrics[h]:<20}" for h in headers))
    
    excel_path = os.path.join(CONFIG["save_dir"], "evaluation_summary.xlsx")
    df = pd.DataFrame.from_dict(results, orient='index')
    df = df[headers]
    df.to_excel(excel_path)
    print(f"\n✅ Saved evaluation summary to {excel_path}")
    print(f"✅ Saved confusion matrices (row/column normalized) to {CONFIG['save_dir']}")

if __name__ == '__main__':
    main()
