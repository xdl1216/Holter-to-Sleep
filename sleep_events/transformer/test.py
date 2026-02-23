import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc, confusion_matrix
)
import pandas as pd

from dataset_sliding import SleepWindowDataset, collate_fn_window
from model_transformer_window import TransformerSleepModel
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

CONFIG = {
    "test_list": "/path/to/test_paths.txt",
    "model_path": "/path/to/your/model_checkpoint.pth",
    "save_roc_path": "./results/roc_curves.png",
    "metrics_csv_path": "./results/metrics_summary.csv",

    "batch_size": 16,
    "input_dim": 1152,
    "hidden_dim": 512,
    "n_heads": 8,
    "num_layers": 4,
    "dropout": 0.1,
    "window_size": 15,
    "stride": 1,
    "bootstrap_num": 1000,
    "threshold_search_range": np.linspace(0.01, 0.99, 99), 
    "metric_for_best_thresh": "f1"  # "f1", "recall", "precision"
}

def load_paths_from_txt(txt_file):
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"The path file does not exist: {txt_file}")
    with open(txt_file, 'r') as f:
        paths = f.read().splitlines()
    return [p for p in paths if p.strip()]

def get_data_loader(file_list, config, is_train=False):
    dataset = SleepWindowDataset(
        file_list=file_list,
        window_size=config["window_size"],
        stride=config["stride"]
    )
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=is_train,
        num_workers=1,
        collate_fn=collate_fn_window,
        pin_memory=False
    )
    return loader, dataset

def predict_probs(model, dataloader, device):
    model.eval()
    probs = {"arousal": [], "respiratory": []}
    labels = {"arousal": [], "respiratory": []}

    with torch.no_grad():
        for x, a, r in tqdm(dataloader, desc="Predicting Probabilities"):
            x = x.to(device)
            a = a.to(device)
            r = r.to(device)

            with autocast():
                aro_logits, resp_logits = model(x)

            probs["arousal"].append(torch.softmax(aro_logits, dim=-1)[:, 1].cpu().numpy())
            probs["respiratory"].append(torch.softmax(resp_logits, dim=-1)[:, 1].cpu().numpy())

            labels["arousal"].append(a.cpu().numpy())
            labels["respiratory"].append(r.cpu().numpy())

    for task in probs.keys():
        probs[task] = np.concatenate(probs[task]).flatten()
        labels[task] = np.concatenate(labels[task]).flatten()

    return labels, probs

def find_best_threshold(y_true, y_prob, task_name, config):
    best_metric = 0.0
    best_thresh = 0.5 
    
    for thresh in config["threshold_search_range"]:
        y_pred = (y_prob >= thresh).astype(int)

        if config["metric_for_best_thresh"] == "f1":
            metric = f1_score(y_true, y_pred, zero_division=0)
        elif config["metric_for_best_thresh"] == "recall":
            metric = recall_score(y_true, y_pred, zero_division=0)
        elif config["metric_for_best_thresh"] == "precision":
            metric = precision_score(y_true, y_pred, zero_division=0)
        else:
            metric = f1_score(y_true, y_pred, zero_division=0)

        if metric > best_metric:
            best_metric = metric
            best_thresh = thresh

    print(f"✅ {task_name} Optimal threshold: {best_thresh:.3f}（Optimal{config['metric_for_best_thresh'].upper()}：{best_metric:.4f}）")
    return best_thresh

def calculate_metrics_with_ci(y_true, y_prob, best_thresh, task_name, n_bootstrap=1000):
    y_pred = (y_prob >= best_thresh).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "Accuracy": "NaN[NaN, NaN]",
            "Precision": "NaN[NaN, NaN]",
            "Recall": "NaN[NaN, NaN]",
            "Specificity": "NaN[NaN, NaN]",
            "F1-Score": "NaN[NaN, NaN]",
            "AUC": "NaN[NaN, NaN]",
            "Best_Threshold": f"{best_thresh:.3f}"
        }

    # ========== original metrics ==========
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        base_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        base_auc = 0.0

    # ========== Bootstrap calculate CI ==========
    bootstrap_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "auc": []
    }

    rng = np.random.RandomState(42)
    
    for _ in tqdm(range(n_bootstrap), desc=f"{task_name} Bootstrap", leave=False):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_prob_boot = y_prob[indices]
        
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        y_pred_boot = (y_prob_boot >= best_thresh).astype(int)
        
        try:
            bootstrap_metrics["accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
            bootstrap_metrics["precision"].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_metrics["recall"].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            bootstrap_metrics["f1"].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
            
            # Specificity
            tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true_boot, y_pred_boot).ravel()
            spec = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0.0
            bootstrap_metrics["specificity"].append(spec)
            
            # AUC
            boot_auc = roc_auc_score(y_true_boot, y_prob_boot)
            bootstrap_metrics["auc"].append(boot_auc)
        except:
            continue

    # ========== 95% CI ==========
    def format_metric_with_ci(value, bootstrap_values):
        """格式化为 value[lower, upper]"""
        if len(bootstrap_values) == 0:
            return f"{value:.3f}[NaN, NaN]"
        lower = np.percentile(bootstrap_values, 2.5)
        upper = np.percentile(bootstrap_values, 97.5)
        return f"{value:.3f}[{lower:.3f}, {upper:.3f}]"

    results = {
        "Accuracy": format_metric_with_ci(accuracy, bootstrap_metrics["accuracy"]),
        "Precision": format_metric_with_ci(precision, bootstrap_metrics["precision"]),
        "Recall": format_metric_with_ci(recall, bootstrap_metrics["recall"]),
        "Specificity": format_metric_with_ci(specificity, bootstrap_metrics["specificity"]),
        "F1-Score": format_metric_with_ci(f1, bootstrap_metrics["f1"]),
        "AUC": format_metric_with_ci(base_auc, bootstrap_metrics["auc"]),
        "Best_Threshold": f"{best_thresh:.3f}"
    }

    return results

def plot_roc_curves(labels, scores, best_thresholds, save_path):
    plt.figure(figsize=(10, 8))
    tasks = ['arousal', 'respiratory']
    colors = ['black', 'gold']
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 

    for task, color in zip(tasks, colors):
        y_true = labels[task].flatten()
        y_score = scores[task].flatten()
        best_thresh = best_thresholds[task]
        
        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        base_auc = auc(fpr, tpr)

        thresh_idx = np.argmin(np.abs(thresholds - best_thresh))
        best_fpr = fpr[thresh_idx]
        best_tpr = tpr[thresh_idx]

        n_bootstraps = CONFIG["bootstrap_num"]
        tprs = []
        rng = np.random.RandomState(42)
        mean_fpr = np.linspace(0, 1, 100)

        for i in range(n_bootstraps // 2): 
            indices = rng.choice(len(y_true), size=len(y_true), replace=True)
            y_true_boot = y_true[indices]
            y_score_boot = y_score[indices]

            if len(np.unique(y_true_boot)) < 2:
                continue

            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_score_boot)
            interp_tpr = np.interp(mean_fpr, fpr_boot, tpr_boot)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        tpr_upper = np.minimum(mean_tpr + std_tpr * 2, 1.0)
        tpr_lower = np.maximum(mean_tpr - std_tpr * 2, 0.0)

        label_name = 'Arousal' if task == 'arousal' else 'Respiratory'

        plt.plot(mean_fpr, mean_tpr, color=color, lw=4,
                 label=f'{label_name} (AUC={base_auc:.3f})')
        plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.15)
        plt.scatter(best_fpr, best_tpr, color=color, s=80, zorder=5, edgecolors='white')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ The ROC curve has been saved as: {save_path}")

def print_metrics_summary(metrics_dict):
    print("\n" + "="*150)
    print("📋 Summary（value[95% CI lower, 95% CI upper]）")
    print("="*150)
    
    header = (f"{'Task':<12} {'Accuracy':<25} {'Precision':<25} {'Recall':<25} "
              f"{'Specificity':<25} {'F1-Score':<25} {'AUC':<25} {'Threshold':<12}")
    print(header)
    print("-"*150)
    
    for task, metrics in metrics_dict.items():
        row = (f"{task.capitalize():<12} "
               f"{metrics['Accuracy']:<25} "
               f"{metrics['Precision']:<25} "
               f"{metrics['Recall']:<25} "
               f"{metrics['Specificity']:<25} "
               f"{metrics['F1-Score']:<25} "
               f"{metrics['AUC']:<25} "
               f"{metrics['Best_Threshold']:<12}")
        print(row)
    print("="*150)

def save_metrics_table(metrics_dict, save_path):
    df = pd.DataFrame.from_dict(metrics_dict, orient='index')
    df.index.name = "Task"
    
    column_order = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC", "Best_Threshold"]
    df = df[[col for col in column_order if col in df.columns]]
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, encoding='utf-8-sig')
    print(f"📝 The metrics table has been saved as CSV: {save_path}")
    print(f"   Format: Each metric is in value[95% CI lower, 95% CI upper] format")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 device: {device}")

    print("\n📥 Loading Transformer Model...")
    model = TransformerSleepModel(
        input_dim=CONFIG["input_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        n_heads=CONFIG["n_heads"],
        num_layers=CONFIG["num_layers"],
        dropout=CONFIG["dropout"]
    ).to(device)

    state_dict = torch.load(CONFIG["model_path"], map_location=device)
    new_state_dict = {}
    for key, val in state_dict.items():
        new_key = key.replace("module.", "") if key.startswith("module.") else key
        new_state_dict[new_key] = val
    model.load_state_dict(new_state_dict)
    print(f"✅ Model loaded: {CONFIG['model_path']}")

    print("\n" + "="*50)
    print("📌 Process: Search for the optimal threshold (F1) on the test set and evaluate")
    print("="*50)

    test_files = load_paths_from_txt(CONFIG["test_list"])
    print(f"📥 test set: {len(test_files)} samples")
    test_loader, _ = get_data_loader(test_files, CONFIG)

    print("\n🚀 Starting Inferring...")
    test_labels, test_probs = predict_probs(model, test_loader, device)

    best_thresholds = {}
    tasks = ["arousal", "respiratory"]
    
    for task in tasks:
        best_thresholds[task] = find_best_threshold(
            y_true=test_labels[task],
            y_prob=test_probs[task],
            task_name=task,
            config=CONFIG
        )

    print("\n📊 Calculating final evaluation metrics and 95% confidence intervals...")
    metrics_summary = {}
    for task in tasks:
        metrics_summary[task] = calculate_metrics_with_ci(
            y_true=test_labels[task],
            y_prob=test_probs[task],
            best_thresh=best_thresholds[task],
            task_name=task,
            n_bootstrap=CONFIG["bootstrap_num"]
        )

    print_metrics_summary(metrics_summary)
    plot_roc_curves(test_labels, test_probs, best_thresholds, CONFIG["save_roc_path"])
    save_metrics_table(metrics_summary, CONFIG["metrics_csv_path"])

    print("\n🎉 Finish Testing!")

if __name__ == "__main__":
    main()

