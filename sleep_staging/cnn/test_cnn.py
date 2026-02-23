'''
test the model
'''
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, roc_curve, auc, cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import pandas as pd
from net1d import Net1DWithAttention
from data_utils import NPZDataset

def load_paths_from_file(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

def load_test_data(test_paths, batch_size=256):
    test_dataset = NPZDataset(test_paths)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net1DWithAttention(
        in_channels=1, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=True, use_do=False, n_classes=5
    )
    state_dict = torch.load(model_path, map_location=device)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

def test_model(model, test_loader, device):
    all_labels, all_preds = [], []
    with torch.no_grad():
        for inputs, freq_features, labels in tqdm(test_loader, desc="Testing Progress"):
            inputs, freq_features, labels = inputs.to(device), freq_features.to(device), labels.to(device)
            outputs = torch.softmax(model(inputs, freq_features), dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def calculate_metrics(all_labels, all_preds, classes):
    print(f"Calculating metrics for classes: {classes}")
    print(f"all_labels shape: {all_labels.shape}")
    print(f"all_preds shape: {all_preds.shape}")
    predicted_classes = np.argmax(all_preds, axis=1)
    cm = confusion_matrix(all_labels, predicted_classes)
    print(f"Confusion matrix:\n{cm}")
    all_labels_binarized = label_binarize(all_labels, classes=classes)
    if len(classes) == 2 and all_labels_binarized.shape[1] == 1:
        all_labels_binarized = np.hstack((1 - all_labels_binarized, all_labels_binarized))
    print(f"all_labels_binarized shape: {all_labels_binarized.shape}")
    fpr, tpr, roc_auc = {}, {}, {}
    for i, class_ in enumerate(classes):
        print(f"Processing class: {class_}")
        fpr[class_], tpr[class_], _ = roc_curve(all_labels_binarized[:, i], all_preds[:, i])
        roc_auc[class_] = auc(fpr[class_], tpr[class_])
        print(f"Class {class_} - AUC: {roc_auc[class_]}")
    metrics = {
        'accuracy': accuracy_score(all_labels, predicted_classes),
        'precision': precision_score(all_labels, predicted_classes, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, predicted_classes, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, predicted_classes, average='weighted', zero_division=0),
        'macro_f1': f1_score(all_labels, predicted_classes, average='macro', zero_division=0),
        'weighted_macro_f1': f1_score(all_labels, predicted_classes, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(all_labels, predicted_classes),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'all_labels_binarized': all_labels_binarized
    }
    return cm, metrics

def plot_confusion_matrix(cm, classes, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    plt.tick_params(labeltop=True, labelbottom=False, top=True, bottom=False)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pct = f"{cm_percent[i, j]:.1%}" if cm_percent[i, j] != 0 else ''
        plt.text(j, i, f"{cm[i, j]}\n({pct})",
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def plot_roc_curve(fpr, tpr, roc_auc, classes, save_path):
    plt.figure(figsize=(10, 8))
    colors = itertools.cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightblue', 'lightgreen'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(save_path)

def plot_total_roc_curve(all_labels_binarized, all_preds, save_path):
    fpr, tpr, _ = roc_curve(all_labels_binarized.ravel(), all_preds.ravel())
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Macro-average ROC curve (area = {roc_auc:0.2f})',
             color='navy', linestyle='-', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Total Receiver Operating Characteristic', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.savefig(save_path, bbox_inches='tight')
    return roc_auc

def calculate_and_plot(all_labels, all_preds, classes, save_path_prefix):
    cm, metrics = calculate_metrics(all_labels, all_preds, classes)
    plot_confusion_matrix(cm, classes, save_path=f"{save_path_prefix}_confusion_matrix.png")
    plot_roc_curve(metrics['fpr'], metrics['tpr'], metrics['roc_auc'], classes, save_path=f"{save_path_prefix}_roc_curve.png")
    total_roc_auc = plot_total_roc_curve(metrics['all_labels_binarized'], all_preds, save_path=f"{save_path_prefix}_total_roc_curve.png")
    return metrics, total_roc_auc

def main(test_dir, model_path, results_dir='results'):
    os.makedirs(results_dir, exist_ok=True)
    
    test_paths = load_paths_from_file(test_dir)
    test_loader = load_test_data(test_paths, batch_size=256)
    model = load_model(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    all_labels, all_preds = test_model(model, test_loader, device)

    # Original 5-class classification
    classes_5 = np.array([0, 1, 2, 3, 4])
    metrics_5, total_roc_auc_5 = calculate_and_plot(all_labels, all_preds, classes_5, os.path.join(results_dir, "5_class"))

    # Binary classification: 0 vs 1,2,3,4
    binary_labels = np.where(all_labels == 0, 0, 1)
    binary_preds = np.column_stack([all_preds[:, 0], all_preds[:, 1:].sum(axis=1)])
    print(f"binary_labels shape: {binary_labels.shape}")
    print(f"binary_preds shape: {binary_preds.shape}")
    metrics_2, total_roc_auc_2 = calculate_and_plot(binary_labels, binary_preds, np.array([0, 1]), os.path.join(results_dir, "binary_class"))

    # Three-class classification: 0 vs 1,2,3 vs 4
    three_labels = np.where(all_labels == 0, 0, np.where(all_labels == 4, 2, 1))
    three_preds = np.column_stack([all_preds[:, 0], all_preds[:, 1:4].sum(axis=1), all_preds[:, 4]])
    print(f"three_labels shape: {three_labels.shape}")
    print(f"three_preds shape: {three_preds.shape}")
    metrics_3, total_roc_auc_3 = calculate_and_plot(three_labels, three_preds, np.array([0, 1, 2]), os.path.join(results_dir, "three_class"))

    # Four-class classification: 0 vs 1,2 vs 3 vs 4
    four_labels = np.where(all_labels == 0, 0, np.where(all_labels == 1, 1, np.where(all_labels == 2, 2, 3)))
    four_preds = np.column_stack([all_preds[:, 0], all_preds[:, 1], all_preds[:, 2], all_preds[:, 3:].sum(axis=1)])
    print(f"four_labels shape: {four_labels.shape}")
    print(f"four_preds shape: {four_preds.shape}")
    metrics_4, total_roc_auc_4 = calculate_and_plot(four_labels, four_preds, np.array([0, 1, 2, 3]), os.path.join(results_dir, "four_class"))

    # Combine metrics for the dataframe
    metrics_data = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1", "Macro F1", "Weighted Macro F1", "Cohen's Kappa", "Total ROC AUC"],
        "5-class": [metrics_5['accuracy'], metrics_5['precision'], metrics_5['recall'], metrics_5['f1'], metrics_5['macro_f1'], metrics_5['weighted_macro_f1'], metrics_5['kappa'], total_roc_auc_5],
        "2-class": [metrics_2['accuracy'], metrics_2['precision'], metrics_2['recall'], metrics_2['f1'], metrics_2['macro_f1'], metrics_2['weighted_macro_f1'], metrics_2['kappa'], total_roc_auc_2],
        "3-class": [metrics_3['accuracy'], metrics_3['precision'], metrics_3['recall'], metrics_3['f1'], metrics_3['macro_f1'], metrics_3['weighted_macro_f1'], metrics_3['kappa'], total_roc_auc_3],
        "4-class": [metrics_4['accuracy'], metrics_4['precision'], metrics_4['recall'], metrics_4['f1'], metrics_4['macro_f1'], metrics_4['weighted_macro_f1'], metrics_4['kappa'], total_roc_auc_4],
    }

    for class_idx, class_name in enumerate(["Wake", "N1", "N2", "N3", "REM"]):
        metrics_data[f"ROC AUC 5-class {class_name}"] = [
            metrics_5['roc_auc'][class_idx] if class_idx in metrics_5['roc_auc'] else None,
            None, None, None, None, None, None, None
        ]

    for class_idx, class_name in enumerate(["Wake", "Non-Wake"]):
        metrics_data[f"ROC AUC 2-class {class_name}"] = [
            None,
            metrics_2['roc_auc'][class_idx] if class_idx in metrics_2['roc_auc'] else None,
            None, None, None, None, None, None
        ]

    for class_idx, class_name in enumerate(["Wake", "Light Sleep", "REM"]):
        metrics_data[f"ROC AUC 3-class {class_name}"] = [
            None, None,
            metrics_3['roc_auc'][class_idx] if class_idx in metrics_3['roc_auc'] else None,
            None, None, None, None, None
        ]

    for class_idx, class_name in enumerate(["Wake", "N1", "N2", "N3, REM"]):
        metrics_data[f"ROC AUC 4-class {class_name}"] = [
            None, None, None,
            metrics_4['roc_auc'][class_idx] if class_idx in metrics_4['roc_auc'] else None,
            None, None, None, None
        ]

    metrics_df = pd.DataFrame(metrics_data)

    print(metrics_df)
    metrics_df.to_csv(os.path.join(results_dir, "classification_metrics.csv"), index=False)

if __name__ == "__main__":
    test_dir = 'path/to/test_paths.txt'
    model_path = 'path/to/model.pth'
    results_dir = 'path/to/results'
    os.makedirs(results_dir, exist_ok=True)
    main(test_dir, model_path, results_dir)
