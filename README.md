# Holter-to-Sleep

Holter-to-Sleep repurposes single-lead ECG for sleep phenotyping. This repository contains model training and evaluation code for **sleep staging** (Wake, N1, N2, N3, and REM) and **sleep event detection** (arousal and respiratory events), together with a web report interface for viewing sleep and cardiac results.

Both modeling tasks use a CNN to combine ECG waveform and frequency-domain features, followed by a Transformer to model temporal context from the extracted features. Respiratory event labels combine obstructive sleep apnea (OSA) and hypopnea.

Related tools: [Holter-to-Sleep web platform](http://ai.heartvoice.com.cn/Holter-to-Sleep/).

## Repository structure

```text
Holter-to-Sleep/
├── sleep_staging/          # Five-class sleep staging
│   ├── cnn/               # ECG segment classification and feature learning
│   └── transformer/       # Sleep staging with temporal context
├── sleep_events/           # Arousal and respiratory event detection
│   ├── cnn/               # Multitask ECG segment classification
│   └── transformer/       # Multitask event detection with temporal context
├── dashboard_demo.html     # Frontend for an integrated sleep and cardiac report
└── README.md
```

### Sleep staging: `sleep_staging/`

| Directory | File | Purpose |
| --- | --- | --- |
| `cnn/` | `dataset_split.py` | Create training, validation, and test file lists from NPZ data and summarize stage-label distributions. |
| `cnn/` | `data_utils.py` | Load ECG segments, frequency-domain features, and stage labels; provide cached datasets and data loaders. |
| `cnn/` | `net1d.py` | Define the 1D CNN and attention-based fusion of waveform and frequency-domain features. |
| `cnn/` | `train_cnn.py` | Train the five-class CNN, with distributed training and mixed precision support. |
| `cnn/` | `test_cnn.py` | Evaluate the CNN and export classification metrics, confusion matrices, and ROC curves. |
| `transformer/` | `feature_extraction.py` | Use a trained CNN to extract and save fused features for each recording. |
| `transformer/` | `split_data.py` | Create file lists for the extracted feature datasets. |
| `transformer/` | `dataset_sliding.py` | Build sliding windows of features and pad recording sequences for batching. |
| `transformer/` | `net1d.py` | Supply the CNN architecture used during feature extraction. |
| `transformer/` | `model_transformer_window.py` | Define the Transformer sleep-stage classifier and positional encoding. |
| `transformer/` | `train.py` | Train the Transformer using extracted feature sequences. |
| `transformer/` | `test.py` | Evaluate five-class staging and merged two-, three-, and four-class results, including bootstrap confidence intervals. |

### Sleep events: `sleep_events/`

| Directory | File | Purpose |
| --- | --- | --- |
| `cnn/` | `dataset_split.py` | Create training, validation, and test file lists from event-labeled NPZ data. |
| `cnn/` | `data_utils.py` | Load ECG and frequency-domain features, combine OSA/hypopnea labels into a respiratory label, and provide cached or chunked datasets. |
| `cnn/` | `net1d.py` | Define a shared CNN with separate arousal and respiratory classification heads. |
| `cnn/` | `train_multitask.py` | Train the multitask CNN for arousal and respiratory event detection. |
| `cnn/` | `test_multitask.py` | Evaluate the CNN separately for the two event tasks. |
| `transformer/` | `extract_feature.py` | Extract fused CNN features and save them with arousal and combined respiratory labels. |
| `transformer/` | `spilit_data.py` | Create file lists for extracted features; the filename retains its existing spelling. |
| `transformer/` | `dataset_sliding.py` | Build feature windows with arousal and respiratory labels for training and evaluation. |
| `transformer/` | `net1d.py` | Supply the CNN architecture used during feature extraction. |
| `transformer/` | `model_transformer_window.py` | Define a Transformer with separate event heads that classify the center of each window. |
| `transformer/` | `train.py` | Train the multitask Transformer. |
| `transformer/` | `test.py` | Evaluate event probabilities, select thresholds, and export metrics, confidence intervals, and ROC plots. |
| `transformer/` | `test_dataset_sliding.py` | Provide an alternative dataset loader with separate arousal, OSA, and hypopnea labels. This is not a unit test and is not the loader used by the current `test.py`. |

## Modeling workflow

1. Prepare labeled ECG segments and frequency-domain features in NPZ files, then create file lists with the relevant CNN splitting script.
2. Train and evaluate the CNN using `train_cnn.py` / `test_cnn.py` for staging, or `train_multitask.py` / `test_multitask.py` for events.
3. Load the trained CNN checkpoint in `feature_extraction.py` (staging) or `extract_feature.py` (events) to produce feature NPZ files.
4. Prepare feature file lists while preserving the original training, validation, and test recording assignments, then use the corresponding Transformer `train.py` and `test.py`.

The staging CNN loader expects `x`, `freq_features`, and `y`; the event CNN loader expects `x`, `arousal`, `osa`, and `hypopnea`, with optional `freq_features`. Extracted staging files contain `features` and `labels`; extracted event files contain `features`, `arousal`, and `respiratory`.

The scripts use PyTorch, NumPy, scikit-learn, pandas, Matplotlib, and tqdm. Before running them, configure data lists, checkpoint/output paths, and GPU settings in each script's arguments or configuration. Run scripts from their respective directories to resolve local imports. The staging Transformer training script requires a distributed launch with CUDA/NCCL. Datasets, pretrained checkpoints, and raw EDF preprocessing code are not included in this repository.

## Web report: `dashboard_demo.html`

The dashboard provides EDF file selection, streamed analysis logs, and an integrated report with:

- Linked ECG and sleep-stage timelines; the display groups N1 and N2 as light sleep.
- Sleep duration, sleep efficiency, stage proportions, apnea–hypopnea index (AHI), and arousal index (ArI).
- Heart-rate summaries, arrhythmia event counts, and heart-rate variability metrics.

This file is the frontend interface. It sends the selected EDF file to `POST /analyze` and expects newline-delimited JSON messages of type `log`, `result`, or `error`. The analysis backend is not included here, so opening the HTML alone does not run ECG analysis or model inference. Charts use ECharts loaded from a CDN.

## Citation

If you find this work helpful, please cite:

```bibtex
@article{xie2026holter,
  title={Holter-to-Sleep: AI-Enabled Repurposing of Single-Lead ECG for Sleep Phenotyping},
  author={Xie, Donglin and Zhao, Qingshuo and Wang, Jingyu and Geng, Shijia and Jin, Jiarui and Li, Jun and Guo, Rongrong and Nie, Guangkun and Tang, Gongzheng and Zhou, Yuxi and others},
  journal={arXiv preprint arXiv:2603.18714},
  year={2026}
}
```
