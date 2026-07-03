# 🍎 Freshness Classification of Fruits & Vegetables using YOLOv8s and VGG16

A two-stage deep learning pipeline for real-time freshness detection of fruits and vegetables. The system first detects and localizes produce using a custom-trained YOLOv8s object detector, then classifies each detected item as **Fresh** or **Rotten** using a VGG16 transfer learning classifier. Low-confidence or unrecognized detections are automatically labeled as **"Others"** and excluded from freshness classification.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Supported Classes](#supported-classes)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset Structure](#dataset-structure)
- [Model Weights](#model-weights)
- [Usage](#usage)
- [Results](#results)
- [Scripts Reference](#scripts-reference)

---

## Overview

| Component | Details |
|-----------|---------|
| Detection Model | YOLOv8s (custom-trained) |
| Classification Model | VGG16 (ImageNet → fine-tuned) |
| Task | Object Detection + Binary Freshness Classification |
| Classes Detected | 6 (Apple, Banana, Carrot, Tomato, Mango, Potato) |
| Freshness Labels | Fresh / Rotten |
| Training Images (VGG16) | 57,000+ |
| Framework | Ultralytics YOLOv8, TensorFlow / Keras |

---

## Pipeline Architecture

```
Input Image
     │
     ▼
┌─────────────────────┐
│   YOLOv8s Detector  │  ← Custom-trained on 6 produce classes
│   (best.pt)         │
└─────────┬───────────┘
          │
          ▼
   Confidence ≥ 0.5?
   Known class?
     │         │
    YES        NO ──────────────► Label: "Others" (skip VGG16)
     │
     ▼
  Crop detected region
     │
     ▼
┌─────────────────────┐
│   VGG16 Classifier  │  ← Fine-tuned on Fresh / Rotten crops
│   (freshness_vgg16  │
│    .keras)          │
└─────────┬───────────┘
          │
          ▼
   Fresh / Rotten
   + Confidence %
```

---

## Supported Classes

| Class ID | Produce  |
|----------|----------|
| 0        | Apple    |
| 1        | Banana   |
| 2        | Carrot   |
| 3        | Tomato   |
| 4        | Mango    |
| 5        | Potato   |

---

## Project Structure

```
Freshness-classification-of-Fruits-or-Vegetables-using-YOLOv8s-and-VGG16/
│
├── Models/
│   ├── best.pt                        # Custom-trained YOLOv8s weights
│   ├── freshness_vgg16.keras          # Trained VGG16 freshness classifier
│   ├── history_stage1.json            # Stage 1 training history
│   ├── history_stage2.json            # Stage 2 training history
│   └── training_curves_*.png          # Training curve plots
│
├── Dataset/
│   ├── train/
│   │   ├── fresh/
│   │   └── rotten/
│   ├── val/
│   │   ├── fresh/
│   │   └── rotten/
│   └── test/
│       ├── fresh/
│       └── rotten/
│
├── Notebooks/
│   └── train_vgg16_final.ipynb        # Full VGG16 training notebook (Colab)
│
├── Scripts/
│   ├── detect_and_classify.py         # Main inference script
│   ├── evaluate_model.py              # VGG16-only evaluation + metrics
│   ├── evaluate_pipeline.py           # Full YOLO+VGG16 pipeline evaluation
│   ├── latency_benchmark.py           # VGG16 latency benchmarking
│   ├── plot_training_curves.py        # Plot training curves from saved history
│   └── check_dataset.py              # Dataset distribution checker
│
├── Evaluation/
│   ├── classification_report.txt
│   ├── metrics_summary.txt
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── confidence_distribution.png
│   ├── calibration_curve.png
│   ├── per_class_metrics.png
│   ├── misclassified_samples.png
│   ├── correct_samples.png
│   ├── gradcam.png
│   ├── threshold_analysis.png
│   └── Pipeline/
│       ├── pipeline_metrics_summary.txt
│       ├── pipeline_latency_benchmark.txt
│       ├── pipeline_confusion_matrix.png
│       ├── pipeline_roc_curve.png
│       ├── pipeline_pr_curve.png
│       ├── pipeline_latency.png
│       └── pipeline_latency_distribution.png
│
├── TestImages/                        # Drop test images here
├── Outputs/                           # Annotated result images saved here
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install ultralytics tensorflow opencv-python matplotlib seaborn scikit-learn
```

> **Note:** This project was developed and tested on a CPU (AMD GPU, MSI Bravo 15).
> For GPU-accelerated inference, a CUDA-compatible NVIDIA GPU is recommended.
> VGG16 training was performed on Google Colab (T4 GPU).

---

## Dataset Structure

VGG16 is trained as a **binary classifier** — Fresh vs Rotten — regardless of fruit/vegetable type, since the YOLO model already identifies the produce category.

Place your cropped fresh/rotten images in:

```
Dataset/
├── train/
│   ├── fresh/     ← fresh produce crops
│   └── rotten/    ← rotten produce crops
├── val/
│   ├── fresh/
│   └── rotten/
└── test/
    ├── fresh/
    └── rotten/
```

Run `check_dataset.py` to verify image counts per split:

```bash
python Scripts/check_dataset.py
```

---

## Model Weights

Availaible in Models folder.
YOLOv8s: `best.pt`
VGG16: `freshness_vgg16.keras`

---

## Usage

### Single image inference

```bash
python Scripts/detect_and_classify.py
```

Edit the `test_image` path at the bottom of the script to point to your image.
Annotated output is saved to `Outputs/`.

### Batch inference (entire folder)

In `detect_and_classify.py`, comment out the single image call and uncomment:

```python
process_folder()
```

This processes all images in `TestImages/` and saves results to `Outputs/`.

### Output format

Each detected object is labeled on-screen as:

```
Apple-Fresh (96.4100%)
Banana-Rotten (88.2300%)
Others (0.31)          ← low confidence, skipped from VGG16
```

- **Green box** → Fresh
- **Orange box** → Rotten
- **Red box** → Others (unrecognized or low-confidence YOLO detection)

---

## Results

### VGG16 Classifier (standalone)

| Metric | Score |
|--------|-------|
| Accuracy | *0.9925* |
| Precision | *0.9925* |
| Recall | *0.9925* |
| F1 Score | *0.9924* |
| ROC AUC | *0.9990* |

### YOLO + VGG16 Pipeline

| Metric | Score |
|--------|-------|
| Accuracy | *0.9440* |
| F1 Score | *0.9439* |
| ROC AUC | *0.9739* |

### Latency (CPU)

| Stage | Mean Latency |
|-------|-------------|
| YOLO Detection | *508.1863* ms|
| VGG16 Classification | *355.2361* ms|
| Total Pipeline | *866.5437* ms|
| Throughput | *1.15* FPS |

> Run `Scripts/evaluate_model.py`, `Scripts/evaluate_pipeline.py`, and
> `Scripts/latency_benchmark.py` to generate all metrics and fill in the table above.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `detect_and_classify.py` | Main inference — runs full YOLO+VGG16 pipeline on images |
| `train_vgg16_final.ipynb` | VGG16 training notebook (designed for Google Colab T4 GPU) |
| `evaluate_model.py` | Standalone VGG16 evaluation: confusion matrix, ROC, PR curve, Grad-CAM, etc. |
| `evaluate_pipeline.py` | End-to-end pipeline evaluation + latency benchmarking |
| `latency_benchmark.py` | VGG16-only latency benchmark across batch sizes |
| `plot_training_curves.py` | Plots Stage 1 / Stage 2 / combined training curves from saved JSON history |
| `check_dataset.py` | Prints image counts per class per split |

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [TensorFlow / Keras](https://www.tensorflow.org/)
- [VGG16 — Very Deep Convolutional Networks (Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)
