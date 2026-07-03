import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ================================================
# PATHS
# ================================================
PROJECT_ROOT = r"C:\Users\Ruby\Freshness-classification-of-Fruits-or-Vegetables-using-YOLOv8s-and-VGG16"

YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "Models", "best.pt")
VGG_WEIGHTS  = os.path.join(PROJECT_ROOT, "Models", "freshness_vgg16.keras")

# Test image folder structure:
# PipelineTest\
#   fresh\      <- images that should be classified Fresh
#   rotten\     <- images that should be classified Rotten
TEST_DIR  = os.path.join(PROJECT_ROOT, "Dataset", "test")
EVAL_DIR  = os.path.join(PROJECT_ROOT, "Evaluation", "Pipeline")
os.makedirs(EVAL_DIR, exist_ok=True)

CONF_THRESHOLD = 0.5
IMG_SIZE_VGG   = (224, 224)
CLASS_NAMES_YOLO = {0: "Apple", 1: "Banana", 2: "Carrot", 3: "Tomato", 4: "Mango", 5: "Potato"}
FRESHNESS_LABELS = {0: "Fresh", 1: "Rotten"}
WARMUP_RUNS  = 5
MEASURE_RUNS = 50

# ================================================
# LOAD MODELS
# ================================================
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading VGG16 model...")
vgg_model = load_model(VGG_WEIGHTS, compile=False)
print("Models loaded.\n")

# ================================================
# PIPELINE FUNCTIONS
# ================================================
def classify_crop(crop_bgr):
    crop_rgb     = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, IMG_SIZE_VGG)
    arr          = np.expand_dims(crop_resized.astype("float32"), axis=0)
    arr          = preprocess_input(arr)
    pred         = vgg_model.predict(arr, verbose=0)[0][0]
    label        = FRESHNESS_LABELS[1] if pred >= 0.5 else FRESHNESS_LABELS[0]
    conf         = pred if pred >= 0.5 else 1 - pred
    return label, float(conf), float(pred)


def run_pipeline(img_path):
    """
    Returns list of dicts, one per detected object:
    { fruit, freshness, freshness_conf, raw_prob, yolo_conf, recognized, bbox }
    """
    img     = cv2.imread(img_path)
    if img is None:
        return []
    results = yolo_model(img, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        recognized = cls_id in CLASS_NAMES_YOLO and conf >= CONF_THRESHOLD

        if not recognized:
            detections.append({
                "fruit": "Others", "freshness": None,
                "freshness_conf": None, "raw_prob": None,
                "yolo_conf": conf, "recognized": False,
                "bbox": (x1, y1, x2, y2)
            })
            continue

        crop = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0:
            continue

        freshness, f_conf, raw_prob = classify_crop(crop)
        detections.append({
            "fruit": CLASS_NAMES_YOLO[cls_id],
            "freshness": freshness,
            "freshness_conf": f_conf,
            "raw_prob": raw_prob,
            "yolo_conf": conf,
            "recognized": True,
            "bbox": (x1, y1, x2, y2)
        })

    return detections


def run_pipeline_timed(img_path):
    """Same as run_pipeline but returns (detections, timing_dict)."""
    img = cv2.imread(img_path)
    if img is None:
        return [], {}

    t0 = time.perf_counter()
    results = yolo_model(img, verbose=False)[0]
    t_yolo = (time.perf_counter() - t0) * 1000

    t_vgg_total = 0
    detections  = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        recognized = cls_id in CLASS_NAMES_YOLO and conf >= CONF_THRESHOLD

        if not recognized:
            detections.append({"fruit":"Others","freshness":None,"recognized":False})
            continue

        crop = img[max(y1,0):max(y2,0), max(x1,0):max(x2,0)]
        if crop.size == 0:
            continue

        t1 = time.perf_counter()
        freshness, f_conf, raw_prob = classify_crop(crop)
        t_vgg_total += (time.perf_counter() - t1) * 1000

        detections.append({
            "fruit": CLASS_NAMES_YOLO[cls_id],
            "freshness": freshness,
            "freshness_conf": f_conf,
            "raw_prob": raw_prob,
            "yolo_conf": conf,
            "recognized": True,
        })

    t_total = (time.perf_counter() - t0) * 1000

    return detections, {
        "yolo_ms":  t_yolo,
        "vgg_ms":   t_vgg_total,
        "total_ms": t_total,
    }

# ================================================
# COLLECT TEST IMAGES + GROUND TRUTH
# ================================================
print("Collecting test images...")
image_paths, y_true_labels = [], []
valid_ext = (".jpg", ".jpeg", ".png")

for cls_folder, true_label in [("fresh", 0), ("rotten", 1)]:
    folder = os.path.join(TEST_DIR, cls_folder)
    if not os.path.exists(folder):
        print(f"  WARNING: {folder} not found, skipping.")
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith(valid_ext):
            image_paths.append(os.path.join(folder, fname))
            y_true_labels.append(true_label)

print(f"Found {len(image_paths)} test images.\n")

if len(image_paths) == 0:
    raise FileNotFoundError(f"No images found in {TEST_DIR}. Check folder structure.")

# ================================================
# RUN PIPELINE ON ALL TEST IMAGES
# ================================================
print("Running pipeline on test set...")
y_true, y_pred, y_prob_list = [], [], []
skipped = 0

for img_path, true_label in zip(image_paths, y_true_labels):
    detections = run_pipeline(img_path)
    recognized = [d for d in detections if d["recognized"]]

    if not recognized:
        skipped += 1
        continue

    # use highest-confidence freshness detection
    best = max(recognized, key=lambda d: d["freshness_conf"])
    pred_label = 1 if best["freshness"] == "Rotten" else 0
    y_true.append(true_label)
    y_pred.append(pred_label)
    y_prob_list.append(best["raw_prob"])

y_true      = np.array(y_true)
y_pred      = np.array(y_pred)
y_prob_arr  = np.array(y_prob_list)

print(f"Evaluated : {len(y_true)} images")
print(f"Skipped   : {skipped} images (YOLO found no recognized objects)\n")

# ================================================
# METRICS
# ================================================
acc   = accuracy_score(y_true, y_pred)
bacc  = balanced_accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
rec   = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1    = f1_score(y_true, y_pred, average="weighted", zero_division=0)
mcc   = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

fpr_v, tpr_v, _ = roc_curve(y_true, y_prob_arr)
roc_auc          = auc(fpr_v, tpr_v)
prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob_arr)
pr_auc            = auc(rec_c, prec_c)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

report = classification_report(y_true, y_pred, target_names=["Fresh", "Rotten"], digits=4)
print("Classification Report:\n", report)

summary_lines = [
    "=" * 45,
    "YOLO + VGG16 PIPELINE EVALUATION",
    "=" * 45,
    f"Total images     : {len(image_paths)}",
    f"Evaluated        : {len(y_true)}",
    f"Skipped (no det) : {skipped}",
    "",
    f"Accuracy         : {acc:.4f}",
    f"Balanced Accuracy: {bacc:.4f}",
    f"Precision        : {prec:.4f}",
    f"Recall           : {rec:.4f}",
    f"F1 Score         : {f1:.4f}",
    f"Specificity      : {specificity:.4f}",
    f"Sensitivity      : {sensitivity:.4f}",
    f"MCC              : {mcc:.4f}",
    f"Cohen Kappa      : {kappa:.4f}",
    f"ROC AUC          : {roc_auc:.4f}",
    f"PR AUC           : {pr_auc:.4f}",
    "",
    f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
    "=" * 45,
]
summary = "\n".join(summary_lines)
print(summary)
with open(os.path.join(EVAL_DIR, "pipeline_metrics_summary.txt"), "w") as f:
    f.write(summary + "\n\n" + report)

# ================================================
# PLOT 1: Confusion matrix (counts + normalized)
# ================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fresh","Rotten"], yticklabels=["Fresh","Rotten"], ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".4f", cmap="Blues",
            xticklabels=["Fresh","Rotten"], yticklabels=["Fresh","Rotten"], ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.suptitle("YOLO + VGG16 Pipeline — Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_confusion_matrix.png"), dpi=150)
plt.show()
print("Saved: pipeline_confusion_matrix.png")

# ================================================
# PLOT 2: ROC Curve
# ================================================
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_v, tpr_v, color="darkorange", lw=2, label=f"YOLO+VGG16 (AUC = {roc_auc:.4f})")
ax.plot([0,1],[0,1], "navy", lw=1, linestyle="--", label="Random")
ax.fill_between(fpr_v, tpr_v, alpha=0.1, color="darkorange")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — YOLO+VGG16 Pipeline")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_roc_curve.png"), dpi=150)
plt.show()
print("Saved: pipeline_roc_curve.png")

# ================================================
# PLOT 3: Precision-Recall Curve
# ================================================
baseline = sum(y_true) / len(y_true)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_c, prec_c, color="green", lw=2, label=f"YOLO+VGG16 (PR AUC = {pr_auc:.4f})")
ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline:.4f})")
ax.fill_between(rec_c, prec_c, alpha=0.1, color="green")
ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve — YOLO+VGG16 Pipeline")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_pr_curve.png"), dpi=150)
plt.show()
print("Saved: pipeline_pr_curve.png")

# ================================================
# PLOT 4: Confidence distribution
# ================================================
fresh_scores  = y_prob_arr[y_true == 0]
rotten_scores = y_prob_arr[y_true == 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(fresh_scores,  bins=40, alpha=0.6, color="green",  label="Fresh (actual)")
ax.hist(rotten_scores, bins=40, alpha=0.6, color="orange", label="Rotten (actual)")
ax.axvline(0.5, color="red", linestyle="--", lw=2, label="Threshold (0.5)")
ax.set_xlabel("Predicted Probability (Rotten)")
ax.set_ylabel("Count")
ax.set_title("Confidence Distribution — YOLO+VGG16 Pipeline")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_confidence_distribution.png"), dpi=150)
plt.show()
print("Saved: pipeline_confidence_distribution.png")

# ================================================
# PLOT 5: Per-class metrics bar chart
# ================================================
metrics_per_class = {
    "Precision": [precision_score(y_true, y_pred, pos_label=0, zero_division=0),
                  precision_score(y_true, y_pred, pos_label=1, zero_division=0)],
    "Recall":    [recall_score(y_true, y_pred, pos_label=0, zero_division=0),
                  recall_score(y_true, y_pred, pos_label=1, zero_division=0)],
    "F1 Score":  [f1_score(y_true, y_pred, pos_label=0, zero_division=0),
                  f1_score(y_true, y_pred, pos_label=1, zero_division=0)],
}
x     = np.arange(2)
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
for i, (metric, values) in enumerate(metrics_per_class.items()):
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xticks(x + width); ax.set_xticklabels(["Fresh", "Rotten"])
ax.set_ylim(0, 1); ax.set_ylabel("Score")
ax.set_title("Per-Class Metrics — YOLO+VGG16 Pipeline")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_per_class_metrics.png"), dpi=150)
plt.show()
print("Saved: pipeline_per_class_metrics.png")

# ================================================
# LATENCY BENCHMARKING
# ================================================
print("\n\nRunning latency benchmark...")

# pick images for benchmarking — one from each class
bench_images = image_paths[:min(MEASURE_RUNS, len(image_paths))]

# warmup
print(f"Warming up ({WARMUP_RUNS} runs)...")
for img_path in bench_images[:WARMUP_RUNS]:
    run_pipeline(img_path)

# measure
print(f"Measuring latency ({MEASURE_RUNS} runs)...")
yolo_times, vgg_times, total_times = [], [], []

for img_path in bench_images[:MEASURE_RUNS]:
    _, timing = run_pipeline_timed(img_path)
    if timing:
        yolo_times.append(timing["yolo_ms"])
        vgg_times.append(timing["vgg_ms"])
        total_times.append(timing["total_ms"])

yolo_times  = np.array(yolo_times)
vgg_times   = np.array(vgg_times)
total_times = np.array(total_times)

def stats(arr):
    return {
        "mean":   np.mean(arr),
        "median": np.median(arr),
        "std":    np.std(arr),
        "min":    np.min(arr),
        "max":    np.max(arr),
        "p95":    np.percentile(arr, 95),
        "p99":    np.percentile(arr, 99),
        "fps":    1000 / np.mean(arr) if np.mean(arr) > 0 else 0,
    }

s_yolo  = stats(yolo_times)
s_vgg   = stats(vgg_times)
s_total = stats(total_times)

latency_lines = [
    "=" * 50,
    "YOLO + VGG16 PIPELINE LATENCY BENCHMARK",
    "=" * 50,
    f"Device       : CPU",
    f"Warmup runs  : {WARMUP_RUNS}",
    f"Measure runs : {MEASURE_RUNS}",
    "",
    f"{'Metric':<20} {'YOLO (ms)':<15} {'VGG16 (ms)':<15} {'Total (ms)':<15}",
    "-" * 65,
    f"{'Mean':<20} {s_yolo['mean']:<15.4f} {s_vgg['mean']:<15.4f} {s_total['mean']:<15.4f}",
    f"{'Median':<20} {s_yolo['median']:<15.4f} {s_vgg['median']:<15.4f} {s_total['median']:<15.4f}",
    f"{'Std Dev':<20} {s_yolo['std']:<15.4f} {s_vgg['std']:<15.4f} {s_total['std']:<15.4f}",
    f"{'Min':<20} {s_yolo['min']:<15.4f} {s_vgg['min']:<15.4f} {s_total['min']:<15.4f}",
    f"{'Max':<20} {s_yolo['max']:<15.4f} {s_vgg['max']:<15.4f} {s_total['max']:<15.4f}",
    f"{'P95':<20} {s_yolo['p95']:<15.4f} {s_vgg['p95']:<15.4f} {s_total['p95']:<15.4f}",
    f"{'P99':<20} {s_yolo['p99']:<15.4f} {s_vgg['p99']:<15.4f} {s_total['p99']:<15.4f}",
    f"{'FPS':<20} {s_yolo['fps']:<15.2f} {s_vgg['fps']:<15.2f} {s_total['fps']:<15.2f}",
    "=" * 50,
]
latency_summary = "\n".join(latency_lines)
print("\n" + latency_summary)
with open(os.path.join(EVAL_DIR, "pipeline_latency_benchmark.txt"), "w") as f:
    f.write(latency_summary)
print("Saved: pipeline_latency_benchmark.txt")

# ================================================
# PLOT 6: Latency breakdown per run
# ================================================
runs = np.arange(1, len(total_times) + 1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(runs, yolo_times,  label=f"YOLO (mean={s_yolo['mean']:.2f}ms)",  alpha=0.8)
axes[0].plot(runs, vgg_times,   label=f"VGG16 (mean={s_vgg['mean']:.2f}ms)",  alpha=0.8)
axes[0].plot(runs, total_times, label=f"Total (mean={s_total['mean']:.2f}ms)", alpha=0.8, lw=2)
axes[0].set_xlabel("Run"); axes[0].set_ylabel("Latency (ms)")
axes[0].set_title("Per-Run Latency: YOLO vs VGG16 vs Total")
axes[0].legend(); axes[0].grid(alpha=0.3)

# stacked bar: avg breakdown
components = ["YOLO Detection", "VGG16 Classification"]
values     = [s_yolo["mean"], s_vgg["mean"]]
colors     = ["steelblue", "orange"]
bars = axes[1].bar(components, values, color=colors, edgecolor="white", width=0.4)
axes[1].set_ylabel("Average Latency (ms)")
axes[1].set_title(f"Average Latency Breakdown\n(Total = {s_total['mean']:.2f} ms | FPS = {s_total['fps']:.2f})")
axes[1].set_ylim(0, max(values) * 1.3)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.2f} ms", ha="center", va="bottom", fontsize=11)
axes[1].grid(axis="y", alpha=0.3)

plt.suptitle("YOLO + VGG16 Pipeline Latency Analysis")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_latency.png"), dpi=150)
plt.show()
print("Saved: pipeline_latency.png")

# ================================================
# PLOT 7: Latency distribution histogram
# ================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, times, label, color in zip(
    axes,
    [yolo_times, vgg_times, total_times],
    ["YOLO", "VGG16", "Total Pipeline"],
    ["steelblue", "orange", "green"]
):
    s = stats(times)
    ax.hist(times, bins=20, color=color, edgecolor="white", alpha=0.85)
    ax.axvline(s["mean"],   color="red",    linestyle="--", lw=2, label=f"Mean ({s['mean']:.2f}ms)")
    ax.axvline(s["median"], color="black",  linestyle="--", lw=1, label=f"Median ({s['median']:.2f}ms)")
    ax.axvline(s["p95"],    color="purple", linestyle=":",  lw=1, label=f"P95 ({s['p95']:.2f}ms)")
    ax.set_xlabel("Latency (ms)"); ax.set_ylabel("Count")
    ax.set_title(f"{label} Latency Distribution")
    ax.legend(fontsize=7)

plt.suptitle("Latency Distributions — YOLO + VGG16 Pipeline")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "pipeline_latency_distribution.png"), dpi=150)
plt.show()
print("Saved: pipeline_latency_distribution.png")

# ================================================
# DONE
# ================================================
print("\n========== ALL PIPELINE OUTPUTS SAVED ==========")
outputs = [
    "pipeline_metrics_summary.txt",
    "pipeline_latency_benchmark.txt",
    "pipeline_confusion_matrix.png",
    "pipeline_roc_curve.png",
    "pipeline_pr_curve.png",
    "pipeline_confidence_distribution.png",
    "pipeline_per_class_metrics.png",
    "pipeline_latency.png",
    "pipeline_latency_distribution.png",
]
for o in outputs:
    print(f"  {EVAL_DIR}\\{o}")
