import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
import random
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ================================================
# PATHS
# ================================================
PROJECT_ROOT = r"C:\\Users\\Ruby\\Freshness-classification-of-Fruits-or-Vegetables-using-YOLOv8s-and-VGG16"

MODEL_PATH  = os.path.join(PROJECT_ROOT, "Models", "freshness_vgg16.keras")
TEST_DIR    = os.path.join(PROJECT_ROOT, "Dataset", "test")
EVAL_DIR    = os.path.join(PROJECT_ROOT, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASS_NAMES = ["Fresh", "Rotten"]

# ================================================
# LOAD MODEL + PREDICTIONS
# ================================================
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)

gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_data = gen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode="binary", classes=["fresh", "rotten"], shuffle=False,
)

print("\nRunning predictions...")
y_prob = model.predict(test_data, verbose=1).flatten()
y_pred = (y_prob >= 0.5).astype(int)
y_true = test_data.classes
filenames = test_data.filenames

# ================================================
# 1) CLASSIFICATION REPORT
# ================================================
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
print("\nClassification Report:\n", report)
with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ================================================
# 2) EXTENDED METRICS SUMMARY
# ================================================
acc   = accuracy_score(y_true, y_pred)
bacc  = balanced_accuracy_score(y_true, y_pred)
prec  = precision_score(y_true, y_pred, average="weighted")
rec   = recall_score(y_true, y_pred, average="weighted")
f1    = f1_score(y_true, y_pred, average="weighted")
mcc   = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
fpr_v, tpr_v, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr_v, tpr_v)
prec_c, rec_c, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(rec_c, prec_c)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

summary_lines = [
    "=" * 40,
    "EVALUATION SUMMARY",
    "=" * 40,
    f"Accuracy           : {acc:.4f}",
    f"Balanced Accuracy  : {bacc:.4f}",
    f"Precision (weighted): {prec:.4f}",
    f"Recall (weighted)  : {rec:.4f}",
    f"F1 Score (weighted): {f1:.4f}",
    f"Specificity        : {specificity:.4f}",
    f"Sensitivity        : {sensitivity:.4f}",
    f"MCC                : {mcc:.4f}",
    f"Cohen Kappa        : {kappa:.4f}",
    f"ROC AUC            : {roc_auc:.4f}",
    f"PR AUC             : {pr_auc:.4f}",
    "=" * 40,
    f"TP={tp}  TN={tn}  FP={fp}  FN={fn}",
    "=" * 40,
]
summary = "\n".join(summary_lines)
print("\n" + summary)
with open(os.path.join(EVAL_DIR, "metrics_summary.txt"), "w") as f:
    f.write(summary)

# ================================================
# 3) CONFUSION MATRIX (normalized + raw side by side)
# ================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt=".4f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[1])
axes[1].set_title("Confusion Matrix (Normalized)")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

plt.suptitle("Confusion Matrices")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"), dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# ================================================
# 4) ROC CURVE
# ================================================
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_v, tpr_v, color="darkorange", lw=2, label=f"VGG16 (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random classifier")
ax.fill_between(fpr_v, tpr_v, alpha=0.1, color="darkorange")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "roc_curve.png"), dpi=150)
plt.show()
print("Saved: roc_curve.png")

# ================================================
# 5) PRECISION-RECALL CURVE
# ================================================
baseline = sum(y_true) / len(y_true)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(rec_c, prec_c, color="green", lw=2, label=f"VGG16 (PR AUC = {pr_auc:.4f})")
ax.axhline(baseline, color="red", linestyle="--", label=f"Baseline ({baseline:.4f})")
ax.fill_between(rec_c, prec_c, alpha=0.1, color="green")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "precision_recall_curve.png"), dpi=150)
plt.show()
print("Saved: precision_recall_curve.png")

# ================================================
# 6) CONFIDENCE DISTRIBUTION
# ================================================
fresh_scores  = y_prob[y_true == 0]
rotten_scores = y_prob[y_true == 1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(fresh_scores,  bins=50, alpha=0.6, color="green",  label="Fresh (actual)")
ax.hist(rotten_scores, bins=50, alpha=0.6, color="orange", label="Rotten (actual)")
ax.axvline(0.5, color="red", linestyle="--", lw=2, label="Decision threshold (0.5)")
ax.set_xlabel("Predicted Probability (Rotten)"); ax.set_ylabel("Count")
ax.set_title("Prediction Confidence Distribution"); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confidence_distribution.png"), dpi=150)
plt.show()
print("Saved: confidence_distribution.png")

# ================================================
# 7) CALIBRATION CURVE (reliability diagram)
# ================================================
fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(mean_pred, fraction_pos, "s-", color="blue", label="VGG16")
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
ax.set_xlabel("Mean Predicted Probability")
ax.set_ylabel("Fraction of Positives")
ax.set_title("Calibration Curve (Reliability Diagram)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "calibration_curve.png"), dpi=150)
plt.show()
print("Saved: calibration_curve.png")

# ================================================
# 8) PER-CLASS METRICS BAR CHART
# ================================================
from sklearn.metrics import precision_score, recall_score, f1_score
metrics_per_class = {
    "Precision": [
        precision_score(y_true, y_pred, pos_label=0),
        precision_score(y_true, y_pred, pos_label=1),
    ],
    "Recall": [
        recall_score(y_true, y_pred, pos_label=0),
        recall_score(y_true, y_pred, pos_label=1),
    ],
    "F1 Score": [
        f1_score(y_true, y_pred, pos_label=0),
        f1_score(y_true, y_pred, pos_label=1),
    ],
}
x = np.arange(len(CLASS_NAMES))
width = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
for i, (metric, values) in enumerate(metrics_per_class.items()):
    ax.bar(x + i * width, values, width, label=metric)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES)
ax.set_ylim(0, 1)
ax.set_ylabel("Score"); ax.set_title("Per-Class Metrics")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "per_class_metrics.png"), dpi=150)
plt.show()
print("Saved: per_class_metrics.png")

# ================================================
# 9) MISCLASSIFIED SAMPLES GRID
# ================================================
misclassified_idx = np.where(y_pred != y_true)[0]
sample_idx = misclassified_idx[:16] if len(misclassified_idx) >= 16 else misclassified_idx

if len(sample_idx) > 0:
    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    axes = axes.flatten()
    for i, idx in enumerate(sample_idx):
        img_path = os.path.join(TEST_DIR, filenames[idx])
        img = load_img(img_path, target_size=IMG_SIZE)
        axes[i].imshow(img)
        true_label  = CLASS_NAMES[y_true[idx]]
        pred_label  = CLASS_NAMES[y_pred[idx]]
        conf        = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
        axes[i].set_title(
            f"True: {true_label}\nPred: {pred_label} ({conf*100:.2f}%)",
            fontsize=8, color="red"
        )
        axes[i].axis("off")
    for j in range(len(sample_idx), 16):
        axes[j].axis("off")
    plt.suptitle("Misclassified Samples (up to 16)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, "misclassified_samples.png"), dpi=150)
    plt.show()
    print("Saved: misclassified_samples.png")
else:
    print("No misclassified samples found.")

# ================================================
# 10) CORRECT PREDICTIONS SAMPLE GRID
# ================================================
correct_idx = np.where(y_pred == y_true)[0]
sample_correct = correct_idx[:16] if len(correct_idx) >= 16 else correct_idx

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes = axes.flatten()
for i, idx in enumerate(sample_correct):
    img_path = os.path.join(TEST_DIR, filenames[idx])
    img = load_img(img_path, target_size=IMG_SIZE)
    axes[i].imshow(img)
    true_label = CLASS_NAMES[y_true[idx]]
    conf       = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
    axes[i].set_title(
        f"{true_label} ({conf*100:.2f}%)",
        fontsize=8, color="green"
    )
    axes[i].axis("off")
for j in range(len(sample_correct), 16):
    axes[j].axis("off")
plt.suptitle("Correctly Classified Samples (up to 16)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "correct_samples.png"), dpi=150)
plt.show()
print("Saved: correct_samples.png")

# ================================================
# 11) GRAD-CAM VISUALIZATION (what the model focuses on)
# ================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3"):
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

# pick 8 random test images for Grad-CAM
sample_for_gradcam = random.sample(range(len(filenames)), min(8, len(filenames)))
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes = axes.flatten()

for i, idx in enumerate(sample_for_gradcam):
    img_path = os.path.join(TEST_DIR, filenames[idx])

    # original
    orig = load_img(img_path, target_size=IMG_SIZE)
    axes[i * 2].imshow(orig)
    axes[i * 2].set_title(
        f"Original\n{CLASS_NAMES[y_true[idx]]}",
        fontsize=8
    )
    axes[i * 2].axis("off")

    # grad-cam
    arr = img_to_array(orig)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    heatmap = make_gradcam_heatmap(arr, model)
    overlay = overlay_gradcam(img_path, heatmap)
    pred_label = CLASS_NAMES[y_pred[idx]]
    conf = y_prob[idx] if y_pred[idx] == 1 else 1 - y_prob[idx]
    axes[i * 2 + 1].imshow(overlay)
    axes[i * 2 + 1].set_title(
        f"Grad-CAM\nPred: {pred_label} ({conf*100:.2f}%)",
        fontsize=8,
        color="green" if y_pred[idx] == y_true[idx] else "red"
    )
    axes[i * 2 + 1].axis("off")

plt.suptitle("Grad-CAM — Model Attention Visualization", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "gradcam.png"), dpi=150)
plt.show()
print("Saved: gradcam.png")

# ================================================
# 12) THRESHOLD ANALYSIS
# ================================================
thresholds = np.arange(0.1, 1.0, 0.05)
metrics_at_thresh = {"threshold": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "specificity": []}
for t in thresholds:
    yp = (y_prob >= t).astype(int)
    cm_t = confusion_matrix(y_true, yp)
    if cm_t.shape == (2, 2):
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
        metrics_at_thresh["threshold"].append(t)
        metrics_at_thresh["accuracy"].append(accuracy_score(y_true, yp))
        metrics_at_thresh["precision"].append(precision_score(y_true, yp, zero_division=0))
        metrics_at_thresh["recall"].append(recall_score(y_true, yp, zero_division=0))
        metrics_at_thresh["f1"].append(f1_score(y_true, yp, zero_division=0))
        metrics_at_thresh["specificity"].append(tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0)

fig, ax = plt.subplots(figsize=(9, 5))
for metric in ["accuracy", "precision", "recall", "f1", "specificity"]:
    ax.plot(metrics_at_thresh["threshold"], metrics_at_thresh[metric], marker="o", markersize=3, label=metric.capitalize())
ax.axvline(0.5, color="black", linestyle="--", lw=1, label="Default threshold (0.5)")
ax.set_xlabel("Classification Threshold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Metrics vs Classification Threshold")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "threshold_analysis.png"), dpi=150)
plt.show()
print("Saved: threshold_analysis.png")

# ================================================
# DONE
# ================================================
print("\n========== ALL OUTPUTS SAVED ==========")
outputs = [
    "classification_report.txt",
    "metrics_summary.txt",
    "confusion_matrix.png",
    "roc_curve.png",
    "precision_recall_curve.png",
    "confidence_distribution.png",
    "calibration_curve.png",
    "per_class_metrics.png",
    "misclassified_samples.png",
    "correct_samples.png",
    "gradcam.png",
    "threshold_analysis.png",
]
for o in outputs:
    print(f"  {EVAL_DIR}\\{o}")