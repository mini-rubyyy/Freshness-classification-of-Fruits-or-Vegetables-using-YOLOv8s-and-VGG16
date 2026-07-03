import os
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# ================================================
# PATHS
# ================================================
PROJECT_ROOT = r"C:\Users\Ruby\Freshness-classification-of-Fruits-or-Vegetables-using-YOLOv8s-and-VGG16"
MODEL_PATH   = os.path.join(PROJECT_ROOT, "Models", "freshness_vgg16.keras")
EVAL_DIR     = os.path.join(PROJECT_ROOT, "Evaluation")
os.makedirs(EVAL_DIR, exist_ok=True)

IMG_SIZE     = (224, 224)
WARMUP_RUNS  = 10    # warmup runs before measuring (lets TF optimize its graph)
MEASURE_RUNS = 100   # number of runs to measure latency over

# ================================================
# LOAD MODEL
# ================================================
print("Loading model...")
model = load_model(MODEL_PATH, compile=False)
print("Model loaded.\n")

# ================================================
# CREATE DUMMY INPUT (single image)
# ================================================
dummy_input = np.random.randint(0, 255, (1, 224, 224, 3)).astype("float32")
dummy_input = preprocess_input(dummy_input)

# ================================================
# WARMUP RUNS (not measured)
# ================================================
print(f"Warming up ({WARMUP_RUNS} runs)...")
for _ in range(WARMUP_RUNS):
    _ = model.predict(dummy_input, verbose=0)

# ================================================
# SINGLE IMAGE LATENCY
# ================================================
print(f"Measuring single-image latency ({MEASURE_RUNS} runs)...")
latencies = []
for _ in range(MEASURE_RUNS):
    start = time.perf_counter()
    _ = model.predict(dummy_input, verbose=0)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)  # ms

latencies = np.array(latencies)

mean_lat   = np.mean(latencies)
median_lat = np.median(latencies)
std_lat    = np.std(latencies)
min_lat    = np.min(latencies)
max_lat    = np.max(latencies)
p95_lat    = np.percentile(latencies, 95)
p99_lat    = np.percentile(latencies, 99)
fps        = 1000 / mean_lat

print("\n========== SINGLE IMAGE LATENCY ==========")
print(f"Mean latency    : {mean_lat:.4f} ms")
print(f"Median latency  : {median_lat:.4f} ms")
print(f"Std deviation   : {std_lat:.4f} ms")
print(f"Min latency     : {min_lat:.4f} ms")
print(f"Max latency     : {max_lat:.4f} ms")
print(f"95th percentile : {p95_lat:.4f} ms")
print(f"99th percentile : {p99_lat:.4f} ms")
print(f"Throughput      : {fps:.2f} FPS")

# ================================================
# BATCH LATENCY
# ================================================
batch_sizes = [1, 2, 4, 8, 16, 32]
batch_results = {}

print("\nMeasuring batch latency...")
for bs in batch_sizes:
    batch_input = np.random.randint(0, 255, (bs, 224, 224, 3)).astype("float32")
    batch_input = preprocess_input(batch_input)

    # warmup
    for _ in range(5):
        _ = model.predict(batch_input, verbose=0)

    times = []
    for _ in range(30):
        start = time.perf_counter()
        _ = model.predict(batch_input, verbose=0)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg = np.mean(times)
    per_image = avg / bs
    batch_results[bs] = {"total_ms": avg, "per_image_ms": per_image, "fps": 1000 / per_image}
    print(f"  Batch {bs:>2}: total={avg:.2f}ms | per image={per_image:.4f}ms | FPS={1000/per_image:.2f}")

# ================================================
# SAVE SUMMARY TO FILE
# ================================================
summary_lines = [
    "=" * 45,
    "LATENCY BENCHMARK SUMMARY",
    "=" * 45,
    f"Device          : CPU",
    f"Warmup runs     : {WARMUP_RUNS}",
    f"Measure runs    : {MEASURE_RUNS}",
    "",
    "--- Single Image ---",
    f"Mean latency    : {mean_lat:.4f} ms",
    f"Median latency  : {median_lat:.4f} ms",
    f"Std deviation   : {std_lat:.4f} ms",
    f"Min latency     : {min_lat:.4f} ms",
    f"Max latency     : {max_lat:.4f} ms",
    f"95th percentile : {p95_lat:.4f} ms",
    f"99th percentile : {p99_lat:.4f} ms",
    f"Throughput      : {fps:.2f} FPS",
    "",
    "--- Batch Latency ---",
    f"{'Batch':<8} {'Total (ms)':<15} {'Per Image (ms)':<18} {'FPS':<10}",
    "-" * 51,
]
for bs, v in batch_results.items():
    summary_lines.append(f"{bs:<8} {v['total_ms']:<15.4f} {v['per_image_ms']:<18.4f} {v['fps']:<10.2f}")
summary_lines.append("=" * 45)

summary = "\n".join(summary_lines)
print("\n" + summary)
with open(os.path.join(EVAL_DIR, "latency_benchmark.txt"), "w") as f:
    f.write(summary)
print(f"\nSaved: latency_benchmark.txt")

# ================================================
# PLOT 1: Latency distribution (single image)
# ================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(latencies, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(mean_lat,   color="red",    linestyle="--", lw=2, label=f"Mean ({mean_lat:.2f} ms)")
axes[0].axvline(median_lat, color="orange", linestyle="--", lw=2, label=f"Median ({median_lat:.2f} ms)")
axes[0].axvline(p95_lat,    color="purple", linestyle=":",  lw=2, label=f"P95 ({p95_lat:.2f} ms)")
axes[0].set_xlabel("Latency (ms)")
axes[0].set_ylabel("Count")
axes[0].set_title("Single Image Inference Latency Distribution")
axes[0].legend()

axes[1].plot(range(1, MEASURE_RUNS + 1), latencies, color="steelblue", alpha=0.7, lw=0.8)
axes[1].axhline(mean_lat, color="red", linestyle="--", lw=1.5, label=f"Mean ({mean_lat:.2f} ms)")
axes[1].set_xlabel("Run")
axes[1].set_ylabel("Latency (ms)")
axes[1].set_title("Latency Over Runs")
axes[1].legend()

plt.suptitle("VGG16 Inference Latency — Single Image")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "latency_single_image.png"), dpi=150)
plt.show()
print("Saved: latency_single_image.png")

# ================================================
# PLOT 2: Batch size vs latency / FPS
# ================================================
bs_list       = list(batch_results.keys())
total_ms_list = [batch_results[b]["total_ms"]    for b in bs_list]
per_img_list  = [batch_results[b]["per_image_ms"] for b in bs_list]
fps_list      = [batch_results[b]["fps"]          for b in bs_list]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(bs_list, total_ms_list, "o-", color="steelblue", lw=2)
axes[0].set_xlabel("Batch Size"); axes[0].set_ylabel("Total Latency (ms)")
axes[0].set_title("Batch Size vs Total Latency"); axes[0].grid(alpha=0.3)

axes[1].plot(bs_list, per_img_list, "o-", color="green", lw=2)
axes[1].set_xlabel("Batch Size"); axes[1].set_ylabel("Per Image Latency (ms)")
axes[1].set_title("Batch Size vs Per-Image Latency"); axes[1].grid(alpha=0.3)

axes[2].plot(bs_list, fps_list, "o-", color="orange", lw=2)
axes[2].set_xlabel("Batch Size"); axes[2].set_ylabel("FPS")
axes[2].set_title("Batch Size vs Throughput (FPS)"); axes[2].grid(alpha=0.3)

plt.suptitle("VGG16 Inference Latency — Batch Analysis")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "latency_batch.png"), dpi=150)
plt.show()
print("Saved: latency_batch.png")

print("\nAll latency results saved to:", EVAL_DIR)
