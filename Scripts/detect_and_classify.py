import os
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = r"C:\Users\Ruby\Freshness classification of Fruits or Vegetables using YOLOv8s and VGG16"

YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "Models", "best.pt")
VGG_WEIGHTS  = os.path.join(PROJECT_ROOT, "Models", "freshness_vgg16.keras")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "Outputs")
TEST_IMG_DIR = os.path.join(PROJECT_ROOT, "TestImages")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
CONF_THRESHOLD = 0.5     # below this -> "Others", skipped from VGG16
IMG_SIZE_VGG = (224, 224)

CLASS_NAMES = {
    0: "Apple",
    1: "Banana",
    2: "Carrot",
    3: "Tomato",
    4: "Mango",
    5: "Potato",
}

FRESHNESS_LABELS = {0: "Fresh", 1: "Rotten"}  # matches classes=["fresh","rotten"] from training

# -----------------------------
# Load models once
# -----------------------------
print("Loading YOLO weights from:", YOLO_WEIGHTS)
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading VGG16 weights from:", VGG_WEIGHTS)
vgg_model = load_model(VGG_WEIGHTS, compile=False)


def classify_crop(crop_bgr):
    """Run VGG16 freshness classification on a single cropped BGR image."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_resized = cv2.resize(crop_rgb, IMG_SIZE_VGG)
    arr = np.expand_dims(crop_resized.astype("float32"), axis=0)
    arr = preprocess_input(arr)

    pred = vgg_model.predict(arr, verbose=0)[0][0]
    label = FRESHNESS_LABELS[1] if pred >= 0.5 else FRESHNESS_LABELS[0]
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, float(confidence)


def process_image(img_path, save_path=None, show=False):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    results = yolo_model(img, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # -----------------------------
        # Confidence / class gate
        # -----------------------------
        recognized = cls_id in CLASS_NAMES and conf >= CONF_THRESHOLD

        if not recognized:
            label_text = "Others"
            color = (0, 0, 255)  # red box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{label_text} ({conf:.2f})", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            continue  # skip VGG16 entirely

        # -----------------------------
        # Recognized & confident -> crop & classify
        # -----------------------------
        crop = img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        if crop.size == 0:
            continue

        fruit_name = CLASS_NAMES[cls_id]
        freshness_label, freshness_conf = classify_crop(crop)

        final_label = f"{fruit_name}-{freshness_label} ({freshness_conf*100:.1f}%)"
        color = (0, 255, 0) if freshness_label == "Fresh" else (0, 165, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, final_label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        print(f"Detected: {fruit_name} -> {freshness_label} (freshness confidence: {freshness_conf*100:.2f}%)")

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"Saved annotated image to {save_path}")

    if show:
        cv2.imshow("Detection + Freshness", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def process_folder(folder_path=TEST_IMG_DIR, output_dir=OUTPUT_DIR):
    """Batch process every image in a folder."""
    valid_ext = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_ext)]

    if not images:
        print(f"No images found in {folder_path}")
        return

    for fname in images:
        img_path = os.path.join(folder_path, fname)
        save_path = os.path.join(output_dir, f"result_{fname}")
        print(f"\n--- Processing {fname} ---")
        process_image(img_path, save_path=save_path, show=False)


if __name__ == "__main__":
    # Single image test (set show=True to pop up a window)
    test_image = os.path.join(TEST_IMG_DIR, "14.jpg")
    if os.path.exists(test_image):
        process_image(test_image, save_path=os.path.join(OUTPUT_DIR, "result_test14.jpg"), show=True)

    # Or batch process the whole TestImages folder:
    # process_folder()