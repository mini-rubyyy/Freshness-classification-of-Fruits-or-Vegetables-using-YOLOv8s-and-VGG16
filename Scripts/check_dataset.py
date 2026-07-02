import os

PROJECT_ROOT = r"C:\Users\Ruby\Freshness classification of Fruits or Vegetables using YOLOv8s and VGG16"
DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")

SPLITS = ["train", "val", "test"]
CLASSES = ["fresh", "rotten"]

print("\nDATASET DISTRIBUTION\n")

for split in SPLITS:
    print(f"--- {split.upper()} ---")
    for cls in CLASSES:
        path = os.path.join(DATASET_DIR, split, cls)
        if not os.path.exists(path):
            print(f"{cls}: 0 (folder missing -> create it: {path})")
            continue

        count = len([
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ])
        print(f"{cls}: {count}")
    print()
