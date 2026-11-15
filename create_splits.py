import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# 1. Point this to your extracted NWPU-RESISC45 folder
# 1. Relative path to the extracted dataset folder in the project root
DATA_DIR = Path("NWPU-RESISC45")

if not DATA_DIR.exists():
  raise FileNotFoundError(
      f"Dataset folder not found at '{DATA_DIR.resolve()}'. "
      "Please download and extract NWPU-RESISC45.zip into the project root."
  )

# 2. Scan all 45 class folders in sorted order for consistent class IDs (0-44)
class_folders = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
class_names = [d.name for d in class_folders]
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

print(f"Found {len(class_names)} classes.")

# 3. Collect all relative image paths and their integer labels
samples = []  # List of {"path": "airplane/airplane_001.jpg", "label": 0}
labels = []

for cls_name in class_names:
  cls_dir = DATA_DIR / cls_name
  # Sort filenames so the order is 100% deterministic across Windows/Mac/Linux
  img_files = sorted(
      [f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg")]
  )
  for img_path in img_files:
    rel_path = f"{cls_name}/{img_path.name}"
    label_id = class_to_idx[cls_name]
    samples.append({"path": rel_path, "label": label_id})
    labels.append(label_id)

total_images = len(samples)
print(f"Total images found: {total_images}")  # Expected: 31,500

# 4. Perform Stratified 70% Train / 30% Temp Split
train_samples, temp_samples, _, temp_labels = train_test_split(
    samples, labels, test_size=0.30, random_state=42, stratify=labels
)

# 5. Split the remaining 30% equally into 15% Validation and 15% Test
val_samples, test_samples = train_test_split(
    temp_samples, test_size=0.50, random_state=42, stratify=temp_labels
)

# 6. Verify split sizes
print(
    f"Train size: {len(train_samples)} ({len(train_samples)/total_images:.1%})"
)  # 22,050 (70.0%)
print(
    f"Val size:   {len(val_samples)} ({len(val_samples)/total_images:.1%})"
)  # 4,725 (15.0%)
print(
    f"Test size:  {len(test_samples)} ({len(test_samples)/total_images:.1%})"
)  # 4,725 (15.0%)

# 7. Save to splits.json (uses relative paths so it works on your friends' computers too!)
splits_data = {
    "seed": 42,
    "num_classes": len(class_names),
    "class_names": class_names,
    "class_to_idx": class_to_idx,
    "train": train_samples,
    "val": val_samples,
    "test": test_samples,
}

with open("splits.json", "w") as f:
  json.dump(splits_data, f, indent=2)

print("Successfully saved splits.json!")
