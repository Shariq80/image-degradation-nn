# image-degradation-nn
Evaluating the robustness of a Convolutional Neural Network (CNN) trained on high-quality images and tested on low-resolution, noisy images.

## 1. Dataset Setup (Download & Extract)
To keep the repository lightweight, the raw images are not stored in GitHub. Follow these steps to set up the dataset locally:

1. **Download:** Go to [isaaccorley/resisc45 on Hugging Face](https://huggingface.co/datasets/isaaccorley/resisc45/tree/main) and download `NWPU-RESISC45.zip`.
2. **Extract:** Unzip `NWPU-RESISC45.zip` into your local workspace so the folder structure looks like this:
   ```text
   NWPU-RESISC45/
   ├── airplane/
   │   ├── airplane_001.jpg
   ├── airport/
   └── ... (45 class folders total, 31,500 images)
   ├── create_splits.py
   ├── splits.json
   └── README.md
   ```
3. **Note on `.gitignore`:** Make sure `NWPU-RESISC45/` is listed in `.gitignore` so you do not accidentally commit raw `.jpg` files to GitHub.


## 2. Reproducible Splits (`splits.json`)
You do **not** need to re-run `create_splits.py` unless you want to verify the split generation—the finalized split file (`splits.json`) is already committed to this repository.
- **Splitting Method:** Stratified split generated via `create_splits.py` using `seed=42`.
- **Partition Sizes:**
  - **Train:** 70% (22,050 images — 490 per class)
  - **Validation:** 15% (4,725 images — 105 per class)
  - **Test:** 15% (4,725 images — 105 per class)

### How to Load `splits.json`
Each entry in `splits.json` stores a relative image path (`"class_name/filename.jpg"`) and an integer class ID (`0`–`44`). Combine `DATA_DIR` with the relative path on your machine:

```python
import json
from pathlib import Path

DATA_DIR = Path("path/to/NWPU-RESISC45")

with open("splits.json", "r") as f:
  splits = json.load(f)

train_records = splits["train"]  # List of {"path": ..., "label": ...}
val_records = splits["val"]
test_records = splits["test"]
class_names = splits["class_names"]  # Index 0 to 44

# Example: full path to the first training image
first_img_path = DATA_DIR / train_records["path"]
first_img_label = train_records["label"]
```