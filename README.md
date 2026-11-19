# image-degradation-nn
Evaluating the robustness of a Convolutional Neural Network (CNN) trained on high-quality images and tested on low-resolution, noisy images.

## Repository Structure
```text
image-degradation-nn/
├── NWPU-RESISC45/          # Local dataset folder (ignored by Git)
│   ├── airplane/
│   │   └── airplane_001.jpg
│   ├── airport/
│   └── ... (45 class folders total, 31,500 images)
├── create_splits.py        # Generates deterministic stratified splits
├── splits.json             # Saved train/val/test partitions (seed=42)
├── dataset.py              # PyTorch Dataset, degradations, and DataLoader helpers
└── README.md
```

---

## 1. Dataset Setup (Download & Extract)
To keep the repository lightweight, raw images are not stored in GitHub. Follow these steps to set up the dataset locally:

1. **Download:** Go to [isaaccorley/resisc45 on Hugging Face](https://huggingface.co/datasets/isaaccorley/resisc45/tree/main) and download `NWPU-RESISC45.zip`.
2. **Extract:** Unzip `NWPU-RESISC45.zip` into the root of this repository so that `NWPU-RESISC45/` sits alongside `splits.json` and `dataset.py`.
3. **Verify `.gitignore`:** Ensure `NWPU-RESISC45/` is listed in `.gitignore` so raw `.jpg` files are never committed to GitHub.

## 2. Reproducible Splits (`splits.json`)
You do **not** need to re-run `create_splits.py` unless you want to verify split generation—the finalized `splits.json` file is already committed to this repository.
- **Splitting Method:** Stratified split generated via `create_splits.py` (`seed=42`).
- **Partition Sizes:**
  - **Train:** 70% (22,050 images — 490 per class)
  - **Validation:** 15% (4,725 images — 105 per class)
  - **Test:** 15% (4,725 images — 105 per class)

## 3. Data Loading & Test-Time Degradations (`dataset.py`)
`dataset.py` reads `splits.json`, converts images to tensors, applies optional test-time degradations on the `[0, 1]` scale, and normalizes using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### Quick Usage for Training & Evaluation
```python
from dataset import get_dataloader

SPLITS_PATH = "splits.json"
DATA_DIR = "NWPU-RESISC45"

# 1. Clean training and validation loaders
train_loader = get_dataloader(SPLITS_PATH, DATA_DIR, split="train", batch_size=64)
val_loader = get_dataloader(SPLITS_PATH, DATA_DIR, split="val", batch_size=64)

# 2. Clean test baseline
clean_test_loader = get_dataloader(SPLITS_PATH, DATA_DIR, split="test")

# 3. Single degradations (RQ1 & RQ2)
res_50_loader = get_dataloader(SPLITS_PATH, DATA_DIR, split="test", res_scale=0.5)
gaussian_loader = get_dataloader(
    SPLITS_PATH, DATA_DIR, split="test", noise_type="gaussian", noise_level=0.1
)
sp_loader = get_dataloader(
    SPLITS_PATH, DATA_DIR, split="test", noise_type="salt_and_pepper", noise_level=0.05
)

# 4. Combined degradation (resolution + noise)
combined_loader = get_dataloader(
    SPLITS_PATH,
    DATA_DIR,
    split="test",
    res_scale=0.5,
    noise_type="gaussian",
    noise_level=0.1,
)
```

### Supported Degradation Parameters
- **`res_scale`** (`float`, default `1.0`): Downsamples via bilinear interpolation and resizes back to `256×256`. Supported values:
  - `1.0` — Native resolution (`256×256`)
  - `0.75` — 25% reduction (`192×192`)
  - `0.5` — 50% reduction (`128×128`)
  - `0.25` — 75% reduction (`64×64`)
- **`noise_type`** (`str | None`, default `None`): `"gaussian"` or `"salt_and_pepper"`.
- **`noise_level`** (`float`, default `0.0`): Standard deviation $\sigma$ for Gaussian noise, or pixel corruption probability $p$ for salt-and-pepper noise. Noise is seeded deterministically per sample (`seed + idx`) so every test evaluation sees identical corruptions.

---
## Environment Setup
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```
2. **Install PyTorch:**
   - **NVIDIA GPU (Recommended):** Install the CUDA-enabled PyTorch build matching your GPU (e.g., CUDA 12.8 for RTX 50-series):
     ```bash
     pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
     ```
   - **CPU / macOS:** Skip this step; `requirements.txt` will install the standard build automatically.
3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```