import json
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NATIVE_SIZE = 256


class DegradationTransform:
    """Applies resolution reduction and/or synthetic noise to [0, 1] image tensors."""

    def __init__(
        self,
        res_scale: float = 1.0,
        noise_type: Optional[Literal["gaussian", "salt_and_pepper"]] = None,
        noise_level: float = 0.0,
        seed: int = 42,
    ):
        if res_scale not in (1.0, 0.75, 0.5, 0.25):
            raise ValueError(f"res_scale must be one of (1.0, 0.75, 0.5, 0.25), got {res_scale}")
        if noise_type not in (None, "gaussian", "salt_and_pepper"):
            raise ValueError(f"Unsupported noise_type: {noise_type}")
        if noise_level < 0.0:
            raise ValueError("noise_level must be >= 0.0")

        self.res_scale = res_scale
        self.noise_type = noise_type
        self.noise_level = noise_level
        self.seed = seed

    def __call__(self, tensor: torch.Tensor, index: int = 0) -> torch.Tensor:
        # 1. Resolution degradation (downsample then upsample back to 256x256)
        if self.res_scale < 1.0:
            target_size = int(round(NATIVE_SIZE * self.res_scale))
            tensor = TF.resize(
                tensor,
                [target_size, target_size],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )
            tensor = TF.resize(
                tensor,
                [NATIVE_SIZE, NATIVE_SIZE],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            )

        # 2. Deterministic per-sample noise injection on [0, 1] range
        if self.noise_type is not None and self.noise_level > 0.0:
            gen = torch.Generator(device=tensor.device)
            gen.manual_seed(self.seed + index)

            if self.noise_type == "gaussian":
                noise = torch.randn(
                    tensor.shape, generator=gen, dtype=tensor.dtype, device=tensor.device
                ) * self.noise_level
                tensor = torch.clamp(tensor + noise, 0.0, 1.0)

            elif self.noise_type == "salt_and_pepper":
                # noise_level = probability p that a pixel is corrupted (half salt=1, half pepper=0)
                rand_map = torch.rand(
                    (1, tensor.shape[1], tensor.shape[2]),
                    generator=gen,
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                salt_mask = (rand_map < (self.noise_level / 2.0)).expand_as(tensor)
                pepper_mask = (
                    (rand_map >= (self.noise_level / 2.0)) & (rand_map < self.noise_level)
                ).expand_as(tensor)
                tensor = tensor.clone()
                tensor[salt_mask] = 1.0
                tensor[pepper_mask] = 0.0

        return tensor


class RESISC45Dataset(Dataset):
    """PyTorch Dataset reading stratified splits from splits.json."""

    def __init__(
        self,
        splits_path: str | Path,
        data_dir: str | Path,
        split: Literal["train", "val", "test"] = "train",
        res_scale: float = 1.0,
        noise_type: Optional[Literal["gaussian", "salt_and_pepper"]] = None,
        noise_level: float = 0.0,
    ):
        self.data_dir = Path(data_dir)
        with open(splits_path, "r") as f:
            payload = json.load(f)

        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split}")

        self.records = payload[split]
        self.class_names = payload["class_names"]
        self.class_to_idx = payload["class_to_idx"]
        self.seed = payload.get("seed", 42)
        self.split = split

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.degrade = DegradationTransform(
            res_scale=res_scale,
            noise_type=noise_type,
            noise_level=noise_level,
            seed=self.seed,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.records[idx]
        img_path = self.data_dir / record["path"]
        label = int(record["label"])

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.to_tensor(img)  # [C, H, W] in [0.0, 1.0]

        # Apply degradation only when requested (typically on test split)
        tensor = self.degrade(tensor, index=idx)
        tensor = self.normalize(tensor)
        return tensor, label


def get_dataloader(
    splits_path: str | Path,
    data_dir: str | Path,
    split: Literal["train", "val", "test"] = "train",
    batch_size: int = 64,
    num_workers: int = 4,
    res_scale: float = 1.0,
    noise_type: Optional[Literal["gaussian", "salt_and_pepper"]] = None,
    noise_level: float = 0.0,
) -> DataLoader:
    dataset = RESISC45Dataset(
        splits_path=splits_path,
        data_dir=data_dir,
        split=split,
        res_scale=res_scale,
        noise_type=noise_type,
        noise_level=noise_level,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
