"""
Dual-view multi-task dataset for:
1. Benign/Malignant classification
2. BI-RADS ordinal classification or continuous risk regression

This dataset reads the unified dual-view manifest and returns paired
CC/MLO images with synchronized augmentations.
"""

import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from multitask_dataset import build_birads_mapping, normalize_birads_label


BIRADS_RISK_MAPPING = {
    "3": 0.02,
    "4A": 0.16,
    "4B": 0.54,
    "4C": 0.88,
    "5": 0.99,
}


def birads_to_risk(birads_value, risk_mapping=None):
    normalized = normalize_birads_label(birads_value)
    if normalized is None:
        return None
    mapping = risk_mapping or BIRADS_RISK_MAPPING
    return mapping.get(normalized)


class DualViewPairTransform:
    """Apply the same random augmentation parameters to both views."""

    def __init__(self, image_size=224, mode="train", augment_policy="base"):
        self.image_size = image_size
        self.mode = mode
        self.augment_policy = augment_policy
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def _resize(self, img):
        return TF.resize(
            img,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
        )

    def _sample_jitter_factor(self, strength):
        if strength <= 0:
            return 1.0
        low = max(0.0, 1.0 - strength)
        high = 1.0 + strength
        return random.uniform(low, high)

    def _to_tensor(self, img):
        tensor = TF.to_tensor(img)
        return TF.normalize(tensor, mean=self.mean, std=self.std)

    def __call__(self, cc_img, mlo_img):
        if self.mode == "train":
            if self.augment_policy in {"medium", "strong"}:
                if self.augment_policy == "strong":
                    crop_scale = (0.75, 1.0)
                    crop_ratio = (0.9, 1.1)
                else:
                    crop_scale = (0.85, 1.0)
                    crop_ratio = (0.95, 1.05)
                crop_params = RandomResizedCrop.get_params(
                    cc_img,
                    scale=crop_scale,
                    ratio=crop_ratio,
                )
                cc_img = TF.resized_crop(
                    cc_img,
                    *crop_params,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                mlo_img = TF.resized_crop(
                    mlo_img,
                    *crop_params,
                    size=[self.image_size, self.image_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
            else:
                cc_img = self._resize(cc_img)
                mlo_img = self._resize(mlo_img)

            if random.random() < 0.5:
                cc_img = TF.hflip(cc_img)
                mlo_img = TF.hflip(mlo_img)

            if random.random() < 0.5:
                cc_img = TF.vflip(cc_img)
                mlo_img = TF.vflip(mlo_img)

            if self.augment_policy == "strong":
                angle_limit = 20.0
            elif self.augment_policy == "medium":
                angle_limit = 12.0
            else:
                angle_limit = 15.0
            angle = random.uniform(-angle_limit, angle_limit)
            cc_img = TF.rotate(cc_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mlo_img = TF.rotate(mlo_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)

            if self.augment_policy in {"medium", "strong"}:
                if self.augment_policy == "strong":
                    translate_limit = 0.06
                    scale_min, scale_max = 0.95, 1.05
                else:
                    translate_limit = 0.03
                    scale_min, scale_max = 0.98, 1.02
                translate_x = random.uniform(-translate_limit, translate_limit) * self.image_size
                translate_y = random.uniform(-translate_limit, translate_limit) * self.image_size
                scale = random.uniform(scale_min, scale_max)
                cc_img = TF.affine(
                    cc_img,
                    angle=0.0,
                    translate=[int(round(translate_x)), int(round(translate_y))],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )
                mlo_img = TF.affine(
                    mlo_img,
                    angle=0.0,
                    translate=[int(round(translate_x)), int(round(translate_y))],
                    scale=scale,
                    shear=[0.0, 0.0],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                )

            if self.augment_policy == "strong":
                color_strength = 0.3
            elif self.augment_policy == "medium":
                color_strength = 0.22
            else:
                color_strength = 0.2
            brightness = self._sample_jitter_factor(color_strength)
            contrast = self._sample_jitter_factor(color_strength)
            cc_img = TF.adjust_brightness(cc_img, brightness)
            mlo_img = TF.adjust_brightness(mlo_img, brightness)
            cc_img = TF.adjust_contrast(cc_img, contrast)
            mlo_img = TF.adjust_contrast(mlo_img, contrast)

            if self.augment_policy == "strong" and random.random() < 0.2:
                cc_img = TF.gaussian_blur(cc_img, kernel_size=[3, 3], sigma=[0.1, 1.2])
                mlo_img = TF.gaussian_blur(mlo_img, kernel_size=[3, 3], sigma=[0.1, 1.2])
        else:
            cc_img = self._resize(cc_img)
            mlo_img = self._resize(mlo_img)

        return self._to_tensor(cc_img), self._to_tensor(mlo_img)


class DualViewMultiTaskManifestDataset(Dataset):
    """Dual-view dataset backed by the unified manifest CSV."""

    def __init__(
        self,
        manifest_csv,
        split,
        transform=None,
        require_both_views=True,
        view_source="roi",
        birads_mapping=None,
        birads_target="class",
        birads_risk_mapping=None,
        drop_missing_birads=True,
    ):
        self.manifest_csv = manifest_csv
        self.split = split
        self.transform = transform
        self.require_both_views = require_both_views
        self.view_source = view_source
        self.birads_target = birads_target
        self.drop_missing_birads = drop_missing_birads
        self.birads_risk_mapping = birads_risk_mapping or BIRADS_RISK_MAPPING

        self.df = self._load_manifest()
        self.df["birads_normalized"] = self.df["birads"].apply(normalize_birads_label)

        if birads_mapping is None:
            self.birads_mapping = build_birads_mapping(self.df["birads_normalized"].tolist())
        else:
            self.birads_mapping = {
                normalize_birads_label(label): int(idx)
                for label, idx in birads_mapping.items()
            }

        self.df["birads_available"] = self.df["birads_normalized"].notna()
        self.df["birads_class"] = self.df["birads_normalized"].map(self.birads_mapping)
        self.df["birads_risk"] = self.df["birads_normalized"].apply(
            lambda value: birads_to_risk(value, self.birads_risk_mapping)
        )
        if self.drop_missing_birads:
            if self.birads_target == "risk":
                self.df = self.df[self.df["birads_risk"].notna()].copy().reset_index(drop=True)
            else:
                self.df = self.df[self.df["birads_class"].notna()].copy().reset_index(drop=True)
            self.df["birads_class"] = self.df["birads_class"].astype(int)
            self.df["birads_risk"] = self.df["birads_risk"].astype(float)
        else:
            self.df["birads_class"] = self.df["birads_class"].fillna(0).astype(int)
            self.df["birads_risk"] = self.df["birads_risk"].fillna(0.0).astype(float)
        self.num_birads_classes = len(self.birads_mapping)

        print(f"[{split}] Loaded {len(self.df)} dual-view multitask samples")
        print(f"[{split}] BI-RADS mapping: {self.birads_mapping}")
        print(f"[{split}] BI-RADS risk mapping: {self.birads_risk_mapping}")
        print(
            f"[{split}] BM distribution - Benign: {(self.df['label'] == 0).sum()}, "
            f"Malignant: {(self.df['label'] == 1).sum()}"
        )

    def _load_manifest(self):
        df = pd.read_csv(self.manifest_csv)
        df["case_id"] = df["case_id"].astype(str)
        if "birads" not in df.columns:
            df["birads"] = pd.NA
        df = df[df["split"] == self.split].copy()
        df = df[df["label"].notna()]

        df["cc_resolved_path"] = df.apply(lambda row: self._resolve_view_path(row, "cc"), axis=1)
        df["mlo_resolved_path"] = df.apply(lambda row: self._resolve_view_path(row, "mlo"), axis=1)
        df["has_cc_resolved"] = df["cc_resolved_path"].notna()
        df["has_mlo_resolved"] = df["mlo_resolved_path"].notna()

        if self.require_both_views:
            df = df[df["has_cc_resolved"] & df["has_mlo_resolved"]]
        else:
            df = df[df["has_cc_resolved"] | df["has_mlo_resolved"]]

        df = df.drop_duplicates("case_id", keep="first").reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        return df

    def _resolve_view_path(self, row, prefix):
        roi_path = row.get(f"{prefix}_roi_path")
        full_path = row.get(f"{prefix}_path")

        if self.view_source == "roi":
            return roi_path if isinstance(roi_path, str) and roi_path else full_path
        if self.view_source == "roi_only":
            return roi_path if isinstance(roi_path, str) and roi_path else None
        return full_path if isinstance(full_path, str) and full_path else roi_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cc_path = row["cc_resolved_path"] if isinstance(row["cc_resolved_path"], str) else None
        mlo_path = row["mlo_resolved_path"] if isinstance(row["mlo_resolved_path"], str) else None

        if cc_path is None and mlo_path is None:
            raise FileNotFoundError(f"Both views are missing for case_id={row['case_id']}")

        if cc_path is None:
            cc_path = mlo_path
        if mlo_path is None:
            mlo_path = cc_path

        cc_img = Image.open(cc_path).convert("RGB")
        mlo_img = Image.open(mlo_path).convert("RGB")

        if self.transform is not None:
            cc_img, mlo_img = self.transform(cc_img, mlo_img)

        bm_label = torch.tensor(int(row["label"]), dtype=torch.long)
        if self.birads_target == "risk":
            birads_label = torch.tensor(float(row["birads_risk"]), dtype=torch.float32)
        else:
            birads_label = torch.tensor(int(row["birads_class"]), dtype=torch.long)
        birads_available = torch.tensor(bool(row["birads_available"]), dtype=torch.bool)
        return cc_img, mlo_img, bm_label, birads_label, birads_available, str(row["case_id"])


def create_dual_view_multitask_dataset(
    manifest_csv,
    split,
    image_size=224,
    is_training=False,
    require_both_views=True,
    view_source="roi",
    birads_mapping=None,
    birads_target="class",
    birads_risk_mapping=None,
    augment_policy="base",
    drop_missing_birads=True,
):
    mode = "train" if is_training else "val"
    transform = DualViewPairTransform(
        image_size=image_size,
        mode=mode,
        augment_policy=augment_policy,
    )
    return DualViewMultiTaskManifestDataset(
        manifest_csv=manifest_csv,
        split=split,
        transform=transform,
        require_both_views=require_both_views,
        view_source=view_source,
        birads_mapping=birads_mapping,
        birads_target=birads_target,
        birads_risk_mapping=birads_risk_mapping,
        drop_missing_birads=drop_missing_birads,
    )
