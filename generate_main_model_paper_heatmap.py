#!/usr/bin/env python3
"""
Generate publication-ready dual-view heatmaps for the main multitask MambaVision model.

The script computes SmoothGrad saliency maps for CC and MLO inputs, averages them
across the 3-fold ensemble, and renders a clean 2x3 paper figure:
Original / Heatmap / Overlay for each view.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from multitask_dual_view_model import create_dual_view_multitask_model_finetune


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest_row(manifest_csv: str, case_id: str, split: str, view_source: str) -> dict:
    df = pd.read_csv(manifest_csv)
    df["case_id"] = df["case_id"].astype(str)
    row_df = df[(df["case_id"] == str(case_id)) & (df["split"] == split)].copy()
    if row_df.empty:
        raise ValueError(f"case_id={case_id} with split={split} not found in {manifest_csv}")
    row = row_df.iloc[0].to_dict()

    def resolve(prefix: str) -> str:
        roi_path = row.get(f"{prefix}_roi_path")
        full_path = row.get(f"{prefix}_path")
        if view_source == "roi":
            if isinstance(roi_path, str) and roi_path:
                return roi_path
            return full_path
        if view_source == "roi_only":
            if isinstance(roi_path, str) and roi_path:
                return roi_path
            raise FileNotFoundError(f"Missing ROI path for {prefix} case_id={case_id}")
        if isinstance(full_path, str) and full_path:
            return full_path
        return roi_path

    row["cc_resolved_path"] = resolve("cc")
    row["mlo_resolved_path"] = resolve("mlo")
    return row


def preprocess_image(image_path: str, image_size: int) -> tuple[np.ndarray, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")
    original_np = np.asarray(image).astype(np.float32) / 255.0
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    resized_np = np.asarray(resized).astype(np.float32) / 255.0
    normalized = (resized_np - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
    return original_np, tensor


def percentile_normalize(x: np.ndarray, high_q: float = 99.0) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    upper = np.percentile(x, high_q)
    if upper <= 1e-8:
        upper = x.max() if x.max() > 0 else 1.0
    x = np.clip(x / upper, 0.0, 1.0)
    return x


def smooth_heatmap(x: np.ndarray, blur_radius: float = 4.0) -> np.ndarray:
    pil_img = Image.fromarray(np.uint8(np.clip(x, 0.0, 1.0) * 255.0))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(blurred).astype(np.float32) / 255.0
    return percentile_normalize(arr, high_q=99.0)


def resize_heatmap(heatmap: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = target_hw
    pil_img = Image.fromarray(np.uint8(np.clip(heatmap, 0.0, 1.0) * 255.0))
    resized = pil_img.resize((width, height), Image.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def to_grayscale(image_np: np.ndarray) -> np.ndarray:
    if image_np.ndim == 2:
        gray = image_np.astype(np.float32)
    else:
        gray = np.dot(image_np[..., :3], np.array([0.2989, 0.5870, 0.1140], dtype=np.float32))
    gray = gray.astype(np.float32)
    gray = gray - gray.min()
    denom = gray.max() if gray.max() > 0 else 1.0
    return gray / denom


def make_overlay(image_np: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45, cmap_name: str = "inferno") -> np.ndarray:
    gray = to_grayscale(image_np)
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    cmap = plt.get_cmap(cmap_name)
    heat_rgb = cmap(np.clip(heatmap, 0.0, 1.0))[..., :3]
    overlay = (1.0 - alpha) * gray_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def instantiate_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    birads_mapping = checkpoint.get("birads_mapping", {"3": 0, "4A": 1, "4B": 2, "4C": 3, "5": 4})
    model = create_dual_view_multitask_model_finetune(
        model_name=saved_args.get("model_name", "mamba_vision_S"),
        num_birads_classes=len(birads_mapping),
        pretrained=False,
        dropout=saved_args.get("dropout", 0.5),
        fusion_mode=saved_args.get("fusion_mode", "gated"),
        view_dropout_prob=saved_args.get("view_dropout_prob", 0.0),
        multitask_mode=saved_args.get("multitask_mode", "simple_regression"),
        num_experts=saved_args.get("num_experts", 4),
        expert_hidden_dims=saved_args.get("expert_hidden_dims", [256, 128]),
        expert_output_dim=saved_args.get("expert_output_dim", 64),
        gate_hidden_dims=saved_args.get("gate_hidden_dims", [64, 32]),
        tower_hidden_dims=saved_args.get("tower_hidden_dims", [128, 64]),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model


def smoothgrad_for_model(
    model: torch.nn.Module,
    cc_tensor: torch.Tensor,
    mlo_tensor: torch.Tensor,
    target_class: int,
    smooth_samples: int,
    noise_std: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    cc_base = cc_tensor.unsqueeze(0).to(device)
    mlo_base = mlo_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(cc_base, mlo_base)
        prob = torch.softmax(logits, dim=1)[0, target_class].item()

    cc_acc = torch.zeros((cc_tensor.shape[1], cc_tensor.shape[2]), device=device)
    mlo_acc = torch.zeros((mlo_tensor.shape[1], mlo_tensor.shape[2]), device=device)

    for _ in range(smooth_samples):
        cc_noisy = (cc_base + torch.randn_like(cc_base) * noise_std).clone().requires_grad_(True)
        mlo_noisy = (mlo_base + torch.randn_like(mlo_base) * noise_std).clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits, _ = model(cc_noisy, mlo_noisy)
        score = logits[:, target_class].sum()
        cc_grad, mlo_grad = torch.autograd.grad(score, [cc_noisy, mlo_noisy], retain_graph=False, create_graph=False)
        cc_acc += cc_grad.detach().abs().mean(dim=1).squeeze(0)
        mlo_acc += mlo_grad.detach().abs().mean(dim=1).squeeze(0)

    cc_map = (cc_acc / smooth_samples).detach().cpu().numpy()
    mlo_map = (mlo_acc / smooth_samples).detach().cpu().numpy()
    return cc_map, mlo_map, prob


def render_paper_figure(
    case_meta: dict,
    cc_original: np.ndarray,
    mlo_original: np.ndarray,
    cc_heat: np.ndarray,
    mlo_heat: np.ndarray,
    cc_overlay: np.ndarray,
    mlo_overlay: np.ndarray,
    output_png: str,
    output_pdf: str | None,
    avg_prob: float,
    fold_probs: list[float],
):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), facecolor="white")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.11, wspace=0.04, hspace=0.08)
    cc_gray = to_grayscale(cc_original)
    mlo_gray = to_grayscale(mlo_original)

    col_titles = ["Original ROI", "Attention Heatmap", "Overlay"]
    row_titles = ["CC view", "MLO view"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=13, pad=10, fontweight="bold")
    for row_idx, row_title in enumerate(row_titles):
        axes[row_idx, 0].text(
            -0.08,
            0.5,
            row_title,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    axes[0, 0].imshow(cc_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0, 1].imshow(cc_heat, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[0, 2].imshow(cc_overlay)

    axes[1, 0].imshow(mlo_gray, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1, 1].imshow(mlo_heat, cmap="inferno", vmin=0.0, vmax=1.0)
    axes[1, 2].imshow(mlo_overlay)

    for ax in axes.ravel():
        ax.axis("off")

    cohort_display = {
        "test": "Internal test cohort",
        "external": "External validation cohort",
        "prospective": "Prospective cohort",
    }.get(str(case_meta.get("split")), str(case_meta.get("split")))
    title_line = "Main model attention heatmap"
    subtitle_line = (
        f"Representative {'malignant' if int(case_meta['label']) == 1 else 'benign'} case from the {cohort_display} | "
        f"BI-RADS {case_meta.get('birads', 'NA')} | Ensemble P(malignant) = {avg_prob:.3f}"
    )
    fig.suptitle(title_line, fontsize=18, fontweight="bold", y=0.96)
    fig.text(0.5, 0.915, subtitle_line, ha="center", va="center", fontsize=12.5, color="#222222")
    footer = "SmoothGrad saliency maps were computed on the malignant logit and averaged across the 3-fold ensemble."
    fig.text(0.06, 0.04, footer, fontsize=10.5, color="#444444")

    fig.savefig(output_png, dpi=350, bbox_inches="tight", facecolor="white")
    if output_pdf:
        fig.savefig(output_pdf, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate publication-ready main-model heatmap")
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--case_id", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--view_source", type=str, default="roi", choices=["roi", "roi_only", "full"])
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--smooth_samples", type=int, default=12)
    parser.add_argument("--noise_std", type=float, default=0.10)
    parser.add_argument("--target_class", type=int, default=1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_meta = load_manifest_row(args.manifest_csv, args.case_id, args.split, args.view_source)
    cc_original, cc_tensor = preprocess_image(case_meta["cc_resolved_path"], args.image_size)
    mlo_original, mlo_tensor = preprocess_image(case_meta["mlo_resolved_path"], args.image_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cc_maps = []
    mlo_maps = []
    fold_probs = []
    for checkpoint_path in args.checkpoints:
        print(f"Loading checkpoint: {checkpoint_path}")
        model = instantiate_model_from_checkpoint(checkpoint_path, device)
        cc_map, mlo_map, prob = smoothgrad_for_model(
            model=model,
            cc_tensor=cc_tensor,
            mlo_tensor=mlo_tensor,
            target_class=args.target_class,
            smooth_samples=args.smooth_samples,
            noise_std=args.noise_std,
            device=device,
        )
        cc_maps.append(cc_map)
        mlo_maps.append(mlo_map)
        fold_probs.append(prob)

    cc_heat = percentile_normalize(np.mean(np.stack(cc_maps, axis=0), axis=0), high_q=99.0)
    mlo_heat = percentile_normalize(np.mean(np.stack(mlo_maps, axis=0), axis=0), high_q=99.0)
    cc_heat = smooth_heatmap(cc_heat, blur_radius=4.0)
    mlo_heat = smooth_heatmap(mlo_heat, blur_radius=4.0)

    cc_heat_resized = resize_heatmap(cc_heat, cc_original.shape[:2])
    mlo_heat_resized = resize_heatmap(mlo_heat, mlo_original.shape[:2])
    cc_overlay = make_overlay(cc_original, cc_heat_resized, alpha=0.42, cmap_name="inferno")
    mlo_overlay = make_overlay(mlo_original, mlo_heat_resized, alpha=0.42, cmap_name="inferno")

    stem = f"main_model_heatmap_case_{args.case_id}"
    png_path = str(output_dir / f"{stem}.png")
    pdf_path = str(output_dir / f"{stem}.pdf")

    render_paper_figure(
        case_meta=case_meta,
        cc_original=cc_original,
        mlo_original=mlo_original,
        cc_heat=cc_heat_resized,
        mlo_heat=mlo_heat_resized,
        cc_overlay=cc_overlay,
        mlo_overlay=mlo_overlay,
        output_png=png_path,
        output_pdf=pdf_path,
        avg_prob=float(np.mean(fold_probs)),
        fold_probs=fold_probs,
    )

    meta = {
        "case_id": str(case_meta["case_id"]),
        "split": case_meta["split"],
        "label": int(case_meta["label"]),
        "birads": case_meta.get("birads"),
        "cc_path": case_meta["cc_resolved_path"],
        "mlo_path": case_meta["mlo_resolved_path"],
        "checkpoints": args.checkpoints,
        "fold_probs": fold_probs,
        "ensemble_prob": float(np.mean(fold_probs)),
        "output_png": png_path,
        "output_pdf": pdf_path,
    }
    with open(output_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    Image.fromarray(np.uint8(np.clip(cc_overlay, 0.0, 1.0) * 255.0)).save(output_dir / f"{stem}_cc_overlay.png")
    Image.fromarray(np.uint8(np.clip(mlo_overlay, 0.0, 1.0) * 255.0)).save(output_dir / f"{stem}_mlo_overlay.png")
    Image.fromarray(np.uint8(np.clip(cc_heat_resized, 0.0, 1.0) * 255.0)).save(output_dir / f"{stem}_cc_heatmap.png")
    Image.fromarray(np.uint8(np.clip(mlo_heat_resized, 0.0, 1.0) * 255.0)).save(output_dir / f"{stem}_mlo_heatmap.png")

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


if __name__ == "__main__":
    main()
