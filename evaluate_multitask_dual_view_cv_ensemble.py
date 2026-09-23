import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from evaluate_multitask_dual_view import plot_confusion_matrix, plot_roc_curve
from multitask_dual_view_dataset import create_dual_view_multitask_dataset
from multitask_dual_view_model import create_dual_view_multitask_model_finetune

if "numpy._core" not in sys.modules:
    sys.modules["numpy._core"] = np.core


def load_model(checkpoint_path, device, dataset, model_name_default="mamba_vision_S"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args", {})
    from scan_backend import SCAN_BACKEND
    recorded_backend = saved_args.get("scan_backend")
    if recorded_backend is not None and recorded_backend != SCAN_BACKEND:
        raise ValueError(f"Checkpoint scan backend {recorded_backend} differs from {SCAN_BACKEND}")
    if recorded_backend is None:
        print("WARNING: checkpoint has no scan provenance; verify historical backend before interpreting results.")
    multitask_mode = saved_args.get("multitask_mode", "ordinal_mmoe")

    model = create_dual_view_multitask_model_finetune(
        model_name=saved_args.get("model_name", model_name_default),
        num_birads_classes=dataset.num_birads_classes,
        pretrained=False,
        dropout=saved_args.get("dropout", 0.5),
        fusion_mode=saved_args.get("fusion_mode", "gated"),
        view_dropout_prob=saved_args.get("view_dropout_prob", 0.0),
        multitask_mode=multitask_mode,
        num_experts=saved_args.get("num_experts", 4),
        expert_hidden_dims=saved_args.get("expert_hidden_dims", [256, 128]),
        expert_output_dim=saved_args.get("expert_output_dim", 64),
        gate_hidden_dims=saved_args.get("gate_hidden_dims", [64, 32]),
        tower_hidden_dims=saved_args.get("tower_hidden_dims", [128, 64]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, checkpoint, multitask_mode


def evaluate_ensemble(models, dataloader, device, birads_mapping, multitask_mode):
    all_case_ids = []
    all_bm_labels = []
    all_bm_probs = []
    all_bm_preds = []
    all_birads_labels = []
    all_birads_preds = []
    all_birads_available = []

    print("\nRunning ensemble inference...")
    with torch.no_grad():
        for batch_idx, (cc_images, mlo_images, bm_labels, birads_labels, birads_available, case_ids) in enumerate(dataloader):
            cc_images = cc_images.to(device)
            mlo_images = mlo_images.to(device)

            bm_prob_list = []
            birads_pred_list = []
            for model in models:
                model_outputs = model(cc_images, mlo_images)
                if multitask_mode == "simple_regression":
                    bm_logits, birads_risk = model_outputs
                    birads_pred_list.append(birads_risk.detach().cpu().numpy())
                else:
                    bm_logits, birads_probs = model_outputs
                    birads_pred_list.append(torch.argmax(birads_probs, dim=1).cpu().numpy())
                bm_prob_list.append(F.softmax(bm_logits, dim=1)[:, 1].cpu().numpy())

            mean_bm_probs = np.mean(np.stack(bm_prob_list, axis=0), axis=0)
            mean_bm_preds = (mean_bm_probs >= 0.5).astype(int)
            mean_birads_preds = np.mean(np.stack(birads_pred_list, axis=0), axis=0)
            if multitask_mode != "simple_regression":
                mean_birads_preds = np.rint(mean_birads_preds).astype(int)

            all_case_ids.extend([str(x) for x in case_ids])
            all_bm_labels.extend(bm_labels.numpy())
            all_bm_probs.extend(mean_bm_probs.tolist())
            all_bm_preds.extend(mean_bm_preds.tolist())
            all_birads_labels.extend(birads_labels.numpy())
            all_birads_preds.extend(mean_birads_preds.tolist())
            all_birads_available.extend(birads_available.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")

    all_bm_labels = np.array(all_bm_labels)
    all_bm_probs = np.array(all_bm_probs)
    all_bm_preds = np.array(all_bm_preds)
    all_birads_labels = np.array(all_birads_labels)
    all_birads_preds = np.array(all_birads_preds)
    all_birads_available = np.array(all_birads_available).astype(bool)

    bm_metrics = {
        "accuracy": accuracy_score(all_bm_labels, all_bm_preds),
        "precision": precision_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
        "recall": recall_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
        "specificity": recall_score(all_bm_labels, all_bm_preds, pos_label=0, average="binary", zero_division=0),
        "f1": f1_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
    }
    try:
        bm_metrics["auc"] = roc_auc_score(all_bm_labels, all_bm_probs)
        bm_fpr, bm_tpr, _ = roc_curve(all_bm_labels, all_bm_probs)
    except Exception:
        bm_metrics["auc"] = 0.0
        bm_fpr, bm_tpr = None, None

    bm_cm = confusion_matrix(all_bm_labels, all_bm_preds)

    birads_class_names = [k for k, v in sorted(birads_mapping.items(), key=lambda item: item[1])]
    if all_birads_available.any():
        valid_birads_labels = all_birads_labels[all_birads_available]
        valid_birads_preds = all_birads_preds[all_birads_available]
        if multitask_mode == "simple_regression":
            birads_metrics = {
                "mae": float(np.mean(np.abs(valid_birads_labels - valid_birads_preds))),
                "rmse": float(np.sqrt(np.mean((valid_birads_labels - valid_birads_preds) ** 2))),
                "n_available": int(all_birads_available.sum()),
            }
            birads_cm = None
            birads_class_names = []
        else:
            birads_metrics = {
                "accuracy": accuracy_score(valid_birads_labels, valid_birads_preds),
                "precision_macro": precision_score(valid_birads_labels, valid_birads_preds, average="macro", zero_division=0),
                "recall_macro": recall_score(valid_birads_labels, valid_birads_preds, average="macro", zero_division=0),
                "f1_macro": f1_score(valid_birads_labels, valid_birads_preds, average="macro", zero_division=0),
                "n_available": int(all_birads_available.sum()),
            }
            birads_cm = confusion_matrix(
                valid_birads_labels,
                valid_birads_preds,
                labels=list(range(len(birads_class_names))),
            )
    else:
        if multitask_mode == "simple_regression":
            birads_metrics = {"mae": None, "rmse": None, "n_available": 0}
            birads_cm = None
            birads_class_names = []
        else:
            birads_metrics = {
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "n_available": 0,
            }
            birads_cm = np.zeros((len(birads_class_names), len(birads_class_names)), dtype=int)

    results = {
        "benign_malignant": bm_metrics,
        "birads": birads_metrics,
        "bm_confusion_matrix": bm_cm.tolist(),
        "birads_confusion_matrix": birads_cm.tolist() if birads_cm is not None else None,
        "birads_class_names": birads_class_names,
        "multitask_mode": multitask_mode,
        "ensemble_size": len(models),
        "predictions": {
            "case_ids": all_case_ids,
            "bm_labels": all_bm_labels.tolist(),
            "bm_preds": all_bm_preds.tolist(),
            "bm_probs": all_bm_probs.tolist(),
            "birads_labels": all_birads_labels.tolist(),
            "birads_preds": all_birads_preds.tolist(),
            "birads_available": all_birads_available.astype(int).tolist(),
        },
    }
    if bm_fpr is not None and bm_tpr is not None:
        results["bm_roc"] = {"fpr": bm_fpr.tolist(), "tpr": bm_tpr.tolist()}
    return results


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    first_checkpoint = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    saved_args = first_checkpoint.get("args", {})
    birads_mapping = first_checkpoint.get("birads_mapping")
    multitask_mode = saved_args.get("multitask_mode", "ordinal_mmoe")

    dataset = create_dual_view_multitask_dataset(
        manifest_csv=args.manifest_csv,
        split=args.split,
        image_size=args.image_size,
        is_training=False,
        require_both_views=not args.allow_missing_views,
        view_source=args.view_source,
        birads_mapping=birads_mapping,
        birads_target="risk" if multitask_mode == "simple_regression" else "class",
        drop_missing_birads=False,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    models = []
    for checkpoint_path in args.checkpoints:
        print(f"Loading checkpoint from: {checkpoint_path}")
        model, _, ckpt_mode = load_model(checkpoint_path, device, dataset, args.model_name)
        if ckpt_mode != multitask_mode:
            raise ValueError("All checkpoints must share the same multitask_mode")
        models.append(model)

    results = evaluate_ensemble(models, dataloader, device, dataset.birads_mapping, multitask_mode)
    results["checkpoints"] = args.checkpoints

    metrics_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in results.items() if k != "predictions"}, f, indent=4)
    print(f"Saved metrics: {metrics_path}")

    pred_df = pd.DataFrame(results["predictions"])
    pred_csv = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions CSV: {pred_csv}")

    plot_confusion_matrix(
        np.array(results["bm_confusion_matrix"]),
        ["Benign", "Malignant"],
        os.path.join(args.output_dir, "confusion_matrix_benign_malignant.png"),
        "Benign/Malignant Confusion Matrix",
    )
    if results["birads"].get("n_available", 0) > 0 and results["birads_confusion_matrix"] is not None:
        plot_confusion_matrix(
            np.array(results["birads_confusion_matrix"]),
            results["birads_class_names"],
            os.path.join(args.output_dir, "confusion_matrix_birads.png"),
            "BI-RADS Confusion Matrix",
        )
    if "bm_roc" in results:
        plot_roc_curve(
            np.array(results["bm_roc"]["fpr"]),
            np.array(results["bm_roc"]["tpr"]),
            results["benign_malignant"]["auc"],
            os.path.join(args.output_dir, "roc_curve_benign_malignant.png"),
            "Benign/Malignant ROC Curve",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CV checkpoint ensemble for dual-view multi-task model")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--manifest_csv", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--view_source", type=str, default="roi", choices=["full", "roi", "roi_only"])
    parser.add_argument("--allow_missing_views", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="mamba_vision_S")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())
