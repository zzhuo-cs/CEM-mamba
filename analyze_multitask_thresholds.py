#!/usr/bin/env python3
"""
Analyze decision thresholds for multitask prediction CSVs.

Thresholds are selected on pooled validation predictions and then applied to
evaluation datasets such as held-out internal test, external, and prospective.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze thresholds for multitask predictions")
    parser.add_argument(
        "--val-predictions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more validation predictions.csv files to pool",
    )
    parser.add_argument("--internal-predictions", type=Path, help="Held-out internal predictions.csv")
    parser.add_argument("--external-predictions", type=Path, help="External predictions.csv")
    parser.add_argument("--prospective-predictions", type=Path, help="Prospective predictions.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--prefix", type=str, default="multitask_thresholds", help="Output file prefix")
    parser.add_argument(
        "--min-sensitivity",
        type=float,
        default=0.90,
        help="Sensitivity constraint for the high-sensitivity operating point",
    )
    return parser.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"case_ids", "bm_labels", "bm_probs"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    out = df.rename(
        columns={
            "case_ids": "case_id",
            "bm_labels": "label",
            "bm_probs": "prob_malignant",
        }
    )[["case_id", "label", "prob_malignant"]].copy()
    out["case_id"] = out["case_id"].astype(str)
    out["label"] = out["label"].astype(int)
    out["prob_malignant"] = out["prob_malignant"].astype(float)
    return out


def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    y_true = df["label"].to_numpy()
    y_prob = df["prob_malignant"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "n": int(len(df)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "youden": float(sensitivity + specificity - 1.0),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def sweep_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    thresholds = np.sort(
        np.unique(np.concatenate(([0.0, 1.0], df["prob_malignant"].to_numpy(dtype=float))))
    )
    rows = [compute_metrics(df, float(thr)) for thr in thresholds]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def evaluate_across_sets(datasets: dict, threshold: float) -> dict:
    return {name: compute_metrics(df, threshold) for name, df in datasets.items()}


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pooled_val = pd.concat([load_predictions(path) for path in args.val_predictions], ignore_index=True)
    pooled_val = pooled_val.drop_duplicates(subset=["case_id"]).reset_index(drop=True)

    datasets = {"pooled_val": pooled_val}
    if args.internal_predictions:
        datasets["internal344"] = load_predictions(args.internal_predictions)
    if args.external_predictions:
        datasets["external97"] = load_predictions(args.external_predictions)
    if args.prospective_predictions:
        datasets["prospective96"] = load_predictions(args.prospective_predictions)

    val_sweep = sweep_thresholds(pooled_val)
    fpr, tpr, thresholds = roc_curve(pooled_val["label"], pooled_val["prob_malignant"])
    youden = tpr - fpr
    youden_idx = int(np.argmax(youden))
    youden_threshold = float(thresholds[youden_idx])

    feasible = val_sweep[val_sweep["sensitivity"] >= args.min_sensitivity].copy()
    feasible = feasible.sort_values(
        ["specificity", "youden", "accuracy", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    summary = {
        "val_prediction_files": [str(p) for p in args.val_predictions],
        "selected_thresholds": {
            "default_0.5": 0.5,
            "youden": youden_threshold,
        },
        "metrics_at_default_0.5": evaluate_across_sets(datasets, 0.5),
        "metrics_at_youden": evaluate_across_sets(datasets, youden_threshold),
        "min_sensitivity_constraint": float(args.min_sensitivity),
        "feasible_val_count": int(len(feasible)),
    }

    if not feasible.empty:
        sens_threshold = float(feasible.iloc[0]["threshold"])
        summary["selected_thresholds"]["high_sensitivity"] = sens_threshold
        summary["metrics_at_high_sensitivity"] = evaluate_across_sets(datasets, sens_threshold)

    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    val_sweep.to_csv(args.output_dir / f"{args.prefix}_pooled_val_sweep.csv", index=False)
    feasible.to_csv(
        args.output_dir / f"{args.prefix}_pooled_val_feasible_sen{int(args.min_sensitivity * 100)}.csv",
        index=False,
    )
    pooled_val.to_csv(args.output_dir / f"{args.prefix}_pooled_val_predictions.csv", index=False)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
