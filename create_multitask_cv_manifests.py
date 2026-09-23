import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def normalize_birads(value):
    if pd.isna(value):
        return None
    value = str(value).strip().upper()
    return value or None


def main():
    parser = argparse.ArgumentParser(description="Create stratified CV manifests for multitask dual-view training.")
    parser.add_argument("--source-manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--pool-splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--heldout-splits", type=str, nargs="+", default=["test"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.source_manifest).copy()
    df["case_id"] = df["case_id"].astype(str)
    df["birads_norm"] = df["birads"].apply(normalize_birads)
    df = df[df["split"] != "exclude_missing_roi"].copy()
    df = df[df["label"].notna()].copy()
    pool_splits = set(args.pool_splits) if args.pool_splits else set(df["split"].unique())
    heldout_splits = set(args.heldout_splits or [])
    if pool_splits & heldout_splits:
        raise ValueError("Development and held-out split names must be disjoint")
    if df["case_id"].duplicated().any():
        raise ValueError("case_id must be unique: one row per patient/lesion pair")
    unknown = set(df["split"].unique()) - pool_splits - heldout_splits
    if unknown:
        raise ValueError(f"Declare all splits explicitly; unassigned splits: {sorted(unknown)}")

    pool_df = df[df["split"].isin(pool_splits)].copy()
    heldout_df = df[df["split"].isin(heldout_splits)].copy()
    pool_df = pool_df[pool_df["birads_norm"].notna()].copy()

    stratify_key = pool_df["label"].astype(int).astype(str) + "|" + pool_df["birads_norm"]
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    summary_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(pool_df, stratify_key)):
        current_pool = pool_df.copy()
        current_pool["split"] = "train"
        current_pool["cv_fold"] = fold_idx
        current_pool.iloc[val_idx, current_pool.columns.get_loc("split")] = "val"

        if len(heldout_df) > 0:
            current_heldout = heldout_df.copy()
            current_heldout["cv_fold"] = fold_idx
            fold_df = pd.concat([current_pool, current_heldout], ignore_index=True)
        else:
            fold_df = current_pool

        output_csv = os.path.join(args.output_dir, f"multitask_cv{args.n_splits}_fold{fold_idx}.csv")
        fold_df.drop(columns=["birads_norm"]).to_csv(output_csv, index=False)

        train_df = fold_df[fold_df["split"] == "train"]
        val_df = fold_df[fold_df["split"] == "val"]
        heldout_fold_df = fold_df[fold_df["split"].isin(heldout_splits)] if heldout_splits else pd.DataFrame(columns=fold_df.columns)
        summary_rows.append(
            {
                "fold": fold_idx,
                "train_n": int(len(train_df)),
                "val_n": int(len(val_df)),
                "heldout_n": int(len(heldout_fold_df)),
                "train_benign": int((train_df["label"] == 0).sum()),
                "train_malignant": int((train_df["label"] == 1).sum()),
                "val_benign": int((val_df["label"] == 0).sum()),
                "val_malignant": int((val_df["label"] == 1).sum()),
                "heldout_benign": int((heldout_fold_df["label"] == 0).sum()) if len(heldout_fold_df) else 0,
                "heldout_malignant": int((heldout_fold_df["label"] == 1).sum()) if len(heldout_fold_df) else 0,
                "manifest_csv": output_csv,
            }
        )

    summary = {
        "source_manifest": args.source_manifest,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "pool_splits": sorted(pool_splits),
        "heldout_splits": sorted(heldout_splits),
        "num_samples_in_pool": int(len(pool_df)),
        "num_samples_in_heldout": int(len(heldout_df)),
        "folds": summary_rows,
    }

    summary_path = os.path.join(args.output_dir, f"multitask_cv{args.n_splits}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
