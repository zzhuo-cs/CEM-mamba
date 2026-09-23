"""
Training script for dual-view multi-task MambaVision.

Keeps the original multi-task learning setup and changes the input
pipeline from single-view to paired CC/MLO views.
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            print("TensorBoard is unavailable in this environment, skipping log writing.")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None

from multitask_dual_view_dataset import create_dual_view_multitask_dataset
from multitask_dual_view_model import create_dual_view_multitask_model_finetune
from ordinal_loss import CombinedOrdinalLoss

warnings.filterwarnings("ignore")


class MultiTaskOrdinalLoss(nn.Module):
    def __init__(self, num_birads_classes=5, task_weights=None, ordinal_alpha=0.5, class_weights=None):
        super().__init__()
        self.task_weights = task_weights if task_weights else [1.0, 1.0]
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.ce_loss = nn.CrossEntropyLoss()
        self.ordinal_loss = CombinedOrdinalLoss(num_birads_classes, alpha=ordinal_alpha)

    def forward(
        self,
        benign_malignant_logits,
        birads_probs,
        birads_cumulative_probs,
        benign_malignant_labels,
        birads_labels,
    ):
        loss_bm = self.ce_loss(benign_malignant_logits, benign_malignant_labels)
        loss_birads, loss_birads_ord, loss_birads_ce = self.ordinal_loss(
            birads_cumulative_probs, birads_probs, birads_labels
        )
        total_loss = self.task_weights[0] * loss_bm + self.task_weights[1] * loss_birads
        return total_loss, loss_bm, loss_birads, loss_birads_ord, loss_birads_ce


class MultiTaskRiskRegressionLoss(nn.Module):
    def __init__(self, task_weights=None, class_weights=None):
        super().__init__()
        self.task_weights = task_weights if task_weights else [1.0, 1.0]
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
            print(f"  Using class weights: {class_weights.cpu().numpy()}")
        else:
            self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        benign_malignant_logits,
        birads_risk_pred,
        benign_malignant_labels,
        birads_risk_targets,
        birads_available=None,
    ):
        loss_bm = self.ce_loss(benign_malignant_logits, benign_malignant_labels)
        if birads_available is None:
            birads_available = torch.ones_like(benign_malignant_labels, dtype=torch.bool)
        if birads_available.any():
            loss_birads = F.smooth_l1_loss(
                birads_risk_pred[birads_available],
                birads_risk_targets[birads_available],
            )
        else:
            loss_birads = benign_malignant_logits.new_tensor(0.0)
        total_loss = self.task_weights[0] * loss_bm + self.task_weights[1] * loss_birads
        return total_loss, loss_bm, loss_birads


def get_birads_weight(epoch, target_weight, warmup_epochs):
    if warmup_epochs <= 0:
        return target_weight
    progress = min((epoch + 1) / warmup_epochs, 1.0)
    return target_weight * progress


def _safe_rmse(targets, preds):
    if len(targets) == 0:
        return None
    targets = np.asarray(targets, dtype=float)
    preds = np.asarray(preds, dtype=float)
    return float(np.sqrt(np.mean((targets - preds) ** 2)))


def _summarize_birads_metrics(labels, preds, multitask_mode):
    if multitask_mode == "simple_regression":
        if len(labels) == 0:
            return {"birads_mae": None, "birads_rmse": None}
        labels = np.asarray(labels, dtype=float)
        preds = np.asarray(preds, dtype=float)
        return {
            "birads_mae": float(np.mean(np.abs(labels - preds))),
            "birads_rmse": _safe_rmse(labels, preds),
        }

    return {
        "birads_acc": accuracy_score(labels, preds),
        "birads_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "birads_recall": recall_score(labels, preds, average="macro", zero_division=0),
        "birads_f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


def _clone_shared_grads(shared_params):
    grads = []
    for param in shared_params:
        if param.grad is None:
            grads.append(None)
        else:
            grads.append(param.grad.detach().clone())
    return grads


def _project_conflicting_gradients(task_grads):
    projected_grads = []
    for i, grad_i_list in enumerate(task_grads):
        adjusted = [None if grad is None else grad.clone() for grad in grad_i_list]
        for j, grad_j_list in enumerate(task_grads):
            if i == j:
                continue
            for k, (grad_i, grad_j) in enumerate(zip(adjusted, grad_j_list)):
                if grad_i is None or grad_j is None:
                    continue
                denom = torch.sum(grad_j * grad_j)
                if denom <= 0:
                    continue
                dot = torch.sum(grad_i * grad_j)
                if dot < 0:
                    adjusted[k] = grad_i - dot / (denom + 1e-8) * grad_j
        projected_grads.append(adjusted)

    merged_grads = []
    for grads_per_param in zip(*projected_grads):
        valid_grads = [grad for grad in grads_per_param if grad is not None]
        if valid_grads:
            merged_grads.append(sum(valid_grads) / len(valid_grads))
        else:
            merged_grads.append(None)
    return merged_grads


def pcgrad_backward(model, total_loss, task_losses, shared_params):
    task_grads = []
    for loss in task_losses:
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        task_grads.append(_clone_shared_grads(shared_params))

    projected_grads = _project_conflicting_gradients(task_grads)

    model.zero_grad(set_to_none=True)
    total_loss.backward()
    for param, projected_grad in zip(shared_params, projected_grads):
        if projected_grad is None:
            param.grad = None
        else:
            param.grad = projected_grad


def build_hard_positive_sampler(dataset_df, args):
    df = dataset_df.reset_index(drop=True).copy()
    birads_series = df["birads_normalized"] if "birads_normalized" in df.columns else df["birads"]
    labels = df["label"].astype(int)

    weights = np.ones(len(df), dtype=np.float64)
    positive_mask = labels == 1
    hard_positive_mask = positive_mask & birads_series.isin(args.hard_positive_birads)

    weights[positive_mask.to_numpy()] = float(args.positive_sample_weight)
    weights[hard_positive_mask.to_numpy()] = float(args.hard_positive_sample_weight)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    benign_weight_mass = float(weights[~positive_mask.to_numpy()].sum())
    positive_weight_mass = float(weights[positive_mask.to_numpy()].sum())
    hard_positive_weight_mass = float(weights[hard_positive_mask.to_numpy()].sum())
    total_weight_mass = float(weights.sum())

    print("\n=== Hard Positive Sampler ===")
    print(
        f"Using weighted sampling with benign=1.0, "
        f"positive={args.positive_sample_weight:.2f}, "
        f"hard_positive={args.hard_positive_sample_weight:.2f}"
    )
    print(f"Hard positive BI-RADS: {args.hard_positive_birads}")
    print(
        f"Original counts - benign: {(~positive_mask).sum()}, "
        f"positive: {positive_mask.sum()}, hard_positive: {hard_positive_mask.sum()}"
    )
    print(
        f"Weighted mass ratio - benign: {benign_weight_mass / total_weight_mass:.3f}, "
        f"positive: {positive_weight_mass / total_weight_mass:.3f}, "
        f"hard_positive: {hard_positive_weight_mass / total_weight_mass:.3f}"
    )
    hard_positive_breakdown = (
        df.loc[hard_positive_mask, "birads_normalized"].value_counts().sort_index().to_dict()
    )
    print(f"Hard positive breakdown: {hard_positive_breakdown}")

    return sampler


def train_one_epoch(model, dataloader, criterion, optimizer, device, multitask_mode, use_pcgrad=False):
    model.train()
    shared_params = model.get_shared_parameters() if use_pcgrad else None

    running_loss = 0.0
    running_loss_bm = 0.0
    running_loss_birads = 0.0
    running_loss_birads_ord = 0.0
    running_loss_birads_ce = 0.0

    all_bm_preds = []
    all_bm_labels = []
    all_birads_preds = []
    all_birads_labels = []

    for batch_idx, (cc_images, mlo_images, bm_labels, birads_labels, birads_available, _) in enumerate(dataloader):
        cc_images = cc_images.to(device)
        mlo_images = mlo_images.to(device)
        bm_labels = bm_labels.to(device)
        birads_labels = birads_labels.to(device)
        birads_available = birads_available.to(device)

        if multitask_mode == "simple_regression":
            bm_logits, birads_risk = model(cc_images, mlo_images)
            loss, loss_bm, loss_birads = criterion(
                bm_logits,
                birads_risk,
                bm_labels,
                birads_labels,
                birads_available=birads_available,
            )
        else:
            bm_logits, birads_probs, birads_cumulative = model(
                cc_images, mlo_images, return_birads_cumulative=True
            )
            loss, loss_bm, loss_birads, loss_birads_ord, loss_birads_ce = criterion(
                bm_logits, birads_probs, birads_cumulative, bm_labels, birads_labels
            )

        optimizer.zero_grad()
        if use_pcgrad:
            pcgrad_backward(model, loss, [loss_bm, loss_birads], shared_params)
        else:
            loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_bm += loss_bm.item()
        running_loss_birads += loss_birads.item()
        if multitask_mode != "simple_regression":
            running_loss_birads_ord += loss_birads_ord.item()
            running_loss_birads_ce += loss_birads_ce.item()

        bm_pred = torch.argmax(bm_logits, dim=1)

        all_bm_preds.extend(bm_pred.cpu().numpy())
        all_bm_labels.extend(bm_labels.cpu().numpy())
        if multitask_mode == "simple_regression":
            valid_mask = birads_available.cpu().numpy().astype(bool)
            birads_pred = birads_risk.detach().cpu().numpy()
            birads_target = birads_labels.detach().cpu().numpy()
            all_birads_preds.extend(birads_pred[valid_mask].tolist())
            all_birads_labels.extend(birads_target[valid_mask].tolist())
        else:
            birads_pred = torch.argmax(birads_probs, dim=1)
            all_birads_preds.extend(birads_pred.cpu().numpy())
            all_birads_labels.extend(birads_labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            if multitask_mode == "simple_regression":
                print(
                    f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(BM: {loss_bm.item():.4f}, BI-RADS Risk: {loss_birads.item():.4f})"
                )
            else:
                print(
                    f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(BM: {loss_bm.item():.4f}, BI-RADS: {loss_birads.item():.4f} "
                    f"[Ord:{loss_birads_ord.item():.4f}, CE:{loss_birads_ce.item():.4f}])"
                )

    metrics = {
        "loss": running_loss / len(dataloader),
        "loss_bm": running_loss_bm / len(dataloader),
        "loss_birads": running_loss_birads / len(dataloader),
        "bm_acc": accuracy_score(all_bm_labels, all_bm_preds),
    }
    if multitask_mode != "simple_regression":
        metrics["loss_birads_ord"] = running_loss_birads_ord / len(dataloader)
        metrics["loss_birads_ce"] = running_loss_birads_ce / len(dataloader)
    metrics.update(_summarize_birads_metrics(all_birads_labels, all_birads_preds, multitask_mode))
    return metrics


def validate(model, dataloader, criterion, device, multitask_mode):
    model.eval()

    running_loss = 0.0
    running_loss_bm = 0.0
    running_loss_birads = 0.0

    all_bm_preds = []
    all_bm_labels = []
    all_bm_probs = []
    all_birads_preds = []
    all_birads_labels = []

    with torch.no_grad():
        for cc_images, mlo_images, bm_labels, birads_labels, birads_available, _ in dataloader:
            cc_images = cc_images.to(device)
            mlo_images = mlo_images.to(device)
            bm_labels = bm_labels.to(device)
            birads_labels = birads_labels.to(device)
            birads_available = birads_available.to(device)

            if multitask_mode == "simple_regression":
                bm_logits, birads_risk = model(cc_images, mlo_images)
                loss, loss_bm, loss_birads = criterion(
                    bm_logits,
                    birads_risk,
                    bm_labels,
                    birads_labels,
                    birads_available=birads_available,
                )
            else:
                bm_logits, birads_probs, birads_cumulative = model(
                    cc_images, mlo_images, return_birads_cumulative=True
                )
                loss, loss_bm, loss_birads, _, _ = criterion(
                    bm_logits, birads_probs, birads_cumulative, bm_labels, birads_labels
                )

            running_loss += loss.item()
            running_loss_bm += loss_bm.item()
            running_loss_birads += loss_birads.item()

            bm_pred = torch.argmax(bm_logits, dim=1)
            bm_prob = torch.softmax(bm_logits, dim=1)[:, 1]

            all_bm_preds.extend(bm_pred.cpu().numpy())
            all_bm_labels.extend(bm_labels.cpu().numpy())
            all_bm_probs.extend(bm_prob.cpu().numpy())
            if multitask_mode == "simple_regression":
                valid_mask = birads_available.cpu().numpy().astype(bool)
                all_birads_preds.extend(birads_risk.detach().cpu().numpy()[valid_mask].tolist())
                all_birads_labels.extend(birads_labels.detach().cpu().numpy()[valid_mask].tolist())
            else:
                birads_pred = torch.argmax(birads_probs, dim=1)
                all_birads_preds.extend(birads_pred.cpu().numpy())
                all_birads_labels.extend(birads_labels.cpu().numpy())

    metrics = {
        "loss": running_loss / len(dataloader),
        "loss_bm": running_loss_bm / len(dataloader),
        "loss_birads": running_loss_birads / len(dataloader),
        "bm_acc": accuracy_score(all_bm_labels, all_bm_preds),
        "bm_precision": precision_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
        "bm_recall": recall_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
        "bm_f1": f1_score(all_bm_labels, all_bm_preds, average="binary", zero_division=0),
        "bm_f2": fbeta_score(all_bm_labels, all_bm_preds, beta=2, average="binary", zero_division=0),
    }
    metrics.update(_summarize_birads_metrics(all_birads_labels, all_birads_preds, multitask_mode))
    try:
        metrics["bm_auc"] = roc_auc_score(all_bm_labels, all_bm_probs)
    except Exception:
        metrics["bm_auc"] = 0.0
    return metrics


def main(args):
    from scan_backend import SCAN_BACKEND
    args.scan_backend = SCAN_BACKEND
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{timestamp}-{args.model_name}-dual_view_multitask")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(f"Output directory: {output_dir}")

    print("\n=== Creating Datasets ===")
    train_dataset = create_dual_view_multitask_dataset(
        manifest_csv=args.manifest_csv,
        split=args.train_split,
        image_size=args.image_size,
        is_training=True,
        require_both_views=not args.allow_missing_views,
        view_source=args.view_source,
        birads_target="risk" if args.multitask_mode == "simple_regression" else "class",
        augment_policy=args.augment_policy,
    )
    val_dataset = create_dual_view_multitask_dataset(
        manifest_csv=args.manifest_csv,
        split=args.val_split,
        image_size=args.image_size,
        is_training=False,
        require_both_views=not args.allow_missing_views,
        view_source=args.view_source,
        birads_mapping=train_dataset.birads_mapping,
        birads_target="risk" if args.multitask_mode == "simple_regression" else "class",
        birads_risk_mapping=train_dataset.birads_risk_mapping,
        augment_policy="base",
    )

    train_sampler = None
    if args.use_hard_positive_sampler:
        train_sampler = build_hard_positive_sampler(train_dataset.df, args)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n=== Computing class weights ===")
    train_df = train_dataset.df
    train_bm_counts = train_df["label"].value_counts().sort_index()

    if args.bm_class_weights is not None:
        class_weights = torch.tensor(args.bm_class_weights, dtype=torch.float32, device=device)
        print(f"Using manual BM class weights: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")
    elif args.use_class_weights:
        n_samples = len(train_df)
        n_classes = 2
        class_weights = torch.tensor(
            [
                n_samples / (n_classes * train_bm_counts.get(0, 1)),
                n_samples / (n_classes * train_bm_counts.get(1, 1)),
            ],
            dtype=torch.float32,
            device=device,
        )
        print(f"Using class weights: [{class_weights[0]:.4f}, {class_weights[1]:.4f}]")
    else:
        class_weights = None
        print("Class weights disabled")

    print("\n=== Creating Model ===")
    model = create_dual_view_multitask_model_finetune(
        model_name=args.model_name,
        num_birads_classes=train_dataset.num_birads_classes,
        pretrained=args.pretrained,
        pretrained_path=args.pretrained_path,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
        fusion_mode=args.fusion_mode,
        view_dropout_prob=args.view_dropout_prob,
        multitask_mode=args.multitask_mode,
        num_experts=args.num_experts,
        expert_hidden_dims=args.expert_hidden_dims,
        expert_output_dim=args.expert_output_dim,
        gate_hidden_dims=args.gate_hidden_dims,
        tower_hidden_dims=args.tower_hidden_dims,
    ).to(device)

    if args.multitask_mode == "simple_regression":
        criterion = MultiTaskRiskRegressionLoss(
            task_weights=[args.weight_bm, args.weight_birads],
            class_weights=class_weights,
        )
    else:
        criterion = MultiTaskOrdinalLoss(
            num_birads_classes=train_dataset.num_birads_classes,
            task_weights=[args.weight_bm, args.weight_birads],
            ordinal_alpha=args.ordinal_alpha,
            class_weights=class_weights,
        )

    if args.optimizer == "adam":
        optimizer = optim.Adam(model.get_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(model.get_trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            model.get_trainable_parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    writer = SummaryWriter(os.path.join(output_dir, "logs"))

    print("\n=== Starting Training ===")
    if args.use_pcgrad:
        print("Using PCGrad on shared parameters")
    best_val_loss = float("inf")
    best_bm_auc = 0.0
    best_bm_recall = 0.0
    best_bm_f2 = 0.0
    epochs_without_auc_improvement = 0

    for epoch in range(args.epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{args.epochs}]")
        print(f"{'=' * 60}")

        current_birads_weight = get_birads_weight(epoch, args.weight_birads, args.birads_warmup_epochs)
        criterion.task_weights = [args.weight_bm, current_birads_weight]
        print(
            f"Task weights this epoch: BM={args.weight_bm:.4f}, "
            f"BI-RADS={current_birads_weight:.4f}"
        )

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.multitask_mode,
            use_pcgrad=args.use_pcgrad,
        )
        val_metrics = validate(model, val_loader, criterion, device, args.multitask_mode)
        scheduler.step()

        if args.multitask_mode == "simple_regression":
            print(
                f"[Train] Loss: {train_metrics['loss']:.4f}, "
                f"BM Acc: {train_metrics['bm_acc']:.4f}, "
                f"BI-RADS MAE: {train_metrics['birads_mae']:.4f}"
            )
            print(
                f"[Val] Loss: {val_metrics['loss']:.4f}, "
                f"BM Acc: {val_metrics['bm_acc']:.4f}, "
                f"BM AUC: {val_metrics['bm_auc']:.4f}, "
                f"BI-RADS MAE: {val_metrics['birads_mae']:.4f}"
            )
        else:
            print(
                f"[Train] Loss: {train_metrics['loss']:.4f}, "
                f"BM Acc: {train_metrics['bm_acc']:.4f}, "
                f"BI-RADS Acc: {train_metrics['birads_acc']:.4f}"
            )
            print(
                f"[Val] Loss: {val_metrics['loss']:.4f}, "
                f"BM Acc: {val_metrics['bm_acc']:.4f}, "
                f"BM AUC: {val_metrics['bm_auc']:.4f}, "
                f"BI-RADS Acc: {val_metrics['birads_acc']:.4f}"
            )

        for key, value in train_metrics.items():
            writer.add_scalar(f"Train/{key}", value, epoch)
        for key, value in val_metrics.items():
            writer.add_scalar(f"Val/{key}", value, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("Train/weight_birads_effective", current_birads_weight, epoch)

        common_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "birads_mapping": train_dataset.birads_mapping,
            "effective_weight_birads": current_birads_weight,
        }

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            state = dict(common_state)
            state["val_loss"] = val_metrics["loss"]
            state["val_metrics"] = val_metrics
            torch.save(state, os.path.join(output_dir, "best_loss_model.pth"))
            print(f"  -> Saved best loss model (loss: {best_val_loss:.4f})")

        if val_metrics["bm_auc"] > best_bm_auc:
            best_bm_auc = val_metrics["bm_auc"]
            epochs_without_auc_improvement = 0
            state = dict(common_state)
            state["val_auc"] = val_metrics["bm_auc"]
            state["val_metrics"] = val_metrics
            torch.save(state, os.path.join(output_dir, "best_auc_model.pth"))
            print(f"  -> Saved best AUC model (AUC: {best_bm_auc:.4f})")
        else:
            epochs_without_auc_improvement += 1

        if val_metrics["bm_recall"] > best_bm_recall:
            best_bm_recall = val_metrics["bm_recall"]
            state = dict(common_state)
            state["val_recall"] = val_metrics["bm_recall"]
            state["val_metrics"] = val_metrics
            torch.save(state, os.path.join(output_dir, "best_recall_model.pth"))
            print(f"  -> Saved best recall model (Recall: {best_bm_recall:.4f})")

        if val_metrics["bm_f2"] > best_bm_f2:
            best_bm_f2 = val_metrics["bm_f2"]
            state = dict(common_state)
            state["val_f2"] = val_metrics["bm_f2"]
            state["val_metrics"] = val_metrics
            torch.save(state, os.path.join(output_dir, "best_f2_model.pth"))
            print(f"  -> Saved best F2 model (F2: {best_bm_f2:.4f})")

        if (epoch + 1) % 10 == 0:
            state = dict(common_state)
            state["val_metrics"] = val_metrics
            torch.save(state, os.path.join(output_dir, f"checkpoint_epoch_{epoch + 1}.pth"))

        if args.early_stop_patience > 0 and epochs_without_auc_improvement >= args.early_stop_patience:
            print(
                f"  -> Early stopping triggered after {epochs_without_auc_improvement} "
                f"epochs without BM AUC improvement"
            )
            break

    final_state = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "birads_mapping": train_dataset.birads_mapping,
        "effective_weight_birads": current_birads_weight,
    }
    torch.save(final_state, os.path.join(output_dir, "final_model.pth"))
    writer.close()

    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best BM AUC: {best_bm_auc:.4f}")
    print(f"Best BM Recall: {best_bm_recall:.4f}")
    print(f"Best BM F2: {best_bm_f2:.4f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-view multi-task training")
    parser.add_argument("--manifest_csv", type=str, required=True, help="Unified dual-view manifest CSV")
    parser.add_argument("--train_split", type=str, default="train", help="Training split name")
    parser.add_argument("--val_split", type=str, default="val", help="Validation split name")
    parser.add_argument("--view_source", type=str, default="roi", choices=["full", "roi", "roi_only"])
    parser.add_argument("--allow_missing_views", action="store_true", help="Allow fallback to a single view")
    parser.add_argument("--output_dir", type=str, default="./output_multitask_dual_view")

    parser.add_argument(
        "--model_name",
        type=str,
        default="mamba_vision_S",
        choices=[
            "mamba_vision_T",
            "mamba_vision_T2",
            "mamba_vision_S",
            "mamba_vision_B",
            "mamba_vision_L",
            "densenet121",
            "resnet50",
            "xception",
            "swin_tiny_patch4_window7_224",
            "mammo_clip_b5",
        ],
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--fusion_mode", type=str, default="gated", choices=["late_concat", "gated", "cross_view"])
    parser.add_argument("--view_dropout_prob", type=float, default=0.15)
    parser.add_argument("--augment_policy", type=str, default="base", choices=["base", "medium", "strong"])
    parser.add_argument(
        "--multitask_mode",
        type=str,
        default="ordinal_mmoe",
        choices=["ordinal_mmoe", "simple_regression"],
    )

    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--expert_hidden_dims", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--expert_output_dim", type=int, default=64)
    parser.add_argument("--gate_hidden_dims", type=int, nargs="+", default=[64, 32])
    parser.add_argument("--tower_hidden_dims", type=int, nargs="+", default=[128, 64])

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--weight_bm", type=float, default=1.0)
    parser.add_argument("--weight_birads", type=float, default=0.15)
    parser.add_argument("--birads_warmup_epochs", type=int, default=10)
    parser.add_argument("--ordinal_alpha", type=float, default=0.6)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--bm_class_weights", type=float, nargs=2, default=None)
    parser.add_argument("--use_pcgrad", action="store_true")
    parser.add_argument("--use_hard_positive_sampler", action="store_true")
    parser.add_argument("--positive_sample_weight", type=float, default=1.5)
    parser.add_argument("--hard_positive_sample_weight", type=float, default=3.0)
    parser.add_argument("--hard_positive_birads", type=str, nargs="+", default=["4A", "4B"])

    main(parser.parse_args())
