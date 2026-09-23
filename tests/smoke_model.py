"""Synthetic CPU forward/backward check for the explicit historical backend.

Run from the repository root: CEM_SCAN_BACKEND=legacy_identity python tests/smoke_model.py
This does not validate the official selective scan or scientific results.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from multitask_dual_view_model import create_dual_view_multitask_model_finetune
from scan_backend import SCAN_BACKEND

assert SCAN_BACKEND == 'legacy_identity', 'This CPU packaging test requires the historical backend explicitly.'
torch.set_num_threads(2)
torch.manual_seed(7)
model = create_dual_view_multitask_model_finetune(
    model_name='mamba_vision_T', pretrained=False, multitask_mode='simple_regression',
    fusion_mode='gated', dropout=0.0, view_dropout_prob=0.0)
model.train()
cc = torch.randn(2, 3, 224, 224)
mlo = torch.randn(2, 3, 224, 224)
logits, risk = model(cc, mlo)
assert logits.shape == (2, 2)
assert risk.numel() == 2
loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
loss = loss + torch.nn.functional.smooth_l1_loss(risk.reshape(-1), torch.tensor([0.16, 0.88]))
loss.backward()
assert torch.isfinite(loss)
assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
print('Synthetic forward/backward passed; backend:', SCAN_BACKEND)
