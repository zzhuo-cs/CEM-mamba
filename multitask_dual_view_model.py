"""
Dual-view multi-task MambaVision model.

Architecture:
1. Shared MambaVision backbone for CC and MLO
2. Feature fusion (late concat / gated / cross-view)
3. Either MMoE or a simple shared projection
4. Primary BM head + auxiliary BI-RADS head
"""

import os
import sys

import torch
import torch.nn as nn

from mmoe_module import MMoELayer

# Bundled dual-view helpers; no dependency on a sibling server checkout.
import dual_view_model as stable_dual_view_model


class OrdinalBiRadsMMoEHead(nn.Module):
    """Ordinal BI-RADS head driven by task-specific MMoE features."""

    def __init__(self, input_dim, num_classes, tower_hidden_dims=[128, 64], dropout=0.5):
        super().__init__()
        self.num_classes = num_classes

        tower_layers = []
        prev_dim = input_dim
        for hidden_dim in tower_hidden_dims:
            tower_layers.append(nn.Linear(prev_dim, hidden_dim))
            tower_layers.append(nn.ReLU(inplace=True))
            tower_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.tower = nn.Sequential(*tower_layers)
        self.classifier = nn.Linear(prev_dim, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.tower(x)
        class_logits = self.classifier(features)
        class_probs = torch.softmax(class_logits, dim=1)
        cumulative_probs = 1.0 - torch.cumsum(class_probs, dim=1)[:, :-1]
        cumulative_probs = torch.clamp(cumulative_probs, min=1e-7, max=1.0 - 1e-7)
        return cumulative_probs, class_probs


class LateConcatFusion(nn.Module):
    def __init__(self, feat_dim, dropout=0.5):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, cc_feat, mlo_feat):
        return self.fusion(torch.cat([cc_feat, mlo_feat], dim=1))


class GatedFusion(nn.Module):
    def __init__(self, feat_dim, dropout=0.5):
        super().__init__()
        self.cc_proj = nn.Linear(feat_dim, feat_dim)
        self.mlo_proj = nn.Linear(feat_dim, feat_dim)
        self.gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, cc_feat, mlo_feat):
        cc_proj = self.cc_proj(cc_feat)
        mlo_proj = self.mlo_proj(mlo_feat)
        gate = self.gate(torch.cat([cc_feat, mlo_feat], dim=1))
        return gate * cc_proj + (1.0 - gate) * mlo_proj


class CrossViewFusion(nn.Module):
    def __init__(self, feat_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn_c2m = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_m2c = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cc = nn.LayerNorm(feat_dim)
        self.norm_mlo = nn.LayerNorm(feat_dim)
        self.channel_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
            nn.Sigmoid(),
        )

    def forward(self, cc_feat, mlo_feat):
        cc_query = cc_feat.unsqueeze(1)
        mlo_query = mlo_feat.unsqueeze(1)

        cc_enhanced, _ = self.cross_attn_c2m(cc_query, mlo_query, mlo_query)
        cc_enhanced = self.norm_cc(cc_feat + cc_enhanced.squeeze(1))

        mlo_enhanced, _ = self.cross_attn_m2c(mlo_query, cc_query, cc_query)
        mlo_enhanced = self.norm_mlo(mlo_feat + mlo_enhanced.squeeze(1))

        gate = self.channel_gate(torch.cat([cc_enhanced, mlo_enhanced], dim=1))
        return gate * cc_enhanced + (1.0 - gate) * mlo_enhanced


class SimpleSharedMultiTaskHead(nn.Module):
    """Small-data-friendly multi-task head without MMoE."""

    def __init__(self, feat_dim):
        super().__init__()
        shared_dim = feat_dim // 2
        self.shared_proj = nn.Sequential(
            nn.Linear(feat_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.bm_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        self.birads_head = nn.Sequential(
            nn.Linear(shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.residual_head = nn.Linear(feat_dim, 2)

    def forward(self, fused_feat):
        shared_feat = self.shared_proj(fused_feat)
        bm_logits = self.bm_head(shared_feat) + self.residual_head(fused_feat)
        birads_risk = self.birads_head(shared_feat).squeeze(1)
        return bm_logits, birads_risk


class DualViewMultiTaskMambaVisionFinetune(nn.Module):
    def __init__(
        self,
        model_name="mamba_vision_T",
        num_birads_classes=5,
        pretrained=False,
        pretrained_path=None,
        freeze_backbone=False,
        dropout=0.5,
        fusion_mode="gated",
        view_dropout_prob=0.0,
        multitask_mode="ordinal_mmoe",
        num_experts=4,
        expert_hidden_dims=[256, 128],
        expert_output_dim=64,
        gate_hidden_dims=[64, 32],
        tower_hidden_dims=[128, 64],
    ):
        super().__init__()

        self.model_name = model_name
        self.num_birads_classes = num_birads_classes
        self.fusion_mode = fusion_mode
        self.view_dropout_prob = view_dropout_prob
        self.multitask_mode = multitask_mode

        self.backbone = self._create_backbone(model_name, pretrained=pretrained)

        if pretrained_path and not os.path.isfile(pretrained_path):
            raise FileNotFoundError(pretrained_path)
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            self._load_pretrained_weights(pretrained_path)

        self.feature_dim = self._infer_feature_dim()
        print(f"Feature dimension: {self.feature_dim}")

        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()

        if freeze_backbone:
            print("Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False

        if fusion_mode == "late_concat":
            self.fusion = LateConcatFusion(self.feature_dim, dropout=dropout)
        elif fusion_mode == "gated":
            self.fusion = GatedFusion(self.feature_dim, dropout=dropout)
        elif fusion_mode == "cross_view":
            self.fusion = CrossViewFusion(self.feature_dim, num_heads=4, dropout=0.1)
        else:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode}")

        if self.multitask_mode == "ordinal_mmoe":
            self.mmoe = MMoELayer(
                input_dim=self.feature_dim,
                num_experts=num_experts,
                expert_hidden_dims=expert_hidden_dims,
                expert_output_dim=expert_output_dim,
                num_tasks=2,
                gate_hidden_dims=gate_hidden_dims,
                dropout=dropout,
            )

            bm_tower_layers = []
            prev_dim = expert_output_dim
            for hidden_dim in tower_hidden_dims:
                bm_tower_layers.extend(
                    [
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout),
                    ]
                )
                prev_dim = hidden_dim
            bm_tower_layers.append(nn.Linear(prev_dim, 2))
            self.benign_malignant_head = nn.Sequential(*bm_tower_layers)

            self.birads_head = OrdinalBiRadsMMoEHead(
                input_dim=expert_output_dim,
                num_classes=num_birads_classes,
                tower_hidden_dims=tower_hidden_dims,
                dropout=dropout,
            )
        elif self.multitask_mode == "simple_regression":
            self.simple_head = SimpleSharedMultiTaskHead(self.feature_dim)
        else:
            raise ValueError(f"Unsupported multitask_mode: {self.multitask_mode}")

        self._initialize_new_layers()

        print(f"✓ Dual-view multi-task model created: {model_name}")
        print(f"  - Fusion mode: {fusion_mode}")
        print(f"  - Multi-task mode: {multitask_mode}")
        print(f"  - Backbone features: {self.feature_dim}")
        if self.multitask_mode == "ordinal_mmoe":
            print(f"  - MMoE experts: {num_experts}")

    def _create_backbone(self, model_name, pretrained=False):
        backbone_key = model_name.lower()
        mamba_supported = {
            "mamba_vision_t",
            "mamba_vision_t2",
            "mamba_vision_s",
            "mamba_vision_b",
            "mamba_vision_l",
        }
        if backbone_key in mamba_supported:
            return stable_dual_view_model.create_model(model_name, pretrained=pretrained, num_classes=1000)
        if backbone_key == "densenet121":
            return stable_dual_view_model.DenseNet121Backbone()
        if backbone_key == "resnet50":
            return stable_dual_view_model.ResNet50Backbone(pretrained=pretrained)
        if backbone_key == "xception":
            if stable_dual_view_model.timm is None:
                raise ImportError("timm is required for xception backbone")
            backbone = stable_dual_view_model.timm.create_model("xception", pretrained=pretrained, num_classes=0)
            backbone.feature_dim = getattr(backbone, "num_features", 2048)
            return backbone
        if backbone_key == "swin_tiny_patch4_window7_224":
            if stable_dual_view_model.timm is None:
                raise ImportError("timm is required for Swin backbone")
            backbone = stable_dual_view_model.timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=pretrained,
                num_classes=0,
            )
            backbone.feature_dim = getattr(backbone, "num_features", 768)
            return backbone
        if backbone_key == "mammo_clip_b5":
            return stable_dual_view_model.MammoClipB5Backbone()
        raise ValueError(f"Unknown model name: {model_name}")

    def _infer_feature_dim(self):
        if hasattr(self.backbone, "feature_dim"):
            return int(self.backbone.feature_dim)
        if hasattr(self.backbone, "head"):
            if isinstance(self.backbone.head, nn.Linear):
                return self.backbone.head.in_features
            for module in self.backbone.head.modules():
                if isinstance(module, nn.Linear):
                    return module.in_features
        was_training = self.backbone.training
        device = next(self.backbone.parameters()).device
        self.backbone.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 3, 224, 224, device=device)
            feat = self.backbone(sample)
        self.backbone.train(was_training)
        if feat.ndim != 2:
            raise ValueError(f"Unable to infer backbone feature dimension from shape {tuple(feat.shape)}")
        return int(feat.shape[1])

    def _load_pretrained_weights(self, checkpoint_path):
        try:
            if checkpoint_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file

                    state_dict = load_file(checkpoint_path)
                except Exception:
                    from safetensors.numpy import load_file as load_numpy_file

                    numpy_state_dict = load_numpy_file(checkpoint_path)
                    state_dict = {key: torch.tensor(value) for key, value in numpy_state_dict.items()}
                print("Loaded .safetensors format")
            else:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif isinstance(checkpoint, dict) and "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                print("Loaded .pth/.pth.tar format")

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("module."):
                    key = key[7:]
                elif key.startswith("model."):
                    key = key[6:]
                if key.startswith("backbone."):
                    key = key[len("backbone."):]
                new_state_dict[key] = value

            if self.model_name.lower() == "mammo_clip_b5":
                image_encoder_state_dict = {
                    key[len("image_encoder."):]: value
                    for key, value in state_dict.items()
                    if key.startswith("image_encoder.")
                }
                if image_encoder_state_dict:
                    incompatible = self.backbone.model.load_state_dict(image_encoder_state_dict, strict=False)
                    print("✓ Loaded Mammo-CLIP image encoder weights")
                    if incompatible.missing_keys:
                        print(f"  Missing keys (expected for wrapper-only params): {len(incompatible.missing_keys)}")
                    if incompatible.unexpected_keys:
                        print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
                    return

            # Prefer weights explicitly belonging to the backbone when loading
            # from a full dual-view checkpoint.
            backbone_only_state_dict = {}
            for key, value in new_state_dict.items():
                if key.startswith(
                    (
                        "fusion",
                        "cc_proj",
                        "mlo_proj",
                        "gate",
                        "classifier",
                        "aux_classifier",
                        "cross_view_fusion",
                        "residual_classifier",
                        "mmoe",
                        "benign_malignant_head",
                        "birads_head",
                        "simple_head",
                    )
                ):
                    continue
                backbone_only_state_dict[key] = value

            incompatible = self.backbone.load_state_dict(backbone_only_state_dict, strict=False)
            print("✓ Loaded pretrained weights")
            if incompatible.missing_keys:
                print(f"  Missing keys (expected for heads): {len(incompatible.missing_keys)}")
            if incompatible.unexpected_keys:
                print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
        except Exception as exc:
            raise RuntimeError(f"Failed to load pretrained weights: {checkpoint_path}") from exc

    def _initialize_new_layers(self):
        modules = [self.fusion]
        if self.multitask_mode == "ordinal_mmoe":
            modules.extend([self.benign_malignant_head, self.birads_head])
        else:
            modules.append(self.simple_head)
        for module_group in modules:
            for module in module_group.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

    def _apply_view_dropout(self, cc_feat, mlo_feat):
        if not self.training or self.view_dropout_prob <= 0:
            return cc_feat, mlo_feat

        batch_size = cc_feat.shape[0]
        drop_choices = torch.rand(batch_size, device=cc_feat.device)
        drop_cc = (drop_choices < self.view_dropout_prob / 2.0).float().unsqueeze(1)
        drop_mlo = (
            (drop_choices >= self.view_dropout_prob / 2.0)
            & (drop_choices < self.view_dropout_prob)
        ).float().unsqueeze(1)

        cc_feat = cc_feat * (1.0 - drop_cc)
        mlo_feat = mlo_feat * (1.0 - drop_mlo)
        return cc_feat, mlo_feat

    def forward(self, cc_img, mlo_img, return_birads_cumulative=False):
        cc_feat = self.backbone(cc_img)
        mlo_feat = self.backbone(mlo_img)
        cc_feat, mlo_feat = self._apply_view_dropout(cc_feat, mlo_feat)

        fused_feat = self.fusion(cc_feat, mlo_feat)
        if self.multitask_mode == "ordinal_mmoe":
            task_inputs = self.mmoe(fused_feat)
            bm_logits = self.benign_malignant_head(task_inputs[0])
            birads_cumulative, birads_probs = self.birads_head(task_inputs[1])

            if return_birads_cumulative:
                return bm_logits, birads_probs, birads_cumulative
            return bm_logits, birads_probs

        bm_logits, birads_risk = self.simple_head(fused_feat)
        if return_birads_cumulative:
            return bm_logits, birads_risk, None
        return bm_logits, birads_risk

    def get_num_parameters(self):
        return sum(param.numel() for param in self.parameters())

    def get_shared_parameters(self):
        shared_modules = [self.backbone, self.fusion]
        if self.multitask_mode == "simple_regression":
            shared_modules.append(self.simple_head.shared_proj)
        elif self.multitask_mode == "ordinal_mmoe":
            shared_modules.append(self.mmoe)

        params = []
        for module in shared_modules:
            params.extend([param for param in module.parameters() if param.requires_grad])
        return params

    def get_trainable_parameters(self):
        return filter(lambda param: param.requires_grad, self.parameters())


def create_dual_view_multitask_model_finetune(
    model_name="mamba_vision_T",
    num_birads_classes=5,
    pretrained=False,
    pretrained_path=None,
    freeze_backbone=False,
    dropout=0.5,
    fusion_mode="gated",
    view_dropout_prob=0.0,
    multitask_mode="ordinal_mmoe",
    num_experts=4,
    expert_hidden_dims=[256, 128],
    expert_output_dim=64,
    gate_hidden_dims=[64, 32],
    tower_hidden_dims=[128, 64],
):
    return DualViewMultiTaskMambaVisionFinetune(
        model_name=model_name,
        num_birads_classes=num_birads_classes,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        fusion_mode=fusion_mode,
        view_dropout_prob=view_dropout_prob,
        multitask_mode=multitask_mode,
        num_experts=num_experts,
        expert_hidden_dims=expert_hidden_dims,
        expert_output_dim=expert_output_dim,
        gate_hidden_dims=gate_hidden_dims,
        tower_hidden_dims=tower_hidden_dims,
    )


if __name__ == "__main__":
    model = create_dual_view_multitask_model_finetune(
        model_name="mamba_vision_T",
        multitask_mode="simple_regression",
    )
    cc = torch.randn(2, 3, 224, 224)
    mlo = torch.randn(2, 3, 224, 224)
    bm_logits, birads_risk, _ = model(cc, mlo, return_birads_cumulative=True)
    print("BM logits:", tuple(bm_logits.shape))
    print("BI-RADS risk:", tuple(birads_risk.shape))
