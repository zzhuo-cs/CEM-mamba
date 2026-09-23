#!/usr/bin/env python3
"""
双视角共享权重模型
支持 late fusion baseline 和 gated fusion 主线。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import re
from pathlib import Path

# 添加mambavision模块路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
# Import mamba_ssm from the installed dependency, never a shadow package.

from mambavision.models.registry import create_model

try:
    import timm
except Exception:  # pragma: no cover - timm should exist in runtime
    timm = None

try:
    from torchvision import models as tv_models
except Exception:  # pragma: no cover - torchvision should exist in runtime
    tv_models = None

try:
    from efficientnet_pytorch import EfficientNet as EfficientNetPyTorch
except Exception:  # pragma: no cover - efficientnet_pytorch should exist in runtime
    EfficientNetPyTorch = None


DEFAULT_PRETRAINED_PATHS = {
    'densenet121': str(Path(torch.hub.get_dir()) / 'checkpoints' / 'densenet121-a639ec97.pth'),
    'xception': str(Path(torch.hub.get_dir()) / 'checkpoints' / 'xception-43020ad28.pth'),
    'swin_tiny_patch4_window7_224': str(Path(torch.hub.get_dir()) / 'checkpoints' / 'swin_tiny_patch4_window7_224.pth'),
    'mammo_clip_b5': str(Path(torch.hub.get_dir()) / 'checkpoints' / 'b5-model-best-epoch-7.tar'),
}


class DenseNet121Backbone(nn.Module):
    """Thin wrapper that exposes DenseNet121 as a feature extractor."""

    def __init__(self):
        super().__init__()
        if tv_models is None:
            raise ImportError('torchvision is required for densenet121 backbone')
        try:
            base_model = tv_models.densenet121(weights=None)
        except TypeError:
            base_model = tv_models.densenet121(pretrained=False)
        self.features = base_model.features
        self.feature_dim = base_model.classifier.in_features

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return torch.flatten(x, 1)


class ResNet50Backbone(nn.Module):
    """Thin wrapper that exposes ResNet50 as a feature extractor."""

    def __init__(self, pretrained=False):
        super().__init__()
        if tv_models is None:
            raise ImportError('torchvision is required for resnet50 backbone')
        try:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = tv_models.resnet50(weights=weights)
        except AttributeError:
            base_model = tv_models.resnet50(pretrained=pretrained)
        except TypeError:
            base_model = tv_models.resnet50(pretrained=pretrained)

        self.stem = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
        )
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool
        self.feature_dim = base_model.fc.in_features

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class MammoClipB5Backbone(nn.Module):
    """EfficientNet-B5 image encoder compatible with official Mammo-CLIP checkpoints."""

    def __init__(self):
        super().__init__()
        if EfficientNetPyTorch is None:
            raise ImportError('efficientnet_pytorch is required for mammo_clip_b5 backbone')
        self.model = EfficientNetPyTorch.from_name('efficientnet-b5', in_channels=3)
        self.feature_dim = 2048
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)


class CrossViewFusion(nn.Module):
    """双向 cross-attention + 通道门控的双视角融合模块。"""

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
        self.fusion_proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
        )

    def forward(self, cc_feat, mlo_feat):
        cc_query = cc_feat.unsqueeze(1)
        mlo_query = mlo_feat.unsqueeze(1)

        cc_enhanced, _ = self.cross_attn_c2m(cc_query, mlo_query, mlo_query)
        cc_enhanced = self.norm_cc(cc_feat + cc_enhanced.squeeze(1))

        mlo_enhanced, _ = self.cross_attn_m2c(mlo_query, cc_query, cc_query)
        mlo_enhanced = self.norm_mlo(mlo_feat + mlo_enhanced.squeeze(1))

        gate = self.channel_gate(torch.cat([cc_enhanced, mlo_enhanced], dim=1))
        fused_base = gate * cc_enhanced + (1.0 - gate) * mlo_enhanced
        fused_feat = self.fusion_proj(fused_base)
        return fused_feat, fused_base


class MultiScaleExtractor(nn.Module):
    """提取多级 stage 特征并投影到统一维度。"""

    def __init__(self, backbone, out_dim=768, img_size=224):
        super().__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.stage_dims = self._infer_stage_dims(img_size=img_size)
        self.proj_dims = self._build_projection_dims(out_dim, len(self.stage_dims))
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(stage_dim, proj_dim),
                nn.ReLU(inplace=True),
            )
            for stage_dim, proj_dim in zip(self.stage_dims, self.proj_dims)
        ])
        self.final_norm = nn.LayerNorm(sum(self.proj_dims))

    def _build_projection_dims(self, out_dim, num_stages):
        base_dim = out_dim // num_stages
        proj_dims = [base_dim] * num_stages
        proj_dims[-1] += out_dim - sum(proj_dims)
        return proj_dims

    def _infer_stage_dims(self, img_size):
        was_training = self.backbone.training
        device = next(self.backbone.parameters()).device
        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size, device=device)
            x = self.backbone.patch_embed(x)
            stage_dims = []
            for level in self.backbone.levels:
                x = level(x)
                stage_dims.append(x.shape[1])
        self.backbone.train(was_training)
        return stage_dims

    def forward(self, x):
        features = []
        x = self.backbone.patch_embed(x)
        num_levels = len(self.backbone.levels)
        for idx, (level, proj) in enumerate(zip(self.backbone.levels, self.proj_layers)):
            x = level(x)
            stage_feat = x
            if idx == num_levels - 1:
                stage_feat = self.backbone.norm(stage_feat)
            pooled = self.avgpool(stage_feat)
            pooled = torch.flatten(pooled, 1)
            features.append(proj(pooled))
        multi_scale_feat = torch.cat(features, dim=1)
        return self.final_norm(multi_scale_feat)


class DualViewSharedModel(nn.Module):
    """
    双视角共享权重模型
    - 一个 backbone 用于两个视角
    - 特征拼接后通过融合层
    使用共享 backbone 提取两个视角特征
    """
    
    def __init__(self, backbone_name='mamba_vision_T', num_classes=2,
                 pretrained=False, checkpoint_path=None, feat_dim=640, dropout=0.5,
                 fusion_mode='late_concat', view_dropout_prob=0.0,
                 enable_aux_logits=False, feature_mode='single_scale'):
        super().__init__()

        self.backbone_name = backbone_name
        self.backbone = self._create_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
        )

        resolved_checkpoint = checkpoint_path
        if not resolved_checkpoint:
            resolved_checkpoint = DEFAULT_PRETRAINED_PATHS.get(backbone_name.lower())
        if resolved_checkpoint:
            self._load_pretrained_weights(resolved_checkpoint)

        inferred_feat_dim = self._infer_backbone_feature_dim()
        self.feat_dim = inferred_feat_dim if feature_mode == 'single_scale' else feat_dim
        self.fusion_mode = fusion_mode
        self.view_dropout_prob = view_dropout_prob
        self.enable_aux_logits = enable_aux_logits
        self.feature_mode = feature_mode
        self.feature_extractor = None
        if self.feature_mode == 'multi_scale':
            if not self._is_mamba_backbone():
                raise ValueError(
                    f'feature_mode=multi_scale 仅支持 MambaVision backbone，当前为 {backbone_name}'
                )
            self.feature_extractor = MultiScaleExtractor(
                backbone=self.backbone,
                out_dim=self.feat_dim,
            )

        # 融合层：将两个视角的特征融合
        self.fusion = nn.Sequential(
            nn.Linear(self.feat_dim * 2, self.feat_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if fusion_mode == 'gated':
            self.cc_proj = nn.Linear(self.feat_dim, self.feat_dim)
            self.mlo_proj = nn.Linear(self.feat_dim, self.feat_dim)
            self.gate = nn.Sequential(
                nn.Linear(self.feat_dim * 2, self.feat_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.Sigmoid(),
            )
            self.gated_fusion = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.residual_classifier = nn.Linear(self.feat_dim, num_classes)
        elif fusion_mode == 'cross_view':
            self.cross_view_fusion = CrossViewFusion(
                feat_dim=self.feat_dim,
                num_heads=4,
                dropout=0.1,
            )
            self.residual_classifier = nn.Linear(self.feat_dim, num_classes)

        # 最终分类头
        self.classifier = nn.Linear(self.feat_dim // 2, num_classes)
        if self.enable_aux_logits:
            self.aux_classifier = nn.Linear(self.feat_dim, num_classes)

    def _is_mamba_backbone(self):
        return self.backbone_name.lower().startswith('mamba_vision')

    def _create_backbone(self, backbone_name, pretrained=False):
        backbone_key = backbone_name.lower()
        if backbone_key == 'densenet121':
            return DenseNet121Backbone()
        if backbone_key == 'resnet50':
            return ResNet50Backbone(pretrained=pretrained)
        if backbone_key == 'xception':
            if timm is None:
                raise ImportError('timm is required for xception backbone')
            backbone = timm.create_model('xception', pretrained=pretrained, num_classes=0)
            backbone.feature_dim = getattr(backbone, 'num_features', 2048)
            return backbone
        if backbone_key == 'swin_tiny_patch4_window7_224':
            if timm is None:
                raise ImportError('timm is required for Swin backbone')
            backbone = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=pretrained,
                num_classes=0,
            )
            backbone.feature_dim = getattr(backbone, 'num_features', 768)
            return backbone
        if backbone_key == 'mammo_clip_b5':
            return MammoClipB5Backbone()
        return create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
        )

    def _infer_backbone_feature_dim(self, img_size=224):
        if hasattr(self.backbone, 'feature_dim'):
            return int(self.backbone.feature_dim)

        was_training = self.backbone.training
        device = next(self.backbone.parameters()).device
        self.backbone.eval()
        with torch.no_grad():
            sample = torch.zeros(1, 3, img_size, img_size, device=device)
            feat = self.backbone(sample)
        self.backbone.train(was_training)
        if feat.ndim != 2:
            raise ValueError(
                f'无法自动推断 backbone 特征维度，输出 shape={tuple(feat.shape)}'
            )
        return int(feat.shape[1])

    def _apply_view_dropout(self, cc_feat, mlo_feat):
        if not self.training or self.view_dropout_prob <= 0:
            return cc_feat, mlo_feat

        batch_size = cc_feat.shape[0]
        drop_choices = torch.rand(batch_size, device=cc_feat.device)
        drop_cc = (drop_choices < self.view_dropout_prob / 2.0).float().unsqueeze(1)
        drop_mlo = (
            (drop_choices >= self.view_dropout_prob / 2.0) &
            (drop_choices < self.view_dropout_prob)
        ).float().unsqueeze(1)

        cc_feat = cc_feat * (1.0 - drop_cc)
        mlo_feat = mlo_feat * (1.0 - drop_mlo)
        return cc_feat, mlo_feat

    def _extract_features(self, img):
        if self.feature_extractor is not None:
            return self.feature_extractor(img)
        return self.backbone(img)

    def _fuse_features(self, cc_feat, mlo_feat):
        if self.fusion_mode == 'gated':
            cc_proj = self.cc_proj(cc_feat)
            mlo_proj = self.mlo_proj(mlo_feat)
            gate = self.gate(torch.cat([cc_feat, mlo_feat], dim=1))
            fused_base = gate * cc_proj + (1.0 - gate) * mlo_proj
            fused_feat = self.gated_fusion(fused_base)
            residual_logits = self.residual_classifier(fused_base)
            return fused_feat, residual_logits
        if self.fusion_mode == 'cross_view':
            fused_feat, fused_base = self.cross_view_fusion(cc_feat, mlo_feat)
            residual_logits = self.residual_classifier(fused_base)
            return fused_feat, residual_logits

        fused_feat = torch.cat([cc_feat, mlo_feat], dim=1)
        fused_feat = self.fusion(fused_feat)
        return fused_feat, None
    
    def _load_pretrained_weights(self, checkpoint_path):
        """加载预训练权重到backbone"""
        print(f"加载预训练权重: {checkpoint_path}")
        
        import os
        
        if not os.path.exists(checkpoint_path):
            print(f"警告: 权重文件不存在 {checkpoint_path}")
            return
        
        # 加载权重
        if checkpoint_path.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                state_dict = load_file(checkpoint_path)
            except Exception:
                from safetensors.numpy import load_file as load_numpy_file
                numpy_state_dict = load_numpy_file(checkpoint_path)
                state_dict = {
                    key: torch.tensor(value)
                    for key, value in numpy_state_dict.items()
                }
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))

        if self.backbone_name.lower() == 'mammo_clip_b5':
            new_state_dict = {}
            for k, v in state_dict.items():
                if not k.startswith('image_encoder.'):
                    continue
                k = k[len('image_encoder.'):]
                new_state_dict[k] = v

            missing, unexpected = self.backbone.model.load_state_dict(new_state_dict, strict=False)
            print(f"  ✓ Mammo-CLIP 图像编码器权重加载完成")
            print(f"    缺失的键: {len(missing)}")
            print(f"    多余的键: {len(unexpected)}")
            return

        # 处理键名前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            # 去掉 'model.' 前缀
            if k.startswith('model.'):
                k = k[6:]
            # 跳过分类头
            if k.startswith('head.') or k.startswith('classifier.'):
                continue
            if self.backbone_name.lower() == 'densenet121':
                k = re.sub(
                    r'^(features\.denseblock\d+\.denselayer\d+\.(?:norm|relu|conv))\.(\d)\.',
                    r'\1\2.',
                    k,
                )
            new_state_dict[k] = v
        
        # 加载到 backbone
        missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
        print(f"  ✓ 权重加载完成")
        print(f"    缺失的键: {len(missing)} (预期包含分类头)")
        print(f"    多余的键: {len(unexpected)}")
    
    def forward(self, cc_img, mlo_img):
        """
        前向传播
        
        Args:
            cc_img: CC视角图像 (B, 3, H, W)
            mlo_img: MLO视角图像 (B, 3, H, W)
        
        Returns:
            logits: 分类输出 (B, num_classes)
        """
        # 两个视角分别通过共享的 backbone
        cc_feat = self._extract_features(cc_img)    # (B, feat_dim)
        mlo_feat = self._extract_features(mlo_img)  # (B, feat_dim)
        
        cc_feat, mlo_feat = self._apply_view_dropout(cc_feat, mlo_feat)
        cc_aux_logits = None
        mlo_aux_logits = None
        if self.enable_aux_logits:
            cc_aux_logits = self.aux_classifier(cc_feat)
            mlo_aux_logits = self.aux_classifier(mlo_feat)
        fused_feat, residual_logits = self._fuse_features(cc_feat, mlo_feat)

        # 分类
        logits = self.classifier(fused_feat)  # (B, num_classes)
        if residual_logits is not None:
            logits = logits + residual_logits
        
        if self.training and self.enable_aux_logits:
            return logits, cc_aux_logits, mlo_aux_logits
        return logits
    
    def get_features(self, cc_img, mlo_img):
        """获取融合后的特征（用于分析）"""
        with torch.no_grad():
            cc_feat = self._extract_features(cc_img)
            mlo_feat = self._extract_features(mlo_img)
            fused_feat, _ = self._fuse_features(cc_feat, mlo_feat)
        return fused_feat


def create_dual_view_model(backbone_name='mamba_vision_T', num_classes=2,
                          checkpoint_path=None, feat_dim=640, dropout=0.5,
                          fusion_mode='late_concat', view_dropout_prob=0.0,
                          enable_aux_logits=False, feature_mode='single_scale'):
    """
    创建双视角模型的便捷函数
    
    Args:
        backbone_name: backbone 模型名称
        num_classes: 分类类别数
        checkpoint_path: 预训练权重路径
        feat_dim: 特征维度
        dropout: Dropout率
    """
    model = DualViewSharedModel(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,  # 我们手动加载权重
        checkpoint_path=checkpoint_path,
        feat_dim=feat_dim,
        dropout=dropout,
        fusion_mode=fusion_mode,
        view_dropout_prob=view_dropout_prob,
        enable_aux_logits=enable_aux_logits,
        feature_mode=feature_mode
    )
    
    return model


if __name__ == '__main__':
    # 测试代码
    print("测试双视角模型...")
    
    model = create_dual_view_model(
        backbone_name='mamba_vision_T',
        num_classes=2,
        checkpoint_path='./pretrained_weights/model.safetensors',
        feat_dim=640
    )
    
    # 测试前向传播
    batch_size = 2
    cc_img = torch.randn(batch_size, 3, 224, 224)
    mlo_img = torch.randn(batch_size, 3, 224, 224)
    
    logits = model(cc_img, mlo_img)
    print(f"\n输入shape: CC {cc_img.shape}, MLO {mlo_img.shape}")
    print(f"输出shape: {logits.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
