"""
Loss functions for Ordinal Classification

Implements various loss functions suitable for ordinal regression tasks
where class labels have a natural ordering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLoss(nn.Module):
    """
    Ordinal Regression Loss using threshold-based approach
    
    For K classes (0, 1, ..., K-1), we predict K-1 cumulative probabilities:
    P(Y > 0), P(Y > 1), ..., P(Y > K-2)
    
    Loss is computed as binary cross-entropy for each threshold.
    """
    
    def __init__(self, num_classes):
        super(OrdinalRegressionLoss, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
    
    def forward(self, cumulative_probs, targets):
        """
        Args:
            cumulative_probs: Predicted cumulative probabilities [B, K-1]
            targets: True class labels [B] with values in [0, K-1]
        
        Returns:
            loss: Scalar loss value
        """
        batch_size = targets.size(0)
        
        # Create ground truth cumulative labels
        # For true class y, we have:
        #   P(Y > k) = 1 if k < y, else 0
        cumulative_targets = torch.zeros_like(cumulative_probs)
        
        for i, target in enumerate(targets):
            # For each sample, set cumulative_targets[i, k] = 1 if k < target
            cumulative_targets[i, :target] = 1.0
        
        # Compute binary cross-entropy for each threshold
        loss = F.binary_cross_entropy(
            cumulative_probs,
            cumulative_targets,
            reduction='mean'
        )
        
        return loss


class OrdinalCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss applied to probabilities derived from ordinal predictions
    
    This loss ensures that:
    1. The ordering constraint is satisfied
    2. The model is optimized for classification accuracy
    """
    
    def __init__(self, num_classes, label_smoothing=0.0):
        super(OrdinalCrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
    
    def forward(self, class_probs, targets):
        """
        Args:
            class_probs: Predicted class probabilities [B, K]
            targets: True class labels [B] with values in [0, K-1]
        
        Returns:
            loss: Scalar loss value
        """
        # Add small epsilon for numerical stability
        class_probs = torch.clamp(class_probs, min=1e-7, max=1.0 - 1e-7)
        
        # Compute log probabilities
        log_probs = torch.log(class_probs)
        
        # Apply label smoothing if requested
        if self.label_smoothing > 0:
            # Create one-hot targets
            one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
            
            # Apply label smoothing
            smooth_targets = (1 - self.label_smoothing) * one_hot + \
                           self.label_smoothing / self.num_classes
            
            # Compute loss
            loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        else:
            # Standard cross-entropy
            loss = F.nll_loss(log_probs, targets)
        
        return loss


class CombinedOrdinalLoss(nn.Module):
    """
    Combined loss for ordinal classification
    
    Combines:
    1. Ordinal regression loss (enforces ordering)
    2. Cross-entropy loss (optimizes classification)
    """
    
    def __init__(self, num_classes, alpha=0.5, label_smoothing=0.0):
        """
        Args:
            num_classes: Number of ordinal classes
            alpha: Weight for ordinal regression loss (1-alpha for CE loss)
            label_smoothing: Label smoothing factor for CE loss
        """
        super(CombinedOrdinalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        
        self.ordinal_loss = OrdinalRegressionLoss(num_classes)
        self.ce_loss = OrdinalCrossEntropyLoss(num_classes, label_smoothing)
    
    def forward(self, cumulative_probs, class_probs, targets):
        """
        Args:
            cumulative_probs: Predicted cumulative probabilities [B, K-1]
            class_probs: Predicted class probabilities [B, K]
            targets: True class labels [B] with values in [0, K-1]
        
        Returns:
            total_loss: Combined loss
            ordinal_loss: Ordinal regression component
            ce_loss: Cross-entropy component
        """
        ord_loss = self.ordinal_loss(cumulative_probs, targets)
        ce_loss = self.ce_loss(class_probs, targets)
        
        total_loss = self.alpha * ord_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, ord_loss, ce_loss


class EMDLoss(nn.Module):
    """
    Earth Mover's Distance (EMD) Loss for Ordinal Classification
    
    Also known as Wasserstein Distance.
    Penalizes predictions based on the distance between predicted and true class.
    """
    
    def __init__(self, num_classes):
        super(EMDLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, class_probs, targets):
        """
        Args:
            class_probs: Predicted class probabilities [B, K]
            targets: True class labels [B] with values in [0, K-1]
        
        Returns:
            loss: EMD loss
        """
        batch_size = targets.size(0)
        
        # Create one-hot ground truth
        true_dist = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Compute cumulative distributions
        pred_cumulative = torch.cumsum(class_probs, dim=1)
        true_cumulative = torch.cumsum(true_dist, dim=1)
        
        # EMD is L1 distance between cumulative distributions
        emd = torch.abs(pred_cumulative - true_cumulative).sum(dim=1).mean()
        
        return emd


if __name__ == '__main__':
    print("Testing Ordinal Loss Functions")
    print("=" * 60)
    
    # Test parameters
    batch_size = 4
    num_classes = 5
    
    # Create dummy predictions
    cumulative_probs = torch.sigmoid(torch.randn(batch_size, num_classes - 1))
    
    # Derive class probabilities (simplified for testing)
    class_probs = torch.softmax(torch.randn(batch_size, num_classes), dim=1)
    
    # Create dummy targets
    targets = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Batch size: {batch_size}")
    print(f"Number of classes: {num_classes}")
    print(f"Targets: {targets.numpy()}")
    print(f"\nClass probabilities:\n{class_probs.detach().numpy()}")
    
    # Test Ordinal Regression Loss
    print("\n" + "=" * 60)
    print("1. Ordinal Regression Loss")
    ord_loss_fn = OrdinalRegressionLoss(num_classes)
    ord_loss = ord_loss_fn(cumulative_probs, targets)
    print(f"   Loss: {ord_loss.item():.4f}")
    
    # Test Ordinal Cross-Entropy Loss
    print("\n" + "=" * 60)
    print("2. Ordinal Cross-Entropy Loss")
    ce_loss_fn = OrdinalCrossEntropyLoss(num_classes)
    ce_loss = ce_loss_fn(class_probs, targets)
    print(f"   Loss: {ce_loss.item():.4f}")
    
    # Test Combined Loss
    print("\n" + "=" * 60)
    print("3. Combined Ordinal Loss")
    combined_loss_fn = CombinedOrdinalLoss(num_classes, alpha=0.5)
    total_loss, ord_comp, ce_comp = combined_loss_fn(
        cumulative_probs, class_probs, targets
    )
    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   Ordinal Component: {ord_comp.item():.4f}")
    print(f"   CE Component: {ce_comp.item():.4f}")
    
    # Test EMD Loss
    print("\n" + "=" * 60)
    print("4. Earth Mover's Distance Loss")
    emd_loss_fn = EMDLoss(num_classes)
    emd_loss = emd_loss_fn(class_probs, targets)
    print(f"   Loss: {emd_loss.item():.4f}")
    
    print("\n✓ All loss functions tested successfully!")
