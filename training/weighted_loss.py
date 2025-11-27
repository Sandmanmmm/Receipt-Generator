"""
Class-Balanced Loss Functions for Imbalanced Entity Recognition
Implements ENS weighting and Focal Loss for rare entities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassBalancedCrossEntropy(nn.Module):
    """
    Class-Balanced Loss using Effective Number of Samples (ENS).
    
    Based on: Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)
    
    Re-weights loss by (1 - beta^n) / (1 - beta) where:
    - n = number of samples per class
    - beta = hyperparameter (0.9999 for extreme imbalance)
    
    Effect: Rare classes get much higher loss weights.
    """
    
    def __init__(
        self,
        label_frequencies: torch.Tensor,
        beta: float = 0.9999,
        ignore_index: int = -100
    ):
        """
        Args:
            label_frequencies: [num_labels] - Number of samples per label
            beta: Balancing parameter (higher = more aggressive reweighting)
            ignore_index: Label index to ignore (padding)
        """
        super().__init__()
        self.beta = beta
        self.ignore_index = ignore_index
        
        # Compute effective number of samples
        effective_num = 1.0 - torch.pow(beta, label_frequencies)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        self.register_buffer("weights", weights)
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_labels] or [batch * seq_len, num_labels]
            labels: [batch, seq_len] or [batch * seq_len]
        
        Returns:
            Weighted cross-entropy loss
        """
        # Flatten if needed
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        
        # Standard cross-entropy with class weights
        loss = F.cross_entropy(
            logits,
            labels,
            weight=self.weights,
            ignore_index=self.ignore_index,
            reduction='mean'
        )
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Based on: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Where:
    - p_t = model's estimated probability for correct class
    - gamma = focusing parameter (higher = more focus on hard examples)
    - alpha_t = class weight
    
    Effect: Down-weights easy examples, focuses learning on hard/rare examples.
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        """
        Args:
            gamma: Focusing parameter (0 = standard CE, higher = more focus on hard)
            alpha: [num_labels] - Class weights (optional)
            ignore_index: Label to ignore
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch, seq_len, num_labels] or [batch * seq_len, num_labels]
            labels: [batch, seq_len] or [batch * seq_len]
        
        Returns:
            Focal loss
        """
        # Flatten
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)  # [N, num_labels]
        
        # Get probability of correct class
        labels_one_hot = F.one_hot(labels.clamp(min=0), num_classes=logits.size(-1))
        p_t = (probs * labels_one_hot).sum(dim=-1)  # [N]
        
        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross-entropy
        ce_loss = F.cross_entropy(
            logits,
            labels,
            reduction='none',
            ignore_index=self.ignore_index
        )  # [N]
        
        # Apply focal weight
        loss = focal_weight * ce_loss
        
        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha[labels.clamp(min=0)]
            loss = alpha_t * loss
        
        # Mask ignored indices
        mask = labels != self.ignore_index
        loss = loss * mask
        
        return loss.sum() / (mask.sum() + 1e-8)


class CombinedLoss(nn.Module):
    """
    Combines Class-Balanced CE and Focal Loss.
    
    Final loss = w1 * CB_CE + w2 * Focal
    
    Best of both:
    - CB_CE: Re-weights by effective sample count
    - Focal: Focuses on hard examples
    """
    
    def __init__(
        self,
        label_frequencies: torch.Tensor,
        beta: float = 0.9999,
        gamma: float = 2.0,
        cb_weight: float = 0.6,
        focal_weight: float = 0.4,
        ignore_index: int = -100
    ):
        """
        Args:
            label_frequencies: [num_labels]
            beta: ENS parameter
            gamma: Focal parameter
            cb_weight: Weight for class-balanced loss
            focal_weight: Weight for focal loss
            ignore_index: Padding index
        """
        super().__init__()
        
        self.cb_loss = ClassBalancedCrossEntropy(
            label_frequencies,
            beta=beta,
            ignore_index=ignore_index
        )
        
        # Use CB weights for focal loss alpha
        self.focal_loss = FocalLoss(
            gamma=gamma,
            alpha=self.cb_loss.weights,
            ignore_index=ignore_index
        )
        
        self.cb_weight = cb_weight
        self.focal_weight = focal_weight
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute combined loss"""
        cb_loss = self.cb_loss(logits, labels)
        focal_loss = self.focal_loss(logits, labels)
        
        return self.cb_weight * cb_loss + self.focal_weight * focal_loss


# Example usage
if __name__ == "__main__":
    # Mock label frequencies (extreme imbalance)
    label_frequencies = torch.tensor([
        10000.0,  # O - very frequent
        5000.0,   # B-INVOICE_NUMBER
        4500.0,   # I-INVOICE_NUMBER
        # ... more frequent labels
        50.0,     # B-CAGE_CODE - rare
        45.0,     # I-CAGE_CODE
        30.0,     # B-LOT_NUMBER - very rare
        28.0,     # I-LOT_NUMBER
    ])
    
    # Compute CB weights
    cb_loss = ClassBalancedCrossEntropy(label_frequencies, beta=0.9999)
    print("Class-Balanced Weights:")
    print(cb_loss.weights)
    
    # Mock predictions
    batch_size, seq_len, num_labels = 4, 128, len(label_frequencies)
    logits = torch.randn(batch_size, seq_len, num_labels)
    labels = torch.randint(0, num_labels, (batch_size, seq_len))
    
    # Compute losses
    cb_loss_val = cb_loss(logits, labels)
    focal_loss = FocalLoss(gamma=2.0, alpha=cb_loss.weights)
    focal_loss_val = focal_loss(logits, labels)
    combined_loss = CombinedLoss(label_frequencies)
    combined_loss_val = combined_loss(logits, labels)
    
    print(f"\nCB Loss: {cb_loss_val.item():.4f}")
    print(f"Focal Loss: {focal_loss_val.item():.4f}")
    print(f"Combined Loss: {combined_loss_val.item():.4f}")
