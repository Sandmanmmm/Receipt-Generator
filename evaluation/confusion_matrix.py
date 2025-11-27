"""
Confusion Matrix Generation and Visualization
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from typing import List, Dict, Optional
from pathlib import Path


class ConfusionMatrixAnalyzer:
    """Generate and visualize confusion matrices for NER evaluation"""
    
    def __init__(self, label_list: List[str]):
        """
        Initialize analyzer
        
        Args:
            label_list: List of all possible labels (including 'O')
        """
        self.label_list = label_list
        self.label_to_id = {label: idx for idx, label in enumerate(label_list)}
    
    def compute_confusion_matrix(self, y_true: List[List[str]], y_pred: List[List[str]]) -> np.ndarray:
        """
        Compute confusion matrix for token-level predictions
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            Confusion matrix as numpy array
        """
        # Flatten sequences
        y_true_flat = [label for seq in y_true for label in seq]
        y_pred_flat = [label for seq in y_pred for label in seq]
        
        # Convert to indices
        y_true_ids = [self.label_to_id.get(label, 0) for label in y_true_flat]
        y_pred_ids = [self.label_to_id.get(label, 0) for label in y_pred_flat]
        
        # Compute confusion matrix
        cm = sk_confusion_matrix(y_true_ids, y_pred_ids, labels=list(range(len(self.label_list))))
        
        return cm
    
    def plot_confusion_matrix(self, cm: np.ndarray, output_path: str, 
                            normalize: bool = False, show_values: bool = True,
                            figsize: tuple = (20, 20), title: str = "Confusion Matrix"):
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            output_path: Path to save figure
            normalize: Whether to normalize by row (true labels)
            show_values: Whether to show numeric values in cells
            figsize: Figure size (width, height)
            title: Plot title
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        if normalize:
            # Normalize by row (true label)
            cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
        else:
            cm_normalized = cm
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(
            cm_normalized,
            annot=show_values and (len(self.label_list) <= 50),  # Don't show values if too many labels
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=self.label_list,
            yticklabels=self.label_list,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, pad=20)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {output_path}")
    
    def plot_entity_confusion_matrix(self, cm: np.ndarray, output_path: str,
                                    top_k: int = 20, normalize: bool = True):
        """
        Plot simplified confusion matrix for top-k most confused entities
        
        Args:
            cm: Full confusion matrix
            output_path: Path to save figure
            top_k: Number of top confused label pairs to show
            normalize: Whether to normalize
        """
        # Remove 'O' label for entity-focused analysis
        if 'O' in self.label_list:
            o_idx = self.label_list.index('O')
            mask = np.ones(len(self.label_list), dtype=bool)
            mask[o_idx] = False
            cm_no_o = cm[mask][:, mask]
            labels_no_o = [l for l in self.label_list if l != 'O']
        else:
            cm_no_o = cm
            labels_no_o = self.label_list
        
        if normalize:
            cm_no_o = cm_no_o.astype('float') / cm_no_o.sum(axis=1, keepdims=True)
            cm_no_o = np.nan_to_num(cm_no_o)
        
        # Find top-k confused pairs
        # Set diagonal to 0 (ignore correct predictions)
        cm_off_diag = cm_no_o.copy()
        np.fill_diagonal(cm_off_diag, 0)
        
        # Get top-k indices
        flat_indices = np.argsort(cm_off_diag.ravel())[-top_k:][::-1]
        top_k_indices = [np.unravel_index(idx, cm_off_diag.shape) for idx in flat_indices]
        
        # Create reduced matrix with only top-k labels
        unique_labels = set()
        for true_idx, pred_idx in top_k_indices:
            unique_labels.add(true_idx)
            unique_labels.add(pred_idx)
        
        unique_labels = sorted(list(unique_labels))
        reduced_cm = cm_no_o[np.ix_(unique_labels, unique_labels)]
        reduced_labels = [labels_no_o[i] for i in unique_labels]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            reduced_cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Reds',
            xticklabels=reduced_labels,
            yticklabels=reduced_labels,
            ax=ax,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title(f'Top-{top_k} Confused Entity Pairs', fontsize=13, pad=15)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Entity confusion matrix saved to: {output_path}")
    
    def get_confusion_summary(self, cm: np.ndarray) -> Dict[str, float]:
        """
        Get summary statistics from confusion matrix
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary with summary statistics
        """
        total = cm.sum()
        correct = np.trace(cm)
        accuracy = correct / total if total > 0 else 0
        
        # Per-class metrics
        per_class_precision = []
        per_class_recall = []
        
        for i in range(len(self.label_list)):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            per_class_precision.append(precision)
            per_class_recall.append(recall)
        
        return {
            'overall_accuracy': accuracy,
            'correct_predictions': int(correct),
            'total_predictions': int(total),
            'macro_precision': np.mean(per_class_precision),
            'macro_recall': np.mean(per_class_recall),
            'weighted_precision': np.average(per_class_precision, weights=cm.sum(axis=1)),
            'weighted_recall': np.average(per_class_recall, weights=cm.sum(axis=1))
        }


__all__ = ['ConfusionMatrixAnalyzer']
