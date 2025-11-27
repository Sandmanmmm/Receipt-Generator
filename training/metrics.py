"""
Training Metrics - Compute metrics during training
"""
import numpy as np
from typing import Dict, List, Tuple
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.metrics import accuracy_score as sklearn_accuracy


class NERMetrics:
    """Compute NER metrics using seqeval"""
    
    def __init__(self, label_list: List[str], ignore_index: int = -100):
        """
        Initialize metrics calculator
        
        Args:
            label_list: List of all labels
            ignore_index: Index to ignore in labels (padding)
        """
        self.label_list = label_list
        self.ignore_index = ignore_index
    
    def compute_metrics(self, predictions: np.ndarray, 
                       labels: np.ndarray) -> Dict[str, float]:
        """
        Compute NER metrics from predictions and labels
        
        Args:
            predictions: Predicted label IDs (batch_size, seq_length)
            labels: True label IDs (batch_size, seq_length)
            
        Returns:
            Dictionary with metrics
        """
        # Convert IDs to label strings
        true_labels = []
        pred_labels = []
        
        for pred_seq, label_seq in zip(predictions, labels):
            true_seq = []
            pred_seq_labels = []
            
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id != self.ignore_index:
                    true_seq.append(self.label_list[label_id])
                    pred_seq_labels.append(self.label_list[pred_id])
            
            if true_seq:  # Only add non-empty sequences
                true_labels.append(true_seq)
                pred_labels.append(pred_seq_labels)
        
        # Compute seqeval metrics
        return {
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'f1': f1_score(true_labels, pred_labels),
        }
    
    def compute_token_accuracy(self, predictions: np.ndarray,
                               labels: np.ndarray) -> float:
        """
        Compute token-level accuracy
        
        Args:
            predictions: Predicted label IDs
            labels: True label IDs
            
        Returns:
            Token accuracy
        """
        # Flatten and filter out ignore_index
        mask = labels != self.ignore_index
        valid_preds = predictions[mask]
        valid_labels = labels[mask]
        
        if len(valid_labels) == 0:
            return 0.0
        
        return (valid_preds == valid_labels).mean()


class MultiTaskMetrics:
    """Compute metrics for multi-task learning"""
    
    def __init__(self, ner_label_list: List[str],
                 table_label_list: List[str],
                 cell_label_list: List[str],
                 ignore_index: int = -100):
        """
        Initialize multi-task metrics
        
        Args:
            ner_label_list: NER labels
            table_label_list: Table detection labels
            cell_label_list: Cell attribute labels
            ignore_index: Padding index to ignore
        """
        self.ner_metrics = NERMetrics(ner_label_list, ignore_index)
        self.table_metrics = NERMetrics(table_label_list, ignore_index)
        self.cell_metrics = NERMetrics(cell_label_list, ignore_index)
        self.ignore_index = ignore_index
    
    def compute_metrics(self, predictions: Dict[str, np.ndarray],
                       labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute all task metrics
        
        Args:
            predictions: Dictionary with task predictions
            labels: Dictionary with task labels
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # NER metrics
        if 'ner' in predictions:
            ner_metrics = self.ner_metrics.compute_metrics(
                predictions['ner'], labels['ner']
            )
            metrics.update({f'ner_{k}': v for k, v in ner_metrics.items()})
        
        # Table metrics
        if 'table' in predictions:
            table_metrics = self.table_metrics.compute_metrics(
                predictions['table'], labels['table']
            )
            metrics.update({f'table_{k}': v for k, v in table_metrics.items()})
        
        # Cell metrics
        if 'cell' in predictions:
            cell_metrics = self.cell_metrics.compute_metrics(
                predictions['cell'], labels['cell']
            )
            metrics.update({f'cell_{k}': v for k, v in cell_metrics.items()})
        
        # Overall F1 (weighted average)
        if all(k in predictions for k in ['ner', 'table', 'cell']):
            metrics['overall_f1'] = (
                0.6 * metrics['ner_f1'] +
                0.3 * metrics['table_f1'] +
                0.1 * metrics['cell_f1']
            )
        
        return metrics


class MetricsTracker:
    """Track metrics over training epochs"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.history = {
            'train': [],
            'val': []
        }
        self.best_val_f1 = 0.0
        self.best_epoch = 0
    
    def update(self, metrics: Dict[str, float], split: str = 'train'):
        """
        Update metrics history
        
        Args:
            metrics: Dictionary with metrics
            split: 'train' or 'val'
        """
        self.history[split].append(metrics)
        
        # Track best validation F1
        if split == 'val' and 'f1' in metrics:
            if metrics['f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['f1']
                self.best_epoch = len(self.history['val']) - 1
    
    def get_best_metrics(self) -> Tuple[int, float]:
        """Get best validation metrics"""
        return self.best_epoch, self.best_val_f1
    
    def get_latest_metrics(self, split: str = 'val') -> Dict[str, float]:
        """Get latest metrics for split"""
        if self.history[split]:
            return self.history[split][-1]
        return {}
    
    def summary(self) -> str:
        """Generate summary string"""
        lines = []
        lines.append("=" * 60)
        lines.append("TRAINING SUMMARY")
        lines.append("=" * 60)
        
        if self.history['val']:
            lines.append(f"Best Epoch: {self.best_epoch + 1}")
            lines.append(f"Best Val F1: {self.best_val_f1:.4f}")
            
            latest = self.get_latest_metrics('val')
            lines.append(f"\nLatest Val Metrics:")
            for key, value in latest.items():
                lines.append(f"  {key}: {value:.4f}")
        
        return "\n".join(lines)


__all__ = ['NERMetrics', 'MultiTaskMetrics', 'MetricsTracker']
