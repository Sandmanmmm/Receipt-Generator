"""
Seqeval-based NER Evaluation Reporting
"""
from seqeval.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from seqeval.scheme import IOB2
from typing import List, Dict, Optional
import json
from pathlib import Path


class SeqevalReporter:
    """Generate comprehensive NER evaluation reports using seqeval"""
    
    def __init__(self, scheme: str = 'IOB2'):
        """
        Initialize reporter
        
        Args:
            scheme: Tagging scheme ('IOB2', 'IOBES', 'BILOU')
        """
        self.scheme = IOB2 if scheme == 'IOB2' else None
    
    def compute_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, float]:
        """
        Compute overall metrics
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            Dictionary with precision, recall, f1, accuracy
        """
        return {
            'precision': precision_score(y_true, y_pred, scheme=self.scheme),
            'recall': recall_score(y_true, y_pred, scheme=self.scheme),
            'f1': f1_score(y_true, y_pred, scheme=self.scheme),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    def generate_report(self, y_true: List[List[str]], y_pred: List[List[str]], 
                       output_path: Optional[str] = None, 
                       digits: int = 4) -> str:
        """
        Generate detailed classification report
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            output_path: Optional path to save report
            digits: Number of decimal places
            
        Returns:
            Report string
        """
        report = classification_report(
            y_true, 
            y_pred, 
            scheme=self.scheme,
            digits=digits,
            output_dict=False
        )
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report
    
    def generate_report_dict(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
        """
        Generate classification report as dictionary
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            Report dictionary with per-entity metrics
        """
        return classification_report(
            y_true, 
            y_pred, 
            scheme=self.scheme,
            output_dict=True
        )
    
    def get_entity_metrics(self, y_true: List[List[str]], y_pred: List[List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Get per-entity type metrics (groups B- and I- tags)
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            
        Returns:
            Dictionary mapping entity type to metrics
        """
        report_dict = self.generate_report_dict(y_true, y_pred)
        
        # Extract per-entity metrics (seqeval groups B- and I- automatically)
        entity_metrics = {}
        for key, value in report_dict.items():
            if key not in ['micro avg', 'macro avg', 'weighted avg']:
                entity_metrics[key] = value
        
        return entity_metrics
    
    def rank_entities_by_metric(self, y_true: List[List[str]], y_pred: List[List[str]], 
                                metric: str = 'f1', ascending: bool = False) -> List[tuple]:
        """
        Rank entity types by performance metric
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            metric: Metric to rank by ('precision', 'recall', 'f1')
            ascending: If True, rank from worst to best
            
        Returns:
            List of (entity_type, score) tuples sorted by metric
        """
        entity_metrics = self.get_entity_metrics(y_true, y_pred)
        
        ranked = [
            (entity, metrics[metric]) 
            for entity, metrics in entity_metrics.items()
            if metric in metrics
        ]
        
        ranked.sort(key=lambda x: x[1], reverse=not ascending)
        return ranked
    
    def generate_json_report(self, y_true: List[List[str]], y_pred: List[List[str]], 
                            output_path: str):
        """
        Generate JSON report with all metrics
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            output_path: Path to save JSON report
        """
        overall_metrics = self.compute_metrics(y_true, y_pred)
        report_dict = self.generate_report_dict(y_true, y_pred)
        
        full_report = {
            'overall': overall_metrics,
            'per_entity': report_dict
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"JSON report saved to: {output_path}")
    
    def compare_models(self, results: Dict[str, Dict], 
                      output_path: Optional[str] = None) -> str:
        """
        Compare multiple model results
        
        Args:
            results: Dictionary mapping model name to (y_true, y_pred) tuple
            output_path: Optional path to save comparison
            
        Returns:
            Comparison report string
        """
        comparison = []
        comparison.append("=" * 80)
        comparison.append("MODEL COMPARISON")
        comparison.append("=" * 80)
        comparison.append("")
        
        # Header
        comparison.append(f"{'Model':<30} {'Precision':>12} {'Recall':>12} {'F1':>12}")
        comparison.append("-" * 80)
        
        # Compute metrics for each model
        model_metrics = {}
        for model_name, (y_true, y_pred) in results.items():
            metrics = self.compute_metrics(y_true, y_pred)
            model_metrics[model_name] = metrics
            
            comparison.append(
                f"{model_name:<30} "
                f"{metrics['precision']:>12.4f} "
                f"{metrics['recall']:>12.4f} "
                f"{metrics['f1']:>12.4f}"
            )
        
        comparison.append("=" * 80)
        
        # Find best model
        best_model = max(model_metrics.items(), key=lambda x: x[1]['f1'])
        comparison.append(f"\nBest Model (by F1): {best_model[0]} (F1={best_model[1]['f1']:.4f})")
        
        report = "\n".join(comparison)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Comparison saved to: {output_path}")
        
        return report
    
    def print_top_bottom_entities(self, y_true: List[List[str]], y_pred: List[List[str]], 
                                  top_n: int = 10, metric: str = 'f1'):
        """
        Print top and bottom performing entities
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            top_n: Number of entities to show
            metric: Metric to rank by
        """
        ranked = self.rank_entities_by_metric(y_true, y_pred, metric=metric, ascending=False)
        
        print(f"\n{'='*60}")
        print(f"TOP {top_n} ENTITIES (by {metric})")
        print(f"{'='*60}")
        for entity, score in ranked[:top_n]:
            print(f"{entity:<40} {score:.4f}")
        
        print(f"\n{'='*60}")
        print(f"BOTTOM {top_n} ENTITIES (by {metric})")
        print(f"{'='*60}")
        for entity, score in ranked[-top_n:]:
            print(f"{entity:<40} {score:.4f}")


__all__ = ['SeqevalReporter']
