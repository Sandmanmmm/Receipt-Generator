"""
Model Evaluation Script
Evaluate LayoutLMv3 model performance on test set
"""
import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from pathlib import Path
import json
from typing import Dict, List
import numpy as np
from seqeval.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from tqdm import tqdm
import click

from training.train import InvoiceDataset
from .confusion_matrix import ConfusionMatrixAnalyzer
from .seqeval_report import SeqevalReporter
from .error_analysis import ErrorAnalyzer


class ModelEvaluator:
    """Evaluate trained LayoutLMv3 model"""
    
    def __init__(self, model_path: str, data_dir: str):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            data_dir: Path to dataset directory
        """
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        
        # Load model
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(self.model_path))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        # Load test dataset
        self.test_dataset = InvoiceDataset(str(self.data_dir), split='test')
        self.id2label = self.test_dataset.id2label
        
        click.echo(f"Loaded model from {model_path}")
        click.echo(f"Using device: {self.device}")
        click.echo(f"Test dataset size: {len(self.test_dataset)}")
    
    def evaluate(self, batch_size: int = 8) -> Dict[str, float]:
        """
        Run evaluation on test set
        
        Args:
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of metrics
        """
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        all_predictions = []
        all_labels = []
        
        click.echo("\nEvaluating model...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                bbox = batch['bbox'].to(self.device)
                pixel_values = batch['pixel_values'].to(self.device)
                labels = batch['labels']
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bbox=bbox,
                    pixel_values=pixel_values
                )
                
                # Get predictions
                predictions = outputs.logits.argmax(dim=-1)
                
                # Move back to CPU
                predictions = predictions.cpu().numpy()
                labels = labels.numpy()
                attention_mask = attention_mask.cpu().numpy()
                
                # Process batch
                for pred, label, mask in zip(predictions, labels, attention_mask):
                    # Remove padding and special tokens
                    valid_indices = (mask == 1) & (label != -100)
                    
                    pred_labels = [self.id2label[p] for p in pred[valid_indices]]
                    true_labels = [self.id2label[l] for l in label[valid_indices]]
                    
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'precision': precision_score(all_labels, all_predictions),
            'recall': recall_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions)
        }
        
        return metrics, all_labels, all_predictions
    
    def generate_report(self) -> str:
        """Generate detailed classification report"""
        metrics, true_labels, predictions = self.evaluate()
        
        report = classification_report(true_labels, predictions, digits=4)
        
        return report, metrics
    
    def evaluate_by_entity(self) -> Dict[str, Dict[str, float]]:
        """Calculate per-entity metrics"""
        metrics, true_labels, predictions = self.evaluate()
        
        # Get unique entity types
        all_entities = set()
        for labels in true_labels:
            all_entities.update(labels)
        
        entity_metrics = {}
        
        for entity in all_entities:
            if entity == 'O':
                continue
            
            # Extract this entity only
            entity_true = []
            entity_pred = []
            
            for true, pred in zip(true_labels, predictions):
                entity_true.append([l if l.endswith(entity.split('-')[-1]) or l == 'O' else 'O' for l in true])
                entity_pred.append([p if p.endswith(entity.split('-')[-1]) or p == 'O' else 'O' for p in pred])
            
            try:
                entity_metrics[entity] = {
                    'precision': precision_score([entity_true], [entity_pred]),
                    'recall': recall_score([entity_true], [entity_pred]),
                    'f1': f1_score([entity_true], [entity_pred])
                }
            except:
                entity_metrics[entity] = {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
        
        return entity_metrics
    
    def confusion_analysis(self, top_n: int = 20) -> List[Dict]:
        """
        Analyze common confusion patterns
        
        Args:
            top_n: Number of top confusions to return
            
        Returns:
            List of confusion patterns
        """
        _, true_labels, predictions = self.evaluate()
        
        # Count confusions
        confusions = {}
        
        for true_seq, pred_seq in zip(true_labels, predictions):
            for true_label, pred_label in zip(true_seq, pred_seq):
                if true_label != pred_label:
                    key = (true_label, pred_label)
                    confusions[key] = confusions.get(key, 0) + 1
        
        # Sort by frequency
        sorted_confusions = sorted(
            confusions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        return [
            {
                'true_label': true,
                'predicted_label': pred,
                'count': count
            }
            for (true, pred), count in sorted_confusions
        ]
    
    def save_results(self, output_path: str):
        """Save evaluation results to file"""
        report, metrics = self.generate_report()
        entity_metrics = self.evaluate_by_entity()
        confusions = self.confusion_analysis()
        
        results = {
            'overall_metrics': metrics,
            'entity_metrics': entity_metrics,
            'classification_report': report,
            'top_confusions': confusions
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        click.echo(f"\n✓ Results saved to {output_path}")


@click.command()
@click.option('--model-path', '-m', required=True, help='Path to trained model')
@click.option('--data-dir', '-d', default='data/layoutlmv3', help='Dataset directory')
@click.option('--output', '-o', default='evaluation/results.json', help='Output file')
@click.option('--batch-size', '-b', default=8, help='Batch size')
def main(model_path, data_dir, output, batch_size):
    """Evaluate LayoutLMv3 model on test set"""
    
    click.echo("="*60)
    click.echo("MODEL EVALUATION")
    click.echo("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model_path, data_dir)
    
    # Run evaluation
    metrics, _, _ = evaluator.evaluate(batch_size)
    
    # Print results
    click.echo("\n" + "="*60)
    click.echo("OVERALL METRICS")
    click.echo("="*60)
    for metric, value in metrics.items():
        click.echo(f"{metric:.<30} {value:.4f}")
    
    # Generate detailed report
    report, _ = evaluator.generate_report()
    click.echo("\n" + "="*60)
    click.echo("CLASSIFICATION REPORT")
    click.echo("="*60)
    click.echo(report)
    
    # Per-entity metrics
    entity_metrics = evaluator.evaluate_by_entity()
    click.echo("\n" + "="*60)
    click.echo("PER-ENTITY METRICS")
    click.echo("="*60)
    for entity, scores in sorted(entity_metrics.items()):
        click.echo(f"\n{entity}:")
        click.echo(f"  Precision: {scores['precision']:.4f}")
        click.echo(f"  Recall:    {scores['recall']:.4f}")
        click.echo(f"  F1:        {scores['f1']:.4f}")
    
    # Confusion analysis
    confusions = evaluator.confusion_analysis(top_n=10)
    click.echo("\n" + "="*60)
    click.echo("TOP CONFUSIONS")
    click.echo("="*60)
    for i, conf in enumerate(confusions, 1):
        click.echo(f"{i}. {conf['true_label']} → {conf['predicted_label']}: {conf['count']} times")
    
    # Save results
    evaluator.save_results(output)
    
    # Run enhanced evaluation with new tools
    click.echo("\n" + "="*60)
    click.echo("ENHANCED EVALUATION (confusion matrix, error analysis)")
    click.echo("="*60)
    
    try:
        # Get predictions and tokens for enhanced analysis
        _, true_labels, pred_labels = evaluator.evaluate(batch_size)
        
        # Get tokens (simplified - in production would extract from dataset)
        tokens = [['token'] * len(seq) for seq in true_labels]  # Placeholder
        
        # Get label list
        label_list = list(evaluator.id2label.values())
        
        # Confusion Matrix Analysis
        cm_analyzer = ConfusionMatrixAnalyzer(label_list)
        cm = cm_analyzer.compute_confusion_matrix(true_labels, pred_labels)
        
        output_dir = Path(output).parent / "enhanced_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cm_analyzer.plot_confusion_matrix(
            cm,
            str(output_dir / "confusion_matrix_normalized.png"),
            normalize=True,
            title="Confusion Matrix (Normalized)"
        )
        
        cm_analyzer.plot_entity_confusion_matrix(
            cm,
            str(output_dir / "confusion_matrix_entities.png"),
            top_k=20
        )
        
        # Seqeval Detailed Report
        seqeval = SeqevalReporter(scheme='IOB2')
        seqeval.generate_report(
            true_labels, pred_labels,
            output_path=str(output_dir / "seqeval_report.txt")
        )
        seqeval.generate_json_report(
            true_labels, pred_labels,
            output_path=str(output_dir / "seqeval_report.json")
        )
        
        # Error Analysis
        error_analyzer = ErrorAnalyzer(label_list)
        errors = error_analyzer.categorize_errors(true_labels, pred_labels, tokens)
        error_analyzer.generate_error_report(
            errors,
            output_path=str(output_dir / "error_report.txt")
        )
        error_analyzer.export_errors_json(
            errors,
            output_path=str(output_dir / "errors.json")
        )
        
        click.echo(f"✓ Enhanced analysis saved to: {output_dir}")
    
    except Exception as e:
        click.echo(f"Warning: Enhanced analysis failed: {e}")
    
    click.echo("\n" + "="*60)
    click.echo("EVALUATION COMPLETE")
    click.echo("="*60)


if __name__ == '__main__':
    main()
