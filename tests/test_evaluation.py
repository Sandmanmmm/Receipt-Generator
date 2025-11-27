"""
Test Evaluation Tools
"""
import pytest
import numpy as np
from evaluation import ConfusionMatrixAnalyzer, SeqevalReporter, ErrorAnalyzer
import tempfile
from pathlib import Path


class TestConfusionMatrixAnalyzer:
    """Test Confusion Matrix generation"""
    
    def test_compute_confusion_matrix(self):
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        analyzer = ConfusionMatrixAnalyzer(label_list)
        
        y_true = [['B-PER', 'I-PER', 'O', 'B-ORG']]
        y_pred = [['B-PER', 'I-PER', 'O', 'O']]
        
        cm = analyzer.compute_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (4, 4)
        assert cm.sum() == 4  # Total tokens
    
    def test_get_confusion_summary(self):
        label_list = ['O', 'B-PER', 'I-PER']
        analyzer = ConfusionMatrixAnalyzer(label_list)
        
        y_true = [['B-PER', 'I-PER', 'O']]
        y_pred = [['B-PER', 'I-PER', 'O']]
        
        cm = analyzer.compute_confusion_matrix(y_true, y_pred)
        summary = analyzer.get_confusion_summary(cm)
        
        assert 'overall_accuracy' in summary
        assert summary['overall_accuracy'] == 1.0
        assert summary['correct_predictions'] == 3


class TestSeqevalReporter:
    """Test Seqeval reporting"""
    
    def test_compute_metrics(self):
        reporter = SeqevalReporter(scheme='IOB2')
        
        y_true = [['B-PER', 'I-PER', 'O', 'B-ORG']]
        y_pred = [['B-PER', 'I-PER', 'O', 'B-ORG']]
        
        metrics = reporter.compute_metrics(y_true, y_pred)
        
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_get_entity_metrics(self):
        reporter = SeqevalReporter(scheme='IOB2')
        
        y_true = [['B-PER', 'I-PER', 'O', 'B-ORG', 'O']]
        y_pred = [['B-PER', 'I-PER', 'O', 'O', 'O']]
        
        entity_metrics = reporter.get_entity_metrics(y_true, y_pred)
        
        assert 'PER' in entity_metrics
        assert 'ORG' in entity_metrics
        assert entity_metrics['PER']['f1'] == 1.0
        assert entity_metrics['ORG']['f1'] == 0.0  # Missed
    
    def test_rank_entities_by_metric(self):
        reporter = SeqevalReporter(scheme='IOB2')
        
        y_true = [['B-PER', 'O', 'B-ORG', 'O', 'B-LOC']]
        y_pred = [['B-PER', 'O', 'O', 'O', 'B-LOC']]
        
        ranked = reporter.rank_entities_by_metric(y_true, y_pred, metric='f1')
        
        assert len(ranked) == 3  # PER, ORG, LOC
        assert ranked[0][1] == 1.0  # Best F1


class TestErrorAnalyzer:
    """Test Error Analysis"""
    
    def test_extract_entities(self):
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        analyzer = ErrorAnalyzer(label_list)
        
        labels = ['B-PER', 'I-PER', 'O', 'B-ORG']
        tokens = ['John', 'Smith', 'works', 'at', 'Google']
        
        entities = analyzer.extract_entities(labels, tokens)
        
        assert len(entities) == 2
        assert entities[0]['type'] == 'PER'
        assert entities[0]['text'] == 'John Smith'
        assert entities[1]['type'] == 'ORG'
    
    def test_categorize_errors(self):
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        analyzer = ErrorAnalyzer(label_list)
        
        y_true = [['B-PER', 'I-PER', 'O', 'B-ORG']]
        y_pred = [['B-PER', 'I-PER', 'O', 'O']]
        tokens = [['John', 'Smith', 'at', 'Google']]
        
        errors = analyzer.categorize_errors(y_true, y_pred, tokens)
        
        assert 'false_positive' in errors
        assert 'false_negative' in errors
        assert 'wrong_type' in errors
        assert len(errors['false_negative']) == 1  # Missed ORG
    
    def test_summarize_errors(self):
        label_list = ['O', 'B-PER']
        analyzer = ErrorAnalyzer(label_list)
        
        errors = {
            'false_positive': [{'text': 'test'}],
            'false_negative': [{'text': 'test1'}, {'text': 'test2'}],
            'wrong_type': []
        }
        
        summary = analyzer.summarize_errors(errors)
        
        assert summary['false_positive'] == 1
        assert summary['false_negative'] == 2
        assert summary['wrong_type'] == 0


class TestModelEvaluatorIntegration:
    """Test integrated evaluation"""
    
    def test_full_evaluation_pipeline(self):
        from evaluation import ModelEvaluator
        
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluator = ModelEvaluator(label_list, output_dir=tmpdir)
            
            y_true = [['B-PER', 'I-PER', 'O', 'B-ORG', 'O']]
            y_pred = [['B-PER', 'I-PER', 'O', 'B-ORG', 'O']]
            tokens = [['John', 'Smith', 'at', 'Google', 'Inc']]
            
            summary = evaluator.evaluate_full(y_true, y_pred, tokens, prefix='test')
            
            assert 'confusion_matrix_summary' in summary
            assert 'overall_metrics' in summary
            assert 'error_summary' in summary
            
            # Check files created
            output_path = Path(tmpdir)
            assert (output_path / 'test_confusion_matrix_full.png').exists()
            assert (output_path / 'test_seqeval_report.txt').exists()
            assert (output_path / 'test_errors.json').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
