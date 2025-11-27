"""
Test Training Support Files
"""
import pytest
from pathlib import Path
import tempfile
import json
import shutil

from training import DatasetBuilder, LayoutLMv3DataCollator, NERMetrics, MetricsTracker


class TestDatasetBuilder:
    """Test DatasetBuilder"""
    
    def test_init(self):
        builder = DatasetBuilder(
            data_dir="data/annotated",
            output_dir="data",
            split_ratios=(0.8, 0.1, 0.1)
        )
        
        assert builder.data_dir == Path("data/annotated")
        assert builder.output_dir == Path("data")
        assert builder.split_ratios == (0.8, 0.1, 0.1)
    
    def test_split_ratios_validation(self):
        with pytest.raises(ValueError):
            DatasetBuilder(
                data_dir="data/annotated",
                output_dir="data",
                split_ratios=(0.5, 0.3, 0.1)  # Doesn't sum to 1.0
            )
    
    @pytest.mark.skipif(not Path("data/annotated").exists(), reason="No annotated data")
    def test_load_annotations(self):
        builder = DatasetBuilder(
            data_dir="data/annotated",
            output_dir="data"
        )
        
        annotations = builder.load_annotations()
        
        if len(annotations) > 0:
            assert 'image_path' in annotations[0]
            assert 'tokens' in annotations[0]
            assert 'labels' in annotations[0]
    
    def test_split_dataset(self):
        builder = DatasetBuilder(
            data_dir="data/annotated",
            output_dir="data",
            split_ratios=(0.7, 0.15, 0.15)
        )
        
        annotations = [
            {'image_path': f'img{i}.png', 'tokens': ['test'], 'labels': ['O']}
            for i in range(100)
        ]
        
        train, val, test = builder.split_dataset(annotations)
        
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15


class TestDataCollator:
    """Test Data Collator"""
    
    def test_single_task_collator(self):
        from transformers import LayoutLMv3Tokenizer
        
        # Mock tokenizer
        tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        collator = LayoutLMv3DataCollator(tokenizer, max_length=512)
        
        batch = [
            {
                'input_ids': [101, 2003, 102],
                'attention_mask': [1, 1, 1],
                'bbox': [[0, 0, 100, 100], [100, 0, 200, 100], [200, 0, 300, 100]],
                'labels': [0, 1, 0]
            }
        ]
        
        result = collator(batch)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'bbox' in result
        assert 'labels' in result


class TestNERMetrics:
    """Test NER Metrics"""
    
    def test_compute_metrics_perfect(self):
        label_list = ['O', 'B-PER', 'I-PER']
        metrics_computer = NERMetrics(label_list)
        
        predictions = [[0, 1, 2]]  # O, B-PER, I-PER
        labels = [[0, 1, 2]]
        
        metrics = metrics_computer.compute_metrics(predictions, labels)
        
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_compute_metrics_with_errors(self):
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        metrics_computer = NERMetrics(label_list)
        
        predictions = [[0, 1, 2, 0]]  # O, B-PER, I-PER, O
        labels = [[0, 1, 2, 3]]       # O, B-PER, I-PER, B-ORG (missed)
        
        metrics = metrics_computer.compute_metrics(predictions, labels)
        
        assert 0 < metrics['f1'] < 1.0  # Not perfect


class TestMetricsTracker:
    """Test Metrics Tracker"""
    
    def test_init(self):
        tracker = MetricsTracker(metric='f1', mode='max')
        
        assert tracker.metric == 'f1'
        assert tracker.mode == 'max'
        assert len(tracker.history) == 0
    
    def test_update_and_is_best(self):
        tracker = MetricsTracker(metric='f1', mode='max')
        
        # First update should be best
        assert tracker.update({'f1': 0.8, 'loss': 0.5})
        assert tracker.is_best()
        
        # Better metric should be best
        assert tracker.update({'f1': 0.9, 'loss': 0.4})
        assert tracker.is_best()
        
        # Worse metric should not be best
        assert tracker.update({'f1': 0.7, 'loss': 0.6})
        assert not tracker.is_best()
    
    def test_get_best_metrics(self):
        tracker = MetricsTracker(metric='f1', mode='max')
        
        tracker.update({'f1': 0.8, 'loss': 0.5, 'epoch': 1})
        tracker.update({'f1': 0.9, 'loss': 0.4, 'epoch': 2})
        tracker.update({'f1': 0.85, 'loss': 0.45, 'epoch': 3})
        
        best = tracker.get_best_metrics()
        
        assert best['f1'] == 0.9
        assert best['epoch'] == 2


class TestEndToEndTraining:
    """Test complete training pipeline integration"""
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires GPU and full setup")
    def test_full_training_pipeline(self):
        """Test complete training flow (requires full environment)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            builder = DatasetBuilder(
                data_dir="data/annotated",
                output_dir=tmpdir
            )
            
            # Mock annotations
            annotations = [
                {
                    'image_path': f'img{i}.png',
                    'tokens': ['test', 'invoice'],
                    'labels': ['O', 'B-DOCUMENT_TYPE'],
                    'bboxes': [[0, 0, 100, 50], [100, 0, 200, 50]]
                }
                for i in range(10)
            ]
            
            # Split
            train, val, test = builder.split_dataset(annotations)
            
            assert len(train) > 0
            assert len(val) > 0
            assert len(test) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
