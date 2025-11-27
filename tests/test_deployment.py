"""
Test Deployment Utilities
"""
import pytest
from pathlib import Path
import tempfile
import json

from deployment import ModelLoader, BatchRunner


class TestModelLoader:
    """Test Model Loader"""
    
    @pytest.mark.skipif(True, reason="Requires trained model")
    def test_load_model(self):
        """Test loading trained model"""
        loader = ModelLoader(
            model_path="models/layoutlmv3_multihead",
            config_path="config/training_config.yaml"
        )
        
        assert loader.model is not None
        assert loader.tokenizer is not None
        assert loader.config is not None
    
    def test_decode_predictions(self):
        """Test prediction decoding"""
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG']
        
        loader = ModelLoader(
            model_path="models/layoutlmv3_multihead",
            config_path="config/training_config.yaml"
        )
        loader.label_list = label_list
        
        predictions = [0, 1, 2, 3]
        decoded = loader.decode_predictions(predictions)
        
        assert decoded == ['O', 'B-PER', 'I-PER', 'B-ORG']


class TestBatchRunner:
    """Test Batch Runner"""
    
    def test_init(self):
        """Test initialization"""
        runner = BatchRunner(
            model_loader=None,
            batch_size=8,
            max_workers=4
        )
        
        assert runner.batch_size == 8
        assert runner.max_workers == 4
    
    @pytest.mark.skipif(True, reason="Requires OCR setup")
    def test_ocr_function(self):
        """Test OCR function"""
        from annotation import OCREngine
        
        runner = BatchRunner(
            model_loader=None,
            batch_size=8,
            ocr_engine='paddleocr'
        )
        
        # Should create OCR function
        assert runner.ocr_fn is not None
    
    @pytest.mark.slow
    @pytest.mark.skipif(True, reason="Requires full setup")
    def test_process_batch(self):
        """Test batch processing"""
        # This would require a complete setup
        pass


class TestAsyncBatchRunner:
    """Test Async Batch Runner"""
    
    def test_init(self):
        """Test initialization"""
        from deployment import AsyncBatchRunner
        
        runner = AsyncBatchRunner(
            model_loader=None,
            batch_size=16,
            max_workers=8
        )
        
        assert runner.batch_size == 16
        assert runner.max_workers == 8


class TestDeploymentIntegration:
    """Test deployment integration"""
    
    @pytest.mark.skipif(not Path("deployment/api.py").exists(),
                       reason="API file not found")
    def test_api_file_exists(self):
        """Test API deployment file exists"""
        assert Path("deployment/api.py").exists()
    
    def test_model_directory_structure(self):
        """Test expected model directory structure"""
        models_dir = Path("models")
        
        if models_dir.exists():
            # Check if any model directories exist
            model_dirs = [d for d in models_dir.iterdir() if d.is_dir()]
            
            # For each model dir, check expected files
            for model_dir in model_dirs:
                if (model_dir / "config.json").exists():
                    assert (model_dir / "pytorch_model.bin").exists() or \
                           (model_dir / "model.safetensors").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
