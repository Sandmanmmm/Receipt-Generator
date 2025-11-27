"""
Model Loader - Utilities for loading trained models
"""
import torch
from pathlib import Path
from typing import Optional, Dict, Any
from transformers import LayoutLMv3Processor
import yaml


class ModelLoader:
    """Load trained LayoutLMv3 models for inference"""
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize model loader
        
        Args:
            model_path: Path to trained model directory
            device: Device to load model on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.processor = None
        self.config = None
        self.label_map = None
    
    def load_model(self):
        """Load model from checkpoint"""
        print(f"Loading model from {self.model_path}...")
        
        # Try to load multi-head model first
        try:
            from training.layoutlmv3_multihead import LayoutLMv3MultiHead
            
            # Load config
            config_path = self.model_path / 'config.yaml'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                
                # Load multi-head model
                self.model = LayoutLMv3MultiHead.from_pretrained(
                    str(self.model_path),
                    num_ner_labels=self.config.get('num_ner_labels', 73),
                    num_table_labels=self.config.get('num_table_labels', 3),
                    num_cell_labels=self.config.get('num_cell_labels', 3),
                    use_crf=self.config.get('use_crf', True)
                )
                print("✓ Loaded multi-head model")
            else:
                raise FileNotFoundError("config.yaml not found")
        
        except (ImportError, FileNotFoundError):
            # Fallback to standard LayoutLMv3
            from transformers import LayoutLMv3ForTokenClassification
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                str(self.model_path)
            )
            print("✓ Loaded standard LayoutLMv3 model")
        
        # Move to device
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            str(self.model_path)
        )
        
        # Load label map
        label_map_path = self.model_path / 'label_map.json'
        if label_map_path.exists():
            import json
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        
        print(f"✓ Model loaded on {self.device}")
    
    def predict(self, image_path: str, words: list, boxes: list) -> Dict[str, Any]:
        """
        Run inference on single document
        
        Args:
            image_path: Path to document image
            words: List of OCR words
            boxes: List of bounding boxes (normalized to 0-1000)
            
        Returns:
            Dictionary with predictions
        """
        if self.model is None:
            self.load_model()
        
        # Prepare inputs
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        encoding = self.processor(
            image,
            words,
            boxes=boxes,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Get predictions
        if hasattr(outputs, 'ner_logits'):
            # Multi-head model
            predictions = {
                'ner': outputs.ner_logits.argmax(dim=-1).cpu().numpy()[0],
                'table': outputs.table_logits.argmax(dim=-1).cpu().numpy()[0],
                'cell': outputs.cell_logits.argmax(dim=-1).cpu().numpy()[0]
            }
        else:
            # Standard model
            predictions = outputs.logits.argmax(dim=-1).cpu().numpy()[0]
        
        return {
            'predictions': predictions,
            'words': words,
            'boxes': boxes
        }
    
    def predict_batch(self, batch_data: list) -> list:
        """
        Run batch inference
        
        Args:
            batch_data: List of (image_path, words, boxes) tuples
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image_path, words, boxes in batch_data:
            result = self.predict(image_path, words, boxes)
            results.append(result)
        return results
    
    def decode_predictions(self, predictions: Dict[str, Any]) -> Dict[str, list]:
        """
        Decode predictions to label strings
        
        Args:
            predictions: Raw prediction dictionary
            
        Returns:
            Dictionary with decoded labels
        """
        if self.label_map is None:
            return predictions
        
        id2label = {int(k): v for k, v in self.label_map.items()}
        
        if isinstance(predictions['predictions'], dict):
            # Multi-head
            decoded = {}
            for task, preds in predictions['predictions'].items():
                decoded[task] = [id2label.get(int(p), 'O') for p in preds]
            return decoded
        else:
            # Standard
            return [id2label.get(int(p), 'O') for p in predictions['predictions']]


__all__ = ['ModelLoader']
