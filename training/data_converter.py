"""
LayoutLMv3 Data Converter
Converts annotated invoices to LayoutLMv3 training format
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import torch
from PIL import Image
from transformers import LayoutLMv3Processor
from annotation.annotator import InvoiceAnnotation


@dataclass
class LayoutLMv3Sample:
    """Single training sample for LayoutLMv3"""
    image: Image.Image
    words: List[str]
    boxes: List[List[int]]  # Normalized [x0, y0, x1, y1] in 0-1000 range
    labels: List[int]  # Entity labels as integers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (without image)"""
        return {
            'words': self.words,
            'boxes': self.boxes,
            'labels': self.labels
        }


class LayoutLMv3Converter:
    """Converts annotations to LayoutLMv3 format"""
    
    def __init__(self, 
                 model_name: str = "microsoft/layoutlmv3-base",
                 label_list: Optional[List[str]] = None):
        """
        Initialize converter
        
        Args:
            model_name: HuggingFace model name
            label_list: List of entity labels (auto-generated if None)
        """
        self.model_name = model_name
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=False)
        
        # Entity labels
        if label_list is None:
            self.label_list = [
                'O',  # Outside any entity
                'B-INVOICE_NUMBER',
                'I-INVOICE_NUMBER',
                'B-DATE',
                'I-DATE',
                'B-TOTAL',
                'I-TOTAL',
                'B-COMPANY_NAME',
                'I-COMPANY_NAME',
                'B-ADDRESS',
                'I-ADDRESS',
                'B-PHONE',
                'I-PHONE',
                'B-EMAIL',
                'I-EMAIL',
                'B-CLIENT_NAME',
                'I-CLIENT_NAME',
                'B-ITEM_DESCRIPTION',
                'I-ITEM_DESCRIPTION',
                'B-QUANTITY',
                'I-QUANTITY',
                'B-RATE',
                'I-RATE',
                'B-AMOUNT',
                'I-AMOUNT',
                'B-SUBTOTAL',
                'I-SUBTOTAL',
                'B-TAX',
                'I-TAX',
                'B-DISCOUNT',
                'I-DISCOUNT',
            ]
        else:
            self.label_list = label_list
        
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for idx, label in enumerate(self.label_list)}
    
    def normalize_box(self, box: List[int], width: int, height: int) -> List[int]:
        """
        Normalize bounding box coordinates to 0-1000 range
        
        Args:
            box: [x, y, x2, y2] in pixel coordinates
            width: Image width
            height: Image height
            
        Returns:
            Normalized box [x, y, x2, y2] in 0-1000 range
        """
        return [
            int(1000 * box[0] / width),
            int(1000 * box[1] / height),
            int(1000 * box[2] / width),
            int(1000 * box[3] / height)
        ]
    
    def annotation_to_sample(self, annotation: InvoiceAnnotation) -> LayoutLMv3Sample:
        """
        Convert InvoiceAnnotation to LayoutLMv3Sample
        
        Args:
            annotation: InvoiceAnnotation object
            
        Returns:
            LayoutLMv3Sample object
        """
        # Load image
        image = Image.open(annotation.image_path).convert("RGB")
        width, height = annotation.image_width, annotation.image_height
        
        words = []
        boxes = []
        labels = []
        
        for box in annotation.boxes:
            # Extract text (may need word-level splitting)
            text = box.text.strip()
            if not text:
                continue
            
            # Split into words
            word_list = text.split()
            
            # Create bounding box for each word (simplified - all get same box)
            # In production, you'd want word-level boxes
            bbox = [box.x, box.y, box.x2, box.y2]
            normalized_bbox = self.normalize_box(bbox, width, height)
            
            # Determine label
            if box.label:
                label_key = box.label.upper()
                # Use BIO tagging
                for i, word in enumerate(word_list):
                    words.append(word)
                    boxes.append(normalized_bbox)
                    
                    if i == 0:
                        label = self.label2id.get(f'B-{label_key}', 0)
                    else:
                        label = self.label2id.get(f'I-{label_key}', 0)
                    
                    labels.append(label)
            else:
                # Outside any entity
                for word in word_list:
                    words.append(word)
                    boxes.append(normalized_bbox)
                    labels.append(0)  # 'O' label
        
        return LayoutLMv3Sample(
            image=image,
            words=words,
            boxes=boxes,
            labels=labels
        )
    
    def prepare_training_sample(self, sample: LayoutLMv3Sample) -> Dict[str, Any]:
        """
        Prepare sample for training using processor
        
        Args:
            sample: LayoutLMv3Sample
            
        Returns:
            Dictionary with model inputs
        """
        encoding = self.processor(
            sample.image,
            sample.words,
            boxes=sample.boxes,
            word_labels=sample.labels,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'bbox': encoding['bbox'].squeeze(0),
            'pixel_values': encoding['pixel_values'].squeeze(0),
            'labels': encoding['labels'].squeeze(0)
        }
    
    def convert_dataset(self,
                       annotation_dir: str,
                       output_dir: str,
                       split: str = 'train') -> List[str]:
        """
        Convert directory of annotations to LayoutLMv3 format
        
        Args:
            annotation_dir: Directory containing JSON annotations
            output_dir: Output directory for processed samples
            split: Dataset split ('train', 'val', 'test')
            
        Returns:
            List of output file paths
        """
        annotation_path = Path(annotation_dir)
        output_path = Path(output_dir) / split
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_files = []
        
        for json_file in annotation_path.glob('*.json'):
            try:
                # Load annotation
                annotation = InvoiceAnnotation.load_json(str(json_file))
                
                # Convert to LayoutLMv3 format
                sample = self.annotation_to_sample(annotation)
                
                # Prepare for training
                training_sample = self.prepare_training_sample(sample)
                
                # Save
                output_file = output_path / f"{json_file.stem}.pt"
                torch.save(training_sample, output_file)
                output_files.append(str(output_file))
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Save label mapping
        label_map_file = output_path / 'label_map.json'
        with open(label_map_file, 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': {str(k): v for k, v in self.id2label.items()}
            }, f, indent=2)
        
        return output_files


class DatasetBuilder:
    """Build train/val/test splits"""
    
    def __init__(self, converter: LayoutLMv3Converter):
        """Initialize dataset builder"""
        self.converter = converter
    
    def build_dataset(self,
                     annotation_dir: str,
                     output_dir: str,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     seed: int = 42):
        """
        Build train/val/test datasets
        
        Args:
            annotation_dir: Directory with JSON annotations
            output_dir: Output directory
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed
        """
        import random
        
        random.seed(seed)
        
        # Get all annotation files
        annotation_path = Path(annotation_dir)
        all_files = list(annotation_path.glob('*.json'))
        random.shuffle(all_files)
        
        # Calculate splits
        total = len(all_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_files = all_files[:train_size]
        val_files = all_files[train_size:train_size + val_size]
        test_files = all_files[train_size + val_size:]
        
        print(f"Dataset splits:")
        print(f"  Train: {len(train_files)}")
        print(f"  Val: {len(val_files)}")
        print(f"  Test: {len(test_files)}")
        
        # Process each split
        for split_name, split_files in [
            ('train', train_files),
            ('val', val_files),
            ('test', test_files)
        ]:
            if not split_files:
                continue
            
            split_output = Path(output_dir) / split_name
            split_output.mkdir(parents=True, exist_ok=True)
            
            print(f"\nProcessing {split_name} split...")
            for idx, json_file in enumerate(split_files, 1):
                if idx % 10 == 0:
                    print(f"  {idx}/{len(split_files)}")
                
                try:
                    annotation = InvoiceAnnotation.load_json(str(json_file))
                    sample = self.converter.annotation_to_sample(annotation)
                    training_sample = self.converter.prepare_training_sample(sample)
                    
                    output_file = split_output / f"{json_file.stem}.pt"
                    torch.save(training_sample, output_file)
                    
                except Exception as e:
                    print(f"Error: {json_file.name}: {e}")
            
            # Save label mapping
            label_map_file = split_output / 'label_map.json'
            with open(label_map_file, 'w') as f:
                json.dump({
                    'label2id': self.converter.label2id,
                    'id2label': {str(k): v for k, v in self.converter.id2label.items()}
                }, f, indent=2)
        
        print("\nDataset conversion complete!")


if __name__ == '__main__':
    # Example usage
    converter = LayoutLMv3Converter()
    
    builder = DatasetBuilder(converter)
    builder.build_dataset(
        annotation_dir='data/annotations',
        output_dir='data/layoutlmv3',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
