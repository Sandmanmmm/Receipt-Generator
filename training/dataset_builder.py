"""
Dataset Builder - Build training datasets from annotations
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm


class DatasetBuilder:
    """Build train/val/test splits from annotated data"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize dataset builder
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def load_annotations(self, annotations_dir: str) -> List[Dict]:
        """
        Load all JSONL annotations
        
        Args:
            annotations_dir: Directory containing .jsonl files
            
        Returns:
            List of annotation dictionaries
        """
        annotations = []
        annotations_path = Path(annotations_dir)
        
        for jsonl_file in annotations_path.glob('*.jsonl'):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        annotations.append(json.loads(line))
        
        return annotations
    
    def split_dataset(self, annotations: List[Dict], 
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split annotations into train/val/test
        
        Args:
            annotations: List of annotation dictionaries
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            
        Returns:
            Tuple of (train, val, test) annotation lists
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        # Shuffle annotations
        shuffled = annotations.copy()
        random.shuffle(shuffled)
        
        total = len(shuffled)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_data = shuffled[:train_size]
        val_data = shuffled[train_size:train_size + val_size]
        test_data = shuffled[train_size + val_size:]
        
        return train_data, val_data, test_data
    
    def save_split(self, annotations: List[Dict], output_dir: str, 
                   split_name: str, copy_images: bool = True):
        """
        Save split to directory
        
        Args:
            annotations: List of annotations
            output_dir: Output directory
            split_name: 'train', 'val', or 'test'
            copy_images: Whether to copy image files
        """
        split_dir = Path(output_dir) / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images subdirectory if copying
        if copy_images:
            images_dir = split_dir / 'images'
            images_dir.mkdir(exist_ok=True)
        
        # Save annotations to JSONL
        annotations_file = split_dir / 'annotations.jsonl'
        with open(annotations_file, 'w', encoding='utf-8') as f:
            for ann in tqdm(annotations, desc=f"Saving {split_name}"):
                # Copy image if needed
                if copy_images:
                    src_image = Path(ann['image_path'])
                    if src_image.exists():
                        dst_image = images_dir / src_image.name
                        shutil.copy2(src_image, dst_image)
                        # Update path in annotation
                        ann['image_path'] = str(dst_image)
                
                f.write(json.dumps(ann) + '\n')
        
        # Save metadata
        metadata = {
            'split': split_name,
            'num_samples': len(annotations),
            'seed': self.seed
        }
        
        metadata_file = split_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {len(annotations)} samples to {split_dir}")
    
    def build_dataset(self, annotations_dir: str, output_dir: str,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1,
                     copy_images: bool = True):
        """
        Complete dataset building pipeline
        
        Args:
            annotations_dir: Directory with annotated JSONL files
            output_dir: Output directory for splits
            train_ratio: Training split ratio
            val_ratio: Validation split ratio
            test_ratio: Test split ratio
            copy_images: Whether to copy images to split directories
        """
        print("=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)
        
        # Load annotations
        print(f"\n[1/3] Loading annotations from {annotations_dir}...")
        annotations = self.load_annotations(annotations_dir)
        print(f"✓ Loaded {len(annotations)} annotations")
        
        # Split dataset
        print(f"\n[2/3] Splitting dataset (train={train_ratio}, val={val_ratio}, test={test_ratio})...")
        train_data, val_data, test_data = self.split_dataset(
            annotations, train_ratio, val_ratio, test_ratio
        )
        print(f"✓ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Save splits
        print(f"\n[3/3] Saving splits to {output_dir}...")
        self.save_split(train_data, output_dir, 'train', copy_images)
        self.save_split(val_data, output_dir, 'val', copy_images)
        self.save_split(test_data, output_dir, 'test', copy_images)
        
        print("\n" + "=" * 60)
        print("DATASET BUILD COMPLETE")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")
    
    def validate_dataset(self, dataset_dir: str) -> Dict[str, any]:
        """
        Validate dataset structure and content
        
        Args:
            dataset_dir: Dataset directory
            
        Returns:
            Validation report dictionary
        """
        dataset_path = Path(dataset_dir)
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check splits exist
        for split in ['train', 'val', 'test']:
            split_dir = dataset_path / split
            if not split_dir.exists():
                report['valid'] = False
                report['errors'].append(f"Missing {split} directory")
                continue
            
            # Check annotations file
            annotations_file = split_dir / 'annotations.jsonl'
            if not annotations_file.exists():
                report['valid'] = False
                report['errors'].append(f"Missing {split}/annotations.jsonl")
                continue
            
            # Count samples
            num_samples = 0
            with open(annotations_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        num_samples += 1
            
            report['statistics'][split] = num_samples
            
            # Check images directory
            images_dir = split_dir / 'images'
            if not images_dir.exists():
                report['warnings'].append(f"{split}/images directory not found")
        
        return report


__all__ = ['DatasetBuilder']
