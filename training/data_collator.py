"""
Data Collator - Batch collation for LayoutLMv3 training
"""
import torch
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class LayoutLMv3DataCollator:
    """
    Collate batches for LayoutLMv3 training
    Handles padding and tensor creation
    """
    
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate list of samples into batch
        
        Args:
            features: List of sample dictionaries
            
        Returns:
            Batched dictionary with padded tensors
        """
        batch = {}
        
        # Get max sequence length in batch
        max_length = max(len(f['input_ids']) for f in features)
        
        # Initialize lists
        input_ids = []
        attention_mask = []
        bbox = []
        pixel_values = []
        labels = []
        
        for feature in features:
            seq_length = len(feature['input_ids'])
            padding_length = max_length - seq_length
            
            # Pad input_ids
            padded_input_ids = feature['input_ids'] + [self.pad_token_id] * padding_length
            input_ids.append(padded_input_ids)
            
            # Pad attention_mask
            padded_attention = feature['attention_mask'] + [0] * padding_length
            attention_mask.append(padded_attention)
            
            # Pad bbox
            padded_bbox = feature['bbox'] + [[0, 0, 0, 0]] * padding_length
            bbox.append(padded_bbox)
            
            # Pad labels
            if 'labels' in feature:
                padded_labels = feature['labels'] + [self.label_pad_token_id] * padding_length
                labels.append(padded_labels)
            
            # Pixel values don't need padding (fixed size)
            pixel_values.append(feature['pixel_values'])
        
        # Convert to tensors
        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        batch['bbox'] = torch.tensor(bbox, dtype=torch.long)
        batch['pixel_values'] = torch.tensor(pixel_values, dtype=torch.float)
        
        if labels:
            batch['labels'] = torch.tensor(labels, dtype=torch.long)
        
        return batch


@dataclass
class LayoutLMv3MultiTaskCollator:
    """
    Collate batches for multi-task LayoutLMv3 (NER + Table + Cell)
    """
    
    pad_token_id: int = 0
    label_pad_token_id: int = -100
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate multi-task batch
        
        Args:
            features: List of sample dictionaries
            
        Returns:
            Batched dictionary with padded tensors for all tasks
        """
        batch = {}
        
        # Get max sequence length
        max_length = max(len(f['input_ids']) for f in features)
        
        # Initialize lists
        input_ids = []
        attention_mask = []
        bbox = []
        pixel_values = []
        ner_labels = []
        table_labels = []
        cell_labels = []
        
        for feature in features:
            seq_length = len(feature['input_ids'])
            padding_length = max_length - seq_length
            
            # Pad input_ids
            input_ids.append(
                feature['input_ids'] + [self.pad_token_id] * padding_length
            )
            
            # Pad attention_mask
            attention_mask.append(
                feature['attention_mask'] + [0] * padding_length
            )
            
            # Pad bbox
            bbox.append(
                feature['bbox'] + [[0, 0, 0, 0]] * padding_length
            )
            
            # Pad labels for each task
            if 'ner_labels' in feature:
                ner_labels.append(
                    feature['ner_labels'] + [self.label_pad_token_id] * padding_length
                )
            
            if 'table_labels' in feature:
                table_labels.append(
                    feature['table_labels'] + [self.label_pad_token_id] * padding_length
                )
            
            if 'cell_labels' in feature:
                cell_labels.append(
                    feature['cell_labels'] + [self.label_pad_token_id] * padding_length
                )
            
            # Pixel values
            pixel_values.append(feature['pixel_values'])
        
        # Convert to tensors
        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
        batch['bbox'] = torch.tensor(bbox, dtype=torch.long)
        batch['pixel_values'] = torch.tensor(pixel_values, dtype=torch.float)
        
        if ner_labels:
            batch['ner_labels'] = torch.tensor(ner_labels, dtype=torch.long)
        if table_labels:
            batch['table_labels'] = torch.tensor(table_labels, dtype=torch.long)
        if cell_labels:
            batch['cell_labels'] = torch.tensor(cell_labels, dtype=torch.long)
        
        return batch


__all__ = ['LayoutLMv3DataCollator', 'LayoutLMv3MultiTaskCollator']
