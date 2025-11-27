"""
Label Mapper - Map extracted text to BIO labels
"""
import re
import yaml
from typing import Dict, List, Optional, Pattern
from pathlib import Path
from .annotation_schema import InvoiceAnnotation, BoundingBox


class LabelMapper:
    """Maps bounding boxes to BIO entity labels"""
    
    def __init__(self, labels_config: Optional[str] = None):
        """
        Initialize label mapper
        
        Args:
            labels_config: Path to labels configuration (uses default rules if None)
        """
        if labels_config and Path(labels_config).exists():
            self.load_config(labels_config)
        else:
            self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default labeling rules"""
        self.entity_patterns = {
            'INVOICE_NUMBER': [
                r'INV[-\s]?\d+',
                r'Invoice\s*[#:]\s*\d+',
                r'#\s*\d{4,}'
            ],
            'PURCHASE_ORDER_NUMBER': [
                r'PO[-\s]?\d+',
                r'Purchase\s*Order\s*[#:]\s*\d+',
                r'Order\s*[#:]\s*\d+'
            ],
            'INVOICE_DATE': [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}\s+[A-Za-z]+\s+\d{4}'
            ],
            'DUE_DATE': [
                r'Due:?\s*\d{4}-\d{2}-\d{2}',
                r'Due\s+Date:?\s*\d{2}/\d{2}/\d{4}'
            ],
            'TOTAL_AMOUNT': [
                r'Total:?\s*[$£€¥]\s*[\d,]+\.?\d*',
                r'Grand\s+Total:?\s*[$£€¥]',
                r'Amount\s+Due:?\s*[$£€¥]'
            ],
            'TAX_AMOUNT': [
                r'Tax:?\s*[$£€¥]\s*[\d,]+\.?\d*',
                r'VAT:?\s*[$£€¥]'
            ],
            'SUBTOTAL': [
                r'Subtotal:?\s*[$£€¥]\s*[\d,]+\.?\d*',
                r'Sub\s+Total:?\s*[$£€¥]'
            ],
            'SUPPLIER_NAME': [
                # Usually at top of document, detected by position
            ],
            'SUPPLIER_VAT': [
                r'VAT\s+No\.?:?\s*[A-Z]{2}\d{9,12}',
                r'Tax\s+ID:?\s*\d{9,11}'
            ],
            'SUPPLIER_PHONE': [
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\(\d{3}\)\s*\d{3}-\d{4}'
            ],
            'SUPPLIER_EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            'CURRENCY': [
                r'\b(USD|GBP|EUR|JPY|CNY|CAD|AUD)\b',
                r'[$£€¥]'
            ],
        }
        
        # Compile patterns
        self.compiled_patterns = {}
        for entity, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def load_config(self, config_path: str):
        """Load labeling rules from YAML config"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        self.entity_patterns = config.get('entity_patterns', {})
        self.compiled_patterns = {}
        for entity, patterns in self.entity_patterns.items():
            self.compiled_patterns[entity] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def map_labels(self, annotation: InvoiceAnnotation, use_bio: bool = True) -> InvoiceAnnotation:
        """
        Map BIO labels to bounding boxes
        
        Args:
            annotation: InvoiceAnnotation to label
            use_bio: Whether to use BIO tagging (B-/I- prefixes)
            
        Returns:
            InvoiceAnnotation with labeled boxes
        """
        # Pattern-based labeling
        for box in annotation.boxes:
            text = box.text.strip()
            
            # Check patterns for each entity type
            for entity_type, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(text):
                        if use_bio:
                            box.label = f"B-{entity_type}"
                        else:
                            box.label = entity_type
                        break
                if box.label:
                    break
        
        # Position-based heuristics
        self._apply_position_heuristics(annotation, use_bio)
        
        # Multi-token entity handling (for BIO tagging)
        if use_bio:
            self._handle_multi_token_entities(annotation)
        
        # Default to 'O' for unlabeled
        for box in annotation.boxes:
            if not box.label:
                box.label = 'O'
        
        return annotation
    
    def _apply_position_heuristics(self, annotation: InvoiceAnnotation, use_bio: bool):
        """Apply position-based labeling heuristics"""
        # Top 15% of document likely contains supplier name
        top_threshold = annotation.image_height * 0.15
        
        for box in annotation.boxes:
            if not box.label and box.y < top_threshold:
                # Check if it's the first unlabeled box at top
                earlier_supplier = any(
                    b.label and 'SUPPLIER_NAME' in b.label 
                    for b in annotation.boxes 
                    if b.y < box.y
                )
                if not earlier_supplier:
                    box.label = 'B-SUPPLIER_NAME' if use_bio else 'SUPPLIER_NAME'
    
    def _handle_multi_token_entities(self, annotation: InvoiceAnnotation):
        """
        Handle multi-token entities by assigning I- tags to continuation tokens
        
        For example: "Acme Corp Inc" -> [B-SUPPLIER_NAME, I-SUPPLIER_NAME, I-SUPPLIER_NAME]
        """
        boxes = annotation.boxes
        
        for i in range(1, len(boxes)):
            curr_box = boxes[i]
            prev_box = boxes[i-1]
            
            # If current box is close to previous and previous has a B- label
            if prev_box.label and prev_box.label.startswith('B-'):
                entity_type = prev_box.label[2:]  # Remove B- prefix
                
                # Check proximity (horizontal and vertical)
                horizontal_distance = curr_box.x - prev_box.x2
                vertical_overlap = self._calc_vertical_overlap(prev_box, curr_box)
                
                # If close enough, tag as continuation
                if horizontal_distance < 50 and vertical_overlap > 0.5:
                    if not curr_box.label or curr_box.label == 'O':
                        curr_box.label = f"I-{entity_type}"
    
    def _calc_vertical_overlap(self, box1: BoundingBox, box2: BoundingBox) -> float:
        """Calculate vertical overlap ratio between two boxes"""
        y1_min, y1_max = box1.y, box1.y2
        y2_min, y2_max = box2.y, box2.y2
        
        overlap_start = max(y1_min, y2_min)
        overlap_end = min(y1_max, y2_max)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_height = overlap_end - overlap_start
        min_height = min(box1.height, box2.height)
        
        return overlap_height / min_height if min_height > 0 else 0.0


__all__ = ['LabelMapper']
