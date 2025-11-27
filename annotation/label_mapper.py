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
        """Initialize retail-specific labeling rules for POS/e-commerce receipts"""
        self.entity_patterns = {
            # Document metadata
            'INVOICE_NUMBER': [
                r'(?:Receipt|Invoice|Order)\s*[#:No.]+\s*[A-Z0-9-]+',
                r'Receipt\s*[#:]\s*\d+',
                r'Transaction\s*[#:]\s*\d+',
                r'Order\s*[#:]\s*[A-Z0-9-]+'
            ],
            'INVOICE_DATE': [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',
                r'Date:\s*\d{2}/\d{2}/\d{4}'
            ],
            'ORDER_DATE': [
                r'Order\s+Date:\s*\d{2}/\d{2}/\d{4}',
                r'Ordered:\s*\d{2}/\d{2}/\d{4}'
            ],
            
            # Merchant information
            'SUPPLIER_NAME': [
                # Usually at top of document, detected by position
            ],
            'SUPPLIER_ADDRESS': [
                r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)',
                r'[A-Za-z\s]+,\s*[A-Z]{2}\s+\d{5}'
            ],
            'SUPPLIER_PHONE': [
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                r'\(\d{3}\)\s*\d{3}-\d{4}',
                r'Phone:\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            ],
            'SUPPLIER_EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            
            # Financial totals
            'TOTAL_AMOUNT': [
                r'(?:Total|TOTAL|Amount\s+Due)[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Grand\s+Total[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Balance[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'SUBTOTAL': [
                r'Subtotal[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Sub\s+Total[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Item\s+Total[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'TAX_AMOUNT': [
                r'(?:Tax|VAT|GST|Sales\s+Tax)[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Tax\s*\(\d+\.?\d*%\)[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'TAX_RATE': [
                r'\d+\.?\d*%',
                r'Tax:\s*\d+\.?\d*%'
            ],
            'DISCOUNT': [
                r'(?:Discount|Savings|Promo)[:=\s]*-?[$£€¥]\s*[\d,]+\.?\d*',
                r'Total\s+Savings[:=\s]*-?[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'CURRENCY': [
                r'\b(USD|GBP|EUR|JPY|CNY|CAD|AUD)\b',
                r'[$£€¥]'
            ],
            
            # Payment information
            'PAYMENT_METHOD': [
                r'(?:Visa|Mastercard|MasterCard|Amex|American\s+Express|Discover|Debit|Credit)',
                r'(?:Cash|Check|Gift\s+Card|Apple\s+Pay|Google\s+Pay|PayPal)',
                r'Payment\s+Method:\s*[A-Za-z\s]+',
                r'(?:Card|Credit\s+Card|Debit\s+Card)'
            ],
            'PAYMENT_TERMS': [
                r'ending\s+in\s+\d{4}',
                r'xxxx\s*-?\s*\d{4}',
                r'Approval\s+Code:\s*[A-Z0-9]+',
                r'Auth\s*[#:]?\s*[A-Z0-9]+',
                r'Transaction\s+ID:\s*[A-Z0-9-]+'
            ],
            
            # Line items
            'ITEM_DESCRIPTION': [
                # Detected by table structure and position
            ],
            'ITEM_QTY': [
                r'\d+\s*@\s*[$£€¥]',  # "2 @ $5.99"
                r'(?:Qty|Quantity)[:=\s]*\d+',
                r'x\s*\d+',  # "x 3"
            ],
            'ITEM_UNIT_COST': [
                r'@\s*[$£€¥]\s*[\d,]+\.?\d*',  # "@ $5.99"
                r'(?:Unit\s+Price|Price)[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'ITEM_TOTAL_COST': [
                r'[$£€¥]\s*[\d,]+\.?\d*$',  # Amount at end of line
            ],
            'ITEM_SKU': [
                r'(?:UPC|SKU|Item\s*#)[:=\s]*[A-Z0-9-]+',
                r'\b\d{12,13}\b',  # UPC barcode
                r'\b[A-Z0-9]{6,12}\b'  # Generic SKU
            ],
            'ITEM_DISCOUNT': [
                r'(?:Promo|Discount|Sale)[:=\s]*-?[$£€¥]\s*[\d,]+\.?\d*',
                r'-\s*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            
            # Retail identifiers
            'REGISTER_NUMBER': [
                r'Register[:=\s]*\d+',
                r'Reg\s*[#:]\s*\d+',
                r'Terminal[:=\s]*\d+'
            ],
            'CASHIER_ID': [
                r'Cashier[:=\s]*[A-Z0-9]+',
                r'Operator[:=\s]*[A-Z0-9]+',
                r'Clerk[:=\s]*[A-Z0-9]+'
            ],
            'TRACKING_NUMBER': [
                r'Tracking[:=\s]*[A-Z0-9-]+',
                r'(?:UPS|FedEx|USPS)[:=\s]*\d[A-Z0-9\s]+',
                r'Ship\s+Track[:=\s]*[A-Z0-9]+'
            ],
            'ACCOUNT_NUMBER': [
                r'Account\s*[#:]\s*[A-Z0-9-]+',
                r'Member\s*[#:]\s*\d+'
            ],
            
            # Product tracking
            'LOT_NUMBER': [
                r'Lot[:=\s]*[A-Z0-9-]+',
                r'Batch[:=\s]*[A-Z0-9-]+'
            ],
            'SERIAL_NUMBER': [
                r'Serial[:=\s]*[A-Z0-9-]+',
                r'S/N[:=\s]*[A-Z0-9-]+'
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
