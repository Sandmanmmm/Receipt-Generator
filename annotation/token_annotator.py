"""
Token Annotator - Maps structured receipt data to token-level BIO labels
Converts receipt entity data to HuggingFace-ready format for LayoutLMv3
"""
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import re
from difflib import SequenceMatcher


class TokenAnnotator:
    """
    Maps structured receipt metadata to token-level BIO tags
    
    Takes receipt data dict + OCR tokens/bboxes, returns HF-ready format:
    {
        "id": str,
        "tokens": List[str],
        "ner_tags": List[int],
        "bboxes": List[List[int]],
        "image_path": str
    }
    """
    
    def __init__(self, label_schema: Dict[str, Any]):
        """
        Initialize token annotator with label schema
        
        Args:
            label_schema: Label schema dict with 'labels' list
        """
        self.label_list = label_schema.get('label_list', label_schema.get('labels', []))
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Entity field mappings (receipt field -> label prefix)
        # Aligned with labels_retail.yaml schema
        self.entity_mappings = {
            # Merchant info (schema uses SUPPLIER_*)
            'supplier_name': 'SUPPLIER_NAME',
            'supplier_address': 'SUPPLIER_ADDRESS',
            'supplier_phone': 'SUPPLIER_PHONE',
            'supplier_email': 'SUPPLIER_EMAIL',
            
            # Document identifiers
            'invoice_number': 'INVOICE_NUMBER',
            'invoice_date': 'INVOICE_DATE',
            'order_date': 'ORDER_DATE',
            
            # Customer info (schema uses BUYER_*)
            'buyer_name': 'BUYER_NAME',
            'buyer_address': 'BUYER_ADDRESS',
            'buyer_phone': 'BUYER_PHONE',
            'buyer_email': 'BUYER_EMAIL',
            'account_number': 'ACCOUNT_NUMBER',
            
            # Financial totals
            'currency': 'CURRENCY',
            'subtotal': 'SUBTOTAL',
            'tax_amount': 'TAX_AMOUNT',
            'tax_rate': 'TAX_RATE',
            'total_amount': 'TOTAL_AMOUNT',
            'discount': 'DISCOUNT',
            'total_discount': 'DISCOUNT',
            
            # Payment info
            'payment_method': 'PAYMENT_METHOD',
            'payment_terms': 'PAYMENT_TERMS',
            'card_type': 'PAYMENT_METHOD',
            
            # Retail-specific
            'register_number': 'REGISTER_NUMBER',
            'cashier_id': 'CASHIER_ID',
            'tracking_number': 'TRACKING_NUMBER',
        }
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for matching"""
        return re.sub(r'\s+', ' ', str(text).strip().lower())
    
    def fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """
        Fuzzy string matching using sequence matcher
        
        Args:
            text1: First string
            text2: Second string
            threshold: Similarity threshold (0-1)
            
        Returns:
            True if strings are similar enough
        """
        ratio = SequenceMatcher(None, 
                               self.normalize_text(text1), 
                               self.normalize_text(text2)).ratio()
        return ratio >= threshold
    
    def find_entity_span(self, entity_text: str, tokens: List[str], 
                        start_idx: int = 0) -> Optional[Tuple[int, int]]:
        """
        Find where entity text appears in token list
        
        Args:
            entity_text: Text to find
            tokens: List of tokens
            start_idx: Start searching from this index
            
        Returns:
            Tuple of (start_idx, end_idx) or None if not found
        """
        entity_normalized = self.normalize_text(entity_text)
        entity_words = entity_normalized.split()
        
        if len(entity_words) == 0:
            return None
        
        # Try exact match first
        for i in range(start_idx, len(tokens) - len(entity_words) + 1):
            token_span = ' '.join(tokens[i:i+len(entity_words)])
            if self.normalize_text(token_span) == entity_normalized:
                return (i, i + len(entity_words))
        
        # Try fuzzy match for single-word entities
        if len(entity_words) == 1:
            for i in range(start_idx, len(tokens)):
                if self.fuzzy_match(tokens[i], entity_text, threshold=0.85):
                    return (i, i + 1)
        
        # Try partial matches for multi-word entities
        for i in range(start_idx, len(tokens) - len(entity_words) + 1):
            matches = 0
            for j, word in enumerate(entity_words):
                if self.fuzzy_match(tokens[i+j], word, threshold=0.75):
                    matches += 1
            
            # If most words match, consider it a match
            if matches >= len(entity_words) * 0.7:
                return (i, i + len(entity_words))
        
        return None
    
    def annotate_tokens(self, receipt_data: Dict, tokens: List[str], 
                       bboxes: List[List[int]], image_path: str,
                       image_width: int = 800, image_height: int = 1200) -> Dict:
        """
        Annotate tokens with BIO labels from receipt data
        
        Args:
            receipt_data: Structured receipt data dict
            tokens: List of token strings from OCR
            bboxes: List of bounding boxes [x_min, y_min, x_max, y_max]
            image_path: Path to receipt image
            image_width: Image width for bbox normalization
            image_height: Image height for bbox normalization
            
        Returns:
            HuggingFace-ready dict with tokens, ner_tags, bboxes, etc.
        """
        # Initialize labels as 'O' (outside)
        labels = ['O'] * len(tokens)
        
        # Track which tokens have been labeled
        labeled_indices = set()
        
        # Process top-level fields
        for field_name, entity_type in self.entity_mappings.items():
            if field_name not in receipt_data:
                continue
            
            field_value = receipt_data[field_name]
            if not field_value or str(field_value).strip() == '':
                continue
            
            # Convert to string
            entity_text = str(field_value)
            
            # Find where this entity appears in tokens
            span = self.find_entity_span(entity_text, tokens)
            
            if span:
                start_idx, end_idx = span
                
                # Only label if not already labeled
                if not any(i in labeled_indices for i in range(start_idx, end_idx)):
                    # Check if entity_type exists in schema (with B- prefix)
                    b_label = f'B-{entity_type}'
                    i_label = f'I-{entity_type}'
                    
                    if b_label in self.label2id:
                        labels[start_idx] = b_label
                        labeled_indices.add(start_idx)
                        
                        # Label continuation tokens
                        for i in range(start_idx + 1, end_idx):
                            if i_label in self.label2id:
                                labels[i] = i_label
                                labeled_indices.add(i)
        
        # Process line items
        # Schema uses: ITEM_DESCRIPTION, ITEM_QTY, ITEM_UNIT_COST, ITEM_TOTAL_COST, ITEM_TAX, ITEM_DISCOUNT, ITEM_SKU
        if 'line_items' in receipt_data:
            for item in receipt_data['line_items']:
                # Item description (schema: ITEM_DESCRIPTION)
                if 'description' in item and item['description']:
                    span = self.find_entity_span(str(item['description']), tokens)
                    if span:
                        start_idx, end_idx = span
                        if not any(i in labeled_indices for i in range(start_idx, end_idx)):
                            if 'B-ITEM_DESCRIPTION' in self.label2id:
                                labels[start_idx] = 'B-ITEM_DESCRIPTION'
                                labeled_indices.add(start_idx)
                                for i in range(start_idx + 1, end_idx):
                                    if 'I-ITEM_DESCRIPTION' in self.label2id:
                                        labels[i] = 'I-ITEM_DESCRIPTION'
                                        labeled_indices.add(i)
                
                # Item quantity (schema: ITEM_QTY)
                if 'quantity' in item and item['quantity']:
                    qty_str = str(item['quantity'])
                    # Look for quantity patterns like "x2", "2", "qty: 2"
                    for i, token in enumerate(tokens):
                        if i in labeled_indices:
                            continue
                        if qty_str in token or token.strip('x') == qty_str:
                            if 'B-ITEM_QTY' in self.label2id:
                                labels[i] = 'B-ITEM_QTY'
                                labeled_indices.add(i)
                            break
                
                # Item unit price (schema: ITEM_UNIT_COST)
                if 'unit_price' in item and item['unit_price']:
                    price_str = str(item['unit_price'])
                    span = self.find_entity_span(price_str, tokens)
                    if span:
                        start_idx, end_idx = span
                        if not any(i in labeled_indices for i in range(start_idx, end_idx)):
                            if 'B-ITEM_UNIT_COST' in self.label2id:
                                labels[start_idx] = 'B-ITEM_UNIT_COST'
                                labeled_indices.add(start_idx)
                                for i in range(start_idx + 1, end_idx):
                                    if 'I-ITEM_UNIT_COST' in self.label2id:
                                        labels[i] = 'I-ITEM_UNIT_COST'
                                        labeled_indices.add(i)
                
                # Item total (schema: ITEM_TOTAL_COST)
                if 'total' in item and item['total']:
                    total_str = str(item['total'])
                    span = self.find_entity_span(total_str, tokens)
                    if span:
                        start_idx, end_idx = span
                        if not any(i in labeled_indices for i in range(start_idx, end_idx)):
                            if 'B-ITEM_TOTAL_COST' in self.label2id:
                                labels[start_idx] = 'B-ITEM_TOTAL_COST'
                                labeled_indices.add(start_idx)
                                for i in range(start_idx + 1, end_idx):
                                    if 'I-ITEM_TOTAL_COST' in self.label2id:
                                        labels[i] = 'I-ITEM_TOTAL_COST'
                                        labeled_indices.add(i)
        
        # Convert labels to IDs
        ner_tags = []
        for label in labels:
            if label in self.label2id:
                ner_tags.append(self.label2id[label])
            else:
                # Fallback to 'O' if label not in schema
                ner_tags.append(self.label2id.get('O', 0))
        
        # Normalize bboxes to [0, 1000] scale
        normalized_bboxes = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            norm_bbox = [
                int((x_min / image_width) * 1000),
                int((y_min / image_height) * 1000),
                int((x_max / image_width) * 1000),
                int((y_max / image_height) * 1000)
            ]
            # Clamp to [0, 1000]
            norm_bbox = [max(0, min(1000, coord)) for coord in norm_bbox]
            normalized_bboxes.append(norm_bbox)
        
        # Create sample ID from receipt data
        sample_id = receipt_data.get('id', 'unknown')
        if 'invoice_number' in receipt_data:
            sample_id = str(receipt_data['invoice_number']).replace('/', '_').replace('\\', '_')
        
        # Return HuggingFace-ready format
        return {
            'id': sample_id,
            'tokens': tokens,
            'ner_tags': ner_tags,
            'bboxes': normalized_bboxes,
            'image_path': str(image_path),
            'image_width': image_width,
            'image_height': image_height
        }
    
    def get_label_statistics(self, ner_tags: List[int]) -> Dict[str, int]:
        """
        Get statistics about label distribution
        
        Args:
            ner_tags: List of label IDs
            
        Returns:
            Dict mapping label names to counts
        """
        stats = {}
        for tag_id in ner_tags:
            label = self.id2label.get(tag_id, 'UNKNOWN')
            stats[label] = stats.get(label, 0) + 1
        return stats
    
    def validate_annotation(self, annotation: Dict) -> Tuple[bool, List[str]]:
        """
        Validate annotation format and consistency
        
        Args:
            annotation: Annotation dict from annotate_tokens
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required fields
        required_fields = ['id', 'tokens', 'ner_tags', 'bboxes', 'image_path']
        for field in required_fields:
            if field not in annotation:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        tokens = annotation['tokens']
        ner_tags = annotation['ner_tags']
        bboxes = annotation['bboxes']
        
        # Check array lengths match
        if not (len(tokens) == len(ner_tags) == len(bboxes)):
            errors.append(
                f"Length mismatch: tokens={len(tokens)}, "
                f"ner_tags={len(ner_tags)}, bboxes={len(bboxes)}"
            )
        
        # Check sequence length
        if len(tokens) > 512:
            errors.append(f"Sequence too long: {len(tokens)} tokens (max 512)")
        
        # Check bbox format and normalization
        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4:
                errors.append(f"Token {i}: bbox must have 4 coordinates")
                continue
            
            x0, y0, x1, y1 = bbox
            
            # Check bounds
            if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 
                   0 <= x1 <= 1000 and 0 <= y1 <= 1000):
                errors.append(f"Token {i}: bbox {bbox} out of range [0, 1000]")
            
            # Check validity
            if x0 >= x1 or y0 >= y1:
                errors.append(f"Token {i}: invalid bbox {bbox} (min >= max)")
        
        # Check BIO tag transitions
        prev_entity = None
        for i, tag_id in enumerate(ner_tags):
            label = self.id2label.get(tag_id, 'UNKNOWN')
            
            if label.startswith('I-'):
                entity = label[2:]
                if i == 0:
                    errors.append(f"Token {i}: I-{entity} at start (should be B-{entity})")
                elif prev_entity != entity:
                    errors.append(
                        f"Token {i}: I-{entity} follows "
                        f"{self.id2label.get(ner_tags[i-1], 'UNKNOWN')} (invalid transition)"
                    )
                prev_entity = entity
            elif label.startswith('B-'):
                prev_entity = label[2:]
            else:
                prev_entity = None
        
        return len(errors) == 0, errors


__all__ = ['TokenAnnotator']
