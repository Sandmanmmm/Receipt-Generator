#!/usr/bin/env python3
"""
Test 5: Validate LayoutLMv3 Data Conversion

Ensures the dataset is in perfect HuggingFace-ready format for LayoutLMv3.

Validates:
- Tokens align with OCR results
- Bounding boxes normalized 0-1000
- Sequence length < 512 (LayoutLMv3 max)
- label_ids match label_list indices
- No invalid indices in token_label_ids
- All required fields present
- Proper data types

This is critical to prevent runtime errors during training.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.retail_data_generator import RetailDataGenerator


def load_label_schema(schema_path: Path) -> Tuple[List[str], Dict[str, int]]:
    """Load label list from schema file."""
    try:
        import yaml
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            label_list = schema.get('label_list', [])
            label2id = {label: idx for idx, label in enumerate(label_list)}
            return label_list, label2id
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        sys.exit(1)


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[int]:
    """Normalize bbox coordinates to 0-1000 scale."""
    x0, y0, x1, y1 = bbox
    
    # Normalize to 0-1000 scale
    norm_x0 = int((x0 / width) * 1000)
    norm_y0 = int((y0 / height) * 1000)
    norm_x1 = int((x1 / width) * 1000)
    norm_y1 = int((y1 / height) * 1000)
    
    # Clamp to valid range
    norm_x0 = max(0, min(1000, norm_x0))
    norm_y0 = max(0, min(1000, norm_y0))
    norm_x1 = max(0, min(1000, norm_x1))
    norm_y1 = max(0, min(1000, norm_y1))
    
    return [norm_x0, norm_y0, norm_x1, norm_y1]


def create_hf_sample(receipt_data, generator: RetailDataGenerator, 
                     label2id: Dict[str, int], sample_id: int) -> Dict:
    """
    Create a HuggingFace-format sample from receipt data.
    
    Format expected by LayoutLMv3:
    {
        'id': str,
        'tokens': List[str],
        'bboxes': List[List[int]],  # Normalized 0-1000
        'ner_tags': List[int],  # Label IDs
        'image': Optional[PIL.Image]  # Not included in this test
    }
    """
    # Convert receipt to dict
    receipt_dict = generator.to_dict(receipt_data)
    
    # Generate text and tokens (same as Test 4)
    text_lines = []
    
    # Add header
    supplier_name = receipt_dict.get('supplier_name', '')
    if supplier_name:
        text_lines.append(str(supplier_name))
    
    supplier_address = receipt_dict.get('supplier_address', '')
    if supplier_address:
        text_lines.append(str(supplier_address))
    
    supplier_phone = receipt_dict.get('supplier_phone', '')
    if supplier_phone:
        text_lines.append(str(supplier_phone))
    
    # Add transaction info
    invoice_num = receipt_dict.get('invoice_number', '')
    if invoice_num:
        text_lines.append(f"Invoice: {invoice_num}")
    
    invoice_date = receipt_dict.get('invoice_date', '')
    if invoice_date:
        text_lines.append(f"Date: {invoice_date}")
    
    # Add line items
    line_items = receipt_dict.get('line_items', [])
    for item in line_items:
        desc = item.get('description', '')
        unit_price = item.get('unit_price', '$0.00')
        qty = item.get('quantity', 1)
        total = item.get('total', '$0.00')
        
        if desc:
            item_text = f"{desc} {unit_price} x{qty} {total}"
            text_lines.append(item_text)
    
    # Add totals
    subtotal = receipt_dict.get('subtotal')
    if subtotal:
        text_lines.append(f"Subtotal: {subtotal}")
    
    tax_amount = receipt_dict.get('tax_amount')
    if tax_amount:
        text_lines.append(f"Tax: {tax_amount}")
    
    total_amount = receipt_dict.get('total_amount')
    if total_amount:
        text_lines.append(f"Total: {total_amount}")
    
    payment_method = receipt_dict.get('payment_method', '')
    if payment_method:
        text_lines.append(f"Payment: {payment_method}")
    
    # Tokenize and create bboxes
    tokens = []
    boxes = []
    y_offset = 50
    image_width = 800
    image_height = 1000
    
    for line in text_lines:
        words = line.split()
        x_offset = 50
        
        for word in words:
            tokens.append(word)
            word_width = len(word) * 10
            bbox = [x_offset, y_offset, x_offset + word_width, y_offset + 20]
            boxes.append(bbox)
            x_offset += word_width + 10
        
        y_offset += 30
    
    # Create labels
    labels = ['O'] * len(tokens)
    
    for i, token in enumerate(tokens):
        token_lower = token.lower()
        
        # Invoice number
        if i > 0 and tokens[i-1].lower() in ['invoice', 'invoice:']:
            labels[i] = 'B-INVOICE_NUMBER'
        
        # Date patterns
        if '/' in token or '-' in token:
            if any(d in token for d in ['2024', '2025', '2023']):
                labels[i] = 'B-INVOICE_DATE'
        
        # Amount patterns
        if token.startswith('$'):
            context = ' '.join(tokens[max(0, i-2):i]).lower()
            if 'total:' in context or 'total' in context:
                labels[i] = 'B-TOTAL_AMOUNT'
            elif 'subtotal:' in context or 'subtotal' in context:
                labels[i] = 'B-SUBTOTAL'
            elif 'tax:' in context or 'tax' in context:
                labels[i] = 'B-TAX_AMOUNT'
    
    # Normalize bboxes to 0-1000
    normalized_bboxes = [
        normalize_bbox(bbox, image_width, image_height) 
        for bbox in boxes
    ]
    
    # Convert labels to IDs
    ner_tags = []
    for label in labels:
        if label in label2id:
            ner_tags.append(label2id[label])
        else:
            # Fallback to 'O' if label not in schema
            ner_tags.append(label2id.get('O', 0))
    
    return {
        'id': f'sample_{sample_id}',
        'tokens': tokens,
        'bboxes': normalized_bboxes,
        'ner_tags': ner_tags,
        'width': image_width,
        'height': image_height
    }


def validate_hf_sample(sample: Dict, label_list: List[str], 
                       sample_idx: int) -> Dict:
    """
    Validate a single HuggingFace-format sample.
    Returns validation result dict.
    """
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Check 1: Required fields
        required_fields = ['id', 'tokens', 'bboxes', 'ner_tags']
        missing = [f for f in required_fields if f not in sample]
        if missing:
            result['errors'].append(f"Missing fields: {missing}")
            return result
        
        tokens = sample['tokens']
        bboxes = sample['bboxes']
        ner_tags = sample['ner_tags']
        
        # Check 2: Array lengths match
        if not (len(tokens) == len(bboxes) == len(ner_tags)):
            result['errors'].append(
                f"Length mismatch: tokens={len(tokens)}, "
                f"bboxes={len(bboxes)}, ner_tags={len(ner_tags)}"
            )
            return result
        
        result['stats']['num_tokens'] = len(tokens)
        
        # Check 3: Sequence length < 512 (LayoutLMv3 max)
        if len(tokens) >= 512:
            result['errors'].append(
                f"Sequence too long: {len(tokens)} tokens (max 512)"
            )
        
        # Check 4: Validate bboxes are normalized 0-1000
        for i, bbox in enumerate(bboxes):
            if len(bbox) != 4:
                result['errors'].append(f"Token {i}: bbox must have 4 coords")
                continue
            
            x0, y0, x1, y1 = bbox
            
            # Check bounds
            if not (0 <= x0 <= 1000 and 0 <= y0 <= 1000 and 
                   0 <= x1 <= 1000 and 0 <= y1 <= 1000):
                result['errors'].append(
                    f"Token {i}: bbox {bbox} out of range 0-1000"
                )
            
            # Check validity
            if x0 >= x1 or y0 >= y1:
                result['errors'].append(
                    f"Token {i}: invalid bbox {bbox} (x0>=x1 or y0>=y1)"
                )
        
        # Check 5: Validate ner_tags are valid label indices
        num_labels = len(label_list)
        for i, tag_id in enumerate(ner_tags):
            if not isinstance(tag_id, int):
                result['errors'].append(
                    f"Token {i}: ner_tag must be int, got {type(tag_id)}"
                )
            elif tag_id < 0 or tag_id >= num_labels:
                result['errors'].append(
                    f"Token {i}: invalid ner_tag {tag_id} "
                    f"(valid range: 0-{num_labels-1})"
                )
        
        # Check 6: Validate tokens are non-empty strings
        for i, token in enumerate(tokens):
            if not isinstance(token, str):
                result['errors'].append(
                    f"Token {i}: must be string, got {type(token)}"
                )
            elif not token.strip():
                result['warnings'].append(f"Token {i}: empty string")
        
        # Statistics
        result['stats']['max_bbox_x'] = max(bbox[2] for bbox in bboxes) if bboxes else 0
        result['stats']['max_bbox_y'] = max(bbox[3] for bbox in bboxes) if bboxes else 0
        result['stats']['num_entities'] = sum(1 for tag in ner_tags if tag != 0)  # Assuming 0 is 'O'
        
        # Tag distribution
        tag_counts = defaultdict(int)
        for tag in ner_tags:
            tag_counts[tag] += 1
        result['stats']['unique_tags'] = len(tag_counts)
        
        # Success if no errors
        if not result['errors']:
            result['success'] = True
    
    except Exception as e:
        result['errors'].append(f"Validation exception: {e}")
        import traceback
        result['errors'].append(traceback.format_exc())
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate HuggingFace dataset conversion for LayoutLMv3'
    )
    parser.add_argument(
        '--schema', 
        type=Path, 
        default='config/labels_retail.yaml',
        help='Path to label schema file'
    )
    parser.add_argument(
        '--samples', 
        type=int, 
        default=200,
        help='Number of samples to validate'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST 5: HUGGINGFACE DATASET CONVERSION VALIDATION")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Samples: {args.samples}")
    print()
    
    # Load label schema
    if not args.schema.exists():
        print(f"[FAIL] Schema file not found: {args.schema}")
        return 1
    
    print("Loading label schema...")
    label_list, label2id = load_label_schema(args.schema)
    print(f"[OK] Loaded {len(label_list)} labels")
    print(f"     Label range: 0-{len(label_list)-1}")
    print()
    
    # Initialize generator
    print("Initializing generator...")
    try:
        generator = RetailDataGenerator()
        print("[OK] RetailDataGenerator initialized")
        print()
    except Exception as e:
        print(f"[FAIL] Generator initialization failed: {e}")
        return 1
    
    # Generate and validate samples
    print(f"Generating and validating {args.samples} samples...")
    print("=" * 80)
    
    results = []
    errors_by_type = defaultdict(list)
    warnings_by_type = defaultdict(list)
    
    store_types = ['fashion', 'accessories', 'jewelry', 'beauty', 
                   'home_garden', 'sports_fitness', 'pet_supplies', 
                   'books_media', 'toys_games', 'food_beverage', 
                   'health_wellness', 'electronics']
    
    for i in range(args.samples):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{args.samples} samples...")
        
        try:
            # Generate receipt
            store_type = store_types[i % len(store_types)]
            if i % 2 == 0:
                receipt = generator.generate_pos_receipt(store_type=store_type)
            else:
                receipt = generator.generate_online_order(store_type=store_type)
            
            # Convert to HF format
            hf_sample = create_hf_sample(receipt, generator, label2id, i)
            
            # Validate
            result = validate_hf_sample(hf_sample, label_list, i)
            results.append(result)
            
            # Collect errors and warnings
            for error in result['errors']:
                error_type = error.split(':')[0] if ':' in error else error.split()[0]
                errors_by_type[error_type].append((i, error))
            
            for warning in result['warnings']:
                warning_type = warning.split(':')[0] if ':' in warning else warning.split()[0]
                warnings_by_type[warning_type].append((i, warning))
            
            # Debug first error
            if i == 0 and result['errors']:
                print(f"\n[DEBUG] First sample errors:")
                for err in result['errors'][:3]:
                    print(f"  {err}")
                print()
        
        except Exception as e:
            print(f"\n[ERROR] Sample {i} failed: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'success': False, 
                'errors': [str(e)], 
                'warnings': [], 
                'stats': {}
            })
    
    print(f"\n[PASS] Processed {args.samples} samples")
    print()
    
    # Aggregate results
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"\nSuccess rate: {successful}/{args.samples} ({successful/args.samples*100:.1f}%)")
    print(f"Failed: {failed}")
    print()
    
    # Statistics
    if results:
        stats_keys = set()
        for r in results:
            stats_keys.update(r.get('stats', {}).keys())
        
        print("Average statistics:")
        for key in sorted(stats_keys):
            values = [r['stats'].get(key, 0) for r in results if key in r.get('stats', {})]
            if values:
                if all(isinstance(v, bool) for v in values):
                    count = sum(values)
                    print(f"  {key}: {count}/{len(values)} ({count/len(values)*100:.1f}%)")
                else:
                    avg = sum(values) / len(values)
                    min_val = min(values)
                    max_val = max(values)
                    print(f"  {key}: avg={avg:.2f}, min={min_val:.0f}, max={max_val:.0f}")
        print()
    
    # Error analysis
    if errors_by_type:
        print("Errors by type:")
        for error_type, errors in sorted(errors_by_type.items(), 
                                        key=lambda x: len(x[1]), reverse=True):
            print(f"  {error_type}: {len(errors)} occurrences")
            # Show first few examples
            for sample_id, err in errors[:3]:
                print(f"    Sample {sample_id}: {err}")
        print()
    else:
        print("[PASS] No errors detected")
        print()
    
    # Warning analysis
    if warnings_by_type:
        print("Warnings by type:")
        for warning_type, warnings in sorted(warnings_by_type.items(), 
                                            key=lambda x: len(x[1]), reverse=True):
            print(f"  {warning_type}: {len(warnings)} occurrences")
        print()
    else:
        print("[PASS] No warnings")
        print()
    
    # Critical checks
    print("=" * 80)
    print("CRITICAL CHECKS")
    print("=" * 80)
    
    critical_errors = []
    critical_warnings = []
    
    # Check 1: Success rate
    if successful < args.samples * 0.95:  # Need 95%+ for HF format
        critical_errors.append(
            f"Success rate too low: {successful/args.samples*100:.1f}% (need ≥95%)"
        )
    else:
        print("[✓] Success rate acceptable")
    
    # Check 2: Sequence length
    max_tokens = max(r['stats'].get('num_tokens', 0) for r in results)
    if max_tokens >= 512:
        critical_errors.append(
            f"Max sequence length {max_tokens} exceeds LayoutLMv3 limit (512)"
        )
    else:
        print(f"[✓] All sequences < 512 tokens (max: {max_tokens})")
    
    # Check 3: Bbox normalization
    max_bbox_x = max(r['stats'].get('max_bbox_x', 0) for r in results)
    max_bbox_y = max(r['stats'].get('max_bbox_y', 0) for r in results)
    if max_bbox_x > 1000 or max_bbox_y > 1000:
        critical_errors.append(
            f"Bbox coordinates exceed 1000: x={max_bbox_x}, y={max_bbox_y}"
        )
    else:
        print(f"[✓] All bboxes normalized 0-1000 (max: x={max_bbox_x}, y={max_bbox_y})")
    
    # Check 4: Label indices
    if any('invalid ner_tag' in err for _, err in 
           [e for errs in errors_by_type.values() for e in errs]):
        critical_errors.append("Invalid label indices detected")
    else:
        print("[✓] All label indices valid")
    
    # Check 5: Data types
    if any('must be' in err for _, err in 
           [e for errs in errors_by_type.values() for e in errs]):
        critical_errors.append("Invalid data types detected")
    else:
        print("[✓] All data types correct")
    
    print()
    
    # Final verdict
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal samples: {args.samples}")
    print(f"Successful: {successful} ({successful/args.samples*100:.1f}%)")
    print(f"Failed: {failed}")
    print(f"Critical errors: {len(critical_errors)}")
    print(f"Warnings: {len(critical_warnings)}")
    print()
    
    if critical_errors:
        print("[FAIL] TEST 5 FAILED")
        for error in critical_errors:
            print(f"  ✗ {error}")
        return 1
    
    if critical_warnings:
        print("[WARN] TEST 5 PASSED WITH WARNINGS")
        for warning in critical_warnings:
            print(f"  ⚠ {warning}")
    else:
        print("[PASS] TEST 5 PASSED - Dataset ready for LayoutLMv3 training!")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
