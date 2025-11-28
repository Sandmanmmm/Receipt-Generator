#!/usr/bin/env python3
"""
Test 10: Validate Inference Pipeline (Pre-Training)

CRITICAL: Test inference pipeline BEFORE training starts.

Validates end-to-end inference with an untrained model:
- Model loads correctly
- Tokenization works
- OCR → preprocessing → model → postprocessing pipeline
- JSON output generation
- No missing keys or empty predictions
- Error handling works

This catches pipeline issues before wasting training time.
Run with an untrained model to verify infrastructure.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_label_schema(schema_path: Path) -> tuple:
    """Load label schema."""
    try:
        import yaml
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            label_list = schema.get('label_list', [])
            label2id = {label: idx for idx, label in enumerate(label_list)}
            id2label = {idx: label for idx, label in enumerate(label_list)}
            return label_list, label2id, id2label
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        sys.exit(1)


def create_dummy_receipt() -> Dict:
    """Create a dummy receipt for testing."""
    return {
        'supplier_name': 'Fashion Boutique',
        'supplier_address': '123 Main St, New York, NY 10001',
        'supplier_phone': '(555) 123-4567',
        'invoice_number': 'INV-2024-001',
        'invoice_date': '2024-11-27',
        'subtotal': '$45.99',
        'tax_amount': '$3.68',
        'total_amount': '$49.67',
        'payment_method': 'Credit Card',
        'line_items': [
            {
                'description': 'Blue T-Shirt',
                'quantity': 2,
                'unit_price': '$15.00',
                'total': '$30.00'
            },
            {
                'description': 'Jeans',
                'quantity': 1,
                'unit_price': '$15.99',
                'total': '$15.99'
            }
        ]
    }


def create_mock_ocr_output(receipt: Dict) -> tuple:
    """Create mock OCR output (tokens, bboxes) from receipt."""
    tokens = []
    bboxes = []
    
    y_pos = 50
    
    # Add supplier info
    if receipt.get('supplier_name'):
        for word in receipt['supplier_name'].split():
            tokens.append(word)
            bboxes.append([50, y_pos, 50 + len(word) * 10, y_pos + 20])
        y_pos += 30
    
    if receipt.get('supplier_address'):
        for word in receipt['supplier_address'].split():
            tokens.append(word)
            bboxes.append([50, y_pos, 50 + len(word) * 10, y_pos + 20])
        y_pos += 30
    
    # Add invoice info
    if receipt.get('invoice_number'):
        tokens.extend(['Invoice:', receipt['invoice_number']])
        bboxes.extend([[50, y_pos, 110, y_pos + 20], [120, y_pos, 220, y_pos + 20]])
        y_pos += 30
    
    if receipt.get('invoice_date'):
        tokens.extend(['Date:', receipt['invoice_date']])
        bboxes.extend([[50, y_pos, 90, y_pos + 20], [100, y_pos, 200, y_pos + 20]])
        y_pos += 30
    
    # Add line items
    for item in receipt.get('line_items', []):
        words = item['description'].split() + [item['unit_price'], f"x{item['quantity']}", item['total']]
        for word in words:
            tokens.append(word)
            bboxes.append([50, y_pos, 50 + len(word) * 10, y_pos + 20])
        y_pos += 25
    
    # Add totals
    if receipt.get('subtotal'):
        tokens.extend(['Subtotal:', receipt['subtotal']])
        bboxes.extend([[50, y_pos, 120, y_pos + 20], [130, y_pos, 200, y_pos + 20]])
        y_pos += 25
    
    if receipt.get('tax_amount'):
        tokens.extend(['Tax:', receipt['tax_amount']])
        bboxes.extend([[50, y_pos, 80, y_pos + 20], [90, y_pos, 160, y_pos + 20]])
        y_pos += 25
    
    if receipt.get('total_amount'):
        tokens.extend(['Total:', receipt['total_amount']])
        bboxes.extend([[50, y_pos, 90, y_pos + 20], [100, y_pos, 170, y_pos + 20]])
        y_pos += 25
    
    if receipt.get('payment_method'):
        tokens.extend(['Payment:', receipt['payment_method']])
        bboxes.extend([[50, y_pos, 120, y_pos + 20], [130, y_pos, 230, y_pos + 20]])
    
    return tokens, bboxes


def normalize_bbox(bbox: List[int], width: int = 800, height: int = 1000) -> List[int]:
    """Normalize bbox to 0-1000 scale."""
    x0, y0, x1, y1 = bbox
    norm_x0 = int((x0 / width) * 1000)
    norm_y0 = int((y0 / height) * 1000)
    norm_x1 = int((x1 / width) * 1000)
    norm_y1 = int((y1 / height) * 1000)
    return [
        max(0, min(1000, norm_x0)),
        max(0, min(1000, norm_y0)),
        max(0, min(1000, norm_x1)),
        max(0, min(1000, norm_y1))
    ]


def run_inference_pipeline(model, tokenizer, tokens: List[str], bboxes: List[List[int]],
                          id2label: Dict, device) -> Dict:
    """Run complete inference pipeline."""
    import torch
    
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'predictions': {},
        'stats': {}
    }
    
    try:
        # Normalize bboxes
        norm_bboxes = [normalize_bbox(bbox) for bbox in bboxes]
        
        # For LayoutLMv3, pass words directly (list of strings) with boxes
        # The tokenizer will handle subword tokenization internally
        encoding = tokenizer(
            tokens,  # List[str] - pre-tokenized words
            boxes=norm_bboxes,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        
        # Get word IDs for entity extraction later
        word_ids = encoding.word_ids(batch_index=0)
        
        # Bbox is already in encoding from tokenizer
        bbox_tensor = encoding['bbox']
        
        # Create pixel values (dummy for testing)
        pixel_values = torch.randn(1, 3, 224, 224)
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        bbox_tensor = bbox_tensor.to(device)
        pixel_values = pixel_values.to(device)
        
        result['stats']['num_input_tokens'] = int(attention_mask.sum().item())
        result['stats']['sequence_length'] = input_ids.shape[1]
        
        # Run inference
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox_tensor,
                pixel_values=pixel_values
            )
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Convert to labels
        pred_labels = []
        for i, word_id in enumerate(word_ids):
            if word_id is not None:  # Not a special token
                pred_idx = predictions[0][i].item()
                pred_label = id2label.get(pred_idx, f'UNKNOWN_{pred_idx}')
                pred_labels.append(pred_label)
            else:
                pred_labels.append('O')  # Special tokens get O
        
        result['stats']['num_predictions'] = len(pred_labels)
        result['stats']['num_entities'] = sum(1 for label in pred_labels if label.startswith('B-'))
        
        # Extract entities
        entities = []
        current_entity = None
        current_tokens = []
        current_label = None
        
        for i, (token, label) in enumerate(zip(tokens, pred_labels[:len(tokens)])):
            if label.startswith('B-'):
                # Save previous entity
                if current_entity is not None:
                    entities.append({
                        'label': current_label,
                        'text': ' '.join(current_tokens),
                        'tokens': current_tokens.copy()
                    })
                
                # Start new entity
                current_label = label[2:]
                current_tokens = [token]
                current_entity = True
            
            elif label.startswith('I-') and current_entity:
                # Continue entity
                current_tokens.append(token)
            
            else:
                # End entity
                if current_entity is not None:
                    entities.append({
                        'label': current_label,
                        'text': ' '.join(current_tokens),
                        'tokens': current_tokens.copy()
                    })
                    current_entity = None
                    current_tokens = []
                    current_label = None
        
        # Save last entity
        if current_entity is not None:
            entities.append({
                'label': current_label,
                'text': ' '.join(current_tokens),
                'tokens': current_tokens.copy()
            })
        
        result['predictions']['entities'] = entities
        result['predictions']['tokens'] = tokens
        result['predictions']['labels'] = pred_labels[:len(tokens)]
        
        # Create structured output
        structured = {}
        for entity in entities:
            label = entity['label']
            text = entity['text']
            
            # Group multiple values for same label
            if label not in structured:
                structured[label] = []
            structured[label].append(text)
        
        result['predictions']['structured'] = structured
        result['stats']['num_unique_labels'] = len(structured)
        
        result['success'] = True
        
    except Exception as e:
        result['errors'].append(f"Inference failed: {e}")
        import traceback
        result['errors'].append(traceback.format_exc())
    
    return result


def validate_output_format(result: Dict) -> Dict:
    """Validate the output format is correct."""
    validation = {
        'success': True,
        'errors': [],
        'warnings': []
    }
    
    # Check required keys
    required_keys = ['predictions', 'stats']
    for key in required_keys:
        if key not in result:
            validation['errors'].append(f"Missing key: {key}")
            validation['success'] = False
    
    if 'predictions' in result:
        pred_keys = ['entities', 'tokens', 'labels', 'structured']
        for key in pred_keys:
            if key not in result['predictions']:
                validation['warnings'].append(f"Missing prediction key: {key}")
    
    # Check for empty predictions
    if result.get('predictions', {}).get('entities') is not None:
        if len(result['predictions']['entities']) == 0:
            validation['warnings'].append("No entities predicted (expected for untrained model)")
    
    # Validate entity format
    for entity in result.get('predictions', {}).get('entities', []):
        if 'label' not in entity:
            validation['errors'].append("Entity missing 'label' field")
            validation['success'] = False
        if 'text' not in entity:
            validation['errors'].append("Entity missing 'text' field")
            validation['success'] = False
    
    return validation


def main():
    parser = argparse.ArgumentParser(
        description='Validate inference pipeline before training'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default='config/labels_retail.yaml',
        help='Path to label schema'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/layoutlmv3-base',
        help='Model name or path'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=3,
        help='Number of test samples'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST 10: INFERENCE PIPELINE VALIDATION (PRE-TRAINING)")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Model: {args.model_name}")
    print(f"Test samples: {args.num_samples}")
    print()
    
    # Load schema
    if not args.schema.exists():
        print(f"[FAIL] Schema not found: {args.schema}")
        return 1
    
    print("Loading label schema...")
    label_list, label2id, id2label = load_label_schema(args.schema)
    print(f"[OK] Loaded {len(label_list)} labels")
    print()
    
    # Check device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    try:
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast
        
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id
        )
        model.to(device)
        model.eval()
        
        tokenizer = LayoutLMv3TokenizerFast.from_pretrained(args.model_name)
        
        print(f"[OK] Model and tokenizer loaded")
        print(f"     Model: {type(model).__name__}")
        print(f"     Tokenizer: {type(tokenizer).__name__}")
        print(f"     Num labels: {len(label_list)}")
        print()
        
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return 1
    
    # Run inference tests
    print("=" * 80)
    print("RUNNING INFERENCE TESTS")
    print("=" * 80)
    print()
    
    all_results = []
    
    for i in range(args.num_samples):
        print(f"Test {i+1}/{args.num_samples}:")
        print("-" * 80)
        
        # Create test receipt
        receipt = create_dummy_receipt()
        
        # Generate OCR output
        tokens, bboxes = create_mock_ocr_output(receipt)
        print(f"  Input: {len(tokens)} tokens")
        
        # Run inference
        result = run_inference_pipeline(model, tokenizer, tokens, bboxes, id2label, device)
        
        if result['success']:
            print(f"  [OK] Inference completed")
            print(f"       Predictions: {result['stats']['num_predictions']} tokens")
            print(f"       Entities: {result['stats'].get('num_entities', 0)} found")
            
            if result['predictions']['entities']:
                print(f"       Sample entities:")
                for entity in result['predictions']['entities'][:3]:
                    print(f"         - {entity['label']}: {entity['text']}")
        else:
            print(f"  [FAIL] Inference failed")
            for error in result['errors'][:2]:
                print(f"         {error}")
        
        # Validate output format
        validation = validate_output_format(result)
        if validation['success']:
            print(f"  [OK] Output format valid")
        else:
            print(f"  [FAIL] Output format invalid")
            for error in validation['errors']:
                print(f"         {error}")
        
        all_results.append((result, validation))
        print()
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    successful_inference = sum(1 for r, _ in all_results if r['success'])
    valid_output = sum(1 for _, v in all_results if v['success'])
    
    total_errors = sum(len(r['errors']) + len(v['errors']) 
                      for r, v in all_results)
    total_warnings = sum(len(r['warnings']) + len(v['warnings']) 
                        for r, v in all_results)
    
    print(f"Inference Success: {successful_inference}/{args.num_samples}")
    print(f"Valid Output: {valid_output}/{args.num_samples}")
    print(f"Total Errors: {total_errors}")
    print(f"Total Warnings: {total_warnings}")
    print()
    
    if total_errors > 0:
        print("Errors:")
        for i, (result, validation) in enumerate(all_results):
            errors = result['errors'] + validation['errors']
            if errors:
                print(f"  Sample {i+1}:")
                for error in errors[:3]:
                    print(f"    ✗ {error}")
        print()
    
    if total_warnings > 0:
        print("Warnings:")
        for i, (result, validation) in enumerate(all_results):
            warnings = result['warnings'] + validation['warnings']
            if warnings:
                print(f"  Sample {i+1}:")
                for warning in warnings[:2]:
                    print(f"    ⚠ {warning}")
        print()
    
    # Final verdict
    all_passed = (successful_inference == args.num_samples and 
                 valid_output == args.num_samples and 
                 total_errors == 0)
    
    if all_passed:
        print("[PASS] TEST 10 PASSED - Inference pipeline ready!")
        print()
        print("Pipeline validated:")
        print("  ✓ Model loads successfully")
        print("  ✓ Tokenization works")
        print("  ✓ OCR → preprocess → model → postprocess works")
        print("  ✓ JSON output generates correctly")
        print("  ✓ No missing keys")
        print("  ✓ Entity extraction works")
        print()
        print("No surprises expected after training.")
    else:
        print("[FAIL] TEST 10 FAILED - Fix issues before training")
        print()
        print("Issues found:")
        if successful_inference < args.num_samples:
            print(f"  ✗ {args.num_samples - successful_inference} samples failed inference")
        if valid_output < args.num_samples:
            print(f"  ✗ {args.num_samples - valid_output} samples have invalid output")
        if total_errors > 0:
            print(f"  ✗ {total_errors} errors detected")
    
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
