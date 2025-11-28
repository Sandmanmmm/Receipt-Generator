#!/usr/bin/env python3
"""
Test 6: Validate Model Forward Pass (Dry-Run)

CRITICAL: Run this before training to catch model configuration issues.

Validates:
- Model loads successfully
- All heads produce correct shapes
- CRF Viterbi decoding works
- No dimension mismatches
- Loss computes without NaNs
- Gradient flow is healthy

If this test fails, DO NOT proceed with training.
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_label_schema(schema_path: Path) -> Tuple[List[str], int]:
    """Load label list from schema file."""
    try:
        import yaml
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            label_list = schema.get('label_list', [])
            num_labels = len(label_list)
            return label_list, num_labels
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        sys.exit(1)


def create_dummy_batch(batch_size: int, seq_length: int, num_labels: int) -> Dict[str, torch.Tensor]:
    """
    Create a dummy batch matching LayoutLMv3 input format.
    
    Returns:
        Dictionary with tensors matching LayoutLMv3 expectations
    """
    # Random input_ids (vocabulary indices)
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))
    
    # Attention mask (1 for real tokens, 0 for padding)
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    # Simulate some padding in last 10% of sequence
    padding_start = int(seq_length * 0.9)
    attention_mask[:, padding_start:] = 0
    
    # Bounding boxes normalized to 0-1000
    bbox = torch.randint(0, 1000, (batch_size, seq_length, 4))
    # Ensure x0 < x1 and y0 < y1
    bbox[:, :, 2] = torch.clamp(bbox[:, :, 0] + torch.randint(10, 100, (batch_size, seq_length)), max=1000)
    bbox[:, :, 3] = torch.clamp(bbox[:, :, 1] + torch.randint(10, 50, (batch_size, seq_length)), max=1000)
    
    # Labels (NER tags)
    labels = torch.randint(0, num_labels, (batch_size, seq_length))
    # Set padding positions to -100 (ignored in loss)
    labels[:, padding_start:] = -100
    
    # Pixel values (for image input) - LayoutLMv3 expects (batch, 3, 224, 224)
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'bbox': bbox,
        'labels': labels,
        'pixel_values': pixel_values
    }


def validate_model_forward(model, batch: Dict[str, torch.Tensor], 
                          num_labels: int, use_crf: bool = False) -> Dict:
    """
    Run forward pass and validate outputs.
    
    Returns:
        Dictionary with validation results
    """
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Forward pass
        with torch.no_grad():
            outputs = model(**batch)
        
        result['stats']['forward_pass'] = 'success'
        
        # Check 1: Loss exists and is valid
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
            result['stats']['loss_value'] = float(loss.item())
            
            if torch.isnan(loss):
                result['errors'].append("Loss is NaN")
            elif torch.isinf(loss):
                result['errors'].append("Loss is Inf")
            else:
                result['stats']['loss_valid'] = True
        else:
            result['warnings'].append("Model output has no loss attribute")
        
        # Check 2: Logits shape
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            expected_shape = (batch['input_ids'].shape[0], batch['input_ids'].shape[1], num_labels)
            actual_shape = tuple(logits.shape)
            
            result['stats']['logits_shape'] = str(actual_shape)
            result['stats']['expected_shape'] = str(expected_shape)
            
            if actual_shape != expected_shape:
                result['errors'].append(
                    f"Logits shape mismatch: got {actual_shape}, expected {expected_shape}"
                )
            else:
                result['stats']['logits_shape_valid'] = True
            
            # Check logits for NaN/Inf
            if torch.isnan(logits).any():
                result['errors'].append("Logits contain NaN values")
            if torch.isinf(logits).any():
                result['errors'].append("Logits contain Inf values")
        else:
            result['errors'].append("Model output has no logits attribute")
        
        # Check 3: Hidden states (if available)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            result['stats']['num_hidden_layers'] = len(outputs.hidden_states)
            result['stats']['hidden_dim'] = outputs.hidden_states[-1].shape[-1]
        
        # Check 4: Attention weights (if available)
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            result['stats']['num_attention_layers'] = len(outputs.attentions)
        
        # Check 5: CRF decoding (if enabled)
        if use_crf and hasattr(model, 'crf'):
            try:
                # Test Viterbi decoding
                emissions = outputs.logits
                mask = batch['attention_mask'].bool()
                
                # Decode
                decoded = model.crf.decode(emissions, mask=mask)
                
                result['stats']['crf_decode'] = 'success'
                result['stats']['num_decoded_sequences'] = len(decoded)
                
                # Validate decoded sequences
                for i, seq in enumerate(decoded):
                    if not isinstance(seq, list):
                        result['errors'].append(f"CRF decoded sequence {i} is not a list")
                    elif not all(isinstance(tag, int) for tag in seq):
                        result['errors'].append(f"CRF decoded sequence {i} contains non-integer tags")
                    elif any(tag < 0 or tag >= num_labels for tag in seq):
                        result['errors'].append(f"CRF decoded sequence {i} has out-of-range tags")
            
            except Exception as e:
                result['errors'].append(f"CRF decoding failed: {e}")
        
        # Success if no errors
        if not result['errors']:
            result['success'] = True
    
    except RuntimeError as e:
        result['errors'].append(f"Runtime error during forward pass: {e}")
    except Exception as e:
        result['errors'].append(f"Forward pass exception: {e}")
        import traceback
        result['errors'].append(traceback.format_exc())
    
    return result


def validate_gradient_flow(model, batch: Dict[str, torch.Tensor]) -> Dict:
    """
    Validate that gradients flow properly through the model.
    
    Returns:
        Dictionary with gradient validation results
    """
    result = {
        'success': False,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        # Enable gradients
        model.train()
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        total_params = 0
        params_with_grad = 0
        grad_norms = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    
                    # Check for NaN/Inf gradients
                    if torch.isnan(param.grad).any():
                        result['warnings'].append(f"NaN gradient in {name}")
                    if torch.isinf(param.grad).any():
                        result['warnings'].append(f"Inf gradient in {name}")
        
        result['stats']['total_trainable_params'] = total_params
        result['stats']['params_with_gradients'] = params_with_grad
        
        if grad_norms:
            result['stats']['mean_grad_norm'] = sum(grad_norms) / len(grad_norms)
            result['stats']['max_grad_norm'] = max(grad_norms)
            result['stats']['min_grad_norm'] = min(grad_norms)
        
        # Validate gradient flow
        if params_with_grad == 0:
            result['errors'].append("No gradients computed")
        elif params_with_grad < total_params * 0.5:
            result['warnings'].append(
                f"Only {params_with_grad}/{total_params} parameters have gradients"
            )
        else:
            result['stats']['gradient_flow'] = 'healthy'
        
        # Clean up gradients
        model.zero_grad()
        
        # Success if no errors
        if not result['errors']:
            result['success'] = True
    
    except Exception as e:
        result['errors'].append(f"Gradient validation exception: {e}")
        import traceback
        result['errors'].append(traceback.format_exc())
    finally:
        model.eval()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate model forward pass before training'
    )
    parser.add_argument(
        '--schema', 
        type=Path, 
        default='config/labels_retail.yaml',
        help='Path to label schema file'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='microsoft/layoutlmv3-base',
        help='HuggingFace model name or path'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=128,
        help='Sequence length for testing'
    )
    parser.add_argument(
        '--use-crf',
        action='store_true',
        help='Test with CRF layer (if model has one)'
    )
    parser.add_argument(
        '--test-gradients',
        action='store_true',
        help='Test gradient flow (slower)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST 6: MODEL FORWARD PASS VALIDATION (DRY-RUN)")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Use CRF: {args.use_crf}")
    print(f"Test gradients: {args.test_gradients}")
    print()
    
    # Load label schema
    if not args.schema.exists():
        print(f"[FAIL] Schema file not found: {args.schema}")
        return 1
    
    print("Loading label schema...")
    label_list, num_labels = load_label_schema(args.schema)
    print(f"[OK] Loaded {num_labels} labels")
    print()
    
    # Check PyTorch and CUDA
    print("Checking environment...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')
    print()
    
    # Load model
    print(f"Loading model: {args.model_name}")
    try:
        from transformers import LayoutLMv3ForTokenClassification
        
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            output_hidden_states=True,
            output_attentions=True
        )
        model.to(device)
        model.eval()
        
        print(f"[OK] Model loaded successfully")
        print(f"     Model type: {type(model).__name__}")
        print(f"     Num labels: {model.num_labels}")
        print(f"     Device: {device}")
        print()
        
    except ImportError:
        print("[FAIL] transformers library not installed")
        print("       Run: pip install transformers")
        return 1
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create dummy batch
    print("Creating dummy batch...")
    try:
        batch = create_dummy_batch(args.batch_size, args.seq_length, num_labels)
        
        # Move to device
        for key in batch:
            batch[key] = batch[key].to(device)
        
        print(f"[OK] Batch created")
        print(f"     Batch size: {batch['input_ids'].shape[0]}")
        print(f"     Sequence length: {batch['input_ids'].shape[1]}")
        print(f"     Input IDs shape: {batch['input_ids'].shape}")
        print(f"     Bbox shape: {batch['bbox'].shape}")
        print(f"     Labels shape: {batch['labels'].shape}")
        print(f"     Pixel values shape: {batch['pixel_values'].shape}")
        print()
        
    except Exception as e:
        print(f"[FAIL] Batch creation failed: {e}")
        return 1
    
    # Run forward pass validation
    print("=" * 80)
    print("RUNNING FORWARD PASS VALIDATION")
    print("=" * 80)
    print()
    
    result = validate_model_forward(model, batch, num_labels, args.use_crf)
    
    # Print results
    print("Forward Pass Results:")
    print("-" * 80)
    
    if result['stats']:
        print("Statistics:")
        for key, value in sorted(result['stats'].items()):
            print(f"  {key}: {value}")
        print()
    
    if result['errors']:
        print("Errors:")
        for error in result['errors']:
            print(f"  ✗ {error}")
        print()
    
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")
        print()
    
    if not result['errors']:
        print("[PASS] Forward pass successful")
    else:
        print("[FAIL] Forward pass failed")
    print()
    
    # Run gradient validation (optional)
    gradient_result = None
    if args.test_gradients:
        print("=" * 80)
        print("RUNNING GRADIENT FLOW VALIDATION")
        print("=" * 80)
        print()
        
        gradient_result = validate_gradient_flow(model, batch)
        
        print("Gradient Flow Results:")
        print("-" * 80)
        
        if gradient_result['stats']:
            print("Statistics:")
            for key, value in sorted(gradient_result['stats'].items()):
                print(f"  {key}: {value}")
            print()
        
        if gradient_result['errors']:
            print("Errors:")
            for error in gradient_result['errors']:
                print(f"  ✗ {error}")
            print()
        
        if gradient_result['warnings']:
            print("Warnings:")
            for warning in gradient_result['warnings']:
                print(f"  ⚠ {warning}")
            print()
        
        if not gradient_result['errors']:
            print("[PASS] Gradient flow healthy")
        else:
            print("[FAIL] Gradient flow issues detected")
        print()
    
    # Final verdict
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    critical_errors = []
    
    # Check forward pass
    if not result['success']:
        critical_errors.append("Forward pass failed")
    
    # Check loss
    if 'loss_valid' not in result['stats']:
        critical_errors.append("Loss validation failed")
    
    # Check logits shape
    if 'logits_shape_valid' not in result['stats']:
        critical_errors.append("Logits shape validation failed")
    
    # Check gradients (if tested)
    if gradient_result and not gradient_result['success']:
        critical_errors.append("Gradient flow validation failed")
    
    print(f"Model: {args.model_name}")
    print(f"Num labels: {num_labels}")
    print(f"Device: {device}")
    print(f"Forward pass: {'✓' if result['success'] else '✗'}")
    if gradient_result:
        print(f"Gradient flow: {'✓' if gradient_result['success'] else '✗'}")
    print()
    
    if critical_errors:
        print("[FAIL] TEST 6 FAILED - DO NOT PROCEED WITH TRAINING")
        for error in critical_errors:
            print(f"  ✗ {error}")
        print()
        print("Fix these issues before training:")
        print("  1. Check model configuration matches label schema")
        print("  2. Verify num_labels is correct")
        print("  3. Check input tensor shapes")
        print("  4. Review model head configuration")
        return 1
    else:
        print("[PASS] TEST 6 PASSED - Model ready for training!")
        print()
        print("All checks passed:")
        print("  ✓ Model loads successfully")
        print("  ✓ Forward pass completes without errors")
        print("  ✓ Loss computes correctly (no NaN/Inf)")
        print("  ✓ Output shapes are correct")
        if gradient_result:
            print("  ✓ Gradients flow properly")
        print()
        print("You can now proceed with training.")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
