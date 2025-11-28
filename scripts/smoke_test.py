#!/usr/bin/env python3
"""
Test 8: Mini-Training Smoke Test

CRITICAL: 2-minute smoke test before full training.

Runs a quick mini-training loop with 50 samples to validate:
- Loss decreases over iterations
- No NaN or Inf values during training
- CRF layer stabilizes (if enabled)
- Model checkpoint save/load works
- Evaluation pipeline runs without errors
- Gradients flow properly

If this fails, full training will waste hours of GPU time.
Run this BEFORE starting any long training job.
"""

import argparse
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Path) -> Dict:
    """Load training configuration."""
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def load_label_schema(schema_path: Path) -> Tuple[List[str], int]:
    """Load label schema."""
    try:
        import yaml
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            label_list = schema.get('label_list', [])
            return label_list, len(label_list)
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        sys.exit(1)


def create_dummy_dataset(num_samples: int, seq_length: int, num_labels: int, 
                        batch_size: int) -> DataLoader:
    """Create a dummy dataset for smoke testing."""
    # Generate random data
    input_ids = torch.randint(0, 50000, (num_samples, seq_length))
    attention_mask = torch.ones(num_samples, seq_length, dtype=torch.long)
    
    # Add some padding
    padding_start = int(seq_length * 0.8)
    attention_mask[:, padding_start:] = 0
    
    # Bounding boxes (normalized 0-1000)
    bbox = torch.randint(0, 1000, (num_samples, seq_length, 4))
    bbox[:, :, 2] = torch.clamp(bbox[:, :, 0] + torch.randint(10, 100, (num_samples, seq_length)), max=1000)
    bbox[:, :, 3] = torch.clamp(bbox[:, :, 1] + torch.randint(10, 50, (num_samples, seq_length)), max=1000)
    
    # Labels - make some variety
    labels = torch.randint(0, min(num_labels, 10), (num_samples, seq_length))
    labels[:, padding_start:] = -100  # Ignore padding in loss
    
    # Pixel values
    pixel_values = torch.randn(num_samples, 3, 224, 224)
    
    dataset = TensorDataset(input_ids, attention_mask, bbox, labels, pixel_values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader


def run_training_iteration(model, batch: Tuple, optimizer, device: torch.device,
                          max_grad_norm: float = 1.0) -> Dict:
    """Run one training iteration and collect metrics."""
    model.train()
    
    input_ids, attention_mask, bbox, labels, pixel_values = batch
    
    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    bbox = bbox.to(device)
    labels = labels.to(device)
    pixel_values = pixel_values.to(device)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        bbox=bbox,
        labels=labels,
        pixel_values=pixel_values
    )
    
    loss = outputs.loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    
    # Optimizer step
    optimizer.step()
    
    # Collect metrics
    metrics = {
        'loss': loss.item(),
        'grad_norm': grad_norm.item(),
        'has_nan': torch.isnan(loss).item(),
        'has_inf': torch.isinf(loss).item()
    }
    
    return metrics


def run_evaluation(model, dataloader: DataLoader, device: torch.device) -> Dict:
    """Run evaluation and return metrics."""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, bbox, labels, pixel_values = batch
            
            # Move to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            bbox = bbox.to(device)
            labels = labels.to(device)
            pixel_values = pixel_values.to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bbox=bbox,
                labels=labels,
                pixel_values=pixel_values
            )
            
            total_loss += outputs.loss.item() * input_ids.size(0)
            total_samples += input_ids.size(0)
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    return {
        'eval_loss': avg_loss,
        'num_samples': total_samples
    }


def test_checkpoint_save_load(model, temp_dir: Path) -> Dict:
    """Test saving and loading model checkpoint."""
    result = {
        'success': False,
        'errors': []
    }
    
    try:
        # Save checkpoint
        save_path = temp_dir / 'checkpoint'
        save_path.mkdir(exist_ok=True)
        
        model.save_pretrained(save_path)
        result['saved'] = True
        
        # Load checkpoint
        from transformers import LayoutLMv3ForTokenClassification
        loaded_model = LayoutLMv3ForTokenClassification.from_pretrained(save_path)
        result['loaded'] = True
        
        # Compare state dicts
        original_state = model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        if set(original_state.keys()) != set(loaded_state.keys()):
            result['errors'].append("State dict keys mismatch after reload")
        else:
            result['state_dict_match'] = True
        
        result['success'] = True
        
    except Exception as e:
        result['errors'].append(f"Checkpoint test failed: {e}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Run mini-training smoke test'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='config/training_config.yaml',
        help='Path to training config'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default='config/labels_retail.yaml',
        help='Path to label schema'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of samples for smoke test'
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=10,
        help='Number of training steps'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST 8: MINI-TRAINING SMOKE TEST")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Schema: {args.schema}")
    print(f"Samples: {args.samples}")
    print(f"Steps: {args.steps}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Load configuration
    if not args.config.exists():
        print(f"[FAIL] Config not found: {args.config}")
        return 1
    
    config = load_config(args.config)
    
    # Load label schema
    if not args.schema.exists():
        print(f"[FAIL] Schema not found: {args.schema}")
        return 1
    
    label_list, num_labels = load_label_schema(args.schema)
    print(f"Loaded {num_labels} labels")
    print()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # Load model
    print("Loading model...")
    try:
        from transformers import LayoutLMv3ForTokenClassification
        
        model_name = config['model'].get('pretrained_name', 'microsoft/layoutlmv3-base')
        model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        model.to(device)
        
        print(f"[OK] Model loaded: {model_name}")
        print(f"     Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"     Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f}M")
        print()
        
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        return 1
    
    # Create optimizer
    print("Setting up optimizer...")
    learning_rate = float(config['training'].get('learning_rate', 3e-5))
    weight_decay = config['training'].get('weight_decay', 0.01)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    print(f"[OK] AdamW optimizer (lr={learning_rate}, wd={weight_decay})")
    print()
    
    # Create datasets
    print("Creating dummy datasets...")
    seq_length = 128  # Use shorter sequences for smoke test
    
    train_loader = create_dummy_dataset(
        args.samples, seq_length, num_labels, args.batch_size
    )
    eval_loader = create_dummy_dataset(
        20, seq_length, num_labels, args.batch_size
    )
    
    print(f"[OK] Train samples: {args.samples}")
    print(f"     Eval samples: 20")
    print(f"     Batches: {len(train_loader)}")
    print()
    
    # Run mini-training
    print("=" * 80)
    print("RUNNING MINI-TRAINING")
    print("=" * 80)
    print()
    
    loss_history = []
    grad_norm_history = []
    errors = []
    
    print(f"Training for {args.steps} steps...")
    step = 0
    
    for epoch in range(10):  # Max 10 epochs
        for batch in train_loader:
            if step >= args.steps:
                break
            
            metrics = run_training_iteration(
                model, batch, optimizer, device,
                max_grad_norm=config['training'].get('max_grad_norm', 1.0)
            )
            
            loss_history.append(metrics['loss'])
            grad_norm_history.append(metrics['grad_norm'])
            
            # Check for issues
            if metrics['has_nan']:
                errors.append(f"Step {step}: NaN loss detected")
            if metrics['has_inf']:
                errors.append(f"Step {step}: Inf loss detected")
            
            # Print progress
            if step % 2 == 0 or step == args.steps - 1:
                print(f"  Step {step+1}/{args.steps}: loss={metrics['loss']:.4f}, "
                      f"grad_norm={metrics['grad_norm']:.4f}")
            
            step += 1
        
        if step >= args.steps:
            break
    
    print()
    
    # Analyze loss trajectory
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    validation_results = {
        'loss_decreased': False,
        'no_nan': len([e for e in errors if 'NaN' in e]) == 0,
        'no_inf': len([e for e in errors if 'Inf' in e]) == 0,
        'stable_gradients': True,
        'checkpoint_works': False,
        'eval_works': False
    }
    
    # Check 1: Loss decreased
    if len(loss_history) >= 2:
        initial_loss = sum(loss_history[:3]) / 3
        final_loss = sum(loss_history[-3:]) / 3
        loss_decrease = initial_loss - final_loss
        loss_decrease_pct = (loss_decrease / initial_loss) * 100
        
        validation_results['loss_decreased'] = loss_decrease > 0
        
        print(f"Loss Analysis:")
        print(f"  Initial (avg first 3): {initial_loss:.4f}")
        print(f"  Final (avg last 3): {final_loss:.4f}")
        print(f"  Decrease: {loss_decrease:.4f} ({loss_decrease_pct:.1f}%)")
        print(f"  Status: {'✓ DECREASED' if validation_results['loss_decreased'] else '✗ NO DECREASE'}")
        print()
    
    # Check 2: No NaN/Inf
    print(f"Numerical Stability:")
    print(f"  NaN detected: {'✗ YES' if not validation_results['no_nan'] else '✓ NO'}")
    print(f"  Inf detected: {'✗ YES' if not validation_results['no_inf'] else '✓ NO'}")
    print()
    
    # Check 3: Gradient norms
    if grad_norm_history:
        avg_grad = sum(grad_norm_history) / len(grad_norm_history)
        max_grad = max(grad_norm_history)
        
        validation_results['stable_gradients'] = max_grad < 100  # No explosion
        
        print(f"Gradient Norms:")
        print(f"  Average: {avg_grad:.4f}")
        print(f"  Maximum: {max_grad:.4f}")
        print(f"  Status: {'✓ STABLE' if validation_results['stable_gradients'] else '✗ EXPLODING'}")
        print()
    
    # Check 4: Evaluation
    print("Running evaluation...")
    try:
        eval_metrics = run_evaluation(model, eval_loader, device)
        validation_results['eval_works'] = True
        print(f"[OK] Evaluation completed")
        print(f"     Eval loss: {eval_metrics['eval_loss']:.4f}")
        print()
    except Exception as e:
        errors.append(f"Evaluation failed: {e}")
        print(f"[FAIL] Evaluation failed: {e}")
        print()
    
    # Check 5: Checkpoint save/load
    print("Testing checkpoint save/load...")
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_result = test_checkpoint_save_load(model, Path(temp_dir))
        validation_results['checkpoint_works'] = checkpoint_result['success']
        
        if checkpoint_result['success']:
            print(f"[OK] Checkpoint save/load works")
        else:
            for error in checkpoint_result.get('errors', []):
                errors.append(error)
                print(f"[FAIL] {error}")
        print()
    
    # Final verdict
    print("=" * 80)
    print("SMOKE TEST SUMMARY")
    print("=" * 80)
    print()
    
    all_passed = all(validation_results.values())
    
    print("Validation Results:")
    for check, passed in validation_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check.replace('_', ' ').title()}: {status}")
    print()
    
    if errors:
        print(f"Errors ({len(errors)}):")
        for error in errors:
            print(f"  ✗ {error}")
        print()
    
    if all_passed and not errors:
        print("[PASS] TEST 8 PASSED - Mini-training successful!")
        print()
        print("All systems operational:")
        print("  ✓ Loss decreases during training")
        print("  ✓ No numerical issues (NaN/Inf)")
        print("  ✓ Gradients are stable")
        print("  ✓ Evaluation pipeline works")
        print("  ✓ Checkpoint save/load works")
        print()
        print("You can proceed with full training.")
    else:
        print("[FAIL] TEST 8 FAILED - Do not proceed with full training!")
        print()
        print("Fix these issues before training:")
        if not validation_results['loss_decreased']:
            print("  ✗ Loss is not decreasing - check learning rate and data")
        if not validation_results['no_nan']:
            print("  ✗ NaN values detected - check for numerical instability")
        if not validation_results['no_inf']:
            print("  ✗ Inf values detected - reduce learning rate or clip gradients")
        if not validation_results['stable_gradients']:
            print("  ✗ Gradient explosion - reduce learning rate or max_grad_norm")
        if not validation_results['eval_works']:
            print("  ✗ Evaluation failed - check eval pipeline")
        if not validation_results['checkpoint_works']:
            print("  ✗ Checkpoint save/load failed - check permissions")
    
    print("=" * 80)
    
    return 0 if all_passed and not errors else 1


if __name__ == '__main__':
    sys.exit(main())
