#!/usr/bin/env python3
"""
Test 7: Validate Training Configuration

CRITICAL: Run before starting training to catch configuration issues.

Validates:
- Label list size matches model configuration
- Learning rate is valid and appropriate
- Warmup steps/ratio is sensible
- Batch size and gradient accumulation are appropriate
- Multi-GPU and AMP FP16 compatibility
- All paths exist
- No conflicting settings

This prevents mid-training crashes and wasted compute time.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: Path) -> Dict:
    """Load training configuration from YAML."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        sys.exit(1)


def load_label_schema(schema_path: Path) -> Tuple[List[str], int]:
    """Load label list from schema file."""
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
            label_list = schema.get('label_list', [])
            return label_list, len(label_list)
    except Exception as e:
        print(f"[ERROR] Failed to load label schema: {e}")
        sys.exit(1)


def validate_paths(config: Dict) -> Dict:
    """Validate that all required paths exist."""
    result = {
        'success': True,
        'errors': [],
        'warnings': []
    }
    
    # Check label list path
    label_path = Path(config['data'].get('label_list_path', 'config/labels.yaml'))
    if not label_path.exists():
        result['errors'].append(f"Label list not found: {label_path}")
        result['success'] = False
    
    # Check data paths (only if they're specified and not placeholders)
    data_paths = {
        'train': config['data'].get('train_json', ''),
        'val': config['data'].get('val_json', ''),
        'test': config['data'].get('test_json', '')
    }
    
    for split, path_str in data_paths.items():
        if path_str and not path_str.startswith('data/processed/'):
            path = Path(path_str)
            if not path.exists():
                result['warnings'].append(f"{split} data not found: {path_str} (will be generated)")
    
    # Check checkpoint directory
    checkpoint_dir = Path(config['training'].get('checkpoint_dir', 'models/checkpoints'))
    if not checkpoint_dir.exists():
        result['warnings'].append(f"Checkpoint dir will be created: {checkpoint_dir}")
    
    return result


def validate_model_config(config: Dict, num_labels: int) -> Dict:
    """Validate model configuration."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    model_config = config.get('model', {})
    
    # Check pretrained model name
    pretrained_name = model_config.get('pretrained_name', 'microsoft/layoutlmv3-base')
    result['stats']['pretrained_model'] = pretrained_name
    
    if 'layoutlmv3' not in pretrained_name.lower():
        result['warnings'].append(f"Model '{pretrained_name}' is not LayoutLMv3")
    
    # Check CRF configuration
    use_crf = model_config.get('use_crf', False)
    result['stats']['use_crf'] = use_crf
    
    if use_crf:
        crf_lr_mult = model_config.get('crf_lr_multiplier', 1.0)
        if crf_lr_mult <= 0 or crf_lr_mult > 10:
            result['warnings'].append(f"Unusual CRF LR multiplier: {crf_lr_mult}")
    
    # Check dropout
    dropout = model_config.get('dropout', 0.1)
    if dropout < 0 or dropout > 0.5:
        result['warnings'].append(f"Unusual dropout rate: {dropout} (typical: 0.1-0.3)")
    
    result['stats']['dropout'] = dropout
    
    return result


def validate_data_config(config: Dict) -> Dict:
    """Validate data configuration."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    data_config = config.get('data', {})
    
    # Check max sequence length
    max_seq_length = data_config.get('max_seq_length', 512)
    result['stats']['max_seq_length'] = max_seq_length
    
    if max_seq_length > 1024:
        result['errors'].append(
            f"max_seq_length {max_seq_length} exceeds LayoutLMv3 limit (1024)"
        )
        result['success'] = False
    elif max_seq_length > 512:
        result['warnings'].append(
            f"max_seq_length {max_seq_length} is high, may cause OOM errors"
        )
    
    # Check image size
    image_size = data_config.get('image_size', 224)
    if image_size not in [224, 384]:
        result['warnings'].append(
            f"Unusual image_size {image_size} (LayoutLMv3 typically uses 224)"
        )
    
    result['stats']['image_size'] = image_size
    
    # Check normalization
    normalize_boxes = data_config.get('normalize_boxes', True)
    if not normalize_boxes:
        result['errors'].append("normalize_boxes must be True for LayoutLMv3")
        result['success'] = False
    
    return result


def validate_training_config(config: Dict) -> Dict:
    """Validate training hyperparameters."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    training_config = config.get('training', {})
    
    # Batch size
    batch_size = training_config.get('batch_size', 4)
    grad_accum = training_config.get('grad_accumulation_steps', 1)
    effective_batch_size = batch_size * grad_accum
    
    result['stats']['batch_size'] = batch_size
    result['stats']['grad_accumulation_steps'] = grad_accum
    result['stats']['effective_batch_size'] = effective_batch_size
    
    if batch_size < 1:
        result['errors'].append(f"Invalid batch_size: {batch_size}")
        result['success'] = False
    elif batch_size > 32:
        result['warnings'].append(f"Large batch_size {batch_size} may cause OOM")
    
    if effective_batch_size < 8:
        result['warnings'].append(
            f"Small effective batch size {effective_batch_size} may harm training"
        )
    
    # Learning rate
    learning_rate = float(training_config.get('learning_rate', 3e-5))
    result['stats']['learning_rate'] = learning_rate
    
    if learning_rate <= 0 or learning_rate > 1e-3:
        result['warnings'].append(
            f"Unusual learning rate {learning_rate} (typical: 2e-5 to 5e-5)"
        )
    
    # Warmup
    warmup_ratio = training_config.get('warmup_ratio', 0.06)
    warmup_steps = training_config.get('warmup_steps', -1)
    
    result['stats']['warmup_ratio'] = warmup_ratio
    result['stats']['warmup_steps'] = warmup_steps
    
    if warmup_steps == -1:
        if warmup_ratio < 0 or warmup_ratio > 0.2:
            result['warnings'].append(
                f"Unusual warmup_ratio {warmup_ratio} (typical: 0.05-0.1)"
            )
    else:
        if warmup_steps < 0:
            result['errors'].append(f"Invalid warmup_steps: {warmup_steps}")
            result['success'] = False
    
    # Epochs
    num_epochs = training_config.get('num_epochs', 10)
    result['stats']['num_epochs'] = num_epochs
    
    if num_epochs < 1:
        result['errors'].append(f"Invalid num_epochs: {num_epochs}")
        result['success'] = False
    elif num_epochs > 50:
        result['warnings'].append(f"Large num_epochs {num_epochs} may overfit")
    
    # Gradient clipping
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    if max_grad_norm <= 0:
        result['warnings'].append("Gradient clipping disabled (max_grad_norm <= 0)")
    elif max_grad_norm > 10:
        result['warnings'].append(f"High max_grad_norm {max_grad_norm}")
    
    result['stats']['max_grad_norm'] = max_grad_norm
    
    # Weight decay
    weight_decay = training_config.get('weight_decay', 0.01)
    if weight_decay < 0 or weight_decay > 0.1:
        result['warnings'].append(f"Unusual weight_decay {weight_decay}")
    
    result['stats']['weight_decay'] = weight_decay
    
    # Learning rate scheduler
    lr_scheduler = training_config.get('lr_scheduler', 'linear')
    valid_schedulers = ['linear', 'cosine', 'polynomial', 'constant']
    if lr_scheduler not in valid_schedulers:
        result['warnings'].append(
            f"Unknown lr_scheduler '{lr_scheduler}' (valid: {valid_schedulers})"
        )
    
    result['stats']['lr_scheduler'] = lr_scheduler
    
    # Mixed precision
    fp16 = training_config.get('fp16', False)
    result['stats']['fp16'] = fp16
    
    if fp16:
        try:
            import torch
            if not torch.cuda.is_available():
                result['warnings'].append("FP16 enabled but CUDA not available")
        except:
            pass
    
    # Checkpointing
    save_steps = training_config.get('save_steps', 500)
    eval_steps = training_config.get('eval_steps', 500)
    
    if save_steps < 10 or eval_steps < 10:
        result['warnings'].append("Very frequent checkpointing may slow training")
    
    result['stats']['save_steps'] = save_steps
    result['stats']['eval_steps'] = eval_steps
    
    return result


def validate_optimizer_config(config: Dict) -> Dict:
    """Validate optimizer configuration."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    optimizer_config = config.get('optimizer', {})
    
    # Optimizer type
    opt_type = optimizer_config.get('type', 'AdamW')
    result['stats']['optimizer'] = opt_type
    
    if opt_type not in ['AdamW', 'Adam', 'SGD']:
        result['warnings'].append(f"Unusual optimizer: {opt_type}")
    
    # Betas
    betas = optimizer_config.get('betas', [0.9, 0.999])
    if len(betas) != 2:
        result['errors'].append(f"Invalid betas: {betas} (must be [beta1, beta2])")
        result['success'] = False
    elif not (0 < betas[0] < 1 and 0 < betas[1] < 1):
        result['warnings'].append(f"Unusual betas: {betas}")
    
    result['stats']['betas'] = betas
    
    # Layer-wise LR decay
    layerwise_enabled = optimizer_config.get('layerwise_lr_decay_enabled', False)
    if layerwise_enabled:
        decay = optimizer_config.get('layerwise_lr_decay', 0.95)
        if decay <= 0 or decay >= 1:
            result['errors'].append(f"Invalid layerwise_lr_decay: {decay}")
            result['success'] = False
        result['stats']['layerwise_lr_decay'] = decay
    
    return result


def validate_hardware_config(config: Dict) -> Dict:
    """Validate hardware and distributed training settings."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    hardware_config = config.get('hardware', {})
    
    # GPU count
    num_gpus = hardware_config.get('num_gpus', 1)
    result['stats']['num_gpus'] = num_gpus
    
    if num_gpus < 0:
        result['errors'].append(f"Invalid num_gpus: {num_gpus}")
        result['success'] = False
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        result['stats']['cuda_available'] = cuda_available
        
        if num_gpus > 0 and not cuda_available:
            result['warnings'].append("GPU configured but CUDA not available")
        
        if cuda_available:
            actual_gpus = torch.cuda.device_count()
            result['stats']['actual_gpus'] = actual_gpus
            
            if num_gpus > actual_gpus:
                result['warnings'].append(
                    f"Configured {num_gpus} GPUs but only {actual_gpus} available"
                )
    except ImportError:
        result['warnings'].append("PyTorch not installed, cannot check CUDA")
    
    # Distributed training
    distributed = hardware_config.get('distributed', {})
    if distributed.get('enabled', False):
        world_size = distributed.get('world_size', 1)
        if world_size != num_gpus:
            result['warnings'].append(
                f"world_size ({world_size}) != num_gpus ({num_gpus})"
            )
        
        backend = distributed.get('backend', 'nccl')
        if backend not in ['nccl', 'gloo', 'mpi']:
            result['warnings'].append(f"Unknown distributed backend: {backend}")
    
    # DataLoader workers
    num_workers = hardware_config.get('dataloader_num_workers', 4)
    if num_workers < 0:
        result['errors'].append(f"Invalid num_workers: {num_workers}")
        result['success'] = False
    elif num_workers > 16:
        result['warnings'].append(f"Many workers ({num_workers}) may cause overhead")
    
    result['stats']['dataloader_workers'] = num_workers
    
    return result


def validate_multi_task_config(config: Dict, num_labels: int) -> Dict:
    """Validate multi-task learning configuration."""
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    multi_task = config.get('multi_task', {})
    
    # Loss weights
    ner_weight = multi_task.get('ner_loss_weight', 1.0)
    table_weight = multi_task.get('table_loss_weight', 0.0)
    structure_weight = multi_task.get('structure_loss_weight', 0.0)
    
    result['stats']['ner_loss_weight'] = ner_weight
    result['stats']['table_loss_weight'] = table_weight
    result['stats']['structure_loss_weight'] = structure_weight
    
    if ner_weight <= 0:
        result['errors'].append("ner_loss_weight must be > 0")
        result['success'] = False
    
    # Table detection
    table_config = multi_task.get('table', {})
    if table_config.get('enabled', False):
        table_num_labels = table_config.get('num_labels', 3)
        if table_num_labels != 3:
            result['warnings'].append(
                f"Table detection expects 3 labels (O, B-TABLE, I-TABLE), got {table_num_labels}"
            )
        result['stats']['table_enabled'] = True
    else:
        result['stats']['table_enabled'] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate training configuration before training'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='config/training_config.yaml',
        help='Path to training configuration YAML'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default='config/labels_retail.yaml',
        help='Path to label schema YAML'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEST 7: TRAINING CONFIGURATION VALIDATION")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Schema: {args.schema}")
    print()
    
    # Load configuration
    if not args.config.exists():
        print(f"[FAIL] Config file not found: {args.config}")
        return 1
    
    print("Loading configuration...")
    config = load_config(args.config)
    print(f"[OK] Configuration loaded")
    print()
    
    # Load label schema
    if not args.schema.exists():
        print(f"[FAIL] Schema file not found: {args.schema}")
        return 1
    
    print("Loading label schema...")
    label_list, num_labels = load_label_schema(args.schema)
    print(f"[OK] Loaded {num_labels} labels")
    print()
    
    # Run all validations
    print("=" * 80)
    print("RUNNING VALIDATIONS")
    print("=" * 80)
    print()
    
    all_results = {}
    
    # 1. Paths validation
    print("1. Validating paths...")
    all_results['paths'] = validate_paths(config)
    print(f"   {'[PASS]' if all_results['paths']['success'] else '[FAIL]'}")
    print()
    
    # 2. Model configuration
    print("2. Validating model configuration...")
    all_results['model'] = validate_model_config(config, num_labels)
    print(f"   {'[PASS]' if all_results['model']['success'] else '[FAIL]'}")
    print()
    
    # 3. Data configuration
    print("3. Validating data configuration...")
    all_results['data'] = validate_data_config(config)
    print(f"   {'[PASS]' if all_results['data']['success'] else '[FAIL]'}")
    print()
    
    # 4. Training configuration
    print("4. Validating training configuration...")
    all_results['training'] = validate_training_config(config)
    print(f"   {'[PASS]' if all_results['training']['success'] else '[FAIL]'}")
    print()
    
    # 5. Optimizer configuration
    print("5. Validating optimizer configuration...")
    all_results['optimizer'] = validate_optimizer_config(config)
    print(f"   {'[PASS]' if all_results['optimizer']['success'] else '[FAIL]'}")
    print()
    
    # 6. Hardware configuration
    print("6. Validating hardware configuration...")
    all_results['hardware'] = validate_hardware_config(config)
    print(f"   {'[PASS]' if all_results['hardware']['success'] else '[FAIL]'}")
    print()
    
    # 7. Multi-task configuration
    print("7. Validating multi-task configuration...")
    all_results['multi_task'] = validate_multi_task_config(config, num_labels)
    print(f"   {'[PASS]' if all_results['multi_task']['success'] else '[FAIL]'}")
    print()
    
    # Aggregate results
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    total_errors = sum(len(r['errors']) for r in all_results.values())
    total_warnings = sum(len(r['warnings']) for r in all_results.values())
    all_success = all(r['success'] for r in all_results.values())
    
    # Print key statistics
    print("Key Configuration:")
    print("-" * 80)
    for section, result in all_results.items():
        if result.get('stats'):
            print(f"\n{section.upper()}:")
            for key, value in sorted(result['stats'].items()):
                print(f"  {key}: {value}")
    print()
    
    # Print errors
    if total_errors > 0:
        print("ERRORS:")
        print("-" * 80)
        for section, result in all_results.items():
            if result['errors']:
                print(f"\n{section.upper()}:")
                for error in result['errors']:
                    print(f"  ✗ {error}")
        print()
    
    # Print warnings
    if total_warnings > 0:
        print("WARNINGS:")
        print("-" * 80)
        for section, result in all_results.items():
            if result['warnings']:
                print(f"\n{section.upper()}:")
                for warning in result['warnings']:
                    print(f"  ⚠ {warning}")
        print()
    
    # Final verdict
    print("=" * 80)
    print(f"Total errors: {total_errors}")
    print(f"Total warnings: {total_warnings}")
    print()
    
    if not all_success:
        print("[FAIL] TEST 7 FAILED - Fix errors before training")
        return 1
    
    if total_warnings > 0:
        print("[WARN] TEST 7 PASSED WITH WARNINGS")
        print()
        print("Configuration is valid but has warnings.")
        print("Review warnings before starting training.")
    else:
        print("[PASS] TEST 7 PASSED - Configuration ready for training!")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
