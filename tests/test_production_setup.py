"""
Quick test script to verify production training setup
Tests model instantiation and forward pass
"""
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_creation():
    """Test creating the multi-head model"""
    print("="*60)
    print("Testing LayoutLMv3MultiHead Model Creation")
    print("="*60)
    
    try:
        from training.layoutlmv3_multihead import create_model
        
        print("\n1. Creating model (this may take a minute)...")
        model = create_model(
            pretrained_name="microsoft/layoutlmv3-base",
            num_ner_labels=73,
            use_crf=True,
            device="cpu"  # Use CPU for testing
        )
        
        print("✓ Model created successfully")
        print(f"  - NER labels: {model.num_ner_labels}")
        print(f"  - Using CRF: {model.use_crf}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with dummy data"""
    print("\n" + "="*60)
    print("Testing Forward Pass")
    print("="*60)
    
    try:
        # Create dummy inputs
        batch_size = 2
        seq_len = 128
        
        print(f"\n2. Creating dummy inputs (batch={batch_size}, seq_len={seq_len})...")
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
        attention_mask = torch.ones(batch_size, seq_len)
        labels = torch.randint(0, 73, (batch_size, seq_len))
        table_labels = torch.randint(0, 3, (batch_size, seq_len))
        attr_labels = torch.rand(batch_size, seq_len, 3)  # Multi-label
        
        print("✓ Dummy inputs created")
        
        print("\n3. Running forward pass...")
        output = model(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            labels=labels,
            table_labels=table_labels,
            attr_labels=attr_labels
        )
        
        print("✓ Forward pass successful")
        print(f"  - NER logits shape: {output.ner_logits.shape}")
        print(f"  - Table logits shape: {output.table_logits.shape}")
        print(f"  - Attr logits shape: {output.attr_logits.shape}")
        print(f"  - Total loss: {output.loss.item():.4f}")
        print(f"  - NER loss: {output.ner_loss.item():.4f}")
        print(f"  - Table loss: {output.table_loss.item():.4f}")
        print(f"  - Attr loss: {output.attr_loss.item():.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference(model):
    """Test inference mode"""
    print("\n" + "="*60)
    print("Testing Inference Mode")
    print("="*60)
    
    try:
        # Create dummy inputs
        batch_size = 1
        seq_len = 64
        
        print(f"\n4. Testing inference with CRF decoding...")
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        bbox = torch.randint(0, 1000, (batch_size, seq_len, 4))
        attention_mask = torch.ones(batch_size, seq_len)
        
        predictions = model.predict(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask
        )
        
        print("✓ Inference successful")
        print(f"  - NER predictions shape: {predictions['ner_predictions'].shape}")
        print(f"  - Table predictions shape: {predictions['table_predictions'].shape}")
        print(f"  - Attr predictions shape: {predictions['attr_predictions'].shape}")
        
        # Test confidence scoring
        print("\n5. Testing confidence scoring...")
        ner_preds, confidence = model.get_ner_predictions_with_confidence(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask
        )
        
        print("✓ Confidence scoring successful")
        print(f"  - Mean confidence: {confidence.mean().item():.4f}")
        print(f"  - Min confidence: {confidence.min().item():.4f}")
        print(f"  - Max confidence: {confidence.max().item():.4f}")
        
        return True
    
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_loading():
    """Test loading labels from config"""
    print("\n" + "="*60)
    print("Testing Label Configuration")
    print("="*60)
    
    try:
        import yaml
        
        label_path = project_root / "config" / "labels.yaml"
        print(f"\n6. Loading labels from {label_path}...")
        
        with open(label_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        labels = config['label_list']
        print(f"✓ Labels loaded successfully")
        print(f"  - Total labels: {len(labels)}")
        print(f"  - First 10: {labels[:10]}")
        
        # Verify label format
        bio_labels = [l for l in labels if l.startswith(('B-', 'I-'))]
        print(f"  - BIO labels: {len(bio_labels)}")
        print(f"  - O labels: {labels.count('O')}")
        
        return True
    
    except Exception as e:
        print(f"✗ Label loading failed: {e}")
        return False


def test_training_config():
    """Test loading training config"""
    print("\n" + "="*60)
    print("Testing Training Configuration")
    print("="*60)
    
    try:
        import yaml
        
        config_path = project_root / "config" / "training_config.yaml"
        print(f"\n7. Loading training config from {config_path}...")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Training config loaded successfully")
        print(f"  - Model: {config['model']['pretrained_name']}")
        print(f"  - Use CRF: {config['model']['use_crf']}")
        print(f"  - Batch size: {config['training']['batch_size']}")
        print(f"  - Learning rate: {config['training']['learning_rate']}")
        print(f"  - Epochs: {config['training']['num_epochs']}")
        print(f"  - FP16: {config['training']['fp16']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Training config loading failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Production Training Setup - Quick Test")
    print("="*60)
    print("\nThis script tests the production training components:")
    print("  - Multi-head model architecture")
    print("  - Forward pass with all heads")
    print("  - CRF inference mode")
    print("  - Configuration loading")
    print("\nNote: This uses CPU for testing. GPU will be used in actual training.")
    print("="*60)
    
    results = {}
    
    # Test label loading
    results['labels'] = test_label_loading()
    
    # Test training config
    results['config'] = test_training_config()
    
    # Test model creation
    model = test_model_creation()
    results['model_creation'] = model is not None
    
    if model:
        # Test forward pass
        results['forward_pass'] = test_forward_pass(model)
        
        # Test inference
        results['inference'] = test_inference(model)
    else:
        results['forward_pass'] = False
        results['inference'] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed! Production setup is ready.")
        print("\nNext steps:")
        print("  1. Generate training data: python scripts/pipeline.py generate -n 1000")
        print("  2. Validate annotations: python scripts/validate_annotations.py --input data/processed/train.jsonl")
        print("  3. Start training: python training/train.py --config config/training_config.yaml")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  - Install pytorch-crf: pip install pytorch-crf")
        print("  - Check dependencies: pip install -r requirements.txt")
        print("  - Verify config files exist in config/ directory")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
