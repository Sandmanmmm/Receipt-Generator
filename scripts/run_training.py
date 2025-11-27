"""
Run Training - Launch training with configuration
"""
import click
import yaml
import torch
from pathlib import Path
from datetime import datetime

from training.train import InvoiceDataset
from training.layoutlmv3_multihead import LayoutLMv3MultiHead
from training.data_collator import LayoutLMv3MultiTaskCollator
from training.metrics import MultiTaskMetrics, MetricsTracker
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm


@click.command()
@click.option('--config', '-c', default='config/training_config.yaml', help='Training config file')
@click.option('--data-dir', '-d', default='data', help='Dataset directory')
@click.option('--output-dir', '-o', default='models', help='Output directory')
@click.option('--resume', '-r', default=None, help='Resume from checkpoint')
@click.option('--dry-run', is_flag=True, help='Dry run (no training)')
def main(config, data_dir, output_dir, resume, dry_run):
    """Launch LayoutLMv3 training"""
    
    click.echo("="*80)
    click.echo("INVOICEGEN MODEL TRAINING")
    click.echo("="*80)
    
    # Load config
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Error: Config file not found: {config}")
        return
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Load labels
    labels_path = Path('config/labels.yaml')
    with open(labels_path, 'r') as f:
        labels_cfg = yaml.safe_load(f)
        ner_labels = labels_cfg.get('labels', [])
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = Path(output_dir)
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = output_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\nConfiguration:")
    click.echo(f"  Device: {device}")
    click.echo(f"  Data dir: {data_dir}")
    click.echo(f"  Output dir: {run_dir}")
    click.echo(f"  Epochs: {cfg['training']['epochs']}")
    click.echo(f"  Batch size: {cfg['training']['batch_size']}")
    click.echo(f"  Learning rate: {cfg['training']['learning_rate']}")
    click.echo(f"  Use CRF: {cfg['model']['use_crf']}")
    
    if dry_run:
        click.echo("\n[DRY RUN MODE - No training will be performed]")
        return
    
    # Load datasets
    click.echo("\n" + "="*80)
    click.echo("LOADING DATASETS")
    click.echo("="*80)
    
    try:
        train_dataset = InvoiceDataset(data_dir, split='train')
        val_dataset = InvoiceDataset(data_dir, split='val')
        
        click.echo(f"✓ Train: {len(train_dataset)} samples")
        click.echo(f"✓ Val: {len(val_dataset)} samples")
    except Exception as e:
        click.echo(f"Error loading datasets: {e}")
        click.echo("Have you built the dataset? Run: python scripts/build_training_set.py")
        return
    
    # Create dataloaders
    collator = LayoutLMv3MultiTaskCollator()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg['training'].get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg['training'].get('num_workers', 4)
    )
    
    # Initialize model
    click.echo("\n" + "="*80)
    click.echo("INITIALIZING MODEL")
    click.echo("="*80)
    
    model = LayoutLMv3MultiHead(
        model_name=cfg['model']['model_name'],
        num_ner_labels=len(ner_labels),
        num_table_labels=3,
        num_cell_labels=3,
        use_crf=cfg['model']['use_crf']
    )
    
    if resume:
        click.echo(f"Resuming from checkpoint: {resume}")
        model.load_state_dict(torch.load(resume, map_location=device))
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    click.echo(f"✓ Model: {cfg['model']['model_name']}")
    click.echo(f"  Total parameters: {total_params:,}")
    click.echo(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    num_training_steps = len(train_loader) * cfg['training']['epochs']
    num_warmup_steps = int(num_training_steps * cfg['training']['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    click.echo("\n" + "="*80)
    click.echo("TRAINING")
    click.echo("="*80)
    
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(cfg['training']['epochs']):
        click.echo(f"\n{'='*80}")
        click.echo(f"Epoch {epoch + 1}/{cfg['training']['epochs']}")
        click.echo(f"{'='*80}")
        
        # Train
        model.train()
        train_loss = 0.0
        
        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Move to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Forward
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['max_grad_norm'])
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        click.echo(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        click.echo(f"Average validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        # Early stopping (simplified)
        if avg_val_loss < best_val_f1:
            best_val_f1 = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_path = run_dir / "best_model.pt"
            model.save_pretrained(str(run_dir / "best"))
            click.echo(f"✓ Saved best model (val_loss: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            
            if patience_counter >= cfg['training']['early_stopping_patience']:
                click.echo(f"\nEarly stopping triggered (patience: {patience_counter})")
                break
    
    # Training complete
    click.echo("\n" + "="*80)
    click.echo("TRAINING COMPLETE")
    click.echo("="*80)
    click.echo(f"Output directory: {run_dir}")
    click.echo(f"Best model: {run_dir / 'best'}")
    click.echo("\nNext steps:")
    click.echo("  1. Evaluate: python evaluation/evaluate.py --model-path models/run_*/best")
    click.echo("  2. Deploy: docker-compose up invoicegen-api")


if __name__ == '__main__':
    main()
