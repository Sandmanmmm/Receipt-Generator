"""
LayoutLMv3 Training Script
Train custom LayoutLMv3 model on invoice dataset
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from pathlib import Path
import json
from typing import Dict, List
from dataclasses import dataclass


class InvoiceDataset(Dataset):
    """PyTorch Dataset for LayoutLMv3 training"""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize dataset
        
        Args:
            data_dir: Directory containing processed .pt files
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir) / split
        self.files = list(self.data_dir.glob('*.pt'))
        
        # Load label mapping
        label_map_file = self.data_dir / 'label_map.json'
        with open(label_map_file, 'r') as f:
            label_map = json.load(f)
        
        self.label2id = label_map['label2id']
        self.id2label = {int(k): v for k, v in label_map['id2label'].items()}
        
        print(f"Loaded {len(self.files)} samples for {split} split")
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a single sample"""
        sample = torch.load(self.files[idx])
        return sample


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_name: str = "microsoft/layoutlmv3-base"
    
    # Data
    data_dir: str = "data/layoutlmv3"
    output_dir: str = "models/layoutlmv3-invoice"
    
    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Evaluation
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_dir: str = "logs"
    logging_steps: int = 10
    report_to: List[str] = None
    
    # Hardware
    fp16: bool = True  # Use mixed precision if GPU available
    dataloader_num_workers: int = 4
    
    # Other
    seed: int = 42
    push_to_hub: bool = False


class InvoiceTrainer:
    """Trainer for LayoutLMv3 on invoices"""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration"""
        self.config = config
        
        # Load datasets
        self.train_dataset = InvoiceDataset(config.data_dir, 'train')
        self.val_dataset = InvoiceDataset(config.data_dir, 'val')
        
        # Get label mapping
        self.label2id = self.train_dataset.label2id
        self.id2label = self.train_dataset.id2label
        self.num_labels = len(self.label2id)
        
        # Initialize model
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Check for GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        if self.device == "cpu":
            config.fp16 = False  # Disable fp16 on CPU
    
    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score
        )
        
        predictions, labels = pred
        predictions = predictions.argmax(axis=-1)
        
        # Remove ignored index (padding)
        true_labels = [
            [self.id2label[l] for l in label if l != -100]
            for label in labels
        ]
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        return {
            "accuracy": accuracy_score(true_labels, true_predictions),
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    
    def train(self):
        """Start training"""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            evaluation_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            seed=self.config.seed,
            push_to_hub=self.config.push_to_hub,
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
        
        # Train
        print("Starting training...")
        train_result = trainer.train()
        
        # Save model
        print(f"Saving model to {self.config.output_dir}")
        trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        
        # Final evaluation
        print("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        return trainer, eval_metrics


def main():
    """Main training function"""
    # Configuration
    config = TrainingConfig(
        model_name="microsoft/layoutlmv3-base",
        data_dir="data/layoutlmv3",
        output_dir="models/layoutlmv3-invoice",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available()
    )
    
    # Initialize trainer
    trainer = InvoiceTrainer(config)
    
    # Train
    trained_model, metrics = trainer.train()
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"\nModel saved to: {config.output_dir}")


if __name__ == '__main__':
    main()
