# Data Directory Structure

This directory contains all data for the InvoiceGen pipeline.

## Directory Layout

```
data/
├── raw/                    # Raw synthetic invoices (PDF/PNG)
├── processed/              # Preprocessed images for training
├── annotations/            # Manual annotations (if any)
├── annotated/             # Auto-annotated JSONL files from pipeline
├── train/                 # Training split (80%)
├── val/                   # Validation split (10%)
└── test/                  # Test split (10%)
```

## Pipeline Flow

1. **Generate** → `raw/`: Generate synthetic invoices using templates
2. **Annotate** → `annotated/`: Auto-annotate with OCR + label mapping
3. **Split** → `train/`, `val/`, `test/`: Split into ML datasets
4. **Augment** → `processed/`: Apply augmentations during training

## Data Formats

### Raw Invoices (raw/)
- PDF files: `invoice_001.pdf`, `invoice_002.pdf`, ...
- PNG images: `invoice_001.png`, `invoice_002.png`, ...
- Metadata JSON: `invoice_001.json` (contains invoice data used for generation)

### Annotated Files (annotated/)
JSONL format with token-level annotations:
```json
{
  "image_path": "path/to/image.png",
  "tokens": ["INVOICE", "#", "INV", "-", "2024", "-", "001"],
  "labels": ["B-DOCUMENT_TYPE", "O", "B-INVOICE_NUMBER", "I-INVOICE_NUMBER", "I-INVOICE_NUMBER", "I-INVOICE_NUMBER", "I-INVOICE_NUMBER"],
  "bboxes": [[10, 20, 100, 40], [105, 20, 110, 40], ...],
  "image_width": 2480,
  "image_height": 3508
}
```

### Training Splits (train/, val/, test/)
Each split contains:
- `images/`: Image files
- `annotations.jsonl`: Token-level annotations
- `metadata.json`: Split statistics

## Usage

### Generate Dataset
```bash
python scripts/build_training_set.py --num-samples 1000 --output data/
```

### Split Dataset
```python
from training.dataset_builder import split_dataset

split_dataset(
    input_dir="data/annotated",
    output_dir="data/",
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

### Load Dataset
```python
from training.train import InvoiceDataset

train_dataset = InvoiceDataset("data/", split="train")
val_dataset = InvoiceDataset("data/", split="val")
test_dataset = InvoiceDataset("data/", split="test")
```

## Storage Requirements

Estimated storage per 1000 invoices:
- Raw PDF: ~50 MB (50 KB each)
- Raw PNG (2480x3508): ~400 MB (400 KB each)
- Annotations JSONL: ~10 MB (10 KB each)
- **Total**: ~460 MB per 1000 invoices

For 10,000 training samples: ~4.6 GB

## Git Ignore

The following are git-ignored (configured in .gitignore):
- `data/raw/*` (except .gitkeep)
- `data/processed/*` (except .gitkeep)
- `data/annotated/*.jsonl`
- `data/train/images/*`
- `data/val/images/*`
- `data/test/images/*`

Annotations and metadata are tracked for reproducibility.
