# InvoiceGen - Production-Ready Invoice Understanding System

A complete end-to-end pipeline for generating synthetic invoices and training custom LayoutLMv3 models for document understanding with **production-grade architecture**.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Overview

InvoiceGen provides a modular system for complete invoice document understanding:

1. **Generate** - Realistic synthetic invoices (3 template styles)
2. **Render** - Multi-format output (HTML → PDF → PNG)
3. **Auto-annotate** - OCR + pattern-based labeling (73 BIO tags)
4. **Augment** - Realistic document distortions
5. **Train** - Multi-task LayoutLMv3 (NER + Table + Cell + CRF)
6. **Evaluate** - Confusion matrices, seqeval reports, error analysis
7. **Deploy** - Docker + FastAPI + batch inference

## ✨ Production Features

- **73 BIO Labels**: Complete invoice/PO extraction (metadata, parties, financial, line items)
- **Multi-Task Learning**: NER + Table Detection + Cell Attributes
- **CRF Layer**: Stable BIO transitions with Viterbi decoding
- **Modular Architecture**: Separated concerns (generators, annotation, training, evaluation, deployment)
- **Docker Support**: `docker-compose up` for instant deployment
- **Batch Inference**: Async processing with high throughput
- **Multiple OCR Backends**: PaddleOCR, Tesseract, EasyOCR

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/InvoiceGen.git
cd InvoiceGen
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Build Dataset (1000 invoices)

```bash
python scripts/build_training_set.py --num-samples 1000
```

**Pipeline**: Generate → Render → Augment → OCR → Annotate → Split (train/val/test 80/10/10)

### Train Model

```bash
python scripts/run_training.py --config config/training_config.yaml
```

### Evaluate

```bash
python evaluation/evaluate.py --model-path models/run_*/best
```

### Deploy

```bash
docker-compose up invoicegen-api
# API: http://localhost:8000
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INVOICEGEN PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [1] GENERATION                                                   │
│      ├─ SyntheticDataGenerator → Random invoice data             │
│      ├─ TemplateRenderer → Jinja2 → HTML                         │
│      ├─ PDFRenderer → WeasyPrint/wkhtmltopdf → PDF               │
│      └─ ImageRenderer → pdf2image → PNG                          │
│                                                                   │
│  [2] ANNOTATION                                                   │
│      ├─ OCREngine → PaddleOCR/Tesseract → Text + Boxes           │
│      ├─ LabelMapper → Pattern matching → BIO labels              │
│      └─ AnnotationWriter → JSONL format                          │
│                                                                   │
│  [3] AUGMENTATION                                                 │
│      └─ Augmenter → Noise/Blur/Rotation → Distorted images       │
│                                                                   │
│  [4] TRAINING                                                     │
│      ├─ DatasetBuilder → Train/Val/Test splits                   │
│      ├─ LayoutLMv3MultiHead → Multi-task model                   │
│      │   ├─ NER Head (73 labels)                                 │
│      │   ├─ Table Head (3 labels)                                │
│      │   ├─ Cell Head (3 labels)                                 │
│      │   └─ CRF Layer                                             │
│      └─ Trainer → AdamW + FP16 + Grad Accumulation               │
│                                                                   │
│  [5] EVALUATION                                                   │
│      ├─ Confusion Matrix → Visualization                         │
│      ├─ Seqeval → Per-entity F1 scores                           │
│      └─ Error Analysis → Categorization                          │
│                                                                   │
│  [6] DEPLOYMENT                                                   │
│      ├─ ModelLoader → Load checkpoints                           │
│      ├─ BatchRunner → Async inference                            │
│      └─ FastAPI → REST endpoints                                 │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 73 BIO Labels

**Document (12)**: INVOICE_NUMBER, PO_NUMBER, INVOICE_DATE, DUE_DATE, CURRENCY_CODE, etc.

**Parties (20)**: SUPPLIER_NAME, SUPPLIER_ADDRESS, CUSTOMER_NAME, SHIP_TO_ADDRESS, etc.

**Financial (14)**: SUBTOTAL, TAX_AMOUNT, TOTAL_AMOUNT, PAYMENT_TERMS, etc.

**Line Items (20)**: ITEM_DESCRIPTION, ITEM_QUANTITY, ITEM_RATE, ITEM_TOTAL, etc.

**Structure (2)**: TABLE_HEADER, TABLE_ROW

**Misc (5)**: NOTES, TERMS_AND_CONDITIONS, SIGNATURE, BARCODE, QR_CODE

See [`config/labels.yaml`](config/labels.yaml) for complete list.

## 🎯 Model: LayoutLMv3MultiHead

```
Input: Image (2480×3508) + Tokens + Bounding Boxes
  ↓
LayoutLMv3-base (125M params)
  ↓
├─→ NER Head (73 classes) → CRF Layer → Entity predictions
├─→ Table Head (3 classes) → Table structure
└─→ Cell Head (3 classes) → Cell attributes

Loss: L_total = 1.0×L_NER + 0.7×L_table + 0.5×L_cell
```

**Training**: AdamW (lr=5e-5), FP16, Batch=4×4 (grad accum), 20 epochs, Early stopping (patience=3)

## 🐳 Docker Deployment

### Services

```yaml
invoicegen-api:        # FastAPI server (port 8000)
invoicegen-training:   # Training service (GPU)
invoicegen-annotation: # Batch annotation
```

### Usage

```bash
# Start API
docker-compose up invoicegen-api

# Start training
docker-compose up invoicegen-training

# Run annotation pipeline
docker-compose up invoicegen-annotation
```

### API Endpoints

```
POST /predict          - Single document
POST /predict/batch    - Batch inference
GET /health            - Health check
GET /metrics           - Prometheus metrics
```

## 📁 Project Structure

```
InvoiceGen/
├── templates/          # Invoice templates (modern/classic/receipt)
├── generators/         # Modular generation (data/template/pdf/image)
├── annotation/         # Modular annotation (ocr/extract/label/write)
├── augmentation/       # Image augmentation
├── training/           # Training infrastructure
├── evaluation/         # Comprehensive evaluation
├── deployment/         # Production deployment
├── data/               # Structured data (raw/annotated/train/val/test)
├── config/             # Configuration files
├── scripts/            # Automation scripts
├── docs/               # Documentation
├── tests/              # Test suite
├── Dockerfile          # Docker image
├── docker-compose.yml  # Multi-service setup
└── requirements.txt    # Dependencies
```

## 📚 Documentation

- **[ANNOTATION_SCHEMA.md](docs/ANNOTATION_SCHEMA.md)** - JSONL format specification
- **[TRAINING_SETUP.md](docs/TRAINING_SETUP.md)** - Complete training guide
- **[PRODUCTION_DEPLOYMENT.md](docs/PRODUCTION_DEPLOYMENT.md)** - Deployment guide
- **[QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - Command cheatsheet

## 🧪 Testing

```bash
# Quick commands (using Makefile)
make test              # Run all tests
make test-fast         # Skip slow and Docker tests
make test-cov          # Run with coverage report
make lint              # Check code quality
make format            # Auto-format code

# Direct pytest commands
pytest tests/ -v                           # All tests
pytest tests/ -m "not slow and not docker" # Fast tests only
pytest tests/ --cov=. --cov-report=html    # With coverage

# Specific test modules
pytest tests/test_annotation_modular.py    # Annotation system
pytest tests/test_evaluation.py            # Evaluation tools
pytest tests/test_training.py              # Training support
pytest tests/test_config.py                # Configuration
```

**Test Coverage:**
- ✅ 1,200+ lines of tests across 9 modules
- ✅ Unit + integration tests
- ✅ 95%+ code coverage
- ✅ CI/CD automated testing
- ✅ Multi-Python version support (3.9-3.12)

## 📈 Evaluation Output

```
outputs/evaluation/
├── eval_confusion_matrix_full.png      # Full CM (73×73)
├── eval_confusion_matrix_entities.png  # Top-20 confused pairs
├── eval_seqeval_report.txt             # Per-entity metrics
├── eval_seqeval_report.json            # JSON format
├── eval_error_report.txt               # Error categorization
├── eval_errors.json                    # Detailed errors
└── eval_summary.json                   # Overall summary
```

## 🔧 Configuration

### Environment Variables
```bash
cp .env.template .env
# Edit .env with your configuration
```

### Training Config (`config/training_config.yaml`)
```yaml
model:
  model_name: microsoft/layoutlmv3-base
  use_crf: true
  
training:
  epochs: 20
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  fp16: true
  early_stopping_patience: 3
```

## 🛠️ Development Tools

**Makefile Commands:**
```bash
make install-dev       # Install all dependencies
make test              # Run tests
make lint              # Check code quality
make format            # Auto-format code
make build-dataset     # Generate training data
make train             # Train model
make docker-up         # Start services
make clean             # Clean artifacts
```

**CI/CD:**
- GitHub Actions workflow for automated testing
- Multi-Python version support (3.9-3.12)
- Automated linting, type checking, security scanning
- Docker build and push automation

**Monitoring:**
- Structured logging (JSON format)
- Prometheus metrics export
- Health check endpoints
- Error tracking

### Augmentation Config (`augmentation/settings.yaml`)

```yaml
augmentation_probability: 0.8

geometric:
  rotation: {enabled: true, probability: 0.5, angle_range: [-5, 5]}
  
noise:
  gaussian_noise: {enabled: true, probability: 0.4, std_range: [0.01, 0.03]}
  
document:
  jpeg_compression: {enabled: true, probability: 0.5, quality_range: [60, 95]}
```

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE).

## 🙏 Acknowledgments

- **LayoutLMv3**: Microsoft Research
- **PaddleOCR**: PaddlePaddle Team
- **Hugging Face**: Transformers library

---

**Built for production document understanding** 🚀
