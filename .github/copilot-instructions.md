# InvoiceGen - Synthetic Invoice Generator with LayoutLMv3 Training

## Project Overview
This workspace contains a complete pipeline for generating synthetic invoices and training LayoutLMv3 models for document understanding.

## Development Guidelines

### Code Organization
- `templates/`: HTML/CSS + Jinja2 invoice templates
- `generators/`: Synthetic data generation modules
- `annotation/`: Auto-annotation system for bounding boxes
- `augmentation/`: Image augmentation pipeline
- `training/`: LayoutLMv3 training scripts
- `evaluation/`: Model evaluation and iteration tools
- `deployment/`: Production deployment scripts
- `config/`: Configuration files
- `scripts/`: Utility and pipeline scripts

### Coding Standards
- Use type hints for all Python functions
- Follow PEP 8 style guidelines
- Document all modules, classes, and functions
- Use dataclasses for configuration objects
- Keep functions focused and modular

### Pipeline Workflow
1. Generate invoices from templates
2. Render to PDF/PNG
3. Auto-annotate with bounding boxes
4. Apply augmentation
5. Convert to LayoutLMv3 format
6. Train model
7. Evaluate and iterate
8. Deploy to production

### Dependencies
- Python 3.9+
- Jinja2 for templating
- wkhtmltopdf or Puppeteer for rendering
- PaddleOCR for text extraction
- transformers for LayoutLMv3
- OpenCV/Pillow for augmentation
- PyTorch for training

### Testing
- Unit tests for each module
- Integration tests for pipeline stages
- Visual inspection of generated invoices
- Model performance metrics
