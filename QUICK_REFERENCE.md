# Quick Reference - Production Training

## 🚀 Quick Start Commands

### 1. Verify Setup
```bash
# Run all tests (should pass)
python tests/test_production_setup.py

# Expected output: ✓ All tests passed! Production setup is ready.
```

### 2. Generate Training Data
```bash
# Generate 1000 synthetic invoices
python scripts/pipeline.py generate -n 1000

# Quick test with 5 samples
python scripts/quickstart.py
```

### 3. Validate Annotations
```bash
# Validate JSONL format
python scripts/validate_annotations.py \
  --input data/processed/train.jsonl \
  --verbose

# Visualize samples
python scripts/visualize_annotations.py \
  --input data/processed/train.jsonl \
  --output-dir visualizations/ \
  --num-samples 20
```

### 4. Start Training
```bash
# Train with production config
python training/train.py --config config/training_config.yaml

# Monitor with TensorBoard
tensorboard --logdir logs/tensorboard
```

### 5. Evaluate Model
```bash
# Evaluate on test set
python evaluation/evaluate.py \
  --model models/checkpoints/best_model.pt \
  --test data/processed/test.jsonl
```

### 6. Deploy API
```bash
# Launch production API
python deployment/api.py

# Test endpoint
curl -X POST http://localhost:8000/extract -F "file=@invoice.pdf"
```

---

## 📁 Key Files Reference

### Configuration Files
- `config/labels.yaml` - 73 BIO labels for NER
- `config/training_config.yaml` - Training hyperparameters
- `config/config.yaml` - Pipeline configuration

### Model Files
- `training/layoutlmv3_multihead.py` - Multi-head model (125M params)
- `training/train.py` - Training loop
- `training/data_converter.py` - Data format conversion

### Utility Scripts
- `scripts/validate_annotations.py` - Schema validator
- `scripts/visualize_annotations.py` - Annotation viewer
- `scripts/pipeline.py` - End-to-end pipeline
- `tests/test_production_setup.py` - Automated tests

### Documentation
- `docs/ANNOTATION_SCHEMA.md` - Format specification
- `docs/TRAINING_SETUP.md` - Detailed training guide
- `IMPLEMENTATION_COMPLETE.md` - Implementation summary
- `README.md` - Project overview

---

## 🎯 Label Categories (73 Labels)

### Document Metadata (12)
```
B/I-DOC_TYPE, B/I-INVOICE_NUMBER, B/I-PURCHASE_ORDER_NUMBER
B/I-INVOICE_DATE, B/I-DUE_DATE, B/I-ORDER_DATE
```

### Party Information (20)
```
# Supplier
B/I-SUPPLIER_NAME, B/I-SUPPLIER_VAT, B/I-SUPPLIER_ADDRESS
B/I-SUPPLIER_PHONE, B/I-SUPPLIER_EMAIL

# Buyer
B/I-BUYER_NAME, B/I-BUYER_ADDRESS
B/I-BUYER_PHONE, B/I-BUYER_EMAIL
```

### Financial (14)
```
B/I-CURRENCY, B/I-TOTAL_AMOUNT, B/I-TAX_AMOUNT
B/I-SUBTOTAL, B/I-DISCOUNT, B/I-TAX_RATE, B/I-PAYMENT_TERMS
```

### Line Items (20)
```
B/I-PO_LINE_ITEM, B/I-ITEM_DESCRIPTION, B/I-ITEM_SKU
B/I-ITEM_QTY, B/I-ITEM_UNIT, B/I-ITEM_UNIT_COST
B/I-ITEM_TOTAL_COST, B/I-ITEM_PACK_SIZE
B/I-ITEM_TAX, B/I-ITEM_DISCOUNT
```

### Other (7)
```
O (outside), B/I-TABLE, B/I-TERMS_AND_CONDITIONS
B/I-NOTE, B/I-GENERIC_LABEL
```

---

## ⚙️ Configuration Quick Tweaks

### Fast Development Mode
Edit `config/training_config.yaml`:
```yaml
training:
  batch_size: 8
  grad_accumulation_steps: 2
  num_epochs: 5
model:
  use_crf: false
```

### Production Mode
```yaml
training:
  batch_size: 4
  grad_accumulation_steps: 8
  num_epochs: 20
model:
  use_crf: true
```

### Large Model Mode
```yaml
model:
  pretrained_name: "microsoft/layoutlmv3-large"
training:
  batch_size: 2
  grad_accumulation_steps: 16
```

---

## 🐛 Troubleshooting

### Model Creation Fails
```bash
# Install CRF
pip install pytorch-crf

# Verify installation
python -c "import torchcrf; print('CRF OK')"
```

### CUDA Out of Memory
Edit `config/training_config.yaml`:
```yaml
training:
  batch_size: 2  # Reduce from 4
  gradient_checkpointing: true  # Enable
```

### Validation Errors
```bash
# Check schema
python scripts/validate_annotations.py \
  --input data/processed/train.jsonl \
  --verbose

# Fix common issues:
# - Ensure all arrays same length
# - Verify labels in labels.yaml
# - Check BIO sequence validity
```

### Poor Model Performance
1. Generate more data: `--num-samples 5000`
2. Increase epochs: `num_epochs: 30`
3. Tune augmentation probabilities
4. Use LayoutLMv3-large model
5. Adjust learning rate

---

## 📊 Model Architecture

```
Input: Image (224×224) + Text Tokens + Bounding Boxes
         ↓
LayoutLMv3 Backbone (125M params)
         ↓
   Hidden States (768-dim)
         ↓
    ┌────┴────┬──────────┐
    ↓         ↓          ↓
NER Head  Table Head  Cell Attr
(73 labs) (3 labels)  (3 attrs)
    ↓         ↓          ↓
   CRF     Softmax    Sigmoid
    ↓         ↓          ↓
  Token    Table     Column
  Labels   Regions    Types
         ↓
   Structured JSON Output
```

---

## 📈 Training Metrics

### Monitor These Metrics:
- **train_loss**: Should decrease steadily
- **eval_f1**: Should increase (target: >0.85)
- **eval_entity_f1**: Entity-level F1 (target: >0.80)
- **learning_rate**: Follows cosine schedule

### TensorBoard Graphs:
```bash
tensorboard --logdir logs/tensorboard

# View at: http://localhost:6006
```

### Early Stopping:
- Monitors `eval_f1` metric
- Patience: 3 epochs
- Saves best checkpoint automatically

---

## 🔍 Inspection Commands

### Check Label Distribution
```python
import yaml
with open('config/labels.yaml') as f:
    labels = yaml.safe_load(f)['label_list']
print(f"Total labels: {len(labels)}")
```

### Inspect Model
```python
from training import create_model
model = create_model(num_ner_labels=73, use_crf=True)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Preview Annotation
```python
import json
with open('data/processed/train.jsonl') as f:
    doc = json.loads(f.readline())
print(f"Tokens: {len(doc['tokens'])}")
print(f"Labels: {set(doc['labels'])}")
```

---

## ⏱️ Training Time Estimates

### Single GPU (RTX 3090 / A100)
- **1k samples**: 2-3 hours (20 epochs)
- **5k samples**: 8-10 hours (20 epochs)
- **10k samples**: 15-18 hours (20 epochs)

### CPU (Not Recommended)
- **1k samples**: 24-30 hours (20 epochs)
- Use GPU for production training

### Multi-GPU (Future)
- Edit `config/training_config.yaml`:
```yaml
hardware:
  distributed:
    enabled: true
    world_size: 4  # Number of GPUs
```

---

## 📦 Export & Deployment

### Save Model for Inference
```python
# Model is auto-saved to:
# models/checkpoints/best_model.pt

# Load for inference:
from training import LayoutLMv3MultiHead
model = LayoutLMv3MultiHead.from_pretrained(
    'models/checkpoints/best_model'
)
```

### Export to HuggingFace Hub
```python
model.push_to_hub("your-username/invoice-extraction-v1")
```

### Docker Deployment
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
COPY . /workspace
RUN pip install -r requirements.txt
CMD ["python", "deployment/api.py"]
```

---

## 🎓 Learning Resources

### Key Papers
- LayoutLMv3: https://arxiv.org/abs/2204.08387
- CRF for NER: https://arxiv.org/abs/1603.01360

### Related Tools
- Label Studio: https://labelstud.io/
- CVAT: https://cvat.org/
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR

### HuggingFace Models
- LayoutLMv3-base: microsoft/layoutlmv3-base
- LayoutLMv3-large: microsoft/layoutlmv3-large

---

## ✅ Pre-Flight Checklist

Before training:
- [ ] All tests pass: `python tests/test_production_setup.py`
- [ ] Config files exist: `config/*.yaml`
- [ ] Labels loaded: 73 labels in `config/labels.yaml`
- [ ] pytorch-crf installed: `pip list | grep pytorch-crf`
- [ ] GPU available (optional): `nvidia-smi`
- [ ] Training data ready: `data/processed/train.jsonl`

---

## 📞 Quick Help

### Check Setup Status
```bash
python tests/test_production_setup.py
```

### View All Labels
```bash
cat config/labels.yaml | grep "^  - " | wc -l
# Should output: 73
```

### Test Model Import
```bash
python -c "from training import create_model; print('✓ Model import OK')"
```

### Verify Dependencies
```bash
pip list | grep -E "torch|transformers|pytorch-crf"
```

---

**Status**: ✅ Ready for production training  
**Next**: Generate training data or start with quickstart
