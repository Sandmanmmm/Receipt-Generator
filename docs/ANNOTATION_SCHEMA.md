# Annotation Schema Specification

## Overview
This document defines the JSONL annotation format for training LayoutLMv3 models on invoice/purchase order documents.

## 🚀 Enhanced Schema Available
**New**: The enhanced label schema with **161 BIO labels (80 entity types)** provides **100% coverage** of real-world invoices including SaaS, telecom, logistics, utilities, healthcare, and government documents.

- **Current Schema**: 73 BIO labels (36 entities) - See `config/labels.yaml`
- **Enhanced Schema**: 161 BIO labels (80 entities) - See `config/labels_enhanced.yaml`
- **Coverage Analysis**: See `docs/LABEL_COVERAGE_ANALYSIS.md`
- **Migration Guide**: See `docs/ENHANCED_SCHEMA_GUIDE.md`

## File Format
- **Format**: JSONL (JSON Lines) - one JSON object per line
- **Encoding**: UTF-8
- **File extension**: `.jsonl`

## Schema Structure

### Document-Level Schema

```json
{
  "id": "doc_0001",
  "image_path": "data/images/doc_0001.png",
  "width": 2480,
  "height": 3508,
  "dpi": 300,
  "tokens": [...],
  "boxes": [...],
  "labels": [...],
  "table_labels": [...],
  "tables": [...]
}
```

### Field Descriptions

#### Required Fields

- **`id`** (string): Unique document identifier
- **`image_path`** (string): Relative or absolute path to the document image
- **`width`** (integer): Image width in pixels
- **`height`** (integer): Image height in pixels
- **`tokens`** (array): Array of token objects (see Token Schema below)
- **`boxes`** (array): Array of bounding boxes (duplicated for convenience)
- **`labels`** (array): Array of BIO labels, same length as tokens

#### Optional Fields

- **`dpi`** (integer): Image resolution in dots per inch (default: 300)
- **`table_labels`** (array): Array of table detection labels (O/B-TABLE/I-TABLE)
- **`tables`** (array): Array of table structure objects (see Table Schema below)
- **`metadata`** (object): Additional document metadata

---

## Token Schema

Each token in the `tokens` array contains:

```json
{
  "text": "Invoice",
  "bbox": [100, 120, 300, 160],
  "token_id": 0,
  "label": "B-DOC_TYPE",
  "confidence": 0.98,
  "font_size": 24,
  "font_family": "Arial"
}
```

### Token Field Descriptions

#### Required Fields

- **`text`** (string): The token text (word or subword)
- **`bbox`** (array[4]): Bounding box coordinates `[x0, y0, x1, y1]` in pixels
  - `x0, y0`: Top-left corner
  - `x1, y1`: Bottom-right corner
- **`token_id`** (integer): Sequential token index (0-based)
- **`label`** (string): BIO label from `config/labels.yaml`

#### Optional Fields

- **`confidence`** (float): OCR confidence score (0.0 to 1.0)
- **`font_size`** (integer): Font size in points
- **`font_family`** (string): Font family name
- **`is_bold`** (boolean): Whether text is bold
- **`is_italic`** (boolean): Whether text is italic
- **`color`** (string): Text color in hex format (e.g., "#000000")

---

## Table Schema

Tables are represented as structured objects with row and column information:

```json
{
  "table_id": 0,
  "row_span": [10, 25],
  "columns": [
    {"start": 10, "end": 12, "type": "description"},
    {"start": 13, "end": 14, "type": "quantity"},
    {"start": 15, "end": 16, "type": "price"}
  ],
  "rows": [
    {
      "row_index": 0,
      "token_span": [10, 16],
      "bbox": [50, 500, 800, 550],
      "cells": [
        {"column": 0, "tokens": [10, 11, 12], "text": "Organic Apples"},
        {"column": 1, "tokens": [13, 14], "text": "5"},
        {"column": 2, "tokens": [15, 16], "text": "$20.00"}
      ]
    }
  ]
}
```

### Table Field Descriptions

- **`table_id`** (integer): Unique table identifier within document
- **`row_span`** (array[2]): Token index range `[start, end]` covering entire table
- **`columns`** (array): Column definitions
  - `start`, `end`: Token index range for column
  - `type`: Column semantic type (description, quantity, price, etc.)
- **`rows`** (array): Row objects
  - `row_index`: Row number (0-based)
  - `token_span`: Token indices for this row
  - `bbox`: Bounding box covering entire row
  - `cells`: Cell objects with column index and tokens

---

## Complete Example

```json
{
  "id": "inv_2025_001",
  "image_path": "data/raw/invoices/inv_2025_001.png",
  "width": 2480,
  "height": 3508,
  "dpi": 300,
  "tokens": [
    {
      "text": "INVOICE",
      "bbox": [100, 120, 300, 160],
      "token_id": 0,
      "label": "B-DOC_TYPE",
      "confidence": 0.99,
      "font_size": 28,
      "is_bold": true
    },
    {
      "text": "INV-2025-001",
      "bbox": [320, 120, 520, 160],
      "token_id": 1,
      "label": "B-INVOICE_NUMBER",
      "confidence": 0.97,
      "font_size": 24
    },
    {
      "text": "Date:",
      "bbox": [100, 200, 180, 230],
      "token_id": 2,
      "label": "O",
      "confidence": 0.98
    },
    {
      "text": "2025-11-26",
      "bbox": [200, 200, 350, 230],
      "token_id": 3,
      "label": "B-INVOICE_DATE",
      "confidence": 0.96
    },
    {
      "text": "Supplier:",
      "bbox": [100, 300, 220, 330],
      "token_id": 4,
      "label": "O",
      "confidence": 0.99
    },
    {
      "text": "Acme",
      "bbox": [240, 300, 320, 330],
      "token_id": 5,
      "label": "B-SUPPLIER_NAME",
      "confidence": 0.95
    },
    {
      "text": "Foods",
      "bbox": [330, 300, 410, 330],
      "token_id": 6,
      "label": "I-SUPPLIER_NAME",
      "confidence": 0.95
    },
    {
      "text": "Ltd",
      "bbox": [420, 300, 470, 330],
      "token_id": 7,
      "label": "I-SUPPLIER_NAME",
      "confidence": 0.94
    }
  ],
  "boxes": [
    [100, 120, 300, 160],
    [320, 120, 520, 160],
    [100, 200, 180, 230],
    [200, 200, 350, 230],
    [100, 300, 220, 330],
    [240, 300, 320, 330],
    [330, 300, 410, 330],
    [420, 300, 470, 330]
  ],
  "labels": [
    "B-DOC_TYPE",
    "B-INVOICE_NUMBER",
    "O",
    "B-INVOICE_DATE",
    "O",
    "B-SUPPLIER_NAME",
    "I-SUPPLIER_NAME",
    "I-SUPPLIER_NAME"
  ],
  "table_labels": [
    "O", "O", "O", "O", "O", "O", "O", "O"
  ],
  "tables": [],
  "metadata": {
    "source": "synthetic",
    "template": "modern_invoice",
    "locale": "en_US",
    "generated_at": "2025-11-26T10:30:00Z"
  }
}
```

---

## BIO Label Set

See `config/labels.yaml` for the complete list of 73 labels:

### Label Categories

1. **Document Metadata** (12 labels)
   - DOC_TYPE, INVOICE_NUMBER, PURCHASE_ORDER_NUMBER, dates

2. **Party Information** (20 labels)
   - Supplier: name, VAT, address, phone, email
   - Buyer: name, address, phone, email

3. **Financial Fields** (14 labels)
   - CURRENCY, TOTAL_AMOUNT, TAX_AMOUNT, SUBTOTAL, etc.

4. **Line Items** (20 labels)
   - ITEM_DESCRIPTION, ITEM_SKU, ITEM_QTY, costs, etc.

5. **Structural** (2 labels)
   - TABLE (for table detection)

6. **Miscellaneous** (5 labels)
   - TERMS_AND_CONDITIONS, NOTE, GENERIC_LABEL

---

## Coordinate System

### Bounding Box Format
- **Format**: `[x0, y0, x1, y1]`
- **Units**: Pixels (absolute coordinates)
- **Origin**: Top-left corner of image (0, 0)
- **Normalization**: For LayoutLMv3, coordinates are normalized to 0-1000 range

### Normalization Formula
```python
def normalize_bbox(bbox, width, height):
    """Normalize bbox to 0-1000 range for LayoutLMv3"""
    return [
        int(1000 * bbox[0] / width),
        int(1000 * bbox[1] / height),
        int(1000 * bbox[2] / width),
        int(1000 * bbox[3] / height),
    ]
```

---

## Data Split Convention

### File Naming
- **Training set**: `train.jsonl`
- **Validation set**: `val.jsonl` or `dev.jsonl`
- **Test set**: `test.jsonl`

### Split Ratios
- Training: 70-80%
- Validation: 10-15%
- Test: 10-15%

### Stratification
- Maintain class balance across splits
- Ensure each split contains examples of all entity types
- Consider document-level stratification for multi-page documents

---

## Quality Guidelines

### Token Quality
- Tokens should be meaningful words or subwords
- Avoid splitting numbers or dates unnecessarily
- Include punctuation as separate tokens when semantically important

### Bounding Box Quality
- Boxes should tightly fit the text
- No overlapping boxes for different tokens
- Boxes should be within image boundaries

### Label Quality
- Use BIO format strictly (B- for begin, I- for inside)
- First token of an entity always gets B- prefix
- Continuation tokens get I- prefix
- O for tokens outside any entity

### Consistency
- Maintain consistent tokenization across documents
- Use same label for same semantic meaning
- Follow template-specific conventions

---

## Validation Rules

### Required Checks
1. All arrays (tokens, boxes, labels) have same length
2. All bounding boxes within image boundaries
3. All labels exist in `config/labels.yaml`
4. Token IDs are sequential starting from 0
5. BIO label sequence is valid (no I- without preceding B-)

### Optional Checks
1. OCR confidence above threshold (e.g., 0.7)
2. Token text not empty
3. Bounding box area > 0
4. No duplicate token_ids

---

## Tools and Utilities

### Validation Script
```bash
python scripts/validate_annotations.py --input data/processed/train.jsonl
```

### Conversion Script
```bash
# Convert from other formats
python scripts/convert_annotations.py \
  --input data/raw/annotations.json \
  --format coco \
  --output data/processed/train.jsonl
```

### Visualization
```bash
# Visualize annotations
python scripts/visualize_annotations.py \
  --input data/processed/train.jsonl \
  --output-dir visualizations/ \
  --num-samples 10
```

---

## Version History

- **v1.0** (2025-11-26): Initial schema with 73 BIO labels
- Multi-task support: NER + table detection + cell attributes
- Compatible with LayoutLMv3 architecture
