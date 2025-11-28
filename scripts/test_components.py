#!/usr/bin/env python3
"""
Quick Test: HTML→PNG Renderer and TokenAnnotator Components
Validates the two new components work correctly before running full Test 11
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
from generators.retail_data_generator import RetailDataGenerator
from generators.html_to_png_renderer import SimplePNGRenderer
from annotation.token_annotator import TokenAnnotator

print("="*80)
print("TESTING NEW COMPONENTS")
print("="*80)
print()

# Test 1: SimplePNGRenderer
print("Test 1: SimplePNGRenderer")
print("-" * 80)

generator = RetailDataGenerator()
receipt_obj = generator.generate_pos_receipt()
receipt_dict = generator.to_dict(receipt_obj)

print(f"Generated receipt: {receipt_dict.get('invoice_number')}")

renderer = SimplePNGRenderer(width=800, height=1200)
output_path = project_root / 'outputs' / 'test_receipt.png'
success = renderer.render_receipt_dict(receipt_dict, str(output_path))

if success:
    print(f"✓ PNG rendered successfully: {output_path}")
    print(f"✓ File exists: {output_path.exists()}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")
else:
    print("✗ PNG rendering failed")

print()

# Test 2: TokenAnnotator
print("Test 2: TokenAnnotator")
print("-" * 80)

# Load schema
schema_path = project_root / 'config' / 'labels_retail.yaml'
with open(schema_path, 'r', encoding='utf-8') as f:
    schema = yaml.safe_load(f)

annotator = TokenAnnotator(schema)
print(f"Loaded {len(annotator.label_list)} labels")

# Simulate OCR tokens and bboxes
tokens = [
    receipt_dict.get('supplier_name', '').split()[0],  # Store name
    'Invoice:', 
    receipt_dict.get('invoice_number', ''),
    'Date:',
    receipt_dict.get('invoice_date', ''),
    'Total:',
    str(receipt_dict.get('total_amount', ''))
]

# Simple bboxes
bboxes = [[50, 50+i*30, 200, 70+i*30] for i in range(len(tokens))]

# Annotate
annotation = annotator.annotate_tokens(
    receipt_dict,
    tokens,
    bboxes,
    str(output_path),
    image_width=800,
    image_height=1200
)

print(f"✓ Annotation created")
print(f"  Tokens: {len(annotation['tokens'])}")
print(f"  NER tags: {len(annotation['ner_tags'])}")
print(f"  Bboxes: {len(annotation['bboxes'])}")
print()

# Show labels
print("Token labels:")
if len(annotator.id2label) > 0:
    for i, (token, tag_id) in enumerate(zip(annotation['tokens'], annotation['ner_tags'])):
        label = annotator.id2label.get(tag_id, 'UNKNOWN')
        if label != 'O' and label != 'UNKNOWN':
            print(f"  {token:20s} → {label}")
else:
    print("  Warning: No labels loaded from schema")

print()

# Validate annotation
is_valid, errors = annotator.validate_annotation(annotation)
if is_valid:
    print("✓ Annotation valid")
else:
    print(f"✗ Annotation validation errors: {errors}")

print()

# Get label statistics
stats = annotator.get_label_statistics(annotation['ner_tags'])
print("Label distribution:")
for label, count in sorted(stats.items(), key=lambda x: -x[1])[:10]:
    print(f"  {label:30s} {count:3d}")

print()
print("="*80)
print("COMPONENT TEST COMPLETE")
print("="*80)
