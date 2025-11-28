#!/usr/bin/env python3
"""
Test 13: Gold Sample Hand-Verification (Human QA)

Purpose: Validate that what the model will learn is semantically correct.
         Automated tests can't catch: wrong SKUs, monetary inconsistencies,
         missing line items, tax miscalculations, unrealistic data.

This test generates gold samples for manual inspection:
- 20 PDF receipts (if renderer available)
- 20 PNG receipts
- Full annotation data
- Visual comparison tool

Manual checks required:
1. Numbers reconcile (subtotal + tax = total)
2. Line items are realistic (prices, quantities, descriptions)
3. Discounts apply correctly
4. OCR has no text duplication
5. Bounding boxes align visually with text
6. No phantom line items
7. Tax rates are reasonable
8. Dates are valid
9. Currency formatting consistent
10. Entity labels match visual content

If even ONE significant failure appears → fix generator/labeler before training.

Usage:
    python scripts/test_13_gold_sample_verification.py
    python scripts/test_13_gold_sample_verification.py --num-samples 20
    python scripts/test_13_gold_sample_verification.py --output-dir outputs/gold_samples
"""

import argparse
import sys
import yaml
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import shutil
from jinja2 import Environment, FileSystemLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator
from generators.modern_invoice_generator import ModernInvoiceGenerator
from generators.html_to_png_renderer import HTMLToPNGRenderer, SimplePNGRenderer
from generators.visual_assets import VisualAssetGenerator
from annotation.ocr_engine import OCREngine
from annotation.token_annotator import TokenAnnotator


class GoldSample:
    """Container for a gold sample with all validation data."""
    
    def __init__(self, sample_id: str, sample_type: str = 'receipt'):
        self.sample_id = sample_id
        self.type = sample_type  # 'receipt' or 'invoice'
        
        # Data
        self.data_dict: Dict = {}
        self.image_path: str = ""
        self.template_name: str = ""
        
        # OCR results
        self.ocr_tokens: List[str] = []
        self.ocr_bboxes: List[List[int]] = []
        self.ocr_confidence: List[float] = []
        
        # Annotation
        self.ner_tags: List[int] = []
        self.ner_labels: List[str] = []
        
        # Validation flags
        self.issues: List[str] = []
        self.warnings: List[str] = []
        
        # Financial validation
        self.subtotal_matches = False
        self.tax_calculation_matches = False
        self.total_matches = False
        self.discount_applied_correctly = False
        
        # Asset validation
        self.has_logo = False
        self.has_qr = False
        self.has_barcode = False


def clean_currency(value: Any) -> Decimal:
    """Convert currency string or number to Decimal."""
    if value is None:
        return Decimal(0)
    
    str_val = str(value)
    # Remove currency symbols and commas
    clean_val = re.sub(r'[^\d.-]', '', str_val)
    
    if not clean_val:
        return Decimal(0)
        
    return Decimal(clean_val)


def validate_financial_math(data: Dict, sample: GoldSample) -> bool:
    """
    Validate that financial calculations are correct.
    Returns True if all calculations match.
    """
    all_valid = True
    
    try:
        # Extract financial values using robust cleaner
        subtotal = clean_currency(data.get('subtotal', 0))
        tax_amount = clean_currency(data.get('tax', data.get('tax_amount', 0)))
        total = clean_currency(data.get('total', data.get('total_amount', 0)))
        discount = clean_currency(data.get('discount', data.get('total_discount', 0)))
        tip = clean_currency(data.get('tip_amount', 0))
        
        # Calculate expected values from line items
        line_items_total = Decimal('0')
        items = data.get('items', data.get('line_items', []))
        
        for item in items:
            qty = clean_currency(item.get('quantity', 0))
            price = clean_currency(item.get('unit_price', 0))
            # Some templates use 'amount', some 'total'
            item_total = clean_currency(item.get('amount', item.get('total', 0)))
            
            expected_item_total = qty * price
            if abs(item_total - expected_item_total) > Decimal('0.05'):
                sample.issues.append(f"Item math mismatch: {qty}x{price} != {item_total}")
                all_valid = False
            
            line_items_total += item_total
        
        # Check 1: Subtotal matches line items sum
        if abs(subtotal - line_items_total) < Decimal('0.05'):
            sample.subtotal_matches = True
        else:
            sample.issues.append(
                f"Subtotal mismatch: Doc shows ${subtotal}, "
                f"line items sum to ${line_items_total}"
            )
            all_valid = False
        
        # Check 2: Tax calculation is reasonable
        if tax_amount > 0:
            tax_rate = (tax_amount / subtotal) * 100 if subtotal > 0 else 0
            if Decimal('0') <= tax_rate <= Decimal('25'):  # Reasonable tax range
                sample.tax_calculation_matches = True
            else:
                sample.warnings.append(
                    f"Unusual tax rate: {tax_rate:.2f}% "
                    f"(${tax_amount} on ${subtotal})"
                )
        else:
            sample.tax_calculation_matches = True  # No tax is valid
        
        # Check 3: Total = Subtotal + Tax + Tip - Discount
        expected_total = subtotal + tax_amount + tip - discount
        if abs(total - expected_total) < Decimal('0.05'):
            sample.total_matches = True
        else:
            sample.issues.append(
                f"Total mismatch: Doc shows ${total}, "
                f"expected ${expected_total} "
                f"(${subtotal} + ${tax_amount} + ${tip} - ${discount})"
            )
            all_valid = False
        
        # Check 4: Discount is reasonable
        if discount > 0:
            discount_pct = (discount / subtotal) * 100 if subtotal > 0 else 0
            if discount_pct <= 50:  # Max 50% discount seems reasonable
                sample.discount_applied_correctly = True
            else:
                sample.warnings.append(
                    f"Unusually high discount: {discount_pct:.1f}% "
                    f"(${discount} on ${subtotal})"
                )
        else:
            sample.discount_applied_correctly = True  # No discount is valid
        
    except Exception as e:
        sample.issues.append(f"Financial validation error: {str(e)}")
        all_valid = False
    
    return all_valid


def validate_line_items(data: Dict, sample: GoldSample) -> bool:
    """
    Validate that line items are realistic.
    Returns True if all line items pass validation.
    """
    all_valid = True
    
    items = data.get('items', data.get('line_items', []))
    if not items:
        sample.issues.append("No line items found")
        return False
    
    for idx, item in enumerate(items):
        item_id = idx + 1
        
        # Check 1: Has description
        if not item.get('description'):
            sample.issues.append(f"Line item {item_id}: Missing description")
            all_valid = False
        
        # Check 2: Has quantity
        qty = clean_currency(item.get('quantity', 0))
        if qty <= 0:
            sample.issues.append(f"Line item {item_id}: Invalid quantity {qty}")
            all_valid = False
        elif qty > 100:
            sample.warnings.append(f"Line item {item_id}: High quantity {qty}")
        
        # Check 3: Has unit price
        unit_price = clean_currency(item.get('unit_price', 0))
        if unit_price <= 0:
            sample.issues.append(f"Line item {item_id}: Invalid unit price ${unit_price}")
            all_valid = False
        elif unit_price > 10000:
            sample.warnings.append(f"Line item {item_id}: Very expensive item ${unit_price}")
        
        # Check 4: Total = Quantity × Unit Price
        total = clean_currency(item.get('amount', item.get('total', 0)))
        expected_total = qty * unit_price
        if abs(total - expected_total) > Decimal('0.05'):
            sample.issues.append(
                f"Line item {item_id}: Total mismatch. "
                f"Shows ${total}, expected ${expected_total} "
                f"({qty} × ${unit_price})"
            )
            all_valid = False
    
    return all_valid


def validate_dates(data: Dict, sample: GoldSample) -> bool:
    """Validate that dates are realistic."""
    all_valid = True
    
    # Check invoice/receipt date
    date_str = data.get('invoice_date', data.get('date'))
    if date_str:
        try:
            # Try to parse date - handle multiple formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d']
            parsed_date = None
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if parsed_date:
                # Check if date is reasonable (not in future, not too old)
                now = datetime.now()
                if parsed_date > now:
                    sample.issues.append(f"Date in future: {date_str}")
                    all_valid = False
                elif (now - parsed_date).days > 365 * 5:  # More than 5 years old
                    sample.warnings.append(f"Date very old: {date_str}")
            else:
                sample.issues.append(f"Invalid date format: {date_str}")
                all_valid = False
                
        except Exception as e:
            sample.issues.append(f"Date validation error: {str(e)}")
            all_valid = False
    
    return all_valid


def validate_ocr_quality(sample: GoldSample) -> bool:
    """Check OCR for common issues."""
    all_valid = True
    
    # Check 1: No empty tokens
    if not sample.ocr_tokens:
        sample.issues.append("OCR returned no tokens")
        return False
    
    # Check 2: No excessive duplication
    token_counts = {}
    for token in sample.ocr_tokens:
        token_lower = token.lower()
        token_counts[token_lower] = token_counts.get(token_lower, 0) + 1
    
    for token, count in token_counts.items():
        if count > 10 and len(token) > 3:  # Same word appears 10+ times
            sample.warnings.append(f"Possible OCR duplication: '{token}' appears {count} times")
    
    # Check 3: Tokens and bboxes match
    if len(sample.ocr_tokens) != len(sample.ocr_bboxes):
        sample.issues.append(
            f"Token/bbox mismatch: {len(sample.ocr_tokens)} tokens, "
            f"{len(sample.ocr_bboxes)} bboxes"
        )
        all_valid = False
    
    # Check 4: Bboxes are valid
    for idx, bbox in enumerate(sample.ocr_bboxes):
        if len(bbox) != 4:
            sample.issues.append(f"Invalid bbox at index {idx}: {bbox}")
            all_valid = False
            continue
        
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            sample.issues.append(
                f"Invalid bbox dimensions at index {idx}: "
                f"[{x_min}, {y_min}, {x_max}, {y_max}]"
            )
            all_valid = False
    
    return all_valid


def validate_annotations(sample: GoldSample, annotator: TokenAnnotator) -> bool:
    """Validate that annotations make sense."""
    all_valid = True
    
    # Check 1: All tokens have labels
    if len(sample.ner_tags) != len(sample.ocr_tokens):
        sample.issues.append(
            f"Label count mismatch: {len(sample.ner_tags)} labels, "
            f"{len(sample.ocr_tokens)} tokens"
        )
        return False
    
    # Convert tag IDs to labels
    sample.ner_labels = []
    for tag_id in sample.ner_tags:
        label = annotator.id2label.get(tag_id, 'UNKNOWN')
        sample.ner_labels.append(label)
    
    # Check 2: No orphan I- tags
    for idx, label in enumerate(sample.ner_labels):
        if label.startswith('I-'):
            if idx == 0 or not sample.ner_labels[idx-1].endswith(label[2:]):
                sample.issues.append(
                    f"Orphan I- tag at token {idx}: {sample.ocr_tokens[idx]} → {label}"
                )
                all_valid = False
    
    # Check 3: Check for reasonable entity distribution
    entity_counts = {}
    for label in sample.ner_labels:
        if label.startswith('B-'):
            entity = label[2:]
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
    
    # Should have at least some basic entities
    critical_entities = ['INVOICE_DATE', 'TOTAL_AMOUNT', 'SUBTOTAL']
    for entity in critical_entities:
        if entity not in entity_counts:
            sample.warnings.append(f"Missing critical entity: {entity}")
    
    return all_valid


def generate_gold_sample(
    sample_id: int,
    retail_gen: RetailDataGenerator,
    invoice_gen: ModernInvoiceGenerator,
    visual_gen: VisualAssetGenerator,
    html_renderer: HTMLToPNGRenderer,
    receipt_renderer: SimplePNGRenderer,
    ocr_engine: OCREngine,
    annotator: TokenAnnotator,
    output_dir: Path,
    env: Environment
) -> GoldSample:
    """Generate a single gold sample with all validation data."""
    
    import random
    
    # 50/50 split between receipt and invoice
    is_invoice = random.random() < 0.5
    sample_type = 'invoice' if is_invoice else 'receipt'
    
    sample = GoldSample(f"gold_{sample_id:03d}", sample_type)
    
    try:
        if is_invoice:
            # Generate Invoice
            data = invoice_gen.generate_modern_invoice()
            
            # Templates
            invoice_templates = [
                "modern_professional/invoice_ecommerce.html",
                "modern_professional/invoice_minimal.html",
                "modern_professional/invoice_a4.html"
            ]
            template_name = random.choice(invoice_templates)
            
            # Page settings
            page_size = 'A4'
            orientation = 'Portrait'
            custom_width = None
            
            # Check assets
            if data.get('logo'): sample.has_logo = True
            if data.get('qr_code'): sample.has_qr = True
            use_receipt_renderer = False  # Invoices use Jinja2 templates
            
        else:
            # Generate Receipt
            store_types = [
                'fashion', 'accessories', 'jewelry', 'beauty', 
                'home_garden', 'sports_fitness', 'pet_supplies', 
                'books_media', 'toys_games', 'food_beverage',
                'health_wellness', 'electronics'
            ]
            store_type = random.choice(store_types)
            
            # Size distribution
            rand_val = random.random()
            if rand_val < 0.2:
                min_items, max_items = 25, 45 # Large
            elif rand_val < 0.3:
                min_items, max_items = 12, 20 # Medium
            else:
                min_items, max_items = 3, 8   # Normal
            
            receipt_obj = retail_gen.generate_pos_receipt(
                store_type=store_type,
                min_items=min_items,
                max_items=max_items
            )
            data = retail_gen.to_dict(receipt_obj)
            data['store_category'] = store_type
            
            # Ensure barcode
            if not data.get('barcode_image') and data.get('barcode_value'):
                data['barcode_image'] = visual_gen.generate_barcode(data['barcode_value'])
            
            if data.get('barcode_image'): sample.has_barcode = True
            
            # Templates
            receipt_templates = [
                "retail/pos_receipt.html",
                "retail/pos_receipt_dense.html"
            ]
            template_name = random.choice(receipt_templates)
            
            # Page settings - receipts use render_receipt_with_data for multipage support
            page_size = None
            orientation = None
            custom_width = 576
            use_receipt_renderer = True  # Flag to use multipage-aware renderer
            
        sample.data_dict = data
        sample.data_dict['id'] = sample.sample_id
        sample.template_name = template_name
        
        # Render to PNG
        image_path = output_dir / 'images' / f"{sample.sample_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not is_invoice and use_receipt_renderer:
            # Use SimplePNGRenderer for receipts - supports multipage
            success = receipt_renderer.render_receipt_dict(data, str(image_path))
        else:
            # Use HTMLToPNGRenderer for invoices (with Jinja2 templates)
            template = env.get_template(template_name)
            html = template.render(**data)
            
            # Embed CSS
            template_dir = project_root / "templates" / Path(template_name).parent
            def replace_css_link(match):
                href = match.group(1)
                css_path = (template_dir / href).resolve()
                if css_path.exists():
                    try:
                        with open(css_path, 'r', encoding='utf-8') as f:
                            css_content = f.read()
                        return f'<style>\n{css_content}\n</style>'
                    except Exception:
                        return match.group(0)
                return match.group(0)
            
            html = re.sub(r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*>', replace_css_link, html)
            
            success = html_renderer.render(
                html, str(image_path),
                page_size=page_size,
                orientation=orientation,
                custom_width=custom_width
            )
        
        if not success:
            sample.issues.append("Failed to render image")
            return sample
        
        # Check if multipage output was created (render_receipt_dict creates _page1.png etc.)
        multipage_marker = image_path.parent / f"{sample.sample_id}_MULTIPAGE.txt"
        if multipage_marker.exists():
            # Use the first page for OCR
            sample.image_path = str(image_path.parent / f"{sample.sample_id}_page1.png")
        else:
            sample.image_path = str(image_path)
        
        # Verify image exists
        if not Path(sample.image_path).exists():
            sample.issues.append(f"Image not found: {sample.image_path}")
            return sample
        
        # Run OCR
        bbox_list = ocr_engine.extract_text(sample.image_path)
        if not bbox_list:
            sample.issues.append("OCR returned no results")
            return sample
        
        # Extract tokens and bboxes
        for bbox_obj in bbox_list:
            if bbox_obj.confidence < 0.5:
                continue
            
            bbox = bbox_obj.to_pascal_voc()
            words = bbox_obj.text.split()
            
            for word in words:
                sample.ocr_tokens.append(word)
                sample.ocr_bboxes.append(bbox)
                sample.ocr_confidence.append(bbox_obj.confidence)
        
        if not sample.ocr_tokens:
            sample.issues.append("No valid OCR tokens after filtering")
            return sample
        
        # Annotate
        annotation = annotator.annotate_tokens(
            sample.data_dict,
            sample.ocr_tokens,
            sample.ocr_bboxes,
            sample.image_path,
            800,
            1200
        )
        
        if not annotation:
            sample.issues.append("Annotation failed")
            return sample
        
        sample.ner_tags = annotation['ner_tags']
        
        # Run validations
        validate_financial_math(sample.data_dict, sample)
        validate_line_items(sample.data_dict, sample)
        validate_dates(sample.data_dict, sample)
        validate_ocr_quality(sample)
        validate_annotations(sample, annotator)
        
        # Save data as JSON
        json_path = output_dir / 'data' / f"{sample.sample_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample.data_dict, f, indent=2, default=str)
        
        # Save annotation data
        annotation_path = output_dir / 'annotations' / f"{sample.sample_id}.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        
        annotation_data = {
            'sample_id': sample.sample_id,
            'type': sample.type,
            'tokens': sample.ocr_tokens,
            'bboxes': sample.ocr_bboxes,
            'ner_tags': sample.ner_tags,
            'ner_labels': sample.ner_labels,
            'confidence': sample.ocr_confidence,
        }
        
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2)
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        sample.issues.append(f"Generation error: {str(e)}")
        print(f"      DEBUG - Full error: {tb[:500]}")
    
    return sample


def format_currency(value) -> str:
    """Convert value to currency string, handling already-formatted values."""
    if isinstance(value, str):
        # Strip existing $ and convert
        value = value.replace('$', '').replace(',', '').strip()
    try:
        return f"${float(value):.2f}"
    except (ValueError, TypeError):
        return "$0.00"


def generate_verification_report(
    samples: List[GoldSample],
    output_dir: Path
):
    """Generate HTML report for manual verification."""
    
    html_parts = []
    
    # Header
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Gold Sample Verification Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        .summary {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .sample {
            background: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .sample-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #eee;
        }
        .sample-title {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .status-pass {
            background: #10b981;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
        }
        .status-warn {
            background: #f59e0b;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
        }
        .status-fail {
            background: #ef4444;
            color: white;
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
        }
        .content-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .receipt-image {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .validation-section {
            margin-top: 20px;
        }
        .validation-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #555;
        }
        .validation-item {
            padding: 8px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            align-items: center;
        }
        .validation-pass {
            background: #d1fae5;
            color: #065f46;
        }
        .validation-warn {
            background: #fef3c7;
            color: #92400e;
        }
        .validation-fail {
            background: #fee2e2;
            color: #991b1b;
        }
        .icon {
            margin-right: 10px;
            font-weight: bold;
        }
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        .data-table th {
            background: #f3f4f6;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }
        .data-table td {
            padding: 8px;
            border-bottom: 1px solid #e5e7eb;
        }
        .token-display {
            font-family: 'Courier New', monospace;
            background: #f9fafb;
            padding: 10px;
            border-radius: 5px;
            max-height: 300px;
            overflow-y: auto;
            margin-top: 10px;
        }
        .token {
            display: inline-block;
            margin: 2px;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }
        .token-o {
            background: #e5e7eb;
        }
        .token-b {
            background: #dbeafe;
            border: 1px solid #3b82f6;
        }
        .token-i {
            background: #fce7f3;
            border: 1px solid #ec4899;
        }
        .instructions {
            background: #fef3c7;
            border-left: 4px solid #f59e0b;
            padding: 15px;
            margin-bottom: 30px;
            border-radius: 5px;
        }
        .instructions-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #92400e;
        }
        .checklist {
            list-style: none;
            padding-left: 0;
        }
        .checklist li {
            padding: 5px 0;
        }
        .checklist li:before {
            content: "☐ ";
            margin-right: 10px;
            font-size: 18px;
        }
        .page-navigator {
            margin-bottom: 15px;
            padding: 10px;
            background: #f3f4f6;
            border-radius: 5px;
        }
        .page-info {
            font-size: 14px;
            font-weight: 600;
            color: #4b5563;
            margin-bottom: 8px;
        }
        .page-buttons {
            display: flex;
            gap: 5px;
        }
        .page-btn {
            padding: 6px 12px;
            border: 1px solid #d1d5db;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .page-btn:hover {
            background: #e5e7eb;
        }
        .page-btn.active {
            background: #3b82f6;
            color: white;
            border-color: #3b82f6;
        }
    </style>
    <script>
        function showPage(sampleId, pageNum) {
            // Hide all pages for this sample
            const allPages = document.querySelectorAll('.page-img-' + sampleId);
            allPages.forEach(img => {
                img.style.display = 'none';
            });
            
            // Show selected page
            const selectedPage = document.querySelector('.page-img-' + sampleId + '[data-page="' + pageNum + '"]');
            if (selectedPage) {
                selectedPage.style.display = 'block';
            }
            
            // Update button states
            const buttons = document.querySelectorAll('#nav-' + sampleId + ' .page-btn');
            buttons.forEach((btn, index) => {
                if (index + 1 === pageNum) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
        }
    </script>
</head>
<body>
    <div class="header">
        <h1>Test 13: Gold Sample Hand-Verification</h1>
        <p>Manual QA required - Validate semantic correctness before training</p>
    </div>
    
    <div class="instructions">
        <div class="instructions-title">Manual Verification Checklist</div>
        <p>For each sample below, verify:</p>
        <ul class="checklist">
            <li>Numbers reconcile (subtotal + tax - discount = total)</li>
            <li>Line items are realistic (prices, quantities, descriptions)</li>
            <li>Discounts apply correctly</li>
            <li>OCR has no text duplication</li>
            <li>Bounding boxes align visually with text</li>
            <li>No phantom line items</li>
            <li>Tax rates are reasonable (3-15%)</li>
            <li>Dates are valid</li>
            <li>Currency formatting consistent</li>
            <li>Entity labels match visual content</li>
        </ul>
        <p><strong>If you find ANY significant failure, stop and fix the generator/labeler before training!</strong></p>
    </div>
""")
    
    # Summary statistics
    total_samples = len(samples)
    samples_with_issues = sum(1 for s in samples if s.issues)
    samples_with_warnings = sum(1 for s in samples if s.warnings and not s.issues)
    samples_clean = total_samples - samples_with_issues - samples_with_warnings
    
    html_parts.append(f"""
    <div class="summary">
        <h2>Summary Statistics</h2>
        <table class="data-table">
            <tr>
                <th>Metric</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            <tr>
                <td>Total Samples</td>
                <td>{total_samples}</td>
                <td>100%</td>
            </tr>
            <tr style="background: #d1fae5;">
                <td>✓ Clean (No Issues)</td>
                <td>{samples_clean}</td>
                <td>{samples_clean/total_samples*100:.1f}%</td>
            </tr>
            <tr style="background: #fef3c7;">
                <td>⚠ Warnings Only</td>
                <td>{samples_with_warnings}</td>
                <td>{samples_with_warnings/total_samples*100:.1f}%</td>
            </tr>
            <tr style="background: #fee2e2;">
                <td>✗ Issues Found</td>
                <td>{samples_with_issues}</td>
                <td>{samples_with_issues/total_samples*100:.1f}%</td>
            </tr>
        </table>
    </div>
""")
    
    # Individual samples
    for sample in samples:
        # Determine status
        if sample.issues:
            status_class = "status-fail"
            status_text = "ISSUES FOUND"
        elif sample.warnings:
            status_class = "status-warn"
            status_text = "WARNINGS"
        else:
            status_class = "status-pass"
            status_text = "PASS"
        
        # Determine image path for HTML (relative to report)
        from pathlib import Path
        if sample.image_path:
            img_filename = Path(sample.image_path).name
        else:
            img_filename = f"{sample.sample_id}.png"
        
        html_parts.append(f"""
    <div class="sample">
        <div class="sample-header">
            <div class="sample-title">{sample.sample_id}</div>
            <div class="{status_class}">{status_text}</div>
        </div>
        
        <div class="content-grid">
            <div>
                <h3>Receipt Image</h3>""")
        
        # Check if multi-page receipt
        multipage_marker = output_dir / "images" / f"{sample.sample_id}_MULTIPAGE.txt"
        if multipage_marker.exists():
            # Read page count
            with open(multipage_marker, 'r') as f:
                lines = f.readlines()
                page_count = int(lines[0].split()[3])
            
            html_parts.append(f"""
                <div class="page-navigator">
                    <div class="page-info">Multi-page invoice ({page_count} pages)</div>
                    <div class="page-buttons" id="nav-{sample.sample_id}">""")
            
            for page_num in range(1, page_count + 1):
                active_class = " active" if page_num == 1 else ""
                html_parts.append(f"""
                        <button class="page-btn{active_class}" onclick="showPage('{sample.sample_id}', {page_num})">{page_num}</button>""")
            
            html_parts.append("""
                    </div>
                </div>""")
            
            # Add all page images (hidden except first)
            for page_num in range(1, page_count + 1):
                page_filename = f"{sample.sample_id}_page{page_num}.png"
                display_style = "" if page_num == 1 else " style='display:none;'"
                html_parts.append(f"""
                <img src="images/{page_filename}" class="receipt-image page-img-{sample.sample_id}"{display_style} data-page="{page_num}" alt="{sample.sample_id} page {page_num}">""")
        else:
            # Single-page receipt
            html_parts.append(f"""
                <img src="images/{img_filename}" class="receipt-image" alt="{sample.sample_id}">""")
        
        html_parts.append("""
            </div>
            
            <div>
                <h3>Financial Summary</h3>
                <table class="data-table">
                    <tr>
                        <td>Subtotal</td>
                        <td>{subtotal}</td>
                        <td>{subtotal_check}</td>
                    </tr>
                    <tr>
                        <td>Tax</td>
                        <td>{tax}</td>
                        <td>{tax_check}</td>
                    </tr>
                    <tr>
                        <td>Discount</td>
                        <td>{discount}</td>
                        <td>{discount_check}</td>
                    </tr>
                    <tr style="font-weight: bold;">
                        <td>Total</td>
                        <td>{total}</td>
                        <td>{total_check}</td>
                    </tr>
                </table>
                
                <h3>Line Items ({item_count})</h3>
                <table class="data-table">
                    <tr>
                        <th>Item</th>
                        <th>Qty</th>
                        <th>Unit Price</th>
                        <th>Total</th>
                    </tr>
""".format(
            subtotal=format_currency(sample.data_dict.get('subtotal', 0)),
            subtotal_check='✓' if sample.subtotal_matches else '✗',
            tax=format_currency(sample.data_dict.get('tax', sample.data_dict.get('tax_amount', 0))),
            tax_check='✓' if sample.tax_calculation_matches else '⚠',
            discount=format_currency(sample.data_dict.get('discount', sample.data_dict.get('total_discount', 0))),
            discount_check='✓' if sample.discount_applied_correctly else '⚠',
            total=format_currency(sample.data_dict.get('total', sample.data_dict.get('total_amount', 0))),
            total_check='✓' if sample.total_matches else '✗',
            item_count=len(sample.data_dict.get('items', sample.data_dict.get('line_items', [])))
        ))
        
        items = sample.data_dict.get('items', sample.data_dict.get('line_items', []))
        for item in items[:5]:  # Show first 5
            html_parts.append(f"""
                    <tr>
                        <td>{item.get('description', 'N/A')[:30]}</td>
                        <td>{item.get('quantity', 0)}</td>
                        <td>{format_currency(item.get('unit_price', 0))}</td>
                        <td>{format_currency(item.get('amount', item.get('total', 0)))}</td>
                    </tr>
""")
        
        if len(items) > 5:
            html_parts.append(f"""
                    <tr>
                        <td colspan="4"><em>... and {len(items) - 5} more items</em></td>
                    </tr>
""")
        
        html_parts.append("""
                </table>
            </div>
        </div>
        
        <div class="validation-section">
            <div class="validation-title">Automated Validation Results</div>
""")
        
        # Show asset checks for invoices
        if sample.type == 'invoice':
             html_parts.append(f"""
            <div class="validation-item {'validation-pass' if sample.has_logo else 'validation-fail'}">
                <span class="icon">{'✓' if sample.has_logo else '✗'}</span>
                <span>Logo Present</span>
            </div>
            <div class="validation-item {'validation-pass' if sample.has_qr else 'validation-fail'}">
                <span class="icon">{'✓' if sample.has_qr else '✗'}</span>
                <span>QR Code Present</span>
            </div>
""")
        elif sample.type == 'receipt':
             html_parts.append(f"""
            <div class="validation-item {'validation-pass' if sample.has_barcode else 'validation-warn'}">
                <span class="icon">{'✓' if sample.has_barcode else '⚠'}</span>
                <span>Barcode Present</span>
            </div>
""")
        
        # Show issues
        if sample.issues:
            for issue in sample.issues:
                html_parts.append(f"""
            <div class="validation-item validation-fail">
                <span class="icon">✗</span>
                <span>{issue}</span>
            </div>
""")
        
        # Show warnings
        if sample.warnings:
            for warning in sample.warnings:
                html_parts.append(f"""
            <div class="validation-item validation-warn">
                <span class="icon">⚠</span>
                <span>{warning}</span>
            </div>
""")
        
        # Show passes
        if not sample.issues and not sample.warnings:
            html_parts.append("""
            <div class="validation-item validation-pass">
                <span class="icon">✓</span>
                <span>All automated checks passed</span>
            </div>
""")
        
        html_parts.append("""
        </div>
        
        <div class="validation-section">
            <div class="validation-title">OCR + Annotations</div>
            <div style="margin-bottom: 10px;">
                <strong>Tokens:</strong> {token_count} |
                <strong>Entities:</strong> {entity_count} |
                <strong>Avg Confidence:</strong> {avg_conf:.2f}
            </div>
            <div class="token-display">
""".format(
            token_count=len(sample.ocr_tokens),
            entity_count=len([l for l in sample.ner_labels if l.startswith('B-')]),
            avg_conf=sum(sample.ocr_confidence) / len(sample.ocr_confidence) if sample.ocr_confidence else 0
        ))
        
        # Show tokens with labels
        for token, label in zip(sample.ocr_tokens[:100], sample.ner_labels[:100]):  # First 100 tokens
            if label == 'O':
                css_class = 'token-o'
            elif label.startswith('B-'):
                css_class = 'token-b'
            else:
                css_class = 'token-i'
            
            html_parts.append(f"""
                <span class="token {css_class}" title="{label}">{token}</span>
""")
        
        if len(sample.ocr_tokens) > 100:
            html_parts.append(f"""
                <div style="margin-top: 10px;"><em>... and {len(sample.ocr_tokens) - 100} more tokens</em></div>
""")
        
        html_parts.append("""
            </div>
        </div>
    </div>
""")
    
    # Footer
    html_parts.append("""
    <div class="summary">
        <h2>Next Steps</h2>
        <ol>
            <li><strong>Review each sample above</strong> - Check visual rendering, OCR, and labels</li>
            <li><strong>Verify financial calculations</strong> - Manually calculate subtotal + tax - discount</li>
            <li><strong>Check entity labels</strong> - Do the highlighted entities match the visual content?</li>
            <li><strong>Look for anomalies</strong> - Unrealistic prices, wrong dates, phantom items</li>
            <li><strong>If ANY significant issue found</strong> - Fix generator/labeler and re-run this test</li>
            <li><strong>If all samples look good</strong> - Proceed to production training!</li>
        </ol>
    </div>
</body>
</html>
""")
    
    # Write report
    report_path = output_dir / 'verification_report.html'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Test 13: Gold Sample Hand-Verification"
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=20,
        help='Number of gold samples to generate (default: 20)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/gold_samples'),
        help='Output directory for gold samples'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default=Path('config/labels_retail.yaml'),
        help='Path to label schema YAML file'
    )
    
    args = parser.parse_args()
    
    # Validate schema
    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TEST 13: GOLD SAMPLE HAND-VERIFICATION (HUMAN QA)")
    print("="*80)
    print(f"Generating {args.num_samples} gold samples for manual inspection")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load schema
    with open(args.schema, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    
    # Initialize components
    print("Initializing components...")
    retail_gen = RetailDataGenerator()
    invoice_gen = ModernInvoiceGenerator()
    visual_gen = VisualAssetGenerator()
    
    # Use HTMLToPNGRenderer for invoices (Jinja2 templates)
    html_renderer = HTMLToPNGRenderer(augment_probability=0.0)
    # Use SimplePNGRenderer for receipts (supports multipage)
    receipt_renderer = SimplePNGRenderer(augment_probability=0.0)
    
    ocr_engine = OCREngine(engine='paddleocr')
    annotator = TokenAnnotator(schema)
    
    # Setup Jinja2
    env = Environment(loader=FileSystemLoader(str(project_root / "templates")))
    print()
    
    # Clean and create output directory (remove old samples to avoid stale data)
    if args.output_dir.exists():
        print("Cleaning previous gold samples...")
        for subdir in ['images', 'data', 'annotations', 'receipts']:
            subdir_path = args.output_dir / subdir
            if subdir_path.exists():
                shutil.rmtree(subdir_path)
        # Also remove old report
        old_report = args.output_dir / 'verification_report.html'
        if old_report.exists():
            old_report.unlink()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate samples
    print("Generating gold samples...")
    samples = []
    
    for i in range(args.num_samples):
        print(f"  Sample {i+1}/{args.num_samples}...", end='', flush=True)
        
        sample = generate_gold_sample(
            i + 1,
            retail_gen,
            invoice_gen,
            visual_gen,
            html_renderer,
            receipt_renderer,
            ocr_engine,
            annotator,
            args.output_dir,
            env
        )
        
        samples.append(sample)
        
        # Show status
        if sample.issues:
            print(" [X] ISSUES")
            if i < 2:  # Print first 2 for debugging
                for issue in sample.issues[:3]:
                    print(f"      - {issue}")
        elif sample.warnings:
            print(" [!] WARNINGS")
            if i < 3:  # Print first 3 warnings for debugging
                for warning in sample.warnings[:2]:
                    print(f"      - {warning}")
        else:
            print(" [OK] PASS")
    
    print()
    
    # Generate verification report
    print("Generating verification report...")
    report_path = generate_verification_report(samples, args.output_dir)
    print(f"  Report saved: {report_path}")
    print()
    
    # Print summary
    print("="*80)
    print("GOLD SAMPLE GENERATION COMPLETE")
    print("="*80)
    print()
    
    total = len(samples)
    with_issues = sum(1 for s in samples if s.issues)
    with_warnings = sum(1 for s in samples if s.warnings and not s.issues)
    clean = total - with_issues - with_warnings
    
    print(f"Summary:")
    print(f"  Total samples:     {total}")
    print(f"  [OK] Clean:        {clean} ({clean/total*100:.1f}%)")
    print(f"  [!] Warnings:      {with_warnings} ({with_warnings/total*100:.1f}%)")
    print(f"  [X] Issues:        {with_issues} ({with_issues/total*100:.1f}%)")
    print()
    
    print("Next Steps:")
    print(f"  1. Open the report: {report_path}")
    print(f"  2. Manually verify each sample")
    print(f"  3. Check for semantic correctness:")
    print(f"     - Financial calculations")
    print(f"     - Realistic line items")
    print(f"     - Correct entity labels")
    print(f"     - Valid OCR and bboxes")
    print(f"  4. If ANY significant issue found → Fix and re-run")
    print(f"  5. If all samples look good → Proceed to training!")
    print()
    
    if with_issues > 0:
        print("[!] WARNING: Some samples have issues. Review carefully!")
        print("="*80)
        sys.exit(1)
    elif with_warnings > 0:
        print("[!] Note: Some samples have warnings. Review recommended.")
        print("="*80)
        sys.exit(0)
    else:
        print("[OK] All automated checks passed. Manual review still required!")
        print("="*80)
        sys.exit(0)


if __name__ == '__main__':
    main()
