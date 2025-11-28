#!/usr/bin/env python3
"""
Test 17: Phase 2 Comprehensive Verification (Mixed Dataset)

Purpose: Validate the complete Phase 2 pipeline including:
         1. Visual Asset Generation (Logos, QR Codes)
         2. New Modern Templates (E-commerce, Minimal)
         3. Mixed Dataset Generation (Receipts + Invoices)
         4. HTML Rendering for all types

This test generates a mixed batch of samples and performs:
- Visual verification (HTML report)
- Financial validation
- Asset presence check (Logos/QR)
- Template usage verification

Usage:
    python scripts/test_17_phase2_verification.py
    python scripts/test_17_phase2_verification.py --num-samples 20
"""

import argparse
import sys
import json
import random
import os
import re
from pathlib import Path
from typing import Dict, List, Any
from decimal import Decimal
from datetime import datetime
from tqdm import tqdm
from jinja2 import Environment, FileSystemLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator
from generators.modern_invoice_generator import ModernInvoiceGenerator
from generators.html_to_png_renderer import HTMLToPNGRenderer
from generators.visual_assets import VisualAssetGenerator

class VerificationSample:
    """Container for a verification sample."""
    
    def __init__(self, sample_id: str, sample_type: str):
        self.sample_id = sample_id
        self.type = sample_type  # 'receipt' or 'invoice'
        self.template = ""
        self.data: Dict = {}
        self.image_path: str = ""
        
        # Validation flags
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.has_logo = False
        self.has_qr = False
        self.financials_valid = False

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

def validate_financials(sample: VerificationSample) -> bool:
    """Validate financial calculations."""
    try:
        data = sample.data
        
        # Normalize field names
        subtotal = clean_currency(data.get('subtotal', 0))
        tax = clean_currency(data.get('tax', data.get('tax_amount', 0)))
        total = clean_currency(data.get('total', data.get('total_amount', 0)))
        discount = clean_currency(data.get('discount', data.get('total_discount', 0)))
        
        # Check items
        items = data.get('items', data.get('line_items', []))
        calculated_subtotal = Decimal(0)
        
        for item in items:
            qty = clean_currency(item.get('quantity', 0))
            price = clean_currency(item.get('unit_price', 0))
            # Some templates use 'amount', some 'total'
            item_total = clean_currency(item.get('amount', item.get('total', 0)))
            
            expected_item_total = qty * price
            if abs(item_total - expected_item_total) > Decimal('0.05'):
                sample.issues.append(f"Item math mismatch: {qty}x{price} != {item_total}")
            
            calculated_subtotal += item_total
            
        # Verify subtotal
        if abs(subtotal - calculated_subtotal) > Decimal('0.05'):
            sample.issues.append(f"Subtotal mismatch: {subtotal} vs calc {calculated_subtotal}")
            
        # Verify total
        expected_total = subtotal + tax - discount
        if abs(total - expected_total) > Decimal('0.05'):
            sample.issues.append(f"Total mismatch: {total} vs calc {expected_total}")
            return False
            
        sample.financials_valid = True
        return True
        
    except Exception as e:
        sample.issues.append(f"Financial validation error: {str(e)}")
        return False

def validate_assets(sample: VerificationSample):
    """Check for presence of visual assets in modern invoices."""
    if sample.type == 'invoice':
        # Check data for logo/qr keys
        if sample.data.get('logo'):
            sample.has_logo = True
        else:
            sample.warnings.append("Missing Logo in invoice data")
            
        if sample.data.get('qr_code'):
            sample.has_qr = True
        else:
            sample.warnings.append("Missing QR Code in invoice data")

def generate_report(samples: List[VerificationSample], output_dir: Path):
    """Generate HTML report."""
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 2 Verification Report</title>
        <style>
            body { font-family: sans-serif; margin: 20px; background: #f0f2f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px 8px 0 0; }
            .status-pass { color: #27ae60; font-weight: bold; }
            .status-fail { color: #c0392b; font-weight: bold; }
            .status-warn { color: #f39c12; font-weight: bold; }
            .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            img { max-width: 100%; border: 1px solid #ddd; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
            .badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
            .badge-invoice { background: #e1f5fe; color: #0288d1; }
            .badge-receipt { background: #f3e5f5; color: #7b1fa2; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card header">
                <h1>Phase 2 Verification Report</h1>
                <p>Mixed Dataset & Visual Assets Validation</p>
            </div>
    """
    
    # Summary
    total = len(samples)
    invoices = sum(1 for s in samples if s.type == 'invoice')
    receipts = sum(1 for s in samples if s.type == 'receipt')
    issues = sum(1 for s in samples if s.issues)
    
    html += f"""
            <div class="card">
                <h2>Summary</h2>
                <p>Total Samples: <strong>{total}</strong></p>
                <p>Distribution: {receipts} Receipts ({receipts/total*100:.1f}%) / {invoices} Invoices ({invoices/total*100:.1f}%)</p>
                <p>Status: {total-issues} Passed, <span class="{ 'status-fail' if issues > 0 else 'status-pass' }">{issues} Failed</span></p>
            </div>
    """
    
    for sample in samples:
        status_class = "status-fail" if sample.issues else ("status-warn" if sample.warnings else "status-pass")
        status_text = "FAIL" if sample.issues else ("WARN" if sample.warnings else "PASS")
        
        html += f"""
            <div class="card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h3>{sample.sample_id} <span class="badge badge-{sample.type}">{sample.type.upper()}</span></h3>
                    <span class="{status_class}">{status_text}</span>
                </div>
                <p>Template: <code>{sample.template}</code></p>
                
                <div class="grid">
                    <div>
                        <img src="images/{Path(sample.image_path).name}" alt="Sample Image">
                    </div>
                    <div>
                        <h4>Validation Details</h4>
                        <ul>
        """
        
        if sample.type == 'invoice':
            html += f"<li>Logo Present: {'✅' if sample.has_logo else '❌'}</li>"
            html += f"<li>QR Code Present: {'✅' if sample.has_qr else '❌'}</li>"
            
        html += f"<li>Financials Valid: {'✅' if sample.financials_valid else '❌'}</li>"
        
        for issue in sample.issues:
            html += f"<li style='color:red'>❌ {issue}</li>"
        for warn in sample.warnings:
            html += f"<li style='color:orange'>⚠️ {warn}</li>"
            
        html += """
                        </ul>
                        <h4>Data Preview</h4>
                        <pre style="background:#f8f9fa; padding:10px; overflow:auto; max-height:200px;">
        """
        html += json.dumps(sample.data, indent=2, default=str)
        html += """
                        </pre>
                    </div>
                </div>
            </div>
        """
        
    html += """
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / "report.html", "w", encoding="utf-8") as f:
        f.write(html)
    
    return output_dir / "report.html"

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Verification")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/phase2_verification")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting Phase 2 Verification with {args.num_samples} samples...")
    
    # Initialize generators
    retail_gen = RetailDataGenerator()
    invoice_gen = ModernInvoiceGenerator()
    visual_gen = VisualAssetGenerator()
    renderer = HTMLToPNGRenderer(augment_probability=0.0) # No augmentation for verification clarity
    
    # Setup Jinja2
    env = Environment(loader=FileSystemLoader(str(project_root / "templates")))
    
    samples = []
    
    # Templates
    receipt_templates = [
        "retail/pos_receipt.html",
        "retail/pos_receipt_dense.html"
    ]
    invoice_templates = [
        "modern_professional/invoice_ecommerce.html",
        "modern_professional/invoice_minimal.html",
        "modern_professional/invoice_a4.html"
    ]
    
    for i in tqdm(range(args.num_samples)):
        # 60/40 split
        is_invoice = random.random() < 0.4
        sample_type = 'invoice' if is_invoice else 'receipt'
        sample_id = f"sample_{i:03d}"
        
        sample = VerificationSample(sample_id, sample_type)
        
        try:
            if is_invoice:
                # Generate Invoice
                data = invoice_gen.generate_modern_invoice()
                template_name = random.choice(invoice_templates)
                
                # Page settings
                page_size = 'A4'
                orientation = 'Portrait'
                
            else:
                # Generate Receipt
                data_obj = retail_gen.generate_pos_receipt()
                data = retail_gen.to_dict(data_obj)
                
                # Ensure barcode image is present
                if not data.get('barcode_image') and data.get('barcode_value'):
                    data['barcode_image'] = visual_gen.generate_barcode(data['barcode_value'])
                
                template_name = random.choice(receipt_templates)
                
                # Page settings for receipt
                page_size = None # Custom width
                orientation = None
                
            sample.data = data
            sample.template = template_name
            
            # Render
            template = env.get_template(template_name)
            html = template.render(**data)
            
            # Embed CSS directly to avoid wkhtmltoimage path issues
            template_dir = project_root / "templates" / Path(template_name).parent
            
            def replace_css_link(match):
                href = match.group(1)
                # Resolve path relative to template directory
                css_path = (template_dir / href).resolve()
                if css_path.exists():
                    try:
                        with open(css_path, 'r', encoding='utf-8') as f:
                            css_content = f.read()
                        return f'<style>\n{css_content}\n</style>'
                    except Exception as e:
                        print(f"Warning: Could not read CSS {css_path}: {e}")
                        return match.group(0)
                return match.group(0)
            
            html = re.sub(r'<link\s+rel="stylesheet"\s+href="([^"]+)"\s*>', replace_css_link, html)
            
            output_path = images_dir / f"{sample_id}.png"
            
            if is_invoice:
                success = renderer.render(
                    html, str(output_path), 
                    page_size=page_size, 
                    orientation=orientation
                )
            else:
                success = renderer.render(
                    html, str(output_path),
                    custom_width=576 # Receipt width
                )
                
            if success:
                sample.image_path = str(output_path)
            else:
                sample.issues.append("Rendering failed")
                
            # Validate
            validate_financials(sample)
            validate_assets(sample)
            
        except Exception as e:
            sample.issues.append(f"Generation error: {str(e)}")
            import traceback
            traceback.print_exc()
            
        samples.append(sample)
        
    # Generate Report
    report_path = generate_report(samples, output_dir)
    print(f"\nVerification complete. Report saved to: {report_path}")
    
    # Check results
    failed = sum(1 for s in samples if s.issues)
    if failed > 0:
        print(f"WARNING: {failed} samples had issues.")
        sys.exit(1)
    else:
        print("SUCCESS: All samples passed verification.")
        sys.exit(0)

if __name__ == "__main__":
    main()
