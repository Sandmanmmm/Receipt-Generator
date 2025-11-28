"""
Mixed Dataset Generator
Generates a dataset containing both retail receipts and modern invoices
Target distribution: 60% Receipts, 40% Modern Invoices
"""
import sys
import os
import random
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator
from generators.modern_invoice_generator import ModernInvoiceGenerator
from generators.html_to_png_renderer import HTMLToPNGRenderer
from jinja2 import Environment, FileSystemLoader

def generate_mixed_dataset(output_dir: str, num_samples: int = 100, augment: bool = True):
    """
    Generate a mixed dataset of receipts and invoices
    
    Args:
        output_dir: Directory to save output
        num_samples: Total number of samples to generate
        augment: Whether to apply augmentation (50% probability per sample)
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    metadata_dir = output_path / "metadata"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generators
    retail_gen = RetailDataGenerator()
    invoice_gen = ModernInvoiceGenerator()
    
    # Initialize renderer
    renderer = HTMLToPNGRenderer(augment_probability=0.5 if augment else 0.0)
    
    # Setup Jinja2 environment
    template_loader = FileSystemLoader(str(project_root / "templates"))
    env = Environment(loader=template_loader)
    
    # Calculate distribution
    num_receipts = int(num_samples * 0.6)
    num_invoices = num_samples - num_receipts
    
    print(f"Generating {num_samples} samples:")
    print(f"  - Receipts: {num_receipts} (60%)")
    print(f"  - Invoices: {num_invoices} (40%)")
    
    # Receipt templates
    receipt_templates = [
        "retail/pos_receipt.html",
        "retail/pos_receipt_dense.html",
        "retail/pos_receipt_premium.html",
        "retail/pos_receipt_qsr.html"
    ]
    
    # Invoice templates
    invoice_templates = [
        "modern_professional/invoice_a4.html",
        "modern_professional/invoice_landscape.html",
        "modern_professional/invoice_ecommerce.html",
        "modern_professional/invoice_minimal.html"
    ]
    
    generated_count = 0
    metadata_list = []
    
    # Generate Receipts
    print("\nGenerating Receipts...")
    for i in tqdm(range(num_receipts)):
        try:
            # Generate data
            data = retail_gen.generate_pos_receipt()
            data_dict = retail_gen.to_dict(data)
            
            # Select template
            template_name = random.choice(receipt_templates)
            template = env.get_template(template_name)
            
            # Render HTML
            html = template.render(**data_dict)
            
            # Output filename
            filename = f"receipt_{i:05d}.png"
            output_file = images_dir / filename
            
            # Render to PNG
            # Receipts are typically long strips, so we don't set page size
            success = renderer.render(
                html_content=html,
                output_path=str(output_file),
                custom_width=576, # Standard receipt width (80mm at 203dpi approx)
                apply_augmentation=None # Use renderer's probability
            )
            
            if success:
                metadata = {
                    "filename": filename,
                    "type": "receipt",
                    "template": template_name,
                    "invoice_number": data.invoice_number,
                    "date": data.invoice_date,
                    "total": data.total_amount
                }
                metadata_list.append(metadata)
                generated_count += 1
                
        except Exception as e:
            print(f"Error generating receipt {i}: {str(e)}")
            
    # Generate Invoices
    print("\nGenerating Invoices...")
    for i in tqdm(range(num_invoices)):
        try:
            # Generate data
            data_dict = invoice_gen.generate_modern_invoice()
            
            # Select template
            template_name = random.choice(invoice_templates)
            template = env.get_template(template_name)
            
            # Determine page settings based on template
            page_size = 'A4'
            orientation = 'Portrait'
            
            if 'landscape' in template_name:
                page_size = 'Letter'
                orientation = 'Landscape'
            elif 'a4' in template_name:
                page_size = 'A4'
            else:
                # Default for others
                page_size = 'A4'
            
            # Render HTML
            html = template.render(**data_dict)
            
            # Output filename
            filename = f"invoice_{i:05d}.png"
            output_file = images_dir / filename
            
            # Render to PNG
            success = renderer.render(
                html_content=html,
                output_path=str(output_file),
                page_size=page_size,
                orientation=orientation,
                apply_augmentation=None # Use renderer's probability
            )
            
            if success:
                metadata = {
                    "filename": filename,
                    "type": "invoice",
                    "template": template_name,
                    "invoice_number": data_dict['invoice_number'],
                    "date": data_dict['invoice_date'],
                    "total": data_dict['total']
                }
                metadata_list.append(metadata)
                generated_count += 1
                
        except Exception as e:
            print(f"Error generating invoice {i}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save metadata
    with open(metadata_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata_list, f, indent=2)
        
    print(f"\nDataset generation complete!")
    print(f"Total samples: {generated_count}/{num_samples}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mixed receipt/invoice dataset")
    parser.add_argument("--output", default="outputs/mixed_dataset", help="Output directory")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples")
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    
    args = parser.parse_args()
    
    generate_mixed_dataset(
        output_dir=args.output,
        num_samples=args.samples,
        augment=not args.no_augment
    )
