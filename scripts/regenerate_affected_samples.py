#!/usr/bin/env python3
"""
Regenerate samples for affected categories.

Based on validation analysis, ~12% of samples may have had issues from
the field mapping bug. This script regenerates samples specifically for
the affected categories to supplement the existing dataset.

Affected categories:
- online_orders (8% weight, 100% affected)
- modern_professional (8% weight, 100% affected)  
- ecommerce templates NOT in ECOMMERCE_TEMPLATE_KEYS (~26% of ecommerce)

Run on vast.ai after pulling the fixed code from GitHub.
"""

import os
import sys
import random
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def compress_image(image_path: str, quality: int = 85, format: str = 'JPEG') -> str:
    """Compress image to reduce file size"""
    from PIL import Image
    import cv2
    import numpy as np
    
    try:
        # Read with OpenCV for speed
        img = cv2.imread(image_path)
        if img is None:
            return image_path
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Save as JPEG with compression
        output_path = image_path.replace('.png', '.jpg')
        pil_img.save(output_path, format, quality=quality, optimize=True)
        
        # Remove original PNG
        if output_path != image_path and os.path.exists(image_path):
            os.remove(image_path)
        
        return output_path
    except Exception as e:
        print(f"Compression error: {e}")
        return image_path


def init_worker(seed_offset):
    """Initialize worker with unique random seed"""
    import random
    import numpy as np
    worker_seed = int(time.time() * 1000) % (2**32) + os.getpid() + seed_offset
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))


def generate_single_invoice(args):
    """Generate a single invoice - worker function"""
    idx, template, category, output_dir, templates_dir, augment_prob = args
    
    try:
        from generators.modern_invoice_generator import ModernInvoiceGenerator
        from generators.renderer import InvoiceRenderer
        
        invoice_gen = ModernInvoiceGenerator()
        renderer = InvoiceRenderer(
            templates_dir=templates_dir,
            output_dir=output_dir,
            augment_probability=augment_prob
        )
        
        # Determine item counts
        min_items, max_items = 3, 12
        if 'compact' in template or 'minimal' in template:
            min_items, max_items = 3, 15
        
        # Generate data - to_dict() now has proper field aliases!
        invoice = invoice_gen.generate_invoice(min_items=min_items, max_items=max_items)
        data_dict = invoice_gen.to_dict(invoice)
        
        invoice_id = f"regen_{idx:05d}"
        
        result = renderer.render_invoice(
            template_name=template,
            data=data_dict,
            invoice_id=invoice_id,
            formats=['png'],
            use_multipage=True
        )
        
        if result['png']:
            # Compress all generated images
            compressed_files = []
            for png_path in result['png']:
                compressed_path = compress_image(png_path, quality=85, format='JPEG')
                compressed_files.append(compressed_path)
            
            inv_num = data_dict.get('invoice_number', 'N/A')
            inv_date = data_dict.get('invoice_date', 'N/A')
            inv_total = data_dict.get('total', data_dict.get('total_amount', 0))
            
            metadata = {
                "filename": Path(compressed_files[0]).name,
                "type": "invoice",
                "template": template,
                "category": category,
                "invoice_number": inv_num,
                "date": inv_date,
                "total": inv_total,
                "pages": result.get('pages', 1),
                "regenerated": True
            }
            return ('success', idx, metadata)
        return ('failed', idx, None)
        
    except Exception as e:
        import traceback
        return ('error', idx, f"{str(e)}\n{traceback.format_exc()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Regenerate affected samples')
    parser.add_argument('--samples', type=int, default=18000,
                       help='Number of samples to regenerate')
    parser.add_argument('--output', type=str, default='data/production_150k_regen',
                       help='Output directory')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers')
    parser.add_argument('--augment', type=float, default=0.5,
                       help='Augmentation probability')
    args = parser.parse_args()
    
    output_path = Path(args.output)
    images_dir = output_path / "images"
    metadata_dir = output_path / "metadata"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    templates_dir = str(Path(__file__).parent.parent / "templates")
    
    # AFFECTED CATEGORIES - these were not properly enriched
    # These use ModernInvoiceGenerator but didn't get the supply_chain enrichment
    affected_templates = {
        'online_orders': [
            "retail/blue_wave_invoice.html",
            "retail/consumer_service_invoice.html",
            "retail/online_order_digital.html",
            "retail/online_order_fashion.html",
            "retail/online_order_grocery.html",
            "retail/online_order_home_improvement.html",
            "retail/online_order_invoice.html",
            "retail/online_order_marketplace.html",
            "retail/online_order_wholesale.html",
        ],
        'modern_professional': [
            "modern_professional/invoice_minimal.html",
            "modern_professional/invoice_minimal_multipage.html",
            "modern_professional/invoice_compact.html",
            "modern_professional/invoice_compact_multipage.html",
            "modern_professional/invoice_elegant.html",
            "modern_professional/invoice_bold.html",
            "modern_professional/invoice_sidebar.html",
            "modern_professional/invoice_landscape.html",
            "modern_professional/invoice_ecommerce.html",
            "modern_professional/invoice_a4.html",
        ],
        'ecommerce_not_in_keys': [
            # These ecommerce templates were NOT in ECOMMERCE_TEMPLATE_KEYS
            # so they used ModernInvoiceGenerator without proper field aliases
            "ecommerce/bigcommerce_standard.html",
            "ecommerce/candytoday_shopify_invoice.html",
            "ecommerce/circlespace_order.html",
            "ecommerce/dingdong_shop_invoice.html",
            "ecommerce/it_supplier_invoice.html",
            "ecommerce/jungle_business_invoice.html",
            "ecommerce/jungle_business_uk_invoice.html",
            "ecommerce/magento_standard.html",
            "ecommerce/shine_invoice.html",
            "ecommerce/waziexpress_invoice.html",
            "ecommerce/waziexpress_invoice_v2.html",
            "ecommerce/woocommerce_standard.html",
            "ecommerce/zemu_invoice.html",
            "ecommerce/zylker_invoice.html",
        ],
    }
    
    # Category weights for regeneration (proportional to original)
    category_weights = {
        'online_orders': 0.33,        # 8/24 = ~33%
        'modern_professional': 0.33,  # 8/24 = ~33%
        'ecommerce_not_in_keys': 0.34 # Remaining
    }
    
    # Prepare tasks
    print(f"\nPreparing {args.samples:,} regeneration tasks...")
    tasks = []
    categories = list(category_weights.keys())
    weights = list(category_weights.values())
    
    for i in range(args.samples):
        category = random.choices(categories, weights=weights)[0]
        template = random.choice(affected_templates[category])
        tasks.append((i, template, category, str(images_dir), templates_dir, args.augment))
    
    # Process
    print(f"Generating {args.samples:,} samples with {num_workers} workers...")
    start_time = time.time()
    
    all_metadata = []
    success_count = 0
    error_count = 0
    
    with mp.Pool(num_workers, initializer=init_worker, initargs=(0,)) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_single_invoice, tasks, chunksize=10)):
            status, idx, data = result
            if status == 'success':
                all_metadata.append(data)
                success_count += 1
            else:
                error_count += 1
                if error_count <= 10:
                    print(f"Error at {idx}: {data}")
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (args.samples - i - 1) / rate / 60
                print(f"Progress: {i+1:,}/{args.samples:,} ({rate:.1f}/sec, ETA: {eta:.1f}min)")
    
    # Save metadata
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    print(f"Success: {success_count:,}, Errors: {error_count:,}")
    
    metadata_file = metadata_dir / "regenerated_metadata.jsonl"
    with open(metadata_file, 'w') as f:
        for meta in all_metadata:
            f.write(json.dumps(meta) + '\n')
    
    print(f"Metadata saved to: {metadata_file}")
    
    # Summary
    print(f"\n=== REGENERATION COMPLETE ===")
    print(f"Samples generated: {success_count:,}")
    print(f"Output directory: {output_path}")
    print(f"Rate: {success_count/elapsed:.1f} samples/sec")


if __name__ == "__main__":
    main()
