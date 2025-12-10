#!/usr/bin/env python3
"""
Parallel Mixed Dataset Generator
Generates a dataset using multiple CPU cores for massive speedup
Optimized for high-core-count servers (128-272 cores)

Features:
- Multi-process rendering with ProcessPoolExecutor
- JPEG compression to reduce file sizes (3.5MB → ~50KB)
- Configurable quality and format options
"""
import sys
import os
import random
import json
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
from functools import partial
import time
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compress_image(image_path: str, quality: int = 85, format: str = 'JPEG') -> bool:
    """
    Compress an image to reduce file size dramatically
    
    Args:
        image_path: Path to PNG image
        quality: JPEG quality (1-100, default 85)
        format: Output format ('JPEG' or 'PNG')
    
    Returns:
        True if successful
    """
    try:
        img = Image.open(image_path)
        
        # Convert RGBA to RGB (JPEG doesn't support transparency)
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img.close()
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        if format == 'JPEG':
            # Save as JPEG with compression
            jpeg_path = str(image_path).replace('.png', '.jpg')
            img.save(jpeg_path, 'JPEG', quality=quality, optimize=True)
            img.close()
            
            # Remove original PNG
            if os.path.exists(image_path) and image_path != jpeg_path:
                os.remove(image_path)
            
            return jpeg_path
        else:
            # Optimize PNG 
            img.save(image_path, 'PNG', optimize=True)
            img.close()
            return image_path
            
    except Exception as e:
        print(f"Warning: Compression failed for {image_path}: {e}")
        return image_path


def init_worker(seed_offset):
    """Initialize worker process with unique random seed"""
    # Each worker gets a unique seed based on PID + offset
    worker_seed = os.getpid() + seed_offset + int(time.time() * 1000) % 10000
    random.seed(worker_seed)
    
    # Set XDG_RUNTIME_DIR to suppress wkhtmltoimage warnings
    os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'
    os.makedirs('/tmp/runtime-root', exist_ok=True)


def generate_single_receipt(args):
    """Generate a single receipt - worker function"""
    idx, template, output_dir, templates_dir = args
    
    try:
        from generators.retail_data_generator import RetailDataGenerator
        from generators.renderer import InvoiceRenderer
        
        retail_gen = RetailDataGenerator()
        renderer = InvoiceRenderer(
            templates_dir=templates_dir,
            output_dir=output_dir
        )
        
        # Generate receipt data
        receipt = retail_gen.generate_pos_receipt(min_items=3, max_items=15)
        data_dict = retail_gen.to_dict(receipt)
        
        receipt_id = f"receipt_{idx:05d}"
        
        result = renderer.render_invoice(
            template_name=template,
            data=data_dict,
            invoice_id=receipt_id,
            formats=['png'],
            use_multipage=True
        )
        
        if result['png']:
            # Compress all generated images
            compressed_files = []
            for png_path in result['png']:
                compressed_path = compress_image(png_path, quality=85, format='JPEG')
                compressed_files.append(compressed_path)
            
            metadata = {
                "filename": Path(compressed_files[0]).name,
                "type": "receipt",
                "template": template,
                "category": "receipt",
                "receipt_number": data_dict.get('receipt_number', 'N/A'),
                "date": data_dict.get('date', 'N/A'),
                "total": data_dict.get('total', 0),
                "pages": result.get('pages', 1)
            }
            return ('success', idx, metadata)
        return ('failed', idx, None)
        
    except Exception as e:
        return ('error', idx, str(e))


def generate_single_invoice(args):
    """Generate a single invoice - worker function"""
    idx, template, category, output_dir, templates_dir = args
    
    try:
        from generators.retail_data_generator import RetailDataGenerator
        from generators.modern_invoice_generator import ModernInvoiceGenerator
        from generators.renderer import InvoiceRenderer
        
        # Import template key mapping
        ECOMMERCE_TEMPLATE_KEYS = {
            "ecommerce/ebay_invoice.html": "ebay_invoice",
            "ecommerce/amazon_business_invoice.html": "amazon_business_invoice",
            "ecommerce/amazon_seller_invoice.html": "amazon_seller_invoice",
            "ecommerce/etsy_invoice.html": "etsy_invoice",
            "ecommerce/faire_invoice.html": "faire_invoice",
            "ecommerce/alibaba_invoice.html": "alibaba_invoice",
            "ecommerce/marketplace_transactional.html": "marketplace_transactional",
            "ecommerce/amazon_invoice.html": "amazon_invoice",
            "ecommerce/shopify_standard.html": "shopify_standard",
            "ecommerce/shopify_modern_clean.html": "shopify_modern_clean",
            "ecommerce/shopify_modern_dark.html": "shopify_modern_dark",
            "ecommerce/shopify_luxe_classic.html": "shopify_luxe_classic",
            "ecommerce/stripe_minimal_invoice.html": "stripe_minimal_invoice",
            "ecommerce/payment_processor_invoice.html": "payment_processor_invoice",
            "ecommerce/modern_wholesale_pro.html": "modern_wholesale_pro",
            "ecommerce/modern_b2b_clean.html": "modern_b2b_clean",
            "ecommerce/nordic_minimalist_wholesale.html": "nordic_minimalist_wholesale",
            "ecommerce/premium_boutique_wholesale.html": "premium_boutique_wholesale",
            "ecommerce/accounting_pro_wholesale.html": "accounting_pro_wholesale",
            "ecommerce/marketplace_wholesale.html": "marketplace_wholesale",
            "ecommerce/enterprise_wholesale_bold.html": "enterprise_wholesale_bold",
            "ecommerce/digital_payment_wholesale.html": "digital_payment_wholesale",
            "ecommerce/modern_cloud_wholesale.html": "modern_cloud_wholesale",
            "ecommerce/wholesale_compact_order.html": "wholesale_compact_order",
            "ecommerce/digital_payment_pro.html": "digital_payment_pro",
            "ecommerce/cloud_accounting_invoice.html": "cloud_accounting_invoice",
            "ecommerce/saas_billing_invoice.html": "saas_billing_invoice",
            "ecommerce/modern_receipt_card.html": "modern_receipt_card",
            "ecommerce/shopify_premium_detailed.html": "shopify_premium_detailed",
            "ecommerce/tech_modern_gradient.html": "tech_modern_gradient",
            "ecommerce/costco_invoice.html": "costco_invoice",
            "ecommerce/walmart_invoice.html": "walmart_invoice",
            "ecommerce/enterprise_accounting_invoice.html": "enterprise_accounting_invoice",
            "ecommerce/logistics_pro_invoice.html": "logistics_pro_invoice",
            "ecommerce/aurora_glass_invoice.html": "aurora_glass_invoice",
            "ecommerce/boutique_luxe_invoice.html": "boutique_luxe_invoice",
            "ecommerce/premium_minimalist_invoice.html": "premium_minimalist_invoice",
            "ecommerce/nordic_statement_invoice.html": "nordic_statement_invoice",
            "ecommerce/amazon_style_order.html": "amazon_style_order",
            "ecommerce/tech_gradient_template.html": "tech_gradient_template"
        }
        
        retail_gen = RetailDataGenerator()
        invoice_gen = ModernInvoiceGenerator()
        renderer = InvoiceRenderer(
            templates_dir=templates_dir,
            output_dir=output_dir
        )
        
        # Determine item counts
        min_items, max_items = 3, 12
        if 'compact' in template or 'minimal' in template:
            min_items, max_items = 3, 15
        elif 'wholesale' in template or 'b2b' in template:
            min_items, max_items = 5, 20
        
        # Generate data based on template type
        if template in ECOMMERCE_TEMPLATE_KEYS:
            template_key = ECOMMERCE_TEMPLATE_KEYS[template]
            receipt = retail_gen.generate_for_template(
                template_name=template_key,
                min_items=min_items,
                max_items=max_items
            )
            data_dict = retail_gen.to_dict(receipt)
        elif category == 'purchase_orders':
            from generators.purchase_order_generator import PurchaseOrderGenerator
            po_gen = PurchaseOrderGenerator()
            data = po_gen.generate_purchase_order(
                po_type='generic',
                min_items=min_items,
                max_items=max_items
            )
            data_dict = po_gen.to_dict(data)
        else:
            # Modern invoice or other templates
            invoice = invoice_gen.generate_invoice(min_items=min_items, max_items=max_items)
            data_dict = invoice_gen.to_dict(invoice)
        
        # Add common fields for supply_chain/wholesale templates
        if category in ['supply_chain', 'wholesale', 'services']:
            company = data_dict.get('company_name', 'Acme Corp')
            data_dict['supplier_name'] = company
            data_dict['exporter'] = {
                'name': company,
                'address': data_dict.get('company_address', '123 Business Ave'),
                'city': data_dict.get('company_city', 'New York'),
                'country': 'USA'
            }
            data_dict['buyer_address'] = f"{data_dict.get('customer_address', '456 Client St')}, {data_dict.get('customer_city', 'Los Angeles')}, CA 90001"
            data_dict['seller_address'] = f"{data_dict.get('company_address', '123 Business Ave')}, {data_dict.get('company_city', 'New York')}, NY 10001"
            data_dict['supplier_address'] = data_dict['seller_address']
            data_dict['subtotal_raw'] = float(str(data_dict.get('subtotal', 0)).replace('$', '').replace(',', ''))
            data_dict['tax_raw'] = float(str(data_dict.get('tax', 0)).replace('$', '').replace(',', ''))
            data_dict['total_raw'] = float(str(data_dict.get('total', 0)).replace('$', '').replace(',', ''))
            data_dict['discount_raw'] = 0.0
            data_dict['total_value'] = data_dict.get('total_raw', random.uniform(500, 10000))
            data_dict['buyer_address_line1'] = data_dict.get('customer_address', '456 Client St')
            data_dict['buyer_address_line2'] = f"{data_dict.get('customer_city', 'Los Angeles')}, CA 90001"
        
        # Add supplier_name for ecommerce templates
        if 'ecommerce' in template:
            if 'supplier_name' not in data_dict:
                data_dict['supplier_name'] = data_dict.get('company_name', 'Supplier Inc.')
            if 'subtotal_raw' not in data_dict:
                data_dict['subtotal_raw'] = float(str(data_dict.get('subtotal', 0)).replace('$', '').replace(',', ''))
            if 'tax_raw' not in data_dict:
                data_dict['tax_raw'] = float(str(data_dict.get('tax', 0)).replace('$', '').replace(',', ''))
            if 'total_raw' not in data_dict:
                data_dict['total_raw'] = float(str(data_dict.get('total', 0)).replace('$', '').replace(',', ''))
            if 'discount_raw' not in data_dict:
                data_dict['discount_raw'] = 0.0
        
        invoice_id = f"invoice_{idx:05d}"
        
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
            
            inv_num = data_dict.get('invoice_number') or data_dict.get('order_number') or data_dict.get('po_number', 'N/A')
            inv_date = data_dict.get('invoice_date') or data_dict.get('order_date') or data_dict.get('po_date', 'N/A')
            inv_total = data_dict.get('total') or data_dict.get('total_amount') or data_dict.get('subtotal', 0)
            
            metadata = {
                "filename": Path(compressed_files[0]).name,
                "type": "invoice",
                "template": template,
                "category": category,
                "invoice_number": inv_num,
                "date": inv_date,
                "total": inv_total,
                "pages": result.get('pages', 1)
            }
            return ('success', idx, metadata)
        return ('failed', idx, None)
        
    except Exception as e:
        import traceback
        return ('error', idx, f"{str(e)}\n{traceback.format_exc()}")


def generate_parallel_dataset(output_dir: str, num_samples: int = 150000, num_workers: int = None):
    """
    Generate dataset using parallel workers
    
    Args:
        output_dir: Output directory
        num_samples: Total samples to generate
        num_workers: Number of parallel workers (default: CPU count)
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    metadata_dir = output_path / "metadata"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    templates_dir = str(project_root / "templates")
    
    # Determine worker count
    if num_workers is None:
        num_workers = min(mp.cpu_count(), 64)  # Cap at 64 to avoid overhead
    
    print(f"=" * 60)
    print(f"Parallel Dataset Generator")
    print(f"=" * 60)
    print(f"Target samples: {num_samples:,}")
    print(f"Workers: {num_workers}")
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Output: {output_dir}")
    print(f"=" * 60)
    
    # Distribution: 20% receipts, 80% invoices
    num_receipts = int(num_samples * 0.20)
    num_invoices = num_samples - num_receipts
    
    print(f"\nDistribution:")
    print(f"  Receipts: {num_receipts:,} (20%)")
    print(f"  Invoices: {num_invoices:,} (80%)")
    
    # Template lists
    receipt_templates = [
        "retail/pos_receipt.html",
        "retail/pos_receipt_auto_parts.html",
        "retail/pos_receipt_bookstore.html",
        "retail/pos_receipt_dense.html",
        "retail/pos_receipt_dollar_store.html",
        "retail/pos_receipt_fuel.html",
        "retail/pos_receipt_hardware.html",
        "retail/pos_receipt_home_center.html",
        "retail/pos_receipt_mini_market.html",
        "retail/pos_receipt_premium.html",
        "retail/pos_receipt_qsr.html",
        "retail/pos_receipt_stationery.html",
        "retail/pos_receipt_superstore.html",
        "retail/pos_receipt_trading.html",
        "retail/pos_receipt_warehouse.html",
        "retail/pos_receipt_wholesale.html",
        "retail/pos_receipt_wholesale_supply.html",
        "retail/pos_receipt_wide.html",
    ]
    
    # Invoice templates by category
    invoice_categories = {
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
        'ecommerce': [
            "ecommerce/accounting_pro_wholesale.html",
            "ecommerce/alibaba_invoice.html",
            "ecommerce/amazon_business_invoice.html",
            "ecommerce/amazon_invoice.html",
            "ecommerce/amazon_seller_invoice.html",
            "ecommerce/amazon_style_order.html",
            "ecommerce/aurora_glass_invoice.html",
            "ecommerce/bigcommerce_standard.html",
            "ecommerce/boutique_luxe_invoice.html",
            "ecommerce/candytoday_shopify_invoice.html",
            "ecommerce/circlespace_order.html",
            "ecommerce/cloud_accounting_invoice.html",
            "ecommerce/costco_invoice.html",
            "ecommerce/digital_payment_pro.html",
            "ecommerce/digital_payment_wholesale.html",
            "ecommerce/dingdong_shop_invoice.html",
            "ecommerce/ebay_invoice.html",
            "ecommerce/enterprise_accounting_invoice.html",
            "ecommerce/enterprise_wholesale_bold.html",
            "ecommerce/etsy_invoice.html",
            "ecommerce/faire_invoice.html",
            "ecommerce/it_supplier_invoice.html",
            "ecommerce/jungle_business_invoice.html",
            "ecommerce/jungle_business_uk_invoice.html",
            "ecommerce/logistics_pro_invoice.html",
            "ecommerce/magento_standard.html",
            "ecommerce/marketplace_transactional.html",
            "ecommerce/marketplace_wholesale.html",
            "ecommerce/modern_b2b_clean.html",
            "ecommerce/modern_cloud_wholesale.html",
            "ecommerce/modern_receipt_card.html",
            "ecommerce/modern_wholesale_pro.html",
            "ecommerce/nordic_minimalist_wholesale.html",
            "ecommerce/nordic_statement_invoice.html",
            "ecommerce/payment_processor_invoice.html",
            "ecommerce/premium_boutique_wholesale.html",
            "ecommerce/premium_minimalist_invoice.html",
            "ecommerce/saas_billing_invoice.html",
            "ecommerce/shine_invoice.html",
            "ecommerce/shopify_luxe_classic.html",
            "ecommerce/shopify_modern_clean.html",
            "ecommerce/shopify_modern_dark.html",
            "ecommerce/shopify_premium_detailed.html",
            "ecommerce/shopify_standard.html",
            "ecommerce/stripe_minimal_invoice.html",
            "ecommerce/tech_gradient_template.html",
            "ecommerce/tech_modern_gradient.html",
            "ecommerce/walmart_invoice.html",
            "ecommerce/waziexpress_invoice.html",
            "ecommerce/waziexpress_invoice_v2.html",
            "ecommerce/wholesale_compact_order.html",
            "ecommerce/woocommerce_standard.html",
            "ecommerce/zemu_invoice.html",
            "ecommerce/zylker_invoice.html",
        ],
        'purchase_orders': [
            "purchase_orders/po_alibaba.html",
            "purchase_orders/po_beauty.html",
            "purchase_orders/po_domestic_distributor.html",
            "purchase_orders/po_dropship.html",
            "purchase_orders/po_electronics.html",
            "purchase_orders/po_fashion_wholesale.html",
            "purchase_orders/po_food_beverage.html",
            "purchase_orders/po_generic.html",
            "purchase_orders/po_home_goods.html",
            "purchase_orders/po_landscape.html",
            "purchase_orders/po_landscape_modern.html",
            "purchase_orders/po_manufacturer_direct.html",
            "purchase_orders/po_receipt_paper.html",
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
        'wholesale': [
            "wholesale/beautyjoint_invoice.html",
            "wholesale/bulk_order_compact.html",
            "wholesale/business_invoice_clean.html",
            "wholesale/candyville_invoice.html",
            "wholesale/exotics_wholesale_invoice.html",
            "wholesale/faire_generic_invoice.html",
            "wholesale/fashiongo_invoice.html",
            "wholesale/global_export_invoice.html",
            "wholesale/ingrammicro_invoice.html",
            "wholesale/international_trade_invoice.html",
            "wholesale/maritime_trade_invoice.html",
            "wholesale/orderchamp_invoice.html",
            "wholesale/orientaltrading_invoice.html",
            "wholesale/petedge_invoice.html",
            "wholesale/stockup_market_invoice.html",
            "wholesale/waliwaba_proforma_invoice.html",
            "wholesale/wayfair_professional_invoice.html",
            "wholesale/zysco_foodservice_invoice.html",
        ],
        'supply_chain': [
            "supply_chain/bill_of_lading.html",
            "supply_chain/packing_slip.html",
            "supply_chain/packing_slip_modern.html",
            "supply_chain/packing_slip_landscape.html",
            "supply_chain/packing_slip_ecommerce.html",
            "supply_chain/delivery_note.html",
            "supply_chain/delivery_note_landscape.html",
            "supply_chain/shipping_manifest.html",
            "supply_chain/shipping_manifest_landscape.html",
            "supply_chain/customs_declaration.html",
            "supply_chain/customs_declaration_form.html",
            "supply_chain/proforma_invoice.html",
            "supply_chain/proforma_invoice_modern.html",
            "supply_chain/rma_form.html",
            "supply_chain/return_label.html",
            "supply_chain/return_label_modern.html",
            "supply_chain/receiving_report.html",
            "supply_chain/receiving_report_compact.html",
            "supply_chain/receiving_report_landscape.html",
            "supply_chain/credit_memo.html",
            "supply_chain/credit_memo_modern.html",
            "supply_chain/debit_memo.html",
            "supply_chain/debit_memo_modern.html",
            "supply_chain/inventory_adjustment_form.html",
            "supply_chain/stock_transfer.html",
            "supply_chain/cycle_count_sheet.html",
            "supply_chain/cycle_count_sheet_modern.html",
        ],
        'services': [
            "services/east_repair_invoice.html",
            "services/business_tax_invoice.html",
        ],
    }
    
    # Category weights
    category_weights = {
        'online_orders': 0.08,
        'ecommerce': 0.40,
        'purchase_orders': 0.10,
        'modern_professional': 0.08,
        'wholesale': 0.14,
        'supply_chain': 0.18,
        'services': 0.02,
    }
    
    # Prepare receipt tasks
    print("\nPreparing tasks...")
    receipt_tasks = []
    for i in range(num_receipts):
        template = random.choice(receipt_templates)
        receipt_tasks.append((i, template, str(images_dir), templates_dir))
    
    # Prepare invoice tasks with weighted category selection
    invoice_tasks = []
    categories = list(category_weights.keys())
    weights = list(category_weights.values())
    
    for i in range(num_invoices):
        category = random.choices(categories, weights=weights)[0]
        template = random.choice(invoice_categories[category])
        invoice_tasks.append((i, template, category, str(images_dir), templates_dir))
    
    # Process receipts
    print(f"\nGenerating {num_receipts:,} receipts with {num_workers} workers...")
    start_time = time.time()
    
    receipt_metadata = []
    success_count = 0
    error_count = 0
    
    with mp.Pool(num_workers, initializer=init_worker, initargs=(0,)) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_single_receipt, receipt_tasks, chunksize=10)):
            status, idx, data = result
            if status == 'success':
                receipt_metadata.append(data)
                success_count += 1
            else:
                error_count += 1
            
            # Progress update every 1000 items
            if (i + 1) % 1000 == 0 or i == num_receipts - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_receipts - i - 1) / rate if rate > 0 else 0
                print(f"  Receipts: {i+1:,}/{num_receipts:,} ({rate:.1f}/s, ETA: {eta/60:.1f}min)")
    
    print(f"  ✓ Receipts complete: {success_count:,} success, {error_count:,} errors")
    
    # Process invoices
    print(f"\nGenerating {num_invoices:,} invoices with {num_workers} workers...")
    invoice_start = time.time()
    
    invoice_metadata = []
    success_count = 0
    error_count = 0
    
    with mp.Pool(num_workers, initializer=init_worker, initargs=(1,)) as pool:
        for i, result in enumerate(pool.imap_unordered(generate_single_invoice, invoice_tasks, chunksize=10)):
            status, idx, data = result
            if status == 'success':
                invoice_metadata.append(data)
                success_count += 1
            elif status == 'error':
                error_count += 1
                if error_count <= 10:  # Only print first 10 errors
                    print(f"    Error in invoice {idx}: {data[:100]}...")
            
            # Progress update every 1000 items
            if (i + 1) % 1000 == 0 or i == num_invoices - 1:
                elapsed = time.time() - invoice_start
                rate = (i + 1) / elapsed
                eta = (num_invoices - i - 1) / rate if rate > 0 else 0
                print(f"  Invoices: {i+1:,}/{num_invoices:,} ({rate:.1f}/s, ETA: {eta/60:.1f}min)")
    
    print(f"  ✓ Invoices complete: {success_count:,} success, {error_count:,} errors")
    
    # Combine and save metadata
    all_metadata = receipt_metadata + invoice_metadata
    
    # Save individual metadata files
    for meta in all_metadata:
        filename = meta['filename'].replace('.png', '.json').replace('_page1', '')
        meta_path = metadata_dir / filename
        with open(meta_path, 'w') as f:
            json.dump([meta], f, indent=2)
    
    # Save combined metadata
    with open(metadata_dir / "dataset_metadata.json", 'w') as f:
        json.dump(all_metadata, f, indent=2)
    
    # Calculate statistics
    total_time = time.time() - start_time
    category_stats = {}
    for meta in all_metadata:
        cat = meta.get('category', 'receipt')
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    print(f"\n{'=' * 60}")
    print(f"Generation Complete!")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(all_metadata):,}/{num_samples:,}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Average rate: {len(all_metadata)/total_time:.1f} samples/second")
    print(f"\nCategory Distribution:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(all_metadata) * 100) if all_metadata else 0
        print(f"  {cat}: {count:,} ({pct:.1f}%)")
    print(f"\nOutput: {output_dir}")
    
    # Create completion marker
    with open(output_path / "COMPLETE.txt", 'w') as f:
        f.write(f"Generation complete: {datetime.now()}\n")
        f.write(f"Total samples: {len(all_metadata)}\n")
        f.write(f"Time: {total_time/60:.1f} minutes\n")


def main():
    parser = argparse.ArgumentParser(description='Parallel Dataset Generator')
    parser.add_argument('--samples', type=int, default=150000, help='Number of samples')
    parser.add_argument('--output', type=str, default='outputs/production_150k', help='Output directory')
    parser.add_argument('--workers', type=int, default=None, help='Number of workers (default: auto)')
    parser.add_argument('--quality', type=int, default=85, help='JPEG compression quality (1-100, default: 85)')
    
    args = parser.parse_args()
    
    print(f"\n📦 JPEG Quality: {args.quality}")
    print(f"   Expected file size: ~{50 + (args.quality - 85) * 2}KB per image\n")
    
    generate_parallel_dataset(
        output_dir=args.output,
        num_samples=args.samples,
        num_workers=args.workers
    )


if __name__ == '__main__':
    main()
