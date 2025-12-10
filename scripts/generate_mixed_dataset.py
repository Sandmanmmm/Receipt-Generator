"""
Mixed Dataset Generator
Generates a dataset containing both retail receipts and modern invoices
Target distribution: 60% Receipts, 40% Modern Invoices
"""
import sys
import os
import random
import json
import csv
import argparse
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator
from generators.modern_invoice_generator import ModernInvoiceGenerator
from generators.renderer import InvoiceRenderer
from generators.shopify_product_csv_generator import ShopifyProductCSVGenerator
from jinja2 import Environment, FileSystemLoader

# E-commerce Template to Generator Key Mapping
# All ecommerce templates require RetailDataGenerator.generate_for_template()
ECOMMERCE_TEMPLATE_KEYS = {
    # Marketplace
    "ecommerce/ebay_invoice.html": "ebay_invoice",
    "ecommerce/amazon_business_invoice.html": "amazon_business_invoice",
    "ecommerce/amazon_seller_invoice.html": "amazon_seller_invoice",
    "ecommerce/etsy_invoice.html": "etsy_invoice",
    "ecommerce/faire_invoice.html": "faire_invoice",
    "ecommerce/alibaba_invoice.html": "alibaba_invoice",
    "ecommerce/marketplace_transactional.html": "marketplace_transactional",
    "ecommerce/amazon_invoice.html": "amazon_invoice",
    
    # Shopify/Stripe
    "ecommerce/shopify_standard.html": "shopify_standard",
    "ecommerce/shopify_modern_clean.html": "shopify_modern_clean",
    "ecommerce/shopify_modern_dark.html": "shopify_modern_dark",
    "ecommerce/shopify_luxe_classic.html": "shopify_luxe_classic",
    "ecommerce/stripe_minimal_invoice.html": "stripe_minimal_invoice",
    "ecommerce/payment_processor_invoice.html": "payment_processor_invoice",
    
    # Wholesale/B2B
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
    
    # Payment Processors
    "ecommerce/digital_payment_pro.html": "digital_payment_pro",
    "ecommerce/cloud_accounting_invoice.html": "cloud_accounting_invoice",
    "ecommerce/saas_billing_invoice.html": "saas_billing_invoice",
    "ecommerce/modern_receipt_card.html": "modern_receipt_card",
    "ecommerce/shopify_premium_detailed.html": "shopify_premium_detailed",
    "ecommerce/tech_modern_gradient.html": "tech_modern_gradient",
    
    # Specialized
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
    
    # Initialize renderer with multipage support
    renderer = InvoiceRenderer(
        templates_dir=str(project_root / "templates"),
        output_dir=str(output_path / "images")
    )
    
    # Setup Jinja2 environment
    template_loader = FileSystemLoader(str(project_root / "templates"))
    env = Environment(loader=template_loader)
    
    # Calculate distribution - ALL 153 TEMPLATES
    # Balanced distribution across all template categories
    num_receipts = int(num_samples * 0.20)  # 20% POS receipts
    num_invoices = num_samples - num_receipts  # 80% invoices/documents
    
    print(f"Generating {num_samples} samples (All 153 Templates):")
    print(f"  - POS Receipts: {num_receipts} (20%)")
    print(f"  - Invoices/Documents: {num_invoices} (80%)")
    print(f"    • Includes all 153 templates across 7 categories")
    
    # ============================================================================
    # ALL 153 TEMPLATES - Comprehensive Coverage
    # ============================================================================
    
    # === RETAIL POS RECEIPTS (18 templates) ===
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
    
    # === RETAIL ONLINE ORDERS (10 templates) ===
    online_order_templates = [
        "retail/blue_wave_invoice.html",
        "retail/consumer_service_invoice.html",
        "retail/online_order_digital.html",
        "retail/online_order_electronics.html",
        "retail/online_order_fashion.html",
        "retail/online_order_grocery.html",
        "retail/online_order_home_improvement.html",
        "retail/online_order_invoice.html",
        "retail/online_order_marketplace.html",
        "retail/online_order_wholesale.html",
    ]
    
    # === ECOMMERCE (55 templates) ===
    ecommerce_templates = [
        "ecommerce/accounting_pro_wholesale.html",
        "ecommerce/alibaba_invoice.html",
        "ecommerce/aliexpress_invoice.html",
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
    ]
    
    # === PURCHASE ORDERS (13 templates) ===
    purchase_order_templates = [
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
    ]
    
    # === MODERN PROFESSIONAL (10 templates) ===
    modern_professional_templates = [
        "modern_professional/invoice_a4.html",
        "modern_professional/invoice_bold.html",
        "modern_professional/invoice_compact.html",
        "modern_professional/invoice_compact_multipage.html",
        "modern_professional/invoice_ecommerce.html",
        "modern_professional/invoice_elegant.html",
        "modern_professional/invoice_landscape.html",
        "modern_professional/invoice_minimal.html",
        "modern_professional/invoice_minimal_multipage.html",
        "modern_professional/invoice_sidebar.html",
    ]
    
    # === WHOLESALE (18 templates) ===
    wholesale_templates = [
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
    ]
    
    # === SUPPLY CHAIN (27 templates) ===
    supply_chain_templates = [
        "supply_chain/bill_of_lading.html",
        "supply_chain/credit_memo.html",
        "supply_chain/credit_memo_modern.html",
        "supply_chain/customs_declaration.html",
        "supply_chain/customs_declaration_form.html",
        "supply_chain/cycle_count_sheet.html",
        "supply_chain/cycle_count_sheet_modern.html",
        "supply_chain/debit_memo.html",
        "supply_chain/debit_memo_modern.html",
        "supply_chain/delivery_note.html",
        "supply_chain/delivery_note_landscape.html",
        "supply_chain/inventory_adjustment_form.html",
        "supply_chain/packing_slip.html",
        "supply_chain/packing_slip_ecommerce.html",
        "supply_chain/packing_slip_landscape.html",
        "supply_chain/packing_slip_modern.html",
        "supply_chain/proforma_invoice.html",
        "supply_chain/proforma_invoice_modern.html",
        "supply_chain/receiving_report.html",
        "supply_chain/receiving_report_compact.html",
        "supply_chain/receiving_report_landscape.html",
        "supply_chain/return_label.html",
        "supply_chain/return_label_modern.html",
        "supply_chain/rma_form.html",
        "supply_chain/shipping_manifest.html",
        "supply_chain/shipping_manifest_landscape.html",
        "supply_chain/stock_transfer.html",
    ]
    
    # === SERVICES (2 templates) ===
    services_templates = [
        "services/business_tax_invoice.html",
        "services/east_repair_invoice.html",
    ]
    
    generated_count = 0
    metadata_list = []
    
    # Generate Receipts
    print("\nGenerating Receipts...")
    for i in tqdm(range(num_receipts)):
        try:
            # Generate data with appropriate item count for receipts (5-15 items for 1-2 pages)
            data = retail_gen.generate_pos_receipt(min_items=5, max_items=15)
            data_dict = retail_gen.to_dict(data)
            
            # Select template
            template_name = random.choice(receipt_templates)
            
            # Output filename
            invoice_id = f"receipt_{i:05d}"
            
            # Render using InvoiceRenderer with multipage support
            result = renderer.render_invoice(
                template_name=template_name,
                data=data_dict,
                invoice_id=invoice_id,
                formats=['png'],
                use_multipage=True
            )
            
            if result['png']:
                # Get the first PNG (or only page if single-page)
                png_file = Path(result['png'][0])
                metadata = {
                    "filename": png_file.name,
                    "type": "receipt",
                    "template": template_name,
                    "invoice_number": data.invoice_number,
                    "date": data.invoice_date,
                    "total": data.total_amount,
                    "pages": result['pages']
                }
                metadata_list.append(metadata)
                generated_count += 1
                
        except Exception as e:
            print(f"Error generating receipt {i}: {str(e)}")
            
    # Generate Invoices with balanced category selection across ALL 153 templates
    print("\nGenerating Invoices/Documents (All 153 Templates - Balanced Distribution)...")
    
    # Define category weights for balanced distribution across all template types
    # Weights proportional to template count in each category
    category_weights = {
        'online_orders': 0.08,           # 10 templates (retail online orders)
        'ecommerce': 0.40,               # 55 templates (largest category)
        'purchase_orders': 0.10,         # 13 templates
        'modern_professional': 0.08,     # 10 templates
        'wholesale': 0.14,               # 18 templates
        'supply_chain': 0.18,            # 27 templates
        'services': 0.02,                # 2 templates
    }
    
    for i in tqdm(range(num_invoices)):
        try:
            # Select category based on weights
            category = random.choices(
                list(category_weights.keys()),
                weights=list(category_weights.values()),
                k=1
            )[0]
            
            # Select template from category
            if category == 'online_orders':
                template_name = random.choice(online_order_templates)
            elif category == 'ecommerce':
                template_name = random.choice(ecommerce_templates)
            elif category == 'purchase_orders':
                template_name = random.choice(purchase_order_templates)
            elif category == 'modern_professional':
                template_name = random.choice(modern_professional_templates)
            elif category == 'wholesale':
                template_name = random.choice(wholesale_templates)
            elif category == 'supply_chain':
                template_name = random.choice(supply_chain_templates)
            else:  # services
                template_name = random.choice(services_templates)
            
            # Determine which generator to use based on template type
            use_retail_generator = 'online_order' in template_name or template_name.startswith('retail/')
            use_purchase_order_generator = 'purchase_order' in template_name or 'po_' in template_name
            use_pos_receipt_generator = 'pos_receipt' in template_name
            
            # Determine realistic item ranges based on template design
            # Distribution: 60% single-page, 30% 2-page, 10% 3-page
            page_distribution = random.random()
            
            # Template-specific item counts (items per page)
            if 'compact' in template_name:
                # 12 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 12
                elif page_distribution < 0.9:
                    min_items, max_items = 13, 30
                else:
                    min_items, max_items = 31, 48
            elif 'minimal' in template_name:
                # 15 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 15
                elif page_distribution < 0.9:
                    min_items, max_items = 16, 35
                else:
                    min_items, max_items = 36, 55
            elif 'marketplace' in template_name or 'online_order' in template_name:
                # 4-5 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 2, 4
                elif page_distribution < 0.9:
                    min_items, max_items = 5, 13
                else:
                    min_items, max_items = 14, 22
            elif 'ebay' in template_name or 'etsy' in template_name or 'amazon' in template_name:
                # E-commerce templates: 8-10 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 10
                elif page_distribution < 0.9:
                    min_items, max_items = 11, 25
                else:
                    min_items, max_items = 26, 40
            elif 'wholesale' in template_name or 'b2b' in template_name:
                # Wholesale/B2B: 10-12 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 12
                elif page_distribution < 0.9:
                    min_items, max_items = 13, 28
                else:
                    min_items, max_items = 29, 45
            elif use_purchase_order_generator:
                # Purchase orders: 12-15 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 15
                elif page_distribution < 0.9:
                    min_items, max_items = 16, 35
                else:
                    min_items, max_items = 36, 55
            else:
                # Default: 10 items/page
                if page_distribution < 0.6:
                    min_items, max_items = 3, 10
                elif page_distribution < 0.9:
                    min_items, max_items = 11, 25
                else:
                    min_items, max_items = 26, 40
            
            # Generate data using appropriate generator based on template type
            if template_name in ECOMMERCE_TEMPLATE_KEYS:
                # E-commerce templates: Use RetailDataGenerator with template key
                template_key = ECOMMERCE_TEMPLATE_KEYS[template_name]
                receipt = retail_gen.generate_for_template(
                    template_name=template_key,
                    min_items=min_items,
                    max_items=max_items
                )
                data_dict = retail_gen.to_dict(receipt)
            
            elif use_purchase_order_generator:
                # Purchase order templates: Use PurchaseOrderGenerator
                from generators.purchase_order_generator import PurchaseOrderGenerator
                po_gen = PurchaseOrderGenerator()
                
                # Map template to industry (Phase 7B)
                template_basename = template_name.split('/')[-1]
                industry_map = {
                    'po_beauty.html': 'beauty',
                    'po_electronics.html': 'electronics',
                    'po_fashion_wholesale.html': 'fashion',
                    'po_food_beverage.html': 'food_beverage',
                    'po_home_goods.html': 'home_goods',
                    'po_manufacturer_direct.html': 'manufacturing',
                    'po_receipt_paper.html': 'paper',
                }
                industry = industry_map.get(template_basename, None)
                
                # Determine PO type from template name
                if 'alibaba' in template_name:
                    po_type = 'alibaba'
                elif 'dropship' in template_name:
                    po_type = 'dropship'
                elif 'landscape' in template_name:
                    po_type = 'landscape'
                else:
                    po_type = 'generic'
                
                data = po_gen.generate_purchase_order(
                    po_type=po_type,
                    min_items=min_items,
                    max_items=max_items,
                    industry=industry  # Phase 7B: Add industry parameter
                )
                data_dict = po_gen.to_dict(data)
            
            elif use_retail_generator:
                # Online order templates: Use RetailDataGenerator
                # Special handling for grocery template
                if 'grocery' in template_name.lower():
                    store_type = 'grocery_delivery'
                else:
                    store_type = random.choice(['fashion', 'home_garden', 'toys_games', 'electronics'])
                
                data = retail_gen.generate_online_order(
                    store_type=store_type,
                    min_items=min_items,
                    max_items=max_items
                )
                data_dict = retail_gen.to_dict(data)
            
            elif use_pos_receipt_generator:
                # POS receipt templates: Use RetailDataGenerator.generate_pos_receipt()
                data = retail_gen.generate_pos_receipt(
                    min_items=min_items,
                    max_items=max_items
                )
                data_dict = retail_gen.to_dict(data)
            
            else:
                # Modern professional, wholesale, supply_chain, services templates
                # Use ModernInvoiceGenerator
                data_dict = invoice_gen.generate_modern_invoice(
                    min_items=min_items,
                    max_items=max_items
                )
                
                # Add comprehensive field mappings for supply_chain/wholesale/services templates
                if 'supply_chain' in template_name or 'wholesale' in template_name or 'services' in template_name:
                    # Company/supplier/exporter names
                    company = data_dict.get('company_name', 'Global Supply Co.')
                    customer = data_dict.get('customer_name', 'Customer Corp.')
                    
                    data_dict['supplier_name'] = company
                    data_dict['exporter_name'] = company
                    data_dict['shipper_name'] = company
                    data_dict['consignor_name'] = company
                    data_dict['vendor_name'] = company
                    data_dict['seller_name'] = company
                    
                    data_dict['recipient_name'] = customer
                    data_dict['consignee_name'] = customer
                    data_dict['importer_name'] = customer
                    data_dict['buyer_name'] = customer
                    
                    # Exporter/Importer objects for complex templates
                    data_dict['exporter'] = {
                        'company_name': company,
                        'address': data_dict.get('company_address', '123 Business Ave'),
                        'city': data_dict.get('company_city', 'New York'),
                        'country': 'USA',
                        'phone': data_dict.get('company_phone', '555-0100'),
                        'email': data_dict.get('company_email', 'info@company.com'),
                    }
                    data_dict['importer'] = {
                        'company_name': customer,
                        'address': data_dict.get('customer_address', '456 Client St'),
                        'city': data_dict.get('customer_city', 'Los Angeles'),
                        'country': 'USA',
                        'phone': '555-0200',
                        'email': 'contact@customer.com',
                    }
                    
                    # Address objects
                    data_dict['buyer_address'] = {
                        'street': data_dict.get('customer_address', '456 Client St'),
                        'city': data_dict.get('customer_city', 'Los Angeles'),
                        'state': 'CA',
                        'zip': '90001',
                        'country': 'USA',
                    }
                    data_dict['seller_address'] = {
                        'street': data_dict.get('company_address', '123 Business Ave'),
                        'city': data_dict.get('company_city', 'New York'),
                        'state': 'NY',
                        'zip': '10001',
                        'country': 'USA',
                    }
                    data_dict['supplier_address'] = data_dict['seller_address']  # Alias
                    data_dict['ship_from_address'] = data_dict['seller_address']  # Alias
                    data_dict['ship_to_address'] = data_dict['buyer_address']  # Alias
                    
                    # Document numbers
                    data_dict['document_number'] = data_dict.get('invoice_number', f'DOC-{random.randint(10000, 99999)}')
                    data_dict['reference_number'] = f'REF-{random.randint(10000, 99999)}'
                    data_dict['tracking_number'] = f'TRK-{random.randint(1000000000, 9999999999)}'
                    data_dict['shipment_number'] = f'SHP-{random.randint(10000, 99999)}'
                    data_dict['order_reference'] = f'ORD-{random.randint(10000, 99999)}'
                    data_dict['memo_number'] = f'MEM-{random.randint(10000, 99999)}'
                    data_dict['rma_number'] = f'RMA-{random.randint(10000, 99999)}'
                    data_dict['bol_number'] = f'BOL-{random.randint(10000, 99999)}'
                    data_dict['customs_entry_number'] = f'CE-{random.randint(100000, 999999)}'
                    
                    # Raw numeric values for templates that need them
                    data_dict['subtotal_raw'] = float(str(data_dict.get('subtotal', 0)).replace('$', '').replace(',', ''))
                    data_dict['tax_raw'] = float(str(data_dict.get('tax', data_dict.get('tax_amount', 0))).replace('$', '').replace(',', ''))
                    data_dict['total_raw'] = float(str(data_dict.get('total', data_dict.get('total_amount', 0))).replace('$', '').replace(',', ''))
                    data_dict['discount_raw'] = 0.0
                    
                    # Dates
                    data_dict['shipment_date'] = data_dict.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))
                    data_dict['delivery_date'] = data_dict.get('due_date', datetime.now().strftime('%Y-%m-%d'))
                    data_dict['count_date'] = data_dict.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))
                    data_dict['memo_date'] = data_dict.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))
                    data_dict['ship_date'] = data_dict.get('invoice_date', datetime.now().strftime('%Y-%m-%d'))
                    
                    # Shipping details
                    data_dict['carrier'] = random.choice(['FedEx', 'UPS', 'DHL', 'USPS', 'Freight Co.'])
                    data_dict['carrier_name'] = data_dict['carrier']
                    data_dict['vessel_name'] = f'MV {random.choice(["Pacific", "Atlantic", "Global", "Express"])} {random.choice(["Star", "Explorer", "Trader", "Voyager"])}'
                    data_dict['port_of_loading'] = random.choice(['Los Angeles', 'Long Beach', 'New York', 'Seattle'])
                    data_dict['port_of_discharge'] = random.choice(['Shanghai', 'Rotterdam', 'Singapore', 'Hamburg'])
                    data_dict['shipping_method'] = random.choice(['Ground', 'Air', 'Ocean', 'Express'])
                    data_dict['weight'] = f'{random.randint(10, 500)} lbs'
                    data_dict['total_weight'] = f'{random.randint(100, 5000)} lbs'
                    data_dict['packages'] = random.randint(1, 50)
                    data_dict['total_packages'] = data_dict['packages']
                    
                    # Inventory-specific fields (for line items)
                    for item in data_dict.get('line_items', []):
                        item['expected_quantity'] = item.get('quantity', 1)
                        item['actual_quantity'] = item['expected_quantity'] + random.randint(-2, 2)
                        item['variance'] = item['actual_quantity'] - item['expected_quantity']
                        item['location'] = f'Bin-{random.choice(["A", "B", "C", "D"])}{random.randint(1, 99):02d}'
                        item['lot_number'] = f'LOT-{random.randint(10000, 99999)}'
                        item['serial_number'] = f'SN-{random.randint(100000, 999999)}'
                        item['weight'] = f'{random.uniform(0.5, 50):.1f} lbs'
                        item['dimensions'] = f'{random.randint(5, 50)}x{random.randint(5, 50)}x{random.randint(5, 50)} in'
                        item['country_of_origin'] = random.choice(['China', 'USA', 'Mexico', 'Vietnam', 'Germany'])
                        item['hs_code'] = f'{random.randint(1000, 9999)}.{random.randint(10, 99)}.{random.randint(10, 99)}'
                        # Add unit_cost for templates that need it
                        if 'unit_price' in item:
                            item['unit_cost'] = item['unit_price']
                        elif 'amount' in item and 'quantity' in item and item['quantity'] > 0:
                            item['unit_cost'] = round(float(str(item['amount']).replace('$', '').replace(',', '')) / item['quantity'], 2)
                        else:
                            item['unit_cost'] = round(random.uniform(5, 200), 2)
                        item['transfer_qty'] = item.get('quantity', 1)
                    
                    # Reason/notes for memos
                    data_dict['reason'] = random.choice([
                        'Damaged goods received',
                        'Price adjustment',
                        'Quantity discrepancy',
                        'Return authorization',
                        'Quality issue',
                        'Shipping error correction'
                    ])
                    data_dict['notes'] = data_dict.get('notes', 'Please reference this document number for all correspondence.')
                    
                    # Additional fields for receiving reports
                    data_dict['items_with_issues'] = random.randint(0, 3)
                    data_dict['inspection_notes'] = random.choice(['', 'Minor damage noted', 'Quality check passed', 'Recount required'])
                    data_dict['received_by'] = random.choice(['John Smith', 'Maria Garcia', 'David Chen', 'Sarah Johnson'])
                    data_dict['inspected_by'] = random.choice(['Quality Team', 'Receiving Dept', 'Warehouse Manager'])
                    
                    # Bank details for proforma invoices
                    data_dict['bank_details'] = {
                        'beneficiary_name': company,
                        'bank_name': random.choice(['Chase Bank', 'Bank of America', 'Wells Fargo', 'Citibank']),
                        'account_number': f'****{random.randint(1000, 9999)}',
                        'routing_number': f'{random.randint(100000000, 999999999)}',
                        'swift_code': f'SWIFT{random.randint(1000, 9999)}',
                        'iban': f'US{random.randint(10, 99)}****{random.randint(1000, 9999)}',
                    }
                    
                    # String address fields for templates that expect strings
                    data_dict['buyer_address_line1'] = data_dict.get('customer_address', '456 Client St')
                    data_dict['buyer_address_line2'] = f"{data_dict.get('customer_city', 'Los Angeles')}, CA 90001"
                    data_dict['seller_address_line1'] = data_dict.get('company_address', '123 Business Ave')
                    data_dict['seller_address_line2'] = f"{data_dict.get('company_city', 'New York')}, NY 10001"
                    
                    # String versions of addresses for templates using .split()
                    # Store dict as _obj and string as base name for compatibility
                    data_dict['buyer_address_obj'] = data_dict['buyer_address']
                    data_dict['seller_address_obj'] = data_dict['seller_address']
                    buyer_addr = data_dict['buyer_address']
                    seller_addr = data_dict['seller_address']
                    data_dict['buyer_address_str'] = f"{buyer_addr['street']}, {buyer_addr['city']}, {buyer_addr['state']} {buyer_addr['zip']}"
                    data_dict['seller_address_str'] = f"{seller_addr['street']}, {seller_addr['city']}, {seller_addr['state']} {seller_addr['zip']}"
                    # Replace dict with string for templates that use .split()
                    data_dict['buyer_address'] = data_dict['buyer_address_str']
                    data_dict['seller_address'] = data_dict['seller_address_str']
                    data_dict['seller_address_line2'] = f"{data_dict.get('company_city', 'New York')}, NY 10001"
                    
                    # Transfer fields for stock transfer
                    data_dict['transfer_qty'] = random.randint(1, 100)
                    data_dict['unit_cost'] = round(random.uniform(5, 500), 2)
                    data_dict['from_location'] = f'Warehouse-{random.choice(["A", "B", "C"])}'
                    data_dict['to_location'] = f'Warehouse-{random.choice(["D", "E", "F"])}'
                    data_dict['transfer_reason'] = random.choice(['Restock', 'Demand Shift', 'Consolidation', 'Seasonal'])
                    data_dict['total_value'] = data_dict.get('total_raw', random.uniform(500, 10000))
                    
                # Add supplier_name for ecommerce templates that need it
                if 'ecommerce' in template_name:
                    if 'supplier_name' not in data_dict:
                        data_dict['supplier_name'] = data_dict.get('company_name', 'Supplier Inc.')
                    # Add raw numeric values for templates that need them
                    if 'subtotal_raw' not in data_dict:
                        data_dict['subtotal_raw'] = float(str(data_dict.get('subtotal', 0)).replace('$', '').replace(',', ''))
                    if 'tax_raw' not in data_dict:
                        data_dict['tax_raw'] = float(str(data_dict.get('tax', data_dict.get('tax_amount', 0))).replace('$', '').replace(',', ''))
                    if 'total_raw' not in data_dict:
                        data_dict['total_raw'] = float(str(data_dict.get('total', data_dict.get('total_amount', 0))).replace('$', '').replace(',', ''))
                    if 'discount_raw' not in data_dict:
                        data_dict['discount_raw'] = 0.0
            
            # Output filename
            invoice_id = f"invoice_{i:05d}"
            
            # Render using InvoiceRenderer with multipage support
            result = renderer.render_invoice(
                template_name=template_name,
                data=data_dict,
                invoice_id=invoice_id,
                formats=['png'],
                use_multipage=True
            )
            
            if result['png']:
                # Get the first PNG (or only page if single-page)
                png_file = Path(result['png'][0])
                # Handle both data structures for metadata
                inv_num = data_dict.get('invoice_number') or data_dict.get('order_number') or data_dict.get('po_number', 'N/A')
                inv_date = data_dict.get('invoice_date') or data_dict.get('order_date') or data_dict.get('po_date', 'N/A')
                inv_total = data_dict.get('total') or data_dict.get('total_amount') or data_dict.get('subtotal', 0)
                metadata = {
                    "filename": png_file.name,
                    "type": "invoice",
                    "template": template_name,
                    "category": category,
                    "invoice_number": inv_num,
                    "date": inv_date,
                    "total": inv_total,
                    "pages": result['pages']
                }
                metadata_list.append(metadata)
                generated_count += 1
                
        except Exception as e:
            print(f"Error generating invoice {i} (template: {template_name}): {str(e)}")
            import traceback
            traceback.print_exc()

    # Save metadata with category statistics
    with open(metadata_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata_list, f, indent=2)
    
    # Calculate and display statistics
    category_stats = {}
    for meta in metadata_list:
        cat = meta.get('category', 'receipt')
        category_stats[cat] = category_stats.get(cat, 0) + 1
    
    print(f"\nDataset generation complete!")
    print(f"Total samples: {generated_count}/{num_samples}")
    print(f"\nCategory Distribution:")
    for cat, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / generated_count * 100) if generated_count > 0 else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")
    print(f"\nOutput directory: {output_dir}")

def generate_shopify_csv(
    output_dir: str,
    num_products: int = 100,
    categories: list = None,
    vendor: str = None,
    seed: int = None
) -> str:
    """
    Generate Shopify-compatible product CSV for bulk import.
    
    Args:
        output_dir: Directory to save output
        num_products: Number of products to generate
        categories: Specific categories to use (None = random mix)
        vendor: Specific vendor/brand name (None = generate random)
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ShopifyProductCSVGenerator(seed=seed)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"shopify_products_{num_products}_{timestamp}.csv"
    csv_path = output_path / csv_filename
    
    print(f"\nGenerating Shopify Product CSV...")
    print(f"  - Products: {num_products}")
    if categories:
        print(f"  - Categories: {', '.join(categories)}")
    else:
        print(f"  - Categories: All (random mix)")
    if vendor:
        print(f"  - Vendor: {vendor}")
    else:
        print(f"  - Vendor: Random (realistic brand names)")
    
    # Generate CSV
    result_path = generator.generate_csv(
        num_products=num_products,
        categories=categories,
        output_path=str(csv_path),
        vendor=vendor
    )
    
    # Read back to get stats
    with open(result_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        rows = list(reader)
    
    print(f"\nShopify CSV generation complete!")
    print(f"  - Output: {result_path}")
    print(f"  - Total CSV rows: {len(rows)} (products + variants + images)")
    print(f"  - Ready for Shopify import")
    
    return result_path


def generate_unified_dataset(
    output_dir: str,
    num_samples: int = 100,
    num_products: int = 50,
    augment: bool = True,
    shopify_categories: list = None,
    shopify_vendor: str = None,
    seed: int = None
):
    """
    Generate BOTH invoice images AND Shopify product CSVs.
    
    This unified command is ideal for Shopify store owners who need:
    1. Invoice/receipt images for OCR training (inventory updates)
    2. Product CSVs for bulk catalog uploads
    
    Args:
        output_dir: Base output directory
        num_samples: Number of invoice/receipt images
        num_products: Number of Shopify products
        augment: Apply augmentation to images
        shopify_categories: Categories for products (None = random)
        shopify_vendor: Vendor name for products (None = random)
        seed: Random seed for reproducibility
    """
    print("="*60)
    print("UNIFIED DATASET GENERATION - Shopify Inventory OCR")
    print("="*60)
    print(f"\nOutput: {output_dir}")
    print(f"Images: {num_samples} invoices/receipts")
    print(f"Products: {num_products} Shopify CSV products")
    
    # Set seed if provided
    if seed:
        random.seed(seed)
    
    # Generate invoice/receipt images
    print("\n" + "-"*40)
    print("PHASE 1: Invoice/Receipt Images")
    print("-"*40)
    generate_mixed_dataset(
        output_dir=output_dir,
        num_samples=num_samples,
        augment=augment
    )
    
    # Generate Shopify product CSV
    print("\n" + "-"*40)
    print("PHASE 2: Shopify Product CSV")
    print("-"*40)
    csv_output_dir = Path(output_dir) / "shopify_csv"
    csv_path = generate_shopify_csv(
        output_dir=str(csv_output_dir),
        num_products=num_products,
        categories=shopify_categories,
        vendor=shopify_vendor,
        seed=seed
    )
    
    print("\n" + "="*60)
    print("UNIFIED GENERATION COMPLETE!")
    print("="*60)
    print(f"\nOutputs:")
    print(f"  📄 Images: {output_dir}/images/")
    print(f"  📊 Metadata: {output_dir}/metadata/")
    print(f"  🛒 Shopify CSV: {csv_path}")
    print(f"\nNext steps:")
    print(f"  1. Upload CSV to Shopify: Products > Import")
    print(f"  2. Use images for OCR model training")
    print(f"  3. Run: python scripts/validate_annotations.py --input {output_dir}/metadata/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate mixed receipt/invoice dataset with optional Shopify CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate images only (default)
  python scripts/generate_mixed_dataset.py --samples 100 --output data/train
  
  # Generate Shopify CSV only
  python scripts/generate_mixed_dataset.py --shopify-only --products 50 --output data/shopify
  
  # Generate BOTH images AND Shopify CSV (unified)
  python scripts/generate_mixed_dataset.py --unified --samples 100 --products 50 --output data/full
  
  # Shopify CSV with specific categories
  python scripts/generate_mixed_dataset.py --shopify-only --products 100 \\
    --shopify-categories apparel shoes bags --output data/fashion
  
  # Shopify CSV with custom brand name
  python scripts/generate_mixed_dataset.py --shopify-only --products 25 \\
    --shopify-vendor "My Store Brand" --output data/branded
        """
    )
    
    # Output options
    parser.add_argument("--output", default="outputs/mixed_dataset", help="Output directory")
    
    # Image generation options
    parser.add_argument("--samples", type=int, default=10, help="Number of invoice/receipt samples")
    parser.add_argument("--no-augment", action="store_true", help="Disable image augmentation")
    
    # Shopify CSV options
    parser.add_argument("--shopify-only", action="store_true", 
                        help="Generate Shopify CSV only (no images)")
    parser.add_argument("--unified", action="store_true",
                        help="Generate BOTH images AND Shopify CSV")
    parser.add_argument("--products", type=int, default=50,
                        help="Number of Shopify products to generate")
    parser.add_argument("--shopify-categories", nargs="+", 
                        choices=["apparel", "shoes", "electronics", "home_decor", "beauty", 
                                 "jewelry", "bags", "fitness", "pet_supplies", "kitchen",
                                 "candy", "books_media", "toys_games", "sports_equipment",
                                 "automotive", "office_supplies", "garden_outdoor", "baby_products"],
                        help="Specific product categories for Shopify CSV")
    parser.add_argument("--shopify-vendor", type=str,
                        help="Custom vendor/brand name for Shopify products")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Determine mode and execute
    if args.shopify_only:
        # Shopify CSV only
        generate_shopify_csv(
            output_dir=args.output,
            num_products=args.products,
            categories=args.shopify_categories,
            vendor=args.shopify_vendor,
            seed=args.seed
        )
    elif args.unified:
        # Both images AND Shopify CSV
        generate_unified_dataset(
            output_dir=args.output,
            num_samples=args.samples,
            num_products=args.products,
            augment=not args.no_augment,
            shopify_categories=args.shopify_categories,
            shopify_vendor=args.shopify_vendor,
            seed=args.seed
        )
    else:
        # Default: Images only (original behavior)
        generate_mixed_dataset(
            output_dir=args.output,
            num_samples=args.samples,
            augment=not args.no_augment
        )
