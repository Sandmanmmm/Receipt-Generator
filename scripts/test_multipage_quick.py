#!/usr/bin/env python3
"""Quick test to verify multipage template generation without OCR."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.modern_invoice_generator import ModernInvoiceGenerator
from generators.template_renderer import TemplateRenderer
from jinja2 import Environment, FileSystemLoader

def main():
    print("Testing multipage template generation...")
    
    # Setup Jinja2 environment
    template_dir = project_root / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    # Generate invoice with many items
    print("\n1. Generating invoice with 20 items...")
    invoice_gen = ModernInvoiceGenerator()
    data = invoice_gen.generate_modern_invoice(min_items=20, max_items=20)
    print(f"   Generated {len(data['items'])} items")
    
    # Test multipage template
    template_name = "modern_professional/invoice_minimal_multipage.html"
    print(f"\n2. Loading template: {template_name}")
    
    try:
        template = env.get_template(template_name)
        print("   Template loaded successfully")
    except Exception as e:
        print(f"   ERROR: {e}")
        return 1
    
    # Simulate pagination
    items_per_page = 12
    num_pages = (len(data['items']) + items_per_page - 1) // items_per_page
    print(f"\n3. Simulating {num_pages} pages @ {items_per_page} items/page")
    
    for page_num in range(num_pages):
        start_idx = page_num * items_per_page
        end_idx = min(start_idx + items_per_page, len(data['items']))
        page_items = data['items'][start_idx:end_idx]
        
        # Create page-specific data
        page_data = data.copy()
        page_data['items'] = page_items
        page_data['_page_number'] = page_num + 1
        page_data['_total_pages'] = num_pages
        page_data['_is_first_page'] = (page_num == 0)
        page_data['_is_last_page'] = (page_num == num_pages - 1)
        
        print(f"   Page {page_num + 1}: {len(page_items)} items")
        
        try:
            html = template.render(**page_data)
            print(f"      Rendered {len(html)} bytes")
            
            # Check for key elements
            if page_data['_is_first_page'] and 'Billed To' not in html:
                print("      WARNING: Missing 'Billed To' on first page")
            if page_data['_is_last_page'] and 'Total' not in html:
                print("      WARNING: Missing 'Total' on last page")
            if not page_data['_is_last_page'] and f"page {page_num + 2}" not in html.lower():
                print(f"      WARNING: Missing continuation notice")
                
        except Exception as e:
            print(f"      ERROR rendering page: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    print("\n✓ All pages rendered successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
