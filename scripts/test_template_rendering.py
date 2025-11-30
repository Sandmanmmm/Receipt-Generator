"""Simple template rendering test - no OCR required."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jinja2 import Environment, FileSystemLoader
from generators.retail_data_generator import RetailDataGenerator

def test_online_order_templates():
    """Test that all online_order templates render without errors."""
    
    env = Environment(
        loader=FileSystemLoader(str(project_root / "templates")),
        autoescape=False
    )
    
    retail_gen = RetailDataGenerator()
    
    online_order_templates = [
        "retail/online_order_electronics.html",
        "retail/online_order_fashion.html",
        "retail/online_order_grocery.html",
        "retail/online_order_home_improvement.html",
        "retail/online_order_invoice.html",
        "retail/online_order_marketplace.html",
        "retail/online_order_digital.html",
        "retail/online_order_wholesale.html",
    ]
    
    print("Testing online_order template rendering...")
    print("=" * 60)
    
    for template_name in online_order_templates:
        print(f"\nTesting: {template_name}")
        
        try:
            # Generate data
            from dataclasses import asdict
            data_obj = retail_gen.generate_online_order(store_type='electronics', min_items=12, max_items=12)
            data = asdict(data_obj)
            
            # Add pagination variables for multi-page test
            data['_is_first_page'] = True
            data['_page_number'] = 1
            data['_total_pages'] = 2
            
            # Wholesale template uses 'items' instead of 'line_items'
            if 'wholesale' in template_name:
                data['items'] = data.get('line_items', [])
            
            # Load template
            template = env.get_template(template_name)
            
            # Render template
            html = template.render(**data)
            
            if html and len(html) > 1000:
                print(f"  ✓ Rendered successfully ({len(html)} chars)")
            else:
                print(f"  ✗ Rendered but output seems short ({len(html)} chars)")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All templates rendered successfully!")
    return True

if __name__ == "__main__":
    success = test_online_order_templates()
    sys.exit(0 if success else 1)
