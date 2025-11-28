"""
Test multi-page receipt generation
Generates receipts with large item counts to trigger multi-page rendering
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.retail_data_generator import RetailDataGenerator
from generators.html_to_png_renderer import SimplePNGRenderer

def main():
    print("=" * 80)
    print("MULTI-PAGE RECEIPT TEST")
    print("=" * 80)
    
    # Initialize components
    generator = RetailDataGenerator()
    renderer = SimplePNGRenderer()
    
    output_dir = Path("outputs/multipage_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test scenarios
    test_cases = [
        {"name": "small_order", "min_items": 3, "max_items": 8, "desc": "Normal retail (should be single page)"},
        {"name": "medium_order", "min_items": 10, "max_items": 15, "desc": "Medium order (might be multi-page)"},
        {"name": "large_order", "min_items": 25, "max_items": 35, "desc": "Large wholesale (should be multi-page)"},
        {"name": "huge_order", "min_items": 50, "max_items": 60, "desc": "Huge e-commerce order (definitely multi-page)"}
    ]
    
    for idx, test_case in enumerate(test_cases, 1):
        print(f"\nTest {idx}/4: {test_case['desc']}")
        print(f"  Item range: {test_case['min_items']}-{test_case['max_items']} items")
        
        # Generate receipt data
        from dataclasses import asdict
        receipt = generator.generate_pos_receipt(
            min_items=test_case['min_items'],
            max_items=test_case['max_items']
        )
        
        # Convert to dict
        receipt_data = asdict(receipt)
        
        print(f"  Generated: {len(receipt_data['line_items'])} items")
        
        # Render to PNG
        output_path = output_dir / f"{test_case['name']}.png"
        success = renderer.render_receipt_dict(receipt_data, str(output_path))
        
        if success:
            # Check if multi-page files were created
            marker_file = output_dir / f"{test_case['name']}_MULTIPAGE.txt"
            if marker_file.exists():
                with open(marker_file, 'r') as f:
                    content = f.read()
                print(f"  ✓ Multi-page: {content.strip().split(chr(10))[0]}")
                
                # Count page files
                page_files = list(output_dir.glob(f"{test_case['name']}_page*.png"))
                print(f"  ✓ Generated {len(page_files)} page files")
            else:
                print(f"  ✓ Single page: {output_path.name}")
        else:
            print(f"  ✗ Failed to render")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()
