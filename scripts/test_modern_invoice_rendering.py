"""
Test script for modern invoice template rendering with landscape and A4 support

Tests:
1. Render A4 portrait invoice
2. Render landscape letter invoice
3. Verify augmentation works on both formats
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.html_to_png_renderer import HTMLToPNGRenderer
from generators.modern_invoice_generator import ModernInvoiceGenerator
from jinja2 import Template
import random
from datetime import datetime, timedelta


def generate_test_invoice_data():
    """Generate sample invoice data using ModernInvoiceGenerator"""
    generator = ModernInvoiceGenerator(locale='en_US')
    return generator.generate_modern_invoice()


def render_invoice_template(template_path: Path, output_path: Path, 
                           page_size: str, orientation: str,
                           apply_augmentation: bool = False):
    """Render invoice template with given page settings"""
    
    print(f"\nRendering {template_path.name}...")
    print(f"  Page size: {page_size} {orientation}")
    print(f"  Augmentation: {apply_augmentation}")
    
    # Load template
    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()
    
    # Generate test data
    data = generate_test_invoice_data()
    
    # Render template with data
    template = Template(template_content)
    html = template.render(**data)
    
    # Initialize renderer
    renderer = HTMLToPNGRenderer(
        augment_probability=1.0 if apply_augmentation else 0.0
    )
    
    # Render to PNG
    success = renderer.render(
        html_content=html,
        output_path=str(output_path),
        page_size=page_size,
        orientation=orientation,
        apply_augmentation=apply_augmentation
    )
    
    if success:
        file_size = output_path.stat().st_size / 1024  # KB
        print(f"  ✓ Success: {output_path.name} ({file_size:.1f} KB)")
    else:
        print(f"  ✗ Failed: {output_path.name}")
    
    return success


def main():
    """Test modern invoice rendering"""
    
    print("=" * 70)
    print("MODERN INVOICE RENDERING TEST")
    print("=" * 70)
    
    # Setup paths
    templates_dir = project_root / "templates" / "modern_professional"
    output_dir = project_root / "outputs" / "modern_invoice_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    tests = [
        {
            'name': 'A4 Portrait (Clean)',
            'template': templates_dir / "invoice_a4.html",
            'output': output_dir / "test_a4_portrait_clean.png",
            'page_size': 'A4',
            'orientation': 'Portrait',
            'augmentation': False
        },
        {
            'name': 'A4 Portrait (Augmented)',
            'template': templates_dir / "invoice_a4.html",
            'output': output_dir / "test_a4_portrait_augmented.png",
            'page_size': 'A4',
            'orientation': 'Portrait',
            'augmentation': True
        },
        {
            'name': 'Landscape Letter (Clean)',
            'template': templates_dir / "invoice_landscape.html",
            'output': output_dir / "test_landscape_clean.png",
            'page_size': 'Letter',
            'orientation': 'Landscape',
            'augmentation': False
        },
        {
            'name': 'Landscape Letter (Augmented)',
            'template': templates_dir / "invoice_landscape.html",
            'output': output_dir / "test_landscape_augmented.png",
            'page_size': 'Letter',
            'orientation': 'Landscape',
            'augmentation': True
        },
    ]
    
    # Run tests
    results = []
    for test in tests:
        print(f"\n{'='*70}")
        print(f"TEST: {test['name']}")
        print('='*70)
        
        success = render_invoice_template(
            template_path=test['template'],
            output_path=test['output'],
            page_size=test['page_size'],
            orientation=test['orientation'],
            apply_augmentation=test['augmentation']
        )
        
        results.append({'name': test['name'], 'success': success})
    
    # Summary
    print(f"\n{'='*70}")
    print("TEST SUMMARY")
    print('='*70)
    
    for result in results:
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status}: {result['name']}")
    
    passed = sum(1 for r in results if r['success'])
    total = len(results)
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Modern invoice rendering is working.")
        print(f"✓ Output location: {output_dir}")
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
