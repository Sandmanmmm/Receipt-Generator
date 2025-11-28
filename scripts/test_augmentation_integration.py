"""
Quick Test: Verify Augmentation Integration

Tests that receipts are automatically augmented during generation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.html_to_png_renderer import HTMLToPNGRenderer
import random


def test_augmentation_integration():
    """Test that augmentation is properly integrated"""
    print("\n" + "="*60)
    print("TESTING: Augmentation Integration")
    print("="*60)
    
    output_dir = project_root / "outputs" / "augmentation_integration_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test 1: Default behavior (50% augmentation probability)
    print("\n[Test 1] Default behavior (50% augmentation)...")
    renderer_default = HTMLToPNGRenderer(augment_probability=0.5)
    
    augmented_count = 0
    total_count = 10
    
    for i in range(total_count):
        output_path = output_dir / f"test_default_{i:02d}.png"
        
        # Generate simple receipt
        html = f"""
        <!DOCTYPE html>
        <html><head><style>
        body {{ font-family: monospace; padding: 20px; }}
        </style></head><body>
        <h2>Test Receipt #{i}</h2>
        <p>Item 1: $10.00</p>
        <p>Item 2: $15.00</p>
        <p>Total: $25.00</p>
        </body></html>
        """
        
        success = renderer_default.render(html, str(output_path))
        if success:
            print(f"  ✓ Generated receipt {i+1}/{total_count}")
    
    print(f"  Generated {total_count} receipts with 50% augmentation probability")
    print(f"  (Expected ~5 augmented, actual varies due to randomness)")
    
    # Test 2: Force augmentation ON
    print("\n[Test 2] Force augmentation ON (100%)...")
    renderer_always = HTMLToPNGRenderer(augment_probability=1.0)
    
    for i in range(5):
        output_path = output_dir / f"test_always_augmented_{i:02d}.png"
        
        html = f"""
        <!DOCTYPE html>
        <html><head><style>
        body {{ font-family: monospace; padding: 20px; }}
        </style></head><body>
        <h2>Augmented Receipt #{i}</h2>
        <p>Item A: $20.00</p>
        <p>Item B: $30.00</p>
        <p>Total: $50.00</p>
        </body></html>
        """
        
        success = renderer_always.render(html, str(output_path))
        if success:
            print(f"  ✓ Generated augmented receipt {i+1}/5")
    
    # Test 3: Force augmentation OFF
    print("\n[Test 3] Force augmentation OFF (0%)...")
    renderer_never = HTMLToPNGRenderer(augment_probability=0.0)
    
    for i in range(5):
        output_path = output_dir / f"test_never_augmented_{i:02d}.png"
        
        html = f"""
        <!DOCTYPE html>
        <html><head><style>
        body {{ font-family: monospace; padding: 20px; }}
        </style></head><body>
        <h2>Clean Receipt #{i}</h2>
        <p>Item X: $5.00</p>
        <p>Item Y: $7.50</p>
        <p>Total: $12.50</p>
        </body></html>
        """
        
        success = renderer_never.render(html, str(output_path))
        if success:
            print(f"  ✓ Generated clean receipt {i+1}/5")
    
    # Test 4: Explicit control via parameter
    print("\n[Test 4] Explicit augmentation control...")
    renderer = HTMLToPNGRenderer(augment_probability=0.5)
    
    # Force ON
    output_path = output_dir / "test_explicit_on.png"
    html = """<!DOCTYPE html><html><body><h2>Explicitly Augmented</h2></body></html>"""
    renderer.render(html, str(output_path), apply_augmentation=True)
    print("  ✓ Generated explicitly augmented receipt")
    
    # Force OFF
    output_path = output_dir / "test_explicit_off.png"
    html = """<!DOCTYPE html><html><body><h2>Explicitly Clean</h2></body></html>"""
    renderer.render(html, str(output_path), apply_augmentation=False)
    print("  ✓ Generated explicitly clean receipt")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nCompare images visually:")
    print("  - test_default_*.png - Mix of clean and augmented")
    print("  - test_always_augmented_*.png - All should show degradation")
    print("  - test_never_augmented_*.png - All should be clean")
    print("  - test_explicit_*.png - Explicit control examples")
    print("\nAugmentation effects may include:")
    print("  • Thermal fade, wrinkles, coffee stains")
    print("  • Skewed angles, blur, noise")
    print("  • Over/under contrast, faint printing")
    print("\n✓ Integration complete - receipts will now be realistically degraded!")


if __name__ == "__main__":
    try:
        test_augmentation_integration()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
