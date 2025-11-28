"""
Side-by-side comparison: Clean vs Augmented
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.html_to_png_renderer import SimplePNGRenderer

def main():
    print("="*80)
    print("CLEAN vs AUGMENTED COMPARISON")
    print("="*80)
    
    # Create output directory
    output_dir = Path("outputs/augmentation_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Same receipt content
    text_lines = []
    text_lines.append(f"{'COMPARISON TEST STORE':^80}")
    text_lines.append(f"{'456 Test Avenue, Suite 100':^80}")
    text_lines.append(f"{'Phone: (555) 123-4567':^80}")
    text_lines.append("="*80)
    text_lines.append(f"Date: 2025-11-27          Receipt #: TEST-001")
    text_lines.append("")
    text_lines.append(f"{'Item Description':<50} {'Amount':>10}")
    text_lines.append("-"*80)
    text_lines.append(f"{'Wireless Mouse - Black':<50} {'$29.99':>10}")
    text_lines.append(f"{'USB-C Cable - 6ft':<50} {'$12.99':>10}")
    text_lines.append(f"{'Laptop Stand - Aluminum':<50} {'$45.99':>10}")
    text_lines.append(f"{'Screen Cleaning Kit':<50} {'$8.99':>10}")
    text_lines.append(f"{'Notebook Set - 3pack':<50} {'$15.99':>10}")
    text_lines.append("-"*80)
    text_lines.append(f"{'Subtotal:':<50} {'$113.95':>10}")
    text_lines.append(f"{'Tax (8.5%):':<50} {'$9.69':>10}")
    text_lines.append("="*80)
    text_lines.append(f"{'TOTAL:':<50} {'$123.64':>10}")
    text_lines.append("="*80)
    text_lines.append(f"{'Payment: VISA ****1234':^80}")
    text_lines.append(f"{'Thank you for your business!':^80}")
    
    print("\n1. Generating CLEAN receipt (0% augmentation)...")
    renderer_clean = SimplePNGRenderer(width=800, height=1200, augment_probability=0.0)
    renderer_clean.render_text_receipt(
        text_lines=text_lines,
        output_path=str(output_dir / "CLEAN_no_augmentation.png"),
        receipt_type='retail'
    )
    print("   ✓ Clean receipt saved")
    
    print("\n2. Generating AUGMENTED receipt (100% augmentation)...")
    renderer_augmented = SimplePNGRenderer(width=800, height=1200, augment_probability=1.0)
    renderer_augmented.render_text_receipt(
        text_lines=text_lines,
        output_path=str(output_dir / "AUGMENTED_with_effects.png"),
        receipt_type='retail'
    )
    print("   ✓ Augmented receipt saved")
    
    print("\n" + "="*80)
    print("COMPARISON READY!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles generated:")
    print("  1. CLEAN_no_augmentation.png   - Original, pristine")
    print("  2. AUGMENTED_with_effects.png  - With degradation effects")
    print("\nOpen both images side-by-side to see the difference!")
    print("Look for:")
    print("  - Slight rotation/skew")
    print("  - Noise or grain")
    print("  - Wrinkles or folds")
    print("  - Stains or discoloration")
    print("  - Fading or contrast changes")
    
    import subprocess
    subprocess.run(['explorer', str(output_dir.absolute())])

if __name__ == '__main__':
    main()
