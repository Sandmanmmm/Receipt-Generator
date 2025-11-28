"""
Quick visual test for augmentation - generates 2 receipts with 100% augmentation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.html_to_png_renderer import SimplePNGRenderer
from generators.retail_data_generator import RetailDataGenerator

def main():
    print("="*80)
    print("AUGMENTATION VISUAL TEST - 100% Augmentation")
    print("="*80)
    
    # Create output directory
    output_dir = Path("outputs/augmentation_visual_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create renderer with 100% augmentation
    renderer = SimplePNGRenderer(width=800, height=1200, augment_probability=1.0)
    
    # Create data generator
    data_gen = RetailDataGenerator()
    
    print("\nGenerating 2 receipts with 100% augmentation...")
    
    for i in range(1, 3):
        print(f"\n  Sample {i}/2...", end='')
        
        # Simple text content
        text_lines = []
        text_lines.append(f"{'TEST STORE':^80}")
        text_lines.append(f"{'123 Main Street':^80}")
        text_lines.append("="*80)
        text_lines.append(f"Date: 2025-11-27")
        text_lines.append("")
        text_lines.append(f"{'Item 1':<50} {'$10.00':>10}")
        text_lines.append(f"{'Item 2':<50} {'$20.00':>10}")
        text_lines.append(f"{'Item 3':<50} {'$15.00':>10}")
        text_lines.append("="*80)
        text_lines.append(f"{'TOTAL:':<50} {'$45.00':>10}")
        text_lines.append("="*80)
        
        # Render with 100% augmentation
        output_path = output_dir / f"augmented_{i:03d}.png"
        success = renderer.render_text_receipt(
            text_lines=text_lines,
            output_path=str(output_path),
            receipt_type='retail'
        )
        
        if success:
            print(" ✓")
        else:
            print(" ✗ FAILED")
    
    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nAll receipts should show visible augmentation effects:")
    print("  - Wrinkles, stains, or fading")
    print("  - Slight rotation/skew")
    print("  - Noise or blur")
    print("\nOpening directory...")
    
    import subprocess
    subprocess.run(['explorer', str(output_dir.absolute())])

if __name__ == '__main__':
    main()
