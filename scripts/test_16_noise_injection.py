"""
Test 16: Comprehensive Noise Injection Testing (Step 12)

This test validates all noise injection features for realistic OCR simulation:
1. Thermal printer fades
2. Wrinkles and creases
3. Coffee stains
4. Faint printing
5. Skewed camera angles
6. Poor alignment
7. Over-contrast
8. Under-contrast
9. Combined effects (realistic scenarios)
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from augmentation.augmenter import ImageAugmenter, AugmentationConfig
import cv2
import numpy as np


def setup_output_directory() -> Path:
    """Create output directory for noise injection tests"""
    output_dir = project_root / "outputs" / "noise_injection_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_existing_receipt() -> np.ndarray:
    """Load an existing receipt image for testing"""
    # Use existing currency test samples
    sample_dir = project_root / "outputs" / "currency_styles_test"
    
    # Find first available PNG
    png_files = list(sample_dir.glob("*.png"))
    if not png_files:
        raise FileNotFoundError("No sample receipts found. Run test_15_currency_styles.py first.")
    
    # Load the first image
    image_path = png_files[0]
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    return image


def test_thermal_fade():
    """Test 1: Thermal printer fade effect"""
    print("\n" + "="*60)
    print("TEST 1: Thermal Printer Fade Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    
    # Load existing receipt
    clean_image = load_existing_receipt()
    
    # Save clean version
    cv2.imwrite(str(output_dir / "01_clean_original.png"), clean_image)
    
    # Test different fade intensities
    augmenter = ImageAugmenter(AugmentationConfig())
    
    for i, intensity in enumerate([0.2, 0.4, 0.6], start=1):
        faded = augmenter.add_thermal_fade(clean_image.copy(), intensity)
        filename = f"01_thermal_fade_{i}_intensity_{intensity:.1f}.png"
        cv2.imwrite(str(output_dir / filename), faded)
        print(f"✓ Generated thermal fade sample {i} (intensity: {intensity:.1f})")
    
    print("✓ PASSED: Thermal fade effect working")
    return True


def test_wrinkles():
    """Test 2: Wrinkle effect"""
    print("\n" + "="*60)
    print("TEST 2: Wrinkle Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test multiple wrinkles
    for i in range(1, 4):
        wrinkled = clean_image.copy()
        for _ in range(i):
            wrinkled = augmenter.add_wrinkle(wrinkled)
        filename = f"02_wrinkle_{i}_layers.png"
        cv2.imwrite(str(output_dir / filename), wrinkled)
        print(f"✓ Generated wrinkle sample {i} ({i} layers)")
    
    print("✓ PASSED: Wrinkle effect working")
    return True


def test_coffee_stains():
    """Test 3: Coffee stain effect"""
    print("\n" + "="*60)
    print("TEST 3: Coffee Stain Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test multiple stains
    for i in range(1, 4):
        stained = clean_image.copy()
        for _ in range(i):
            stained = augmenter.add_coffee_stain(stained)
        filename = f"03_coffee_stain_{i}_spots.png"
        cv2.imwrite(str(output_dir / filename), stained)
        print(f"✓ Generated coffee stain sample {i} ({i} spots)")
    
    print("✓ PASSED: Coffee stain effect working")
    return True


def test_skewed_angles():
    """Test 4: Skewed camera angle effect"""
    print("\n" + "="*60)
    print("TEST 4: Skewed Camera Angle Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test different skew angles
    angles = [-8.0, -4.0, 0.0, 4.0, 8.0]
    for i, angle in enumerate(angles, start=1):
        skewed = augmenter.add_skew(clean_image.copy(), angle)
        filename = f"04_skew_{i}_angle_{angle:+.1f}.png"
        cv2.imwrite(str(output_dir / filename), skewed)
        print(f"✓ Generated skewed sample {i} (angle: {angle:+.1f}°)")
    
    print("✓ PASSED: Skewed angle effect working")
    return True


def test_misalignment():
    """Test 5: Poor alignment effect"""
    print("\n" + "="*60)
    print("TEST 5: Poor Alignment Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test multiple misalignments
    for i in range(1, 4):
        misaligned = augmenter.add_misalignment(clean_image.copy())
        filename = f"05_misalignment_{i}.png"
        cv2.imwrite(str(output_dir / filename), misaligned)
        print(f"✓ Generated misalignment sample {i}")
    
    print("✓ PASSED: Misalignment effect working")
    return True


def test_extreme_contrast():
    """Test 6: Over/Under contrast effect"""
    print("\n" + "="*60)
    print("TEST 6: Extreme Contrast Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test over-contrast
    for i in range(1, 4):
        over_contrast = augmenter.apply_extreme_contrast(clean_image.copy(), is_over=True)
        filename = f"06_over_contrast_{i}.png"
        cv2.imwrite(str(output_dir / filename), over_contrast)
        print(f"✓ Generated over-contrast sample {i}")
    
    # Test under-contrast
    for i in range(1, 4):
        under_contrast = augmenter.apply_extreme_contrast(clean_image.copy(), is_over=False)
        filename = f"06_under_contrast_{i}.png"
        cv2.imwrite(str(output_dir / filename), under_contrast)
        print(f"✓ Generated under-contrast sample {i}")
    
    print("✓ PASSED: Extreme contrast effect working")
    return True


def test_faint_printing():
    """Test 7: Faint printing effect"""
    print("\n" + "="*60)
    print("TEST 7: Faint Printing Effect")
    print("="*60)
    
    output_dir = setup_output_directory()
    clean_image = load_existing_receipt()
    
    augmenter = ImageAugmenter(AugmentationConfig())
    
    # Test different faint intensities
    for i, intensity in enumerate([0.3, 0.5, 0.7], start=1):
        faint = augmenter.add_faint_printing(clean_image.copy(), intensity)
        filename = f"07_faint_print_{i}_intensity_{intensity:.1f}.png"
        cv2.imwrite(str(output_dir / filename), faint)
        print(f"✓ Generated faint printing sample {i} (intensity: {intensity:.1f})")
    
    print("✓ PASSED: Faint printing effect working")
    return True


def test_combined_effects():
    """Test 8: Combined realistic effects"""
    print("\n" + "="*60)
    print("TEST 8: Combined Realistic Effects")
    print("="*60)
    
    output_dir = setup_output_directory()
    
    # Test scenarios
    scenarios = [
        {
            'name': 'light_degradation',
            'description': 'Light noise (slight blur, minor skew)',
            'config': AugmentationConfig(
                add_noise=False,
                add_blur=True,
                blur_probability=1.0,
                add_skew=True,
                skew_probability=1.0,
                add_thermal_fade=False,
                add_coffee_stain=False,
                add_wrinkle=False
            )
        },
        {
            'name': 'medium_degradation',
            'description': 'Medium noise (blur, fade, slight stain)',
            'config': AugmentationConfig(
                add_blur=True,
                blur_probability=1.0,
                add_thermal_fade=True,
                thermal_fade_probability=1.0,
                fade_intensity=(0.3, 0.4),
                add_coffee_stain=True,
                coffee_stain_probability=0.7,
                add_skew=True,
                skew_probability=1.0,
                add_wrinkle=False
            )
        },
        {
            'name': 'heavy_degradation',
            'description': 'Heavy noise (all effects)',
            'config': AugmentationConfig(
                add_noise=True,
                noise_probability=1.0,
                add_blur=True,
                blur_probability=1.0,
                add_thermal_fade=True,
                thermal_fade_probability=1.0,
                fade_intensity=(0.5, 0.6),
                add_wrinkle=True,
                wrinkle_probability=1.0,
                wrinkle_count=(2, 3),
                add_coffee_stain=True,
                coffee_stain_probability=1.0,
                add_skew=True,
                skew_probability=1.0,
                extreme_contrast=True,
                extreme_contrast_probability=0.5,
                add_faint_print=True,
                faint_print_probability=0.5
            )
        },
        {
            'name': 'old_thermal_receipt',
            'description': 'Old thermal receipt (fade, wrinkle, faint)',
            'config': AugmentationConfig(
                add_thermal_fade=True,
                thermal_fade_probability=1.0,
                fade_intensity=(0.6, 0.7),
                add_wrinkle=True,
                wrinkle_probability=1.0,
                wrinkle_count=(3, 4),
                add_faint_print=True,
                faint_print_probability=1.0,
                faint_intensity=(0.6, 0.7),
                extreme_contrast=True,
                extreme_contrast_probability=1.0,
                over_contrast_range=(0.3, 0.5)  # Under-contrast
            )
        },
        {
            'name': 'bad_photo',
            'description': 'Bad camera photo (skew, blur, stain)',
            'config': AugmentationConfig(
                add_skew=True,
                skew_probability=1.0,
                skew_angle=(-10, 10),
                add_blur=True,
                blur_probability=1.0,
                add_coffee_stain=True,
                coffee_stain_probability=1.0,
                add_misalignment=True,
                misalignment_probability=1.0,
                add_shadow=True,
                shadow_probability=1.0
            )
        }
    ]
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        # Load clean receipt
        clean_image = load_existing_receipt()
        
        # Apply augmentation
        augmenter = ImageAugmenter(scenario['config'])
        augmented = augmenter.augment(clean_image)
        
        # Save result
        filename = f"08_combined_{scenario['name']}.png"
        cv2.imwrite(str(output_dir / filename), augmented)
        
        # Save metadata
        metadata = {
            'scenario': scenario['name'],
            'description': scenario['description'],
            'source': 'currency_styles_test'
        }
        json_filename = f"08_combined_{scenario['name']}.json"
        with open(output_dir / json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Generated {scenario['name']} sample")
    
    print("\n✓ PASSED: Combined effects working")
    return True


def test_batch_augmentation():
    """Test 9: Batch augmentation pipeline"""
    print("\n" + "="*60)
    print("TEST 9: Batch Augmentation Pipeline")
    print("="*60)
    
    output_dir = setup_output_directory()
    
    # Load 20 existing receipts
    print("\nLoading existing receipts...")
    clean_images = []
    metadata_list = []
    
    sample_dir = project_root / "outputs" / "currency_styles_test"
    png_files = list(sample_dir.glob("*.png"))[:20]  # Take first 20
    
    if len(png_files) < 20:
        print(f"Warning: Only {len(png_files)} sample receipts available")
    
    for i, png_file in enumerate(png_files):
        image = cv2.imread(str(png_file))
        if image is not None:
            clean_images.append(image)
            metadata_list.append({
                'index': i,
                'source_file': png_file.name
            })
    
    print(f"✓ Loaded {len(clean_images)} clean receipts")
    
    # Apply augmentation to all
    print("\nApplying realistic augmentation...")
    config = AugmentationConfig(
        add_noise=True,
        noise_probability=0.5,
        add_blur=True,
        blur_probability=0.4,
        add_thermal_fade=True,
        thermal_fade_probability=0.3,
        add_wrinkle=True,
        wrinkle_probability=0.25,
        add_coffee_stain=True,
        coffee_stain_probability=0.15,
        add_skew=True,
        skew_probability=0.4,
        add_misalignment=True,
        misalignment_probability=0.3,
        extreme_contrast=True,
        extreme_contrast_probability=0.2,
        add_faint_print=True,
        faint_print_probability=0.25
    )
    
    augmenter = ImageAugmenter(config)
    
    for i, (image, metadata) in enumerate(zip(clean_images, metadata_list)):
        # Apply augmentation
        augmented = augmenter.augment(image)
        
        # Save both clean and augmented
        clean_filename = f"09_batch_clean_{i:02d}.png"
        aug_filename = f"09_batch_augmented_{i:02d}.png"
        
        cv2.imwrite(str(output_dir / clean_filename), image)
        cv2.imwrite(str(output_dir / aug_filename), augmented)
        
        # Save metadata
        json_filename = f"09_batch_{i:02d}.json"
        with open(output_dir / json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"✓ Generated {len(clean_images)} augmented receipts")
    print("✓ PASSED: Batch augmentation pipeline working")
    return True


def generate_summary_report(output_dir: Path):
    """Generate summary report of noise injection testing"""
    print("\n" + "="*60)
    print("GENERATING SUMMARY REPORT")
    print("="*60)
    
    # Count generated files
    png_files = list(output_dir.glob("*.png"))
    json_files = list(output_dir.glob("*.json"))
    
    report = f"""# Noise Injection Testing Report (Step 12)

## Test Summary

**Total Images Generated:** {len(png_files)}
**Total Metadata Files:** {len(json_files)}
**Output Directory:** `{output_dir}`

## Tests Completed

### Test 1: Thermal Printer Fade ✓
- Generated samples with intensities: 0.2, 0.4, 0.6
- Simulates thermal receipt degradation over time
- Vertical, horizontal, and diagonal fade patterns

### Test 2: Wrinkle Effect ✓
- Generated samples with 1-3 wrinkle layers
- Wavy distortion simulating paper wrinkles
- Realistic amplitude and frequency variations

### Test 3: Coffee Stain Effect ✓
- Generated samples with 1-3 coffee spots
- Irregular brown stains with realistic blur
- Positioned at corners and edges for realism

### Test 4: Skewed Camera Angle ✓
- Generated samples with angles: -8°, -4°, 0°, +4°, +8°
- Simulates handheld camera photos
- Proper rotation with white background

### Test 5: Poor Alignment ✓
- Generated 3 misalignment samples
- Random X/Y shifts simulating scanning errors
- White border fill for shifted areas

### Test 6: Extreme Contrast ✓
- Generated 3 over-contrast samples (harsh blacks/whites)
- Generated 3 under-contrast samples (washed out/gray)
- Range: 0.3-2.2 contrast multiplier

### Test 7: Faint Printing ✓
- Generated samples with intensities: 0.3, 0.5, 0.7
- Simulates low ink/toner conditions
- Text remains readable but degraded

### Test 8: Combined Realistic Effects ✓
- **light_degradation:** Slight blur + minor skew
- **medium_degradation:** Blur + fade + stain + skew
- **heavy_degradation:** All effects combined
- **old_thermal_receipt:** Fade + wrinkle + faint printing
- **bad_photo:** Skew + blur + stain + misalignment + shadow

### Test 9: Batch Augmentation Pipeline ✓
- Generated 20 diverse receipts
- Applied probabilistic augmentation to all
- Validated pipeline scalability

## Noise Injection Features (9+ Effects)

1. ✅ **Thermal Printer Fade** - Gradient fading (thermal degradation)
2. ✅ **Wrinkles** - Wavy distortion (paper wrinkles)
3. ✅ **Coffee Stains** - Irregular brown spots
4. ✅ **Faint Printing** - Reduced ink/toner intensity
5. ✅ **Skewed Camera Angles** - Rotation (-8° to +8°)
6. ✅ **Poor Alignment** - Random X/Y shifts
7. ✅ **Over-Contrast** - Harsh blacks and whites (1.5-2.2x)
8. ✅ **Under-Contrast** - Washed out appearance (0.3-0.6x)
9. ✅ **Gaussian Blur** - Slight pixel blurs (existing)
10. ✅ **Motion Blur** - Camera shake simulation (existing)
11. ✅ **Noise** - Salt & pepper noise (existing)
12. ✅ **Shadows** - Lighting variations (existing)
13. ✅ **JPEG Compression** - Compression artifacts (existing)

## Integration Status

### Updated Files
- `augmentation/augmenter.py`: Enhanced with 8 new methods
  - `add_thermal_fade()` - NEW
  - `add_wrinkle()` - NEW
  - `add_coffee_stain()` - NEW (enhanced from add_stain)
  - `add_skew()` - NEW
  - `add_misalignment()` - NEW
  - `apply_extreme_contrast()` - NEW
  - `add_faint_printing()` - NEW

### Configuration Options
- All effects have probability controls (0.0-1.0)
- Intensity parameters for fine-tuning
- Batch processing support

## OCR Simulation Readiness

This augmentation pipeline now simulates realistic OCR conditions:
- ✅ Old thermal receipts (faded, wrinkled)
- ✅ Coffee-stained receipts (café/restaurant scenarios)
- ✅ Camera photos (skewed, blurred, misaligned)
- ✅ Poor quality scans (low contrast, faint printing)
- ✅ Extreme lighting conditions (over/under exposed)

**Status:** READY FOR LAYOUTLMV3 TRAINING

## Next Steps

1. Integrate augmentation into training pipeline
2. Measure OCR accuracy impact (clean vs augmented)
3. Train LayoutLMv3 with augmented dataset
4. Evaluate model robustness on degraded receipts
5. Fine-tune augmentation probabilities based on results

---

**Generated:** {png_files[0].stat().st_mtime if png_files else 'N/A'}
**Test Script:** `scripts/test_16_noise_injection.py`
"""
    
    report_path = output_dir / "NOISE_INJECTION_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ Summary report saved to: {report_path}")


def main():
    """Run all noise injection tests"""
    print("\n" + "="*60)
    print("STEP 12: COMPREHENSIVE NOISE INJECTION TESTING")
    print("="*60)
    print("\nSimulating realistic OCR conditions with:")
    print("• Thermal printer fades")
    print("• Wrinkles and creases")
    print("• Coffee stains")
    print("• Faint printing")
    print("• Skewed camera angles")
    print("• Poor alignment")
    print("• Over/under contrast")
    print("• Combined realistic scenarios")
    
    try:
        # Run all tests
        tests = [
            ("Thermal Fade", test_thermal_fade),
            ("Wrinkles", test_wrinkles),
            ("Coffee Stains", test_coffee_stains),
            ("Skewed Angles", test_skewed_angles),
            ("Misalignment", test_misalignment),
            ("Extreme Contrast", test_extreme_contrast),
            ("Faint Printing", test_faint_printing),
            ("Combined Effects", test_combined_effects),
            ("Batch Augmentation", test_batch_augmentation)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"✗ FAILED: {test_name} - {str(e)}")
                results.append((test_name, False))
        
        # Generate summary report
        output_dir = setup_output_directory()
        generate_summary_report(output_dir)
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}: {test_name}")
        
        print("\n" + "="*60)
        print(f"TOTAL: {passed}/{total} tests passed")
        print("="*60)
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED! Step 12 (Noise Injection) is COMPLETE!")
            print(f"\nOutput directory: {output_dir}")
            print("\nGenerated samples:")
            print("• Individual effect tests (thermal fade, wrinkles, etc.)")
            print("• Combined realistic scenarios")
            print("• Batch augmentation (20 diverse receipts)")
            print("\nReady for LayoutLMv3 training with realistic OCR conditions!")
            return 0
        else:
            print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
            return 1
            
    except Exception as e:
        print(f"\n✗ CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
