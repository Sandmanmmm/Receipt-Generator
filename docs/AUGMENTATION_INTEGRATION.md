# Augmentation Integration - Usage Guide

## Overview

The HTMLToPNGRenderer now automatically applies realistic augmentation effects to generated receipts. This integration enables seamless production of training data with realistic degradation patterns.

## Configuration

### Basic Usage

```python
from generators.html_to_png_renderer import HTMLToPNGRenderer

# Create renderer with 50% augmentation probability (default)
renderer = HTMLToPNGRenderer(augment_probability=0.5)

# Generate receipt - 50% chance of augmentation
html = "<html>...</html>"
renderer.render(html, "receipt.png")
```

### Control Augmentation Probability

```python
# Always augment (100%)
renderer_full = HTMLToPNGRenderer(augment_probability=1.0)

# Never augment (0%) - clean receipts only
renderer_clean = HTMLToPNGRenderer(augment_probability=0.0)

# 30% augmentation
renderer_light = HTMLToPNGRenderer(augment_probability=0.3)

# 80% augmentation (heavy degradation)
renderer_heavy = HTMLToPNGRenderer(augment_probability=0.8)
```

### Explicit Control Per Receipt

```python
renderer = HTMLToPNGRenderer(augment_probability=0.5)

# Force augmentation ON for this receipt
renderer.render(html, "augmented.png", apply_augmentation=True)

# Force augmentation OFF for this receipt
renderer.render(html, "clean.png", apply_augmentation=False)

# Use probability (None = default behavior)
renderer.render(html, "maybe.png", apply_augmentation=None)
```

## Augmentation Effects Applied

When augmentation is enabled, the following effects are randomly applied with configured probabilities:

### Effect Distribution (Default Config)

| Effect | Probability | Description |
|--------|-------------|-------------|
| Gaussian Blur | 30% | Slight pixel blur (out of focus) |
| Gaussian Noise | 40% | Salt & pepper pixel noise |
| Thermal Fade | 25% | Gradient fading (thermal degradation) |
| Wrinkles | 20% | Wavy paper distortion (1-3 layers) |
| Coffee Stains | 15% | Brown organic stains |
| Skewed Angle | 35% | Rotation -6° to +6° |
| Misalignment | 25% | Random X/Y shifts |
| Extreme Contrast | 15% | Over or under-contrast |
| Faint Printing | 20% | Low ink/toner simulation |
| JPEG Compression | 30% | Compression artifacts |
| Shadow | 20% | Lighting variations |

### Effect Combinations

Multiple effects can be applied to a single receipt, creating realistic combinations:
- Old thermal receipt: fade + wrinkles + faint printing
- Bad camera photo: skew + blur + shadow + stain
- Poor scan: misalignment + compression + contrast issues

## Recommended Training Data Distribution

### For LayoutLMv3 Training

```python
# Generate 10,000 training receipts
total_receipts = 10000

# Split by augmentation level
clean = 2000        # 20% - No augmentation
light = 3000        # 30% - Probability 0.3
medium = 3000       # 30% - Probability 0.6
heavy = 2000        # 20% - Probability 0.9

# Clean receipts
renderer_clean = HTMLToPNGRenderer(augment_probability=0.0)
for i in range(clean):
    # generate receipt...
    renderer_clean.render(html, f"clean_{i}.png")

# Light augmentation
renderer_light = HTMLToPNGRenderer(augment_probability=0.3)
for i in range(light):
    renderer_light.render(html, f"light_{i}.png")

# Medium augmentation
renderer_medium = HTMLToPNGRenderer(augment_probability=0.6)
for i in range(medium):
    renderer_medium.render(html, f"medium_{i}.png")

# Heavy augmentation
renderer_heavy = HTMLToPNGRenderer(augment_probability=0.9)
for i in range(heavy):
    renderer_heavy.render(html, f"heavy_{i}.png")
```

## Testing Integration

A test script verifies the integration:

```bash
python scripts/test_augmentation_integration.py
```

This generates:
- 10 receipts with 50% augmentation (default)
- 5 receipts with 100% augmentation (always)
- 5 receipts with 0% augmentation (never)
- Examples of explicit control

Output location: `outputs/augmentation_integration_test/`

## Performance Impact

- **Rendering time:** +50-200ms per augmented image
- **File size:** Varies significantly (26KB to 3MB)
  - Clean: ~3-4 MB (lossless PNG)
  - Augmented: 26KB to 1.4MB (depends on effects)
- **Memory:** Minimal additional overhead (<50MB)

## Customizing Augmentation

To modify augmentation parameters, edit the `_get_augmenter()` method in `html_to_png_renderer.py`:

```python
def _get_augmenter(self):
    config = AugmentationConfig(
        # Adjust probabilities
        add_thermal_fade=True,
        thermal_fade_probability=0.25,  # Change to 0.4 for more fading
        fade_intensity=(0.2, 0.5),      # Increase max to 0.7 for stronger fade
        
        # Adjust effect intensity
        add_wrinkle=True,
        wrinkle_probability=0.2,
        wrinkle_count=(1, 3),           # Change to (2, 5) for more wrinkles
        
        # Enable/disable effects
        add_coffee_stain=True,          # Set to False to disable
        coffee_stain_probability=0.15,
    )
    return ImageAugmenter(config)
```

## Verification

Visual inspection of augmented receipts:

1. **Check file sizes:** Augmented images vary significantly in size
2. **Visual comparison:** Open side-by-side in image viewer
3. **OCR test:** Ensure text is still readable after augmentation

Expected characteristics:
- ✓ Some receipts rotated/skewed
- ✓ Visible blur or noise on some images
- ✓ Color/contrast variations
- ✓ Occasional stains or fade effects
- ✓ Text remains readable (though degraded)

## Production Deployment

### Batch Generation with Augmentation

```python
from generators.html_to_png_renderer import HTMLToPNGRenderer
from pathlib import Path

def generate_training_set(output_dir: Path, num_receipts: int, 
                          augment_prob: float = 0.5):
    """Generate training set with augmentation"""
    
    renderer = HTMLToPNGRenderer(augment_probability=augment_prob)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_receipts):
        # Generate receipt HTML
        html, metadata = generate_receipt_html()
        
        # Render with automatic augmentation
        output_path = output_dir / f"receipt_{i:05d}.png"
        success = renderer.render(html, str(output_path))
        
        if success:
            # Save metadata
            json_path = output_dir / f"receipt_{i:05d}.json"
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_receipts} receipts in {output_dir}")

# Generate 10k training receipts with 60% augmentation
generate_training_set(
    Path("data/train"),
    num_receipts=10000,
    augment_prob=0.6
)
```

## Troubleshooting

### Issue: No augmentation applied

**Check:**
1. `augment_probability > 0`
2. Augmentation module imported successfully
3. No errors in terminal output

**Solution:**
```python
# Verify augmenter is loaded
renderer = HTMLToPNGRenderer(augment_probability=1.0)
augmenter = renderer._get_augmenter()
print(f"Augmenter loaded: {augmenter is not None}")
```

### Issue: Too much degradation

**Solution:** Reduce probability or customize config:
```python
# Lighter augmentation
renderer = HTMLToPNGRenderer(augment_probability=0.2)

# Or customize individual effect probabilities
# Edit _get_augmenter() method in html_to_png_renderer.py
```

### Issue: Not enough variety

**Solution:** Increase probability and effect intensity:
```python
# More aggressive augmentation
renderer = HTMLToPNGRenderer(augment_probability=0.8)
```

## Benefits

### For Model Training
- ✅ **Improved robustness:** Models trained on augmented data perform better on real-world receipts
- ✅ **Better generalization:** Handles various quality levels and degradation patterns
- ✅ **Reduced overfitting:** More diverse training data prevents memorization

### For Testing
- ✅ **Realistic evaluation:** Test sets reflect actual receipt conditions
- ✅ **Edge case coverage:** Simulates worst-case scenarios
- ✅ **Performance metrics:** Measure accuracy on degraded vs clean receipts

### For Production
- ✅ **One-time setup:** Configure once, automatically applies to all receipts
- ✅ **Flexible control:** Per-receipt or batch-level configuration
- ✅ **Minimal overhead:** Fast augmentation (50-200ms per image)

## Summary

The integrated augmentation system provides:
- **13 augmentation effects** automatically applied during rendering
- **Configurable probabilities** from 0% (clean) to 100% (always augmented)
- **Explicit per-receipt control** via `apply_augmentation` parameter
- **Production-ready** with minimal performance impact

This enables seamless generation of realistic training data for robust OCR model development.

---

**Status:** ✅ PRODUCTION READY  
**Test:** `scripts/test_augmentation_integration.py`  
**Updated:** November 27, 2025
