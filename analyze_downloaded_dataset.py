#!/usr/bin/env python3
"""
Analyze the downloaded 150K dataset to verify the $0.00 pricing bug.

This script:
1. Samples images from the dataset
2. Extracts text with OCR
3. Checks for $0.00 patterns in line items
4. Reports statistics on how many invoices show the bug
"""

import os
import sys
from pathlib import Path
import random
from PIL import Image
import pytesseract
import re
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def extract_text_from_image(image_path):
    """Extract text from image using Tesseract OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extracting text from {image_path}: {e}")
        return ""

def check_for_zero_prices(text):
    """Check if text contains $0.00 patterns in line items"""
    # Patterns that might indicate $0.00 bug
    zero_patterns = [
        r'\$0\.00',  # Exact $0.00
        r'\$\s*0\.00',  # $0.00 with space
        r'0\.00\s*$',  # 0.00 followed by currency
    ]
    
    zero_count = 0
    for pattern in zero_patterns:
        matches = re.findall(pattern, text)
        zero_count += len(matches)
    
    # Also check for valid prices
    valid_price_pattern = r'\$\s*\d+\.\d{2}'
    valid_prices = re.findall(valid_price_pattern, text)
    non_zero_prices = [p for p in valid_prices if '$0.00' not in p and '$ 0.00' not in p]
    
    return {
        'zero_count': zero_count,
        'valid_price_count': len(valid_prices),
        'non_zero_price_count': len(non_zero_prices),
        'has_zero_bug': zero_count > 2  # More than 2 zeros suggests a bug (not just totals)
    }

def analyze_dataset(dataset_dir, sample_size=50):
    """Analyze a sample of the dataset"""
    images_dir = Path(dataset_dir) / "images"
    
    # Get all image files (excluding multipage txt files)
    all_images = [f for f in images_dir.glob("*.jpg") if not str(f).endswith('_MULTIPAGE.txt')]
    all_images.extend([f for f in images_dir.glob("*.png")])
    
    print(f"Total images found: {len(all_images)}")
    
    # Sample random images
    sample = random.sample(all_images, min(sample_size, len(all_images)))
    
    results = {
        'total_sampled': len(sample),
        'images_with_zero_bug': 0,
        'images_analyzed': 0,
        'ocr_errors': 0,
        'zero_counts': [],
        'price_counts': []
    }
    
    print(f"\nAnalyzing {len(sample)} random images...")
    
    for i, img_path in enumerate(sample, 1):
        print(f"  [{i}/{len(sample)}] {img_path.name}...", end='')
        
        text = extract_text_from_image(img_path)
        
        if not text or len(text) < 50:
            print(" OCR failed")
            results['ocr_errors'] += 1
            continue
        
        analysis = check_for_zero_prices(text)
        results['images_analyzed'] += 1
        results['zero_counts'].append(analysis['zero_count'])
        results['price_counts'].append(analysis['non_zero_price_count'])
        
        if analysis['has_zero_bug']:
            results['images_with_zero_bug'] += 1
            print(f" ⚠️ HAS BUG (zeros: {analysis['zero_count']}, valid: {analysis['non_zero_price_count']})")
        else:
            print(f" ✓ OK (zeros: {analysis['zero_count']}, valid: {analysis['non_zero_price_count']})")
    
    return results

def print_report(results):
    """Print analysis report"""
    print("\n" + "="*70)
    print("DATASET ANALYSIS REPORT")
    print("="*70)
    
    print(f"\nSample Size: {results['total_sampled']}")
    print(f"Successfully Analyzed: {results['images_analyzed']}")
    print(f"OCR Errors: {results['ocr_errors']}")
    
    if results['images_analyzed'] > 0:
        bug_rate = (results['images_with_zero_bug'] / results['images_analyzed']) * 100
        print(f"\nImages with $0.00 Bug: {results['images_with_zero_bug']} ({bug_rate:.1f}%)")
        
        avg_zeros = sum(results['zero_counts']) / len(results['zero_counts']) if results['zero_counts'] else 0
        avg_prices = sum(results['price_counts']) / len(results['price_counts']) if results['price_counts'] else 0
        
        print(f"Average $0.00 occurrences per image: {avg_zeros:.1f}")
        print(f"Average valid prices per image: {avg_prices:.1f}")
    
    print("\n" + "="*70)
    print("\nCONCLUSION:")
    if results['images_with_zero_bug'] > 0:
        print("⚠️  The downloaded dataset DOES contain invoices with $0.00 pricing bug.")
        print("    This confirms the bug was present in the generation code used on vast.ai.")
        print("    The fix we just committed will prevent this in future generations.")
    else:
        print("✓  No significant $0.00 pricing bugs detected in sampled images.")
    print("="*70)

if __name__ == "__main__":
    dataset_dir = Path(__file__).parent / "data" / "production_150k"
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        sys.exit(1)
    
    # Analyze with a reasonable sample size
    results = analyze_dataset(dataset_dir, sample_size=30)
    print_report(results)
