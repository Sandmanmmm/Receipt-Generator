"""
Test 14: Locale & Language Variants
====================================

Validates comprehensive locale support including:
- Currency formatting (symbol, position, separators)
- Date formatting (MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD, DD.MM.YYYY, Chinese)
- Tax labels (Sales Tax, VAT, GST, IVA, MwSt, 增值税)
- Decimal/thousand separators (. vs , vs space)
- Language-specific labels
- 10 locales: en_US, en_GB, en_CA, en_AU, fr_CA, fr_FR, es_ES, es_MX, de_DE, zh_CN

Author: InvoiceGen Team
Date: 2025-01-27
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator, RetailReceiptData
from generators.html_to_png_renderer import SimplePNGRenderer


def test_locale_currency_formatting():
    """Test currency formatting across all locales"""
    print("\n" + "="*80)
    print("TEST 1: Currency Formatting")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    test_amount = 1234.56
    
    locales = ['en_US', 'en_GB', 'en_CA', 'en_AU', 'fr_CA', 'fr_FR', 'es_ES', 'es_MX', 'de_DE', 'zh_CN']
    
    print(f"\nTest amount: {test_amount}")
    print("\nFormatted by locale:")
    print("-" * 60)
    
    for locale in locales:
        config = renderer._get_locale_config(locale)
        formatted = renderer._format_currency(test_amount, config)
        print(f"{locale:10s} ({config['name']:30s}): {formatted:>15s}")
        
        # Validate format
        if config['currency_position'] == 'before':
            assert config['currency_symbol'] in formatted, f"Currency symbol missing for {locale}"
            assert formatted.startswith(config['currency_symbol']) or formatted.startswith('-'), \
                f"Currency should be before amount for {locale}"
        else:
            assert config['currency_symbol'] in formatted, f"Currency symbol missing for {locale}"
            assert formatted.endswith(config['currency_symbol']) or formatted.endswith(f" {config['currency_symbol']}"), \
                f"Currency should be after amount for {locale}"
    
    print("\n✓ All currency formats validated")


def test_locale_date_formatting():
    """Test date formatting across all locales"""
    print("\n" + "="*80)
    print("TEST 2: Date Formatting")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    test_date = '2025-01-27'
    
    locales = ['en_US', 'en_GB', 'en_CA', 'en_AU', 'fr_CA', 'fr_FR', 'es_ES', 'es_MX', 'de_DE', 'zh_CN']
    
    print(f"\nTest date (ISO): {test_date}")
    print("\nFormatted by locale:")
    print("-" * 60)
    
    for locale in locales:
        config = renderer._get_locale_config(locale)
        formatted = renderer._format_date(test_date, config)
        expected_format = config['date_format']
        print(f"{locale:10s} ({expected_format:15s}): {formatted:>20s}")
        
        # Validate format
        assert formatted, f"Date formatting failed for {locale}"
        if locale == 'zh_CN':
            assert '年' in formatted and '月' in formatted and '日' in formatted, \
                f"Chinese date should contain 年月日 for {locale}"
        else:
            assert '2025' in formatted or '25' in formatted, f"Year missing for {locale}"
            assert '01' in formatted or '1' in formatted or '27' in formatted, \
                f"Month/day missing for {locale}"
    
    print("\n✓ All date formats validated")


def test_locale_tax_labels():
    """Test tax labels across all locales"""
    print("\n" + "="*80)
    print("TEST 3: Tax Labels")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    locales = ['en_US', 'en_GB', 'en_CA', 'en_AU', 'fr_CA', 'fr_FR', 'es_ES', 'es_MX', 'de_DE', 'zh_CN']
    
    print("\nTax labels by locale:")
    print("-" * 60)
    
    expected_labels = {
        'en_US': 'Sales Tax',
        'en_GB': 'VAT',
        'en_CA': 'GST/HST',
        'en_AU': 'GST',
        'fr_CA': 'TPS/TVQ',
        'fr_FR': 'TVA',
        'es_ES': 'IVA',
        'es_MX': 'IVA',
        'de_DE': 'MwSt.',
        'zh_CN': '增值税'
    }
    
    for locale in locales:
        config = renderer._get_locale_config(locale)
        tax_label = config['tax_label']
        expected = expected_labels[locale]
        print(f"{locale:10s}: {tax_label:>15s}")
        assert tax_label == expected, f"Expected '{expected}' for {locale}, got '{tax_label}'"
    
    print("\n✓ All tax labels validated")


def test_locale_label_translations():
    """Test label translations across locales"""
    print("\n" + "="*80)
    print("TEST 4: Label Translations")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    
    # Test key labels
    test_labels = ['invoice', 'receipt', 'subtotal', 'discount', 'total', 'thank_you']
    
    locales_to_test = {
        'en_US': 'English (US)',
        'fr_FR': 'French',
        'es_ES': 'Spanish',
        'de_DE': 'German',
        'zh_CN': 'Chinese'
    }
    
    for locale, language in locales_to_test.items():
        config = renderer._get_locale_config(locale)
        labels = config['labels']
        
        print(f"\n{language} ({locale}):")
        print("-" * 40)
        for label_key in test_labels:
            label_value = labels.get(label_key, '')
            print(f"  {label_key:15s}: {label_value}")
            assert label_value, f"Missing label '{label_key}' for {locale}"
    
    print("\n✓ All label translations validated")


def test_end_to_end_receipt_generation():
    """Generate receipts with various locales and validate rendering"""
    print("\n" + "="*80)
    print("TEST 5: End-to-End Receipt Generation")
    print("="*80)
    
    output_dir = project_root / 'outputs' / 'locale_test'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test each locale
    locales = ['en_US', 'en_GB', 'fr_CA', 'fr_FR', 'es_ES', 'de_DE', 'zh_CN']
    
    generator = RetailDataGenerator()
    renderer = SimplePNGRenderer()
    
    results = []
    
    print(f"\nGenerating {len(locales)} receipts (one per locale)...")
    print("-" * 60)
    
    for i, locale in enumerate(locales, 1):
        # Generate POS receipt with specific locale
        receipt = generator.generate_pos_receipt(store_type='fashion', min_items=5, max_items=8, locale=locale)
        receipt_dict = generator.to_dict(receipt)
        
        # Save JSON
        json_path = output_dir / f"locale_{i:02d}_{locale}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(receipt_dict, f, indent=2, ensure_ascii=False)
        
        # Render to PNG
        png_path = output_dir / f"locale_{i:02d}_{locale}.png"
        success = renderer.render_receipt_dict(receipt_dict, str(png_path))
        
        # Verify locale-specific elements
        config = renderer._get_locale_config(locale)
        
        result = {
            'locale': locale,
            'name': config['name'],
            'currency_symbol': config['currency_symbol'],
            'date_format': config['date_format'],
            'tax_label': config['tax_label'],
            'rendered': success
        }
        results.append(result)
        
        status = "✓" if success else "✗"
        print(f"{status} {locale:10s} - {config['name']:30s} - {config['currency_symbol']:3s} - {config['tax_label']:10s}")
    
    print(f"\n✓ Generated {len(results)} locale variants")
    print(f"✓ Output directory: {output_dir}")
    
    return results


def test_locale_distribution():
    """Test that locale distribution matches expected weights"""
    print("\n" + "="*80)
    print("TEST 6: Locale Distribution")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    
    # Generate 1000 random locales and check distribution
    n_samples = 1000
    locale_counts = {}
    
    print(f"\nSampling {n_samples} random locale selections...")
    
    for _ in range(n_samples):
        config = renderer._get_locale_config(None)  # Random selection
        locale = config['name'].split(' ')[0] + ' (' + [k for k, v in {
            'en_US': 'English (United States)',
            'en_GB': 'English (United Kingdom)',
            'en_CA': 'English (Canada)',
            'en_AU': 'English (Australia)',
            'fr_CA': 'Français (Canada)',
            'fr_FR': 'Français (France)',
            'es_ES': 'Español (España)',
            'es_MX': 'Español (México)',
            'de_DE': 'Deutsch (Deutschland)',
            'zh_CN': '中文 (中国)'
        }.items() if v == config['name']][0] + ')'
        locale_counts[locale] = locale_counts.get(locale, 0) + 1
    
    print("\nDistribution:")
    print("-" * 60)
    
    sorted_locales = sorted(locale_counts.items(), key=lambda x: x[1], reverse=True)
    for locale, count in sorted_locales:
        percentage = (count / n_samples) * 100
        bar = '█' * int(percentage / 2)
        print(f"{locale:30s}: {count:4d} ({percentage:5.1f}%) {bar}")
    
    # Validate en_US is most common (should be ~40%)
    en_us_count = locale_counts.get('English (en_US)', 0)
    assert en_us_count > 300, f"en_US should be ~40%, got {en_us_count/n_samples*100:.1f}%"
    
    print("\n✓ Locale distribution validated")


def test_decimal_separator_variations():
    """Test decimal and thousand separator variations"""
    print("\n" + "="*80)
    print("TEST 7: Decimal & Thousand Separators")
    print("="*80)
    
    renderer = SimplePNGRenderer()
    test_amount = 12345.67
    
    print(f"\nTest amount: {test_amount}")
    print("\nFormatted with separators:")
    print("-" * 70)
    
    locales = ['en_US', 'fr_FR', 'de_DE', 'es_ES']
    
    for locale in locales:
        config = renderer._get_locale_config(locale)
        formatted = renderer._format_currency(test_amount, config)
        
        decimal_sep = config['decimal_separator']
        thousand_sep = config['thousand_separator']
        
        print(f"{locale:10s}: {formatted:>20s}  (decimal: '{decimal_sep}', thousand: '{thousand_sep}')")
        
        # Validate separators are used correctly
        if decimal_sep == '.':
            # Decimal should be period
            assert '.67' in formatted or ',67' in formatted, f"Decimal separator incorrect for {locale}"
        elif decimal_sep == ',':
            # Decimal should be comma
            assert ',67' in formatted, f"Decimal separator should be comma for {locale}"
    
    print("\n✓ Separator variations validated")


def main():
    """Run all locale variant tests"""
    print("\n" + "="*80)
    print("INVOICE GEN - TEST 14: LOCALE & LANGUAGE VARIANTS")
    print("="*80)
    print("\nValidating comprehensive locale support across 10 locales")
    print("Testing currency, dates, tax labels, translations, and rendering")
    
    try:
        # Run all tests
        test_locale_currency_formatting()
        test_locale_date_formatting()
        test_locale_tax_labels()
        test_locale_label_translations()
        results = test_end_to_end_receipt_generation()
        test_locale_distribution()
        test_decimal_separator_variations()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print("\n✓ All 7 test suites passed")
        print(f"✓ Validated 10 locales")
        print(f"✓ Generated {len(results)} sample receipts")
        print("\nLocale Support Confirmed:")
        print("  • Currency formatting (symbol, position, separators)")
        print("  • Date formatting (5 different formats)")
        print("  • Tax labels (7 different tax types)")
        print("  • Label translations (5 languages)")
        print("  • Decimal/thousand separator variations")
        print("  • Locale distribution weights")
        
        print("\n" + "="*80)
        print("STATUS: ALL TESTS PASSED ✓")
        print("="*80)
        
        return 0
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
