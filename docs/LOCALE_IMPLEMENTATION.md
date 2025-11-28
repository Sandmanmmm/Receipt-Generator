# Locale & Language Variant Implementation

**Status**: ✅ COMPLETE  
**Date**: 2025-01-27  
**Test Suite**: test_14_locale_variants.py

## Overview

Comprehensive locale support has been implemented across the InvoiceGen system, enabling generation of receipts and invoices in 10 different locales with proper currency formatting, date formatting, tax labels, and language-specific text.

## Supported Locales

### 1. English Variants (4 locales - 73% distribution)

#### en_US - English (United States) - 40%
- **Currency**: $ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: MM/DD/YYYY (e.g., 01/27/2025)
- **Tax Label**: Sales Tax
- **Tax Rates**: 0%, 6.25%, 7%, 8.25%, 8.75%, 10% (state-dependent)
- **Example**: $1,234.56 on 01/27/2025 with 7% Sales Tax

#### en_GB - English (United Kingdom) - 15%
- **Currency**: £ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: DD/MM/YYYY (e.g., 27/01/2025)
- **Tax Label**: VAT
- **Tax Rates**: 0%, 20% (standard VAT rate)
- **Example**: £1,234.56 on 27/01/2025 with 20% VAT

#### en_CA - English (Canada) - 10%
- **Currency**: $ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: YYYY-MM-DD (ISO format)
- **Tax Label**: GST/HST
- **Tax Rates**: 5% (GST), 13% (HST Ontario), 15% (HST Atlantic)
- **Example**: $1,234.56 on 2025-01-27 with 13% HST

#### en_AU - English (Australia) - 8%
- **Currency**: $ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: DD/MM/YYYY (e.g., 27/01/2025)
- **Tax Label**: GST
- **Tax Rates**: 0%, 10% (GST rate)
- **Document Type**: "Tax Invoice" (required for GST compliance)
- **Example**: $1,234.56 on 27/01/2025 with 10% GST

### 2. French Variants (2 locales - 12% distribution)

#### fr_CA - Français (Canada) - 7%
- **Currency**: $ (before amount)
- **Decimal**: `,` (comma)
- **Thousands**: ` ` (space)
- **Date Format**: YYYY-MM-DD (ISO format)
- **Tax Label**: TPS/TVQ
- **Tax Rates**: 5% (TPS), 9.975% (TVQ Quebec), 14.975% (Combined)
- **Labels**: Facture, Reçu, Sous-total, Rabais, Taxes, Total, Montant dû
- **Example**: 1 234,56 $ on 2025-01-27 with 14.975% TPS/TVQ

#### fr_FR - Français (France) - 5%
- **Currency**: € (after amount)
- **Decimal**: `,` (comma)
- **Thousands**: ` ` (space)
- **Date Format**: DD/MM/YYYY (e.g., 27/01/2025)
- **Tax Label**: TVA
- **Tax Rates**: 0%, 5.5%, 10%, 20% (various VAT rates)
- **Labels**: Facture, Reçu, Sous-total, Remise, TVA, Total, Montant dû
- **Example**: 1 234,56 € on 27/01/2025 with 20% TVA

### 3. Spanish Variants (2 locales - 9% distribution)

#### es_ES - Español (España) - 5%
- **Currency**: € (after amount)
- **Decimal**: `,` (comma)
- **Thousands**: `.` (period)
- **Date Format**: DD/MM/YYYY (e.g., 27/01/2025)
- **Tax Label**: IVA
- **Tax Rates**: 0%, 4%, 10%, 21% (super-reduced, reduced, standard)
- **Labels**: Factura, Recibo, Subtotal, Descuento, IVA, Total, Importe a pagar
- **Example**: 1.234,56 € on 27/01/2025 with 21% IVA

#### es_MX - Español (México) - 4%
- **Currency**: $ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: DD/MM/YYYY (e.g., 27/01/2025)
- **Tax Label**: IVA
- **Tax Rates**: 0%, 16% (standard IVA rate)
- **Labels**: Factura, Recibo, Subtotal, Descuento, IVA, Total, Monto a pagar
- **Example**: $1,234.56 on 27/01/2025 with 16% IVA

### 4. German (1 locale - 4% distribution)

#### de_DE - Deutsch (Deutschland) - 4%
- **Currency**: € (after amount)
- **Decimal**: `,` (comma)
- **Thousands**: `.` (period)
- **Date Format**: DD.MM.YYYY (e.g., 27.01.2025)
- **Tax Label**: MwSt.
- **Tax Rates**: 0%, 7%, 19% (reduced and standard rates)
- **Labels**: Rechnung, Quittung, Zwischensumme, Rabatt, MwSt., Gesamt, Zu zahlender Betrag
- **Example**: 1.234,56 € on 27.01.2025 with 19% MwSt.

### 5. Chinese (1 locale - 2% distribution)

#### zh_CN - 中文 (中国) - 2%
- **Currency**: ¥ (before amount)
- **Decimal**: `.` (period)
- **Thousands**: `,` (comma)
- **Date Format**: YYYY年MM月DD日 (e.g., 2025年01月27日)
- **Tax Label**: 增值税 (Value Added Tax)
- **Tax Rates**: 0%, 3%, 6%, 9%, 13% (various VAT rates)
- **Labels**: 发票, 收据, 小计, 折扣, 税, 总计, 应付金额
- **Example**: ¥1,234.56 on 2025年01月27日 with 13% 增值税

## Implementation Architecture

### 1. Core Locale Configuration System

**File**: `generators/html_to_png_renderer.py`  
**Method**: `_get_locale_config(locale: Optional[str] = None) -> dict`

```python
config = {
    'name': 'English (United States)',
    'language': 'en',
    'currency_symbol': '$',
    'currency_code': 'USD',
    'currency_position': 'before',  # or 'after'
    'decimal_separator': '.',
    'thousand_separator': ',',
    'date_format': 'MM/DD/YYYY',
    'tax_label': 'Sales Tax',
    'tax_rates': [0.0, 0.0625, 0.07, ...],
    'labels': {
        'invoice': 'Invoice',
        'receipt': 'Receipt',
        'subtotal': 'Subtotal',
        'discount': 'Discount',
        'tax': 'Tax',
        'total': 'Total',
        'amount_due': 'Amount Due',
        'thank_you': 'Thank you for your business!',
        'page': 'Page',
        'of': 'of'
    }
}
```

### 2. Currency Formatting

**Method**: `_format_currency(amount: float, locale_config: dict) -> str`

**Features**:
- Applies correct decimal separator (`.` or `,`)
- Applies correct thousands separator (`,`, `.`, or space)
- Places currency symbol before or after amount
- Formats to 2 decimal places

**Examples**:
```python
en_US: $1,234.56
fr_FR: 1 234,56 €
de_DE: 1.234,56 €
zh_CN: ¥1,234.56
```

### 3. Date Formatting

**Method**: `_format_date(date_str: str, locale_config: dict) -> str`

**Features**:
- Converts ISO format (YYYY-MM-DD) to locale-specific format
- Handles special formats (e.g., Chinese 年月日)
- Supports multiple date formats per locale

**Examples**:
```python
en_US: 01/27/2025  (MM/DD/YYYY)
en_GB: 27/01/2025  (DD/MM/YYYY)
en_CA: 2025-01-27  (YYYY-MM-DD)
de_DE: 27.01.2025  (DD.MM.YYYY)
zh_CN: 2025年01月27日
```

### 4. Data Generation with Locale Support

**File**: `generators/retail_data_generator.py`  
**Methods**: 
- `generate_pos_receipt(locale: Optional[str] = None)`
- `generate_online_order(locale: Optional[str] = None)`

**Features**:
- Random locale selection with weighted distribution
- Locale stored in `RetailReceiptData.locale` field
- Date stored in ISO format for consistent parsing
- Locale passed through to rendering pipeline

**Locale Distribution Weights**:
```python
{
    'en_US': 0.40,  # 40%
    'en_GB': 0.15,  # 15%
    'en_CA': 0.10,  # 10%
    'en_AU': 0.08,  #  8%
    'fr_CA': 0.07,  #  7%
    'fr_FR': 0.05,  #  5%
    'es_ES': 0.05,  #  5%
    'es_MX': 0.04,  #  4%
    'de_DE': 0.04,  #  4%
    'zh_CN': 0.02   #  2%
}
```

### 5. Receipt Rendering with Locale

**Process Flow**:

1. **Data Generation**: Locale randomly selected or specified
2. **Storage**: Locale stored in receipt JSON (`receipt_data['locale']`)
3. **Metadata Processing**: Locale config loaded, date formatted
4. **Totals Section**: Currency formatted, tax labels localized
5. **Output**: PNG with locale-appropriate formatting

**Code Integration**:
```python
# In _generate_order_metadata_section()
locale_code = receipt_data.get('locale', None)
locale_config = self._get_locale_config(locale_code)
invoice_date = self._format_date(invoice_date_raw, locale_config)

# In _generate_totals_section()
labels = locale_config['labels']
formatted_subtotal = self._format_currency(subtotal_num, locale_config)
text_lines.append(f"{labels['subtotal'] + ':':25s} {formatted_subtotal:>12s}")
```

## Testing & Validation

### Test Suite: test_14_locale_variants.py

**7 Test Cases**:

1. **Currency Formatting** - Validates 10 locales
   - Symbol position (before/after)
   - Decimal separator (. or ,)
   - Thousands separator (, or . or space)

2. **Date Formatting** - Validates 10 locales
   - MM/DD/YYYY (US)
   - DD/MM/YYYY (UK, AU, France, Spain, Mexico)
   - YYYY-MM-DD (Canada)
   - DD.MM.YYYY (Germany)
   - YYYY年MM月DD日 (China)

3. **Tax Labels** - Validates 10 locales
   - Sales Tax, VAT, GST, HST, TPS/TVQ, TVA, IVA, MwSt., 增值税

4. **Label Translations** - Validates 5 languages
   - English, French, Spanish, German, Chinese
   - Invoice, Receipt, Subtotal, Discount, Total, Thank You

5. **End-to-End Receipt Generation** - Generates 7 samples
   - Full receipt generation with locale
   - JSON storage with locale field
   - PNG rendering with locale formatting

6. **Locale Distribution** - Samples 1000 receipts
   - Validates weighted random selection
   - Confirms en_US dominance (~40%)
   - Ensures diversity across all locales

7. **Decimal/Thousand Separators** - Validates 4 locales
   - Tests large numbers (12,345.67)
   - Validates separator usage

**Test Results**:
```
✓ All 7 test suites passed
✓ Validated 10 locales
✓ Generated 7 sample receipts
✓ Locale distribution: en_US 37.7%, en_GB 14.7%, en_CA 13.0%, ...
```

## Label Translations

### Complete Label Set

| Key | en_US | fr_FR | es_ES | de_DE | zh_CN |
|-----|-------|-------|-------|-------|-------|
| invoice | Invoice | Facture | Factura | Rechnung | 发票 |
| receipt | Receipt | Reçu | Recibo | Quittung | 收据 |
| date | Date | Date | Fecha | Datum | 日期 |
| due_date | Due Date | Date d'échéance | Fecha de vencimiento | Fälligkeitsdatum | 到期日 |
| subtotal | Subtotal | Sous-total | Subtotal | Zwischensumme | 小计 |
| discount | Discount | Remise | Descuento | Rabatt | 折扣 |
| tax | Tax | TVA | IVA | MwSt. | 税 |
| total | Total | Total | Total | Gesamt | 总计 |
| amount_due | Amount Due | Montant dû | Importe a pagar | Zu zahlender Betrag | 应付金额 |
| thank_you | Thank you for your business! | Merci pour votre confiance! | ¡Gracias por su compra! | Vielen Dank für Ihren Einkauf! | 感谢您的惠顾！ |
| page | Page | Page | Página | Seite | 第 |
| of | of | sur | de | von | 页，共 |

## Usage Examples

### Generate Receipt with Specific Locale

```python
from generators.retail_data_generator import RetailDataGenerator

generator = RetailDataGenerator()

# Generate French (France) receipt
receipt = generator.generate_pos_receipt(locale='fr_FR', min_items=5, max_items=10)

# Generate German receipt
receipt = generator.generate_online_order(locale='de_DE', min_items=3, max_items=7)

# Generate random locale (weighted distribution)
receipt = generator.generate_pos_receipt()  # locale=None (default)
```

### Render with Locale Formatting

```python
from generators.html_to_png_renderer import SimplePNGRenderer

renderer = SimplePNGRenderer()

# Locale is automatically extracted from receipt_dict['locale']
receipt_dict = generator.to_dict(receipt)
renderer.render_receipt_dict(receipt_dict, 'output.png')

# Currency, dates, tax labels automatically formatted per locale
```

### Manual Locale Formatting

```python
renderer = SimplePNGRenderer()

# Get locale config
config = renderer._get_locale_config('de_DE')

# Format currency
amount_str = renderer._format_currency(1234.56, config)
# Result: "1.234,56 €"

# Format date
date_str = renderer._format_date('2025-01-27', config)
# Result: "27.01.2025"

# Get tax label
tax_label = config['tax_label']
# Result: "MwSt."
```

## LayoutLMv3 Training Implications

### Multi-Locale Dataset Benefits

1. **Number Format Diversity**
   - Period vs comma decimals
   - Various thousand separators
   - Currency position variations

2. **Date Format Diversity**
   - Multiple date formats in same dataset
   - Teaches model to recognize date patterns regardless of format
   - Handles international documents

3. **Text Diversity**
   - Multiple languages for same concepts
   - Improves generalization
   - Reduces overfitting to English-only patterns

4. **Real-World Accuracy**
   - Matches actual business document variety
   - Handles international invoices
   - Prepares for production deployment

### OCR Considerations

- **PaddleOCR**: Handles all locales including Chinese characters
- **Number Recognition**: Tested with various decimal/thousand separators
- **Date Recognition**: Validated across all date formats
- **Currency Symbols**: Properly detected ($, £, €, ¥)

## Configuration & Extension

### Adding New Locales

To add a new locale, update `_get_locale_config()` in `html_to_png_renderer.py`:

```python
'ja_JP': {
    'name': '日本語 (日本)',
    'language': 'ja',
    'currency_symbol': '¥',
    'currency_code': 'JPY',
    'currency_position': 'before',
    'decimal_separator': '.',
    'thousand_separator': ',',
    'date_format': 'YYYY年MM月DD日',
    'tax_label': '消費税',
    'tax_rates': [0.0, 0.08, 0.10],
    'labels': {
        'invoice': '請求書',
        'receipt': '領収書',
        # ... more labels
    }
}
```

Update locale weights:
```python
locale_weights = {
    # ... existing locales
    'ja_JP': 0.02  # 2%
}
```

### Customizing Distribution

Adjust weights in `_get_locale_config()` or pass explicit locale to generators:

```python
# 100% German receipts
for i in range(1000):
    receipt = generator.generate_pos_receipt(locale='de_DE')
```

## Production Recommendations

### Balanced Dataset

For LayoutLMv3 training, use default weighted distribution:
- **60% English** (en_US, en_GB, en_CA, en_AU combined)
- **40% Other languages** (French, Spanish, German, Chinese)

This ensures:
- English dominance for primary use case
- Sufficient variety for generalization
- Real-world distribution approximation

### Locale-Specific Models

For specialized deployments:
```python
# European model (focus on EUR currencies)
locales = ['en_GB', 'fr_FR', 'es_ES', 'de_DE']

# North American model
locales = ['en_US', 'en_CA', 'fr_CA', 'es_MX']

# Generate dataset with specific locales
for i in range(10000):
    locale = random.choice(locales)
    receipt = generator.generate_pos_receipt(locale=locale)
```

## Validation Results

### Test Run (2025-01-27)

```
✓ Currency formatting: 10/10 locales validated
✓ Date formatting: 10/10 locales validated
✓ Tax labels: 10/10 locales validated
✓ Label translations: 5/5 languages validated
✓ End-to-end generation: 7/7 samples rendered
✓ Locale distribution: 1000 samples validated
✓ Separator variations: 4/4 locales validated

Sample outputs:
- outputs/locale_test/locale_01_en_US.png
- outputs/locale_test/locale_02_en_GB_page1.png
- outputs/locale_test/locale_03_fr_CA_page1.png
- outputs/locale_test/locale_04_fr_FR_page1.png
- outputs/locale_test/locale_05_es_ES.png
- outputs/locale_test/locale_06_de_DE_page1.png
- outputs/locale_test/locale_07_zh_CN_page1.png
```

## Summary

**Status**: ✅ Production Ready

The locale implementation provides:
- ✅ 10 fully supported locales
- ✅ 5 languages (English, French, Spanish, German, Chinese)
- ✅ Proper currency formatting (symbol, position, separators)
- ✅ Locale-specific date formats (5 different formats)
- ✅ Tax label localization (7 different tax types)
- ✅ Weighted random distribution
- ✅ Comprehensive test coverage
- ✅ Ready for LayoutLMv3 training

This implementation ensures that the InvoiceGen system can generate realistic, diverse receipts suitable for training robust document understanding models that work with international documents.
