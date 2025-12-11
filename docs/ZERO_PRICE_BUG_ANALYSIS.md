# $0.00 Pricing Bug - Root Cause Analysis & Fix

**Date:** December 10, 2025  
**Issue:** Generated invoices showing $0.00 for line item unit prices and totals  
**Status:** ✅ FIXED (commit f906f79)

---

## Problem Description

The 150K production dataset generated on vast.ai contained invoices with **$0.00** displayed for:
- Unit prices (line items)
- Line totals
- Individual item amounts

However, **subtotals and grand totals were CORRECT**, indicating the data existed but wasn't accessible to templates.

### Evidence
User provided 4 screenshots showing:
1. **candyville.ca invoice** - Missing unit prices in line items
2. **Flores Group invoice** - All prices showing $0.00, amounts $0.00
3. **Kent, Cohen and Richards** - Unit price $0.00, Amount $0.00 (but Subtotal correct: $67,516.06)
4. **Waliwaba proforma** - TOTAL (INR) column missing price data

---

## Root Cause Analysis

### Investigation Process

1. **Initial Hypothesis:** Wrong method called in `generate_parallel_dataset.py`
   - Line 341: `invoice_gen.generate_invoice()` 
   - But `ModernInvoiceGenerator` only has `generate_modern_invoice()`
   - Falls back to parent `SyntheticDataGenerator.generate_invoice()`

2. **Testing Revealed:** Method fallback actually works correctly
   - `to_dict()` properly adds aliases from `rate` → `unit_price`
   - Items contained correct numeric values

3. **Real Issue Discovered:** Missing field aliases
   - Templates use **different field names** for the same data:
     - `item.unit_cost` (PO/supply_chain templates)
     - `item.name` or `item.product_name` (vs. `item.description`)
     - `item.subtotal` (item-level, vs. invoice-level)
   
4. **The Bug:**
   ```python
   # Template uses: {{ item.unit_cost|to_float }}
   # But item only has: unit_price, rate, price
   # to_float filter returns: 0.0 (default when field is None)
   # Result: $0.00 displayed
   ```

### Affected Templates

Templates requiring `unit_cost`:
- `po_food_beverage.html`
- `po_electronics.html`
- `stock_transfer.html`
- `inventory_adjustment_form.html`
- `po_alibaba.html`
- `po_domestic_distributor.html`
- `po_dropship.html`
- `po_fashion_wholesale.html`

Templates requiring `name`/`product_name`:
- Many ecommerce templates
- Purchase order templates
- Supply chain templates

---

## The Fix

### Changes to `generators/modern_invoice_generator.py`

Added comprehensive field aliases in the `to_dict()` method:

```python
# Item-level field aliases (lines 225-234)
if 'unit_cost' not in item:
    item['unit_cost'] = item.get('unit_price', item.get('rate', item.get('price', 0)))

if 'name' not in item:
    item['name'] = item.get('description', 'Item')
    
if 'product_name' not in item:
    item['product_name'] = item.get('description', 'Item')

if 'subtotal' not in item:
    item['subtotal'] = item.get('total', item.get('amount', 0))
```

### Complete Field Coverage

After the fix, every item now has ALL these fields:
```python
{
    'description': 'Lipstick Collection',
    'quantity': 15,
    'rate': 888.75,          # Original from dataclass
    'unit_price': 888.75,    # Alias for rate
    'price': 888.75,         # Alias for unit_price
    'unit_cost': 888.75,     # Alias for unit_price (NEW)
    'total': 13331.25,       # Calculated from qty * unit_price
    'amount': 13331.25,      # Alias for total
    'line_total': 13331.25,  # Alias for total
    'subtotal': 13331.25,    # Alias for total (NEW)
    'name': 'Lipstick Collection',         # Alias for description (NEW)
    'product_name': 'Lipstick Collection', # Alias for description (NEW)
    'tax_rate': 0
}
```

---

## Impact Assessment

### Dataset Generated on vast.ai (150K samples)
- ❌ **Contains the bug** - generated with OLD code
- Missing `unit_cost`, `name`, `product_name`, `subtotal` aliases
- Templates using these fields showed $0.00
- **Estimated affected:** 20-40% of invoices (supply_chain, wholesale, PO categories)

### Future Generations
- ✅ **Bug fixed** - commit f906f79
- All templates now have access to price data
- No more $0.00 line items

---

## Recommendations

### Immediate Actions
1. ✅ **DONE:** Commit fix to repository
2. ⚠️ **TODO:** Regenerate the 150K dataset with corrected code
3. ⚠️ **TODO:** Clear old data on vast.ai to free disk space
4. ⚠️ **TODO:** Run new generation with `generate_parallel_dataset.py`

### Validation Steps
Before using regenerated dataset:
1. Sample 50-100 random invoices
2. Visually inspect for $0.00 patterns
3. Check across all categories (especially supply_chain, wholesale, purchase_orders)
4. Verify unit prices, line totals, and amounts are non-zero

### Prevention
- Add unit test to verify all field aliases exist
- Add template compatibility test that checks field access
- Document required fields for each template category

---

## Technical Details

### Why Subtotals Were Correct
The invoice-level `subtotal` was calculated correctly:
```python
# In InvoiceData.calculate_totals()
self.subtotal = sum(item.amount for item in self.items)
```

The `item.amount` is a `@property` that calculates `quantity * rate` dynamically. The **calculation** worked, but the `@property` isn't included when converting dataclass to dict via `asdict()`.

### Why to_dict() Now Adds amount
```python
if 'total' not in item and 'amount' not in item:
    qty = item.get('quantity', 1)
    unit_price = item.get('unit_price', item.get('rate', item.get('price', 0)))
    item['total'] = round(qty * unit_price, 2)
    item['amount'] = item['total']  # Explicitly add amount field
```

---

## Commit Details

**Commit:** f906f79  
**Message:** `fix: add missing field aliases to ModernInvoiceGenerator.to_dict()`

**Files Changed:**
- `generators/modern_invoice_generator.py` (+14 lines)

**Changes:**
1. Added `unit_cost` alias (line 230-231)
2. Added `name` alias (line 225-226)
3. Added `product_name` alias (line 228-229)
4. Added item-level `subtotal` alias (line 247-248)

---

## Conclusion

The $0.00 pricing bug was caused by **incomplete field name aliasing** in the data generator. Different template categories used different field names (`unit_cost` vs. `unit_price`, `name` vs. `description`), and when templates requested a missing field, the Jinja2 `to_float` filter defaulted to `0.0`.

The fix ensures **comprehensive field coverage** so all 153 templates can access price data regardless of which field name they use. The 150K dataset currently downloaded still contains the bug and should be regenerated with the corrected code.
