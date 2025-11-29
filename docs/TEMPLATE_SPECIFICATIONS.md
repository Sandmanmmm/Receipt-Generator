# Comprehensive Template Specifications Analysis

**Generated:** November 28, 2025  
**Updated:** November 28, 2025 (Specialty templates removed)  
**Total Templates:** 30  
**Multipage-Aware:** 2  
**Needs Multipage Support:** 14+

---

## Executive Summary

### Templates by Category

| Category | Count | Multipage-Aware | Needs Multipage |
|----------|-------|-----------------|-----------------|
| **modern_professional** | 10 | 2 | 6 |
| **retail/online_order** | 8 | 0 | 8 |
| **retail/pos_receipt** | 9 | 0 | 2 |
| **retail/consumer** | 1 | 0 | 1 |
| **classic** | 1 | 0 | 0 |
| **modern** | 1 | 0 | 0 |
| **receipt** | 1 | 0 | 0 |

### Multipage Status Overview

- ✅ **Multipage-Ready:** 2 templates (invoice_minimal_multipage, invoice_compact_multipage)
- ⚠️ **High Priority for Multipage:** 8 online_order templates (590-206 lines, used frequently)
- 📋 **Medium Priority:** 6 modern_professional invoices (508-190 lines)
- 🔍 **Low Priority:** Specialty templates (hospital_bill, contract_invoice, etc.)
- ✔️ **No Multipage Needed:** POS receipts (handled by SimplePNGRenderer), basic templates

---

## Detailed Template Specifications

### 1. Modern Professional Invoices (10 templates)

#### 1.1 invoice_minimal_multipage.html ✅ MULTIPAGE-AWARE
- **Lines:** 219
- **Style:** Clean, minimal design with subtle colors
- **Multipage Support:** ✅ YES
  - Uses `{% if _is_first_page|default(true) %}` for full header
  - Uses `{% if _is_last_page|default(true) %}` for totals section
  - Page indicators: "Page X of Y"
  - Continuation text: "Continued on page X+1..."
- **Sections:**
  - Full header (logo area, company info, invoice details) - first page only
  - Continuation header (compact invoice info) - subsequent pages
  - Items table (always shown)
  - Totals section - last page only
  - Footer with page numbers
- **Data Fields:** company_name, invoice_number, invoice_date, due_date, customer_name, customer_address, line_items[], subtotal, tax, total_amount, payment_info
- **Design Features:** Clean typography, subtle borders, professional layout
- **Estimated Height:** 900 + 45×items + 350 = ~1295px for 10 items

#### 1.2 invoice_compact_multipage.html ✅ MULTIPAGE-AWARE
- **Lines:** 273
- **Style:** Dense business design with dark header
- **Multipage Support:** ✅ YES
  - Uses `{% if _is_first_page|default(true) %}` for full header/customer info
  - Uses `{% if _is_last_page|default(true) %}` for totals/payment terms
  - Page indicators in header and footer
  - Conditional display based on _total_pages
- **Sections:**
  - Full header (dark banner, company info, invoice details) - first page only
  - Customer information section - first page only
  - Items table with alternating rows (always shown)
  - Totals section with payment terms - last page only
  - Footer with page navigation
- **Data Fields:** Same as minimal + payment_terms, bank_info
- **Design Features:** Dark header bar, alternating row colors, dense layout
- **Estimated Height:** 950 + 50×items + 400 = ~1350px for 10 items

#### 1.3 invoice_a4.html ❌ NEEDS MULTIPAGE
- **Lines:** 508
- **Style:** Professional A4 format with European styling
- **Multipage Support:** ❌ NO
- **Sections:**
  - Company header with logo and contact details
  - Invoice details grid (number, date, due date)
  - Customer billing section
  - Detailed items table
  - Subtotals, tax, discount sections
  - Payment terms and bank details
  - Footer with legal text
- **Data Fields:** All standard fields + vat_number, registration_number, bank_details
- **Design Features:** A4 dimensions, formal layout, European business standards
- **Complexity:** HIGH - 508 lines, many sections
- **Multipage Priority:** HIGH - Large templates likely to exceed 1400px

#### 1.4 invoice_landscape.html ❌ NEEDS MULTIPAGE
- **Lines:** 420
- **Style:** Wide landscape format for detailed items
- **Multipage Support:** ❌ NO
- **Sections:**
  - Wide header with company branding
  - Horizontal invoice info bar
  - Wide items table (more columns)
  - Totals on the right side
  - Wide footer
- **Data Fields:** Standard fields + additional item columns (SKU, category, notes)
- **Design Features:** Landscape orientation, wide tables, horizontal flow
- **Complexity:** HIGH - 420 lines
- **Multipage Priority:** HIGH - Landscape may need special handling

#### 1.5 invoice_ecommerce.html ❌ NEEDS MULTIPAGE
- **Lines:** 285
- **Style:** Modern e-commerce design with product images
- **Multipage Support:** ❌ NO
- **Sections:**
  - E-commerce header with branding
  - Order summary box
  - Items with product images
  - Shipping information
  - Payment method details
  - Totals with shipping costs
- **Data Fields:** Standard + shipping_address, shipping_method, tracking_number, payment_method
- **Design Features:** Product thumbnails, shipping tracking, modern colors
- **Complexity:** MEDIUM - 285 lines
- **Multipage Priority:** MEDIUM - E-commerce orders can be large

#### 1.6 invoice_elegant.html ❌ NEEDS MULTIPAGE
- **Lines:** 234
- **Style:** Sophisticated design with elegant typography
- **Multipage Support:** ❌ NO
- **Sections:**
  - Elegant header with serif fonts
  - Styled invoice details
  - Refined items table
  - Subtotals with decorative elements
  - Elegant footer
- **Data Fields:** Standard invoice fields
- **Design Features:** Serif fonts, subtle decorations, refined aesthetics
- **Complexity:** MEDIUM - 234 lines
- **Multipage Priority:** MEDIUM

#### 1.7 invoice_sidebar.html ❌ NEEDS MULTIPAGE
- **Lines:** 221
- **Style:** Two-column layout with sidebar
- **Multipage Support:** ❌ NO
- **Sections:**
  - Left sidebar (company info, payment details)
  - Main content area (invoice details, items)
  - Two-column footer
- **Data Fields:** Standard fields
- **Design Features:** Sidebar layout, column-based design
- **Complexity:** MEDIUM - 221 lines
- **Multipage Priority:** MEDIUM - Sidebar may need special multipage handling

#### 1.8 invoice_compact.html ❌ SINGLE-PAGE VARIANT
- **Lines:** 224
- **Style:** Dense single-page layout
- **Multipage Support:** ❌ NO (Single-page by design)
- **Multipage Priority:** LOW - This is the single-page version of invoice_compact_multipage

#### 1.9 invoice_bold.html ❌ NEEDS MULTIPAGE
- **Lines:** 190
- **Style:** Bold colors and strong typography
- **Multipage Support:** ❌ NO
- **Sections:**
  - Bold header with accent colors
  - Strong visual hierarchy
  - Bold items table
  - Prominent totals
- **Data Fields:** Standard invoice fields
- **Design Features:** Bold colors, strong contrasts, modern typography
- **Complexity:** LOW-MEDIUM - 190 lines
- **Multipage Priority:** MEDIUM

#### 1.10 invoice_minimal.html ❌ SINGLE-PAGE VARIANT
- **Lines:** 180
- **Style:** Minimal single-page design
- **Multipage Support:** ❌ NO (Single-page by design)
- **Multipage Priority:** LOW - This is the single-page version of invoice_minimal_multipage

---

### 2. Retail Online Orders (8 templates)

#### 2.1 online_order_wholesale.html ❌ HIGH PRIORITY
- **Lines:** 590 (LARGEST TEMPLATE)
- **Style:** Complex B2B wholesale order with business terms
- **Multipage Support:** ❌ NO - **CRITICAL ISSUE**
- **Sections:**
  - Full header (company branding, order number, date) - Lines 1-26
  - Business account banner (credit info, account status) - Lines 28-43
  - Customer information (billing + shipping addresses) - Lines 45-100
  - Order items table with tiered pricing - Lines 150-350
  - Order summary with totals - Lines 400-420
  - Shipping details (carrier, tracking, freight) - Lines 425-490
  - Account information and payment terms - Lines 500-590
- **Data Fields:** customer_name, customer_address, shipping_address, business_account_number, credit_limit, line_items[], shipping_method, freight_charges, payment_terms, account_manager
- **Design Features:** Tiered pricing tables, credit status, freight calculations, business account info
- **Complexity:** VERY HIGH - 590 lines, most complex template
- **Multipage Priority:** ⚠️ **CRITICAL** - Frequently generates 12-18 items, always exceeds 1400px
- **Multipage Strategy:**
  - Wrap lines 1-100 (header + customer) with `{% if _is_first_page|default(true) %}`
  - Keep lines 150-350 (items loop) unconditional
  - Wrap lines 400-590 (totals + shipping + account) with `{% if _is_last_page|default(true) %}`

#### 2.2 online_order_grocery.html ❌ HIGH PRIORITY
- **Lines:** 435
- **Style:** Fresh grocery delivery with categories
- **Multipage Support:** ❌ NO
- **Sections:**
  - Grocery store header with fresh branding
  - Delivery information with time slot
  - Categorized items (produce, dairy, meat, etc.)
  - Substitution preferences
  - Delivery instructions
  - Totals with delivery fee
- **Data Fields:** Standard + delivery_time_slot, substitution_preferences, delivery_instructions, perishable_handling
- **Design Features:** Category grouping, freshness indicators, delivery scheduling
- **Complexity:** HIGH - 435 lines
- **Multipage Priority:** ⚠️ **HIGH** - Grocery orders often have 15-20+ items

#### 2.3 online_order_electronics.html ❌ HIGH PRIORITY
- **Lines:** 424
- **Style:** Tech/electronics order with specifications
- **Multipage Support:** ❌ NO
- **Sections:**
  - Tech store header
  - Order details with tracking
  - Items with specs and warranty info
  - Extended warranty options
  - Technical support information
  - Totals with installation services
- **Data Fields:** Standard + warranty_info, technical_specs, support_contact, installation_options
- **Design Features:** Spec tables, warranty details, tech support info
- **Complexity:** HIGH - 424 lines
- **Multipage Priority:** ⚠️ **HIGH**

#### 2.4 online_order_home_improvement.html ❌ HIGH PRIORITY
- **Lines:** 446
- **Style:** Hardware/home improvement order
- **Multipage Support:** ❌ NO
- **Sections:**
  - Hardware store header
  - Project information
  - Items with availability and stock location
  - Installation services
  - Delivery scheduling (bulk items)
  - Totals with delivery/installation
- **Data Fields:** Standard + stock_location, installation_required, bulk_delivery, project_name
- **Design Features:** Stock indicators, installation options, project tracking
- **Complexity:** HIGH - 446 lines
- **Multipage Priority:** ⚠️ **HIGH**

#### 2.5 online_order_digital.html ❌ MEDIUM PRIORITY
- **Lines:** 368
- **Style:** Digital goods/software order
- **Multipage Support:** ❌ NO
- **Sections:**
  - Digital storefront header
  - License information
  - Digital items with download links
  - Activation keys
  - Digital rights information
  - Totals (no shipping)
- **Data Fields:** Standard + license_key, download_link, activation_info, license_type
- **Design Features:** Download buttons, license keys, digital rights
- **Complexity:** MEDIUM - 368 lines
- **Multipage Priority:** MEDIUM - Digital orders typically smaller

#### 2.6 online_order_fashion.html ❌ HIGH PRIORITY
- **Lines:** 297
- **Style:** Fashion/apparel order
- **Multipage Support:** ❌ NO
- **Sections:**
  - Fashion brand header
  - Order details
  - Items with size/color/style
  - Returns policy
  - Shipping information
  - Totals
- **Data Fields:** Standard + size, color, style, returns_deadline
- **Design Features:** Style information, size charts, returns policy
- **Complexity:** MEDIUM - 297 lines
- **Multipage Priority:** ⚠️ **HIGH** - Fashion orders can be large

#### 2.7 online_order_marketplace.html ❌ MEDIUM PRIORITY
- **Lines:** 223
- **Style:** Multi-vendor marketplace order
- **Multipage Support:** ❌ NO
- **Sections:**
  - Marketplace header
  - Multiple vendor sections
  - Items grouped by vendor
  - Per-vendor shipping
  - Combined totals
- **Data Fields:** Standard + vendor_name, vendor_id, per_vendor_shipping
- **Design Features:** Vendor grouping, multiple shipments
- **Complexity:** MEDIUM - 223 lines
- **Multipage Priority:** MEDIUM

#### 2.8 online_order_invoice.html ❌ MEDIUM PRIORITY
- **Lines:** 206
- **Style:** Generic online order invoice
- **Multipage Support:** ❌ NO
- **Sections:**
  - Standard online order header
  - Order details
  - Items table
  - Shipping info
  - Totals
- **Data Fields:** Standard online order fields
- **Design Features:** Generic modern design
- **Complexity:** MEDIUM - 206 lines
- **Multipage Priority:** MEDIUM

---

### 3. Retail POS Receipts (9 templates)

**Note:** POS receipts use SimplePNGRenderer which handles native multipage rendering (cuts at fixed heights). These do NOT need Jinja2 multipage conditionals.

#### 3.1 pos_receipt_wholesale.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 416
- **Style:** B2B wholesale receipt
- **Renderer:** SimplePNGRenderer (native multipage at 800px)
- **Multipage Priority:** ✔️ NONE - Handled by renderer

#### 3.2 pos_receipt_pharmacy.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 282
- **Style:** Pharmacy receipt with Rx info
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.3 pos_receipt_fuel.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 274
- **Style:** Gas station receipt with fuel info
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.4 pos_receipt_qsr.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 250
- **Style:** Quick service restaurant receipt
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.5 pos_receipt_premium.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 245
- **Style:** Premium retail receipt
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.6 pos_receipt_wide.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 213
- **Style:** Wide format receipt
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.7 pos_receipt.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 207
- **Style:** Standard POS receipt
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

#### 3.8 pos_receipt_dense.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 153
- **Style:** Compact dense receipt
- **Renderer:** SimplePNGRenderer
- **Multipage Priority:** ✔️ NONE

---

### 4. Retail Consumer Services (1 template)

#### 4.1 consumer_service_invoice.html ❌ MEDIUM PRIORITY
- **Lines:** 268
- **Category:** retail
- **Style:** Service invoice (repair, maintenance, etc.)
- **Multipage Support:** ❌ NO
- **Sections:** Service details, labor, parts, totals
- **Multipage Priority:** MEDIUM - Service invoices can be detailed

---

**Note:** Specialty templates (saas, telecom, utility, medical, logistics, government) have been removed from the codebase to focus on core retail and professional invoice templates.

---

### 5. Basic Templates (3 templates)

#### 5.1 classic/invoice.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 151
- **Style:** Classic simple invoice
- **Multipage Priority:** ✔️ NONE - Basic template, typically small

#### 5.2 modern/invoice.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 150
- **Style:** Basic modern invoice
- **Multipage Priority:** ✔️ NONE - Basic template

#### 5.3 receipt/invoice.html ✔️ NO MULTIPAGE NEEDED
- **Lines:** 97
- **Style:** Minimal receipt
- **Multipage Priority:** ✔️ NONE - Very basic

---

## Multipage Implementation Strategy

### Phase 1: Critical Priority (IMMEDIATE) ⚠️
**Target:** 8 online_order templates - Used most frequently, consistently exceed 1400px with 12-18 items

1. **online_order_wholesale.html** (590 lines) - MOST CRITICAL
2. **online_order_grocery.html** (435 lines)
3. **online_order_home_improvement.html** (446 lines)
4. **online_order_electronics.html** (424 lines)
5. **online_order_fashion.html** (297 lines)
6. **online_order_digital.html** (368 lines)
7. **online_order_marketplace.html** (223 lines)
8. **online_order_invoice.html** (206 lines)

**Implementation Pattern:**
```jinja2
{# First page: Full header + customer info #}
{% if _is_first_page|default(true) %}
    <div class="header">...</div>
    <div class="customer-info">...</div>
    <div class="business-details">...</div>
{% else %}
    {# Continuation header: Compact info #}
    <div class="continuation-header">
        Order {{ order_number }} (Continued) - Page {{ _page_number }} of {{ _total_pages }}
    </div>
{% endif %}

{# Always show items table #}
<table class="items-table">
    {% for item in line_items %}
        <tr>...</tr>
    {% endfor %}
</table>

{# Last page: Totals + shipping + payment terms #}
{% if _is_last_page|default(true) %}
    <div class="order-summary">...</div>
    <div class="shipping-details">...</div>
    <div class="payment-terms">...</div>
{% else %}
    <div class="continuation-footer">
        Continued on page {{ (_page_number|default(1)) + 1 }}...
    </div>
{% endif %}
```

### Phase 2: High Priority (NEXT SPRINT)
**Target:** 6 modern_professional invoices - Professional invoices that can grow large

1. **invoice_a4.html** (508 lines)
2. **invoice_landscape.html** (420 lines) - May need special landscape handling
3. **invoice_ecommerce.html** (285 lines)
4. **invoice_elegant.html** (234 lines)
5. **invoice_sidebar.html** (221 lines) - Sidebar may need special handling
6. **invoice_bold.html** (190 lines)

### Phase 3: Low Priority (BACKLOG)
**Target:** Remaining templates - Used less frequently, typically smaller

1. **consumer_service_invoice.html** (268 lines)

### No Multipage Needed ✔️
- **POS receipts** (9 templates) - SimplePNGRenderer handles multipage natively
- **Basic templates** (3 templates) - Simple designs, typically stay under 1400px
- **Single-page variants** - invoice_minimal.html, invoice_compact.html (by design)

---

## Technical Implementation Details

### Conditional Structure Pattern

```jinja2
{# Title with page numbers #}
<title>{{ template_name }}{% if _page_number|default(none) %} - Page {{ _page_number }} of {{ _total_pages }}{% endif %}</title>

{# First page full header #}
{% if _is_first_page|default(true) %}
    <div class="full-header">
        <!-- Company branding, logos, full invoice details -->
    </div>
    <div class="customer-section">
        <!-- Customer info, addresses, account details -->
    </div>
{% else %}
    {# Continuation header for subsequent pages #}
    <div class="continuation-header">
        <!-- Minimal: Invoice/Order number, page indicator -->
    </div>
{% endif %}

{# Items always displayed (pagination handles subset) #}
<table class="items-table">
    <thead>
        <tr><th>Item</th><th>Qty</th><th>Price</th><th>Total</th></tr>
    </thead>
    <tbody>
        {% for item in line_items %}
        <tr>
            <td>{{ item.description }}</td>
            <td>{{ item.quantity }}</td>
            <td>{{ item.rate|currency }}</td>
            <td>{{ item.amount|currency }}</td>
        </tr>
        {% endfor %}
    </tbody>
</table>

{# Last page totals and footer #}
{% if _is_last_page|default(true) %}
    <div class="totals-section">
        <div class="subtotal">Subtotal: {{ subtotal|currency }}</div>
        <div class="tax">Tax: {{ tax|currency }}</div>
        <div class="total">Total: {{ total_amount|currency }}</div>
    </div>
    <div class="payment-terms">
        <!-- Payment instructions, bank details, terms -->
    </div>
    <div class="footer">
        <!-- Legal text, contact info, thank you message -->
    </div>
{% else %}
    <div class="continuation-footer">
        Continued on page {{ (_page_number|default(1)) + 1 }}...
    </div>
{% endif %}
```

### Height Calculation Parameters

From `test_13_gold_sample_verification.py`:

```python
if 'online_order' in template_name_lower:
    base_height = 1200       # Header + customer info + margins
    item_height = 80         # Per line item (more complex than invoices)
    footer_height = 500      # Totals + shipping + payment terms
    items_per_page = 10      # Fewer items due to complexity
elif 'invoice' in template_name_lower:
    base_height = 900        # Header + margins
    item_height = 50         # Per line item
    footer_height = 400      # Totals + footer
    items_per_page = 12      # Standard invoice items
```

### Page Variables Reference

| Variable | Type | Purpose | Default |
|----------|------|---------|---------|
| `_page_number` | int | Current page number (1-indexed) | none |
| `_total_pages` | int | Total number of pages | none |
| `_is_first_page` | bool | True if current page is page 1 | true |
| `_is_last_page` | bool | True if current page is final page | true |
| `_hide_totals` | bool | True if totals should be hidden (non-last pages) | false |

**Important:** Always use `|default()` filters for backward compatibility with single-page rendering:
- `_is_first_page|default(true)` - Treats single-page as first page
- `_is_last_page|default(true)` - Treats single-page as last page
- `_page_number|default(none)` - Only shows page numbers in multipage context

---

## Testing Requirements

### Per-Template Testing Checklist

For each template modified with multipage support:

1. **Single-page render** (5 items)
   - ✅ Full header displays correctly
   - ✅ All customer info displays
   - ✅ Items table renders
   - ✅ Totals section displays
   - ✅ Footer displays
   - ✅ No page numbers shown (|default(none) working)

2. **Two-page render** (15 items)
   - ✅ Page 1: Full header, customer info, items 1-10, no totals, continuation text
   - ✅ Page 2: Continuation header, items 11-15, totals, full footer
   - ✅ Page numbers display correctly
   - ✅ No duplicate sections

3. **Three-page render** (25 items)
   - ✅ Page 1: Full header, items 1-10, continuation
   - ✅ Page 2: Continuation header, items 11-20, continuation
   - ✅ Page 3: Continuation header, items 21-25, totals, footer
   - ✅ All page numbers correct

4. **Height validation**
   - ✅ Each page ≤ 1400px
   - ✅ No content cutoff
   - ✅ Proper spacing maintained

### Integration Testing

Run `test_13_gold_sample_verification.py` with:
- 20 samples
- Mixed templates (invoices + online_orders)
- Expected: 5-7 multipage samples
- Verify: All multipage samples paginate correctly

---

## Current Test Results (Before Phase 1)

**Test:** `test_13_gold_sample_verification.py --num-samples 20`  
**Date:** November 28, 2025

### Issues Identified

- ❌ **gold_001:** online_order_grocery - Single tall page (should be 2 pages)
- ❌ **gold_003:** online_order_wholesale - Single tall page (should be 2 pages)
- ❌ **gold_005:** online_order_electronics - Single tall page (should be 2 pages)
- ❌ **gold_006:** online_order_fashion - Single tall page (should be 2 pages)
- ❌ **gold_011:** online_order_grocery - Single tall page (should be 2 pages)
- ❌ **gold_013:** online_order_wholesale - Single tall page (should be 2 pages)
- ❌ **gold_014:** online_order_home_improvement - Single tall page (should be 2 pages)
- ✅ **gold_015:** invoice_compact_multipage - Correctly paginated (2 pages)

**Success Rate:** 1/8 multipage templates working (12.5%)  
**Root Cause:** online_order templates lack multipage conditionals

---

## Recommendations

### Immediate Actions

1. ✅ **Document complete** - This comprehensive analysis
2. ⚠️ **Start Phase 1** - Add multipage conditionals to 8 online_order templates
   - Begin with online_order_wholesale.html (most complex, highest priority)
   - Test with single sample after each template modification
   - Use proven pattern from invoice_minimal_multipage.html

3. **Update test_13** - Adjust template selection weights
   - Increase multipage template selection to 50% once Phase 1 complete
   - Add validation for multipage conditional presence

### Quality Assurance

- Run comprehensive testing after each phase completion
- Visual inspection of generated samples in verification report
- Height validation for each page
- OCR accuracy testing (ensure bounding boxes handle multipage)

### Future Enhancements

- Create multipage detection tool (scan templates for conditionals)
- Automated testing for multipage templates
- Template validation in CI/CD pipeline
- Documentation generator from template analysis

---

## Appendix: Template Selection Logic

From `test_13_gold_sample_verification.py` lines 397-451:

```python
# Select template FIRST
all_invoice_templates = [t for t in all_templates if 'invoice' in t.lower()]
all_online_order_templates = [t for t in all_templates if 'online_order' in t.lower()]

# Weight toward multipage
multipage_invoice_templates = [t for t in all_invoice_templates if '_multipage' in t.lower()]
invoice_selection = multipage_invoice_templates * 4 + all_invoice_templates

# Select template
if random.random() < 0.6:  # 60% invoice
    template_name = random.choice(invoice_selection)
else:  # 40% online_order
    template_name = random.choice(all_online_order_templates)

# Determine item count BASED ON TEMPLATE
is_multipage_template = '_multipage' in template_name.lower()
is_online_order = 'online_order' in template_name.lower()

if is_multipage_template:
    min_items, max_items = 15, 25  # Force multipage with many items
elif is_online_order:
    min_items, max_items = 12, 18  # Moderate items for online orders
else:
    min_items, max_items = 3, 8    # Regular invoices

# Generate data with correct item count
num_items = random.randint(min_items, max_items)
data = generate_invoice_data(num_items=num_items, locale=locale)
```

**Current Weights:**
- Multipage invoice templates: 31% chance (4/13 weighted selection)
- Regular invoice templates: 29% chance
- Online order templates: 40% chance

**After Phase 1 completion, should adjust to:**
- Multipage-aware templates (2 invoice + 8 online_order): 60-70% selection
- Regular templates: 30-40% selection

---

**End of Comprehensive Template Specifications Analysis**

