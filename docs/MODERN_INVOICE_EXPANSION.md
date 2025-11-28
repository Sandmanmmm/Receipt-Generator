# Modern Invoice Expansion Plan

## Current State
✅ **Receipts**: Excellent coverage with 40+ variants
- POS receipts (retail, restaurant, gas station, pharmacy)
- Online order receipts
- Multiple page formats (continuous roll, standard letter)
- 50% augmentation working perfectly

❌ **Invoices**: Limited to old receipt-style formatting
- Only text-based portrait format
- No modern professional layouts
- Missing visual elements (logos, colors, borders)
- No landscape or A4 formats

## Gap Analysis

### Missing Invoice Types

#### 1. **Professional B2B Invoices**
- Corporate letterhead style
- Detailed line items with product codes
- Payment terms (Net 30, Net 60)
- Company logos and branding
- Multiple currency support

#### 2. **Service Invoices**
- Hourly billing
- Project-based pricing
- Consulting/freelance layouts
- Time tracking details

#### 3. **E-commerce Invoices**
- Order confirmation style
- Shipping details prominent
- Tracking numbers
- Return policies

#### 4. **Wholesale/Manufacturing Invoices**
- Purchase orders
- Batch numbers
- Warehouse locations
- Volume discounts

#### 5. **International Invoices**
- Multiple languages
- VAT/GST formats
- Import/export declarations
- Currency conversions

### Missing Visual Elements

#### Layouts
- ❌ Landscape orientation (11" x 8.5")
- ❌ A4 format (210mm x 297mm)
- ❌ Multi-column designs
- ❌ Sidebar layouts
- ❌ Header/footer with borders

#### Visual Components
- ❌ Company logos (SVG/PNG)
- ❌ Color schemes (brand colors)
- ❌ Background watermarks
- ❌ QR codes (payment, tracking)
- ❌ Barcode images
- ❌ Product thumbnails
- ❌ Signatures (digital/scanned)
- ❌ Stamps/seals

#### Typography
- ❌ Multiple font families
- ❌ Font weights (bold headers)
- ❌ Font sizes hierarchy
- ❌ Color text (not just black)

#### Table Styles
- ❌ Bordered tables
- ❌ Alternating row colors
- ❌ Column spanning
- ❌ Nested tables
- ❌ Rounded corners

## Implementation Plan

### Phase 1: Template Infrastructure (HIGH PRIORITY)
**Goal**: Create 10+ modern invoice HTML templates with CSS

**Tasks**:
1. Create modern invoice templates:
   - `templates/modern_professional/` - Corporate B2B
   - `templates/modern_minimal/` - Clean, simple design
   - `templates/modern_creative/` - Designer/agency style
   - `templates/modern_colorful/` - Bright, branded
   - `templates/landscape_corporate/` - Landscape format
   - `templates/a4_european/` - A4 with VAT
   - `templates/service_invoice/` - Hourly/project billing
   - `templates/ecommerce_order/` - Online order style
   - `templates/wholesale/` - B2B wholesale
   - `templates/international/` - Multi-language

2. Each template should include:
   - HTML structure with Jinja2 variables
   - Dedicated CSS file
   - Responsive design (for different page sizes)
   - Print-optimized styles

3. Visual elements support:
   - Logo placeholder areas
   - Color scheme variables in CSS
   - QR code/barcode placeholders
   - Signature areas

### Phase 2: Modern Invoice Data Generator (HIGH PRIORITY)
**Goal**: Generate realistic data for modern invoices

**Create**: `generators/modern_invoice_generator.py`

**Features**:
- Company profiles with branding (colors, logo paths)
- Professional line items (product codes, detailed descriptions)
- Payment terms generation
- Shipping/billing address separation
- Tax variations (VAT, GST, sales tax)
- Multi-currency support
- Terms & conditions text
- Notes/special instructions

**Data Classes**:
```python
@dataclass
class CompanyProfile:
    name: str
    logo_path: Optional[str]
    address: str
    phone: str
    email: str
    website: str
    tax_id: str
    brand_colors: Dict[str, str]  # primary, secondary, accent
    
@dataclass
class ModernInvoiceData:
    invoice_number: str
    invoice_date: str
    due_date: str
    company: CompanyProfile
    client: ClientInfo
    line_items: List[InvoiceLineItem]
    subtotal: float
    tax: float
    total: float
    payment_terms: str
    notes: Optional[str]
    terms: Optional[str]
```

### Phase 3: Enhanced Renderer (MEDIUM PRIORITY)
**Goal**: Render modern templates with visual elements

**Enhance**: `generators/html_to_png_renderer.py`

**New Features**:
1. **Landscape rendering**:
   - Support 11" x 8.5" (landscape letter)
   - Adjust wkhtmltoimage parameters
   
2. **A4 rendering**:
   - Support 210mm x 297mm
   - DPI adjustments for metric sizes

3. **Logo injection**:
   - Generate placeholder logos (colored rectangles/circles)
   - SVG generation for simple shapes
   - Base64 embedding in HTML

4. **QR code generation**:
   - Invoice number QR codes
   - Payment QR codes (for crypto/mobile payments)
   - Library: `qrcode` or `segno`

5. **Barcode generation**:
   - Code128 for invoice numbers
   - Library: `python-barcode`

6. **Color scheme application**:
   - Dynamic CSS variable injection
   - Random brand color generation

### Phase 4: Integration & Distribution (MEDIUM PRIORITY)
**Goal**: Mix modern invoices into training data

**Modify**: `scripts/pipeline.py` or create `scripts/generate_mixed_dataset.py`

**Distribution Target**:
```
Training Data Mix (10,000 samples):
├── Receipts (60%)
│   ├── POS receipts: 40%
│   ├── Online orders: 20%
│   
├── Modern Invoices (40%)
│   ├── Professional B2B: 15%
│   ├── Service invoices: 8%
│   ├── E-commerce: 8%
│   ├── Wholesale: 5%
│   └── International: 4%

Formats:
├── Portrait (70%)
├── Landscape (20%)
└── A4 (10%)

Visual Complexity:
├── Text-only (30%)
├── Text + Simple borders (40%)
├── Text + Colors + Borders (20%)
└── Full visual (logos, colors, QR, images) (10%)

Augmentation:
├── Clean (20%)
├── Light degradation (30%)
├── Medium degradation (30%)
└── Heavy degradation (20%)
```

### Phase 5: Visual Assets Generation (LOW PRIORITY)
**Goal**: Generate realistic visual elements

**Create**: `generators/visual_assets.py`

**Features**:
1. **Logo generation**:
   - Simple geometric shapes
   - Text-based logos (company initials)
   - SVG generation
   - Random color schemes

2. **QR code generation**:
   - Invoice number encoded
   - Payment URLs
   - Variable sizes

3. **Barcode generation**:
   - Code128, Code39
   - Invoice/product numbers

4. **Signature simulation**:
   - Simple SVG signatures
   - Handwriting-style fonts

5. **Product images** (optional):
   - Colored placeholder boxes
   - Generic product icons

## Technical Considerations

### Page Size Handling

**Portrait Letter** (current):
```python
width = 816px (8.5")
height = 1056px (11")
dpi = 96
```

**Landscape Letter** (new):
```python
width = 1056px (11")
height = 816px (8.5")
dpi = 96
```

**A4 Portrait** (new):
```python
width = 794px (210mm)
height = 1123px (297mm)
dpi = 96
```

**A4 Landscape** (new):
```python
width = 1123px (297mm)
height = 794px (210mm)
dpi = 96
```

### wkhtmltoimage Parameters

```python
# Landscape
wkhtmltoimage --orientation Landscape ...

# A4
wkhtmltoimage --page-size A4 ...

# Custom size
wkhtmltoimage --width 1056 --height 816 ...
```

### Augmentation Compatibility

Modern invoices with colors/logos need special handling:
- Color augmentation should preserve brand colors (don't make logos unreadable)
- QR codes should remain scannable (exclude from heavy distortion)
- Logos should stay within reasonable bounds
- Text in colored backgrounds needs contrast preservation

**Recommendation**: Create `augmentation_mode` parameter:
- `text_only`: Full augmentation (current)
- `simple_visual`: Moderate augmentation (preserve basic visuals)
- `complex_visual`: Light augmentation (preserve colors, logos, QR codes)

## Libraries to Add

```txt
# requirements.txt additions
qrcode[pil]==7.4.2          # QR code generation
python-barcode==0.15.1       # Barcode generation
cairosvg==2.7.1             # SVG to PNG conversion (for logos)
Pillow>=10.0.0              # Already installed, for image manipulation
```

## Priority Ranking

### Immediate (Week 1)
1. ✅ Analyze current state (DONE)
2. 🔨 Create 5 modern invoice templates (HTML + CSS)
3. 🔨 Build ModernInvoiceGenerator class
4. 🔨 Add landscape/A4 rendering support

### Short-term (Week 2)
5. 🔨 Implement logo/QR/barcode generation
6. 🔨 Create mixed dataset generation script
7. 🔨 Test augmentation on modern invoices
8. 🔨 Generate 1,000 mixed samples for validation

### Medium-term (Week 3+)
9. 🔨 Add remaining 5 invoice templates
10. 🔨 Implement advanced visual elements
11. 🔨 Fine-tune augmentation modes
12. 🔨 Generate full 10,000 sample dataset

## Next Steps

**Immediate Action Items**:
1. Create first modern invoice template (professional B2B)
2. Implement landscape rendering in HTMLToPNGRenderer
3. Build basic ModernInvoiceGenerator
4. Generate 10 test samples
5. Verify augmentation works on modern invoices

**Decision Points**:
- Which invoice types are highest priority? (B2B, service, e-commerce?)
- Should we use real logo images or generate simple ones?
- What percentage of training data should be modern invoices vs receipts?
- Do we need multi-page modern invoices (like receipts)?

