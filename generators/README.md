# Generators Module

Modular invoice generation system with separate concerns for data generation, templating, and rendering.

## Module Structure

```
generators/
├── data_generator.py        # Synthetic invoice data generation
├── template_renderer.py     # Jinja2 template rendering
├── pdf_renderer.py          # HTML to PDF conversion
├── image_renderer.py        # PDF to image conversion
├── randomizers.py           # Randomization utilities
├── synthetic_data.py        # Legacy (backward compatibility)
└── renderer.py              # Legacy (backward compatibility)
```

## New Modular API

### 1. Data Generation

```python
from generators import SyntheticDataGenerator, InvoiceRandomizer

# Generate synthetic invoice data
generator = SyntheticDataGenerator(locale='en_US', seed=42)
invoice = generator.generate_invoice(
    min_items=3,
    max_items=8,
    include_tax=True,
    include_discount=True
)

# Use randomizers for custom data
randomizer = InvoiceRandomizer(seed=42)
currency = randomizer.random_currency(['USD', 'EUR', 'GBP'])
tax_rate = randomizer.random_tax_rate(region='US')
invoice_num = randomizer.random_invoice_number(prefix='INV', year=True)
```

### 2. Template Rendering

```python
from generators import TemplateRenderer

# Render Jinja2 templates to HTML
renderer = TemplateRenderer(templates_dir='templates')

html = renderer.render(
    template_name='modern/invoice.html',
    data=invoice_dict
)

# Or save to file
html = renderer.render_to_file(
    template_name='modern/invoice.html',
    data=invoice_dict,
    output_path='output/invoice.html'
)
```

### 3. PDF Rendering

```python
from generators import PDFRenderer

# Convert HTML to PDF
pdf_renderer = PDFRenderer(backend='weasyprint')  # or 'wkhtmltopdf'

# From HTML file
pdf_renderer.render_from_html_file(
    html_path='output/invoice.html',
    pdf_path='output/invoice.pdf'
)

# From HTML string
pdf_renderer.render_from_html_string(
    html_content=html,
    pdf_path='output/invoice.pdf',
    base_url='templates'
)
```

### 4. Image Rendering

```python
from generators import ImageRenderer

# Convert PDF to images
img_renderer = ImageRenderer(dpi=150)

# Single page
img_renderer.pdf_to_image(
    pdf_path='output/invoice.pdf',
    image_path='output/invoice.png',
    format='PNG'
)

# All pages
image_paths = img_renderer.pdf_to_images(
    pdf_path='output/document.pdf',
    output_dir='output/pages',
    format='PNG'
)
```

## Complete Pipeline Example

```python
from generators import (
    SyntheticDataGenerator,
    TemplateRenderer,
    PDFRenderer,
    ImageRenderer
)

# 1. Generate data
data_gen = SyntheticDataGenerator(seed=42)
invoice = data_gen.generate_invoice()
invoice_dict = data_gen.invoice_to_dict(invoice)

# 2. Render template to HTML
template_renderer = TemplateRenderer('templates')
html = template_renderer.render('modern/invoice.html', invoice_dict)

# 3. Convert to PDF
pdf_renderer = PDFRenderer(backend='weasyprint')
pdf_renderer.render_from_html_string(
    html_content=html,
    pdf_path=f'output/{invoice.invoice_number}.pdf',
    base_url='templates/modern'
)

# 4. Convert to image
img_renderer = ImageRenderer(dpi=150)
img_renderer.pdf_to_image(
    pdf_path=f'output/{invoice.invoice_number}.pdf',
    image_path=f'output/{invoice.invoice_number}.png'
)
```

## Legacy API (Backward Compatibility)

The original `InvoiceRenderer` and `SyntheticDataGenerator` classes are still available:

```python
from generators import InvoiceRenderer, BatchRenderer
from generators import LegacySyntheticDataGenerator

# Legacy unified renderer
renderer = InvoiceRenderer(
    templates_dir='templates/html',
    output_dir='data/raw'
)

results = renderer.render_invoice(
    template_name='modern_invoice.html',
    data=invoice_dict,
    invoice_id=invoice.invoice_number,
    formats=['html', 'pdf', 'png']
)
```

## Migration Guide

### From Legacy to Modular API

**Before (Legacy):**
```python
from generators import InvoiceRenderer

renderer = InvoiceRenderer('templates/html', 'output')
results = renderer.render_invoice(
    'modern_invoice.html',
    data,
    'INV-001',
    formats=['html', 'pdf', 'png']
)
```

**After (Modular):**
```python
from generators import TemplateRenderer, PDFRenderer, ImageRenderer

# Separate concerns
template_renderer = TemplateRenderer('templates')
pdf_renderer = PDFRenderer('weasyprint')
img_renderer = ImageRenderer(dpi=150)

# Render HTML
html = template_renderer.render_to_file(
    'modern/invoice.html', data, 'output/INV-001.html'
)

# Convert to PDF
pdf_renderer.render_from_html_file(
    'output/INV-001.html', 'output/INV-001.pdf'
)

# Convert to image
img_renderer.pdf_to_image(
    'output/INV-001.pdf', 'output/INV-001.png'
)
```

## Benefits of Modular Design

1. **Separation of Concerns**: Each module has a single responsibility
2. **Testability**: Easier to unit test individual components
3. **Flexibility**: Mix and match renderers (e.g., use wkhtmltopdf instead of weasyprint)
4. **Reusability**: Use components independently in different contexts
5. **Maintainability**: Smaller, focused files are easier to understand and modify

## Template Locations

After restructuring:
- Modern templates: `templates/modern/invoice.html`, `templates/modern/styles.css`
- Classic templates: `templates/classic/invoice.html`, `templates/classic/styles.css`
- Receipt templates: `templates/receipt/invoice.html`, `templates/receipt/styles.css`

## Dependencies

- **Jinja2**: Template rendering
- **WeasyPrint** or **wkhtmltopdf**: PDF generation
- **pdf2image** + **Poppler**: PDF to image conversion
- **Pillow**: Image manipulation
- **Faker**: Synthetic data generation
