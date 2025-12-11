"""
Invoice Renderer
Renders Jinja2 templates to HTML, PDF, and PNG

Supports multipage rendering using config-driven pagination from template_pagination.yaml.
This ensures consistent behavior between test scripts and production dataset generation.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
from jinja2 import Environment, FileSystemLoader, Template
from PIL import Image
import base64
from io import BytesIO

# Add project root to path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import config manager for multipage support
from config.template_config import get_template_config_manager, TemplatePaginationConfig


class InvoiceRenderer:
    """Renders invoice templates to various formats with multipage support"""
    
    def __init__(self, templates_dir: str, output_dir: str = "data/raw", augment_probability: float = 0.0):
        """
        Initialize renderer
        
        Args:
            templates_dir: Path to directory containing HTML templates
            output_dir: Directory for output files
            augment_probability: Probability of applying augmentation (0.0-1.0)
        """
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.augment_probability = augment_probability
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Add custom filters for safe numeric handling
        self.env.filters['to_float'] = self._to_float_filter
        self.env.filters['safe_format'] = self._safe_format_filter
        self.env.filters['currency'] = self._currency_filter
        
        # Template config manager for multipage support
        self.config_manager = get_template_config_manager()
        
        # HTML to PNG renderer (lazy loaded)
        self._html_renderer = None
        self._simple_renderer = None
    
    @staticmethod
    def _to_float_filter(value, default=0.0):
        """
        Jinja2 filter to safely convert any value to float.
        Handles strings like "$12.34", "12.34%", None, etc.
        """
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            # Remove currency symbols and percentage signs
            cleaned = value.replace('$', '').replace('€', '').replace('£', '')
            cleaned = cleaned.replace('%', '').replace(',', '').strip()
            try:
                return float(cleaned) if cleaned else default
            except ValueError:
                return default
        return default
    
    @staticmethod
    def _safe_format_filter(value, format_spec="%.2f", default=0.0):
        """
        Jinja2 filter for safe numeric formatting.
        Usage: {{ tax_amount|safe_format }}
               {{ tax_amount|safe_format("%.0f") }}
        """
        num_value = InvoiceRenderer._to_float_filter(value, default)
        return format_spec % num_value
    
    @staticmethod
    def _currency_filter(value, symbol="$", default=0.0):
        """
        Jinja2 filter to format a numeric value as currency.
        Handles already-formatted strings (passes through if already has symbol).
        Usage: {{ subtotal|currency }}
               {{ total_amount|currency("€") }}
        """
        if value is None:
            return f"{symbol}{default:,.2f}"
        
        # If already a formatted string with currency symbol, pass through
        if isinstance(value, str):
            value_stripped = value.strip()
            if value_stripped.startswith(('$', '€', '£')) or 'FREE' in value_stripped.upper():
                return value_stripped
            # Try to convert to float and format
            num_value = InvoiceRenderer._to_float_filter(value_stripped, default)
            return f"{symbol}{num_value:,.2f}"
        
        # Numeric value - format it
        if isinstance(value, (int, float)):
            return f"{symbol}{value:,.2f}"
        
        return f"{symbol}{default:,.2f}"
    
    def _get_html_renderer(self):
        """Lazy load HTMLToPNGRenderer"""
        if self._html_renderer is None:
            from generators.html_to_png_renderer import HTMLToPNGRenderer
            self._html_renderer = HTMLToPNGRenderer(augment_probability=self.augment_probability)
        return self._html_renderer
    
    def _get_simple_renderer(self):
        """Lazy load SimplePNGRenderer for POS receipts"""
        if self._simple_renderer is None:
            from generators.html_to_png_renderer import SimplePNGRenderer
            self._simple_renderer = SimplePNGRenderer()
        return self._simple_renderer
    
    def _fix_css_paths(self, html_content: str, template_name: str) -> str:
        """
        Convert relative CSS paths to absolute file:// URLs.
        
        This is needed because wkhtmltoimage writes HTML to a temp file,
        breaking relative paths like '../css/style.css'.
        
        Args:
            html_content: HTML string with relative CSS paths
            template_name: Template name to determine base directory
            
        Returns:
            HTML with absolute CSS paths
        """
        import re
        
        # Get the template's directory (e.g., templates/retail/)
        template_path = self.templates_dir / template_name
        template_dir = template_path.parent
        
        def resolve_path(match):
            href = match.group(1)
            # Skip external URLs
            if href.startswith(('http://', 'https://', 'file://')):
                return match.group(0)
            
            # Resolve relative path from template directory
            css_path = (template_dir / href).resolve()
            
            # Convert to file:// URL (Windows compatible)
            file_url = css_path.as_uri()
            
            return f'href="{file_url}"'
        
        # Replace href="..." in link tags
        fixed_html = re.sub(r'href="([^"]+\.css)"', resolve_path, html_content)
        
        return fixed_html
    
    def _inline_css(self, html_content: str, template_name: str) -> str:
        """
        Inline CSS into HTML for wkhtmltoimage compatibility.
        
        wkhtmltoimage has issues loading external CSS even with file:// URLs,
        so we inline the CSS content directly into <style> tags.
        
        Args:
            html_content: HTML string with CSS link tags
            template_name: Template name to determine base directory
            
        Returns:
            HTML with inlined CSS in <style> tags
        """
        import re
        
        # Get the template's directory
        template_path = self.templates_dir / template_name
        template_dir = template_path.parent
        
        css_content_blocks = []
        
        def inline_css(match):
            href = match.group(1)
            # Skip external URLs
            if href.startswith(('http://', 'https://', 'file://')):
                return match.group(0)
            
            # Resolve relative path from template directory
            css_path = (template_dir / href).resolve()
            
            # Read CSS file if it exists
            if css_path.exists():
                try:
                    with open(css_path, 'r', encoding='utf-8') as f:
                        css_content = f.read()
                        css_content_blocks.append(css_content)
                except Exception as e:
                    print(f"Warning: Could not read CSS file {css_path}: {e}")
            
            # Return empty string to remove the link tag
            return ''
        
        # Remove all link tags and collect CSS content
        html_without_links = re.sub(r'<link\s+rel="stylesheet"\s+href="([^"]+\.css)"[^>]*>', inline_css, html_content)
        
        # Insert collected CSS as inline <style> in the <head>
        if css_content_blocks:
            inline_style = f'<style>\n{chr(10).join(css_content_blocks)}\n</style>'
            # Insert before </head>
            html_with_inline = html_without_links.replace('</head>', f'{inline_style}\n</head>')
            return html_with_inline
        
        return html_without_links
    
    def render_html(self,
                   template_name: str,
                   data: Dict[str, Any],
                   output_path: Optional[str] = None) -> str:
        """
        Render template to HTML with inline CSS
        
        Args:
            template_name: Name of template file (e.g., 'modern_invoice.html')
            data: Dictionary with template variables
            output_path: Optional path to save HTML file
            
        Returns:
            Rendered HTML string with inlined CSS
        """
        template = self.env.get_template(template_name)
        html_content = template.render(**data)
        
        # Inline CSS for compatibility
        html_content = self._inline_css(html_content, template_name)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return html_content
    
    def render_to_pdf_wkhtmltopdf(self,
                                  html_path: str,
                                  pdf_path: str,
                                  page_size: str = 'A4',
                                  margin_top: str = '10mm',
                                  margin_right: str = '10mm',
                                  margin_bottom: str = '10mm',
                                  margin_left: str = '10mm') -> bool:
        """
        Convert HTML to PDF using wkhtmltopdf
        
        Args:
            html_path: Path to HTML file
            pdf_path: Output PDF path
            page_size: Page size (A4, Letter, etc.)
            margin_top: Top margin
            margin_right: Right margin
            margin_bottom: Bottom margin
            margin_left: Left margin
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                'wkhtmltopdf',
                '--page-size', page_size,
                '--margin-top', margin_top,
                '--margin-right', margin_right,
                '--margin-bottom', margin_bottom,
                '--margin-left', margin_left,
                '--enable-local-file-access',
                html_path,
                pdf_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            return Path(pdf_path).exists()
        
        except subprocess.CalledProcessError as e:
            print(f"wkhtmltopdf error: {e.stderr}")
            return False
        except FileNotFoundError:
            print("wkhtmltopdf not found. Please install: https://wkhtmltopdf.org/")
            return False
    
    def render_to_pdf_weasyprint(self,
                                 html_content: str,
                                 pdf_path: str) -> bool:
        """
        Convert HTML to PDF using WeasyPrint (alternative to wkhtmltopdf)
        
        Args:
            html_content: HTML string
            pdf_path: Output PDF path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from weasyprint import HTML, CSS
            
            output_file = Path(pdf_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Render PDF
            HTML(string=html_content, base_url=str(self.templates_dir)).write_pdf(
                pdf_path
            )
            
            return output_file.exists()
        
        except ImportError:
            print("WeasyPrint not installed. Install with: pip install weasyprint")
            return False
        except Exception as e:
            print(f"WeasyPrint error: {e}")
            return False
    
    def pdf_to_png(self,
                   pdf_path: str,
                   png_path: str,
                   dpi: int = 150) -> bool:
        """
        Convert PDF to PNG image
        
        Args:
            pdf_path: Path to PDF file
            png_path: Output PNG path
            dpi: Resolution in DPI
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pdf2image import convert_from_path
            
            # Convert PDF to images (returns list, we take first page)
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=1,
                last_page=1
            )
            
            if images:
                output_file = Path(png_path)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                images[0].save(png_path, 'PNG')
                return True
            
            return False
        
        except ImportError:
            print("pdf2image not installed. Install with: pip install pdf2image")
            print("Also requires poppler: https://github.com/oschwartz10612/poppler-windows/releases/")
            return False
        except Exception as e:
            print(f"PDF to PNG conversion error: {e}")
            return False
    
    def render_invoice(self,
                      template_name: str,
                      data: Dict[str, Any],
                      invoice_id: str,
                      formats: list = ['html', 'pdf', 'png'],
                      pdf_backend: str = 'wkhtmltopdf',
                      png_dpi: int = 150,
                      use_multipage: bool = True) -> Dict[str, Any]:
        """
        Render invoice to multiple formats with automatic multipage support.
        
        Uses template configuration from template_pagination.yaml to determine:
        - Whether the template supports multipage rendering
        - Page size and orientation
        - How many items fit per page
        - Pagination context variables for templates
        
        Args:
            template_name: Template filename (e.g., 'retail/online_order_electronics.html')
            data: Invoice data dictionary (must include 'line_items' or 'items')
            invoice_id: Unique invoice identifier
            formats: List of formats to generate ('html', 'pdf', 'png')
            pdf_backend: 'wkhtmltopdf' or 'weasyprint' (for legacy single-page)
            png_dpi: DPI for PNG output (96 recommended for standard page sizes)
            use_multipage: Enable multipage rendering (default: True)
            
        Returns:
            Dictionary with:
                - 'files': List of paths to generated files
                - 'pages': Number of pages rendered
                - 'html': List of HTML file paths (if requested)
                - 'png': List of PNG file paths (if requested)
        """
        results = {
            'files': [],
            'pages': 1,
            'html': [],
            'png': []
        }
        
        # Get template configuration
        config = self.config_manager.get_config(template_name)
        
        # Known item list field names used across templates
        # Order matters - more specific fields first to avoid double-counting
        ITEM_LIST_FIELDS = [
            # Grocery template categories
            'frozen_items', 'refrigerated_items', 'pantry_items', 'produce_items',
            # Pharmacy template categories  
            'rx_items', 'otc_items',
            # Fuel station
            'store_items',
            # Home improvement
            'item_categories',
            # Generic (checked last)
            'products', 'services', 'order_items',
            # Standard fields (most common, checked last)
            'line_items', 'items',
        ]
        
        # Collect all items from all item list fields
        # Use a set to track which items we've already counted
        # Deduplicate by content (description + quantity + unit_price) rather than object id
        all_items = []
        item_field_mapping = []  # Track which field each item came from
        seen_items = set()  # Track unique items by content signature
        
        for field in ITEM_LIST_FIELDS:
            field_items = data.get(field, [])
            if field_items and isinstance(field_items, list):
                for item in field_items:
                    # Create a content-based signature for deduplication
                    # Use description, quantity, and a price field (try multiple formats)
                    desc = item.get('description', item.get('name', item.get('product_name', '')))
                    qty = item.get('quantity', item.get('qty', item.get('quantity_ordered', 0)))
                    # Try to get price in various formats (string or number)
                    price = item.get('unit_price', item.get('price', item.get('rate', item.get('unit_cost', 0))))
                    if isinstance(price, str):
                        # Strip currency symbols and convert to float for comparison
                        price = price.replace('$', '').replace(',', '').strip()
                        try:
                            price = float(price)
                        except (ValueError, AttributeError):
                            price = 0
                    
                    item_signature = (desc, qty, float(price) if price else 0)
                    
                    if item_signature not in seen_items:
                        seen_items.add(item_signature)
                        all_items.append(item)
                        item_field_mapping.append(field)
        
        # Fallback: if no items found in known fields, check line_items/items directly
        if not all_items:
            all_items = data.get('line_items', data.get('items', []))
            item_field_mapping = ['line_items'] * len(all_items)
        
        num_items = len(all_items)
        
        # Calculate pages needed based on config
        if use_multipage and config.supports_multipage:
            num_pages = config.calculate_pages_needed(num_items)
        else:
            num_pages = 1
        
        results['pages'] = num_pages
        
        template = self.env.get_template(template_name)
        
        if num_pages == 1:
            # Single page rendering
            page_data = data.copy()
            page_data['_page_number'] = 1
            page_data['_total_pages'] = 1
            page_data['_is_first_page'] = True
            page_data['_is_last_page'] = True
            
            html_content = template.render(**page_data)
            
            # Inline CSS for wkhtmltoimage compatibility
            html_content = self._inline_css(html_content, template_name)
            
            # Save HTML if requested
            if 'html' in formats:
                html_path = self.output_dir / f"{invoice_id}.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                results['html'].append(str(html_path))
                results['files'].append(str(html_path))
            
            # Render PNG using appropriate renderer based on config
            if 'png' in formats:
                png_path = self.output_dir / f"{invoice_id}.png"
                
                renderer = self._get_html_renderer()
                
                # Handle Custom page sizes
                if config.page_size == 'Custom' and config.custom_width > 0 and config.custom_height > 0:
                    success = renderer.render(
                        html_content,
                        str(png_path),
                        custom_width=config.custom_width,
                        custom_height=config.custom_height,
                        apply_augmentation=None  # Use probability-based augmentation
                    )
                else:
                    success = renderer.render(
                        html_content,
                        str(png_path),
                        page_size=config.page_size,
                        orientation=config.orientation,
                        apply_augmentation=None  # Use probability-based augmentation
                    )
                
                if success:
                    results['png'].append(str(png_path))
                    results['files'].append(str(png_path))
        
        else:
            # Multipage rendering
            for page_num in range(num_pages):
                # Get items for this page using config
                page_items = config.get_items_for_page(page_num, all_items)
                
                # Get the field mapping for this page's items
                start_idx = 0
                if page_num == 0:
                    start_idx = 0
                else:
                    first_page_capacity = config.calculate_first_page_capacity()
                    continuation_capacity = config.calculate_continuation_page_capacity(is_last_page=False)
                    start_idx = first_page_capacity + ((page_num - 1) * continuation_capacity)
                
                end_idx = start_idx + len(page_items)
                page_field_mapping = item_field_mapping[start_idx:end_idx]
                
                # Create page-specific data
                page_data = data.copy()
                
                # Reconstruct all item list fields with only this page's items
                # First, clear all item list fields
                for field in ITEM_LIST_FIELDS:
                    if field in page_data:
                        page_data[field] = []
                
                # Then populate with this page's items, preserving original field assignment
                for item, field in zip(page_items, page_field_mapping):
                    if field not in page_data:
                        page_data[field] = []
                    page_data[field].append(item)
                
                # Also set the generic fields for templates that use them
                page_data['line_items'] = page_items
                page_data['items'] = page_items
                
                # Calculate page subtotal (sum of item totals on this page)
                page_subtotal = 0
                for item in page_items:
                    if isinstance(item, dict):
                        # Skip category rows that don't have totals
                        if item.get('is_category'):
                            continue
                        total_value = item.get('total', item.get('amount', 0))
                        # Handle string totals (e.g., "$10.50", "€15.00", "£20.00", "₹1,000.00", "RM100.00")
                        if isinstance(total_value, str):
                            total_value = float(total_value.replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').replace('RM', '').replace('RM ', '').replace(',', '').replace(' ', ''))
                        page_subtotal += total_value
                    else:
                        total_value = getattr(item, 'total', getattr(item, 'amount', 0))
                        if isinstance(total_value, str):
                            total_value = float(total_value.replace('$', '').replace('€', '').replace('£', '').replace('¥', '').replace('₹', '').replace('RM', '').replace('RM ', '').replace(',', '').replace(' ', ''))
                        page_subtotal += total_value
                page_data['_page_subtotal'] = round(page_subtotal, 2)
                
                # Set pagination context
                page_data['_page_number'] = page_num + 1
                page_data['_current_page'] = page_num + 1
                page_data['_total_pages'] = num_pages
                page_data['_is_first_page'] = (page_num == 0)
                page_data['_is_last_page'] = (page_num == num_pages - 1)
                
                # Note: Don't blank out financial totals here - templates should use
                # _is_last_page conditional to control where totals display
                # Some templates (like Faire) show totals in header on first page
                
                # Render page HTML
                page_html = template.render(**page_data)
                
                # Inline CSS for wkhtmltoimage compatibility
                page_html = self._inline_css(page_html, template_name)
                
                # Create page-specific filenames
                page_suffix = f"_page{page_num + 1}"
                
                # Save HTML if requested
                if 'html' in formats:
                    html_path = self.output_dir / f"{invoice_id}{page_suffix}.html"
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(page_html)
                    results['html'].append(str(html_path))
                    results['files'].append(str(html_path))
                
                # Render PNG using appropriate renderer based on config
                if 'png' in formats:
                    png_path = self.output_dir / f"{invoice_id}{page_suffix}.png"
                    
                    renderer = self._get_html_renderer()
                    
                    # Handle Custom page sizes
                    if config.page_size == 'Custom' and config.custom_width > 0 and config.custom_height > 0:
                        success = renderer.render(
                            page_html,
                            str(png_path),
                            custom_width=config.custom_width,
                            custom_height=config.custom_height,
                            apply_augmentation=None  # Use probability-based augmentation
                        )
                    else:
                        success = renderer.render(
                            page_html,
                            str(png_path),
                            page_size=config.page_size,
                            orientation=config.orientation,
                            apply_augmentation=None  # Use probability-based augmentation
                        )
                    
                    if success:
                        results['png'].append(str(png_path))
                        results['files'].append(str(png_path))
            
            # Create multipage marker file for pipeline tracking
            marker_path = self.output_dir / f"{invoice_id}_MULTIPAGE.txt"
            marker_path.write_text(f"{num_pages} pages\n" + "\n".join(results['png']))
        
        return results
    
    def render_invoice_legacy(self,
                      template_name: str,
                      data: Dict[str, Any],
                      invoice_id: str,
                      formats: list = ['html', 'pdf', 'png'],
                      pdf_backend: str = 'wkhtmltopdf',
                      png_dpi: int = 150) -> Dict[str, str]:
        """
        Legacy render method - single page only via PDF conversion.
        Kept for backward compatibility.
        
        Args:
            template_name: Template filename
            data: Invoice data dictionary
            invoice_id: Unique invoice identifier
            formats: List of formats to generate ('html', 'pdf', 'png')
            pdf_backend: 'wkhtmltopdf' or 'weasyprint'
            png_dpi: DPI for PNG output
            
        Returns:
            Dictionary with paths to generated files
        """
        results = {}
        
        # HTML
        if 'html' in formats:
            html_path = self.output_dir / f"{invoice_id}.html"
            html_content = self.render_html(template_name, data, str(html_path))
            results['html'] = str(html_path)
        else:
            # Generate HTML anyway for PDF conversion
            html_content = self.render_html(template_name, data)
            html_path = self.output_dir / f"{invoice_id}_temp.html"
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        # PDF
        if 'pdf' in formats:
            pdf_path = self.output_dir / f"{invoice_id}.pdf"
            
            if pdf_backend == 'wkhtmltopdf':
                success = self.render_to_pdf_wkhtmltopdf(str(html_path), str(pdf_path))
            elif pdf_backend == 'weasyprint':
                success = self.render_to_pdf_weasyprint(html_content, str(pdf_path))
            else:
                raise ValueError(f"Unknown PDF backend: {pdf_backend}")
            
            if success:
                results['pdf'] = str(pdf_path)
        
        # PNG
        if 'png' in formats and 'pdf' in results:
            png_path = self.output_dir / f"{invoice_id}.png"
            if self.pdf_to_png(results['pdf'], str(png_path), dpi=png_dpi):
                results['png'] = str(png_path)
        
        # Clean up temp HTML if not in formats
        if 'html' not in formats and html_path.exists():
            html_path.unlink()
        
        return results


class BatchRenderer:
    """Batch rendering of multiple invoices"""
    
    def __init__(self, renderer: InvoiceRenderer):
        """
        Initialize batch renderer
        
        Args:
            renderer: InvoiceRenderer instance
        """
        self.renderer = renderer
    
    def render_batch(self,
                    invoices: list,
                    template_name: str,
                    formats: list = ['html', 'pdf', 'png'],
                    pdf_backend: str = 'wkhtmltopdf',
                    png_dpi: int = 150,
                    callback=None) -> list:
        """
        Render multiple invoices
        
        Args:
            invoices: List of (invoice_id, data) tuples
            template_name: Template filename
            formats: Output formats
            pdf_backend: PDF rendering backend
            png_dpi: PNG resolution
            callback: Optional callback function(index, total, invoice_id)
            
        Returns:
            List of results dictionaries
        """
        results = []
        total = len(invoices)
        
        for idx, (invoice_id, data) in enumerate(invoices, 1):
            if callback:
                callback(idx, total, invoice_id)
            
            result = self.renderer.render_invoice(
                template_name=template_name,
                data=data,
                invoice_id=invoice_id,
                formats=formats,
                pdf_backend=pdf_backend,
                png_dpi=png_dpi
            )
            
            results.append({
                'invoice_id': invoice_id,
                'files': result
            })
        
        return results


if __name__ == '__main__':
    # Example usage - demonstrates multipage rendering
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from generators.retail_data_generator import RetailDataGenerator
    from generators.modern_invoice_generator import ModernInvoiceGenerator
    
    print("="*60)
    print("MULTIPAGE INVOICE RENDERER TEST")
    print("="*60)
    
    # Initialize renderer
    renderer = InvoiceRenderer(
        templates_dir='templates',
        output_dir='outputs/renderer_test'
    )
    
    # Test 1: Modern professional invoice with many items
    print("\n1. Testing modern_professional multipage template...")
    invoice_gen = ModernInvoiceGenerator()
    data = invoice_gen.generate_modern_invoice(min_items=15, max_items=15)
    
    results = renderer.render_invoice(
        template_name='modern_professional/invoice_minimal_multipage.html',
        data=data,
        invoice_id='TEST_MODERN_001',
        formats=['png'],
        use_multipage=True
    )
    
    print(f"   Pages rendered: {results['pages']}")
    for png_path in results['png']:
        print(f"   PNG: {png_path}")
    
    # Test 2: Retail electronics order
    print("\n2. Testing retail electronics multipage template...")
    retail_gen = RetailDataGenerator()
    receipt = retail_gen.generate_online_order(
        store_type='electronics',
        min_items=10,
        max_items=10
    )
    data = retail_gen.to_dict(receipt)
    
    results = renderer.render_invoice(
        template_name='retail/online_order_electronics.html',
        data=data,
        invoice_id='TEST_ELECTRONICS_001',
        formats=['png'],
        use_multipage=True
    )
    
    print(f"   Pages rendered: {results['pages']}")
    for png_path in results['png']:
        print(f"   PNG: {png_path}")
    
    # Test 3: Single page (few items)
    print("\n3. Testing single page rendering (few items)...")
    receipt = retail_gen.generate_online_order(
        store_type='grocery',
        min_items=3,
        max_items=3
    )
    data = retail_gen.to_dict(receipt)
    
    results = renderer.render_invoice(
        template_name='retail/online_order_grocery.html',
        data=data,
        invoice_id='TEST_GROCERY_001',
        formats=['png'],
        use_multipage=True
    )
    
    print(f"   Pages rendered: {results['pages']}")
    for png_path in results['png']:
        print(f"   PNG: {png_path}")
    
    print("\n" + "="*60)
    print("All tests completed!")
    print(f"Output directory: outputs/renderer_test")
    print("="*60)
