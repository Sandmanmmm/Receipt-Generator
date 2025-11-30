"""
HTML to PNG Renderer using wkhtmltoimage
Converts rendered HTML templates to PNG images for OCR processing
"""
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import os
import random
import cv2
import numpy as np


class HTMLToPNGRenderer:
    """Renders HTML to PNG using wkhtmltoimage"""
    
    def __init__(self, width: int = 800, height: int = 1200, dpi: int = 96, 
                 augment_probability: float = 0.5):
        """
        Initialize HTML to PNG renderer
        
        Args:
            width: Image width in pixels (default: 800)
            height: Image height in pixels (default: 1200)
            dpi: DPI for rendering (default: 96)
            augment_probability: Probability of applying augmentation (0.0-1.0, default: 0.5)
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.augment_probability = augment_probability
        self._augmenter = None  # Lazy load
        self.wkhtmltoimage_path = 'wkhtmltoimage'  # Default to PATH
        
        # Check if wkhtmltoimage is available
        self._check_wkhtmltoimage()
    
    def _check_wkhtmltoimage(self):
        """Check if wkhtmltoimage is installed"""
        # First check if it's in PATH
        try:
            result = subprocess.run(
                ['wkhtmltoimage', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return  # Found in PATH
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        # Check common Windows installation paths
        common_paths = [
            r"C:\Program Files\wkhtmltopdf\bin\wkhtmltoimage.exe",
            r"C:\Program Files (x86)\wkhtmltopdf\bin\wkhtmltoimage.exe",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                self.wkhtmltoimage_path = path
                return
                
        # If we get here, it wasn't found
        raise RuntimeError(
            "wkhtmltoimage not found. Please install:\n"
            "  Windows: Download from https://wkhtmltopdf.org/downloads.html\n"
            "  Linux: sudo apt-get install wkhtmltopdf\n"
            "  Mac: brew install wkhtmltopdf"
        )
    
    def _get_page_dimensions(self, page_size: str, orientation: Optional[str] = None):
        """
        Get pixel dimensions for standard page sizes at 96 DPI
        
        Args:
            page_size: Page size ('Letter', 'A4', 'Legal', etc.)
            orientation: 'Portrait' or 'Landscape' (defaults to Portrait)
            
        Returns:
            Tuple of (width, height) in pixels
        """
        orientation = orientation or 'Portrait'
        
        # Standard page sizes at 96 DPI
        page_sizes = {
            'Letter': (816, 1056),    # 8.5" x 11"
            'A4': (794, 1123),         # 210mm x 297mm
            'Legal': (816, 1344),      # 8.5" x 14"
            'A3': (1123, 1587),        # 297mm x 420mm
            'Tabloid': (1056, 1632),   # 11" x 17"
        }
        
        width, height = page_sizes.get(page_size, (816, 1056))
        
        # Swap dimensions for landscape
        if orientation.lower() == 'landscape':
            width, height = height, width
        
        return width, height
    
    def _get_augmenter(self):
        """Lazy load augmenter to avoid circular imports"""
        if self._augmenter is None:
            try:
                from augmentation.augmenter import ImageAugmenter, AugmentationConfig
                
                # Create realistic augmentation config with varied effects
                # REDUCED probabilities to prevent over-distortion
                config = AugmentationConfig(
                    # Blur effects
                    add_blur=True,
                    blur_probability=0.3,
                    
                    # Noise
                    add_noise=True,
                    noise_probability=0.4,
                    
                    # Thermal fade
                    add_thermal_fade=True,
                    thermal_fade_probability=0.25,
                    fade_intensity=(0.2, 0.5),
                    
                    # Wrinkles - REDUCED to prevent severe distortion
                    add_wrinkle=True,
                    wrinkle_probability=0.1,  # Reduced from 0.2
                    wrinkle_count=(1, 2),  # Max 2 instead of 3
                    
                    # Coffee stains - REDUCED
                    add_coffee_stain=True,
                    coffee_stain_probability=0.08,  # Reduced from 0.15
                    
                    # Skewed camera - REDUCED angle
                    add_skew=True,
                    skew_probability=0.35,
                    skew_angle=(-3.0, 3.0),  # Reduced from ±6° to ±3°
                    
                    # Misalignment
                    add_misalignment=True,
                    misalignment_probability=0.25,
                    
                    # Contrast variations
                    extreme_contrast=True,
                    extreme_contrast_probability=0.15,
                    
                    # Faint printing
                    add_faint_print=True,
                    faint_print_probability=0.2,
                    faint_intensity=(0.3, 0.6),
                    
                    # Compression
                    add_compression=True,
                    compression_probability=0.3,
                    
                    # Shadow
                    add_shadow=True,
                    shadow_probability=0.2
                )
                
                self._augmenter = ImageAugmenter(config)
            except ImportError:
                print("Warning: Augmentation module not available, skipping augmentation")
                self._augmenter = None
        return self._augmenter
    
    def _apply_augmentation(self, image_path: str, seed: Optional[int] = None) -> bool:
        """
        Apply realistic augmentation to rendered image
        
        Args:
            image_path: Path to image file to augment
            seed: Random seed for consistent augmentation across pages (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            augmenter = self._get_augmenter()
            if augmenter is None:
                return False
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image for augmentation: {image_path}")
                return False
            
            # Apply augmentation with optional seed for consistency
            augmented = augmenter.augment(image, seed=seed)
            
            # Save augmented image
            cv2.imwrite(image_path, augmented)
            return True
            
        except Exception as e:
            print(f"Warning: Augmentation failed: {str(e)}")
            return False
    
    def render(self, html_content: str, output_path: str, 
               custom_width: Optional[int] = None,
               custom_height: Optional[int] = None,
               apply_augmentation: Optional[bool] = None,
               page_size: Optional[str] = None,
               orientation: Optional[str] = None) -> bool:
        """
        Render HTML content to PNG file with optional augmentation
        
        Args:
            html_content: HTML string to render
            output_path: Path to save PNG file
            custom_width: Override default width (optional)
            custom_height: Override default height (optional)
            apply_augmentation: Force augmentation on/off (None = use probability)
            page_size: Page size ('Letter', 'A4', 'Legal', or None for custom)
            orientation: Page orientation ('Portrait' or 'Landscape')
            
        Returns:
            True if successful, False otherwise
        """
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate dimensions based on page size and orientation
        if page_size and not (custom_width and custom_height):
            width, height = self._get_page_dimensions(page_size, orientation)
        else:
            # Use custom dimensions or defaults
            width = custom_width or self.width
            height = custom_height or self.height
        
        # Create temporary HTML file
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.html', 
            delete=False,
            encoding='utf-8'
        ) as tmp_html:
            tmp_html.write(html_content)
            tmp_html_path = tmp_html.name
        
        try:
            # Build wkhtmltoimage command
            cmd = [
                self.wkhtmltoimage_path,
                '--format', 'png',
                '--quality', '100',
                '--enable-local-file-access',  # Allow loading local CSS/images
                '--quiet',  # Suppress output
            ]
            
            # Add dimensions (wkhtmltoimage doesn't support --page-size/--orientation)
            cmd.extend(['--width', str(width), '--height', str(height)])
            
            cmd.extend([tmp_html_path, str(output_file)])
            
            # Run wkhtmltoimage
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                print(f"wkhtmltoimage error: {result.stderr}")
                return False
            
            # Verify output file was created
            if not output_file.exists():
                print(f"Output file not created: {output_path}")
                return False
            
            # Apply augmentation if requested
            should_augment = apply_augmentation if apply_augmentation is not None \
                            else (random.random() < self.augment_probability)
            
            if should_augment and self.augment_probability > 0:
                self._apply_augmentation(str(output_file))
            
            return True
            
        except subprocess.TimeoutExpired:
            print(f"Timeout rendering {output_path}")
            return False
        except Exception as e:
            print(f"Error rendering HTML to PNG: {str(e)}")
            return False
        finally:
            # Clean up temporary HTML file
            try:
                os.unlink(tmp_html_path)
            except:
                pass
    
    def render_multipage_html(self, html_content: str, output_path: str,
                               data: dict, template_name: str,
                               page_width: int = 816, page_height: int = 1056,
                               items_per_page: int = 8,
                               apply_augmentation: Optional[bool] = None) -> bool:
        """
        Render HTML content as multi-page document with proper page breaks.
        Splits line items across pages while preserving header/footer styling.
        
        Args:
            html_content: Original HTML content (used for styling extraction)
            output_path: Base path for output (will create _page1.png, _page2.png, etc.)
            data: Data dictionary with line_items to split
            template_name: Name of template for structure detection
            page_width: Page width in pixels
            page_height: Page height in pixels  
            items_per_page: Maximum items per page
            apply_augmentation: Force augmentation on/off
            
        Returns:
            True if successful, False otherwise
        """
        from pathlib import Path
        import re
        
        output_file = Path(output_path)
        output_dir = output_file.parent
        base_name = output_file.stem
        
        # Get line items from data
        line_items = data.get('line_items', [])
        num_items = len(line_items)
        
        # Calculate number of pages needed
        if num_items <= items_per_page:
            # Single page - use normal render
            return self.render(html_content, output_path, page_width, page_height, apply_augmentation)
        
        num_pages = (num_items + items_per_page - 1) // items_per_page
        
        # Extract CSS from the HTML content
        css_match = re.search(r'<style[^>]*>(.*?)</style>', html_content, re.DOTALL)
        embedded_css = css_match.group(1) if css_match else ""
        
        # Create pages
        pages_created = []
        
        for page_num in range(num_pages):
            start_idx = page_num * items_per_page
            end_idx = min(start_idx + items_per_page, num_items)
            page_items = line_items[start_idx:end_idx]
            
            # Create page data with subset of items
            page_data = data.copy()
            page_data['line_items'] = page_items
            page_data['items'] = page_items  # Alias
            page_data['_page_number'] = page_num + 1
            page_data['_total_pages'] = num_pages
            page_data['_is_first_page'] = (page_num == 0)
            page_data['_is_last_page'] = (page_num == num_pages - 1)
            
            # Only show totals on last page
            if not page_data['_is_last_page']:
                page_data['_hide_totals'] = True
            
            # Generate page HTML
            page_html = self._generate_multipage_html(
                page_data, template_name, embedded_css, page_width, page_height
            )
            
            # Render this page
            page_path = output_dir / f"{base_name}_page{page_num + 1}.png"
            success = self.render(page_html, str(page_path), page_width, page_height, apply_augmentation)
            
            if success:
                pages_created.append(str(page_path))
            else:
                print(f"Failed to render page {page_num + 1}")
                return False
        
        # Create multipage marker file
        marker_path = output_dir / f"{base_name}_MULTIPAGE.txt"
        with open(marker_path, 'w') as f:
            f.write(f"Total pages: {num_pages}\n")
            for i, page_path in enumerate(pages_created, 1):
                f.write(f"Page {i}: {page_path}\n")
        
        return True
    
    def _generate_multipage_html(self, page_data: dict, template_name: str,
                                  css_content: str, page_width: int, page_height: int) -> str:
        """
        Generate HTML for a single page of a multi-page document.
        Creates a standardized layout that works across different template styles.
        """
        page_num = page_data.get('_page_number', 1)
        total_pages = page_data.get('_total_pages', 1)
        is_first = page_data.get('_is_first_page', True)
        is_last = page_data.get('_is_last_page', True)
        hide_totals = page_data.get('_hide_totals', False)
        
        # Extract brand color
        brand_color = page_data.get('brand_primary_color', '#2c3e50')
        
        # Build HTML
        html_parts = [f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Page {page_num} of {total_pages}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 12px;
            line-height: 1.4;
            color: #333;
            width: {page_width}px;
            min-height: {page_height}px;
            padding: 30px;
            background: white;
        }}
        .page-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 3px solid {brand_color};
            margin-bottom: 20px;
        }}
        .company-info h1 {{
            font-size: 22px;
            color: {brand_color};
            margin-bottom: 5px;
        }}
        .company-info p {{
            font-size: 11px;
            color: #666;
        }}
        .page-indicator {{
            text-align: right;
            font-size: 11px;
            color: #888;
        }}
        .page-indicator .page-num {{
            font-size: 16px;
            font-weight: bold;
            color: {brand_color};
        }}
        .doc-info {{
            background: #f8f9fa;
            padding: 15px;
            margin-bottom: 20px;
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
        }}
        .doc-info-item {{
            font-size: 11px;
        }}
        .doc-info-item .label {{
            color: #888;
            text-transform: uppercase;
            font-size: 9px;
            letter-spacing: 0.5px;
        }}
        .doc-info-item .value {{
            font-weight: 600;
            margin-top: 3px;
        }}
        .addresses {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .address-box {{
            padding: 15px;
            background: #f8f9fa;
            border-left: 3px solid {brand_color};
        }}
        .address-box h3 {{
            font-size: 10px;
            text-transform: uppercase;
            color: #888;
            margin-bottom: 8px;
        }}
        .address-box p {{
            font-size: 12px;
            line-height: 1.5;
        }}
        .items-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        .items-table thead {{
            background: {brand_color};
            color: white;
        }}
        .items-table th {{
            padding: 12px 10px;
            text-align: left;
            font-size: 10px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        .items-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid #e0e0e0;
            font-size: 11px;
        }}
        .items-table tr:hover {{
            background: #f8f9fa;
        }}
        .item-desc {{
            font-weight: 500;
        }}
        .item-sku {{
            font-size: 9px;
            color: #888;
        }}
        .totals-section {{
            margin-top: 20px;
            display: flex;
            justify-content: flex-end;
        }}
        .totals-box {{
            width: 280px;
            border: 2px solid {brand_color};
            padding: 15px;
        }}
        .total-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 12px;
        }}
        .total-row.subtotal {{
            border-bottom: 1px solid #e0e0e0;
        }}
        .total-row.grand {{
            border-top: 2px solid {brand_color};
            margin-top: 10px;
            padding-top: 12px;
            font-size: 16px;
            font-weight: bold;
            color: {brand_color};
        }}
        .page-footer {{
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid #e0e0e0;
            text-align: center;
            font-size: 10px;
            color: #888;
        }}
        .continuation-notice {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 10px;
            text-align: center;
            font-size: 11px;
            color: #856404;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
''']
        
        # Page header (all pages)
        supplier_name = page_data.get('supplier_name', page_data.get('company_name', 'Company'))
        supplier_address = page_data.get('supplier_address', page_data.get('company_address', ''))
        supplier_phone = page_data.get('supplier_phone', page_data.get('company_phone', ''))
        
        html_parts.append(f'''
    <div class="page-header">
        <div class="company-info">
            <h1>{supplier_name}</h1>
            <p>{supplier_address}</p>
            <p>{supplier_phone}</p>
        </div>
        <div class="page-indicator">
            <div>Page</div>
            <div class="page-num">{page_num} / {total_pages}</div>
        </div>
    </div>
''')
        
        # Document info (first page only - full, other pages - minimal)
        invoice_num = page_data.get('invoice_number', page_data.get('order_number', 'N/A'))
        invoice_date = page_data.get('invoice_date', page_data.get('order_date', ''))
        
        if is_first:
            html_parts.append(f'''
    <div class="doc-info">
        <div class="doc-info-item">
            <div class="label">Invoice/Order #</div>
            <div class="value">{invoice_num}</div>
        </div>
        <div class="doc-info-item">
            <div class="label">Date</div>
            <div class="value">{invoice_date}</div>
        </div>
        <div class="doc-info-item">
            <div class="label">Payment Method</div>
            <div class="value">{page_data.get('payment_method', 'N/A')}</div>
        </div>
    </div>
''')
            
            # Addresses (first page only)
            buyer_name = page_data.get('buyer_name', page_data.get('client_name', ''))
            buyer_address = page_data.get('buyer_address', page_data.get('client_address', ''))
            
            html_parts.append(f'''
    <div class="addresses">
        <div class="address-box">
            <h3>Bill To</h3>
            <p><strong>{buyer_name}</strong></p>
            <p>{buyer_address}</p>
        </div>
        <div class="address-box">
            <h3>Ship To</h3>
            <p><strong>{page_data.get('shipping_name', buyer_name)}</strong></p>
            <p>{page_data.get('shipping_address', buyer_address)}</p>
        </div>
    </div>
''')
        else:
            # Minimal info for continuation pages
            html_parts.append(f'''
    <div class="doc-info" style="grid-template-columns: 1fr 1fr;">
        <div class="doc-info-item">
            <div class="label">Invoice/Order #</div>
            <div class="value">{invoice_num}</div>
        </div>
        <div class="doc-info-item">
            <div class="label">Date</div>
            <div class="value">{invoice_date}</div>
        </div>
    </div>
''')
        
        # Items table
        currency = page_data.get('currency', page_data.get('currency_symbol', '$'))
        
        html_parts.append('''
    <table class="items-table">
        <thead>
            <tr>
                <th style="width: 45%;">Description</th>
                <th style="width: 15%;">SKU</th>
                <th style="width: 10%; text-align: center;">Qty</th>
                <th style="width: 15%; text-align: right;">Unit Price</th>
                <th style="width: 15%; text-align: right;">Total</th>
            </tr>
        </thead>
        <tbody>
''')
        
        for item in page_data.get('line_items', []):
            desc = item.get('description', '')
            sku = item.get('sku', item.get('upc', ''))
            qty = item.get('quantity', 1)
            
            # Parse unit_price and total - handle both string ($5.99) and numeric formats
            unit_price_val = item.get('unit_price', item.get('rate', 0))
            if isinstance(unit_price_val, str):
                try:
                    unit_price_val = float(unit_price_val.replace('$', '').replace(',', ''))
                except (ValueError, AttributeError):
                    unit_price_val = 0.0
            unit_price = float(unit_price_val)
            
            total_val = item.get('total', item.get('amount', unit_price * qty))
            if isinstance(total_val, str):
                try:
                    total_val = float(total_val.replace('$', '').replace(',', ''))
                except (ValueError, AttributeError):
                    total_val = unit_price * qty
            total = float(total_val)
            
            html_parts.append(f'''
            <tr>
                <td class="item-desc">{desc}</td>
                <td class="item-sku">{sku}</td>
                <td style="text-align: center;">{qty}</td>
                <td style="text-align: right;">{currency}{unit_price:.2f}</td>
                <td style="text-align: right;"><strong>{currency}{total:.2f}</strong></td>
            </tr>
''')
        
        html_parts.append('''
        </tbody>
    </table>
''')
        
        # Continuation notice or totals
        if not is_last:
            html_parts.append(f'''
    <div class="continuation-notice">
        Continued on page {page_num + 1}...
    </div>
''')
        else:
            # Totals section (last page only)
            # Parse currency strings to floats
            def parse_amount(val, default=0.0):
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, str):
                    try:
                        return float(val.replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        return default
                return default
            
            subtotal = parse_amount(page_data.get('subtotal'))
            tax = parse_amount(page_data.get('tax', page_data.get('tax_amount', 0)))
            total = parse_amount(page_data.get('total', page_data.get('total_amount')))
            if total == 0:
                total = subtotal + tax
            tax_rate = page_data.get('tax_rate', '')
            
            html_parts.append(f'''
    <div class="totals-section">
        <div class="totals-box">
            <div class="total-row subtotal">
                <span>Subtotal</span>
                <span>{currency}{subtotal:.2f}</span>
            </div>
            <div class="total-row">
                <span>Tax ({tax_rate})</span>
                <span>{currency}{tax:.2f}</span>
            </div>
            <div class="total-row grand">
                <span>TOTAL</span>
                <span>{currency}{total:.2f}</span>
            </div>
        </div>
    </div>
''')
        
        # Footer
        notes = page_data.get('notes', page_data.get('footer_message', 'Thank you for your business!'))
        html_parts.append(f'''
    <div class="page-footer">
        <p>{notes}</p>
        <p style="margin-top: 5px;">{supplier_name} • {supplier_phone}</p>
    </div>
</body>
</html>
''')
        
        return ''.join(html_parts)

    def render_file(self, html_path: str, output_path: str,
                    custom_width: Optional[int] = None,
                    custom_height: Optional[int] = None) -> bool:
        """
        Render HTML file to PNG
        
        Args:
            html_path: Path to HTML file
            output_path: Path to save PNG file
            custom_width: Override default width (optional)
            custom_height: Override default height (optional)
            
        Returns:
            True if successful, False otherwise
        """
        # Read HTML file
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return self.render(html_content, output_path, custom_width, custom_height, None)
    
    def batch_render(self, html_files: list, output_dir: str) -> dict:
        """
        Render multiple HTML files to PNG
        
        Args:
            html_files: List of HTML file paths
            output_dir: Directory to save PNG files
            
        Returns:
            Dict with success/failure counts and failed files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {
            'success': 0,
            'failed': 0,
            'failed_files': []
        }
        
        for html_file in html_files:
            html_path = Path(html_file)
            output_file = output_path / f"{html_path.stem}.png"
            
            success = self.render_file(str(html_path), str(output_file))
            
            if success:
                results['success'] += 1
            else:
                results['failed'] += 1
                results['failed_files'].append(str(html_file))
        
        return results


# Fallback: PIL-based text renderer (if wkhtmltoimage not available)
class SimplePNGRenderer:
    """
    Fallback renderer using PIL for simple text-based receipts
    Use when wkhtmltoimage is not available
    """
    
    def __init__(self, width: int = 800, height: int = 1200, 
                 font_size: int = 16, line_spacing: int = 30,
                 augment_probability: float = 0.5):
        """
        Initialize simple PNG renderer
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            font_size: Font size for text
            line_spacing: Pixels between lines
            augment_probability: Probability of applying augmentation (0.0-1.0, default: 0.5)
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.line_spacing = line_spacing
        self.augment_probability = augment_probability
        self._augmenter = None  # Lazy load
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            self.Image = Image
            self.ImageDraw = ImageDraw
            self.ImageFont = ImageFont
        except ImportError:
            raise RuntimeError("Pillow not installed. Install with: pip install Pillow")
    
    def _get_augmenter(self):
        """Lazy load augmenter (shared with HTMLToPNGRenderer)"""
        if self._augmenter is None:
            try:
                from augmentation.augmenter import ImageAugmenter, AugmentationConfig
                
                # REDUCED probabilities to prevent over-distortion
                config = AugmentationConfig(
                    add_blur=True, blur_probability=0.3,
                    add_noise=True, noise_probability=0.4,
                    add_thermal_fade=True, thermal_fade_probability=0.25, fade_intensity=(0.2, 0.5),
                    add_wrinkle=True, wrinkle_probability=0.1, wrinkle_count=(1, 2),  # REDUCED
                    add_coffee_stain=True, coffee_stain_probability=0.08,  # REDUCED
                    add_skew=True, skew_probability=0.35, skew_angle=(-3.0, 3.0),  # REDUCED angle
                    add_misalignment=True, misalignment_probability=0.25,
                    extreme_contrast=True, extreme_contrast_probability=0.15,
                    add_faint_print=True, faint_print_probability=0.2, faint_intensity=(0.3, 0.6),
                    add_compression=True, compression_probability=0.3,
                    add_shadow=True, shadow_probability=0.2
                )
                
                self._augmenter = ImageAugmenter(config)
            except ImportError:
                self._augmenter = None
        return self._augmenter
    
    def _apply_augmentation(self, image_path: str, seed: Optional[int] = None) -> bool:
        """Apply augmentation to rendered image with optional seed for consistency"""
        try:
            augmenter = self._get_augmenter()
            if augmenter is None:
                return False
            
            image = cv2.imread(image_path)
            if image is None:
                return False
            
            # Apply augmentation with optional seed for consistency
            augmented = augmenter.augment(image, seed=seed)
            cv2.imwrite(image_path, augmented)
            return True
        except Exception as e:
            print(f"Warning: Augmentation failed: {str(e)}")
            return False
    
    def render_text_receipt(self, text_lines: list, output_path: str, receipt_type: str = 'retail') -> bool:
        """
        Render text lines to PNG image
        
        Args:
            text_lines: List of text lines to render
            output_path: Path to save PNG file
            receipt_type: 'retail' for continuous roll (no height limit), 
                         'invoice' for standard page size (letter/A4)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate dynamic height based on receipt type
            if receipt_type == 'retail':
                # Continuous roll - calculate exact height needed
                # Add margin at top (50px) and bottom (100px) plus line spacing
                calculated_height = 50 + (len(text_lines) * self.line_spacing) + 100
                
                # CRITICAL: Cap maximum height to prevent OCR failures
                # PaddleOCR has max_side_limit of 4000px by default
                # Keep images under 3500px to be safe
                MAX_SAFE_HEIGHT = 3500
                
                if calculated_height > MAX_SAFE_HEIGHT:
                    # Image would be too tall - this should have been handled by multipage logic
                    # Force it to max height and log warning
                    print(f"Warning: Receipt height {calculated_height}px exceeds max {MAX_SAFE_HEIGHT}px. Content may be truncated.")
                    print(f"  Receipt has {len(text_lines)} lines. Consider using multipage rendering.")
                    image_height = MAX_SAFE_HEIGHT
                else:
                    image_height = calculated_height
            else:
                # Invoice/standard page - use standard letter size height
                # Letter: 8.5" x 11" at 96 DPI = 816 x 1056 pixels
                # A4: 210mm x 297mm at 96 DPI = 794 x 1123 pixels
                image_height = 1056  # Letter size height
            
            # Create white background with calculated height
            image = self.Image.new('RGB', (self.width, image_height), 'white')
            draw = self.ImageDraw.Draw(image)
            
            # Try to load a monospace font
            try:
                font = self.ImageFont.truetype('consola.ttf', self.font_size)
            except:
                try:
                    font = self.ImageFont.truetype('courier.ttf', self.font_size)
                except:
                    # Fall back to default font
                    font = self.ImageFont.load_default()
            
            # Draw text lines
            y_offset = 50
            for line in text_lines:
                draw.text((50, y_offset), line, fill='black', font=font)
                y_offset += self.line_spacing
                
                # For invoice type, if content exceeds page, we need to expand the canvas
                # This is a safety mechanism - proper pagination should have happened earlier
                if receipt_type == 'invoice' and y_offset > image_height - 50:
                    # Expand the image to accommodate more content
                    new_height = image_height + 1056  # Add another page worth of space
                    new_image = self.Image.new('RGB', (self.width, new_height), 'white')
                    new_image.paste(image, (0, 0))
                    image = new_image
                    draw = self.ImageDraw.Draw(image)
                    image_height = new_height
            
            # Save image
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(output_file), 'PNG')
            
            # Apply augmentation if enabled
            if self.augment_probability > 0 and random.random() < self.augment_probability:
                self._apply_augmentation(str(output_file))
            
            return True
            
        except Exception as e:
            print(f"Error rendering text to PNG: {str(e)}")
            return False
    
    def _generate_header(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate header with 40+ archetype variations
        
        Returns:
            List of text lines for header section
        """
        import random
        text_lines = []
        
        # Safe string extraction with fallbacks
        store_name = str(receipt_data.get('supplier_name', 'Store Name'))[:width-4]
        address = str(receipt_data.get('supplier_address', ''))[:width-4]
        phone = str(receipt_data.get('supplier_phone', ''))[:width-4]
        email = str(receipt_data.get('supplier_email', ''))[:width-4]
        website = str(receipt_data.get('store_website', ''))[:width-4]
        store_category = receipt_data.get('store_category', '')
        
        # Choose divider chars for this header (ASCII-safe fallbacks)
        try:
            heavy_div = random.choice(['=', '#', '*', '█']) * width
        except:
            heavy_div = '=' * width
            
        try:
            light_div = random.choice(['-', '.', '_', '~']) * width
        except:
            light_div = '-' * width
        
        # 40+ header archetypes - base types (always available)
        base_header_types = [
            # === SIMPLE ALIGNMENT VARIATIONS (1-8) ===
            'centered_simple', 'left_aligned', 'right_aligned', 'centered_uppercase',
            'left_uppercase', 'right_uppercase', 'staggered_center', 'justified_spread',
            
            # === BOXED VARIATIONS (9-16) ===
            'full_box_single', 'full_box_double', 'top_bottom_box', 'side_frame_box',
            'rounded_corners', 'thick_border_box', 'minimal_corner_box', 'shadow_box',
            
            # === BANNER STYLES (17-24) ===
            'star_banner', 'hash_banner', 'double_line_banner', 'wave_banner',
            'bracket_banner', 'arrow_banner', 'diamond_banner', 'vertical_bar_banner',
            
            # === LOGO REPRESENTATIONS (25-32) ===
            'large_logo_center', 'small_logo_left', 'small_logo_right', 'no_logo_minimal',
            'ascii_logo_art', 'logo_with_tagline', 'logo_with_seal', 'stacked_logo',
            
            # === SECTIONAL HEADERS (33-40) ===
            'two_column_layout', 'three_section_header', 'blocked_sections', 'inline_sections',
            'tabular_header', 'receipt_title_header', 'invoice_style_header', 'formal_letterhead',
            
            # === UNIVERSAL SPECIAL FORMATS (41-42) ===
            'watermark_style', 'retail_pos_style'
        ]
        
        # Category-specific header styles (only used when category matches)
        category_specific_styles = {
            'food_beverage': ['restaurant_style', 'grocery_style'],
            'health_wellness': ['pharmacy_style'],
            'fashion': ['boutique_style', 'department_store_style'],
            'accessories': ['boutique_style', 'department_store_style'],
            'jewelry': ['boutique_style'],
            'beauty': ['boutique_style', 'department_store_style']
        }
        
        # Build available header types based on store category
        header_types = base_header_types.copy()
        if store_category in category_specific_styles:
            header_types.extend(category_specific_styles[store_category])
        
        header_type = random.choice(header_types)
        
        # === IMPLEMENTATION OF EACH ARCHETYPE ===
        
        # --- SIMPLE ALIGNMENT VARIATIONS (1-8) ---
        if header_type == 'centered_simple':
            text_lines.append(f"{store_name:^{width}}")
            if random.random() < 0.5:
                text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        elif header_type == 'left_aligned':
            text_lines.append(store_name)
            text_lines.append(address)
            text_lines.append(f"Phone: {phone}")
            
        elif header_type == 'right_aligned':
            text_lines.append(f"{store_name:>{width}}")
            text_lines.append(f"{address:>{width}}")
            text_lines.append(f"{phone:>{width}}")
            
        elif header_type == 'centered_uppercase':
            text_lines.append(f"{store_name.upper():^{width}}")
            text_lines.append(light_div)
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        elif header_type == 'left_uppercase':
            text_lines.append(store_name.upper())
            text_lines.append(address)
            text_lines.append(phone)
            if random.random() < 0.4 and email:
                text_lines.append(email)
                
        elif header_type == 'right_uppercase':
            text_lines.append(f"{store_name.upper():>{width}}")
            text_lines.append(f"{address:>{width}}")
            text_lines.append(f"Tel: {phone:>{width-5}}")
            
        elif header_type == 'staggered_center':
            text_lines.append(f"{store_name:^{width}}")
            text_lines.append(f"{address:<{width}}")
            text_lines.append(f"{phone:>{width}}")
            
        elif header_type == 'justified_spread':
            text_lines.append(f"{store_name:<{width//2}}{phone:>{width//2}}")
            text_lines.append(f"{address:^{width}}")
            
        # --- BOXED VARIATIONS (9-16) ---
        elif header_type == 'full_box_single':
            text_lines.append(heavy_div)
            text_lines.append(f"| {store_name:^{width-4}} |")
            text_lines.append(f"| {address:^{width-4}} |")
            text_lines.append(f"| {phone:^{width-4}} |")
            text_lines.append(heavy_div)
            
        elif header_type == 'full_box_double':
            text_lines.append('╔' + '═' * (width-2) + '╗')
            text_lines.append(f"║ {store_name:^{width-4}} ║")
            text_lines.append(f"║ {address:^{width-4}} ║")
            text_lines.append(f"║ {phone:^{width-4}} ║")
            text_lines.append('╚' + '═' * (width-2) + '╝')
            
        elif header_type == 'top_bottom_box':
            text_lines.append(heavy_div)
            text_lines.append(f"{store_name.upper():^{width}}")
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            text_lines.append(heavy_div)
            
        elif header_type == 'side_frame_box':
            text_lines.append(f"║ {store_name:<{width-4}} ║")
            text_lines.append(f"║ {address:<{width-4}} ║")
            text_lines.append(f"║ {phone:<{width-4}} ║")
            
        elif header_type == 'rounded_corners':
            text_lines.append('╭' + '─' * (width-2) + '╮')
            text_lines.append(f"│ {store_name:^{width-4}} │")
            text_lines.append(f"│ {address:^{width-4}} │")
            text_lines.append('╰' + '─' * (width-2) + '╯')
            
        elif header_type == 'thick_border_box':
            text_lines.append('█' * width)
            text_lines.append(f"█ {store_name.upper():^{width-4}} █")
            text_lines.append(f"█ {address:^{width-4}} █")
            text_lines.append('█' * width)
            
        elif header_type == 'minimal_corner_box':
            text_lines.append(f"┌{'─' * (width-2)}┐")
            text_lines.append(f"  {store_name:^{width-4}}  ")
            text_lines.append(f"  {address:^{width-4}}  ")
            text_lines.append(f"└{'─' * (width-2)}┘")
            
        elif header_type == 'shadow_box':
            text_lines.append('▄' * width)
            text_lines.append(f"█ {store_name:^{width-4}} █")
            text_lines.append(f"█ {address:^{width-4}} █")
            text_lines.append('▀' * width)
            
        # --- BANNER STYLES (17-24) ---
        elif header_type == 'star_banner':
            text_lines.append('*' * width)
            text_lines.append(f"***  {store_name:^{width-8}}  ***")
            text_lines.append('*' * width)
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'hash_banner':
            text_lines.append('#' * width)
            text_lines.append(f"#  {store_name.upper():^{width-6}}  #")
            text_lines.append('#' * width)
            
        elif header_type == 'double_line_banner':
            text_lines.append('=' * width)
            text_lines.append('-' * width)
            text_lines.append(f"{store_name:^{width}}")
            text_lines.append('-' * width)
            text_lines.append('=' * width)
            
        elif header_type == 'wave_banner':
            text_lines.append('~' * width)
            text_lines.append(f"~ {store_name:^{width-4}} ~")
            text_lines.append('~' * width)
            
        elif header_type == 'bracket_banner':
            text_lines.append(f"[[ {store_name.upper():^{width-6}} ]]")
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        elif header_type == 'arrow_banner':
            text_lines.append(f">>> {store_name} <<<")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'diamond_banner':
            text_lines.append(f"◆ {store_name:^{width-4}} ◆")
            text_lines.append(f"◇ {address:^{width-4}} ◇")
            
        elif header_type == 'vertical_bar_banner':
            text_lines.append('|' * width)
            text_lines.append(f"| {store_name:^{width-4}} |")
            text_lines.append('|' * width)
            
        # --- LOGO REPRESENTATIONS (25-32) ---
        elif header_type == 'large_logo_center':
            # ASCII art logo representation
            text_lines.append(f"{'╔═══╗':^{width}}")
            text_lines.append(f"{'║ ◊ ║':^{width}}")
            text_lines.append(f"{'╚═══╝':^{width}}")
            text_lines.append('')
            text_lines.append(f"{store_name:^{width}}")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'small_logo_left':
            text_lines.append(f"[◊] {store_name}")
            text_lines.append(f"    {address}")
            text_lines.append(f"    {phone}")
            
        elif header_type == 'small_logo_right':
            text_lines.append(f"{store_name:<{width-6}} [◊]")
            text_lines.append(f"{address:>{width}}")
            
        elif header_type == 'no_logo_minimal':
            text_lines.append(store_name)
            text_lines.append(phone)
            
        elif header_type == 'ascii_logo_art':
            text_lines.append(f"  ___  ")
            text_lines.append(f" /   \\ ")
            text_lines.append(f"|  ◊  |")
            text_lines.append(f" \\___/ ")
            text_lines.append(f"{store_name:^{width}}")
            
        elif header_type == 'logo_with_tagline':
            text_lines.append(f"{'[LOGO]':^{width}}")
            text_lines.append(f"{store_name.upper():^{width}}")
            taglines = ["Est. 1995", "Family Owned", "Since 2005", "Trusted Quality"]
            text_lines.append(f"{random.choice(taglines):^{width}}")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'logo_with_seal':
            text_lines.append(f"  ◉ {store_name} ◉")
            text_lines.append(f"{'CERTIFIED':^{width}}")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'stacked_logo':
            text_lines.append(f"{'▓▓▓':^{width}}")
            text_lines.append(f"{'▓ ◊ ▓':^{width}}")
            text_lines.append(f"{'▓▓▓':^{width}}")
            text_lines.append(f"{store_name:^{width}}")
            
        # --- SECTIONAL HEADERS (33-40) ---
        elif header_type == 'two_column_layout':
            # Seller left, Buyer right (if buyer exists)
            buyer = receipt_data.get('buyer_name', '')
            if buyer and len(buyer) > 0:
                col_width = width // 2 - 2
                text_lines.append(f"{'SELLER':<{col_width}} | {'BUYER':>{col_width}}")
                text_lines.append(light_div)
                text_lines.append(f"{store_name:<{col_width}} | {buyer:>{col_width}}")
                text_lines.append(f"{phone:<{col_width}} |")
            else:
                text_lines.append(store_name)
                text_lines.append(address)
                
        elif header_type == 'three_section_header':
            text_lines.append(f"{'RECEIPT':^{width}}")
            text_lines.append(light_div)
            text_lines.append(f"From: {store_name}")
            text_lines.append(f"Location: {address}")
            text_lines.append(f"Contact: {phone}")
            
        elif header_type == 'blocked_sections':
            text_lines.append("┌─ MERCHANT INFO ─┐")
            text_lines.append(f"  {store_name}")
            text_lines.append(f"  {address}")
            text_lines.append(f"  {phone}")
            text_lines.append("└──────────────────┘")
            
        elif header_type == 'inline_sections':
            text_lines.append(f"STORE: {store_name} | TEL: {phone}")
            text_lines.append(f"ADDR: {address}")
            
        elif header_type == 'tabular_header':
            text_lines.append(f"{'Store:':<10} {store_name}")
            text_lines.append(f"{'Address:':<10} {address}")
            text_lines.append(f"{'Phone:':<10} {phone}")
            if email and random.random() < 0.5:
                text_lines.append(f"{'Email:':<10} {email}")
                
        elif header_type == 'receipt_title_header':
            text_lines.append(f"{'RECEIPT':^{width}}")
            text_lines.append(heavy_div)
            text_lines.append(f"{store_name:^{width}}")
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        elif header_type == 'invoice_style_header':
            text_lines.append(f"{store_name.upper():<{width//2}}{'INVOICE':>{width//2}}")
            text_lines.append(light_div)
            text_lines.append(address)
            text_lines.append(phone)
            
        elif header_type == 'formal_letterhead':
            text_lines.append('')
            text_lines.append(f"{store_name.upper():^{width}}")
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"Phone: {phone:^{width-7}}")
            if email:
                text_lines.append(f"Email: {email:^{width-7}}")
            text_lines.append('')
            text_lines.append(heavy_div)
            
        # --- SPECIAL FORMATS (41-48) ---
        elif header_type == 'watermark_style':
            text_lines.append('░'*5 + f" {store_name} " + '░'*5)
            text_lines.append(f"{'▒'*width:^{width}}")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'retail_pos_style':
            text_lines.append(f"╔{'═'*20}╗")
            text_lines.append(f"║ {store_name[:18]:^18} ║")
            text_lines.append(f"╚{'═'*20}╝")
            text_lines.append(f"{address}")
            text_lines.append(f"Tel: {phone}")
            
        elif header_type == 'restaurant_style':
            # Only for food_beverage category
            text_lines.append(f"~*~ {store_name} ~*~")
            text_lines.append(f"{address:^{width}}")
            tagline = random.choice(['Reservations:', 'Call to Order:', 'Dine In or Takeout:'])
            text_lines.append(f"{tagline:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        elif header_type == 'pharmacy_style':
            # Only for health_wellness category
            text_lines.append(f"{'℞'*3} {store_name} {'℞'*3}")
            text_lines.append(f"{'HEALTH & WELLNESS':^{width}}")
            text_lines.append(f"{address}")
            text_lines.append(f"Phone: {phone}")
            
        elif header_type == 'grocery_style':
            # Only for food_beverage category
            text_lines.append(f"{store_name.upper()}")
            tagline = random.choice(['Fresh Foods & More', 'Quality Ingredients', 'Gourmet Selections'])
            text_lines.append(f"{tagline:^{width}}")
            text_lines.append(light_div)
            text_lines.append(f"{address}")
            
        elif header_type == 'boutique_style':
            # For fashion, accessories, jewelry, beauty
            text_lines.append(f"✦ ✦ ✦")
            text_lines.append(f"{store_name:^{width}}")
            # Category-appropriate taglines
            if store_category == 'fashion':
                tagline = 'Curated Fashion'
            elif store_category == 'jewelry':
                tagline = 'Fine Jewelry'
            elif store_category == 'beauty':
                tagline = 'Beauty & Style'
            elif store_category == 'accessories':
                tagline = 'Accessories & More'
            else:
                tagline = 'Quality Products'
            text_lines.append(f"{tagline:^{width}}")
            text_lines.append(f"✦ ✦ ✦")
            text_lines.append(f"{address:^{width}}")
            
        elif header_type == 'department_store_style':
            # For fashion, accessories, beauty
            text_lines.append(heavy_div)
            text_lines.append(f"{store_name.upper():^{width}}")
            # Generic enough for multiple categories
            tagline = random.choice(['RETAIL STORE', 'SHOP & SAVE', 'QUALITY PRODUCTS'])
            text_lines.append(f"{tagline:^{width}}")
            text_lines.append(heavy_div)
            text_lines.append(f"{address:^{width}}")
            text_lines.append(f"{phone:^{width}}")
            
        # Add optional tagline/motto (20% of receipts)
        if random.random() < 0.2:
            taglines = [
                "Thank you for shopping with us!",
                "Quality You Can Trust",
                "Where Value Meets Style",
                "Your Satisfaction, Our Priority",
                "Serving the Community Since 1995",
                "Est. 2005 - Family Owned",
                "Low Prices, High Quality",
                "The Store That Saves You More",
                "We Love Our Customers!",
                "Shop Local, Save Big"
            ]
            tagline = random.choice(taglines)
            try:
                text_lines.append(f"{tagline:^{width}}")
            except:
                text_lines.append(tagline)
            
        return text_lines

    def _generate_supplier_section(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate supplier/merchant section with 12 variants
        
        Returns:
            List of text lines for supplier section
        """
        import random
        text_lines = []
        
        # Extract supplier data safely
        supplier_name = str(receipt_data.get('supplier_name', 'Merchant'))[:width-4]
        address = str(receipt_data.get('supplier_address', ''))[:width-4]
        phone = str(receipt_data.get('supplier_phone', ''))[:width-4]
        email = str(receipt_data.get('supplier_email', ''))[:width-4]
        website = str(receipt_data.get('store_website', ''))[:width-4]
        tax_id = str(receipt_data.get('tax_id', ''))[:width-4] if 'tax_id' in receipt_data else ''
        
        # 12 supplier section variants
        supplier_variants = [
            'minimal_company_only',
            'full_block_left',
            'full_block_centered',
            'right_aligned_full',
            'bordered_card',
            'boxed_formal',
            'vertical_layout',
            'horizontal_compact',
            'tabular_format',
            'labeled_fields',
            'two_column_split',
            'minimal_with_divider'
        ]
        
        variant = random.choice(supplier_variants)
        
        # === IMPLEMENTATION OF EACH VARIANT ===
        
        if variant == 'minimal_company_only':
            # Just company name and phone
            text_lines.append(supplier_name)
            text_lines.append(phone)
            
        elif variant == 'full_block_left':
            # Traditional left-aligned block
            text_lines.append("MERCHANT INFORMATION")
            text_lines.append('-' * min(width, 25))
            text_lines.append(supplier_name)
            if address:
                text_lines.append(address)
            if phone:
                text_lines.append(f"Phone: {phone}")
            if email and random.random() < 0.6:
                text_lines.append(f"Email: {email}")
            if tax_id and random.random() < 0.4:
                text_lines.append(f"Tax ID: {tax_id}")
                
        elif variant == 'full_block_centered':
            # Centered block with all info
            text_lines.append(f"{supplier_name:^{width}}")
            if address:
                text_lines.append(f"{address:^{width}}")
            if phone:
                text_lines.append(f"{phone:^{width}}")
            if email and random.random() < 0.5:
                text_lines.append(f"{email:^{width}}")
                
        elif variant == 'right_aligned_full':
            # Everything right-aligned
            text_lines.append("SELLER DETAILS".rjust(width))
            text_lines.append(supplier_name.rjust(width))
            if address:
                text_lines.append(address.rjust(width))
            if phone:
                text_lines.append(f"Tel: {phone}".rjust(width))
            if website and random.random() < 0.5:
                text_lines.append(website.rjust(width))
                
        elif variant == 'bordered_card':
            # Supplier info in a card with border
            card_width = min(width - 4, 50)
            text_lines.append('┌' + '─' * card_width + '┐')
            text_lines.append(f"│ {'SELLER':^{card_width-2}} │")
            text_lines.append('├' + '─' * card_width + '┤')
            text_lines.append(f"│ {supplier_name:<{card_width-2}} │")
            if address:
                # Split long addresses
                if len(address) > card_width - 4:
                    addr_parts = address.split(', ')
                    for part in addr_parts[:2]:  # Max 2 lines
                        text_lines.append(f"│ {part:<{card_width-2}} │")
                else:
                    text_lines.append(f"│ {address:<{card_width-2}} │")
            if phone:
                text_lines.append(f"│ {phone:<{card_width-2}} │")
            text_lines.append('└' + '─' * card_width + '┘')
            
        elif variant == 'boxed_formal':
            # Formal box with headers
            text_lines.append('═' * width)
            text_lines.append(f"║ MERCHANT: {supplier_name}")
            if address:
                text_lines.append(f"║ ADDRESS:  {address}")
            if phone:
                text_lines.append(f"║ PHONE:    {phone}")
            if email and random.random() < 0.5:
                text_lines.append(f"║ EMAIL:    {email}")
            text_lines.append('═' * width)
            
        elif variant == 'vertical_layout':
            # Stacked vertical with labels
            text_lines.append("┌─ SELLER ─┐")
            text_lines.append(f"  {supplier_name}")
            text_lines.append("  " + "─" * min(width-2, 30))
            if address:
                text_lines.append(f"  Location: {address}")
            if phone:
                text_lines.append(f"  Contact:  {phone}")
            if website and random.random() < 0.4:
                text_lines.append(f"  Web:      {website}")
            text_lines.append("└" + "─" * min(width-1, 15) + "┘")
            
        elif variant == 'horizontal_compact':
            # Single or two-line compact format
            if phone:
                text_lines.append(f"{supplier_name} | {phone}")
            else:
                text_lines.append(supplier_name)
            if address:
                text_lines.append(address)
                
        elif variant == 'tabular_format':
            # Table-like format with aligned columns
            text_lines.append("SELLER INFORMATION")
            text_lines.append("─" * width)
            text_lines.append(f"{'Company:':<12} {supplier_name}")
            if address:
                text_lines.append(f"{'Address:':<12} {address}")
            if phone:
                text_lines.append(f"{'Phone:':<12} {phone}")
            if email and random.random() < 0.6:
                text_lines.append(f"{'Email:':<12} {email}")
                
        elif variant == 'labeled_fields':
            # Each field clearly labeled
            text_lines.append("[Merchant Details]")
            text_lines.append(f"Name: {supplier_name}")
            if address:
                text_lines.append(f"Addr: {address}")
            if phone:
                text_lines.append(f"Tel:  {phone}")
            if email and random.random() < 0.5:
                text_lines.append(f"Mail: {email}")
                
        elif variant == 'two_column_split':
            # Split into two columns if data available
            col_width = width // 2 - 2
            text_lines.append(f"{'SELLER INFO':^{width}}")
            text_lines.append("─" * width)
            if phone:
                text_lines.append(f"{supplier_name:<{col_width}} | {phone:>{col_width}}")
            else:
                text_lines.append(supplier_name)
            if address:
                text_lines.append(address)
                
        elif variant == 'minimal_with_divider':
            # Minimal with visual separator
            text_lines.append(supplier_name)
            text_lines.append("~" * len(supplier_name))
            if address:
                text_lines.append(address)
            if phone:
                text_lines.append(phone)
                
        return text_lines

    def _generate_buyer_section(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate buyer/customer section with 10 variants
        
        Returns:
            List of text lines for buyer section
        """
        import random
        text_lines = []
        
        # Extract buyer data safely
        buyer_name = str(receipt_data.get('buyer_name', ''))[:width-4]
        buyer_address = str(receipt_data.get('buyer_address', ''))[:width-4]
        buyer_phone = str(receipt_data.get('buyer_phone', ''))[:width-4]
        buyer_email = str(receipt_data.get('buyer_email', ''))[:width-4]
        customer_id = str(receipt_data.get('customer_id', ''))[:width-4]
        account_number = str(receipt_data.get('account_number', ''))[:width-4]
        
        # Only generate buyer section if we have buyer data
        if not buyer_name and not customer_id and not account_number:
            return text_lines
        
        # 10 buyer section variants
        buyer_variants = [
            'minimal_name_only',
            'customer_id_focus',
            'full_customer_block',
            'loyalty_member_card',
            'ship_to_format',
            'bill_to_format',
            'ecommerce_order_style',
            'bordered_customer_info',
            'inline_customer_details',
            'vip_member_highlight'
        ]
        
        variant = random.choice(buyer_variants)
        
        # === IMPLEMENTATION OF EACH VARIANT ===
        
        if variant == 'minimal_name_only':
            # Just customer name
            if buyer_name:
                text_lines.append(f"Customer: {buyer_name}")
                
        elif variant == 'customer_id_focus':
            # Emphasize customer ID/account
            if customer_id or account_number:
                text_lines.append("CUSTOMER ACCOUNT")
                text_lines.append("-" * 20)
                if buyer_name:
                    text_lines.append(f"Name: {buyer_name}")
                if customer_id:
                    text_lines.append(f"ID: {customer_id}")
                elif account_number:
                    text_lines.append(f"Account: {account_number}")
                    
        elif variant == 'full_customer_block':
            # Complete customer info block
            text_lines.append("CUSTOMER INFORMATION")
            text_lines.append("─" * min(width, 25))
            if buyer_name:
                text_lines.append(f"Name:    {buyer_name}")
            if customer_id:
                text_lines.append(f"Cust ID: {customer_id}")
            if buyer_email and random.random() < 0.6:
                text_lines.append(f"Email:   {buyer_email}")
            if buyer_phone and random.random() < 0.5:
                text_lines.append(f"Phone:   {buyer_phone}")
                
        elif variant == 'loyalty_member_card':
            # Loyalty/rewards member format
            if account_number or customer_id:
                text_lines.append("╔═══ REWARDS MEMBER ═══╗")
                if buyer_name:
                    text_lines.append(f"║ {buyer_name:^21} ║")
                member_num = account_number or customer_id
                text_lines.append(f"║ Member: {member_num:<12} ║")
                text_lines.append("╚═══════════════════════╝")
            elif buyer_name:
                text_lines.append(f"** CUSTOMER: {buyer_name} **")
                
        elif variant == 'ship_to_format':
            # E-commerce ship-to style
            if buyer_name or buyer_address:
                text_lines.append("┌─ SHIP TO ─┐")
                if buyer_name:
                    text_lines.append(f"  {buyer_name}")
                if buyer_address:
                    text_lines.append(f"  {buyer_address}")
                if buyer_phone:
                    text_lines.append(f"  {buyer_phone}")
                text_lines.append("└──────────┘")
                
        elif variant == 'bill_to_format':
            # Bill-to address format
            if buyer_name:
                text_lines.append("BILL TO:")
                text_lines.append(buyer_name)
                if buyer_address:
                    text_lines.append(buyer_address)
                if customer_id:
                    text_lines.append(f"Account #: {customer_id}")
                    
        elif variant == 'ecommerce_order_style':
            # E-commerce order format
            order_num = receipt_data.get('invoice_number', 'N/A')
            text_lines.append(f"Order: {order_num}")
            if buyer_name:
                text_lines.append(f"Customer: {buyer_name}")
            if buyer_email and random.random() < 0.7:
                text_lines.append(f"Email: {buyer_email}")
            if customer_id:
                text_lines.append(f"Customer ID: {customer_id}")
                
        elif variant == 'bordered_customer_info':
            # Customer info in bordered box
            if buyer_name:
                box_width = min(width - 4, 40)
                text_lines.append("┌" + "─" * box_width + "┐")
                text_lines.append(f"│ {'CUSTOMER':^{box_width-2}} │")
                text_lines.append("├" + "─" * box_width + "┤")
                text_lines.append(f"│ {buyer_name:<{box_width-2}} │")
                if customer_id or account_number:
                    member = customer_id or account_number
                    text_lines.append(f"│ ID: {member:<{box_width-6}} │")
                text_lines.append("└" + "─" * box_width + "┘")
                
        elif variant == 'inline_customer_details':
            # Single-line or compact inline format
            if buyer_name and (customer_id or account_number):
                member = customer_id or account_number
                text_lines.append(f"Customer: {buyer_name} (ID: {member})")
            elif buyer_name:
                text_lines.append(f"Customer: {buyer_name}")
                
        elif variant == 'vip_member_highlight':
            # VIP/premium member emphasis
            if account_number or customer_id:
                text_lines.append("*** VIP MEMBER ***")
                if buyer_name:
                    text_lines.append(f"{buyer_name:^{width}}")
                member = account_number or customer_id
                text_lines.append(f"Member #{member:^{width}}")
                text_lines.append("*" * min(width, 20))
            elif buyer_name:
                text_lines.append(f"Customer: {buyer_name}")
                
        return text_lines

    def _generate_order_metadata_section(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate order/transaction metadata section with 20 variants
        
        Returns:
            List of text lines for order metadata section
        """
        import random
        text_lines = []
        
        # Get locale configuration
        locale_code = receipt_data.get('locale', None)
        locale_config = self._get_locale_config(locale_code)
        
        # Extract metadata safely
        invoice_number = str(receipt_data.get('invoice_number', ''))[:width-4]
        invoice_date_raw = str(receipt_data.get('invoice_date', ''))[:width-4]
        # Format date according to locale
        invoice_date = self._format_date(invoice_date_raw, locale_config) if invoice_date_raw else ''
        transaction_number = str(receipt_data.get('transaction_number', ''))[:width-4]
        transaction_time = str(receipt_data.get('transaction_time', ''))[:width-4]
        register_number = str(receipt_data.get('register_number', ''))[:width-4]
        cashier_id = str(receipt_data.get('cashier_id', ''))[:width-4]
        payment_method = str(receipt_data.get('payment_method', ''))[:width-4]
        
        # 20 order metadata variants
        metadata_variants = [
            'minimal_date_invoice',
            'full_transaction_block',
            'tabular_metadata',
            'boxed_order_info',
            'inline_compact',
            'vertical_labeled',
            'two_column_metadata',
            'receipt_header_style',
            'invoice_professional',
            'po_number_emphasis',
            'vendor_reference_style',
            'sales_rep_included',
            'payment_terms_focus',
            'shipping_method_block',
            'delivery_slot_style',
            'account_number_card',
            'transaction_id_banner',
            'timestamp_detailed',
            'minimal_modern',
            'comprehensive_all_fields'
        ]
        
        variant = random.choice(metadata_variants)
        
        # === IMPLEMENTATION OF EACH VARIANT ===
        
        if variant == 'minimal_date_invoice':
            # Just date and receipt number
            if invoice_number:
                text_lines.append(f"Receipt #: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
                
        elif variant == 'full_transaction_block':
            # Complete transaction details in a block
            text_lines.append("TRANSACTION DETAILS")
            text_lines.append("-" * min(width, 30))
            if transaction_number:
                text_lines.append(f"Transaction: {transaction_number}")
            if invoice_number:
                text_lines.append(f"Receipt #:   {invoice_number}")
            if invoice_date and transaction_time:
                text_lines.append(f"Date/Time:   {invoice_date} {transaction_time}")
            elif invoice_date:
                text_lines.append(f"Date:        {invoice_date}")
            if register_number:
                text_lines.append(f"Register:    {register_number}")
            if cashier_id:
                text_lines.append(f"Cashier:     {cashier_id}")
                
        elif variant == 'tabular_metadata':
            # Table format with aligned columns
            text_lines.append("ORDER INFORMATION")
            text_lines.append("─" * min(width, 35))
            if invoice_number:
                text_lines.append(f"{'Order #:':<15} {invoice_number}")
            if invoice_date:
                text_lines.append(f"{'Date:':<15} {invoice_date}")
            if transaction_number:
                text_lines.append(f"{'Transaction:':<15} {transaction_number}")
            # Generate PO number (30% chance)
            if random.random() < 0.3:
                po_number = f"PO-{random.randint(100000, 999999)}"
                text_lines.append(f"{'PO Number:':<15} {po_number}")
                
        elif variant == 'boxed_order_info':
            # Order info in a bordered box
            box_width = min(width - 4, 45)
            text_lines.append("┌" + "─" * box_width + "┐")
            text_lines.append(f"│ {'ORDER DETAILS':^{box_width-2}} │")
            text_lines.append("├" + "─" * box_width + "┤")
            if invoice_number:
                text_lines.append(f"│ Order: {invoice_number:<{box_width-10}} │")
            if invoice_date:
                text_lines.append(f"│ Date:  {invoice_date:<{box_width-10}} │")
            if transaction_time:
                text_lines.append(f"│ Time:  {transaction_time:<{box_width-10}} │")
            text_lines.append("└" + "─" * box_width + "┘")
            
        elif variant == 'inline_compact':
            # Single or two-line compact format
            if invoice_number and invoice_date:
                text_lines.append(f"Order {invoice_number} | {invoice_date}")
            elif invoice_number:
                text_lines.append(f"Order: {invoice_number}")
            if transaction_number:
                text_lines.append(f"Txn: {transaction_number}")
                
        elif variant == 'vertical_labeled':
            # Vertical stack with clear labels
            text_lines.append("┌─ ORDER INFO ─┐")
            if invoice_number:
                text_lines.append(f"  Receipt #: {invoice_number}")
            if invoice_date:
                text_lines.append(f"  Date:      {invoice_date}")
            if transaction_time:
                text_lines.append(f"  Time:      {transaction_time}")
            if register_number:
                text_lines.append(f"  Register:  {register_number}")
            text_lines.append("└" + "─" * min(width-1, 20) + "┘")
            
        elif variant == 'two_column_metadata':
            # Split metadata across two columns
            col_width = width // 2 - 2
            if invoice_number and invoice_date:
                text_lines.append(f"{'Order: ' + invoice_number:<{col_width}} | {'Date: ' + invoice_date:>{col_width}}")
            if transaction_number and register_number:
                text_lines.append(f"{'Txn: ' + transaction_number:<{col_width}} | {'Reg: ' + register_number:>{col_width}}")
            elif transaction_number:
                text_lines.append(f"Transaction: {transaction_number}")
                
        elif variant == 'receipt_header_style':
            # Receipt-style header
            text_lines.append(f"{'RECEIPT':^{width}}")
            text_lines.append("=" * width)
            if invoice_number:
                text_lines.append(f"No: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
            if transaction_time:
                text_lines.append(f"Time: {transaction_time}")
                
        elif variant == 'invoice_professional':
            # Professional invoice style
            text_lines.append(f"{'INVOICE':>{width}}")
            if invoice_number:
                text_lines.append(f"{'Invoice No: ' + invoice_number:>{width}}")
            if invoice_date:
                text_lines.append(f"{'Date: ' + invoice_date:>{width}}")
            # Add vendor ID (20% chance)
            if random.random() < 0.2:
                vendor_id = f"VND-{random.randint(1000, 9999)}"
                text_lines.append(f"{'Vendor ID: ' + vendor_id:>{width}}")
                
        elif variant == 'po_number_emphasis':
            # Emphasize PO number
            po_number = f"PO-{random.randint(100000, 999999)}"
            text_lines.append("╔" + "═" * min(width-2, 40) + "╗")
            text_lines.append(f"║ PURCHASE ORDER: {po_number:<{min(width-20, 22)}} ║")
            text_lines.append("╚" + "═" * min(width-2, 40) + "╝")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
            if invoice_number:
                text_lines.append(f"Invoice: {invoice_number}")
                
        elif variant == 'vendor_reference_style':
            # Vendor reference format
            text_lines.append("VENDOR REFERENCE")
            text_lines.append("─" * min(width, 25))
            if invoice_number:
                text_lines.append(f"Ref #: {invoice_number}")
            # Add account number (40% chance)
            if random.random() < 0.4:
                account_num = f"ACCT-{random.randint(10000, 99999)}"
                text_lines.append(f"Account: {account_num}")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
                
        elif variant == 'sales_rep_included':
            # Include sales rep information
            text_lines.append("ORDER SUMMARY")
            text_lines.append("~" * min(width, 25))
            if invoice_number:
                text_lines.append(f"Order #: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
            # Add sales rep (25% chance)
            if random.random() < 0.25:
                sales_reps = ["John Smith", "Sarah Johnson", "Mike Davis", "Emily Chen"]
                text_lines.append(f"Sales Rep: {random.choice(sales_reps)}")
            if cashier_id:
                text_lines.append(f"Cashier: {cashier_id}")
                
        elif variant == 'payment_terms_focus':
            # Emphasize payment terms
            text_lines.append("PAYMENT INFORMATION")
            text_lines.append("─" * min(width, 30))
            if invoice_number:
                text_lines.append(f"Invoice: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
            # Add payment terms (50% chance)
            if random.random() < 0.5:
                terms = random.choice(["Net 30", "Net 15", "Due on Receipt", "COD", "Net 60"])
                text_lines.append(f"Terms: {terms}")
            if payment_method:
                text_lines.append(f"Method: {payment_method}")
                
        elif variant == 'shipping_method_block':
            # Include shipping details
            text_lines.append("SHIPPING & ORDER INFO")
            text_lines.append("─" * min(width, 30))
            if invoice_number:
                text_lines.append(f"Order #: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Order Date: {invoice_date}")
            # Add shipping method (60% chance)
            if random.random() < 0.6:
                methods = ["Standard", "Express", "Next Day", "Ground", "2-Day Air", "In-Store Pickup"]
                text_lines.append(f"Shipping: {random.choice(methods)}")
            # Add tracking (30% chance)
            if random.random() < 0.3:
                tracking = f"TRK-{random.randint(1000000000, 9999999999)}"
                text_lines.append(f"Tracking: {tracking}")
                
        elif variant == 'delivery_slot_style':
            # E-commerce delivery slot format
            text_lines.append("ORDER & DELIVERY")
            text_lines.append("─" * min(width, 25))
            if invoice_number:
                text_lines.append(f"Order: {invoice_number}")
            if invoice_date:
                text_lines.append(f"Placed: {invoice_date}")
            # Add delivery slot (40% chance)
            if random.random() < 0.4:
                slots = ["9AM-12PM", "12PM-3PM", "3PM-6PM", "6PM-9PM", "Next Day"]
                delivery_date = invoice_date  # Simplified
                text_lines.append(f"Delivery: {delivery_date} {random.choice(slots)}")
                
        elif variant == 'account_number_card':
            # Account number emphasis
            text_lines.append("╔" + "═" * min(width-2, 38) + "╗")
            text_lines.append(f"║ {'ACCOUNT TRANSACTION':^{min(width-4, 36)}} ║")
            text_lines.append("╠" + "═" * min(width-2, 38) + "╣")
            # Generate account number
            account = f"ACCT-{random.randint(100000, 999999)}"
            text_lines.append(f"║ Account: {account:<{min(width-14, 26)}} ║")
            if invoice_number:
                text_lines.append(f"║ Invoice: {invoice_number:<{min(width-14, 26)}} ║")
            if invoice_date:
                text_lines.append(f"║ Date: {invoice_date:<{min(width-11, 29)}} ║")
            text_lines.append("╚" + "═" * min(width-2, 38) + "╝")
            
        elif variant == 'transaction_id_banner':
            # Transaction ID banner
            if transaction_number:
                text_lines.append("*" * width)
                text_lines.append(f"{'TRANSACTION ID':^{width}}")
                text_lines.append(f"{transaction_number:^{width}}")
                text_lines.append("*" * width)
            if invoice_date and transaction_time:
                text_lines.append(f"{invoice_date} @ {transaction_time}")
            elif invoice_date:
                text_lines.append(f"Date: {invoice_date}")
                
        elif variant == 'timestamp_detailed':
            # Detailed timestamp format
            text_lines.append("TRANSACTION TIMESTAMP")
            text_lines.append("═" * min(width, 30))
            if invoice_date:
                text_lines.append(f"Date: {invoice_date}")
            if transaction_time:
                text_lines.append(f"Time: {transaction_time}")
            if transaction_number:
                text_lines.append(f"ID: {transaction_number}")
            # Add timezone (10% chance)
            if random.random() < 0.1:
                timezone = random.choice(["EST", "PST", "CST", "MST"])
                text_lines.append(f"Timezone: {timezone}")
            if register_number:
                text_lines.append(f"Terminal: {register_number}")
                
        elif variant == 'minimal_modern':
            # Minimal modern style
            if invoice_number:
                text_lines.append(f"#{invoice_number}")
            if invoice_date:
                text_lines.append(invoice_date)
            if transaction_time:
                text_lines.append(transaction_time)
                
        elif variant == 'comprehensive_all_fields':
            # Include all available metadata
            text_lines.append("═" * width)
            text_lines.append(f"{'TRANSACTION RECORD':^{width}}")
            text_lines.append("═" * width)
            
            if invoice_number:
                text_lines.append(f"Receipt #:     {invoice_number}")
            if transaction_number:
                text_lines.append(f"Transaction:   {transaction_number}")
            
            # Add PO number
            po_number = f"PO-{random.randint(100000, 999999)}"
            text_lines.append(f"PO Number:     {po_number}")
            
            if invoice_date:
                text_lines.append(f"Date:          {invoice_date}")
            if transaction_time:
                text_lines.append(f"Time:          {transaction_time}")
            
            if register_number:
                text_lines.append(f"Register:      {register_number}")
            if cashier_id:
                text_lines.append(f"Operator:      {cashier_id}")
            
            # Add vendor ID
            vendor_id = f"VND-{random.randint(1000, 9999)}"
            text_lines.append(f"Vendor ID:     {vendor_id}")
            
            # Add payment terms
            terms = random.choice(["Net 30", "Due on Receipt", "COD"])
            text_lines.append(f"Terms:         {terms}")
            
            text_lines.append("─" * width)
            
        return text_lines

    def _generate_line_items_table(self, receipt_data: dict, width: int, style: str, divider_light: str, layout: str = '') -> list:
        """
        Generate line item table with 25 layout variants
        
        Args:
            receipt_data: Receipt data dictionary
            width: Line width
            style: Receipt style
            divider_light: Light divider character pattern
            layout: Optional specific layout to use (for consistency across pages)
        
        Returns:
            List of text lines for line items table
        """
        import random
        text_lines = []
        
        line_items = receipt_data.get('line_items', [])
        if not line_items:
            return text_lines
        
        # 25 line item table layout variants
        layout_variants = [
            'classic_table',
            'complex_table',
            'multi_line_description',
            'attribute_columns',
            'nested_description',
            'alternating_shaded',
            'borderless_minimal',
            'condensed_rows',
            'wide_table',
            'narrow_table',
            'sku_emphasis',
            'price_breakdown',
            'quantity_focus',
            'discount_inline',
            'discount_separate',
            'tabular_boxed',
            'list_style',
            'receipt_tape_style',
            'invoice_professional',
            'retail_pos_style',
            'ecommerce_order',
            'grouped_categories',
            'two_column_items',
            'description_first',
            'prices_right_aligned'
        ]
        
        # Use provided layout or choose randomly
        if not layout:
            layout = random.choice(layout_variants)
        
        # === IMPLEMENTATION OF EACH LAYOUT ===
        
        if layout == 'classic_table':
            # Classic table: SKU | QTY | PRICE | TOTAL
            text_lines.append("ITEMS PURCHASED")
            text_lines.append(divider_light)
            header_width = min(width, 60)
            text_lines.append(f"{'SKU':<12} {'QTY':>4} {'PRICE':>10} {'TOTAL':>10}")
            text_lines.append("-" * header_width)
            
            for item in line_items:
                sku = item.get('sku', 'N/A')[:10]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{sku:<12} {qty:>4} {unit_price:>10} {total:>10}")
                # Add description on second line
                desc = item.get('description', 'Item')[:45]
                text_lines.append(f"  {desc}")
                
        elif layout == 'complex_table':
            # Complex table: SKU | Name | Color | Size | Qty | Unit | Rate | Discount | Total
            text_lines.append("ORDER DETAILS")
            text_lines.append("=" * min(width, 70))
            text_lines.append(f"{'SKU':<8} {'ITEM':<15} {'QTY':>3} {'RATE':>8} {'DISC':>6} {'TOTAL':>9}")
            text_lines.append("-" * min(width, 70))
            
            for item in line_items:
                sku = item.get('sku', 'N/A')[:7]
                desc = item.get('description', 'Item')[:14]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                discount = item.get('discount', 0)
                total = str(item.get('total', '$0.00'))
                
                # Format discount
                if discount and discount not in [0, 0.0]:
                    disc_str = f"-{discount}" if isinstance(discount, str) else f"-${discount:.2f}"
                else:
                    disc_str = "--"
                
                text_lines.append(f"{sku:<8} {desc:<15} {qty:>3} {unit_price:>8} {disc_str:>6} {total:>9}")
                
                # Add attributes if available
                if random.random() < 0.3:
                    attrs = random.choice([
                        "Color: Blue, Size: M",
                        "Material: Cotton",
                        "Size: Large",
                        "Variant: Standard"
                    ])
                    text_lines.append(f"         └─ {attrs}")
                    
        elif layout == 'multi_line_description':
            # Multi-line item descriptions with details
            text_lines.append("ITEMS:")
            text_lines.append(divider_light)
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                # Item number and description
                text_lines.append(f"{idx}. {desc}")
                
                # Second line with details
                text_lines.append(f"   Quantity: {qty} × {unit_price} = {total}")
                
                # Third line with SKU if available
                if 'sku' in item:
                    text_lines.append(f"   SKU: {item['sku']}")
                
                # Optional attributes
                if random.random() < 0.25:
                    attrs = random.choice([
                        "Size: Medium, Color: Navy Blue",
                        "Material: 100% Cotton",
                        "Model: 2024 Edition",
                        "Condition: New, Warranty: 1 Year"
                    ])
                    text_lines.append(f"   ({attrs})")
                
                text_lines.append("")  # Blank line between items
                
        elif layout == 'attribute_columns':
            # Attribute columns: Material, Color, Size, etc.
            text_lines.append("PRODUCT DETAILS")
            text_lines.append("=" * min(width, 65))
            text_lines.append(f"{'ITEM':<20} {'ATTR':<15} {'QTY':>3} {'PRICE':>8} {'TOTAL':>9}")
            text_lines.append("-" * min(width, 65))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:19]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                # Generate attribute
                attr = random.choice([
                    "Material:Cotton",
                    "Size:Medium",
                    "Color:Blue",
                    "Type:Standard",
                    "Variant:Basic",
                    "--"
                ])[:14]
                
                text_lines.append(f"{desc:<20} {attr:<15} {qty:>3} {unit_price:>8} {total:>9}")
                
        elif layout == 'nested_description':
            # Nested description blocks with indentation
            text_lines.append("╔═══ ITEMS ORDERED ═══╗")
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"║")
                text_lines.append(f"║ [{idx}] {desc}")
                text_lines.append(f"║     ├─ Quantity: {qty}")
                text_lines.append(f"║     ├─ Unit Price: {unit_price}")
                text_lines.append(f"║     └─ Subtotal: {total}")
                
                if 'sku' in item and random.random() < 0.5:
                    text_lines.append(f"║        SKU: {item['sku']}")
                    
            text_lines.append("╚" + "═" * min(width-1, 30) + "╝")
            
        elif layout == 'alternating_shaded':
            # Alternating shaded rows (simulated with characters)
            text_lines.append("ORDER SUMMARY")
            text_lines.append(divider_light)
            
            for idx, item in enumerate(line_items):
                desc = item.get('description', 'Item')[:35]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                # Alternate shading
                if idx % 2 == 0:
                    # Light row
                    text_lines.append(f"{desc:<35} {qty:>3}x {unit_price:>8} = {total:>9}")
                else:
                    # "Shaded" row (with background indicator)
                    text_lines.append(f"░ {desc:<33} {qty:>3}x {unit_price:>8} = {total:>9}")
                    
        elif layout == 'borderless_minimal':
            # Borderless minimalist table
            text_lines.append("")
            text_lines.append(f"{'ITEM':<30} {'QTY':>4}  {'AMOUNT':>10}")
            text_lines.append("")
            
            for item in line_items:
                desc = item.get('description', 'Item')[:29]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{desc:<30} {qty:>4}  {total:>10}")
            
            text_lines.append("")
            
        elif layout == 'condensed_rows':
            # Condensed item rows - no spacing
            text_lines.append("ITEMS")
            text_lines.append("-" * min(width, 40))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:28]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{desc:<28} {qty}x {total:>8}")
                
        elif layout == 'wide_table':
            # Wide table format (uses full width)
            text_lines.append("PURCHASE DETAILS")
            text_lines.append("=" * min(width, 80))
            text_lines.append(f"{'DESCRIPTION':<30} {'SKU':<12} {'QTY':>4} {'UNIT':>9} {'TOTAL':>10}")
            text_lines.append("-" * min(width, 80))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:29]
                sku = item.get('sku', 'N/A')[:11]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{desc:<30} {sku:<12} {qty:>4} {unit_price:>9} {total:>10}")
                
        elif layout == 'narrow_table':
            # Narrow table format (compact width)
            text_lines.append("ITEMS:")
            text_lines.append("-" * min(width, 35))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:20]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{desc:<20} {qty}x")
                text_lines.append(f"{' '*20} {total:>10}")
                
        elif layout == 'sku_emphasis':
            # SKU-emphasized format
            text_lines.append("PRODUCT LIST")
            text_lines.append(divider_light)
            
            for item in line_items:
                sku = item.get('sku', 'N/A')
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"SKU: {sku}")
                text_lines.append(f"  {desc}")
                text_lines.append(f"  Qty: {qty} @ {unit_price} = {total}")
                text_lines.append("")
                
        elif layout == 'price_breakdown':
            # Detailed price breakdown per item
            text_lines.append("PRICE BREAKDOWN")
            text_lines.append("=" * min(width, 50))
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                discount = item.get('discount', 0)
                
                text_lines.append(f"Item #{idx}: {desc}")
                text_lines.append(f"  Base Price:     {unit_price} × {qty}")
                
                if discount and discount not in [0, 0.0]:
                    disc_str = f"-{discount}" if isinstance(discount, str) else f"-${discount:.2f}"
                    text_lines.append(f"  Discount:       {disc_str}")
                
                text_lines.append(f"  Line Total:     {total}")
                text_lines.append("")
                
        elif layout == 'quantity_focus':
            # Quantity-focused format
            text_lines.append("QUANTITY | ITEM | PRICE")
            text_lines.append("-" * min(width, 45))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:30]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"   {qty:>2}    | {desc:<30} | {total:>9}")
                
        elif layout == 'discount_inline':
            # Discount shown inline with item
            text_lines.append("ITEMS (with discounts)")
            text_lines.append(divider_light)
            
            for item in line_items:
                desc = item.get('description', 'Item')[:25]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                discount = item.get('discount', 0)
                
                if discount and discount not in [0, 0.0]:
                    disc_str = f"(-{discount})" if isinstance(discount, str) else f"(-${discount:.2f})"
                    text_lines.append(f"{desc:<25} {qty}x {unit_price} {disc_str} = {total}")
                else:
                    text_lines.append(f"{desc:<25} {qty}x {unit_price} = {total}")
                    
        elif layout == 'discount_separate':
            # Discount shown on separate line
            text_lines.append("ITEMS ORDERED")
            text_lines.append(divider_light)
            
            for item in line_items:
                desc = item.get('description', 'Item')[:35]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                discount = item.get('discount', 0)
                
                text_lines.append(f"{desc:<35} {qty}x {unit_price}")
                
                if discount and discount not in [0, 0.0]:
                    disc_str = f"-{discount}" if isinstance(discount, str) else f"-${discount:.2f}"
                    text_lines.append(f"  {'Item Discount:':<33} {disc_str:>9}")
                
                text_lines.append(f"  {'Line Total:':<33} {total:>9}")
                text_lines.append("")
                
        elif layout == 'tabular_boxed':
            # Tabular format with box borders
            text_lines.append("┌" + "─" * min(width-2, 60) + "┐")
            text_lines.append(f"│ {'ITEM':<25} {'QTY':>4} {'PRICE':>9} {'TOTAL':>10} │")
            text_lines.append("├" + "─" * min(width-2, 60) + "┤")
            
            for item in line_items:
                desc = item.get('description', 'Item')[:24]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"│ {desc:<25} {qty:>4} {unit_price:>9} {total:>10} │")
            
            text_lines.append("└" + "─" * min(width-2, 60) + "┘")
            
        elif layout == 'list_style':
            # Simple list style
            text_lines.append("Your Items:")
            text_lines.append("")
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"{idx}. {desc}")
                text_lines.append(f"   {qty} × {unit_price} = {total}")
                
        elif layout == 'receipt_tape_style':
            # Classic receipt tape style
            text_lines.append("ITEMS PURCHASED")
            text_lines.append("*" * min(width, 40))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:30]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(desc)
                text_lines.append(f"  {qty} @ {unit_price}")
                text_lines.append(f"  {' '*20} {total:>10}")
                
        elif layout == 'invoice_professional':
            # Professional invoice format
            text_lines.append("LINE ITEMS")
            text_lines.append("=" * min(width, 70))
            text_lines.append(f"{'#':<3} {'DESCRIPTION':<30} {'QTY':>5} {'RATE':>10} {'AMOUNT':>11}")
            text_lines.append("-" * min(width, 70))
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')[:29]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"{idx:<3} {desc:<30} {qty:>5} {unit_price:>10} {total:>11}")
                
        elif layout == 'retail_pos_style':
            # Retail POS receipt style
            text_lines.append("*** ITEMS ***")
            text_lines.append("")
            
            for item in line_items:
                desc = item.get('description', 'Item')[:35]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"{desc:<35}")
                text_lines.append(f"  {qty} @ {item.get('unit_price', '$0.00'):>8} {total:>10}")
                
        elif layout == 'ecommerce_order':
            # E-commerce order style
            text_lines.append("YOUR ORDER")
            text_lines.append(divider_light)
            
            for idx, item in enumerate(line_items, 1):
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"Item {idx}")
                text_lines.append(f"  {desc}")
                text_lines.append(f"  Qty: {qty} | Price: {unit_price} | Total: {total}")
                
                # Add SKU for e-commerce
                if 'sku' in item:
                    text_lines.append(f"  SKU: {item['sku']}")
                
                text_lines.append("")
                
        elif layout == 'grouped_categories':
            # Group items by implicit category (every 2-3 items)
            text_lines.append("ORDER BREAKDOWN")
            text_lines.append(divider_light)
            
            group_size = random.randint(2, 3)
            for idx, item in enumerate(line_items):
                # Add category header every N items
                if idx % group_size == 0:
                    categories = ["Electronics", "Apparel", "Home Goods", "Accessories", "Food Items"]
                    text_lines.append(f"--- {random.choice(categories)} ---")
                
                desc = item.get('description', 'Item')[:30]
                qty = item.get('quantity', 1)
                total = str(item.get('total', '$0.00'))
                text_lines.append(f"  {desc:<30} {qty}x  {total:>9}")
                
        elif layout == 'two_column_items':
            # Two-column item layout (if width allows)
            if width >= 60:
                text_lines.append("ITEMS PURCHASED")
                text_lines.append(divider_light)
                
                for i in range(0, len(line_items), 2):
                    item1 = line_items[i]
                    desc1 = item1.get('description', 'Item')[:20]
                    total1 = str(item1.get('total', '$0.00'))
                    
                    if i + 1 < len(line_items):
                        item2 = line_items[i + 1]
                        desc2 = item2.get('description', 'Item')[:20]
                        total2 = str(item2.get('total', '$0.00'))
                        text_lines.append(f"{desc1:<20} {total1:>8}  |  {desc2:<20} {total2:>8}")
                    else:
                        text_lines.append(f"{desc1:<20} {total1:>8}")
            else:
                # Fallback to single column
                text_lines.append("ITEMS:")
                for item in line_items:
                    desc = item.get('description', 'Item')[:25]
                    total = str(item.get('total', '$0.00'))
                    text_lines.append(f"{desc:<25} {total:>8}")
                    
        elif layout == 'description_first':
            # Description-first layout
            text_lines.append("PURCHASED ITEMS")
            text_lines.append("=" * min(width, 50))
            
            for item in line_items:
                desc = item.get('description', 'Item')
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(desc)
                text_lines.append(f"  Quantity: {qty} | Unit Price: {unit_price} | Total: {total}")
                text_lines.append("")
                
        elif layout == 'prices_right_aligned':
            # All prices right-aligned
            text_lines.append(f"{'ITEM':<40} {'AMOUNT':>10}")
            text_lines.append("-" * min(width, 52))
            
            for item in line_items:
                desc = item.get('description', 'Item')[:39]
                qty = item.get('quantity', 1)
                unit_price = str(item.get('unit_price', '$0.00'))
                total = str(item.get('total', '$0.00'))
                
                text_lines.append(f"{desc:<40} {total:>10}")
                text_lines.append(f"  {'(Qty: ' + str(qty) + ' @ ' + unit_price + ')':<40}")
        
        return text_lines

    def _generate_totals_section(self, receipt_data: dict, width: int, style: str, divider_full: str, divider_light: str) -> list:
        """
        Generate totals section with 20 layout variations
        
        Variations include:
        - Currency placement (left/right, symbol/code)
        - Tax shown/hidden/detailed
        - Discount above/below subtotal
        - Rounding adjustments
        - Shipping split by package
        - Multiple calculation formats
        
        Args:
            receipt_data: Receipt data dictionary
            width: Line width
            style: Receipt style
            divider_full: Full divider character pattern
            divider_light: Light divider character pattern
            
        Returns:
            List of text lines for totals section
        """
        import random
        text_lines = []
        
        # Extract values
        subtotal = receipt_data.get('subtotal', '0')
        discount = receipt_data.get('discount', 0)
        tax_amount = receipt_data.get('tax_amount', '0')
        tax_rate = receipt_data.get('tax_rate', 0)
        tip_amount = receipt_data.get('tip_amount', 0)
        total_amount = receipt_data.get('total_amount', '0')
        shipping_cost = receipt_data.get('shipping_cost', 0)
        
        # Parse numeric values
        def parse_currency(val):
            if val in [None, 0, '0', '$0.00', '']:
                return 0.0
            try:
                return float(str(val).replace('$', '').replace(',', ''))
            except (ValueError, TypeError):
                return 0.0
        
        subtotal_num = parse_currency(subtotal)
        discount_num = parse_currency(discount)
        tax_num = parse_currency(tax_amount)
        tip_num = parse_currency(tip_amount)
        total_num = parse_currency(total_amount)
        shipping_num = parse_currency(shipping_cost)
        
        # 20 totals layout variants
        totals_variants = [
            'standard_right_aligned',
            'left_labels_right_amounts',
            'boxed_totals',
            'discount_first',
            'discount_after_subtotal',
            'tax_breakdown_detailed',
            'tax_hidden',
            'currency_code_suffix',
            'currency_symbol_left',
            'tabular_grid',
            'compact_inline',
            'verbose_labels',
            'minimal_two_line',
            'shipping_split_packages',
            'rounding_adjustment',
            'grand_total_emphasized',
            'invoice_professional',
            'receipt_casual',
            'ecommerce_detailed',
            'retail_pos_style'
        ]
        
        variant = random.choice(totals_variants)
        
        # === IMPLEMENTATION OF EACH VARIANT ===
        
        # Get locale configuration (pass through from receipt_data if available)
        locale_code = receipt_data.get('locale', None)
        locale_config = self._get_locale_config(locale_code)
        labels = locale_config['labels']
        
        # Select consistent currency style for this receipt (15 variants)
        # Store in receipt_data if not already set for consistency across sections
        if 'currency_style' not in receipt_data:
            import random
            currency_styles = [
                'symbol_before', 'symbol_after', 'symbol_space_before', 'symbol_space_after',
                'code_before', 'code_after', 'code_no_space_before', 'code_no_space_after',
                'symbol_parentheses', 'code_parentheses', 'tax_included_suffix', 
                'tax_included_code', 'with_currency_name', 'accounting_negative', 'code_hyphen'
            ]
            # Favor standard styles (first 4) 70% of the time
            if random.random() < 0.70:
                currency_style = random.choice(currency_styles[:4])
            else:
                currency_style = random.choice(currency_styles[4:])
            receipt_data['currency_style'] = currency_style
        else:
            currency_style = receipt_data['currency_style']
        
        if variant == 'standard_right_aligned':
            # Classic format: labels on left, amounts right-aligned
            text_lines.append(divider_full)
            if subtotal_num > 0:
                text_lines.append(f"{labels['subtotal'] + ':':25s} {self._format_currency(subtotal_num, locale_config, currency_style):>15s}")
            if discount_num > 0:
                text_lines.append(f"{labels['discount'] + ':':25s} -{self._format_currency(discount_num, locale_config, currency_style):>14s}")
            if tax_num > 0:
                if tax_rate:
                    tax_label = f"{labels['tax']} ({tax_rate}%)"
                    text_lines.append(f"{tax_label + ':':25s} {self._format_currency(tax_num, locale_config, currency_style):>15s}")
                else:
                    text_lines.append(f"{labels['tax'] + ':':25s} {self._format_currency(tax_num, locale_config, currency_style):>15s}")
            if tip_num > 0:
                text_lines.append(f"{'Tip:':25s} {self._format_currency(tip_num, locale_config, currency_style):>15s}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping:':25s} {self._format_currency(shipping_num, locale_config, currency_style):>15s}")
            text_lines.append(divider_light)
            text_lines.append(f"{labels['total'].upper() + ':':25s} {self._format_currency(total_num, locale_config, currency_style):>15s}")
            text_lines.append(divider_full)
            
        elif variant == 'left_labels_right_amounts':
            # Labels left-aligned, amounts far right
            text_lines.append(divider_full)
            col_width = max(width - 15, 40)
            text_lines.append(f"Subtotal{' ' * (col_width - 8 - 10)}${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"Discount{' ' * (col_width - 8 - 11)}-${discount_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"Tax{' ' * (col_width - 3 - 10)}${tax_num:.2f}")
            if shipping_num > 0:
                text_lines.append(f"Shipping{' ' * (col_width - 8 - 10)}${shipping_num:.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"TOTAL{' ' * (col_width - 5 - 10)}${total_num:.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'boxed_totals':
            # Totals in a box
            box_width = min(width, 50)
            text_lines.append('┌' + '─' * (box_width - 2) + '┐')
            text_lines.append(f"│ {'Subtotal:':20s} ${subtotal_num:>10.2f}  │")
            if discount_num > 0:
                text_lines.append(f"│ {'Discount:':20s} -${discount_num:>9.2f}  │")
            if tax_num > 0:
                text_lines.append(f"│ {'Tax:':20s} ${tax_num:>10.2f}  │")
            if shipping_num > 0:
                text_lines.append(f"│ {'Shipping:':20s} ${shipping_num:>10.2f}  │")
            text_lines.append('├' + '─' * (box_width - 2) + '┤')
            text_lines.append(f"│ {'TOTAL:':20s} ${total_num:>10.2f}  │")
            text_lines.append('└' + '─' * (box_width - 2) + '┘')
            
        elif variant == 'discount_first':
            # Show discount before subtotal
            text_lines.append(divider_full)
            if discount_num > 0:
                original = subtotal_num + discount_num
                text_lines.append(f"{'Original Price:':25s} ${original:>10.2f}")
                text_lines.append(f"{'Discount Applied:':25s} -${discount_num:>9.2f}")
                text_lines.append(divider_light)
            text_lines.append(f"{'Subtotal:':25s} ${subtotal_num:>10.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Tax:':25s} ${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping:':25s} ${shipping_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'TOTAL DUE:':25s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'discount_after_subtotal':
            # Standard order
            text_lines.append(divider_full)
            text_lines.append(f"{'Subtotal:':25s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Less Discount:':25s} -${discount_num:>9.2f}")
                after_discount = subtotal_num - discount_num
                text_lines.append(f"{'After Discount:':25s} ${after_discount:>10.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Plus Tax:':25s} ${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping & Handling:':25s} ${shipping_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'AMOUNT DUE:':25s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'tax_breakdown_detailed':
            # Detailed tax calculation
            text_lines.append(divider_full)
            text_lines.append("CALCULATION BREAKDOWN")
            text_lines.append(divider_light)
            text_lines.append(f"{'Merchandise:':25s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount:':25s} -${discount_num:>9.2f}")
                taxable = subtotal_num - discount_num
                text_lines.append(f"{'Taxable Amount:':25s} ${taxable:>10.2f}")
            if tax_num > 0:
                if tax_rate:
                    text_lines.append(f"{'Sales Tax (' + str(tax_rate) + '%):':25s} ${tax_num:>10.2f}")
                else:
                    text_lines.append(f"{'Sales Tax:':25s} ${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping (non-taxable):':25s} ${shipping_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'GRAND TOTAL:':25s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'tax_hidden':
            # Tax included in total, not itemized
            text_lines.append(divider_full)
            text_lines.append(f"{'Subtotal:':25s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount:':25s} -${discount_num:>9.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping:':25s} ${shipping_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'TOTAL (incl. tax):':25s} ${total_num:>10.2f}")
            if tax_num > 0:
                text_lines.append(f"{'  (includes tax:':25s}  ${tax_num:>.2f})")
            text_lines.append(divider_full)
            
        elif variant == 'currency_code_suffix':
            # Use currency codes instead of symbols
            text_lines.append(divider_full)
            text_lines.append(f"{'Subtotal:':25s} {subtotal_num:>10.2f} USD")
            if discount_num > 0:
                text_lines.append(f"{'Discount:':25s} -{discount_num:>9.2f} USD")
            if tax_num > 0:
                text_lines.append(f"{'Tax:':25s} {tax_num:>10.2f} USD")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping:':25s} {shipping_num:>10.2f} USD")
            text_lines.append(divider_light)
            text_lines.append(f"{'TOTAL:':25s} {total_num:>10.2f} USD")
            text_lines.append(divider_full)
            
        elif variant == 'currency_symbol_left':
            # Dollar sign before the number, close together
            text_lines.append(divider_full)
            text_lines.append(f"{'Subtotal':20s} ${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount':20s} -${discount_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Tax':20s} ${tax_num:.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping':20s} ${shipping_num:.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'TOTAL':20s} ${total_num:.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'tabular_grid':
            # Grid-like table format
            text_lines.append(divider_full)
            text_lines.append(f"{'ITEM':<20} | {'AMOUNT':>12}")
            text_lines.append(divider_light)
            text_lines.append(f"{'Subtotal':<20} | ${subtotal_num:>11.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount':<20} | -${discount_num:>10.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Tax':<20} | ${tax_num:>11.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping':<20} | ${shipping_num:>11.2f}")
            text_lines.append(divider_full)
            text_lines.append(f"{'TOTAL':<20} | ${total_num:>11.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'compact_inline':
            # Everything on fewer lines
            text_lines.append(divider_full)
            line_parts = []
            line_parts.append(f"Sub: ${subtotal_num:.2f}")
            if discount_num > 0:
                line_parts.append(f"Disc: -${discount_num:.2f}")
            if tax_num > 0:
                line_parts.append(f"Tax: ${tax_num:.2f}")
            if shipping_num > 0:
                line_parts.append(f"Ship: ${shipping_num:.2f}")
            if line_parts:
                text_lines.append(" | ".join(line_parts))
            text_lines.append(divider_light)
            text_lines.append(f"TOTAL: ${total_num:.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'verbose_labels':
            # Very descriptive labels
            text_lines.append(divider_full)
            text_lines.append(f"{'Merchandise Subtotal:':30s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Promotional Discount:':30s} -${discount_num:>9.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Estimated Sales Tax:':30s} ${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping & Handling Fee:':30s} ${shipping_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'Order Total:':30s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'minimal_two_line':
            # Just subtotal and total
            text_lines.append(divider_light)
            text_lines.append(f"Subtotal: ${subtotal_num:.2f}")
            text_lines.append(f"TOTAL: ${total_num:.2f}")
            text_lines.append(divider_light)
            
        elif variant == 'shipping_split_packages':
            # Split shipping by packages
            text_lines.append(divider_full)
            text_lines.append(f"{'Items Subtotal:':25s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount:':25s} -${discount_num:>9.2f}")
            if shipping_num > 0:
                # Split shipping into multiple packages (randomize)
                num_packages = random.randint(1, 3)
                if num_packages == 1:
                    text_lines.append(f"{'Shipping (1 package):':25s} ${shipping_num:>10.2f}")
                else:
                    per_package = shipping_num / num_packages
                    text_lines.append(f"{'Shipping:':25s}")
                    for i in range(num_packages):
                        text_lines.append(f"{'  Package ' + str(i+1) + ':':25s} ${per_package:>10.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Sales Tax:':25s} ${tax_num:>10.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'ORDER TOTAL:':25s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'rounding_adjustment':
            # Show rounding adjustment
            text_lines.append(divider_full)
            text_lines.append(f"{'Subtotal:':25s} ${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"{'Discount:':25s} -${discount_num:>9.2f}")
            if tax_num > 0:
                text_lines.append(f"{'Tax:':25s} ${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"{'Shipping:':25s} ${shipping_num:>10.2f}")
            # Add small rounding adjustment (0-0.04)
            rounding = random.uniform(-0.04, 0.04)
            if abs(rounding) > 0.001:
                sign = '+' if rounding > 0 else ''
                text_lines.append(f"{'Rounding Adjustment:':25s} {sign}${rounding:>9.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"{'TOTAL:':25s} ${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'grand_total_emphasized':
            # Total very emphasized
            text_lines.append(divider_full)
            text_lines.append(f"Subtotal: ${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"Discount: -${discount_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"Tax: ${tax_num:.2f}")
            if shipping_num > 0:
                text_lines.append(f"Shipping: ${shipping_num:.2f}")
            text_lines.append(divider_full)
            text_lines.append("")
            text_lines.append(f"{'*** GRAND TOTAL ***':^{width}}")
            text_lines.append(f"{'$' + f'{total_num:.2f}':^{width}}")
            text_lines.append("")
            text_lines.append(divider_full)
            
        elif variant == 'invoice_professional':
            # Professional invoice style
            text_lines.append(divider_full)
            text_lines.append("AMOUNT DUE CALCULATION")
            text_lines.append(divider_light)
            text_lines.append(f"  Net Amount{' ' * 20}${subtotal_num:>10.2f}")
            if discount_num > 0:
                text_lines.append(f"  Less: Discount{' ' * 16}-${discount_num:>9.2f}")
            if tax_num > 0:
                text_lines.append(f"  Add: Sales Tax{' ' * 16}${tax_num:>10.2f}")
            if shipping_num > 0:
                text_lines.append(f"  Add: Freight{' ' * 19}${shipping_num:>10.2f}")
            text_lines.append(divider_full)
            text_lines.append(f"  AMOUNT DUE{' ' * 20}${total_num:>10.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'receipt_casual':
            # Casual retail style
            text_lines.append(divider_light)
            text_lines.append(f"Items Total ........... ${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"You Saved ............. ${discount_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"Tax ................... ${tax_num:.2f}")
            text_lines.append(divider_light)
            text_lines.append(f"TOTAL ................. ${total_num:.2f}")
            text_lines.append(divider_light)
            
        elif variant == 'ecommerce_detailed':
            # E-commerce order summary
            text_lines.append(divider_full)
            text_lines.append("ORDER SUMMARY")
            text_lines.append(divider_light)
            item_count = len(receipt_data.get('line_items', []))
            text_lines.append(f"Items ({item_count}): ${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"Promo Code Applied: -${discount_num:.2f}")
            if shipping_num > 0:
                text_lines.append(f"Shipping & Handling: ${shipping_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"Estimated Tax: ${tax_num:.2f}")
            text_lines.append(divider_full)
            text_lines.append(f"Order Total: ${total_num:.2f}")
            text_lines.append(divider_full)
            
        elif variant == 'retail_pos_style':
            # POS terminal style
            text_lines.append("=" * min(width, 40))
            text_lines.append(f"SUBTOTAL{' ' * 15}${subtotal_num:.2f}")
            if discount_num > 0:
                text_lines.append(f"DISCOUNT{' ' * 15}-${discount_num:.2f}")
            if tax_num > 0:
                text_lines.append(f"TAX{' ' * 20}${tax_num:.2f}")
            text_lines.append("-" * min(width, 40))
            text_lines.append(f"TOTAL{' ' * 18}${total_num:.2f}")
            text_lines.append("=" * min(width, 40))
        
        return text_lines

    def _detect_receipt_type(self, receipt_data: dict) -> str:
        """
        Detect whether this is a retail/POS receipt or online/wholesale invoice
        
        Detection logic:
        - Retail/POS: Small orders (1-8 items), casual products, no PO numbers
        - Invoice: Large orders, has PO number, wholesale context, formal structure
        
        Args:
            receipt_data: Receipt data dictionary
            
        Returns:
            'retail' for POS/retail receipts (continuous roll)
            'invoice' for online/wholesale invoices (standard pages)
        """
        import random
        
        # Check for explicit invoice indicators
        has_po_number = bool(receipt_data.get('po_number'))
        has_vendor_id = bool(receipt_data.get('vendor_id'))
        has_account_number = bool(receipt_data.get('account_number'))
        has_payment_terms = bool(receipt_data.get('payment_terms'))
        
        # Strong invoice indicators
        if has_po_number or has_vendor_id or has_payment_terms:
            return 'invoice'
        
        # Check item count (large orders are usually invoices)
        line_items = receipt_data.get('line_items', [])
        num_items = len(line_items)
        
        if num_items > 15:
            return 'invoice'
        elif num_items > 8:
            # 8-15 items: could be either, 70% invoice
            return 'invoice' if random.random() < 0.7 else 'retail'
        else:
            # 1-8 items: mostly retail, 20% invoice
            return 'invoice' if random.random() < 0.2 else 'retail'

    def _generate_footer_section(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate footer section with 30 variants covering warranty text, return policies,
        supplier terms, contact details, payment footers, loyalty messages, and tracking info.
        
        Args:
            receipt_data: Receipt data dictionary
            width: Display width in characters
            style: Receipt style (centered, boxed, etc.)
            
        Returns:
            List of text lines for footer section
        """
        import random
        
        footer_lines = []
        
        # Get some context from receipt data
        supplier_name = receipt_data.get('supplier_name', 'Our Company')
        buyer_company = receipt_data.get('buyer_company', receipt_data.get('customer_name', 'Customer'))
        invoice_number = receipt_data.get('invoice_number', receipt_data.get('transaction_number', 'N/A'))
        payment_method = receipt_data.get('payment_method', 'Credit Card')
        
        # 30 footer variants
        footer_variants = [
            # ========== WARRANTY TEXT (5 variants) ==========
            {
                'type': '1_year_warranty',
                'lines': [
                    "═" * width,
                    "WARRANTY INFORMATION",
                    "This product is covered by a 1-year limited warranty from date of purchase.",
                    "For warranty service, contact us with your receipt and product details.",
                    f"Keep this receipt for warranty claims. Ref: {invoice_number}"
                ]
            },
            {
                'type': '90_day_guarantee',
                'lines': [
                    "─" * width,
                    "90-DAY MONEY BACK GUARANTEE",
                    "Not satisfied? Return within 90 days for a full refund.",
                    "No questions asked. We stand behind our products.",
                    "Visit our returns center or contact customer service."
                ]
            },
            {
                'type': 'lifetime_warranty',
                'lines': [
                    "★" * width,
                    "LIFETIME WARRANTY ON MANUFACTURING DEFECTS",
                    "We guarantee this product against defects in materials and workmanship.",
                    "Valid for the original purchaser only. Proof of purchase required.",
                    "Contact warranty@company.com for claims."
                ]
            },
            {
                'type': 'extended_warranty',
                'lines': [
                    "┄" * width,
                    "EXTENDED WARRANTY AVAILABLE",
                    "Purchase extended protection for up to 3 additional years.",
                    "Covers accidental damage, wear and tear, and technical support.",
                    "Ask about warranty plans at checkout or online."
                ]
            },
            {
                'type': 'no_warranty',
                'lines': [
                    "─" * width,
                    "SOLD AS-IS - NO WARRANTY",
                    "This item is sold without warranty, express or implied.",
                    "All sales are final. Inspect goods before purchase.",
                    "For questions, contact customer service within 7 days."
                ]
            },
            
            # ========== RETURN POLICIES (5 variants) ==========
            {
                'type': '30_day_return',
                'lines': [
                    "═" * width,
                    "RETURN POLICY",
                    "Returns accepted within 30 days of purchase with original receipt.",
                    "Items must be unused, in original packaging with all tags attached.",
                    f"Return authorization: {invoice_number} | Visit returns.{supplier_name.lower().replace(' ', '')}.com"
                ]
            },
            {
                'type': 'no_returns',
                'lines': [
                    "▬" * width,
                    "FINAL SALE - NO RETURNS OR EXCHANGES",
                    "All sales are final. Items cannot be returned or exchanged.",
                    "Please inspect your purchase carefully before leaving.",
                    "Questions? Contact us within 48 hours of purchase."
                ]
            },
            {
                'type': 'exchange_only',
                'lines': [
                    "─" * width,
                    "EXCHANGE POLICY",
                    "Exchanges only - no refunds. Exchange for store credit within 14 days.",
                    "Defective items will be replaced at no charge with proof of purchase.",
                    "Bring this receipt to any location for exchanges."
                ]
            },
            {
                'type': '14_day_window',
                'lines': [
                    "┅" * width,
                    "14-DAY RETURN WINDOW",
                    "Return unused items within 14 days for refund to original payment method.",
                    "Opened items subject to 15% restocking fee. Sale items are final.",
                    f"Keep this receipt. Return code: {invoice_number}"
                ]
            },
            {
                'type': 'restocking_fee',
                'lines': [
                    "━" * width,
                    "RETURN POLICY - RESTOCKING FEE APPLIES",
                    "Returns accepted within 30 days. 20% restocking fee on all returns.",
                    "Defective items eligible for full refund with no restocking fee.",
                    "Contact customer service to initiate return process."
                ]
            },
            
            # ========== SUPPLIER TERMS (5 variants) ==========
            {
                'type': 'net_30',
                'lines': [
                    "═" * width,
                    "PAYMENT TERMS: NET 30",
                    f"Payment due 30 days from invoice date. Invoice: {invoice_number}",
                    f"Make checks payable to: {supplier_name}",
                    "Late payments subject to 1.5% monthly interest. Thank you for your business."
                ]
            },
            {
                'type': 'early_payment_discount',
                'lines': [
                    "─" * width,
                    "PAYMENT TERMS: 2/10 NET 30",
                    f"Take 2% discount if paid within 10 days. Full payment due in 30 days.",
                    f"Invoice Date: {receipt_data.get('issue_date', 'N/A')} | Invoice: {invoice_number}",
                    f"Remit to: {supplier_name} Accounts Receivable"
                ]
            },
            {
                'type': 'payment_due_now',
                'lines': [
                    "▬" * width,
                    "PAYMENT TERMS: DUE UPON RECEIPT",
                    "Payment is due immediately upon receipt of goods/services.",
                    f"Invoice: {invoice_number} | Payment Method: {payment_method}",
                    "Contact our billing department with payment questions."
                ]
            },
            {
                'type': 'net_60',
                'lines': [
                    "═" * width,
                    "EXTENDED PAYMENT TERMS: NET 60",
                    f"Payment due 60 days from invoice date for qualified accounts.",
                    f"Invoice: {invoice_number} | Account: {buyer_company}",
                    "Questions? Contact accounts receivable at billing@company.com"
                ]
            },
            {
                'type': 'cod_only',
                'lines': [
                    "─" * width,
                    "PAYMENT TERMS: COD OR CREDIT CARD ONLY",
                    "Cash on delivery or credit card payment required.",
                    "No checks accepted. No credit terms available.",
                    f"Payment received: {payment_method} | Reference: {invoice_number}"
                ]
            },
            
            # ========== CONTACT DETAILS (5 variants) ==========
            {
                'type': 'phone_email',
                'lines': [
                    "═" * width,
                    f"{supplier_name} - CONTACT INFORMATION",
                    "Phone: (555) 123-4567 | Email: support@company.com",
                    "Hours: Monday-Friday 9AM-6PM, Saturday 10AM-4PM",
                    "We're here to help! Contact us with any questions."
                ]
            },
            {
                'type': 'website_hours',
                'lines': [
                    "─" * width,
                    "VISIT US ONLINE OR IN STORE",
                    f"Website: www.{supplier_name.lower().replace(' ', '')}.com",
                    "Store Hours: Mon-Sat 10AM-8PM, Sun 12PM-6PM",
                    "Online orders ship within 1-2 business days"
                ]
            },
            {
                'type': 'social_media',
                'lines': [
                    "┄" * width,
                    "FOLLOW US ON SOCIAL MEDIA",
                    f"Instagram: @{supplier_name.lower().replace(' ', '')} | Facebook: /{supplier_name.replace(' ', '')}",
                    f"Twitter: @{supplier_name.lower().replace(' ', '')}_official",
                    "Tag us in your photos! Use #MyCompanyPurchase for a chance to be featured"
                ]
            },
            {
                'type': 'multiple_locations',
                'lines': [
                    "═" * width,
                    f"{supplier_name} - STORE LOCATIONS",
                    "Downtown: 123 Main St | West Side: 456 Oak Ave | East Mall: 789 Elm Rd",
                    "Find your nearest location at www.company.com/locations",
                    "All locations accept returns with receipt"
                ]
            },
            {
                'type': 'customer_service',
                'lines': [
                    "─" * width,
                    "CUSTOMER SERVICE HOTLINE",
                    "Need help? Call 1-800-555-HELP (4357) anytime, 24/7",
                    "Live chat available on our website during business hours",
                    "Email: customercare@company.com | Response within 24 hours"
                ]
            },
            
            # ========== PAYMENT FOOTERS (5 variants) ==========
            {
                'type': 'authorization_code',
                'lines': [
                    "═" * width,
                    "PAYMENT AUTHORIZATION",
                    f"Authorization Code: AUTH{random.randint(100000, 999999)}",
                    f"Transaction ID: {invoice_number}",
                    "Keep this receipt for your records. Payment approved."
                ]
            },
            {
                'type': 'card_details',
                'lines': [
                    "─" * width,
                    "PAYMENT DETAILS",
                    f"Card Type: {payment_method} | Last 4 Digits: {random.randint(1000, 9999)}",
                    f"Transaction Date: {receipt_data.get('issue_date', 'N/A')}",
                    "Thank you for your payment. Receipt for your records."
                ]
            },
            {
                'type': 'payment_confirmation',
                'lines': [
                    "▬" * width,
                    "PAYMENT CONFIRMATION",
                    f"Confirmation Number: CONF-{random.randint(1000000, 9999999)}",
                    f"Payment Method: {payment_method} | Status: APPROVED",
                    f"Reference: {invoice_number} | Keep for your records"
                ]
            },
            {
                'type': 'thank_you_payment',
                'lines': [
                    "═" * width,
                    "THANK YOU FOR YOUR PAYMENT",
                    f"Payment received via {payment_method}",
                    "Your account has been credited. Thank you for prompt payment!",
                    f"Questions? Reference invoice {invoice_number} when contacting us."
                ]
            },
            {
                'type': 'tax_receipt',
                'lines': [
                    "─" * width,
                    "RECEIPT FOR TAX PURPOSES",
                    f"Tax ID: {random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
                    f"Invoice: {invoice_number} | Keep for tax records",
                    "Consult your tax advisor regarding deductibility."
                ]
            },
            
            # ========== LOYALTY MESSAGES (5 variants) ==========
            {
                'type': 'points_earned',
                'lines': [
                    "★" * width,
                    "REWARDS POINTS EARNED",
                    f"You earned {random.randint(50, 500)} points with this purchase!",
                    f"Current Balance: {random.randint(1000, 5000)} points",
                    "Redeem points online or in-store. 100 points = $1 off!"
                ]
            },
            {
                'type': 'join_rewards',
                'lines': [
                    "═" * width,
                    "JOIN OUR REWARDS PROGRAM TODAY!",
                    "Earn 1 point per $1 spent. Get $10 off your next $50 purchase when you join!",
                    "Sign up at www.company.com/rewards or ask at checkout",
                    "Members get exclusive deals, birthday gifts, and early access to sales!"
                ]
            },
            {
                'type': 'member_exclusive',
                'lines': [
                    "─" * width,
                    "MEMBER EXCLUSIVE DISCOUNTS",
                    "As a valued member, you saved 15% on this purchase!",
                    "Check your email for members-only flash sales and promotions",
                    f"Member ID: {buyer_company[:20]} | Points never expire!"
                ]
            },
            {
                'type': 'referral_bonus',
                'lines': [
                    "┄" * width,
                    "REFER A FRIEND - GET $20!",
                    "Love our products? Refer a friend and you both get $20 off!",
                    "Share your unique referral code: FRIEND20",
                    "No limit on referrals. More friends = more savings!"
                ]
            },
            {
                'type': 'vip_tier',
                'lines': [
                    "★" * width,
                    "VIP TIER BENEFITS",
                    "You're a VIP! Enjoy free shipping, priority support, and exclusive access.",
                    f"VIP Status: Gold | Member Since: {receipt_data.get('issue_date', 'N/A')[:4] if receipt_data.get('issue_date') else '2024'}",
                    "Thank you for being a loyal customer!"
                ]
            },
            
            # ========== TRACKING INFO (5 variants) ==========
            {
                'type': 'order_tracking_url',
                'lines': [
                    "═" * width,
                    "TRACK YOUR ORDER ONLINE",
                    f"Order Number: {invoice_number}",
                    f"Track at: www.company.com/track?order={invoice_number}",
                    "You'll receive email updates as your order ships and delivers."
                ]
            },
            {
                'type': 'estimated_delivery',
                'lines': [
                    "─" * width,
                    "SHIPMENT INFORMATION",
                    f"Order: {invoice_number} | Estimated Delivery: 3-5 business days",
                    "Tracking number will be emailed within 24 hours of shipment.",
                    "Questions? Contact shipping@company.com"
                ]
            },
            {
                'type': 'tracking_number',
                'lines': [
                    "▬" * width,
                    "SHIPMENT TRACKING",
                    f"Tracking Number: 1Z{random.randint(100000000, 999999999)}",
                    f"Carrier: UPS Ground | Expected Delivery: 5-7 business days",
                    "Track your package at www.ups.com or www.company.com/track"
                ]
            },
            {
                'type': 'track_online',
                'lines': [
                    "═" * width,
                    "TRACK YOUR PACKAGE",
                    f"Visit www.company.com/tracking and enter: {invoice_number}",
                    "Real-time updates from warehouse to your door.",
                    "Download our mobile app for push notifications!"
                ]
            },
            {
                'type': 'shipping_notifications',
                'lines': [
                    "─" * width,
                    "SHIPPING NOTIFICATIONS ENABLED",
                    f"You'll receive SMS and email updates for order {invoice_number}",
                    "Updates include: Order confirmed, Shipped, Out for delivery, Delivered",
                    "Manage notification preferences at www.company.com/account"
                ]
            }
        ]
        
        # Choose a random footer variant
        footer_variant = random.choice(footer_variants)
        
        # Format lines based on style
        for line in footer_variant['lines']:
            if style in ['centered', 'boxed', 'banner']:
                # Center the line
                footer_lines.append(f"{line:^{width}}")
            else:
                # Left-aligned or default
                footer_lines.append(line)
        
        return footer_lines

    def _generate_barcode_section(self, receipt_data: dict, width: int, style: str) -> list:
        """
        Generate barcode/QR code section with 15 realistic variants
        
        Barcode types represented:
        - UPC-A (12 digits)
        - EAN-13 (13 digits)
        - Code 128 (alphanumeric)
        - Code 39 (alphanumeric with *)
        - ITF (Interleaved 2 of 5)
        - QR codes (URL/transaction data)
        - DataMatrix (compact 2D)
        - MicroQR (small format)
        - Template-specific barcodes
        
        Args:
            receipt_data: Receipt data dictionary
            width: Line width
            style: Receipt style
            
        Returns:
            List of text lines for barcode section
        """
        import random
        text_lines = []
        
        # Extract barcodeable data
        transaction_num = receipt_data.get('transaction_number', '')
        invoice_num = receipt_data.get('invoice_number', '')
        store_id = receipt_data.get('store_id', '')
        
        # Generate barcode number if needed
        if not transaction_num and not invoice_num:
            barcode_num = f"{random.randint(100000000000, 999999999999)}"
        else:
            barcode_num = str(transaction_num or invoice_num)
        
        # 15 barcode/QR variants
        barcode_variants = [
            'upc_a',
            'ean_13',
            'code_128',
            'code_39',
            'itf',
            'qr_url',
            'qr_transaction',
            'datamatrix',
            'micro_qr',
            'simple_bars',
            'double_bars',
            'box_code',
            'receipt_id_barcode',
            'order_tracking_barcode',
            'mixed_2d_1d'
        ]
        
        variant = random.choice(barcode_variants)
        
        # === IMPLEMENTATION OF EACH VARIANT ===
        
        if variant == 'upc_a':
            # UPC-A: 12 digits with guard patterns
            upc_digits = str(barcode_num).replace('-', '').replace(' ', '')[:12].zfill(12)
            text_lines.append('|' + ' |' * (width // 2 - 1) + ' |')
            text_lines.append(f"{upc_digits[:1]} {upc_digits[1:6]} {upc_digits[6:11]} {upc_digits[11:]:^{width-15}}")
            text_lines.append('| ' + '|' * (width // 2 - 1) + '|')
            text_lines.append(f"{'UPC-A':^{width}}")
            
        elif variant == 'ean_13':
            # EAN-13: 13 digits with country code
            ean_digits = str(barcode_num).replace('-', '').replace(' ', '')[:13].zfill(13)
            text_lines.append('| ' + '|| ' * 10 + '| ')
            text_lines.append(f"{ean_digits:^{width}}")
            text_lines.append('| ' + '|| ' * 10 + '| ')
            
        elif variant == 'code_128':
            # Code 128: Alphanumeric with START/STOP
            code_value = str(barcode_num)[:20]
            text_lines.append('█' + '▌▌' * 8 + '█')
            text_lines.append(f"{'START':>8} {code_value:^{width-20}} {'STOP':<8}")
            text_lines.append('█' + '▌▌' * 8 + '█')
            text_lines.append(f"{'Code 128':^{width}}")
            
        elif variant == 'code_39':
            # Code 39: With asterisk delimiters
            code_value = str(barcode_num)[:15]
            text_lines.append('*' + '|' * (width - 2) + '*')
            text_lines.append(f"* {code_value:^{width-4}} *")
            text_lines.append('*' + '|' * (width - 2) + '*')
            text_lines.append(f"{'*CODE39*':^{width}}")
            
        elif variant == 'itf':
            # ITF (Interleaved 2 of 5): Paired thick/thin bars
            itf_digits = str(barcode_num).replace('-', '')[:14].zfill(14)
            # Represent paired bars
            text_lines.append('▐ ▌▐  ▌▐ ▌▐  ▌▐ ▌▐  ▌▐ ▌')
            text_lines.append(f"{itf_digits:^{width}}")
            text_lines.append('▐ ▌▐  ▌▐ ▌▐  ▌▐ ▌▐  ▌▐ ▌')
            text_lines.append(f"{'ITF-14':^{width}}")
            
        elif variant == 'qr_url':
            # QR Code with URL
            qr_size = random.choice([5, 7, 9])
            text_lines.append('╔' + '═' * (qr_size * 2) + '╗')
            for _ in range(qr_size):
                # Random QR pattern
                pattern = ''.join(random.choice(['██', '  ']) for _ in range(qr_size))
                text_lines.append('║' + pattern + '║')
            text_lines.append('╚' + '═' * (qr_size * 2) + '╝')
            text_lines.append(f"{'QR: example.com/r/' + str(barcode_num)[:8]:^{width}}")
            
        elif variant == 'qr_transaction':
            # QR Code with transaction data
            qr_size = random.choice([5, 7])
            text_lines.append('┌' + '─' * (qr_size * 2) + '┐')
            for _ in range(qr_size):
                pattern = ''.join(random.choice(['▓▓', '░░']) for _ in range(qr_size))
                text_lines.append('│' + pattern + '│')
            text_lines.append('└' + '─' * (qr_size * 2) + '┘')
            text_lines.append(f"{'Scan for receipt details':^{width}}")
            
        elif variant == 'datamatrix':
            # DataMatrix: Small 2D barcode
            dm_size = random.choice([5, 7])
            text_lines.append('┏' + '━' * dm_size + '┓')
            for _ in range(dm_size):
                pattern = ''.join(random.choice(['█', '·']) for _ in range(dm_size))
                text_lines.append('┃' + pattern + '┃')
            text_lines.append('┗' + '━' * dm_size + '┛')
            text_lines.append(f"{'DataMatrix':^{width}}")
            
        elif variant == 'micro_qr':
            # MicroQR: Compact QR variant
            text_lines.append('╔══╗')
            text_lines.append('║▓░║')
            text_lines.append('║░▓║')
            text_lines.append('╚══╝')
            text_lines.append(f"{'µQR':^{width}}")
            
        elif variant == 'simple_bars':
            # Simple vertical bars (classic receipt)
            text_lines.append('|' * width)
            text_lines.append(f"{barcode_num:^{width}}")
            text_lines.append('|' * width)
            
        elif variant == 'double_bars':
            # Double-line bars
            text_lines.append('║' * (width // 2))
            text_lines.append(f"{barcode_num:^{width}}")
            text_lines.append('║' * (width // 2))
            
        elif variant == 'box_code':
            # Boxed barcode with label
            box_width = min(width, len(barcode_num) + 10)
            text_lines.append('┌' + '─' * (box_width - 2) + '┐')
            text_lines.append('│' + '▌' * (box_width - 2) + '│')
            text_lines.append(f"│{barcode_num:^{box_width-2}}│")
            text_lines.append('│' + '▌' * (box_width - 2) + '│')
            text_lines.append('└' + '─' * (box_width - 2) + '┘')
            
        elif variant == 'receipt_id_barcode':
            # Receipt-specific with ID label
            text_lines.append('*' * width)
            text_lines.append(f"{'RECEIPT ID':^{width}}")
            text_lines.append(f"{'#' + barcode_num:^{width}}")
            text_lines.append('*' * width)
            
        elif variant == 'order_tracking_barcode':
            # Order tracking style
            tracking = f"TRK-{barcode_num[:10]}"
            text_lines.append('=' * width)
            text_lines.append(f"{'ORDER TRACKING':^{width}}")
            text_lines.append('|' + ' ' + '|' * (width // 2 - 2) + ' ' + '|')
            text_lines.append(f"{tracking:^{width}}")
            text_lines.append('|' + ' ' + '|' * (width // 2 - 2) + ' ' + '|')
            text_lines.append('=' * width)
            
        elif variant == 'mixed_2d_1d':
            # Mixed: QR code + 1D barcode
            # Small QR
            text_lines.append('  ╔═══╗')
            text_lines.append('  ║▓░▓║')
            text_lines.append('  ║░▓░║')
            text_lines.append('  ╚═══╝')
            text_lines.append('')
            # 1D barcode below
            text_lines.append('|' * width)
            text_lines.append(f"{barcode_num:^{width}}")
            text_lines.append('|' * width)
        
        return text_lines

    def _get_locale_config(self, locale: Optional[str] = None) -> dict:
        """
        Get comprehensive locale configuration for language and regional formatting.
        
        Handles:
        - Date formats (MM/DD/YYYY vs DD/MM/YYYY vs YYYY-MM-DD)
        - Currency symbols and placement
        - Decimal separators (. vs ,)
        - Thousand separators (, vs . vs space)
        - Tax terminology and rates
        - Language-specific labels
        
        Args:
            locale: Locale code (e.g., 'en_US', 'fr_CA', 'de_DE')
                   If None, randomly selects a locale
        
        Returns:
            Dictionary with locale-specific formatting rules
        """
        import random
        
        # Comprehensive locale configurations
        locale_configs = {
            'en_US': {
                'name': 'English (United States)',
                'language': 'en',
                'currency_symbol': '$',
                'currency_code': 'USD',
                'currency_position': 'before',  # $100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'MM/DD/YYYY',
                'date_format_short': 'M/D/YY',
                'tax_label': 'Sales Tax',
                'tax_rates': [0.0, 0.0625, 0.07, 0.0825, 0.0875, 0.10],  # State-dependent
                'labels': {
                    'invoice': 'Invoice',
                    'receipt': 'Receipt',
                    'date': 'Date',
                    'due_date': 'Due Date',
                    'subtotal': 'Subtotal',
                    'discount': 'Discount',
                    'tax': 'Tax',
                    'total': 'Total',
                    'amount_due': 'Amount Due',
                    'thank_you': 'Thank you for your business!',
                    'page': 'Page',
                    'of': 'of'
                }
            },
            'en_GB': {
                'name': 'English (United Kingdom)',
                'language': 'en',
                'currency_symbol': '£',
                'currency_code': 'GBP',
                'currency_position': 'before',  # £100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'DD/MM/YYYY',
                'date_format_short': 'D/M/YY',
                'tax_label': 'VAT',
                'tax_rates': [0.0, 0.20],  # Standard VAT rate
                'labels': {
                    'invoice': 'Invoice',
                    'receipt': 'Receipt',
                    'date': 'Date',
                    'due_date': 'Due Date',
                    'subtotal': 'Subtotal',
                    'discount': 'Discount',
                    'tax': 'VAT',
                    'total': 'Total',
                    'amount_due': 'Amount Due',
                    'thank_you': 'Thank you for your custom!',
                    'page': 'Page',
                    'of': 'of'
                }
            },
            'en_CA': {
                'name': 'English (Canada)',
                'language': 'en',
                'currency_symbol': '$',
                'currency_code': 'CAD',
                'currency_position': 'before',  # $100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'YYYY-MM-DD',  # ISO format common in Canada
                'date_format_short': 'YY-MM-DD',
                'tax_label': 'GST/HST',
                'tax_rates': [0.05, 0.13, 0.15],  # GST, HST Ontario, HST Atlantic
                'labels': {
                    'invoice': 'Invoice',
                    'receipt': 'Receipt',
                    'date': 'Date',
                    'due_date': 'Due Date',
                    'subtotal': 'Subtotal',
                    'discount': 'Discount',
                    'tax': 'Tax',
                    'total': 'Total',
                    'amount_due': 'Amount Due',
                    'thank_you': 'Thank you for your business!',
                    'page': 'Page',
                    'of': 'of'
                }
            },
            'en_AU': {
                'name': 'English (Australia)',
                'language': 'en',
                'currency_symbol': '$',
                'currency_code': 'AUD',
                'currency_position': 'before',  # $100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'DD/MM/YYYY',
                'date_format_short': 'D/M/YY',
                'tax_label': 'GST',
                'tax_rates': [0.0, 0.10],  # GST rate
                'labels': {
                    'invoice': 'Tax Invoice',  # Required for GST
                    'receipt': 'Receipt',
                    'date': 'Date',
                    'due_date': 'Due Date',
                    'subtotal': 'Subtotal',
                    'discount': 'Discount',
                    'tax': 'GST',
                    'total': 'Total',
                    'amount_due': 'Amount Due',
                    'thank_you': 'Thank you for your business!',
                    'page': 'Page',
                    'of': 'of'
                }
            },
            'fr_CA': {
                'name': 'Français (Canada)',
                'language': 'fr',
                'currency_symbol': '$',
                'currency_code': 'CAD',
                'currency_position': 'before',  # 100,00 $
                'decimal_separator': ',',
                'thousand_separator': ' ',  # Space
                'date_format': 'YYYY-MM-DD',
                'date_format_short': 'AA-MM-JJ',
                'tax_label': 'TPS/TVQ',
                'tax_rates': [0.05, 0.09975, 0.14975],  # TPS, TVQ Quebec, Combined
                'labels': {
                    'invoice': 'Facture',
                    'receipt': 'Reçu',
                    'date': 'Date',
                    'due_date': 'Date d\'échéance',
                    'subtotal': 'Sous-total',
                    'discount': 'Rabais',
                    'tax': 'Taxes',
                    'total': 'Total',
                    'amount_due': 'Montant dû',
                    'thank_you': 'Merci pour votre entreprise!',
                    'page': 'Page',
                    'of': 'de'
                }
            },
            'fr_FR': {
                'name': 'Français (France)',
                'language': 'fr',
                'currency_symbol': '€',
                'currency_code': 'EUR',
                'currency_position': 'after',  # 100,00 €
                'decimal_separator': ',',
                'thousand_separator': ' ',  # Space
                'date_format': 'DD/MM/YYYY',
                'date_format_short': 'JJ/MM/AA',
                'tax_label': 'TVA',
                'tax_rates': [0.0, 0.055, 0.10, 0.20],  # Various VAT rates
                'labels': {
                    'invoice': 'Facture',
                    'receipt': 'Reçu',
                    'date': 'Date',
                    'due_date': 'Date d\'échéance',
                    'subtotal': 'Sous-total',
                    'discount': 'Remise',
                    'tax': 'TVA',
                    'total': 'Total',
                    'amount_due': 'Montant dû',
                    'thank_you': 'Merci pour votre confiance!',
                    'page': 'Page',
                    'of': 'sur'
                }
            },
            'es_ES': {
                'name': 'Español (España)',
                'language': 'es',
                'currency_symbol': '€',
                'currency_code': 'EUR',
                'currency_position': 'after',  # 100,00 €
                'decimal_separator': ',',
                'thousand_separator': '.',
                'date_format': 'DD/MM/YYYY',
                'date_format_short': 'DD/MM/AA',
                'tax_label': 'IVA',
                'tax_rates': [0.0, 0.04, 0.10, 0.21],  # Super-reduced, reduced, standard
                'labels': {
                    'invoice': 'Factura',
                    'receipt': 'Recibo',
                    'date': 'Fecha',
                    'due_date': 'Fecha de vencimiento',
                    'subtotal': 'Subtotal',
                    'discount': 'Descuento',
                    'tax': 'IVA',
                    'total': 'Total',
                    'amount_due': 'Importe a pagar',
                    'thank_you': '¡Gracias por su compra!',
                    'page': 'Página',
                    'of': 'de'
                }
            },
            'es_MX': {
                'name': 'Español (México)',
                'language': 'es',
                'currency_symbol': '$',
                'currency_code': 'MXN',
                'currency_position': 'before',  # $100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'DD/MM/YYYY',
                'date_format_short': 'DD/MM/AA',
                'tax_label': 'IVA',
                'tax_rates': [0.0, 0.16],  # Standard IVA rate
                'labels': {
                    'invoice': 'Factura',
                    'receipt': 'Recibo',
                    'date': 'Fecha',
                    'due_date': 'Fecha de vencimiento',
                    'subtotal': 'Subtotal',
                    'discount': 'Descuento',
                    'tax': 'IVA',
                    'total': 'Total',
                    'amount_due': 'Monto a pagar',
                    'thank_you': '¡Gracias por su compra!',
                    'page': 'Página',
                    'of': 'de'
                }
            },
            'de_DE': {
                'name': 'Deutsch (Deutschland)',
                'language': 'de',
                'currency_symbol': '€',
                'currency_code': 'EUR',
                'currency_position': 'after',  # 100,00 €
                'decimal_separator': ',',
                'thousand_separator': '.',
                'date_format': 'DD.MM.YYYY',
                'date_format_short': 'DD.MM.YY',
                'tax_label': 'MwSt.',
                'tax_rates': [0.0, 0.07, 0.19],  # Reduced and standard rates
                'labels': {
                    'invoice': 'Rechnung',
                    'receipt': 'Quittung',
                    'date': 'Datum',
                    'due_date': 'Fälligkeitsdatum',
                    'subtotal': 'Zwischensumme',
                    'discount': 'Rabatt',
                    'tax': 'MwSt.',
                    'total': 'Gesamt',
                    'amount_due': 'Zu zahlender Betrag',
                    'thank_you': 'Vielen Dank für Ihren Einkauf!',
                    'page': 'Seite',
                    'of': 'von'
                }
            },
            'zh_CN': {
                'name': '中文 (中国)',
                'language': 'zh',
                'currency_symbol': '¥',
                'currency_code': 'CNY',
                'currency_position': 'before',  # ¥100.00
                'decimal_separator': '.',
                'thousand_separator': ',',
                'date_format': 'YYYY年MM月DD日',
                'date_format_short': 'YY-MM-DD',
                'tax_label': '增值税',
                'tax_rates': [0.0, 0.03, 0.06, 0.09, 0.13],  # Various VAT rates
                'labels': {
                    'invoice': '发票',
                    'receipt': '收据',
                    'date': '日期',
                    'due_date': '到期日',
                    'subtotal': '小计',
                    'discount': '折扣',
                    'tax': '税',
                    'total': '总计',
                    'amount_due': '应付金额',
                    'thank_you': '感谢您的惠顾！',
                    'page': '第',
                    'of': '页，共'
                }
            }
        }
        
        # If no locale specified, randomly choose one
        if locale is None:
            # Weight distribution to favor English variants but include diversity
            locale_weights = {
                'en_US': 0.40,
                'en_GB': 0.15,
                'en_CA': 0.10,
                'en_AU': 0.08,
                'fr_CA': 0.07,
                'fr_FR': 0.05,
                'es_ES': 0.05,
                'es_MX': 0.04,
                'de_DE': 0.04,
                'zh_CN': 0.02
            }
            locales = list(locale_weights.keys())
            weights = list(locale_weights.values())
            locale = random.choices(locales, weights=weights)[0]
        
        return locale_configs.get(locale, locale_configs['en_US'])

    def _format_currency(self, amount: float, locale_config: dict, style: Optional[str] = None) -> str:
        """
        Format currency amount according to locale rules with 15 style variants.
        
        Args:
            amount: Numeric amount
            locale_config: Locale configuration from _get_locale_config()
            style: Currency formatting style (if None, randomly selected)
            
        Returns:
            Formatted currency string
            
        Style Options (15 variants):
        1. symbol_before: $14.99
        2. symbol_after: 14.99$
        3. symbol_space_before: $ 14.99
        4. symbol_space_after: 14.99 $
        5. code_before: USD 14.99
        6. code_after: 14.99 USD
        7. code_no_space_before: USD14.99
        8. code_no_space_after: 14.99USD
        9. symbol_parentheses: $(14.99)
        10. code_parentheses: USD(14.99)
        11. tax_included_suffix: 14.99 $ (tax incl.)
        12. tax_included_code: 14.99 USD (tax incl.)
        13. with_currency_name: $14.99 USD
        14. accounting_negative: (14.99)
        15. code_hyphen: USD-14.99
        """
        import random
        
        # Format number with proper decimal/thousand separators
        amount_str = f"{abs(amount):.2f}"
        integer_part, decimal_part = amount_str.split('.')
        
        # Add thousand separators
        if len(integer_part) > 3:
            thousand_sep = locale_config['thousand_separator']
            formatted_int = ''
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_int = thousand_sep + formatted_int
                formatted_int = digit + formatted_int
            integer_part = formatted_int
        
        # Combine with locale decimal separator
        decimal_sep = locale_config['decimal_separator']
        formatted_amount = f"{integer_part}{decimal_sep}{decimal_part}"
        
        symbol = locale_config['currency_symbol']
        code = locale_config['currency_code']
        
        # Currency formatting styles (15 variants)
        styles = [
            'symbol_before',           # $14.99
            'symbol_after',            # 14.99$
            'symbol_space_before',     # $ 14.99
            'symbol_space_after',      # 14.99 $
            'code_before',             # USD 14.99
            'code_after',              # 14.99 USD
            'code_no_space_before',    # USD14.99
            'code_no_space_after',     # 14.99USD
            'symbol_parentheses',      # $(14.99)
            'code_parentheses',        # USD(14.99)
            'tax_included_suffix',     # 14.99 $ (tax incl.)
            'tax_included_code',       # 14.99 USD (tax incl.)
            'with_currency_name',      # $14.99 USD
            'accounting_negative',     # (14.99) for negative/special
            'code_hyphen'              # USD-14.99
        ]
        
        # Select style
        if style is None:
            # Favor standard styles (first 4) 70% of the time
            if random.random() < 0.70:
                style = random.choice(styles[:4])
            else:
                style = random.choice(styles[4:])
        
        # Apply formatting style
        if style == 'symbol_before':
            result = f"{symbol}{formatted_amount}"
        
        elif style == 'symbol_after':
            result = f"{formatted_amount}{symbol}"
        
        elif style == 'symbol_space_before':
            result = f"{symbol} {formatted_amount}"
        
        elif style == 'symbol_space_after':
            result = f"{formatted_amount} {symbol}"
        
        elif style == 'code_before':
            result = f"{code} {formatted_amount}"
        
        elif style == 'code_after':
            result = f"{formatted_amount} {code}"
        
        elif style == 'code_no_space_before':
            result = f"{code}{formatted_amount}"
        
        elif style == 'code_no_space_after':
            result = f"{formatted_amount}{code}"
        
        elif style == 'symbol_parentheses':
            result = f"{symbol}({formatted_amount})"
        
        elif style == 'code_parentheses':
            result = f"{code}({formatted_amount})"
        
        elif style == 'tax_included_suffix':
            result = f"{formatted_amount} {symbol} (tax incl.)"
        
        elif style == 'tax_included_code':
            result = f"{formatted_amount} {code} (tax incl.)"
        
        elif style == 'with_currency_name':
            result = f"{symbol}{formatted_amount} {code}"
        
        elif style == 'accounting_negative':
            # Accounting style - parentheses for emphasis (not necessarily negative)
            result = f"({formatted_amount})"
        
        elif style == 'code_hyphen':
            result = f"{code}-{formatted_amount}"
        
        else:
            # Fallback to locale default
            if locale_config['currency_position'] == 'before':
                result = f"{symbol}{formatted_amount}"
            else:
                result = f"{formatted_amount} {symbol}"
        
        # Handle negative amounts (prefix with minus for most styles)
        if amount < 0 and style not in ['accounting_negative', 'symbol_parentheses', 'code_parentheses']:
            result = f"-{result}"
        
        return result

    def _format_date(self, date_str: str, locale_config: dict) -> str:
        """
        Format date string according to locale rules.
        
        Args:
            date_str: Date in YYYY-MM-DD format
            locale_config: Locale configuration from _get_locale_config()
            
        Returns:
            Formatted date string
        """
        from datetime import datetime
        
        try:
            # Parse the date
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Format according to locale
            date_format = locale_config['date_format']
            
            # Map format tokens to strftime codes
            format_map = {
                'YYYY': '%Y',
                'YY': '%y',
                'MM': '%m',
                'M': '%-m' if hasattr(datetime, '%-m') else '%m',
                'DD': '%d',
                'D': '%-d' if hasattr(datetime, '%-d') else '%d',
                '年': '年',
                '月': '月',
                '日': '日'
            }
            
            # Special handling for Chinese format
            if '年' in date_format:
                return date_obj.strftime('%Y年%m月%d日')
            
            # Replace tokens
            strftime_format = date_format
            for token, code in format_map.items():
                strftime_format = strftime_format.replace(token, code)
            
            return date_obj.strftime(strftime_format)
        except:
            # Fallback to original if parsing fails
            return date_str

    def _should_use_multipage(self, receipt_data: dict) -> bool:
        """
        Determine if receipt should use multi-page rendering
        Based on receipt type, number of line items, and estimated content length
        
        IMPORTANT DISTINCTION:
        - Retail/POS receipts: Always single continuous roll (thermal paper)
        - Online/Wholesale/Invoice: Standard page size, can be multi-page
        
        Multi-page receipts are typical for:
        - E-commerce orders (often 15+ items)
        - Wholesale/B2B invoices (large quantities)
        - Corporate procurement orders
        
        Args:
            receipt_data: Receipt data dictionary
            
        Returns:
            True if multi-page should be used
        """
        import random
        
        # Determine receipt type based on context clues
        receipt_type = self._detect_receipt_type(receipt_data)
        
        # Retail/POS receipts are ALWAYS single continuous page
        if receipt_type == 'retail':
            return False
        
        # For invoice/online/wholesale types, check item count
        line_items = receipt_data.get('line_items', [])
        num_items = len(line_items)
        
        # Estimate total content lines to detect if we'll overflow single page
        # Letter size page can fit approximately 50-55 lines comfortably (with 20px line spacing)
        estimated_lines = 0
        
        # Header section: 10-20 lines depending on style
        estimated_lines += 15
        
        # Transaction/buyer/metadata sections: 5-15 lines
        estimated_lines += 10
        
        # Line items: Each item is ~2-4 lines depending on table format
        estimated_lines += num_items * 3
        
        # Totals section: 8-15 lines
        estimated_lines += 12
        
        # Payment, footer, barcode sections: 10-20 lines
        estimated_lines += 15
        
        # CRITICAL: If estimated content exceeds single page capacity, FORCE multi-page
        # This must happen BEFORE random logic to prevent overflow
        if estimated_lines > 52:
            return True
        
        # Always use multi-page for large orders (>15 items)
        if num_items > 15:
            return True
        
        # For smaller orders that fit on one page, use random distribution
        # Medium orders (10-15 items): 60% chance
        if num_items >= 10:
            return random.random() < 0.6
        
        # Small-medium orders (7-9 items): 30% chance
        if num_items >= 7:
            return random.random() < 0.3
        
        # Small orders (5-6 items): 10% chance
        if num_items >= 5:
            return random.random() < 0.1
        
        # Very small orders: never multi-page
        return False
    
    def _generate_page_header(self, receipt_data: dict, width: int, page_num: int, total_pages: int) -> list:
        """
        Generate page header for multi-page documents
        
        Returns:
            List of text lines for page header
        """
        import random
        text_lines = []
        
        # Page indicator styles
        header_styles = [
            'top_right',
            'centered',
            'top_left',
            'boxed',
            'minimal',
            'professional'
        ]
        
        style = random.choice(header_styles)
        
        if style == 'top_right':
            text_lines.append(f"{'Page ' + str(page_num) + ' of ' + str(total_pages):>{width}}")
            text_lines.append("-" * width)
            
        elif style == 'centered':
            text_lines.append(f"{'═' * width}")
            text_lines.append(f"{'Page ' + str(page_num) + ' of ' + str(total_pages):^{width}}")
            text_lines.append(f"{'═' * width}")
            
        elif style == 'top_left':
            text_lines.append(f"Page {page_num} of {total_pages}")
            text_lines.append("-" * width)
            
        elif style == 'boxed':
            text_lines.append("┌" + "─" * (width - 2) + "┐")
            text_lines.append(f"│ {'Page ' + str(page_num) + '/' + str(total_pages):<{width-4}} │")
            text_lines.append("└" + "─" * (width - 2) + "┘")
            
        elif style == 'minimal':
            text_lines.append(f"[{page_num}/{total_pages}]")
            text_lines.append("")
            
        elif style == 'professional':
            invoice_num = receipt_data.get('invoice_number', 'N/A')
            text_lines.append(f"Invoice: {invoice_num:<30} Page {page_num} of {total_pages:>{width-45}}")
            text_lines.append("─" * width)
        
        return text_lines
    
    def _generate_page_footer(self, receipt_data: dict, width: int, page_num: int, total_pages: int, is_last_page: bool) -> list:
        """
        Generate page footer for multi-page documents
        
        Returns:
            List of text lines for page footer
        """
        import random
        text_lines = []
        
        if not is_last_page:
            # Continuation markers
            continuation_styles = [
                'arrow',
                'continued',
                'see_next',
                'ellipsis',
                'professional'
            ]
            
            style = random.choice(continuation_styles)
            
            if style == 'arrow':
                text_lines.append("")
                text_lines.append(f"{'▼ Continued on next page ▼':^{width}}")
                text_lines.append("═" * width)
                
            elif style == 'continued':
                text_lines.append("")
                text_lines.append(f"{'*** CONTINUED ***':^{width}}")
                
            elif style == 'see_next':
                text_lines.append("")
                text_lines.append(f"{'(See next page for more items)':^{width}}")
                text_lines.append("-" * width)
                
            elif style == 'ellipsis':
                text_lines.append("")
                text_lines.append(f"{'...':^{width}}")
                text_lines.append("")
                
            elif style == 'professional':
                text_lines.append("-" * width)
                text_lines.append(f"Continued on page {page_num + 1}")
                text_lines.append("-" * width)
        else:
            # Last page footer
            text_lines.append("")
            text_lines.append(f"{'─' * width}")
            text_lines.append(f"{'End of Document':^{width}}")
        
        return text_lines
    
    def _split_items_into_pages(self, line_items: list, items_per_page: int = 15) -> list:
        """
        Split line items into pages
        
        Args:
            line_items: All line items
            items_per_page: Maximum items per page (default: 15)
            
        Returns:
            List of lists, each representing items for one page
        """
        pages = []
        for i in range(0, len(line_items), items_per_page):
            pages.append(line_items[i:i + items_per_page])
        return pages
    
    def _render_multipage_receipt(self, receipt_data: dict, output_path: str) -> bool:
        """
        Render multi-page receipt for e-commerce/wholesale orders
        
        Args:
            receipt_data: Receipt data dictionary
            output_path: Path to save PNG file
            
        Returns:
            True if successful, False otherwise
        """
        import random
        from pathlib import Path
        
        # Decide ONCE whether to augment this entire multipage receipt
        # This ensures consistent augmentation across all pages
        should_augment_multipage = (self.augment_probability > 0 and 
                                   random.random() < self.augment_probability)
        
        # Generate a seed for consistent augmentation effects across all pages
        # Same seed = same wrinkles, stains, angles, etc. on every page
        augmentation_seed = random.randint(0, 999999) if should_augment_multipage else None
        
        # Multi-page receipts are typically more formal/professional
        style = random.choice(['detailed', 'professional', 'invoice', 'ecommerce'])
        width = random.choice([60, 70, 80])  # Wider for e-commerce
        
        divider_char = random.choice(['=', '-'])
        divider_full = divider_char * width
        divider_light = '-' * width
        
        # Choose line item table layout ONCE for entire multi-page document (consistency)
        line_item_layout = random.choice([
            'classic_table', 'complex_table', 'multi_line_description', 'attribute_columns',
            'nested_description', 'alternating_shaded', 'borderless_minimal', 'condensed_rows',
            'wide_table', 'narrow_table', 'sku_emphasis', 'price_breakdown', 'quantity_focus',
            'discount_inline', 'discount_separate', 'tabular_boxed', 'list_style',
            'receipt_tape_style', 'invoice_professional', 'retail_pos_style', 'ecommerce_order',
            'grouped_categories', 'two_column_items', 'description_first', 'prices_right_aligned'
        ])
        
        # Split line items into pages (10-20 items per page)
        items_per_page = random.randint(10, 20)
        line_items = receipt_data.get('line_items', [])
        item_pages = self._split_items_into_pages(line_items, items_per_page)
        total_pages = len(item_pages) if item_pages else 1
        
        # For multi-page, create separate files for each page
        # Modify output path to include page number
        output_file = Path(output_path)
        base_name = output_file.stem
        output_dir = output_file.parent
        
        all_pages_rendered = True
        
        for page_num in range(1, total_pages + 1):
            is_first_page = (page_num == 1)
            is_last_page = (page_num == total_pages)
            
            text_lines = []
            
            # Add spacing helper
            def add_spacing():
                if style in ['spacious', 'ecommerce']:
                    text_lines.append('')
                elif random.random() < 0.3:
                    text_lines.append('')
            
            # === PAGE HEADER ===
            if not is_first_page:
                page_header = self._generate_page_header(receipt_data, width, page_num, total_pages)
                text_lines.extend(page_header)
                add_spacing()
            
            # === DOCUMENT HEADER (First page only) ===
            if is_first_page:
                try:
                    header_lines = self._generate_header(receipt_data, width, style)
                    text_lines.extend(header_lines)
                except Exception as e:
                    print(f"Warning: Header generation failed: {e}")
                    store_name = receipt_data.get('supplier_name', 'Store')
                    text_lines.append(f"{store_name:^{width}}")
                
                add_spacing()
                
                # Supplier section (first page only)
                if random.random() < 0.5:
                    try:
                        supplier_lines = self._generate_supplier_section(receipt_data, width, style)
                        if supplier_lines:
                            text_lines.extend(supplier_lines)
                            add_spacing()
                    except Exception as e:
                        print(f"Warning: Supplier section failed: {e}")
                
                # Buyer section (first page only)
                if random.random() < 0.8:
                    try:
                        buyer_lines = self._generate_buyer_section(receipt_data, width, style)
                        if buyer_lines:
                            text_lines.extend(buyer_lines)
                            add_spacing()
                    except Exception as e:
                        print(f"Warning: Buyer section failed: {e}")
                
                # Order metadata (first page only)
                if random.random() < 0.9:
                    try:
                        metadata_lines = self._generate_order_metadata_section(receipt_data, width, style)
                        if metadata_lines:
                            text_lines.extend(metadata_lines)
                            add_spacing()
                    except Exception as e:
                        print(f"Warning: Order metadata section failed: {e}")
                
                text_lines.append(divider_full)
                add_spacing()
            
            # === LINE ITEMS FOR THIS PAGE ===
            if page_num <= len(item_pages):
                page_items = item_pages[page_num - 1]
                
                # Create temporary receipt data with only this page's items
                page_receipt_data = receipt_data.copy()
                page_receipt_data['line_items'] = page_items
                
                try:
                    item_lines = self._generate_line_items_table(page_receipt_data, width, style, divider_light, layout=line_item_layout)
                    if item_lines:
                        text_lines.extend(item_lines)
                except Exception as e:
                    print(f"Warning: Line items generation failed: {e}")
                    # Fallback
                    for item in page_items:
                        desc = item.get('description', 'Item')[:40]
                        total = item.get('total', '$0.00')
                        text_lines.append(f"{desc:<40} {total:>10}")
            
            add_spacing()
            
            # === TOTALS SECTION (Last page only) ===
            if is_last_page:
                add_spacing()
                
                # Use modularized totals section (20 variants)
                try:
                    totals_lines = self._generate_totals_section(receipt_data, width, style, divider_full, divider_light)
                    if totals_lines:
                        text_lines.extend(totals_lines)
                except Exception as e:
                    print(f"Warning: Totals section generation failed: {e}")
                    # Fallback to simple total
                    text_lines.append(divider_full)
                    total_amount = receipt_data.get('total_amount', '0')
                    text_lines.append(f"{'TOTAL:':20s} {str(total_amount):>12s}")
                    text_lines.append(divider_full)
                
                add_spacing()
                
                # Payment details (last page only)
                if 'payment_method' in receipt_data:
                    payment = receipt_data['payment_method']
                    text_lines.append(f"Payment Method: {payment}")
                    
                    if payment in ['Credit Card', 'Debit Card']:
                        if 'card_last_four' in receipt_data and receipt_data['card_last_four']:
                            text_lines.append(f"Card: ****{receipt_data['card_last_four']}")
                    
                    add_spacing()
                
                # Footer (last page only)
                # Footer section (last page only, 70% of invoices)
                if random.random() < 0.7:
                    try:
                        footer_lines = self._generate_footer_section(receipt_data, width, style)
                        if footer_lines:
                            text_lines.extend(footer_lines)
                    except Exception as e:
                        print(f"Warning: Footer generation failed: {e}")
                        # Fallback to simple thank you message
                        text_lines.append(f"{'Thank you for your order!':^{width}}")
                
                add_spacing()
                
                # Barcode section (last page only, 60% of invoices)
                if random.random() < 0.6:
                    try:
                        barcode_lines = self._generate_barcode_section(receipt_data, width, style)
                        if barcode_lines:
                            text_lines.extend(barcode_lines)
                    except Exception as e:
                        print(f"Warning: Barcode generation failed: {e}")
            
            # === PAGE FOOTER ===
            page_footer = self._generate_page_footer(receipt_data, width, page_num, total_pages, is_last_page)
            text_lines.extend(page_footer)
            
            # Render this page as invoice type (standard page size)
            # Always use page format in multipage branch (even if only 1 page)
            page_output_path = output_dir / f"{base_name}_page{page_num}.png"
            
            # Temporarily disable automatic augmentation for multipage rendering
            # We'll apply it manually after all pages are rendered
            original_aug_prob = self.augment_probability
            self.augment_probability = 0
            
            success = self.render_text_receipt(text_lines, str(page_output_path), receipt_type='invoice')
            
            # Restore original probability
            self.augment_probability = original_aug_prob
            
            # Apply augmentation with SAME SEED to this page if decided for the whole receipt
            # This ensures all pages have identical augmentation effects
            if success and should_augment_multipage:
                self._apply_augmentation(str(page_output_path), seed=augmentation_seed)
            if not success:
                all_pages_rendered = False
                print(f"Failed to render page {page_num} of {total_pages}")
        
        # For multi-page, create a marker file and clean up base file
        marker_path = output_dir / f"{base_name}_MULTIPAGE.txt"
        with open(marker_path, 'w') as f:
            f.write(f"This receipt has {total_pages} pages\n")
            for i in range(1, total_pages + 1):
                f.write(f"Page {i}: {base_name}_page{i}.png\n")
        
        # CRITICAL: Delete the base file if it exists
        # This prevents confusion in verification reports
        base_file = Path(output_path)
        if base_file.exists():
            print(f"Deleting base file for multi-page: {base_file.name}")
            base_file.unlink()
        
        return all_pages_rendered
    
    def render_receipt_dict(self, receipt_data: dict, output_path: str) -> bool:
        """
        Render receipt dictionary to PNG with realistic variety
        Supports both:
        - Retail/POS receipts: Single continuous roll (thermal paper, no height limit)
        - Online/Wholesale invoices: Standard page size, can be multi-page
        
        Args:
            receipt_data: Receipt data dictionary
            output_path: Path to save PNG file
            
        Returns:
            True if successful, False otherwise
        """
        import random
        
        # Detect receipt type
        receipt_type = self._detect_receipt_type(receipt_data)
        
        # Check if multi-page rendering is needed (only for invoice types)
        if self._should_use_multipage(receipt_data):
            return self._render_multipage_receipt(receipt_data, output_path)
        
        # Single-page rendering (existing retail receipt logic)
        # Convert receipt dict to text lines
        text_lines = []
        
        # Choose receipt style with more dramatic variations
        style = random.choice(['standard', 'detailed', 'minimal', 'modern', 'compact', 'spacious'])
        
        # Choose width variation (40-70 characters)
        width = random.choice([40, 48, 60, 70])
        
        # Choose divider style
        divider_char = random.choice(['=', '-', '*', '#'])
        divider_full = divider_char * width
        divider_light = '-' * width
        
        # Spacing variation - some receipts have extra blank lines
        def add_spacing():
            if style == 'spacious':
                text_lines.append('')
            elif style == 'compact':
                pass  # No extra lines
            elif random.random() < 0.4:
                text_lines.append('')
        
        # === MODULARIZED HEADER GENERATION (40+ archetypes) ===
        try:
            header_lines = self._generate_header(receipt_data, width, style)
            text_lines.extend(header_lines)
        except Exception as e:
            # Fallback to simple header if generation fails
            print(f"Warning: Header generation failed, using simple fallback: {e}")
            store_name = receipt_data.get('supplier_name', 'Store')
            text_lines.append(f"{store_name:^{width}}")
            if 'supplier_address' in receipt_data:
                text_lines.append(f"{receipt_data['supplier_address']:^{width}}")
        
        add_spacing()
        
        # === MODULARIZED SUPPLIER SECTION (12 variants) ===
        # Note: Some header types already include full supplier info, so conditionally add
        # Only add separate supplier section for minimal header types (30% of time)
        if random.random() < 0.3:
            try:
                supplier_lines = self._generate_supplier_section(receipt_data, width, style)
                if supplier_lines:
                    text_lines.extend(supplier_lines)
                    add_spacing()
            except Exception as e:
                print(f"Warning: Supplier section generation failed: {e}")
        
        # Transaction details - vary presence and formatting
        show_transaction_header = random.random() < 0.3
        if show_transaction_header:
            text_lines.append("--- Transaction Details ---")
        
        if 'register_number' in receipt_data and random.random() < 0.7:
            reg_format = random.choice([
                f"Register: {receipt_data['register_number']}",
                f"Reg #{receipt_data['register_number']}",
                f"Till: {receipt_data['register_number']}"
            ])
            text_lines.append(reg_format)
            
        if 'cashier_id' in receipt_data and random.random() < 0.6:
            cashier_format = random.choice([
                f"Cashier: {receipt_data['cashier_id']}",
                f"Served by: {receipt_data['cashier_id']}",
                f"Operator: {receipt_data['cashier_id']}"
            ])
            text_lines.append(cashier_format)
            
        if 'transaction_number' in receipt_data and random.random() < 0.8:
            trn_format = random.choice([
                f"Transaction: {receipt_data['transaction_number']}",
                f"Txn #: {receipt_data['transaction_number']}",
                f"Trans ID: {receipt_data['transaction_number']}"
            ])
            text_lines.append(trn_format)
        
        # Invoice info
        if 'invoice_number' in receipt_data:
            receipt_format = random.choice([
                f"Receipt #: {receipt_data['invoice_number']}",
                f"Receipt No: {receipt_data['invoice_number']}",
                f"Invoice: {receipt_data['invoice_number']}"
            ])
            text_lines.append(receipt_format)
            
        if 'invoice_date' in receipt_data:
            date_str = receipt_data['invoice_date']
            if 'transaction_time' in receipt_data and random.random() < 0.7:
                # Varied date/time formats
                date_format = random.choice([
                    f"Date: {date_str}  Time: {receipt_data['transaction_time']}",
                    f"{date_str} @ {receipt_data['transaction_time']}",
                    f"Date/Time: {date_str} {receipt_data['transaction_time']}"
                ])
                text_lines.append(date_format)
            else:
                text_lines.append(f"Date: {date_str}")
        
        # === MODULARIZED BUYER/CUSTOMER SECTION (10 variants) ===
        # Generate buyer section with varied formats (70% of receipts have some buyer info)
        if random.random() < 0.7:
            add_spacing()
            try:
                buyer_lines = self._generate_buyer_section(receipt_data, width, style)
                if buyer_lines:
                    text_lines.extend(buyer_lines)
            except Exception as e:
                print(f"Warning: Buyer section generation failed: {e}")
        
        # === MODULARIZED ORDER METADATA SECTION (20 variants) ===
        # Generate order/transaction metadata with varied formats (80% of receipts)
        # This can replace or supplement the basic transaction details above
        if random.random() < 0.8:
            add_spacing()
            try:
                metadata_lines = self._generate_order_metadata_section(receipt_data, width, style)
                if metadata_lines:
                    text_lines.extend(metadata_lines)
            except Exception as e:
                print(f"Warning: Order metadata section generation failed: {e}")
        
        add_spacing()
        text_lines.append(divider_full)
        
        # === MODULARIZED LINE ITEM TABLE LAYOUTS (25 variants) ===
        if 'line_items' in receipt_data:
            try:
                item_lines = self._generate_line_items_table(receipt_data, width, style, divider_light)
                if item_lines:
                    text_lines.extend(item_lines)
            except Exception as e:
                print(f"Warning: Line items generation failed: {e}")
                # Fallback to simple format
                for item in receipt_data['line_items']:
                    desc = item.get('description', 'Item')
                    total = item.get('total', '$0.00')
                    text_lines.append(f"{desc[:30]:30s} {total:>8s}")
        
        add_spacing()
        
        # === CHECK IF CONTENT IS TOO LONG FOR SINGLE PAGE ===
        # If we're building an invoice-type receipt and content is getting long, switch to multi-page
        if receipt_type == 'invoice':
            # Estimate lines needed for remaining sections (totals, payment, footer, barcode)
            estimated_remaining_lines = 30  # Conservative estimate
            total_estimated_lines = len(text_lines) + estimated_remaining_lines
            
            # Letter size page can fit approximately 50-55 lines comfortably (with 20px line spacing)
            # If we'll exceed this, force multi-page rendering
            if total_estimated_lines > 50:
                print(f"Content too long for single page ({total_estimated_lines} lines estimated), switching to multi-page")
                return self._render_multipage_receipt(receipt_data, output_path)
        
        # === MODULARIZED TOTALS SECTION (20 variants) ===
        try:
            totals_lines = self._generate_totals_section(receipt_data, width, style, divider_full, divider_light)
            if totals_lines:
                text_lines.extend(totals_lines)
        except Exception as e:
            print(f"Warning: Totals section generation failed: {e}")
            # Fallback to simple format
            text_lines.append(divider_full)
            total_amount = receipt_data.get('total_amount', '0')
            text_lines.append(f"{'TOTAL:':20s} {str(total_amount):>12s}")
            text_lines.append(divider_full)
        
        add_spacing()
        
        # Payment details
        if 'payment_method' in receipt_data:
            payment = receipt_data['payment_method']
            payment_header = random.choice([
                f"Payment Method: {payment}",
                f"Paid by: {payment}",
                f"Payment: {payment}"
            ])
            text_lines.append(payment_header)
            
            # Card details (if card payment)
            if payment in ['Credit Card', 'Debit Card']:
                if 'card_type' in receipt_data and receipt_data['card_type']:
                    text_lines.append(f"Card Type: {receipt_data['card_type']}")
                if 'card_last_four' in receipt_data and receipt_data['card_last_four']:
                    card_format = random.choice([
                        f"Card: ****{receipt_data['card_last_four']}",
                        f"XXXX-XXXX-XXXX-{receipt_data['card_last_four']}",
                        f"Card ending in {receipt_data['card_last_four']}"
                    ])
                    text_lines.append(card_format)
                if 'approval_code' in receipt_data and receipt_data['approval_code']:
                    text_lines.append(f"Approval: {receipt_data['approval_code']}")
            
            # Cash payment details
            elif payment == 'Cash':
                if 'cash_tendered' in receipt_data and receipt_data['cash_tendered']:
                    text_lines.append(f"Cash Tendered: {receipt_data['cash_tendered']}")
                if 'change_amount' in receipt_data and receipt_data['change_amount']:
                    text_lines.append(f"Change: {receipt_data['change_amount']}")
        
        add_spacing()
        
        # === MODULARIZED FOOTER SECTION (30 variants) ===
        # 80% of receipts have footer messages
        if random.random() < 0.8:
            try:
                footer_lines = self._generate_footer_section(receipt_data, width, style)
                if footer_lines:
                    text_lines.extend(footer_lines)
                    add_spacing()
            except Exception as e:
                print(f"Warning: Footer generation failed: {e}")
                # Fallback to simple thank you message
                text_lines.append(f"{'Thank you for your purchase!':^{width}}")
                add_spacing()
        
        # === MODULARIZED BARCODE SECTION (15 variants) ===
        # 60% of receipts have barcodes (increased from 50%)
        if random.random() < 0.6:
            try:
                barcode_lines = self._generate_barcode_section(receipt_data, width, style)
                if barcode_lines:
                    text_lines.extend(barcode_lines)
            except Exception as e:
                print(f"Warning: Barcode generation failed: {e}")
                # Fallback to simple bars
                if 'transaction_number' in receipt_data or 'invoice_number' in receipt_data:
                    num = receipt_data.get('transaction_number') or receipt_data.get('invoice_number')
                    text_lines.append('|' * width)
                    text_lines.append(f"{num:^{width}}")
                    text_lines.append('|' * width)
        
        # Render to PNG with appropriate receipt type
        return self.render_text_receipt(text_lines, output_path, receipt_type=receipt_type)


# Factory function to get appropriate renderer
def get_renderer(prefer_wkhtmltoimage: bool = True, **kwargs):
    """
    Get appropriate renderer based on availability
    
    Args:
        prefer_wkhtmltoimage: Try to use wkhtmltoimage first (default: True)
        **kwargs: Arguments passed to renderer constructor
        
    Returns:
        HTMLToPNGRenderer or SimplePNGRenderer instance
    """
    if prefer_wkhtmltoimage:
        try:
            return HTMLToPNGRenderer(**kwargs)
        except RuntimeError as e:
            print(f"Warning: {str(e)}")
            print("Falling back to SimplePNGRenderer")
            return SimplePNGRenderer(**kwargs)
    else:
        return SimplePNGRenderer(**kwargs)


__all__ = ['HTMLToPNGRenderer', 'SimplePNGRenderer', 'get_renderer']
