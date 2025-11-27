"""
Invoice Renderer
Renders Jinja2 templates to HTML, PDF, and PNG
"""
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader, Template
from PIL import Image
import base64
from io import BytesIO


class InvoiceRenderer:
    """Renders invoice templates to various formats"""
    
    def __init__(self, templates_dir: str, output_dir: str = "data/raw"):
        """
        Initialize renderer
        
        Args:
            templates_dir: Path to directory containing HTML templates
            output_dir: Directory for output files
        """
        self.templates_dir = Path(templates_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
    
    def render_html(self,
                   template_name: str,
                   data: Dict[str, Any],
                   output_path: Optional[str] = None) -> str:
        """
        Render template to HTML
        
        Args:
            template_name: Name of template file (e.g., 'modern_invoice.html')
            data: Dictionary with template variables
            output_path: Optional path to save HTML file
            
        Returns:
            Rendered HTML string
        """
        template = self.env.get_template(template_name)
        html_content = template.render(**data)
        
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
                      png_dpi: int = 150) -> Dict[str, str]:
        """
        Render invoice to multiple formats
        
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
    # Example usage
    from generators.synthetic_data import SyntheticDataGenerator
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(seed=42)
    invoice = generator.generate_invoice()
    invoice_dict = generator.invoice_to_dict(invoice)
    
    # Render invoice
    renderer = InvoiceRenderer(
        templates_dir='templates/html',
        output_dir='data/raw'
    )
    
    results = renderer.render_invoice(
        template_name='modern_invoice.html',
        data=invoice_dict,
        invoice_id=invoice.invoice_number,
        formats=['html', 'pdf', 'png'],
        pdf_backend='weasyprint'  # or 'wkhtmltopdf'
    )
    
    print("Generated files:")
    for format_type, path in results.items():
        print(f"  {format_type}: {path}")
