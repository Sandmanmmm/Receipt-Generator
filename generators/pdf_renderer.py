"""
PDF Renderer - HTML to PDF Conversion
"""
import subprocess
from pathlib import Path
from typing import Optional


class PDFRenderer:
    """Converts HTML to PDF using multiple backends"""
    
    def __init__(self, backend: str = 'wkhtmltopdf'):
        """
        Initialize PDF renderer
        
        Args:
            backend: 'wkhtmltopdf' or 'weasyprint'
        """
        self.backend = backend
    
    def render_from_html_file(self, html_path: str, pdf_path: str, **options) -> bool:
        """
        Convert HTML file to PDF
        
        Args:
            html_path: Path to HTML file
            pdf_path: Output PDF path
            **options: Backend-specific options
            
        Returns:
            True if successful, False otherwise
        """
        if self.backend == 'wkhtmltopdf':
            return self._render_wkhtmltopdf_file(html_path, pdf_path, **options)
        elif self.backend == 'weasyprint':
            # Read HTML content and use string method
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            base_url = Path(html_path).parent
            return self._render_weasyprint_string(html_content, pdf_path, str(base_url))
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def render_from_html_string(self, html_content: str, pdf_path: str, 
                               base_url: Optional[str] = None) -> bool:
        """
        Convert HTML string to PDF
        
        Args:
            html_content: HTML string
            pdf_path: Output PDF path
            base_url: Base URL for resolving relative paths
            
        Returns:
            True if successful, False otherwise
        """
        if self.backend == 'weasyprint':
            return self._render_weasyprint_string(html_content, pdf_path, base_url)
        elif self.backend == 'wkhtmltopdf':
            # wkhtmltopdf requires a file, so create temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', 
                                            delete=False, encoding='utf-8') as f:
                f.write(html_content)
                temp_path = f.name
            
            try:
                return self._render_wkhtmltopdf_file(temp_path, pdf_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _render_wkhtmltopdf_file(self, html_path: str, pdf_path: str,
                                 page_size: str = 'A4',
                                 margin_top: str = '10mm',
                                 margin_right: str = '10mm',
                                 margin_bottom: str = '10mm',
                                 margin_left: str = '10mm',
                                 **kwargs) -> bool:
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
            **kwargs: Additional wkhtmltopdf options
            
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
            ]
            
            # Add any additional options
            for key, value in kwargs.items():
                cmd.extend([f'--{key.replace("_", "-")}', str(value)])
            
            cmd.extend([html_path, pdf_path])
            
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
            print("wkhtmltopdf not found. Install from: https://wkhtmltopdf.org/")
            return False
    
    def _render_weasyprint_string(self, html_content: str, pdf_path: str, 
                                  base_url: Optional[str] = None) -> bool:
        """
        Convert HTML string to PDF using WeasyPrint
        
        Args:
            html_content: HTML string
            pdf_path: Output PDF path
            base_url: Base URL for resolving relative paths
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from weasyprint import HTML
            
            output_file = Path(pdf_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Render PDF
            HTML(string=html_content, base_url=base_url).write_pdf(pdf_path)
            
            return output_file.exists()
        
        except ImportError:
            print("WeasyPrint not installed. Install with: pip install weasyprint")
            return False
        except Exception as e:
            print(f"WeasyPrint error: {e}")
            return False


__all__ = ['PDFRenderer']
