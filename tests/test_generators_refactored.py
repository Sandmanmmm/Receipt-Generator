"""
Test Refactored Generators
"""
import pytest
from pathlib import Path
import tempfile

from generators import (
    SyntheticDataGenerator,
    InvoiceData,
    TemplateRenderer,
    PDFRenderer,
    ImageRenderer,
    InvoiceRandomizer
)


class TestSyntheticDataGenerator:
    """Test data generation"""
    
    def test_generate_invoice(self):
        generator = SyntheticDataGenerator(seed=42)
        invoice = generator.generate_invoice()
        
        assert isinstance(invoice, InvoiceData)
        assert invoice.company_name
        assert invoice.invoice_number
        assert len(invoice.items) > 0
        assert invoice.total > 0
    
    def test_generate_with_constraints(self):
        generator = SyntheticDataGenerator(seed=42)
        invoice = generator.generate_invoice(
            min_items=5,
            max_items=10,
            include_tax=True,
            include_discount=True
        )
        
        assert 5 <= len(invoice.items) <= 10
        assert invoice.tax >= 0
    
    def test_invoice_to_dict(self):
        generator = SyntheticDataGenerator(seed=42)
        invoice = generator.generate_invoice()
        invoice_dict = generator.invoice_to_dict(invoice)
        
        assert isinstance(invoice_dict, dict)
        assert 'company_name' in invoice_dict
        assert 'items' in invoice_dict
        assert isinstance(invoice_dict['items'], list)


class TestTemplateRenderer:
    """Test template rendering"""
    
    @pytest.mark.skipif(not Path("templates/modern/invoice.html").exists(),
                       reason="Template not found")
    def test_render_template(self):
        renderer = TemplateRenderer('templates')
        
        data = {
            'company_name': 'Test Corp',
            'invoice_number': 'INV-001',
            'items': []
        }
        
        html = renderer.render('modern/invoice.html', data)
        
        assert 'Test Corp' in html
        assert 'INV-001' in html
    
    @pytest.mark.skipif(not Path("templates/modern/invoice.html").exists(),
                       reason="Template not found")
    def test_render_to_file(self):
        renderer = TemplateRenderer('templates')
        
        data = {'company_name': 'Test', 'items': []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name
        
        try:
            html = renderer.render_to_file('modern/invoice.html', data, temp_path)
            assert Path(temp_path).exists()
            assert 'Test' in html
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestPDFRenderer:
    """Test PDF rendering"""
    
    @pytest.mark.skipif(True, reason="Requires WeasyPrint/wkhtmltopdf")
    def test_render_from_html_string(self):
        renderer = PDFRenderer(backend='weasyprint')
        
        html = "<html><body><h1>Test Invoice</h1></body></html>"
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            temp_path = f.name
        
        try:
            success = renderer.render_from_html_string(html, temp_path)
            assert success
            assert Path(temp_path).exists()
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestImageRenderer:
    """Test image rendering"""
    
    @pytest.mark.skipif(True, reason="Requires pdf2image")
    def test_pdf_to_image(self):
        renderer = ImageRenderer(dpi=150)
        
        # Assuming we have a test PDF
        pdf_path = "tests/fixtures/test.pdf"
        
        if Path(pdf_path).exists():
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = f.name
            
            try:
                success = renderer.pdf_to_image(pdf_path, temp_path)
                assert success
                assert Path(temp_path).exists()
            finally:
                Path(temp_path).unlink(missing_ok=True)


class TestInvoiceRandomizer:
    """Test randomizer utilities"""
    
    def test_random_currency(self):
        randomizer = InvoiceRandomizer(seed=42)
        currency = randomizer.random_currency(['USD', 'EUR'])
        
        assert 'symbol' in currency
        assert 'code' in currency
        assert currency['code'] in ['USD', 'EUR']
    
    def test_random_tax_rate(self):
        randomizer = InvoiceRandomizer(seed=42)
        tax_rate = randomizer.random_tax_rate(region='US')
        
        assert tax_rate >= 0
        assert tax_rate <= 20
    
    def test_random_invoice_number(self):
        randomizer = InvoiceRandomizer(seed=42)
        inv_num = randomizer.random_invoice_number(prefix='INV', year=True)
        
        assert 'INV' in inv_num
        assert '2024' in inv_num or '2025' in inv_num
    
    def test_random_price(self):
        randomizer = InvoiceRandomizer(seed=42)
        price = randomizer.random_price(min_price=10.0, max_price=100.0)
        
        assert 10.0 <= price <= 100.0
        assert isinstance(price, float)


class TestEndToEndGeneration:
    """Test complete generation pipeline"""
    
    @pytest.mark.skipif(not Path("templates/modern/invoice.html").exists(),
                       reason="Template not found")
    def test_full_pipeline(self):
        # Generate data
        data_gen = SyntheticDataGenerator(seed=42)
        invoice = data_gen.generate_invoice()
        invoice_dict = data_gen.invoice_to_dict(invoice)
        
        # Render template
        template_renderer = TemplateRenderer('templates')
        html = template_renderer.render('modern/invoice.html', invoice_dict)
        
        assert html
        assert invoice.company_name in html
        assert invoice.invoice_number in html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
