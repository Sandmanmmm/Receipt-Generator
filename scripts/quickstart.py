"""
Quick Start Script
Generate a small sample dataset and test the pipeline
"""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.synthetic_data import SyntheticDataGenerator
from generators.renderer import InvoiceRenderer
import click


@click.command()
@click.option('--samples', '-n', default=5, help='Number of samples')
def quickstart(samples):
    """Generate a few sample invoices to test the pipeline"""
    
    click.echo("="*60)
    click.echo("INVOICEGEN - QUICK START")
    click.echo("="*60)
    
    # Create output directory
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\nGenerating {samples} sample invoices...")
    
    # Initialize
    generator = SyntheticDataGenerator(locale='en_US', seed=42)
    renderer = InvoiceRenderer(
        templates_dir='templates/html',
        output_dir='data/raw'
    )
    
    # Generate samples
    for i in range(samples):
        click.echo(f"  Generating invoice {i+1}/{samples}...")
        
        # Generate data
        invoice = generator.generate_invoice(min_items=2, max_items=5)
        invoice_dict = generator.invoice_to_dict(invoice)
        
        # Render (HTML only for quick test)
        try:
            results = renderer.render_invoice(
                template_name='modern_invoice.html',
                data=invoice_dict,
                invoice_id=invoice.invoice_number,
                formats=['html'],  # HTML only for speed
                pdf_backend='weasyprint'
            )
            
            click.echo(f"    ✓ {invoice.invoice_number}")
        except Exception as e:
            click.echo(f"    ✗ Error: {e}", err=True)
    
    click.echo(f"\n✓ Generated {samples} invoices in data/raw/")
    click.echo("\nNext steps:")
    click.echo("  1. View generated invoices in data/raw/")
    click.echo("  2. Install PDF dependencies: pip install weasyprint")
    click.echo("  3. Run full pipeline: python scripts/pipeline.py pipeline -n 100")
    click.echo("  4. See README.md for more details")


if __name__ == '__main__':
    quickstart()
