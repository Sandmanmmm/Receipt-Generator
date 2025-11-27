"""
Build Training Set - End-to-end dataset generation pipeline
"""
import click
from pathlib import Path
import yaml
from tqdm import tqdm

from generators import SyntheticDataGenerator, TemplateRenderer, PDFRenderer, ImageRenderer
from annotation import OCREngine, BBoxExtractor, LabelMapper, AnnotationWriter
from augmentation.augmenter import Augmenter
from training.dataset_builder import DatasetBuilder


@click.command()
@click.option('--num-samples', '-n', default=1000, help='Number of invoices to generate')
@click.option('--output-dir', '-o', default='data', help='Output directory')
@click.option('--config', '-c', default='config/config.yaml', help='Config file')
@click.option('--templates', '-t', multiple=True, 
              default=['modern/invoice.html', 'classic/invoice.html', 'receipt/invoice.html'],
              help='Template names to use')
@click.option('--augment/--no-augment', default=True, help='Apply augmentation')
@click.option('--split/--no-split', default=True, help='Split into train/val/test')
@click.option('--seed', default=42, help='Random seed')
@click.option('--ocr-engine', default='paddleocr', 
              type=click.Choice(['paddleocr', 'tesseract', 'easyocr']),
              help='OCR engine')
def main(num_samples, output_dir, config, templates, augment, split, seed, ocr_engine):
    """Build complete training dataset from scratch"""
    
    click.echo("="*80)
    click.echo("INVOICEGEN TRAINING SET BUILDER")
    click.echo("="*80)
    
    # Load config
    config_path = Path(config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}
    
    # Setup directories
    output_path = Path(output_dir)
    raw_dir = output_path / 'raw'
    annotated_dir = output_path / 'annotated'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"\nConfiguration:")
    click.echo(f"  Samples: {num_samples}")
    click.echo(f"  Templates: {', '.join(templates)}")
    click.echo(f"  Augmentation: {augment}")
    click.echo(f"  Split dataset: {split}")
    click.echo(f"  OCR Engine: {ocr_engine}")
    click.echo(f"  Seed: {seed}")
    
    # Step 1: Generate synthetic invoices
    click.echo("\n" + "="*80)
    click.echo("[1/5] GENERATING SYNTHETIC INVOICES")
    click.echo("="*80)
    
    data_generator = SyntheticDataGenerator(locale='en_US', seed=seed)
    template_renderer = TemplateRenderer('templates')
    pdf_renderer = PDFRenderer(backend='weasyprint')
    image_renderer = ImageRenderer(dpi=150)
    
    samples_per_template = num_samples // len(templates)
    
    for template_name in templates:
        click.echo(f"\nGenerating {samples_per_template} invoices with {template_name}...")
        
        for i in tqdm(range(samples_per_template), desc=f"  {template_name}"):
            # Generate data
            invoice = data_generator.generate_invoice()
            invoice_dict = data_generator.invoice_to_dict(invoice)
            invoice_id = f"{Path(template_name).stem}_{i:04d}"
            
            # Render to HTML
            html = template_renderer.render(template_name, invoice_dict)
            
            # Convert to PDF
            pdf_path = raw_dir / f"{invoice_id}.pdf"
            pdf_renderer.render_from_html_string(
                html, str(pdf_path), base_url=f'templates/{Path(template_name).parent}'
            )
            
            # Convert to PNG
            png_path = raw_dir / f"{invoice_id}.png"
            image_renderer.pdf_to_image(str(pdf_path), str(png_path))
            
            # Save invoice data
            import json
            data_path = raw_dir / f"{invoice_id}.json"
            with open(data_path, 'w') as f:
                json.dump(invoice_dict, f, indent=2)
    
    click.echo(f"\n✓ Generated {num_samples} invoices in {raw_dir}")
    
    # Step 2: Apply augmentation
    if augment:
        click.echo("\n" + "="*80)
        click.echo("[2/5] APPLYING AUGMENTATION")
        click.echo("="*80)
        
        aug_config_path = Path('augmentation/settings.yaml')
        if aug_config_path.exists():
            augmenter = Augmenter(str(aug_config_path))
        else:
            click.echo("Warning: augmentation/settings.yaml not found, using defaults")
            augmenter = Augmenter()
        
        image_files = list(raw_dir.glob('*.png'))
        for img_path in tqdm(image_files, desc="  Augmenting"):
            augmenter.augment_image(str(img_path), str(img_path))
        
        click.echo(f"✓ Augmented {len(image_files)} images")
    else:
        click.echo("\n[2/5] SKIPPING AUGMENTATION")
    
    # Step 3: Auto-annotation
    click.echo("\n" + "="*80)
    click.echo("[3/5] AUTO-ANNOTATION WITH OCR")
    click.echo("="*80)
    
    # Load labels
    labels_path = Path('config/labels.yaml')
    if labels_path.exists():
        with open(labels_path, 'r') as f:
            labels_cfg = yaml.safe_load(f)
            label_list = labels_cfg.get('labels', [])
    else:
        label_list = ['O']  # Default
    
    ocr = OCREngine(engine=ocr_engine)
    extractor = BBoxExtractor(ocr)
    labeler = LabelMapper(label_list)
    writer = AnnotationWriter()
    
    image_files = list(raw_dir.glob('*.png'))
    
    for img_path in tqdm(image_files, desc="  Annotating"):
        try:
            # Extract text and boxes
            bboxes = extractor.extract(str(img_path))
            
            if not bboxes:
                continue
            
            # Map labels
            tokens = [bbox.text for bbox in bboxes]
            boxes = [[bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height] 
                    for bbox in bboxes]
            
            labels = labeler.map_labels(tokens)
            
            # Get image dimensions
            from PIL import Image
            img = Image.open(img_path)
            width, height = img.size
            
            # Create annotation
            annotation = {
                'image_path': str(img_path),
                'tokens': tokens,
                'labels': labels,
                'bboxes': boxes,
                'image_width': width,
                'image_height': height
            }
            
            # Save to JSONL
            output_file = annotated_dir / f"{img_path.stem}.jsonl"
            writer.write_jsonl([annotation], str(output_file))
        
        except Exception as e:
            click.echo(f"Error annotating {img_path.name}: {e}")
    
    click.echo(f"✓ Annotated {len(list(annotated_dir.glob('*.jsonl')))} images")
    
    # Step 4: Split dataset
    if split:
        click.echo("\n" + "="*80)
        click.echo("[4/5] SPLITTING DATASET")
        click.echo("="*80)
        
        builder = DatasetBuilder(seed=seed)
        builder.build_dataset(
            annotations_dir=str(annotated_dir),
            output_dir=str(output_path),
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            copy_images=True
        )
    else:
        click.echo("\n[4/5] SKIPPING DATASET SPLIT")
    
    # Step 5: Validation
    click.echo("\n" + "="*80)
    click.echo("[5/5] VALIDATING DATASET")
    click.echo("="*80)
    
    if split:
        builder = DatasetBuilder(seed=seed)
        report = builder.validate_dataset(str(output_path))
        
        if report['valid']:
            click.echo("✓ Dataset validation passed")
            click.echo(f"  Train: {report['statistics'].get('train', 0)} samples")
            click.echo(f"  Val: {report['statistics'].get('val', 0)} samples")
            click.echo(f"  Test: {report['statistics'].get('test', 0)} samples")
        else:
            click.echo("✗ Dataset validation failed:")
            for error in report['errors']:
                click.echo(f"  - {error}")
        
        if report['warnings']:
            click.echo("Warnings:")
            for warning in report['warnings']:
                click.echo(f"  - {warning}")
    
    # Summary
    click.echo("\n" + "="*80)
    click.echo("DATASET BUILD COMPLETE")
    click.echo("="*80)
    click.echo(f"Output directory: {output_path}")
    click.echo(f"Total samples: {num_samples}")
    click.echo("\nNext steps:")
    click.echo("  1. Review annotations: python scripts/visualize_annotations.py")
    click.echo("  2. Start training: python scripts/run_training.py")


if __name__ == '__main__':
    main()
