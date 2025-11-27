"""
End-to-End Invoice Generation Pipeline
Orchestrates the complete process from generation to training
"""
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
import click
from tqdm import tqdm

from generators.synthetic_data import SyntheticDataGenerator
from generators.renderer import InvoiceRenderer, BatchRenderer
from annotation.annotator import OCRAnnotator, EntityLabeler, AnnotationVisualizer
from augmentation.augmenter import ImageAugmenter, BatchAugmenter, AugmentationConfig
from training.data_converter import LayoutLMv3Converter, DatasetBuilder


class InvoicePipeline:
    """Complete pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline with configuration"""
        self.config = self.load_config(config_path)
        self.setup_directories()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs = self.config['directories']
        for key, path in dirs.items():
            if isinstance(path, dict):
                for subkey, subpath in path.items():
                    Path(subpath).mkdir(parents=True, exist_ok=True)
            else:
                Path(path).mkdir(parents=True, exist_ok=True)
    
    def step1_generate_invoices(self, num_samples: int = 100, seed: Optional[int] = None):
        """Step 1: Generate synthetic invoices"""
        click.echo(f"\n{'='*60}")
        click.echo("STEP 1: Generating Synthetic Invoices")
        click.echo(f"{'='*60}")
        
        gen_config = self.config['generation']
        locales = gen_config['locales']
        templates = gen_config['templates']
        
        # Initialize generator and renderer
        renderer = InvoiceRenderer(
            templates_dir='templates/html',
            output_dir=self.config['directories']['data']['raw']
        )
        
        invoices = []
        
        click.echo(f"Generating {num_samples} invoices...")
        for i in tqdm(range(num_samples)):
            # Rotate through locales
            locale = locales[i % len(locales)]
            template = templates[i % len(templates)]
            
            # Generate data
            generator = SyntheticDataGenerator(locale=locale, seed=seed)
            invoice = generator.generate_invoice(
                min_items=gen_config['min_items'],
                max_items=gen_config['max_items']
            )
            invoice_dict = generator.invoice_to_dict(invoice)
            
            invoices.append((invoice.invoice_number, invoice_dict, template))
        
        # Render batch
        click.echo("Rendering to PDF/PNG...")
        batch_renderer = BatchRenderer(renderer)
        
        for invoice_id, data, template in tqdm(invoices):
            renderer.render_invoice(
                template_name=template,
                data=data,
                invoice_id=invoice_id,
                formats=gen_config['rendering']['formats'],
                pdf_backend=gen_config['rendering']['pdf_backend'],
                png_dpi=gen_config['rendering']['png_dpi']
            )
        
        click.echo(f"✓ Generated {num_samples} invoices")
        return invoices
    
    def step2_annotate(self):
        """Step 2: Auto-annotate with OCR"""
        click.echo(f"\n{'='*60}")
        click.echo("STEP 2: Auto-Annotating Invoices")
        click.echo(f"{'='*60}")
        
        ocr_config = self.config['ocr']
        
        # Initialize annotator
        annotator = OCRAnnotator(ocr_engine=ocr_config['engine'])
        labeler = EntityLabeler()
        
        # Get all PNG files
        raw_dir = Path(self.config['directories']['data']['raw'])
        png_files = list(raw_dir.glob('*.png'))
        
        click.echo(f"Annotating {len(png_files)} images...")
        for png_file in tqdm(png_files):
            try:
                # Extract text and boxes
                annotation = annotator.annotate_image(str(png_file))
                
                # Label entities
                annotation = labeler.label_boxes(annotation)
                
                # Save annotation
                json_path = Path(self.config['directories']['data']['annotations']) / f"{png_file.stem}.json"
                annotation.save_json(str(json_path))
                
            except Exception as e:
                click.echo(f"Error processing {png_file.name}: {e}", err=True)
        
        click.echo(f"✓ Annotated {len(png_files)} images")
    
    def step3_augment(self):
        """Step 3: Apply augmentation"""
        click.echo(f"\n{'='*60}")
        click.echo("STEP 3: Augmenting Images")
        click.echo(f"{'='*60}")
        
        aug_config = self.config['augmentation']
        
        if not aug_config['enabled']:
            click.echo("Augmentation disabled, skipping...")
            return
        
        # Create augmentation config
        config = AugmentationConfig(
            add_noise=aug_config['noise']['enabled'],
            noise_probability=aug_config['noise']['probability'],
            add_blur=aug_config['blur']['enabled'],
            blur_probability=aug_config['blur']['probability'],
            add_compression=aug_config['compression']['enabled'],
            compression_probability=aug_config['compression']['probability'],
            add_rotation=aug_config['rotation']['enabled'],
            rotation_probability=aug_config['rotation']['probability']
        )
        
        augmenter = ImageAugmenter(config)
        batch_augmenter = BatchAugmenter(augmenter)
        
        # Get all PNG files
        raw_dir = Path(self.config['directories']['data']['raw'])
        png_files = [str(f) for f in raw_dir.glob('*.png')]
        
        click.echo(f"Augmenting {len(png_files)} images ({aug_config['copies_per_image']} copies each)...")
        batch_augmenter.augment_batch(
            input_paths=png_files,
            output_dir=self.config['directories']['data']['processed'],
            copies_per_image=aug_config['copies_per_image'],
            callback=lambda i, t, f: None  # Silent
        )
        
        click.echo(f"✓ Created {len(png_files) * aug_config['copies_per_image']} augmented images")
    
    def step4_convert_to_layoutlmv3(self):
        """Step 4: Convert to LayoutLMv3 format"""
        click.echo(f"\n{'='*60}")
        click.echo("STEP 4: Converting to LayoutLMv3 Format")
        click.echo(f"{'='*60}")
        
        train_config = self.config['training']
        
        # Initialize converter
        converter = LayoutLMv3Converter(
            model_name=train_config['model_name'],
            label_list=self.config['entity_labels']
        )
        
        builder = DatasetBuilder(converter)
        
        # Build dataset splits
        builder.build_dataset(
            annotation_dir=self.config['directories']['data']['annotations'],
            output_dir=self.config['directories']['data']['layoutlmv3'],
            train_ratio=train_config['train_ratio'],
            val_ratio=train_config['val_ratio'],
            test_ratio=train_config['test_ratio']
        )
        
        click.echo("✓ Dataset converted to LayoutLMv3 format")
    
    def step5_train(self):
        """Step 5: Train LayoutLMv3 model"""
        click.echo(f"\n{'='*60}")
        click.echo("STEP 5: Training LayoutLMv3 Model")
        click.echo(f"{'='*60}")
        
        from training.train import InvoiceTrainer, TrainingConfig
        import torch
        
        train_config = self.config['training']
        
        # Create training config
        config = TrainingConfig(
            model_name=train_config['model_name'],
            data_dir=self.config['directories']['data']['layoutlmv3'],
            output_dir=train_config['output_dir'],
            num_train_epochs=train_config['num_epochs'],
            per_device_train_batch_size=train_config['batch_size'],
            learning_rate=train_config['learning_rate'],
            weight_decay=train_config['weight_decay'],
            warmup_ratio=train_config['warmup_ratio'],
            fp16=train_config['use_fp16'] and torch.cuda.is_available()
        )
        
        # Train
        trainer = InvoiceTrainer(config)
        model, metrics = trainer.train()
        
        click.echo("\n✓ Training complete!")
        click.echo(f"Model saved to: {train_config['output_dir']}")
    
    def run_full_pipeline(self, num_samples: int = 100, seed: Optional[int] = None):
        """Run complete end-to-end pipeline"""
        click.echo(f"\n{'#'*60}")
        click.echo("# INVOICEGEN - FULL PIPELINE")
        click.echo(f"{'#'*60}")
        
        try:
            self.step1_generate_invoices(num_samples, seed)
            self.step2_annotate()
            self.step3_augment()
            self.step4_convert_to_layoutlmv3()
            self.step5_train()
            
            click.echo(f"\n{'#'*60}")
            click.echo("# PIPELINE COMPLETE!")
            click.echo(f"{'#'*60}")
            
        except Exception as e:
            click.echo(f"\n❌ Pipeline failed: {e}", err=True)
            raise


@click.group()
def cli():
    """InvoiceGen - Synthetic Invoice Generator with LayoutLMv3 Training"""
    pass


@cli.command()
@click.option('--num-samples', '-n', default=100, help='Number of invoices to generate')
@click.option('--seed', '-s', default=None, type=int, help='Random seed')
def generate(num_samples, seed):
    """Generate synthetic invoices"""
    pipeline = InvoicePipeline()
    pipeline.step1_generate_invoices(num_samples, seed)


@cli.command()
def annotate():
    """Auto-annotate invoices with OCR"""
    pipeline = InvoicePipeline()
    pipeline.step2_annotate()


@cli.command()
def augment():
    """Apply image augmentation"""
    pipeline = InvoicePipeline()
    pipeline.step3_augment()


@cli.command()
def convert():
    """Convert to LayoutLMv3 format"""
    pipeline = InvoicePipeline()
    pipeline.step4_convert_to_layoutlmv3()


@cli.command()
def train():
    """Train LayoutLMv3 model"""
    pipeline = InvoicePipeline()
    pipeline.step5_train()


@cli.command()
@click.option('--num-samples', '-n', default=100, help='Number of invoices to generate')
@click.option('--seed', '-s', default=None, type=int, help='Random seed')
def pipeline(num_samples, seed):
    """Run complete end-to-end pipeline"""
    pipe = InvoicePipeline()
    pipe.run_full_pipeline(num_samples, seed)


if __name__ == '__main__':
    cli()
