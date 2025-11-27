"""Test suite for InvoiceGen"""
import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.synthetic_data import SyntheticDataGenerator, InvoiceData
from generators.renderer import InvoiceRenderer
from annotation.annotator import OCRAnnotator, EntityLabeler
from augmentation.augmenter import ImageAugmenter, AugmentationConfig


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation"""
    
    def test_generate_invoice(self):
        """Test invoice generation"""
        generator = SyntheticDataGenerator(seed=42)
        invoice = generator.generate_invoice(min_items=3, max_items=5)
        
        assert isinstance(invoice, InvoiceData)
        assert invoice.company_name
        assert invoice.client_name
        assert len(invoice.items) >= 3
        assert len(invoice.items) <= 5
        assert invoice.total > 0
    
    def test_invoice_to_dict(self):
        """Test invoice serialization"""
        generator = SyntheticDataGenerator(seed=42)
        invoice = generator.generate_invoice()
        invoice_dict = generator.invoice_to_dict(invoice)
        
        assert isinstance(invoice_dict, dict)
        assert 'company_name' in invoice_dict
        assert 'items' in invoice_dict
        assert isinstance(invoice_dict['items'], list)
    
    def test_reproducibility(self):
        """Test that same seed produces same results"""
        gen1 = SyntheticDataGenerator(seed=42)
        gen2 = SyntheticDataGenerator(seed=42)
        
        inv1 = gen1.generate_invoice()
        inv2 = gen2.generate_invoice()
        
        assert inv1.company_name == inv2.company_name
        assert inv1.invoice_number == inv2.invoice_number


class TestImageAugmenter:
    """Tests for image augmentation"""
    
    def test_augmentation_config(self):
        """Test augmentation configuration"""
        config = AugmentationConfig(
            add_noise=True,
            noise_probability=0.8
        )
        
        assert config.add_noise is True
        assert config.noise_probability == 0.8
    
    def test_augmenter_initialization(self):
        """Test augmenter initialization"""
        config = AugmentationConfig()
        augmenter = ImageAugmenter(config)
        
        assert augmenter.config == config


class TestAnnotation:
    """Tests for annotation system"""
    
    def test_entity_labeler_initialization(self):
        """Test entity labeler setup"""
        labeler = EntityLabeler()
        
        assert hasattr(labeler, 'entity_patterns')
        assert 'invoice_number' in labeler.entity_patterns
        assert 'date' in labeler.entity_patterns


def test_project_structure():
    """Test that key directories exist"""
    root = Path(__file__).parent.parent
    
    required_dirs = [
        'templates/html',
        'templates/css',
        'generators',
        'annotation',
        'augmentation',
        'training',
        'evaluation',
        'deployment',
        'config',
        'scripts'
    ]
    
    for dir_path in required_dirs:
        assert (root / dir_path).exists(), f"Missing directory: {dir_path}"


def test_config_exists():
    """Test that configuration file exists"""
    root = Path(__file__).parent.parent
    config_path = root / 'config' / 'config.yaml'
    
    assert config_path.exists(), "config.yaml not found"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
