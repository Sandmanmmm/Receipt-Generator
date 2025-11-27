"""
Test Configuration Files
"""
import pytest
from pathlib import Path
import yaml


class TestConfigFiles:
    """Test configuration files"""
    
    def test_main_config_exists(self):
        """Test main config file exists"""
        assert Path("config/config.yaml").exists()
    
    def test_labels_config_exists(self):
        """Test labels config exists"""
        assert Path("config/labels.yaml").exists()
    
    def test_training_config_exists(self):
        """Test training config exists"""
        assert Path("config/training_config.yaml").exists()
    
    def test_augmentation_config_exists(self):
        """Test augmentation config exists"""
        assert Path("augmentation/settings.yaml").exists()
    
    def test_load_main_config(self):
        """Test loading main config"""
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
    
    def test_load_labels_config(self):
        """Test loading labels config"""
        with open("config/labels.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
        if 'labels' in config:
            assert isinstance(config['labels'], list)
    
    def test_load_training_config(self):
        """Test loading training config"""
        with open("config/training_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
        
        # Check expected keys
        expected_keys = ['model', 'training']
        for key in expected_keys:
            assert key in config
    
    def test_load_augmentation_config(self):
        """Test loading augmentation config"""
        with open("augmentation/settings.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict)
        
        # Check expected sections
        expected_sections = ['enabled', 'geometric', 'color', 'noise']
        for section in expected_sections:
            assert section in config


class TestDataDirectoryStructure:
    """Test data directory structure"""
    
    def test_data_dir_exists(self):
        """Test data directory exists"""
        assert Path("data").exists()
    
    def test_required_subdirs(self):
        """Test required subdirectories exist"""
        required_dirs = [
            "data/raw",
            "data/processed",
            "data/annotated",
            "data/annotations",
            "data/train",
            "data/val",
            "data/test"
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"{dir_path} should exist"


class TestTemplateStructure:
    """Test template structure"""
    
    def test_template_dirs(self):
        """Test template directories exist"""
        template_dirs = [
            "templates/modern",
            "templates/classic",
            "templates/receipt"
        ]
        
        for dir_path in template_dirs:
            assert Path(dir_path).exists(), f"{dir_path} should exist"
    
    def test_template_files(self):
        """Test template files exist"""
        for template_type in ['modern', 'classic', 'receipt']:
            template_dir = Path(f"templates/{template_type}")
            
            assert (template_dir / "invoice.html").exists()
            assert (template_dir / "styles.css").exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
