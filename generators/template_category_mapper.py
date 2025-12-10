"""
Template Category Mapper
Maps invoice templates to appropriate product categories for realistic data generation
"""
import random
from typing import List, Tuple
import yaml
from pathlib import Path


class TemplateCategoryMapper:
    """Maps templates to product categories based on business type"""
    
    def __init__(self, config_path: str = 'config/template_categories.yaml'):
        """Load template category mappings from config file"""
        config_file = Path(config_path)
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.templates = config.get('templates', {})
                self.category_info = config.get('category_info', {})
        else:
            # Fallback to default mappings if config not found
            self.templates = {}
            self.category_info = {}
    
    def get_categories_for_template(self, template_name: str) -> List[str]:
        """
        Get weighted category list for a template
        
        Args:
            template_name: Name of template (without .html extension)
            
        Returns:
            List of categories to use for item generation
        """
        # Remove .html extension and path if present
        clean_name = template_name.replace('.html', '').split('/')[-1]
        
        # Get template config
        template_config = self.templates.get(clean_name)
        
        if not template_config:
            # Default to fashion if template not found
            return ['fashion'], [1.0]
        
        primary = template_config.get('primary', 'fashion')
        secondary = template_config.get('secondary', [])
        weights = template_config.get('weights', [1.0])
        
        # Build category list with weights
        categories = [primary] + secondary
        
        # Normalize weights if needed
        if len(weights) != len(categories):
            weights = [0.6] + [0.4 / len(secondary)] * len(secondary)
        
        return categories, weights
    
    def generate_mixed_categories(self, template_name: str, num_items: int) -> List[str]:
        """
        Generate a list of categories for invoice items with realistic mixing
        
        Args:
            template_name: Name of template
            num_items: Number of items to generate categories for
            
        Returns:
            List of category names (length = num_items)
        """
        categories, weights = self.get_categories_for_template(template_name)
        
        # Use weighted random selection
        selected_categories = random.choices(
            categories,
            weights=weights,
            k=num_items
        )
        
        return selected_categories
    
    def get_primary_category(self, template_name: str) -> str:
        """Get the primary category for a template"""
        categories, _ = self.get_categories_for_template(template_name)
        return categories[0] if categories else 'fashion'
    
    def get_template_info(self, template_name: str) -> dict:
        """Get full configuration for a template"""
        clean_name = template_name.replace('.html', '').split('/')[-1]
        return self.templates.get(clean_name, {
            'primary': 'fashion',
            'secondary': [],
            'weights': [1.0]
        })


# Global instance for easy import
_mapper_instance = None

def get_category_mapper() -> TemplateCategoryMapper:
    """Get singleton instance of category mapper"""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = TemplateCategoryMapper()
    return _mapper_instance
