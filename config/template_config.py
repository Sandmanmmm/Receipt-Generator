"""
Template Configuration Manager

Centralizes template-specific parameters for pagination, rendering, and layout.
This ensures consistent behavior between test scripts and production dataset generation.
"""

import yaml
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class TemplatePaginationConfig:
    """Configuration for how a template handles multipage rendering"""
    template_name: str
    supports_multipage: bool = False
    renderer: str = 'html'  # 'html' or 'simple'
    
    # Page size configuration
    page_size: str = 'Letter'  # Letter, A4, Legal, Tabloid, Custom
    orientation: str = 'Portrait'  # Portrait or Landscape
    custom_width: int = 0  # For Custom page size (pixels)
    custom_height: int = 0  # For Custom page size (pixels)
    
    # Pagination parameters
    first_page_items: int = 10
    continuation_items_per_page: int = 10
    last_page_items: int = 10
    
    # Section heights (in pixels at 96 DPI) - measured from actual templates
    # These are used to calculate if items fit on a page
    header_height: int = 300        # Company info, logo, invoice header
    item_row_height: int = 50       # Single line item height
    totals_height: int = 200        # Subtotal, tax, total section
    footer_height: int = 150        # Payment terms, notes, thank you
    continuation_header_height: int = 150  # Simplified header for pages 2+
    continuation_footer_height: int = 100  # Simplified footer for pages 2+
    margin_top: int = 40            # Top margin/padding
    margin_bottom: int = 40         # Bottom margin/padding
    
    def get_page_height_px(self) -> int:
        """Get available page height in pixels for current page size and orientation"""
        # Handle custom page sizes first
        if self.page_size == 'Custom' and self.custom_height > 0:
            return self.custom_height
            
        # Standard page sizes at 96 DPI (pixels)
        page_dimensions = {
            'Letter': {'Portrait': (816, 1056), 'Landscape': (1056, 816)},
            'A4': {'Portrait': (794, 1123), 'Landscape': (1123, 794)},
            'Legal': {'Portrait': (816, 1344), 'Landscape': (1344, 816)},
            'Tabloid': {'Portrait': (1056, 1632), 'Landscape': (1632, 1056)},
        }
        return page_dimensions.get(self.page_size, {}).get(self.orientation, (816, 1056))[1]
    
    def calculate_first_page_capacity(self) -> int:
        """Calculate how many items can fit on the first page"""
        page_height = self.get_page_height_px()
        
        # First page sections
        used_height = (
            self.margin_top +
            self.header_height +
            self.totals_height +
            self.footer_height +
            self.margin_bottom
        )
        
        available_for_items = page_height - used_height
        max_items = max(1, available_for_items // self.item_row_height)
        
        return min(max_items, self.first_page_items)  # Cap at configured limit
    
    def calculate_continuation_page_capacity(self, is_last_page: bool = False) -> int:
        """Calculate how many items can fit on a continuation page"""
        page_height = self.get_page_height_px()
        
        # Continuation page sections
        used_height = (
            self.margin_top +
            self.continuation_header_height +
            self.margin_bottom
        )
        
        # Add totals and full footer on last page
        if is_last_page:
            used_height += self.totals_height + self.footer_height
        else:
            used_height += self.continuation_footer_height
        
        available_for_items = page_height - used_height
        max_items = max(1, available_for_items // self.item_row_height)
        
        # Use configured limits
        limit = self.last_page_items if is_last_page else self.continuation_items_per_page
        return min(max_items, limit)
    
    def calculate_pages_needed(self, num_items: int) -> int:
        """
        Calculate the number of pages needed for given number of items
        based on actual page size and section heights.
        
        Args:
            num_items: Total number of line items
            
        Returns:
            Number of pages required
        """
        if num_items == 0:
            return 1
            
        if not self.supports_multipage:
            return 1
        
        # Calculate actual capacity of first page
        first_page_capacity = self.calculate_first_page_capacity()
        
        if num_items <= first_page_capacity:
            return 1
        
        remaining_items = num_items - first_page_capacity
        
        # Calculate how many continuation pages we need
        continuation_capacity = self.calculate_continuation_page_capacity(is_last_page=False)
        last_page_capacity = self.calculate_continuation_page_capacity(is_last_page=True)
        
        # Try to fit with: (n-1) middle pages + 1 last page
        for additional_pages in range(1, 100):  # Sanity check
            if additional_pages == 1:
                can_fit = remaining_items <= last_page_capacity
            else:
                middle_pages = additional_pages - 1
                can_fit = remaining_items <= (middle_pages * continuation_capacity + last_page_capacity)
            
            if can_fit:
                return 1 + additional_pages
        
        # Fallback
        return 1 + ((remaining_items + continuation_capacity - 1) // continuation_capacity)
    
    def get_items_for_page(self, page_num: int, all_items: list) -> list:
        """
        Get the items that should appear on a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            all_items: Complete list of all items
            
        Returns:
            List of items for this page
        """
        num_items = len(all_items)
        total_pages = self.calculate_pages_needed(num_items)
        is_last_page = (page_num == total_pages - 1)
        
        if page_num == 0:
            # First page
            first_page_capacity = self.calculate_first_page_capacity()
            return all_items[:first_page_capacity]
        
        # Calculate start index for continuation pages
        first_page_capacity = self.calculate_first_page_capacity()
        continuation_capacity = self.calculate_continuation_page_capacity(is_last_page=False)
        
        start_idx = first_page_capacity + ((page_num - 1) * continuation_capacity)
        
        # Calculate end index
        if is_last_page:
            # Last page: take remaining items up to last_page_items limit
            last_page_capacity = self.calculate_continuation_page_capacity(is_last_page=True)
            end_idx = start_idx + last_page_capacity
        else:
            # Middle continuation page: full continuation_items_per_page
            end_idx = start_idx + continuation_capacity
        
        return all_items[start_idx:end_idx]
    
    def calculate_actual_page_height(self, page_num: int, total_pages: int, items_on_page: int) -> int:
        """
        Calculate the actual content height for a specific page.
        This is used to verify the page will fit within the page size.
        
        Args:
            page_num: Page number (0-indexed)
            total_pages: Total number of pages
            items_on_page: Number of items on this page
            
        Returns:
            Actual content height in pixels
        """
        is_first_page = (page_num == 0)
        is_last_page = (page_num == total_pages - 1)
        
        # Calculate all section heights
        height = self.margin_top
        
        if is_first_page:
            height += self.header_height
        else:
            height += self.continuation_header_height
        
        # Items
        height += items_on_page * self.item_row_height
        
        # Totals section (only on last page)
        if is_last_page:
            height += self.totals_height
            height += self.footer_height
        else:
            height += self.continuation_footer_height
        
        height += self.margin_bottom
        
        return height
    
    def get_page_dimensions(self) -> tuple:
        """Get (width, height) in pixels for current page size and orientation"""
        # Handle custom page sizes first
        if self.page_size == 'Custom' and self.custom_width > 0 and self.custom_height > 0:
            return (self.custom_width, self.custom_height)
            
        page_dimensions = {
            'Letter': {'Portrait': (816, 1056), 'Landscape': (1056, 816)},
            'A4': {'Portrait': (794, 1123), 'Landscape': (1123, 794)},
            'Legal': {'Portrait': (816, 1344), 'Landscape': (1344, 816)},
            'Tabloid': {'Portrait': (1056, 1632), 'Landscape': (1632, 1056)},
        }
        return page_dimensions.get(self.page_size, {}).get(self.orientation, (816, 1056))


class TemplateConfigManager:
    """Manages template configuration from YAML file"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the template config manager.
        
        Args:
            config_path: Path to template_pagination.yaml. 
                        If None, uses default location in config/
        """
        if config_path is None:
            config_path = Path(__file__).parent / 'template_pagination.yaml'
        
        self.config_path = config_path
        self._configs: Dict[str, TemplatePaginationConfig] = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load template configurations from YAML file"""
        if not self.config_path.exists():
            print(f"Warning: Template config not found at {self.config_path}")
            return
        
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        
        for template_name, params in data.items():
            if params is None:
                params = {}
            
            self._configs[template_name] = TemplatePaginationConfig(
                template_name=template_name,
                supports_multipage=params.get('supports_multipage', False),
                renderer=params.get('renderer', 'html'),
                page_size=params.get('page_size', 'Letter'),
                orientation=params.get('orientation', 'Portrait'),
                custom_width=params.get('custom_width', 0),
                custom_height=params.get('custom_height', 0),
                first_page_items=params.get('first_page_items', 10),
                continuation_items_per_page=params.get('continuation_items_per_page', 10),
                last_page_items=params.get('last_page_items', 10),
                header_height=params.get('header_height', 300),
                item_row_height=params.get('item_row_height', 50),
                totals_height=params.get('totals_height', 200),
                footer_height=params.get('footer_height', 150),
                continuation_header_height=params.get('continuation_header_height', 150),
                continuation_footer_height=params.get('continuation_footer_height', 100),
                margin_top=params.get('margin_top', 40),
                margin_bottom=params.get('margin_bottom', 40),
            )
    
    def get_config(self, template_name: str) -> TemplatePaginationConfig:
        """
        Get configuration for a template.
        
        Args:
            template_name: Template filename (e.g., 'retail/online_order_electronics.html')
            
        Returns:
            TemplatePaginationConfig for the template, or default config if not found
        """
        # Normalize template name
        template_key = template_name.replace('\\', '/')
        
        if template_key in self._configs:
            return self._configs[template_key]
        
        # Return default config
        return TemplatePaginationConfig(
            template_name=template_name,
            supports_multipage=False,
        )
    
    def list_templates(self) -> list:
        """Get list of all configured templates"""
        return list(self._configs.keys())


# Global instance for easy access
_manager_instance = None

def get_template_config_manager() -> TemplateConfigManager:
    """Get global TemplateConfigManager instance (singleton)"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = TemplateConfigManager()
    return _manager_instance


if __name__ == '__main__':
    # Test the configuration loader
    manager = get_template_config_manager()
    
    print(f"Loaded {len(manager.list_templates())} template configurations\n")
    
    # Test a few templates
    test_templates = [
        'retail/online_order_electronics.html',
        'retail/online_order_digital.html',
        'modern_professional/invoice_minimal_multipage.html',
    ]
    
    for template in test_templates:
        config = manager.get_config(template)
        print(f"\n{template}:")
        print(f"  Supports multipage: {config.supports_multipage}")
        print(f"  Page size: {config.page_size} ({config.orientation})")
        print(f"  First page capacity: {config.calculate_first_page_capacity()} items")
        print(f"  Continuation page capacity: {config.calculate_continuation_page_capacity()} items")
        print(f"  Item row height: {config.item_row_height}px")
        
        # Test pagination calculation
        for num_items in [6, 12, 18]:
            pages = config.calculate_pages_needed(num_items)
            print(f"  {num_items} items → {pages} pages")
