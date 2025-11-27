"""Invoice Generators Package"""

# Lazy imports to avoid dependency issues - import when needed
# from .data_generator import SyntheticDataGenerator, InvoiceData, InvoiceItem
# from .template_renderer import TemplateRenderer
# from .pdf_renderer import PDFRenderer
# from .image_renderer import ImageRenderer
# from .randomizers import Randomizer, InvoiceRandomizer
# from .synthetic_data import SyntheticDataGenerator as LegacySyntheticDataGenerator
# from .renderer import InvoiceRenderer, BatchRenderer
# from .retail_data_generator import RetailDataGenerator, RetailReceiptData, RetailLineItem

__all__ = [
    # New modular API
    'SyntheticDataGenerator',
    'InvoiceData',
    'InvoiceItem',
    'TemplateRenderer',
    'PDFRenderer',
    'ImageRenderer',
    'Randomizer',
    'InvoiceRandomizer',
    # Legacy API (backward compatibility)
    'LegacySyntheticDataGenerator',
    'InvoiceRenderer',
    'BatchRenderer',
]
