"""Invoice Generators Package"""

# New modular API
from .data_generator import SyntheticDataGenerator, InvoiceData, InvoiceItem
from .template_renderer import TemplateRenderer
from .pdf_renderer import PDFRenderer
from .image_renderer import ImageRenderer
from .randomizers import Randomizer, InvoiceRandomizer

# Legacy imports (backward compatibility)
from .synthetic_data import SyntheticDataGenerator as LegacySyntheticDataGenerator
from .renderer import InvoiceRenderer, BatchRenderer

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
