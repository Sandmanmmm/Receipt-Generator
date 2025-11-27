"""
Annotation Package - Modular annotation system
"""
from .annotation_schema import BoundingBox, InvoiceAnnotation
from .ocr_engine import OCREngine
from .bbox_extractor import BBoxExtractor
from .label_mapper import LabelMapper
from .annotation_writer import AnnotationWriter

# Legacy imports for backward compatibility
from .annotator import OCRAnnotator, EntityLabeler, AnnotationVisualizer

__all__ = [
    # New modular API
    'BoundingBox',
    'InvoiceAnnotation',
    'OCREngine',
    'BBoxExtractor',
    'LabelMapper',
    'AnnotationWriter',
    # Legacy API (deprecated)
    'OCRAnnotator',
    'EntityLabeler',
    'AnnotationVisualizer',
]
