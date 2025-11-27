"""
Bounding Box Extractor - Extract boxes from annotated images
"""
from typing import List, Optional, Dict, Any
from .annotation_schema import BoundingBox, InvoiceAnnotation
from .ocr_engine import OCREngine
from PIL import Image


class BBoxExtractor:
    """Extract bounding boxes from images using OCR"""
    
    def __init__(self, ocr_engine: str = 'paddleocr', **ocr_kwargs):
        """
        Initialize bbox extractor
        
        Args:
            ocr_engine: OCR engine type
            **ocr_kwargs: Additional OCR engine parameters
        """
        self.ocr = OCREngine(engine=ocr_engine, **ocr_kwargs)
    
    def extract(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> InvoiceAnnotation:
        """
        Extract bounding boxes from image
        
        Args:
            image_path: Path to image
            metadata: Optional metadata
            
        Returns:
            InvoiceAnnotation with extracted boxes
        """
        # Get image dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Extract boxes using OCR
        boxes = self.ocr.extract_text(image_path)
        
        return InvoiceAnnotation(
            image_path=image_path,
            image_width=width,
            image_height=height,
            boxes=boxes,
            metadata=metadata
        )
    
    def extract_batch(self, image_paths: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[InvoiceAnnotation]:
        """
        Extract bounding boxes from multiple images
        
        Args:
            image_paths: List of image paths
            metadata_list: Optional list of metadata dicts
            
        Returns:
            List of InvoiceAnnotation objects
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)
        
        annotations = []
        for image_path, metadata in zip(image_paths, metadata_list):
            annotation = self.extract(image_path, metadata)
            annotations.append(annotation)
        
        return annotations


__all__ = ['BBoxExtractor']
