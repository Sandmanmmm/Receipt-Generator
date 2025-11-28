"""
OCR Engine - Multi-backend OCR support
"""
from typing import List, Optional
from .annotation_schema import BoundingBox
from PIL import Image


class OCREngine:
    """Base OCR engine interface"""
    
    def __init__(self, engine: str = 'paddleocr', **kwargs):
        """
        Initialize OCR engine
        
        Args:
            engine: Engine type ('paddleocr', 'tesseract', 'easyocr')
            **kwargs: Engine-specific parameters
        """
        self.engine_type = engine
        self.engine = None
        self.kwargs = kwargs
        
        if engine == 'paddleocr':
            self._init_paddleocr()
        elif engine == 'tesseract':
            self._init_tesseract()
        elif engine == 'easyocr':
            self._init_easyocr()
        else:
            raise ValueError(f"Unknown OCR engine: {engine}")
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            # Note: use_angle_cls is deprecated, use use_textline_orientation
            self.engine = PaddleOCR(
                use_textline_orientation=self.kwargs.get('use_textline_orientation', True),
                lang=self.kwargs.get('lang', 'en')
            )
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def _init_tesseract(self):
        """Initialize Tesseract"""
        try:
            import pytesseract
            self.engine = pytesseract
            # Check if tesseract is installed
            pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")
        except Exception as e:
            raise RuntimeError(f"Tesseract not properly configured: {e}")
    
    def _init_easyocr(self):
        """Initialize EasyOCR"""
        try:
            import easyocr
            langs = self.kwargs.get('languages', ['en'])
            gpu = self.kwargs.get('gpu', True)
            self.engine = easyocr.Reader(langs, gpu=gpu)
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def extract_text(self, image_path: str) -> List[BoundingBox]:
        """
        Extract text and bounding boxes from image
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of BoundingBox objects
        """
        if self.engine_type == 'paddleocr':
            return self._extract_paddleocr(image_path)
        elif self.engine_type == 'tesseract':
            return self._extract_tesseract(image_path)
        elif self.engine_type == 'easyocr':
            return self._extract_easyocr(image_path)
        else:
            return []
    
    def _extract_paddleocr(self, image_path: str) -> List[BoundingBox]:
        """Extract with PaddleOCR (updated for v5+ API)"""
        # Use predict() instead of deprecated ocr()
        result = self.engine.predict(image_path)
        boxes = []
        
        if not result:
            return boxes
        
        # Result is a list of OCRResult objects (dict-like)
        ocr_result = result[0]
        
        # Get recognized texts, scores, and polygons
        rec_texts = ocr_result.get('rec_texts', [])
        rec_scores = ocr_result.get('rec_scores', [])
        rec_polys = ocr_result.get('rec_polys', [])
        
        if not rec_texts:
            return boxes
        
        for i, (text, confidence, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
            if not text:
                continue
                
            # Convert polygon coords to bbox format
            # poly is numpy array of shape (N, 2) with points
            x_coords = [p[0] for p in poly]
            y_coords = [p[1] for p in poly]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            boxes.append(BoundingBox(
                text=text,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=float(confidence)
            ))
        
        return boxes
    
    def _extract_tesseract(self, image_path: str) -> List[BoundingBox]:
        """Extract with Tesseract"""
        import pytesseract
        
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            conf = int(data['conf'][i])
            if conf > 0:  # Filter out low confidence
                text = data['text'][i].strip()
                if text:
                    boxes.append(BoundingBox(
                        text=text,
                        x=int(data['left'][i]),
                        y=int(data['top'][i]),
                        width=int(data['width'][i]),
                        height=int(data['height'][i]),
                        confidence=conf / 100.0
                    ))
        
        return boxes
    
    def _extract_easyocr(self, image_path: str) -> List[BoundingBox]:
        """Extract with EasyOCR"""
        result = self.engine.readtext(image_path)
        boxes = []
        
        for detection in result:
            coords = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = detection[1]
            confidence = detection[2]
            
            # Convert coords to bbox format
            x_coords = [p[0] for p in coords]
            y_coords = [p[1] for p in coords]
            
            x = int(min(x_coords))
            y = int(min(y_coords))
            width = int(max(x_coords) - min(x_coords))
            height = int(max(y_coords) - min(y_coords))
            
            boxes.append(BoundingBox(
                text=text,
                x=x,
                y=y,
                width=width,
                height=height,
                confidence=float(confidence)
            ))
        
        return boxes


__all__ = ['OCREngine']
