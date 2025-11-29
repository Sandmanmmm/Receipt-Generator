"""
Auto-annotation System
Extracts bounding boxes from rendered invoices using OCR
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import cv2
import numpy as np
from PIL import Image


@dataclass
class BoundingBox:
    """Represents a bounding box with text"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: Optional[str] = None  # Entity label (e.g., 'company_name', 'total', etc.)
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_coco_format(self) -> List[int]:
        """Convert to COCO format [x, y, width, height]"""
        return [self.x, self.y, self.width, self.height]
    
    def to_pascal_voc(self) -> List[int]:
        """Convert to Pascal VOC format [xmin, ymin, xmax, ymax]"""
        return [self.x, self.y, self.x2, self.y2]


@dataclass
class InvoiceAnnotation:
    """Complete annotation for an invoice"""
    image_path: str
    image_width: int
    image_height: int
    boxes: List[BoundingBox]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'image_path': self.image_path,
            'image_width': self.image_width,
            'image_height': self.image_height,
            'boxes': [box.to_dict() for box in self.boxes],
            'metadata': self.metadata
        }
    
    def save_json(self, output_path: str):
        """Save annotation to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InvoiceAnnotation':
        """Load from dictionary"""
        boxes = [BoundingBox(**box) for box in data['boxes']]
        return cls(
            image_path=data['image_path'],
            image_width=data['image_width'],
            image_height=data['image_height'],
            boxes=boxes,
            metadata=data.get('metadata')
        )
    
    @classmethod
    def load_json(cls, json_path: str) -> 'InvoiceAnnotation':
        """Load annotation from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


class OCRAnnotator:
    """Extracts text and bounding boxes using OCR"""
    
    def __init__(self, ocr_engine: str = 'paddleocr'):
        """
        Initialize OCR annotator
        
        Args:
            ocr_engine: OCR engine to use ('paddleocr', 'tesseract', 'easyocr')
        """
        self.ocr_engine = ocr_engine
        self.ocr = None
        
        if ocr_engine == 'paddleocr':
            self._init_paddleocr()
        elif ocr_engine == 'tesseract':
            self._init_tesseract()
        elif ocr_engine == 'easyocr':
            self._init_easyocr()
        else:
            raise ValueError(f"Unknown OCR engine: {ocr_engine}")
    
    def _init_paddleocr(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                show_log=False
            )
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")
    
    def _init_tesseract(self):
        """Initialize Tesseract"""
        try:
            import pytesseract
            self.ocr = pytesseract
        except ImportError:
            raise ImportError("pytesseract not installed. Install with: pip install pytesseract")
    
    def _init_easyocr(self):
        """Initialize EasyOCR"""
        try:
            import easyocr
            self.ocr = easyocr.Reader(['en'])
        except ImportError:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
    
    def extract_boxes_paddleocr(self, image_path: str) -> List[BoundingBox]:
        """Extract bounding boxes using PaddleOCR"""
        result = self.ocr.ocr(image_path, cls=True)
        boxes = []
        
        for line in result[0]:
            coords = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = line[1][1]
            
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
                confidence=confidence
            ))
        
        return boxes
    
    def extract_boxes_tesseract(self, image_path: str) -> List[BoundingBox]:
        """Extract bounding boxes using Tesseract"""
        import pytesseract
        from PIL import Image
        
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Filter out low confidence
                text = data['text'][i].strip()
                if text:
                    boxes.append(BoundingBox(
                        text=text,
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=data['conf'][i] / 100.0
                    ))
        
        return boxes
    
    def extract_boxes_easyocr(self, image_path: str) -> List[BoundingBox]:
        """Extract bounding boxes using EasyOCR"""
        result = self.ocr.readtext(image_path)
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
                confidence=confidence
            ))
        
        return boxes
    
    def annotate_image(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> InvoiceAnnotation:
        """
        Extract annotations from an image
        
        Args:
            image_path: Path to image file
            metadata: Optional metadata to include
            
        Returns:
            InvoiceAnnotation object
        """
        # Get image dimensions
        image = Image.open(image_path)
        width, height = image.size
        
        # Extract boxes based on engine
        if self.ocr_engine == 'paddleocr':
            boxes = self.extract_boxes_paddleocr(image_path)
        elif self.ocr_engine == 'tesseract':
            boxes = self.extract_boxes_tesseract(image_path)
        elif self.ocr_engine == 'easyocr':
            boxes = self.extract_boxes_easyocr(image_path)
        else:
            boxes = []
        
        return InvoiceAnnotation(
            image_path=image_path,
            image_width=width,
            image_height=height,
            boxes=boxes,
            metadata=metadata
        )


class EntityLabeler:
    """Labels bounding boxes with entity types using retail schema"""
    
    def __init__(self, label_schema_path: Optional[str] = None):
        """
        Initialize entity labeler with retail-specific rules
        
        Args:
            label_schema_path: Optional path to load label schema
        """
        if label_schema_path:
            self._load_from_schema(label_schema_path)
        else:
            self._init_retail_patterns()
    
    def _init_retail_patterns(self):
        """Initialize retail-specific entity patterns"""
        self.entity_patterns = {
            # Document metadata
            'INVOICE_NUMBER': [
                r'(?:Receipt|Invoice|Order)\s*[#:No.]+\s*[A-Z0-9-]+',
                r'Receipt\s*[#:]\s*\d+',
                r'Transaction\s*[#:]\s*\d+'
            ],
            'INVOICE_DATE': [
                r'Date:\s*\d{2}/\d{2}/\d{4}',
                r'Date:\s*\d{2}\.\d{2}\.\d{4}',  # German format with dots
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{2}\.\d{2}\.\d{4}'  # DD.MM.YYYY format
            ],
            # Merchant info
            'SUPPLIER_NAME': [],  # Position-based
            'SUPPLIER_PHONE': [
                r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
            ],
            'SUPPLIER_EMAIL': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            # Financial totals
            'TOTAL_AMOUNT': [
                r'(?:Total|TOTAL)[:=\s]*[$£€¥]\s*[\d,]+\.?\d*',
                r'Grand\s+Total[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'SUBTOTAL': [
                r'Subtotal[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'TAX_AMOUNT': [
                r'Tax[:=\s]*[$£€¥]\s*[\d,]+\.?\d*'
            ],
            'DISCOUNT': [
                r'(?:Discount|Savings)[:=\s]*-?[$£€¥]\s*[\d,]+\.?\d*'
            ],
            # Payment info
            'PAYMENT_METHOD': [
                r'(?:Visa|Mastercard|Amex|Discover|Cash|Debit|Credit)',
                r'Payment\s+Method:\s*[A-Za-z\s]+'
            ],
            'PAYMENT_TERMS': [
                r'ending\s+in\s+\d{4}',
                r'Approval\s+Code:\s*[A-Z0-9]+',
                r'Auth\s*[#:]?\s*[A-Z0-9]+'
            ],
            # Retail identifiers
            'REGISTER_NUMBER': [
                r'Register[:=\s]*\d+'
            ],
            'CASHIER_ID': [
                r'Cashier[:=\s]*[A-Z0-9]+'
            ],
            # Line items
            'ITEM_QTY': [
                r'\d+\s*@\s*[$£€¥]'
            ],
            'ITEM_SKU': [
                r'(?:UPC|SKU)[:=\s]*[A-Z0-9-]+'
            ]
        }
    
    def _load_from_schema(self, schema_path: str):
        """Load patterns from label schema YAML"""
        import yaml
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = yaml.safe_load(f)
        
        # Extract entity names from label_list
        self.entity_patterns = {}
        for label in schema.get('label_list', []):
            if label.startswith('B-'):
                entity = label[2:]  # Remove B- prefix
                self.entity_patterns[entity] = []  # Will use default patterns
    
    def label_boxes(self, annotation: InvoiceAnnotation) -> InvoiceAnnotation:
        """
        Apply entity labels to bounding boxes
        
        Args:
            annotation: InvoiceAnnotation to label
            
        Returns:
            InvoiceAnnotation with labeled boxes
        """
        import re
        
        for box in annotation.boxes:
            text = box.text.strip()
            
            # Check patterns for each entity type
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    if pattern and re.search(pattern, text, re.IGNORECASE):
                        box.label = entity_type
                        break
                if box.label:
                    break
            
            # Position-based heuristics for SUPPLIER_NAME
            if not box.label:
                # Top 15% of document likely contains supplier name
                if box.y < annotation.image_height * 0.15:
                    if not any(b.label == 'SUPPLIER_NAME' for b in annotation.boxes):
                        # Check if text looks like a business name (not a number or single word)
                        if len(text.split()) >= 2 and not text.replace('.', '').isdigit():
                            box.label = 'SUPPLIER_NAME'
        
        return annotation


class AnnotationVisualizer:
    """Visualize annotations on images"""
    
    @staticmethod
    def draw_boxes(image_path: str,
                   annotation: InvoiceAnnotation,
                   output_path: str,
                   show_labels: bool = True,
                   show_text: bool = False):
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to input image
            annotation: InvoiceAnnotation with boxes
            output_path: Path to save annotated image
            show_labels: Whether to show entity labels
            show_text: Whether to show extracted text
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Colors for retail entity types (BGR format for OpenCV)
        colors = {
            'INVOICE_NUMBER': (255, 0, 0),      # Blue
            'INVOICE_DATE': (0, 255, 0),        # Green
            'TOTAL_AMOUNT': (0, 0, 255),        # Red
            'SUPPLIER_NAME': (0, 165, 255),     # Orange
            'SUPPLIER_ADDRESS': (255, 255, 0),  # Cyan
            'SUPPLIER_PHONE': (255, 0, 255),    # Magenta
            'SUPPLIER_EMAIL': (0, 255, 255),    # Yellow
            'PAYMENT_METHOD': (147, 20, 255),   # Deep Pink
            'PAYMENT_TERMS': (230, 216, 173),   # Light Blue
            'SUBTOTAL': (0, 128, 255),          # Light Orange
            'TAX_AMOUNT': (128, 0, 128),        # Purple
            'DISCOUNT': (0, 255, 0),            # Lime
            'ITEM_DESCRIPTION': (203, 192, 255), # Pink
            'ITEM_QTY': (42, 42, 165),          # Brown
            'ITEM_UNIT_COST': (92, 92, 205),    # Indian Red
            'ITEM_TOTAL_COST': (0, 140, 255),   # Dark Orange
            'ITEM_SKU': (140, 230, 240),        # Khaki
            'REGISTER_NUMBER': (180, 105, 255), # Hot Pink
            'CASHIER_ID': (255, 191, 0),        # Deep Sky Blue
            None: (128, 128, 128)               # Gray for unlabeled
        }
        
        # Draw boxes
        for box in annotation.boxes:
            color = colors.get(box.label, colors[None])
            
            # Draw rectangle
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x2, box.y2),
                color,
                2
            )
            
            # Draw label
            if show_labels and box.label:
                label_text = box.label
                cv2.putText(
                    image,
                    label_text,
                    (box.x, box.y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )
            
            # Draw text
            if show_text:
                cv2.putText(
                    image,
                    box.text[:30],  # Limit text length
                    (box.x, box.y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1
                )
        
        # Save
        cv2.imwrite(output_path, image)


if __name__ == '__main__':
    # Example usage
    annotator = OCRAnnotator(ocr_engine='paddleocr')
    annotation = annotator.annotate_image('data/raw/sample_invoice.png')
    
    # Label entities
    labeler = EntityLabeler()
    annotation = labeler.label_boxes(annotation)
    
    # Save annotation
    annotation.save_json('data/annotations/sample_invoice.json')
    
    # Visualize
    visualizer = AnnotationVisualizer()
    visualizer.draw_boxes(
        'data/raw/sample_invoice.png',
        annotation,
        'data/annotations/sample_invoice_annotated.png',
        show_labels=True
    )
    
    print(f"Found {len(annotation.boxes)} text boxes")
    print(f"Labeled: {sum(1 for b in annotation.boxes if b.label)}")
