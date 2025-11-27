"""
Annotation Schema - Data structures for annotations
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
import json


@dataclass
class BoundingBox:
    """Represents a bounding box with text"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: Optional[str] = None  # Entity label (e.g., 'B-COMPANY_NAME', 'I-TOTAL', etc.)
    
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
    
    def to_layoutlmv3_format(self, img_width: int, img_height: int) -> List[int]:
        """Convert to LayoutLMv3 format [x0, y0, x1, y1] normalized to 0-1000"""
        return [
            int(1000 * self.x / img_width),
            int(1000 * self.y / img_height),
            int(1000 * self.x2 / img_width),
            int(1000 * self.y2 / img_height),
        ]


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
    
    def to_jsonl_format(self, doc_id: str) -> Dict[str, Any]:
        """
        Convert to JSONL format for LayoutLMv3 training
        
        Format:
        {
          "id": "doc_0001",
          "image_path": "...",
          "width": 2480,
          "height": 3508,
          "tokens": [{"text": "...", "bbox": [...], "token_id": 0, "label": "B-DOC_TYPE"}],
          "boxes": [[x0,y0,x1,y1], ...],
          "labels": ["B-DOC_TYPE", ...]
        }
        """
        tokens = []
        boxes = []
        labels = []
        
        for idx, box in enumerate(self.boxes):
            tokens.append({
                "text": box.text,
                "bbox": box.to_pascal_voc(),
                "token_id": idx,
                "label": box.label or "O",
                "confidence": box.confidence
            })
            boxes.append(box.to_pascal_voc())
            labels.append(box.label or "O")
        
        return {
            "id": doc_id,
            "image_path": self.image_path,
            "width": self.image_width,
            "height": self.image_height,
            "tokens": tokens,
            "boxes": boxes,
            "labels": labels,
            "metadata": self.metadata
        }
    
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


__all__ = ['BoundingBox', 'InvoiceAnnotation']
