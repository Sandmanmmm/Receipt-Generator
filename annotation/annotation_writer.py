"""
Annotation Writer - Write annotations to various formats
"""
import json
from pathlib import Path
from typing import List, Optional
from .annotation_schema import InvoiceAnnotation


class AnnotationWriter:
    """Write annotations to disk in various formats"""
    
    @staticmethod
    def write_json(annotation: InvoiceAnnotation, output_path: str):
        """
        Write annotation to JSON format
        
        Args:
            annotation: InvoiceAnnotation to write
            output_path: Path to output file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(annotation.to_dict(), f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def write_jsonl(annotations: List[InvoiceAnnotation], output_path: str, doc_id_prefix: str = "doc"):
        """
        Write annotations to JSONL format (one JSON object per line)
        
        Args:
            annotations: List of InvoiceAnnotation objects
            output_path: Path to output JSONL file
            doc_id_prefix: Prefix for document IDs
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, annotation in enumerate(annotations):
                doc_id = f"{doc_id_prefix}_{idx:06d}"
                jsonl_data = annotation.to_jsonl_format(doc_id)
                f.write(json.dumps(jsonl_data, ensure_ascii=False) + '\n')
    
    @staticmethod
    def write_coco(annotations: List[InvoiceAnnotation], output_path: str, dataset_name: str = "invoices"):
        """
        Write annotations to COCO format
        
        Args:
            annotations: List of InvoiceAnnotation objects
            output_path: Path to output JSON file
            dataset_name: Dataset name
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        coco_format = {
            "info": {
                "description": dataset_name,
                "version": "1.0",
                "year": 2025,
                "contributor": "InvoiceGen",
                "date_created": "2025-11-26"
            },
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Extract unique categories from labels
        categories = set()
        for annotation in annotations:
            for box in annotation.boxes:
                if box.label and box.label != 'O':
                    # Remove B-/I- prefix
                    entity = box.label[2:] if box.label.startswith(('B-', 'I-')) else box.label
                    categories.add(entity)
        
        # Add categories
        for idx, category in enumerate(sorted(categories)):
            coco_format["categories"].append({
                "id": idx + 1,
                "name": category,
                "supercategory": "text"
            })
        
        category_map = {cat["name"]: cat["id"] for cat in coco_format["categories"]}
        
        # Add images and annotations
        annotation_id = 1
        for image_id, annotation in enumerate(annotations, 1):
            # Add image
            coco_format["images"].append({
                "id": image_id,
                "file_name": Path(annotation.image_path).name,
                "width": annotation.image_width,
                "height": annotation.image_height
            })
            
            # Add annotations (boxes)
            for box in annotation.boxes:
                if box.label and box.label != 'O':
                    entity = box.label[2:] if box.label.startswith(('B-', 'I-')) else box.label
                    
                    coco_format["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map.get(entity, 0),
                        "bbox": box.to_coco_format(),
                        "area": box.width * box.height,
                        "segmentation": [],
                        "iscrowd": 0,
                        "attributes": {
                            "text": box.text,
                            "confidence": box.confidence
                        }
                    })
                    annotation_id += 1
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def read_json(json_path: str) -> InvoiceAnnotation:
        """
        Read annotation from JSON file
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            InvoiceAnnotation object
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return InvoiceAnnotation.from_dict(data)
    
    @staticmethod
    def read_jsonl(jsonl_path: str) -> List[InvoiceAnnotation]:
        """
        Read annotations from JSONL file
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            List of InvoiceAnnotation objects
        """
        annotations = []
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # Convert from JSONL format to InvoiceAnnotation
                # Note: JSONL format has different structure, need to adapt
                # This is a simplified version
                annotations.append(InvoiceAnnotation.from_dict(data))
        
        return annotations


__all__ = ['AnnotationWriter']
