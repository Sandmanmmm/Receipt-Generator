"""
Annotation Visualization Tool
Visualizes token-level annotations with bounding boxes
"""
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import random


class AnnotationVisualizer:
    """Visualize annotations on images"""
    
    # Color palette for different entity types (BGR format)
    COLORS = {
        'DOC_TYPE': (255, 0, 0),      # Blue
        'INVOICE_NUMBER': (0, 255, 0),  # Green
        'PURCHASE_ORDER_NUMBER': (0, 200, 200),  # Yellow
        'DATE': (255, 0, 255),         # Magenta
        'SUPPLIER': (255, 165, 0),     # Orange
        'BUYER': (180, 105, 255),      # Pink
        'AMOUNT': (0, 255, 255),       # Cyan
        'ITEM': (255, 255, 0),         # Light blue
        'TABLE': (128, 128, 128),      # Gray
        'OTHER': (200, 200, 200),      # Light gray
    }
    
    def __init__(self, color_by_entity: bool = True):
        """
        Initialize visualizer
        
        Args:
            color_by_entity: Color boxes by entity type (vs random)
        """
        self.color_by_entity = color_by_entity
        self.entity_colors = {}
    
    def _get_entity_type(self, label: str) -> str:
        """Extract entity type from BIO label"""
        if label == 'O':
            return 'OTHER'
        
        # Remove B-/I- prefix
        entity = label[2:] if label.startswith(('B-', 'I-')) else label
        
        # Map to color category
        if 'DATE' in entity or 'DUE' in entity or 'ORDER' in entity:
            return 'DATE'
        elif 'INVOICE' in entity or 'NUMBER' in entity:
            return 'INVOICE_NUMBER'
        elif 'PURCHASE_ORDER' in entity or 'PO' in entity:
            return 'PURCHASE_ORDER_NUMBER'
        elif 'SUPPLIER' in entity:
            return 'SUPPLIER'
        elif 'BUYER' in entity or 'MERCHANT' in entity:
            return 'BUYER'
        elif 'AMOUNT' in entity or 'TOTAL' in entity or 'TAX' in entity or 'SUBTOTAL' in entity:
            return 'AMOUNT'
        elif 'ITEM' in entity:
            return 'ITEM'
        elif 'TABLE' in entity:
            return 'TABLE'
        elif 'DOC_TYPE' in entity:
            return 'DOC_TYPE'
        else:
            return 'OTHER'
    
    def _get_color(self, label: str) -> Tuple[int, int, int]:
        """Get color for a label"""
        if not self.color_by_entity:
            # Random color per label
            if label not in self.entity_colors:
                self.entity_colors[label] = (
                    random.randint(50, 255),
                    random.randint(50, 255),
                    random.randint(50, 255)
                )
            return self.entity_colors[label]
        
        # Color by entity type
        entity_type = self._get_entity_type(label)
        return self.COLORS.get(entity_type, self.COLORS['OTHER'])
    
    def visualize_document(
        self,
        doc: Dict,
        output_path: str,
        show_labels: bool = True,
        show_confidence: bool = False,
        thickness: int = 2,
        font_scale: float = 0.5
    ):
        """
        Visualize a single document
        
        Args:
            doc: Document dictionary from JSONL
            output_path: Path to save visualization
            show_labels: Whether to show label text
            show_confidence: Whether to show confidence scores
            thickness: Bbox line thickness
            font_scale: Font scale for text
        """
        # Load image
        image_path = doc['image_path']
        if not Path(image_path).exists():
            print(f"Warning: Image not found: {image_path}")
            return
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image: {image_path}")
            return
        
        # Create overlay
        overlay = image.copy()
        
        # Draw bounding boxes and labels
        tokens = doc['tokens']
        
        for token in tokens:
            bbox = token['bbox']
            label = token['label']
            text = token.get('text', '')
            confidence = token.get('confidence', None)
            
            # Skip 'O' labels if desired
            if label == 'O':
                continue
            
            # Get color
            color = self._get_color(label)
            
            # Draw rectangle
            x0, y0, x1, y1 = map(int, bbox)
            cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
            
            # Draw label text
            if show_labels:
                label_text = label
                if show_confidence and confidence is not None:
                    label_text = f"{label} ({confidence:.2f})"
                
                # Background for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, 1)
                
                # Position text above bbox
                text_x = x0
                text_y = max(y0 - 5, text_h + 5)
                
                # Draw text background
                cv2.rectangle(
                    overlay,
                    (text_x, text_y - text_h - 2),
                    (text_x + text_w + 2, text_y + 2),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    overlay,
                    label_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        # Blend overlay with original
        alpha = 0.6
        result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Add legend
        result = self._add_legend(result)
        
        # Save
        cv2.imwrite(output_path, result)
        print(f"Saved visualization: {output_path}")
    
    def _add_legend(self, image: np.ndarray) -> np.ndarray:
        """Add color legend to image"""
        if not self.color_by_entity:
            return image
        
        # Legend dimensions
        legend_height = 30 * len(self.COLORS)
        legend_width = 250
        margin = 10
        
        # Create legend background
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        legend[:] = (255, 255, 255)  # White background
        
        # Draw legend items
        y_offset = 20
        for entity_type, color in self.COLORS.items():
            # Draw color box
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), color, -1)
            cv2.rectangle(legend, (10, y_offset - 10), (30, y_offset + 5), (0, 0, 0), 1)
            
            # Draw text
            cv2.putText(
                legend,
                entity_type,
                (40, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA
            )
            
            y_offset += 30
        
        # Paste legend on image (top-right corner)
        h, w = image.shape[:2]
        x_pos = w - legend_width - margin
        y_pos = margin
        
        # Add border
        cv2.rectangle(
            image,
            (x_pos - 2, y_pos - 2),
            (x_pos + legend_width + 2, y_pos + legend_height + 2),
            (0, 0, 0),
            2
        )
        
        # Paste legend
        image[y_pos:y_pos + legend_height, x_pos:x_pos + legend_width] = legend
        
        return image
    
    def visualize_batch(
        self,
        jsonl_path: str,
        output_dir: str,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Visualize multiple documents from JSONL
        
        Args:
            jsonl_path: Path to JSONL file
            output_dir: Output directory for visualizations
            num_samples: Number of samples to visualize (None = all)
            **kwargs: Additional arguments for visualize_document
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if num_samples and count >= num_samples:
                    break
                
                doc = json.loads(line)
                doc_id = doc.get('id', f'doc_{count}')
                
                output_file = output_path / f"{doc_id}_annotated.png"
                self.visualize_document(doc, str(output_file), **kwargs)
                
                count += 1
        
        print(f"\nVisualized {count} documents in {output_dir}")


def main():
    """CLI for annotation visualization"""
    parser = argparse.ArgumentParser(
        description="Visualize token-level annotations"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to JSONL annotation file'
    )
    parser.add_argument(
        '--output-dir',
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to visualize (default: all)'
    )
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='Hide label text'
    )
    parser.add_argument(
        '--show-confidence',
        action='store_true',
        help='Show confidence scores'
    )
    parser.add_argument(
        '--thickness',
        type=int,
        default=2,
        help='Bounding box line thickness'
    )
    parser.add_argument(
        '--random-colors',
        action='store_true',
        help='Use random colors instead of entity-based colors'
    )
    
    args = parser.parse_args()
    
    visualizer = AnnotationVisualizer(color_by_entity=not args.random_colors)
    
    visualizer.visualize_batch(
        jsonl_path=args.input,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        show_labels=not args.no_labels,
        show_confidence=args.show_confidence,
        thickness=args.thickness
    )


if __name__ == '__main__':
    main()
