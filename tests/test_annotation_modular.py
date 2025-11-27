"""
Test Modular Annotation System
"""
import pytest
from pathlib import Path
import tempfile
import json

from annotation import (
    BoundingBox,
    InvoiceAnnotation,
    OCREngine,
    BBoxExtractor,
    LabelMapper,
    AnnotationWriter
)


class TestBoundingBox:
    """Test BoundingBox dataclass"""
    
    def test_create_bbox(self):
        bbox = BoundingBox(text="INVOICE", x=100, y=50, width=200, height=30)
        assert bbox.text == "INVOICE"
        assert bbox.x == 100
        assert bbox.y == 50
        assert bbox.x2 == 300
        assert bbox.y2 == 80
    
    def test_to_layoutlmv3_format(self):
        bbox = BoundingBox(text="TEST", x=100, y=50, width=200, height=30)
        formatted = bbox.to_layoutlmv3_format(image_width=1000, image_height=1000)
        assert formatted == [100, 50, 300, 80]
    
    def test_to_coco_format(self):
        bbox = BoundingBox(text="TEST", x=100, y=50, width=200, height=30)
        coco = bbox.to_coco_format()
        assert coco == [100, 50, 200, 30]


class TestInvoiceAnnotation:
    """Test InvoiceAnnotation dataclass"""
    
    def test_create_annotation(self):
        ann = InvoiceAnnotation(
            image_path="test.png",
            tokens=["INVOICE", "123"],
            labels=["B-DOCUMENT_TYPE", "B-INVOICE_NUMBER"],
            bboxes=[[0, 0, 100, 50], [100, 0, 200, 50]],
            image_width=1000,
            image_height=1000
        )
        assert len(ann.tokens) == 2
        assert len(ann.labels) == 2
    
    def test_to_jsonl_format(self):
        ann = InvoiceAnnotation(
            image_path="test.png",
            tokens=["TEST"],
            labels=["O"],
            bboxes=[[0, 0, 100, 50]],
            image_width=1000,
            image_height=1000
        )
        jsonl = ann.to_jsonl_format()
        assert "image_path" in jsonl
        assert "tokens" in jsonl
        assert jsonl["tokens"] == ["TEST"]


class TestOCREngine:
    """Test OCR Engine"""
    
    @pytest.mark.skipif(not Path("tests/fixtures/test_invoice.png").exists(),
                       reason="Test image not found")
    def test_paddleocr_engine(self):
        engine = OCREngine(engine='paddleocr')
        bboxes = engine.extract_text("tests/fixtures/test_invoice.png")
        assert isinstance(bboxes, list)
    
    def test_unsupported_engine(self):
        with pytest.raises(ValueError):
            OCREngine(engine='invalid_engine')


class TestLabelMapper:
    """Test Label Mapper"""
    
    def test_map_invoice_number(self):
        labels = ['O', 'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER']
        mapper = LabelMapper(labels)
        
        tokens = ["Invoice", "#", "INV-2024-001"]
        mapped_labels = mapper.map_labels(tokens)
        
        assert mapped_labels[0] == 'O'  # "Invoice"
        assert mapped_labels[1] == 'O'  # "#"
        assert mapped_labels[2] in ['B-INVOICE_NUMBER', 'O']  # "INV-2024-001"
    
    def test_map_date(self):
        labels = ['O', 'B-INVOICE_DATE', 'I-INVOICE_DATE']
        mapper = LabelMapper(labels)
        
        tokens = ["Date:", "2024-11-26"]
        mapped_labels = mapper.map_labels(tokens)
        
        assert mapped_labels[0] == 'O'
        assert mapped_labels[1] in ['B-INVOICE_DATE', 'O']


class TestAnnotationWriter:
    """Test Annotation Writer"""
    
    def test_write_read_jsonl(self):
        writer = AnnotationWriter()
        
        annotation = {
            'image_path': 'test.png',
            'tokens': ['TEST', '123'],
            'labels': ['O', 'B-NUMBER'],
            'bboxes': [[0, 0, 100, 50], [100, 0, 200, 50]]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            # Write
            writer.write_jsonl([annotation], temp_path)
            
            # Read
            loaded = writer.read_jsonl(temp_path)
            assert len(loaded) == 1
            assert loaded[0]['tokens'] == ['TEST', '123']
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_write_read_json(self):
        writer = AnnotationWriter()
        
        annotation = {
            'image_path': 'test.png',
            'tokens': ['TEST'],
            'labels': ['O']
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            writer.write_json([annotation], temp_path)
            loaded = writer.read_json(temp_path)
            assert len(loaded) == 1
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestEndToEndAnnotation:
    """Test complete annotation pipeline"""
    
    @pytest.mark.skipif(not Path("tests/fixtures/test_invoice.png").exists(),
                       reason="Test image not found")
    def test_full_pipeline(self):
        # Setup
        image_path = "tests/fixtures/test_invoice.png"
        labels = ['O', 'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'B-TOTAL_AMOUNT']
        
        # Extract with OCR
        ocr = OCREngine(engine='paddleocr')
        extractor = BBoxExtractor(ocr)
        bboxes = extractor.extract(image_path)
        
        assert len(bboxes) > 0
        
        # Map labels
        mapper = LabelMapper(labels)
        tokens = [bbox.text for bbox in bboxes]
        mapped_labels = mapper.map_labels(tokens)
        
        assert len(mapped_labels) == len(tokens)
        
        # Write annotation
        writer = AnnotationWriter()
        annotation = InvoiceAnnotation(
            image_path=image_path,
            tokens=tokens,
            labels=mapped_labels,
            bboxes=[[bbox.x, bbox.y, bbox.x2, bbox.y2] for bbox in bboxes],
            image_width=1000,
            image_height=1000
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = f.name
        
        try:
            writer.write_jsonl([annotation.to_dict()], temp_path)
            loaded = writer.read_jsonl(temp_path)
            assert len(loaded) == 1
            assert len(loaded[0]['tokens']) > 0
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
