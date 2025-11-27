"""
FastAPI Deployment Server
Serve LayoutLMv3 model for invoice extraction
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from PIL import Image
import io
import numpy as np
from pathlib import Path
import json
import uvicorn

from annotation.annotator import OCRAnnotator


# Response models
class BoundingBox(BaseModel):
    """Bounding box with entity"""
    text: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    entity: Optional[str] = None


class ExtractionResult(BaseModel):
    """Invoice extraction result"""
    invoice_number: Optional[str] = None
    date: Optional[str] = None
    total: Optional[str] = None
    company_name: Optional[str] = None
    client_name: Optional[str] = None
    all_entities: List[BoundingBox]
    metadata: Dict


class InvoiceExtractor:
    """Invoice extraction service"""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize extractor
        
        Args:
            model_path: Path to trained model
            device: Device to use (cuda/cpu)
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load processor
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )
        
        # Load OCR
        self.ocr = OCRAnnotator(ocr_engine='paddleocr')
        
        # Load label mapping
        label_map_path = Path(model_path) / 'config.json'
        with open(label_map_path, 'r') as f:
            config = json.load(f)
        
        self.id2label = config['id2label']
        
        print(f"Model loaded on {self.device}")
    
    def extract(self, image: Image.Image) -> ExtractionResult:
        """
        Extract entities from invoice image
        
        Args:
            image: PIL Image
            
        Returns:
            ExtractionResult
        """
        # Run OCR
        annotation = self.ocr.annotate_image_pil(image)
        
        # Prepare inputs
        words = [box.text for box in annotation.boxes]
        boxes = [[box.x, box.y, box.x2, box.y2] for box in annotation.boxes]
        
        # Normalize boxes
        width, height = image.size
        normalized_boxes = [
            [
                int(1000 * x / width),
                int(1000 * y / height),
                int(1000 * x2 / width),
                int(1000 * y2 / height)
            ]
            for x, y, x2, y2 in boxes
        ]
        
        # Prepare model inputs
        encoding = self.processor(
            image,
            words,
            boxes=normalized_boxes,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        # Move to device
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**encoding)
            predictions = outputs.logits.argmax(dim=-1)
        
        # Process predictions
        predictions = predictions.cpu().numpy()[0]
        
        # Map predictions to entities
        entities = []
        key_entities = {
            'invoice_number': None,
            'date': None,
            'total': None,
            'company_name': None,
            'client_name': None
        }
        
        for word, box, pred in zip(words, annotation.boxes, predictions):
            if pred != 0:  # Not 'O'
                entity_label = self.id2label[str(pred)]
                
                # Remove B-/I- prefix
                entity_type = entity_label.split('-')[-1].lower()
                
                bbox = BoundingBox(
                    text=word,
                    x=box.x,
                    y=box.y,
                    width=box.width,
                    height=box.height,
                    confidence=box.confidence,
                    entity=entity_type
                )
                
                entities.append(bbox)
                
                # Capture key entities
                if entity_type in key_entities and key_entities[entity_type] is None:
                    key_entities[entity_type] = word
        
        return ExtractionResult(
            invoice_number=key_entities['invoice_number'],
            date=key_entities['date'],
            total=key_entities['total'],
            company_name=key_entities['company_name'],
            client_name=key_entities['client_name'],
            all_entities=entities,
            metadata={
                'image_width': width,
                'image_height': height,
                'num_entities': len(entities),
                'num_words': len(words)
            }
        )


# Initialize FastAPI
app = FastAPI(
    title="InvoiceGen Extraction API",
    description="Extract entities from invoice images using LayoutLMv3",
    version="1.0.0"
)

# Global extractor instance
extractor = None


@app.on_event("startup")
async def startup_event():
    """Initialize extractor on startup"""
    global extractor
    
    model_path = "models/layoutlmv3-invoice"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    extractor = InvoiceExtractor(model_path, device)
    print("Model loaded successfully!")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "InvoiceGen Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "extract": "/extract",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": extractor is not None,
        "device": extractor.device if extractor else None
    }


@app.post("/extract", response_model=ExtractionResult)
async def extract_invoice(file: UploadFile = File(...)):
    """
    Extract entities from invoice image
    
    Args:
        file: Invoice image file (PNG, JPG, PDF)
        
    Returns:
        ExtractionResult with extracted entities
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Extract entities
        result = extractor.extract(image)
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@app.post("/batch_extract")
async def batch_extract(files: List[UploadFile] = File(...)):
    """
    Extract entities from multiple invoice images
    
    Args:
        files: List of invoice image files
        
    Returns:
        List of ExtractionResults
    """
    if extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            result = extractor.extract(image)
            results.append({
                "filename": file.filename,
                "result": result
            })
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return results


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
