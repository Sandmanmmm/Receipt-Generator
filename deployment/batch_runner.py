"""
Batch Runner - Process multiple documents efficiently
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Callable
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PIL import Image


class BatchRunner:
    """Run inference on batches of documents"""
    
    def __init__(self, model_loader, batch_size: int = 8, num_workers: int = 4):
        """
        Initialize batch runner
        
        Args:
            model_loader: ModelLoader instance
            batch_size: Number of documents per batch
            num_workers: Number of parallel workers for preprocessing
        """
        self.model_loader = model_loader
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def process_directory(self, input_dir: str, output_dir: str,
                         ocr_function: Optional[Callable] = None,
                         save_format: str = 'json') -> Dict[str, int]:
        """
        Process all images in directory
        
        Args:
            input_dir: Directory with images
            output_dir: Directory for results
            ocr_function: Function to extract text and boxes from image
            save_format: 'json' or 'jsonl'
            
        Returns:
            Statistics dictionary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get all images
        image_files = list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg'))
        
        print(f"Found {len(image_files)} images")
        
        # Process in batches
        stats = {
            'total': len(image_files),
            'processed': 0,
            'errors': 0
        }
        
        for i in tqdm(range(0, len(image_files), self.batch_size), desc="Processing batches"):
            batch_files = image_files[i:i + self.batch_size]
            
            # Prepare batch data
            batch_data = []
            for image_file in batch_files:
                try:
                    # Extract text and boxes using OCR
                    if ocr_function:
                        words, boxes = ocr_function(str(image_file))
                    else:
                        # Default: use PaddleOCR
                        words, boxes = self._default_ocr(str(image_file))
                    
                    batch_data.append((str(image_file), words, boxes))
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    stats['errors'] += 1
            
            # Run inference
            if batch_data:
                results = self.model_loader.predict_batch(batch_data)
                
                # Save results
                for (image_path, _, _), result in zip(batch_data, results):
                    image_name = Path(image_path).stem
                    output_file = output_path / f"{image_name}.{save_format}"
                    
                    # Decode predictions
                    decoded = self.model_loader.decode_predictions(result)
                    
                    # Save
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(decoded, f, indent=2)
                    
                    stats['processed'] += 1
        
        return stats
    
    def process_file_list(self, file_list: List[str], output_dir: str,
                         ocr_function: Optional[Callable] = None) -> List[Dict]:
        """
        Process list of files
        
        Args:
            file_list: List of image paths
            output_dir: Output directory
            ocr_function: OCR function
            
        Returns:
            List of results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i in tqdm(range(0, len(file_list), self.batch_size), desc="Processing"):
            batch_files = file_list[i:i + self.batch_size]
            
            # Prepare batch
            batch_data = []
            for image_file in batch_files:
                try:
                    if ocr_function:
                        words, boxes = ocr_function(image_file)
                    else:
                        words, boxes = self._default_ocr(image_file)
                    
                    batch_data.append((image_file, words, boxes))
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
            # Inference
            if batch_data:
                results = self.model_loader.predict_batch(batch_data)
                
                for (image_path, _, _), result in zip(batch_data, results):
                    decoded = self.model_loader.decode_predictions(result)
                    decoded['image_path'] = image_path
                    all_results.append(decoded)
                    
                    # Save individual result
                    image_name = Path(image_path).stem
                    output_file = output_path / f"{image_name}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(decoded, f, indent=2)
        
        return all_results
    
    def _default_ocr(self, image_path: str) -> tuple:
        """
        Default OCR using PaddleOCR
        
        Args:
            image_path: Path to image
            
        Returns:
            Tuple of (words, boxes)
        """
        try:
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            result = ocr.ocr(image_path, cls=True)
            
            words = []
            boxes = []
            
            if result and result[0]:
                for line in result[0]:
                    bbox = line[0]
                    text = line[1][0]
                    
                    # Normalize box to 0-1000
                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]
                    
                    # Get image size
                    img = Image.open(image_path)
                    width, height = img.size
                    
                    normalized_box = [
                        int(min(x_coords) / width * 1000),
                        int(min(y_coords) / height * 1000),
                        int(max(x_coords) / width * 1000),
                        int(max(y_coords) / height * 1000)
                    ]
                    
                    words.append(text)
                    boxes.append(normalized_box)
            
            return words, boxes
        
        except ImportError:
            print("PaddleOCR not available. Install with: pip install paddleocr")
            return [], []
        except Exception as e:
            print(f"OCR error: {e}")
            return [], []


class AsyncBatchRunner(BatchRunner):
    """Asynchronous batch runner for better throughput"""
    
    def __init__(self, model_loader, batch_size: int = 8, 
                 num_workers: int = 4, use_threads: bool = True):
        """
        Initialize async batch runner
        
        Args:
            model_loader: ModelLoader instance
            batch_size: Batch size
            num_workers: Number of workers
            use_threads: Use threads (True) or processes (False)
        """
        super().__init__(model_loader, batch_size, num_workers)
        self.use_threads = use_threads
    
    def process_directory_async(self, input_dir: str, output_dir: str,
                                ocr_function: Optional[Callable] = None) -> Dict[str, int]:
        """
        Process directory asynchronously
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            ocr_function: OCR function
            
        Returns:
            Statistics dictionary
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get images
        image_files = list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg'))
        
        # Use thread/process pool for OCR preprocessing
        Executor = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        
        stats = {'total': len(image_files), 'processed': 0, 'errors': 0}
        
        with Executor(max_workers=self.num_workers) as executor:
            # Submit OCR tasks
            ocr_futures = []
            for image_file in image_files:
                future = executor.submit(
                    ocr_function or self._default_ocr,
                    str(image_file)
                )
                ocr_futures.append((image_file, future))
            
            # Collect results and run inference in batches
            batch_data = []
            for image_file, future in tqdm(ocr_futures, desc="OCR + Inference"):
                try:
                    words, boxes = future.result()
                    batch_data.append((str(image_file), words, boxes))
                    
                    # Process batch when full
                    if len(batch_data) >= self.batch_size:
                        self._process_and_save_batch(batch_data, output_path, stats)
                        batch_data = []
                
                except Exception as e:
                    print(f"Error: {e}")
                    stats['errors'] += 1
            
            # Process remaining
            if batch_data:
                self._process_and_save_batch(batch_data, output_path, stats)
        
        return stats
    
    def _process_and_save_batch(self, batch_data: list, output_path: Path,
                                stats: dict):
        """Process batch and save results"""
        results = self.model_loader.predict_batch(batch_data)
        
        for (image_path, _, _), result in zip(batch_data, results):
            decoded = self.model_loader.decode_predictions(result)
            
            image_name = Path(image_path).stem
            output_file = output_path / f"{image_name}.json"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(decoded, f, indent=2)
            
            stats['processed'] += 1


__all__ = ['BatchRunner', 'AsyncBatchRunner']
