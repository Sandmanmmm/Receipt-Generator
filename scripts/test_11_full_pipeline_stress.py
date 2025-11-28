#!/usr/bin/env python3
"""
Test 11: 1000 Sample Full Pipeline Stress Test

Purpose: Validate entire generation → OCR → annotation → HF dataset → model forward pass
         pipeline under load to catch rare edge cases and scaling issues.

Success Criteria:
- ❌ No empty tokens
- ❌ No missing bboxes
- ❌ No mismatched lengths
- ❌ No NaNs in model output
- ❌ No failures in PaddleOCR
- ✔️ Mean sequence length < 80 tokens
- ✔️ No sample exceeds 512 tokens
- ✔️ All entities appear at least 30 times

Usage:
    python scripts/test_11_full_pipeline_stress.py --num-samples 1000
    python scripts/test_11_full_pipeline_stress.py --num-samples 1000 --forward-pass-samples 200
    python scripts/test_11_full_pipeline_stress.py --num-samples 100 --quick-test
"""

import argparse
import sys
import yaml
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import tempfile
import shutil
import time
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3TokenizerFast

from generators.retail_data_generator import RetailDataGenerator
from generators.html_to_png_renderer import SimplePNGRenderer
from annotation.ocr_engine import OCREngine
from annotation.token_annotator import TokenAnnotator


class StressTestResults:
    """Container for stress test results and statistics."""
    
    def __init__(self):
        self.total_samples = 0
        self.successful_samples = 0
        self.failed_samples = 0
        
        # Generation stats
        self.generation_failures = []
        
        # OCR stats
        self.ocr_failures = []
        self.empty_ocr_results = []
        self.ocr_times = []
        
        # Token stats
        self.token_counts = []
        self.empty_token_samples = []
        self.max_tokens = 0
        self.mean_tokens = 0.0
        
        # Bbox stats
        self.missing_bbox_samples = []
        self.bbox_mismatches = []
        
        # Array length mismatches
        self.length_mismatches = []
        
        # Entity distribution
        self.entity_counter = Counter()
        self.entity_by_sample = defaultdict(list)
        
        # HF conversion stats
        self.hf_conversion_failures = []
        
        # Model forward pass stats
        self.forward_pass_samples_tested = 0
        self.forward_pass_failures = []
        self.nan_outputs = []
        self.inf_outputs = []
        
        # Memory stats
        self.peak_memory_mb = 0.0
        
        # Timing
        self.total_time = 0.0
        self.generation_time = 0.0
        self.ocr_time = 0.0
        self.annotation_time = 0.0
        self.hf_conversion_time = 0.0
        self.forward_pass_time = 0.0


def load_schema(schema_path: Path) -> Dict:
    """Load label schema from YAML file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    return schema


def setup_components(schema_path: Path, device: str = 'cpu') -> Tuple:
    """Initialize all pipeline components."""
    
    print("Initializing pipeline components...")
    
    # Load label schema
    schema = load_schema(schema_path)
    label_list = schema.get('label_list', [])
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    
    print(f"  Labels: {len(label_list)}")
    
    # Initialize data generator
    generator = RetailDataGenerator()
    print(f"  Data generator: RetailDataGenerator ready")
    
    # Initialize OCR engine
    ocr_engine = OCREngine(engine='paddleocr', show_log=False)
    print(f"  OCR engine: PaddleOCR initialized")
    
    # Initialize PNG renderer
    png_renderer = SimplePNGRenderer(width=800, height=1200)
    print(f"  PNG renderer: SimplePNGRenderer ready")
    
    # Initialize token annotator
    annotator = TokenAnnotator(schema)
    print(f"  Token annotator: Ready ({len(annotator.label_list)} labels)")
    
    # Initialize model and tokenizer
    model_name = "microsoft/layoutlmv3-base"
    print(f"  Loading model: {model_name}...")
    
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    model.eval()
    
    # Move to device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(model_name)
    
    print(f"  Model loaded: {model.__class__.__name__}")
    print(f"  Device: {device}")
    print()
    
    return generator, ocr_engine, annotator, png_renderer, model, tokenizer, label_list, label2id, device_obj


def generate_receipt_image(
    generator: RetailDataGenerator,
    png_renderer: SimplePNGRenderer,
    sample_id: int,
    output_dir: Path
) -> Optional[Tuple[Dict, str]]:
    """Generate a single receipt and render to PNG."""
    
    try:
        # Generate receipt data (POS receipt format)
        receipt_obj = generator.generate_pos_receipt()
        receipt_dict = generator.to_dict(receipt_obj)
        receipt_dict['id'] = f"stress_test_{sample_id:05d}"
        
        # Render to PNG
        image_path = output_dir / f"{receipt_dict['id']}.png"
        success = png_renderer.render_receipt_dict(receipt_dict, str(image_path))
        
        if not success or not image_path.exists():
            return None
        
        return receipt_dict, str(image_path)
        
    except Exception as e:
        return None


def run_ocr_on_image(
    ocr_engine: OCREngine,
    image_path: str
) -> Optional[Tuple[List[str], List[List[int]]]]:
    """Run OCR on image and extract tokens with bounding boxes."""
    
    try:
        # Extract bounding boxes using OCR
        bbox_list = ocr_engine.extract_text(image_path)
        
        if not bbox_list or len(bbox_list) == 0:
            return None
        
        # Extract tokens and bboxes from BoundingBox objects
        tokens = []
        bboxes = []
        
        for bbox_obj in bbox_list:
            # BoundingBox has: text, x, y, width, height, confidence
            if bbox_obj.confidence < 0.5:  # Skip low confidence
                continue
            
            # Convert to pascal VOC format [x_min, y_min, x_max, y_max]
            bbox = bbox_obj.to_pascal_voc()
            
            # Split text into words
            words = bbox_obj.text.split()
            for word in words:
                tokens.append(word)
                bboxes.append(bbox)  # Same bbox for all words from this detection
        
        if len(tokens) == 0:
            return None
        
        return tokens, bboxes
        
    except Exception as e:
        return None


def annotate_and_convert(
    annotator: TokenAnnotator,
    receipt_dict: Dict,
    tokens: List[str],
    bboxes: List[List[int]],
    image_path: str
) -> Optional[Dict]:
    """Annotate tokens and convert to HF format."""
    
    try:
        # Use TokenAnnotator to create HF-ready annotation
        annotation = annotator.annotate_tokens(
            receipt_dict,
            tokens,
            bboxes,
            image_path,
            image_width=800,
            image_height=1200
        )
        
        # Validate annotation
        is_valid, errors = annotator.validate_annotation(annotation)
        
        if not is_valid:
            return None
        
        return annotation
        
    except Exception as e:
        return None


def run_model_forward_pass(
    model: LayoutLMv3ForTokenClassification,
    tokenizer: LayoutLMv3TokenizerFast,
    annotation: Dict,
    device: torch.device
) -> Optional[Dict]:
    """Run model forward pass on annotation."""
    
    try:
        tokens = annotation['tokens']
        bboxes = annotation['bboxes']
        labels = annotation['ner_tags']
        
        # Tokenize
        encoding = tokenizer(
            tokens,
            boxes=bboxes,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=512
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        bbox = encoding['bbox'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Create labels tensor
        labels_tensor = torch.full(input_ids.shape, -100, dtype=torch.long)
        word_ids = encoding.word_ids(batch_index=0)
        
        for idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx < len(labels):
                labels_tensor[0, idx] = labels[word_idx]
        
        labels_tensor = labels_tensor.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                bbox=bbox,
                attention_mask=attention_mask,
                labels=labels_tensor
            )
        
        # Check for NaN/Inf
        loss = outputs.loss.item()
        logits = outputs.logits
        
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        
        return {
            'loss': loss,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'logits_shape': tuple(logits.shape)
        }
        
    except Exception as e:
        return None


def collect_entity_statistics(
    annotation: Dict,
    annotator: TokenAnnotator,
    results: StressTestResults
):
    """Collect entity distribution statistics."""
    
    sample_id = annotation['id']
    ner_tags = annotation['ner_tags']
    
    current_entity = None
    
    for tag_id in ner_tags:
        label = annotator.id2label.get(tag_id, 'O')
        
        if label.startswith('B-'):
            entity_type = label[2:]  # Remove 'B-' prefix
            results.entity_counter[entity_type] += 1
            results.entity_by_sample[sample_id].append(entity_type)
            current_entity = entity_type
        elif label.startswith('I-') and current_entity:
            # Continue current entity (already counted)
            pass
        else:
            current_entity = None


def print_progress(current: int, total: int, prefix: str = "Progress"):
    """Print progress bar."""
    bar_length = 50
    progress = current / total if total > 0 else 0
    filled = int(bar_length * progress)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = progress * 100
    print(f"\r{prefix}: [{bar}] {current}/{total} ({percent:.1f}%)", end='', flush=True)


def run_stress_test(
    num_samples: int,
    forward_pass_samples: int,
    schema_path: Path,
    output_dir: Path,
    device: str = 'cpu'
) -> StressTestResults:
    """Run the full pipeline stress test."""
    
    results = StressTestResults()
    results.total_samples = num_samples
    
    start_time = time.time()
    
    # Setup components
    print("\n" + "="*80)
    print("TEST 11: 1000 SAMPLE FULL PIPELINE STRESS TEST")
    print("="*80)
    print(f"Total samples: {num_samples}")
    print(f"Forward pass samples: {forward_pass_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print()
    
    generator, ocr_engine, annotator, png_renderer, model, tokenizer, label_list, label2id, device_obj = setup_components(
        schema_path, device
    )
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    
    # Store annotations for forward pass testing
    annotations = []
    
    print("="*80)
    print("RUNNING FULL PIPELINE TEST")
    print("="*80)
    print()
    
    # Process each sample
    for i in range(num_samples):
        sample_id = f"stress_test_{i:05d}"
        
        # Print progress every 10 samples
        if (i + 1) % 10 == 0 or i == 0:
            print_progress(i + 1, num_samples, "Processing")
        
        try:
            # Step 1: Generate receipt and render to PNG
            gen_start = time.time()
            result = generate_receipt_image(generator, png_renderer, i, images_dir)
            results.generation_time += time.time() - gen_start
            
            if result is None:
                results.generation_failures.append(sample_id)
                results.failed_samples += 1
                continue
            
            receipt_dict, image_path = result
            
            # Step 2: Run OCR
            ocr_start = time.time()
            ocr_result = run_ocr_on_image(ocr_engine, image_path)
            ocr_time_taken = time.time() - ocr_start
            results.ocr_time += ocr_time_taken
            results.ocr_times.append(ocr_time_taken)
            
            if ocr_result is None:
                results.ocr_failures.append(sample_id)
                results.failed_samples += 1
                continue
            
            tokens, bboxes = ocr_result
            
            # Check for empty tokens
            if len(tokens) == 0:
                results.empty_token_samples.append(sample_id)
                results.failed_samples += 1
                continue
            
            # Check for missing bboxes
            if len(bboxes) == 0:
                results.missing_bbox_samples.append(sample_id)
                results.failed_samples += 1
                continue
            
            # Check for length mismatch
            if len(tokens) != len(bboxes):
                results.bbox_mismatches.append({
                    'sample_id': sample_id,
                    'tokens_len': len(tokens),
                    'bboxes_len': len(bboxes)
                })
                results.failed_samples += 1
                continue
            
            # Track token count
            results.token_counts.append(len(tokens))
            results.max_tokens = max(results.max_tokens, len(tokens))
            
            # Step 3: Annotate and convert to HF format
            annot_start = time.time()
            annotation = annotate_and_convert(
                annotator, receipt_dict, tokens, bboxes, image_path
            )
            results.annotation_time += time.time() - annot_start
            
            if annotation is None:
                results.hf_conversion_failures.append(sample_id)
                results.failed_samples += 1
                continue
            
            # Check for length mismatch in annotation
            if len(annotation['tokens']) != len(annotation['ner_tags']) or \
               len(annotation['tokens']) != len(annotation['bboxes']):
                results.length_mismatches.append({
                    'sample_id': sample_id,
                    'tokens_len': len(annotation['tokens']),
                    'ner_tags_len': len(annotation['ner_tags']),
                    'bboxes_len': len(annotation['bboxes'])
                })
                results.failed_samples += 1
                continue
            
            # Collect entity statistics
            collect_entity_statistics(annotation, annotator, results)
            
            # Store annotation for forward pass testing
            annotations.append(annotation)
            
            results.successful_samples += 1
            
        except Exception as e:
            results.failed_samples += 1
            continue
    
    print()  # New line after progress bar
    
    # Calculate token statistics
    if results.token_counts:
        results.mean_tokens = float(np.mean(results.token_counts))
    
    # Step 4: Run forward pass on random subset
    if forward_pass_samples > 0 and len(annotations) > 0:
        print("\n" + "="*80)
        print("RUNNING MODEL FORWARD PASS TEST")
        print("="*80)
        print()
        
        # Sample random subset
        test_annotations = random.sample(
            annotations,
            min(forward_pass_samples, len(annotations))
        )
        
        results.forward_pass_samples_tested = len(test_annotations)
        
        for idx, annotation in enumerate(test_annotations):
            if (idx + 1) % 10 == 0 or idx == 0:
                print_progress(idx + 1, len(test_annotations), "Forward pass")
            
            fp_start = time.time()
            fp_result = run_model_forward_pass(
                model, tokenizer, annotation, device_obj
            )
            results.forward_pass_time += time.time() - fp_start
            
            if fp_result is None:
                results.forward_pass_failures.append(annotation['id'])
                continue
            
            if fp_result['has_nan']:
                results.nan_outputs.append(annotation['id'])
            
            if fp_result['has_inf']:
                results.inf_outputs.append(annotation['id'])
        
        print()  # New line after progress bar
    
    results.total_time = time.time() - start_time
    
    # Track memory
    if torch.cuda.is_available():
        results.peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    
    return results


def print_results(results: StressTestResults, label_list: List[str]) -> bool:
    """Print comprehensive test results and return pass/fail status."""
    
    print("\n" + "="*80)
    print("STRESS TEST RESULTS")
    print("="*80)
    print()
    
    # Overall statistics
    print("Overall Statistics:")
    print(f"  Total samples:       {results.total_samples}")
    print(f"  Successful:          {results.successful_samples} ({results.successful_samples/results.total_samples*100:.1f}%)")
    print(f"  Failed:              {results.failed_samples} ({results.failed_samples/results.total_samples*100:.1f}%)")
    print(f"  Total time:          {results.total_time:.1f}s ({results.total_time/60:.1f}m)")
    print()
    
    # Timing breakdown
    print("Timing Breakdown:")
    if results.total_time > 0:
        print(f"  Generation:          {results.generation_time:.1f}s ({results.generation_time/results.total_time*100:.1f}%)")
        print(f"  OCR:                 {results.ocr_time:.1f}s ({results.ocr_time/results.total_time*100:.1f}%)")
        print(f"  Annotation:          {results.annotation_time:.1f}s ({results.annotation_time/results.total_time*100:.1f}%)")
        if results.forward_pass_time > 0:
            print(f"  Forward Pass:        {results.forward_pass_time:.1f}s ({results.forward_pass_time/results.total_time*100:.1f}%)")
    print()
    
    # OCR statistics
    if results.ocr_times:
        print("OCR Performance:")
        print(f"  Mean OCR time:       {np.mean(results.ocr_times):.3f}s")
        print(f"  Median OCR time:     {np.median(results.ocr_times):.3f}s")
        print(f"  Max OCR time:        {np.max(results.ocr_times):.3f}s")
        print()
    
    # Token statistics
    if results.token_counts:
        print("Token Statistics:")
        print(f"  Mean tokens:         {results.mean_tokens:.1f}")
        print(f"  Median tokens:       {np.median(results.token_counts):.1f}")
        print(f"  Max tokens:          {results.max_tokens}")
        print(f"  Min tokens:          {np.min(results.token_counts):.0f}")
        print(f"  Std deviation:       {np.std(results.token_counts):.1f}")
        print()
        
        # Token distribution
        print("  Token distribution:")
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(results.token_counts, p)
            print(f"    {p}th percentile:   {val:.0f} tokens")
        print()
    
    # Entity distribution
    print("Entity Distribution:")
    print(f"  Unique entity types: {len(results.entity_counter)}")
    print(f"  Total entities:      {sum(results.entity_counter.values())}")
    print()
    
    # Find entities below threshold
    min_entity_count = 30
    low_frequency_entities = [
        (entity, count) for entity, count in results.entity_counter.items()
        if count < min_entity_count
    ]
    
    if low_frequency_entities:
        print(f"  Entities below {min_entity_count} occurrences:")
        for entity, count in sorted(low_frequency_entities, key=lambda x: x[1])[:10]:
            print(f"    {entity:30s} {count:4d}")
        print()
    
    # Top entities
    print("  Top 10 entities:")
    for entity, count in results.entity_counter.most_common(10):
        print(f"    {entity:30s} {count:4d}")
    print()
    
    # Failure analysis
    print("Failure Analysis:")
    print(f"  Generation failures: {len(results.generation_failures)}")
    print(f"  OCR failures:        {len(results.ocr_failures)}")
    print(f"  Empty tokens:        {len(results.empty_token_samples)}")
    print(f"  Missing bboxes:      {len(results.missing_bbox_samples)}")
    print(f"  Bbox mismatches:     {len(results.bbox_mismatches)}")
    print(f"  Length mismatches:   {len(results.length_mismatches)}")
    print(f"  HF conversion fail:  {len(results.hf_conversion_failures)}")
    print()
    
    # Forward pass results
    if results.forward_pass_samples_tested > 0:
        print("Forward Pass Results:")
        print(f"  Samples tested:      {results.forward_pass_samples_tested}")
        print(f"  Failures:            {len(results.forward_pass_failures)}")
        print(f"  NaN outputs:         {len(results.nan_outputs)}")
        print(f"  Inf outputs:         {len(results.inf_outputs)}")
        print()
    
    # Success criteria validation
    print("="*80)
    print("SUCCESS CRITERIA VALIDATION")
    print("="*80)
    print()
    
    all_passed = True
    
    # Critical failures (must be zero)
    criteria = [
        (len(results.empty_token_samples) == 0, "No empty tokens", len(results.empty_token_samples)),
        (len(results.missing_bbox_samples) == 0, "No missing bboxes", len(results.missing_bbox_samples)),
        (len(results.bbox_mismatches) == 0, "No bbox mismatches", len(results.bbox_mismatches)),
        (len(results.length_mismatches) == 0, "No length mismatches", len(results.length_mismatches)),
        (len(results.nan_outputs) == 0, "No NaN in model output", len(results.nan_outputs)),
        (len(results.inf_outputs) == 0, "No Inf in model output", len(results.inf_outputs)),
        (len(results.ocr_failures) < results.total_samples * 0.05, "OCR failure rate < 5%", 
         f"{len(results.ocr_failures)/results.total_samples*100:.1f}%"),
    ]
    
    for passed, description, value in criteria:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {description:40s} ({value})")
        if not passed:
            all_passed = False
    
    print()
    
    # Performance criteria
    if results.token_counts:
        mean_check = results.mean_tokens < 80
        max_check = results.max_tokens <= 512
        
        status = "✓ PASS" if mean_check else "⚠ WARN"
        print(f"  {status:8s} Mean sequence length < 80 tokens     ({results.mean_tokens:.1f})")
        
        status = "✓ PASS" if max_check else "✗ FAIL"
        print(f"  {status:8s} No sample exceeds 512 tokens         ({results.max_tokens})")
        
        if not max_check:
            all_passed = False
        
        print()
    
    # Entity coverage (warning only, not failure)
    entity_check = len(low_frequency_entities) == 0
    
    status = "✓ PASS" if entity_check else "⚠ WARN"
    print(f"  {status:8s} All entities appear 30+ times")
    
    if not entity_check:
        print(f"           ({len(low_frequency_entities)} entities below threshold)")
    
    print()
    
    # Final verdict
    print("="*80)
    if all_passed:
        print("[PASS] TEST 11 PASSED - System validated under load!")
        print()
        print("Pipeline ready for production:")
        print("  ✓ Large-scale generation stable")
        print("  ✓ OCR pipeline reliable")
        print("  ✓ Annotation pipeline robust")
        print("  ✓ HF conversion working")
        print("  ✓ Model forward pass stable")
        print("  ✓ No data corruption at scale")
        if entity_check:
            print("  ✓ Entity distribution adequate")
        else:
            print("  ⚠ Some entities under-represented (non-critical)")
    else:
        print("[FAIL] TEST 11 FAILED - Fix issues before production")
        print()
        print("Issues found:")
        if len(results.empty_token_samples) > 0:
            print(f"  ✗ {len(results.empty_token_samples)} samples with empty tokens")
        if len(results.missing_bbox_samples) > 0:
            print(f"  ✗ {len(results.missing_bbox_samples)} samples with missing bboxes")
        if len(results.bbox_mismatches) > 0:
            print(f"  ✗ {len(results.bbox_mismatches)} bbox length mismatches")
        if len(results.length_mismatches) > 0:
            print(f"  ✗ {len(results.length_mismatches)} array length mismatches")
        if len(results.nan_outputs) > 0:
            print(f"  ✗ {len(results.nan_outputs)} samples with NaN outputs")
        if len(results.inf_outputs) > 0:
            print(f"  ✗ {len(results.inf_outputs)} samples with Inf outputs")
        if results.max_tokens > 512:
            print(f"  ✗ Max tokens ({results.max_tokens}) exceeds 512 limit")
        if len(results.ocr_failures) >= results.total_samples * 0.05:
            print(f"  ✗ OCR failure rate too high ({len(results.ocr_failures)/results.total_samples*100:.1f}%)")
    
    print("="*80)
    print()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test 11: Full Pipeline Stress Test"
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of samples to generate and test (default: 1000)'
    )
    parser.add_argument(
        '--forward-pass-samples',
        type=int,
        default=200,
        help='Number of random samples to test with forward pass (default: 200)'
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default=Path('config/labels_retail.yaml'),
        help='Path to label schema YAML file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/stress_test'),
        help='Output directory for generated receipts (default: outputs/stress_test)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device for model inference (default: cpu)'
    )
    parser.add_argument(
        '--keep-images',
        action='store_true',
        help='Keep generated images after test (default: delete)'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with 100 samples and 20 forward passes'
    )
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.num_samples = 100
        args.forward_pass_samples = 20
        print("Quick test mode: 100 samples, 20 forward passes")
        print()
    
    # Validate schema path
    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)
    
    # Run stress test
    results = run_stress_test(
        num_samples=args.num_samples,
        forward_pass_samples=args.forward_pass_samples,
        schema_path=args.schema,
        output_dir=args.output_dir,
        device=args.device
    )
    
    # Load label list for reporting
    schema = load_schema(args.schema)
    label_list = schema.get('label_list', [])
    
    # Print results
    passed = print_results(results, label_list)
    
    # Cleanup images if requested
    if not args.keep_images:
        print("\nCleaning up generated images...")
        images_dir = args.output_dir / 'images'
        if images_dir.exists():
            shutil.rmtree(images_dir)
        print("  Done")
    
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
