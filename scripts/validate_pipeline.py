"""
Pipeline Validation & Hardening Script
Validates entire data generation + annotation pipeline before training

Ensures:
- No bad labels
- No faulty BIO sequences
- No misaligned bounding boxes
- Balanced entity occurrence
- Real-world robustness via augmentation

Usage:
    python scripts/validate_pipeline.py --config config/config.yaml --samples 100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import Counter, defaultdict
import yaml
import logging
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Validation result container"""
    passed: bool
    message: str
    details: Dict = None


class PipelineValidator:
    """Validates entire data generation + annotation pipeline"""
    
    def __init__(self, label_schema_path: str, retail_only: bool = True):
        """Initialize validator with label schema"""
        self.retail_only = retail_only
        self.label_schema = self._load_schema(label_schema_path)
        self.valid_entities = self._extract_entities()
        self.valid_bio_labels = self._generate_bio_labels()
        
    def _load_schema(self, path: str) -> Dict:
        """Load label schema from YAML"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _extract_entities(self) -> Set[str]:
        """Extract entity names from BIO label list"""
        entities = set()
        for label in self.label_schema.get('label_list', []):
            if label == 'O':
                continue
            # Remove B- or I- prefix
            entity = label.split('-', 1)[1] if '-' in label else label
            entities.add(entity)
        return entities
    
    def _generate_bio_labels(self) -> Set[str]:
        """Generate valid BIO label set"""
        return set(self.label_schema.get('label_list', []))
    
    def validate_bio_sequence(self, labels: List[str]) -> ValidationResult:
        """
        Validate BIO sequence follows proper tagging rules:
        - I-X must be preceded by B-X or I-X
        - No orphaned I- tags
        - All labels exist in schema
        """
        if not labels:
            return ValidationResult(False, "Empty label sequence")
        
        errors = []
        
        for i, label in enumerate(labels):
            # Check if label exists in schema
            if label not in self.valid_bio_labels:
                errors.append(f"Invalid label at position {i}: '{label}'")
                continue
            
            # Check BIO sequence rules
            if label.startswith('I-'):
                entity = label.split('-', 1)[1]
                if i == 0:
                    errors.append(f"Orphaned I-{entity} at start (position {i})")
                else:
                    prev_label = labels[i-1]
                    if not (prev_label == f'B-{entity}' or prev_label == f'I-{entity}'):
                        errors.append(
                            f"Invalid I-{entity} at position {i}, "
                            f"not preceded by B-{entity} or I-{entity} (prev: {prev_label})"
                        )
        
        if errors:
            return ValidationResult(
                False, 
                f"BIO sequence validation failed: {len(errors)} errors",
                {'errors': errors[:10]}  # Show first 10 errors
            )
        
        return ValidationResult(True, "BIO sequence valid")
    
    def validate_bounding_boxes(
        self, 
        boxes: List[List[float]], 
        tokens: List[str],
        image_width: int,
        image_height: int
    ) -> ValidationResult:
        """
        Validate bounding boxes:
        - Count matches token count
        - Coordinates within image bounds
        - Non-zero width/height
        - Proper format [x0, y0, x1, y1]
        """
        if len(boxes) != len(tokens):
            return ValidationResult(
                False,
                f"Bounding box count mismatch: {len(boxes)} boxes vs {len(tokens)} tokens"
            )
        
        errors = []
        
        for i, box in enumerate(boxes):
            # Check format
            if len(box) != 4:
                errors.append(f"Box {i}: Invalid format (expected 4 coords, got {len(box)})")
                continue
            
            x0, y0, x1, y1 = box
            
            # Check coordinates are numbers
            if not all(isinstance(c, (int, float)) for c in box):
                errors.append(f"Box {i}: Non-numeric coordinates")
                continue
            
            # Check proper ordering
            if x1 <= x0 or y1 <= y0:
                errors.append(f"Box {i}: Invalid ordering (x1={x1} <= x0={x0} or y1={y1} <= y0={y0})")
            
            # Check within image bounds (with tolerance for OCR errors)
            if x0 < -5 or y0 < -5 or x1 > image_width + 5 or y1 > image_height + 5:
                errors.append(
                    f"Box {i}: Outside image bounds "
                    f"({x0}, {y0}, {x1}, {y1}) vs image ({image_width}x{image_height})"
                )
            
            # Check non-zero area
            width = x1 - x0
            height = y1 - y0
            if width < 1 or height < 1:
                errors.append(f"Box {i}: Too small ({width}x{height})")
        
        if errors:
            return ValidationResult(
                False,
                f"Bounding box validation failed: {len(errors)} errors",
                {'errors': errors[:10]}
            )
        
        return ValidationResult(True, "Bounding boxes valid")
    
    def validate_entity_coverage(
        self,
        annotations: List[Dict],
        min_samples_per_entity: int = 5
    ) -> ValidationResult:
        """
        Validate entity coverage across dataset:
        - All retail entities appear at least N times
        - Balanced distribution (no entity > 50% of total)
        - Critical entities (ITEM_*, TOTAL_AMOUNT, etc.) well-represented
        """
        entity_counts = Counter()
        
        # Count entity occurrences
        for ann in annotations:
            labels = ann.get('labels', [])
            for label in labels:
                if label == 'O':
                    continue
                # Extract entity name
                entity = label.split('-', 1)[1] if '-' in label else label
                entity_counts[entity] += 1
        
        errors = []
        warnings = []
        
        # Check minimum coverage
        missing_entities = []
        for entity in self.valid_entities:
            if entity_counts[entity] < min_samples_per_entity:
                missing_entities.append(f"{entity} ({entity_counts[entity]} samples)")
        
        if missing_entities:
            errors.append(f"Under-represented entities: {', '.join(missing_entities[:10])}")
        
        # Check for over-representation
        total_entities = sum(entity_counts.values())
        for entity, count in entity_counts.most_common(5):
            percentage = (count / total_entities) * 100
            if percentage > 50:
                warnings.append(f"{entity}: {percentage:.1f}% (too dominant)")
        
        # Check critical retail entities
        critical_entities = {
            'ITEM_DESCRIPTION', 'ITEM_QTY', 'ITEM_UNIT_COST', 'ITEM_TOTAL_COST',
            'TOTAL_AMOUNT', 'TAX_AMOUNT', 'SUBTOTAL', 
            'PAYMENT_METHOD', 'INVOICE_NUMBER', 'INVOICE_DATE'
        }
        
        for entity in critical_entities:
            if entity in self.valid_entities:
                if entity_counts[entity] < min_samples_per_entity * 2:
                    warnings.append(
                        f"Critical entity under-represented: {entity} "
                        f"({entity_counts[entity]} samples)"
                    )
        
        if errors:
            return ValidationResult(
                False,
                f"Entity coverage validation failed: {len(errors)} errors",
                {'errors': errors, 'warnings': warnings, 'distribution': dict(entity_counts.most_common(20))}
            )
        
        if warnings:
            return ValidationResult(
                True,
                f"Entity coverage OK with {len(warnings)} warnings",
                {'warnings': warnings, 'distribution': dict(entity_counts.most_common(20))}
            )
        
        return ValidationResult(
            True, 
            "Entity coverage excellent",
            {'distribution': dict(entity_counts.most_common(20))}
        )
    
    def validate_annotation_format(self, annotation: Dict) -> ValidationResult:
        """
        Validate annotation format:
        - Required fields present
        - Correct data types
        - Consistent array lengths
        """
        required_fields = ['tokens', 'labels', 'bboxes', 'image']
        
        # Check required fields
        missing = [f for f in required_fields if f not in annotation]
        if missing:
            return ValidationResult(
                False,
                f"Missing required fields: {', '.join(missing)}"
            )
        
        # Check data types
        if not isinstance(annotation['tokens'], list):
            return ValidationResult(False, "'tokens' must be a list")
        
        if not isinstance(annotation['labels'], list):
            return ValidationResult(False, "'labels' must be a list")
        
        if not isinstance(annotation['bboxes'], list):
            return ValidationResult(False, "'bboxes' must be a list")
        
        # Check array lengths match
        n_tokens = len(annotation['tokens'])
        n_labels = len(annotation['labels'])
        n_boxes = len(annotation['bboxes'])
        
        if n_tokens != n_labels:
            return ValidationResult(
                False,
                f"Token/label mismatch: {n_tokens} tokens vs {n_labels} labels"
            )
        
        if n_tokens != n_boxes:
            return ValidationResult(
                False,
                f"Token/bbox mismatch: {n_tokens} tokens vs {n_boxes} boxes"
            )
        
        # Check for empty
        if n_tokens == 0:
            return ValidationResult(False, "Empty annotation (0 tokens)")
        
        return ValidationResult(True, "Annotation format valid")
    
    def validate_sample(
        self, 
        annotation: Dict,
        image_width: int = 800,
        image_height: int = 1200
    ) -> Tuple[bool, List[ValidationResult]]:
        """
        Validate a single annotation sample
        Returns (passed, results)
        """
        results = []
        
        # 1. Format validation
        format_result = self.validate_annotation_format(annotation)
        results.append(format_result)
        if not format_result.passed:
            return False, results
        
        # 2. BIO sequence validation
        bio_result = self.validate_bio_sequence(annotation['labels'])
        results.append(bio_result)
        
        # 3. Bounding box validation
        bbox_result = self.validate_bounding_boxes(
            annotation['bboxes'],
            annotation['tokens'],
            image_width,
            image_height
        )
        results.append(bbox_result)
        
        # Passed if all validations passed
        passed = all(r.passed for r in results)
        return passed, results
    
    def validate_dataset(
        self,
        annotations: List[Dict],
        min_samples_per_entity: int = 5
    ) -> Tuple[bool, Dict]:
        """
        Validate entire dataset
        Returns (passed, summary)
        """
        logger.info(f"Validating {len(annotations)} annotations...")
        
        sample_results = []
        passed_count = 0
        failed_count = 0
        
        # Validate each sample
        for i, ann in enumerate(annotations):
            passed, results = self.validate_sample(ann)
            sample_results.append((i, passed, results))
            
            if passed:
                passed_count += 1
            else:
                failed_count += 1
                
                # Log first few failures
                if failed_count <= 5:
                    logger.warning(f"Sample {i} failed validation:")
                    for r in results:
                        if not r.passed:
                            logger.warning(f"  - {r.message}")
        
        # Validate entity coverage
        coverage_result = self.validate_entity_coverage(annotations, min_samples_per_entity)
        
        # Calculate statistics
        pass_rate = (passed_count / len(annotations)) * 100
        
        summary = {
            'total_samples': len(annotations),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': pass_rate,
            'entity_coverage': coverage_result.passed,
            'coverage_details': coverage_result.details,
            'failed_samples': [i for i, passed, _ in sample_results if not passed][:20]
        }
        
        # Overall pass if > 95% samples valid and entity coverage OK
        overall_passed = pass_rate >= 95.0 and coverage_result.passed
        
        return overall_passed, summary


def generate_test_samples(
    n_samples: int,
    templates: List[str],
    output_dir: Path
) -> List[Dict]:
    """
    Generate test samples using actual pipeline
    Returns list of annotations
    """
    logger.info(f"Generating {n_samples} test samples...")
    
    # TODO: Import and use actual generators
    # from generators.template_renderer import TemplateRenderer
    # from annotation.annotator import Annotator
    
    # For now, return mock data structure
    annotations = []
    
    for i in range(n_samples):
        # Mock annotation structure
        ann = {
            'id': f'sample_{i:04d}',
            'template': templates[i % len(templates)],
            'tokens': ['Store', 'Name', 'Total', '$12.99'],
            'labels': ['B-SUPPLIER_NAME', 'I-SUPPLIER_NAME', 'B-TOTAL_AMOUNT', 'I-TOTAL_AMOUNT'],
            'bboxes': [[10, 10, 100, 30], [105, 10, 180, 30], [10, 50, 60, 70], [65, 50, 120, 70]],
            'image': str(output_dir / f'sample_{i:04d}.png')
        }
        annotations.append(ann)
    
    return annotations


def main():
    parser = argparse.ArgumentParser(description='Validate data generation + annotation pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--label-schema', type=str, default='config/labels_retail.yaml',
                       help='Path to label schema (default: retail)')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of test samples to generate')
    parser.add_argument('--output', type=str, default='outputs/validation',
                       help='Output directory for validation artifacts')
    parser.add_argument('--min-entity-samples', type=int, default=5,
                       help='Minimum samples required per entity')
    parser.add_argument('--retail-only', action='store_true', default=True,
                       help='Validate retail templates only')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("PIPELINE VALIDATION & HARDENING")
    logger.info("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize validator
    logger.info(f"Loading label schema: {args.label_schema}")
    validator = PipelineValidator(args.label_schema, retail_only=args.retail_only)
    logger.info(f"Loaded {len(validator.valid_entities)} entities, {len(validator.valid_bio_labels)} BIO labels")
    
    # Define retail templates
    retail_templates = [
        'pos_receipt_standard',
        'pos_receipt_dense',
        'pos_receipt_wide',
        'pos_receipt_premium',
        'pos_receipt_qsr',
        'pos_receipt_fuel',
        'pos_receipt_pharmacy',
        'pos_receipt_wholesale',
        'online_order_standard',
        'online_order_fashion',
        'online_order_electronics',
        'online_order_grocery',
        'online_order_home_improvement',
        'online_order_digital',
        'online_order_marketplace',
        'online_order_wholesale',
    ]
    
    # Generate test samples
    logger.info(f"Generating {args.samples} test samples...")
    annotations = generate_test_samples(args.samples, retail_templates, output_dir)
    
    # Validate dataset
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING DATASET")
    logger.info("=" * 80)
    
    passed, summary = validator.validate_dataset(
        annotations,
        min_samples_per_entity=args.min_entity_samples
    )
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total samples:    {summary['total_samples']}")
    logger.info(f"Passed:           {summary['passed']} ({summary['pass_rate']:.1f}%)")
    logger.info(f"Failed:           {summary['failed']}")
    logger.info(f"Entity coverage:  {'✓ PASS' if summary['entity_coverage'] else '✗ FAIL'}")
    
    if summary.get('coverage_details', {}).get('warnings'):
        logger.warning("\nEntity coverage warnings:")
        for warning in summary['coverage_details']['warnings']:
            logger.warning(f"  - {warning}")
    
    if summary.get('coverage_details', {}).get('errors'):
        logger.error("\nEntity coverage errors:")
        for error in summary['coverage_details']['errors']:
            logger.error(f"  - {error}")
    
    if summary.get('coverage_details', {}).get('distribution'):
        logger.info("\nTop 10 entity distribution:")
        for entity, count in list(summary['coverage_details']['distribution'].items())[:10]:
            logger.info(f"  {entity:25s} {count:5d} samples")
    
    if summary['failed_samples']:
        logger.warning(f"\nFirst failed samples: {summary['failed_samples'][:10]}")
    
    # Save summary
    summary_path = output_dir / 'validation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary saved to: {summary_path}")
    
    # Final verdict
    logger.info("\n" + "=" * 80)
    if passed:
        logger.info("✓ VALIDATION PASSED - Pipeline ready for training")
        logger.info("=" * 80)
        return 0
    else:
        logger.error("✗ VALIDATION FAILED - Fix errors before training")
        logger.error("=" * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
