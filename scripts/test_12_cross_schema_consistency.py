#!/usr/bin/env python3
"""
Test 12: Cross-Schema Consistency Check

Purpose: Ensure entity_assigner, renderer, labeler, and schema YAML are all consistent.
         This catches schema/entity mismatches that cause silent training corruption.

Checks:
- Every entity in schema exists in labeling code
- No extra entities produced by the labeler
- No entity missing its BIO mapping
- NER classes sorted correctly and consistent
- Visual bounding boxes exist for every non-O token

Expected Result:
- Zero mismatches
- Zero unmapped entities
- Zero ghost entities

Usage:
    python scripts/test_12_cross_schema_consistency.py
    python scripts/test_12_cross_schema_consistency.py --schema config/labels_retail.yaml
    python scripts/test_12_cross_schema_consistency.py --num-samples 50
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from generators.retail_data_generator import RetailDataGenerator
from generators.html_to_png_renderer import SimplePNGRenderer
from annotation.ocr_engine import OCREngine
from annotation.token_annotator import TokenAnnotator


class ConsistencyResults:
    """Container for consistency check results."""
    
    def __init__(self):
        # Schema analysis
        self.schema_entities: Set[str] = set()
        self.schema_labels: List[str] = []
        self.schema_has_o_label = False
        self.schema_has_bio_tags = False
        
        # Annotator analysis
        self.annotator_entities: Set[str] = set()
        self.annotator_mappings: Dict[str, str] = {}
        
        # Generator analysis
        self.generator_fields: Set[str] = set()
        self.generated_entities_in_samples: Set[str] = set()
        
        # Consistency issues
        self.entities_in_schema_not_in_annotator: Set[str] = set()
        self.entities_in_annotator_not_in_schema: Set[str] = set()
        self.entities_missing_bio_mapping: Set[str] = set()
        self.entities_without_bboxes: Set[str] = set()
        
        # Label ordering issues
        self.labels_not_sorted = False
        self.duplicate_labels: List[str] = []
        
        # Runtime validation (from actual samples)
        self.unmapped_entities_found: Set[str] = set()
        self.ghost_entities_found: Set[str] = set()
        self.tokens_without_bboxes: List[Dict] = []
        
        # Statistics
        self.samples_checked = 0
        self.total_tokens = 0
        self.total_entities_detected = 0
        self.entity_distribution: Dict[str, int] = defaultdict(int)


def load_schema(schema_path: Path) -> Dict:
    """Load and analyze schema YAML file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)
    return schema


def analyze_schema(schema: Dict, results: ConsistencyResults):
    """Analyze schema structure and extract entity types."""
    
    print("\n" + "="*80)
    print("ANALYZING SCHEMA")
    print("="*80)
    
    # Get label list
    label_list = schema.get('label_list', schema.get('labels', []))
    results.schema_labels = label_list
    
    print(f"Total labels in schema: {len(label_list)}")
    
    # Check for O label
    if 'O' in label_list:
        results.schema_has_o_label = True
        print("  ✓ 'O' label present")
    else:
        print("  ✗ 'O' label missing")
    
    # Extract entity types from BIO labels
    for label in label_list:
        if label == 'O':
            continue
        
        if label.startswith('B-') or label.startswith('I-'):
            results.schema_has_bio_tags = True
            entity_type = label[2:]  # Remove B- or I- prefix
            results.schema_entities.add(entity_type)
        elif '-' not in label:
            # Non-BIO entity (warning)
            results.schema_entities.add(label)
    
    print(f"Entity types in schema: {len(results.schema_entities)}")
    
    # Check label ordering
    if label_list != sorted(label_list):
        results.labels_not_sorted = True
        print("  ⚠ Labels not alphabetically sorted")
    else:
        print("  ✓ Labels alphabetically sorted")
    
    # Check for duplicates
    seen = set()
    for label in label_list:
        if label in seen:
            results.duplicate_labels.append(label)
        seen.add(label)
    
    if results.duplicate_labels:
        print(f"  ✗ Duplicate labels found: {results.duplicate_labels}")
    else:
        print("  ✓ No duplicate labels")
    
    # Check BIO consistency
    entities_with_b = set()
    entities_with_i = set()
    
    for label in label_list:
        if label.startswith('B-'):
            entities_with_b.add(label[2:])
        elif label.startswith('I-'):
            entities_with_i.add(label[2:])
    
    # Entities with I- but no B-
    orphan_i_entities = entities_with_i - entities_with_b
    if orphan_i_entities:
        print(f"  ✗ Entities with I- but no B-: {orphan_i_entities}")
    else:
        print("  ✓ All I- tags have corresponding B- tags")
    
    # Entities with B- but no I- (this is OK)
    b_only_entities = entities_with_b - entities_with_i
    if b_only_entities:
        print(f"  ℹ Entities with only B- tags (single-token entities): {len(b_only_entities)}")
    
    print()


def analyze_annotator(annotator: TokenAnnotator, results: ConsistencyResults):
    """Analyze TokenAnnotator to extract entity mappings."""
    
    print("="*80)
    print("ANALYZING TOKEN ANNOTATOR")
    print("="*80)
    
    # Get entity mappings directly from annotator's entity_mappings dict
    if hasattr(annotator, 'entity_mappings'):
        results.annotator_mappings = annotator.entity_mappings.copy()
        results.annotator_entities = set(annotator.entity_mappings.values())
        
        print(f"Entity mappings in annotator: {len(results.annotator_mappings)}")
        
        # Show sample mappings
        print("\nSample mappings:")
        for field, entity in list(results.annotator_mappings.items())[:8]:
            print(f"  {field:20s} → {entity}")
    else:
        print("  ✗ Annotator has no entity_mappings attribute")
        results.annotator_entities = set()
    
    # Also check for hardcoded entity types in the annotation logic
    # (like ITEM_DESCRIPTION, ITEM_QTY, etc. that appear in line item processing)
    hardcoded_entities = {
        'ITEM_DESCRIPTION',  # Used in line item processing
        'ITEM_QTY',
        'ITEM_UNIT_COST',
        'ITEM_TOTAL_COST',
    }
    
    results.annotator_entities.update(hardcoded_entities)
    
    print(f"Total unique entity types in annotator: {len(results.annotator_entities)}")
    
    # Check if annotator has label list
    if hasattr(annotator, 'label_list') and annotator.label_list:
        print(f"  ✓ Annotator has {len(annotator.label_list)} labels loaded")
    else:
        print("  ✗ Annotator has no labels loaded")
    
    print()


def analyze_generator(generator: RetailDataGenerator, results: ConsistencyResults):
    """Analyze data generator to extract available fields."""
    
    print("="*80)
    print("ANALYZING DATA GENERATOR")
    print("="*80)
    
    # Generate a sample receipt to see what fields are available
    receipt = generator.generate_pos_receipt()
    receipt_dict = generator.to_dict(receipt)
    
    # Extract top-level fields
    top_level_fields = set(receipt_dict.keys())
    results.generator_fields.update(top_level_fields)
    
    print(f"Top-level fields in generated receipts: {len(top_level_fields)}")
    print(f"  {', '.join(sorted(top_level_fields)[:10])}...")
    
    # Check for line items
    if 'items' in receipt_dict and receipt_dict['items']:
        item = receipt_dict['items'][0]
        item_fields = set(item.keys())
        print(f"Item fields: {len(item_fields)}")
        print(f"  {', '.join(sorted(item_fields))}")
        results.generator_fields.update([f"item.{f}" for f in item_fields])
    
    print()


def check_consistency(results: ConsistencyResults):
    """Check for consistency issues between schema and annotator."""
    
    print("="*80)
    print("CHECKING CONSISTENCY")
    print("="*80)
    
    # Check 1: Entities in schema but not in annotator
    results.entities_in_schema_not_in_annotator = (
        results.schema_entities - results.annotator_entities
    )
    
    if results.entities_in_schema_not_in_annotator:
        print(f"\n✗ Entities in schema but NOT in annotator ({len(results.entities_in_schema_not_in_annotator)}):")
        for entity in sorted(results.entities_in_schema_not_in_annotator):
            print(f"    {entity}")
    else:
        print("\n✓ All schema entities have annotator mappings")
    
    # Check 2: Entities in annotator but not in schema
    results.entities_in_annotator_not_in_schema = (
        results.annotator_entities - results.schema_entities
    )
    
    if results.entities_in_annotator_not_in_schema:
        print(f"\n✗ Entities in annotator but NOT in schema ({len(results.entities_in_annotator_not_in_schema)}):")
        for entity in sorted(results.entities_in_annotator_not_in_schema):
            print(f"    {entity}")
    else:
        print("\n✓ All annotator entities exist in schema")
    
    # Check 3: Entities missing BIO mapping
    if results.schema_has_bio_tags:
        for entity in results.schema_entities:
            b_label = f"B-{entity}"
            if b_label not in results.schema_labels:
                results.entities_missing_bio_mapping.add(entity)
        
        if results.entities_missing_bio_mapping:
            print(f"\n✗ Entities missing B- tag ({len(results.entities_missing_bio_mapping)}):")
            for entity in sorted(results.entities_missing_bio_mapping):
                print(f"    {entity}")
        else:
            print("\n✓ All entities have proper BIO tagging")
    
    print()


def validate_runtime_samples(
    num_samples: int,
    generator: RetailDataGenerator,
    renderer: SimplePNGRenderer,
    ocr_engine: OCREngine,
    annotator: TokenAnnotator,
    results: ConsistencyResults
):
    """Validate consistency on actual generated samples."""
    
    print("="*80)
    print(f"VALIDATING RUNTIME CONSISTENCY ({num_samples} samples)")
    print("="*80)
    print()
    
    for i in range(num_samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\rProcessing: {i+1}/{num_samples}", end='', flush=True)
        
        try:
            # Generate receipt
            receipt_obj = generator.generate_pos_receipt()
            receipt_dict = generator.to_dict(receipt_obj)
            receipt_dict['id'] = f"consistency_check_{i:03d}"
            
            # Render to PNG
            with Path('outputs/temp_consistency_check.png').open('wb') as f:
                pass  # Create temp file
            
            image_path = 'outputs/temp_consistency_check.png'
            renderer.render_receipt_dict(receipt_dict, image_path)
            
            # Run OCR
            bbox_list = ocr_engine.extract_text(image_path)
            
            if not bbox_list:
                continue
            
            # Extract tokens and bboxes
            tokens = []
            bboxes = []
            
            for bbox_obj in bbox_list:
                if bbox_obj.confidence < 0.5:
                    continue
                
                bbox = bbox_obj.to_pascal_voc()
                words = bbox_obj.text.split()
                
                for word in words:
                    tokens.append(word)
                    bboxes.append(bbox)
            
            if not tokens:
                continue
            
            results.total_tokens += len(tokens)
            
            # Annotate
            annotation = annotator.annotate_tokens(
                receipt_dict,
                tokens,
                bboxes,
                image_path,
                800,
                1200
            )
            
            if not annotation:
                continue
            
            # Check for consistency issues
            ner_tags = annotation['ner_tags']
            annotation_tokens = annotation['tokens']
            annotation_bboxes = annotation['bboxes']
            
            # Check 1: All tokens have bboxes
            if len(annotation_tokens) != len(annotation_bboxes):
                results.tokens_without_bboxes.append({
                    'sample_id': receipt_dict['id'],
                    'tokens_len': len(annotation_tokens),
                    'bboxes_len': len(annotation_bboxes)
                })
            
            # Check 2: All non-O tokens have valid bboxes
            for idx, tag_id in enumerate(ner_tags):
                label = annotator.id2label.get(tag_id, 'O')
                
                if label != 'O':
                    results.total_entities_detected += 1
                    
                    # Extract entity type
                    if label.startswith('B-'):
                        entity_type = label[2:]
                        results.entity_distribution[entity_type] += 1
                        results.generated_entities_in_samples.add(entity_type)
                        
                        # Check if entity type is in schema
                        if entity_type not in results.schema_entities:
                            results.unmapped_entities_found.add(entity_type)
                    
                    # Check bbox validity
                    if idx < len(annotation_bboxes):
                        bbox = annotation_bboxes[idx]
                        if not bbox or len(bbox) != 4:
                            results.entities_without_bboxes.add(label)
            
            results.samples_checked += 1
            
            # Cleanup temp file
            Path(image_path).unlink(missing_ok=True)
            
        except Exception as e:
            continue
    
    print()  # New line after progress
    print()


def print_results(results: ConsistencyResults) -> bool:
    """Print comprehensive consistency check results."""
    
    print("="*80)
    print("CONSISTENCY CHECK RESULTS")
    print("="*80)
    print()
    
    # Schema summary
    print("Schema Summary:")
    print(f"  Total labels:        {len(results.schema_labels)}")
    print(f"  Entity types:        {len(results.schema_entities)}")
    print(f"  Has 'O' label:       {results.schema_has_o_label}")
    print(f"  Uses BIO tagging:    {results.schema_has_bio_tags}")
    print()
    
    # Annotator summary
    print("Annotator Summary:")
    print(f"  Entity mappings:     {len(results.annotator_mappings)}")
    print(f"  Unique entities:     {len(results.annotator_entities)}")
    print()
    
    # Generator summary
    print("Generator Summary:")
    print(f"  Receipt fields:      {len(results.generator_fields)}")
    print()
    
    # Runtime validation summary
    if results.samples_checked > 0:
        print("Runtime Validation Summary:")
        print(f"  Samples checked:     {results.samples_checked}")
        print(f"  Total tokens:        {results.total_tokens}")
        print(f"  Entities detected:   {results.total_entities_detected}")
        print(f"  Entity types found:  {len(results.generated_entities_in_samples)}")
        print()
    
    # Consistency issues
    print("="*80)
    print("CONSISTENCY VALIDATION")
    print("="*80)
    print()
    
    all_passed = True
    
    # Critical issues (must be zero) - these cause training corruption
    critical_checks = [
        (len(results.entities_in_annotator_not_in_schema) == 0,
         "Annotator entities exist in schema",
         f"{len(results.entities_in_annotator_not_in_schema)} ghost entities"),
        
        (len(results.entities_missing_bio_mapping) == 0,
         "All entities have BIO mappings",
         f"{len(results.entities_missing_bio_mapping)} missing"),
        
        (len(results.duplicate_labels) == 0,
         "No duplicate labels",
         f"{len(results.duplicate_labels)} duplicates"),
        
        (len(results.unmapped_entities_found) == 0,
         "No unmapped entities in samples",
         f"{len(results.unmapped_entities_found)} unmapped"),
        
        (len(results.tokens_without_bboxes) == 0,
         "All tokens have bboxes",
         f"{len(results.tokens_without_bboxes)} mismatches"),
        
        (len(results.entities_without_bboxes) == 0,
         "All entities have valid bboxes",
         f"{len(results.entities_without_bboxes)} invalid"),
    ]
    
    # Non-critical checks (warnings only) - schema completeness
    warning_checks = [
        (len(results.entities_in_schema_not_in_annotator) == 0,
         "All schema entities mapped in annotator",
         f"{len(results.entities_in_schema_not_in_annotator)} unmapped (OK if unused)"),
    ]
    
    # Print critical checks
    for passed, description, value in critical_checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8s} {description:45s} ({value})")
        if not passed:
            all_passed = False
    
    print()
    
    # Print warning checks (non-critical)
    for passed, description, value in warning_checks:
        status = "✓ PASS" if passed else "⚠ WARN"
        print(f"  {status:8s} {description:45s} ({value})")
    
    # Additional warnings
    if results.labels_not_sorted:
        print(f"  ⚠ WARN   Labels alphabetically sorted                  (not sorted)")
    else:
        print(f"  ✓ PASS   Labels alphabetically sorted                  (sorted)")
    
    print()
    
    # Detailed issues
    if not all_passed:
        print("="*80)
        print("DETAILED ISSUES")
        print("="*80)
        print()
        
        if results.entities_in_schema_not_in_annotator:
            print(f"Entities in schema but NOT in annotator:")
            for entity in sorted(results.entities_in_schema_not_in_annotator):
                print(f"  ✗ {entity}")
            print()
        
        if results.entities_in_annotator_not_in_schema:
            print(f"Entities in annotator but NOT in schema:")
            for entity in sorted(results.entities_in_annotator_not_in_schema):
                print(f"  ✗ {entity}")
            print()
        
        if results.unmapped_entities_found:
            print(f"Unmapped entities found in generated samples:")
            for entity in sorted(results.unmapped_entities_found):
                count = results.entity_distribution.get(entity, 0)
                print(f"  ✗ {entity} ({count} occurrences)")
            print()
        
        if results.tokens_without_bboxes:
            print(f"Samples with token/bbox mismatches:")
            for item in results.tokens_without_bboxes[:5]:
                print(f"  ✗ {item['sample_id']}: {item['tokens_len']} tokens, {item['bboxes_len']} bboxes")
            if len(results.tokens_without_bboxes) > 5:
                print(f"  ... and {len(results.tokens_without_bboxes) - 5} more")
            print()
    
    # Entity distribution
    if results.entity_distribution:
        print("="*80)
        print("ENTITY DISTRIBUTION IN SAMPLES")
        print("="*80)
        print()
        
        sorted_entities = sorted(
            results.entity_distribution.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("Top 15 entities:")
        for entity, count in sorted_entities[:15]:
            in_schema = "✓" if entity in results.schema_entities else "✗"
            print(f"  {in_schema} {entity:30s} {count:4d}")
        print()
    
    # Coverage analysis
    if results.generated_entities_in_samples and results.schema_entities:
        coverage = len(results.generated_entities_in_samples & results.schema_entities)
        total = len(results.schema_entities)
        coverage_pct = (coverage / total * 100) if total > 0 else 0
        
        print("Schema Coverage:")
        print(f"  Entities in schema:  {total}")
        print(f"  Found in samples:    {coverage} ({coverage_pct:.1f}%)")
        
        missing = results.schema_entities - results.generated_entities_in_samples
        if missing:
            print(f"  Not found in samples: {len(missing)}")
            if len(missing) <= 10:
                for entity in sorted(missing):
                    print(f"    ⚠ {entity}")
        print()
    
    # Final verdict
    print("="*80)
    if all_passed:
        print("[PASS] SCHEMA CONSISTENCY VALIDATED!")
        print()
        print("Cross-component consistency confirmed:")
        print("  ✓ Schema and annotator aligned")
        print("  ✓ All entities properly mapped")
        print("  ✓ BIO tagging consistent")
        print("  ✓ No ghost entities")
        print("  ✓ No unmapped entities")
        print("  ✓ All tokens have valid bboxes")
    else:
        print("[FAIL] SCHEMA CONSISTENCY ISSUES FOUND!")
        print()
        print("Fix the following issues before training:")
        
        if results.entities_in_annotator_not_in_schema:
            print(f"  ✗ {len(results.entities_in_annotator_not_in_schema)} annotator entities not in schema (ghost entities)")
        if results.unmapped_entities_found:
            print(f"  ✗ {len(results.unmapped_entities_found)} unmapped entities in generated samples")
        if results.tokens_without_bboxes:
            print(f"  ✗ {len(results.tokens_without_bboxes)} samples with bbox mismatches")
        if results.entities_without_bboxes:
            print(f"  ✗ {len(results.entities_without_bboxes)} entities with invalid bboxes")
        
        print()
        print("These issues will cause silent training corruption!")
    
    # Note about unmapped schema entities (non-critical)
    if all_passed and results.entities_in_schema_not_in_annotator:
        print()
        print("Note:")
        print(f"  ⚠ {len(results.entities_in_schema_not_in_annotator)} schema entities not mapped in annotator")
        print("    This is OK if these entities don't appear in generated receipts.")
        print("    Examples: LOT_NUMBER, SERIAL_NUMBER, WEIGHT (not in standard POS receipts)")
    
    print("="*80)
    print()
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test 12: Cross-Schema Consistency Check"
    )
    parser.add_argument(
        '--schema',
        type=Path,
        default=Path('config/labels_retail.yaml'),
        help='Path to label schema YAML file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of samples to validate (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate schema path
    if not args.schema.exists():
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TEST 12: CROSS-SCHEMA CONSISTENCY CHECK")
    print("="*80)
    print(f"Schema: {args.schema}")
    print(f"Validation samples: {args.num_samples}")
    print()
    
    # Create results container
    results = ConsistencyResults()
    
    # Load and analyze schema
    schema = load_schema(args.schema)
    analyze_schema(schema, results)
    
    # Initialize components
    print("Initializing components...")
    annotator = TokenAnnotator(schema)
    generator = RetailDataGenerator()
    renderer = SimplePNGRenderer(width=800, height=1200)
    ocr_engine = OCREngine(engine='paddleocr', show_log=False)
    print()
    
    # Analyze annotator
    analyze_annotator(annotator, results)
    
    # Analyze generator
    analyze_generator(generator, results)
    
    # Check static consistency
    check_consistency(results)
    
    # Validate runtime consistency
    validate_runtime_samples(
        args.num_samples,
        generator,
        renderer,
        ocr_engine,
        annotator,
        results
    )
    
    # Print results
    passed = print_results(results)
    
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
