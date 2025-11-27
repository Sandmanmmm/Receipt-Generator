"""
Validate Label Schema
Ensures YAML is valid, BIO labels well-formed, no duplicates, complete consistency
"""
import yaml
import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple


def load_schema(schema_path: str) -> Dict:
    """Load label schema from YAML"""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def check_duplicates(label_list: List[str]) -> List[str]:
    """Check for duplicate labels"""
    counts = Counter(label_list)
    return [label for label, count in counts.items() if count > 1]


def check_bio_consistency(label_list: List[str]) -> List[str]:
    """Check for I- tags without corresponding B- tags"""
    errors = []
    b_entities = set()
    i_entities = set()
    
    for label in label_list:
        if label == 'O':
            continue
        if label.startswith('B-'):
            entity = label[2:]
            b_entities.add(entity)
        elif label.startswith('I-'):
            entity = label[2:]
            i_entities.add(entity)
        else:
            errors.append(f"Invalid label format: {label} (must be O, B-*, or I-*)")
    
    # Check for orphaned I- tags
    orphaned = i_entities - b_entities
    for entity in orphaned:
        errors.append(f"Orphaned I-tag: I-{entity} has no corresponding B-{entity}")
    
    return errors


def check_bio_pairs(label_list: List[str]) -> List[str]:
    """Check that every B- tag has an I- tag (multi-token entities)"""
    warnings = []
    b_entities = set()
    i_entities = set()
    
    for label in label_list:
        if label.startswith('B-'):
            b_entities.add(label[2:])
        elif label.startswith('I-'):
            i_entities.add(label[2:])
    
    # Entities with B- but no I- (single-token only - acceptable but worth noting)
    single_token_only = b_entities - i_entities
    if single_token_only:
        warnings.append(f"Single-token-only entities: {', '.join(sorted(single_token_only))}")
    
    return warnings


def extract_entities_from_schema(schema: Dict) -> Set[str]:
    """Extract all entity names from schema documentation"""
    entities = set()
    
    # From label_list
    for label in schema.get('label_list', []):
        if label.startswith('B-'):
            entities.add(label[2:])
    
    return entities


def check_entity_definitions(schema: Dict, entities: Set[str]) -> List[str]:
    """Check that all entities are documented"""
    errors = []
    
    # Check if entity_descriptions exists
    descriptions = schema.get('entity_descriptions', {})
    if not descriptions:
        errors.append("No entity_descriptions section found in schema")
        return errors
    
    # Check for missing descriptions
    missing_desc = entities - set(descriptions.keys())
    if missing_desc:
        errors.append(f"Missing descriptions for: {', '.join(sorted(missing_desc))}")
    
    # Check for extra descriptions
    extra_desc = set(descriptions.keys()) - entities
    if extra_desc:
        errors.append(f"Extra descriptions (no label): {', '.join(sorted(extra_desc))}")
    
    return errors


def validate_label_counts(schema: Dict) -> List[str]:
    """Validate that entity counts are correct"""
    errors = []
    
    label_list = schema.get('label_list', [])
    metadata = schema.get('metadata', {})
    
    # Count entities (B- tags)
    entities = [label[2:] for label in label_list if label.startswith('B-')]
    num_entities = len(entities)
    
    # Count BIO labels (excluding O)
    bio_labels = [label for label in label_list if label != 'O']
    num_bio_labels = len(bio_labels)
    
    # Check metadata counts
    if metadata.get('num_entities') != num_entities:
        errors.append(f"Entity count mismatch: metadata says {metadata.get('num_entities')}, "
                     f"but found {num_entities}")
    
    if metadata.get('num_bio_labels') != num_bio_labels:
        errors.append(f"BIO label count mismatch: metadata says {metadata.get('num_bio_labels')}, "
                     f"but found {num_bio_labels}")
    
    total_labels = len(label_list)
    if metadata.get('total_labels') != total_labels:
        errors.append(f"Total label count mismatch: metadata says {metadata.get('total_labels')}, "
                     f"but found {total_labels}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description='Validate label schema')
    parser.add_argument('--schema', type=str, default='config/labels_retail.yaml',
                       help='Path to label schema')
    args = parser.parse_args()
    
    print("=" * 80)
    print("LABEL SCHEMA VALIDATION")
    print("=" * 80)
    print(f"Schema: {args.schema}\n")
    
    # Load schema
    try:
        schema = load_schema(args.schema)
        print("[PASS] Schema loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load schema: {e}")
        return 1
    
    label_list = schema.get('label_list', [])
    if not label_list:
        print("✗ No label_list found in schema")
        return 1
    
    print(f"[PASS] Found {len(label_list)} labels\n")
    
    all_errors = []
    all_warnings = []
    
    # Test 1: Check for duplicates
    print("Test 1: Checking for duplicate labels...")
    duplicates = check_duplicates(label_list)
    if duplicates:
        all_errors.append(f"Duplicate labels found: {', '.join(duplicates)}")
        print(f"[FAIL] Found duplicates: {', '.join(duplicates)}")
    else:
        print("[PASS] No duplicates found")
    
    # Test 2: Check BIO consistency
    print("\nTest 2: Checking BIO tag consistency...")
    bio_errors = check_bio_consistency(label_list)
    if bio_errors:
        all_errors.extend(bio_errors)
        for error in bio_errors:
            print(f"[FAIL] {error}")
    else:
        print("[PASS] All BIO tags consistent")
    
    # Test 3: Check B-/I- pairing
    print("\nTest 3: Checking B-/I- pairing...")
    bio_warnings = check_bio_pairs(label_list)
    if bio_warnings:
        all_warnings.extend(bio_warnings)
        for warning in bio_warnings:
            print(f"[WARN] {warning}")
    else:
        print("[PASS] All entities have both B- and I- tags")
    
    # Test 4: Extract entities
    print("\nTest 4: Extracting entity list...")
    entities = extract_entities_from_schema(schema)
    print(f"[PASS] Found {len(entities)} unique entities")
    
    # Test 5: Check entity definitions
    print("\nTest 5: Checking entity descriptions...")
    desc_errors = check_entity_definitions(schema, entities)
    if desc_errors:
        all_warnings.extend(desc_errors)
        for error in desc_errors:
            print(f"[WARN] {error}")
    else:
        print("[PASS] All entities have descriptions")
    
    # Test 6: Validate counts
    print("\nTest 6: Validating metadata counts...")
    count_errors = validate_label_counts(schema)
    if count_errors:
        all_errors.extend(count_errors)
        for error in count_errors:
            print(f"[FAIL] {error}")
    else:
        print("[PASS] All counts match metadata")
    
    # Test 7: Check for 'O' label
    print("\nTest 7: Checking for 'O' (outside) label...")
    if 'O' not in label_list:
        all_errors.append("Missing 'O' label")
        print("[FAIL] No 'O' label found")
    else:
        print("[PASS] 'O' label present")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total labels: {len(label_list)}")
    print(f"Unique entities: {len(entities)}")
    print(f"Errors: {len(all_errors)}")
    print(f"Warnings: {len(all_warnings)}")
    
    if all_errors:
        print("\n[FAIL] VALIDATION FAILED")
        print("\nErrors:")
        for error in all_errors:
            print(f"  - {error}")
        return 1
    
    if all_warnings:
        print("\n[WARN] VALIDATION PASSED WITH WARNINGS")
        print("\nWarnings:")
        for warning in all_warnings:
            print(f"  - {warning}")
    else:
        print("\n[PASS] VALIDATION PASSED - Schema is production-ready!")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
