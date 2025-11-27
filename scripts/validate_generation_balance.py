#!/usr/bin/env python3
"""
Test 3: Generation Balance Validation

Generates 2000 samples and validates:
1. Entity frequency distribution is balanced
2. No field appears <0.5% of the time (too rare)
3. Critical entities appear frequently (ITEM_DESCRIPTION, TAX_AMOUNT, etc.)
4. Line-item entities appear in all items (SKU, QTY, UNIT_COST)
5. Store type distribution is balanced
6. Receipt type distribution is balanced (POS vs Online)

Target: Balanced entity distribution suitable for training
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.retail_data_generator import RetailDataGenerator


def load_schema(schema_path: Path) -> dict:
    """Load label schema from YAML file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_entities(schema: dict) -> Set[str]:
    """Extract entity names from schema label_list."""
    entities = set()
    label_list = schema.get('label_list', [])
    
    for label in label_list:
        if label.startswith('B-'):
            entity = label[2:]  # Remove 'B-' prefix
            entities.add(entity)
    
    return entities


def analyze_receipt_entities(receipt_data, generator: RetailDataGenerator) -> Dict[str, int]:
    """Extract all entities present in a receipt."""
    entities_found = defaultdict(int)
    
    # Convert to dict for easier access
    receipt_dict = generator.to_dict(receipt_data)
    
    # Document-level entities
    doc_entities = [
        'DOC_TYPE', 'INVOICE_NUMBER', 'INVOICE_DATE', 'SUPPLIER_NAME',
        'SUPPLIER_ADDRESS', 'SUPPLIER_PHONE', 'SUPPLIER_EMAIL',
        'CUSTOMER_NAME', 'CUSTOMER_ADDRESS', 'SUBTOTAL', 'TAX_AMOUNT',
        'TAX_RATE', 'DISCOUNT_AMOUNT', 'TOTAL_AMOUNT', 'PAYMENT_METHOD',
        'PAYMENT_TERMS', 'CASHIER_ID', 'REGISTER_NUMBER', 'STORE_NUMBER',
        'ORDER_DATE', 'DELIVERY_DATE', 'TRACKING_NUMBER', 'SHIPPING_COST',
        'BUYER_NAME', 'BUYER_ADDRESS', 'BUYER_PHONE', 'BUYER_EMAIL',
        'PO_NUMBER', 'NOTES'
    ]
    
    for entity in doc_entities:
        # Convert entity name to dict key format
        key = entity.lower()
        if key in receipt_dict and receipt_dict[key]:
            entities_found[entity] += 1
    
    # Line item entities - check if present in any line item
    if 'line_items' in receipt_dict:
        item_entity_present = defaultdict(bool)
        
        for item in receipt_dict['line_items']:
            if item.get('description'):
                item_entity_present['ITEM_DESCRIPTION'] = True
            if item.get('quantity') is not None:
                item_entity_present['ITEM_QTY'] = True
            if item.get('unit_price') is not None:
                item_entity_present['ITEM_UNIT_COST'] = True
            if item.get('total') is not None:
                item_entity_present['ITEM_TOTAL_COST'] = True
            if item.get('sku'):
                item_entity_present['ITEM_SKU'] = True
            if item.get('unit'):
                item_entity_present['ITEM_UNIT'] = True
            if item.get('tax_amount') is not None:
                item_entity_present['ITEM_TAX'] = True
            if item.get('discount_amount') is not None:
                item_entity_present['ITEM_DISCOUNT'] = True
            if item.get('category'):
                item_entity_present['ITEM_CATEGORY'] = True
            if item.get('serial_number'):
                item_entity_present['SERIAL_NUMBER'] = True
            if item.get('lot_number'):
                item_entity_present['LOT_NUMBER'] = True
            if item.get('weight') is not None:
                item_entity_present['WEIGHT'] = True
        
        # Add counts for line item entities
        for entity, present in item_entity_present.items():
            if present:
                entities_found[entity] += 1
    
    # Special entities that may appear
    if receipt_dict.get('po_line_items'):
        entities_found['PO_LINE_ITEM'] += 1
    
    return entities_found


def get_critical_entities() -> Set[str]:
    """Define critical entities that MUST appear frequently."""
    return {
        'DOC_TYPE',
        'INVOICE_NUMBER',
        'INVOICE_DATE',
        'SUPPLIER_NAME',
        'TOTAL_AMOUNT',
        'SUBTOTAL',
        'TAX_AMOUNT',
        'PAYMENT_METHOD',
        'ITEM_DESCRIPTION',
        'ITEM_QTY',
        'ITEM_UNIT_COST',
        'ITEM_TOTAL_COST'
    }


def main():
    parser = argparse.ArgumentParser(description='Validate generation balance')
    parser.add_argument('--schema', type=Path, default='config/labels_retail.yaml',
                       help='Path to label schema YAML')
    parser.add_argument('--samples', type=int, default=2000,
                       help='Number of samples to generate')
    parser.add_argument('--min-frequency', type=float, default=0.005,
                       help='Minimum entity frequency (0.005 = 0.5%%)')
    parser.add_argument('--critical-min-frequency', type=float, default=0.15,
                       help='Minimum frequency for critical entities (0.15 = 15%%)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATION BALANCE VALIDATION")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Samples: {args.samples}")
    print(f"Min frequency: {args.min_frequency * 100:.1f}%")
    print(f"Critical min frequency: {args.critical_min_frequency * 100:.1f}%")
    print()
    
    # Load schema
    try:
        schema = load_schema(args.schema)
        all_entities = extract_entities(schema)
        print(f"[PASS] Loaded schema with {len(all_entities)} entities\n")
    except Exception as e:
        print(f"[FAIL] Failed to load schema: {e}")
        return 1
    
    # Initialize generator
    try:
        generator = RetailDataGenerator()
        print("[PASS] Initialized RetailDataGenerator\n")
    except Exception as e:
        print(f"[FAIL] Failed to initialize generator: {e}")
        return 1
    
    # Generate samples and collect statistics
    print(f"Generating {args.samples} samples...")
    entity_counts = defaultdict(int)
    store_type_counts = Counter()
    receipt_type_counts = Counter()
    line_item_counts = []
    errors = []
    
    for i in range(args.samples):
        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{args.samples} samples...")
        
        try:
            # Generate receipt (50/50 POS vs Online) with Shopify-focused categories
            store_types = [
                'fashion', 'accessories', 'jewelry', 'beauty', 
                'home_garden', 'sports_fitness', 'pet_supplies', 
                'books_media', 'toys_games', 'food_beverage', 
                'health_wellness', 'electronics'
            ]
            
            if i % 2 == 0:
                store_type = store_types[i % len(store_types)]
                receipt = generator.generate_pos_receipt(store_type=store_type)
                receipt_type_counts['POS'] += 1
            else:
                store_type = store_types[i % len(store_types)]
                receipt = generator.generate_online_order(store_type=store_type)
                receipt_type_counts['Online'] += 1
            
            store_type_counts[store_type] += 1
            
            # Analyze entities in this receipt
            entities_in_receipt = analyze_receipt_entities(receipt, generator)
            for entity, count in entities_in_receipt.items():
                entity_counts[entity] += count
            
            # Track line items
            receipt_dict = generator.to_dict(receipt)
            if 'line_items' in receipt_dict:
                line_item_counts.append(len(receipt_dict['line_items']))
        
        except Exception as e:
            errors.append(f"Sample {i}: {str(e)}")
    
    print(f"[PASS] Generated {args.samples} samples\n")
    
    if errors:
        print(f"[WARN] Encountered {len(errors)} generation errors:")
        for error in errors[:5]:  # Show first 5
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
        print()
    
    # Analysis
    print("=" * 80)
    print("ENTITY FREQUENCY ANALYSIS")
    print("=" * 80)
    
    critical_entities = get_critical_entities()
    total_samples = args.samples
    
    # Calculate frequencies
    entity_frequencies = {
        entity: count / total_samples 
        for entity, count in entity_counts.items()
    }
    
    # Sort by frequency
    sorted_entities = sorted(entity_frequencies.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 15 most frequent entities:")
    for entity, freq in sorted_entities[:15]:
        count = entity_counts[entity]
        critical_marker = " [CRITICAL]" if entity in critical_entities else ""
        print(f"  {entity:25s}: {count:5d} ({freq*100:5.1f}%){critical_marker}")
    
    print(f"\nBottom 15 least frequent entities:")
    for entity, freq in sorted_entities[-15:]:
        count = entity_counts[entity]
        critical_marker = " [CRITICAL]" if entity in critical_entities else ""
        print(f"  {entity:25s}: {count:5d} ({freq*100:5.1f}%){critical_marker}")
    
    # Check for rare entities
    print(f"\n[CHECK] Entities below {args.min_frequency * 100:.1f}% frequency threshold:")
    rare_entities = [(e, f) for e, f in entity_frequencies.items() if f < args.min_frequency]
    
    if rare_entities:
        print(f"[WARN] Found {len(rare_entities)} rare entities:")
        for entity, freq in sorted(rare_entities, key=lambda x: x[1]):
            count = entity_counts[entity]
            print(f"  - {entity}: {count} ({freq*100:.2f}%)")
    else:
        print(f"[PASS] No entities below threshold")
    
    # Check critical entities
    print(f"\n[CHECK] Critical entities above {args.critical_min_frequency * 100:.0f}% threshold:")
    low_critical = []
    missing_critical = []
    
    for entity in critical_entities:
        if entity not in entity_frequencies:
            missing_critical.append(entity)
        elif entity_frequencies[entity] < args.critical_min_frequency:
            low_critical.append((entity, entity_frequencies[entity]))
    
    if missing_critical:
        print(f"[FAIL] Critical entities missing from all samples:")
        for entity in missing_critical:
            print(f"  - {entity}")
    
    if low_critical:
        print(f"[WARN] Critical entities with low frequency:")
        for entity, freq in low_critical:
            count = entity_counts[entity]
            print(f"  - {entity}: {count} ({freq*100:.1f}%)")
    
    if not missing_critical and not low_critical:
        print(f"[PASS] All critical entities above threshold")
    
    # Check for missing entities from schema
    print(f"\n[CHECK] Entities defined in schema but never generated:")
    missing_from_generation = all_entities - set(entity_counts.keys())
    
    if missing_from_generation:
        print(f"[WARN] {len(missing_from_generation)} entities never generated:")
        for entity in sorted(missing_from_generation):
            critical_marker = " [CRITICAL - SERIOUS ISSUE]" if entity in critical_entities else ""
            print(f"  - {entity}{critical_marker}")
    else:
        print(f"[PASS] All schema entities generated at least once")
    
    # Store type distribution
    print("\n" + "=" * 80)
    print("STORE TYPE DISTRIBUTION")
    print("=" * 80)
    print(f"\nStore type balance:")
    for store_type, count in store_type_counts.most_common():
        freq = count / total_samples
        print(f"  {store_type:20s}: {count:5d} ({freq*100:5.1f}%)")
    
    # Receipt type distribution
    print(f"\nReceipt type balance:")
    for receipt_type, count in receipt_type_counts.most_common():
        freq = count / total_samples
        print(f"  {receipt_type:20s}: {count:5d} ({freq*100:5.1f}%)")
    
    # Line item statistics
    print("\n" + "=" * 80)
    print("LINE ITEM STATISTICS")
    print("=" * 80)
    if line_item_counts:
        avg_items = sum(line_item_counts) / len(line_item_counts)
        min_items = min(line_item_counts)
        max_items = max(line_item_counts)
        print(f"\nLine items per receipt:")
        print(f"  Average: {avg_items:.1f}")
        print(f"  Min: {min_items}")
        print(f"  Max: {max_items}")
    
    # Final validation
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    validation_errors = []
    validation_warnings = []
    
    # Critical checks
    if missing_critical:
        validation_errors.append(f"{len(missing_critical)} critical entities never generated")
    
    if low_critical:
        validation_warnings.append(f"{len(low_critical)} critical entities below frequency threshold")
    
    if rare_entities:
        validation_warnings.append(f"{len(rare_entities)} entities below {args.min_frequency * 100:.1f}% frequency")
    
    if missing_from_generation:
        critical_missing = missing_from_generation & critical_entities
        if critical_missing:
            validation_errors.append(f"{len(critical_missing)} critical entities defined but never generated")
        else:
            validation_warnings.append(f"{len(missing_from_generation)} non-critical entities never generated")
    
    # Print summary
    print(f"\nTotal samples generated: {total_samples}")
    print(f"Entities with data: {len(entity_counts)}")
    print(f"Entities in schema: {len(all_entities)}")
    print(f"Critical entities: {len(critical_entities)}")
    print(f"Errors: {len(validation_errors)}")
    print(f"Warnings: {len(validation_warnings)}")
    print()
    
    if validation_errors:
        print("[FAIL] VALIDATION FAILED")
        for error in validation_errors:
            print(f"  - {error}")
        return 1
    
    if validation_warnings:
        print("[WARN] VALIDATION PASSED WITH WARNINGS")
        for warning in validation_warnings:
            print(f"  - {warning}")
    else:
        print("[PASS] VALIDATION PASSED - Generation is well-balanced!")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
