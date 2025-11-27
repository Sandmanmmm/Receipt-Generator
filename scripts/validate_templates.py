"""
Validate Template Coverage
Ensures every HTML/Jinja template populates ALL required retail fields
"""
import yaml
import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


def load_schema(schema_path: str) -> Dict:
    """Load label schema from YAML"""
    with open(schema_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def extract_entities(schema: Dict) -> Set[str]:
    """Extract entity names from schema"""
    entities = set()
    for label in schema.get('label_list', []):
        if label.startswith('B-'):
            entities.add(label[2:])
    return entities


def entity_to_template_var(entity: str) -> List[str]:
    """Map entity name to possible template variable names"""
    # Convert SCREAMING_SNAKE to snake_case
    var_name = entity.lower()
    
    variations = [var_name]
    
    # Add common variations
    if entity == 'INVOICE_NUMBER':
        variations.extend(['invoice_number', 'receipt_number', 'order_number', 'invoice_no'])
    elif entity == 'INVOICE_DATE':
        variations.extend(['invoice_date', 'date', 'order_date'])
    elif entity == 'SUPPLIER_NAME':
        variations.extend(['supplier_name', 'store_name', 'company_name', 'merchant_name'])
    elif entity == 'SUPPLIER_ADDRESS':
        variations.extend(['supplier_address', 'store_address', 'company_address'])
    elif entity == 'SUPPLIER_PHONE':
        variations.extend(['supplier_phone', 'store_contact', 'phone'])
    elif entity == 'SUPPLIER_EMAIL':
        variations.extend(['supplier_email', 'email', 'contact_email'])
    elif entity == 'TOTAL_AMOUNT':
        variations.extend(['total_amount', 'total', 'grand_total'])
    elif entity == 'SUBTOTAL':
        variations.extend(['subtotal', 'sub_total'])
    elif entity == 'TAX_AMOUNT':
        variations.extend(['tax_amount', 'tax'])
    elif entity == 'TAX_RATE':
        variations.extend(['tax_rate'])
    elif entity == 'PAYMENT_METHOD':
        variations.extend(['payment_method', 'payment_type'])
    elif entity == 'REGISTER_NUMBER':
        variations.extend(['register_number', 'register'])
    elif entity == 'CASHIER_ID':
        variations.extend(['cashier_id', 'cashier', 'cashier_name'])
    elif entity == 'TRACKING_NUMBER':
        variations.extend(['tracking_number', 'tracking'])
    elif entity == 'LINE_ITEMS' or entity == 'TABLE':
        variations.extend(['line_items', 'items', 'order_items'])
    elif entity == 'ITEM_QTY':
        variations.extend(['item.quantity', 'quantity', 'item_qty', 'qty'])
    elif entity == 'ITEM_UNIT_COST':
        variations.extend(['item.unit_price', 'unit_price', 'item_unit_cost', 'price'])
    elif entity == 'ITEM_TOTAL_COST':
        variations.extend(['item.total', 'total', 'item_total_cost', 'item_total', 'amount'])
    elif entity == 'ITEM_DESCRIPTION':
        variations.extend(['item.description', 'description', 'item_description', 'product'])
    elif entity == 'ITEM_SKU':
        variations.extend(['item.upc', 'item.sku', 'sku', 'upc'])
    elif entity == 'ITEM_UNIT':
        variations.extend(['item.unit', 'unit'])
    elif entity == 'ITEM_DISCOUNT':
        variations.extend(['item.discount', 'discount'])
    elif entity == 'ITEM_TAX':
        variations.extend(['item.tax', 'tax'])
    elif entity == 'PO_LINE_ITEM':
        variations.extend(['line_item', 'item', 'line'])
    elif entity == 'DOC_TYPE':
        variations.extend(['doc_type', 'document_type', 'receipt', 'invoice'])
    elif entity == 'CURRENCY':
        variations.extend(['currency', 'currency_symbol', '$', '€', '£'])
    elif entity == 'GENERIC_LABEL':
        variations.extend(['generic', 'misc', 'other'])
    elif entity.startswith('ITEM_'):
        item_field = entity[5:].lower()
        variations.extend([
            f'item.{item_field}',
            item_field,
            f'{item_field}'
        ])
    
    return variations


def scan_template(template_path: Path) -> Set[str]:
    """Scan template for variable references"""
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all Jinja2 variables: {{ variable }} or {{ item.field }}
    pattern = r'\{\{\s*([a-z_][a-z0-9_\.]*)\s*\}\}'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    # Also find variables in {% for %} loops
    for_pattern = r'\{%\s*for\s+\w+\s+in\s+([a-z_][a-z0-9_\.]*)\s*%\}'
    for_matches = re.findall(for_pattern, content, re.IGNORECASE)
    matches.extend(for_matches)
    
    # Also find {% if %} conditions
    if_pattern = r'\{%\s*if\s+([a-z_][a-z0-9_\.]*)'
    if_matches = re.findall(if_pattern, content, re.IGNORECASE)
    matches.extend(if_matches)
    
    return set(matches)


def check_template_coverage(template_path: Path, entities: Set[str]) -> Dict:
    """Check which entities are covered by template"""
    template_vars = scan_template(template_path)
    
    covered = set()
    missing = set()
    
    for entity in entities:
        variations = entity_to_template_var(entity)
        found = False
        for var in variations:
            if any(var in tv for tv in template_vars):
                found = True
                break
        
        if found:
            covered.add(entity)
        else:
            missing.add(entity)
    
    return {
        'covered': covered,
        'missing': missing,
        'variables': template_vars
    }


def get_critical_entities() -> Set[str]:
    """Return set of critical retail entities that MUST appear"""
    return {
        'INVOICE_NUMBER', 'INVOICE_DATE', 
        'SUPPLIER_NAME', 'TOTAL_AMOUNT', 'SUBTOTAL', 'TAX_AMOUNT',
        'PAYMENT_METHOD', 'ITEM_DESCRIPTION', 'ITEM_QTY', 
        'ITEM_UNIT_COST', 'ITEM_TOTAL_COST'
    }


def main():
    parser = argparse.ArgumentParser(description='Validate template coverage')
    parser.add_argument('--schema', type=str, default='config/labels_retail.yaml',
                       help='Path to label schema')
    parser.add_argument('--templates-dir', type=str, default='templates/retail',
                       help='Path to templates directory')
    args = parser.parse_args()
    
    print("=" * 80)
    print("TEMPLATE COVERAGE VALIDATION")
    print("=" * 80)
    print(f"Schema: {args.schema}")
    print(f"Templates: {args.templates_dir}\n")
    
    # Load schema
    try:
        schema = load_schema(args.schema)
        entities = extract_entities(schema)
        print(f"[PASS] Loaded schema with {len(entities)} entities")
    except Exception as e:
        print(f"[FAIL] Failed to load schema: {e}")
        return 1
    
    # Find templates
    templates_dir = Path(args.templates_dir)
    if not templates_dir.exists():
        print(f"[FAIL] Templates directory not found: {templates_dir}")
        return 1
    
    templates = list(templates_dir.glob('*.html'))
    if not templates:
        print(f"[FAIL] No templates found in {templates_dir}")
        return 1
    
    print(f"[PASS] Found {len(templates)} templates\n")
    
    # Analyze each template
    all_covered = set()
    template_results = {}
    critical_entities = get_critical_entities()
    
    for template_path in sorted(templates):
        print(f"Analyzing: {template_path.name}")
        result = check_template_coverage(template_path, entities)
        template_results[template_path.name] = result
        all_covered.update(result['covered'])
        
        coverage_pct = (len(result['covered']) / len(entities)) * 100
        print(f"  Coverage: {len(result['covered'])}/{len(entities)} ({coverage_pct:.1f}%)")
        
        # Check critical entities
        missing_critical = critical_entities & result['missing']
        if missing_critical:
            print(f"  [WARN] Missing critical: {', '.join(sorted(missing_critical))}")
        else:
            print(f"  [PASS] All critical entities present")
        print()
    
    # Overall coverage
    print("=" * 80)
    print("OVERALL COVERAGE")
    print("=" * 80)
    
    never_covered = entities - all_covered
    
    print(f"Total entities: {len(entities)}")
    print(f"Covered by at least one template: {len(all_covered)}")
    print(f"Never covered: {len(never_covered)}")
    
    if never_covered:
        print(f"\n[WARN] Entities not covered by ANY template:")
        for entity in sorted(never_covered):
            print(f"  - {entity}")
    
    # Critical entity coverage
    print(f"\n" + "=" * 80)
    print("CRITICAL ENTITY ANALYSIS")
    print("=" * 80)
    
    critical_missing = critical_entities - all_covered
    if critical_missing:
        print(f"[FAIL] Critical entities missing from ALL templates:")
        for entity in sorted(critical_missing):
            print(f"  - {entity}")
    else:
        print("[PASS] All critical entities covered by at least one template")
    
    # Per-entity coverage
    print(f"\n" + "=" * 80)
    print("PER-ENTITY TEMPLATE COVERAGE")
    print("=" * 80)
    
    entity_coverage = defaultdict(list)
    for template_name, result in template_results.items():
        for entity in result['covered']:
            entity_coverage[entity].append(template_name)
    
    # Show entities with low coverage
    low_coverage = []
    for entity in sorted(entities):
        count = len(entity_coverage[entity])
        if count == 0:
            low_coverage.append((entity, count))
        elif count < 3 and entity in critical_entities:
            low_coverage.append((entity, count))
    
    if low_coverage:
        print("[WARN] Entities with low template coverage:")
        for entity, count in low_coverage:
            templates_str = ', '.join(entity_coverage[entity]) if entity_coverage[entity] else 'NONE'
            print(f"  - {entity}: {count} templates ({templates_str})")
    else:
        print("[PASS] All entities have good template coverage")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    errors = []
    warnings = []
    
    if critical_missing:
        errors.append(f"{len(critical_missing)} critical entities missing from all templates")
    
    if never_covered:
        warnings.append(f"{len(never_covered)} entities never used in any template")
    
    if low_coverage:
        warnings.append(f"{len(low_coverage)} entities have low template coverage")
    
    if errors:
        print("[FAIL] VALIDATION FAILED")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    if warnings:
        print("[WARN] VALIDATION PASSED WITH WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("[PASS] VALIDATION PASSED - All templates ready for production!")
    
    print("=" * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
