"""
Annotation Validation Utilities
Validates JSONL annotation files against schema
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
import argparse


@dataclass
class ValidationResult:
    """Validation result for a document"""
    doc_id: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class AnnotationValidator:
    """Validates annotation files against schema"""
    
    def __init__(self, label_list_path: str = "config/labels.yaml"):
        """
        Initialize validator
        
        Args:
            label_list_path: Path to labels.yaml file
        """
        self.valid_labels = self._load_labels(label_list_path)
    
    def _load_labels(self, path: str) -> Set[str]:
        """Load valid labels from YAML file"""
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return set(config['label_list'])
        except Exception as e:
            print(f"Warning: Could not load labels from {path}: {e}")
            print("Using default label set")
            return {'O'}  # Minimal fallback
    
    def validate_document(self, doc: Dict) -> ValidationResult:
        """
        Validate a single document
        
        Args:
            doc: Document dictionary from JSONL
        
        Returns:
            ValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        doc_id = doc.get('id', 'unknown')
        
        # Required fields check
        required_fields = ['id', 'image_path', 'width', 'height', 'tokens', 'boxes', 'labels']
        for field in required_fields:
            if field not in doc:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return ValidationResult(doc_id, False, errors, warnings)
        
        # Array length consistency
        tokens = doc['tokens']
        boxes = doc['boxes']
        labels = doc['labels']
        
        num_tokens = len(tokens)
        num_boxes = len(boxes)
        num_labels = len(labels)
        
        if not (num_tokens == num_boxes == num_labels):
            errors.append(
                f"Array length mismatch: tokens={num_tokens}, boxes={num_boxes}, labels={num_labels}"
            )
        
        # Validate tokens
        for i, token in enumerate(tokens):
            # Required token fields
            if 'text' not in token:
                errors.append(f"Token {i}: missing 'text' field")
            elif not token['text']:
                warnings.append(f"Token {i}: empty text")
            
            if 'bbox' not in token:
                errors.append(f"Token {i}: missing 'bbox' field")
            else:
                bbox = token['bbox']
                if len(bbox) != 4:
                    errors.append(f"Token {i}: bbox must have 4 coordinates, got {len(bbox)}")
                else:
                    # Validate bbox coordinates
                    x0, y0, x1, y1 = bbox
                    if x0 >= x1 or y0 >= y1:
                        errors.append(f"Token {i}: invalid bbox {bbox} (x0>=x1 or y0>=y1)")
                    
                    # Check within image boundaries
                    width = doc['width']
                    height = doc['height']
                    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                        warnings.append(
                            f"Token {i}: bbox {bbox} outside image bounds ({width}×{height})"
                        )
            
            if 'token_id' not in token:
                warnings.append(f"Token {i}: missing 'token_id' field")
            elif token['token_id'] != i:
                warnings.append(f"Token {i}: token_id={token['token_id']} (expected {i})")
            
            if 'label' not in token:
                errors.append(f"Token {i}: missing 'label' field")
        
        # Validate labels
        for i, label in enumerate(labels):
            if label not in self.valid_labels:
                errors.append(f"Token {i}: invalid label '{label}'")
        
        # Validate BIO sequence
        bio_errors = self._validate_bio_sequence(labels)
        errors.extend([f"BIO sequence: {err}" for err in bio_errors])
        
        # Validate boxes array (duplicate check)
        for i, bbox in enumerate(boxes):
            if len(bbox) != 4:
                errors.append(f"Box {i}: must have 4 coordinates, got {len(bbox)}")
        
        # Optional: validate table_labels if present
        if 'table_labels' in doc:
            table_labels = doc['table_labels']
            if len(table_labels) != num_labels:
                errors.append(
                    f"table_labels length mismatch: {len(table_labels)} vs {num_labels}"
                )
            
            valid_table_labels = {'O', 'B-TABLE', 'I-TABLE'}
            for i, label in enumerate(table_labels):
                if label not in valid_table_labels:
                    errors.append(f"Token {i}: invalid table_label '{label}'")
        
        is_valid = len(errors) == 0
        return ValidationResult(doc_id, is_valid, errors, warnings)
    
    def _validate_bio_sequence(self, labels: List[str]) -> List[str]:
        """
        Validate BIO label sequence
        
        Returns:
            List of error messages
        """
        errors = []
        prev_label = 'O'
        
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # I- must follow B- or I- of same entity type
                if prev_label == 'O':
                    errors.append(f"Token {i}: I-{entity_type} without preceding B-")
                elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                    prev_entity = prev_label[2:]
                    if prev_entity != entity_type:
                        errors.append(
                            f"Token {i}: I-{entity_type} follows {prev_label} (type mismatch)"
                        )
            
            prev_label = label
        
        return errors
    
    def validate_file(self, filepath: str) -> Tuple[int, int, List[ValidationResult]]:
        """
        Validate entire JSONL file
        
        Args:
            filepath: Path to JSONL file
        
        Returns:
            Tuple of (total_docs, valid_docs, results)
        """
        results = []
        total_docs = 0
        valid_docs = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    result = self.validate_document(doc)
                    results.append(result)
                    
                    total_docs += 1
                    if result.is_valid:
                        valid_docs += 1
                
                except json.JSONDecodeError as e:
                    results.append(ValidationResult(
                        f"line_{line_num}",
                        False,
                        [f"JSON decode error: {e}"],
                        []
                    ))
                    total_docs += 1
        
        return total_docs, valid_docs, results
    
    def print_report(self, results: List[ValidationResult], verbose: bool = False):
        """Print validation report"""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        invalid = total - valid
        
        print(f"\n{'='*60}")
        print(f"Validation Report")
        print(f"{'='*60}")
        print(f"Total documents: {total}")
        print(f"Valid: {valid} ({100*valid/total:.1f}%)")
        print(f"Invalid: {invalid} ({100*invalid/total:.1f}%)")
        
        # Count errors and warnings
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        print(f"Total errors: {total_errors}")
        print(f"Total warnings: {total_warnings}")
        
        # Print invalid documents
        if invalid > 0:
            print(f"\n{'='*60}")
            print(f"Invalid Documents:")
            print(f"{'='*60}")
            
            for result in results:
                if not result.is_valid:
                    print(f"\nDocument: {result.doc_id}")
                    print(f"  Errors ({len(result.errors)}):")
                    for error in result.errors:
                        print(f"    - {error}")
                    
                    if result.warnings and verbose:
                        print(f"  Warnings ({len(result.warnings)}):")
                        for warning in result.warnings:
                            print(f"    - {warning}")
        
        # Print warnings for valid documents
        if verbose and total_warnings > 0:
            print(f"\n{'='*60}")
            print(f"Warnings (Valid Documents):")
            print(f"{'='*60}")
            
            for result in results:
                if result.is_valid and result.warnings:
                    print(f"\nDocument: {result.doc_id}")
                    for warning in result.warnings:
                        print(f"  - {warning}")
        
        print(f"\n{'='*60}\n")


def main():
    """CLI for annotation validation"""
    parser = argparse.ArgumentParser(
        description="Validate JSONL annotation files"
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to JSONL file to validate'
    )
    parser.add_argument(
        '--labels',
        default='config/labels.yaml',
        help='Path to labels.yaml file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show warnings for valid documents'
    )
    parser.add_argument(
        '--fail-on-warnings',
        action='store_true',
        help='Exit with error code if warnings found'
    )
    
    args = parser.parse_args()
    
    # Check input file exists
    if not Path(args.input).exists():
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    
    # Validate
    validator = AnnotationValidator(args.labels)
    total_docs, valid_docs, results = validator.validate_file(args.input)
    validator.print_report(results, verbose=args.verbose)
    
    # Exit code
    has_errors = total_docs != valid_docs
    has_warnings = any(r.warnings for r in results)
    
    if has_errors:
        sys.exit(1)
    elif has_warnings and args.fail_on_warnings:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
