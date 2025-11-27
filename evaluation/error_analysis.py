"""
Error Analysis Tools for NER Models
"""
import json
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from pathlib import Path
import re


class ErrorAnalyzer:
    """Analyze and categorize prediction errors for NER models"""
    
    def __init__(self, label_list: List[str]):
        """
        Initialize analyzer
        
        Args:
            label_list: List of all possible labels
        """
        self.label_list = label_list
        self.entity_types = self._extract_entity_types(label_list)
    
    def _extract_entity_types(self, label_list: List[str]) -> set:
        """Extract unique entity types from BIO labels"""
        entity_types = set()
        for label in label_list:
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]
                entity_types.add(entity_type)
        return entity_types
    
    def extract_entities(self, labels: List[str], tokens: List[str]) -> List[Dict]:
        """
        Extract entities from BIO-tagged sequence
        
        Args:
            labels: BIO label sequence
            tokens: Token sequence
            
        Returns:
            List of entity dictionaries with {type, text, start, end}
        """
        entities = []
        current_entity = None
        
        for idx, (label, token) in enumerate(zip(labels, tokens)):
            if label.startswith('B-'):
                # Start new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'text': token,
                    'start': idx,
                    'end': idx
                }
            elif label.startswith('I-'):
                # Continue entity
                if current_entity:
                    current_entity['text'] += f" {token}"
                    current_entity['end'] = idx
                else:
                    # Orphaned I- tag (error in sequence)
                    entity_type = label[2:]
                    current_entity = {
                        'type': entity_type,
                        'text': token,
                        'start': idx,
                        'end': idx
                    }
            else:
                # O tag - end current entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def categorize_errors(self, y_true: List[List[str]], y_pred: List[List[str]], 
                         tokens: List[List[str]]) -> Dict[str, List[Dict]]:
        """
        Categorize prediction errors
        
        Args:
            y_true: True labels (list of sequences)
            y_pred: Predicted labels (list of sequences)
            tokens: Token sequences
            
        Returns:
            Dictionary mapping error type to list of error instances
        """
        errors = {
            'false_positive': [],  # Predicted entity that doesn't exist
            'false_negative': [],  # Missed entity
            'wrong_type': [],      # Correct span, wrong type
            'wrong_boundary': [],  # Correct type, wrong span
            'partial_match': []    # Overlapping but not exact match
        }
        
        for seq_idx, (true_labels, pred_labels, token_seq) in enumerate(zip(y_true, y_pred, tokens)):
            true_entities = self.extract_entities(true_labels, token_seq)
            pred_entities = self.extract_entities(pred_labels, token_seq)
            
            # Track matched entities
            matched_true = set()
            matched_pred = set()
            
            # Find exact matches and wrong types
            for pred_idx, pred_entity in enumerate(pred_entities):
                for true_idx, true_entity in enumerate(true_entities):
                    if true_idx in matched_true:
                        continue
                    
                    # Exact match (correct!)
                    if (pred_entity['start'] == true_entity['start'] and 
                        pred_entity['end'] == true_entity['end'] and
                        pred_entity['type'] == true_entity['type']):
                        matched_true.add(true_idx)
                        matched_pred.add(pred_idx)
                        break
                    
                    # Wrong type (correct span, wrong label)
                    elif (pred_entity['start'] == true_entity['start'] and 
                          pred_entity['end'] == true_entity['end'] and
                          pred_entity['type'] != true_entity['type']):
                        errors['wrong_type'].append({
                            'sequence': seq_idx,
                            'tokens': token_seq[pred_entity['start']:pred_entity['end']+1],
                            'true_type': true_entity['type'],
                            'pred_type': pred_entity['type'],
                            'text': pred_entity['text']
                        })
                        matched_true.add(true_idx)
                        matched_pred.add(pred_idx)
                        break
                    
                    # Partial overlap
                    elif (pred_entity['start'] <= true_entity['end'] and 
                          pred_entity['end'] >= true_entity['start']):
                        # Wrong boundary if same type
                        if pred_entity['type'] == true_entity['type']:
                            errors['wrong_boundary'].append({
                                'sequence': seq_idx,
                                'true_span': (true_entity['start'], true_entity['end']),
                                'pred_span': (pred_entity['start'], pred_entity['end']),
                                'true_text': true_entity['text'],
                                'pred_text': pred_entity['text'],
                                'entity_type': true_entity['type']
                            })
                        else:
                            # Partial match with wrong type
                            errors['partial_match'].append({
                                'sequence': seq_idx,
                                'true_span': (true_entity['start'], true_entity['end']),
                                'pred_span': (pred_entity['start'], pred_entity['end']),
                                'true_text': true_entity['text'],
                                'pred_text': pred_entity['text'],
                                'true_type': true_entity['type'],
                                'pred_type': pred_entity['type']
                            })
                        matched_true.add(true_idx)
                        matched_pred.add(pred_idx)
                        break
            
            # False positives (predicted but not true)
            for pred_idx, pred_entity in enumerate(pred_entities):
                if pred_idx not in matched_pred:
                    errors['false_positive'].append({
                        'sequence': seq_idx,
                        'span': (pred_entity['start'], pred_entity['end']),
                        'text': pred_entity['text'],
                        'pred_type': pred_entity['type']
                    })
            
            # False negatives (true but not predicted)
            for true_idx, true_entity in enumerate(true_entities):
                if true_idx not in matched_true:
                    errors['false_negative'].append({
                        'sequence': seq_idx,
                        'span': (true_entity['start'], true_entity['end']),
                        'text': true_entity['text'],
                        'true_type': true_entity['type']
                    })
        
        return errors
    
    def summarize_errors(self, errors: Dict[str, List[Dict]]) -> Dict[str, int]:
        """Get error counts by category"""
        return {error_type: len(error_list) for error_type, error_list in errors.items()}
    
    def get_error_breakdown_by_entity(self, errors: Dict[str, List[Dict]]) -> Dict[str, Dict[str, int]]:
        """Get error counts broken down by entity type"""
        breakdown = defaultdict(lambda: defaultdict(int))
        
        for error_type, error_list in errors.items():
            for error in error_list:
                if error_type == 'false_positive':
                    entity_type = error['pred_type']
                elif error_type in ['false_negative', 'wrong_boundary']:
                    entity_type = error.get('true_type') or error.get('entity_type')
                elif error_type == 'wrong_type':
                    entity_type = error['true_type']
                elif error_type == 'partial_match':
                    entity_type = error['true_type']
                else:
                    entity_type = 'UNKNOWN'
                
                breakdown[entity_type][error_type] += 1
        
        return dict(breakdown)
    
    def generate_error_report(self, errors: Dict[str, List[Dict]], 
                            output_path: Optional[str] = None) -> str:
        """
        Generate human-readable error report
        
        Args:
            errors: Categorized errors from categorize_errors()
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ERROR ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Overall counts
        lines.append("OVERALL ERROR COUNTS")
        lines.append("-" * 80)
        summary = self.summarize_errors(errors)
        total_errors = sum(summary.values())
        
        for error_type, count in summary.items():
            percentage = (count / total_errors * 100) if total_errors > 0 else 0
            lines.append(f"{error_type.replace('_', ' ').title():<30} {count:>6} ({percentage:>5.1f}%)")
        lines.append(f"{'Total Errors':<30} {total_errors:>6}")
        lines.append("")
        
        # Per-entity breakdown
        lines.append("=" * 80)
        lines.append("ERROR BREAKDOWN BY ENTITY TYPE")
        lines.append("=" * 80)
        breakdown = self.get_error_breakdown_by_entity(errors)
        
        for entity_type in sorted(breakdown.keys()):
            entity_errors = breakdown[entity_type]
            total_entity_errors = sum(entity_errors.values())
            lines.append(f"\n{entity_type} (Total: {total_entity_errors})")
            lines.append("-" * 60)
            for error_type, count in sorted(entity_errors.items(), key=lambda x: -x[1]):
                lines.append(f"  {error_type.replace('_', ' ').title():<25} {count:>5}")
        
        # Sample errors (first 5 of each type)
        lines.append("\n" + "=" * 80)
        lines.append("SAMPLE ERRORS (first 5 of each type)")
        lines.append("=" * 80)
        
        for error_type, error_list in errors.items():
            if error_list:
                lines.append(f"\n{error_type.replace('_', ' ').title()}:")
                lines.append("-" * 60)
                for error in error_list[:5]:
                    lines.append(f"  Sequence {error['sequence']}: {json.dumps(error, indent=4)}")
        
        report = "\n".join(lines)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            print(f"Error report saved to: {output_path}")
        
        return report
    
    def export_errors_json(self, errors: Dict[str, List[Dict]], output_path: str):
        """
        Export all errors to JSON file
        
        Args:
            errors: Categorized errors
            output_path: Path to save JSON
        """
        error_export = {
            'summary': self.summarize_errors(errors),
            'breakdown_by_entity': self.get_error_breakdown_by_entity(errors),
            'detailed_errors': errors
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(error_export, f, indent=2)
        
        print(f"Errors exported to: {output_path}")


__all__ = ['ErrorAnalyzer']
