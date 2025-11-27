"""
Test Fixtures for Invoice Generation
"""
import pytest
from pathlib import Path
import tempfile
import json


@pytest.fixture
def sample_invoice_data():
    """Sample invoice data"""
    return {
        'company_name': 'Test Corp Inc.',
        'company_address': '123 Test St, Test City, TC 12345',
        'invoice_number': 'INV-2024-001',
        'invoice_date': '2024-11-26',
        'due_date': '2024-12-26',
        'customer_name': 'John Doe',
        'customer_address': '456 Customer Ave, Customer City, CC 67890',
        'items': [
            {
                'description': 'Test Product 1',
                'quantity': 2,
                'unit_price': 50.00,
                'total': 100.00
            },
            {
                'description': 'Test Product 2',
                'quantity': 1,
                'unit_price': 75.00,
                'total': 75.00
            }
        ],
        'subtotal': 175.00,
        'tax': 14.00,
        'discount': 0.00,
        'total': 189.00,
        'currency': 'USD',
        'notes': 'Thank you for your business!'
    }


@pytest.fixture
def sample_annotation():
    """Sample annotation data"""
    return {
        'image_path': 'test_invoice.png',
        'tokens': ['INVOICE', '#', 'INV-2024-001', 'Date:', '2024-11-26', 'Total:', '$189.00'],
        'labels': ['B-DOCUMENT_TYPE', 'O', 'B-INVOICE_NUMBER', 'O', 'B-INVOICE_DATE', 'O', 'B-TOTAL_AMOUNT'],
        'bboxes': [
            [50, 50, 150, 80],
            [150, 50, 160, 80],
            [160, 50, 280, 80],
            [50, 100, 90, 130],
            [90, 100, 200, 130],
            [50, 600, 90, 630],
            [90, 600, 170, 630]
        ],
        'image_width': 1000,
        'image_height': 1000
    }


@pytest.fixture
def temp_dir():
    """Temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_model_config():
    """Mock model configuration"""
    return {
        'model': {
            'name': 'microsoft/layoutlmv3-base',
            'num_labels': 73,
            'max_position_embeddings': 512
        },
        'ner': {
            'num_labels': 73
        },
        'table': {
            'num_labels': 3
        },
        'cell': {
            'num_labels': 3
        }
    }


@pytest.fixture
def label_list():
    """Standard label list"""
    return [
        'O',
        'B-DOCUMENT_TYPE', 'I-DOCUMENT_TYPE',
        'B-INVOICE_NUMBER', 'I-INVOICE_NUMBER',
        'B-INVOICE_DATE', 'I-INVOICE_DATE',
        'B-DUE_DATE', 'I-DUE_DATE',
        'B-TOTAL_AMOUNT', 'I-TOTAL_AMOUNT',
        'B-COMPANY_NAME', 'I-COMPANY_NAME',
        'B-CUSTOMER_NAME', 'I-CUSTOMER_NAME'
    ]


@pytest.fixture
def sample_predictions():
    """Sample model predictions"""
    return {
        'true_labels': [['B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'O', 'B-TOTAL_AMOUNT']],
        'pred_labels': [['B-INVOICE_NUMBER', 'I-INVOICE_NUMBER', 'O', 'B-TOTAL_AMOUNT']],
        'tokens': [['INV', '-2024-001', 'Total:', '$189.00']]
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
