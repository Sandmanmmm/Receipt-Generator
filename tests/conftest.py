"""
Pytest Configuration
"""
import pytest


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "docker: marks tests requiring Docker"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU"
    )


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create temporary test data directory"""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration"""
    return {
        'model': {
            'name': 'microsoft/layoutlmv3-base',
            'num_labels': 73
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 2e-5,
            'num_epochs': 3
        }
    }
