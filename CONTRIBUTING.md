# Contributing to InvoiceGen

Thank you for your interest in contributing to InvoiceGen! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository**
2. **Clone your fork:**
   ```bash
   git clone https://github.com/yourusername/InvoiceGen.git
   cd InvoiceGen
   ```
3. **Set up development environment:**
   ```bash
   python setup.py
   ```
4. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Document all modules, classes, and functions with docstrings
- Keep functions focused and modular (< 50 lines when possible)
- Use meaningful variable names

Example:
```python
def process_invoice(image_path: str, model: LayoutLMv3) -> Dict[str, Any]:
    """
    Process an invoice image and extract entities.
    
    Args:
        image_path: Path to the invoice image
        model: Trained LayoutLMv3 model
        
    Returns:
        Dictionary containing extracted entities
    """
    # Implementation
    pass
```

### Project Structure

- **generators/**: Data generation and rendering
- **annotation/**: OCR and annotation tools
- **augmentation/**: Image augmentation pipeline
- **training/**: Model training scripts
- **evaluation/**: Evaluation and metrics
- **deployment/**: Production deployment code
- **scripts/**: Utility and pipeline scripts
- **tests/**: Unit and integration tests

### Adding New Features

#### New Invoice Template

1. Create HTML template in `templates/html/your_template.html`
2. Create CSS stylesheet in `templates/css/your_template.css`
3. Use Jinja2 syntax for dynamic content
4. Test rendering with sample data
5. Add to `config/config.yaml` templates list

#### New Augmentation

1. Add method to `ImageAugmenter` class in `augmentation/augmenter.py`
2. Add configuration parameters to `AugmentationConfig`
3. Update `augment()` method to include new augmentation
4. Add tests in `tests/test_augmentation.py`

#### New Entity Type

1. Update entity labels in `config/config.yaml`
2. Add patterns to `EntityLabeler` in `annotation/annotator.py`
3. Update label mapping in training converter
4. Retrain model with new entity type

### Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

Add tests for new features:
```python
def test_new_feature():
    """Test description"""
    # Test implementation
    assert expected == actual
```

### Documentation

- Update README.md for user-facing changes
- Update docstrings for API changes
- Add examples for new features
- Update config.yaml with new parameters

## Contribution Areas

### High Priority

- [ ] Additional invoice templates (minimalist, colorful, etc.)
- [ ] Support for more languages and locales
- [ ] Improved entity labeling accuracy
- [ ] Model quantization for faster inference
- [ ] Docker deployment configuration
- [ ] CI/CD pipeline setup

### Enhancement Ideas

- [ ] Web UI for invoice generation
- [ ] Real-time inference API with streaming
- [ ] Support for multi-page invoices
- [ ] Invoice validation and error detection
- [ ] Export to structured formats (JSON, XML)
- [ ] Integration with accounting software APIs

### Documentation Needs

- [ ] Video tutorials
- [ ] API documentation with Swagger
- [ ] Training guide for custom datasets
- [ ] Deployment best practices
- [ ] Performance optimization tips

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes:**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

3. **Test your changes:**
   ```bash
   pytest tests/
   python scripts/quickstart.py
   ```

4. **Commit with descriptive messages:**
   ```bash
   git commit -m "Add: New minimalist invoice template"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Create Pull Request:**
   - Provide clear description of changes
   - Reference any related issues
   - Include screenshots for UI changes
   - Ensure tests pass

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] Documentation updated
- [ ] No unnecessary dependencies added
- [ ] Commit messages are clear
- [ ] PR description explains changes

## Code Review Process

1. Maintainers will review within 1-2 weeks
2. Address review comments
3. Once approved, PR will be merged
4. Your contribution will be credited!

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
- Error messages and stack traces
- Minimal code example

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Potential impact on existing functionality

## Community

- Be respectful and constructive
- Help others in issues and discussions
- Share your use cases and results
- Spread the word about InvoiceGen!

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open an issue or discussion on GitHub!

---

**Thank you for contributing to InvoiceGen! 🙏**
