# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: security@yourproject.com

You should receive a response within 48 hours. If for some reason you do not, please follow up via email.

Please include the following information:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## Security Best Practices

### API Deployment

1. **API Keys**: Always use API key authentication in production
   ```python
   API_KEY = os.getenv('INVOICEGEN_API_KEY')
   ```

2. **Rate Limiting**: Configure rate limits in production
   ```yaml
   rate_limit: 100  # requests per minute
   ```

3. **HTTPS**: Always use HTTPS in production
   - Configure SSL certificates in Nginx/load balancer
   - Redirect HTTP to HTTPS

4. **Input Validation**: Validate all uploaded files
   - Check file types (only PNG, JPG, PDF)
   - Limit file size (max 10MB)
   - Scan for malware

### Docker Security

1. **Non-root User**: Run containers as non-root
   ```dockerfile
   USER appuser
   ```

2. **Secret Management**: Use Docker secrets or environment variables
   ```bash
   docker secret create api_key /path/to/key
   ```

3. **Image Scanning**: Scan images for vulnerabilities
   ```bash
   trivy image invoicegen:latest
   ```

### Data Security

1. **Encryption at Rest**: Encrypt sensitive data
   - Use encrypted volumes for model storage
   - Encrypt database if used

2. **Encryption in Transit**: Use TLS for all network communication

3. **Access Control**: Implement proper access controls
   - Use IAM roles for cloud deployments
   - Restrict network access with security groups

### Dependencies

1. **Keep Updated**: Regularly update dependencies
   ```bash
   pip list --outdated
   pip install --upgrade package_name
   ```

2. **Vulnerability Scanning**: Use tools like `safety`
   ```bash
   pip install safety
   safety check
   ```

3. **Pin Versions**: Use exact versions in production
   ```
   torch==2.0.1  # not torch>=2.0.0
   ```

## Known Security Considerations

### 1. Model Security
- Trained models can be large targets for theft
- Use model encryption or obfuscation for proprietary models
- Implement model versioning and access logs

### 2. OCR Engine Security
- OCR engines may have vulnerabilities
- Keep PaddleOCR, Tesseract updated
- Validate OCR output before processing

### 3. API Endpoint Security
- Implement request validation
- Use CORS appropriately
- Log all API requests for audit

### 4. Data Privacy
- Invoice data may contain PII
- Implement data retention policies
- Comply with GDPR/CCPA if applicable

## Compliance

This project aims to comply with:

- **OWASP Top 10** security risks
- **CWE Top 25** most dangerous software weaknesses
- **PCI DSS** for payment card data (if applicable)
- **GDPR** for EU data privacy
- **SOC 2** controls for security

## Security Checklist

Before production deployment:

- [ ] API authentication enabled
- [ ] Rate limiting configured
- [ ] HTTPS/TLS enabled
- [ ] Input validation implemented
- [ ] Error messages don't leak sensitive info
- [ ] Logging configured (but doesn't log secrets)
- [ ] Dependencies updated and scanned
- [ ] Docker images scanned
- [ ] Access controls configured
- [ ] Secrets properly managed
- [ ] Backup and recovery tested
- [ ] Monitoring and alerting enabled
