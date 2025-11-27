# Production Deployment Guide

Complete guide for deploying InvoiceGen in production environments.

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Docker Deployment](#docker-deployment)
3. [Kubernetes Deployment](#kubernetes-deployment)
4. [API Server Configuration](#api-server-configuration)
5. [Model Serving](#model-serving)
6. [Monitoring & Logging](#monitoring--logging)
7. [Scaling](#scaling)
8. [Security](#security)
9. [Performance Optimization](#performance-optimization)
10. [Troubleshooting](#troubleshooting)

## Deployment Options

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

**Best for**: Single server, development, small teams

**Pros**:
- Simple setup
- Easy management
- Local development-production parity

**Cons**:
- Limited scaling
- Single point of failure

### Option 2: Kubernetes (Recommended for Large Scale)

**Best for**: Multi-server, high availability, auto-scaling

**Pros**:
- Horizontal scaling
- High availability
- Rolling updates
- Resource management

**Cons**:
- Complex setup
- Operational overhead

### Option 3: Cloud Managed Services

**Best for**: Minimal ops overhead

**Options**:
- AWS SageMaker
- Azure ML
- Google Vertex AI

## Docker Deployment

### Prerequisites

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Build Image

```bash
# Build production image
docker build -t invoicegen:latest .

# Build with specific tag
docker build -t invoicegen:v1.0.0 .

# Multi-stage build (smaller image)
docker build -f Dockerfile.prod -t invoicegen:prod .
```

### Run Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d invoicegen-api

# View logs
docker-compose logs -f invoicegen-api

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Environment Variables

Create `.env` file:

```env
# Model Configuration
MODEL_PATH=/app/models/checkpoint
MODEL_NAME=layoutlmv3-invoice

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
LOG_LEVEL=INFO

# Inference Configuration
MAX_BATCH_SIZE=16
INFERENCE_TIMEOUT=30
USE_GPU=true
FP16_INFERENCE=true

# OCR Configuration
OCR_ENGINE=paddleocr
OCR_LANG=en

# Monitoring
PROMETHEUS_ENABLED=true
METRICS_PORT=9090

# Security
API_KEY_REQUIRED=true
CORS_ORIGINS=["https://yourdomain.com"]
```

Load with docker-compose:

```yaml
services:
  invoicegen-api:
    env_file:
      - .env
```

### Volume Mounts

```yaml
volumes:
  - ./models:/app/models:ro          # Read-only models
  - ./data:/app/data                  # Data directory
  - ./outputs:/app/outputs            # Output results
  - ./logs:/app/logs                  # Log files
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## Kubernetes Deployment

### Namespace

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: invoicegen
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: invoicegen-api
  namespace: invoicegen
spec:
  replicas: 3
  selector:
    matchLabels:
      app: invoicegen-api
  template:
    metadata:
      labels:
        app: invoicegen-api
    spec:
      containers:
      - name: api
        image: invoicegen:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /app/models/checkpoint
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: invoicegen-models-pvc
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: invoicegen-api
  namespace: invoicegen
spec:
  selector:
    app: invoicegen-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: invoicegen-ingress
  namespace: invoicegen
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: api.invoicegen.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: invoicegen-api
            port:
              number: 80
```

### Deploy

```bash
kubectl apply -f namespace.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check status
kubectl get pods -n invoicegen
kubectl get svc -n invoicegen
kubectl logs -f deployment/invoicegen-api -n invoicegen
```

## API Server Configuration

### FastAPI Setup

```python
# deployment/api.py (enhanced)
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

app = FastAPI(
    title="InvoiceGen API",
    description="Production invoice understanding API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Load model once at startup
@app.on_event("startup")
async def startup_event():
    from deployment.model_loader import ModelLoader
    global model_loader
    model_loader = ModelLoader(
        model_path=os.getenv("MODEL_PATH"),
        device="cuda" if os.getenv("USE_GPU") == "true" else "cpu"
    )
    model_loader.load_model()

@app.post("/predict")
async def predict(file: UploadFile):
    # Implementation
    pass

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        access_log=True
    )
```

### Nginx Reverse Proxy

```nginx
# nginx.conf
upstream invoicegen {
    least_conn;
    server api1:8000;
    server api2:8000;
    server api3:8000;
}

server {
    listen 80;
    server_name api.invoicegen.com;

    location / {
        proxy_pass http://invoicegen;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        
        # Timeouts for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req zone=api_limit burst=20 nodelay;
}
```

## Model Serving

### Model Optimization

```python
# Quantization (INT8)
from torch.quantization import quantize_dynamic

model_fp32 = ModelLoader("models/checkpoint").model
model_int8 = quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)
torch.save(model_int8.state_dict(), "models/checkpoint_int8.pth")

# ONNX Export
import torch.onnx

dummy_input = {
    'input_ids': torch.zeros(1, 512, dtype=torch.long),
    'bbox': torch.zeros(1, 512, 4, dtype=torch.long),
    'pixel_values': torch.zeros(1, 3, 224, 224)
}

torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    opset_version=14,
    input_names=['input_ids', 'bbox', 'pixel_values'],
    output_names=['ner_logits', 'table_logits', 'cell_logits'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'bbox': {0: 'batch', 1: 'sequence'},
        'pixel_values': {0: 'batch'}
    }
)
```

### Batching Strategy

```python
from deployment.batch_runner import AsyncBatchRunner

runner = AsyncBatchRunner(
    model_loader=model_loader,
    batch_size=16,
    num_workers=4,
    use_threads=True
)

# Process directory with async batching
stats = runner.process_directory_async(
    input_dir="data/incoming",
    output_dir="data/results"
)
```

## Monitoring & Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram

# Custom metrics
prediction_counter = Counter(
    'predictions_total',
    'Total predictions made'
)

prediction_duration = Histogram(
    'prediction_duration_seconds',
    'Prediction duration'
)

@app.post("/predict")
async def predict(file: UploadFile):
    with prediction_duration.time():
        result = model_loader.predict(...)
        prediction_counter.inc()
        return result
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "InvoiceGen API",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(predictions_total[5m])"
        }]
      },
      {
        "title": "Latency p95",
        "targets": [{
          "expr": "histogram_quantile(0.95, prediction_duration_seconds)"
        }]
      }
    ]
  }
}
```

### Structured Logging

```python
import structlog

logger = structlog.get_logger()

@app.post("/predict")
async def predict(file: UploadFile):
    logger.info(
        "prediction_started",
        filename=file.filename,
        size=file.size
    )
    
    try:
        result = model_loader.predict(...)
        logger.info("prediction_success", filename=file.filename)
        return result
    except Exception as e:
        logger.error("prediction_failed", filename=file.filename, error=str(e))
        raise
```

## Scaling

### Horizontal Pod Autoscaler (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: invoicegen-hpa
  namespace: invoicegen
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: invoicegen-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Testing

```python
# locustfile.py
from locust import HttpUser, task, between

class InvoiceGenUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def predict(self):
        with open("test_invoice.png", "rb") as f:
            self.client.post(
                "/predict",
                files={"file": f}
            )
```

Run: `locust -f locustfile.py --host=http://api.invoicegen.com`

## Security

### API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict(
    file: UploadFile,
    api_key: str = Depends(verify_api_key)
):
    ...
```

### Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(request: Request, file: UploadFile):
    ...
```

## Performance Optimization

### Caching

```python
from functools import lru_cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@lru_cache(maxsize=100)
def get_model_prediction(image_hash: str):
    # Cache frequent predictions
    pass
```

### Async Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@app.post("/predict")
async def predict(file: UploadFile):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        model_loader.predict,
        file
    )
    return result
```

## Troubleshooting

### Common Issues

**Issue**: OOM (Out of Memory)
```bash
# Solution: Reduce batch size
export MAX_BATCH_SIZE=8

# Or increase memory limit
docker update --memory=16g invoicegen-api
```

**Issue**: Slow inference
```bash
# Enable FP16
export FP16_INFERENCE=true

# Use GPU
export USE_GPU=true

# Check GPU usage
nvidia-smi
```

**Issue**: Model not loading
```bash
# Check model path
docker exec invoicegen-api ls -la /app/models

# Check logs
docker logs invoicegen-api

# Verify checkpoint
python -c "from transformers import LayoutLMv3ForTokenClassification; model = LayoutLMv3ForTokenClassification.from_pretrained('models/checkpoint')"
```

## Backup & Recovery

```bash
# Backup models
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/

# Backup data
tar -czf data-backup-$(date +%Y%m%d).tar.gz data/

# Restore
tar -xzf models-backup-20241126.tar.gz
```

---

**Production deployment checklist:**
- [ ] Configure environment variables
- [ ] Set up monitoring (Prometheus + Grafana)
- [ ] Enable logging (structured logs)
- [ ] Configure autoscaling (HPA)
- [ ] Set up backup strategy
- [ ] Load test (Locust)
- [ ] Security audit (API keys, rate limiting)
- [ ] Documentation (API docs, runbooks)
