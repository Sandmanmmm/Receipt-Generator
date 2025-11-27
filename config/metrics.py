"""
Prometheus Metrics for InvoiceGen API
Export custom metrics for monitoring
"""
from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable


# Request metrics
REQUEST_COUNT = Counter(
    'invoicegen_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'invoicegen_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Model inference metrics
INFERENCE_COUNT = Counter(
    'invoicegen_inference_total',
    'Total number of model inferences',
    ['model_type', 'status']
)

INFERENCE_DURATION = Histogram(
    'invoicegen_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type']
)

# OCR metrics
OCR_COUNT = Counter(
    'invoicegen_ocr_total',
    'Total number of OCR operations',
    ['ocr_engine', 'status']
)

OCR_DURATION = Histogram(
    'invoicegen_ocr_duration_seconds',
    'OCR processing duration in seconds',
    ['ocr_engine']
)

# Batch processing metrics
BATCH_SIZE = Histogram(
    'invoicegen_batch_size',
    'Batch processing size',
    buckets=[1, 5, 10, 20, 50, 100, 200]
)

BATCH_DURATION = Histogram(
    'invoicegen_batch_duration_seconds',
    'Batch processing duration in seconds'
)

# System metrics
ACTIVE_REQUESTS = Gauge(
    'invoicegen_active_requests',
    'Number of active requests'
)

MODEL_LOAD_TIME = Gauge(
    'invoicegen_model_load_seconds',
    'Time taken to load model'
)

# Error metrics
ERROR_COUNT = Counter(
    'invoicegen_errors_total',
    'Total number of errors',
    ['error_type', 'module']
)

# Model info
MODEL_INFO = Info(
    'invoicegen_model',
    'Information about the loaded model'
)


# Decorators for automatic metric collection
def track_request_metrics(endpoint: str):
    """Decorator to track request metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            method = kwargs.get('request', args[0] if args else None)
            method_name = method.method if method else 'UNKNOWN'
            
            ACTIVE_REQUESTS.inc()
            start_time = time.time()
            status = 'success'
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                ERROR_COUNT.labels(
                    error_type=type(e).__name__,
                    module=func.__module__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                ACTIVE_REQUESTS.dec()
                REQUEST_COUNT.labels(
                    method=method_name,
                    endpoint=endpoint,
                    status=status
                ).inc()
                REQUEST_DURATION.labels(
                    method=method_name,
                    endpoint=endpoint
                ).observe(duration)
        
        return wrapper
    return decorator


def track_inference_metrics(model_type: str = 'layoutlmv3'):
    """Decorator to track inference metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                INFERENCE_COUNT.labels(
                    model_type=model_type,
                    status=status
                ).inc()
                INFERENCE_DURATION.labels(
                    model_type=model_type
                ).observe(duration)
        
        return wrapper
    return decorator


def track_ocr_metrics(ocr_engine: str):
    """Decorator to track OCR metrics"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                raise
            finally:
                duration = time.time() - start_time
                OCR_COUNT.labels(
                    ocr_engine=ocr_engine,
                    status=status
                ).inc()
                OCR_DURATION.labels(
                    ocr_engine=ocr_engine
                ).observe(duration)
        
        return wrapper
    return decorator


def track_batch_metrics(func: Callable) -> Callable:
    """Decorator to track batch processing metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        batch = kwargs.get('batch', args[0] if args else [])
        batch_size = len(batch) if isinstance(batch, (list, tuple)) else 1
        
        BATCH_SIZE.observe(batch_size)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            BATCH_DURATION.observe(duration)
    
    return wrapper


# Initialize model info
def set_model_info(model_name: str, model_version: str, labels: int):
    """Set model information"""
    MODEL_INFO.info({
        'model_name': model_name,
        'model_version': model_version,
        'num_labels': str(labels)
    })
