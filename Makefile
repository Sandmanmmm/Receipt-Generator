# Makefile for InvoiceGen

.PHONY: help install install-dev test test-fast test-cov lint format type-check clean docker-build docker-up docker-down build-dataset train deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -r requirements_crf.txt

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements_crf.txt
	pip install pytest pytest-cov flake8 black mypy isort

test: ## Run all tests
	pytest tests/ -v

test-fast: ## Run fast tests only (skip slow and docker)
	pytest tests/ -v -m "not slow and not docker"

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

test-integration: ## Run integration tests only
	pytest tests/ -v -m integration

lint: ## Run linting checks
	flake8 annotation/ training/ deployment/ generators/ evaluation/ scripts/
	black --check .
	isort --check-only .

format: ## Format code with black and isort
	black .
	isort .

type-check: ## Run type checking with mypy
	mypy annotation/ training/ deployment/ generators/ --ignore-missing-imports

clean: ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache .coverage htmlcov/ dist/ build/

docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start Docker services
	docker-compose up -d

docker-down: ## Stop Docker services
	docker-compose down

docker-logs: ## View Docker logs
	docker-compose logs -f

build-dataset: ## Build training dataset (1000 invoices)
	python scripts/build_training_set.py \
		--num-invoices 1000 \
		--template-type modern \
		--augment \
		--output-dir data

build-dataset-large: ## Build large training dataset (10000 invoices)
	python scripts/build_training_set.py \
		--num-invoices 10000 \
		--template-type modern \
		--augment \
		--output-dir data

train: ## Train model
	python scripts/run_training.py \
		--config config/training_config.yaml \
		--train-dir data/train \
		--val-dir data/val \
		--output-dir models/layoutlmv3_multihead

train-resume: ## Resume training from checkpoint
	python scripts/run_training.py \
		--config config/training_config.yaml \
		--train-dir data/train \
		--val-dir data/val \
		--output-dir models/layoutlmv3_multihead \
		--resume

evaluate: ## Evaluate trained model
	python -m evaluation.evaluate \
		--model-path models/layoutlmv3_multihead \
		--test-dir data/test \
		--output-dir outputs/evaluation

deploy-local: ## Deploy API locally
	uvicorn deployment.api:app --host 0.0.0.0 --port 8000 --reload

deploy-prod: ## Deploy with Docker (production)
	docker-compose -f docker-compose.yml up -d

validate-annotations: ## Validate annotation files
	python scripts/validate_annotations.py --data-dir data/annotated

visualize: ## Visualize annotations
	python scripts/visualize_annotations.py \
		--annotation-file data/annotations/sample.jsonl \
		--output-dir outputs/visualizations

setup: ## Initial setup (check dependencies)
	python setup.py

all: clean install test ## Clean, install, and test
