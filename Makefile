PYTHON ?= python3
VENV ?= .venv
IMAGE ?= mixing-forum-analyzer

.PHONY: install dev-install lint format test coverage notebooks docker-build docker-run clean ci help

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	$(PYTHON) -m pip install -r requirements.txt

dev-install: ## Install production + development dependencies
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

lint: ## Run flake8, black (check), and isort (check)
	flake8 src presets evaluation tests
	black --check .
	isort --check-only .

format: ## Format code with black and isort
	black .
	isort .

test: ## Run pytest suite
	pytest

coverage: ## Run pytest with coverage report
	pytest --cov --cov-report=term-missing

notebooks: ## Launch Jupyter Notebook
	jupyter notebook notebooks

docker-build: ## Build Docker image
	docker build -t $(IMAGE) .

docker-run: ## Run Docker container
	docker run --rm -p 8501:8501 $(IMAGE)

clean: ## Remove caches and coverage artifacts
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -f .coverage coverage.xml

ci: lint test ## Run linting and tests (shortcut for CI)
