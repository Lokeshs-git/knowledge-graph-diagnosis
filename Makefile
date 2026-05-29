.PHONY: help install test lint format check eval graph trace clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync --all-groups
	uv run pre-commit install

test:  ## Run tests
	uv run pytest

lint:  ## Lint with ruff
	uv run ruff check .

format:  ## Format code with ruff
	uv run ruff format .
	uv run ruff check --fix .

check-graph:  ## Validate the knowledge graph (IDs, bidirectional links)
	uv run python tools/check_links.py

graph:  ## Generate a Mermaid diagram of the knowledge graph
	uv run python tools/graph.py --output graph.mmd
	@echo "Wrote graph.mmd — view at https://mermaid.live or in any Mermaid renderer"

trace:  ## Trace an artifact: make trace ID=PRD-003
	@if [ -z "$(ID)" ]; then echo "Usage: make trace ID=PRD-003"; exit 1; fi
	uv run qs trace $(ID)

check: lint test check-graph  ## Run lint + tests + graph check

eval:  ## Run the example eval
	uv run python -m evals.run_example

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build graph.mmd
	find . -type d -name __pycache__ -exec rm -rf {} +
