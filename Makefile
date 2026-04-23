.PHONY: install lint format run-all-checks test
.SILENT: test lint format run-all-checks

help:
	@echo "Usage: make <target>"
	@echo "Targets:"
	@echo "  install: Install dependencies"
	@echo "  lint: Run linting"
	@echo "  format: Run formatting"
	@echo "  test: Run tests"
	@echo "  run-all-checks: Run all checks"

install:
	uv sync --active --all-groups --all-extras

lint:
	uv run --no-sync --active ruff check

format:
	uv run --no-sync --active ruff format

test:
	uv run --no-sync --active pytest --cov=pydantic_ai_provenance --cov-report=term-missing -v

run-all-checks:
	-$(MAKE) lint
	-$(MAKE) format
	-$(MAKE) test