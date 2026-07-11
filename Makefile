.PHONY: lint format typecheck test check clean

lint:
	python -m ruff check aetherml/ --no-fix

format:
	python -m ruff format aetherml/
	python -m ruff check aetherml/ --fix

typecheck:
	python -m mypy aetherml/ --ignore-missing-imports

test:
	python -m pytest tests/ -q --tb=short

check: lint typecheck test
	@echo "All checks passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
