.PHONY: check format lint imports typecheck test

# Everything CI and pre-commit enforce.
check: lint imports typecheck test

format:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

imports:
	uv run python scripts/check_imports.py $$(git ls-files '*.py')

typecheck:
	uv run pyright

test:
	uv run pytest
