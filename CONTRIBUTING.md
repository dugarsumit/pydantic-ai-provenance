# Contributing

Thank you for considering a contribution! Below are the steps to get up and running.

## Development setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone https://github.com/dugarsumit/pydantic-ai-provenance.git
cd pydantic-ai-provenance

# Install all dependencies including dev extras
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

## Running tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=pydantic_ai_provenance --cov-report=term-missing
```

## Code style

Formatting and linting are enforced by [ruff](https://docs.astral.sh/ruff/).

```bash
uv run ruff check .
uv run ruff format .
```

Pre-commit runs both checks automatically on every commit.

## Submitting changes

1. Fork the repository and create a feature branch from `main`.
2. Add tests for any new behaviour.
3. Ensure the full test suite passes (`uv run pytest`).
4. Open a pull request with a clear description of the change and the motivation.

## Reporting bugs

Please open an issue on GitHub describing:
- Python version and OS.
- `pydantic-ai` version.
- A minimal reproducible example.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
