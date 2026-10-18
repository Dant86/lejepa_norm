# lejepa_norm

Studying the role of the embedding radius in LeJEPA. See `CLAUDE.md` for the
engineering rules and `docs/architecture.md` for module and data-flow diagrams.

```bash
uv sync
cp .env.example .env   # fill in real paths
uv run pre-commit install
make check             # ruff, import rule, pyright, pytest (same as CI)
```
