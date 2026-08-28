.PHONY: sync lint typecheck test smoke verify-paper-artifacts reproduce-paper \
	reproduce-paper-tables reproduce-paper-figures table-time table-tonal \
	table-relation table-carrier table-commu

sync:
	uv sync --frozen --extra dev

lint:
	uv run ruff check src/el_tokenization tests scripts

typecheck:
	uv run mypy src/el_tokenization scripts

test:
	uv run pytest

smoke:
	uv run el-token smoke --steps 100

verify-paper-artifacts:
	uv run el-token paper verify

reproduce-paper:
	uv run python scripts/reproduce_paper.py --artifact-root artifacts/paper_v1

reproduce-paper-tables:
	uv run el-token paper tables --artifact-root artifacts/paper_v1 --output-root paper/figure_data

reproduce-paper-figures:
	uv run el-token paper figures --artifact-root artifacts/paper_v1 --output-root paper/figure_data

table-time table-tonal table-relation table-carrier table-commu: reproduce-paper-tables
