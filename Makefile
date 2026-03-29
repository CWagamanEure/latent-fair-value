PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MARKET ?= btc
TEST_ARGS ?=

.PHONY: run test lint format

run:
	$(PYTHON) -m src.main $(MARKET)

test:
	$(PYTHON) -m pytest $(TEST_ARGS)

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
