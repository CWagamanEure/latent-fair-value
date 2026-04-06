PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
MARKET ?= btc
TEST_ARGS ?=
UI_DB ?= data/$(MARKET)_prices.sqlite3
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8000

.PHONY: run live test lint format ui ui-server

run:
	$(PYTHON) -m src.main $(MARKET) --db-dir $(dir $(UI_DB))

live: run

ui:
	@set -eu; \
	$(PYTHON) -m src.main $(MARKET) --db-dir "$(dir $(UI_DB))" & \
	collector_pid=$$!; \
	trap 'kill $$collector_pid' EXIT INT TERM; \
	$(PYTHON) -m ui.server --db $(UI_DB) --host $(UI_HOST) --port $(UI_PORT)

ui-server:
	$(PYTHON) -m ui.server --db $(UI_DB) --host $(UI_HOST) --port $(UI_PORT)

test:
	$(PYTHON) -m pytest $(TEST_ARGS)

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .
