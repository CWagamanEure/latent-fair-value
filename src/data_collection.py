from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.filters.latent_state_types import PriceBasisState
from src.measurement_types import ActiveAssetContextMeasurement, BBOMeasurement


@dataclass
class BBOState:
    price: Decimal | None = None
    microprice: Decimal | None = None


class SQLiteDataCollector:
    LEGACY_TABLE_NAME = "price_snapshots"
    MARKET_TABLE_NAME = "market_snapshots"
    FILTER_TABLE_NAME = "filter_snapshots"
    ASSET_CONTEXT_TABLE_NAME = "asset_context_snapshots"

    _MARKET_COLUMNS: dict[str, str] = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "asset": "TEXT NOT NULL",
        "measurement_timestamp": "INTEGER",
        "observed_market": "TEXT NOT NULL",
        "observed_market_type": "TEXT NOT NULL",
        "observed_asset": "TEXT",
        "observed_channel": "TEXT NOT NULL",
        "observed_bid_price": "REAL",
        "observed_bid_size": "REAL",
        "observed_ask_price": "REAL",
        "observed_ask_size": "REAL",
        "observed_mid_price": "REAL",
        "observed_microprice": "REAL",
        "spot_mid_price": "REAL",
        "perp_mid_price": "REAL",
        "spot_microprice": "REAL",
        "perp_microprice": "REAL",
        "raw_message_json": "TEXT NOT NULL DEFAULT '{}'",
        "recorded_at_ms": "INTEGER NOT NULL",
    }
    _FILTER_COLUMNS: dict[str, str] = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "market_snapshot_id": f"INTEGER NOT NULL REFERENCES {MARKET_TABLE_NAME}(id) ON DELETE CASCADE",
        "asset": "TEXT NOT NULL",
        "filter_name": "TEXT NOT NULL",
        "price_choice": "TEXT NOT NULL",
        "filter_timestamp": "INTEGER",
        "filtered_price": "REAL NOT NULL",
        "basis": "REAL NOT NULL",
        "spot_error": "REAL",
        "perp_error": "REAL",
        "temporary_dislocation": "REAL",
        "quoted_spot_price": "REAL",
        "quoted_perp_price": "REAL",
        "quoted_basis": "REAL",
        "raw_state_vector_json": "TEXT",
        "raw_covariance_matrix_json": "TEXT",
        "recorded_at_ms": "INTEGER NOT NULL",
    }
    _ASSET_CONTEXT_EXPECTED_COLUMNS: dict[str, str] = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "asset": "TEXT NOT NULL",
        "measurement_timestamp": "INTEGER",
        "observed_market": "TEXT NOT NULL DEFAULT ''",
        "observed_market_type": "TEXT NOT NULL",
        "observed_asset": "TEXT",
        "observed_channel": "TEXT NOT NULL DEFAULT ''",
        "funding_rate": "REAL",
        "open_interest": "REAL",
        "oracle_price": "REAL",
        "mark_price": "REAL",
        "mid_price": "REAL",
        "premium": "REAL",
        "impact_bid_price": "REAL",
        "impact_ask_price": "REAL",
        "day_notional_volume": "REAL",
        "prev_day_price": "REAL",
        "is_snapshot": "INTEGER NOT NULL DEFAULT 0",
        "raw_message_json": "TEXT NOT NULL DEFAULT '{}'",
        "raw_context_json": "TEXT NOT NULL DEFAULT '{}'",
        "recorded_at_ms": "INTEGER NOT NULL DEFAULT 0",
    }

    def __init__(self, asset: str, db_dir: str | Path = "data") -> None:
        self.asset = asset.upper()
        self.db_path = self.build_db_path(self.asset, db_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.execute("PRAGMA foreign_keys = ON")
        self._spot = BBOState()
        self._perp = BBOState()
        self._create_tables()

    @staticmethod
    def build_db_path(asset: str, db_dir: str | Path = "data") -> Path:
        return Path(db_dir) / f"{asset.lower()}_prices.sqlite3"

    def close(self) -> None:
        self.connection.close()

    def record(
        self,
        measurement: BBOMeasurement,
        filtered_states: dict[str, PriceBasisState],
    ) -> None:
        if not filtered_states:
            raise ValueError("filtered_states must include at least one filter entry")

        snapshot = BBOState(price=measurement.mid(), microprice=measurement.microprice())
        if measurement.is_spot:
            self._spot = snapshot
        elif measurement.is_perp:
            self._perp = snapshot
        else:
            raise ValueError("BBOMeasurement must have either spot or perp market_type")

        recorded_at_ms = time.time_ns() // 1_000_000
        cursor = self.connection.execute(
            f"""
            INSERT INTO {self.MARKET_TABLE_NAME} (
                asset,
                measurement_timestamp,
                observed_market,
                observed_market_type,
                observed_asset,
                observed_channel,
                observed_bid_price,
                observed_bid_size,
                observed_ask_price,
                observed_ask_size,
                observed_mid_price,
                observed_microprice,
                spot_mid_price,
                perp_mid_price,
                spot_microprice,
                perp_microprice,
                raw_message_json,
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.asset,
                measurement.timestamp,
                measurement.market,
                measurement.market_type,
                measurement.asset,
                measurement.channel,
                self._as_float(measurement.bid_price),
                self._as_float(measurement.bid_size),
                self._as_float(measurement.ask_price),
                self._as_float(measurement.ask_size),
                self._as_float(snapshot.price),
                self._as_float(snapshot.microprice),
                self._as_float(self._spot.price),
                self._as_float(self._perp.price),
                self._as_float(self._spot.microprice),
                self._as_float(self._perp.microprice),
                json.dumps(measurement.raw_message, sort_keys=True, separators=(",", ":")),
                recorded_at_ms,
            ),
        )
        market_snapshot_id = cursor.lastrowid
        if market_snapshot_id is None:
            raise RuntimeError("Failed to persist market snapshot row")

        for filter_name, state in filtered_states.items():
            self.connection.execute(
                f"""
                INSERT INTO {self.FILTER_TABLE_NAME} (
                    market_snapshot_id,
                    asset,
                    filter_name,
                    price_choice,
                    filter_timestamp,
                    filtered_price,
                    basis,
                    spot_error,
                    perp_error,
                    temporary_dislocation,
                    quoted_spot_price,
                    quoted_perp_price,
                    quoted_basis,
                    raw_state_vector_json,
                    raw_covariance_matrix_json,
                    recorded_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    market_snapshot_id,
                    self.asset,
                    filter_name,
                    self._infer_price_choice(filter_name),
                    state.timestamp,
                    state.price,
                    state.basis,
                    state.spot_error,
                    state.perp_error,
                    state.temporary_dislocation,
                    state.quoted_spot_price,
                    state.quoted_perp_price,
                    state.quoted_basis,
                    state.raw_state_vector_json,
                    state.raw_covariance_matrix_json,
                    recorded_at_ms,
                ),
            )
        self.connection.commit()

    def record_active_asset_context(self, measurement: ActiveAssetContextMeasurement) -> None:
        context = measurement.context
        self.connection.execute(
            f"""
            INSERT INTO {self.ASSET_CONTEXT_TABLE_NAME} (
                asset,
                measurement_timestamp,
                observed_market,
                observed_market_type,
                observed_asset,
                observed_channel,
                funding_rate,
                open_interest,
                oracle_price,
                mark_price,
                mid_price,
                premium,
                impact_bid_price,
                impact_ask_price,
                day_notional_volume,
                prev_day_price,
                is_snapshot,
                raw_message_json,
                raw_context_json,
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.asset,
                measurement.timestamp,
                measurement.market,
                measurement.market_type,
                measurement.asset,
                measurement.channel,
                self._context_float(context, "funding"),
                self._context_float(context, "openInterest"),
                self._context_float(context, "oraclePx"),
                self._context_float(context, "markPx"),
                self._context_float(context, "midPx"),
                self._context_float(context, "premium"),
                self._context_float(context, "impactPxs", 0),
                self._context_float(context, "impactPxs", 1),
                self._context_float(context, "dayNtlVlm"),
                self._context_float(context, "prevDayPx"),
                1 if measurement.is_snapshot else 0,
                json.dumps(measurement.raw_message, sort_keys=True, separators=(",", ":")),
                json.dumps(context, sort_keys=True, separators=(",", ":")),
                time.time_ns() // 1_000_000,
            ),
        )
        self.connection.commit()

    def _create_tables(self) -> None:
        self._drop_legacy_price_table()
        self._recreate_table_if_schema_changed(self.MARKET_TABLE_NAME, self._MARKET_COLUMNS)
        self._recreate_table_if_schema_changed(self.FILTER_TABLE_NAME, self._FILTER_COLUMNS)
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.MARKET_TABLE_NAME} (
                {", ".join(f"{column} {definition}" for column, definition in self._MARKET_COLUMNS.items())}
            )
            """
        )
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.FILTER_TABLE_NAME} (
                {", ".join(f"{column} {definition}" for column, definition in self._FILTER_COLUMNS.items())}
            )
            """
        )
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.MARKET_TABLE_NAME}_timestamp
            ON {self.MARKET_TABLE_NAME} (measurement_timestamp)
            """
        )
        self.connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_{self.FILTER_TABLE_NAME}_filter_time
            ON {self.FILTER_TABLE_NAME} (filter_name, filter_timestamp, market_snapshot_id)
            """
        )
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.ASSET_CONTEXT_TABLE_NAME} (
                {", ".join(f"{column} {definition}" for column, definition in self._ASSET_CONTEXT_EXPECTED_COLUMNS.items())}
            )
            """
        )
        self._migrate_asset_context_table()
        self.connection.commit()

    def _drop_legacy_price_table(self) -> None:
        table_names = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if self.LEGACY_TABLE_NAME in table_names:
            self.connection.execute(f"DROP TABLE {self.LEGACY_TABLE_NAME}")

    def _recreate_table_if_schema_changed(
        self,
        table_name: str,
        expected_columns: dict[str, str],
    ) -> None:
        table_names = {
            row[0]
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if table_name not in table_names:
            return
        existing_columns = [
            row[1] for row in self.connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        ]
        if existing_columns == list(expected_columns.keys()):
            return
        self.connection.execute(f"DROP TABLE {table_name}")

    def _migrate_asset_context_table(self) -> None:
        existing_columns = {
            row[1]
            for row in self.connection.execute(
                f"PRAGMA table_info({self.ASSET_CONTEXT_TABLE_NAME})"
            ).fetchall()
        }
        for column, definition in self._ASSET_CONTEXT_EXPECTED_COLUMNS.items():
            if column in existing_columns:
                continue
            self.connection.execute(
                f"ALTER TABLE {self.ASSET_CONTEXT_TABLE_NAME} ADD COLUMN {column} {definition}"
            )

    @staticmethod
    def _infer_price_choice(filter_name: str) -> str:
        return "microprice" if "microprice" in filter_name.lower() else "midprice"

    @staticmethod
    def _as_float(value: Decimal | None) -> float | None:
        if value is None:
            return None
        return float(value)

    @classmethod
    def _context_float(
        cls,
        context: dict[str, Any],
        key: str,
        index: int | None = None,
    ) -> float | None:
        value: Any = context.get(key)
        if index is not None:
            if not isinstance(value, list) or index >= len(value):
                return None
            value = value[index]
        decimal_value = cls._coerce_decimal(value)
        return cls._as_float(decimal_value)

    @staticmethod
    def _coerce_decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int | float | str):
            try:
                return Decimal(str(value))
            except Exception:
                return None
        return None
