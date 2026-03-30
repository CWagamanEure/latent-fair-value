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


FILTER_COLUMN_NAMES: dict[str, tuple[str, str, str]] = {
    "midprice": (
        "midprice_filtered_timestamp",
        "midprice_filtered_price",
        "midprice_basis",
    ),
    "microprice_1p5x": (
        "microprice_1p5x_filtered_timestamp",
        "microprice_1p5x_filtered_price",
        "microprice_1p5x_basis",
    ),
    "microprice_3x": (
        "microprice_3x_filtered_timestamp",
        "microprice_3x_filtered_price",
        "microprice_3x_basis",
    ),
}


@dataclass
class BBOState:
    price: Decimal | None = None
    microprice: Decimal | None = None


class SQLiteDataCollector:
    TABLE_NAME = "price_snapshots"
    ASSET_CONTEXT_TABLE_NAME = "asset_context_snapshots"
    _OBSOLETE_COLUMNS = {
        "microprice_2x_filtered_timestamp",
        "microprice_2x_filtered_price",
        "microprice_2x_basis",
        "microprice_4x_filtered_timestamp",
        "microprice_4x_filtered_price",
        "microprice_4x_basis",
    }
    _EXPECTED_COLUMNS: dict[str, str] = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "asset": "TEXT NOT NULL",
        "measurement_timestamp": "INTEGER",
        "filtered_timestamp": "INTEGER",
        "observed_market_type": "TEXT NOT NULL",
        "observed_asset": "TEXT",
        "observed_bid_price": "REAL",
        "observed_bid_size": "REAL",
        "observed_ask_price": "REAL",
        "observed_ask_size": "REAL",
        "observed_mid_price": "REAL",
        "observed_microprice": "REAL",
        "spot_price": "REAL",
        "perp_price": "REAL",
        "spot_microprice": "REAL",
        "perp_microprice": "REAL",
        "filtered_price": "REAL NOT NULL",
        "basis": "REAL NOT NULL",
        "midprice_filtered_timestamp": "INTEGER",
        "midprice_filtered_price": "REAL",
        "midprice_basis": "REAL",
        "microprice_1p5x_filtered_timestamp": "INTEGER",
        "microprice_1p5x_filtered_price": "REAL",
        "microprice_1p5x_basis": "REAL",
        "microprice_3x_filtered_timestamp": "INTEGER",
        "microprice_3x_filtered_price": "REAL",
        "microprice_3x_basis": "REAL",
        "recorded_at_ms": "INTEGER NOT NULL",
    }
    _ASSET_CONTEXT_EXPECTED_COLUMNS: dict[str, str] = {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "asset": "TEXT NOT NULL",
        "measurement_timestamp": "INTEGER",
        "observed_market_type": "TEXT NOT NULL",
        "observed_asset": "TEXT",
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
        "raw_context_json": "TEXT NOT NULL DEFAULT '{}'",
        "recorded_at_ms": "INTEGER NOT NULL DEFAULT 0",
    }

    def __init__(self, asset: str, db_dir: str | Path = "data") -> None:
        self.asset = asset.upper()
        self.db_path = self.build_db_path(self.asset, db_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self._spot = BBOState()
        self._perp = BBOState()
        self._create_table()

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
        if "midprice" not in filtered_states:
            raise ValueError("filtered_states must include a 'midprice' entry")

        snapshot = BBOState(price=measurement.mid(), microprice=measurement.microprice())
        if measurement.is_spot:
            self._spot = snapshot
        elif measurement.is_perp:
            self._perp = snapshot
        else:
            raise ValueError("BBOMeasurement must have either spot or perp market_type")

        baseline_state = filtered_states["midprice"]
        filter_column_values: dict[str, float | int | None] = {}
        for filter_name, (ts_column, price_column, basis_column) in FILTER_COLUMN_NAMES.items():
            state = filtered_states.get(filter_name)
            filter_column_values[ts_column] = None if state is None else state.timestamp
            filter_column_values[price_column] = None if state is None else state.price
            filter_column_values[basis_column] = None if state is None else state.basis

        self.connection.execute(
            """
            INSERT INTO price_snapshots (
                asset,
                measurement_timestamp,
                filtered_timestamp,
                observed_market_type,
                observed_asset,
                observed_bid_price,
                observed_bid_size,
                observed_ask_price,
                observed_ask_size,
                observed_mid_price,
                observed_microprice,
                spot_price,
                perp_price,
                spot_microprice,
                perp_microprice,
                filtered_price,
                basis,
                midprice_filtered_timestamp,
                midprice_filtered_price,
                midprice_basis,
                microprice_1p5x_filtered_timestamp,
                microprice_1p5x_filtered_price,
                microprice_1p5x_basis,
                microprice_3x_filtered_timestamp,
                microprice_3x_filtered_price,
                microprice_3x_basis,
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.asset,
                measurement.timestamp,
                baseline_state.timestamp,
                measurement.market_type,
                measurement.asset,
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
                baseline_state.price,
                baseline_state.basis,
                filter_column_values["midprice_filtered_timestamp"],
                filter_column_values["midprice_filtered_price"],
                filter_column_values["midprice_basis"],
                filter_column_values["microprice_1p5x_filtered_timestamp"],
                filter_column_values["microprice_1p5x_filtered_price"],
                filter_column_values["microprice_1p5x_basis"],
                filter_column_values["microprice_3x_filtered_timestamp"],
                filter_column_values["microprice_3x_filtered_price"],
                filter_column_values["microprice_3x_basis"],
                time.time_ns() // 1_000_000,
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
                observed_market_type,
                observed_asset,
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
                raw_context_json,
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.asset,
                measurement.timestamp,
                measurement.market_type,
                measurement.asset,
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
                json.dumps(context, sort_keys=True, separators=(",", ":")),
                time.time_ns() // 1_000_000,
            ),
        )
        self.connection.commit()

    def _create_table(self) -> None:
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                {", ".join(f"{column} {definition}" for column, definition in self._EXPECTED_COLUMNS.items())}
            )
            """
        )
        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.ASSET_CONTEXT_TABLE_NAME} (
                {", ".join(f"{column} {definition}" for column, definition in self._ASSET_CONTEXT_EXPECTED_COLUMNS.items())}
            )
            """
        )
        self._migrate_table()
        self._migrate_asset_context_table()
        self.connection.commit()

    def _migrate_table(self) -> None:
        existing_columns = {
            row[1]
            for row in self.connection.execute(f"PRAGMA table_info({self.TABLE_NAME})").fetchall()
        }
        if existing_columns & self._OBSOLETE_COLUMNS:
            self.connection.execute(f"DROP TABLE {self.TABLE_NAME}")
            self.connection.execute(
                f"""
                CREATE TABLE {self.TABLE_NAME} (
                    {", ".join(f"{column} {definition}" for column, definition in self._EXPECTED_COLUMNS.items())}
                )
                """
            )
            return
        for column, definition in self._EXPECTED_COLUMNS.items():
            if column in existing_columns:
                continue
            self.connection.execute(
                f"ALTER TABLE {self.TABLE_NAME} ADD COLUMN {column} {definition}"
            )

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
