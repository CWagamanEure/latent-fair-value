import sqlite3
from decimal import Decimal

from src.data_collection import SQLiteDataCollector
from src.filters.latent_state_types import PriceBasisState
from src.measurement_types import ActiveAssetContextMeasurement, BBOMeasurement


def test_build_db_path_is_asset_scoped(tmp_path) -> None:
    btc_path = SQLiteDataCollector.build_db_path("BTC", tmp_path)
    hype_path = SQLiteDataCollector.build_db_path("HYPE", tmp_path)

    assert btc_path == tmp_path / "btc_prices.sqlite3"
    assert hype_path == tmp_path / "hype_prices.sqlite3"
    assert btc_path != hype_path


def test_record_writes_latest_spot_and_perp_values(tmp_path) -> None:
    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)

    spot_measurement = BBOMeasurement(
        timestamp=101,
        market="BTC",
        market_type="spot",
        asset="BTC",
        channel="bbo",
        raw_message={},
        bid_price=Decimal("99"),
        bid_size=Decimal("2"),
        ask_price=Decimal("101"),
        ask_size=Decimal("1"),
    )
    perp_measurement = BBOMeasurement(
        timestamp=102,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="bbo",
        raw_message={},
        bid_price=Decimal("100"),
        bid_size=Decimal("4"),
        ask_price=Decimal("102"),
        ask_size=Decimal("2"),
    )

    spot_states = {
        "midprice": PriceBasisState(timestamp=101, price=100.25, basis=0.4),
        "microprice_1p5x": PriceBasisState(timestamp=101, price=100.2, basis=0.35),
        "microprice_3x": PriceBasisState(timestamp=101, price=100.15, basis=0.3),
    }
    perp_states = {
        "midprice": PriceBasisState(timestamp=102, price=100.5, basis=0.6),
        "microprice_1p5x": PriceBasisState(timestamp=102, price=100.45, basis=0.55),
        "microprice_3x": PriceBasisState(timestamp=102, price=100.4, basis=0.5),
    }

    collector.record(spot_measurement, spot_states)
    collector.record(perp_measurement, perp_states)
    collector.close()

    connection = sqlite3.connect(tmp_path / "btc_prices.sqlite3")
    rows = connection.execute(
        """
        SELECT
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
            microprice_3x_basis
        FROM price_snapshots
        ORDER BY id
        """
    ).fetchall()
    connection.close()

    assert rows == [
        (
            "BTC",
            101,
            101,
            "spot",
            "BTC",
            99.0,
            2.0,
            101.0,
            1.0,
            100.0,
            100.33333333333333,
            100.0,
            None,
            100.33333333333333,
            None,
            100.25,
            0.4,
            101,
            100.25,
            0.4,
            101,
            100.2,
            0.35,
            101,
            100.15,
            0.3,
        ),
        (
            "BTC",
            102,
            102,
            "perp",
            "BTC",
            100.0,
            4.0,
            102.0,
            2.0,
            101.0,
            101.33333333333333,
            100.0,
            101.0,
            100.33333333333333,
            101.33333333333333,
            100.5,
            0.6,
            102,
            100.5,
            0.6,
            102,
            100.45,
            0.55,
            102,
            100.4,
            0.5,
        ),
    ]


def test_existing_table_is_migrated_with_new_observation_columns(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER,
            observed_market_type TEXT NOT NULL,
            spot_price REAL,
            perp_price REAL,
            spot_microprice REAL,
            perp_microprice REAL,
            filtered_price REAL NOT NULL,
            basis REAL NOT NULL,
            recorded_at_ms INTEGER NOT NULL
        )
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    columns = [
        row[1]
        for row in collector.connection.execute("PRAGMA table_info(price_snapshots)").fetchall()
    ]
    collector.close()

    assert "filtered_timestamp" in columns
    assert "observed_asset" in columns
    assert "observed_bid_price" in columns
    assert "observed_mid_price" in columns
    assert "midprice_filtered_price" in columns
    assert "microprice_1p5x_basis" in columns
    assert "microprice_3x_filtered_timestamp" in columns


def test_asset_context_table_is_created_and_records_structured_context(tmp_path) -> None:
    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    measurement = ActiveAssetContextMeasurement(
        timestamp=201,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="activeAssetCtx",
        raw_message={},
        context={
            "funding": "0.0001",
            "openInterest": "12345.6",
            "oraclePx": "100.1",
            "markPx": "100.2",
            "midPx": "100.15",
            "premium": "0.0003",
            "impactPxs": ["99.9", "100.4"],
            "dayNtlVlm": "1234567.89",
            "prevDayPx": "98.7",
        },
        is_snapshot=True,
    )

    collector.record_active_asset_context(measurement)
    collector.close()

    connection = sqlite3.connect(tmp_path / "btc_prices.sqlite3")
    row = connection.execute(
        """
        SELECT
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
            raw_context_json
        FROM asset_context_snapshots
        """
    ).fetchone()
    connection.close()

    assert row == (
        "BTC",
        201,
        "perp",
        "BTC",
        0.0001,
        12345.6,
        100.1,
        100.2,
        100.15,
        0.0003,
        99.9,
        100.4,
        1234567.89,
        98.7,
        1,
        (
            '{"dayNtlVlm":"1234567.89","funding":"0.0001","impactPxs":["99.9","100.4"],'
            '"markPx":"100.2","midPx":"100.15","openInterest":"12345.6","oraclePx":"100.1",'
            '"premium":"0.0003","prevDayPx":"98.7"}'
        ),
    )


def test_existing_asset_context_table_is_migrated_with_new_columns(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE asset_context_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER,
            observed_market_type TEXT NOT NULL,
            observed_asset TEXT
        )
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    columns = [
        row[1]
        for row in collector.connection.execute("PRAGMA table_info(asset_context_snapshots)").fetchall()
    ]
    collector.close()

    assert "funding_rate" in columns
    assert "open_interest" in columns
    assert "oracle_price" in columns
    assert "mark_price" in columns
    assert "mid_price" in columns
    assert "premium" in columns
    assert "impact_bid_price" in columns
    assert "impact_ask_price" in columns
    assert "day_notional_volume" in columns
    assert "prev_day_price" in columns
    assert "raw_context_json" in columns


def test_existing_asset_context_table_with_rows_is_migrated_without_failure(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE asset_context_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER,
            observed_market_type TEXT NOT NULL,
            observed_asset TEXT
        )
        """
    )
    connection.execute(
        """
        INSERT INTO asset_context_snapshots (
            asset,
            measurement_timestamp,
            observed_market_type,
            observed_asset
        ) VALUES ('BTC', 123, 'perp', 'BTC')
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    row = collector.connection.execute(
        """
        SELECT
            asset,
            measurement_timestamp,
            observed_market_type,
            observed_asset,
            is_snapshot,
            raw_context_json,
            recorded_at_ms
        FROM asset_context_snapshots
        """
    ).fetchone()
    collector.close()

    assert row == ("BTC", 123, "perp", "BTC", 0, "{}", 0)


def test_obsolete_filter_columns_trigger_table_rebuild(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER,
            observed_market_type TEXT NOT NULL,
            microprice_2x_filtered_timestamp INTEGER,
            microprice_2x_filtered_price REAL,
            microprice_2x_basis REAL,
            filtered_price REAL NOT NULL,
            basis REAL NOT NULL,
            recorded_at_ms INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO price_snapshots (
            asset,
            measurement_timestamp,
            observed_market_type,
            microprice_2x_filtered_timestamp,
            microprice_2x_filtered_price,
            microprice_2x_basis,
            filtered_price,
            basis,
            recorded_at_ms
        ) VALUES ('BTC', 1, 'spot', 1, 100.0, 0.1, 100.0, 0.1, 1)
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    columns = [
        row[1]
        for row in collector.connection.execute("PRAGMA table_info(price_snapshots)").fetchall()
    ]
    row_count = collector.connection.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    collector.close()

    assert "microprice_2x_filtered_timestamp" not in columns
    assert "microprice_1p5x_filtered_timestamp" in columns
    assert row_count == 0
