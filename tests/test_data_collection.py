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


def test_record_writes_market_snapshots_and_filter_snapshots(tmp_path) -> None:
    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)

    spot_measurement = BBOMeasurement(
        timestamp=101,
        market="BTC",
        market_type="spot",
        asset="BTC",
        channel="bbo",
        raw_message={"channel": "bbo", "data": {"time": 101}},
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
        raw_message={"channel": "bbo", "data": {"time": 102}},
        bid_price=Decimal("100"),
        bid_size=Decimal("4"),
        ask_price=Decimal("102"),
        ask_size=Decimal("2"),
    )

    spot_states = {
        "microprice": PriceBasisState(
            timestamp=101,
            price=100.25,
            basis=0.4,
            spot_error=-0.01,
            perp_error=0.02,
            temporary_dislocation=0.03,
            quoted_spot_price=100.24,
            quoted_perp_price=100.64,
            quoted_basis=0.4,
            raw_state_vector_json="[4.61,0.004,-0.0001,0.0002]",
            raw_covariance_matrix_json="[[1.0,0.0],[0.0,2.0]]",
        ),
    }
    perp_states = {
        "microprice": PriceBasisState(
            timestamp=102,
            price=100.5,
            basis=0.6,
            spot_error=-0.04,
            perp_error=0.05,
            temporary_dislocation=0.09,
            quoted_spot_price=100.46,
            quoted_perp_price=101.15,
            quoted_basis=0.69,
            raw_state_vector_json="[4.62,0.006,-0.0004,0.0005]",
            raw_covariance_matrix_json="[[3.0,0.0],[0.0,4.0]]",
        ),
    }

    collector.record(spot_measurement, spot_states)
    collector.record(perp_measurement, perp_states)
    collector.close()

    connection = sqlite3.connect(tmp_path / "btc_prices.sqlite3")
    market_rows = connection.execute(
        """
        SELECT
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
            raw_message_json
        FROM market_snapshots
        ORDER BY id
        """
    ).fetchall()
    filter_rows = connection.execute(
        """
        SELECT
            fs.asset,
            fs.filter_name,
            fs.price_choice,
            fs.filter_timestamp,
            fs.filtered_price,
            fs.basis,
            fs.spot_error,
            fs.perp_error,
            fs.temporary_dislocation,
            fs.quoted_spot_price,
            fs.quoted_perp_price,
            fs.quoted_basis,
            fs.raw_state_vector_json,
            fs.raw_covariance_matrix_json,
            ms.measurement_timestamp
        FROM filter_snapshots fs
        JOIN market_snapshots ms ON ms.id = fs.market_snapshot_id
        ORDER BY fs.id
        """
    ).fetchall()
    connection.close()

    assert market_rows == [
        (
            "BTC",
            101,
            "BTC",
            "spot",
            "BTC",
            "bbo",
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
            '{"channel":"bbo","data":{"time":101}}',
        ),
        (
            "BTC",
            102,
            "BTC",
            "perp",
            "BTC",
            "bbo",
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
            '{"channel":"bbo","data":{"time":102}}',
        ),
    ]
    assert filter_rows == [
        (
            "BTC",
            "microprice",
            "microprice",
            101,
            100.25,
            0.4,
            -0.01,
            0.02,
            0.03,
            100.24,
            100.64,
            0.4,
            "[4.61,0.004,-0.0001,0.0002]",
            "[[1.0,0.0],[0.0,2.0]]",
            101,
        ),
        (
            "BTC",
            "microprice",
            "microprice",
            102,
            100.5,
            0.6,
            -0.04,
            0.05,
            0.09,
            100.46,
            101.15,
            0.69,
            "[4.62,0.006,-0.0004,0.0005]",
            "[[3.0,0.0],[0.0,4.0]]",
            102,
        ),
    ]


def test_legacy_price_snapshots_table_is_removed(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE price_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER
        )
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    tables = {
        row[0]
        for row in collector.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }
    collector.close()

    assert "price_snapshots" not in tables
    assert "market_snapshots" in tables
    assert "filter_snapshots" in tables


def test_existing_market_snapshot_table_with_wrong_shape_is_rebuilt(tmp_path) -> None:
    db_path = tmp_path / "btc_prices.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER
        )
        """
    )
    connection.execute(
        """
        INSERT INTO market_snapshots (asset, measurement_timestamp) VALUES ('BTC', 1)
        """
    )
    connection.commit()
    connection.close()

    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    columns = [
        row[1]
        for row in collector.connection.execute("PRAGMA table_info(market_snapshots)").fetchall()
    ]
    row_count = collector.connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]
    collector.close()

    assert columns == [
        "id",
        "asset",
        "measurement_timestamp",
        "observed_market",
        "observed_market_type",
        "observed_asset",
        "observed_channel",
        "observed_bid_price",
        "observed_bid_size",
        "observed_ask_price",
        "observed_ask_size",
        "observed_mid_price",
        "observed_microprice",
        "spot_mid_price",
        "perp_mid_price",
        "spot_microprice",
        "perp_microprice",
        "raw_message_json",
        "recorded_at_ms",
    ]
    assert row_count == 0


def test_asset_context_table_is_created_and_records_structured_context(tmp_path) -> None:
    collector = SQLiteDataCollector("BTC", db_dir=tmp_path)
    measurement = ActiveAssetContextMeasurement(
        timestamp=201,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="activeAssetCtx",
        raw_message={"channel": "activeAssetCtx", "data": {"coin": "BTC"}},
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
            raw_context_json
        FROM asset_context_snapshots
        """
    ).fetchone()
    connection.close()

    assert row == (
        "BTC",
        201,
        "BTC",
        "perp",
        "BTC",
        "activeAssetCtx",
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
        '{"channel":"activeAssetCtx","data":{"coin":"BTC"}}',
        (
            '{"dayNtlVlm":"1234567.89","funding":"0.0001","impactPxs":["99.9","100.4"],'
            '"markPx":"100.2","midPx":"100.15","openInterest":"12345.6","oraclePx":"100.1",'
            '"premium":"0.0003","prevDayPx":"98.7"}'
        ),
    )


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
            observed_market,
            observed_market_type,
            observed_asset,
            observed_channel,
            is_snapshot,
            raw_message_json,
            raw_context_json,
            recorded_at_ms
        FROM asset_context_snapshots
        """
    ).fetchone()
    collector.close()

    assert row == ("BTC", 123, "", "perp", "BTC", "", 0, "{}", "{}", 0)
