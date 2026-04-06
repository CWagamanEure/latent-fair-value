import runpy
from pathlib import Path
import sqlite3

from ui.server import PriceSeriesRepository


def _create_snapshot_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT,
            measurement_timestamp INTEGER,
            observed_market_type TEXT,
            observed_asset TEXT,
            observed_bid_price REAL,
            observed_bid_size REAL,
            observed_ask_price REAL,
            observed_ask_size REAL,
            observed_mid_price REAL,
            observed_microprice REAL,
            spot_mid_price REAL,
            perp_mid_price REAL,
            spot_microprice REAL,
            perp_microprice REAL,
            recorded_at_ms INTEGER
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE filter_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_snapshot_id INTEGER NOT NULL,
            asset TEXT NOT NULL,
            filter_name TEXT NOT NULL,
            price_choice TEXT NOT NULL,
            filter_timestamp INTEGER,
            filtered_price REAL NOT NULL,
            basis REAL NOT NULL,
            spot_error REAL,
            perp_error REAL,
            temporary_dislocation REAL,
            quoted_spot_price REAL,
            quoted_perp_price REAL,
            quoted_basis REAL,
            recorded_at_ms INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE asset_context_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset TEXT NOT NULL,
            measurement_timestamp INTEGER,
            observed_market_type TEXT NOT NULL,
            observed_asset TEXT,
            funding_rate REAL,
            open_interest REAL,
            oracle_price REAL,
            mark_price REAL,
            mid_price REAL,
            premium REAL,
            impact_bid_price REAL,
            impact_ask_price REAL,
            day_notional_volume REAL,
            prev_day_price REAL,
            is_snapshot INTEGER NOT NULL DEFAULT 0,
            raw_context_json TEXT NOT NULL DEFAULT '{}',
            recorded_at_ms INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def _build_test_db(db_path: Path, row_count: int) -> None:
    connection = sqlite3.connect(db_path)
    _create_snapshot_tables(connection)
    for index in range(row_count):
        cursor = connection.execute(
            """
            INSERT INTO market_snapshots (
                asset,
                measurement_timestamp,
                observed_market_type,
                observed_asset,
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
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BTC",
                1_000 + index * 100,
                "perp" if index % 2 else "spot",
                "BTC",
                99.0 + index,
                2.0 + index,
                101.0 + index,
                3.0 + index,
                100.0 + index,
                100.2 + index,
                100.0 + index,
                101.0 + index,
                100.1 + index,
                100.9 + index,
                1_010 + index * 100,
            ),
        )
        connection.execute(
            """
            INSERT INTO filter_snapshots (
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
                recorded_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cursor.lastrowid,
                "BTC",
                "microprice",
                "microprice",
                1_000 + index * 100,
                100.5 + index,
                1.0 + index * 0.01,
                -0.001 * index,
                0.002 * index,
                0.003 * index,
                100.0 + index - 0.001 * index,
                101.0 + index + 0.002 * index,
                1.0 + index * 0.01 + 0.003 * index,
                1_010 + index * 100,
            ),
        )

    connection.execute(
        """
        INSERT INTO asset_context_snapshots (
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
            "BTC",
            6_000,
            "perp",
            "BTC",
            0.0001,
            12_345.0,
            106.2,
            106.1,
            106.15,
            0.0012,
            105.9,
            106.3,
            9_876_543.0,
            101.0,
            1,
            "{}",
            6_050,
        ),
    )
    connection.commit()
    connection.close()


def test_fetch_series_returns_raw_points_when_window_is_small(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    _build_test_db(db_path, row_count=20)
    repository = PriceSeriesRepository(db_path)

    payload = repository.fetch_series(window_ms=10_000, max_points=50)

    assert payload["meta"]["downsampled"] is False
    assert payload["meta"]["pointCount"] == 20
    assert payload["meta"]["filterName"] == "microprice"
    assert payload["meta"]["priceChoice"] == "microprice"
    assert payload["points"][0]["timestamp"] == 1_000
    assert payload["points"][-1]["timestamp"] == 2_900
    assert payload["points"][0]["selectedSpotPrice"] == 100.1
    assert payload["points"][0]["selectedPerpPrice"] == 100.9
    assert payload["points"][-1]["temporaryDislocation"] == 0.057


def test_fetch_series_downsamples_large_ranges(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    _build_test_db(db_path, row_count=500)
    repository = PriceSeriesRepository(db_path)

    payload = repository.fetch_series(window_ms=60_000, max_points=40)

    assert payload["meta"]["downsampled"] is True
    assert payload["meta"]["pointCount"] <= 40
    assert payload["meta"]["bucketMs"] is not None


def test_fetch_series_since_timestamp_returns_incremental_points(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    _build_test_db(db_path, row_count=50)
    repository = PriceSeriesRepository(db_path)

    payload = repository.fetch_series(window_ms=60_000, max_points=10, since_ts=4_000)

    assert payload["meta"]["downsampled"] is False
    assert payload["points"][0]["timestamp"] == 4_100
    assert payload["points"][-1]["timestamp"] == 5_000
    assert len(payload["points"]) == 10


def test_fetch_live_snapshot_returns_latest_measurement_state_and_context(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    _build_test_db(db_path, row_count=50)
    repository = PriceSeriesRepository(db_path)

    payload = repository.fetch_live_snapshot()

    assert payload["measurement"]["timestamp"] == 5_900
    assert payload["measurement"]["marketType"] == "perp"
    assert payload["measurement"]["midPrice"] == 149.0
    assert payload["measurement"]["microprice"] == 149.2
    assert payload["measurement"]["spotMidPrice"] == 149.0
    assert payload["measurement"]["perpMidPrice"] == 150.0
    assert payload["measurement"]["spotMicroprice"] == 149.1
    assert payload["measurement"]["perpMicroprice"] == 149.9
    assert payload["filter_state"]["filterName"] == "microprice"
    assert payload["filter_state"]["priceChoice"] == "microprice"
    assert payload["filter_state"]["timestamp"] == 5_900
    assert payload["filter_state"]["price"] == 149.5
    assert payload["filter_state"]["basis"] == 1.49
    assert payload["filter_state"]["spotError"] == -0.049
    assert payload["filter_state"]["perpError"] == 0.098
    assert payload["filter_state"]["temporaryDislocation"] == 0.147
    assert payload["filter_state"]["quotedSpotPrice"] == 148.951
    assert payload["filter_state"]["quotedPerpPrice"] == 150.098
    assert payload["filter_state"]["quotedBasis"] == 1.637
    assert payload["context"]["timestamp"] == 6_000
    assert payload["context"]["fundingRate"] == 0.0001
    assert payload["context"]["openInterest"] == 12_345.0


def test_repository_preserves_zero_values_in_live_and_series_payloads(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    connection = sqlite3.connect(db_path)
    _create_snapshot_tables(connection)
    cursor = connection.execute(
        """
        INSERT INTO market_snapshots (
            asset,
            measurement_timestamp,
            observed_market_type,
            observed_asset,
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
            recorded_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "BTC",
            100,
            "spot",
            "BTC",
            99.0,
            1.0,
            101.0,
            1.0,
            100.0,
            100.0,
            100.0,
            0.0,
            100.0,
            0.0,
            500,
        ),
    )
    connection.execute(
        """
        INSERT INTO filter_snapshots (
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
            recorded_at_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cursor.lastrowid,
            "BTC",
            "microprice",
            "microprice",
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            500,
        ),
    )
    connection.commit()
    connection.close()

    repository = PriceSeriesRepository(db_path)

    live_payload = repository.fetch_live_snapshot()
    series_payload = repository.fetch_series(window_ms=10_000, max_points=10)

    assert live_payload["filter_state"]["timestamp"] == 0
    assert live_payload["filter_state"]["price"] == 0.0
    assert live_payload["filter_state"]["basis"] == 0.0
    assert live_payload["filter_state"]["temporaryDislocation"] == 0.0
    assert live_payload["filter_state"]["quotedSpotPrice"] == 0.0
    assert live_payload["filter_state"]["quotedPerpPrice"] == 0.0
    assert live_payload["filter_state"]["quotedBasis"] == 0.0
    assert series_payload["points"][0]["filterPrice"] == 0.0
    assert series_payload["points"][0]["filterBasis"] == 0.0
    assert series_payload["points"][0]["temporaryDislocation"] == 0.0
    assert series_payload["points"][0]["quotedSpotPrice"] == 0.0
    assert series_payload["points"][0]["quotedPerpPrice"] == 0.0
    assert series_payload["points"][0]["quotedBasis"] == 0.0


def test_fetch_live_snapshot_returns_empty_payload_when_no_measurements_exist(tmp_path) -> None:
    db_path = tmp_path / "prices.sqlite3"
    connection = sqlite3.connect(db_path)
    _create_snapshot_tables(connection)
    connection.commit()
    connection.close()
    repository = PriceSeriesRepository(db_path)

    payload = repository.fetch_live_snapshot()

    assert payload == {"measurement": None, "filter_state": None, "context": None}


def test_ui_module_entrypoint_invokes_main(monkeypatch) -> None:
    called = False

    def fake_main() -> None:
        nonlocal called
        called = True

    monkeypatch.setattr("src.ui.server.main", fake_main)

    runpy.run_module("ui.server", run_name="__main__")

    assert called is True
