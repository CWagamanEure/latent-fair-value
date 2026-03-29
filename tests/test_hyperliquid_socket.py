from decimal import Decimal

from src.measurement_manager import MeasurementManager
from src.measurement_types import (
    ActiveAssetContextMeasurement,
    BBOMeasurement,
    CandleMeasurement,
    L2BookMeasurement,
    RawMeasurement,
    TradeMeasurement,
)
from src.sockets.hyperliquid_socket import HyperliquidSocket


def test_subscriptions_include_expected_channels() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval="5m")

    assert socket._subscriptions() == [
        {"type": "trades", "coin": "BTC"},
        {"type": "l2Book", "coin": "BTC"},
        {"type": "bbo", "coin": "BTC"},
        {"type": "activeAssetCtx", "coin": "BTC"},
        {"type": "candle", "coin": "BTC", "interval": "5m"},
    ]


def test_subscriptions_use_explicit_subscription_coin_when_provided() -> None:
    socket = HyperliquidSocket(
        market="btc",
        market_type="spot",
        subscription_coin="@107",
        candle_interval="5m",
    )

    assert socket._subscriptions() == [
        {"type": "trades", "coin": "@107"},
        {"type": "l2Book", "coin": "@107"},
        {"type": "bbo", "coin": "@107"},
        {"type": "activeAssetCtx", "coin": "@107"},
        {"type": "candle", "coin": "@107", "interval": "5m"},
    ]


def test_extract_timestamp_handles_dict_and_list_payloads() -> None:
    assert MeasurementManager.extract_timestamp({"data": {"timestamp": 123}}) == 123
    assert MeasurementManager.extract_timestamp({"data": [{"t": 456}]}) == 456
    assert MeasurementManager.extract_timestamp({"data": "ignored"}) is None


def test_stream_market_updates_emits_trade_measurements() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {
        "channel": "trades",
        "data": [
            {"coin": "BTC", "side": "B", "px": "101.5", "sz": "0.25", "time": 111, "tid": 7},
            {"coin": "BTC", "side": "A", "px": "102.0", "sz": "0.10", "time": 112, "hash": "0xabc"},
        ],
    }

    measurements = list(socket._convert_message_to_measurements(message))

    assert measurements == [
        TradeMeasurement(
            timestamp=111,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="trades",
            raw_message=message,
            side="B",
            price=Decimal("101.5"),
            size=Decimal("0.25"),
            trade_id=7,
            hash=None,
        ),
        TradeMeasurement(
            timestamp=112,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="trades",
            raw_message=message,
            side="A",
            price=Decimal("102.0"),
            size=Decimal("0.10"),
            trade_id=None,
            hash="0xabc",
        ),
    ]


def test_stream_market_updates_emits_l2_book_measurement() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": 123,
            "isSnapshot": True,
            "levels": [
                [{"px": "100", "sz": "1.2", "n": 2}],
                [{"px": "101", "sz": "0.8", "n": 1}],
            ],
        },
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert isinstance(measurement, L2BookMeasurement)
    assert measurement.timestamp == 123
    assert measurement.market_type == "perp"
    assert measurement.is_perp is True
    assert measurement.is_snapshot is True
    assert measurement.bids[0].price == Decimal("100")
    assert measurement.asks[0].size == Decimal("0.8")


def test_stream_market_updates_emits_bbo_measurement() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {
        "channel": "bbo",
        "data": {
            "coin": "BTC",
            "time": 200,
            "bbo": [
                {"px": "99", "sz": "3", "n": 1},
                {"px": "100", "sz": "4", "n": 2},
            ],
        },
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == BBOMeasurement(
        timestamp=200,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="bbo",
        raw_message=message,
        bid_price=Decimal("99"),
        bid_size=Decimal("3"),
        ask_price=Decimal("100"),
        ask_size=Decimal("4"),
    )
    assert measurement.mid() == Decimal("99.5")
    assert measurement.microprice() == Decimal("99.42857142857142857142857143")


def test_stream_market_updates_emits_bbo_measurement_from_legacy_dict_shape() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {
        "channel": "bbo",
        "data": {"coin": "BTC", "time": 200, "bbo": {"bid": "99", "bidSz": "3", "ask": "100", "askSz": "4"}},
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == BBOMeasurement(
        timestamp=200,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="bbo",
        raw_message=message,
        bid_price=Decimal("99"),
        bid_size=Decimal("3"),
        ask_price=Decimal("100"),
        ask_size=Decimal("4"),
    )


def test_stream_market_updates_normalizes_spot_subscription_asset_aliases() -> None:
    socket = HyperliquidSocket(
        market="btc",
        market_type="spot",
        subscription_coin="@107",
        candle_interval=None,
    )
    message = {
        "channel": "bbo",
        "data": {
            "coin": "@107",
            "time": 200,
            "bbo": [
                {"px": "99", "sz": "3", "n": 1},
                {"px": "100", "sz": "4", "n": 2},
            ],
        },
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == BBOMeasurement(
        timestamp=200,
        market="BTC",
        market_type="spot",
        asset="BTC",
        channel="bbo",
        raw_message=message,
        bid_price=Decimal("99"),
        bid_size=Decimal("3"),
        ask_price=Decimal("100"),
        ask_size=Decimal("4"),
    )


def test_bbo_helpers_return_none_for_incomplete_or_zero_size_inputs() -> None:
    missing_prices = BBOMeasurement(
        timestamp=200,
        market="BTC",
        market_type="unknown",
        asset="BTC",
        channel="bbo",
        raw_message={},
        bid_price=None,
        bid_size=Decimal("3"),
        ask_price=Decimal("100"),
        ask_size=Decimal("4"),
    )
    zero_size = BBOMeasurement(
        timestamp=200,
        market="BTC",
        market_type="unknown",
        asset="BTC",
        channel="bbo",
        raw_message={},
        bid_price=Decimal("99"),
        bid_size=Decimal("0"),
        ask_price=Decimal("100"),
        ask_size=Decimal("0"),
    )

    assert missing_prices.mid() is None
    assert missing_prices.microprice() is None
    assert zero_size.microprice() is None


def test_stream_market_updates_emits_candle_measurement() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval="1m")
    message = {
        "channel": "candle",
        "data": {
            "s": "BTC",
            "i": "1m",
            "t": 300,
            "T": 359,
            "o": "10",
            "h": "11",
            "l": "9",
            "c": "10.5",
            "v": "42",
            "n": 8,
        },
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == CandleMeasurement(
        timestamp=300,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="candle",
        raw_message=message,
        interval="1m",
        open_time=300,
        close_time=359,
        open_price=Decimal("10"),
        high_price=Decimal("11"),
        low_price=Decimal("9"),
        close_price=Decimal("10.5"),
        volume=Decimal("42"),
        trade_count=8,
    )


def test_stream_market_updates_emits_active_asset_context_measurement() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {
        "channel": "activeAssetCtx",
        "data": {"coin": "BTC", "ctx": {"funding": "0.0001"}, "isSnapshot": True},
    }

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == ActiveAssetContextMeasurement(
        timestamp=None,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="activeAssetCtx",
        raw_message=message,
        context={"funding": "0.0001"},
        is_snapshot=True,
    )


def test_stream_market_updates_falls_back_to_raw_measurement() -> None:
    socket = HyperliquidSocket(market="btc", market_type="perp", candle_interval=None)
    message = {"channel": "mids", "data": {"coin": "BTC", "time": 999, "value": "1"}}

    measurement = socket._convert_message_to_measurements(message)[0]

    assert measurement == RawMeasurement(
        timestamp=999,
        market="BTC",
        market_type="perp",
        asset="BTC",
        channel="mids",
        raw_message=message,
        data={"coin": "BTC", "time": 999, "value": "1"},
        is_snapshot=False,
    )
