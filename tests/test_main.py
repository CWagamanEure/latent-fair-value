import pytest

from src import main
from src.main import build_parser
from src.market_feeds import build_market_feeds, lookup_spot_subscription_coin, resolve_spot_subscription_coin


def test_parser_defaults_to_both_market_feeds() -> None:
    args = build_parser().parse_args(["btc"])

    assert args.market == "btc"
    assert args.market_scope == "both"
    assert args.db_dir == "data"
    assert args.spot_coin is None


def test_build_market_feeds_includes_spot_and_perp_by_default() -> None:
    from src import market_feeds

    original_resolver = market_feeds.resolve_spot_subscription_coin
    market_feeds.resolve_spot_subscription_coin = lambda market, *, testnet: "@107"
    try:
        feeds = build_market_feeds(
            "btc",
            market_scope="both",
            spot_coin=None,
            testnet=False,
            candle_interval="1m",
        )
    finally:
        market_feeds.resolve_spot_subscription_coin = original_resolver

    assert [feed.socket.market_type for feed in feeds] == ["perp", "spot"]
    assert [feed.measurement_manager.market_type for feed in feeds] == ["perp", "spot"]
    assert all(feed.socket.market == "BTC" for feed in feeds)
    assert [feed.socket.subscription_coin for feed in feeds] == ["BTC", "@107"]


def test_build_market_feeds_can_limit_to_single_market_type() -> None:
    from src import market_feeds

    original_resolver = market_feeds.resolve_spot_subscription_coin
    market_feeds.resolve_spot_subscription_coin = lambda market, *, testnet: "@42"
    try:
        feeds = build_market_feeds(
            "eth",
            market_scope="spot",
            spot_coin=None,
            testnet=True,
            candle_interval=None,
        )
    finally:
        market_feeds.resolve_spot_subscription_coin = original_resolver

    assert len(feeds) == 1
    assert feeds[0].socket.market == "ETH"
    assert feeds[0].socket.market_type == "spot"
    assert feeds[0].socket.subscription_coin == "@42"
    assert feeds[0].measurement_manager.market_type == "spot"


def test_build_market_feeds_uses_explicit_spot_override_when_provided() -> None:
    feeds = build_market_feeds(
        "btc",
        market_scope="both",
        spot_coin="@999",
        testnet=False,
        candle_interval="1m",
    )

    assert [feed.socket.subscription_coin for feed in feeds] == ["BTC", "@999"]


def test_lookup_spot_subscription_coin_prefers_pair_index_for_at_pairs() -> None:
    spot_meta = {
        "tokens": [
            {"index": 0, "name": "USDC"},
            {"index": 150, "name": "HYPE"},
        ],
        "universe": [
            {"index": 107, "name": "@107", "tokens": [150, 0]},
        ],
    }

    assert lookup_spot_subscription_coin("HYPE", spot_meta) == "@107"


def test_lookup_spot_subscription_coin_uses_aliases_for_ui_symbol_remaps() -> None:
    spot_meta = {
        "tokens": [
            {"index": 0, "name": "USDC"},
            {"index": 12, "name": "UBTC"},
        ],
        "universe": [
            {"index": 5, "name": "@5", "tokens": [12, 0]},
        ],
    }

    assert lookup_spot_subscription_coin("BTC", spot_meta) is None
    assert lookup_spot_subscription_coin("UBTC", spot_meta) == "@5"


def test_resolve_spot_subscription_coin_raises_for_unknown_market() -> None:
    from src import market_feeds

    original_fetch = market_feeds.fetch_spot_meta
    market_feeds.fetch_spot_meta = lambda *, testnet: {"tokens": [], "universe": []}
    try:
        with pytest.raises(ValueError, match="Use --spot-coin to override"):
            resolve_spot_subscription_coin("btc", testnet=False)
    finally:
        market_feeds.fetch_spot_meta = original_fetch


def test_build_filter_runs_uses_requested_initial_state_and_params() -> None:
    filter_runs = main.build_filter_runs(
        market="BTC",
        first_timestamp=123,
        first_spot_mid=100.0,
        first_perp_mid=101.5,
    )

    assert [filter_run.name for filter_run in filter_runs] == [
        "midprice",
        "microprice_2x",
        "microprice_4x",
    ]

    expected_multipliers = {
        "midprice": (1.0, "midprice"),
        "microprice_2x": (2.0, "microprice"),
        "microprice_4x": (4.0, "microprice"),
    }
    for filter_run in filter_runs:
        kalman_filter = filter_run.kalman_filter
        expected_microprice_r_mult, expected_price_choice = expected_multipliers[filter_run.name]
        assert kalman_filter.price_choice == expected_price_choice
        assert kalman_filter.microprice_r_mult == expected_microprice_r_mult
        assert kalman_filter.spot_r_mult == 1.0
        assert kalman_filter.perp_r_mult == 1.25
        assert kalman_filter.min_measurement_var == 0.25
        assert kalman_filter.state.timestamp == 123
        assert kalman_filter.state.price == 100.0
        assert kalman_filter.state.basis == 1.5
        assert kalman_filter.cov.timestamp == 123
        assert kalman_filter.cov.matrix.tolist() == [[100.0, 0.0], [0.0, 25.0]]
        assert kalman_filter.F.tolist() == [[1.0, 0.0], [0.0, 1.0]]
        assert kalman_filter.Q.tolist() == [[0.1, 0.0], [0.0, 0.01]]
