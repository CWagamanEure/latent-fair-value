import asyncio
from decimal import Decimal

import numpy as np
import pytest

from src import main
from src.filters.kalman_filter import KalmanFilter
from src.filters.latent_state_types import FilterSettings, PriceBasisErrorState, StateCovariance
from src.main import build_parser
from src.measurement_types import ActiveAssetContextMeasurement, BBOMeasurement
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

    assert [filter_run.name for filter_run in filter_runs] == ["microprice"]

    kalman_filter = filter_runs[0].kalman_filter
    assert kalman_filter.price_choice == "microprice"
    assert kalman_filter.microprice_r_mult == 1.0
    assert kalman_filter.spot_r_mult == 1.0
    assert kalman_filter.perp_r_mult == 1.25
    assert kalman_filter.min_measurement_var == 5.0e-11
    assert kalman_filter.price_var_per_sec == 1.0e-8
    assert kalman_filter.basis_var_per_sec == 2.5e-8
    assert kalman_filter.error_kappa == 2.0
    assert kalman_filter.basis_kappa == 0.05
    assert kalman_filter.basis_long_run_mean == 0.0
    assert kalman_filter.spot_error_var_per_sec == 1.0e-7
    assert kalman_filter.perp_error_var_per_sec == 1.0e-7
    assert kalman_filter.state.timestamp == 123
    assert kalman_filter.state.log_spot == pytest.approx(np.log(100.0))
    assert kalman_filter.state.log_basis == pytest.approx(np.log(101.5) - np.log(100.0))
    assert kalman_filter.state.spot_error == 0.0
    assert kalman_filter.state.perp_error == 0.0
    assert kalman_filter.cov.timestamp == 123
    assert kalman_filter.cov.matrix.tolist() == [
        [2.5e-06, 0.0, 0.0, 0.0],
        [0.0, 1e-05, 0.0, 0.0],
        [0.0, 0.0, 2.5e-06, 0.0],
        [0.0, 0.0, 0.0, 2.5e-06],
    ]


def test_linearize_filter_state_uses_dollar_denominated_filter_properties() -> None:
    state = PriceBasisErrorState(
        timestamp=123,
        log_spot=np.log(100.0),
        log_basis=np.log(101.5) - np.log(100.0),
        spot_error=-0.01,
        perp_error=0.02,
    )
    settings = FilterSettings(
        asset="BTC",
        price_choice="midprice",
        init_state=state,
        init_cov=StateCovariance(timestamp=123, matrix=np.eye(4, dtype=np.float64)),
        price_var_per_sec=1.0e-8,
        basis_var_per_sec=2.5e-8,
        error_kappa=2.0,
        basis_kappa=0.05,
        basis_long_run_mean=0.0,
        spot_error_var_per_sec=1.0e-7,
        perp_error_var_per_sec=1.0e-7,
        microprice_r_mult=1.0,
        perp_r_mult=1.25,
        spot_r_mult=1.0,
        min_measurement_var=5.0e-11,
        covariates=None,
    )
    kalman_filter = KalmanFilter(settings)

    linearized = main.linearize_filter_state(kalman_filter)

    assert linearized.timestamp == 123
    assert linearized.price == pytest.approx(kalman_filter.equilibrium_spot_price)
    assert linearized.basis == pytest.approx(kalman_filter.equilibrium_dollar_basis)
    assert linearized.spot_error == pytest.approx(kalman_filter.spot_error_dollars)
    assert linearized.perp_error == pytest.approx(kalman_filter.perp_error_dollars)
    assert linearized.temporary_dislocation == pytest.approx(
        kalman_filter.temporary_dislocation_dollars
    )
    assert linearized.quoted_spot_price == pytest.approx(kalman_filter.quoted_spot_price)
    assert linearized.quoted_perp_price == pytest.approx(kalman_filter.quoted_perp_price)
    assert linearized.quoted_basis == pytest.approx(kalman_filter.quoted_dollar_basis)


def test_stream_market_measurements_surfaces_feed_errors() -> None:
    class Boom(Exception):
        pass

    class StubSocket:
        async def stream_raw_messages(self):
            raise Boom("feed failed")
            yield {}

        async def close(self) -> None:
            return None

    class StubMeasurementManager:
        def build_measurements(self, _raw_message):
            return []

    feed = main.MarketFeed(
        socket=StubSocket(),
        measurement_manager=StubMeasurementManager(),
    )

    async def consume() -> None:
        async for _measurement in main.stream_market_measurements([feed]):
            pass

    with pytest.raises(Boom, match="feed failed"):
        asyncio.run(consume())


def test_process_measurements_does_not_double_apply_second_seed_measurement() -> None:
    class StubCollector:
        def __init__(self) -> None:
            self.records: list[tuple[BBOMeasurement, dict[str, object]]] = []
            self.closed = False

        def record(self, measurement, filtered_states) -> None:
            self.records.append((measurement, filtered_states))

        def close(self) -> None:
            self.closed = True

    class StubKalmanFilter:
        def __init__(self) -> None:
            self.state = PriceBasisErrorState(
                timestamp=2_000,
                log_spot=np.log(100.0),
                log_basis=np.log(110.0) - np.log(100.0),
                spot_error=0.0,
                perp_error=0.0,
            )
            self.update_calls: list[int | None] = []
            self.equilibrium_spot_price = 100.0
            self.equilibrium_dollar_basis = 10.0
            self.spot_error_dollars = 0.0
            self.perp_error_dollars = 0.0
            self.temporary_dislocation_dollars = 0.0
            self.quoted_spot_price = 100.0
            self.quoted_perp_price = 110.0
            self.quoted_dollar_basis = 10.0

        def update(self, measurement):
            self.update_calls.append(measurement.timestamp)
            return self.state

    measurements = [
        BBOMeasurement(
            timestamp=1_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("99"),
            bid_size=Decimal("1"),
            ask_price=Decimal("101"),
            ask_size=Decimal("1"),
        ),
        BBOMeasurement(
            timestamp=2_000,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("109"),
            bid_size=Decimal("1"),
            ask_price=Decimal("111"),
            ask_size=Decimal("1"),
        ),
    ]

    async def fake_stream_market_measurements(_feeds):
        for measurement in measurements:
            yield measurement

    stub_filter = StubKalmanFilter()
    collector = StubCollector()
    original_stream = main.stream_market_measurements
    original_build = main.build_filter_runs
    main.stream_market_measurements = fake_stream_market_measurements
    main.build_filter_runs = lambda **_kwargs: [main.FilterRun(name="microprice", kalman_filter=stub_filter)]
    try:
        asyncio.run(main.process_measurements([], [], collector))
    finally:
        main.stream_market_measurements = original_stream
        main.build_filter_runs = original_build

    assert stub_filter.update_calls == []
    assert len(collector.records) == 1
    assert collector.records[0][0].timestamp == 2_000
    assert collector.closed is True


def test_process_measurements_seeds_filter_with_latest_funding_context() -> None:
    class StubCollector:
        def __init__(self) -> None:
            self.records: list[tuple[BBOMeasurement, dict[str, object]]] = []
            self.active_contexts: list[ActiveAssetContextMeasurement] = []
            self.closed = False

        def record(self, measurement, filtered_states) -> None:
            self.records.append((measurement, filtered_states))

        def record_active_asset_context(self, measurement) -> None:
            self.active_contexts.append(measurement)

        def close(self) -> None:
            self.closed = True

    class StubKalmanFilter:
        def __init__(self) -> None:
            self.state = PriceBasisErrorState(
                timestamp=2_000,
                log_spot=np.log(100.0),
                log_basis=np.log(110.0) - np.log(100.0),
                spot_error=0.0,
                perp_error=0.0,
            )
            self.current_funding_rate: float | None = None
            self.current_funding_timestamp: int | None = None
            self.equilibrium_spot_price = 100.0
            self.equilibrium_dollar_basis = 10.0
            self.spot_error_dollars = 0.0
            self.perp_error_dollars = 0.0
            self.temporary_dislocation_dollars = 0.0
            self.quoted_spot_price = 100.0
            self.quoted_perp_price = 110.0
            self.quoted_dollar_basis = 10.0

        def update(self, measurement):
            return self.state

        def update_covariates(self, timestamp=None, funding_rate=None) -> None:
            self.current_funding_timestamp = timestamp
            self.current_funding_rate = funding_rate

    measurements = [
        ActiveAssetContextMeasurement(
            timestamp=500,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "0.0001"},
            is_snapshot=True,
        ),
        BBOMeasurement(
            timestamp=1_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("99"),
            bid_size=Decimal("1"),
            ask_price=Decimal("101"),
            ask_size=Decimal("1"),
        ),
        BBOMeasurement(
            timestamp=2_000,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("109"),
            bid_size=Decimal("1"),
            ask_price=Decimal("111"),
            ask_size=Decimal("1"),
        ),
    ]

    async def fake_stream_market_measurements(_feeds):
        for measurement in measurements:
            yield measurement

    stub_filter = StubKalmanFilter()
    collector = StubCollector()
    original_stream = main.stream_market_measurements
    original_build = main.build_filter_runs
    main.stream_market_measurements = fake_stream_market_measurements
    main.build_filter_runs = lambda **_kwargs: [main.FilterRun(name="microprice", kalman_filter=stub_filter)]
    try:
        asyncio.run(main.process_measurements([], [], collector))
    finally:
        main.stream_market_measurements = original_stream
        main.build_filter_runs = original_build

    assert stub_filter.current_funding_rate == pytest.approx(0.0001)
    assert stub_filter.current_funding_timestamp == 500
    assert collector.closed is True


def test_process_measurements_updates_live_filter_when_funding_context_changes() -> None:
    class StubCollector:
        def __init__(self) -> None:
            self.records: list[tuple[BBOMeasurement, dict[str, object]]] = []
            self.active_contexts: list[ActiveAssetContextMeasurement] = []
            self.closed = False

        def record(self, measurement, filtered_states) -> None:
            self.records.append((measurement, filtered_states))

        def record_active_asset_context(self, measurement) -> None:
            self.active_contexts.append(measurement)

        def close(self) -> None:
            self.closed = True

    class StubKalmanFilter:
        def __init__(self) -> None:
            self.state = PriceBasisErrorState(
                timestamp=0,
                log_spot=np.log(100.0),
                log_basis=np.log(110.0) - np.log(100.0),
                spot_error=0.0,
                perp_error=0.0,
            )
            self.current_funding_rate: float | None = None
            self.current_funding_timestamp: int | None = None
            self.funding_updates: list[float | None] = []
            self.funding_timestamps: list[int | None] = []
            self.update_calls: list[int | None] = []
            self.equilibrium_spot_price = 100.0
            self.equilibrium_dollar_basis = 10.0
            self.spot_error_dollars = 0.0
            self.perp_error_dollars = 0.0
            self.temporary_dislocation_dollars = 0.0
            self.quoted_spot_price = 100.0
            self.quoted_perp_price = 110.0
            self.quoted_dollar_basis = 10.0

        def update(self, measurement):
            self.state = PriceBasisErrorState(
                timestamp=measurement.timestamp or self.state.timestamp,
                log_spot=self.state.log_spot,
                log_basis=self.state.log_basis,
                spot_error=self.state.spot_error,
                perp_error=self.state.perp_error,
            )
            self.update_calls.append(measurement.timestamp)
            return self.state

        def update_covariates(self, timestamp=None, funding_rate=None) -> None:
            self.current_funding_timestamp = timestamp
            self.current_funding_rate = funding_rate
            self.funding_timestamps.append(timestamp)
            self.funding_updates.append(funding_rate)

    measurements = [
        BBOMeasurement(
            timestamp=1_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("99"),
            bid_size=Decimal("1"),
            ask_price=Decimal("101"),
            ask_size=Decimal("1"),
        ),
        BBOMeasurement(
            timestamp=2_000,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("109"),
            bid_size=Decimal("1"),
            ask_price=Decimal("111"),
            ask_size=Decimal("1"),
        ),
        ActiveAssetContextMeasurement(
            timestamp=2_500,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "0.0002"},
            is_snapshot=False,
        ),
        BBOMeasurement(
            timestamp=3_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("100"),
            bid_size=Decimal("1"),
            ask_price=Decimal("102"),
            ask_size=Decimal("1"),
        ),
    ]

    async def fake_stream_market_measurements(_feeds):
        for measurement in measurements:
            yield measurement

    stub_filter = StubKalmanFilter()
    collector = StubCollector()
    original_stream = main.stream_market_measurements
    original_build = main.build_filter_runs
    main.stream_market_measurements = fake_stream_market_measurements
    main.build_filter_runs = lambda **_kwargs: [main.FilterRun(name="microprice", kalman_filter=stub_filter)]
    try:
        asyncio.run(main.process_measurements([], [], collector))
    finally:
        main.stream_market_measurements = original_stream
        main.build_filter_runs = original_build

    assert stub_filter.funding_updates == [pytest.approx(0.0002)]
    assert stub_filter.funding_timestamps == [2_500]
    assert stub_filter.update_calls == [3_000]
    assert collector.closed is True


def test_process_measurements_ignores_stale_funding_context_updates() -> None:
    class StubCollector:
        def __init__(self) -> None:
            self.records: list[tuple[BBOMeasurement, dict[str, object]]] = []
            self.active_contexts: list[ActiveAssetContextMeasurement] = []
            self.closed = False

        def record(self, measurement, filtered_states) -> None:
            self.records.append((measurement, filtered_states))

        def record_active_asset_context(self, measurement) -> None:
            self.active_contexts.append(measurement)

        def close(self) -> None:
            self.closed = True

    class StubKalmanFilter:
        def __init__(self) -> None:
            self.state = PriceBasisErrorState(
                timestamp=0,
                log_spot=np.log(100.0),
                log_basis=np.log(110.0) - np.log(100.0),
                spot_error=0.0,
                perp_error=0.0,
            )
            self.current_funding_rate: float | None = None
            self.funding_updates: list[float | None] = []
            self.equilibrium_spot_price = 100.0
            self.equilibrium_dollar_basis = 10.0
            self.spot_error_dollars = 0.0
            self.perp_error_dollars = 0.0
            self.temporary_dislocation_dollars = 0.0
            self.quoted_spot_price = 100.0
            self.quoted_perp_price = 110.0
            self.quoted_dollar_basis = 10.0

        def update(self, measurement):
            self.state = PriceBasisErrorState(
                timestamp=measurement.timestamp or self.state.timestamp,
                log_spot=self.state.log_spot,
                log_basis=self.state.log_basis,
                spot_error=self.state.spot_error,
                perp_error=self.state.perp_error,
            )
            return self.state

        def update_covariates(self, timestamp=None, funding_rate=None) -> None:
            self.current_funding_rate = funding_rate
            self.funding_updates.append(funding_rate)

    measurements = [
        BBOMeasurement(
            timestamp=1_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("99"),
            bid_size=Decimal("1"),
            ask_price=Decimal("101"),
            ask_size=Decimal("1"),
        ),
        BBOMeasurement(
            timestamp=2_000,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("109"),
            bid_size=Decimal("1"),
            ask_price=Decimal("111"),
            ask_size=Decimal("1"),
        ),
        ActiveAssetContextMeasurement(
            timestamp=2_500,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "0.0002"},
            is_snapshot=False,
        ),
        ActiveAssetContextMeasurement(
            timestamp=2_400,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "0.0001"},
            is_snapshot=False,
        ),
    ]

    async def fake_stream_market_measurements(_feeds):
        for measurement in measurements:
            yield measurement

    stub_filter = StubKalmanFilter()
    collector = StubCollector()
    original_stream = main.stream_market_measurements
    original_build = main.build_filter_runs
    main.stream_market_measurements = fake_stream_market_measurements
    main.build_filter_runs = lambda **_kwargs: [main.FilterRun(name="microprice", kalman_filter=stub_filter)]
    try:
        asyncio.run(main.process_measurements([], [], collector))
    finally:
        main.stream_market_measurements = original_stream
        main.build_filter_runs = original_build

    assert stub_filter.current_funding_rate == pytest.approx(0.0002)
    assert stub_filter.funding_updates == [pytest.approx(0.0002)]
    assert collector.closed is True


def test_process_measurements_preserves_last_valid_funding_when_context_is_invalid() -> None:
    class StubCollector:
        def __init__(self) -> None:
            self.records: list[tuple[BBOMeasurement, dict[str, object]]] = []
            self.active_contexts: list[ActiveAssetContextMeasurement] = []
            self.closed = False

        def record(self, measurement, filtered_states) -> None:
            self.records.append((measurement, filtered_states))

        def record_active_asset_context(self, measurement) -> None:
            self.active_contexts.append(measurement)

        def close(self) -> None:
            self.closed = True

    class StubKalmanFilter:
        def __init__(self) -> None:
            self.state = PriceBasisErrorState(
                timestamp=2_000,
                log_spot=np.log(100.0),
                log_basis=np.log(110.0) - np.log(100.0),
                spot_error=0.0,
                perp_error=0.0,
            )
            self.current_funding_rate: float | None = None
            self.equilibrium_spot_price = 100.0
            self.equilibrium_dollar_basis = 10.0
            self.spot_error_dollars = 0.0
            self.perp_error_dollars = 0.0
            self.temporary_dislocation_dollars = 0.0
            self.quoted_spot_price = 100.0
            self.quoted_perp_price = 110.0
            self.quoted_dollar_basis = 10.0

        def update(self, measurement):
            return self.state

        def update_covariates(self, timestamp=None, funding_rate=None) -> None:
            self.current_funding_rate = funding_rate

    measurements = [
        ActiveAssetContextMeasurement(
            timestamp=500,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "0.0001"},
            is_snapshot=True,
        ),
        ActiveAssetContextMeasurement(
            timestamp=600,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={"funding": "nan"},
            is_snapshot=True,
        ),
        ActiveAssetContextMeasurement(
            timestamp=700,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="activeAssetCtx",
            raw_message={},
            context={},
            is_snapshot=True,
        ),
        BBOMeasurement(
            timestamp=1_000,
            market="BTC",
            market_type="spot",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("99"),
            bid_size=Decimal("1"),
            ask_price=Decimal("101"),
            ask_size=Decimal("1"),
        ),
        BBOMeasurement(
            timestamp=2_000,
            market="BTC",
            market_type="perp",
            asset="BTC",
            channel="bbo",
            raw_message={},
            bid_price=Decimal("109"),
            bid_size=Decimal("1"),
            ask_price=Decimal("111"),
            ask_size=Decimal("1"),
        ),
    ]

    async def fake_stream_market_measurements(_feeds):
        for measurement in measurements:
            yield measurement

    stub_filter = StubKalmanFilter()
    collector = StubCollector()
    original_stream = main.stream_market_measurements
    original_build = main.build_filter_runs
    main.stream_market_measurements = fake_stream_market_measurements
    main.build_filter_runs = lambda **_kwargs: [main.FilterRun(name="microprice", kalman_filter=stub_filter)]
    try:
        asyncio.run(main.process_measurements([], [], collector))
    finally:
        main.stream_market_measurements = original_stream
        main.build_filter_runs = original_build

    assert stub_filter.current_funding_rate == pytest.approx(0.0001)
    assert collector.closed is True
