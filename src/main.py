from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data_collection import SQLiteDataCollector
from src.filters.kalman_filter import KalmanFilter
from src.filters.latent_state_types import FilterSettings, PriceBasisState, StateCovariance
from src.market_feeds import MarketFeed, build_market_feeds
from src.measurement_types import ActiveAssetContextMeasurement, BBOMeasurement, Measurement
from src.sockets.hyperliquid_socket import HyperliquidSocket


@dataclass(frozen=True)
class FilterRun:
    name: str
    kalman_filter: KalmanFilter


def linearize_filter_state(state: PriceBasisState) -> PriceBasisState:
    spot_price = float(np.exp(state.price))
    perp_price = float(np.exp(state.price + state.basis))
    return PriceBasisState(
        timestamp=state.timestamp,
        price=spot_price,
        basis=perp_price - spot_price,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Hyperliquid websocket updates for a market."
    )
    parser.add_argument("market", help="Market symbol to subscribe to, e.g. BTC or SOL.")
    parser.add_argument(
        "--market-scope",
        choices=("both", "perp", "spot"),
        default="both",
        help="Markets to stream into the filter. Defaults to both spot and perp.",
    )
    parser.add_argument(
        "--testnet",
        action="store_true",
        help="Use the Hyperliquid testnet websocket endpoint.",
    )
    parser.add_argument(
        "--candle-interval",
        default="1m",
        help="Optional candle interval, e.g. 1m, 5m, 15m. Use 'none' to disable.",
    )
    parser.add_argument(
        "--db-dir",
        default="data",
        help="Directory for asset-scoped SQLite files. Defaults to ./data.",
    )
    parser.add_argument(
        "--spot-coin",
        help="Optional Hyperliquid spot subscription override, for example an @index.",
    )
    return parser


async def run_socket(socket: HyperliquidSocket) -> AsyncIterator[dict[str, Any]]:
    try:
        async for raw_message in socket.stream_raw_messages():
            if raw_message.get("channel") == "subscriptionResponse":
                continue
            yield raw_message
    finally:
        await socket.close()


async def stream_market_measurements(feeds: list[MarketFeed]) -> AsyncIterator[Measurement]:
    queue: asyncio.Queue[Measurement | None] = asyncio.Queue()

    async def pump_feed(feed: MarketFeed) -> None:
        try:
            async for raw_message in run_socket(feed.socket):
                for measurement in feed.measurement_manager.build_measurements(raw_message):
                    if isinstance(measurement, (BBOMeasurement, ActiveAssetContextMeasurement)):
                        await queue.put(measurement)
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(pump_feed(feed)) for feed in feeds]
    completed_feeds = 0

    try:
        while completed_feeds < len(feeds):
            measurement = await queue.get()
            if measurement is None:
                completed_feeds += 1
                continue
            yield measurement
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def process_measurements(
    feeds: list[MarketFeed],
    filter_runs: list[FilterRun] | None = None,
    data_collector: SQLiteDataCollector | None = None,
) -> None:
    pending_spot_mid: float | None = None
    pending_spot_timestamp: int | None = None
    pending_perp_mid: float | None = None
    pending_perp_timestamp: int | None = None

    try:
        async for measurement in stream_market_measurements(feeds):
            if isinstance(measurement, ActiveAssetContextMeasurement):
                if data_collector is not None:
                    data_collector.record_active_asset_context(measurement)
                continue

            if not isinstance(measurement, BBOMeasurement):
                continue

            if filter_runs is None:
                continue

            measurement_mid = measurement.mid()
            if measurement_mid is None:
                continue

            if pending_spot_mid is None and measurement.is_spot:
                pending_spot_mid = float(measurement_mid)
                pending_spot_timestamp = measurement.timestamp
            elif pending_perp_mid is None and measurement.is_perp:
                pending_perp_mid = float(measurement_mid)
                pending_perp_timestamp = measurement.timestamp

            if not filter_runs:
                if pending_spot_mid is None or pending_perp_mid is None:
                    continue
                known_timestamps = [
                    timestamp
                    for timestamp in (pending_spot_timestamp, pending_perp_timestamp)
                    if timestamp is not None
                ]
                first_timestamp = max(known_timestamps) if known_timestamps else 0
                filter_runs = build_filter_runs(
                    market=measurement.asset,
                    first_timestamp=first_timestamp,
                    first_spot_mid=pending_spot_mid,
                    first_perp_mid=pending_perp_mid,
                )

            states_by_filter: dict[str, PriceBasisState] = {}
            try:
                for filter_run in filter_runs:
                    state = filter_run.kalman_filter.update(measurement)
                    states_by_filter[filter_run.name] = linearize_filter_state(state)
            except ValueError:
                print(measurement, flush=True)
                continue
            if data_collector is not None:
                data_collector.record(measurement, states_by_filter)
            print(
                "Estimated states: "
                + ", ".join(
                    (
                        f"{filter_name}="
                        f"(price={state.price:.6f}, basis={state.basis:.6f}, ts={state.timestamp})"
                    )
                    for filter_name, state in states_by_filter.items()
                ),
                flush=True,
            )
    finally:
        if data_collector is not None:
            data_collector.close()


def build_filter_runs(
    *,
    market: str,
    first_timestamp: int,
    first_spot_mid: float,
    first_perp_mid: float,
) -> list[FilterRun]:
    first_spot_mid = float(first_spot_mid)
    first_perp_mid = float(first_perp_mid)
    if first_spot_mid <= 0.0 or first_perp_mid <= 0.0:
        raise ValueError("Initial spot and perp prices must be positive for log-price filtering")

    init_price = float(np.log(first_spot_mid))
    init_basis = float(np.log(first_perp_mid) - np.log(first_spot_mid))
    init_state = PriceBasisState(
        timestamp=first_timestamp,
        price=init_price,
        basis=init_basis,
    )
    init_cov = StateCovariance(
        timestamp=first_timestamp,
        matrix=np.array(
            [
                [2.5e-6, 0.0],
                [0.0, 1.0e-5],
            ],
            dtype=np.float64,
        ),
    )
    settings_by_name = (
        (
            "midprice",
            FilterSettings(
                asset=market,
                price_choice="midprice",
                init_state=init_state,
                init_cov=init_cov,
                price_var_per_sec=1.0e-8,
                basis_var_per_sec=2.5e-8,
                basis_kappa=0.0,
                basis_long_run_mean=0.0,
                microprice_r_mult=1.0,
                spot_r_mult=1.0,
                perp_r_mult=1.25,
                min_measurement_var=5.0e-11,
                covariates=None,
            ),
        ),
        (
            "microprice_1p5x",
            FilterSettings(
                asset=market,
                price_choice="microprice",
                init_state=init_state,
                init_cov=init_cov,
                price_var_per_sec=1.0e-8,
                basis_var_per_sec=2.5e-8,
                basis_kappa=0.0,
                basis_long_run_mean=0.0,
                microprice_r_mult=1.5,
                spot_r_mult=1.0,
                perp_r_mult=1.25,
                min_measurement_var=5.0e-11,
                covariates=None,
            ),
        ),
        (
            "microprice_3x",
            FilterSettings(
                asset=market,
                price_choice="microprice",
                init_state=init_state,
                init_cov=init_cov,
                price_var_per_sec=1.0e-8,
                basis_var_per_sec=2.5e-8,
                basis_kappa=0.0,
                basis_long_run_mean=0.0,
                microprice_r_mult=3.0,
                spot_r_mult=1.0,
                perp_r_mult=1.25,
                min_measurement_var=5.0e-11,
                covariates=None,
            ),
        ),
    )
    return [
        FilterRun(name=name, kalman_filter=KalmanFilter(filter_settings))
        for name, filter_settings in settings_by_name
    ]


def main() -> None:
    args = build_parser().parse_args()
    candle_interval = None if args.candle_interval.lower() == "none" else args.candle_interval
    market = args.market.upper()
    feeds = build_market_feeds(
        args.market,
        market_scope=args.market_scope,
        spot_coin=args.spot_coin,
        testnet=args.testnet,
        candle_interval=candle_interval,
    )
    data_collector = SQLiteDataCollector(market, db_dir=Path(args.db_dir))
    print(f"Writing market data to {data_collector.db_path}", flush=True)
    asyncio.run(
        process_measurements(
            feeds,
            [],
            data_collector,
        )
    )


if __name__ == "__main__":
    main()
