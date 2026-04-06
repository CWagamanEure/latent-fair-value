from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.data_collection import SQLiteDataCollector
from src.exceptions import StaleMeasurementException
from src.filters.kalman_filter import KalmanFilter
from src.filters.latent_state_types import FilterSettings, PriceBasisErrorState, PriceBasisState, StateCovariance
from src.market_feeds import MarketFeed, build_market_feeds
from src.measurement_types import ActiveAssetContextMeasurement, BBOMeasurement, Measurement
from src.sockets.hyperliquid_socket import HyperliquidSocket


@dataclass(frozen=True)
class FilterRun:
    name: str
    kalman_filter: KalmanFilter


def linearize_filter_state(kalman_filter: KalmanFilter) -> PriceBasisState:
    covariance_matrix = getattr(getattr(kalman_filter, "cov", None), "matrix", None)
    return PriceBasisState(
        timestamp=kalman_filter.state.timestamp,
        price=kalman_filter.equilibrium_spot_price,
        basis=kalman_filter.equilibrium_dollar_basis,
        spot_error=kalman_filter.spot_error_dollars,
        perp_error=kalman_filter.perp_error_dollars,
        temporary_dislocation=kalman_filter.temporary_dislocation_dollars,
        quoted_spot_price=kalman_filter.quoted_spot_price,
        quoted_perp_price=kalman_filter.quoted_perp_price,
        quoted_basis=kalman_filter.quoted_dollar_basis,
        raw_state_vector_json=json.dumps(kalman_filter.state.vector.tolist()),
        raw_covariance_matrix_json=(
            json.dumps(covariance_matrix.tolist()) if covariance_matrix is not None else None
        ),
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
    queue: asyncio.Queue[Measurement | Exception | None] = asyncio.Queue()

    async def pump_feed(feed: MarketFeed) -> None:
        try:
            async for raw_message in run_socket(feed.socket):
                for measurement in feed.measurement_manager.build_measurements(raw_message):
                    if isinstance(measurement, (BBOMeasurement, ActiveAssetContextMeasurement)):
                        await queue.put(measurement)
        except Exception as exc:
            await queue.put(exc)
        finally:
            await queue.put(None)

    tasks = [asyncio.create_task(pump_feed(feed)) for feed in feeds]
    completed_feeds = 0

    try:
        while completed_feeds < len(feeds):
            item = await queue.get()
            if isinstance(item, Exception):
                raise item
            if item is None:
                completed_feeds += 1
                continue
            yield item
    finally:
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                raise result


def linearize_filter_runs(filter_runs: list[FilterRun]) -> dict[str, PriceBasisState]:
    return {
        filter_run.name: linearize_filter_state(filter_run.kalman_filter)
        for filter_run in filter_runs
    }


def extract_funding_rate(measurement: ActiveAssetContextMeasurement) -> float | None:
    funding_value = measurement.context.get("funding")
    if funding_value is None:
        return None
    try:
        funding_rate = float(funding_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(funding_rate):
        return None
    return funding_rate


def update_filter_covariates(
    filter_runs: list[FilterRun],
    *,
    timestamp: int | None = None,
    funding_rate: float | None,
) -> None:
    if funding_rate is None:
        return

    for filter_run in filter_runs:
        update_covariates = getattr(filter_run.kalman_filter, "update_covariates", None)
        if callable(update_covariates):
            update_covariates(timestamp=timestamp, funding_rate=funding_rate)


def record_and_print_states(
    measurement: BBOMeasurement,
    states_by_filter: dict[str, PriceBasisState],
    data_collector: SQLiteDataCollector | None,
) -> None:
    if data_collector is not None:
        data_collector.record(measurement, states_by_filter)
    print(
        "Estimated states: "
        + ", ".join(
            (
                f"{filter_name}="
                f"(filtered_price={state.price:.6f}, filtered_basis={state.basis:.6f}, "
                f"quoted_spot_price={state.quoted_spot_price if state.quoted_spot_price is not None else float('nan'):.6f}, "
                f"quoted_perp_price={state.quoted_perp_price if state.quoted_perp_price is not None else float('nan'):.6f}, "
                f"quoted_basis={state.quoted_basis if state.quoted_basis is not None else float('nan'):.6f}, "
                f"spot_error_dollars={state.spot_error if state.spot_error is not None else float('nan'):.6f}, "
                f"perp_error_dollars={state.perp_error if state.perp_error is not None else float('nan'):.6f}, "
                f"temporary_dislocation_dollars={state.temporary_dislocation if state.temporary_dislocation is not None else float('nan'):.6f}, "
                f"ts={state.timestamp})"
            )
            for filter_name, state in states_by_filter.items()
        ),
        flush=True,
    )


async def process_measurements(
    feeds: list[MarketFeed],
    filter_runs: list[FilterRun] | None = None,
    data_collector: SQLiteDataCollector | None = None,
) -> None:
    pending_spot_mid: float | None = None
    pending_spot_timestamp: int | None = None
    pending_perp_mid: float | None = None
    pending_perp_timestamp: int | None = None
    latest_funding_rate: float | None = None
    latest_funding_timestamp: int | None = None

    try:
        async for measurement in stream_market_measurements(feeds):
            if isinstance(measurement, ActiveAssetContextMeasurement):
                if data_collector is not None:
                    data_collector.record_active_asset_context(measurement)

                funding_rate = extract_funding_rate(measurement)
                measurement_timestamp = measurement.timestamp
                is_stale_context = (
                    measurement_timestamp is not None
                    and latest_funding_timestamp is not None
                    and measurement_timestamp < latest_funding_timestamp
                )
                if is_stale_context or funding_rate is None:
                    continue

                latest_funding_rate = funding_rate
                latest_funding_timestamp = measurement_timestamp
                if filter_runs:
                    update_filter_covariates(
                        filter_runs,
                        timestamp=measurement_timestamp,
                        funding_rate=funding_rate,
                    )
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
                update_filter_covariates(
                    filter_runs,
                    timestamp=latest_funding_timestamp,
                    funding_rate=latest_funding_rate,
                )
                record_and_print_states(
                    measurement,
                    linearize_filter_runs(filter_runs),
                    data_collector,
                )
                continue

            try:
                if measurement.timestamp is not None and any(
                    measurement.timestamp < filter_run.kalman_filter.state.timestamp
                    for filter_run in filter_runs
                ):
                    raise StaleMeasurementException(
                        f"Skipping stale measurement at {measurement.timestamp}"
                    )

                states_by_filter: dict[str, PriceBasisState] = {}
                for filter_run in filter_runs:
                    filter_run.kalman_filter.update(measurement)
                    states_by_filter[filter_run.name] = linearize_filter_state(
                        filter_run.kalman_filter
                    )
            except StaleMeasurementException as exc:
                print(f"{exc}: {measurement}", flush=True)
                continue
            except ValueError:
                print(measurement, flush=True)
                continue
            record_and_print_states(measurement, states_by_filter, data_collector)
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

    init_log_spot = float(np.log(first_spot_mid))
    init_log_basis = float(np.log(first_perp_mid) - np.log(first_spot_mid))
    init_state = PriceBasisErrorState(
        timestamp=first_timestamp,
        log_spot=init_log_spot,
        log_basis=init_log_basis,
        spot_error=0.0,
        perp_error=0.0,
    )
    init_cov = StateCovariance(
        timestamp=first_timestamp,
        matrix=np.array(
            [
                [2.5e-6, 0.0, 0.0, 0.0],
                [0.0, 1.0e-5, 0.0, 0.0],
                [0.0, 0.0, 2.5e-6, 0.0],
                [0.0, 0.0, 0.0, 2.5e-6],
            ],
            dtype=np.float64,
        ),
    )
    settings = FilterSettings(
        asset=market,
        price_choice="microprice",
        init_state=init_state,
        init_cov=init_cov,
        price_var_per_sec=1.0e-8,
        basis_var_per_sec=2.5e-8,
        error_kappa=2.0,
        basis_kappa=0.05,
        basis_long_run_mean=0.0,
        spot_error_var_per_sec=1.0e-7,
        perp_error_var_per_sec=1.0e-7,
        microprice_r_mult=1.0,
        spot_r_mult=1.0,
        perp_r_mult=1.25,
        min_measurement_var=5.0e-11,
        covariates=None,
    )

    return [FilterRun(name="microprice", kalman_filter=KalmanFilter(settings))]


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
