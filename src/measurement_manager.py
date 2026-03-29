from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from src.measurement_types import (
    ActiveAssetContextMeasurement,
    BBOMeasurement,
    BookLevel,
    CandleMeasurement,
    MarketType,
    L2BookMeasurement,
    Measurement,
    RawMeasurement,
    TradeMeasurement,
)


class MeasurementManager:
    def __init__(self, market: str, market_type: MarketType = "unknown") -> None:
        self.market = market
        self.market_type = market_type

    def build_measurements(self, raw_message: dict[str, Any]) -> list[Measurement]:
        channel = raw_message.get("channel") or "unknown"
        data = raw_message.get("data")

        if channel == "trades" and isinstance(data, list):
            return [
                self._build_trade_measurement(raw_message, item)
                for item in data
                if isinstance(item, dict)
            ]

        if channel == "l2Book" and isinstance(data, dict):
            return [self._build_l2_book_measurement(raw_message, data)]

        if channel == "bbo" and isinstance(data, dict):
            return [self._build_bbo_measurement(raw_message, data)]

        if channel == "candle" and isinstance(data, dict):
            return [self._build_candle_measurement(raw_message, data)]

        if channel == "activeAssetCtx" and isinstance(data, dict):
            return [self._build_active_asset_context_measurement(raw_message, data)]

        return [
            RawMeasurement(
                timestamp=self.extract_timestamp(raw_message),
                market=self.market,
                market_type=self.market_type,
                asset=self._extract_asset(data),
                channel=channel,
                raw_message=raw_message,
                data=data,
                is_snapshot=self._extract_is_snapshot(data),
            )
        ]

    @staticmethod
    def extract_timestamp(message: dict[str, Any]) -> int | None:
        data = message.get("data")
        if isinstance(data, dict):
            for key in ("time", "timestamp", "statusTimestamp", "t"):
                value = MeasurementManager._coerce_int(data.get(key))
                if value is not None:
                    return value
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                for key in ("time", "timestamp", "statusTimestamp", "t"):
                    value = MeasurementManager._coerce_int(item.get(key))
                    if value is not None:
                        return value
        return None

    def _build_trade_measurement(
        self,
        raw_message: dict[str, Any],
        trade: dict[str, Any],
    ) -> TradeMeasurement:
        return TradeMeasurement(
            timestamp=self._coerce_int(trade.get("time") or trade.get("timestamp") or trade.get("t")),
            market=self.market,
            market_type=self.market_type,
            asset=self._extract_asset(trade),
            channel="trades",
            raw_message=raw_message,
            side=self._coerce_str(trade.get("side")),
            price=self._coerce_decimal(trade.get("px")),
            size=self._coerce_decimal(trade.get("sz")),
            trade_id=self._coerce_int(trade.get("tid")),
            hash=self._coerce_str(trade.get("hash")),
        )

    def _build_l2_book_measurement(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
    ) -> L2BookMeasurement:
        levels = data.get("levels")
        bids_raw = (
            levels[0]
            if isinstance(levels, list) and len(levels) > 0 and isinstance(levels[0], list)
            else []
        )
        asks_raw = (
            levels[1]
            if isinstance(levels, list) and len(levels) > 1 and isinstance(levels[1], list)
            else []
        )

        return L2BookMeasurement(
            timestamp=self.extract_timestamp(raw_message),
            market=self.market,
            market_type=self.market_type,
            asset=self._extract_asset(data),
            channel="l2Book",
            raw_message=raw_message,
            bids=tuple(self._build_book_level(level) for level in bids_raw if isinstance(level, dict)),
            asks=tuple(self._build_book_level(level) for level in asks_raw if isinstance(level, dict)),
            is_snapshot=self._extract_is_snapshot(data),
        )

    def _build_book_level(self, level: dict[str, Any]) -> BookLevel:
        return BookLevel(
            price=self._coerce_decimal(level.get("px")),
            size=self._coerce_decimal(level.get("sz")),
            order_count=self._coerce_int(level.get("n")),
        )

    def _build_bbo_measurement(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
    ) -> BBOMeasurement:
        bid_level: dict[str, Any] = {}
        ask_level: dict[str, Any] = {}

        bbo = data.get("bbo")
        if isinstance(bbo, list):
            if len(bbo) > 0 and isinstance(bbo[0], dict):
                bid_level = bbo[0]
            if len(bbo) > 1 and isinstance(bbo[1], dict):
                ask_level = bbo[1]
        else:
            payload = bbo if isinstance(bbo, dict) else data
            bid_level = payload
            ask_level = payload

        return BBOMeasurement(
            timestamp=self.extract_timestamp(raw_message),
            market=self.market,
            market_type=self.market_type,
            asset=self._extract_asset(data),
            channel="bbo",
            raw_message=raw_message,
            bid_price=self._coerce_decimal(
                bid_level.get("bid") or bid_level.get("bidPx") or bid_level.get("px")
            ),
            bid_size=self._coerce_decimal(
                bid_level.get("bidSz") or bid_level.get("size") or bid_level.get("sz")
            ),
            ask_price=self._coerce_decimal(
                ask_level.get("ask") or ask_level.get("askPx") or ask_level.get("px")
            ),
            ask_size=self._coerce_decimal(
                ask_level.get("askSz") or ask_level.get("size") or ask_level.get("sz")
            ),
        )

    def _build_candle_measurement(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
    ) -> CandleMeasurement:
        return CandleMeasurement(
            timestamp=self._coerce_int(data.get("t") or data.get("time") or data.get("timestamp")),
            market=self.market,
            market_type=self.market_type,
            asset=self._extract_asset(data),
            channel="candle",
            raw_message=raw_message,
            interval=self._coerce_str(data.get("i") or data.get("interval")),
            open_time=self._coerce_int(data.get("t")),
            close_time=self._coerce_int(data.get("T")),
            open_price=self._coerce_decimal(data.get("o")),
            high_price=self._coerce_decimal(data.get("h")),
            low_price=self._coerce_decimal(data.get("l")),
            close_price=self._coerce_decimal(data.get("c")),
            volume=self._coerce_decimal(data.get("v")),
            trade_count=self._coerce_int(data.get("n")),
        )

    def _build_active_asset_context_measurement(
        self,
        raw_message: dict[str, Any],
        data: dict[str, Any],
    ) -> ActiveAssetContextMeasurement:
        context = data.get("ctx") if isinstance(data.get("ctx"), dict) else data

        return ActiveAssetContextMeasurement(
            timestamp=self.extract_timestamp(raw_message),
            market=self.market,
            market_type=self.market_type,
            asset=self._extract_asset(data),
            channel="activeAssetCtx",
            raw_message=raw_message,
            context=context,
            is_snapshot=self._extract_is_snapshot(data),
        )

    def _extract_asset(self, data: Any) -> str:
        if isinstance(data, dict):
            asset = self._coerce_str(data.get("coin") or data.get("asset") or data.get("s"))
            if asset:
                if asset.startswith("@"):
                    return self.market
                return asset
        return self.market

    @staticmethod
    def _extract_is_snapshot(data: Any) -> bool:
        return bool(data.get("isSnapshot", False)) if isinstance(data, dict) else False

    @staticmethod
    def _coerce_decimal(value: Any) -> Decimal | None:
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, int | str):
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None
        if isinstance(value, float):
            return Decimal(str(value))
        return None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        return value if isinstance(value, str) else None
