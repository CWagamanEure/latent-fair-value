from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Iterable, Literal


MarketType = Literal["spot", "perp", "unknown"]


@dataclass(frozen=True)
class BaseMeasurement:
    timestamp: int | None
    market: str
    market_type: MarketType
    asset: str
    channel: str
    raw_message: dict[str, Any]

    @property
    def display_name(self) -> str:
        return self.__class__.__name__

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            ("market", self.market),
            ("market_type", self.market_type),
            ("asset", self.asset),
            ("channel", self.channel),
            ("timestamp", self.timestamp),
        )

    @property
    def is_spot(self) -> bool:
        return self.market_type == "spot"

    @property
    def is_perp(self) -> bool:
        return self.market_type == "perp"

    @staticmethod
    def _format_value(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, tuple):
            return ", ".join(str(item) for item in value) if value else "-"
        return str(value)

    def __str__(self) -> str:
        lines = [self.display_name]
        lines.extend(f"  {label}: {self._format_value(value)}" for label, value in self._display_fields())
        return "\n".join(lines)


@dataclass(frozen=True)
class TradeMeasurement(BaseMeasurement):
    side: str | None
    price: Decimal | None
    size: Decimal | None
    trade_id: int | None
    hash: str | None

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            *super()._display_fields(),
            ("side", self.side),
            ("price", self.price),
            ("size", self.size),
            ("trade_id", self.trade_id),
            ("hash", self.hash),
        )





@dataclass(frozen=True)
class BookLevel:
    price: Decimal | None
    size: Decimal | None
    order_count: int | None


@dataclass(frozen=True)
class L2BookMeasurement(BaseMeasurement):
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]
    is_snapshot: bool

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        bid_levels = tuple(
            f"{level.price}@{level.size} (orders={level.order_count})" for level in self.bids
        )
        ask_levels = tuple(
            f"{level.price}@{level.size} (orders={level.order_count})" for level in self.asks
        )
        return (
            *super()._display_fields(),
            ("is_snapshot", self.is_snapshot),
            ("bids", bid_levels),
            ("asks", ask_levels),
        )


@dataclass(frozen=True)
class BBOMeasurement(BaseMeasurement):
    bid_price: Decimal | None
    bid_size: Decimal | None
    ask_price: Decimal | None
    ask_size: Decimal | None

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            *super()._display_fields(),
            ("bid_price", self.bid_price),
            ("bid_size", self.bid_size),
            ("ask_price", self.ask_price),
            ("ask_size", self.ask_size),
        )

    def mid(self) -> Decimal | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return (self.bid_price + self.ask_price) / Decimal("2")

    def microprice(self) -> Decimal | None:
        if (
            self.bid_price is None
            or self.ask_price is None
            or self.bid_size is None
            or self.ask_size is None
        ):
            return None
        total_size = self.bid_size + self.ask_size
        if total_size == 0:
            return None
        return (
            self.bid_size * self.ask_price + self.ask_size * self.bid_price
        ) / total_size



@dataclass(frozen=True)
class CandleMeasurement(BaseMeasurement):
    interval: str | None
    open_time: int | None
    close_time: int | None
    open_price: Decimal | None
    high_price: Decimal | None
    low_price: Decimal | None
    close_price: Decimal | None
    volume: Decimal | None
    trade_count: int | None

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            *super()._display_fields(),
            ("interval", self.interval),
            ("open_time", self.open_time),
            ("close_time", self.close_time),
            ("open_price", self.open_price),
            ("high_price", self.high_price),
            ("low_price", self.low_price),
            ("close_price", self.close_price),
            ("volume", self.volume),
            ("trade_count", self.trade_count),
        )


@dataclass(frozen=True)
class ActiveAssetContextMeasurement(BaseMeasurement):
    context: dict[str, Any]
    is_snapshot: bool

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            *super()._display_fields(),
            ("is_snapshot", self.is_snapshot),
            ("context", self.context),
        )


@dataclass(frozen=True)
class RawMeasurement(BaseMeasurement):
    data: Any
    is_snapshot: bool

    def _display_fields(self) -> Iterable[tuple[str, object]]:
        return (
            *super()._display_fields(),
            ("is_snapshot", self.is_snapshot),
            ("data", self.data),
        )


Measurement = (
    TradeMeasurement
    | L2BookMeasurement
    | BBOMeasurement
    | CandleMeasurement
    | ActiveAssetContextMeasurement
    | RawMeasurement
)
