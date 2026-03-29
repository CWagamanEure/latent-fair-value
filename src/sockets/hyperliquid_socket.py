from __future__ import annotations

from typing import Any, AsyncIterator

try:
    from src.measurement_manager import MeasurementManager
    from src.measurement_types import MarketType, Measurement
    from src.sockets.base_socket import BaseSocket
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution from src/.
    from measurement_manager import MeasurementManager
    from measurement_types import MarketType, Measurement
    from sockets.base_socket import BaseSocket


class HyperliquidSocket(BaseSocket):
    """Stream market-scoped updates from Hyperliquid."""

    MAINNET_WS_URL = "wss://api.hyperliquid.xyz/ws"
    TESTNET_WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"

    def __init__(
        self,
        market: str,
        *,
        market_type: MarketType = "unknown",
        subscription_coin: str | None = None,
        testnet: bool = False,
        candle_interval: str | None = "1m",
        include_trades: bool = True,
        include_l2_book: bool = True,
        include_bbo: bool = True,
        include_active_asset_ctx: bool = True,
    ) -> None:
        super().__init__(self.TESTNET_WS_URL if testnet else self.MAINNET_WS_URL)
        self.market = market.upper()
        self.market_type = market_type
        self.subscription_coin = subscription_coin or self.market
        self.candle_interval = candle_interval
        self.include_trades = include_trades
        self.include_l2_book = include_l2_book
        self.include_bbo = include_bbo
        self.include_active_asset_ctx = include_active_asset_ctx
        self.measurement_manager = MeasurementManager(self.market, market_type=self.market_type)

    async def after_connect(self) -> None:
        for subscription in self._subscriptions():
            await self.send_json(
                {
                    "method": "subscribe",
                    "subscription": subscription,
                }
            )

    def _subscriptions(self) -> list[dict[str, Any]]:
        subscriptions: list[dict[str, Any]] = []

        if self.include_trades:
            subscriptions.append({"type": "trades", "coin": self.subscription_coin})
        if self.include_l2_book:
            subscriptions.append({"type": "l2Book", "coin": self.subscription_coin})
        if self.include_bbo:
            subscriptions.append({"type": "bbo", "coin": self.subscription_coin})
        if self.include_active_asset_ctx:
            subscriptions.append({"type": "activeAssetCtx", "coin": self.subscription_coin})
        if self.candle_interval:
            subscriptions.append(
                {
                    "type": "candle",
                    "coin": self.subscription_coin,
                    "interval": self.candle_interval,
                }
            )

        return subscriptions

    async def stream_market_updates(
        self,
        *,
        include_subscription_responses: bool = False,
    ) -> AsyncIterator[Measurement]:
        async for raw_message in self.stream_raw_messages():
            channel = raw_message.get("channel")

            if channel == "subscriptionResponse" and not include_subscription_responses:
                continue

            for measurement in self._convert_message_to_measurements(raw_message):
                yield measurement

    def _convert_message_to_measurements(
        self,
        raw_message: dict[str, Any],
    ) -> list[Measurement]:
        return self.measurement_manager.build_measurements(raw_message)

    @staticmethod
    def _extract_timestamp(message: dict[str, Any]) -> int | None:
        return MeasurementManager.extract_timestamp(message)
