from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

from src.measurement_manager import MeasurementManager
from src.sockets.hyperliquid_socket import HyperliquidSocket


@dataclass(frozen=True)
class MarketFeed:
    socket: HyperliquidSocket
    measurement_manager: MeasurementManager


_SPOT_SYMBOL_ALIASES = {
    "BTC": "UBTC",
}


def build_market_feeds(
    market: str,
    *,
    market_scope: str,
    spot_coin: str | None,
    testnet: bool,
    candle_interval: str | None,
) -> list[MarketFeed]:
    market_types = ["perp", "spot"] if market_scope == "both" else [market_scope]
    resolved_spot_coin = (
        spot_coin
        if spot_coin is not None
        else resolve_spot_subscription_coin(market, testnet=testnet)
        if "spot" in market_types
        else None
    )
    return [
        MarketFeed(
            socket=HyperliquidSocket(
                market=market,
                market_type=market_type,
                subscription_coin=resolve_subscription_coin(
                    market=market,
                    market_type=market_type,
                    spot_coin=resolved_spot_coin,
                ),
                testnet=testnet,
                candle_interval=candle_interval,
            ),
            measurement_manager=MeasurementManager(market.upper(), market_type=market_type),
        )
        for market_type in market_types
    ]


def resolve_subscription_coin(
    *,
    market: str,
    market_type: str,
    spot_coin: str | None,
) -> str:
    if market_type == "spot":
        if not spot_coin:
            raise ValueError("Spot feeds require a resolved Hyperliquid spot subscription identifier.")
        return spot_coin
    return market.upper()


def resolve_spot_subscription_coin(market: str, *, testnet: bool) -> str:
    market_key = market.upper()
    canonical_market = _SPOT_SYMBOL_ALIASES.get(market_key, market_key)
    spot_meta = fetch_spot_meta(testnet=testnet)
    subscription_coin = lookup_spot_subscription_coin(canonical_market, spot_meta)
    if subscription_coin is None:
        raise ValueError(
            f"Unable to resolve Hyperliquid spot subscription coin for {market_key}. "
            "Use --spot-coin to override."
        )
    return subscription_coin


def fetch_spot_meta(*, testnet: bool) -> dict[str, Any]:
    base_url = (
        "https://api.hyperliquid-testnet.xyz/info"
        if testnet
        else "https://api.hyperliquid.xyz/info"
    )
    request = Request(
        base_url,
        data=json.dumps({"type": "spotMeta"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Unexpected spotMeta payload from Hyperliquid.")
    return payload


def lookup_spot_subscription_coin(
    market: str,
    spot_meta: dict[str, Any],
) -> str | None:
    tokens = spot_meta.get("tokens")
    universe = spot_meta.get("universe")
    if not isinstance(tokens, list) or not isinstance(universe, list):
        return None

    token_names_by_index: dict[int, str] = {}
    for token in tokens:
        if not isinstance(token, dict):
            continue
        index = token.get("index")
        name = token.get("name")
        if isinstance(index, int) and isinstance(name, str):
            token_names_by_index[index] = name.upper()

    for pair in universe:
        if not isinstance(pair, dict):
            continue
        name = pair.get("name")
        token_indexes = pair.get("tokens")
        if not isinstance(name, str) or not isinstance(token_indexes, list) or len(token_indexes) < 2:
            continue
        base_token_name = token_names_by_index.get(token_indexes[0])
        quote_token_name = token_names_by_index.get(token_indexes[1])
        if base_token_name != market or quote_token_name != "USDC":
            continue
        pair_index = pair.get("index")
        if isinstance(pair_index, int) and name.startswith("@"):
            return f"@{pair_index}"
        return name

    return None
