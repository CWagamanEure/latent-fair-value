## Live Filter

Run the live collector directly:

```bash
python -m src.main btc --db-dir data
```

Or use the make target:

```bash
make live MARKET=btc
```

This writes to:

```text
data/btc_prices.sqlite3
```

The SQLite database stores:

- `market_snapshots`: normalized BBO measurement data, running spot/perp mids and microprices, and the raw BBO websocket payload.
- `filter_snapshots`: linearized filter outputs plus the raw latent state vector and raw covariance matrix for each recorded filter update.
- `asset_context_snapshots`: structured active-asset context fields plus the raw websocket payload and raw context JSON.

Useful flags:

```bash
python -m src.main btc --market-scope both --db-dir data
python -m src.main btc --spot-coin @107 --db-dir data
python -m src.main btc --testnet --db-dir data
```
