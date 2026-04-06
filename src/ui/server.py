from __future__ import annotations

import argparse
import json
import math
import sqlite3
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


class PriceSeriesRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def fetch_series(
        self,
        *,
        window_ms: int,
        max_points: int,
        since_ts: int | None = None,
    ) -> dict[str, Any]:
        with self._connect() as connection:
            active_filter = self._fetch_active_filter(connection)
            if active_filter is None:
                return {
                    "meta": {
                        "downsampled": False,
                        "pointCount": 0,
                        "bucketMs": None,
                        "windowMs": window_ms,
                        "sinceTs": since_ts,
                        "filterName": None,
                        "priceChoice": None,
                    },
                    "points": [],
                }
            if since_ts is not None:
                rows = connection.execute(
                    """
                    SELECT
                        COALESCE(fs.filter_timestamp, ms.measurement_timestamp) AS ts,
                        ms.measurement_timestamp,
                        ms.observed_market_type,
                        ms.observed_mid_price,
                        ms.observed_microprice,
                        ms.spot_mid_price,
                        ms.perp_mid_price,
                        ms.spot_microprice,
                        ms.perp_microprice,
                        fs.filter_name,
                        fs.price_choice,
                        fs.filtered_price,
                        fs.basis,
                        fs.temporary_dislocation,
                        fs.quoted_spot_price,
                        fs.quoted_perp_price,
                        fs.quoted_basis
                    FROM filter_snapshots fs
                    JOIN market_snapshots ms ON ms.id = fs.market_snapshot_id
                    WHERE fs.filter_name = ?
                      AND COALESCE(fs.filter_timestamp, ms.measurement_timestamp) > ?
                    ORDER BY ts ASC
                    LIMIT ?
                    """,
                    (active_filter["filter_name"], since_ts, max_points),
                ).fetchall()
                points = [self._serialize_row(row) for row in rows]
                return {
                    "meta": {
                        "downsampled": False,
                        "pointCount": len(points),
                        "bucketMs": None,
                        "windowMs": window_ms,
                        "sinceTs": since_ts,
                        "filterName": active_filter["filter_name"],
                        "priceChoice": active_filter["price_choice"],
                    },
                    "points": points,
                }

            latest_timestamp = connection.execute(
                """
                SELECT MAX(COALESCE(fs.filter_timestamp, ms.measurement_timestamp))
                FROM filter_snapshots fs
                JOIN market_snapshots ms ON ms.id = fs.market_snapshot_id
                WHERE fs.filter_name = ?
                """
                ,
                (active_filter["filter_name"],),
            ).fetchone()[0]
            if latest_timestamp is None:
                return {
                    "meta": {
                        "downsampled": False,
                        "pointCount": 0,
                        "bucketMs": None,
                        "windowMs": window_ms,
                        "sinceTs": None,
                        "filterName": active_filter["filter_name"],
                        "priceChoice": active_filter["price_choice"],
                    },
                    "points": [],
                }

            start_timestamp = max(int(latest_timestamp) - window_ms, 0)
            rows = connection.execute(
                """
                SELECT
                    COALESCE(fs.filter_timestamp, ms.measurement_timestamp) AS ts,
                    ms.measurement_timestamp,
                    ms.observed_market_type,
                    ms.observed_mid_price,
                    ms.observed_microprice,
                    ms.spot_mid_price,
                    ms.perp_mid_price,
                    ms.spot_microprice,
                    ms.perp_microprice,
                    fs.filter_name,
                    fs.price_choice,
                    fs.filtered_price,
                    fs.basis,
                    fs.temporary_dislocation,
                    fs.quoted_spot_price,
                    fs.quoted_perp_price,
                    fs.quoted_basis
                FROM filter_snapshots fs
                JOIN market_snapshots ms ON ms.id = fs.market_snapshot_id
                WHERE fs.filter_name = ?
                  AND COALESCE(fs.filter_timestamp, ms.measurement_timestamp) >= ?
                ORDER BY ts ASC
                """,
                (active_filter["filter_name"], start_timestamp),
            ).fetchall()

        raw_points = [self._serialize_row(row) for row in rows]
        if len(raw_points) <= max_points:
            return {
                "meta": {
                    "downsampled": False,
                    "pointCount": len(raw_points),
                    "bucketMs": None,
                    "windowMs": window_ms,
                    "sinceTs": None,
                    "filterName": active_filter["filter_name"],
                    "priceChoice": active_filter["price_choice"],
                },
                "points": raw_points,
            }

        bucket_ms = max(int(math.ceil(window_ms / max_points)), 1)
        bucketed: list[dict[str, Any]] = []
        current_bucket = -1
        for point in raw_points:
            bucket_index = max((point["timestamp"] - start_timestamp) // bucket_ms, 0)
            if bucket_index != current_bucket:
                bucketed.append(point)
                current_bucket = bucket_index
                continue
            bucketed[-1] = point

        if len(bucketed) > max_points:
            bucketed = bucketed[-max_points:]

        return {
            "meta": {
                "downsampled": True,
                "pointCount": len(bucketed),
                "bucketMs": bucket_ms,
                "windowMs": window_ms,
                "sinceTs": None,
                "filterName": active_filter["filter_name"],
                "priceChoice": active_filter["price_choice"],
            },
            "points": bucketed,
        }

    def fetch_live_snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            active_filter = self._fetch_active_filter(connection)
            if active_filter is None:
                return {"measurement": None, "filter_state": None, "context": None}
            measurement_row = connection.execute(
                """
                SELECT
                    ms.measurement_timestamp,
                    ms.observed_market_type,
                    ms.observed_mid_price,
                    ms.observed_microprice,
                    ms.spot_mid_price,
                    ms.perp_mid_price,
                    ms.spot_microprice,
                    ms.perp_microprice,
                    fs.filter_name,
                    fs.price_choice,
                    fs.filter_timestamp,
                    fs.filtered_price,
                    fs.basis,
                    fs.spot_error,
                    fs.perp_error,
                    fs.temporary_dislocation,
                    fs.quoted_spot_price,
                    fs.quoted_perp_price,
                    fs.quoted_basis
                FROM filter_snapshots fs
                JOIN market_snapshots ms ON ms.id = fs.market_snapshot_id
                WHERE fs.filter_name = ?
                ORDER BY COALESCE(fs.filter_timestamp, ms.measurement_timestamp) DESC
                LIMIT 1
                """,
                (active_filter["filter_name"],),
            ).fetchone()
            context_row = connection.execute(
                """
                SELECT
                    measurement_timestamp,
                    funding_rate,
                    open_interest,
                    oracle_price,
                    mark_price,
                    mid_price,
                    premium
                FROM asset_context_snapshots
                ORDER BY COALESCE(measurement_timestamp, recorded_at_ms) DESC
                LIMIT 1
                """
            ).fetchone()

        measurement = {
            "timestamp": measurement_row["measurement_timestamp"],
            "marketType": measurement_row["observed_market_type"],
            "midPrice": _as_float(measurement_row["observed_mid_price"]),
            "microprice": _as_float(measurement_row["observed_microprice"]),
            "spotMidPrice": _as_float(measurement_row["spot_mid_price"]),
            "perpMidPrice": _as_float(measurement_row["perp_mid_price"]),
            "spotMicroprice": _as_float(measurement_row["spot_microprice"]),
            "perpMicroprice": _as_float(measurement_row["perp_microprice"]),
        }
        filter_state = {
            "filterName": measurement_row["filter_name"],
            "priceChoice": measurement_row["price_choice"],
            "timestamp": measurement_row["filter_timestamp"],
            "price": _as_float(measurement_row["filtered_price"]),
            "basis": _as_float(measurement_row["basis"]),
            "spotError": _as_float(measurement_row["spot_error"]),
            "perpError": _as_float(measurement_row["perp_error"]),
            "temporaryDislocation": _as_float(measurement_row["temporary_dislocation"]),
            "quotedSpotPrice": _as_float(measurement_row["quoted_spot_price"]),
            "quotedPerpPrice": _as_float(measurement_row["quoted_perp_price"]),
            "quotedBasis": _as_float(measurement_row["quoted_basis"]),
        }
        context = None
        if context_row is not None:
            context = {
                "timestamp": context_row["measurement_timestamp"],
                "fundingRate": _as_float(context_row["funding_rate"]),
                "openInterest": _as_float(context_row["open_interest"]),
                "oraclePrice": _as_float(context_row["oracle_price"]),
                "markPrice": _as_float(context_row["mark_price"]),
                "midPrice": _as_float(context_row["mid_price"]),
                "premium": _as_float(context_row["premium"]),
            }

        return {
            "measurement": measurement,
            "filter_state": filter_state,
            "context": context,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _fetch_active_filter(connection: sqlite3.Connection) -> sqlite3.Row | None:
        return connection.execute(
            """
            SELECT
                filter_name,
                price_choice
            FROM filter_snapshots
            ORDER BY recorded_at_ms DESC, id DESC
            LIMIT 1
            """
        ).fetchone()

    @staticmethod
    def _serialize_row(row: sqlite3.Row) -> dict[str, Any]:
        selected_spot_price = (
            row["spot_microprice"] if row["price_choice"] == "microprice" else row["spot_mid_price"]
        )
        selected_perp_price = (
            row["perp_microprice"] if row["price_choice"] == "microprice" else row["perp_mid_price"]
        )
        return {
            "timestamp": row["ts"],
            "measurementTimestamp": row["measurement_timestamp"],
            "marketType": row["observed_market_type"],
            "measurementMidPrice": _as_float(row["observed_mid_price"]),
            "measurementMicroprice": _as_float(row["observed_microprice"]),
            "spotMidPrice": _as_float(row["spot_mid_price"]),
            "perpMidPrice": _as_float(row["perp_mid_price"]),
            "spotMicroprice": _as_float(row["spot_microprice"]),
            "perpMicroprice": _as_float(row["perp_microprice"]),
            "selectedSpotPrice": _as_float(selected_spot_price),
            "selectedPerpPrice": _as_float(selected_perp_price),
            "filterName": row["filter_name"],
            "priceChoice": row["price_choice"],
            "filterPrice": _as_float(row["filtered_price"]),
            "filterBasis": _as_float(row["basis"]),
            "temporaryDislocation": _as_float(row["temporary_dislocation"]),
            "quotedSpotPrice": _as_float(row["quoted_spot_price"]),
            "quotedPerpPrice": _as_float(row["quoted_perp_price"]),
            "quotedBasis": _as_float(row["quoted_basis"]),
        }


class UIServerHandler(BaseHTTPRequestHandler):
    repository: PriceSeriesRepository
    default_window_ms: int
    default_max_points: int
    default_poll_ms: int

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(self._build_index_html())
            return
        if parsed.path == "/api/series":
            params = parse_qs(parsed.query)
            payload = self.repository.fetch_series(
                window_ms=self._int_param(params, "window_ms", self.default_window_ms),
                max_points=self._int_param(params, "max_points", self.default_max_points),
                since_ts=self._optional_int_param(params, "since_ts"),
            )
            self._send_json(payload)
            return
        if parsed.path == "/api/live":
            self._send_json(self.repository.fetch_live_snapshot())
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def log_message(self, format: str, *args: object) -> None:
        return None

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _int_param(params: dict[str, list[str]], key: str, default: int) -> int:
        values = params.get(key)
        if not values:
            return default
        try:
            return max(int(values[0]), 1)
        except ValueError:
            return default

    @staticmethod
    def _optional_int_param(params: dict[str, list[str]], key: str) -> int | None:
        values = params.get(key)
        if not values:
            return None
        try:
            return int(values[0])
        except ValueError:
            return None

    def _build_index_html(self) -> str:
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live Filter State Monitor</title>
  <style>
    :root {{
      --bg: #071018;
      --panel: rgba(10, 22, 33, 0.9);
      --panel-border: rgba(120, 175, 211, 0.22);
      --text: #e7f2ff;
      --muted: #8ea7bf;
      --spot: #ffb703;
      --perp: #fb8500;
      --filter: #8ecae6;
      --basis: #90be6d;
      --grid: rgba(142, 167, 191, 0.18);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "SF Mono", "Menlo", monospace;
      background:
        radial-gradient(circle at top left, rgba(142, 202, 230, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(251, 133, 0, 0.14), transparent 24%),
        linear-gradient(180deg, #08121c 0%, #04070b 100%);
      color: var(--text);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1400px, calc(100vw - 32px));
      margin: 24px auto;
      display: grid;
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 16px;
      backdrop-filter: blur(8px);
      box-shadow: 0 20px 60px rgba(0, 0, 0, 0.25);
    }}
    .topbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }}
    h1, h2 {{ margin: 0; font-size: 16px; letter-spacing: 0.08em; text-transform: uppercase; }}
    .muted {{ color: var(--muted); }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    select {{
      background: rgba(255, 255, 255, 0.04);
      color: var(--text);
      border: 1px solid var(--panel-border);
      border-radius: 10px;
      padding: 8px 10px;
      font: inherit;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }}
    .stat {{
      padding: 12px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .stat .label {{
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .stat .value {{
      margin-top: 8px;
      font-size: 22px;
      line-height: 1.2;
    }}
    .charts {{
      display: grid;
      gap: 16px;
    }}
    canvas {{
      width: 100%;
      height: 320px;
      display: block;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.02);
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .swatch {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      margin-right: 6px;
      transform: translateY(1px);
    }}
    @media (max-width: 760px) {{
      .shell {{ width: min(100vw - 16px, 1400px); margin: 8px auto 16px; }}
      .panel {{ padding: 12px; border-radius: 14px; }}
      canvas {{ height: 240px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel">
      <div class="topbar">
        <div>
          <h1>Live Filter State Monitor</h1>
          <div class="muted" id="status">Connecting to SQLite stream...</div>
        </div>
        <div class="controls">
          <label>Window
            <select id="window">
              <option value="60000">1 minute</option>
              <option value="300000" selected>5 minutes</option>
              <option value="900000">15 minutes</option>
              <option value="3600000">1 hour</option>
            </select>
          </label>
          <label>Refresh
            <select id="poll">
              <option value="250">250 ms</option>
              <option value="{self.default_poll_ms}" selected>{self.default_poll_ms} ms</option>
              <option value="1000">1000 ms</option>
              <option value="2000">2000 ms</option>
            </select>
          </label>
        </div>
      </div>
    </section>
    <section class="panel">
      <div class="stats" id="stats"></div>
    </section>
    <section class="panel charts">
      <div>
        <div class="topbar">
          <h2>Venue Prices vs Filter Price</h2>
          <div class="legend">
            <span><span class="swatch" style="background: var(--spot)"></span>Spot input price</span>
            <span><span class="swatch" style="background: var(--perp)"></span>Perp input price</span>
            <span><span class="swatch" style="background: var(--filter)"></span>Filtered equilibrium spot</span>
          </div>
        </div>
        <canvas id="priceChart"></canvas>
      </div>
        <div>
          <div class="topbar">
          <h2>Filter Basis and Dislocation</h2>
          <div class="legend">
            <span><span class="swatch" style="background: var(--basis)"></span>Filtered equilibrium basis</span>
            <span><span class="swatch" style="background: #f94144"></span>Temporary dislocation</span>
          </div>
        </div>
        <canvas id="basisChart"></canvas>
      </div>
    </section>
  </main>
  <script>
    const state = {{
      points: [],
      windowMs: {self.default_window_ms},
      maxPoints: {self.default_max_points},
      pollMs: {self.default_poll_ms},
      lastTs: null,
      timer: null,
    }};

    const statusEl = document.getElementById("status");
    const statsEl = document.getElementById("stats");
    const windowEl = document.getElementById("window");
    const pollEl = document.getElementById("poll");
    const priceCanvas = document.getElementById("priceChart");
    const basisCanvas = document.getElementById("basisChart");

    function formatNumber(value) {{
      return value == null ? "n/a" : Number(value).toFixed(6);
    }}

    function formatTimestamp(value) {{
      if (value == null) return "n/a";
      return new Date(Number(value)).toLocaleTimeString();
    }}

    function titleCase(value) {{
      if (!value) return "n/a";
      return String(value).charAt(0).toUpperCase() + String(value).slice(1);
    }}

    function measurementValueForChoice(measurement, side, priceChoice) {{
      if (priceChoice === "microprice") {{
        return side === "spot" ? measurement.spotMicroprice : measurement.perpMicroprice;
      }}
      return side === "spot" ? measurement.spotMidPrice : measurement.perpMidPrice;
    }}

    function renderStats(snapshot) {{
      const measurement = snapshot.measurement || {{}};
      const filterState = snapshot.filter_state || {{}};
      const context = snapshot.context || {{}};
      const priceChoice = filterState.priceChoice || "midprice";
      const cards = [
        ["Last update", formatTimestamp(filterState.timestamp || measurement.timestamp)],
        ["Active filter", `${{titleCase(filterState.filterName)}} / ${{titleCase(priceChoice)}}`],
        [`Spot ${{priceChoice}}`, formatNumber(measurementValueForChoice(measurement, "spot", priceChoice))],
        [`Perp ${{priceChoice}}`, formatNumber(measurementValueForChoice(measurement, "perp", priceChoice))],
        ["Observed mid", formatNumber(measurement.midPrice)],
        ["Observed microprice", formatNumber(measurement.microprice)],
        ["Filtered equilibrium spot", formatNumber(filterState.price)],
        ["Filtered equilibrium basis", formatNumber(filterState.basis)],
        ["Quoted spot", formatNumber(filterState.quotedSpotPrice)],
        ["Quoted perp", formatNumber(filterState.quotedPerpPrice)],
        ["Quoted basis", formatNumber(filterState.quotedBasis)],
        ["Spot error ($)", formatNumber(filterState.spotError)],
        ["Perp error ($)", formatNumber(filterState.perpError)],
        ["Temporary dislocation ($)", formatNumber(filterState.temporaryDislocation)],
        ["Funding", formatNumber(context.fundingRate)],
        ["Open interest", context.openInterest == null ? "n/a" : Number(context.openInterest).toLocaleString()],
        ["Mark price", formatNumber(context.markPrice)],
      ];
      statsEl.innerHTML = cards.map(([label, value]) => `
        <div class="stat">
          <div class="label">${{label}}</div>
          <div class="value">${{value}}</div>
        </div>
      `).join("");
    }}

    function trimPoints() {{
      if (!state.points.length) return;
      const latestTs = state.points[state.points.length - 1].timestamp;
      const cutoff = latestTs - state.windowMs;
      state.points = state.points.filter((point) => point.timestamp >= cutoff);
      if (state.points.length > state.maxPoints * 4) {{
        state.points = state.points.slice(-state.maxPoints * 4);
      }}
    }}

    function setupCanvas(canvas) {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(Math.floor(rect.width * ratio), 1);
      canvas.height = Math.max(Math.floor(rect.height * ratio), 1);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return ctx;
    }}

    function drawChart(canvas, series, options) {{
      const ctx = setupCanvas(canvas);
      const width = canvas.getBoundingClientRect().width;
      const height = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, width, height);

      const padding = {{ top: 16, right: 16, bottom: 22, left: 56 }};
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      if (!series.length || plotWidth <= 0 || plotHeight <= 0) {{
        ctx.fillStyle = "#8ea7bf";
        ctx.fillText("Waiting for live data...", 16, 24);
        return;
      }}

      const timestamps = series.flatMap((item) => item.points.map((point) => point.timestamp));
      const values = series.flatMap((item) => item.points.map((point) => point.value)).filter((value) => value != null);
      if (!timestamps.length || !values.length) {{
        ctx.fillStyle = "#8ea7bf";
        ctx.fillText("Waiting for live data...", 16, 24);
        return;
      }}

      let minValue = Math.min(...values);
      let maxValue = Math.max(...values);
      if (minValue === maxValue) {{
        minValue -= 1;
        maxValue += 1;
      }}
      const minTs = Math.min(...timestamps);
      const maxTs = Math.max(...timestamps);
      const safeMaxTs = maxTs === minTs ? minTs + 1 : maxTs;
      const valuePad = (maxValue - minValue) * 0.1;
      minValue -= valuePad;
      maxValue += valuePad;

      ctx.strokeStyle = "rgba(142, 167, 191, 0.18)";
      ctx.lineWidth = 1;
      for (let index = 0; index <= 4; index += 1) {{
        const y = padding.top + (plotHeight * index) / 4;
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
      }}

      ctx.fillStyle = "#8ea7bf";
      ctx.font = '11px "SF Mono", "Menlo", monospace';
      ctx.fillText(maxValue.toFixed(6), 8, padding.top + 8);
      ctx.fillText(minValue.toFixed(6), 8, height - padding.bottom);
      ctx.fillText(new Date(minTs).toLocaleTimeString(), padding.left, height - 6);
      ctx.fillText(new Date(maxTs).toLocaleTimeString(), width - padding.right - 72, height - 6);

      for (const item of series) {{
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        for (const point of item.points) {{
          if (point.value == null) continue;
          const x = padding.left + ((point.timestamp - minTs) / (safeMaxTs - minTs)) * plotWidth;
          const y = padding.top + (1 - (point.value - minValue) / (maxValue - minValue)) * plotHeight;
          if (!started) {{
            ctx.moveTo(x, y);
            started = true;
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
        ctx.stroke();
      }}

      if (options && options.zeroLine && minValue < 0 && maxValue > 0) {{
        const y = padding.top + (1 - (0 - minValue) / (maxValue - minValue)) * plotHeight;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.18)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(width - padding.right, y);
        ctx.stroke();
        ctx.setLineDash([]);
      }}
    }}

    function renderCharts() {{
      const priceChoice = state.points[0]?.priceChoice || "midprice";
      const priceSeries = [
        {{
          color: "#ffb703",
          points: state.points.map((point) => ({{ timestamp: point.timestamp, value: point.selectedSpotPrice }})),
        }},
        {{
          color: "#fb8500",
          points: state.points.map((point) => ({{ timestamp: point.timestamp, value: point.selectedPerpPrice }})),
        }},
        {{
          color: "#8ecae6",
          points: state.points.map((point) => ({{ timestamp: point.timestamp, value: point.filterPrice }})),
        }},
      ];
      const basisSeries = [
        {{
          color: "#90be6d",
          points: state.points.map((point) => ({{ timestamp: point.timestamp, value: point.filterBasis }})),
        }},
        {{
          color: "#f94144",
          points: state.points.map((point) => ({{ timestamp: point.timestamp, value: point.temporaryDislocation }})),
        }},
      ];
      drawChart(priceCanvas, priceSeries);
      drawChart(basisCanvas, basisSeries, {{ zeroLine: true }});
    }}

    async function fetchJson(url) {{
      const response = await fetch(url, {{ cache: "no-store" }});
      if (!response.ok) {{
        throw new Error(`Request failed with ${{response.status}}`);
      }}
      return response.json();
    }}

    async function hydrateFullSeries() {{
      const payload = await fetchJson(`/api/series?window_ms=${{state.windowMs}}&max_points=${{state.maxPoints}}`);
      state.points = payload.points || [];
      state.lastTs = state.points.length ? state.points[state.points.length - 1].timestamp : null;
      renderCharts();
      statusEl.textContent = payload.meta.downsampled
        ? `Live polling SQLite, ${{payload.meta.filterName || "n/a"}} / ${{payload.meta.priceChoice || "n/a"}}, downsampled to ${{payload.meta.pointCount}} points`
        : `Live polling SQLite, ${{payload.meta.filterName || "n/a"}} / ${{payload.meta.priceChoice || "n/a"}}, displaying ${{payload.meta.pointCount}} raw points`;
    }}

    async function pollIncremental() {{
      const query = state.lastTs == null
        ? `/api/series?window_ms=${{state.windowMs}}&max_points=${{state.maxPoints}}`
        : `/api/series?window_ms=${{state.windowMs}}&max_points=${{state.maxPoints}}&since_ts=${{state.lastTs}}`;
      const payload = await fetchJson(query);
      if (state.lastTs == null || payload.meta.downsampled) {{
        state.points = payload.points || [];
      }} else if ((payload.points || []).length) {{
        state.points.push(...payload.points);
      }}
      trimPoints();
      state.lastTs = state.points.length ? state.points[state.points.length - 1].timestamp : state.lastTs;
      renderCharts();
    }}

    async function refresh() {{
      try {{
        await Promise.all([
          pollIncremental(),
          fetchJson("/api/live").then(renderStats),
        ]);
      }} catch (error) {{
        statusEl.textContent = `Polling error: ${{error.message}}`;
      }}
    }}

    function restartPolling() {{
      if (state.timer) window.clearInterval(state.timer);
      state.timer = window.setInterval(refresh, state.pollMs);
    }}

    windowEl.addEventListener("change", async (event) => {{
      state.windowMs = Number(event.target.value);
      await hydrateFullSeries();
    }});

    pollEl.addEventListener("change", (event) => {{
      state.pollMs = Number(event.target.value);
      restartPolling();
    }});

    window.addEventListener("resize", renderCharts);

    (async () => {{
      await hydrateFullSeries();
      await refresh();
      restartPolling();
    }})();
  </script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a live UI for filter states and venue prices."
    )
    parser.add_argument("--db", required=True, help="Path to the SQLite price snapshot database.")
    parser.add_argument(
        "--host", default="127.0.0.1", help="HTTP bind host. Defaults to 127.0.0.1."
    )
    parser.add_argument("--port", type=int, default=8000, help="HTTP port. Defaults to 8000.")
    parser.add_argument(
        "--window-ms",
        type=int,
        default=300_000,
        help="Initial chart window in milliseconds. Defaults to 300000.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=600,
        help="Maximum points returned for a full history fetch. Defaults to 600.",
    )
    parser.add_argument(
        "--poll-ms",
        type=int,
        default=500,
        help="Default browser poll interval in milliseconds. Defaults to 500.",
    )
    return parser


def make_handler(
    repository: PriceSeriesRepository,
    *,
    default_window_ms: int,
    default_max_points: int,
    default_poll_ms: int,
) -> type[UIServerHandler]:
    class BoundHandler(UIServerHandler):
        pass

    BoundHandler.repository = repository
    BoundHandler.default_window_ms = default_window_ms
    BoundHandler.default_max_points = default_max_points
    BoundHandler.default_poll_ms = default_poll_ms
    return BoundHandler


def main() -> None:
    args = build_parser().parse_args()
    repository = PriceSeriesRepository(args.db)
    handler = make_handler(
        repository,
        default_window_ms=max(args.window_ms, 1),
        default_max_points=max(args.max_points, 1),
        default_poll_ms=max(args.poll_ms, 100),
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving live UI for {repository.db_path} at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
