#!/usr/bin/env python3
"""
Bybit → CSV mit Features für die Anomalieerkennung in der Clustering-App (Isolation Forest, LOF, DBSCAN, K-Means).

Nur öffentliche Marktdaten der Bybit-API v5 – kein API-Schlüssel, kein Konto nötig:
  /v5/market/kline             Kerzen (OHLC, Volumen, Umsatz)
  /v5/market/funding/history   Funding-Raten (Perpetuals, category linear/inverse)
  /v5/market/open-interest     Open Interest (linear/inverse)
  /v5/market/tickers           24h-Umsatz → --top N liquideste Symbole

Zwei Formen der CSV:
  series (Standard)  eine Zeile je Symbol und Kerze → „Welche Stunden/Tage waren ungewöhnlich?“
  cross              eine Zeile je Symbol           → „Welche Tokens verhalten sich anders als der Rest?“

Kein Blick in die Zukunft: Jede Kennzahl einer Kerze nutzt nur Daten bis zu ihrem Schluss – z-Werte vergleichen mit
den VORIGEN Kerzen, Funding und Open Interest sind die zum Kerzenschluss bekannten Werte, die laufende Kerze fehlt.
Open Interest vorsichtig: Bybit sagt nicht, ob der Zeitstempel Beginn oder Messzeitpunkt meint – ein Wert zählt daher
erst ein Raster nach seinem Zeitstempel (series: Raster höchstens ¼ Kerze, mindestens 5 min – bei 1h-Kerzen endet die
OI-Änderung also 15 min vor dem Kerzenschluss). Bei 1- und 3-Minuten-Kerzen fehlt Open Interest (nur alle 5 min).
Mit --split lässt sich daher auf einem früheren Zeitraum trainieren und ein späterer in Schritt 10 der App bewerten –
wie im Live-Betrieb.

Beispiele (im Ordner apps/Clustering):
  python tools/bybit_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 120
  python tools/bybit_features.py --top 20 --interval 4h --days 365 --split 2026-09-01
  python tools/bybit_features.py --mode cross --top 100 --interval 1h --days 30

Danach in der App: Schritt 1 CSV laden → Schritt 2 „Beispiel-Konfiguration laden“ (wird an den Spalten erkannt) →
Übernehmen → Schritt 4 vorbereiten → Schritt 6 trainieren (Isolation Forest ist voreingestellt).

Nur Standardbibliothek + numpy/pandas (beides braucht die App ohnehin). Meldungen auf Englisch und nur ASCII, damit
sie auch in einer Windows-Konsole lesbar sind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://api.bybit.com"
USER_AGENT = "ML-HUB-bybit-features/1.0"
MS_MIN = 60_000
MIN_PER_YEAR = 365 * 1440  # Krypto handelt rund um die Uhr

# Kerzenlängen der API in Minuten (M = Monat fehlt absichtlich: zu wenige Kerzen für z-Werte)
INTERVALS = {"1": 1, "3": 3, "5": 5, "15": 15, "30": 30, "60": 60, "120": 120, "240": 240, "360": 360, "720": 720,
             "D": 1440, "W": 10080}
ALIASES = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120", "4h": "240",
           "6h": "360", "12h": "720", "1d": "D", "d": "D", "1w": "W", "w": "W"}
OI_INTERVALS = [(5, "5min"), (15, "15min"), (30, "30min"), (60, "1h"), (240, "4h"), (1440, "1d")]
KLINE_LIMIT, FUNDING_LIMIT, OI_LIMIT = 1000, 200, 200  # Höchstwerte je Seite laut API
MAX_PAGES = 10_000  # Schutz gegen Endlosschleifen beim Blättern
FUNDING_MAX_AGE = 24 * 60 * MS_MIN  # älteres Funding gilt als unbekannt (übliche Intervalle: 1–8 h)
RETRY_CODES = {10000, 10006, 10016}  # Zeitüberschreitung, zu viele Anfragen, Serverfehler → erneut versuchen
Z_CLIP = 50.0  # z-Werte begrenzen: eine einzige Extremkerze soll die Skalierung nicht sprengen
STABLECOINS = {"USDC", "USDE", "FDUSD", "TUSD", "DAI", "PYUSD", "USDD", "BUSD", "USDP", "EURC", "USD1", "RLUSD",
               "USTC", "EUR", "BRZ"}  # bei --top ausgelassen: ohne Kursbewegung keine sinnvollen Kennzahlen

SERIES_FEATURES = ["ret_z", "range_z", "turnover_z", "vola_regime", "er", "cs_spread_pct", "corr_btc", "excess_z",
                   "funding_bp", "funding_z", "oi_z"]
SERIES_INFO = ["ret_pct", "range_pct", "oi_chg_pct"]  # zum Lesen (Prozent), in der Beispiel-Konfiguration kein Feature
SERIES_ORDER = ["ret_pct", "ret_z", "range_pct", "range_z", "turnover_z", "vola_regime", "er", "cs_spread_pct",
                "corr_btc", "excess_z", "funding_bp", "funding_z", "oi_chg_pct", "oi_z"]
CROSS_FEATURES = ["ret_window_pct", "vola_ann_pct", "er", "cs_spread_pct", "max_dd_pct", "corr_btc", "beta_btc",
                  "vola_trend", "turnover_log10", "ret_skew", "ret_kurt", "funding_mean_bp", "oi_chg_pct"]
FUNDING_COLS = {"funding_bp", "funding_z", "funding_mean_bp"}
OI_COLS = {"oi_chg_pct", "oi_z"}
BENCH_COLS = {"corr_btc", "excess_z", "beta_btc"}
OHLCV = ["open", "high", "low", "close", "volume", "turnover"]
MAX_NAN_SHARE = 0.2  # Feature-Spalte mit mehr Lücken wird weggelassen (statt viele Zeilen zu verlieren)

DESCRIPTIONS = {  # für <name>_columns.csv – in der App: Schritt 1 „Spalten-Beschreibungen laden“
    "row": "row name: symbol + candle start (UTC) - not a feature",
    "symbol": "Bybit symbol - not a feature (can be the comparison group in step 4)",
    "time": "candle start in UTC - not a feature",
    "ret_pct": "return of the candle in % (close vs. previous close)",
    "ret_z": "return as z-score vs. the previous baseline candles (log returns)",
    "range_pct": "high-low range of the candle in % of the open",
    "range_z": "range as z-score vs. the previous baseline candles (log1p of range_pct)",
    "turnover_z": "traded value in USD as z-score vs. the previous baseline candles (log1p)",
    "vola_regime": "log2(volatility of the last window candles / volatility of the baseline before) - 0 normal, +1 double",
    "er": "efficiency ratio over the window: net move / sum of all moves - 0 sideways, 1 straight trend",
    "cs_spread_pct": "estimated bid-ask spread in % (Corwin-Schultz from high/low), mean over the window",
    "corr_btc": "correlation of the returns with the benchmark (default BTC) over the window",
    "excess_z": "return minus benchmark return as z-score - coin-specific move",
    "funding_bp": "funding rate known at candle close in basis points (1 bp = 0.01 %)",
    "funding_z": "funding rate as z-score vs. the previous baseline candles",
    "oi_chg_pct": "change of the open interest in % over one candle length, ending one open interest interval before "
                  "the candle close (1h candles: 15 min) - only values surely known at the close",
    "oi_z": "open interest change as z-score vs. the previous baseline candles",
    "ret_window_pct": "return over the whole period in %",
    "vola_ann_pct": "annualised volatility in % (std of the log returns)",
    "max_dd_pct": "largest drop from a previous high in %",
    "beta_btc": "beta vs. the benchmark (default BTC)",
    "vola_trend": "log2(volatility second half / first half of the period) - >0 rising",
    "turnover_log10": "log10 of the mean traded value per day in USD",
    "ret_skew": "skewness of the returns",
    "ret_kurt": "excess kurtosis of the returns - fat tails",
    "funding_mean_bp": "mean funding rate in the period in basis points",
    "open": "open price", "high": "high price", "low": "low price", "close": "close price",
    "volume": "volume (linear/spot: coins, inverse: USD)", "turnover": "turnover (linear/spot: USD, inverse: coins)",
}


class BybitError(RuntimeError):
    """Fehler der Bybit-API (retCode ≠ 0) – betrifft meist nur ein Symbol (z. B. unbekannter Name)."""


class FatalError(BybitError):
    """Kein Weiterarbeiten möglich: Zugriff verweigert (403), Netz weg, nichts im Cache (--offline)."""


# ---------------------------------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------------------------------
class Client:
    """Kleiner REST-Client für die öffentlichen Endpunkte (urllib, keine Zusatzpakete).

    Wartet zwischen Anfragen (Bybit erlaubt 600 Anfragen je 5 s und IP), wiederholt bei Überlastung mit wachsender
    Pause und kann Antworten in einem Ordner ablegen (--cache) – mit festem --start/--end lässt sich ein Lauf dann
    ohne Netz wiederholen (--offline)."""

    def __init__(self, base_url: str = BASE_URL, pause: float = 0.12, retries: int = 5, timeout: float = 20.0,
                 cache_dir=None, offline: bool = False, log=None, opener=None, sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.pause, self.retries, self.timeout = pause, retries, timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.offline = offline
        self.log = log or (lambda msg: None)
        self.opener = opener or urllib.request.urlopen
        self.sleep = sleep
        self.calls = 0  # tatsächliche Anfragen ans Netz
        self._last = 0.0

    def url(self, path: str, params: dict) -> str:
        query = urllib.parse.urlencode(sorted((k, v) for k, v in params.items() if v is not None))
        return f"{self.base_url}{path}?{query}"

    def get(self, path: str, **params) -> dict:
        """GET → Inhalt von 'result'."""
        url = self.url(path, params)
        file = self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".json") if self.cache_dir else None
        if file is not None and file.exists():
            return json.loads(file.read_text(encoding="utf-8"))["result"]
        if self.offline:
            raise FatalError(f"--offline: answer not in the cache folder ({url})")
        body = self._request(url)
        if file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(json.dumps(body), encoding="utf-8")
        return body["result"]

    def _request(self, url: str) -> dict:
        delay = 1.0
        for attempt in range(1, self.retries + 1):
            wait = self.pause - (time.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            self.calls += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                with self.opener(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 403:  # laut Bybit: IP-Limit überschritten oder Region gesperrt – Wiederholen hilft nicht
                    raise FatalError("HTTP 403 from Bybit: access denied - the IP rate limit was exceeded (wait at "
                                     "least 10 minutes before the next try) or access from this network/region is "
                                     "blocked.") from exc
                if exc.code != 429 and exc.code < 500:
                    raise BybitError(f"HTTP {exc.code} for {url}") from exc
                problem = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError) as exc:  # ValueError: kein JSON
                problem = str(exc) or type(exc).__name__
            else:
                code = body.get("retCode") if isinstance(body, dict) else None
                if code == 0:
                    return body
                if code not in RETRY_CODES:
                    msg = body.get("retMsg") if isinstance(body, dict) else body
                    raise BybitError(f"Bybit error {code}: {msg} ({url})")
                problem = f"retCode {code}: {body.get('retMsg')}"
            if attempt == self.retries:
                raise FatalError(f"giving up after {self.retries} attempts: {problem} ({url})")
            self.log(f"  retry in {delay:.0f} s ({problem})")
            self.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise FatalError(f"no answer ({url})")  # nicht erreichbar – retries ≥ 1


# ---------------------------------------------------------------------------------------------------------------------
# Daten holen (blättern)
# ---------------------------------------------------------------------------------------------------------------------
def fetch_klines(client, category: str, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Kerzen mit Beginn in [start_ms, end_ms], aufsteigend. Rückwärts blättern: jede Anfrage nur mit end liefert
    die neuesten höchstens 1000 Kerzen bis end; die nächste Seite endet vor der ältesten Kerze der vorigen – so hängt
    nichts davon ab, wie die API start und end zusammen auslegt oder sortiert."""
    rows, cur_end = {}, end_ms
    for _ in range(MAX_PAGES):
        if cur_end < start_ms:
            break
        page = client.get("/v5/market/kline", category=category, symbol=symbol, interval=interval,
                          end=cur_end, limit=KLINE_LIMIT).get("list") or []
        if not page:
            break
        for item in page:
            rows[int(item[0])] = item[:7]
        oldest = min(int(item[0]) for item in page)
        if len(page) < KLINE_LIMIT or oldest <= start_ms or oldest - 1 >= cur_end:
            break
        cur_end = oldest - 1
    df = pd.DataFrame([rows[k] for k in sorted(rows)], columns=["start", *OHLCV])
    df = df.astype({c: "float64" for c in OHLCV}).astype({"start": "int64"})
    return df[(df["start"] >= start_ms) & (df["start"] <= end_ms)].reset_index(drop=True)


def fetch_funding(client, category: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Funding-Raten (Zeitpunkt der Abrechnung) in [start_ms, end_ms], aufsteigend. Laut API liefert endTime allein
    die 200 Einträge bis endTime (startTime allein wäre ein Fehler) → rückwärts blättern wie bei den Kerzen."""
    rows, cur_end = {}, end_ms
    for _ in range(MAX_PAGES):
        if cur_end < start_ms:
            break
        page = client.get("/v5/market/funding/history", category=category, symbol=symbol, endTime=cur_end,
                          limit=FUNDING_LIMIT).get("list") or []
        if not page:
            break
        for item in page:
            rows[int(item["fundingRateTimestamp"])] = float(item["fundingRate"])
        oldest = min(int(item["fundingRateTimestamp"]) for item in page)
        if len(page) < FUNDING_LIMIT or oldest <= start_ms or oldest - 1 >= cur_end:
            break
        cur_end = oldest - 1
    ts = sorted(t for t in rows if start_ms <= t <= end_ms)
    return pd.DataFrame({"ts": np.array(ts, dtype="int64"), "rate": [rows[t] for t in ts]})


def fetch_open_interest(client, category: str, symbol: str, interval_time: str, start_ms: int,
                        end_ms: int) -> pd.DataFrame:
    """Open Interest (Stand zum Zeitstempel) in [start_ms, end_ms], aufsteigend – Seiten über nextPageCursor."""
    rows, cursor = {}, None
    for _ in range(MAX_PAGES):
        res = client.get("/v5/market/open-interest", category=category, symbol=symbol, intervalTime=interval_time,
                         startTime=start_ms, endTime=end_ms, limit=OI_LIMIT, cursor=cursor)
        page = res.get("list") or []
        for item in page:
            value = item.get("openInterest") or item.get("singleOpenInterest")
            if value not in (None, ""):
                rows[int(item["timestamp"])] = float(value)
        cursor = res.get("nextPageCursor") or None
        if not page or not cursor:
            break
    ts = sorted(t for t in rows if start_ms <= t <= end_ms)
    return pd.DataFrame({"ts": np.array(ts, dtype="int64"), "oi": [rows[t] for t in ts]})


def top_symbols(client, category: str, n: int) -> list[str]:
    """Die n Symbole mit dem größten 24h-Umsatz in USD (ohne Stablecoins und Futures mit Laufzeit)."""
    quote = "USD" if category == "inverse" else "USDT"
    usd_field = "volume24h" if category == "inverse" else "turnover24h"  # inverse: Volumen in USD, Umsatz in Coins
    found = []
    for item in client.get("/v5/market/tickers", category=category).get("list") or []:
        sym = str(item.get("symbol", ""))
        base = sym[: -len(quote)]
        if not sym.endswith(quote) or "-" in sym or not base or base in STABLECOINS:
            continue
        try:
            usd = float(item.get(usd_field) or 0)
        except ValueError:
            continue
        if usd > 0:
            found.append((usd, sym))
    found.sort(reverse=True)
    return [sym for _, sym in found[:n]]


def oi_interval(minutes: int, fine: bool = False) -> tuple[int, str]:
    """Open-Interest-Raster zur Kerzenlänge: das gröbste, das nicht länger als die Kerze ist (mindestens 5 min).
    fine=True (series, eine Zeile je Kerze): höchstens ¼ der Kerze – ein OI-Wert gilt erst ein Raster nach seinem
    Zeitstempel als bekannt (oi_step_ms), so deckt die OI-Änderung trotzdem gut ¾ der Kerze ab."""
    limit = minutes / 4 if fine else minutes
    fit = [x for x in OI_INTERVALS if x[0] <= limit]
    return fit[-1] if fit else OI_INTERVALS[0]


def oi_step_ms(oi: pd.DataFrame | None, minutes: int) -> int:
    """Raster der Open-Interest-Daten in ms (typischer Abstand der Zeitstempel), ohne Daten das zur Kerze passende.

    Bybit dokumentiert nicht, ob der Zeitstempel eines OI-Werts den Beginn seines Rasters meint (Wert erst am Ende
    bekannt) oder den Zeitpunkt der Messung. Vorsichtig: Ein Wert gilt erst ein Raster nach seinem Zeitstempel als
    bekannt – so gibt es in keinem der beiden Fälle einen Blick in die Zukunft."""
    if oi is not None and len(oi) >= 2:
        return int(np.median(np.diff(oi["ts"].to_numpy(dtype="int64"))))
    return oi_interval(minutes, fine=True)[0] * MS_MIN


# ---------------------------------------------------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Settings:
    category: str = "linear"
    interval: str = "60"
    window: int = 24  # Kerzen für ER, Spread, Korrelation, kurze Volatilität
    baseline: int = 720  # Kerzen für die z-Werte (Vergleich mit den VORIGEN Kerzen)
    benchmark: str | None = "BTCUSDT"
    funding: bool = True
    oi: bool = True

    @property
    def minutes(self) -> int:
        return INTERVALS[self.interval]


def default_window(minutes: int) -> int:
    return max(10, round(1440 / minutes))  # 1 Tag, mindestens 10 Kerzen


def default_baseline(minutes: int) -> int:
    return max(20, round(30 * 1440 / minutes))  # 30 Tage, mindestens 20 Kerzen


def zscore(x, n: int, floor: float) -> pd.Series:
    """z-Wert gegen die VORIGEN n Werte (der aktuelle zählt nicht mit – kein Blick in die Zukunft, und ein Ausreißer
    dämpft sich nicht selbst). floor: kleinste Streuung, damit fast konstante Reihen keine Riesenwerte liefern.
    Bis zu ¼ Lücken im Vergleichsfenster sind erlaubt."""
    x = pd.Series(x, dtype="float64")
    prev = x.shift(1).rolling(n, min_periods=max(10, n * 3 // 4))
    z = (x - prev.mean()) / prev.std().clip(lower=floor)
    return z.clip(-Z_CLIP, Z_CLIP)


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Geschätzter Bid-Ask-Spread (Anteil, nicht %) aus Hoch/Tief zweier Kerzen – Corwin & Schultz (2012).
    Negative Schätzungen → 0."""
    k = 3 - 2 * math.sqrt(2)
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    gamma = np.log(np.maximum(high, high.shift(1)) / np.minimum(low, low.shift(1))) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.clip(lower=0)


def asof(times_ms, table: pd.DataFrame | None, col: str, max_age_ms: int) -> np.ndarray:
    """Letzter bekannter Wert zum Zeitpunkt t: Zeitstempel ≤ t und höchstens max_age_ms alt – sonst NaN."""
    times_ms = np.asarray(times_ms, dtype="int64")
    out = np.full(len(times_ms), np.nan)
    if table is None or table.empty:
        return out
    ts = table["ts"].to_numpy(dtype="int64")
    vals = table[col].to_numpy(dtype="float64")
    idx = np.searchsorted(ts, times_ms, side="right") - 1
    ok = idx >= 0
    ok[ok] = times_ms[ok] - ts[idx[ok]] <= max_age_ms
    out[ok] = vals[idx[ok]]
    return out


def usd_turnover(k: pd.DataFrame, category: str) -> pd.Series:
    """Gehandelter Wert in USD: linear/spot = turnover, inverse = volume (dort in USD)."""
    return k["volume"] if category == "inverse" else k["turnover"]


def closed(k: pd.DataFrame, minutes: int, end_ms: int) -> pd.DataFrame:
    """Nur abgeschlossene Kerzen: Beginn + Dauer ≤ Ende. Die laufende Kerze fehlt – ihr Schlusskurs ist noch offen."""
    return k[k["start"] + minutes * MS_MIN <= end_ms]


def log_returns(k: pd.DataFrame) -> pd.Series:
    return np.log(k.set_index("start")["close"]).diff()


def series_features(k: pd.DataFrame, s: Settings, bench_r: pd.Series | None = None, is_bench: bool = False,
                    funding: pd.DataFrame | None = None, oi: pd.DataFrame | None = None) -> pd.DataFrame:
    """Kennzahlen je Kerze eines Symbols (Index: Kerzenbeginn in ms). Nur Daten bis zum Schluss der Kerze."""
    k = k.set_index("start")
    o, h, l, c = (k[x] for x in ("open", "high", "low", "close"))
    w, b = s.window, s.baseline
    r = np.log(c).diff()
    f = pd.DataFrame(index=k.index)
    f["ret_pct"] = 100 * (c / c.shift(1) - 1)
    f["ret_z"] = zscore(r, b, 1e-4)
    f["range_pct"] = 100 * (h - l) / o
    f["range_z"] = zscore(np.log1p(f["range_pct"]), b, 0.01)
    f["turnover_z"] = zscore(np.log1p(usd_turnover(k, s.category)), b, 0.05)
    short = r.rolling(w, min_periods=w).std()
    long = r.shift(w).rolling(b, min_periods=max(10, b * 3 // 4)).std()  # die b Kerzen VOR dem kurzen Fenster
    f["vola_regime"] = np.log2(short.clip(lower=1e-8) / long.clip(lower=1e-8))
    path = c.diff().abs().rolling(w, min_periods=w).sum()
    f["er"] = ((c - c.shift(w)).abs() / path).mask(path == 0, 0.0)  # keine Bewegung → 0
    f["cs_spread_pct"] = 100 * corwin_schultz(h, l).rolling(w, min_periods=w).mean()
    if is_bench:
        f["corr_btc"], f["excess_z"] = 1.0, 0.0
    elif bench_r is not None:
        rb = bench_r.reindex(k.index)
        corr = r.rolling(w, min_periods=w).corr(rb).replace([np.inf, -np.inf], np.nan)
        both = (r.notna() & rb.notna()).astype("float64").rolling(w, min_periods=w).sum() == w
        f["corr_btc"] = corr.mask(both & corr.isna(), 0.0).clip(-1.0, 1.0)  # ohne Bewegung: keine Korrelation
        f["excess_z"] = zscore(r - rb, b, 1e-4)
    close_ms = k.index.to_numpy(dtype="int64") + s.minutes * MS_MIN
    if funding is not None:
        f["funding_bp"] = asof(close_ms, funding, "rate", FUNDING_MAX_AGE) * 1e4
        f["funding_z"] = zscore(f["funding_bp"], b, 0.5)
    if oi is not None:  # OI-Stand, der zum Kerzenschluss sicher bekannt ist: Zeitstempel + Raster ≤ Schluss
        step = oi_step_ms(oi, s.minutes)
        level = pd.Series(asof(close_ms - step, oi, "oi", 3 * step), index=k.index)
        f["oi_chg_pct"] = (100 * (level / level.shift(1) - 1)).replace([np.inf, -np.inf], np.nan)
        f["oi_z"] = zscore(f["oi_chg_pct"], b, 0.05)
    return f


def time_text(start_ms, minutes: int) -> pd.Series:
    t = pd.to_datetime(pd.Series(start_ms, dtype="int64"), unit="ms", utc=True)
    return t.dt.strftime("%Y-%m-%d" if minutes >= 1440 else "%Y-%m-%d %H:%M")


def drop_gappy(df: pd.DataFrame, cols: list[str], log) -> list[str]:
    """Feature-Spalten mit mehr als 20 % Lücken weglassen (statt so viele Zeilen zu verlieren)."""
    keep = []
    for col in cols:
        share = float(df[col].isna().mean()) if len(df) else 1.0
        if share > MAX_NAN_SHARE:
            hint = " (try --no-oi)" if col in OI_COLS else " (try --no-funding)" if col in FUNDING_COLS else ""
            log(f"note: column {col} left out - {share:.0%} of the rows have no value{hint}")
        else:
            keep.append(col)
    return keep


def build_series(raw: dict, s: Settings, start_ms: int, end_ms: int, keep_ohlcv: bool = False,
                 log=None) -> pd.DataFrame:
    """raw: {Symbol: {"kline", "funding", "oi"}} → eine Zeile je Symbol und abgeschlossener Kerze mit Beginn in
    [start_ms, end_ms]. Ältere Kerzen dienen nur als Vorlauf (z-Werte, Fenster). Interne Spalte _start (ms)."""
    log = log or (lambda msg: None)
    bench = raw.get(s.benchmark) if s.benchmark else None
    bench_r = log_returns(closed(bench["kline"], s.minutes, end_ms)) if bench is not None else None
    if s.benchmark and bench_r is None:
        log(f"note: benchmark {s.benchmark} missing - corr_btc and excess_z are left out")
    parts = []
    for sym, d in raw.items():
        if d.get("only_benchmark"):
            continue
        k = closed(d["kline"], s.minutes, end_ms)
        if len(k) < 2:
            log(f"note: {sym} skipped - no closed candles")
            continue
        f = series_features(k, s, bench_r=bench_r, is_bench=(sym == s.benchmark),
                            funding=d.get("funding") if s.funding else None, oi=d.get("oi") if s.oi else None)
        if keep_ohlcv:
            f = f.join(k.set_index("start")[OHLCV])
        f = f[f.index >= start_ms]
        if f.empty:
            log(f"note: {sym} skipped - no candles in the period")
            continue
        f.insert(0, "symbol", sym)
        parts.append(f.reset_index(names="_start"))
    if not parts:
        return pd.DataFrame(columns=["row", "symbol", "time", "_start"])
    df = pd.concat(parts, ignore_index=True)
    feats = drop_gappy(df, [c for c in SERIES_ORDER if c in df.columns], log)
    before = len(df)
    df = df.dropna(subset=feats)
    lost = before - len(df)
    if lost:
        log(f"note: {lost} of {before} rows without all values dropped (warm-up of the z-scores, gaps)")
    df = df.sort_values(["_start", "symbol"], kind="stable").reset_index(drop=True)
    df["time"] = time_text(df["_start"], s.minutes).to_numpy()
    df["row"] = df["symbol"] + " " + df["time"]
    df[feats] = df[feats].astype("float64").round(5)
    extra = OHLCV if keep_ohlcv else []
    return df[["row", "symbol", "time", *feats, *extra, "_start"]]


def cross_features(k: pd.DataFrame, s: Settings, start_ms: int, end_ms: int, bench_r: pd.Series | None = None,
                   is_bench: bool = False, funding: pd.DataFrame | None = None,
                   oi: pd.DataFrame | None = None) -> dict:
    """Kennzahlen eines Symbols über den ganzen Zeitraum (eine Zeile je Symbol)."""
    c = k["close"].reset_index(drop=True)
    r = log_returns(k).dropna()
    out = {"ret_window_pct": 100 * (c.iloc[-1] / k["open"].iloc[0] - 1),
           "vola_ann_pct": 100 * r.std() * math.sqrt(MIN_PER_YEAR / s.minutes)}
    path = float(c.diff().abs().sum())
    out["er"] = abs(c.iloc[-1] - c.iloc[0]) / path if path > 0 else 0.0
    out["cs_spread_pct"] = 100 * float(corwin_schultz(k["high"], k["low"]).mean())
    out["max_dd_pct"] = 100 * float((1 - c / c.cummax()).max())
    if is_bench:
        out["corr_btc"], out["beta_btc"] = 1.0, 1.0
    elif bench_r is not None:
        both = pd.concat([r, bench_r], axis=1, join="inner").dropna()
        if len(both) >= 10:
            a, bm = both.iloc[:, 0], both.iloc[:, 1]
            corr = a.corr(bm)
            out["corr_btc"] = float(corr) if np.isfinite(corr) else 0.0
            var = float(bm.var())
            out["beta_btc"] = float(a.cov(bm) / var) if var > 0 else 0.0
    half = len(r) // 2
    s1, s2 = float(r.iloc[:half].std()), float(r.iloc[half:].std())
    out["vola_trend"] = math.log2(max(s2, 1e-8) / max(s1, 1e-8)) if np.isfinite(s1) and np.isfinite(s2) else np.nan
    days = len(k) * s.minutes / 1440
    out["turnover_log10"] = math.log10(max(float(usd_turnover(k, s.category).sum()) / days, 1.0))
    out["ret_skew"], out["ret_kurt"] = float(r.skew()), float(r.kurt())
    if funding is not None:
        sel = funding[(funding["ts"] > start_ms) & (funding["ts"] <= end_ms)]
        out["funding_mean_bp"] = 1e4 * float(sel["rate"].mean()) if len(sel) else np.nan
    if oi is not None:  # wie bei series: ein OI-Wert gilt erst ein Raster nach seinem Zeitstempel als bekannt
        step = oi_step_ms(oi, s.minutes)
        first, last = asof([start_ms - step, end_ms - step], oi, "oi", 3 * step)
        out["oi_chg_pct"] = 100 * (last / first - 1) if first > 0 and np.isfinite(last) else np.nan
    return out


def build_cross(raw: dict, s: Settings, start_ms: int, end_ms: int, log=None, min_candles: int = 20) -> pd.DataFrame:
    """raw → eine Zeile je Symbol mit den Kennzahlen über [start_ms, end_ms]."""
    log = log or (lambda msg: None)
    bench = raw.get(s.benchmark) if s.benchmark else None
    bench_r = None
    if bench is not None:
        bk = closed(bench["kline"], s.minutes, end_ms)
        bench_r = log_returns(bk[bk["start"] >= start_ms])
    elif s.benchmark:
        log(f"note: benchmark {s.benchmark} missing - corr_btc and beta_btc are left out")
    rows = []
    for sym, d in raw.items():
        if d.get("only_benchmark"):
            continue
        k = closed(d["kline"], s.minutes, end_ms)
        k = k[k["start"] >= start_ms].reset_index(drop=True)
        if len(k) < min_candles:
            log(f"note: {sym} skipped - only {len(k)} candles (at least {min_candles})")
            continue
        row = {"symbol": sym}
        row.update(cross_features(k, s, start_ms, end_ms, bench_r=bench_r, is_bench=(sym == s.benchmark),
                                  funding=d.get("funding") if s.funding else None,
                                  oi=d.get("oi") if s.oi else None))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["symbol"])
    df = pd.DataFrame(rows)
    feats = drop_gappy(df, [c for c in CROSS_FEATURES if c in df.columns], log)
    before = len(df)
    df = df.dropna(subset=feats)
    if before - len(df):
        log(f"note: {before - len(df)} of {before} symbols without all values dropped")
    df[feats] = df[feats].astype("float64").round(5)
    return df[["symbol", *feats]].sort_values("symbol").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------------------------------
# Ablauf
# ---------------------------------------------------------------------------------------------------------------------
def collect(client, symbols: list[str], s: Settings, fetch_start_ms: int, end_ms: int, log=None,
            fine_oi: bool = True) -> dict:
    """Rohdaten je Symbol holen. Ein Fehler bei einem Symbol (z. B. unbekannter Name) überspringt nur dieses.
    fine_oi: Open Interest im feinen Raster (series, siehe oi_interval) – cross braucht nur Anfang und Ende."""
    log = log or (lambda msg: None)
    oi_min, oi_name = oi_interval(s.minutes, fine=fine_oi)
    todo = list(dict.fromkeys(symbols))
    if s.benchmark and s.benchmark not in todo:
        todo.append(s.benchmark)
    raw = {}
    for i, sym in enumerate(todo, 1):
        try:
            k = fetch_klines(client, s.category, sym, s.interval, fetch_start_ms, end_ms)
            fund = fetch_funding(client, s.category, sym, fetch_start_ms - FUNDING_MAX_AGE, end_ms) \
                if s.funding else None
            oi = fetch_open_interest(client, s.category, sym, oi_name, fetch_start_ms - 4 * oi_min * MS_MIN, end_ms) \
                if s.oi else None
        except FatalError:
            raise
        except BybitError as exc:
            log(f"[{i}/{len(todo)}] {sym}: skipped - {exc}")
            continue
        if k.empty:
            log(f"[{i}/{len(todo)}] {sym}: skipped - no candles (unknown symbol or not listed in the period?)")
            continue
        raw[sym] = {"kline": k, "funding": fund, "oi": oi, "only_benchmark": sym not in symbols}
        parts = [f"{len(k):,} candles"]
        if fund is not None:
            parts.append(f"{len(fund):,} funding")
        if oi is not None:
            parts.append(f"{len(oi):,} open interest")
        log(f"[{i}/{len(todo)}] {sym}: " + ", ".join(parts))
    return raw


def norm_interval(text: str) -> str:
    t = str(text).strip()
    t = ALIASES.get(t.lower(), t.upper() if t.upper() in ("D", "W") else t)
    if t not in INTERVALS:
        raise ValueError(f"unknown interval {text!r} - use one of {', '.join(INTERVALS)} or 15m, 1h, 4h, 1d, 1w")
    return t


def parse_time(text: str) -> int:
    """'2026-01-01', '2026-01-01 12:00' oder ISO-Zeit → ms (UTC, wenn keine Zeitzone angegeben)."""
    t = pd.Timestamp(str(text).strip())
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    return int(t.value // 1_000_000)


def ms_text(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


EXAMPLES = """examples (run in the folder apps/Clustering):
  python tools/bybit_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 120
  python tools/bybit_features.py --top 20 --interval 4h --days 365 --split 2026-09-01
  python tools/bybit_features.py --mode cross --top 100 --interval 1h --days 30

then in the Clustering app: step 1 load the CSV -> step 2 'Load example configuration' -> Apply ->
step 4 'Prepare data for the model' -> step 6 train (Isolation Forest is preset) -> steps 7 and 8."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="bybit_features.py", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EXAMPLES,
        description="Bybit public market data -> CSV with features for anomaly detection in the Clustering app "
                    "(Isolation Forest, LOF, DBSCAN). No API key needed.")
    p.add_argument("--mode", choices=["series", "cross"], default="series",
                   help="series: one row per symbol and candle (default) - cross: one row per symbol")
    p.add_argument("--category", choices=["linear", "inverse", "spot"], default="linear",
                   help="linear = USDT perpetuals (default), inverse = coin-margined, spot (no funding/open interest)")
    p.add_argument("--symbols", help="comma-separated, e.g. BTCUSDT,ETHUSDT (default: --top)")
    p.add_argument("--top", type=int, help="the N most traded symbols of the last 24 h (default: 20 for series, "
                                           "100 for cross)")
    p.add_argument("--interval", default="60", help="candle length: 1 3 5 15 30 60 120 240 360 720 D W or "
                                                    "15m, 1h, 4h, 1d, 1w (default 1h)")
    p.add_argument("--days", type=float, help="length of the period up to --end (default: 90 for series, 30 for cross)")
    p.add_argument("--start", help="start in UTC instead of --days, e.g. 2026-01-01 or '2026-01-01 12:00'")
    p.add_argument("--end", help="end in UTC (default: now - the candle that is still running is left out)")
    p.add_argument("--window", type=int, help="candles for er, spread, correlation and the short volatility "
                                              "(default: 1 day, at least 10)")
    p.add_argument("--baseline", type=int, help="earlier candles the z-scores compare with (default: 30 days, "
                                                "at least 20)")
    p.add_argument("--benchmark", help="symbol for corr_btc, excess_z and beta_btc (default BTCUSDT, inverse: BTCUSD); "
                                       "'none' = without")
    p.add_argument("--no-funding", action="store_true", help="without funding rate columns")
    p.add_argument("--no-oi", action="store_true", help="without open interest columns")
    p.add_argument("--split", help="series only: rows before this time -> <out>_train.csv, from it on -> "
                                   "<out>_new.csv (train on the first, score the second in step 10)")
    p.add_argument("--out", help="CSV file (default: bybit_<mode>_<category>_<interval>.csv)")
    p.add_argument("--keep-ohlcv", action="store_true", help="series only: also write open/high/low/close/volume/"
                                                             "turnover (not used as features)")
    p.add_argument("--cache", help="folder for the raw API answers - repeat a run with fixed --start/--end "
                                   "without new requests")
    p.add_argument("--offline", action="store_true", help="only use the --cache folder, no network")
    p.add_argument("--base-url", default=BASE_URL, help=f"API address (default {BASE_URL}, testnet: "
                                                        "https://api-testnet.bybit.com)")
    p.add_argument("--pause", type=float, default=0.12, help="seconds between requests (Bybit allows 600 per 5 s "
                                                             "and IP; default 0.12)")
    args = p.parse_args(argv)
    try:
        args.interval = norm_interval(args.interval)
        args.start_ms = parse_time(args.start) if args.start else None
        args.end_ms = parse_time(args.end) if args.end else None
        args.split_ms = parse_time(args.split) if args.split else None
    except ValueError as exc:
        p.error(str(exc))
    if args.mode == "cross" and (args.split or args.keep_ohlcv):
        p.error("--split and --keep-ohlcv only work with --mode series")
    if args.offline and not args.cache:
        p.error("--offline needs --cache (the folder with the saved answers)")
    for name in ("top", "window", "baseline"):
        if getattr(args, name) is not None and getattr(args, name) < (1 if name == "top" else 2):
            p.error(f"--{name} is too small")
    if args.days is not None and args.days <= 0:
        p.error("--days must be greater than 0")
    return args


def write_outputs(df: pd.DataFrame, out: Path, split_ms: int | None, log) -> list[Path]:
    """CSV (bzw. _train/_new bei --split) und <name>_columns.csv mit den Spaltenbeschreibungen schreiben."""
    out.parent.mkdir(parents=True, exist_ok=True)
    data = df.drop(columns=["_start"], errors="ignore")
    written = []
    if split_ms is None:
        data.to_csv(out, index=False)
        written.append(out)
    else:
        before = df["_start"] < split_ms
        if before.all() or not before.any():
            raise ValueError(f"--split {ms_text(split_ms)} UTC leaves one part empty - choose a time inside "
                             f"{df['time'].iloc[0]} ... {df['time'].iloc[-1]}")
        for suffix, mask in (("_train", before), ("_new", ~before)):
            path = out.with_name(f"{out.stem}{suffix}{out.suffix}")
            data[mask.to_numpy()].to_csv(path, index=False)
            written.append(path)
    info = out.with_name(f"{out.stem}_columns.csv")
    pd.DataFrame({"column": list(data.columns), "description": [DESCRIPTIONS.get(c, "") for c in data.columns]}) \
        .to_csv(info, index=False, sep=";")
    written.append(info)
    return written


def main(argv=None, client=None) -> int:
    args = parse_args(argv)

    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    minutes = INTERVALS[args.interval]
    end_ms = args.end_ms if args.end_ms is not None else int(time.time() // 60 * 60 * 1000)
    days = args.days if args.days is not None else (90.0 if args.mode == "series" else 30.0)
    start_ms = args.start_ms if args.start_ms is not None else end_ms - int(days * 1440 * MS_MIN)
    if start_ms >= end_ms:
        log("error: the start must be before the end")
        return 2
    funding = oi = args.category != "spot"
    if args.category == "spot" and not (args.no_funding and args.no_oi):
        log("note: spot has no funding rate and no open interest - these columns are left out")
    bench = args.benchmark or ("BTCUSD" if args.category == "inverse" else "BTCUSDT")
    s = Settings(category=args.category, interval=args.interval,
                 window=args.window or default_window(minutes), baseline=args.baseline or default_baseline(minutes),
                 benchmark=None if bench.lower() == "none" else bench.upper(),
                 funding=funding and not args.no_funding, oi=oi and not args.no_oi)
    if s.oi and args.mode == "series" and minutes < OI_INTERVALS[0][0]:  # sonst: OI-Sprung nur jede 5. Minute
        log(f"note: open interest exists only every {OI_INTERVALS[0][0]} minutes - left out for {minutes}-minute "
            "candles (use 5m or longer)")
        s.oi = False
    client = client or Client(base_url=args.base_url, pause=args.pause, cache_dir=args.cache, offline=args.offline,
                              log=log)
    out = Path(args.out or f"bybit_{args.mode}_{args.category}_{args.interval}.csv")
    try:
        if args.symbols:
            symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        else:
            n = args.top or (20 if args.mode == "series" else 100)
            symbols = top_symbols(client, args.category, n)
            log(f"top {len(symbols)} by 24h turnover: {', '.join(symbols)}")
        if not symbols:
            log("error: no symbols")
            return 2
        warmup = s.baseline + s.window + 2 if args.mode == "series" else 0  # Vorlauf für z-Werte und Fenster
        fetch_start = start_ms - warmup * minutes * MS_MIN
        log(f"{args.mode}: {len(symbols)} symbol(s), {args.category}, interval {args.interval}, "
            f"{ms_text(start_ms)} ... {ms_text(end_ms)} UTC" + (f", warm-up from {ms_text(fetch_start)}" if warmup
                                                                else ""))
        raw = collect(client, symbols, s, fetch_start, end_ms, log, fine_oi=args.mode == "series")
        if args.mode == "series":
            df = build_series(raw, s, start_ms, end_ms, keep_ohlcv=args.keep_ohlcv, log=log)
        else:
            df = build_cross(raw, s, start_ms, end_ms, log=log)
        if df.empty:
            log("error: no rows - check the symbols and the period")
            return 1
        written = write_outputs(df, out, args.split_ms, log)
    except FatalError as exc:
        log(f"error: {exc}")
        return 2
    except (BybitError, ValueError, OSError) as exc:
        log(f"error: {exc}")
        return 1
    feats = [c for c in df.columns if c in SERIES_ORDER or c in CROSS_FEATURES]
    n_sym = df["symbol"].nunique()
    log(f"done: {len(df):,} rows x {len(feats)} feature columns, {n_sym} symbol(s), "
        f"{getattr(client, 'calls', 0)} API requests")
    for path in written:
        log(f"  -> {path}")
    log("next: Clustering app -> step 1 load the CSV -> step 2 'Load example configuration' -> Apply -> step 4 "
        "'Prepare data for the model' -> step 6 train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
