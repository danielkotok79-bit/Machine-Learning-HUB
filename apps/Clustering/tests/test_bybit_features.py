"""
tools/bybit_features.py – Bybit → CSV mit Features für die Anomalieerkennung. Ohne Netz: eine simulierte API
(FakeBybit in synthetic.py) liefert Kerzen, Funding und Open Interest wie die echte (neueste zuerst, Seitenlimit,
Cursor) – mit einem eingebauten Kurssturz, den Isolation Forest finden muss.

    pytest -q tests/test_bybit_features.py
"""
import ast
import io
import json
import sys
import urllib.error
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as pl  # noqa: E402
from synthetic import HOUR_MS, FakeBybit  # noqa: E402
from tools import bybit_features as bf  # noqa: E402

CONFIGS = ROOT / "configs"
SERIES_CFG, CROSS_CFG = "bybit_zeitreihe_beispiel.json", "bybit_querschnitt_beispiel.json"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def run(tmp_path, *args, fake=None, name="out.csv"):
    """Skript wie auf der Kommandozeile – mit der simulierten API und festem Ende."""
    fake = fake or FakeBybit()
    out = tmp_path / name
    rc = bf.main([*args, "--end", bf.ms_text(fake.now_ms), "--out", str(out)], client=fake)
    return rc, out, fake


def examples_in_app() -> dict:
    """EXAMPLES aus app.py (Spalten, an denen die Beispiel-Konfiguration erkannt wird) – ohne Streamlit zu starten."""
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "EXAMPLES":
            return {e.elts[1].value: ast.literal_eval(e.elts[0]) for e in node.value.elts}
    raise AssertionError("EXAMPLES fehlt in app.py")


def prepare_like_app(csv_path, config_name):
    """Schritt 1, 2 und 4 wie in der App nach „Beispiel-Konfiguration laden“ → (viz, prep, ClusterConfig, cfg)."""
    cfg = json.loads((CONFIGS / config_name).read_text(encoding="utf-8"))
    df, _ = pl.read_csv(csv_path)
    prof = pl.column_profile(df)
    for a in cfg["step2"]:
        i = prof.index[prof["Spalte"] == a["Spalte"]][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [a["Aktion"], a["Parameter"], a["Neuer Name"]]
    viz, fitted, _, lineage = pl.apply_column_actions(df, pl.actions_from_table(prof))
    w = cfg["widgets"]
    feats = [c for c in w["w_features"] if c in viz.columns]
    prep = pl.prepare_cluster_data(viz, feats, scaler=w["w_scaler"], column_actions=fitted, lineage=lineage,
                                   group_col=None if w["w_group"] == "(keine)" else w["w_group"],
                                   name_col=w["w_name_col"])
    ccfg = pl.ClusterConfig(model=w["w_model"], contamination=0.0 if w["w_contam_auto"] else w["w_contam_pct"] / 100,
                            n_estimators=w["w_n_estimators"], max_samples=w["w_max_samples"], seed=w["w_seed"])
    return viz, prep, ccfg, cfg


# --- Eingaben ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text, api", [("1h", "60"), ("4H", "240"), ("15m", "15"), ("60", "60"), ("d", "D"),
                                       ("1d", "D"), ("W", "W"), ("1w", "W"), ("720", "720")])
def test_interval_names(text, api):
    assert bf.norm_interval(text) == api


@pytest.mark.parametrize("text", ["M", "2d", "7", "", "1y"])
def test_unknown_interval_is_refused(text):
    with pytest.raises(ValueError, match="unknown interval"):
        bf.norm_interval(text)


def test_default_window_and_baseline_follow_the_candle_length():
    assert (bf.default_window(60), bf.default_baseline(60)) == (24, 720)  # 1 Tag / 30 Tage
    assert (bf.default_window(240), bf.default_baseline(240)) == (10, 180)
    assert (bf.default_window(1440), bf.default_baseline(1440)) == (10, 30)  # mindestens 10 Kerzen im Fenster
    assert (bf.default_window(10080), bf.default_baseline(10080)) == (10, 20)  # mindestens 20 zum Vergleich
    assert bf.oi_interval(60) == (60, "1h") and bf.oi_interval(1) == (5, "5min")
    assert bf.oi_interval(720) == (240, "4h") and bf.oi_interval(10080) == (1440, "1d")


@pytest.mark.parametrize("args", [["--mode", "cross", "--split", "2026-01-01"], ["--mode", "cross", "--keep-ohlcv"],
                                  ["--offline"], ["--interval", "M"], ["--days", "0"], ["--top", "0"],
                                  ["--baseline", "1"], ["--split", "kein Datum"]])
def test_wrong_command_line_is_refused(args):
    with pytest.raises(SystemExit) as e:
        bf.parse_args(args)
    assert e.value.code == 2


def test_times_are_utc():
    assert bf.parse_time("2026-01-01") == 1_767_225_600_000
    assert bf.parse_time("2026-01-01 12:00") == 1_767_225_600_000 + 12 * HOUR_MS
    assert bf.parse_time("2026-01-01T14:00+02:00") == 1_767_225_600_000 + 12 * HOUR_MS
    assert bf.ms_text(1_767_225_600_000) == "2026-01-01 00:00"


# --- API: Blättern, Fehler, Cache ---------------------------------------------------------------------------------
def test_paging_follows_the_api_limits():
    fake = FakeBybit(n=2600)
    lo, hi = 0, fake.now_ms
    k = bf.fetch_klines(fake, "linear", "ETHUSDT", "60", lo, hi)
    assert len(k) == 2600 and k["start"].is_monotonic_increasing and k["start"].is_unique
    assert k["start"].tolist() == [int(x[0]) for x in fake.kl["ETHUSDT"]]
    assert np.allclose(k["close"], [float(x[4]) for x in fake.kl["ETHUSDT"]])
    pages = [p for path, p in fake.requests if path == "/v5/market/kline"]
    assert len(pages) == 3 and all(int(p["limit"]) <= 1000 for p in pages)  # 1000 + 1000 + 600, rückwärts
    f = bf.fetch_funding(fake, "linear", "ETHUSDT", lo, hi)
    assert f["ts"].tolist() == [t for t, _ in fake.fund["ETHUSDT"]] and f["ts"].is_monotonic_increasing
    oi = bf.fetch_open_interest(fake, "linear", "ETHUSDT", "1h", lo, hi)
    assert oi["ts"].tolist() == [t for t, _ in fake.oi["ETHUSDT"]]
    oi_pages = [p for path, p in fake.requests if path == "/v5/market/open-interest"]
    assert len(oi_pages) == -(-len(fake.oi["ETHUSDT"]) // 200) and oi_pages[1]["cursor"] == "200"
    part = bf.fetch_klines(fake, "linear", "ETHUSDT", "60", int(k["start"][100]), int(k["start"][150]))
    assert part["start"].tolist() == k["start"][100:151].tolist()  # start und end inklusive


def test_top_symbols_skip_stablecoins_and_dated_futures():
    fake = FakeBybit()
    assert bf.top_symbols(fake, "linear", 10) == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # nach Umsatz, ohne USDC/Futures
    assert bf.top_symbols(fake, "linear", 2) == ["BTCUSDT", "ETHUSDT"]


class Answer:
    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_client_retries_when_overloaded_and_uses_the_cache(tmp_path):
    answers = [{"retCode": 10006, "retMsg": "Too many visits"}, {"retCode": 0, "retMsg": "OK", "result": {"list": [1]}}]
    urls, sleeps = [], []

    def opener(req, timeout):
        urls.append(req.full_url)
        return Answer(answers.pop(0))

    c = bf.Client(opener=opener, sleep=sleeps.append, pause=0, cache_dir=tmp_path / "cache")
    assert c.get("/v5/market/tickers", category="linear", cursor=None) == {"list": [1]}
    assert c.calls == 2 and sleeps == [1.0]  # einmal 1 s warten, dann klappt es
    assert urls[0] == "https://api.bybit.com/v5/market/tickers?category=linear"  # None-Parameter fehlen
    offline = bf.Client(offline=True, cache_dir=tmp_path / "cache", opener=lambda *a, **k: pytest.fail("Netz"))
    assert offline.get("/v5/market/tickers", category="linear") == {"list": [1]}  # aus dem Cache
    with pytest.raises(bf.FatalError, match="offline"):
        offline.get("/v5/market/tickers", category="spot")


def test_client_errors():
    def forbidden(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b""))

    with pytest.raises(bf.FatalError, match="403"):  # IP-Limit oder Region – Wiederholen hilft nicht
        bf.Client(opener=forbidden, pause=0, sleep=lambda s: pytest.fail("kein Warten")).get("/v5/market/kline")
    bad = bf.Client(opener=lambda req, timeout: Answer({"retCode": 10001, "retMsg": "params error"}), pause=0)
    with pytest.raises(bf.BybitError, match="10001") as e:  # betrifft nur dieses Symbol
        bad.get("/v5/market/kline", symbol="NOPE")
    assert not isinstance(e.value, bf.FatalError)

    def down(req, timeout):
        raise urllib.error.URLError("no route to host")

    slept = []
    with pytest.raises(bf.FatalError, match="3 attempts"):
        bf.Client(opener=down, pause=0, retries=3, sleep=slept.append).get("/v5/market/kline")
    assert slept == [1.0, 2.0]  # wachsende Pause


# --- Kennzahlen ---------------------------------------------------------------------------------------------------
def collect_series(fake, s, days=25):
    end = fake.now_ms
    start = end - days * 24 * HOUR_MS
    fetch_start = start - (s.baseline + s.window + 2) * HOUR_MS
    return bf.collect(fake, SYMBOLS, s, fetch_start, end), start, end


def test_no_look_ahead():
    """Jede Zeile hängt nur von Daten bis zum Schluss ihrer Kerze ab: Schneidet man alles Spätere ab (Kerzen, Funding,
    Open Interest), bleiben die früheren Zeilen bitgenau gleich."""
    fake = FakeBybit(n=1500)
    s = bf.Settings(window=24, baseline=200)
    raw, start, end = collect_series(fake, s)
    full = bf.build_series(raw, s, start, end)
    cut = start + 300 * HOUR_MS + 17 * 60_000  # mitten in einer Kerze

    def until(d):
        k, f, oi = d["kline"], d["funding"], d["oi"]
        return {"kline": k[k["start"] + HOUR_MS <= cut], "funding": f[f["ts"] <= cut], "oi": oi[oi["ts"] <= cut],
                "only_benchmark": d["only_benchmark"]}

    part = bf.build_series({sym: until(d) for sym, d in raw.items()}, s, start, cut)
    same = full[full["_start"] + HOUR_MS <= cut].reset_index(drop=True)
    assert len(part) > 800
    pd.testing.assert_frame_equal(part, same)


def test_series_rows_and_values():
    fake = FakeBybit(n=1500)
    s = bf.Settings(window=24, baseline=200)
    raw, start, end = collect_series(fake, s)
    df = bf.build_series(raw, s, start, end, keep_ohlcv=True)
    assert list(df.columns[:3]) == ["row", "symbol", "time"] and not df.drop(columns="_start").isna().any().any()
    last = (fake.now_ms // HOUR_MS) * HOUR_MS - HOUR_MS  # jüngste ABGESCHLOSSENE Kerze
    assert df["_start"].max() == last and df["_start"].min() >= start
    first = -(-start // HOUR_MS) * HOUR_MS  # erste Kerze, die nach dem Start beginnt
    assert (df.groupby("symbol").size() == (last - first) // HOUR_MS + 1).all()  # Vorlauf weg, laufende Kerze weg
    btc = df[df["symbol"] == "BTCUSDT"]
    assert (btc["corr_btc"] == 1.0).all() and (btc["excess_z"] == 0.0).all()  # Benchmark gegen sich selbst
    assert df[["ret_z", "range_z", "turnover_z", "oi_z"]].abs().max().max() <= bf.Z_CLIP
    assert df["er"].between(0, 1).all() and df["corr_btc"].between(-1, 1).all() and (df["cs_spread_pct"] >= 0).all()
    k = raw["ETHUSDT"]["kline"].set_index("start")
    eth = df[df["symbol"] == "ETHUSDT"].set_index("_start")
    t = eth.index[50]
    assert eth.at[t, "ret_pct"] == pytest.approx(100 * (k.at[t, "close"] / k.at[t - HOUR_MS, "close"] - 1), abs=1e-5)
    assert eth.at[t, "range_pct"] == pytest.approx(100 * (k.at[t, "high"] - k.at[t, "low"]) / k.at[t, "open"],
                                                   abs=1e-5)
    r = np.log(k["close"]).diff()
    prev = r.loc[:t - HOUR_MS].iloc[-200:]  # die 200 Kerzen davor – ohne die Kerze selbst
    assert eth.at[t, "ret_z"] == pytest.approx((r.at[t] - prev.mean()) / prev.std(), abs=1e-4)
    fund = raw["ETHUSDT"]["funding"]
    known = fund[fund["ts"] <= t + HOUR_MS]["rate"].iloc[-1]  # zum Kerzenschluss bekanntes Funding
    assert eth.at[t, "funding_bp"] == pytest.approx(known * 1e4, abs=1e-5)
    assert eth.at[t, "time"] == bf.ms_text(t) and eth.at[t, "row"] == f"ETHUSDT {bf.ms_text(t)}"


def test_short_history_and_missing_benchmark_do_not_break_it():
    fake = FakeBybit(n=1500)
    s = bf.Settings(window=24, baseline=200)
    raw, start, end = collect_series(fake, s)
    raw["SOLUSDT"]["kline"] = raw["SOLUSDT"]["kline"].iloc[-150:]  # erst kürzlich gelistet: kein Vorlauf
    notes = []
    df = bf.build_series(raw, s, start, end, log=notes.append)
    sol = df[df["symbol"] == "SOLUSDT"]
    assert len(sol) < 150 and any("without all values dropped" in n for n in notes)
    raw.pop("BTCUSDT")
    df = bf.build_series(raw, s, start, end, log=notes.append)
    assert "corr_btc" not in df.columns and "excess_z" not in df.columns and len(df)
    assert any("benchmark BTCUSDT missing" in n for n in notes)


def test_cross_section_values(tmp_path):
    rc, out, _ = run(tmp_path, "--mode", "cross", "--days", "40", fake=FakeBybit(n=1500))
    assert rc == 0
    df = pd.read_csv(out)
    assert df["symbol"].tolist() == SYMBOLS and list(df.columns) == ["symbol", *bf.CROSS_FEATURES]
    btc = df.set_index("symbol").loc["BTCUSDT"]
    assert btc["corr_btc"] == 1.0 and btc["beta_btc"] == 1.0
    assert (df["vola_ann_pct"] > 0).all() and (df["max_dd_pct"] >= 0).all() and df["er"].between(0, 1).all()
    assert df["funding_mean_bp"].between(0.3, 1.5).all()  # simuliert: etwa 1 bp je Abrechnung
    sol = df.set_index("symbol").loc["SOLUSDT"]
    eth = df.set_index("symbol").loc["ETHUSDT"]
    assert sol["ret_kurt"] > 3 > eth["ret_kurt"]  # Kurssturz → dicke Ränder
    assert sol["beta_btc"] > eth["beta_btc"] > 1  # simuliert: SOL schwankt stärker mit BTC als ETH


def test_spot_and_without_benchmark(tmp_path):
    rc, out, _ = run(tmp_path, "--category", "spot", "--symbols", "ETHUSDT,SOLUSDT", "--days", "20",
                     "--benchmark", "none")
    assert rc == 0
    cols = set(pd.read_csv(out).columns)
    assert not cols & (bf.FUNDING_COLS | bf.OI_COLS | bf.BENCH_COLS)
    assert {"ret_z", "range_z", "turnover_z", "vola_regime", "er", "cs_spread_pct"} <= cols


def test_unknown_symbol_is_skipped(tmp_path, capsys):
    rc, out, _ = run(tmp_path, "--symbols", "ETHUSDT,NOPEUSDT", "--days", "20")
    assert rc == 0
    df = pd.read_csv(out)
    assert set(df["symbol"]) == {"ETHUSDT"}  # BTCUSDT nur als Benchmark geholt, nicht in der CSV
    err = capsys.readouterr().err
    assert "NOPEUSDT: skipped" in err and "10001" in err
    rc, _, _ = run(tmp_path, "--symbols", "NOPEUSDT", "--days", "20", "--benchmark", "none", name="leer.csv")
    assert rc == 1 and "no rows" in capsys.readouterr().err


# --- CSV ↔ App ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mode, config", [("series", SERIES_CFG), ("cross", CROSS_CFG)])
def test_example_configuration_matches_the_csv(tmp_path, mode, config):
    rc, out, _ = run(tmp_path, "--mode", mode, "--days", "30")
    assert rc == 0
    df = pd.read_csv(out)
    cols = set(df.columns)
    cfg = json.loads((CONFIGS / config).read_text(encoding="utf-8"))
    w = cfg["widgets"]
    assert set(w["w_features"]) <= cols  # alle Features der Beispiel-Konfiguration stehen in der CSV …
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in w["w_features"])  # … als Zahlen
    assert {a["Spalte"] for a in cfg["step2"]} <= cols and w["w_name_col"] in cols
    assert w["w_model"] in pl.ANOMALY_MODELS and set(w) <= set(pl.SETTING_TYPES)
    found = [file for file, need in examples_in_app().items() if need <= cols]
    assert found == [config]  # die App erkennt genau diese Beispiel-Konfiguration
    info = pl.read_feature_info(out.with_name(out.stem + "_columns.csv"))
    assert list(info) == list(df.columns) and all(v.strip() for v in info.values())  # Schritt 1: Beschreibungen


def test_the_crash_is_the_top_anomaly(tmp_path):
    """Ende zu Ende: CSV → Schritt 2/4 mit der Beispiel-Konfiguration → Isolation Forest (1 %) → der eingebaute
    Kurssturz steht ganz oben."""
    rc, out, fake = run(tmp_path, "--days", "60")
    assert rc == 0
    viz, prep, ccfg, _ = prepare_like_app(out, SERIES_CFG)
    assert prep.X.shape[1] == len(bf.SERIES_FEATURES) and ccfg.contamination == 0.01
    model, hist, info = pl.train_anomaly(ccfg, prep)
    assert info["share"] == pytest.approx(0.01, abs=0.002)
    top = pl.top_anomalies(prep, hist["score"], hist["flag"], n=3)
    shock = f"SOLUSDT {bf.ms_text(fake.shock_start)}"
    assert top["Bezeichnung"].iloc[0] == shock and top["Anomalie"].iloc[0], top
    devs = top["größte Abweichungen (z)"].iloc[0]  # was die Zeile auffällig macht: Sprünge bei OI, Funding, Rendite …
    assert sum(c in devs for c in ("oi_z", "excess_z", "funding_z", "ret_z", "turnover_z", "range_z")) == 3


def test_split_trains_on_the_past_and_scores_the_rest(tmp_path):
    fake = FakeBybit()
    split = bf.ms_text(fake.shock_start - 3 * 24 * HOUR_MS)  # Kurssturz liegt im neuen Teil
    rc, out, _ = run(tmp_path, "--days", "60", "--split", split, fake=fake)
    assert rc == 0
    train, new = out.with_name("out_train.csv"), out.with_name("out_new.csv")
    a, b = pd.read_csv(train), pd.read_csv(new)
    assert a["time"].max() < split <= b["time"].min() and list(a.columns) == list(b.columns)
    _, prep, ccfg, _ = prepare_like_app(train, SERIES_CFG)
    model, hist, info = pl.train_anomaly(ccfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, ccfg, info)
    folder = pl.save_bundle(model, recipe, tmp_path / "models" / "bybit")
    model2, recipe2 = pl.load_bundle(folder)
    res, rep, cmp = pl.score_new(model2, recipe2, pd.read_csv(new))
    assert cmp is None and len(res) == len(b)
    worst = res.sort_values(pl.RESULT_SCORE, ascending=False).iloc[0]
    assert worst["row"] == f"SOLUSDT {bf.ms_text(fake.shock_start)}" and worst[pl.RESULT_ANOMALY] == 1
    assert set(pl.required_raw_columns(recipe2)) <= set(b.columns)


def test_split_outside_the_period_is_an_error(tmp_path, capsys):
    rc, _, _ = run(tmp_path, "--days", "20", "--split", "2020-01-01")
    assert rc == 1 and "leaves one part empty" in capsys.readouterr().err
