"""
Regressionstests zu Audit 6 (Anomalieerkennung und Bybit-Skript aus Version 2): jeder Test hat einen gefundenen
Fehler reproduziert und schlägt mit dem Code vor dem Audit fehl.

    pytest -q tests/test_audit6_regressions.py
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
import test_anomaly as ta  # noqa: E402
from synthetic import HOUR_MS, FakeBybit, anomaly_like  # noqa: E402
from test_app import button, go, load, ok  # noqa: E402
from test_bybit_features import SYMBOLS, run  # noqa: E402
from tools import bybit_features as bf  # noqa: E402

APP = str(ROOT / "app.py")
IF, LOF, DB = pl.MODEL_IFOREST, pl.MODEL_LOF, pl.MODEL_DBSCAN


def prep_of(df, group=None, name=None, scaler="StandardScaler"):
    """Schritt 2 (Vorschläge, die Bezeichnung bleibt) und 4 wie in der App – alle übrigen Spalten als Features."""
    prof = pl.column_profile(df)
    if name is not None:
        prof.loc[prof["Spalte"] == name, "Aktion"] = pl.A_KEEP
    viz, fitted, _, lin = pl.apply_column_actions(df, pl.actions_from_table(prof))
    feats = [c for c in viz.columns if c not in (group, name)]
    return pl.prepare_cluster_data(viz, feats, scaler=scaler, column_actions=fitted, lineage=lin, group_col=group,
                                   name_col=name)


def app_and_code(tmp_path, raw, cfg, with_curve=False):
    """App (Schritt 4, 6, 7) und der angezeigte Code bis Schritt 7 → (prep, hist, info, ns). with_curve: der Code
    bekommt das Ergebnis aus der App (wie in Schritt 7 und im Skript aus Schritt 9)."""
    df, viz, fitted, lineage, code = ta.load(tmp_path, raw)
    prep = pl.prepare_cluster_data(viz, list(viz.columns), column_actions=fitted, lineage=lineage)
    model, hist, info = pl.train_anomaly(cfg, prep)
    n = len(prep.X)
    curve = cg.curve_code(cfg, n, hist["curve"]) if with_curve else cg.curve_code(cfg, n)
    src = "\n".join([*code, cg.step4_code(prep.recipe), cg.model_code(cfg, n), curve])
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, "generierter_code", "exec"), ns)  # noqa: S102 – eigener, erzeugter Code
    plt.close("all")
    return prep, hist, info, ns


# --- A1: positive Gruppe in Schritt 10 ---------------------------------------------------------------------------------
def test_scoring_new_data_keeps_the_positive_class_from_training():
    """A1: In Schritt 10 galt die in den NEUEN Daten seltenere Gruppe als 'positiv'. Eine Stichprobe mit vielen
    Betrugsfällen (15 Betrug, 5 normal) wurde so gegen 'normal' bewertet: Precision, Recall und ROC-AUC 0."""
    train = anomaly_like(1000, seed=3)
    prep = prep_of(train, group="Class", name="ID")
    cfg = pl.ClusterConfig(model=IF, contamination=0.02)
    model, hist, info = pl.train_anomaly(cfg, prep)
    cmp_train = pl.compare_anomalies(hist["flag"], hist["score"], prep.groups)
    assert cmp_train["positive"] == "1"  # im Training die seltenere Gruppe
    recipe = pl.finalize_recipe(prep.recipe, cfg, info, None, cmp_train)
    assert recipe["anomaly_metrics"]["positive"] == "1"

    pool = anomaly_like(400, share=0.1, seed=9)
    new = pd.concat([pool[pool["Class"] == 1].head(15), pool[pool["Class"] == 0].head(5)], ignore_index=True)
    res, rep, cmp = pl.score_new(model, recipe, new)
    assert cmp["positive"] == "1"  # vorher: '0' – in den neuen Daten die seltenere Gruppe
    assert cmp["recall"] > 0.8 and cmp["precision"] > 0.9 and cmp["roc_auc"] > 0.9  # vorher alle 0
    assert cmp["positive_source"] == "training" and cmp_train["positive_source"] == "rarer"
    text = " ".join(pl.assess_scoring(recipe, res, rep, cmp)["meaning"])
    assert "zu '1'" in text

    old = {k: v for k, v in recipe.items() if k != "anomaly_metrics"}  # Rezept ohne Vergleich im Training
    assert pl.score_new(model, old, new)[2]["positive_source"] == "rarer"
    for value in (1, 1.0, " 1 "):  # Zahl oder Text – wie die Gruppen normalisiert
        assert pl.compare_anomalies(hist["flag"], hist["score"], prep.groups, positive=value)["positive"] == "1"
    assert pl.compare_anomalies(hist["flag"], hist["score"], prep.groups, positive="x")["positive_source"] == "rarer"


# --- B1: eps aus einer anderen Vorbereitung ---------------------------------------------------------------------------
def test_start_eps_follows_a_new_preparation(tmp_path, monkeypatch):
    """B1: Der Startwert für eps (Schritt 6) wurde nur einmal berechnet. Nach einer neuen Vorbereitung mit anderem
    Scaler (StandardScaler → MinMaxScaler, Abstände rund 6-mal kleiner) blieb das alte eps stehen – DBSCAN fand dann
    keine einzige Anomalie. Selbst eingegebene Werte bleiben erhalten."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    anomaly_like(600, seed=3).to_csv(tmp_path / "daten.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "daten.csv")
    go(at, 1)
    button(at, "Übernehmen").click().run()

    def prepare(scaler):
        go(at, 3)
        at.selectbox(key="w_scaler").set_value(scaler).run()
        button(at, "Daten fürs Modell vorbereiten").click().run()
        ok(at, f"Schritt 4 ({scaler})")
        go(at, 5)
        at.radio(key="w_model").set_value(DB).run()
        ok(at, "Schritt 6")
        p, ss = at.session_state["prep"], at.session_state
        return ss["w_eps"], pl.suggest_eps(p.X, ss["w_min_samples"], ss["w_metric"])

    eps_std, fresh_std = prepare("StandardScaler")
    assert eps_std == fresh_std
    eps_mm, fresh_mm = prepare("MinMaxScaler")
    assert fresh_mm < eps_std / 3
    assert eps_mm == fresh_mm  # vorher: weiter der Wert des StandardScalers
    at.number_input(key="w_eps").set_value(0.123).run()  # selbst eingegeben → bleibt
    eps_own, _ = prepare("StandardScaler")
    assert eps_own == 0.123

    # Vorschlag aus Schritt 7 übernommen: stammt aus den Daten → nach neuer Vorbereitung wieder neu berechnet
    at.number_input(key="w_eps").set_value(3.0).run()
    button(at, "Training starten").click().run()
    ok(at, "Training")
    go(at, 6)
    offers = [b for b in at.button if b.label.startswith("eps = ") and b.label.endswith("übernehmen")]
    offers[0].click().run()
    applied = at.session_state["w_eps"]
    assert applied < 3.0
    eps_mm2, fresh_mm2 = prepare("MinMaxScaler")
    assert eps_mm2 == fresh_mm2 != applied


# --- B7: eps-Vorschlag 0 ----------------------------------------------------------------------------------------------
def categorical_only(n=300, seed=5):
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame({"color": rng.choice(["rot", "grün", "blau"], n, p=[0.6, 0.3, 0.1]),
                        "size": rng.choice(["S", "M", "L"], n, p=[0.5, 0.4, 0.1])})
    raw.loc[:5, "color"] = "lila"  # seltene Kombinationen
    return raw


def test_dbscan_eps_suggestion_is_never_zero(tmp_path):
    """B7: Nur kategoriale Features → viele genau gleiche Zeilen, k-Distanz meist 0 → Schritt 7 bot 'eps = 0
    übernehmen' an (eps muss größer als 0 sein)."""
    cfg = pl.ClusterConfig(model=DB, eps=0.5, min_samples=5)
    prep, hist, info, ns = app_and_code(tmp_path, categorical_only(), cfg)
    curve = hist["curve"]
    assert curve["knee"] is not None and curve["kdist"][curve["knee"]] == 0
    assert curve["suggest"] == 1e-6  # vorher 0.0
    assert ns["eps_knee"] == curve["suggest"]  # angezeigter Code = App
    fz = pl.threshold_assessment(curve, info, cfg)
    assert [v for _, v in fz["suggest"]] == [1e-6]
    assert any("Viele genau gleiche Zeilen" in t for t in fz["notes"])
    _, h, small = pl.train_anomaly(pl.ClusterConfig(model=DB, eps=pl.EPS_MIN, min_samples=5), prep)
    assert small["k"] >= 3 and 0 < small["share"] < 0.2  # gleiche Zeilen = Cluster, seltene Kombinationen = Rauschen


# --- P-a: Speicher bei DBSCAN und LOF ---------------------------------------------------------------------------------
def blobs(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])


def test_neighbor_pairs_estimates_what_dbscan_keeps_in_memory(monkeypatch):
    """P-a: DBSCAN hält alle Nachbarschaften gleichzeitig im Speicher (scikit-learn) – bei großem eps oder vielen
    gleichen Zeilen wächst das quadratisch (gemessen: 6 000 Zeilen, großes eps → 0,8–1,3 GB; 20 000 Zeilen → Absturz
    der App). Die Schätzung aus einer Stichprobe trifft die genaue Zahl."""
    prep = prep_of(blobs())
    X = prep.X
    assert pl.neighbor_pairs(X, 0.3) == 3000.0 ** 2  # wenige Zeilen: unkritisch, nichts zu messen
    monkeypatch.setattr(pl, "MAX_NEIGHBOR_PAIRS", 1)
    eps = [0.2, 0.6, 2.0]
    est = pl.neighbor_pairs(X, eps)
    exact = [float(np.count_nonzero(pl.pairwise_distances(X) <= e)) for e in eps]
    assert est == sorted(est) and all(v == pytest.approx(x, rel=0.25) for v, x in zip(est, exact))
    assert pl.neighbor_pairs(X, 0.6) == est[1]  # eine Zahl oder eine Liste: gleiche Stichprobe


def test_training_refuses_settings_that_would_fill_the_memory(monkeypatch):
    prep = prep_of(blobs())
    monkeypatch.setattr(pl, "MAX_NEIGHBOR_PAIRS", 200_000)  # wie ein großer Datensatz
    with pytest.raises(ValueError, match="Nachbarpaare"):  # vorher: DBSCAN.fit ohne Schutz
        pl.train_anomaly(pl.ClusterConfig(model=DB, eps=3.0, min_samples=5), prep)
    with pytest.raises(ValueError, match="n_neighbors"):
        pl.train_anomaly(pl.ClusterConfig(model=LOF, n_neighbors=100), prep)
    pl.train_anomaly(pl.ClusterConfig(model=DB, eps=0.15, min_samples=5, curve=False), prep)  # kleines eps: geht
    pl.train_anomaly(pl.ClusterConfig(model=LOF, n_neighbors=20), prep)


def test_eps_table_skips_values_that_would_fill_the_memory(tmp_path, monkeypatch):
    """P-a: Schritt 7 lässt DBSCAN für Quantile der k-Distanz laufen (bis 99 %) – auch dort zu große eps überspringen.
    Der angezeigte Code überspringt dieselben."""
    raw = blobs()
    prep = prep_of(raw)
    cfg = pl.ClusterConfig(model=DB, eps=pl.suggest_eps(prep.X, 5), min_samples=5)
    monkeypatch.setattr(pl, "MAX_NEIGHBOR_PAIRS", 1)  # Grenze 1: alle eps der Tabelle übersprungen, Schätzung erzwungen
    values = pl.anomaly_curve(cfg, prep.X, np.zeros(len(prep.X)))["grid"]["skipped"]
    est = pl.neighbor_pairs(prep.X, values)
    i = values.index(cfg.eps)
    assert i < len(values) - 1 and est[i] < est[-1]  # es gibt größere eps mit mehr Nachbarpaaren
    monkeypatch.setattr(pl, "MAX_NEIGHBOR_PAIRS", int((est[i] + est[-1]) / 2))  # eps aus Schritt 6 geht, größere nicht
    _, hist, info, ns = app_and_code(tmp_path, raw, cfg, with_curve=True)
    eps = cfg.eps
    grid = hist["curve"]["grid"]
    assert grid["skipped"] and max(grid["eps"]) < min(grid["skipped"]) and eps in grid["eps"]
    assert ns["eps_values"] == grid["eps"] and ns["clusters"] == grid["clusters"]
    np.testing.assert_array_equal(ns["noise"], grid["noise"])
    notes = pl.threshold_assessment(hist["curve"], info, cfg)["notes"]
    assert any("übersprungen" in t for t in notes)
    assert "übersprungen" not in cg.curve_code(cfg, len(prep.X))  # ohne Ergebnis aus der App: alle eps


# --- C1, C2: Randfälle --------------------------------------------------------------------------------------------
def test_identical_rows_are_refused_instead_of_100_percent_anomalies():
    """C1: Alle Zeilen gleich → Isolation Forest meldete ALLE 50 Zeilen als auffällig (Score = Schwelle 0.5, Rundung)."""
    prep = prep_of(pd.DataFrame({"a": [1.0] * 50, "b": [2.0] * 50}))
    for key in (IF, LOF, DB):
        with pytest.raises(ValueError, match="Alle Zeilen sind nach der Vorbereitung gleich"):
            pl.train_anomaly(pl.ClusterConfig(model=key, eps=0.5), prep)


@pytest.mark.parametrize("key", [IF, LOF])
def test_small_data_does_not_suggest_half_of_the_rows(tmp_path, key):
    """C2: Unter 100 Zeilen wurde der Knick in mehr als den auffälligsten 20 % gesucht (mindestens 20 Zeilen) – bei
    12 Zeilen schlug Schritt 7 contamination = 50 % vor."""
    rng = np.random.default_rng(3)
    raw = pd.DataFrame({"a": rng.normal(size=12), "b": rng.normal(size=12)})
    cfg = pl.ClusterConfig(model=key)  # LOF: n_neighbors 20 → höchstens Zeilen − 1 = 11
    prep, hist, info, ns = app_and_code(tmp_path, raw, cfg)
    curve = hist["curve"]
    assert curve["knee"] is None or curve["knee"] + 1 <= 3  # höchstens die auffälligsten 20 % (12 × 0.2 → 3)
    assert ns["knee"] == curve["knee"]  # angezeigter Code = App
    assert all(v <= 0.25 for _, v in pl.threshold_assessment(curve, info, cfg)["suggest"])


def test_kdist_chart_always_shows_the_eps_line(recwarn):
    """C3: Lag eps über allen k-Distanzen (alles Kernpunkte – z. B. eps aus einer anderen Skalierung, oder nur gleiche
    Zeilen), endete die y-Achse unter eps: Linie und Beschriftung 'eingestellt: eps' fehlten, tight_layout warnte."""
    import plots

    for kd, eps in ((np.zeros(50), 0.5), (np.sort(np.random.default_rng(0).gamma(2, 0.05, 600)), 1.049)):
        fig = plots.fig_kdist({"kdist": kd, "k": 5, "knee": None, "grid": None}, eps)
        lo, hi = fig.axes[0].get_ylim()
        assert lo == 0 and hi > eps  # vorher: 0.055 bzw. 0.46
    assert not [w for w in recwarn if "tight_layout" in str(w.message) or "Tight layout" in str(w.message)]
    plt.close("all")


# --- Bybit: Open Interest -----------------------------------------------------------------------------------------
def test_open_interest_only_counts_once_its_interval_is_over():
    """Bybit dokumentiert nicht, ob der Zeitstempel eines OI-Werts den Beginn seines Rasters meint. Dann ist der Wert
    erst am Ende des Rasters bekannt – die Kerze, die zum Zeitstempel schließt, nutzte ihn aber schon (Blick in die
    Zukunft). Jetzt: Schneidet man alle OI-Werte ab, deren Raster beim Schnitt noch läuft, bleibt alles gleich."""
    fake = FakeBybit(n=1500)
    s = bf.Settings(window=24, baseline=200)
    end = fake.now_ms
    start = end - 25 * 24 * HOUR_MS
    raw = bf.collect(fake, SYMBOLS, s, start - (s.baseline + s.window + 2) * HOUR_MS, end)
    full = bf.build_series(raw, s, start, end)
    step = int(np.median(np.diff(raw["ETHUSDT"]["oi"]["ts"])))  # Raster der geholten OI-Werte
    cut = (start // HOUR_MS + 300) * HOUR_MS  # genau ein Kerzenschluss

    def until(d):
        k, f, oi = d["kline"], d["funding"], d["oi"]
        return {"kline": k[k["start"] + HOUR_MS <= cut], "funding": f[f["ts"] <= cut],
                "oi": oi[oi["ts"] + step <= cut], "only_benchmark": d["only_benchmark"]}

    part = bf.build_series({sym: until(d) for sym, d in raw.items()}, s, start, cut)
    same = full[full["_start"] + HOUR_MS <= cut].reset_index(drop=True)
    assert len(part) > 800
    pd.testing.assert_frame_equal(part, same)
    sol = full[full["symbol"] == "SOLUSDT"].set_index("_start")
    assert sol.at[fake.shock_start, "oi_chg_pct"] == pytest.approx(-15, abs=1)  # Rückgang trotzdem in der Sturz-Kerze


def test_series_asks_for_a_finer_open_interest_grid(tmp_path):
    rc, _, fake = run(tmp_path, "--symbols", "ETHUSDT", "--days", "20")
    assert rc == 0
    asked = {p["intervalTime"] for path, p in fake.requests if path == "/v5/market/open-interest"}
    assert asked == {"15min"}  # ¼ der 1h-Kerze
    rc, _, fake = run(tmp_path, "--mode", "cross", "--symbols", "ETHUSDT", "--days", "20", name="quer.csv")
    assert rc == 0
    assert {p["intervalTime"] for path, p in fake.requests if path == "/v5/market/open-interest"} == {"1h"}


def test_one_minute_candles_leave_out_open_interest(tmp_path, capsys):
    """Open Interest gibt es nur alle 5 Minuten: Bei 1-Minuten-Kerzen war oi_chg_pct in 4 von 5 Kerzen 0 und sprang
    in jeder fünften – ein künstliches Muster, das die Anomalieerkennung als 'auffällig' lernen kann."""
    fake = FakeBybit(minutes=1, n=4000)
    rc, out, _ = run(tmp_path, "--interval", "1", "--symbols", "ETHUSDT", "--days", "1", "--baseline", "200",
                     "--window", "30", fake=fake)
    assert rc == 0
    assert not set(pd.read_csv(out).columns) & bf.OI_COLS
    assert "open interest exists only every 5 minutes" in capsys.readouterr().err
