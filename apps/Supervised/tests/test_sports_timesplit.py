"""
Zeitlicher Split, Gradient Boosting und Vergleich mit einer Wahrscheinlichkeit aus den Daten (z. B. Elo) – für
Datensätze wie die des Sportdaten-Tools (pre_*/lineup_*/inplay_*/post_*). Der angezeigte Code rechnet dasselbe wie
die App, auch beim zeitlichen Split.

    pytest -q tests/test_sports_timesplit.py
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from synthetic import sports_like  # noqa: E402

APP = str(ROOT / "app.py")
RNG = np.random.default_rng(17)
pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning",
    "ignore:The inplace keyword in DataFrame.reset_index is deprecated:DeprecationWarning:streamlit")


def _viz(raw):
    """Schritt 2 mit den vorgeschlagenen Aktionen (wie 'Übernehmen' in der App)."""
    viz, fitted, _, lineage = pl.apply_column_actions(raw, pl.actions_from_table(pl.column_profile(raw)))
    return viz, fitted, lineage


@pytest.fixture(scope="module")
def games():
    return sports_like(300, seed=11)


# --- Erkennung des Formats -----------------------------------------------------------------------------------------
def test_layout_of_the_sports_tool_is_detected(games):
    lay = pl.match_data_layout(games.columns)
    assert lay["label"] == "home_win" and lay["time"] == "date" and lay["baseline"] == "pre_elo_prob_home"
    assert all(c.startswith(("pre_", "lineup_")) for c in lay["features"])
    assert set(lay["leaky"]) == {"inplay_ht_diff", "post_margin", "post_total"}
    # Schritt 2: die Datumsspalte bleibt für den zeitlichen Split erhalten (sonst 'Datum → Jahr')
    prof = pl.column_profile(games).set_index("Spalte")
    assert prof.at["date", "Aktion"] == pl.A_KEEP
    # Audit 5: erkannt am Label (home_win/p1_win) + pre_* – auch ohne post_/inplay_ (im Tool abwählbar) …
    assert pl.column_profile(games.drop(columns=["post_margin", "post_total", "inplay_ht_diff"])).set_index(
        "Spalte").at["date", "Aktion"] == pl.A_KEEP
    # … aber ohne Label kein Sportdatensatz: dann wieder der normale Vorschlag 'Datum → Jahr'
    assert pl.column_profile(games.drop(columns=["home_win"])).set_index("Spalte").at["date", "Aktion"] == pl.A_YEAR
    assert pl.match_data_layout(["pre_x", "y"]) is None and pl.match_data_layout(["post_code", "y"]) is None


def test_probability_columns_and_time_column_suggestion(games):
    probs = pl.probability_columns(games)
    assert "pre_elo_prob_home" in probs
    assert "home_win" not in probs  # 0/1 ist keine Wahrscheinlichkeit, sondern ein Flag
    assert pl.suggest_time_column(games.columns) == "date"
    assert pl.suggest_time_column(["a", "Saison", "b"]) == "Saison"
    assert pl.suggest_time_column(["a", "b"]) is None


# --- zeitlicher Split ----------------------------------------------------------------------------------------------
def test_time_split_trains_on_the_past_and_keeps_days_together(games):
    viz, fitted, lineage = _viz(games)
    lay = pl.match_data_layout(viz.columns)
    prep = pl.prepare_ml_data(viz, "home_win", lay["features"], pl.T_BIN, positive_class="1", test_size=0.25,
                              split_mode=pl.SPLIT_TIME, time_column="date", column_actions=fitted, lineage=lineage,
                              missing={c: {"strategy": pl.M_MEDIAN} for c in lay["features"]})
    sp = prep.recipe["split"]
    assert sp["mode"] == pl.SPLIT_TIME and sp["time_kind"] == "date" and sp["time_format"] == "%Y-%m-%d"
    assert not sp["stratify"] and not sp["stratify_used"]
    d_tr = pd.to_datetime(viz.loc[prep.idx_train, "date"])
    d_te = pd.to_datetime(viz.loc[prep.idx_test, "date"])
    assert d_tr.max() < d_te.min()  # Training liegt vollständig vor dem Test – kein Tag in beiden Teilen
    assert d_tr.is_monotonic_increasing and d_te.is_monotonic_increasing  # nach der Zeit sortiert
    assert abs(sp["test_share"] - 0.25) < 0.02
    assert sp["train_until"] == d_tr.max().strftime("%Y-%m-%d") and sp["test_from"] == d_te.min().strftime("%Y-%m-%d")
    assert any("Zeitlicher Split nach 'date'" in line for line in prep.log)


def test_time_split_by_season_number_and_errors():
    viz, *_ = _viz(sports_like(700, seed=3))  # Saisons 2022, 2023 und der Anfang von 2024
    feats = ["pre_diff_elo", "pre_home_rest_days"]
    prep = pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, test_size=0.3, split_mode=pl.SPLIT_TIME,
                              time_column="season")
    sp = prep.recipe["split"]
    assert sp["time_kind"] == "number" and sp["test_from"] == "2023"  # ganze Saisons im Test
    assert any("Testanteil" in w for w in prep.warnings)  # 2023 bleibt zusammen: Anteil weicht von 0.3 ab
    one_season = viz[viz["season"] == 2022]
    with pytest.raises(ValueError, match="denselben Zeitwert"):
        pl.prepare_ml_data(one_season, "home_win", feats, pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="season")
    with pytest.raises(ValueError, match="nicht das Label"):
        pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="home_win")
    with pytest.raises(ValueError, match="weder eine Zahl"):
        pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="home_team")
    with pytest.raises(ValueError, match="Zeitspalte wählen"):
        pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column=None)


@pytest.mark.parametrize("time_col", ["date", "season"])
def test_time_split_code_computes_the_same_as_the_app(tmp_path, time_col):
    raw = sports_like(700, seed=3)  # mehrere Saisons
    raw.loc[[3, 40, 41], "date"] = np.nan  # Zeilen ohne Zeitwert werden in App und Code gleich entfernt
    raw.loc[[7, 99], "season"] = np.nan
    raw.loc[::9, "pre_home_rest_days"] = np.nan
    raw["season_txt"] = raw["season"]  # Zeitspalte darf auch selbst ein Feature mit vielen Lücken sein …
    raw.loc[::2, "season_txt"] = np.nan  # … das „Spalte entfernen“ wird (Code: Zeitspalte bleibt für den Split)
    feats = ["pre_diff_elo", "pre_elo_prob_home", "pre_home_rest_days", "pre_home_l10_net", "home_team", "season_txt"]
    missing = {"pre_home_rest_days": {"strategy": pl.M_MEDIAN}, "pre_home_l10_net": {"strategy": pl.M_MEAN},
               "season_txt": {"strategy": pl.M_DROP_COL}}
    tc = "season_txt" if time_col == "season" else time_col
    prep, ns = parity.run_both(tmp_path, raw, {"date": (pl.A_KEEP, "", "")}, "home_win", pl.T_BIN,
                               positive_class="1", features=feats, missing=missing, test_size=0.2,
                               split_mode=pl.SPLIT_TIME, time_column=tc, baseline_column="pre_elo_prob_home",
                               scaler="StandardScaler")
    assert prep.recipe["split"]["time_removed"] > 0
    assert "season_txt" in prep.recipe["dropped_columns"]
    parity.assert_same(prep, ns)
    assert "train_test_split(df" not in cg.step4_code(prep.recipe)


# --- Gradient Boosting --------------------------------------------------------------------------------------------
def test_boosting_curve_finds_the_best_number_of_rounds():
    """Verrauschte Daten + hohe learning_rate: nach wenigen Runden lernt das Modell Rauschen → bester Wert < max_iter,
    Vorschlag für Schritt 6. Die Kurve stammt aus EINEM Training (staged_predict) und ist das fertige Modell."""
    x = np.random.default_rng(2).normal(size=(900, 4))
    df = pd.DataFrame(x, columns=list("pqrs")).assign(y=2 * x[:, 0] + np.random.default_rng(3).normal(0, 2, 900))
    prep = pl.prepare_ml_data(df, "y", list("pqrs"), pl.T_REG)
    cfg = pl.ModelConfig(model=pl.MODEL_GB, gb_iter=300, gb_lr=0.5, gb_leaf=2)
    model, hist, info = pl.train_model(cfg, prep)
    curve = hist["curve"]
    assert curve["param"] == "max_iter" and curve["values"] == pl.forest_steps(300)
    assert model.n_iter_ == 300 and not model.early_stopping
    direct = pl.make_model(cfg, pl.T_REG).fit(prep.X_train, prep.y_train)
    np.testing.assert_allclose(model.predict(prep.X_test), direct.predict(prep.X_test))
    a = pl.curve_assessment(curve, info, cfg, pl.T_REG)
    assert a["suggest"]["param"] == "max_iter" and a["suggest"]["value"] < 300
    assert a["verdict"] == "overfit"


def test_boosting_note_when_more_rounds_would_help():
    x = RNG.normal(size=(800, 3))
    df = pd.DataFrame(x, columns=list("abc")).assign(y=3 * x[:, 0] + x[:, 1] ** 2 + RNG.normal(0, 0.3, 800))
    prep = pl.prepare_ml_data(df, "y", list("abc"), pl.T_REG)
    cfg = pl.ModelConfig(model=pl.MODEL_GB, gb_iter=25, gb_lr=0.03)  # viel zu wenige, zu kleine Schritte
    _, hist, info = pl.train_model(cfg, prep)
    notes = " ".join(pl.curve_assessment(hist["curve"], info, cfg, pl.T_REG)["notes"])
    assert "sinkt bis zur letzten Runde" in notes


def test_time_split_validation_and_grid_search_respect_time(games):
    viz, *_ = _viz(games)
    feats = ["pre_diff_elo", "pre_home_rest_days", "pre_away_rest_days", "lineup_home_out_n"]
    prep = pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="date")
    tr, va = pl._holdout(prep.y_train, pl.T_BIN, 0.2, 1, time_order=True)
    assert va.min() > tr.max()  # Validierung = jüngster Teil der (zeitlich sortierten) Trainingsdaten
    cfg = pl.ModelConfig(model=pl.MODEL_SVM, grid_search=True, cv=3, curve=False)
    _, hist, _ = pl.train_model(cfg, prep)
    assert hist["grid"]["time_order"]
    code = cg.model_code(cfg, pl.T_BIN, 5, 3, time_split=True)
    assert "cv=TimeSeriesSplit(n_splits=3)" in code
    assert "shuffle=False" in cg.curve_code(replace(cfg, curve=True, validation="split"), pl.T_BIN, False, True)


@pytest.mark.parametrize("cfg", [pl.ModelConfig(model=pl.MODEL_GB, gb_iter=50, gb_lr=0.1, validation="split"),
                                 pl.ModelConfig(model=pl.MODEL_SVM, grid_search=True, cv=3, validation="split")])
def test_model_and_curve_code_with_time_split(tmp_path, games, cfg):
    feats = ["pre_diff_elo", "pre_elo_prob_home", "pre_home_rest_days", "pre_away_rest_days", "pre_home_l10_net"]
    prep, ns = parity.run_both(tmp_path, games, {"date": (pl.A_KEEP, "", "")}, "home_win", pl.T_BIN,
                               positive_class="1", features=feats, split_mode=pl.SPLIT_TIME, time_column="date",
                               missing={"pre_home_l10_net": {"strategy": pl.M_MEDIAN}})
    model, hist, _ = pl.train_model(cfg, prep)
    calib = pl.calibration_cv(prep.y_train) if cfg.model == pl.MODEL_SVM else 5
    gcv = pl.grid_cv(cfg, pl.T_BIN, prep.y_train) if cfg.grid_search else None
    exec(cg.model_code(cfg, pl.T_BIN, calib, gcv, time_split=True), ns)  # noqa: S102 – erzeugter Code
    np.testing.assert_allclose(ns["model"].predict_proba(ns["X_test"]), model.predict_proba(prep.X_test),
                               rtol=1e-9, atol=1e-12)
    exec(cg.curve_code(cfg, pl.T_BIN, False, time_split=True), ns)  # noqa: S102
    ns["plt"].close("all")
    np.testing.assert_allclose(ns["error_val"], hist["curve"]["val"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(ns["error_train"], hist["curve"]["train"], rtol=1e-9, atol=1e-12)


# --- Vergleich mit einer Wahrscheinlichkeit aus den Daten (Elo) ----------------------------------------------------
def test_compare_probabilities_verdicts():
    n = 4000
    p_true = RNG.uniform(0.2, 0.8, n)
    y = (RNG.random(n) < p_true).astype(int)
    noisy = np.clip(p_true + RNG.normal(0, 0.15, n), 0.01, 0.99)
    better = pl.compare_probabilities(y, p_true, noisy, 0.5)  # die wahre Wahrscheinlichkeit schlägt eine verrauschte
    assert better["verdict"] == "better" and better["delta"] < -2 * better["se"]
    worse = pl.compare_probabilities(y, noisy, p_true, 0.5)
    assert worse["verdict"] == "worse"
    same = pl.compare_probabilities(y, p_true, p_true, 0.5)
    assert same["verdict"] == "same" and same["delta"] == 0
    names = [r["name"] for r in better["rows"]]
    assert names == ["Modell", "Vergleich", "naive Vorhersage"]
    assert better["rows"][0]["log_loss"] == pytest.approx(log_loss(y, p_true))
    base = np.where(np.arange(n) % 10 == 0, np.nan, noisy)  # Zeilen ohne Wert zählen nicht
    assert pl.compare_probabilities(y, p_true, base, 0.5)["n"] == n - n // 10


def test_baseline_column_is_checked(games):
    viz, *_ = _viz(games)
    feats = ["pre_diff_elo", "pre_home_rest_days"]
    prep = pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, baseline_column="pre_elo_prob_home")
    assert prep.recipe["baseline_column"] == "pre_elo_prob_home"
    np.testing.assert_allclose(prep.base_test, viz.loc[prep.idx_test, "pre_elo_prob_home"].to_numpy())
    with pytest.raises(ValueError, match="zwischen 0 und 1"):
        pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, baseline_column="pre_diff_elo")
    with pytest.raises(ValueError, match="nicht in den Daten"):
        pl.prepare_ml_data(viz, "home_win", feats, pl.T_BIN, baseline_column="gibt_es_nicht")
    reg = pl.prepare_ml_data(viz, "pre_diff_elo", ["pre_home_rest_days"], pl.T_REG,
                             baseline_column="pre_elo_prob_home")
    assert reg.base_test is None and any("ignoriert" in w for w in reg.warnings)


def test_eval_code_compares_with_the_baseline(tmp_path, games):
    feats = ["pre_diff_elo", "pre_home_rest_days", "pre_away_rest_days"]
    prep, ns = parity.run_both(tmp_path, games, {"date": (pl.A_KEEP, "", "")}, "home_win", pl.T_BIN,
                               positive_class="1", features=feats, split_mode=pl.SPLIT_TIME, time_column="date",
                               baseline_column="pre_elo_prob_home")
    cfg = pl.ModelConfig(model=pl.MODEL_GB, gb_iter=30)
    model, _, _ = pl.train_model(replace(cfg, curve=False), prep)
    exec(cg.model_code(cfg, pl.T_BIN), ns)  # noqa: S102
    exec(cg.eval_code(pl.T_BIN, 0.5, False, cfg, "pre_elo_prob_home"), ns)  # noqa: S102
    cmp = pl.compare_probabilities(prep.y_test, pl.predict_scores(model, prep.X_test, pl.T_BIN, prep.recipe),
                                   prep.base_test, 0.5)
    ok = ns["ok"]
    assert log_loss(ns["y_test"][ok], ns["p_base"][ok]) == pytest.approx(cmp["rows"][1]["log_loss"], rel=1e-9)
    assert log_loss(ns["y_test"][ok], ns["proba"][ok]) == pytest.approx(cmp["rows"][0]["log_loss"], rel=1e-9)
    ns["plt"].close("all")


def test_full_script_with_time_split_and_baseline_runs(tmp_path, games):
    path = tmp_path / "nba_games.csv"
    games.to_csv(path, index=False)
    df, _ = pl.read_csv(path)
    viz, fitted, lineage = _viz(df)
    lay = pl.match_data_layout(viz.columns)
    prep = pl.prepare_ml_data(viz, "home_win", lay["features"], pl.T_BIN, positive_class="1",
                              split_mode=pl.SPLIT_TIME, time_column="date", baseline_column="pre_elo_prob_home",
                              missing={c: {"strategy": pl.M_MEDIAN} for c in lay["features"] if viz[c].isna().any()},
                              column_actions=fitted, lineage=lineage)
    cfg = pl.ModelConfig(model=pl.MODEL_GB, gb_iter=40, validation="split")
    model, _, info = pl.train_model(cfg, prep)
    src = {"name": "nba_games.csv", "path": str(path), "sep": ",", "decimal": ".", "encoding": "utf-8"}
    script = cg.full_script(src, fitted, {c for c in df.columns if pl.is_numeric(df[c])},
                            pl.finalize_recipe(prep.recipe, cfg, info), cfg, folder=str(tmp_path / "m"))
    ns = {}
    exec(compile(script, "ml_pipeline.py", "exec"), ns)  # noqa: S102 – erzeugter Code
    ns["plt"].close("all")
    np.testing.assert_allclose(ns["model"].predict_proba(ns["X_test"]), model.predict_proba(prep.X_test),
                               rtol=1e-9, atol=1e-12)
    assert "p_base" in ns and "TimeSeriesSplit" in script


# --- Oberfläche ------------------------------------------------------------------------------------------------------
def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


def test_app_sports_dataset_time_split_boosting_and_elo(tmp_path, monkeypatch):
    """Klick-Test mit einem Datensatz im Format des Sportdaten-Tools: Vorschläge in Schritt 4 (nur pre_*/lineup_*,
    zeitlicher Split nach date, Vergleich mit pre_elo_prob_home), Gradient Boosting auf Platz 1, besten Wert der Kurve
    übernehmen, Vergleich mit Elo in Schritt 8, Speichern als .skops und Schritt 10."""
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    games = sports_like(360, seed=5)
    games.iloc[:-60].to_csv(tmp_path / "nba_games.csv", index=False)
    games.iloc[-60:].to_csv(tmp_path / "neu.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "nba_games.csv")).run()
    _button(at, "Daten laden").click().run()
    nav = lambda i: at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()  # noqa: E731
    nav(1)
    _button(at, "Übernehmen").click().run()
    _ok(at, "Schritt 2")
    assert "date" in at.session_state["df_viz"].columns and not pl.is_numeric(at.session_state["df_viz"]["date"])

    nav(3)
    _ok(at, "Schritt 4 – Vorschläge")
    assert at.session_state["w_label"] == "home_win"
    feats = at.session_state["w_features"]
    assert feats and all(c.startswith(("pre_", "lineup_")) for c in feats)
    assert at.session_state["w_split_mode"] == pl.SPLIT_TIME and at.session_state["w_time_col"] == "date"
    assert at.session_state["w_base_col"] == "pre_elo_prob_home"
    assert any("Zeitraum in 'date'" in c.value for c in at.caption)
    assert any("pre_" in i.value and "erkannt" in i.value for i in at.info)
    at.multiselect(key="w_features").set_value(feats + ["post_margin"]).run()
    assert any("erst während oder nach dem Spiel" in w.value for w in at.warning)
    at.multiselect(key="w_features").set_value(feats).run()
    _button(at, "Daten für ML vorbereiten").click().run()
    _ok(at, "Schritt 4")
    r = at.session_state["prep"].recipe
    assert r["split"]["mode"] == pl.SPLIT_TIME and r["baseline_column"] == "pre_elo_prob_home"
    assert any("Zeitlicher Split nach 'date'" in m.value for m in at.markdown)

    nav(5)
    assert at.radio(key="w_model").options[0].startswith("1 · Gradient Boosting")
    assert at.session_state["w_model"] == pl.MODEL_GB  # Platz 1 ist vorausgewählt
    at.number_input(key="w_gb_iter").set_value(400).run()  # bewusst überangepasst: 400 Runden, learning_rate 1,
    at.number_input(key="w_gb_lr").set_value(1.0).run()     # min_samples_leaf 2 → Schritt 7 schlägt weniger Runden vor
    at.number_input(key="w_gb_leaf").set_value(2).run()
    at.radio(key="w_val").set_value("split").run()
    assert ("HistGradientBoostingClassifier(max_iter=400, learning_rate=1, min_samples_leaf=2, early_stopping=False"
            in "\n".join(c.value for c in at.code))
    _button(at, "Training starten").click().run()
    _ok(at, "Training")
    nav(6)
    _ok(at, "Schritt 7")
    curve = at.session_state["history"]["curve"]
    assert curve["param"] == "max_iter"
    best = curve["values"][curve["best_index"]]
    assert best < 400  # das Modell lernt bald nur noch Rauschen
    _button(at, "Besten Wert in Schritt 6 übernehmen").click().run()  # Audit 5: früher nur „falls vorhanden“ geprüft
    assert at.session_state["w_gb_iter"] == best and at.session_state["w_model"] == pl.MODEL_GB
    assert any("shuffle=False" in c.value for c in at.code)

    nav(7)
    _ok(at, "Schritt 8")
    assert any(h.value == "Vergleich mit 'pre_elo_prob_home'" for h in at.subheader)
    assert any("Log Loss" in m.label for m in at.metric)
    assert any(("schlägt" in m.value or "Kein klarer Unterschied" in m.value or "ist besser als das Modell" in m.value)
               for m in at.markdown)

    nav(8)
    at.radio(key="w_save_fmt").set_value(".skops").run()
    at.text_input(key="w_save_name").input("nba_gb").run()
    _button(at, "Speichern").click().run()
    _ok(at, "Speichern")
    saved = json.loads((tmp_path / "models" / "nba_gb" / "preprocessing.json").read_text(encoding="utf-8"))
    assert saved["training"]["model"] == pl.MODEL_GB and "log_loss" in saved["metrics"]
    _button(at, "Laden").click().run()
    _ok(at, "Laden (skops)")

    nav(9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "neu.csv")).run()
    _button(at, "Vorhersage berechnen").click().run()
    _ok(at, "Schritt 10")
    assert at.session_state["pred_result"]["ev"]["n"] == 60


def test_config_with_the_new_settings():
    cfg = {"widgets": {"w_gb_iter": 0, "w_gb_lr": 5, "w_gb_leaves": 1, "w_gb_depth": -3, "w_gb_leaf": "20",
                       "w_split_mode": "time", "w_time_col": "date", "w_base_col": "pre_elo_prob_home",
                       "w_model": "gb"}}
    out, skipped = pl.sanitize_config(cfg)
    w = out["widgets"]
    assert (w["w_gb_iter"], w["w_gb_lr"], w["w_gb_leaves"], w["w_gb_depth"], w["w_gb_leaf"]) == (1, 1.0, 2, 0, 20)
    assert (w["w_split_mode"], w["w_time_col"], w["w_base_col"], w["w_model"]) == ("time", "date",
                                                                                   "pre_elo_prob_home", "gb")
    assert not skipped
