"""
Audit 5 (Gradient Boosting, zeitlicher Split, Vergleich mit Elo): je ein Test pro gefundenem Problem – siehe AUDIT.md,
Abschnitt „Audyt 5“. Auf dem Code vor dem Audit (Commit 3f349fd) schlagen alle Tests fehl.

    pytest -q tests/test_audit5_regressions.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
import plots  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from synthetic import lending_club_like, sports_like  # noqa: E402

APP = str(ROOT / "app.py")
RNG = np.random.default_rng(55)
pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning",
    "ignore:The inplace keyword in DataFrame.reset_index is deprecated:DeprecationWarning:streamlit")


def _viz(raw):
    viz, fitted, _, lineage = pl.apply_column_actions(raw, pl.actions_from_table(pl.column_profile(raw)))
    return viz


@pytest.fixture(scope="module")
def games():
    return _viz(sports_like(300, seed=8))


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


# --- A1: Vergleichsspalte gehört zur anderen Klasse -----------------------------------------------------------------
def test_comparison_for_the_other_class_is_flagged(games):
    """Positive Klasse '0' (Auswärtssieg), Vergleich = Siegchance der Heimmannschaft (Klasse '1'): vorher kein Hinweis,
    und Schritt 8 meldete „Modell schlägt Elo“ (NBA-Daten: Log Loss 0.62 gegen 0.94) – ein wertloser Vergleich."""
    feats = ["pre_diff_elo", "pre_home_rest_days"]
    wrong = pl.prepare_ml_data(games, "home_win", feats, pl.T_BIN, positive_class="0", split_mode=pl.SPLIT_TIME,
                               time_column="date", baseline_column="pre_elo_prob_home")
    assert wrong.recipe["baseline_auc_train"] < 0.5
    assert any("schlechter vorher als Zufall" in w and "'1' als positive Klasse" in w for w in wrong.warnings)
    right = pl.prepare_ml_data(games, "home_win", feats, pl.T_BIN, positive_class="1", split_mode=pl.SPLIT_TIME,
                               time_column="date", baseline_column="pre_elo_prob_home")
    assert right.recipe["baseline_auc_train"] > 0.5 and not right.warnings


def test_app_proposes_elo_only_for_the_matching_class(tmp_path, monkeypatch):
    """Schritt 4 schlug pre_elo_prob_home auch bei positiver Klasse '0' vor; Schritt 8 warnt jetzt zusätzlich."""
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    sports_like(300, seed=8).to_csv(tmp_path / "g.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "g.csv")).run()
    _button(at, "Daten laden").click().run()
    nav = lambda i: at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()  # noqa: E731
    nav(1)
    _button(at, "Übernehmen").click().run()
    at.session_state["w_pos"] = "0"  # Auswärtssieg als positive Klasse, bevor Schritt 4 zum ersten Mal läuft
    nav(3)
    _ok(at, "Schritt 4")
    assert at.session_state["w_pos"] == "0" and at.session_state["w_base_col"] == "(keine)"
    at.selectbox(key="w_base_col").set_value("pre_elo_prob_home").run()  # trotzdem gewählt → Warnung
    _button(at, "Daten für ML vorbereiten").click().run()
    _ok(at, "Schritt 4 vorbereitet")
    assert any("schlechter vorher als Zufall" in w.value for w in at.warning)
    nav(5)
    at.radio(key="w_model").set_value("logreg").run()
    _button(at, "Training starten").click().run()
    nav(7)
    _ok(at, "Schritt 8")
    assert any("Vergleich wertlos" in w.value for w in at.warning)


# --- A2: zeitlicher Split ohne beide Klassen / mit winzigem Training --------------------------------------------
def test_time_split_without_both_classes_in_training():
    """Früher Zeitraum nur Klasse 0: Gradient Boosting „trainierte“ ein Modell, das immer 0 sagt – ohne Hinweis."""
    x = RNG.normal(size=(60, 2))
    df = pd.DataFrame({"a": x[:, 0], "b": x[:, 1], "t": np.arange(60),
                       "y": np.r_[np.zeros(40, int), RNG.integers(0, 2, 20)]})
    with pytest.raises(ValueError, match="kommt nur die Klasse '0' vor"):
        pl.prepare_ml_data(df, "y", ["a", "b"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="t", test_size=0.33)
    multi = df.assign(y=np.r_[RNG.choice(["a", "b"], 45), ["c"] * 15])  # Klasse c erst ganz am Ende
    prep = pl.prepare_ml_data(multi, "y", ["a", "b"], pl.T_MULTI, split_mode=pl.SPLIT_TIME, time_column="t",
                              test_size=0.25)
    assert any("Klassen ohne eine einzige Trainingszeile" in w and "'c'" in w for w in prep.warnings)


def test_time_split_with_too_few_training_rows():
    """2 Zeilen 2020, 118 Zeilen 2021, test_size 0.3: Die ganze Saison 2021 kam in den Test, trainiert wurde auf
    2 Zeilen – nur mit dem Hinweis „Testanteil 98 %“."""
    df = pd.DataFrame({"a": RNG.normal(size=120), "y": RNG.integers(0, 2, 120), "season": [2020] * 2 + [2021] * 118})
    with pytest.raises(ValueError, match="Nur 2 Trainingszeilen"):
        pl.prepare_ml_data(df, "y", ["a"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="season", test_size=0.3)


# --- A4: Erkennung des Formats -------------------------------------------------------------------------------------
def test_sports_format_without_post_columns_and_no_false_positive():
    """Im Tool lassen sich post_*/inplay_* abwählen – solche Exporte wurden nicht erkannt (Datum → Jahr, keine
    Vorschläge). Umgekehrt galten fremde Daten mit pre_tax_income und post_code als Sportdaten."""
    g = sports_like(60, seed=2)
    only_pre = g[[c for c in g.columns if not c.startswith(("post_", "inplay_"))]]
    lay = pl.match_data_layout(only_pre.columns)
    assert lay and lay["label"] == "home_win" and lay["leaky"] == [] and lay["time"] == "date"
    assert pl.column_profile(only_pre).set_index("Spalte").at["date", "Aktion"] == pl.A_KEEP
    assert pl.match_data_layout(["pre_tax_income", "post_code", "churn"]) is None
    tennis = pl.match_data_layout(["match_date_est", "p1_win", "pre_p1_elo", "pre_elo_prob_p1"])
    assert tennis["label"] == "p1_win" and tennis["time"] == "match_date_est" and tennis["baseline"] == "pre_elo_prob_p1"


# --- C1: erzeugter Code, Vergleichsspalte in Schritt 4 entfernt ----------------------------------------------------
def test_generated_code_keeps_a_dropped_comparison_column(tmp_path):
    """Vergleichsspalte mit 50 % Lücken + „Spalte entfernen“: Der Code von Schritt 8 endete mit KeyError."""
    raw = sports_like(300, seed=8)
    raw.loc[raw.index[::2], "pre_elo_prob_home"] = np.nan
    feats = ["pre_diff_elo", "pre_elo_prob_home", "pre_home_rest_days"]
    prep, ns = parity.run_both(tmp_path, raw, {"date": (pl.A_KEEP, "", "")}, "home_win", pl.T_BIN,
                               positive_class="1", features=feats,
                               missing={"pre_elo_prob_home": {"strategy": pl.M_DROP_COL}},
                               split_mode=pl.SPLIT_TIME, time_column="date", baseline_column="pre_elo_prob_home")
    assert "pre_elo_prob_home" in prep.recipe["dropped_columns"]
    parity.assert_same(prep, ns)
    cfg = pl.ModelConfig(model=pl.MODEL_GB, gb_iter=20, curve=False)
    model, _, _ = pl.train_model(cfg, prep)
    exec(cg.model_code(cfg, pl.T_BIN), ns)  # noqa: S102 – erzeugter Code
    exec(cg.eval_code(pl.T_BIN, 0.5, False, cfg, "pre_elo_prob_home"), ns)  # noqa: S102
    plt.close("all")
    c = pl.compare_probabilities(prep.y_test, pl.predict_scores(model, prep.X_test, pl.T_BIN, prep.recipe),
                                 prep.base_test, 0.5)
    ok = ns["ok"]
    assert log_loss(ns["y_test"][ok], ns["p_base"][ok]) == pytest.approx(c["rows"][1]["log_loss"], rel=1e-9)


# --- C2: geänderte Vergleichsspalte → Hinweis „neu vorbereiten“ ----------------------------------------------------
def test_changed_comparison_column_marks_step4_as_stale(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    sports_like(300, seed=3).to_csv(tmp_path / "g.csv", index=False)  # 200 Tage: season konstant
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "g.csv")).run()
    _button(at, "Daten laden").click().run()
    nav = lambda i: at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()  # noqa: E731
    nav(1)
    _button(at, "Übernehmen").click().run()
    nav(3)
    _button(at, "Daten für ML vorbereiten").click().run()
    stale = "seit dem Vorbereiten geändert"
    assert not any(stale in i.value for i in at.info)
    at.selectbox(key="w_base_col").set_value("(keine)").run()
    assert any(stale in i.value for i in at.info)


# --- C3–C5: Meldungen und Prüfungen ---------------------------------------------------------------------------------
def test_time_boundary_on_the_oldest_value_explains_the_limit():
    """Die Meldung sagte „die jüngsten Zeilen haben alle denselben Zeitwert wie die ältesten“ – hier falsch; jetzt mit
    dem größten möglichen test_size, und der funktioniert."""
    df = pd.DataFrame({"a": RNG.normal(size=120), "y": RNG.integers(0, 2, 120), "season": [2020] * 100 + [2021] * 20})
    with pytest.raises(ValueError, match=r"ältesten Zeitwert \(2020\).*höchstens 0\.16"):
        pl.prepare_ml_data(df, "y", ["a"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="season", test_size=0.5)
    prep = pl.prepare_ml_data(df, "y", ["a"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="season",
                              test_size=0.16)
    assert prep.recipe["split"]["test_from"] == "2021" and len(prep.y_test) == 20
    with pytest.raises(ValueError, match="Alle Zeilen haben denselben Zeitwert"):
        pl.prepare_ml_data(df.assign(season=2020), "y", ["a"], pl.T_BIN, split_mode=pl.SPLIT_TIME,
                           time_column="season")
    few = df.assign(season=[2020] * 117 + [2021] * 3)  # nur 2,5 % jünger: kleiner als der Regler (ab 0.05) erlaubt
    with pytest.raises(ValueError, match=r"Nur 2\.5% der Zeilen liegen nach dem ältesten Zeitwert \(2020\)"):
        pl.prepare_ml_data(few, "y", ["a"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="season", test_size=0.2)


def test_comparison_column_checks(games):
    g = games.copy()
    g.loc[g.index[len(g) // 2:], "pre_elo_prob_home"] = np.nan  # die jüngste Hälfte (= Test) ohne Wert
    with pytest.raises(ValueError, match="keinen einzigen Wert"):
        pl.prepare_ml_data(g, "home_win", ["pre_diff_elo"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="date",
                           baseline_column="pre_elo_prob_home")
    with pytest.raises(ValueError, match="nicht das Label"):
        pl.prepare_ml_data(games, "home_win", ["pre_diff_elo"], pl.T_BIN, baseline_column="home_win")


@pytest.mark.parametrize("mode", [pl.SPLIT_RANDOM, pl.SPLIT_TIME])
@pytest.mark.parametrize("size", [0.0, 1.0])
def test_test_size_out_of_range(games, mode, size):
    """test_size 0 (nur über pipeline.py möglich): zeitlicher Split → IndexError statt einer Meldung."""
    with pytest.raises(ValueError, match="test_size muss zwischen 0 und 1"):
        pl.prepare_ml_data(games, "home_win", ["pre_diff_elo"], pl.T_BIN, split_mode=mode, time_column="date",
                           test_size=size)


# --- C6: GridSearchCV mit TimeSeriesSplit -------------------------------------------------------------------------
def test_grid_search_message_for_time_ordered_folds():
    n = 300
    y = np.r_[np.zeros(120, int), (RNG.random(n - 120) < 0.5).astype(int)]
    df = pd.DataFrame({"a": RNG.normal(size=n) + y, "b": RNG.normal(size=n), "t": np.arange(n), "y": y})
    prep = pl.prepare_ml_data(df, "y", ["a", "b"], pl.T_BIN, split_mode=pl.SPLIT_TIME, time_column="t")
    with pytest.warns(Warning):  # FitFailedWarning / nicht endliche Bewertungen – gewollt
        with pytest.raises(ValueError, match="TimeSeriesSplit"):
            pl.train_model(pl.ModelConfig(model=pl.MODEL_SVM, grid_search=True, cv=5, curve=False), prep)


# --- P1, P2: breite Daten (Sportdaten-Tool) in Schritt 4 ----------------------------------------------------------
def test_correlation_help_draws_only_the_strongest():
    """160 Spalten → ein 56 Zoll hohes Bild, das st.pyplot bei jedem Klick in Schritt 4 neu zeichnete (Sekunden)."""
    corr = pd.Series(np.linspace(-0.8, 0.9, 160), index=[f"c{i}" for i in range(160)]).sort_values()
    fig = plots.fig_corr_target(corr, "ziel")
    ax = fig.axes[0]
    assert len(ax.patches) == 30 and fig.get_size_inches()[1] < 13
    assert "die 30 stärksten von 160" in ax.get_title("left") + ax.get_title()  # Stil: Titel links
    shown = {t.get_text() for t in ax.get_yticklabels()}
    assert "c159" in shown and "c0" in shown and "c80" not in shown  # stärkste positive und negative
    assert len(plots.fig_corr_target(corr.head(12), "ziel").axes[0].patches) == 12  # wenige Spalten: alle
    plt.close("all")


@pytest.mark.parametrize("dataset", ["lending", "sports"])
def test_missing_suggestions_in_one_pass_are_identical(dataset, monkeypatch):
    """Schritt 4 rechnete die Vorschläge Spalte für Spalte (ATP: 67 × 49 Korrelationen, ca. 20 s beim ersten Aufruf).
    suggest_missing_all prüft die Kandidaten einmal und rechnet alle Korrelationen in einem Schritt – gleiche
    Vorschläge."""
    if dataset == "lending":
        df = lending_club_like(1500, seed=9).drop(columns=["emp_title", "title", "issue_d", "earliest_cr_line",
                                                           "address"])
    else:
        df = _viz(sports_like(400, seed=4))
        for c in ("pre_home_rest_days", "pre_away_l10_net", "lineup_home_out_n"):
            df.loc[RNG.choice(len(df), 40, replace=False), c] = np.nan
    feats = [c for c in df.columns if c not in ("loan_status", "home_win")]
    cols = [c for c in feats if df[c].isna().any()]
    one_by_one = {c: pl.suggest_missing(df, c, feats) for c in cols}
    calls = []
    original = pd.Series.corr
    monkeypatch.setattr(pd.Series, "corr", lambda self, *a, **k: calls.append(1) or original(self, *a, **k))
    assert pl.suggest_missing_all(df, cols, feats) == one_by_one
    assert not calls  # keine Korrelation mehr Spalte für Spalte
    assert any(s == pl.M_GROUP for s, _ in one_by_one.values()) or dataset == "sports"
