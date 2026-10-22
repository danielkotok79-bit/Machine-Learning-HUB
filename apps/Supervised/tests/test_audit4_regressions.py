"""
Audit 4 (Version 9, scikit-learn): je ein Test für die Fehler, die im Audit der neuen Engine gefunden wurden –
siehe AUDIT.md, Abschnitt „Audyt 4“.

    pytest -q tests/test_audit4_regressions.py
"""
import copy
import io
import json
import sys
import zipfile
from dataclasses import replace
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
import i18n  # noqa: E402
import pipeline as pl  # noqa: E402
import plots  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from synthetic import house_like, iris_like, lending_club_like  # noqa: E402

APP = str(ROOT / "app.py")
RNG = np.random.default_rng(44)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _nav(at, i):
    at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()


def _ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


def _frame(task, n=300):
    raw = pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.uniform(1, 5, n), "c": RNG.choice(list("xyz"), n)})
    if task == pl.T_REG:
        raw["y"] = 40 + 7 * raw["a"] + 3 * raw["b"] + RNG.normal(size=n)
    else:
        k = 2 if task == pl.T_BIN else 3
        s = raw["a"] + raw["b"] / 2 + RNG.normal(size=n)
        raw["y"] = np.asarray(["k0", "k1", "k2"])[np.digitize(s, np.quantile(s, np.linspace(0, 1, k + 1)[1:-1]))]
    return raw


# --- B1 + B2: „Besten Wert übernehmen“ gehört zum TRAINIERTEN Modell -----------------------------------------------
def test_apply_best_value_goes_to_the_trained_model(tmp_path, monkeypatch):
    """B1: Logistische Regression trainiert, danach in Schritt 6 SVM ausgewählt → das beste C landete in der SVM.
    B2: SVM mit GridSearchCV → der übernommene Wert blieb wirkungslos (die Suche ersetzte C beim nächsten Training)."""
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    lending_club_like(1200, seed=4).drop(columns=["emp_title", "title", "issue_d", "earliest_cr_line"]).to_csv(
        tmp_path / "d.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "d.csv")).run()
    _button(at, "Daten laden").click().run()
    _nav(at, 1)
    _button(at, "Übernehmen").click().run()
    _nav(at, 3)
    at.selectbox(key="w_label").set_value("loan_status").run()
    _button(at, "Daten für ML vorbereiten").click().run()
    _ok(at, "Schritt 4")

    _nav(at, 5)
    at.radio(key="w_model").set_value("logreg").run()
    at.number_input(key="w_log_c").set_value(1000.0).run()
    _button(at, "Training starten").click().run()
    _ok(at, "Logistische Regression")
    hist = copy.deepcopy(at.session_state["history"])
    hist["curve"]["val"] = [0.01 if v == 0.01 else 0.4 for v in hist["curve"]["values"]]  # Kurve: C = 0.01 am besten
    at.session_state["history"] = hist
    at.radio(key="w_model").set_value("svm").run()  # Auswahl gewechselt, aber nicht trainiert
    _nav(at, 6)
    _button(at, "Besten Wert in Schritt 6 übernehmen").click().run()
    assert at.session_state["w_model"] == "logreg"  # Auswahl zurück beim trainierten Modell
    assert at.session_state["w_log_c"] == 0.01
    assert at.session_state["w_svm_c"] == 1.0  # die SVM bleibt unberührt

    _nav(at, 5)
    at.radio(key="w_model").set_value("svm").run()
    at.checkbox(key="w_svm_grid").check().run()
    at.number_input(key="w_svm_cv").set_value(3).run()
    _button(at, "Training starten").click().run()
    _ok(at, "SVM + GridSearchCV")
    gamma = at.session_state["train_info"]["best_params"]["gamma"]
    hist = copy.deepcopy(at.session_state["history"])
    curve = hist["curve"]
    other = next(v for v in curve["values"] if v != curve["chosen"])
    curve["val"] = [0.01 if v == other else 0.4 for v in curve["values"]]  # Kurve bevorzugt ein anderes C als die Suche
    at.session_state["history"] = hist
    _nav(at, 6)
    _button(at, "Besten Wert in Schritt 6 übernehmen").click().run()
    assert at.session_state["w_svm_c"] == other and at.session_state["w_svm_grid"] is False
    assert at.session_state["w_svm_gamma"] == f"{float(gamma):g}"
    assert any("GridSearchCV ist dafür ausgeschaltet" in s.value for s in at.success)
    _nav(at, 5)
    _button(at, "Training starten").click().run()
    _ok(at, "SVM mit übernommenem C")
    assert at.session_state["history"]["curve"]["chosen"] == other and "grid" not in at.session_state["history"]


# --- B3 + C1: angezeigter Code rechnet mit exakt denselben Werten -------------------------------------------------
@pytest.mark.parametrize("name, task, cfg", [
    ("svm", pl.T_BIN, pl.ModelConfig(model=pl.MODEL_SVM, svm_c=1.234567, gamma="0.1")),
    ("svr", pl.T_REG, pl.ModelConfig(model=pl.MODEL_SVM, svm_c=3.3333333, epsilon=0.123456789)),
    ("logreg", pl.T_MULTI, pl.ModelConfig(model=pl.MODEL_LOGREG, logreg_c=0.123456789)),
    ("knn50", pl.T_BIN, pl.ModelConfig(model=pl.MODEL_KNN, n_neighbors=50)),
    ("split", pl.T_BIN, pl.ModelConfig(model=pl.MODEL_LOGREG, validation="split", validation_split=0.15)),
])
def test_generated_code_uses_exact_values(tmp_path, name, task, cfg):
    """B3: f"{C:g}" rundete auf 6 Stellen (C = 1.234567 → 1.23457) – der Code trainierte ein anderes Modell.
    C1: der Kurven-Code ließ den eingestellten Wert außerhalb des Rasters weg (C = 1.234567, K = 50)."""
    prep, ns = parity.run_both(tmp_path, _frame(task), {}, "y", task, **({"scale_label": True} if task == pl.T_REG else {}))
    model, hist, _ = pl.train_model(cfg, prep)
    calib = pl.calibration_cv(prep.y_train) if cfg.model == pl.MODEL_SVM and task != pl.T_REG else 5
    exec(cg.model_code(cfg, task, calib), ns)  # noqa: S102 – erzeugter Code
    if task == pl.T_REG:
        np.testing.assert_allclose(ns["model"].predict(ns["X_test"]), model.predict(prep.X_test), rtol=1e-9, atol=1e-12)
    else:
        np.testing.assert_allclose(ns["model"].predict_proba(ns["X_test"]), model.predict_proba(prep.X_test),
                                   rtol=1e-9, atol=1e-12)
    exec(cg.curve_code(cfg, task, bool(prep.recipe.get("label_scaler"))), ns)  # noqa: S102
    plt.close("all")
    curve = hist["curve"]
    assert list(ns["values"]) == [int(v) if float(v).is_integer() else v for v in curve["values"]]
    np.testing.assert_allclose(ns["error_val"], curve["val"], rtol=1e-9, atol=1e-12)
    assert "1.23457" not in cg.model_code(cfg, task) and "0.123457" not in cg.model_code(cfg, task)
    assert "proba > 0.333333333)" in cg.eval_code(pl.T_BIN, 0.333333333)


@pytest.mark.parametrize("cfg", [pl.ModelConfig(model=pl.MODEL_KNN, n_neighbors=12, validation="split"),
                                 pl.ModelConfig(model=pl.MODEL_LINREG)])
def test_curve_code_runs_with_the_minimum_of_20_rows(tmp_path, cfg):
    """C1: Bei 20 Zeilen (Minimum in Schritt 4) bricht der Kurven-Code nicht mehr ab (K ≤ Zeilen, Lernkurve ≥ 2 Zeilen)
    und rechnet dieselben Werte wie die App."""
    task = pl.T_REG if cfg.model == pl.MODEL_LINREG else pl.T_BIN
    prep, ns = parity.run_both(tmp_path, _frame(task, 20), {}, "y", task)
    _, hist, _ = pl.train_model(cfg, prep)
    exec(cg.model_code(cfg, task), ns)  # noqa: S102
    exec(cg.curve_code(cfg, task), ns)  # noqa: S102
    plt.close("all")
    assert list(ns["values"]) == hist["curve"]["values"]
    np.testing.assert_allclose(ns["error_val"], hist["curve"]["val"], rtol=1e-9, atol=1e-12)


# --- B4: GridSearchCV mit wenigen Zeilen je Fold ------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore:One or more of the test scores are non-finite")  # gewollt im letzten Teil
def test_grid_search_folds_fit_small_data(tmp_path, monkeypatch):
    """B4: SVR + GridSearchCV mit 13 Trainingszeilen und cv=10 → Folds mit 1 Zeile, R² = NaN für alle Kombinationen,
    zufällige „beste“ Kombination und Absturz der Heatmap in Schritt 7."""
    df = pd.DataFrame({"a": RNG.normal(size=20), "b": RNG.normal(size=20)})
    df["y"] = df["a"] * 3 + df["b"]
    prep = pl.prepare_ml_data(df, "y", ["a", "b"], pl.T_REG)
    cfg = pl.ModelConfig(model=pl.MODEL_SVM, grid_search=True, cv=10)
    n = len(prep.y_train)
    assert pl.grid_cv(cfg, pl.T_REG, prep.y_train) == n // 2 == pl.grid_cv(cfg, pl.T_REG, recipe=prep.recipe)
    _, hist, info = pl.train_model(cfg, prep)
    assert np.isfinite(hist["grid"]["best_score"]) and not any(np.isnan(r["val"]) for r in hist["grid"]["rows"])
    plots.fig_grid(hist["grid"]).canvas.draw()
    plt.close("all")
    assert f"cv={n // 2}," in cg.model_code(cfg, pl.T_REG, grid_cv=pl.grid_cv(cfg, pl.T_REG, prep.y_train))
    # Klassen: höchstens so viele Folds wie Zeilen der seltensten Klasse – auch im exportierten Skript
    prep_c = pl.prepare_ml_data(iris_like().head(110), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    counts = np.bincount(prep_c.y_train.astype(int))
    assert pl.grid_cv(cfg, pl.T_MULTI, prep_c.y_train) == min(10, counts.min())
    assert pl.grid_cv(cfg, pl.T_MULTI, recipe=prep_c.recipe) == pl.grid_cv(cfg, pl.T_MULTI, prep_c.y_train)
    # bleibt trotzdem keine gültige Bewertung → verständliche Meldung statt zufälliger Kombination
    monkeypatch.setattr(pl, "grid_cv", lambda *a, **k: n)
    with pytest.raises(ValueError, match="keine gültige Bewertung"):
        pl.train_model(cfg, prep)


# --- C2: Rückfrage vor pickle gilt der Datei, die tatsächlich geladen wird ------------------------------------------
def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_pickle_question_matches_the_file_that_is_loaded(tmp_path):
    prep = pl.prepare_ml_data(iris_like(), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    model = pl.train_model(pl.ModelConfig(model=pl.MODEL_LOGREG, curve=False), prep)[0]
    sk = pl.save_bundle(model, prep.recipe, tmp_path / "s", ".skops")
    jb = pl.save_bundle(model, prep.recipe, tmp_path / "j", ".joblib")
    both = {"model.skops": (sk / "model.skops").read_bytes(), "model.joblib": (jb / "model.joblib").read_bytes()}
    names_joblib = _zip({"preprocessing.json": (jb / "preprocessing.json").read_text(), **both})
    assert pl.zip_needs_pickle(names_joblib)  # früher False – die App bot kein Häkchen an
    with pytest.raises(ValueError, match="pickle"):
        pl.load_bundle_zip(names_joblib)
    assert pl.load_bundle_zip(names_joblib, allow_pickle=True)[0] is not None
    names_skops = _zip({"preprocessing.json": (sk / "preprocessing.json").read_text(), **both})
    assert not pl.zip_needs_pickle(names_skops) and pl.load_bundle_zip(names_skops)[0] is not None
    # Rezept zeigt auf eine andere Datei → nur model.skops/model.joblib sind möglich
    odd = json.loads((sk / "preprocessing.json").read_text()) | {"model_file": "../../x.joblib"}
    data = _zip({"preprocessing.json": json.dumps(odd), "model.skops": both["model.skops"]})
    assert not pl.zip_needs_pickle(data) and pl.load_bundle_zip(data)[0] is not None
    assert not pl.zip_needs_pickle(b"kein zip")


# --- C3: Namen aus Dateien nicht als Markdown ---------------------------------------------------------------------
def test_unknown_model_names_are_not_shown_as_markdown():
    assert pl.model_name("![x](http://example.invalid/t.png)") == "?"
    assert pl.model_name(pl.MODEL_KNN) == "K-Nearest Neighbors"


def test_names_from_files_are_escaped_and_linreg_dummies_warn(tmp_path, monkeypatch):
    """C3: Name eines hochgeladenen ZIP und Label aus einem fremden Rezept kommen wörtlich in die Anzeige.
    C7: Lineare Regression ohne drop_first – Hinweis, dass die Dummy-Koeffizienten nicht eindeutig sind."""
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    house_like(400).assign(lage=RNG.choice(["nord", "sued", "mitte"], 400)).to_csv(tmp_path / "haus.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "haus.csv")).run()
    _button(at, "Daten laden").click().run()
    _nav(at, 1)
    _button(at, "Übernehmen").click().run()
    _nav(at, 3)
    at.selectbox(key="w_label").set_value("price").run()
    at.checkbox(key="w_drop_first").uncheck().run()
    _button(at, "Daten für ML vorbereiten").click().run()
    _ok(at, "Schritt 4")
    assert at.session_state["prep"].recipe["categorical_features"] and not at.session_state["prep"].recipe["drop_first"]
    _nav(at, 5)
    at.radio(key="w_model").set_value("linreg").run()
    _button(at, "Training starten").click().run()
    _nav(at, 7)
    _ok(at, "Schritt 8")
    assert any("Ohne drop_first" in w.value for w in at.warning)

    bad = "![x](http://example.invalid/t.png)"
    r = dict(at.session_state["prep"].recipe, label="price` ![y](http://example.invalid/u.png) `",
             training={"model": bad})
    at.session_state["bundle"] = {"model": at.session_state["model"], "recipe": r, "name": bad, "token": "x"}
    _nav(at, 8)
    _ok(at, "Schritt 9")
    shown = [m.value for m in at.markdown if "Geladen" in m.value]
    assert shown and "\\!\\[x\\]\\(http://example.invalid/t.png\\)" in shown[0]  # Name wörtlich, kein Bild
    assert "`price' ![y](http://example.invalid/u.png) '`" in shown[0]  # Label bleibt in EINEM Code-Span
    assert " · ? · " in shown[0]  # Modellname aus dem Rezept unbekannt → „?“
    _nav(at, 9)
    _ok(at, "Schritt 10")
    labels = [o for o in at.radio(key="w_pred_model").options if "example.invalid" in o]
    assert labels and all("\\!\\[x\\]" in o for o in labels)


# --- C4: erzeugter Lade-Code für .skops vertraut nur scikit-learn ---------------------------------------------------
def _helper(x):  # eigene Funktion → für skops ein fremder Typ
    return x


def test_generated_skops_code_trusts_only_sklearn(tmp_path):
    import skops.io as sio
    from sklearn.preprocessing import FunctionTransformer

    code = cg.save_code(str(tmp_path / "m"), ".skops")
    assert "startswith('sklearn.')" in code and "trusted=untrusted" not in code
    prep = pl.prepare_ml_data(iris_like(), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    for model, ok in [(pl.train_model(pl.ModelConfig(n_estimators=5, curve=False), prep)[0], True),
                      (FunctionTransformer(_helper).fit(prep.X_train), False)]:
        ns = {"model": model}
        if ok:
            exec(code, ns)  # noqa: S102
            assert type(ns["later_model"]) is type(model)
        else:
            with pytest.raises(Exception, match="Untrusted|untrusted"):
                exec(code, ns)  # noqa: S102
    assert sio.get_untrusted_types(file=tmp_path / "m" / "model.skops")


# --- C5: gescheitertes Speichern lässt das vorhandene Modell intakt -------------------------------------------------
def test_failed_save_keeps_the_previous_model(tmp_path, monkeypatch):
    import skops.io as sio

    prep = pl.prepare_ml_data(iris_like(), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    model = pl.train_model(pl.ModelConfig(model=pl.MODEL_LOGREG, curve=False), prep)[0]
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "m", ".joblib")

    def broken(*a, **k):
        raise OSError("Datenträger voll")

    monkeypatch.setattr(sio, "dump", broken)
    with pytest.raises(OSError):
        pl.save_bundle(model, prep.recipe, folder, ".skops")
    m2, r2 = pl.load_bundle(folder)  # das alte .joblib-Modell ist noch da und passt zum Rezept
    assert r2["model_file"] == "model.joblib" and np.allclose(m2.predict_proba(prep.X_test), model.predict_proba(prep.X_test))
    assert sorted(p.name for p in folder.iterdir()) == ["model.joblib", "preprocessing.json"]
    monkeypatch.undo()
    pl.save_bundle(model, prep.recipe, folder, ".skops")  # anderes Format → altes Modell weg, Rezept neu
    assert sorted(p.name for p in folder.iterdir()) == ["model.skops", "preprocessing.json"]


# --- C6: entpackte Größe einer .skops-Datei ist begrenzt ------------------------------------------------------------
def test_skops_unpacked_size_is_limited(tmp_path, monkeypatch):
    prep = pl.prepare_ml_data(iris_like(), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    folder = pl.save_bundle(pl.train_model(pl.ModelConfig(n_estimators=5, curve=False), prep)[0], prep.recipe,
                            tmp_path / "m")
    monkeypatch.setattr(pl, "MAX_BUNDLE_BYTES", 1000)
    with pytest.raises(ValueError, match="zu groß"):
        pl.load_bundle(folder)


# --- E3 (Browser-Test, Version 9): Achsentexte der Kurve in der aktuellen Sprache -----------------------------------
@pytest.mark.parametrize("code, expected", [("de", "Anzahl Bäume"), ("en", "Number of trees"), ("pl", "Liczba drzew")])
def test_curve_labels_follow_the_current_language(code, expected):
    prep = pl.prepare_ml_data(iris_like(), "species", ["sepal_length", "petal_length"], pl.T_MULTI)
    i18n.set_lang("de")  # trainiert auf Deutsch …
    _, hist, _ = pl.train_model(pl.ModelConfig(n_estimators=10), prep)
    try:
        i18n.set_lang(code)  # … angezeigt in der gewählten Sprache
        ax = plots.fig_curve(hist["curve"], "x").axes[0]
        assert ax.get_xlabel().startswith(expected)
        if code == "pl":
            assert ax.get_title(loc="left") == "x a liczba drzew (n_estimators)"
    finally:
        i18n.set_lang("de")
        plt.close("all")
