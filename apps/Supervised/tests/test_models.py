"""
Tests der Modelle (Schritt 6–9): die fünf Modelle der Rangliste für Klassifikation und Regression, Validierungskurve
und Einordnung (Schritt 7), Modell-Details (Schritt 8), Speichern/Laden als .skops oder .joblib (Schritt 9) und der
angezeigte scikit-learn-Code – er rechnet dasselbe wie die App.

    pytest -q tests/test_models.py
"""
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import i18n  # noqa: E402
import pipeline as pl  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from synthetic import house_like, iris_like, lending_club_like  # noqa: E402

APP = str(ROOT / "app.py")
RNG = np.random.default_rng(5)
# veraltete scikit-learn-Aufrufe (z. B. SVC(probability=True), LogisticRegression(penalty=…)) wären Fehler.
# Ausnahme (letzter Filter = Vorrang): Streamlits eigenes st.line_chart ruft intern df.reset_index(inplace=True) auf –
# in pandas 3.1 veraltet (PDEP-8). Das ist Streamlit-Code, nicht die App; gilt nur für Warnungen aus dem Modul streamlit.
pytestmark = pytest.mark.filterwarnings(
    "error::FutureWarning", "error::DeprecationWarning",
    "ignore:The inplace keyword in DataFrame.reset_index is deprecated:DeprecationWarning:streamlit")

CONFIGS = {  # Name → Einstellungen aus Schritt 6 (klein und schnell)
    "forest": pl.ModelConfig(model=pl.MODEL_TREES, n_estimators=30),
    "tree": pl.ModelConfig(model=pl.MODEL_TREES, tree_variant=pl.TREE_SINGLE, max_depth=6),
    "svm": pl.ModelConfig(model=pl.MODEL_SVM),
    "logreg": pl.ModelConfig(model=pl.MODEL_LOGREG),
    "knn": pl.ModelConfig(model=pl.MODEL_KNN, n_neighbors=7),
    "knn_distance": pl.ModelConfig(model=pl.MODEL_KNN, weights="distance", p=1, validation="split"),
    "linreg": pl.ModelConfig(model=pl.MODEL_LINREG),
}


def _prep(raw, label, task, actions=None, **kw):
    prof = pl.column_profile(raw)
    for col, (act, param, new) in (actions or {}).items():
        prof.loc[prof.index[prof["Spalte"] == col][0], ["Aktion", "Parameter", "Neuer Name"]] = [act, param, new]
    viz, fitted, _, lin = pl.apply_column_actions(raw, pl.actions_from_table(prof))
    feats = [c for c in viz.columns if c != label]
    return pl.prepare_ml_data(viz, label, feats, task, column_actions=fitted, lineage=lin, **kw)


@pytest.fixture(scope="module")
def data():
    # LendingClub-ähnlich mit zip_code aus der Adresse (wie im Kurs) – dort steckt das stärkste Signal
    lc = lending_club_like(900, seed=21).drop(columns=["emp_title", "title", "issue_d", "earliest_cr_line"])
    house = house_like(700)
    iris = iris_like()
    return {
        pl.T_BIN: (lc, _prep(lc, "loan_status", pl.T_BIN, {"address": (pl.A_LAST, "5", "zip_code")},
                             positive_class="Fully Paid",
                             missing={"mort_acc": {"strategy": pl.M_MEDIAN}, "emp_length": {"strategy": pl.M_MODE}})),
        pl.T_REG: (house, _prep(house, "price", pl.T_REG, scale_label=True,
                                missing={"bathrooms": {"strategy": pl.M_MEDIAN}})),
        pl.T_MULTI: (iris, _prep(iris, "species", pl.T_MULTI)),
    }


# --- Rangliste ------------------------------------------------------------------------------------------------------
def test_ranking_order_and_descriptions():
    assert pl.MODELS == [pl.MODEL_TREES, pl.MODEL_SVM, pl.MODEL_LOGREG, pl.MODEL_KNN, pl.MODEL_LINREG]
    i18n.set_lang("pl")
    try:
        assert [pl.model_name(m) for m in pl.MODELS] == ["Decision Trees + Random Forests", "Support Vector Machines",
                                                         "Logistic Regression", "K-Nearest Neighbors",
                                                         "Linear Regression"]
        assert i18n.tx(pl.MODEL_WHY[pl.MODEL_TREES]).startswith("Najlepszy z tej listy.")
        assert i18n.tx(pl.MODEL_BEST[pl.MODEL_SVM]) == "Klasyfikacja kierunku (buy/sell)"
        assert i18n.tx(pl.MODEL_BEST[pl.MODEL_LINREG]) == "Tylko jako bardzo prosty punkt odniesienia"
    finally:
        i18n.set_lang("de")
    assert pl.MODEL_TASKS[pl.MODEL_LOGREG] == (pl.T_BIN, pl.T_MULTI)
    assert pl.MODEL_TASKS[pl.MODEL_LINREG] == (pl.T_REG,)


# --- jedes Modell für jeden passenden Problemtyp -------------------------------------------------------------------
CASES = [(name, task) for name, cfg in CONFIGS.items() for task in (pl.T_BIN, pl.T_REG, pl.T_MULTI)
         if task in pl.MODEL_TASKS[cfg.model]]


@pytest.mark.parametrize("name, task", CASES)
def test_train_curve_details_save_load(data, tmp_path, name, task):
    raw, prep = data[task]
    cfg = CONFIGS[name]
    model, hist, info = pl.train_model(cfg, prep)
    r = prep.recipe
    s = pl.predict_scores(model, prep.X_test, task, r)
    if task == pl.T_BIN:
        assert s.shape == (len(prep.y_test),) and ((s >= 0) & (s <= 1)).all()
    elif task == pl.T_MULTI:
        assert s.shape == (len(prep.y_test), 3) and np.allclose(s.sum(axis=1), 1)
    else:
        assert s.mean() > 10_000  # zurückgerechnet in Originaleinheiten (Preise)
    assert np.isfinite([info["train_error"], info["test_error"], info["baseline_error"]]).all()
    assert info["test_error"] < info["baseline_error"]  # besser als die naive Vorhersage
    # Schritt 7: Kurve + Einordnung
    curve = hist["curve"]
    assert len(curve["values"]) == len(curve["train"]) == len(curve["val"]) >= 2
    assert curve["chosen"] in curve["values"] and curve["chosen_x"] in curve["x"]
    assert curve["best_index"] is None or 0 <= curve["best_index"] < len(curve["values"])
    a = pl.curve_assessment(curve, info, cfg, task)
    assert a["notes"] and a["verdict"] in ("ok", "overfit", "underfit", "nosignal", "memorize")
    # Schritt 8: Modell-Details
    det = pl.model_details(model, r)
    if cfg.model == pl.MODEL_TREES:
        assert det["kind"] == "importance" and det["table"]["Wichtigkeit"].sum() == pytest.approx(1.0)
        assert set(det["table"]["Feature"]) == set(r["features"])
    elif cfg.model in (pl.MODEL_LOGREG, pl.MODEL_LINREG):
        assert det["kind"] == "coef" and len(det["table"]) == len(r["feature_columns"])
    else:
        assert det["facts"]
    # Schritt 9 + 10: Speichern/Laden in beiden Formaten → neue Daten = gleiche Vorhersage
    for fmt in (".skops", ".joblib"):
        folder = pl.save_bundle(model, pl.finalize_recipe(r, cfg, info), tmp_path / fmt[1:], fmt)
        m2, r2 = pl.load_bundle(folder)
        assert r2["engine"] == "scikit-learn" and r2["training"]["model"] == cfg.model
        res, _, ev = pl.predict_new(m2, r2, raw.loc[prep.idx_test])
        if task == pl.T_BIN:
            assert np.allclose(res[f"P({r['classes'][1]})"], np.round(s, 4), atol=1e-4)
        elif task == pl.T_MULTI:
            assert (res["Vorhersage"].to_numpy() == np.asarray(r["classes"])[s.argmax(axis=1)]).all()
        else:
            assert np.allclose(res["Vorhersage"], s, rtol=1e-9)
        assert ev is not None and ev["n"] == len(prep.idx_test)


def test_unsuitable_models_are_refused(data):
    _, prep_bin = data[pl.T_BIN]
    _, prep_reg = data[pl.T_REG]
    with pytest.raises(ValueError, match="Lineare Regression sagt eine Zahl"):
        pl.train_model(CONFIGS["linreg"], prep_bin)
    with pytest.raises(ValueError, match="Logistische Regression sagt Klassen"):
        pl.train_model(CONFIGS["logreg"], prep_reg)
    with pytest.raises(ValueError, match="größer als die Anzahl"):
        pl.train_model(pl.ModelConfig(model=pl.MODEL_KNN, n_neighbors=10_000), prep_bin)


def test_random_forest_curve_reuses_the_final_model(data):
    """Validierung mit Testdaten: der Wald der Kurve (warm_start) ist schon das fertige Modell – identisch zu fit()."""
    _, prep = data[pl.T_BIN]
    cfg = replace(CONFIGS["forest"], class_weight=True)
    model, hist, info = pl.train_model(cfg, prep)
    direct = pl.make_model(cfg, pl.T_BIN).fit(prep.X_train, prep.y_train.astype(int))
    assert np.array_equal(model.predict_proba(prep.X_test), direct.predict_proba(prep.X_test))
    assert hist["curve"]["values"] == pl.forest_steps(30) and info["class_weight"]


def test_svm_grid_search_like_the_course(data):
    _, prep = data[pl.T_MULTI]
    cfg = replace(CONFIGS["svm"], grid_search=True, cv=3)
    model, hist, info = pl.train_model(cfg, prep)
    grid = hist["grid"]
    assert len(grid["rows"]) == len(pl.GRID_C) * len(pl.GRID_GAMMA) and grid["cv"] == 3
    best = grid["best"]
    svc = pl.base_estimator(model)
    assert svc.C == best["C"] and svc.gamma == best["gamma"] and info["best_params"] == best
    assert grid["best_score"] == max(row["val"] for row in grid["rows"])
    lin, hist_lin, _ = pl.train_model(replace(cfg, kernel="linear"), prep)
    assert len(hist_lin["grid"]["rows"]) == len(pl.GRID_C)  # linearer Kernel: nur C


@pytest.mark.parametrize("name", ["svm", "logreg", "knn", "tree", "forest", "linreg"])
def test_curve_figures_have_readable_axes(data, name):
    """Browser-Test: bei C (logarithmische Achse) stand '$\\mathdefault{10^{-2}}$' unter der Achse – wegen
    text.parse_math=False (Audit 3). Jetzt Zahlen wie 0.01 … 100; alle Diagramme der Schritte 7 und 8 zeichnen."""
    import matplotlib.pyplot as plt
    import plots

    task = pl.T_REG if name == "linreg" else pl.T_BIN
    _, prep = data[task]
    cfg = replace(CONFIGS[name], svm_c=0.37, logreg_c=0.37, grid_search=name == "svm", cv=3)
    model, hist, _ = pl.train_model(cfg, prep)
    curve = hist["curve"]
    j = curve["best_index"]
    figs = [plots.fig_curve(curve, "MAE" if task == pl.T_REG else "Fehlerrate",
                            pl.curve_value_text(curve, curve["values"][j]) if j is not None else None)]
    if hist.get("grid"):
        figs.append(plots.fig_grid(hist["grid"]))
    det = pl.model_details(model, prep.recipe)
    if det["table"] is not None:
        col = "Wichtigkeit" if det["kind"] == "importance" else "Koeffizient (skaliert)"
        figs.append(plots.fig_bars(det["table"]["Feature"], det["table"][col], "t", "x", signed=col != "Wichtigkeit"))
    if cfg.model == pl.MODEL_TREES and cfg.tree_variant == pl.TREE_SINGLE:
        figs.append(plots.fig_tree(pl.base_estimator(model), prep.recipe["feature_columns"], prep.recipe["classes"]))
    for fig in figs:
        fig.canvas.draw()
        for ax in fig.axes:
            labels = [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels()]
            assert not any("$" in t or "mathdefault" in t for t in labels), labels
    if curve["log_x"]:
        shown = [t.get_text() for t in figs[0].axes[0].get_xticklabels()]
        assert {"0.01", "1", "100"} <= set(shown), shown
        chosen = f"({curve['chosen_x']:g})"  # SVM mit GridSearchCV: das beste C der Suche statt 0.37
        assert chosen == ("(0.37)" if name != "svm" else f"({hist['grid']['best']['C']:g})")
        assert any(chosen in t.get_text() for t in figs[0].axes[0].texts)  # eingestellter Wert steht an der Linie
    plt.close("all")


def test_assessment_verdicts(data):
    _, prep = data[pl.T_BIN]
    _, tree_hist, tree_info = pl.train_model(replace(CONFIGS["tree"], max_depth=0), prep)
    a = pl.curve_assessment(tree_hist["curve"], tree_info, replace(CONFIGS["tree"], max_depth=0), pl.T_BIN)
    assert a["verdict"] == "overfit" and a["suggest"]["param"] == "max_depth"  # unbegrenzter Baum lernt auswendig
    _, rf_hist, rf_info = pl.train_model(CONFIGS["forest"], prep)
    assert pl.curve_assessment(rf_hist["curve"], rf_info, CONFIGS["forest"], pl.T_BIN)["verdict"] == "memorize"
    # Browser-Test: Wald mit min_samples_leaf=50 lernt nicht auswendig → keine Erklärung „Trainingsfehler ist normal“
    leafy = replace(CONFIGS["forest"], min_samples_leaf=50)
    _, h_leafy, i_leafy = pl.train_model(leafy, prep)
    a_leafy = pl.curve_assessment(h_leafy["curve"], i_leafy, leafy, pl.T_BIN)
    assert a_leafy["train"] > 0.5 * a_leafy["val"] and a_leafy["verdict"] != "memorize"
    # Zufalls-Label: kein Signal → nicht besser als die häufigste Klasse
    noise = pd.DataFrame({"a": RNG.normal(size=600), "b": RNG.normal(size=600), "y": RNG.choice(["u", "v"], 600)})
    prep_noise = pl.prepare_ml_data(noise, "y", ["a", "b"], pl.T_BIN)
    _, h, i = pl.train_model(CONFIGS["logreg"], prep_noise)
    assert pl.curve_assessment(h["curve"], i, CONFIGS["logreg"], pl.T_BIN)["verdict"] == "nosignal"
    # ohne Kurve: Training gegen Test
    _, h2, i2 = pl.train_model(replace(CONFIGS["logreg"], curve=False), prep)
    assert "curve" not in h2 and pl.curve_assessment(None, i2, CONFIGS["logreg"], pl.T_BIN)["notes"]


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
@pytest.mark.parametrize("task", [pl.T_BIN, pl.T_REG])
def test_fast_elbow_equals_sklearn_predict(weights, task, ties):
    """Elbow-Methode: EINE Nachbarsuche für alle K liefert dieselben Vorhersagen wie predict() je K (Schleife im
    Kurs). Einzige Ausnahme: gleich weit entfernte Nachbarn an der K-ten Stelle (Ties, z. B. doppelte Zeilen) –
    welche davon zählen, legt auch scikit-learn nicht fest."""
    rng = np.random.default_rng(11)
    X = rng.normal(size=(300, 4))
    Xq = np.vstack([X[:40] + 1e-3 * rng.normal(size=(40, 4)), rng.normal(size=(40, 4))])
    if ties:
        X, Xq = X.round(1), Xq.round(1)
        X[50:60] = X[0]  # Duplikate → Abstand 0
        Xq[0] = X[0]
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int) if task == pl.T_BIN else X[:, 1] * 3 + rng.normal(size=300)
    values = list(range(1, 16))
    dist, ind = NearestNeighbors(n_neighbors=16).fit(X).kneighbors(Xq)
    pred = pl._knn_predict_all(dist, ind, y, values, weights, task)
    cls = KNeighborsClassifier if task == pl.T_BIN else KNeighborsRegressor
    differ = 0
    for k in values:
        expected = cls(n_neighbors=k, weights=weights).fit(X, y).predict(Xq)
        diff = ~np.isclose(pred[k], expected, rtol=1e-9, atol=1e-12)
        tie = dist[:, k - 1] == dist[:, k]  # K-ter und (K+1)-ter Nachbar gleich weit entfernt
        assert not (diff & ~tie).any(), (k, np.where(diff & ~tie))
        differ += int(diff.sum())
    assert differ == 0 or ties


@pytest.mark.parametrize("scaler", ["MinMaxScaler", "StandardScaler", "keiner"])
@pytest.mark.parametrize("scale_label", [True, False])
def test_linear_coefficients_in_original_units(scaler, scale_label):
    """Koeffizienten in Originaleinheiten (wie lm.coef_ im Kurs): Rohwerte · Koeffizient + Achsenabschnitt = Vorhersage."""
    raw = house_like(400)
    prep = _prep(raw, "price", pl.T_REG, scaler=scaler, scale_label=scale_label,
                 missing={"bathrooms": {"strategy": pl.M_MEDIAN}})
    model, _, _ = pl.train_model(replace(CONFIGS["linreg"], curve=False), prep)
    w, b = pl.unscaled_coefficients(model.coef_, model.intercept_, prep.recipe)
    rows = raw.loc[prep.idx_test]
    viz, *_ = pl.apply_column_actions(rows, prep.recipe["column_actions"], fit=False)
    X_raw, _ = pl.transform_features(viz, prep.recipe)
    np.testing.assert_allclose(X_raw.to_numpy(dtype="float64") @ w + b,
                               pl.predict_scores(model, prep.X_test, pl.T_REG, prep.recipe), rtol=1e-4)
    det = pl.model_details(model, prep.recipe)
    assert det["table"].set_index("Feature").loc["sqft_living", "Koeffizient"] == pytest.approx(
        w[prep.recipe["feature_columns"].index("sqft_living")])


# --- Speichern & Laden: Sicherheit -----------------------------------------------------------------------------------
def _helper(x):  # eigene Funktion → für skops ein unbekannter (nicht vertrauenswürdiger) Typ
    return x


def test_skops_refuses_unknown_types(tmp_path, data):
    import skops.io as sio
    from sklearn.preprocessing import FunctionTransformer

    _, prep = data[pl.T_BIN]
    folder = tmp_path / "m"
    pl.save_bundle(pl.train_model(replace(CONFIGS["forest"], curve=False), prep)[0], prep.recipe, folder)
    sio.dump(FunctionTransformer(_helper).fit(prep.X_train), folder / "model.skops")  # fremder Code im Modell
    with pytest.raises(ValueError, match="nicht als sicher"):
        pl.load_bundle(folder)


def test_joblib_from_a_zip_needs_permission(tmp_path, data):
    _, prep = data[pl.T_BIN]
    model = pl.train_model(replace(CONFIGS["logreg"], curve=False), prep)[0]
    data_zip = pl.bundle_zip_bytes(pl.save_bundle(model, prep.recipe, tmp_path / "j", ".joblib"))
    assert pl.zip_needs_pickle(data_zip)
    with pytest.raises(ValueError, match="pickle"):
        pl.load_bundle_zip(data_zip)
    m2, _ = pl.load_bundle_zip(data_zip, allow_pickle=True)
    assert np.allclose(m2.predict_proba(prep.X_test), model.predict_proba(prep.X_test))
    skops_zip = pl.bundle_zip_bytes(pl.save_bundle(model, prep.recipe, tmp_path / "s", ".skops"))
    assert not pl.zip_needs_pickle(skops_zip) and pl.load_bundle_zip(skops_zip)[0] is not None


def test_models_of_the_deep_learning_version_are_not_offered(tmp_path):
    keras = tmp_path / "models" / "dl_modell"
    keras.mkdir(parents=True)
    (keras / "preprocessing.json").write_text(json.dumps({"label": "y", "model_file": "model.keras"}), encoding="utf-8")
    (keras / "model.keras").write_bytes(b"keras")
    assert pl.list_bundles(tmp_path / "models") == []
    with pytest.raises(ValueError, match="Deep-Learning-Version"):
        pl.load_bundle(keras)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(keras / "preprocessing.json", arcname="preprocessing.json")
        zf.write(keras / "model.keras", arcname="model.keras")
    with pytest.raises(ValueError, match="Deep-Learning-Version"):
        pl.load_bundle_zip(buf.getvalue())


# --- angezeigter Code = App ------------------------------------------------------------------------------------------
PARITY = [("forest", pl.T_BIN), ("tree", pl.T_MULTI), ("svm", pl.T_BIN), ("svm", pl.T_REG), ("logreg", pl.T_MULTI),
          ("knn", pl.T_BIN), ("knn", pl.T_REG), ("linreg", pl.T_REG)]


def _parity_data(tmp_path, task):
    if task == pl.T_REG:
        raw = pd.DataFrame({"a": RNG.normal(size=400), "b": RNG.uniform(1, 5, 400), "c": RNG.choice(list("xyz"), 400)})
        raw["y"] = 40 + 7 * raw["a"] + 3 * raw["b"] + RNG.normal(size=400)
        return parity.run_both(tmp_path, raw, {}, "y", task, scale_label=True)
    raw = pd.DataFrame({"a": RNG.normal(size=400), "b": RNG.uniform(1, 5, 400), "c": RNG.choice(list("xyz"), 400)})
    k = 2 if task == pl.T_BIN else 3
    raw["y"] = np.asarray(["k0", "k1", "k2"])[(raw["a"] + raw["b"] / 2 + RNG.normal(size=400) > 1.2).astype(int)
                                                 + (k == 3) * (raw["a"] > 1).astype(int)]
    return parity.run_both(tmp_path, raw, {}, "y", task)


@pytest.mark.parametrize("name, task", PARITY)
def test_model_and_curve_code_compute_the_same_as_the_app(tmp_path, name, task):
    prep, ns = _parity_data(tmp_path, task)
    parity.assert_same(prep, ns)
    cfg = CONFIGS[name] if name != "knn" else replace(CONFIGS[name], validation="test")
    model, hist, _ = pl.train_model(cfg, prep)
    calib = pl.calibration_cv(prep.y_train) if cfg.model == pl.MODEL_SVM and task != pl.T_REG else 5
    exec(cg.model_code(cfg, task, calib), ns)  # noqa: S102 – erzeugter Code
    if task == pl.T_REG:
        np.testing.assert_allclose(ns["model"].predict(ns["X_test"]), model.predict(prep.X_test), rtol=1e-9, atol=1e-12)
    else:
        p_code, p_app = ns["model"].predict_proba(ns["X_test"]), model.predict_proba(prep.X_test)
        np.testing.assert_allclose(p_code, p_app, rtol=1e-9, atol=1e-12)
    ns["plt"].close("all")
    exec(cg.curve_code(cfg, task, bool(prep.recipe.get("label_scaler"))), ns)  # noqa: S102
    ns["plt"].close("all")
    curve = hist["curve"]
    if cfg.model == pl.MODEL_TREES and cfg.tree_variant == pl.TREE_SINGLE:
        assert len(ns["error_val"]) == len(ns["values"])  # andere Tiefen als in der App, aber lauffähig
    else:
        assert list(ns["values"]) == [int(v) if float(v).is_integer() else v for v in curve["values"]]
        np.testing.assert_allclose(ns["error_val"], curve["val"], rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(ns["error_train"], curve["train"], rtol=1e-9, atol=1e-12)


def test_eval_and_save_code_run(tmp_path):
    prep, ns = _parity_data(tmp_path, pl.T_BIN)
    for name in ("forest", "svm", "logreg"):
        cfg = CONFIGS[name]
        exec(cg.model_code(cfg, pl.T_BIN), ns)  # noqa: S102
        exec(cg.eval_code(pl.T_BIN, 0.4, False, cfg), ns)  # noqa: S102
        for fmt in (".skops", ".joblib"):
            folder = str(tmp_path / f"m_{name}")
            exec(cg.save_code(folder, fmt), ns)  # noqa: S102
            np.testing.assert_allclose(ns["later_model"].predict_proba(ns["X_test"]),
                                       ns["model"].predict_proba(ns["X_test"]))


# --- Oberfläche --------------------------------------------------------------------------------------------------------
def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


def test_app_regression_models_elbow_and_best_value(tmp_path, monkeypatch):
    """Regression in der App: Lineare Regression (Lernkurve, Koeffizienten), KNN (Elbow-Methode, besten Wert
    übernehmen), SVM mit GridSearchCV, Logistische Regression gesperrt; Speichern als .skops und Schritt 10."""
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    house_like(500).to_csv(tmp_path / "haus.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "haus.csv")).run()
    _button(at, "Daten laden").click().run()
    nav = lambda i: at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()  # noqa: E731
    nav(1)
    _button(at, "Übernehmen").click().run()
    nav(3)
    at.selectbox(key="w_label").set_value("price").run()
    assert at.session_state["w_task"] == "regression"
    _button(at, "Daten für ML vorbereiten").click().run()
    _ok(at, "Schritt 4")

    nav(5)
    at.radio(key="w_model").set_value("logreg").run()
    assert _button(at, "Training starten").disabled and any("Logistische Regression" in w.value for w in at.warning)
    at.radio(key="w_model").set_value("linreg").run()
    _button(at, "Training starten").click().run()
    _ok(at, "Lineare Regression")
    nav(6)
    _ok(at, "Lernkurve")
    assert at.session_state["history"]["curve"]["param"] == "n_rows"
    nav(7)
    _ok(at, "Koeffizienten")
    assert any("ändert sich `price` um" in m.value for m in at.markdown)

    nav(5)
    at.radio(key="w_model").set_value("knn").run()
    at.number_input(key="w_knn_k").set_value(1).run()
    _button(at, "Training starten").click().run()
    _ok(at, "KNN")
    nav(6)
    _ok(at, "Elbow-Methode")
    curve = at.session_state["history"]["curve"]
    assert curve["values"][:3] == [1, 2, 3] and len(curve["values"]) == pl.KNN_MAX_K
    best = curve["values"][curve["best_index"]]
    _button(at, "Besten Wert in Schritt 6 übernehmen").click().run()
    assert at.session_state["w_knn_k"] == best and any("jetzt in Schritt 6 neu trainieren" in s.value
                                                       for s in at.success)

    nav(5)
    at.radio(key="w_model").set_value("svm").run()
    at.checkbox(key="w_svm_grid").check().run()
    at.number_input(key="w_svm_cv").set_value(3).run()
    assert "GridSearchCV(SVR(kernel='rbf', epsilon=0.1), param_grid, cv=3" in "\n".join(c.value for c in at.code)
    _button(at, "Training starten").click().run()
    _ok(at, "SVM + GridSearchCV")
    nav(6)
    _ok(at, "GridSearchCV in Schritt 7")
    assert any(h.value == "GridSearchCV" for h in at.subheader)
    nav(8)
    at.radio(key="w_save_fmt").set_value(".skops").run()
    at.text_input(key="w_save_name").input("svr").run()
    _button(at, "Speichern").click().run()
    _ok(at, "Speichern .skops")
    assert (tmp_path / "models" / "svr" / "model.skops").exists()
    _button(at, "Laden").click().run()
    nav(9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "haus.csv")).run()
    _button(at, "Vorhersage berechnen").click().run()
    _ok(at, "Schritt 10")
    assert at.session_state["pred_result"]["ev"]["MAE"] > 0
