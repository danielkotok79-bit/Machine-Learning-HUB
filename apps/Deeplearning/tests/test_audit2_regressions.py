"""
Regressionstests zum zweiten Audit (nach den Erweiterungen: neue Spalten, Daten speichern, Fazit).
Jeder Test sichert einen gefundenen und behobenen Fehler.

    pytest -q tests/test_audit2_regressions.py
"""
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline as pl  # noqa: E402
import plots  # noqa: E402

RNG = np.random.default_rng(11)
N = 300


def base(n=N):
    return pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.uniform(1, 5, n), "c": RNG.choice(list("xyz"), n)})


def tiny_model(prep):
    cfg = pl.ModelConfig(epochs=1, layers=[{"units": 4, "activation": "relu", "dropout": 0}])
    return pl.build_model(prep.X_train.shape[1], pl.n_outputs(prep.recipe), cfg, prep.recipe["task"])


def formula(expr, df=None):
    df = base(5) if df is None else df
    return pl.apply_new_columns(df, [{"name": "neu", "kind": pl.N_FORMULA, "definition": expr}])[0]["neu"]


# --- Formeln: nichts darf das Programm blockieren oder abstürzen lassen ---------------------------------------
@pytest.mark.parametrize("expr, message", [
    ("9 ** 9 ** 9 + a", "Exponent"),                 # riesige Ganzzahl → blockierte minutenlang
    ("a * 9 ** 999999", "Exponent"),
    ("a ** 1e3", "Exponent"),
    ("'x' * 10**8 == c", "Text in Anführungszeichen"),  # riesiger Text im Speicher
    ("-" * 990 + "a", "verschachtelt"),              # RecursionError statt Meldung
    ("<".join(["a"] * 450), "verschachtelt"),
    ("+".join(["a"] * 300), "zu lang|verschachtelt"),
    ("a + " * 300 + "a", "zu lang"),
    ("round(a, 1000)", "round"),
    ("round(a, b)", "round"),
    ("abs(a, b)", "nicht erlaubt"),
    ("(9 ** 99) ** 99 * a", "berechnen"),            # Überlauf → Meldung
])
def test_formula_limits_fail_fast(expr, message):
    t = time.perf_counter()
    with pytest.raises(ValueError, match=message):
        formula(expr)
    assert time.perf_counter() - t < 2


@pytest.mark.parametrize("expr, expected", [
    ("a ** (1/2)", lambda d: np.sqrt(d["a"])),
    ("b ** 2 + round(b, 2)", lambda d: d["b"] ** 2 + np.round(d["b"], 2)),
    ("b ** -2", lambda d: d["b"] ** -2.0),
    ("b ** a", lambda d: d["b"] ** d["a"]),
    ("+".join(["b"] * 60), lambda d: d["b"] * 60),   # lange Summen bleiben erlaubt
])
def test_normal_formulas_still_work(expr, expected):
    d = base(8).assign(a=lambda x: x["a"].abs())
    np.testing.assert_allclose(formula(expr, d).to_numpy(), expected(d).to_numpy(), rtol=1e-12)


def test_mapping_without_any_match_explains_why():
    df = pd.DataFrame({"status": ["Fully Paid", "Charged Off"] * 5})
    with pytest.raises(ValueError, match="Kein Wert von 'status' passt"):
        pl.apply_new_columns(df, [{"name": "ok", "kind": pl.N_MAP, "source": "status",
                                   "definition": "fully paid=1; charged off=0"}])
    # neue Daten (fit=False): keine Fehlermeldung, nur leere Werte
    out, *_ = pl.apply_new_columns(df, [{"name": "ok", "kind": pl.N_MAP, "source": "status",
                                         "definition": "fully paid=1"}], fit=False)
    assert out["ok"].isna().all()


# --- importierte Konfiguration --------------------------------------------------------------------------------
def test_config_must_be_an_object():
    for bad in ([1, 2], "text", 5, None):
        with pytest.raises(ValueError, match="JSON-Objekt"):
            pl.sanitize_config(bad)


def test_config_with_wrong_types_is_cleaned():
    cfg = json.loads("""{
        "step2": "kaputt",
        "missing": [{"Spalte": "a", "Strategie": "Median"}, {"Spalte": ["x"]}, 7,
                    {"Spalte": "b", "Strategie": "gibt es nicht"}],
        "layers": [{"Neuronen": 16, "Aktivierung": "relu", "Dropout": 0.2},
                   {"Neuronen": 1e9, "Aktivierung": "relu"}, {"Neuronen": Infinity, "Aktivierung": "relu"},
                   {"Neuronen": "abc"}, {"Neuronen": 8, "Aktivierung": "eval"}, {"Neuronen": 8, "Dropout": NaN}],
        "new_columns": [{"name": "n1", "kind": "Formel", "definition": "a + b"}, {"name": "", "kind": "Formel"},
                        {"name": "n2", "kind": "unbekannt"}, "x"],
        "widgets": {"w_features": 5, "w_viz_cols": ["a", {"x": 1}, 3], "w_epochs": "Infinity", "w_lr": NaN,
                    "w_label": ["a"], "w_test_size": "0,25", "w_batch": 1e300}
    }""")
    out, skipped = pl.sanitize_config(cfg)
    assert "step2" not in out and "step2" in skipped
    assert [m["Spalte"] for m in out["missing"]] == ["a"]
    assert out["layers"] == [{"Neuronen": 16, "Aktivierung": "relu", "Dropout": 0.2}]
    assert [s["name"] for s in out["new_columns"]] == ["n1"]
    w = out["widgets"]
    assert w["w_viz_cols"] == ["a", "3"] and w["w_test_size"] == 0.25 and w["w_batch"] == 65_536
    assert not {"w_features", "w_epochs", "w_lr", "w_label"} & set(w)
    assert {"w_features", "w_epochs", "w_lr", "w_label"} <= set(skipped)


def test_config_limits_size_of_lists_and_texts():
    cfg = {"layers": [{"Neuronen": 4, "Aktivierung": "relu"}] * 500,
           "new_columns": [{"name": f"n{i}", "kind": pl.N_FORMULA, "definition": "a" * 5000} for i in range(500)]}
    out, _ = pl.sanitize_config(cfg)
    assert len(out["layers"]) == pl.MAX_LAYERS
    assert len(out["new_columns"]) == pl.MAX_NEW_COLUMNS
    assert all(len(s["definition"]) <= pl.MAX_FORMULA_LEN for s in out["new_columns"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "inf", "-Infinity"])
def test_non_finite_numbers_in_config_are_ignored(value):
    assert pl.coerce_setting("w_epochs", value) is None
    assert pl.coerce_setting("w_threshold", value) is None


def test_layer_table_rejects_absurd_sizes():
    for units in (0, pl.MAX_UNITS + 1, float("inf")):
        with pytest.raises(ValueError):
            pl.layers_from_table(pd.DataFrame([{"Neuronen": units, "Aktivierung": "relu", "Dropout": 0.0}]))
    with pytest.raises(ValueError, match="Höchstens"):
        pl.layers_from_table(pd.DataFrame([{"Neuronen": 2, "Aktivierung": "relu", "Dropout": 0.0}] * 51))


# --- Label mit ±inf ---------------------------------------------------------------------------------------
def test_label_with_inf_rows_are_removed():
    df = base().assign(y=RNG.normal(10, 2, N))
    df.loc[[3, 50, 120], "y"] = [np.inf, -np.inf, np.inf]
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_REG)
    assert prep.recipe["label_inf_removed"] is True
    assert len(prep.y_train) + len(prep.y_test) == N - 3
    assert np.isfinite(prep.y_train).all() and np.isfinite(prep.y_test).all()
    assert any("inf" in line for line in prep.log)
    res, rep, ev = pl.predict_new(tiny_model(prep), prep.recipe, df)
    assert len(res) == N and ev["n"] == N - 3 and np.isfinite(ev["MAE"])


# --- Daten speichern (Schritt 5) -----------------------------------------------------------------------------
def test_feature_named_split_keeps_its_values():
    df = pd.DataFrame({"split": RNG.normal(size=N), "y": RNG.choice(["a", "b"], N)})
    prep = pl.prepare_ml_data(df, "y", ["split"], pl.T_BIN)
    ml = pl.ml_ready_frame(df, prep)
    assert pd.api.types.is_numeric_dtype(ml["split"])
    assert set(ml["split_train_test"]) == {"train", "test"}


def test_excel_export_neutralises_formulas():
    df = pd.DataFrame({"name": ['=HYPERLINK("http://x","klick")', "+1+1", "@SUM(A1)", "-2+3", "normal", None],
                       "wert": [1.5, -2.0, 3.0, -4.25, 5.0, 6.0]})
    de = pd.read_csv(io.BytesIO(pl.to_csv_bytes(df, german=True)), sep=";", decimal=",", encoding="utf-8-sig")
    assert de["name"].tolist()[:5] == ["'=HYPERLINK(\"http://x\",\"klick\")", "'+1+1", "'@SUM(A1)", "'-2+3", "normal"]
    assert de["wert"].tolist() == df["wert"].tolist()  # Zahlen (auch negative) unverändert
    intl = pd.read_csv(io.BytesIO(pl.to_csv_bytes(df)))
    assert intl["name"].iloc[0] == df["name"].iloc[0]  # internationales Format: Daten unverändert
    assert df["name"].iloc[0].startswith("=")  # Original nicht verändert


# --- Diagramme mit leeren Daten -----------------------------------------------------------------------------
def test_charts_with_empty_columns_do_not_crash():
    n = 120
    df = pd.DataFrame({"leer": np.nan, "x": RNG.normal(size=n), "k": RNG.choice(list("ab"), n),
                       "text_leer": pd.Series([None] * n, dtype=object)})
    assert plots.fig_hist(df, "leer", "k") is not None  # seaborn: TypeError bei hue ohne Daten
    for draw in (lambda: plots.fig_hist(df, "leer"), lambda: plots.fig_box(df, "k", "leer"),
                 lambda: plots.fig_scatter(df, "x", "leer", "k"), lambda: plots.fig_count(df, "text_leer", "k"),
                 lambda: plots.fig_corr_heatmap(plots.corr_matrix(df, ["x", "leer"])),
                 lambda: plots.fig_pairplot(df, ["x", "leer"], "k", 100)):
        draw()  # darf nicht abstürzen
    plt.close("all")


# --- Modell-ZIP ------------------------------------------------------------------------------------------------
def test_zip_bomb_is_rejected(tmp_path, monkeypatch):
    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    folder = pl.save_bundle(tiny_model(prep), prep.recipe, tmp_path / "m")
    data = pl.bundle_zip_bytes(folder)
    pl.load_bundle_zip(data)  # normal: lädt
    monkeypatch.setattr(pl, "MAX_BUNDLE_BYTES", 1000)  # entpackt größer als erlaubt
    with pytest.raises(ValueError, match="ZIP zu groß"):
        pl.load_bundle_zip(data)


def test_zip_without_recipe_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.keras", b"x")
    with pytest.raises(ValueError, match="preprocessing.json"):
        pl.load_bundle_zip(buf.getvalue())


# --- Fazit (Schritt 10) -----------------------------------------------------------------------------------------
def test_fazit_share_uses_the_prediction_column():
    """P(…) ist auf 4 Stellen gerundet: 0.50004 steht dort als 0.5, vorhergesagt ist aber die positive Klasse."""
    df = base().assign(y=RNG.choice(["ja", "nein"], N))
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN, positive_class="ja")
    res, rep, ev = pl.predict_new(tiny_model(prep), prep.recipe, df)
    pos = prep.recipe["classes"][1]
    res[f"P({pos})"] = 0.5
    res["Vorhersage"] = pos
    fz = pl.assess_prediction(prep.recipe, res, rep, ev, threshold=0.5)
    assert any(f"Vorhergesagt '{pos}': 100.0%" in m for m in fz["meaning"])
