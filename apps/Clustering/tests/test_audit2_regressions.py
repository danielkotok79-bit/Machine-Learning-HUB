"""
Regressionstests zum zweiten Audit des Grundgerüsts (neue Spalten, Daten speichern, Fazit) – übertragen auf die
Clustering-App. Jeder Test sichert einen gefundenen und behobenen Fehler.

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


def tiny_kmeans(prep, k=3):
    """Kleines K-Means (2 Starts, ohne Kurve) – schnell für Tests."""
    return pl.train_kmeans(pl.ClusterConfig(n_clusters=k, n_init=2, curve=False), prep)[0]


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
        "layers": [{"Neuronen": 16, "Aktivierung": "relu", "Dropout": 0.2}],
        "new_columns": [{"name": "n1", "kind": "Formel", "definition": "a + b"}, {"name": "", "kind": "Formel"},
                        {"name": "n2", "kind": "unbekannt"}, "x"],
        "widgets": {"w_features": 5, "w_viz_cols": ["a", {"x": 1}, 3], "w_k": "Infinity", "w_n_init": NaN,
                    "w_group": ["a"], "w_k_max": "7,0", "w_max_iter": 1e300, "w_epochs": 25, "w_rf_trees": 600}
    }""")
    out, skipped = pl.sanitize_config(cfg)
    assert "step2" not in out and "step2" in skipped
    assert [m["Spalte"] for m in out["missing"]] == ["a"]
    assert "layers" not in out and "layers" not in skipped  # Deep-Learning-Version: still übergangen
    assert [s["name"] for s in out["new_columns"]] == ["n1"]
    w = out["widgets"]
    assert w["w_viz_cols"] == ["a", "3"] and w["w_k_max"] == 7 and w["w_max_iter"] == 100_000
    assert not {"w_features", "w_k", "w_n_init", "w_group", "w_epochs", "w_rf_trees"} & set(w)
    assert {"w_features", "w_k", "w_n_init", "w_group"} <= set(skipped)
    assert not {"w_epochs", "w_rf_trees"} & set(skipped)  # Einstellungen anderer Versionen ohne Meldung


def test_config_limits_size_of_lists_and_texts():
    cfg = {"new_columns": [{"name": f"n{i}", "kind": pl.N_FORMULA, "definition": "a" * 5000} for i in range(500)]}
    out, _ = pl.sanitize_config(cfg)
    assert len(out["new_columns"]) == pl.MAX_NEW_COLUMNS
    assert all(len(s["definition"]) <= pl.MAX_FORMULA_LEN for s in out["new_columns"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "inf", "-Infinity"])
def test_non_finite_numbers_in_config_are_ignored(value):
    assert pl.coerce_setting("w_k", value) is None
    assert pl.coerce_setting("w_n_init", value) is None


def test_model_settings_are_limited():
    """Absurde Modell-Einstellungen werden begrenzt oder klar abgelehnt (statt eines Absturzes in scikit-learn)."""
    km = pl.make_kmeans(pl.ClusterConfig(init="gibts nicht", algorithm="schnell"))
    assert km.init == "k-means++" and km.algorithm == "lloyd"
    prep = pl.prepare_cluster_data(base(40), ["a", "c"])
    with pytest.raises(ValueError, match="zu groß"):
        pl.train_kmeans(pl.ClusterConfig(n_clusters=40, curve=False), prep)
    with pytest.raises(ValueError, match="zwischen 2 und"):
        pl.train_kmeans(pl.ClusterConfig(n_clusters=1, curve=False), prep)
    assert pl.k_values(pl.ClusterConfig(n_clusters=3, k_max=30), 12) == list(range(1, 12))  # höchstens n − 1


# --- Daten speichern (Schritt 5) -----------------------------------------------------------------------------
def test_feature_named_cluster_keeps_its_values():
    """Heißt eine Spalte schon 'Cluster', bekommt die neue Spalte im Export einen freien Namen (Cluster_2)."""
    df = pd.DataFrame({"Cluster": RNG.normal(size=N), "b": RNG.normal(size=N)})
    prep = pl.prepare_cluster_data(df, ["Cluster", "b"])
    model, hist, _ = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, n_init=2, curve=False), prep)
    out = pl.clusters_frame(df, prep, model.labels_, hist["dist"])
    assert out["Cluster"].tolist() == df["Cluster"].tolist()
    assert set(out["Cluster_2"]) == {0, 1} and "Abstand" in out.columns


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
    prep = pl.prepare_cluster_data(base(), ["a", "c"])
    folder = pl.save_bundle(tiny_kmeans(prep), prep.recipe, tmp_path / "m")
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
def test_fazit_counts_far_rows_and_compares_with_training():
    """Zeilen weit weg von allen Clustern werden gezählt und gemeldet; die Verteilung wird mit dem Training verglichen."""
    prep = pl.prepare_cluster_data(base(), ["a", "b"])
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, n_init=2, curve=False), prep)
    recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=3), info)
    new = base(50)
    new.loc[new.index[:5], "a"] = 40.0  # weit außerhalb
    res, rep, cmp = pl.assign_new(model, recipe, new)
    fz = pl.assess_assignment(recipe, res, rep, cmp)
    assert cmp is None and rep["row_far"][:5].all() and fz["level"] != pl.LEVEL_OK
    assert any("weit weg von allen Clustern" in m for m in fz["meaning"])
    assert any(m.startswith("Verteilung auf die Cluster: ") for m in fz["meaning"])
    assert res.loc[:4, "Hinweis"].str.contains("weit weg von allen Clustern").all()
