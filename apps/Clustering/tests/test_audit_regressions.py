"""
Regressionstests zu den Fehlern aus den Code-Audits des Grundgerüsts (Supervised-Version) – übertragen auf die
Clustering-App: Schritt 1–3 sind gleich geblieben, Schritt 4 und 10 rechnen jetzt ohne Label und ohne Split.

    pytest -q tests/test_audit_regressions.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline as pl  # noqa: E402

RNG = np.random.default_rng(7)
N = 300


def base(n=N):
    return pd.DataFrame({"a": RNG.normal(size=n), "b": RNG.integers(0, 5, n).astype(float),
                         "c": RNG.choice(list("xyz"), n)})


def tiny_kmeans(prep, k=3):
    """Kleines K-Means (2 Starts, ohne Kurve) – schnell für Tests."""
    return pl.train_kmeans(pl.ClusterConfig(n_clusters=k, n_init=2, curve=False), prep)[0]


# --- Zahlen & Datum ----------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("$1,200", 1200.0), ("1,234,567", 1234567.0), ("1.234,5", 1234.5), ("1,234.5", 1234.5),
    ("13,5 %", 13.5), ("1.234.567", 1234567.0), (" 36 months", 36.0), ("-2.5 kg", -2.5),
    ("True", 1.0), ("false", 0.0), ("kein Wert", np.nan),
])
def test_number_extraction(text, expected):
    got = pl.parse_numbers(pd.Series([text])).iloc[0]
    assert (np.isnan(got) and np.isnan(expected)) or got == pytest.approx(expected)


def test_date_as_integer_yyyymmdd():
    df = pd.DataFrame({"d": [20141013, 20150101, 20160505], "m": [20141013, 20150101, 20160505]})
    out, _, _, _ = pl.apply_column_actions(df, [{"column": "d", "action": pl.A_YEAR},
                                                {"column": "m", "action": pl.A_MONTH}])
    assert out["d"].tolist() == [2014.0, 2015.0, 2016.0]
    assert out["m"].tolist() == [10.0, 1.0, 5.0]
    # bereits Jahreszahlen (z. B. in neuen Daten) bleiben unverändert
    out2, _, _, _ = pl.apply_column_actions(pd.DataFrame({"d": [2014, 2015]}), [{"column": "d", "action": pl.A_YEAR}])
    assert out2["d"].tolist() == [2014.0, 2015.0]


# --- Vergleichsgruppe (statt Label) ---------------------------------------------------------------
@pytest.mark.parametrize("values, expected", [
    ([1.0, 0.0, 1.0], ["1", "0", "1"]), (["yes ", "yes", " no"], ["yes", "yes", "no"]), ([True, False, True],
                                                                                          ["True", "False", "True"]),
])
def test_group_values_are_text_without_spaces_and_dot_zero(values, expected):
    assert pl.group_values(pd.Series(values)).tolist() == expected


def test_group_with_too_many_values_is_ignored_with_warning():
    df = base().assign(g=[f"k{i}" for i in range(N)])
    prep = pl.prepare_cluster_data(df, ["a", "b"], group_col="g")
    assert prep.groups is None and prep.recipe["group_col"] is None and prep.recipe["group_requested"] == "g"
    assert any("Vergleichsgruppe 'g'" in w for w in prep.warnings)


def test_group_and_name_are_never_features():
    df = base().assign(g=RNG.choice(["p", "q"], N), name=[f"n{i}" for i in range(N)])
    prep = pl.prepare_cluster_data(df, ["a", "g", "name", "c"], group_col="g", name_col="name")
    assert prep.recipe["features"] == ["a", "c"] and prep.names.tolist()[:2] == ["n0", "n1"]
    assert any("nicht als Feature" in w for w in prep.warnings)


# --- fehlende / kaputte Werte ---------------------------------------------------------------------
def test_inf_values_are_treated_as_missing():
    df = base()
    df.loc[df.index[:5], "a"] = np.inf
    prep = pl.prepare_cluster_data(df, ["a", "b"])
    assert np.isfinite(prep.X).all() and prep.recipe["inf_columns"] == ["a"]
    X, _, rep = pl.prepare_new_data(df.head(3).copy(), prep.recipe)
    assert np.isfinite(X).all() and rep["imputed"].get("a") == 3


def test_all_empty_column_is_dropped_with_warning():
    df = base().assign(e=np.nan)
    prep = pl.prepare_cluster_data(df, ["a", "e"], missing={"e": {"strategy": pl.M_MODE}})
    assert "e" not in prep.recipe["features"] and prep.recipe["empty_columns"] == ["e"]
    assert any("ohne Werte" in w for w in prep.warnings)


def test_bool_feature_with_gap_in_new_data():
    df = base().assign(f=RNG.random(N) > 0.5)
    prep = pl.prepare_cluster_data(df, ["a", "f"])
    new = pd.read_csv(io.StringIO("a,f\n0.1,True\n0.2,\n0.3,False\n"))  # object: True/NaN/False
    X, _, rep = pl.prepare_new_data(new, prep.recipe)
    i = prep.recipe["feature_columns"].index("f")
    raw = pl.invert_scaler(X, prep.recipe["scaler"])
    assert rep["invalid_numeric"] == {} and rep["imputed"] == {"f": 1}
    assert raw[0, i] == pytest.approx(1.0) and raw[2, i] == pytest.approx(0.0)


def test_whitespace_in_categories_of_new_data():
    df = base()
    prep = pl.prepare_cluster_data(df, ["a", "c"])
    new = df.head(3).copy()
    new["c"] = new["c"] + " "
    _, _, rep = pl.prepare_new_data(new, prep.recipe)
    assert rep["unknown"] == {}


def test_too_few_rows_after_dropping_gaps():
    df = base(30)
    df.loc[df.index[:25], "a"] = np.nan
    with pytest.raises(ValueError, match="zu wenig für ein Clustering"):
        pl.prepare_cluster_data(df, ["a", "b"], missing={"a": {"strategy": pl.M_DROP_ROWS}})


# --- Dateien ------------------------------------------------------------------------------------
def test_csv_semicolon_and_decimal_comma_are_detected():
    raw = base().to_csv(index=False, sep=";", decimal=",").encode()
    df, _ = pl.read_csv(raw)
    assert df.shape == (N, 3) and pl.is_numeric(df["a"])
    assert df.attrs["csv"]["sep"] == ";" and df.attrs["csv"]["decimal"] == ","


def test_quoted_windows_style_path(tmp_path):
    p = tmp_path / "daten mit leerzeichen.csv"
    base().to_csv(p, index=False)
    df, _ = pl.read_csv(f'  "{p}" ')
    assert df.shape == (N, 3)


def test_column_names_are_stripped_and_flagged():
    df, _ = pl.read_csv(b" a ,b\n1,2\n")
    assert list(df.columns) == ["a", "b"] and df.attrs["csv"]["strip_columns"] is True


def test_index_column_of_college_data_is_suggested_for_dropping_but_can_be_kept():
    """College_Data hat eine erste Spalte ohne Namen (im Kurs index_col=0) → 'Unnamed: 0'."""
    df, _ = pl.read_csv(b",Private,Apps\nA College,Yes,10\nB College,No,20\n")
    prof = pl.column_profile(df)
    assert prof.loc[prof["Spalte"] == "Unnamed: 0", "Aktion"].iloc[0] == pl.A_DROP
    out, _, _, lin = pl.apply_column_actions(df, [{"column": "Unnamed: 0", "action": pl.A_KEEP,
                                                   "new_name": "College"}])
    assert out["College"].tolist() == ["A College", "B College"] and lin["College"] == "Unnamed: 0"


def test_empty_new_file_gives_clear_error():
    prep = pl.prepare_cluster_data(base(), ["a", "c"])
    with pytest.raises(ValueError, match="keine Datenzeilen"):
        pl.assign_new(tiny_kmeans(prep), pl.finalize_recipe(prep.recipe), base().head(0))


def test_zip_cannot_point_outside_the_bundle(tmp_path):
    prep = pl.prepare_cluster_data(base(), ["a", "c"])
    model = tiny_kmeans(prep)
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "m")
    decoy = tmp_path / "fremd" / "model.skops"  # fremdes Modell außerhalb des ZIP (7 Eingänge)
    decoy.parent.mkdir()
    import skops.io as sio
    from sklearn.cluster import KMeans

    sio.dump(KMeans(n_clusters=2, n_init=1).fit(RNG.normal(size=(20, 7))), decoy)
    recipe = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    recipe["model_file"] = str(decoy)  # absoluter Pfad in der JSON → darf nicht geladen werden
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("preprocessing.json", json.dumps(recipe))
        zf.write(folder / "model.skops", arcname="model.skops")
    loaded, r = pl.load_bundle_zip(buf.getvalue())  # nur der Dateiname zählt → lädt model.skops aus dem ZIP
    assert loaded.n_features_in_ == len(r["feature_columns"])
    np.testing.assert_array_equal(loaded.cluster_centers_, model.cluster_centers_)


# --- Konfiguration ------------------------------------------------------------------------------
@pytest.mark.parametrize("key, value, expected", [
    ("w_k", "6", 6), ("w_k", 6.0, 6), ("w_k", 1, 2), ("w_k", 10**9, pl.MAX_K), ("w_k", "viele", None),
    ("w_n_init", "0", 1), ("w_k_max", 99, 30), ("w_curve", "false", False), ("w_drop_first", "ja", True),
    ("w_group", "Private", "Private"), ("w_init", "random", "random"), ("w_seed", "1,0", 1),
])
def test_config_values_are_coerced(key, value, expected):
    got = pl.coerce_setting(key, value)
    assert got == expected and type(got) is type(expected)


def test_supervised_settings_in_a_config_are_ignored():
    cfg, skipped = pl.sanitize_config({"widgets": {"w_label": "loan_status", "w_rf_trees": 600, "w_features": ["a"],
                                                   "w_scaler": "MinMaxScaler", "w_k": 4}})
    assert cfg["widgets"] == {"w_features": ["a"], "w_scaler": "MinMaxScaler", "w_k": 4} and skipped == []


# --- Füllwerte & Kategorien aus allen Zeilen (Clustering: kein Split) --------------------------------
def _gappy_frame(n=N):
    rng = np.random.default_rng(11)
    g = rng.integers(1, 6, n).astype(float)
    df = pd.DataFrame({"a": rng.lognormal(3, 1, n), "g": g, "m": g * 2 + rng.normal(size=n),
                       "c": rng.choice(["x", "y", "z"], n, p=[0.6, 0.3, 0.1])})
    df.loc[df.index[::7], "a"] = np.nan
    df.loc[df.index[::5], "m"] = np.nan
    df.loc[df.index[::9], "c"] = np.nan
    return df


def test_fill_values_and_scaler_come_from_all_rows():
    df = _gappy_frame()
    missing = {"a": {"strategy": pl.M_MEDIAN}, "m": {"strategy": pl.M_GROUP, "param": "g"},
               "c": {"strategy": pl.M_MODE}}
    prep = pl.prepare_cluster_data(df, ["a", "g", "m", "c"], missing=missing)
    r = prep.recipe
    assert r["fit_on"] == "all" and r["n_rows"] == {"train": N} and len(prep.X) == N
    assert r["fills"]["a"]["value"] == pytest.approx(df["a"].median())
    assert dict(map(tuple, r["fills"]["m"]["group_means"])) == pytest.approx(df.groupby("g")["m"].mean().to_dict())
    assert r["fills"]["m"]["overall"] == pytest.approx(df["m"].mean())
    assert r["fills"]["c"]["value"] == df["c"].mode()[0]
    assert r["categories"]["c"] == ["x", "y", "z"]
    np.testing.assert_allclose(prep.X.mean(axis=0), 0, atol=1e-12)  # StandardScaler auf allen Zeilen
    # dieselben Zeilen über das Rezept (wie neue Daten) → dieselbe Matrix
    X_again, _, _ = pl.prepare_new_data(df, r)
    np.testing.assert_array_equal(X_again, prep.X)
