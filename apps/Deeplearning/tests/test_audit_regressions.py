"""
Regressionstests zu den Fehlern aus dem Code-Audit (jeder Test = ein gefundener und behobener Fehler).

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


def tiny_model(prep):
    cfg = pl.ModelConfig(epochs=1, layers=[{"units": 4, "activation": "relu", "dropout": 0}])
    return pl.build_model(prep.X_train.shape[1], pl.n_outputs(prep.recipe), cfg, prep.recipe["task"])


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


# --- Label --------------------------------------------------------------------------------------
def test_bool_label_and_new_data_variants():
    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN, positive_class="True")
    r = prep.recipe
    assert r["label_kind"] == "bool" and r["classes"] == ["False", "True"]
    new = pd.Series([True, "False", "true", "1", 0, None], dtype=object)
    assert pl.encode_label(new, r)[:5].tolist() == [1.0, 0.0, 1.0, 1.0, 0.0]


@pytest.mark.parametrize("values", [[0.5, 1.0, 1.5], [1, 2, 3]])
def test_numeric_multiclass_encoding_by_value(values):
    df = base().assign(y=RNG.choice(values, N))
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_MULTI)
    r = prep.recipe
    new = pd.Series([float(values[2]), str(values[0]), values[1]], dtype=object)
    assert pl.encode_label(new, r).tolist() == [2.0, 0.0, 1.0]


def test_text_label_whitespace_counts_as_same_class():
    df = base().assign(y=RNG.choice(["yes", "yes ", "no"], N))
    assert pl.validate_task(df["y"], pl.T_BIN) is None
    assert pl.suggest_task(df["y"]) == pl.T_BIN
    prep = pl.prepare_ml_data(df, "y", ["a"], pl.T_BIN, positive_class="yes")
    assert prep.recipe["classes"] == ["no", "yes"]


# --- fehlende / kaputte Werte ---------------------------------------------------------------------
def test_inf_values_are_treated_as_missing():
    df = base().assign(y=RNG.random(N) > 0.5)
    df.loc[df.index[:5], "a"] = np.inf
    prep = pl.prepare_ml_data(df, "y", ["a", "b"], pl.T_BIN)
    assert np.isfinite(prep.X_train).all() and prep.recipe["inf_columns"] == ["a"]
    new = df.head(3).copy()
    X, _, rep, _ = pl.prepare_new_data(new, prep.recipe)
    assert np.isfinite(X).all() and rep["imputed"].get("a") == 3


def test_all_empty_column_is_dropped_with_warning():
    df = base().assign(e=np.nan, y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "e"], pl.T_BIN, missing={"e": {"strategy": pl.M_MODE}})
    assert "e" not in prep.recipe["features"] and prep.recipe["empty_columns"] == ["e"]
    assert any("ohne Werte" in w for w in prep.warnings)


def test_bool_feature_with_gap_in_new_data():
    df = base().assign(f=RNG.random(N) > 0.5, y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "f"], pl.T_BIN)
    new = pd.read_csv(io.StringIO("a,f\n0.1,True\n0.2,\n0.3,False\n"))  # object: True/NaN/False
    X, _, rep, _ = pl.prepare_new_data(new, prep.recipe)
    i = prep.recipe["feature_columns"].index("f")
    assert rep["invalid_numeric"] == {} and rep["imputed"] == {"f": 1}
    assert X[0, i] == pytest.approx(1.0) and X[2, i] == pytest.approx(0.0)


def test_whitespace_in_categories_of_new_data():
    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    new = df.head(3).copy()
    new["c"] = new["c"] + " "
    _, _, rep, _ = pl.prepare_new_data(new, prep.recipe)
    assert rep["unknown"] == {}


# --- Split --------------------------------------------------------------------------------------
def test_stratify_with_many_classes_and_small_test_set():
    df = base().assign(y=[f"k{i % 40}" for i in range(N)])
    prep = pl.prepare_ml_data(df, "y", ["a"], pl.T_MULTI, test_size=0.05)
    assert prep.recipe["split"]["stratify_used"] is False
    assert any("Stratify nicht möglich" in w for w in prep.warnings)


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


def test_empty_new_file_gives_clear_error():
    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    with pytest.raises(ValueError, match="keine Datenzeilen"):
        pl.predict_new(tiny_model(prep), prep.recipe, df.head(0))


def test_zip_cannot_point_outside_the_bundle(tmp_path):
    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    folder = pl.save_bundle(tiny_model(prep), prep.recipe, tmp_path / "m")
    decoy = tmp_path / "fremd" / "model.keras"  # fremdes Modell außerhalb des ZIP (7 Eingänge)
    decoy.parent.mkdir()
    pl.build_model(7, 1, pl.ModelConfig(epochs=1), pl.T_BIN).save(decoy)
    recipe = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    recipe["model_file"] = str(decoy)  # absoluter Pfad in der JSON → darf nicht geladen werden
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("preprocessing.json", json.dumps(recipe))
        zf.write(folder / "model.keras", arcname="model.keras")
    model, r = pl.load_bundle_zip(buf.getvalue())  # nur der Dateiname zählt → lädt model.keras aus dem ZIP
    assert model.input_shape[1] == len(r["feature_columns"])


# --- Konfiguration ------------------------------------------------------------------------------
@pytest.mark.parametrize("key, value, expected", [
    ("w_epochs", "600", 600), ("w_epochs", 600.0, 600), ("w_epochs", -5, 1), ("w_test_size", 0.9, 0.5),
    ("w_lr", 1, 1.0), ("w_lr", "0,01", 0.01), ("w_stratify", "false", False), ("w_epochs", "viele", None),
    ("w_label", "price", "price"),
])
def test_config_values_are_coerced(key, value, expected):
    got = pl.coerce_setting(key, value)
    assert got == expected and type(got) is type(expected)


def test_recipes_from_version_1_still_work():
    """Modelle, die mit der ersten Version gespeichert wurden (ohne die neuen Rezept-Felder)."""
    import codegen as cg

    df = base().assign(y=RNG.random(N) > 0.5)
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN, positive_class="True")
    old = {k: v for k, v in prep.recipe.items()
           if k not in ("label_kind", "class_values", "inf_columns", "empty_columns")}
    old["split"] = {k: v for k, v in old["split"].items() if k != "stratify_used"}
    old["template_version"] = 1
    res, _, ev = pl.predict_new(tiny_model(prep), old, df.head(50))
    assert len(res) == 50 and ev is not None and ev["n"] == 50
    compile(cg.step4_code(old), "v1", "exec")


# --- Füllwerte & Kategorien nur aus den Trainingszeilen (kein Datenleck) --------------------------
def _leaky_frame(n=N):
    rng = np.random.default_rng(11)
    g = rng.integers(1, 6, n).astype(float)
    df = pd.DataFrame({"a": rng.lognormal(3, 1, n), "g": g, "m": g * 2 + rng.normal(size=n),
                       "c": rng.choice(["x", "y", "z"], n, p=[0.6, 0.3, 0.1]), "y": rng.random(n) > 0.5})
    df.loc[df.index[::7], "a"] = np.nan
    df.loc[df.index[::5], "m"] = np.nan
    df.loc[df.index[::9], "c"] = np.nan
    return df


def test_fill_values_come_from_training_rows_only():
    df = _leaky_frame()
    missing = {"a": {"strategy": pl.M_MEDIAN}, "m": {"strategy": pl.M_GROUP, "param": "g"},
               "c": {"strategy": pl.M_MODE}}
    prep = pl.prepare_ml_data(df, "y", ["a", "g", "m", "c"], pl.T_BIN, missing=missing)
    r, tr = prep.recipe, df.loc[prep.idx_train]
    assert r["fit_on"] == "train"
    assert r["fills"]["a"]["value"] == pytest.approx(tr["a"].median())
    assert tr["a"].median() != pytest.approx(df["a"].median())  # der Test unterscheidet wirklich
    assert dict(map(tuple, r["fills"]["m"]["group_means"])) == pytest.approx(tr.groupby("g")["m"].mean().to_dict())
    assert r["fills"]["m"]["overall"] == pytest.approx(tr["m"].mean())
    assert r["fills"]["c"]["value"] == tr["c"].mode()[0]
    assert r["fallback_fill"]["a"] == pytest.approx(tr["a"].median())
    assert any(line.startswith("Split zuerst") for line in prep.log)
    # Testzeilen mit Lücke bekommen den Trainingswert (unskaliert über das Rezept nachgerechnet)
    row = df.loc[[i for i in prep.idx_test if pd.isna(df.at[i, "a"])][:1]]
    X_row, _ = pl.transform_features(row, r)
    assert X_row["a"].iloc[0] == pytest.approx(tr["a"].median(), rel=1e-6)


def test_split_rows_are_the_same_as_before():
    """Die Aufteilung hängt nur von Zeilenzahl, Seed und y ab – gleiche Testzeilen wie train_test_split(X, y)."""
    from sklearn.model_selection import train_test_split

    df = _leaky_frame()
    prep = pl.prepare_ml_data(df, "y", ["a", "g", "m", "c"], pl.T_BIN, test_size=0.25, random_state=5)
    y = pl.encode_label(df["y"], prep.recipe)
    _, _, _, idx_te = train_test_split(np.zeros((len(df), 1)), df.index.to_numpy(), test_size=0.25,
                                       random_state=5, stratify=y)
    assert list(prep.idx_test) == list(idx_te)


def test_category_only_in_test_rows_is_unknown():
    df = _leaky_frame()
    first = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    df.loc[first.idx_test[0], "c"] = "selten"  # die Aufteilung ändert sich dadurch nicht
    prep = pl.prepare_ml_data(df, "y", ["a", "c"], pl.T_BIN)
    assert list(prep.idx_test) == list(first.idx_test)
    assert "selten" not in prep.recipe["categories"]["c"]
    assert not any("selten" in col for col in prep.recipe["feature_columns"])
    assert any(line.startswith("Testdaten: 1 Zeilen") for line in prep.log)


def test_column_with_values_only_in_test_rows_is_dropped():
    df = _leaky_frame().assign(nur_test=np.nan)
    first = pl.prepare_ml_data(df.drop(columns="nur_test"), "y", ["a", "c"], pl.T_BIN)
    df.loc[first.idx_test[:5], "nur_test"] = 1.0
    prep = pl.prepare_ml_data(df, "y", ["a", "c", "nur_test"], pl.T_BIN)
    assert "nur_test" not in prep.recipe["features"] and "nur_test" in prep.recipe["empty_columns"]
    assert any("ohne Werte in den Trainingsdaten" in w for w in prep.warnings)
