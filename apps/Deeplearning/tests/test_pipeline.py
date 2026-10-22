"""
Tests für pipeline.py und codegen.py (synthetische Daten, kleine Modelle).

    pip install pytest
    pytest -q
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
from synthetic import house_like, iris_like, lending_club_like  # noqa: E402

LENDING = {  # Schritt 2 – Beispiel-Konfiguration LendingClub
    "grade": (pl.A_DROP, "", ""), "emp_title": (pl.A_DROP, "", ""), "emp_length": (pl.A_DROP, "", ""),
    "title": (pl.A_DROP, "", ""), "issue_d": (pl.A_DROP, "", ""),
    "home_ownership": (pl.A_REPLACE, "NONE,ANY=OTHER", ""),
    "earliest_cr_line": (pl.A_YEAR, "%b-%Y", "earliest_cr_year"),
    "address": (pl.A_LAST, "5", "zip_code"),
}
SMALL = dict(epochs=4, patience=2, batch_size=128)


def lending_actions(raw):
    prof = pl.column_profile(raw)
    for c, (a, p, n) in LENDING.items():
        i = prof.index[prof["Spalte"] == c][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [a, p, n]
    return pl.actions_from_table(prof)


@pytest.fixture(scope="module")
def lc(tmp_path_factory):
    folder = tmp_path_factory.mktemp("lc")
    path = folder / "lc.csv"
    lending_club_like(3000).to_csv(path, index=False)
    raw, _ = pl.read_csv(path)
    viz, fitted, log, lineage = pl.apply_column_actions(raw, lending_actions(raw))
    feats = [c for c in viz.columns if c != "loan_status"]
    missing = {"revol_util": {"strategy": pl.M_DROP_ROWS}, "pub_rec_bankruptcies": {"strategy": pl.M_DROP_ROWS},
               "mort_acc": {"strategy": pl.M_GROUP, "param": "total_acc"}}
    prep = pl.prepare_ml_data(viz, "loan_status", feats, pl.T_BIN, positive_class="Fully Paid", missing=missing,
                              column_actions=fitted, lineage=lineage, test_size=0.2)
    cfg = pl.ModelConfig(**SMALL)
    model = pl.build_model(prep.X_train.shape[1], 1, cfg, pl.T_BIN)
    hist, info = pl.train_model(model, prep, cfg)
    return dict(folder=folder, path=path, raw=raw, viz=viz, fitted=fitted, prep=prep, model=model, cfg=cfg,
                info=info, hist=hist)


def test_suggestions():
    raw = lending_club_like(1500)
    sug = {c: pl.suggest_action(raw[c], c) for c in raw.columns}
    assert sug["term"][0] == pl.A_NUM
    assert sug["earliest_cr_line"] == (pl.A_YEAR, "%b-%Y")
    assert sug["emp_title"][0] == pl.A_DROP
    assert sug["address"][0] == pl.A_DROP
    assert sug["sub_grade"][0] == pl.A_KEEP
    assert sug["loan_amnt"][0] == pl.A_KEEP


def test_step2_lending_example(lc):
    v = lc["viz"]
    assert {"earliest_cr_year", "zip_code"} <= set(v.columns)
    assert not {"grade", "emp_title", "emp_length", "title", "issue_d", "address"} & set(v.columns)
    assert pl.is_numeric(v["term"]) and set(v["term"].unique()) == {36, 60}
    assert set(v["home_ownership"].unique()) <= {"MORTGAGE", "RENT", "OWN", "OTHER"}
    assert v["zip_code"].str.len().eq(5).all()


def test_feature_layout(lc):
    r = lc["prep"].recipe
    n_num = len(r["numeric_features"])
    n_dummies = sum(len(r["categories"][c]) - 1 for c in r["categorical_features"])
    assert len(r["feature_columns"]) == n_num + n_dummies
    assert r["classes"] == ["Charged Off", "Fully Paid"]
    assert not np.isnan(lc["prep"].X_train).any()
    assert lc["prep"].X_train.min() >= -1e-6 and lc["prep"].X_train.max() <= 1 + 1e-6  # MinMax auf Train


@pytest.mark.parametrize("fmt", [".keras", ".h5"])
def test_save_load_and_new_data_consistency(lc, fmt):
    prep, model = lc["prep"], lc["model"]
    scores = pl.predict_scores(model, prep.X_test, pl.T_BIN)
    folder = pl.save_bundle(model, pl.finalize_recipe(prep.recipe, lc["cfg"], lc["info"]), lc["folder"] / fmt[1:], fmt)
    m2, r2 = pl.load_bundle(folder)
    # Rohzeilen der Testmenge → exakt dieselben Vorhersagen wie im Training (Schritt 10 korrekt)
    res, rep, ev = pl.predict_new(m2, r2, lc["raw"].loc[prep.idx_test])
    assert np.allclose(res["P(Fully Paid)"].to_numpy(), np.round(scores, 4), atol=1e-4)
    assert ev is not None and ev["n"] == len(prep.idx_test)
    assert set(rep["extra_columns"]) >= {"grade", "emp_title", "issue_d"}
    m3, _ = pl.load_bundle_zip(pl.bundle_zip_bytes(folder))
    assert np.allclose(pl.predict_scores(m3, prep.X_test, pl.T_BIN), scores, atol=1e-6)


def test_new_data_problems(lc):
    prep, model, raw = lc["prep"], lc["model"], lc["raw"]
    r = prep.recipe
    with pytest.raises(pl.MissingColumnsError) as err:
        pl.predict_new(model, r, raw.drop(columns=["address"]).head(5))
    assert err.value.raw == ["address"]
    x = raw.head(4).copy()
    x.loc[x.index[0], "purpose"] = "raumschiff"
    x.loc[x.index[1], "revol_util"] = np.nan
    x.loc[x.index[2], "loan_amnt"] = 10_000_000
    res, rep, _ = pl.predict_new(model, r, x)
    assert rep["unknown"] == {"purpose": {"raumschiff": 1}}
    assert rep["imputed"].get("revol_util") == 1
    assert rep["out_of_range"].get("loan_amnt") == 1
    assert "unbekannte Kategorie" in res["Hinweis"].iloc[0]
    assert "Lücken ergänzt" in res["Hinweis"].iloc[1]
    # Einzelner Datensatz in der Form nach Schritt 2 (Formular in Schritt 10)
    one = lc["viz"].loc[[prep.idx_test[0]], r["features"]]
    res1, _, _ = pl.predict_new(model, r, one)
    assert abs(res1["P(Fully Paid)"].iloc[0] - pl.predict_scores(model, prep.X_test[:1], pl.T_BIN)[0]) < 1e-4


def test_threshold_and_evaluation(lc):
    prep, model = lc["prep"], lc["model"]
    s = pl.predict_scores(model, prep.X_test, pl.T_BIN)
    ev5 = pl.evaluate(prep.y_test, s, pl.T_BIN, prep.recipe["classes"], 0.5)
    ev9 = pl.evaluate(prep.y_test, s, pl.T_BIN, prep.recipe["classes"], 0.9)
    assert ev5["cm"].sum() == len(prep.y_test)
    assert ev9["cm"][:, 1].sum() <= ev5["cm"][:, 1].sum()  # höhere Schwelle → weniger Klasse 1
    imp = pl.permutation_importance(model, prep.X_test, prep.y_test, prep.recipe, n_rows=300, n_repeats=1)
    assert set(imp["Feature"]) == set(prep.recipe["features"])


def test_regression_label_scaling_h5(tmp_path):
    raw = house_like(1200)
    viz, fitted, _, lin = pl.apply_column_actions(raw, pl.actions_from_table(pl.column_profile(raw)))
    feats = [c for c in viz.columns if c != "price"]
    prep = pl.prepare_ml_data(viz, "price", feats, pl.T_REG, missing={"bathrooms": {"strategy": pl.M_MEDIAN}},
                              column_actions=fitted, lineage=lin, scale_label=True)
    assert prep.recipe["label_scaler"] and abs(prep.y_train_fit.mean()) < 1e-3
    cfg = pl.ModelConfig(layers=[{"units": 8, "activation": "relu", "dropout": 0}], output_activation="linear",
                         loss="mean_squared_error", **SMALL)
    model = pl.build_model(prep.X_train.shape[1], 1, cfg, pl.T_REG)
    pl.train_model(model, prep, cfg)
    s = pl.predict_scores(model, prep.X_test, pl.T_REG, prep.recipe)
    assert s.mean() > 10_000  # zurückgerechnet in Originaleinheiten
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "reg", ".h5")  # .h5 + mse: in Keras 3 heikel
    m2, r2 = pl.load_bundle(folder)
    res, _, ev = pl.predict_new(m2, r2, raw.loc[prep.idx_test])
    assert np.allclose(res["Vorhersage"].to_numpy(), s, rtol=1e-5)
    assert ev["MAE"] > 0


def test_multiclass():
    ir = iris_like()
    viz, fitted, _, lin = pl.apply_column_actions(ir, pl.actions_from_table(pl.column_profile(ir)))
    assert pl.suggest_task(viz["species"]) == pl.T_MULTI
    prep = pl.prepare_ml_data(viz, "species", [c for c in viz.columns if c != "species"], pl.T_MULTI,
                              column_actions=fitted, lineage=lin)
    cfg = pl.ModelConfig(layers=[{"units": 8, "activation": "relu", "dropout": 0}], output_activation="softmax",
                         loss="sparse_categorical_crossentropy", epochs=3, batch_size=16, class_weight=True)
    model = pl.build_model(prep.X_train.shape[1], 3, cfg, pl.T_MULTI)
    pl.train_model(model, prep, cfg)
    res, _, ev = pl.predict_new(model, prep.recipe, ir)
    assert set(res["Vorhersage"]) <= set(prep.recipe["classes"])
    assert np.allclose(res[[f"P({c})" for c in prep.recipe["classes"]]].sum(axis=1), 1, atol=1e-3)
    assert ev["cm"].shape == (3, 3)


def test_validation_errors():
    df = pd.DataFrame({"a": [1, 2, 3] * 10, "y": ["x", "y", "z"] * 10})
    with pytest.raises(ValueError):
        pl.prepare_ml_data(df, "y", ["a"], pl.T_BIN)  # 3 Klassen ≠ binär
    with pytest.raises(ValueError):
        pl.prepare_ml_data(df, "y", ["a"], pl.T_REG)  # Text-Label ≠ Regression
    with pytest.raises(ValueError):
        pl.layers_from_table(pd.DataFrame([{"Neuronen": 0, "Aktivierung": "relu", "Dropout": 0}]))
    with pytest.raises(ValueError):
        pl.apply_column_actions(df, [{"column": "y", "action": pl.A_REPLACE, "param": "kaputt"}])


def test_generated_script_runs(lc, tmp_path):
    """Exportiertes Python-Skript (Schritt 9) läuft und baut dieselben Features wie die App."""
    r = lc["prep"].recipe
    cfg = pl.ModelConfig(**SMALL)
    numeric_raw = {c for c in lc["raw"].columns if pl.is_numeric(lc["raw"][c])}
    script = cg.full_script({"name": str(lc["path"])}, lc["fitted"], numeric_raw, r, cfg,
                            folder=str(tmp_path / "gen"))
    script = script.replace("model_loss[['loss', 'val_loss']].plot()", "pass")
    script += (f"\nassert feature_columns == {r['feature_columns']!r}\n"
               f"assert X_train.shape == {lc['prep'].X_train.shape}, X_train.shape\nprint('OK', X_train.shape)\n")
    (tmp_path / "gen_script.py").write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg", TF_CPP_MIN_LOG_LEVEL="3")
    out = subprocess.run([sys.executable, str(tmp_path / "gen_script.py")], capture_output=True, text=True,
                         cwd=tmp_path, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout
