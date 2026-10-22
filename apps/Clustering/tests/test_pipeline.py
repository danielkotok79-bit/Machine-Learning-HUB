"""
Tests für pipeline.py und codegen.py im ganzen Ablauf (synthetische Daten, kleine K-Means-Modelle).

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
from synthetic import lending_club_like  # noqa: E402

LENDING = {  # Schritt 2 – wie die Beispiel-Konfiguration der Supervised-Version (gleiche Rohdaten)
    "grade": (pl.A_DROP, "", ""), "emp_title": (pl.A_DROP, "", ""), "emp_length": (pl.A_DROP, "", ""),
    "title": (pl.A_DROP, "", ""), "issue_d": (pl.A_DROP, "", ""), "sub_grade": (pl.A_DROP, "", ""),
    "home_ownership": (pl.A_REPLACE, "NONE,ANY=OTHER", ""),
    "earliest_cr_line": (pl.A_YEAR, "%b-%Y", "earliest_cr_year"),
    "address": (pl.A_LAST, "5", "zip_code"),
    "revol_util": (pl.A_CLIP, ",100", ""),
}
SMALL = dict(n_clusters=4, n_init=3, curve=False)  # klein und schnell


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
    prep = pl.prepare_cluster_data(viz, feats, missing=missing, column_actions=fitted, lineage=lineage,
                                   group_col="loan_status")
    cfg = pl.ClusterConfig(**SMALL)
    model, hist, info = pl.train_kmeans(cfg, prep)
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
    assert v["revol_util"].max() <= 100


def test_feature_layout(lc):
    prep = lc["prep"]
    r = prep.recipe
    n_num = len(r["numeric_features"])
    n_dummies = sum(len(r["categories"][c]) for c in r["categorical_features"])  # drop_first=False: alle Kategorien
    assert len(r["feature_columns"]) == n_num + n_dummies and r["drop_first"] is False
    assert "loan_status" not in r["features"] and r["group_col"] == "loan_status"
    assert not np.isnan(prep.X).any() and prep.X.shape == (len(prep.index), len(r["feature_columns"]))
    np.testing.assert_allclose(prep.X.mean(axis=0), 0, atol=1e-9)  # StandardScaler auf allen Zeilen
    assert set(prep.groups.unique()) == {"Charged Off", "Fully Paid"}


@pytest.mark.parametrize("fmt", [".skops", ".joblib"])
def test_save_load_and_new_data_consistency(lc, fmt):
    prep, model = lc["prep"], lc["model"]
    recipe = pl.finalize_recipe(prep.recipe, lc["cfg"], lc["info"])
    folder = pl.save_bundle(model, recipe, lc["folder"] / fmt[1:], fmt)
    m2, r2 = pl.load_bundle(folder)
    np.testing.assert_array_equal(m2.cluster_centers_, model.cluster_centers_)
    assert r2["model_file"] == f"model{fmt}" and r2["training"]["n_clusters"] == 4
    # dieselben Rohdaten über das Rezept → dieselben Cluster wie im Training (gleiche Zeilen)
    res, rep, cmp = pl.assign_new(m2, r2, lc["raw"])
    used = lc["viz"].index.get_indexer(prep.index)
    assert res["Cluster"].to_numpy()[used].tolist() == model.labels_.tolist()
    assert cmp is not None and cmp["n"] == len(lc["raw"])
    m3, r3 = pl.load_bundle_zip(pl.bundle_zip_bytes(folder), allow_pickle=fmt == ".joblib")
    assert r3 == r2 and np.array_equal(m3.predict(prep.X[:50]), model.labels_[:50])


def test_new_data_problems(lc):
    model, recipe = lc["model"], pl.finalize_recipe(lc["prep"].recipe, lc["cfg"], lc["info"])
    raw_new = lending_club_like(200, seed=9)
    raw_new.loc[raw_new.index[:3], "home_ownership"] = "HAUSBOOT"
    raw_new.loc[raw_new.index[3:6], "annual_inc"] = 1e9
    raw_new["extra"] = 1
    res, rep, cmp = pl.assign_new(model, recipe, raw_new)
    assert rep["unknown"] == {"home_ownership": {"HAUSBOOT": 3}} and rep["row_unknown"][:3].all()
    assert rep["out_of_range"].get("annual_inc", 0) >= 3 and rep["row_out_of_range"][3:6].all()
    assert "extra" in rep["extra_columns"] and "loan_status" not in rep["extra_columns"]
    assert res.loc[0, "Hinweis"].startswith("unbekannte Kategorie")
    assert rep["row_far"][3:6].all()  # Einkommen 1e9 → weit weg von allen Clustern
    with pytest.raises(pl.MissingColumnsError) as err:
        pl.assign_new(model, recipe, raw_new.drop(columns=["address"]))
    assert err.value.raw == ["address"] and "zip_code" in str(err.value)


def test_validation_errors():
    df = pd.DataFrame({"a": [1, 2, 3] * 10, "y": ["x", "y", "z"] * 10})
    with pytest.raises(ValueError, match="mindestens ein Feature"):
        pl.prepare_cluster_data(df, [])
    with pytest.raises(ValueError, match="Unbekannte Features"):
        pl.prepare_cluster_data(df, ["gibts_nicht"])
    with pytest.raises(ValueError, match="nicht in den Daten"):
        pl.prepare_cluster_data(df, ["a"], group_col="gibts_nicht")
    with pytest.raises(ValueError, match="nur bei Zahlen"):
        pl.prepare_cluster_data(df.assign(y=df["y"].where(df.index > 2)), ["a", "y"],
                                missing={"y": {"strategy": pl.M_MEDIAN}})
    with pytest.raises(ValueError):
        pl.apply_column_actions(df, [{"column": "y", "action": pl.A_REPLACE, "param": "kaputt"}])


def test_generated_script_runs(lc, tmp_path):
    """Exportiertes Python-Skript (Schritt 9) läuft und rechnet dasselbe wie die App."""
    prep = lc["prep"]
    r = prep.recipe
    cfg = pl.ClusterConfig(n_clusters=4, n_init=3, curve=True, k_max=5)
    numeric_raw = {c for c in lc["raw"].columns if pl.is_numeric(lc["raw"][c])}
    script = cg.full_script({"name": str(lc["path"])}, lc["fitted"], numeric_raw, r, cfg,
                            folder=str(tmp_path / "gen"))
    script = script.replace("plt.show()", "plt.close('all')")
    script += (f"\nassert feature_columns == {r['feature_columns']!r}\n"
               f"assert X.shape == {prep.X.shape}, X.shape\n"
               f"assert labels.tolist() == {lc['model'].labels_.tolist()!r}\n"
               "assert later_model.cluster_centers_.shape == kmeans.cluster_centers_.shape\n"
               "print('OK', X.shape)\n")
    (tmp_path / "gen_script.py").write_text(script, encoding="utf-8")
    env = dict(os.environ, MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, str(tmp_path / "gen_script.py")], capture_output=True, text=True,
                         cwd=tmp_path, env=env, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert "OK" in out.stdout
