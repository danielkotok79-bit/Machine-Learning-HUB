"""
Regressionstests zu Audit 5 (erster Audit der Clustering-App): jeder Test hat einen gefundenen Fehler reproduziert
und schlägt mit dem Code vor dem Audit fehl.

    pytest -q tests/test_audit5_regressions.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from synthetic import college_like, iris_like, mall_like  # noqa: E402
from test_app import button, go, load  # noqa: E402

APP = str(ROOT / "app.py")


def _app(tmp_path, monkeypatch, df, name="daten.csv"):
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    df.to_csv(tmp_path / name, index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / name)
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    return at


# --- B1: Export-Code (Schritt 5) nach dem Code von Schritt 4 ------------------------------------------------------
def test_b1_export_code_matches_the_app_after_rows_and_columns_were_removed(tmp_path):
    """Schritt 4 entfernt Zeilen (Lücken), eine Spalte (zu viele Lücken) und ersetzt inf – der erzeugte Code änderte
    dabei df selbst: der Export 'Daten nach Schritt 2' hatte dann weniger Zeilen und Spalten als in der App, der
    Export mit Clustern eine Spalte zu wenig."""
    raw = college_like(300, seed=11)
    raw.loc[raw.index[:20], "Books"] = np.nan
    raw.loc[raw.index[5:200], "Personal"] = np.nan
    raw["Expend"] = raw["Expend"].astype(float)
    raw.loc[raw.index[30:33], "Expend"] = np.inf
    missing = {"Books": {"strategy": pl.M_DROP_ROWS}, "Personal": {"strategy": pl.M_DROP_COL}}
    r = parity.run_both(tmp_path, raw, {"Unnamed: 0": (pl.A_KEEP, "", "College")},
                        features=["Apps", "Books", "Personal", "Expend", "Outstate"], group="Private", name="College",
                        index=True, cfg=pl.ClusterConfig(n_clusters=3, curve=False), missing=missing)
    parity.assert_same(r)
    assert r["ns"]["df"].shape == r["viz"].shape  # df (Daten nach Schritt 2) bleibt unverändert
    for what in ("step2", "ml", "ml_scaled", "clusters"):
        for excel in (None, "de"):
            app, gen = parity.export(r, what, tmp_path, excel)
            assert app == gen, (what, excel)


def test_b1_without_removals_the_code_stays_simple():
    r = pl.prepare_cluster_data(college_like(120), ["Apps", "Outstate"], group_col="Private").recipe
    code = cg.step4_code(r)
    assert cg.rows_var(r) == "df" and "data = " not in code and "data[" not in code


# --- B2: keine Vergleichsgruppe ohne passenden Namen ---------------------------------------------------------------
def test_b2_last_column_does_not_silently_become_the_comparison_group(tmp_path, monkeypatch):
    """Wie CC GENERAL: die letzte Spalte TENURE hat 7 Werte – sie wurde ungefragt zur Vergleichsgruppe und fiel aus
    den Features."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"BALANCE": rng.gamma(2, 800, n).round(2), "PURCHASES": rng.gamma(1.5, 500, n).round(2),
                       "TENURE": rng.choice([6, 7, 8, 9, 10, 11, 12], n)})
    at = _app(tmp_path, monkeypatch, df)
    assert at.session_state["w_group"] == "(keine)"
    assert at.session_state["w_features"] == ["BALANCE", "PURCHASES", "TENURE"]


def test_b2_a_deselected_comparison_group_returns_to_the_features(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, iris_like())
    assert at.session_state["w_group"] == "species" and "species" not in at.session_state["w_features"]
    at.selectbox(key="w_group").set_value("(keine)").run()
    assert "species" in at.session_state["w_features"]
    at.selectbox(key="w_group").set_value("species").run()
    assert "species" not in at.session_state["w_features"]


# --- B3: ID-Spalten ------------------------------------------------------------------------------------------------
def test_b3_looks_like_id():
    n = 50
    assert pl.looks_like_id(pd.Series(np.arange(1, n + 1)), "CustomerID")
    assert pl.looks_like_id(pd.Series(np.arange(n)), "x")  # 0, 1, 2 … ohne passenden Namen
    assert pl.looks_like_id(pd.Series(np.random.default_rng(0).permutation(np.arange(1000, 1000 + n))), "kunden_nr")
    assert not pl.looks_like_id(pd.Series(np.arange(2000, 2000 + n)), "Jahr")  # Jahreszahlen: kein ID-Name
    assert not pl.looks_like_id(pd.Series(np.r_[np.arange(1, n), 1]), "CustomerID")  # nicht eindeutig
    assert not pl.looks_like_id(pd.Series(np.arange(n) + 0.5), "id")  # keine ganzen Zahlen
    assert not pl.looks_like_id(pd.Series(["a"] * n), "id")


def test_b3_id_column_is_the_row_name_not_a_feature(tmp_path, monkeypatch):
    """Mall Customers: CustomerID (1, 2, 3 …) war Standard-Feature – K-Means gruppierte teils nach Zeilennummer."""
    at = _app(tmp_path, monkeypatch, mall_like(150))
    assert at.session_state["w_name_col"] == "CustomerID"
    assert "CustomerID" not in at.session_state["w_features"]
    assert not any("sieht aus wie eine ID" in w.value for w in at.warning)
    at.selectbox(key="w_name_col").set_value("(keine)").run()
    feats = at.session_state["w_features"]
    at.multiselect(key="w_features").set_value(["CustomerID"] + feats).run()
    assert any("CustomerID: sieht aus wie eine ID" in w.value for w in at.warning)


# --- C1: Text 'Wie im Kurs' mit den echten Gruppen -----------------------------------------------------------------
def test_c1_course_comparison_names_the_real_group(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, iris_like())
    button(at, "Daten für K-Means vorbereiten").click().run()
    go(at, 5)
    at.number_input(key="w_k").set_value(3).run()
    button(at, "Training starten").click().run()
    go(at, 7)
    text = " ".join(i.value for i in at.info if "Wie im Kurs" in i.value)
    assert "setosa = 0" in text and "'Yes'" not in text
    assert "'versicolor'" in text or "mehrere Cluster zur selben Gruppe" in text or "passen hier schon" in text


# --- C2: Markdown aus einem geladenen Modell -----------------------------------------------------------------------
EVIL = "![x](http://example.invalid/t.png)"


def _bundle(tmp_path, feature=EVIL, names=None):
    df = college_like(150, seed=3).rename(columns={"Apps": feature, "Private": EVIL + "g"})
    prep = pl.prepare_cluster_data(df, [feature, "Outstate"], group_col=EVIL + "g")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep)
    cmp = pl.compare_groups(model.labels_, prep.groups, 2)
    recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=2), info, names or [EVIL, "ok"], cmp)
    return df, model, recipe


def test_c2_summary_of_step_10_does_not_render_markdown_from_the_model(tmp_path):
    df, model, recipe = _bundle(tmp_path)
    new = df.head(30).copy()
    new.loc[new.index[:3], EVIL] = np.nan  # Lücken → Spaltenname im Fazit
    res, rep, cmp = pl.assign_new(model, recipe, new)
    fz = pl.assess_assignment(recipe, res, rep, cmp)
    texts = [q["text"] for q in fz["quality"]] + [d for q in fz["quality"] for d in q["details"]] + fz["meaning"]
    assert texts and not any("![x](http" in t for t in texts)
    assert any(r"\!\[x\]\(http" in t for t in texts)  # sichtbar als Text
    assert pl.md_escape("a*b_c\n![x](y)") == r"a\*b\_c \!\[x\]\(y\)"


def test_c2_step_10_form_labels_and_heading_are_escaped(tmp_path, monkeypatch):
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    _, model, recipe = _bundle(tmp_path)
    pl.save_bundle(model, recipe, tmp_path / "models" / "fremd")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    go(at, 8)
    button(at, "Laden").click().run()
    assert not at.exception and not at.error
    go(at, 9)
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    labels = [w.label for w in at.number_input]
    assert labels and not any("![x](" in lb for lb in labels)
    markdown = " ".join(m.value for m in at.markdown)
    assert "![x](http" not in markdown


# --- C3: beschädigtes preprocessing.json ---------------------------------------------------------------------------
@pytest.mark.parametrize("change", [
    lambda r: r.pop("features"),
    lambda r: r.update(features="Apps"),
    lambda r: r.update(scaler={"type": "RobustScaler"}),  # wirkte vorher still wie 'keiner'
    lambda r: r["scaler"].update(mean=r["scaler"]["mean"][:1]),
    lambda r: r.update(train_info=dict(r["train_info"], dist_p99=["a", "b"])),
    lambda r: r.update(group_mapping={"x": "Yes"}),
    lambda r: r.update(fallback_fill={**r["fallback_fill"], "Apps": "viel"}),
    lambda r: r.update(training=dict(r["training"], n_clusters="2")),
])
def test_c3_broken_recipes_are_refused_when_loading(tmp_path, change):
    df = college_like(120, seed=5)
    prep = pl.prepare_cluster_data(df, ["Apps", "Outstate"], group_col="Private")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep)
    cmp = pl.compare_groups(model.labels_, prep.groups, 2)
    folder = pl.save_bundle(model, pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=2), info, None, cmp),
                            tmp_path / "m")
    good = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    assert pl.load_bundle(folder)[1]["features"] == ["Apps", "Outstate"]
    broken = json.loads(json.dumps(good))
    change(broken)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("preprocessing.json", json.dumps(broken))
        z.write(folder / "model.skops", "model.skops")
    with pytest.raises(ValueError, match="unvollständig oder beschädigt"):
        pl.load_bundle_zip(buf.getvalue())


def test_c3_model_and_recipe_must_fit_together(tmp_path):
    df = college_like(120, seed=5)
    prep3 = pl.prepare_cluster_data(df, ["Apps", "Outstate", "Grad.Rate"])
    prep2 = pl.prepare_cluster_data(df, ["Apps", "Outstate"])
    model2, _, _ = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep2)
    folder = pl.save_bundle(model2, prep3.recipe, tmp_path / "m")
    with pytest.raises(ValueError, match="passen nicht zusammen"):
        pl.load_bundle(folder)


# --- C4–C6 ---------------------------------------------------------------------------------------------------------
def test_c4_constant_column_with_rounding_noise_is_reported():
    df = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": [0.1] * 50})
    assert np.std(df["b"].to_numpy()) > 0  # 2.8e-17 durch Rundung
    prep = pl.prepare_cluster_data(df, ["a", "b"])
    assert any("Ohne Streuung" in w and "'b'" in w for w in prep.warnings)


def test_c5_empty_clusters_are_called_empty():
    df = pd.DataFrame({"a": [1.0, 2.0] * 20, "b": [5.0, 7.0] * 20})
    prep = pl.prepare_cluster_data(df, ["a", "b"])
    cfg = pl.ClusterConfig(n_clusters=3, curve=False)
    _, _, info = pl.train_kmeans(cfg, prep)
    assert sorted(info["sizes"]) == [0, 20, 20]
    notes = " ".join(pl.k_assessment(None, info, cfg)["notes"])
    assert "Leerer Cluster" in notes and "Sehr kleiner Cluster" not in notes


def test_c6_changed_scaler_or_drop_first_shows_the_stale_hint(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, college_like(150))
    button(at, "Daten für K-Means vorbereiten").click().run()
    hint = "seit dem Vorbereiten geändert"
    assert not any(hint in i.value for i in at.info)
    at.selectbox(key="w_scaler").set_value("MinMaxScaler").run()
    assert any(hint in i.value for i in at.info)
    at.selectbox(key="w_scaler").set_value("StandardScaler").run()
    assert not any(hint in i.value for i in at.info)
    at.checkbox(key="w_drop_first").check().run()
    assert any(hint in i.value for i in at.info)
