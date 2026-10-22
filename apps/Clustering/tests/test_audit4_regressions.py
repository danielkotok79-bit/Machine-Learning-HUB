"""
Audit 4 des Grundgerüsts (Version 9 der Supervised-Version, scikit-learn): die Tests zu Speichern, Laden und Anzeige –
übertragen auf K-Means. Dazu: Modelle der Supervised-Version und fremde Modelltypen werden abgelehnt.

    pytest -q tests/test_audit4_regressions.py
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
from synthetic import college_like, iris_like  # noqa: E402

APP = str(ROOT / "app.py")
RNG = np.random.default_rng(44)


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _nav(at, i):
    at.sidebar.radio(key="nav").set_value(at.sidebar.radio(key="nav").options[i]).run()


def _ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


def _iris():
    prep = pl.prepare_cluster_data(iris_like(), ["sepal_length", "petal_length"], group_col="species")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, n_init=3, curve=False), prep)
    return prep, model, info


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


# --- B4: die Pickle-Rückfrage passt zur Datei, die wirklich geladen wird --------------------------------------------
def test_pickle_question_matches_the_file_that_is_loaded(tmp_path):
    prep, model, _ = _iris()
    sk = pl.save_bundle(model, prep.recipe, tmp_path / "s", ".skops")
    jb = pl.save_bundle(model, prep.recipe, tmp_path / "j", ".joblib")
    both = {"model.skops": (sk / "model.skops").read_bytes(), "model.joblib": (jb / "model.joblib").read_bytes()}
    names_joblib = _zip({"preprocessing.json": (jb / "preprocessing.json").read_text(), **both})
    assert pl.zip_needs_pickle(names_joblib)
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


# --- neu: nur K-Means-Modelle der Clustering-Version --------------------------------------------------------------
def test_models_of_the_supervised_version_are_refused(tmp_path):
    prep, model, _ = _iris()
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "models" / "cl")
    other = pl.save_bundle(model, dict(prep.recipe, task="binary"), tmp_path / "models" / "sup")
    with pytest.raises(ValueError, match="nicht aus der Clustering-Version"):
        pl.load_bundle(other)
    assert [b["name"] for b in pl.list_bundles(tmp_path / "models")] == ["cl"]
    assert pl.list_bundles(tmp_path / "models")[0]["k"] is None  # ohne finalize_recipe steht K nicht im Rezept
    assert pl.load_bundle(folder)[0].n_clusters == 3


def test_a_foreign_model_type_is_refused(tmp_path):
    import skops.io as sio
    from sklearn.linear_model import LogisticRegression

    prep, model, _ = _iris()
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "m")
    sio.dump(LogisticRegression().fit(prep.X, (prep.X[:, 0] > 0).astype(int)), folder / "model.skops")
    with pytest.raises(ValueError, match="kein K-Means-Modell"):
        pl.load_bundle(folder)


# --- C3: Namen aus Dateien nicht als Markdown ---------------------------------------------------------------------
def test_unknown_model_names_are_not_shown_as_markdown():
    assert pl.model_name("![x](http://example.invalid/t.png)") == "?"
    assert pl.model_name(pl.MODEL_KMEANS) == "K-Means"


def test_names_from_files_are_escaped(tmp_path, monkeypatch):
    """Name eines hochgeladenen ZIP und Cluster-Namen aus einem fremden Rezept kommen wörtlich in die Anzeige."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    college_like(200, seed=4).to_csv(tmp_path / "college.csv")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "college.csv")).run()
    _button(at, "Daten laden").click().run()
    _nav(at, 1)
    _button(at, "Übernehmen").click().run()
    _nav(at, 3)
    _button(at, "Daten für K-Means vorbereiten").click().run()
    _nav(at, 5)
    at.checkbox(key="w_curve").uncheck().run()
    _button(at, "Training starten").click().run()
    _ok(at, "Training")

    bad = "![x](http://example.invalid/t.png)"
    r = pl.finalize_recipe(at.session_state["prep"].recipe, pl.ClusterConfig(), at.session_state["train_info"],
                           [bad, "*fett*", "ok"][:at.session_state["train_info"]["k"]])
    at.session_state["bundle"] = {"model": at.session_state["model"], "recipe": r, "name": bad, "token": "x"}
    _nav(at, 8)
    _ok(at, "Schritt 9")
    shown = [m.value for m in at.markdown if "Geladen" in m.value]
    assert shown and "\\!\\[x\\]\\(http://example.invalid/t.png\\)" in shown[0]  # Name wörtlich, kein Bild
    names = [m.value for m in at.markdown if m.value.startswith("Cluster: ")]
    assert names and "\\!\\[x\\]" in names[0] and "\\*fett\\*" in names[0]
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
    prep, model, _ = _iris()
    for obj, ok in [(model, True), (FunctionTransformer(_helper).fit(prep.X), False)]:
        ns = {"kmeans": obj}
        if ok:
            exec(code, ns)  # noqa: S102
            assert type(ns["later_model"]) is type(obj)
            np.testing.assert_array_equal(ns["later_model"].cluster_centers_, model.cluster_centers_)
        else:
            with pytest.raises(Exception, match="Untrusted|untrusted"):
                exec(code, ns)  # noqa: S102
    assert sio.get_untrusted_types(file=tmp_path / "m" / "model.skops")


# --- C5: gescheitertes Speichern lässt das vorhandene Modell intakt -------------------------------------------------
def test_failed_save_keeps_the_previous_model(tmp_path, monkeypatch):
    import skops.io as sio

    prep, model, _ = _iris()
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "m", ".joblib")

    def broken(*a, **k):
        raise OSError("Datenträger voll")

    monkeypatch.setattr(sio, "dump", broken)
    with pytest.raises(OSError):
        pl.save_bundle(model, prep.recipe, folder, ".skops")
    m2, r2 = pl.load_bundle(folder)  # das alte .joblib-Modell ist noch da und passt zum Rezept
    assert r2["model_file"] == "model.joblib" and np.array_equal(m2.predict(prep.X), model.predict(prep.X))
    assert sorted(p.name for p in folder.iterdir()) == ["model.joblib", "preprocessing.json"]
    monkeypatch.undo()
    pl.save_bundle(model, prep.recipe, folder, ".skops")  # anderes Format → altes Modell weg, Rezept neu
    assert sorted(p.name for p in folder.iterdir()) == ["model.skops", "preprocessing.json"]


# --- C6: entpackte Größe einer .skops-Datei ist begrenzt ------------------------------------------------------------
def test_skops_unpacked_size_is_limited(tmp_path, monkeypatch):
    prep, model, _ = _iris()
    folder = pl.save_bundle(model, prep.recipe, tmp_path / "m")
    monkeypatch.setattr(pl, "MAX_BUNDLE_BYTES", 1000)
    with pytest.raises(ValueError, match="zu groß"):
        pl.load_bundle(folder)


def test_saved_recipe_contains_names_mapping_and_metrics(tmp_path):
    prep, model, info = _iris()
    cmp = pl.compare_groups(model.labels_, prep.groups, 3)
    recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=3), info, ["  klein ", None, "nan"], cmp)
    assert recipe["cluster_names"] == ["klein", "", ""]
    assert set(recipe["group_mapping"].values()) <= {"setosa", "versicolor", "virginica"}
    assert recipe["metrics"]["silhouette"] == pytest.approx(info["silhouette"])
    assert sum(recipe["cluster_share"]) == pytest.approx(1.0)
    folder = pl.save_bundle(model, recipe, tmp_path / "m")
    _, loaded = pl.load_bundle(folder)
    assert loaded["cluster_names"] == ["klein", "", ""] and loaded["group_mapping"] == recipe["group_mapping"]
    res, rep, cmp_new = pl.assign_new(model, loaded, iris_like())
    assert "Cluster-Name" in res.columns and cmp_new["accuracy"] == pytest.approx(cmp["accuracy"])
    assert pd.Series(res["Cluster"]).tolist() == model.labels_.tolist()  # dieselben Zeilen → derselbe Cluster
