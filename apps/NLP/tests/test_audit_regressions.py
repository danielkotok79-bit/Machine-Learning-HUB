"""
Sicherheit und Robustheit (übernommen aus den Audits der Supervised- und Clustering-Version und für Texte erweitert):
Modelldateien nur sicher laden, Rezepte prüfen, ZIP-Grenzen, Formeln und Konfigurationen begrenzen, keine Code- oder
Markdown-Einschleusung über Namen aus Dateien, gescheitertes Speichern lässt das alte Modell intakt.

    pytest -q tests/test_audit_regressions.py
"""
import ast
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
from synthetic import yelp_like  # noqa: E402


def _text_process(text):  # wie im Kurs: CountVectorizer(analyzer=text_process) – eigene Funktion im Modell
    return text.split()


@pytest.fixture(scope="module")
def trained():
    prep = pl.prepare_text_data(yelp_like(1200, seed=21), "text", "stars", classes=["1", "5"])
    cfg = pl.TextModelConfig(model=pl.MODEL_LOGREG, curve=False)
    model, _, info = pl.train_model(cfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info, {"accuracy": 0.9}, 0.5)
    return prep, model, recipe


@pytest.fixture()
def bundle(trained, tmp_path):
    prep, model, recipe = trained
    return pl.save_bundle(pl.make_pipeline(prep.vectorizer, model), recipe, tmp_path / "m")


# --- Modelldateien --------------------------------------------------------------------------------------------
def test_zip_roundtrip_and_zip_bomb(bundle, monkeypatch):
    data = pl.bundle_zip_bytes(bundle)
    pipe, r = pl.load_bundle_zip(data)
    assert r["app"] == pl.APP_ID and len(pipe.steps) == 2
    monkeypatch.setattr(pl, "MAX_BUNDLE_BYTES", 1000)  # entpackt größer als erlaubt
    with pytest.raises(ValueError, match="ZIP zu groß"):
        pl.load_bundle_zip(data)
    with pytest.raises(ValueError, match="zu groß"):  # auch die .skops-Datei selbst ist ein ZIP
        pl.load_bundle(bundle)


def test_zip_without_recipe_is_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("model.skops", b"x")
    with pytest.raises(ValueError, match="preprocessing.json"):
        pl.load_bundle_zip(buf.getvalue())


def test_joblib_from_zip_needs_explicit_trust(trained, tmp_path):
    prep, model, recipe = trained
    folder = pl.save_bundle(pl.make_pipeline(prep.vectorizer, model), recipe, tmp_path / "j", ".joblib")
    data = pl.bundle_zip_bytes(folder)
    assert pl.zip_needs_pickle(data)
    with pytest.raises(ValueError, match="pickle"):
        pl.load_bundle_zip(data)
    pipe, _ = pl.load_bundle_zip(data, allow_pickle=True)
    assert pipe.steps[1][1].__class__.__name__ == "LogisticRegression"


def test_model_file_in_recipe_cannot_point_outside(bundle, tmp_path):
    recipe = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    recipe["model_file"] = str(tmp_path / "fremd" / "model.joblib")  # absoluter Pfad → nur der Dateiname zählt
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("preprocessing.json", json.dumps(recipe))
        zf.write(bundle / "model.skops", arcname="model.skops")
    assert not pl.zip_needs_pickle(buf.getvalue())
    pipe, r = pl.load_bundle_zip(buf.getvalue())  # lädt model.skops aus dem ZIP
    assert len(pipe.steps[0][1].vocabulary_) == r["vocab_size"]


def test_skops_with_own_function_is_refused(trained, tmp_path):
    """Wie im Kurs CountVectorizer(analyzer=text_process): eine eigene Funktion lädt skops nicht ungefragt."""
    import skops.io as sio
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline

    prep, _, recipe = trained
    pipe = Pipeline([("vect", CountVectorizer(analyzer=_text_process)), ("clf", MultinomialNB())])
    pipe.fit(list(prep.text_train), prep.y_train)
    folder = tmp_path / "eigen"
    folder.mkdir()
    sio.dump(pipe, folder / "model.skops")
    (folder / "preprocessing.json").write_text(json.dumps(pl._jsonable(recipe)), encoding="utf-8")
    with pytest.raises(ValueError, match="nicht als sicher"):
        pl.load_bundle(folder)


def test_generated_skops_code_trusts_only_sklearn(trained, tmp_path):
    prep, model, _ = trained
    code = cg.save_code(str(tmp_path / "m"), ".skops")
    assert "startswith('sklearn.')" in code
    ns = {"vectorizer": prep.vectorizer, "model": model}
    exec(cg.IMPORTS + "\n" + code, ns)  # noqa: S102
    assert ns["later_pipe"].predict(["great food"]).shape == (1,)


def test_failed_save_keeps_the_previous_model(trained, tmp_path, monkeypatch):
    import skops.io as sio

    prep, model, recipe = trained
    pipe = pl.make_pipeline(prep.vectorizer, model)
    folder = pl.save_bundle(pipe, recipe, tmp_path / "m", ".joblib")

    def broken(*a, **k):
        raise OSError("Datenträger voll")

    monkeypatch.setattr(sio, "dump", broken)
    with pytest.raises(OSError):
        pl.save_bundle(pipe, recipe, folder, ".skops")
    p2, r2 = pl.load_bundle(folder)
    assert r2["model_file"] == "model.joblib"
    assert sorted(p.name for p in folder.iterdir()) == ["model.joblib", "preprocessing.json"]
    monkeypatch.undo()
    pl.save_bundle(pipe, recipe, folder, ".skops")  # anderes Format → altes Modell weg
    assert sorted(p.name for p in folder.iterdir()) == ["model.skops", "preprocessing.json"]


# --- Rezepte prüfen -------------------------------------------------------------------------------------------
def test_models_of_other_versions_are_refused(bundle):
    recipe = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    for app in (None, "clustering", "![x](http://evil)"):
        r = dict(recipe, app=app) if app else {k: v for k, v in recipe.items() if k != "app"}
        with pytest.raises(ValueError, match="nicht aus der NLP-Version") as exc:
            pl.check_recipe(r)
        assert "![" not in str(exc.value)  # Name aus der Datei escaped (Markdown)
    assert pl.list_bundles(bundle.parent)  # das eigene Modell wird aufgelistet


@pytest.mark.parametrize("key, value", [
    ("classes", ["1"]), ("classes", ["1", "1"]), ("classes", "15"), ("task", "multiclass"), ("label_kind", "zahl"),
    ("class_values", [1]), ("class_values", ["a", "b"]), ("threshold", 1.5), ("threshold", True),
    ("text_col", ""), ("label", 5), ("column_actions", {"a": 1}), ("lineage", []), ("training", {"model": "rf"}),
    ("label_stats", {"class_share": {"1": "viel"}}), ("train_fingerprints", ["x"]), ("text_stats", {"mean": "x"}),
    ("vectorizer", "count"), ("n_rows", {"train": "x"}),
])
def test_broken_recipe_fails_at_loading(trained, key, value):
    _, _, recipe = trained
    r = json.loads(json.dumps(pl._jsonable(recipe)))
    r[key] = value
    with pytest.raises(ValueError, match="unvollständig oder beschädigt"):
        pl.check_recipe(r)


def test_pipeline_must_match_the_recipe(trained):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB

    prep, model, recipe = trained
    r = json.loads(json.dumps(pl._jsonable(recipe)))
    other_vec = CountVectorizer().fit(["ganz andere wörter hier", "noch mehr"])
    with pytest.raises(ValueError, match="Vektorisierer und Modell passen nicht"):
        pl.check_recipe(r, pl.make_pipeline(other_vec, model))
    nb = MultinomialNB().fit(prep.X_train, prep.y_train)
    with pytest.raises(ValueError, match="laut Rezept"):
        pl.check_recipe(r, pl.make_pipeline(prep.vectorizer, nb))  # Rezept sagt logreg
    three = MultinomialNB().fit(prep.X_train[:30], np.arange(30) % 3)
    with pytest.raises(ValueError, match="Klassen"):
        pl.check_recipe(dict(r, training={}), pl.make_pipeline(prep.vectorizer, three))
    with pytest.raises(ValueError, match="keine Pipeline"):
        pl.check_recipe(r, model)
    with pytest.raises(ValueError, match="nicht trainiert"):
        pl.check_recipe(r, pl.make_pipeline(CountVectorizer(), MultinomialNB()))


# --- Code und Markdown aus Namen ------------------------------------------------------------------------------
def test_names_with_newlines_cannot_inject_code():
    evil_col, evil_cls = "text\nraise SystemExit('col')", "x\nraise SystemExit('cls')"
    rng = np.random.default_rng(3)
    y = rng.choice([evil_cls, "ok"], 200)
    df = pd.DataFrame({evil_col: [("good " if v == "ok" else "bad ") + "words here" for v in y], "y": y})
    prep = pl.prepare_text_data(df, evil_col, "y", classes=[evil_cls, "ok"])
    code = "\n".join([cg.step4_code(prep.recipe), cg.eval_code(pl.T_BIN, 0.5, pl.TextModelConfig(),
                                                              prep.recipe["classes"]),
                      cg.export_code("split", prep.recipe), cg.predictions_export_code(prep.recipe)])
    assert not [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Raise)]
    assert all(not ln.lstrip().startswith("raise") for ln in code.splitlines())


def test_fazit_escapes_names_from_files(trained):
    prep, model, recipe = trained
    r = dict(recipe, classes=["![a](http://evil)", "*b*"], text_col="[t](x)")
    res = pd.DataFrame({"P(*b*)": [0.9, 0.2], "Vorhersage": ["*b*", "![a](http://evil)"], "Hinweis": ["", ""]})
    rep = {"y": None, "total": np.array([3, 3]), "known": np.array([3, 3]), "row_empty": np.zeros(2, bool),
           "row_no_known": np.zeros(2, bool), "row_mostly_unknown": np.zeros(2, bool), "mean_words": 3.0}
    fz = pl.assess_prediction(r, res, rep, None)
    text = " ".join([q["text"] for q in fz["quality"]] + fz["meaning"] + fz["advice"])
    assert "![" not in text and "\\*b\\*" in text and "\\[t\\]" in text


# --- Formeln und Konfiguration (Schritt 2, aus den Audits übernommen) -------------------------------------------
def formula(expr):
    df = pd.DataFrame({"a": np.arange(5.0), "b": np.arange(5.0) + 1, "c": list("xyzxy")})
    return pl.apply_new_columns(df, [{"name": "neu", "kind": pl.N_FORMULA, "definition": expr}])[0]["neu"]


@pytest.mark.parametrize("expr, message", [
    ("9 ** 9 ** 9 + a", "Exponent"), ("'x' * 10**8 == c", "Text in Anführungszeichen"),
    ("-" * 990 + "a", "verschachtelt"), ("a + " * 300 + "a", "zu lang"), ("abs(a, b)", "nicht erlaubt"),
    ("__import__('os').system('x')", "nicht erlaubt|Unbekannte|unbekannt"), ("a.__class__", "nicht erlaubt"),
])
def test_formula_limits_fail_fast(expr, message):
    t = time.perf_counter()
    with pytest.raises(ValueError, match=message):
        formula(expr)
    assert time.perf_counter() - t < 2


@pytest.mark.parametrize("key, value, expected", [
    ("w_ngram", 7, 3), ("w_ngram", "2", 2), ("w_max_df", 5, 1.0), ("w_max_df", "0,8", 0.8), ("w_min_df", 0, 1),
    ("w_test_size", 0.9, 0.5), ("w_alpha", "0,5", 0.5), ("w_svc_c", -1, 1e-6), ("w_stratify", "false", False),
    ("w_classes", ["1", 5, {"x": 1}], ["1", "5"]), ("w_classes", "1,5", None), ("w_model", "linsvc", "linsvc"),
    ("w_threshold", float("nan"), None), ("w_max_feat", "Infinity", None), ("w_text", ["a"], None),
])
def test_config_values_are_coerced(key, value, expected):
    got = pl.coerce_setting(key, value)
    assert got == expected and type(got) is type(expected)


def test_config_from_other_versions_is_cleaned():
    cfg = json.loads("""{"step2": [{"Spalte": "text", "Aktion": "gibt es nicht"}, {"Spalte": ["x"]}],
        "missing": [{"Spalte": "a", "Strategie": "Median"}], "new_columns": [{"name": "n", "kind": "Formel",
        "definition": "a + b"}, {"name": "", "kind": "Formel"}], "widgets": {"w_features": ["a"], "w_k": 3,
        "w_rf_trees": 600, "w_alpha": "abc", "w_text": "text"}}""")
    out, skipped = pl.sanitize_config(cfg)
    assert out["step2"] == [{"Spalte": "text", "Aktion": pl.A_KEEP, "Parameter": "", "Neuer Name": ""}]
    assert "missing" not in out and [s["name"] for s in out["new_columns"]] == ["n"]
    assert out["widgets"] == {"w_text": "text"} and skipped == ["w_alpha"]
    for bad in ([1, 2], "text", None):
        with pytest.raises(ValueError, match="JSON-Objekt"):
            pl.sanitize_config(bad)


def test_excel_export_neutralises_formulas():
    df = pd.DataFrame({"text": ["=HYPERLINK(\"http://evil\")", "+1", "-2 stars", "@x", "fine"], "n": [1, 2, 3, 4, 5]})
    back = pd.read_csv(io.BytesIO(pl.to_csv_bytes(df, excel="pl")), sep=";", encoding="utf-8-sig")
    assert back["text"].tolist() == ["'" + t for t in df["text"][:4]] + ["fine"]  # Excel führt nichts aus
    assert back["n"].tolist() == [1, 2, 3, 4, 5]


# --- Texte ----------------------------------------------------------------------------------------------------
def test_vocabulary_coverage_counts_like_the_model(trained):
    prep, _, _ = trained
    total, known = pl.vocabulary_coverage(prep.vectorizer, ["great food", "zzz qqq great", ""])
    assert total.tolist() == [2, 3, 0] and known.tolist() == [2, 1, 0]


def test_missing_text_column_names_the_raw_column():
    df = pd.DataFrame({"Spalte_1": ["ham", "spam"] * 20, "Spalte_2": [f"msg {i} words" for i in range(40)]})
    out, fitted, _, lineage = pl.apply_column_actions(df, [
        {"column": "Spalte_1", "action": pl.A_KEEP, "new_name": "label"},
        {"column": "Spalte_2", "action": pl.A_KEEP, "new_name": "message"}])
    prep = pl.prepare_text_data(out, "message", "label", column_actions=fitted, lineage=lineage)
    with pytest.raises(pl.MissingColumnsError) as exc:
        pl.prepare_new_texts(pd.DataFrame({"x": ["a"]}), prep.recipe)
    assert exc.value.raw == ["Spalte_2"] and "Spalte_2 (→ message)" in str(exc.value)
    assert pl.template_csv(prep.recipe) == b"Spalte_2\n"


def test_fingerprints_are_limited_and_can_be_left_out(trained, tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "FINGERPRINTS", 50)
    prep = pl.prepare_text_data(yelp_like(800, seed=4), "text", "stars")
    assert len(prep.recipe["train_fingerprints"]) == 50 and prep.recipe["text_stats"]["unique_train"] > 50
    _, model, recipe = trained
    r = {k: v for k, v in recipe.items() if k != "train_fingerprints"}
    folder = pl.save_bundle(pl.make_pipeline(trained[0].vectorizer, model), r, tmp_path / "ohne")
    pipe, loaded = pl.load_bundle(folder)
    _, rep, _ = pl.predict_texts(pipe, loaded, yelp_like(20, seed=1))
    assert "train_overlap" not in rep


def test_fazit_share_uses_the_prediction_column(trained):
    """P(…) ist auf 4 Stellen gerundet: 0.50004 steht dort als 0.5, vorhergesagt ist aber die positive Klasse."""
    prep, model, recipe = trained
    res, rep, ev = pl.predict_texts(pl.make_pipeline(prep.vectorizer, model), recipe, yelp_like(50, seed=3))
    res["P(5)"] = 0.5
    res["Vorhersage"] = "5"
    fz = pl.assess_prediction(recipe, res, rep, ev, threshold=0.5)
    assert any("Vorhergesagt '5': 100.0%" in m for m in fz["meaning"])
