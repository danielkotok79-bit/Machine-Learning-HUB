"""
Regressionstests zu Audit 6 (AUDIT.md, „Audyt 6“): Rezept-Prüfung beim Laden, leere Labels, kurze Texte in
Schritt 2, Erkennung der Kopfzeile, Confusion Matrix bei vielen Klassen, Code der Beispieltexte, naive Vorhersage
aus den Trainingsdaten.

    pytest -q tests/test_audit6_regressions.py
"""
import io
import json
import shutil
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
import plots  # noqa: E402
import test_codegen_parity as par  # noqa: E402
from synthetic import news_like, sms_like, yelp_like  # noqa: E402

APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Gespeichertes Modell (Yelp 1 gegen 5 Sterne) – Vorlage für manipulierte Rezepte."""
    prep = pl.prepare_text_data(yelp_like(800, seed=3), "text", "stars", classes=["1", "5"])
    cfg = pl.TextModelConfig(curve=False)
    model, _, info = pl.train_model(cfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info, {"accuracy": 0.95, "f1_macro": 0.9, "roc_auc": 0.99}, 0.5)
    folder = tmp_path_factory.mktemp("m") / "gut"
    return pl.save_bundle(pl.make_pipeline(prep.vectorizer, model), recipe, folder)


def zip_with(folder, **changes) -> bytes:
    r = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    r.update(changes)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(folder / "model.skops", "model.skops")
        zf.writestr("preprocessing.json", json.dumps(r))
    return buf.getvalue()


# --- B1: jeder Eintrag, den Schritt 9 und 10 benutzen, wird beim Laden geprüft ---------------------------------
@pytest.mark.parametrize("key, value", [
    ("classes_selected", 5),  # vorher: TypeError im Fazit von Schritt 10 (Seite bricht ab)
    ("classes_selected", ["2", "3"]),  # vorher: Fazit nannte falsche Klassen
    ("metrics", {"accuracy": "0.9"}),  # vorher: TypeError im Fazit
    ("metrics", {"accuracy": 0.9, "roc_auc": None}),
    ("threshold", 0.999),  # vorher: Schritt 10 brach ab (Schieberegler 0.05–0.95)
    ("threshold", 0.01),
    ("column_actions", [{"action": "behalten"}]),  # vorher: KeyError 'column' bei der Vorhersage
    ("column_actions", [{"column": "text", "action": pl.A_TOPN, "fitted": "x"}]),  # vorher: AttributeError
    ("column_actions", [{"column": "text", "action": "rm -rf"}]),
    ("new_columns", [{"kind": pl.N_COPY, "source": "text"}]),
    ("new_columns", [{"name": "x", "kind": "Python", "definition": "1"}]),
    ("lineage", {"text": 5}),
    ("split", "x"),
    ("missing_text", "egal"),
])
def test_broken_recipe_entries_fail_at_loading(bundle, key, value):
    with pytest.raises(ValueError, match=f"unvollständig oder beschädigt \\(Eintrag: {key}\\)"):
        pl.load_bundle_zip(zip_with(bundle, **{key: value}))


def test_valid_recipe_still_loads_and_predicts(bundle):
    pipe, recipe = pl.load_bundle_zip(pl.bundle_zip_bytes(bundle))
    res, rep, ev = pl.predict_texts(pipe, recipe, yelp_like(150, seed=9))
    fz = pl.assess_prediction(recipe, res, rep, ev)
    assert rep["label_unknown"] > 0 and any("nur mit den Klassen 1, 5" in a for a in fz["advice"])


def test_vectorizer_in_recipe_must_match_the_pipeline(bundle):
    r = json.loads((bundle / "preprocessing.json").read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="Vektorisierer und preprocessing.json passen nicht zusammen"):
        pl.load_bundle_zip(zip_with(bundle, vectorizer={**r["vectorizer"], "kind": "tfidf"}))


def test_oversized_recipe_is_refused(bundle, monkeypatch):
    monkeypatch.setattr(pl, "MAX_RECIPE_BYTES", 1000)
    with pytest.raises(ValueError, match="preprocessing.json ist zu groß"):
        pl.load_bundle(bundle)
    assert not pl.zip_needs_pickle(pl.bundle_zip_bytes(bundle))  # zu groß → wie ohne Rezept: model.skops


def test_app_refuses_broken_models_without_crashing(bundle, tmp_path, monkeypatch):
    models = tmp_path / "models"
    for name, change in (("sel5", {"classes_selected": 5}), ("schwelle", {"threshold": 0.999})):
        shutil.copytree(bundle, models / name)
        r = json.loads((models / name / "preprocessing.json").read_text(encoding="utf-8"))
        r.update(change)
        (models / name / "preprocessing.json").write_text(json.dumps(r), encoding="utf-8")
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(models))
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    nav = at.sidebar.radio(key="nav")
    nav.set_value(nav.options[8]).run()
    for name in ("sel5", "schwelle"):
        at.session_state["w_bundle_sel"] = name
        at.run()
        next(b for b in at.button if b.label == "Laden").click().run()
        assert not at.exception
        assert any("unvollständig oder beschädigt" in e.value for e in at.error)
        assert "bundle" not in at.session_state
    nav.set_value(nav.options[9]).run()
    assert not at.exception


# --- B2: Label aus nur Leerzeichen = kein Label (keine Klasse '') ----------------------------------------------
def blank_sms(n=600, seed=2, blanks=9):
    raw = sms_like(n, seed)
    raw.loc[raw.index[:blanks], "label"] = "  "
    return raw


def test_blank_labels_are_missing_not_a_class():
    raw = blank_sms()
    assert pl.label_values(raw["label"]) == ["ham", "spam"]
    assert int(pl.missing_label(raw["label"]).sum()) == 9
    prep = pl.prepare_text_data(raw, "message", "label")
    assert prep.recipe["classes"] == ["ham", "spam"] and prep.recipe["task"] == pl.T_BIN  # vorher: 3 Klassen
    assert any("9 Zeilen ohne Label entfernt" in line for line in prep.log)


def test_blank_labels_in_new_data_are_not_unknown_classes():
    prep = pl.prepare_text_data(sms_like(400, 1), "message", "label")
    model, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep)
    recipe = pl.finalize_recipe(prep.recipe, pl.TextModelConfig(curve=False))
    new = blank_sms(80, 5, blanks=10)
    res, rep, ev = pl.predict_texts(pl.make_pipeline(prep.vectorizer, model), recipe, new)
    assert rep["label_unknown"] == 0 and ev["n"] == 70  # vorher: 10 „unbekannte Klassen“
    assert not any("nicht kennt" in q["text"] for q in pl.assess_prediction(recipe, res, rep, ev)["quality"])


def test_generated_code_drops_blank_labels_like_the_app(tmp_path):
    res = par.run_both(tmp_path, blank_sms(), {}, "message", "label")
    assert res[0].recipe["classes"] == ["ham", "spam"]
    par.assert_same(*res)


# --- B3: kurze Texte werden in Schritt 2 nicht zum Entfernen vorgeschlagen ----------------------------------------
def test_short_texts_are_kept_and_ids_dropped():
    rng = np.random.default_rng(1)
    words = ["great", "bad", "awful", "nice", "love", "hate", "super", "poor", "best", "food", "service", "price"]
    titles = [" ".join(rng.choice(words, rng.integers(1, 4))) for _ in range(600)]  # Ø 2 Wörter, ~400 verschiedene
    df = pd.DataFrame({"title": titles, "score": rng.choice([1, 5], 600), "review_id": [f"r{i}x" for i in range(600)]})
    assert pl.mean_words(df["title"]) < 3 and df["title"].nunique() > 50
    actions = dict(zip(*pl.column_profile(df)[["Spalte", "Aktion"]].to_numpy().T))
    assert actions == {"title": pl.A_KEEP, "score": pl.A_KEEP, "review_id": pl.A_DROP}  # vorher: title entfernen


def test_app_offers_short_texts_instead_of_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(2)
    y = rng.choice(["pos", "neg"], 300)
    titles = [" ".join(rng.choice(["great", "love", "best"] if v == "pos" else ["bad", "awful", "worst"],
                                  rng.integers(1, 3))) + f" no{i}" for i, v in enumerate(y)]
    pd.DataFrame({"title": titles, "sentiment": y}).to_csv(tmp_path / "kurz.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "kurz.csv")).run()
    next(b for b in at.button if b.label == "Daten laden").click().run()
    assert any("Nur kurze Texte gefunden" in i.value for i in at.info)
    assert not any("Keine Spalte mit Freitext" in w.value for w in at.warning)
    prof = at.session_state["ed_actions__base"]
    assert dict(zip(prof["Spalte"], prof["Aktion"]))["title"] == pl.A_KEEP


# --- B4: Kopfzeile nur dann als Datenzeile, wenn ihr Wert in der Spalte häufig ist ---------------------------------
def test_header_is_kept_for_rare_value_and_repeated_header():
    c = pd.DataFrame({"text": [f"text number {i} with words" for i in range(300)],
                      "typ": ["typ" if i == 5 else "a" if i % 2 else "b" for i in range(300)]})
    df, _ = pl.read_csv(c.to_csv(index=False).encode())
    assert list(df.columns) == ["text", "typ"] and df.attrs["csv"]["header"] is True  # vorher: Spalte_1, Spalte_2
    raw = sms_like(400, 1).to_csv(index=False) + sms_like(300, 2).to_csv(index=False)  # zwei Exporte aneinander
    df2, _ = pl.read_csv(raw.encode())
    assert list(df2.columns) == ["label", "message"] and len(df2) == 701


def test_file_without_header_is_still_detected():
    s = pd.concat([pd.DataFrame({"label": ["spam"], "message": ["WIN a prize now"]}), sms_like(500, 3)])
    df, _ = pl.read_csv(s.to_csv(sep="\t", header=False, index=False).encode())
    assert list(df.columns) == ["Spalte_1", "Spalte_2"] and len(df) == 501


# --- C1: Confusion Matrix und Klassen-Balken bei vielen Klassen ----------------------------------------------------
def test_confusion_matrix_size_is_limited():
    import matplotlib.pyplot as plt

    fig = plots.fig_confusion(np.ones((50, 50), dtype=int), [f"Klasse {i}" for i in range(50)])
    w, h = fig.get_size_inches()
    assert w <= plots.CM_MAX_INCHES[0] and h <= plots.CM_MAX_INCHES[1]  # vorher: 67.6 × 56.8 Zoll
    plt.close(fig)
    fig = plots.fig_confusion(np.array([[5, 1], [2, 7]]), ["ham", "spam"])
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "5\n83%" in texts  # wenige Klassen: Anzahl und Anteil wie bisher
    plt.close(fig)
    fig = plots.fig_class_counts(pd.Series([f"k{i % 30}" for i in range(300)]), None)
    assert fig.axes[0].get_xticklabels()[0].get_rotation() == 45
    plt.close(fig)


# --- C2: Code der Beispieltexte = App, auch mit fehlenden Texten ---------------------------------------------------
def test_example_texts_code_matches_the_app_with_missing_texts():
    df = news_like(200, 3)
    df.loc[df.index[::4], "text"] = np.nan
    disp = pl.label_display(df["thema"])
    app = {v: df.loc[disp == v, "text"].dropna().sample(min(3, len(df.loc[disp == v, "text"].dropna())),
                                                       random_state=7).tolist() for v in disp.dropna().unique()}
    printed = []
    exec(cg.chart_code("examples", text="text", group="thema", n=3, seed=7),  # noqa: S102
         {"df": df, "print": lambda *a: printed.append(a)})
    assert {v: lst for v, lst in printed} == app  # vorher: NaN unter den Beispielen, andere Texte
    ns = {"df": df}
    exec(cg.chart_code("examples", text="text", n=500, seed=7), ns)  # noqa: S102 – vorher: ValueError
    assert len(ns["texts"]) == df["text"].notna().sum()


# --- C3: naive Vorhersage = häufigste Klasse der Trainingsdaten -------------------------------------------------
def test_naive_baseline_uses_the_training_majority():
    prep = pl.prepare_text_data(yelp_like(1500, seed=4), "text", "stars", classes=["1", "5"])
    model, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep)
    recipe = pl.finalize_recipe(prep.recipe, pl.TextModelConfig(curve=False))
    assert pl.training_majority(recipe) == 1  # '5' ist im Training häufiger
    new = yelp_like(2000, seed=11)
    new = new[new["stars"] == 1].head(60)  # nur 1 Stern: die naive Vorhersage ('5') liegt immer falsch
    _, _, ev = pl.predict_texts(pl.make_pipeline(prep.vectorizer, model), recipe, new)
    assert ev["baseline"] == 0.0  # vorher: 1.0 (häufigste Klasse der neuen Daten)
    ev2 = pl.evaluate(prep.y_test, pl.predict_scores(model, prep.X_test, pl.T_BIN), pl.T_BIN, recipe["classes"],
                      majority=1)
    assert ev2["baseline"] == pytest.approx(float(np.mean(prep.y_test == 1)))


def test_polish_recipe_term_is_consistent():
    from locales import CATALOG

    assert CATALOG["Modell + Rezept als ZIP"][1] == "Model + przepis jako ZIP"
