"""
Regressionstests zur Prüfung der NLP-Version vor der Auslieferung (AUDIT.md, „Znalezione w trakcie przeglądu“):
seltene Klassen und LinearSVC, Validierungskurve mit sehr wenigen Texten, CSV für Excel in Schritt 8 und 10,
Meldungen mit Namen aus den Daten wörtlich.

    pytest -q tests/test_review_regressions.py
"""
import io
import sys
from dataclasses import replace
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
from synthetic import news_like  # noqa: E402

APP = str(ROOT / "app.py")


def rare_news(rare_rows=1, seed=4):
    """Mehrklassen-Daten mit einer seltenen Klasse 'Kultur' (rare_rows Zeilen insgesamt)."""
    df = news_like(300, seed)
    df.loc[list(range(rare_rows)), "thema"] = "Kultur"
    return df


def prep_with_single_training_text():
    """Split so wählen, dass 'Kultur' genau einmal im Training steht (ohne stratify, wie in der App möglich)."""
    df = rare_news(2)
    for seed in range(100):
        try:
            prep = pl.prepare_text_data(df, "text", "thema", stratify=False, test_size=0.2, random_state=seed)
        except ValueError:  # Kultur nur im Test
            continue
        k = prep.recipe["classes"].index("Kultur")
        if int((prep.y_train == k).sum()) == 1:
            return prep
    raise AssertionError("kein passender Seed gefunden")


# --- R1: LinearSVC mit einer Klasse, die nur einen Trainingstext hat -------------------------------------------
def test_single_training_text_warns_in_step4_and_blocks_only_linearsvc():
    prep = prep_with_single_training_text()
    assert any("Sehr wenige Trainingstexte für Kultur" in w and "LinearSVC nicht möglich" in w for w in prep.warnings)
    with pytest.raises(ValueError, match="mindestens 2 Trainingstexte je Klasse – Kultur: nur einer"):
        pl.train_model(pl.TextModelConfig(model=pl.MODEL_SVC, curve=False), prep)
    for m in (pl.MODEL_NB, pl.MODEL_LOGREG):  # die anderen beiden Modelle trainieren normal
        pl.train_model(pl.TextModelConfig(model=m, curve=False), prep)
    table = pl.compare_models(pl.TextModelConfig(), prep)
    assert list(table["model"]) == [pl.MODEL_NB, pl.MODEL_LOGREG]
    assert "Kultur" in table.attrs["skipped"][pl.MODEL_SVC]
    code = cg.compare_code(pl.TextModelConfig(), prep.recipe["task"], 5, list(table["model"]))
    assert "LinearSVC(" not in code and "Nicht im Vergleich: Lineare SVM (LinearSVC)" in code
    assert "LinearSVC(" in cg.compare_code(pl.TextModelConfig(), prep.recipe["task"], 5)  # Standard: alle drei


def test_two_training_texts_are_enough_for_linearsvc():
    df = rare_news(3)
    prep = pl.prepare_text_data(df, "text", "thema", stratify=True, test_size=0.2, random_state=1)
    k = prep.recipe["classes"].index("Kultur")
    assert int((prep.y_train == k).sum()) == 2
    assert any("Sehr wenige Trainingstexte" in w and "LinearSVC" not in w for w in prep.warnings)
    _, _, info = pl.train_model(pl.TextModelConfig(model=pl.MODEL_SVC, curve=False), prep)
    assert info["calib_cv"] == 2


# --- R2: Validierungskurve „Anteil aus Training“, wenn im Trainingsteil nur eine Klasse bleibt --------------------
def test_validation_split_with_one_class_left_gives_a_clear_message():
    df = pd.DataFrame({"t": [f"text number {i} good" for i in range(19)] + ["bad bad"], "y": ["a"] * 19 + ["b"]})
    prep = pl.prepare_text_data(df, "t", "y", stratify=True, test_size=0.2)
    hit = None
    for seed in range(200):
        cfg = pl.TextModelConfig(model=pl.MODEL_LOGREG, validation="split", validation_split=0.2, seed=seed)
        if len(np.unique(pl.curve_data(cfg, prep)[1])) < 2:
            hit = cfg
            break
    assert hit is not None
    for m in (pl.MODEL_NB, pl.MODEL_LOGREG):  # vorher: „This solver needs samples of at least 2 classes“ (scikit-learn)
        with pytest.raises(ValueError, match="Validierungskurve nicht möglich"):
            pl.train_model(replace(hit, model=m), prep)
    with pytest.raises(ValueError, match="LinearSVC braucht"):  # 'b' hat nur einen Trainingstext (R1)
        pl.train_model(replace(hit, model=pl.MODEL_SVC), prep)
    pl.train_model(replace(hit, curve=False), prep)  # ohne Kurve geht es


# --- R1 + R3 + R4 in der Oberfläche ------------------------------------------------------------------------------
def button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise KeyError(f"Button '{label}' fehlt: {[b.label for b in at.button]}")


def go(at, index):
    nav = at.sidebar.radio(key="nav")
    nav.set_value(nav.options[index]).run()


def test_app_rare_class_linearsvc_comparison_and_excel_downloads(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    prep = prep_with_single_training_text()
    seed = prep.recipe["split"]["random_state"]
    df = rare_news(2)
    df["thema"] = df["thema"].replace({"Kultur": "*Kultur*"})  # Markdown im Klassennamen
    df.to_csv(tmp_path / "rare.csv", index=False)
    new = news_like(40, 9)
    new.loc[0, "text"] = "=HYPERLINK(\"http://example.com\",\"x\") Tor Spiel"  # Text wie eine Excel-Formel
    new.to_csv(tmp_path / "neu.csv", index=False)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    go(at, 0)
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "rare.csv")).run()
    button(at, "Daten laden").click().run()
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    at.selectbox(key="w_label").set_value("thema").run()
    at.checkbox(key="w_stratify").uncheck().run()
    at.slider(key="w_test_size").set_value(0.2).run()
    at.number_input(key="w_split_seed").set_value(seed).run()
    button(at, "Texte vektorisieren").click().run()
    assert not at.exception and not at.error
    warn = [w.value for w in at.warning if "Sehr wenige Trainingstexte" in w.value]
    assert warn and "\\*Kultur\\*" in warn[0]  # Klassenname wörtlich, nicht kursiv

    go(at, 5)
    at.radio(key="w_model").set_value("linsvc").run()
    assert any("LinearSVC arbeitet am besten mit TF-IDF" in c.value for c in at.caption)  # CountVectorizer gewählt
    button(at, "Training starten").click().run()
    assert not at.exception
    assert any("mindestens 2 Trainingstexte" in e.value for e in at.error)
    button(at, "Modelle vergleichen").click().run()  # der Vergleich bleibt erreichbar und läuft ohne LinearSVC
    assert not at.exception
    table, _ = at.session_state["compare"]
    assert list(table["model"]) == ["nb", "logreg"]
    assert any("fehlt im Vergleich" in i.value for i in at.info)

    at.radio(key="w_model").set_value("nb").run()
    button(at, "Training starten").click().run()
    assert not at.exception and not at.error
    go(at, 7)
    labels = [b.label for b in at.get("download_button")]
    assert "Testtexte mit Vorhersage als CSV" in labels and "CSV für Excel" in labels

    go(at, 9)
    at.radio(key="w_pred_mode").set_value("CSV-Datei").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "neu.csv")).run()
    button(at, "Vorhersage berechnen").click().run()
    assert not at.exception and not at.error
    labels = [b.label for b in at.get("download_button")]
    assert "Ergebnis als CSV" in labels and "CSV für Excel" in labels
    res = at.session_state["pred_result"]["res"]
    back = pd.read_csv(io.StringIO(pl.to_csv_bytes(res, excel="de").decode("utf-8-sig")), sep=";")
    assert back.loc[0, "text"].startswith("'=HYPERLINK")  # in der Excel-Datei entschärft
    assert res.loc[0, "text"].startswith("=HYPERLINK")  # im Ergebnis selbst unverändert


def test_no_signal_verdict_names_the_parameter_when_the_curve_is_better():
    """Naive Bayes mit TF-IDF und alpha = 1: kaum besser als naiv – die Kurve zeigt aber bessere Werte. Die Einordnung
    darf dann nicht „in diesen Texten steckt kaum ein Signal“ sagen."""
    curve = {"values": [0.01, 0.1, 1.0], "chosen": 1.0, "train": [0.0, 0.02, 0.45], "val": [0.08, 0.1, 0.5],
             "baseline": 0.5, "n_val": 300, "validation": "split", "param": "alpha", "best_index": 0}
    info = {"train_error": 0.45, "test_error": 0.5, "baseline_error": 0.5}
    a = pl.curve_assessment(curve, info, pl.TextModelConfig())
    assert a["verdict"] == "nosignal" and a["suggest"] == {"param": "alpha", "value": 0.01}
    assert a["notes"][0].startswith("**Mit alpha = 1 nicht besser als eine naive Vorhersage")
    assert not any("kaum ein Signal" in n for n in a["notes"])
    flat = dict(curve, val=[0.5, 0.5, 0.5], best_index=2)  # wirklich kein Signal → wie bisher
    assert "kaum ein Signal" in pl.curve_assessment(flat, info, pl.TextModelConfig())["notes"][0]


def test_polish_model_names_are_polish():
    import i18n
    try:
        i18n.set_lang("pl")
        assert [pl.model_name(m) for m in pl.MODELS] == ["Naiwny Bayes (MultinomialNB)", "Regresja logistyczna",
                                                         "Liniowa SVM (LinearSVC)"]
    finally:
        i18n.set_lang("de")
