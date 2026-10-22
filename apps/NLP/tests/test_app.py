"""
Klick-Test der Streamlit-App ohne Browser (streamlit.testing.AppTest): alle 10 Schritte einmal durch – mit einer
Yelp-ähnlichen CSV (wie das NLP-Projekt im Kurs) und einer SMS-Datei ohne Kopfzeile (wie SMSSpamCollection).

    pytest -q tests/test_app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import write_sms_file, yelp_like  # noqa: E402

APP = str(ROOT / "app.py")


def button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise KeyError(f"Button '{label}' fehlt: {[b.label for b in at.button]}")


def ok(at, where):
    assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
    assert not at.error, f"{where}: {[e.value for e in at.error]}"


def go(at, index):
    nav = at.sidebar.radio(key="nav")
    nav.set_value(nav.options[index]).run()


def load(at, path):
    go(at, 0)
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(path)).run()
    button(at, "Daten laden").click().run()
    ok(at, "Laden")


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    yelp_like(2500, seed=101).to_csv(tmp_path / "yelp.csv", index=False)
    yelp_like(400, seed=7).to_csv(tmp_path / "neu.csv", index=False)
    return tmp_path


def test_full_workflow_yelp_like_the_course(data):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    ok(at, "Start")
    nav = at.sidebar.radio(key="nav").options
    assert [o.split(" · ")[0] for o in nav] == [str(i) for i in range(1, 11)]  # Schritte 1–10 vollständig
    for i in range(len(nav)):  # alle Seiten ohne Daten
        go(at, i)
        ok(at, f"leere Seite {i}")

    # 1) Laden über Pfad, Stichprobe an/aus
    load(at, data / "yelp.csv")
    assert at.session_state["df_raw"].shape == (2500, 10)
    assert any("Freitext erkannt" in c.value for c in at.caption)
    at.sidebar.checkbox(key="w_use_sample").check().run()
    assert len(at.session_state["df_raw"]) == 250
    at.sidebar.checkbox(key="w_use_sample").uncheck().run()
    assert len(at.session_state["df_raw"]) == 2500

    # 2) Beispiel-Konfiguration (wie im Kurs) + Übernehmen, Schnell-Button für die Wortzahl
    go(at, 1)
    assert any("Yelp-Datensatz" in i.value for i in at.info)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    viz = at.session_state["df_viz"]
    assert list(viz.columns) == ["stars", "text", "cool", "useful", "funny", "text length"]
    assert viz["text length"].iloc[0] == len(viz["text"].iloc[0])
    button(at, "Anzahl Wörter").click().run()
    ok(at, "Wortzahl")
    assert "text words" in at.session_state["df_viz"].columns

    # 3) jedes Diagramm einmal
    go(at, 2)
    for chart in at.radio(key="w_chart").options:
        at.radio(key="w_chart").set_value(chart).run()
        ok(at, f"Diagramm {chart}")
        assert not any("Diagramm nicht möglich" in w.value for w in at.warning), chart
    at.radio(key="w_chart").set_value("Beispieltexte").run()
    button(at, "Andere Texte").click().run()
    ok(at, "andere Beispieltexte")

    # 4) Klassen 1 und 5, CountVectorizer wie im Kurs
    go(at, 3)
    assert at.session_state["w_text"] == "text" and at.session_state["w_label"] == "stars"
    assert at.session_state["w_classes"] == ["1", "5"] and at.session_state["w_pos"] == "5"
    button(at, "Texte vektorisieren").click().run()
    ok(at, "Schritt 4")
    prep = at.session_state["prep"]
    r = prep.recipe
    assert r["classes"] == ["1", "5"] and r["split"]["test_size"] == 0.3 and not r["split"]["stratify"]
    assert r["new_columns"] and r["vectorizer"]["kind"] == "count"
    button(at, "Anderer Text").click().run()
    ok(at, "Beispieltext als Zahlen")
    at.number_input(key="w_min_df").set_value(2).run()
    assert any("seit dem Vektorisieren geändert" in i.value for i in at.info)
    at.number_input(key="w_min_df").set_value(1).run()

    # 5) Exporte
    go(at, 4)
    for what in ("step2", "split", "vocab"):
        at.radio(key="w_exp_what").set_value(what).run()
        button(at, "CSV erstellen").click().run()
        ok(at, f"Export {what}")
    assert at.session_state["_export"][3][1] == 3  # Vokabular: Wort, Texte, Anzahl
    at.text_input(key="w_exp_path").input(str(data / "exports")).run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export in Ordner")
    assert list((data / "exports").glob("*_vokabular.csv"))

    # 6) Naive Bayes (Beispiel-Konfiguration) + Vergleich aller drei Modelle
    go(at, 5)
    assert at.session_state["w_model"] == "nb"
    button(at, "Training starten").click().run()
    ok(at, "Training")
    info = at.session_state["train_info"]
    assert info["model"] == "nb" and info["test_error"] < info["baseline_error"]
    button(at, "Modelle vergleichen").click().run()
    ok(at, "Vergleich")
    table, _ = at.session_state["compare"]
    assert list(table["model"]) == ["nb", "logreg", "linsvc"]

    # 7) Kurve + bester Wert übernehmen (falls vorgeschlagen)
    go(at, 6)
    ok(at, "Schritt 7")
    best = [b for b in at.button if b.label == "Besten Wert in Schritt 6 übernehmen"]
    if best:
        best[0].click().run()
        ok(at, "bester Wert")
        assert at.session_state["w_alpha"] in at.session_state["history"]["curve"]["values"]

    # 8) Schwelle, Einzelfall, eigener Text
    go(at, 7)
    at.slider(key="w_threshold").set_value(0.6).run()
    button(at, "Zufälligen Testtext ziehen").click().run()
    ok(at, "Einzelfall")
    at.text_area(key="w_try_text").input("The food was great and the staff friendly")  # Formular: erst beim Absenden
    button(at, "Einordnen").click().run()
    ok(at, "eigener Text")
    assert any(":blue[**great**]" in m.value for m in at.markdown)
    button(at, "Beispiel aus den Testdaten einsetzen").click().run()
    ok(at, "Beispiel eingesetzt")
    assert at.session_state["w_try_text"]

    # 9) speichern (.joblib) und wieder laden
    go(at, 8)
    at.radio(key="w_save_fmt").set_value(".joblib").run()
    at.text_input(key="w_save_name").input("yelp_nb").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern")
    assert (data / "models" / "yelp_nb" / "model.joblib").exists()
    button(at, "Laden").click().run()
    ok(at, "Laden Modell")
    assert at.session_state["bundle"]["recipe"]["threshold"] == pytest.approx(0.6)

    # 10) einzelner Text und CSV (mit Fazit) – auch die Trainingsdaten noch einmal
    go(at, 9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    assert at.slider(key="w_pred_thr").value == pytest.approx(0.6)
    at.text_area(key="w_pred_text").input("Terrible rude staff, the worst food")  # Formular: erst beim Absenden
    button(at, "Einordnen").click().run()
    ok(at, "einzelner Text")
    assert any(m.label == "Vorhersage" and m.value == "1" for m in at.metric)
    at.radio(key="w_pred_mode").set_value("CSV-Datei").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(data / "neu.csv")).run()
    button(at, "Vorhersage berechnen").click().run()
    ok(at, "Vorhersage CSV")
    res = at.session_state["pred_result"]["res"]
    assert len(res) == 400 and {"Vorhersage", "P(5)", "Hinweis"} <= set(res.columns)
    assert any(h.value == "Fazit" for h in at.subheader)
    at.text_input(key="w_pred_path").input(str(data / "yelp.csv")).run()  # Trainingsdaten erneut einordnen
    button(at, "Vorhersage berechnen").click().run()
    ok(at, "Vorhersage Trainingsdaten")
    assert at.session_state["pred_result"]["rep"]["train_overlap"]["share"] > 0.15
    assert any("wortgleich aus dem Training" in m.value for m in at.markdown)


def test_sms_file_without_header_and_svc(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    write_sms_file(tmp_path / "SMSSpamCollection", 1500)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "SMSSpamCollection")
    src = at.session_state["source"]
    assert src["header"] is False and src["sep"] == "\t"
    assert any("keine Kopfzeile" in i.value for i in at.info)

    go(at, 1)
    assert any("SMSSpamCollection" in i.value for i in at.info)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    assert list(at.session_state["df_viz"].columns) == ["label", "message", "length"]

    go(at, 3)
    assert at.session_state["w_vec"] == "tfidf" and at.session_state["w_stop"] == "english"
    button(at, "Texte vektorisieren").click().run()
    ok(at, "Schritt 4")
    r = at.session_state["prep"].recipe
    assert r["classes"] == ["ham", "spam"] and r["meta"]["header"] is False
    assert any("wortgleich" in w.value for w in at.warning)  # SMS-Vorlagen: doppelte Nachrichten

    go(at, 5)
    at.radio(key="w_model").set_value("linsvc").run()
    at.checkbox(key="w_cw").check().run()
    at.radio(key="w_val").set_value("split").run()
    button(at, "Training starten").click().run()
    ok(at, "Training LinearSVC")
    assert at.session_state["history"]["curve"]["validation"] == "split"
    go(at, 6)
    ok(at, "Kurve")
    go(at, 7)
    ok(at, "Evaluation")

    go(at, 8)
    at.text_input(key="w_save_name").input("sms_svc").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern")
    scripts = [b for b in at.get("download_button") if b.label.startswith("Komplette Pipeline")]
    assert scripts
    go(at, 9)
    at.radio(key="w_pred_mode").set_value("CSV-Datei").run()
    assert any("ohne Kopfzeile trainiert" in c.value for c in at.caption)
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "SMSSpamCollection")).run()
    button(at, "Vorhersage berechnen").click().run()
    ok(at, "Vorhersage gleiche Datei (ohne Kopfzeile)")
    assert "P(spam)" in at.session_state["pred_result"]["res"].columns


def test_guards_and_errors(tmp_path, monkeypatch):
    """Label mit zu vielen Werten, nur eine Klasse gewählt, kein Freitext, Überschreib-Schutz beim Speichern,
    Modell ohne Prüfsummen, Fehler beim Zeichnen als Hinweis."""
    sys.path.insert(0, str(ROOT))
    import plots

    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(1)
    n = 300
    good, bad = ["great", "tasty", "lovely"], ["awful", "cold", "rude"]
    y = rng.choice(["pos", "neg"], n)
    text = [" ".join(rng.choice(good if v == "pos" else bad, 3)) + " food here" for v in y]
    pd.DataFrame({"id": [f"r{i}" for i in range(n)], "text": text, "y": y, "score": rng.normal(size=n)}).to_csv(
        tmp_path / "d.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "d.csv")
    go(at, 1)
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    assert "id" not in at.session_state["df_viz"].columns  # ID-Spalte als Vorschlag entfernt

    go(at, 3)
    at.selectbox(key="w_label").set_value("score").run()  # 300 verschiedene Werte → kein Label
    assert any("kein Label" in e.value for e in at.error)
    assert button(at, "Texte vektorisieren").disabled
    at.selectbox(key="w_label").set_value("y").run()
    at.multiselect(key="w_classes").set_value(["neg"]).run()
    assert any("mindestens 2 Klassen" in w.value for w in at.warning)
    assert button(at, "Texte vektorisieren").disabled
    at.multiselect(key="w_classes").set_value([]).run()
    button(at, "Texte vektorisieren").click().run()
    ok(at, "Schritt 4")

    # Fehler beim Zeichnen → Hinweis statt Absturz der Seite
    go(at, 2)

    def boom(*args, **kwargs):
        raise TypeError("Testfehler")

    with monkeypatch.context() as m:
        m.setattr(plots, "fig_top_terms", boom)
        at.radio(key="w_chart").set_value("Häufigste Wörter").run()
        assert not at.exception
        assert any("Diagramm nicht möglich" in w.value for w in at.warning)

    def train():
        go(at, 5)
        button(at, "Training starten").click().run()
        ok(at, "Training")

    train()
    go(at, 8)
    at.text_input(key="w_save_name").input("modell").run()
    at.checkbox(key="w_save_fp").uncheck().run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern ohne Prüfsummen")
    saved = json.loads((tmp_path / "models" / "modell" / "preprocessing.json").read_text(encoding="utf-8"))
    assert "train_fingerprints" not in saved and saved["text_col"] == "text"

    train()  # neues Modell, gleicher Name → Warnung, Speichern erst nach Bestätigung
    go(at, 8)
    at.text_input(key="w_save_name").input("modell").run()
    assert any("liegt schon ein Modell" in w.value for w in at.warning)
    assert button(at, "Speichern").disabled
    at.checkbox(key="overwrite_modell").check().run()
    assert not button(at, "Speichern").disabled


def test_no_free_text_column(tmp_path, monkeypatch):
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    pd.DataFrame({"a": range(50), "b": [1.5] * 50}).to_csv(tmp_path / "zahlen.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "zahlen.csv")
    assert any("Keine Spalte mit Freitext" in w.value for w in at.warning)
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    assert any("Keine Textspalte" in e.value for e in at.error)
    assert not at.exception


def test_config_roundtrip(tmp_path, monkeypatch):
    """Konfiguration exportieren und in einer neuen Sitzung importieren – Schritt 2, 4 und 6 kommen zurück."""
    sys.path.insert(0, str(ROOT))
    monkeypatch.setenv("NLP_TEMPLATE_MODELS", str(tmp_path / "models"))
    yelp_like(800, seed=3).to_csv(tmp_path / "y.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "y.csv")
    go(at, 1)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    go(at, 3)
    button(at, "Texte vektorisieren").click().run()
    go(at, 5)
    at.radio(key="w_model").set_value("logreg").run()
    at.number_input(key="w_log_c").set_value(7.5).run()
    exported = at.session_state  # Export wie in der App: dieselbe Funktion über den Zustand
    cfg = {"template_config": 1, "app": "nlp",
           "step2": exported["ed_actions__draft"][["Spalte", "Aktion", "Parameter", "Neuer Name"]].to_dict("records"),
           "new_columns": exported["new_cols"],
           "widgets": {k: exported[k] for k in ("w_text", "w_label", "w_classes", "w_vec", "w_model", "w_log_c")}}
    (tmp_path / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")

    at2 = AppTest.from_file(APP, default_timeout=300)
    at2.run()
    load(at2, tmp_path / "y.csv")
    import pipeline as pl  # wie import_config der App (der Upload selbst lässt sich in AppTest nicht klicken)

    clean, skipped = pl.sanitize_config(json.loads((tmp_path / "cfg.json").read_text(encoding="utf-8")))
    assert not skipped and clean["widgets"]["w_log_c"] == 7.5 and clean["widgets"]["w_classes"] == ["1", "5"]
    for k, v in clean["widgets"].items():
        at2.session_state[k] = v
    at2.session_state["_classes_for_label"] = "stars"
    at2.session_state["new_cols"] = clean["new_columns"]
    go(at2, 1)
    button(at2, "Übernehmen").click().run()
    ok(at2, "Schritt 2 nach Import")
    go(at2, 3)
    button(at2, "Texte vektorisieren").click().run()
    ok(at2, "Schritt 4 nach Import")
    assert at2.session_state["prep"].recipe["classes"] == ["1", "5"]
    go(at2, 5)
    assert at2.session_state["w_model"] == "logreg" and at2.session_state["w_log_c"] == 7.5
