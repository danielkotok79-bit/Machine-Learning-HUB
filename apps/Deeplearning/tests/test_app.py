"""
Klick-Test der Streamlit-App ohne Browser (streamlit.testing.AppTest): alle 10 Schritte einmal durch.

    pytest -q tests/test_app.py
"""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import lending_club_like  # noqa: E402

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


@pytest.fixture()
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("DL_TEMPLATE_MODELS", str(tmp_path / "models"))
    lending_club_like(2500, seed=101).to_csv(tmp_path / "train.csv", index=False)
    lending_club_like(400, seed=7).to_csv(tmp_path / "neu.csv", index=False)
    return tmp_path


def test_full_workflow(data):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    ok(at, "Start")
    nav = at.sidebar.radio(key="nav").options
    assert [o.split(" · ")[0] for o in nav] == [str(i) for i in range(1, 11)]  # Schritte 1–10 vollständig
    for i in range(len(nav)):  # alle Seiten ohne Daten
        go(at, i)
        ok(at, f"leere Seite {i}")

    # 1) Laden über Pfad
    go(at, 0)
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(data / "train.csv")).run()
    button(at, "Daten laden").click().run()
    ok(at, "Laden")
    assert at.session_state["df_raw"].shape == (2500, 27)

    # Stichprobe an/aus (Seitenleiste)
    at.sidebar.checkbox(key="w_use_sample").check().run()
    assert len(at.session_state["df_raw"]) == 250
    at.sidebar.checkbox(key="w_use_sample").uncheck().run()
    assert len(at.session_state["df_raw"]) == 2500

    # 2) Beispiel-Konfiguration + Übernehmen, dann neue Spalten
    go(at, 1)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    assert {"zip_code", "earliest_cr_year"} <= set(at.session_state["df_viz"].columns)
    at.radio(key="w_new_kind").set_value("Formel").run()
    at.text_input(key="w_new_name").input("kredit_einkommen").run()
    at.text_input(key="w_new_formula").input("loan_amnt / annual_inc").run()
    button(at, "Spalte hinzufügen").click().run()
    ok(at, "neue Spalte Formel")
    at.radio(key="w_new_kind").set_value("Werte zuordnen").run()
    at.text_input(key="w_new_name").input("loan_repaid").run()
    at.selectbox(key="w_new_src").set_value("loan_status").run()
    at.text_input(key="w_new_map").input("Fully Paid=1; Charged Off=0").run()
    button(at, "Spalte hinzufügen").click().run()
    ok(at, "neue Spalte Zuordnung")
    viz = at.session_state["df_viz"]
    assert {"kredit_einkommen", "loan_repaid"} <= set(viz.columns)
    assert "kredit_einkommen" in at.session_state["w_features"] or "w_features" not in at.session_state
    at.text_input(key="w_new_name").input("kaputt").run()
    at.radio(key="w_new_kind").set_value("Formel").run()
    at.text_input(key="w_new_formula").input("gibts_nicht * 2").run()
    button(at, "Spalte hinzufügen").click().run()
    assert any("Unbekannte Spalte" in e.value for e in at.error)  # verständliche Meldung statt Absturz

    # 3) Auswahl der Spalten jetzt auf Seite 3 – und jedes Diagramm einmal
    go(at, 2)
    at.multiselect(key="w_viz_cols").set_value(["loan_amnt", "int_rate", "kredit_einkommen", "loan_status"]).run()
    ok(at, "Auswahl Seite 3")
    for chart in at.radio(key="w_chart").options:
        at.radio(key="w_chart").set_value(chart).run()
        ok(at, f"Diagramm {chart}")
        assert not any("Diagramm nicht möglich" in w.value for w in at.warning), chart

    # 4) vorbereiten (loan_repaid stammt aus loan_status → darf kein Feature sein)
    go(at, 3)
    assert "loan_repaid" not in at.session_state["w_features"]
    button(at, "Daten für ML vorbereiten").click().run()
    ok(at, "Schritt 4")
    prep = at.session_state["prep"]
    assert prep.recipe["task"] == "binary" and prep.recipe["classes"][1] == "Fully Paid"
    assert "kredit_einkommen" in prep.recipe["features"] and prep.recipe["new_columns"]

    # 5) Daten speichern
    go(at, 4)
    for what in ("step2", "ml"):
        at.radio(key="w_exp_what").set_value(what).run()
        button(at, "CSV erstellen").click().run()
        ok(at, f"Export {what}")
        assert at.session_state["_export"][3][0] > 2000
    at.text_input(key="w_exp_path").input(str(data / "exports")).run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export in Ordner")
    assert list((data / "exports").glob("*_ml_daten.csv"))

    # 6) trainieren (kurz)
    go(at, 5)
    at.number_input(key="w_epochs").set_value(3).run()
    button(at, "Training starten").click().run()
    ok(at, "Training")
    assert at.session_state["train_info"]["epochs_run"] == 3

    # 7) + 8)
    go(at, 6)
    ok(at, "Loss")
    go(at, 7)
    at.slider(key="w_threshold").set_value(0.55).run()
    button(at, "Zufälligen Datensatz ziehen").click().run()
    ok(at, "Evaluation")

    # 9) speichern (.h5) und wieder laden
    go(at, 8)
    at.radio(key="w_save_fmt").set_value(".h5").run()
    at.text_input(key="w_save_name").input("test_modell").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern")
    assert (data / "models" / "test_modell" / "model.h5").exists()
    button(at, "Laden").click().run()
    ok(at, "Laden Modell")
    assert at.session_state["bundle"]["recipe"]["threshold"] == pytest.approx(0.55)

    # 10) Vorhersage für neue CSV (mit Fazit) und Formular
    go(at, 9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(data / "neu.csv")).run()
    button(at, "Vorhersage berechnen").click().run()
    ok(at, "Vorhersage CSV")
    res = at.session_state["pred_result"]["res"]
    assert len(res) == 400 and {"Vorhersage", "P(Fully Paid)"} <= set(res.columns)
    assert any(h.value == "Fazit" for h in at.subheader)
    at.text_input(key="w_pred_path").input(str(data / "train.csv")).run()  # Trainingsdaten erneut vorhersagen
    button(at, "Vorhersage berechnen").click().run()
    ok(at, "Vorhersage Trainingsdaten")
    assert at.session_state["pred_result"]["rep"]["train_overlap"]["share"] > 0.5
    assert any("kennt das Modell schon" in m.value for m in at.markdown)
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    button(at, "Vorhersage").click().run()
    ok(at, "Formular")


def test_bool_label_semicolon_csv_and_overwrite_guard(tmp_path, monkeypatch):
    """Audit-Regression: Bool-Label (Seite 4 stürzte ab), Semikolon-CSV mit Dezimalkomma, Pfad in
    Anführungszeichen, Schutz vor dem Überschreiben eines vorhandenen Modells."""
    import numpy as np
    import pandas as pd

    monkeypatch.setenv("DL_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(0)
    n = 600
    betrag = rng.uniform(10, 2000, n).round(2)
    df = pd.DataFrame({"betrag": betrag, "typ": rng.choice(["A", "B", "C"], n),
                       "betrug": (betrag > 1500) ^ (rng.random(n) < 0.1)})
    df.to_csv(tmp_path / "betrug.csv", sep=";", decimal=",", index=False)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(f'"{tmp_path / "betrug.csv"}"').run()  # wie „Als Pfad kopieren“
    button(at, "Daten laden").click().run()
    ok(at, "Laden")
    assert at.session_state["df_raw"].shape == (n, 3)
    assert at.session_state["source"]["sep"] == ";" and at.session_state["source"]["decimal"] == ","
    assert any("Automatisch erkannt" in i.value for i in at.info)
    assert at.session_state["w_sep"] == ";"  # Auswahl zeigt das erkannte Trennzeichen

    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    at.selectbox(key="w_label").set_value("betrug").run()
    assert at.session_state["w_task"] == "binary"
    button(at, "Daten für ML vorbereiten").click().run()
    ok(at, "Schritt 4 mit Bool-Label")
    assert at.session_state["prep"].recipe["label_kind"] == "bool"

    def train():
        go(at, 5)
        at.number_input(key="w_epochs").set_value(2).run()
        button(at, "Training starten").click().run()
        ok(at, "Training")

    train()
    go(at, 8)
    at.text_input(key="w_save_name").input("betrug_modell").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern + Skript-Export")
    assert not at.warning or not any("liegt schon ein Modell" in w.value for w in at.warning)

    train()  # neues Modell, gleicher Name → Warnung, Speichern erst nach Bestätigung
    go(at, 8)
    at.text_input(key="w_save_name").input("betrug_modell").run()
    assert any("liegt schon ein Modell" in w.value for w in at.warning)
    assert button(at, "Speichern").disabled
    at.checkbox(key="overwrite_betrug_modell").check().run()
    assert not button(at, "Speichern").disabled


def test_guards_from_second_audit(tmp_path, monkeypatch):
    """Zweiter Audit: Entfernen einer benutzten neuen Spalte, unbrauchbare Werte im Sitzungszustand, Fehler beim
    Zeichnen, Schutz vor dem Überschreiben beim CSV-Export, Modell ohne Prüfsummen speichern."""
    import json

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT))
    import plots

    monkeypatch.setenv("DL_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(1)
    n = 400
    pd.DataFrame({"x": rng.normal(size=n), "z": rng.uniform(1, 5, n), "k": rng.choice(["a", "b"], n),
                  "y": rng.choice(["ja", "nein"], n)}).to_csv(tmp_path / "d.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "d.csv")).run()
    button(at, "Daten laden").click().run()
    ok(at, "Laden")
    go(at, 1)
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")

    # 1) 'doppelt' benutzt 'summe' → 'summe' allein nicht entfernbar, klare Meldung, nichts verändert
    for name, formula in (("summe", "x + z"), ("doppelt", "summe * 2")):
        at.radio(key="w_new_kind").set_value("Formel").run()
        at.text_input(key="w_new_name").input(name).run()
        at.text_input(key="w_new_formula").input(formula).run()
        button(at, "Spalte hinzufügen").click().run()
        ok(at, f"neue Spalte {name}")
    at.multiselect(key="w_new_rm").set_value(["summe"]).run()
    button(at, "Entfernen").click().run()
    assert not at.exception
    assert any("'doppelt' benutzt 'summe'" in e.value for e in at.error)
    assert [s["name"] for s in at.session_state["new_cols"]] == ["summe", "doppelt"]
    at.multiselect(key="w_new_rm").set_value(["summe", "doppelt"]).run()
    button(at, "Entfernen").click().run()
    ok(at, "beide entfernt")
    assert not at.session_state["new_cols"] and "summe" not in at.session_state["df_viz"].columns

    # 2) Fehler beim Zeichnen → Hinweis statt Absturz der Seite
    go(at, 2)

    def boom(*args, **kwargs):
        raise TypeError("Testfehler")

    with monkeypatch.context() as m:
        m.setattr(plots, "fig_hist", boom)
        at.radio(key="w_chart").set_value("Histogramm").run()
        assert not at.exception
        assert any("Diagramm nicht möglich" in w.value for w in at.warning)
    at.run()
    assert not any("Diagramm nicht möglich" in w.value for w in at.warning)

    # 3) unbrauchbarer Wert (z. B. aus einer von Hand bearbeiteten Datei) → Vorschlag statt TypeError
    go(at, 3)
    at.selectbox(key="w_label").set_value("y").run()
    go(at, 2)  # auf einer anderen Seite setzen (AppTest kann ein angezeigtes Widget nicht mit 5 füllen)
    at.session_state["w_features"] = 5
    at.session_state["_features_for_label"] = "anders"
    go(at, 3)
    ok(at, "w_features = 5")
    assert at.session_state["w_features"] == ["x", "z", "k"]
    button(at, "Daten für ML vorbereiten").click().run()
    ok(at, "Schritt 4")

    # 4) CSV-Export: vorhandene Datei nur nach Bestätigung überschreiben
    go(at, 4)
    button(at, "CSV erstellen").click().run()
    at.text_input(key="w_exp_path").input(str(tmp_path / "exports")).run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export")
    target = next((tmp_path / "exports").glob("*.csv"))
    at.run()
    assert not any("gibt es schon" in w.value for w in at.warning)  # eigene Datei aus dieser Sitzung
    at.session_state["_exp_saved"] = None  # wie eine fremde, schon vorhandene Datei
    target.write_text("alt", encoding="utf-8")
    at.run()
    assert any("gibt es schon" in w.value for w in at.warning)
    assert button(at, "In Ordner speichern").disabled
    at.checkbox(key=f"exp_overwrite_{target.name}").check().run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export überschrieben")
    assert target.read_text(encoding="utf-8").startswith("x,z,k,y")

    # 5) Modell ohne Prüfsummen der Trainingszeilen speichern
    go(at, 5)
    at.number_input(key="w_epochs").set_value(1).run()
    button(at, "Training starten").click().run()
    ok(at, "Training")
    go(at, 8)
    at.text_input(key="w_save_name").input("ohne_pruefsummen").run()
    at.checkbox(key="w_save_fp").uncheck().run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern ohne Prüfsummen")
    saved = json.loads((tmp_path / "models" / "ohne_pruefsummen" / "preprocessing.json").read_text(encoding="utf-8"))
    assert "train_fingerprints" not in saved and saved["features"] == ["x", "z", "k"]
