"""
Klick-Test der Streamlit-App ohne Browser (streamlit.testing.AppTest): alle 10 Schritte einmal durch – mit dem
College-Datensatz wie im K-Means-Kurs – und die Schutzmechanismen (Überschreiben, unbrauchbare Werte, Konfiguration).

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
from synthetic import college_like  # noqa: E402

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
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    college_like().to_csv(tmp_path / "College_Data.csv")  # erste Spalte ohne Namen = Name der Hochschule
    college_like(80, seed=7).to_csv(tmp_path / "neu.csv")
    return tmp_path


def test_full_workflow_like_the_course(data):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    ok(at, "Start")
    nav = at.sidebar.radio(key="nav").options
    assert [o.split(" · ")[0] for o in nav] == [str(i) for i in range(1, 11)]  # Schritte 1–10 vollständig
    for i in range(len(nav)):  # alle Seiten ohne Daten
        go(at, i)
        ok(at, f"leere Seite {i}")

    # 1) Laden über Pfad, Stichprobe an/aus
    load(at, data / "College_Data.csv")
    assert at.session_state["df_raw"].shape == (777, 19)
    at.sidebar.checkbox(key="w_use_sample").check().run()
    assert len(at.session_state["df_raw"]) == 78
    at.sidebar.checkbox(key="w_use_sample").uncheck().run()
    assert len(at.session_state["df_raw"]) == 777

    # 2) College-Datensatz erkannt → Beispiel-Konfiguration (Name behalten, Grad.Rate ≤ 100) + neue Spalte
    go(at, 1)
    assert any("College-Datensatz" in i.value for i in at.info)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    viz = at.session_state["df_viz"]
    assert viz.columns[0] == "College" and viz["Grad.Rate"].max() == 100
    assert any("begrenzt (1 Werte geändert)" in m.value for m in at.markdown)
    at.radio(key="w_new_kind").set_value("Formel").run()
    at.text_input(key="w_new_name").input("annahmequote").run()
    at.text_input(key="w_new_formula").input("Accept / Apps").run()
    button(at, "Spalte hinzufügen").click().run()
    ok(at, "neue Spalte")

    # 3) wie im Kurs: Scatterplot mit Farbe = Private, Histogramm je Gruppe
    go(at, 2)
    at.radio(key="w_chart").set_value("Scatterplot").run()
    at.selectbox(key="w_sc_x").set_value("Room.Board").run()
    at.selectbox(key="w_sc_y").set_value("Grad.Rate").run()
    at.selectbox(key="w_sc_hue").set_value("Private").run()
    ok(at, "Scatterplot")
    at.radio(key="w_chart").set_value("Histogramm je Gruppe").run()
    ok(at, "Histogramm je Gruppe")
    assert at.session_state["w_hg_group"] == "Private"

    # 4) Private nur zum Vergleich, College als Bezeichnung, neue Spalte automatisch als Feature
    go(at, 3)
    assert at.session_state["w_group"] == "Private" and at.session_state["w_name_col"] == "College"
    assert "annahmequote" in at.session_state["w_features"] and "Private" not in at.session_state["w_features"]
    button(at, "Daten für K-Means vorbereiten").click().run()
    ok(at, "Schritt 4")
    prep = at.session_state["prep"]
    assert prep.X.shape == (777, 18) and prep.recipe["group_col"] == "Private" and prep.names is not None
    assert prep.recipe["scaler"]["type"] == "StandardScaler" and prep.recipe["fit_on"] == "all"

    # 5) Daten speichern
    go(at, 4)
    for what in ("step2", "ml", "ml_scaled"):
        at.radio(key="w_exp_what").set_value(what).run()
        button(at, "CSV erstellen").click().run()
        ok(at, f"Export {what}")
    at.radio(key="w_exp_what").set_value("clusters").run()
    assert any("zuerst in Schritt 6 trainieren" in i.value for i in at.info)

    # 6) K-Means wie im Kurs mit K = 2 (Beispiel-Konfiguration)
    go(at, 5)
    assert at.session_state["w_k"] == 2 and at.session_state["w_n_init"] == 10
    button(at, "Training starten").click().run()
    ok(at, "Training")
    info = at.session_state["train_info"]
    assert info["k"] == 2 and sum(info["sizes"]) == 777
    assert at.session_state["history"]["curve"]["values"] == list(range(1, 11))

    # 7) Wahl von K: Vorschlag übernehmen (falls einer da ist), sonst K = 3 von Hand
    go(at, 6)
    ok(at, "Schritt 7")
    proposals = [b for b in at.button if b.label.startswith("K = ") and "übernehmen" in b.label]
    if proposals:
        proposals[0].click().run()
        assert any("jetzt in Schritt 6 neu trainieren" in s.value for s in at.success)
    go(at, 5)
    at.number_input(key="w_k").set_value(3).run()
    at.checkbox(key="w_curve").uncheck().run()
    button(at, "Training starten").click().run()
    ok(at, "Training K = 3")
    assert at.session_state["train_info"]["k"] == 3 and at.session_state["history"]["curve"] is None
    go(at, 6)
    ok(at, "Schritt 7 ohne Kurve")
    assert any("ausgeschaltet" in i.value for i in at.info)

    # 8) Auswertung: Profile, PCA (Farbe = Private), zwei Features, Vergleich, Vertreter, Einzelfall
    go(at, 7)
    ok(at, "Schritt 8")
    assert any(s.value.startswith("Vergleich mit der bekannten Einteilung") for s in at.subheader)
    at.radio(key="w_pca_color").set_value("group").run()
    button(at, "Zufällige Zeile ziehen").click().run()
    ok(at, "Einzelfall")
    names = at.session_state["ed_names__base"].copy()
    names["Name"] = ["eins", "", "drei"]
    go(at, 4)
    at.session_state["ed_names__draft"] = names  # wie eine Eingabe im Editor von Schritt 8 (dann Seitenwechsel)
    at.radio(key="w_exp_what").set_value("clusters").run()
    button(at, "CSV erstellen").click().run()
    ok(at, "Export Cluster")
    head = at.session_state["_export"][1].decode("utf-8").splitlines()[0]
    assert head.startswith("College,Private,") and head.endswith(",Cluster,Cluster-Name,Abstand")

    # 9) speichern (.skops) + Skript, laden
    go(at, 8)
    at.text_input(key="w_save_name").input("college_k3").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern")
    saved = json.loads((data / "models" / "college_k3" / "preprocessing.json").read_text(encoding="utf-8"))
    assert saved["cluster_names"] == ["eins", "", "drei"] and saved["training"]["n_clusters"] == 3
    assert set(saved["group_mapping"].values()) <= {"Yes", "No"} and saved["task"] == "clustering"
    button(at, "Laden").click().run()
    ok(at, "Laden Modell")
    assert at.session_state["bundle"]["recipe"]["cluster_names"] == ["eins", "", "drei"]

    # 10) neue Hochschulen zuordnen (mit Fazit und Vergleich) und Formular
    go(at, 9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(data / "neu.csv")).run()
    button(at, "Cluster zuordnen").click().run()
    ok(at, "Zuordnung CSV")
    res = at.session_state["pred_result"]["res"]
    assert len(res) == 80 and {"Cluster", "Cluster-Name", "Abstand", "Hinweis"} <= set(res.columns)
    assert at.session_state["pred_result"]["cmp"]["n"] == 80
    assert any(h.value == "Fazit" for h in at.subheader)
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    ok(at, "Formular")


def test_semicolon_csv_overwrite_guard_and_stale_settings(tmp_path, monkeypatch):
    """Semikolon-CSV mit Dezimalkomma in Anführungszeichen, Schutz vor dem Überschreiben eines vorhandenen Modells,
    Hinweis auf geänderte Einstellungen in Schritt 4, unbrauchbare Werte im Sitzungszustand."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"betrag": rng.uniform(10, 2000, n).round(2), "dauer": rng.uniform(1, 60, n).round(1),
                       "typ": rng.choice(["A", "B", "C"], n), "kanal": rng.choice(["web", "laden"], n)})
    df.to_csv(tmp_path / "kunden.csv", sep=";", decimal=",", index=False)

    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(f'"{tmp_path / "kunden.csv"}"').run()  # wie „Als Pfad kopieren“
    button(at, "Daten laden").click().run()
    ok(at, "Laden")
    assert at.session_state["source"]["sep"] == ";" and at.session_state["source"]["decimal"] == ","
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    assert at.session_state["w_group"] == "kanal" or at.session_state["w_group"] == "(keine)"
    at.selectbox(key="w_group").set_value("kanal").run()
    assert at.session_state["w_features"] == ["betrag", "dauer", "typ"]
    button(at, "Daten für K-Means vorbereiten").click().run()
    ok(at, "Schritt 4")
    at.multiselect(key="w_features").set_value(["betrag", "dauer"]).run()
    assert any("seit dem Vorbereiten geändert" in i.value for i in at.info)
    button(at, "Daten für K-Means vorbereiten").click().run()

    # unbrauchbarer Wert (z. B. aus einer von Hand bearbeiteten Datei) → Vorschlag statt TypeError
    go(at, 2)
    at.session_state["w_features"] = 5
    at.session_state["_features_for_group"] = ("anders", None)
    go(at, 3)
    ok(at, "w_features = 5")
    assert at.session_state["w_features"] == ["betrag", "dauer", "typ"]

    def train():
        go(at, 5)
        at.checkbox(key="w_curve").uncheck().run()
        button(at, "Training starten").click().run()
        ok(at, "Training")

    button(at, "Daten für K-Means vorbereiten").click().run()
    train()
    go(at, 8)
    at.text_input(key="w_save_name").input("kunden").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern + Skript-Export")
    assert not any("liegt schon ein Modell" in w.value for w in at.warning)
    train()  # neues Modell, gleicher Name → Warnung, Speichern erst nach Bestätigung
    go(at, 8)
    at.text_input(key="w_save_name").input("kunden").run()
    assert any("liegt schon ein Modell" in w.value for w in at.warning)
    assert button(at, "Speichern").disabled
    at.checkbox(key="overwrite_kunden").check().run()
    assert not button(at, "Speichern").disabled


def test_new_columns_guard_chart_errors_and_export_overwrite(tmp_path, monkeypatch):
    """Benutzte neue Spalten sind nicht einzeln entfernbar; Fehler beim Zeichnen als Hinweis; der CSV-Export
    überschreibt fremde Dateien nur nach Bestätigung."""
    sys.path.insert(0, str(ROOT))
    import plots

    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    college_like(300, seed=5).to_csv(tmp_path / "c.csv")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "c.csv")
    go(at, 1)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")

    # 1) 'doppelt' benutzt 'summe' → 'summe' allein nicht entfernbar, klare Meldung, nichts verändert
    for name, formula in (("summe", "Apps + Accept"), ("doppelt", "summe * 2")):
        at.radio(key="w_new_kind").set_value("Formel").run()
        at.text_input(key="w_new_name").input(name).run()
        at.text_input(key="w_new_formula").input(formula).run()
        button(at, "Spalte hinzufügen").click().run()
        ok(at, f"neue Spalte {name}")
    at.multiselect(key="w_new_rm").set_value(["summe"]).run()
    button(at, "Entfernen").click().run()
    assert not at.exception and any("'doppelt' benutzt 'summe'" in e.value for e in at.error)
    at.multiselect(key="w_new_rm").set_value(["summe", "doppelt"]).run()
    button(at, "Entfernen").click().run()
    ok(at, "beide entfernt")

    # 2) Fehler beim Zeichnen → Hinweis statt Absturz der Seite
    go(at, 2)

    def boom(*args, **kwargs):
        raise TypeError("Testfehler")

    with monkeypatch.context() as m:
        m.setattr(plots, "fig_hist", boom)
        at.radio(key="w_chart").set_value("Histogramm").run()
        assert not at.exception and any("Diagramm nicht möglich" in w.value for w in at.warning)

    # 3) CSV-Export: vorhandene Datei nur nach Bestätigung überschreiben
    go(at, 4)
    button(at, "CSV erstellen").click().run()
    at.text_input(key="w_exp_path").input(str(tmp_path / "exports")).run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export")
    target = next((tmp_path / "exports").glob("*.csv"))
    at.session_state["_exp_saved"] = None  # wie eine fremde, schon vorhandene Datei
    target.write_text("alt", encoding="utf-8")
    at.run()
    assert any("gibt es schon" in w.value for w in at.warning)
    assert button(at, "In Ordner speichern").disabled
    at.checkbox(key=f"exp_overwrite_{target.name}").check().run()
    button(at, "In Ordner speichern").click().run()
    ok(at, "Export überschrieben")
    assert target.read_text(encoding="utf-8").startswith("College,Private,")
