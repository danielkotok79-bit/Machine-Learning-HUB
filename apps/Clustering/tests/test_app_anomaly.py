"""
Klick-Test der Anomalieerkennung (streamlit.testing.AppTest): Isolation Forest, LOF und DBSCAN durch die Schritte
4–10 – und die Bybit-CSV aus tools/bybit_features.py mit ihrer Beispiel-Konfiguration (simulierte API, kein Netz).

    pytest -q tests/test_app_anomaly.py
"""
import json
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import HOUR_MS, FakeBybit, anomaly_like  # noqa: E402
from test_app import button, go, load, ok  # noqa: E402
from tools import bybit_features as bf  # noqa: E402

APP = str(ROOT / "app.py")


@pytest.fixture()
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    return tmp_path


def start(path):
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    ok(at, "Start")
    load(at, path)
    return at


def apply_suggestion(at, prefix):
    """Schritt 7: Vorschlag (contamination bzw. eps) übernehmen, falls angeboten → True."""
    offers = [b for b in at.button if b.label.startswith(prefix) and b.label.endswith("übernehmen")]
    if not offers:
        return False
    offers[0].click().run()
    assert any("jetzt in Schritt 6 neu trainieren" in s.value for s in at.success)
    return True


@pytest.mark.parametrize("model", ["iforest", "lof", "dbscan", "autoencoder"])
def test_anomaly_workflow(folder, model):
    anomaly_like(1000, seed=3).to_csv(folder / "train.csv", index=False)
    anomaly_like(150, share=0.05, seed=9).to_csv(folder / "neu.csv", index=False)
    at = start(folder / "train.csv")
    go(at, 1)
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")

    # 4) Class nur zum Vergleich (wie bei Betrugsdaten), alle anderen Spalten als Features
    go(at, 3)
    at.selectbox(key="w_group").set_value("Class").run()
    assert "Class" not in at.session_state["w_features"] and "channel" in at.session_state["w_features"]
    button(at, "Daten fürs Modell vorbereiten").click().run()
    ok(at, "Schritt 4")

    # 6) Methode wählen, Parameter, Training
    go(at, 5)
    at.radio(key="w_model").set_value(model).run()
    ok(at, "Methode")
    assert any("So arbeitet" in e.label for e in at.expander)
    if model == "lof":
        at.number_input(key="w_n_neighbors").set_value(60).run()  # mehr Nachbarn als Ausreißer (Gruppe von 20)
    if model in ("iforest", "lof"):
        at.checkbox(key="w_contam_auto").uncheck().run()
        at.number_input(key="w_contam_pct").set_value(2.0).run()
    elif model == "autoencoder":  # keine eingebaute Schwelle: nur der Anteil, Schichten aus der Zahl der Features
        assert not [c for c in at.checkbox if c.key == "w_contam_auto"]
        assert at.session_state["w_ae_layers"] == "8,2,8"
        at.number_input(key="w_contam_pct").set_value(2.0).run()
    else:
        assert at.session_state["w_eps"] > 0 and at.session_state["w_min_samples"] == 16  # 2 × 8 Features
    button(at, "Training starten").click().run()
    ok(at, "Training")
    info = at.session_state["train_info"]
    assert info["model"] == model and info["n_rows"] == 1000 and info["n_anomalies"] > 0
    assert any("fertig" in s.value for s in at.success)
    sidebar = [m.value for m in at.sidebar.markdown] + [c.value for c in at.sidebar.caption]
    assert any("trainiert (" in v and "auffällig" in v for v in sidebar)
    if model == "autoencoder":  # Lernkurve und Aufbau des Netzes
        assert any(s.value == "Lernkurve des Autoencoders" for s in at.subheader)
        assert any("Engpass 2 Neuronen" in m.value for m in at.markdown)

    # 7) Schwelle bzw. eps: Vorschlag übernehmen und neu trainieren
    go(at, 6)
    ok(at, "Schritt 7")
    assert ("eps" if model == "dbscan" else "Schwelle") in at.header[0].value
    if apply_suggestion(at, "eps = " if model == "dbscan" else "contamination = "):
        go(at, 5)
        button(at, "Training starten").click().run()
        ok(at, "Training mit Vorschlag")
        go(at, 6)
        ok(at, "Schritt 7 nach Vorschlag")

    # 8) Auswertung: Tabelle, Profil, PCA, Vergleich mit Class, Einzelfall
    go(at, 7)
    ok(at, "Schritt 8")
    heads = [s.value for s in at.subheader]
    assert "Die auffälligsten Zeilen" in heads and "Vergleich mit der bekannten Einteilung: Class" in heads
    assert ("Cluster und Rauschen" in heads) == (model == "dbscan")
    top = next(d.value for d in at.dataframe if "Anomalie-Score" in d.value.columns)
    assert top["Anomalie-Score"].is_monotonic_decreasing and len(top) == 20
    assert ("am schlechtesten nachgebaut" in top.columns) == (model == "autoencoder")
    assert {"Precision", "Recall", "ROC-AUC"} <= {m.label for m in at.metric}
    button(at, "Zufällige Zeile ziehen").click().run()
    ok(at, "Einzelfall")
    assert {"Anomalie", "Anomalie-Score", "Rang"} <= {m.label for m in at.metric}

    # 5) Daten mit Anomalien speichern
    go(at, 4)
    at.radio(key="w_exp_what").set_value("anomalies").run()
    button(at, "CSV erstellen").click().run()
    ok(at, "Export Anomalien")
    head = at.session_state["_export"][1].decode("utf-8").splitlines()[0]
    assert head.endswith(("Cluster," if model == "dbscan" else "Class,") + "Anomalie,Anomalie-Score")

    # 9) speichern und laden
    go(at, 8)
    at.text_input(key="w_save_name").input(f"modell_{model}").run()
    button(at, "Speichern").click().run()
    ok(at, "Speichern")
    saved = json.loads((folder / "models" / f"modell_{model}" / "preprocessing.json").read_text(encoding="utf-8"))
    assert saved["training"]["model"] == model and saved["task"] == "clustering"
    assert saved["train_info"]["n_anomalies"] == at.session_state["train_info"]["n_anomalies"]
    assert saved["train_info"]["threshold"] == at.session_state["train_info"]["threshold"]
    button(at, "Laden").click().run()
    ok(at, "Laden Modell")

    # 10) neue Daten bewerten (mit Vergleich) und ein Datensatz über das Formular
    go(at, 9)
    assert at.header[0].value.startswith("10 · Neue Daten bewerten")
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(folder / "neu.csv")).run()
    button(at, "Neue Daten bewerten").click().run()
    ok(at, "Bewertung CSV")
    pr = at.session_state["pred_result"]
    assert len(pr["res"]) == 150 and {"Anomalie", "Anomalie-Score", "Hinweis"} <= set(pr["res"].columns)
    assert ("Cluster" in pr["res"].columns) == (model == "dbscan")
    assert pr["cmp"]["binary"] and pr["cmp"]["n"] == 150
    assert pr["cmp"]["positive"] == "1" and pr["cmp"]["positive_source"] == "training"  # wie im Training
    assert any(h.value == "Fazit" for h in at.subheader)
    assert any("Was die Bewertung aussagt" in m.value for m in at.markdown)
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    ok(at, "Formular")
    button(at, "Neue Daten bewerten").click().run()
    ok(at, "Formular bewertet")
    assert any(m.label == "Anomalie" and m.value in ("ja", "nein") for m in at.metric)


def test_bybit_csv_with_its_example_configuration(folder):
    """tools/bybit_features.py → CSV → Beispiel-Konfiguration erkannt → Isolation Forest auf dem früheren Teil →
    der spätere Teil (mit dem eingebauten Kurssturz) in Schritt 10 bewertet."""
    fake = FakeBybit()
    split = bf.ms_text(fake.shock_start - 3 * 24 * HOUR_MS)
    rc = bf.main(["--days", "60", "--split", split, "--end", bf.ms_text(fake.now_ms), "--out",
                  str(folder / "bybit.csv")], client=fake)
    assert rc == 0
    at = start(folder / "bybit_train.csv")
    go(at, 1)
    assert any("Bybit-Daten" in i.value and "Zeitreihe" in i.value for i in at.info)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    go(at, 3)
    assert at.session_state["w_name_col"] == "row" and at.session_state["w_group"] == "(keine)"
    assert at.session_state["w_features"] == bf.SERIES_FEATURES  # Reihenfolge wie in der CSV
    button(at, "Daten fürs Modell vorbereiten").click().run()
    ok(at, "Schritt 4")
    go(at, 5)
    assert at.session_state["w_model"] == "iforest" and at.session_state["w_n_estimators"] == 200
    assert not at.session_state["w_contam_auto"] and at.session_state["w_contam_pct"] == 1.0
    button(at, "Training starten").click().run()
    ok(at, "Training")
    assert at.session_state["train_info"]["share"] == pytest.approx(0.01, abs=0.002)
    go(at, 7)
    ok(at, "Schritt 8")
    top = next(d.value for d in at.dataframe if "Bezeichnung" in d.value.columns)
    assert top["Bezeichnung"].str.match(r"(BTC|ETH|SOL)USDT 20\d\d-\d\d-\d\d \d\d:00").all()

    go(at, 9)
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(folder / "bybit_new.csv")).run()
    button(at, "Neue Daten bewerten").click().run()
    ok(at, "Bewertung")
    res = at.session_state["pred_result"]["res"]
    worst = res.sort_values("Anomalie-Score", ascending=False).iloc[0]
    assert worst["row"] == f"SOLUSDT {bf.ms_text(fake.shock_start)}" and worst["Anomalie"] == 1
    assert any(f"SOLUSDT {bf.ms_text(fake.shock_start)}" in m.value for m in at.markdown)  # im Fazit genannt


def test_cross_section_csv_is_recognised(folder):
    rc = bf.main(["--mode", "cross", "--days", "30", "--end", bf.ms_text(FakeBybit().now_ms), "--out",
                  str(folder / "tokens.csv")], client=FakeBybit())
    assert rc == 0
    at = start(folder / "tokens.csv")
    go(at, 1)
    assert any("Querschnitt" in i.value for i in at.info)
    button(at, "Beispiel-Konfiguration laden").click().run()
    button(at, "Übernehmen").click().run()
    ok(at, "Schritt 2")
    go(at, 3)
    assert at.session_state["w_name_col"] == "symbol" and len(at.session_state["w_features"]) == 13
    assert at.session_state["w_contam_pct"] == 5.0
