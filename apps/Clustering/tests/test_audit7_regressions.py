"""
Regressionstests zu Audyt 7 (Autoencoder, Version 3) – jeder Test schlägt auf dem Stand vor dem Audit (e0fe3a0) wegen
des Verhaltens fehl, nicht wegen einer neuen Schnittstelle.

  B1  Schichten des Autoencoders folgen der Zahl der Features (Vorschlag nicht mehr veraltet)
  B2  Ein einziges Feature: Score = echter Rekonstruktionsfehler (vorher n × n-Matrix), Code läuft
  C1  min_samples (DBSCAN) folgt ebenso der Zahl der Features
  C2  Tote Neuronen im Engpass (relu) werden gemeldet

    pytest -q tests/test_audit7_regressions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as pl  # noqa: E402
from synthetic import anomaly_like  # noqa: E402
from test_anomaly import assert_same, prepare, run_both  # noqa: E402
from test_app import button, go, load, ok  # noqa: E402
from test_autoencoder import combinations  # noqa: E402

AE = pl.MODEL_AUTOENCODER
APP = str(ROOT / "app.py")


def step6_with_features(at, features):
    """Schritt 4 mit diesen Features vorbereiten, dann Schritt 6 öffnen."""
    go(at, 3)
    at.multiselect(key="w_features").set_value(features).run()
    button(at, "Daten fürs Modell vorbereiten").click().run()
    ok(at, f"Schritt 4 mit {len(features)} Features")
    go(at, 5)


@pytest.fixture()
def app_400(tmp_path, monkeypatch):
    """App mit 400 Betrugsdaten bis Schritt 4 (Vergleichsgruppe Class) – und die Liste aller Features."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    anomaly_like(400, seed=3).to_csv(tmp_path / "daten.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "daten.csv")
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    at.selectbox(key="w_group").set_value("Class").run()
    return at, list(at.session_state["w_features"])


def test_b1_layers_follow_the_number_of_features(app_400):
    """B1: Der Vorschlag 8,1,8 aus 2 Features blieb bei 8 Features stehen – Engpass 1 statt 2 (bei ungewöhnlichen
    Kombinationen Average Precision 0.22 statt 0.85). Jetzt neu berechnet; selbst eingegebene Schichten bleiben."""
    at, features = app_400
    step6_with_features(at, ["amount", "items"])
    at.radio(key="w_model").set_value(AE).run()
    assert at.session_state["w_ae_layers"] == "8,1,8"
    step6_with_features(at, features)  # 8 Modell-Eingänge (channel als 3 Dummies)
    assert at.session_state["prep"].X.shape[1] == 8
    assert at.session_state["w_ae_layers"] == "8,2,8"  # vorher: 8,1,8
    assert any("Engpass 2 Neuronen" in c.value for c in at.caption)
    at.text_input(key="w_ae_layers").input("16,4,16").run()  # eigene Wahl …
    step6_with_features(at, ["amount", "items"])
    assert at.session_state["w_ae_layers"] == "16,4,16"  # … bleibt, auch bei anderen Features
    assert any("Kein echter Engpass" in w.value for w in at.warning)  # 4 Neuronen bei 2 Features


def test_c1_min_samples_follow_the_number_of_features(app_400):
    """C1: min_samples (Faustregel 2 × Features) blieb nach einer Änderung der Features beim alten Vorschlag."""
    at, features = app_400
    step6_with_features(at, ["amount", "items"])
    at.radio(key="w_model").set_value(pl.MODEL_DBSCAN).run()
    assert at.session_state["w_min_samples"] == 5
    step6_with_features(at, features)
    assert at.session_state["w_min_samples"] == 16  # vorher: 5
    at.number_input(key="w_min_samples").set_value(9).run()
    step6_with_features(at, ["amount", "items"])
    assert at.session_state["w_min_samples"] == 9  # eigene Wahl bleibt


def test_b2_single_feature_score_is_the_reconstruction_error(tmp_path):
    """B2: Bei einem Feature liefert MLPRegressor.predict eine flache Reihe (n,) – X − predict(X) wurde zur n × n-Matrix:
    Score = Abstand zu ALLEN Nachbauten (Korrelation mit dem echten Fehler 0.45), n² Speicher (100 000 Zeilen = 80 GB),
    englische Warnung von scikit-learn in der App, der angezeigte Code brach in Schritt 8 ab (IndexError)."""
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({"x0": np.r_[rng.normal(size=n - 3), [6.0, -7.0, 8.0]],
                       "grp": np.r_[np.zeros(n - 3, dtype=int), [1, 1, 1]]})
    prep, _ = prepare(tmp_path, df, name=None, group="grp")
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=AE), prep)
    pred = model.predict(prep.X)
    assert pred.shape == (n,)  # so liefert scikit-learn den Nachbau bei einem Feature
    np.testing.assert_allclose(hist["score"], np.abs(prep.X[:, 0] - pred), rtol=1e-12)
    assert pl.reconstruction(model, prep.X).shape == prep.X.shape == (n, 1)
    assert info["warnings"] == []  # vorher: 'A column-vector y was passed …'
    assert any(t.startswith(":orange[**Nur ein Feature:**]") for t in pl.autoencoder_notes(info))
    assert set(np.argsort(-hist["score"])[:3]) == set(np.flatnonzero((prep.groups == "1").to_numpy()))
    r = run_both(tmp_path, df, pl.ClusterConfig(model=AE), group="grp")  # Schritte 1–8 als Code
    assert_same(r)
    assert "X_rec = autoencoder.predict(X).reshape(X.shape)" in r["code"]


def test_c2_dead_bottleneck_neurons_are_reported(tmp_path):
    """C2: Mit relu „stirbt“ oft ein Neuron des Engpasses (immer 0) – bei 2 Neuronen bleibt effektiv 1: Average
    Precision 0.18 statt 0.95 bei ungewöhnlichen Kombinationen, ohne jeden Hinweis. Jetzt eine Warnung."""
    prep, _ = prepare(tmp_path, combinations(600, seed=2), name=None)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=AE, ae_activation="relu", seed=1), prep)
    notes = pl.autoencoder_notes(info)
    assert any("Tote Neuronen im Engpass" in t and "1 von 2 Neuronen" in t for t in notes)
    assert int((np.ptp(pl.autoencoder_codes(model, prep.X), axis=0) == 0).sum()) == info["dead_code"] == 1
    _, _, fine = pl.train_anomaly(pl.ClusterConfig(model=AE, seed=1), prep)  # tanh (Standard): kein totes Neuron
    assert fine["dead_code"] == 0 and not any("Tote Neuronen" in t for t in pl.autoencoder_notes(fine))
