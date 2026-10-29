"""
Autoencoder (neuronales Netz, unsupervised) als vierte Methode der Anomalieerkennung: Schichten, Training mit
Eingabe = Ziel, Schwelle = Quantil der Rekonstruktionsfehler, Engpass, am schlechtesten nachgebaute Features,
Speichern/Laden (.skops ohne pickle), Lernkurve und die Oberfläche von Schritt 6.

    pytest -q tests/test_autoencoder.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as pl  # noqa: E402
import plots  # noqa: E402
from synthetic import anomaly_like  # noqa: E402
from test_anomaly import prepare  # noqa: E402
from test_app import button, go, load, ok  # noqa: E402

AE = pl.MODEL_AUTOENCODER
APP = str(ROOT / "app.py")


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    prep, viz = prepare(tmp_path_factory.mktemp("ae"))
    cfg = pl.ClusterConfig(model=AE, contamination=0.02)
    model, hist, info = pl.train_anomaly(cfg, prep)
    return prep, viz, cfg, model, hist, info


def test_layers_default_and_parsing():
    assert [pl.default_ae_layers(p) for p in (1, 2, 3, 8, 11, 30, 100)] == [
        (8, 1, 8), (8, 1, 8), (8, 2, 8), (8, 2, 8), (16, 3, 16), (32, 8, 32), (64, 8, 64)]
    assert all(min(pl.default_ae_layers(p)) < p for p in range(2, 200))  # immer ein echter Engpass
    assert pl.parse_layers(" 16, 3 ;16 ") == (16, 3, 16) and pl.parse_layers("32 16 4 16 32") == (32, 16, 4, 16, 32)
    assert pl.parse_layers("", 11) == (16, 3, 16) and pl.layers_text((16, 3, 16)) == "16,3,16"
    for bad in ("16,x,16", "16,0,16", "2000", ",".join(["4"] * 10), "1.5"):
        with pytest.raises(ValueError, match="Schichten"):
            pl.parse_layers(bad)


def test_training_threshold_and_settings(trained):
    prep, _, cfg, model, hist, info = trained
    assert cfg.ae_layers == "8,2,8"  # leer → aus der Zahl der Features (8) eingetragen
    assert model.hidden_layer_sizes == (8, 2, 8) and model.activation == "tanh" and model.early_stopping
    score = pl.reconstruction_error(model, prep.X)
    np.testing.assert_array_equal(hist["score"], score)
    assert info["threshold"] == model.threshold_ == np.percentile(score, 98)
    np.testing.assert_array_equal(hist["flag"], score > model.threshold_)
    assert info["share"] == pytest.approx(0.02, abs=0.002) and hist["labels"] is None
    assert info["layers"] == [8, 2, 8] and info["code_size"] == 2 and info["n_iter"] >= 1
    assert info["best_validation"] is not None and 0 < info["best_validation"] <= 1
    assert not any("Maximum iterations" in w for w in info["warnings"])  # verständlich in autoencoder_notes erklärt
    zero = pl.ClusterConfig(model=AE, ae_layers="8,2,8", ae_epochs=40)
    _, _, info0 = pl.train_anomaly(zero, prep)
    assert zero.contamination == pl.AE_DEFAULT_CONTAMINATION and info0["share"] == pytest.approx(0.01, abs=0.002)
    assert any("Höchstzahl von 40 Epochen" in t for t in pl.autoencoder_notes(info0))


def test_the_network_finds_the_planted_anomalies(trained):
    prep, _, _, model, hist, info = trained
    y = (prep.groups == "1").to_numpy()
    assert average_precision_score(y, hist["score"]) > 0.85
    top = np.argsort(-hist["score"])[:10]
    assert y[top].mean() >= 0.9  # die auffälligsten Zeilen sind die eingebauten Betrugsfälle


@pytest.mark.parametrize("activation", ["tanh", "relu"])
def test_codes_are_the_encoder_output(tmp_path, activation):
    prep, _ = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=AE, ae_layers="12,6,3,6,12", ae_activation=activation, ae_epochs=60)
    model, hist, _ = pl.train_anomaly(cfg, prep)
    act = np.tanh if activation == "tanh" else (lambda v: np.maximum(v, 0))
    a = prep.X
    for i in range(3):  # 12 → 6 → 3: der Engpass ist die dritte verborgene Schicht
        a = act(a @ model.coefs_[i] + model.intercepts_[i])
    np.testing.assert_allclose(pl.autoencoder_codes(model, prep.X), a, rtol=1e-12, atol=1e-12)
    assert pl.code_layer(model) == 2
    proj = pl.code_projection(prep, model, priority=hist["flag"])
    assert proj["code_size"] == 3 and proj["coords"].shape == (len(prep.X), 2) and "Code-PC1" in proj["axes"][0]
    fig = plots.fig_anomaly_pca(proj, hist["flag"][proj["rows"]])
    assert "Engpass" in fig.axes[0].get_title(loc="left") and "Code-PC2" in fig.axes[0].get_ylabel()
    one, _, _ = pl.train_anomaly(pl.ClusterConfig(model=AE, ae_layers="4,1,4", ae_epochs=30), prep)
    assert pl.code_projection(prep, one) is None  # 1 Neuron: keine 2D-Karte
    plt.close("all")


def test_worst_reconstructed_features(trained):
    prep, _, _, model, hist, _ = trained
    recon = pl.reconstruction(model, prep.X)
    np.testing.assert_allclose(np.sqrt((recon ** 2).mean(axis=1)), hist["score"], rtol=1e-12)
    top = pl.top_anomalies(prep, hist["score"], hist["flag"], 5, recon=recon)
    assert list(top.columns)[-1] == "am schlechtesten nachgebaut"
    i = int(np.argsort(-hist["score"], kind="stable")[0])
    worst = prep.recipe["feature_columns"][int(np.argmax(np.abs(recon[i])))]
    assert top["am schlechtesten nachgebaut"].iloc[0].startswith(worst + " ")
    assert "σ" not in top["am schlechtesten nachgebaut"].iloc[0]  # skalierte Einheiten, nicht immer σ
    assert "am schlechtesten nachgebaut" not in pl.top_anomalies(prep, hist["score"], hist["flag"], 5).columns


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_save_load_without_pickle_and_broken_models(trained, tmp_path):
    prep, viz, cfg, model, hist, info = trained
    recipe = pl.finalize_recipe(prep.recipe, cfg, info)
    folder = pl.save_bundle(model, recipe, tmp_path / "models" / "ae")
    loaded, rec2 = pl.load_bundle_zip(pl.bundle_zip_bytes(folder), allow_pickle=False)  # .skops: kein Code beim Laden
    assert loaded.threshold_ == model.threshold_ and rec2["training"]["ae_layers"] == "8,2,8"
    np.testing.assert_array_equal(loaded.predict(prep.X), model.predict(prep.X))
    res, rep, cmp = pl.score_new(loaded, rec2, viz)  # die Trainingsdaten noch einmal: gleiche Scores und Anomalien
    np.testing.assert_allclose(rep["score"][np.argsort(prep.index)], hist["score"][np.argsort(prep.index)])
    assert int(res[pl.RESULT_ANOMALY].sum()) == info["n_anomalies"]
    del loaded.threshold_
    with pytest.raises(ValueError, match="threshold_"):
        pl.check_recipe(rec2, loaded)
    wrong = pl.make_anomaly_model(cfg, len(prep.X), prep.X.shape[1]).set_params(max_iter=5).fit(prep.X, prep.X[:, :3])
    wrong.threshold_ = 1.0
    with pytest.raises(ValueError, match="unvollständig"):  # Ausgänge ≠ Eingänge: kein Autoencoder
        pl.check_recipe(rec2, wrong)
    json.dumps(pl._jsonable(recipe))  # Rezept mit den Angaben zum Netz bleibt JSON


def test_notes_learning_curve_and_no_bottleneck(trained, tmp_path):
    prep, _, cfg, model, hist, info = trained
    notes = pl.threshold_assessment(hist["curve"], info, cfg)["notes"]
    assert any(t.startswith("**Netz:** 8 Features → Schichten 8 – 2 – 8") for t in notes)
    assert any("Early Stopping: bestes R²" in t for t in notes)
    fig = plots.fig_learning_curve(model)
    assert len(fig.axes) == 2 and "beste Epoche" in fig.axes[1].texts[0].get_text()
    wide, _, info_w = pl.train_anomaly(pl.ClusterConfig(model=AE, ae_layers="16,16", ae_epochs=30,
                                                        ae_early_stopping=False), prep)
    assert len(plots.fig_learning_curve(wide).axes) == 1  # ohne Early Stopping keine Validierungskurve
    assert any("Kein echter Engpass" in t for t in pl.autoencoder_notes(info_w))
    plt.close("all")


def test_configuration_keeps_the_autoencoder_settings():
    cfg, skipped = pl.sanitize_config({"widgets": {"w_model": "autoencoder", "w_ae_layers": "32,8,32",
                                                   "w_ae_activation": "relu", "w_ae_epochs": "99999", "w_ae_alpha": "x",
                                                   "w_ae_lr": 0.01, "w_ae_early": "false"}})
    w = cfg["widgets"]
    assert w["w_ae_layers"] == "32,8,32" and w["w_ae_activation"] == "relu" and w["w_ae_epochs"] == 5000
    assert w["w_ae_lr"] == 0.01 and w["w_ae_early"] is False and "w_ae_alpha" in skipped


def test_step_6_checks_the_layers(tmp_path, monkeypatch):
    """Schritt 6: unbrauchbare Schichten → Fehler und kein Training; kein Engpass → Warnung."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    anomaly_like(400, seed=3).to_csv(tmp_path / "daten.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    load(at, tmp_path / "daten.csv")
    go(at, 1)
    button(at, "Übernehmen").click().run()
    go(at, 3)
    at.selectbox(key="w_group").set_value("Class").run()
    button(at, "Daten fürs Modell vorbereiten").click().run()
    go(at, 5)
    at.radio(key="w_model").set_value(AE).run()
    ok(at, "Autoencoder gewählt")
    assert any("Engpass 2 Neuronen" in c.value for c in at.caption)
    at.text_input(key="w_ae_layers").input("16,x,16").run()
    assert any("Schichten" in e.value for e in at.error) and button(at, "Training starten").disabled
    at.text_input(key="w_ae_layers").input("16,16,16").run()
    assert any("Kein echter Engpass" in w.value for w in at.warning)
    assert not button(at, "Training starten").disabled
