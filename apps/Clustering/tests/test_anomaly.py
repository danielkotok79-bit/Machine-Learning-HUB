"""
Anomalieerkennung – Isolation Forest, Local Outlier Factor (LOF) und DBSCAN (Rauschen = Anomalie):
Rechenlogik (Score, Schwelle, Kurven für Schritt 7), Auswertung (Schritt 8), Speichern/Laden (Schritt 9), Bewertung
neuer Daten (Schritt 10) – und der angezeigte Python-Code rechnet exakt dasselbe wie die App.

    pytest -q tests/test_anomaly.py
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import i18n  # noqa: E402
import pipeline as pl  # noqa: E402
from synthetic import anomaly_like, college_like, mall_like  # noqa: E402

pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning", "error::FutureWarning")
IF, LOF, DB = pl.MODEL_IFOREST, pl.MODEL_LOF, pl.MODEL_DBSCAN
ANOMALY_ACTIONS = {"ID": (pl.A_KEEP, "", "")}


def load(tmp_path, raw, actions=None, sample=None, index=False):
    """CSV schreiben und wie in Schritt 1 + 2 laden → (df, viz, fitted, lineage, Lade-Code-Teile)."""
    path = tmp_path / "daten.csv"
    raw.to_csv(path, index=index)
    df, enc = pl.read_csv(path)
    meta = df.attrs["csv"]
    df = pl.take_sample(df, sample)
    prof = pl.column_profile(df)
    for col, (act, param, new) in (actions or {}).items():
        i = prof.index[prof["Spalte"] == col][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [act, param, new]
    viz, fitted, _, lineage = pl.apply_column_actions(df, pl.actions_from_table(prof))
    numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
    code = [cg.IMPORTS, cg.load_code(str(path), meta["sep"], meta["decimal"], enc, sample,
                                     strip_columns=meta["strip_columns"]), cg.step2_code(fitted, numeric_raw)]
    return df, viz, fitted, lineage, code


def run_both(tmp_path, raw, cfg, actions=None, features=None, group=None, name=None, sample=None, index=False,
             **prep_kw):
    """(a) App-Pipeline bis Schritt 8, (b) der erzeugte Code bis Schritt 8 – Ergebnis-dict für assert_same."""
    df, viz, fitted, lineage, code = load(tmp_path, raw, actions, sample, index)
    feats = features or [c for c in viz.columns if c not in (group, name)]
    prep = pl.prepare_cluster_data(viz, feats, column_actions=fitted, lineage=lineage, group_col=group,
                                   name_col=name, **prep_kw)
    model, hist, info = pl.train_anomaly(cfg, prep)
    n = len(prep.X)
    code = "\n".join([*code, cg.step4_code(prep.recipe), cg.model_code(cfg, n), cg.curve_code(cfg, n),
                      cg.eval_code(prep.recipe, cfg)])
    ns = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "generierter_code", "exec"), ns)  # noqa: S102 – eigener, erzeugter Code
    plt.close("all")
    return dict(prep=prep, model=model, hist=hist, info=info, ns=ns, viz=viz, cfg=cfg, code=code)


def assert_same(r):
    prep, ns, hist, info, cfg = r["prep"], r["ns"], r["hist"], r["info"], r["cfg"]
    assert ns["feature_columns"] == prep.recipe["feature_columns"]
    np.testing.assert_array_equal(ns["X"], prep.X)  # gleiche Matrix
    np.testing.assert_array_equal(ns["score"], hist["score"])  # gleicher Score
    np.testing.assert_array_equal(ns["anomaly"], hist["flag"])  # gleiche Anomalien
    assert ns["threshold"] == info["threshold"]
    curve = hist["curve"]
    if cfg.model == DB:
        np.testing.assert_array_equal(ns["labels"], hist["labels"])
        np.testing.assert_array_equal(ns["kdist"], curve["kdist"])
        assert ns["knee"] == curve["knee"] and ns["eps_knee"] == curve["suggest"]
        if curve["grid"] is not None:
            assert ns["eps_values"] == curve["grid"]["eps"] and ns["clusters"] == curve["grid"]["clusters"]
            np.testing.assert_array_equal(ns["noise"], curve["grid"]["noise"])
        else:
            assert "eps_values" not in ns
    else:
        np.testing.assert_array_equal(ns["thresholds"], curve["thresholds"])
        np.testing.assert_array_equal(ns["desc"], curve["sorted"])
        assert ns["knee"] == curve["knee"]
    prof = pl.anomaly_profiles(prep, hist["flag"])
    cols = ["Alle"] + [j for j in (0, 1) if (hist["flag"].astype(int) == j).any()]
    np.testing.assert_allclose(ns["profile"].to_numpy(), prof["means"][cols].to_numpy(), rtol=1e-9, atol=1e-9)
    top = np.argsort(-hist["score"], kind="stable")[:10]
    np.testing.assert_array_equal(ns["top"], top)
    for i, devs in zip(top, pl.row_deviations(prep, top)):  # stärkste Abweichungen der auffälligsten Zeilen
        biggest = np.argsort(-np.abs(ns["z"][i]), kind="stable")[:3]
        assert [ns["feature_columns"][j] for j in biggest] == [c for c, _ in devs]
        np.testing.assert_allclose([ns["z"][i, j] for j in biggest], [z for _, z in devs], rtol=1e-9, atol=1e-9)
    if prep.groups is not None:
        cmp = pl.compare_anomalies(hist["flag"], hist["score"], prep.groups)
        assert ns["g"].tolist() == prep.groups.dropna().tolist()
        if cmp["binary"]:
            from sklearn.metrics import roc_auc_score

            assert ns["positive"] == cmp["positive"]
            assert roc_auc_score(ns["y"], ns["s"]) == pytest.approx(cmp["roc_auc"], abs=1e-12)


def export(r, tmp_path, excel=None, columns=None):
    """Export 'Daten mit Anomalien' (Schritt 5): App gegen erzeugten Code, als Bytes."""
    prep, viz, hist, cfg = r["prep"], r["viz"], r["hist"], r["cfg"]
    frame = pl.anomalies_frame(viz, prep, hist["flag"], hist["score"], hist["labels"], columns)
    code = cg.export_code("anomalies", prep.recipe, excel, None, columns, list(viz.columns), model=cfg.model)
    file = i18n.tx(cg.EXPORT_FILES["anomalies"]) + ".csv"
    assert repr(file) in code
    exec(code.replace(repr(file), repr(str(tmp_path / "out.csv"))), dict(r["ns"]))  # noqa: S102
    return pl.to_csv_bytes(frame, excel=excel), (tmp_path / "out.csv").read_bytes()


def prepare(tmp_path, raw=None, group="Class", name="ID", **kw):
    raw = anomaly_like() if raw is None else raw
    _, viz, fitted, lineage, _ = load(tmp_path, raw, ANOMALY_ACTIONS if "ID" in raw.columns else None)
    feats = [c for c in viz.columns if c not in (group, name)]
    return pl.prepare_cluster_data(viz, feats, column_actions=fitted, lineage=lineage, group_col=group,
                                   name_col=name, **kw), viz


# --- Rechenlogik ----------------------------------------------------------------------------------------------------
def test_isolation_forest_auto_and_fixed_share(tmp_path):
    prep, _ = prepare(tmp_path)
    assert prep.recipe["feature_columns"][:5] == ["amount", "items", "hour", "merchant_age", "distance_km"]
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=IF), prep)
    assert info["threshold"] == pytest.approx(0.5)  # contamination = auto → Score 0.5 wie im Original-Paper
    np.testing.assert_array_equal(hist["flag"], hist["score"] > info["threshold"])  # auffällig ⇔ Score über Schwelle
    assert ((hist["score"] > 0) & (hist["score"] < 1)).all() and hist["labels"] is None
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.02, n_estimators=150), prep)
    assert abs(info["n_anomalies"] - 24) <= 1 and info["contamination"] == 0.02
    cmp = pl.compare_anomalies(hist["flag"], hist["score"], prep.groups)
    assert cmp["binary"] and cmp["positive"] == "1" and cmp["base_rate"] == pytest.approx(0.02)
    assert cmp["roc_auc"] > 0.95 and cmp["recall"] > 0.8 and cmp["tp"] + cmp["fn"] == 24
    assert cmp["precision"] == pytest.approx(cmp["tp"] / (cmp["tp"] + cmp["fp"]))
    again = pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.02, n_estimators=150), prep)[1]
    np.testing.assert_array_equal(again["score"], hist["score"])  # gleicher Seed = gleiches Ergebnis


def test_lof_scores_training_rows_and_new_rows(tmp_path):
    prep, _ = prepare(tmp_path)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=LOF, n_neighbors=25), prep)
    assert model.novelty and info["threshold"] == pytest.approx(1.5)  # auto → LOF 1.5
    np.testing.assert_array_equal(hist["flag"], hist["score"] > info["threshold"])
    assert np.median(hist["score"]) == pytest.approx(1.0, abs=0.1)  # normale Zeilen ≈ 1
    new = np.vstack([prep.X[:40], prep.X[:1] + 30.0])  # weit weg von allem
    score, flag, labels = pl.anomaly_scores(model, new, LOF)
    np.testing.assert_array_equal(flag, score > info["threshold"])
    assert labels is None and flag[-1] and score[-1] > 5
    m2, h2, i2 = pl.train_anomaly(pl.ClusterConfig(model=LOF, contamination=0.01, metric="manhattan"), prep)
    assert m2.metric == "manhattan" and abs(i2["n_anomalies"] - 12) <= 1
    tiny = pl.anomaly_params(pl.ClusterConfig(model=LOF, n_neighbors=50), 30)
    assert tiny["n_neighbors"] == 29  # höchstens Zeilen − 1


def test_dbscan_noise_is_the_anomaly(tmp_path):
    prep, _ = prepare(tmp_path)
    eps = pl.suggest_eps(prep.X, min_samples=12)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=DB, eps=eps, min_samples=12), prep)
    labels, score, flag = hist["labels"], hist["score"], hist["flag"]
    np.testing.assert_array_equal(flag, labels == -1)  # Rauschen = Anomalie
    near_one = np.abs(score - 1) < 1e-9
    assert ((score > 1) == flag)[~near_one].all()  # Score = Abstand zum nächsten Kernpunkt / eps
    assert info["threshold"] == 1.0 and info["k"] >= 1 and sum(info["sizes"]) == (~flag).sum()
    assert 0 < info["share"] < 0.2 and (score[model.core_sample_indices_] == 0).all()
    core = model.components_[:3]
    s, f, lab = pl.anomaly_scores(model, np.vstack([core, core[:1] + 50.0]), DB)
    assert (s[:3] == 0).all() and (lab[:3] == model.labels_[model.core_sample_indices_[:3]]).all()
    assert f[-1] and lab[-1] == -1 and s[-1] > 1  # neue Zeile weit weg → Rauschen
    cmp = pl.compare_anomalies(flag, score, prep.groups)
    assert cmp["roc_auc"] > 0.9


def test_dbscan_with_too_small_eps_has_no_core_point(tmp_path):
    prep, _ = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=DB, eps=1e-6, min_samples=10)
    model, hist, info = pl.train_anomaly(cfg, prep)
    assert hist["flag"].all() and np.isposinf(hist["score"]).all() and info["n_core"] == 0 and info["k"] == 0
    s, f, lab = pl.anomaly_scores(model, prep.X[:5], DB)
    assert f.all() and (lab == -1).all() and np.isposinf(s).all()
    fz = pl.threshold_assessment(hist["curve"], info, cfg)
    assert any("Kein Kernpunkt" in t for t in fz["notes"]) and fz["suggest"]  # Knick → eps vorschlagen
    assert fz["suggest"][0][1] == hist["curve"]["suggest"] > 1e-6
    cmp = pl.compare_anomalies(hist["flag"], hist["score"], prep.groups)  # ∞ überall: kein Absturz
    assert cmp["roc_auc"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="eps"):
        pl.train_anomaly(pl.ClusterConfig(model=DB, eps=0.0), prep)
    with pytest.raises(ValueError, match="contamination"):
        pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.6), prep)
    with pytest.raises(ValueError, match="Unbekanntes Modell"):
        pl.train_anomaly(pl.ClusterConfig(model="kmeans"), prep)


def test_knee_and_threshold_curve(tmp_path):
    assert pl.knee_index([1, 2, 3]) is None and pl.knee_index(np.linspace(0, 1, 50)) is None  # gerade: kein Knick
    assert pl.knee_index([5.0] * 10) is None and pl.knee_index([1, np.nan, 3, 4, 5, 6]) is None
    y = np.r_[np.linspace(10, 2, 10), np.linspace(1.99, 1.5, 90)]  # steil, dann flach
    assert pl.knee_index(y) == 9
    assert pl.suggest_share(12, 1200) == 0.01 and pl.suggest_share(1, 10**6) == 0.001
    assert pl.suggest_share(900, 1000) == pl.MAX_CONTAMINATION
    prep, _ = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=IF, contamination=0.05)
    model, hist, info = pl.train_anomaly(cfg, prep)
    curve = hist["curve"]
    assert curve["levels"] == pl.CONTAM_LEVELS and np.all(np.diff(curve["thresholds"]) <= 0)  # mehr Anteil → tiefer
    assert curve["counts"] == [int((hist["score"] > t).sum()) for t in curve["thresholds"]]
    assert len(curve["sorted"]) == int(np.ceil(0.2 * len(prep.X))) and np.all(np.diff(curve["sorted"]) <= 0)
    assert curve["knee"] is not None and 0.005 <= curve["suggest"] <= 0.05  # Knick nahe beim echten Anteil (2 %)
    fz = pl.threshold_assessment(curve, info, cfg)
    assert fz["notes"][0].startswith(f"**{pl._n(info['n_anomalies'])} von {pl._n(1200)} Zeilen auffällig")
    assert fz["suggest"] and fz["suggest"][0][1] == curve["suggest"]  # 5 % weicht deutlich ab → übernehmen anbieten


def test_dbscan_curve_and_eps_grid(tmp_path, monkeypatch):
    prep, _ = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=DB, eps=0.9, min_samples=12)
    model, hist, info = pl.train_anomaly(cfg, prep)
    curve = hist["curve"]
    assert curve["kind"] == "kdist" and curve["k"] == 12 and not curve["sampled"]
    assert np.all(np.diff(curve["kdist"]) >= 0)
    assert curve["core_share"] == pytest.approx(info["n_core"] / len(prep.X), abs=2 / len(prep.X))
    grid = curve["grid"]
    assert 0.9 in grid["eps"] and grid["eps"] == sorted(grid["eps"])
    assert np.all(np.diff(grid["noise"]) <= 0)  # größeres eps → weniger Rauschen
    assert grid["noise"][grid["eps"].index(0.9)] == pytest.approx(info["share"])
    monkeypatch.setattr(pl, "CURVE_ROWS", 500)  # große Daten: k-Distanz auf einer Stichprobe, kein eps-Raster
    model, hist, info = pl.train_anomaly(cfg, prep)
    assert hist["curve"]["sampled"] and hist["curve"]["n_rows"] == 500 and hist["curve"]["grid"] is None
    notes = pl.threshold_assessment(hist["curve"], info, cfg)["notes"]
    assert any("500 zufällige Zeilen" in t for t in notes) and any("nur bis 500 Zeilen" in t for t in notes)


def test_top_anomalies_and_deviations(tmp_path):
    prep, _ = prepare(tmp_path)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.02), prep)
    top = pl.top_anomalies(prep, hist["score"], hist["flag"], n=5)
    assert list(top.columns) == ["Rang", "Zeile", "Bezeichnung", "Anomalie-Score", "Anomalie", "größte Abweichungen (z)"]
    assert top["Anomalie-Score"].is_monotonic_decreasing and top["Anomalie"].all()
    assert top["Bezeichnung"].str.fullmatch(r"1\d{5}").all()  # ID als Bezeichnung
    assert pl.deviation_text([("vol_z", 3.24), ("ret_z", -2.9)]) == "vol_z +3.2 σ · ret_z −2.9 σ"
    assert pl.deviation_text([("ret-z", -1.0)]) == "ret-z −1.0 σ"  # Bindestrich im Namen bleibt
    prof = pl.anomaly_profiles(prep, hist["flag"])
    amount = prof["means"].loc["amount"]
    assert amount[1] > 5 * amount[0]  # auffällige Zeilen: viel höhere Beträge


def test_pca_keeps_the_anomalies_in_the_picture(tmp_path):
    prep, _ = prepare(tmp_path)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.02), prep)
    p = pl.pca_projection(prep, model, max_points=300, labels=hist["flag"].astype(int), priority=hist["flag"])
    shown = p["rows"]
    assert not len(p["centers"])  # Zentren nur bei K-Means
    assert hist["flag"][shown].sum() == hist["flag"].sum()  # alle 24 Anomalien im Bild …
    assert len(shown) == 300 and (p["labels"] == hist["flag"][shown]).all()  # … in der Stichprobe von 300 Zeilen
    assert not pl.pca_projection(prep, model, max_points=300)["labels"].any()  # ohne labels_: alle 0
    many = np.ones(len(prep.X), dtype=bool)
    assert len(pl.pca_projection(prep, model, max_points=300, priority=many)["rows"]) == 300  # höchstens die Hälfte


# --- Speichern, Laden, neue Daten -----------------------------------------------------------------------------------
@pytest.mark.parametrize("key, fmt", [(IF, ".skops"), (LOF, ".skops"), (DB, ".skops"), (IF, ".joblib")])
def test_bundle_roundtrip_and_new_data(tmp_path, key, fmt):
    prep, viz = prepare(tmp_path)
    # LOF: mehr Nachbarn als die Gruppe der Ausreißer groß ist – sonst sind sie sich gegenseitig 'normal'
    cfg = pl.ClusterConfig(model=key, contamination=0.02 if key != DB else 0.0, eps=pl.suggest_eps(prep.X, 10),
                           min_samples=10, n_neighbors=60)
    model, hist, info = pl.train_anomaly(cfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info)
    assert recipe["training"]["model"] == key and recipe["train_info"]["threshold"] == info["threshold"]
    folder = pl.save_bundle(model, recipe, tmp_path / "models" / key, fmt=fmt)
    loaded, rec2 = pl.load_bundle(folder)
    assert type(loaded) is type(model) and pl.recipe_model(rec2) == key
    listed = pl.list_bundles(tmp_path / "models")
    assert listed[0]["share"] == pytest.approx(info["share"])
    raw_new = anomaly_like(300, share=0.05, seed=11)
    res, rep, cmp = pl.score_new(loaded, rec2, raw_new)
    cols = [pl.RESULT_ANOMALY, pl.RESULT_SCORE, pl.RESULT_NOTE]
    assert list(res.columns[-len(cols):]) == cols and (pl.RESULT_CLUSTER in res.columns) == (key == DB)
    s, f, lab = pl.anomaly_scores(model, rep["X"] if "X" in rep else pl.prepare_new_data(raw_new, rec2)[0], key)
    np.testing.assert_array_equal(res[pl.RESULT_ANOMALY].to_numpy(), f.astype(int))
    np.testing.assert_allclose(res[pl.RESULT_SCORE].to_numpy(), np.round(s, 4))
    assert cmp is not None and cmp["binary"] and cmp["roc_auc"] > 0.9  # Class in den neuen Daten → Vergleich
    fz = pl.assess_scoring(rec2, res, rep, cmp)
    text = " ".join(fz["meaning"])
    assert "im Training" in text and "Am auffälligsten" in text and "Vergleich mit 'Class'" in text
    assert ("Verteilung (DBSCAN)" in text) == (key == DB)
    if key == IF:  # auf den Trainingszeilen: derselbe Score wie im Training
        res_train, _, _ = pl.score_new(loaded, rec2, pl.read_csv(tmp_path / "daten.csv")[0])
        np.testing.assert_allclose(res_train[pl.RESULT_SCORE].to_numpy()[prep.index], np.round(hist["score"], 4))


def test_kmeans_and_anomaly_models_are_not_mixed_up(tmp_path):
    prep, viz = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=IF)
    model, hist, info = pl.train_anomaly(cfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info)
    with pytest.raises(ValueError, match="assign_new"):
        pl.score_new(model, {**recipe, "training": {**recipe["training"], "model": "kmeans"}}, viz)
    with pytest.raises(ValueError, match="score_new"):
        pl.assign_new(model, recipe, viz)
    km_cfg = pl.ClusterConfig(n_clusters=3, curve=False)
    km, km_hist, km_info = pl.train_kmeans(km_cfg, prep)
    km_recipe = pl.finalize_recipe(prep.recipe, km_cfg, km_info)
    with pytest.raises(ValueError, match="kein Isolation Forest-Modell"):  # Rezept sagt IF, Datei enthält K-Means
        pl.check_recipe(recipe, km)
    with pytest.raises(ValueError, match="kein K-Means-Modell"):
        pl.check_recipe(km_recipe, model)
    lof = pl.make_anomaly_model(pl.ClusterConfig(model=LOF), len(prep.X)).set_params(novelty=False).fit(prep.X)
    lof_recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(model=LOF), info)
    with pytest.raises(ValueError, match="novelty=True"):
        pl.check_recipe(lof_recipe, lof)
    small = pl.make_anomaly_model(cfg, 50).fit(prep.X[:50, :3])
    with pytest.raises(ValueError, match="erwartet 3 Eingänge"):
        pl.check_recipe(recipe, small)
    old = json.loads(json.dumps(km_recipe))
    old["training"].pop("model", None)  # Rezept einer älteren Version → K-Means
    assert pl.recipe_model(old) == pl.MODEL_KMEANS
    pl.check_recipe(old, km)


@pytest.mark.parametrize("broken", [
    lambda r: r["train_info"].update(threshold=float("nan")),
    lambda r: r["train_info"].update(threshold="hoch"),
    lambda r: r["train_info"].update(share=7),
    lambda r: r["training"].update(model="svm"),
])
def test_broken_anomaly_recipes_are_refused(tmp_path, broken):
    prep, _ = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=IF)
    model, hist, info = pl.train_anomaly(cfg, prep)
    recipe = json.loads(json.dumps(pl._jsonable(pl.finalize_recipe(prep.recipe, cfg, info))))
    broken(recipe)
    assert pl.recipe_problem(recipe)
    with pytest.raises(ValueError):
        pl.check_recipe(recipe, model)


# --- angezeigter Code = App -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("cfg", [
    pl.ClusterConfig(model=IF),
    pl.ClusterConfig(model=IF, contamination=0.015, n_estimators=150, max_samples=300, seed=5),
    pl.ClusterConfig(model=LOF),
    pl.ClusterConfig(model=LOF, contamination=0.03, n_neighbors=15, metric="manhattan"),
    pl.ClusterConfig(model=DB, eps=0.9, min_samples=12),
    pl.ClusterConfig(model=DB, eps=2.5, min_samples=6, metric="manhattan", curve=False),
], ids=["if-auto", "if-fixed", "lof-auto", "lof-manhattan", "dbscan", "dbscan-manhattan-no-grid"])
def test_generated_code_matches_the_app(tmp_path, cfg):
    r = run_both(tmp_path, anomaly_like(), cfg, actions=ANOMALY_ACTIONS, group="Class", name="ID")
    assert pl.MODEL_CLASSES[cfg.model].__name__ + "(" in r["code"]
    assert_same(r)
    app, code = export(r, tmp_path)
    assert app == code
    if cfg.model == DB:
        assert "'Cluster'" in cg.export_code("anomalies", r["prep"].recipe, model=DB)


def test_generated_code_with_categories_gaps_minmax_and_sample(tmp_path):
    missing = {"Annual Income (k$)": {"strategy": pl.M_GROUP, "param": "Age"}, "Region": {"strategy": pl.M_MODE}}
    r = run_both(tmp_path, mall_like(600), pl.ClusterConfig(model=IF, contamination=0.05),
                 actions={"CustomerID": (pl.A_DROP, "", "")}, group="Segment", missing=missing,
                 scaler="MinMaxScaler", drop_first=True, sample=0.9)
    assert r["prep"].groups.nunique() > 2
    assert_same(r)  # mehr als zwei Gruppen: nur die Kreuztabelle
    for excel in ("de", "en"):
        app, code = export(r, tmp_path, excel=excel)
        assert app == code


def test_generated_code_with_college_names_and_dbscan_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "CURVE_ROWS", 300)  # k-Distanz auf einer Stichprobe, kein eps-Raster
    raw = college_like(500, seed=2)
    actions = {"Unnamed: 0": (pl.A_KEEP, "", "College"), "Grad.Rate": (pl.A_CLIP, ",100", "")}
    r = run_both(tmp_path, raw, pl.ClusterConfig(model=DB, eps=2.2, min_samples=8), actions=actions,
                 group="Private", name="College", index=True)
    assert r["hist"]["curve"]["sampled"] and "choice(len(X), 300" in r["code"]
    assert_same(r)
    columns = {pl.RESULT_ANOMALY: "Outlier", pl.RESULT_SCORE: "Apps"}  # Name schon vergeben → Apps_2
    app, code = export(r, tmp_path, excel="pl", columns=columns)
    assert app == code and b"Apps_2" in code


def test_knick_in_the_code_is_knee_index():
    ns = {}
    exec(cg.KNEE, {"np": np}, ns)  # noqa: S102
    rng = np.random.default_rng(0)
    cases = [np.sort(rng.exponential(size=n))[::-1] for n in (5, 12, 200)] + \
            [np.sort(rng.normal(size=300)), np.linspace(0, 1, 30), [1, 2, 3], [2.0] * 8, [1, np.inf, 3, 4, 5]]
    for y in cases:
        assert ns["knick"](np.asarray(y, dtype=float)) == pl.knee_index(y)


@pytest.mark.parametrize("key", [IF, LOF, DB])
def test_predict_code_scores_like_the_app(tmp_path, monkeypatch, key):
    prep, viz = prepare(tmp_path)
    cfg = pl.ClusterConfig(model=key, contamination=0.02 if key != DB else 0.0, eps=pl.suggest_eps(prep.X, 10),
                           min_samples=10)
    model, hist, info = pl.train_anomaly(cfg, prep)
    folder = pl.save_bundle(model, pl.finalize_recipe(prep.recipe, cfg, info), tmp_path / "m")
    new = anomaly_like(200, seed=21)
    new.to_csv(tmp_path / "neu.csv", index=False)
    code = cg.predict_code(str(folder), key)
    code = code.replace(repr(i18n.tx("neue_daten.csv")), repr(str(tmp_path / "neu.csv")))
    code = code.replace(repr(i18n.tx("anomalien.csv")), repr(str(tmp_path / "out.csv")))
    exec(code, {})  # noqa: S102
    expected, rep, _ = pl.score_new(*pl.load_bundle(folder), pd.read_csv(tmp_path / "neu.csv"))
    out = pd.read_csv(tmp_path / "out.csv")
    np.testing.assert_array_equal(out[pl.RESULT_ANOMALY], expected[pl.RESULT_ANOMALY])
    np.testing.assert_allclose(out[pl.RESULT_SCORE], expected[pl.RESULT_SCORE])
    formula = next(line for line in code.splitlines() if line.startswith("# score_new = ")).split(" = ", 1)[1]
    X_new = pl.prepare_new_data(new, pl.load_bundle(folder)[1])[0]
    from sklearn.neighbors import NearestNeighbors

    ns = {pl.MODEL_VARS[key]: model, "X_new": X_new, "NearestNeighbors": NearestNeighbors}
    np.testing.assert_allclose(eval(formula, ns), rep["score"])  # noqa: S307 – Formel im Kommentar stimmt


# --- Diagramme ------------------------------------------------------------------------------------------------------
def test_anomaly_charts_and_edge_cases(tmp_path):
    import plots

    prep, _ = prepare(tmp_path)
    model, hist, info = pl.train_anomaly(pl.ClusterConfig(model=IF, contamination=0.02), prep)
    fig = plots.fig_scores(hist["score"], info["threshold"], hist["curve"])
    fig.canvas.draw()
    labels = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert labels and not any("$" in t for t in labels)  # Log-Achse ohne mathtext-Reste
    assert any(t.get_text().startswith("Schwelle") for t in fig.axes[0].get_legend().get_texts())
    for score, thr in ((np.full(5, np.inf), 1.0), (np.array([0.4, 0.5, 0.7]), float("nan"))):
        plots.fig_scores(score, thr, None)  # keine endlichen Scores / keine Schwelle → trotzdem ein Bild
    kd = {"kdist": np.r_[np.linspace(1, 2, 500), [40.0, 90.0]], "k": 5, "knee": 499, "grid": None}
    fig = plots.fig_kdist(kd, 2.5)
    lo, hi = fig.axes[0].get_ylim()
    assert lo == 0 and hi < 40  # extreme Ausreißer abgeschnitten, damit der Knick sichtbar bleibt
    plots.fig_kdist({"kdist": np.zeros(50), "k": 5, "knee": None, "grid": None}, 0.5)
    plots.fig_kdist({"kdist": np.r_[np.ones(20), np.inf], "k": 5, "knee": None,
                     "grid": {"eps": [0.5, 1.0], "clusters": [0, 1], "noise": [1.0, 0.2]}}, 0.5)
    fig = plots.fig_dbscan_sizes([7162], 33)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("0.46%" in t for t in texts) and any("99.5%" in t for t in texts)  # nicht 0 % / 100 %
    plots.fig_dbscan_sizes([], 50)  # kein Cluster: alles Rauschen
    proj = pl.pca_projection(prep, model, labels=np.full(len(prep.X), -1), priority=hist["flag"])
    plots.fig_anomaly_pca(proj, np.ones(len(proj["rows"]), dtype=bool), dbscan=True)
    cmp = pl.compare_anomalies(hist["flag"], hist["score"], prep.groups)
    plots.fig_anomaly_groups(cmp["table"], cmp["share"], "Class")
    plt.close("all")


@pytest.mark.parametrize("key", [IF, LOF, DB])
def test_exported_script_runs(tmp_path, key):
    """Schritt 9: die ganze Pipeline als Skript – läuft für sich allein und rechnet dasselbe wie die App."""
    import os
    import subprocess

    df, viz, fitted, lineage, _ = load(tmp_path, anomaly_like(), ANOMALY_ACTIONS)
    feats = [c for c in viz.columns if c not in ("Class", "ID")]
    prep = pl.prepare_cluster_data(viz, feats, column_actions=fitted, lineage=lineage, group_col="Class",
                                   name_col="ID")
    cfg = pl.ClusterConfig(model=key, contamination=0.02 if key != DB else 0.0, eps=pl.suggest_eps(prep.X, 10),
                           min_samples=10, n_neighbors=60)
    model, hist, info = pl.train_anomaly(cfg, prep)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info)
    numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
    script = cg.full_script({"name": str(tmp_path / "daten.csv")}, fitted, numeric_raw, recipe, cfg,
                            folder=str(tmp_path / "gen"))
    script = script.replace("plt.show()", "plt.close('all')")
    script += (f"\nassert X.shape == {prep.X.shape}, X.shape\n"
               f"assert anomaly.tolist() == {hist['flag'].tolist()!r}\n"
               f"assert type(later_model).__name__ == {type(model).__name__!r}\n"
               "print('OK', int(anomaly.sum()))\n")
    (tmp_path / "gen_script.py").write_text(script, encoding="utf-8")
    out = subprocess.run([sys.executable, str(tmp_path / "gen_script.py")], capture_output=True, text=True,
                         cwd=tmp_path, env=dict(os.environ, MPLBACKEND="Agg"), timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    assert f"OK {int(hist['flag'].sum())}" in out.stdout
