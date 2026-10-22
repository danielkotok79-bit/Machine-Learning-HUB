"""
Tests der Clustering-Engine (pipeline.py, Schritt 6–8 und 10): K-Means wie im Kurs, Elbow-Methode und Silhouette,
Profile, Trennkraft, PCA, Vergleich mit einer bekannten Einteilung, Zuordnung neuer Daten und das Fazit dazu.

    pytest -q tests/test_clustering.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as pl  # noqa: E402
from synthetic import college_like, iris_like, mall_like  # noqa: E402

COLLEGE_FEATURES = ["Apps", "Accept", "Enroll", "Top10perc", "Top25perc", "F.Undergrad", "P.Undergrad", "Outstate",
                    "Room.Board", "Books", "Personal", "PhD", "Terminal", "S.F.Ratio", "perc.alumni", "Expend",
                    "Grad.Rate"]


CORNERS = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10], [10, 10, 10], [10, 10, 0]], dtype=float)


def blobs(k=4, n=400, seed=1, spread=0.6):
    """k klar getrennte Gruppen in 3 Dimensionen (Ecken eines Würfels)."""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, k, n)
    X = CORNERS[lab] + rng.normal(0, spread, (n, 3))
    return pd.DataFrame(X, columns=["a", "b", "c"]).assign(truth=[f"g{j}" for j in lab])


@pytest.fixture(scope="module")
def college():
    df = college_like().reset_index(names="College")
    df["Grad.Rate"] = df["Grad.Rate"].clip(upper=100)
    prep = pl.prepare_cluster_data(df, COLLEGE_FEATURES, group_col="Private", name_col="College")
    model, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2), prep)
    return dict(df=df, prep=prep, model=model, hist=hist, info=info)


# --- Schritt 6: Training -----------------------------------------------------------------------------------------
def test_kmeans_like_the_course_without_scaling():
    """Wie im Kurs: KMeans(n_clusters=2) auf den unskalierten Zahlen – die Zahlen der App = scikit-learn direkt."""
    from sklearn.cluster import KMeans

    df = college_like(seed=3)
    prep = pl.prepare_cluster_data(df, COLLEGE_FEATURES, scaler="keiner", group_col="Private")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep)
    direct = KMeans(n_clusters=2, n_init=10, random_state=pl.RANDOM_STATE).fit(df[COLLEGE_FEATURES].astype(float))
    np.testing.assert_array_equal(model.labels_, direct.labels_)
    assert info["inertia"] == pytest.approx(direct.inertia_)
    cmp = pl.compare_groups(model.labels_, prep.groups, 2)
    assert cmp["raw_accuracy"] is not None  # Vergleich wie im Kurs (Gruppen-Nummer = Cluster-Nummer)
    assert cmp["accuracy"] == pytest.approx(max(cmp["raw_accuracy"], 1 - cmp["raw_accuracy"]))


def test_training_is_reproducible_and_sizes_add_up(college):
    prep, info = college["prep"], college["info"]
    again = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep)[0]
    np.testing.assert_array_equal(again.labels_, college["model"].labels_)
    assert sum(info["sizes"]) == len(prep.X) == info["n_rows"] and info["converged"]
    assert 0 < info["r2"] < 1 and -1 <= info["silhouette"] <= 1 and info["davies_bouldin"] > 0
    assert len(info["dist_p99"]) == 2 and all(p > m for p, m in zip(info["dist_p99"], info["dist_median"]))
    np.testing.assert_allclose(college["hist"]["dist"], np.sqrt(((prep.X - college["model"].cluster_centers_[
        college["model"].labels_]) ** 2).sum(axis=1)))


# --- Schritt 7: Wahl von K -------------------------------------------------------------------------------------
def test_clear_groups_are_found_by_elbow_and_silhouette():
    prep = pl.prepare_cluster_data(blobs(4), ["a", "b", "c"], group_col="truth")
    model, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, k_max=8), prep)
    curve = hist["curve"]
    assert curve["values"] == list(range(1, 9)) and curve["elbow"] == 4 and curve["best_silhouette"] == 4
    assert np.all(np.diff(curve["inertia"]) < 0) and np.isnan(curve["silhouette"][0])
    a = pl.k_assessment(curve, info, pl.ClusterConfig(n_clusters=2))
    assert a["suggest"] == [("K = 4 übernehmen", 4)]
    assert any("Beide Methoden zeigen **K = 4**" in n for n in a["notes"])
    model4, hist4, info4 = pl.train_kmeans(pl.ClusterConfig(n_clusters=4, k_max=8), prep)
    assert pl.k_assessment(hist4["curve"], info4, pl.ClusterConfig(n_clusters=4))["suggest"] == []
    assert info4["silhouette"] > 0.7 and "starke Struktur" in pl.silhouette_verdict(info4["silhouette"])
    cmp = pl.compare_groups(model4.labels_, prep.groups, 4)
    assert cmp["accuracy"] == 1.0 and cmp["ari"] == pytest.approx(1.0) and not cmp["unmatched"]
    # gleiche Kurve für das eingestellte K wie das Modell selbst (wiederverwendet, nicht neu gerechnet)
    assert hist4["curve"]["inertia"][3] == pytest.approx(info4["inertia"])


def test_curve_uses_a_fixed_sample_for_large_data(monkeypatch):
    monkeypatch.setattr(pl, "CURVE_ROWS", 150)
    monkeypatch.setattr(pl, "SIL_ROWS", 100)
    prep = pl.prepare_cluster_data(blobs(3, n=400), ["a", "b", "c"])
    _, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, k_max=5), prep)
    curve = hist["curve"]
    assert curve["sampled"] and curve["n_rows"] == 150 and curve["values"] == [1, 2, 3, 4, 5]
    rows = pl.curve_rows(400, pl.RANDOM_STATE)
    assert len(rows) == 150 and len(set(rows)) == 150
    a = pl.k_assessment(curve, info, pl.ClusterConfig(n_clusters=3))
    assert any("Stichprobe von 150 Zeilen" in n for n in a["notes"])
    sv = pl.silhouette_values(prep.X, np.asarray(pl.make_kmeans(pl.ClusterConfig(n_clusters=3)).fit(prep.X).labels_))
    assert sv["sampled"] and sv["n"] == 100


def test_k_values_include_the_chosen_k():
    assert pl.k_values(pl.ClusterConfig(n_clusters=15, k_max=10), 1000) == list(range(1, 11)) + [15]
    assert pl.k_values(pl.ClusterConfig(n_clusters=3, k_max=10), 6) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("inertia, expected", [
    ([100, 40, 20, 15, 12, 10, 9], 3),          # deutlicher Knick bei 3 (danach 4-mal langsamer)
    ([1200, 543, 138, 19, 17, 16, 15, 14], 4),  # Kneedle (Abstand zur Geraden) hätte hier 3 gewählt
    ([100, 90, 80, 70, 60, 50, 40], None),      # Gerade → kein Knick
    ([2000, 1631, 1392, 1211, 1083, 994, 930, 859, 815, 770], None),  # Rauschen ohne Gruppen
    ([100, 100, 100], None),                    # keine Abnahme
])
def test_elbow_detection(inertia, expected):
    assert pl.elbow_k(list(range(1, len(inertia) + 1)), inertia) == expected


def test_no_suggestion_without_structure():
    rng = np.random.default_rng(0)
    prep = pl.prepare_cluster_data(pd.DataFrame(rng.normal(size=(400, 4)), columns=list("abcd")), list("abcd"))
    _, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, k_max=8), prep)
    a = pl.k_assessment(hist["curve"], info, pl.ClusterConfig(n_clusters=3))
    assert a["suggest"] == [] and any("keinen Vorschlag" in n for n in a["notes"])


@pytest.mark.parametrize("s, word", [(0.8, "starke"), (0.6, "vernünftige"), (0.3, "schwache"), (0.1, "kaum"),
                                     (float("nan"), "nicht berechenbar")])
def test_silhouette_verdicts(s, word):
    assert word in pl.silhouette_verdict(s)


def test_silhouette_values_match_the_score(college):
    sv = pl.silhouette_values(college["prep"].X, college["model"].labels_)
    assert sv["mean"] == pytest.approx(college["info"]["silhouette"]) and not sv["sampled"]
    assert len(sv["values"]) == len(college["prep"].X)


def test_assessment_warns_about_tiny_dominant_and_unconverged_clusters():
    df = blobs(2, n=300)
    df.loc[0, ["a", "b", "c"]] = [500.0, 500.0, 500.0]  # ein extremer Ausreißer bildet einen eigenen Cluster
    prep = pl.prepare_cluster_data(df, ["a", "b", "c"], scaler="keiner")
    model, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, k_max=4), prep)
    notes = " ".join(pl.k_assessment(hist["curve"], info, pl.ClusterConfig(n_clusters=2))["notes"])
    assert "Sehr kleiner Cluster" in notes and "Ein Cluster dominiert" in notes
    _, _, info1 = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, max_iter=10, init="random", n_init=1, curve=False,
                                                   seed=0), pl.prepare_cluster_data(mall_like(), ["Age", "Gender"]))
    assert isinstance(info1["converged"], bool)
    info1 = dict(info1, converged=False, warnings=["Number of distinct clusters (2) found smaller than n_clusters (3)"])
    notes = " ".join(pl.k_assessment(None, info1, pl.ClusterConfig(n_clusters=3))["notes"])
    assert "Nicht konvergiert" in notes and "Hinweis von scikit-learn" in notes


def test_duplicate_rows_give_a_scikit_learn_note_instead_of_an_error():
    df = pd.DataFrame({"a": [1.0, 2.0] * 20, "b": [5.0, 7.0] * 20})
    prep = pl.prepare_cluster_data(df, ["a", "b"])
    model, hist, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, k_max=4), prep)
    assert info["warnings"] and "distinct clusters" in info["warnings"][0]
    assert hist["curve"]["values"] == [1, 2, 3, 4]


# --- Schritt 8: Auswertung -------------------------------------------------------------------------------------
def test_profiles_are_means_in_original_units(college):
    prep, labels = college["prep"], college["model"].labels_
    prof = pl.cluster_profiles(prep, labels, 2)
    raw = pd.DataFrame(pl.invert_scaler(prep.X, prep.recipe["scaler"]), columns=prep.recipe["feature_columns"])
    expected = raw.groupby(labels).mean().T
    np.testing.assert_allclose(prof["means"][[0, 1]].to_numpy(), expected.to_numpy(), rtol=1e-9)
    np.testing.assert_allclose(prof["means"]["Alle"].to_numpy(), raw.mean().to_numpy(), rtol=1e-9)
    z = (raw.groupby(labels).mean() - raw.mean()) / raw.std(ddof=0)
    np.testing.assert_allclose(prof["z"].to_numpy(), z.T.to_numpy(), atol=1e-9)
    # Originaleinheiten: Outstate in Dollar, nicht skaliert
    assert prof["means"].loc["Outstate", "Alle"] == pytest.approx(college["df"]["Outstate"].mean())
    desc = pl.describe_clusters(prof["z"])
    assert len(desc) == 2 and all(set(d) == {"cluster", "high", "low"} for d in desc)
    for d in desc:
        zs = [z for _, z in d["high"]]
        assert zs == sorted(zs, reverse=True) and all(v >= 0.3 for v in zs)


def test_feature_separation_adds_up_to_the_explained_variance(college):
    sep = pl.feature_separation(college["prep"], college["model"].labels_, 2)
    assert list(sep.columns) == ["Feature", "R²"] and sep["R²"].between(0, 1).all()
    assert sep["R²"].is_monotonic_decreasing and set(sep["Feature"]) == set(COLLEGE_FEATURES)
    X = college["prep"].X
    total = ((X - X.mean(axis=0)) ** 2).sum(axis=0)  # StandardScaler: jede Spalte hat dieselbe Gesamtstreuung
    weighted = float((sep.set_index("Feature")["R²"] * pd.Series(total, index=COLLEGE_FEATURES)).sum() / total.sum())
    assert weighted == pytest.approx(college["info"]["r2"], rel=1e-6)


def test_feature_separation_counts_dummies_of_one_category_together():
    df = mall_like()
    prep = pl.prepare_cluster_data(df, ["Age", "Annual Income (k$)", "Spending Score (1-100)", "Region"])
    model = pl.train_kmeans(pl.ClusterConfig(n_clusters=5, curve=False), prep)[0]
    sep = pl.feature_separation(prep, model.labels_, 5)
    assert set(sep["Feature"]) == {"Age", "Annual Income (k$)", "Spending Score (1-100)", "Region"}


def test_pca_projection(college):
    proj = pl.pca_projection(college["prep"], college["model"])
    assert proj["coords"].shape == (len(college["prep"].X), 2) and proj["centers"].shape == (2, 2)
    assert 0 < sum(proj["explained"]) <= 1 and list(proj["loadings"].columns) == ["PC1", "PC2"]
    one = pl.prepare_cluster_data(college["df"], ["Outstate"])
    m1 = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), one)[0]
    p1 = pl.pca_projection(one, m1, max_points=100)
    assert p1["coords"].shape == (100, 2) and (p1["coords"][:, 1] == 0).all() and p1["explained"][1] == 0


def test_compare_groups_majority_mapping_and_unmatched_groups():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    groups = pd.Series(["a", "a", "b", "a", "a", "a", "b", None])
    cmp = pl.compare_groups(labels, groups, 3)
    assert cmp["mapping"] == {0: "a", 1: "a", 2: "b"} and cmp["n"] == 7 and cmp["missing"] == 1
    assert cmp["accuracy"] == pytest.approx(6 / 7) and cmp["baseline"] == pytest.approx(5 / 7)
    assert cmp["raw_accuracy"] is None and cmp["unmatched"] == []
    cmp2 = pl.compare_groups(np.array([0, 0, 1, 1]), pd.Series(["x", "x", "x", "y"]), 2)
    assert cmp2["mapping"] == {0: "x", 1: "x"} and cmp2["unmatched"] == ["y"]
    # ARI hängt nicht von den Cluster-Nummern ab; eine Zuordnung aus dem Training wird übernommen
    swapped = pl.compare_groups(1 - np.array([0, 0, 1, 1]), pd.Series(["p", "p", "q", "q"]), 2)
    assert swapped["ari"] == pytest.approx(1.0) and swapped["raw_accuracy"] == 0.0 and swapped["accuracy"] == 1.0
    fixed = pl.compare_groups(np.array([0, 1]), pd.Series(["q", "q"]), 2, mapping={"0": "p", "1": "q"})
    assert fixed["mapping"] == {0: "p", 1: "q"} and fixed["accuracy"] == 0.5 and fixed["classes"] == ["p", "q"]
    assert pl.compare_groups(np.array([0, 1]), pd.Series([None, None]), 2) is None


# --- Schritt 10: neue Daten --------------------------------------------------------------------------------------
def test_assignment_flags_far_and_ambiguous_rows():
    df = blobs(2, n=300, spread=0.4)
    prep = pl.prepare_cluster_data(df, ["a", "b", "c"], scaler="keiner")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=2, curve=False), prep)
    recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=2), info)
    c0, c1 = model.cluster_centers_
    new = pd.DataFrame([c0, c1, (c0 + c1) / 2 + 1e-3, c0 + 100], columns=["a", "b", "c"])
    res, rep, _ = pl.assign_new(model, recipe, new)
    assert res["Cluster"].tolist()[:2] == [0, 1] and res["Abstand"].tolist()[:2] == [0.0, 0.0]
    assert rep["row_far"].tolist() == [False, False, True, True]  # die Mitte ist weit von beiden Zentren
    assert rep["row_ambiguous"].tolist() == [False, False, False, False]  # weit weg zählt nicht als knapp
    mid = pd.DataFrame([(c0 + c1) / 2 + np.array([0.01, 0, 0])], columns=["a", "b", "c"])
    recipe_wide = dict(recipe, train_info=dict(recipe["train_info"], dist_p99=[1e9, 1e9]))
    _, rep_mid, _ = pl.assign_new(model, recipe_wide, mid)
    assert rep_mid["row_ambiguous"].tolist() == [True]


def test_assignment_report_and_names_for_new_data():
    df = iris_like()
    prep = pl.prepare_cluster_data(df, ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                                   group_col="species")
    model, _, info = pl.train_kmeans(pl.ClusterConfig(n_clusters=3, curve=False), prep)
    cmp = pl.compare_groups(model.labels_, prep.groups, 3)
    recipe = pl.finalize_recipe(prep.recipe, pl.ClusterConfig(n_clusters=3), info, ["A", "", "C"], cmp)
    res, rep, cmp_new = pl.assign_new(model, recipe, df.sample(60, random_state=1))
    assert set(res["Cluster-Name"]) <= {"A", "", "C"} and cmp_new["n"] == 60
    fz = pl.assess_assignment(recipe, res, rep, cmp_new)
    assert fz["level"] == pl.LEVEL_OK
    text = " ".join(fz["meaning"])
    assert "0 · A" in text and "Cluster 1" in text and "(Training" in text and "Vergleich mit 'species'" in text
    assert pl.cluster_label(1, ["A", "", "C"]) == "Cluster 1" and pl.cluster_label(2, ["A", "", "C"]) == "2 · C"
