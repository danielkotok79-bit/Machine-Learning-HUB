"""
Regressionstests: Der angezeigte / exportierte Python-Code (codegen.py) rechnet exakt dasselbe wie die App
(pipeline.py) – Schritt 2 und 4 (gleiche Matrix X), K-Means (gleiche Cluster, gleiche Inertia), Elbow- und
Silhouette-Kurve, Profile, der Vergleich mit der bekannten Einteilung und der Export (Schritt 5) – auch in
Grenzfällen (Tausendertrennzeichen, Datum als Zahl, inf, Gruppen-Füllwerte, Werte begrenzen …).

    pytest -q tests/test_codegen_parity.py
"""
import contextlib
import io
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
from synthetic import college_like, lending_club_like, mall_like  # noqa: E402

RNG = np.random.default_rng(42)
N = 400
# Der erzeugte Code soll auch mit künftigen pandas-Versionen laufen → Deprecation-Warnungen sind Fehler
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning", "error::FutureWarning")


def run_both(tmp_path, raw, actions, features=None, group=None, name=None, csv_kw=None, sample=None,
             new_columns=None, cfg=None, index=False, names=None, **prep_kw):
    """CSV schreiben → (a) App-Pipeline, (b) erzeugter Code bis Schritt 8. Rückgabe: Ergebnis-dict."""
    csv_kw = csv_kw or {}
    path = tmp_path / "daten.csv"
    raw.to_csv(path, index=index, **csv_kw)
    df, enc = pl.read_csv(path)
    meta = df.attrs["csv"]
    df = pl.take_sample(df, sample)
    prof = pl.column_profile(df)
    for col, (act, param, new) in actions.items():
        i = prof.index[prof["Spalte"] == col][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [act, param, new]
    viz, fitted, _, lineage = pl.apply_column_actions(df, pl.actions_from_table(prof))
    viz, new_fitted, _, sources = pl.apply_new_columns(viz, new_columns or [])
    lineage = pl.merge_lineage(lineage, sources)
    feats = features or [c for c in viz.columns if c not in (group, name)]
    prep = pl.prepare_cluster_data(viz, feats, column_actions=fitted, lineage=lineage, new_columns=new_fitted,
                                   group_col=group, name_col=name, **prep_kw)
    cfg = cfg or pl.ClusterConfig(n_clusters=3, k_max=5)
    with warnings_off():
        model, hist, info = pl.train_kmeans(cfg, prep)
    n = len(prep.X)
    numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
    names = names or [""] * cfg.n_clusters
    code = "\n".join([
        cg.IMPORTS,
        cg.load_code(str(path), meta["sep"], meta["decimal"], enc, sample, strip_columns=meta["strip_columns"]),
        cg.step2_code(fitted, numeric_raw, new_fitted),
        cg.step4_code(prep.recipe),
        cg.model_code(cfg, n),
        cg.curve_code(cfg, n),
        cg.eval_code(prep.recipe, cfg),
    ])
    ns = {}
    with warnings_off(), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, "generierter_code", "exec"), ns)  # noqa: S102 – eigener, erzeugter Code
    plt.close("all")
    return dict(prep=prep, model=model, hist=hist, info=info, ns=ns, viz=viz, cfg=cfg, names=names, code=code)


@contextlib.contextmanager
def warnings_off():
    """K-Means mit doppelten Zeilen bzw. K = 1 und elkan warnt – das gehört zum Ergebnis, nicht zum Test."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*elkan.*", category=RuntimeWarning)
        yield


def assert_same(r):
    prep, ns, model, hist = r["prep"], r["ns"], r["model"], r["hist"]
    assert ns["feature_columns"] == prep.recipe["feature_columns"]
    np.testing.assert_array_equal(ns["X"], prep.X)  # float64: exakt dieselbe Matrix
    np.testing.assert_array_equal(ns["labels"], model.labels_)
    assert ns["kmeans"].inertia_ == model.inertia_
    if r["cfg"].curve:
        assert ns["values"] == hist["curve"]["values"]
        np.testing.assert_array_equal(ns["inertia"], hist["curve"]["inertia"])
        np.testing.assert_array_equal(np.asarray(ns["silhouette"], dtype=float), hist["curve"]["silhouette"])
    prof = pl.cluster_profiles(prep, model.labels_, r["cfg"].n_clusters)
    k = r["cfg"].n_clusters
    np.testing.assert_allclose(ns["profile"].to_numpy(), prof["means"][["Alle"] + list(range(k))].to_numpy(),
                               rtol=1e-9, atol=1e-9)
    if prep.groups is not None:
        cmp = pl.compare_groups(model.labels_, prep.groups, k)
        assert float((ns["predicted"] == ns["g"]).mean()) == pytest.approx(cmp["accuracy"])
        assert ns["g"].tolist() == prep.groups.dropna().tolist()


def export(r, what, tmp_path, excel=None, columns=None):
    """Export der App und des erzeugten Codes (Schritt 5) als Bytes."""
    prep, viz = r["prep"], r["viz"]
    if what == "step2":
        frame = viz
    elif what == "clusters":
        frame = pl.clusters_frame(viz, prep, r["model"].labels_, r["hist"]["dist"], r["names"], columns)
    else:
        frame = pl.cluster_ready_frame(viz, prep, scaled=what == "ml_scaled")
    code = cg.export_code(what, prep.recipe, excel, r["names"], columns, list(viz.columns))
    file = i18n.tx(cg.EXPORT_FILES[what]) + ".csv"
    exec(code.replace(repr(file), repr(str(tmp_path / "out.csv"))), dict(r["ns"]))  # noqa: S102
    return pl.to_csv_bytes(frame, excel=excel), (tmp_path / "out.csv").read_bytes()


# --- wie im Kurs: College_Data -------------------------------------------------------------------------------
def test_college_like_the_course(tmp_path):
    raw = college_like(500, seed=2)
    actions = {"Unnamed: 0": (pl.A_KEEP, "", "College"), "Grad.Rate": (pl.A_CLIP, ",100", "")}
    r = run_both(tmp_path, raw, actions, group="Private", name="College", index=True,
                 cfg=pl.ClusterConfig(n_clusters=2, k_max=6))
    assert "confusion_matrix(codes, lab)" in r["code"]  # wie im Kurs: Private → 0/1 gegen kmeans.labels_
    assert "clip(upper=100)" in r["code"]
    assert_same(r)


def test_lending_pipeline_with_group_fill_and_sample(tmp_path):
    raw = lending_club_like(1500, seed=3)
    actions = {
        "grade": (pl.A_DROP, "", ""), "emp_title": (pl.A_DROP, "", ""), "emp_length": (pl.A_DROP, "", ""),
        "title": (pl.A_DROP, "", ""), "issue_d": (pl.A_DROP, "", ""), "sub_grade": (pl.A_DROP, "", ""),
        "home_ownership": (pl.A_REPLACE, "NONE,ANY=OTHER", ""),
        "earliest_cr_line": (pl.A_YEAR, "%b-%Y", "earliest_cr_year"),
        "address": (pl.A_LAST, "5", "zip_code"), "revol_util": (pl.A_CLIP, "0,100", ""),
    }
    missing = {"revol_util": {"strategy": pl.M_DROP_ROWS}, "pub_rec_bankruptcies": {"strategy": pl.M_DROP_ROWS},
               "mort_acc": {"strategy": pl.M_GROUP, "param": "total_acc"}}
    r = run_both(tmp_path, raw, actions, group="loan_status", sample=0.8, missing=missing,
                 cfg=pl.ClusterConfig(n_clusters=4, n_init=3, k_max=5))
    assert_same(r)


def test_mall_categories_minmax_drop_first_elkan_random_init(tmp_path):
    raw = mall_like(500)
    missing = {"Annual Income (k$)": {"strategy": pl.M_GROUP, "param": "Age"}, "Region": {"strategy": pl.M_MODE}}
    cfg = pl.ClusterConfig(n_clusters=5, init="random", n_init=4, max_iter=50, algorithm="elkan", seed=3, k_max=6)
    r = run_both(tmp_path, raw, {"CustomerID": (pl.A_DROP, "", "")}, group="Segment", missing=missing,
                 scaler="MinMaxScaler", drop_first=True, cfg=cfg, names=["A", "", "C", "", ""])
    assert "init='random'" in r["code"] and "algorithm='elkan'" in r["code"] and "max_iter=50" in r["code"]
    assert_same(r)


def test_numeric_group_values_and_no_scaling(tmp_path):
    raw = pd.DataFrame({"a": RNG.normal(size=N), "b": RNG.normal(size=N) * 100,
                        "g": RNG.choice([1.0, 2.0, np.nan], N)})
    r = run_both(tmp_path, raw, {}, features=["a", "b"], group="g", scaler="keiner",
                 cfg=pl.ClusterConfig(n_clusters=2, curve=False))
    assert set(r["prep"].groups.dropna()) == {"1", "2"} and "no_curve" not in r["code"]
    assert_same(r)


def test_curve_and_silhouette_on_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "CURVE_ROWS", 250)
    monkeypatch.setattr(pl, "SIL_ROWS", 150)
    raw = pd.DataFrame(RNG.normal(size=(N, 3)), columns=list("abc"))
    r = run_both(tmp_path, raw, {}, cfg=pl.ClusterConfig(n_clusters=3, k_max=4))
    assert r["hist"]["curve"]["sampled"] and "choice(len(X), 250" in r["code"] and "sample_size=150" in r["code"]
    assert_same(r)


# --- Schritt 2 – Grenzfälle (wie in der Supervised-Version) -------------------------------------------------------
def test_number_extraction_thousands_and_bool_text(tmp_path):
    amounts = RNG.integers(500, 90_000, N)
    raw = pd.DataFrame({
        "amount": [f"${v:,}" for v in amounts],                     # $1,200
        "rate": [f"{v:.1f} %".replace(".", ",") for v in RNG.uniform(5, 25, N)],  # 13,5 %
        "term": RNG.choice([" 36 months", " 60 months"], N),
        "flag": RNG.choice(["True", "False"], N),
        "pct": [f"{v} %" for v in RNG.integers(0, 130, N)],        # Text → Zahl → begrenzt
    })
    actions = {c: (pl.A_NUM, "", "") for c in ("amount", "rate", "term", "flag")}
    actions["pct"] = (pl.A_CLIP, "0,100", "")
    r = run_both(tmp_path, raw, actions)
    assert set(r["prep"].recipe["numeric_features"]) == {"amount", "rate", "term", "flag", "pct"}
    assert_same(r)


def test_numeric_dates_float_ints_topn_inf_and_whitespace(tmp_path):
    days = pd.date_range("2010-01-01", periods=N, freq="7D")
    zip_codes = RNG.choice([22690.0, 5113.0, 70466.0, 30723.0, np.nan], N)
    x = RNG.normal(size=N)
    x[:5] = np.inf
    x[5:8] = -np.inf
    raw = pd.DataFrame({
        "issue": days.strftime("%Y%m%d").astype(int),       # 20141013 als Zahl
        "issue2": days.strftime("%Y%m%d").astype(int),
        "zip": zip_codes,                                     # float mit ganzen Zahlen → Kategorie
        "zip_first": zip_codes,
        "code": RNG.integers(1, 30, N),                       # int → Top-N
        "home": RNG.choice(["RENT", "RENT ", " OWN", "MORTGAGE"], N),
        "x": x,
    })
    actions = {
        "issue": (pl.A_YEAR, "", "issue_year"), "issue2": (pl.A_MONTH, "", ""),
        "zip": (pl.A_CAT, "", ""), "zip_first": (pl.A_FIRST, "2", ""), "code": (pl.A_TOPN, "5", ""),
    }
    r = run_both(tmp_path, raw, actions, missing={"zip": {"strategy": pl.M_CONST, "param": "unbekannt"},
                                                  "zip_first": {"strategy": pl.M_MODE}, "x": {"strategy": pl.M_MEAN}})
    rec = r["prep"].recipe
    assert rec["inf_columns"] == ["x"] and rec["categories"]["home"] == ["MORTGAGE", "OWN", "RENT"]
    assert {"issue_year", "issue2"} <= set(rec["numeric_features"])
    assert_same(r)


def test_group_fill_edge_cases(tmp_path):
    g = RNG.integers(1, 8, N).astype(float)
    g[:20] = np.nan                                           # Gruppenspalte selbst mit Lücken …
    base = np.nan_to_num(g, nan=1.0)
    m1 = base * 2 + RNG.normal(size=N)
    m2 = base * 3 + RNG.normal(size=N)
    m1[20:60] = np.nan
    m2[40:90] = np.nan
    raw = pd.DataFrame({"total acc": g, "mort acc": m1, "other": m2, "c": RNG.choice(["u", "v"], N),
                        "leer": np.nan})
    missing = {"total acc": {"strategy": pl.M_MEDIAN},
               "mort acc": {"strategy": pl.M_GROUP, "param": "total acc"},
               "other": {"strategy": pl.M_GROUP, "param": "total acc"}}
    r = run_both(tmp_path, raw, {"leer": (pl.A_KEEP, "", "")}, missing=missing,
                 features=["total acc", "mort acc", "other", "c", "leer"])
    assert r["prep"].recipe["empty_columns"] == ["leer"]
    assert_same(r)


def test_semicolon_decimal_comma_and_spaces_in_header(tmp_path):
    raw = pd.DataFrame({" preis ": RNG.uniform(1, 100, N).round(2), "typ": RNG.choice(["A", "B"], N),
                        "menge": RNG.uniform(0, 5, N).round(3)})
    r = run_both(tmp_path, raw, {}, csv_kw={"sep": ";", "decimal": ","})
    assert "preis" in r["prep"].recipe["features"]
    assert_same(r)


def test_replace_text_and_numeric(tmp_path):
    raw = pd.DataFrame({"n": RNG.choice([1.0, 2.0, 3.0, 99.0], N), "t": RNG.choice(["NONE", "ANY", "OWN"], N),
                        "w": RNG.normal(size=N)})
    actions = {"n": (pl.A_REPLACE, "99=0", ""), "t": (pl.A_REPLACE, "NONE,ANY=OTHER", "")}
    r = run_both(tmp_path, raw, actions)
    assert r["prep"].recipe["categories"]["t"] == ["OTHER", "OWN"]
    assert_same(r)


def test_generated_to_number_matches_pipeline():
    s = pd.Series(["$1,200", "1.234,5", "13,5 %", "1.234.567", " 36 months", "2.5", "-3", "1,2,3", "kein", None,
                   "True", "false", "1,234.5 kg"])
    ns = {}
    exec(cg.TO_NUMBER, {"pd": pd}, ns)  # noqa: S102
    np.testing.assert_array_equal(ns["to_number"](s).to_numpy(), pl.parse_numbers(s).to_numpy())


def test_year_month_and_new_columns(tmp_path):
    """Datum → Jahr + Monat (Text und Zahl 20141013) und alle vier Arten neuer Spalten – App = Code."""
    days = pd.date_range("2010-01-01", periods=N, freq="5D")
    raw = pd.DataFrame({
        "issue_d": days.strftime("%b-%Y"),
        "stichtag": days.strftime("%Y%m%d").astype(int),
        "loan_amnt": RNG.integers(1000, 40000, N).astype(float),
        "annual_inc": np.where(RNG.random(N) < 0.02, 0.0, RNG.uniform(20000, 200000, N)),  # 0 → Division durch 0
        "dti": np.where(RNG.random(N) < 0.05, np.nan, RNG.uniform(0, 40, N)),
        "purpose": RNG.choice(["debt", "card", "home", "car", "other"], N),
        "status": RNG.choice(["Fully Paid", "Charged Off", "Current"], N, p=[0.6, 0.3, 0.1]),
    })
    actions = {"issue_d": (pl.A_YEAR_MONTH, "%b-%Y", "issue"), "stichtag": (pl.A_YEAR_MONTH, "", "")}
    new_columns = [
        {"name": "repaid", "kind": pl.N_MAP, "source": "status", "definition": "Fully Paid,Current=1; Charged Off=0"},
        {"name": "zweck", "kind": pl.N_MAP, "source": "purpose", "definition": "debt,card=schulden; sonst=anderes"},
        {"name": "quote", "kind": pl.N_FORMULA, "definition": "loan_amnt / annual_inc"},
        {"name": "log_inc", "kind": pl.N_FORMULA, "definition": "log(annual_inc + 1) * 2"},
        {"name": "hohe_dti", "kind": pl.N_COND, "definition": "dti > 30 or purpose == 'car'"},
        {"name": "zweck_top", "kind": pl.N_COPY, "source": "purpose", "action": pl.A_TOPN, "param": "2"},
        {"name": "jahr_text", "kind": pl.N_COPY, "source": "issue_jahr", "action": pl.A_CAT},
        {"name": "dti_max", "kind": pl.N_COPY, "source": "dti", "action": pl.A_CLIP, "param": ",30"},
    ]
    feats = ["issue_jahr", "issue_monat", "stichtag_jahr", "stichtag_monat", "loan_amnt", "annual_inc", "dti",
             "zweck", "quote", "log_inc", "hohe_dti", "zweck_top", "jahr_text", "dti_max"]
    r = run_both(tmp_path, raw, actions, features=feats, group="repaid", new_columns=new_columns,
                 missing={"quote": {"strategy": pl.M_MEDIAN}, "dti": {"strategy": pl.M_MEDIAN},
                          "hohe_dti": {"strategy": pl.M_MODE}, "dti_max": {"strategy": pl.M_MEDIAN}})
    assert len(r["prep"].recipe["new_columns"]) == len(new_columns)
    assert_same(r)


# --- Schritt 5: Export – Code = App -------------------------------------------------------------------------------
@pytest.mark.parametrize("what", ["step2", "ml", "ml_scaled", "clusters"])
@pytest.mark.parametrize("excel", [None, "de", "en"])
def test_export_code_matches_app(tmp_path, what, excel):
    raw = college_like(200, seed=6)
    raw["Cluster"] = "schon da"  # eine Spalte heißt schon wie die neue → beide nehmen Cluster_2
    actions = {"Unnamed: 0": (pl.A_KEEP, "", "College")}
    feats = ["Apps", "Outstate", "Grad.Rate", "perc.alumni"]
    r = run_both(tmp_path, raw, actions, features=feats, group="Private", name="College", index=True,
                 cfg=pl.ClusterConfig(n_clusters=3, curve=False), names=["klein", "", "=gefährlich"])
    app, gen = export(r, what, tmp_path, excel)
    assert app == gen
    if what == "clusters":
        head = gen.decode("utf-8-sig").splitlines()[0]
        assert "Cluster_2" in head and "Cluster-Name" in head and "Abstand" in head


@pytest.mark.parametrize("lang", ["en", "pl"])
def test_export_column_names_follow_the_language(tmp_path, lang):
    raw = college_like(120, seed=7)
    r = run_both(tmp_path, raw, {"Unnamed: 0": (pl.A_KEEP, "", "College")}, features=["Apps", "Outstate"],
                 group="Private", name="College", index=True, cfg=pl.ClusterConfig(n_clusters=2, curve=False),
                 names=["x", "y"])
    i18n.set_lang(lang)
    try:
        columns = {c: i18n.tx(c) for c in (pl.RESULT_CLUSTER, pl.RESULT_NAME, pl.RESULT_DIST)}
        app, gen = export(r, "clusters", tmp_path, lang, columns)
    finally:
        i18n.set_lang("de")
    assert app == gen
    assert (("Klaster" if lang == "pl" else "Cluster name") in gen.decode("utf-8-sig").splitlines()[0])
