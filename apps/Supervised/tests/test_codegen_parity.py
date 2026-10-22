"""
Regressionstests: Der angezeigte / exportierte Python-Code (codegen.py) rechnet exakt dasselbe wie die App
(pipeline.py) – auch in Grenzfällen (Bool-Label, Tausendertrennzeichen, Datum als Zahl, inf, Gruppen-Füllwerte …).

    pytest -q tests/test_codegen_parity.py
"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import pipeline as pl  # noqa: E402
from synthetic import lending_club_like  # noqa: E402

RNG = np.random.default_rng(42)
N = 400
# Der erzeugte Code soll auch mit künftigen pandas-Versionen laufen → Deprecation-Warnungen sind Fehler
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning", "error::FutureWarning")


def run_both(tmp_path, raw: pd.DataFrame, actions: dict, label: str, task: str, csv_kw=None, sample=None,
             new_columns=None, **prep_kw):
    """CSV schreiben → (a) App-Pipeline, (b) generierter Code. Rückgabe: (PrepResult, Namespace des Codes)."""
    csv_kw = csv_kw or {}
    path = tmp_path / "daten.csv"
    raw.to_csv(path, index=False, **csv_kw)
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
    feats = prep_kw.pop("features", None) or [c for c in viz.columns if c != label
                                              and not set(pl.raw_sources(lineage, c)) & set(pl.raw_sources(lineage, label))]
    prep = pl.prepare_ml_data(viz, label, feats, task, column_actions=fitted, lineage=lineage,
                              new_columns=new_fitted, **prep_kw)

    numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
    code = "\n".join([
        cg.IMPORTS,
        cg.load_code(str(path), meta["sep"], meta["decimal"], enc, sample, strip_columns=meta["strip_columns"]),
        cg.step2_code(fitted, numeric_raw, new_fitted),
        cg.step4_code(prep.recipe),
    ])
    ns = {}
    exec(compile(code, "generierter_code", "exec"), ns)  # noqa: S102 – eigener, erzeugter Code
    prep.df_viz = viz  # für den Vergleich des Exports (Schritt 5)
    return prep, ns


def assert_same(prep, ns):
    assert ns["feature_columns"] == prep.recipe["feature_columns"]
    np.testing.assert_allclose(ns["X_train"], prep.X_train, atol=2e-5)
    np.testing.assert_allclose(ns["X_test"], prep.X_test, atol=2e-5)
    y_tr, y_te = prep.fit_targets()
    np.testing.assert_allclose(np.asarray(ns["y_train"], dtype=float), y_tr, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ns["y_test"], dtype=float), y_te, rtol=1e-5, atol=1e-5)


def test_lending_pipeline_with_group_fill_and_sample(tmp_path):
    raw = lending_club_like(1500, seed=3)
    actions = {
        "grade": (pl.A_DROP, "", ""), "emp_title": (pl.A_DROP, "", ""), "emp_length": (pl.A_DROP, "", ""),
        "title": (pl.A_DROP, "", ""), "issue_d": (pl.A_DROP, "", ""),
        "home_ownership": (pl.A_REPLACE, "NONE,ANY=OTHER", ""),
        "earliest_cr_line": (pl.A_YEAR, "%b-%Y", "earliest_cr_year"),
        "address": (pl.A_LAST, "5", "zip_code"),
    }
    missing = {"revol_util": {"strategy": pl.M_DROP_ROWS}, "pub_rec_bankruptcies": {"strategy": pl.M_DROP_ROWS},
               "mort_acc": {"strategy": pl.M_GROUP, "param": "total_acc"}}
    prep, ns = run_both(tmp_path, raw, actions, "loan_status", pl.T_BIN, sample=0.8, positive_class="Fully Paid",
                        missing=missing, test_size=0.2)
    assert_same(prep, ns)


@pytest.mark.parametrize("positive", ["True", "False"])
def test_bool_label(tmp_path, positive):
    raw = pd.DataFrame({"a": RNG.normal(size=N), "c": RNG.choice(["x", "y"], N), "y": RNG.random(N) > 0.4})
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_BIN, positive_class=positive)
    assert prep.recipe["classes"][1] == positive
    assert_same(prep, ns)


@pytest.mark.parametrize("values", [[1, 2, 3], [0.5, 1.0, 1.5], [10, 20, 30, 40]])
def test_numeric_multiclass_labels(tmp_path, values):
    raw = pd.DataFrame({"a": RNG.normal(size=N), "b": RNG.normal(size=N), "y": RNG.choice(values, N)})
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_MULTI)
    assert_same(prep, ns)


def test_text_label_and_categories_with_whitespace(tmp_path):
    raw = pd.DataFrame({
        "a": RNG.normal(size=N),
        "home": RNG.choice(["RENT", "RENT ", " OWN", "MORTGAGE"], N),
        "y": RNG.choice(["yes", "yes ", "no"], N),
    })
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_BIN, positive_class="yes")
    assert prep.recipe["categories"]["home"] == ["MORTGAGE", "OWN", "RENT"]
    assert_same(prep, ns)


def test_number_extraction_thousands_and_bool_text(tmp_path):
    amounts = RNG.integers(500, 90_000, N)
    raw = pd.DataFrame({
        "amount": [f"${v:,}" for v in amounts],                     # $1,200
        "rate": [f"{v:.1f} %".replace(".", ",") for v in RNG.uniform(5, 25, N)],  # 13,5 %
        "term": RNG.choice([" 36 months", " 60 months"], N),
        "flag": RNG.choice(["True", "False"], N),
        "y": RNG.choice(["a", "b"], N),
    })
    actions = {c: (pl.A_NUM, "", "") for c in ("amount", "rate", "term", "flag")}
    prep, ns = run_both(tmp_path, raw, actions, "y", pl.T_BIN)
    assert set(prep.recipe["numeric_features"]) == {"amount", "rate", "term", "flag"}
    assert_same(prep, ns)


def test_numeric_dates_float_ints_topn_inf(tmp_path):
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
        "x": x,
        "y": RNG.choice(["p", "q"], N),
    })
    actions = {
        "issue": (pl.A_YEAR, "", "issue_year"), "issue2": (pl.A_MONTH, "", ""),
        "zip": (pl.A_CAT, "", ""), "zip_first": (pl.A_FIRST, "2", ""), "code": (pl.A_TOPN, "5", ""),
    }
    prep, ns = run_both(tmp_path, raw, actions, "y", pl.T_BIN,
                        missing={"zip": {"strategy": pl.M_CONST, "param": "unbekannt"},
                                 "zip_first": {"strategy": pl.M_MODE}, "x": {"strategy": pl.M_MEAN}})
    assert prep.recipe["inf_columns"] == ["x"]
    assert {"issue_year", "issue2"} <= set(prep.recipe["numeric_features"])
    assert_same(prep, ns)


def test_group_fill_edge_cases(tmp_path):
    g = RNG.integers(1, 8, N).astype(float)
    g[:20] = np.nan                                           # Gruppenspalte selbst mit Lücken …
    base = np.nan_to_num(g, nan=1.0)                          # … die Zeilen haben aber Werte in mort acc
    m1 = base * 2 + RNG.normal(size=N)
    m2 = base * 3 + RNG.normal(size=N)
    m1[20:60] = np.nan
    m2[40:90] = np.nan
    raw = pd.DataFrame({"total acc": g, "mort acc": m1, "other": m2, "c": RNG.choice(["u", "v"], N),
                        "leer": np.nan, "y": RNG.choice(["k1", "k2", "k3"], N)})
    missing = {"total acc": {"strategy": pl.M_MEDIAN},
               "mort acc": {"strategy": pl.M_GROUP, "param": "total acc"},
               "other": {"strategy": pl.M_GROUP, "param": "total acc"}}
    prep, ns = run_both(tmp_path, raw, {"leer": (pl.A_KEEP, "", "")}, "y", pl.T_MULTI, missing=missing,
                        features=["total acc", "mort acc", "other", "c", "leer"], scaler="StandardScaler")
    assert prep.recipe["empty_columns"] == ["leer"]
    assert_same(prep, ns)


def test_stratify_fallback_and_rare_class(tmp_path):
    y = np.array(["a"] * 200 + ["b"] * 199 + ["c"])        # Klasse c nur 1× → kein stratify möglich
    raw = pd.DataFrame({"a": RNG.normal(size=N), "y": y})
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_MULTI, stratify=True)
    assert prep.recipe["split"]["stratify_used"] is False
    assert "stratify=y" not in cg.step4_code(prep.recipe)
    assert_same(prep, ns)


def test_semicolon_decimal_comma_and_spaces_in_header(tmp_path):
    raw = pd.DataFrame({" preis ": RNG.uniform(1, 100, N).round(2), "typ": RNG.choice(["A", "B"], N),
                        "menge": RNG.uniform(0, 5, N).round(3), "y": RNG.uniform(10, 1000, N).round(2)})
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_REG, csv_kw={"sep": ";", "decimal": ","},
                        scale_label=True, scaler="keiner")
    assert "preis" in prep.recipe["features"]
    assert_same(prep, ns)


def test_replace_text_and_numeric(tmp_path):
    raw = pd.DataFrame({"n": RNG.choice([1.0, 2.0, 3.0, 99.0], N), "t": RNG.choice(["NONE", "ANY", "OWN"], N),
                        "y": RNG.choice([0, 1], N)})
    actions = {"n": (pl.A_REPLACE, "99=0", ""), "t": (pl.A_REPLACE, "NONE,ANY=OTHER", "")}
    prep, ns = run_both(tmp_path, raw, actions, "y", pl.T_BIN)
    assert prep.recipe["categories"]["t"] == ["OTHER", "OWN"]
    assert_same(prep, ns)


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
        {"name": "stichtag2", "kind": pl.N_COPY, "source": "stichtag_jahr", "action": pl.A_KEEP},
    ]
    prep, ns = run_both(tmp_path, raw, actions, "repaid", pl.T_BIN, new_columns=new_columns, positive_class="1",
                        missing={"quote": {"strategy": pl.M_MEDIAN}, "dti": {"strategy": pl.M_MEDIAN},
                                 "hohe_dti": {"strategy": pl.M_MODE}})
    r = prep.recipe
    assert {"issue_jahr", "issue_monat", "stichtag_jahr", "stichtag_monat"} <= set(r["features"])
    assert "status" not in r["features"]  # Quelle des Labels → kein Feature
    assert len(r["new_columns"]) == len(new_columns)
    assert_same(prep, ns)



def test_label_with_inf_values(tmp_path):
    """Zweiter Audit: Zeilen mit ±inf im Label werden in App und Code gleich entfernt (Regression)."""
    raw = pd.DataFrame({"a": RNG.normal(size=N), "b": RNG.choice(list("xyz"), N), "y": RNG.normal(10, 2, N)})
    raw.loc[[3, 50, 120], "y"] = [np.inf, -np.inf, np.inf]
    prep, ns = run_both(tmp_path, raw, {}, "y", pl.T_REG, missing={})
    assert prep.recipe["label_inf_removed"] and len(prep.y_train) + len(prep.y_test) == N - 3
    assert_same(prep, ns)


@pytest.mark.parametrize("task, scaled", [(pl.T_BIN, False), (pl.T_BIN, True), (pl.T_REG, True)])
def test_export_code_matches_app(tmp_path, task, scaled):
    """Zweiter Audit: Export der ML-Daten (Schritt 5) – Code = App, auch Reihenfolge der Zeilen, ein Feature,
    das selbst 'split' heißt, und ein skaliertes Regressions-Label (exportiert wird der Originalwert)."""
    y = RNG.choice(["ja", "nein"], N) if task == pl.T_BIN else RNG.normal(50, 10, N).round(3)
    raw = pd.DataFrame({"split": RNG.normal(size=N), "b": RNG.choice(list("xyz"), N), "y": y})
    raw.loc[::17, "split"] = np.nan
    kw = {"positive_class": "ja"} if task == pl.T_BIN else {"scale_label": True}
    prep, ns = run_both(tmp_path, raw, {}, "y", task, missing={"split": {"strategy": pl.M_MEDIAN}}, **kw)
    app = pl.ml_ready_frame(prep.df_viz, prep, scaled=scaled)
    assert list(app.columns)[-1] == "split_train_test"
    code = cg.export_code("ml", False, "y", scaled=scaled,
                          split_col=pl.split_column_name(prep.recipe["feature_columns"] + ["y"]))
    exec(code.replace("'ml_daten.csv'", repr(str(tmp_path / "ml.csv"))), ns)  # noqa: S102
    gen = pd.read_csv(tmp_path / "ml.csv")
    pd.testing.assert_frame_equal(gen, pd.read_csv(io.BytesIO(pl.to_csv_bytes(app))), check_exact=False, atol=2e-5)
