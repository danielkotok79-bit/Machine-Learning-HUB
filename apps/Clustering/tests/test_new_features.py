"""
Tests für die Erweiterungen des Grundgerüsts: Datum → Jahr + Monat, neue Spalten (Zuordnung, Formel, Bedingung,
Kopie + Aktion), Werte begrenzen – und im Clustering-Ablauf: neue Spalten für neue Daten, Daten speichern (Schritt 5)
und das Fazit in Schritt 10.

    pytest -q tests/test_new_features.py
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

import pipeline as pl  # noqa: E402
from synthetic import lending_club_like  # noqa: E402

RNG = np.random.default_rng(3)


# --- Datum → Jahr + Monat ----------------------------------------------------------------------
def test_year_month_from_text_and_number():
    df = pd.DataFrame({"d": ["Oct-2014", "Jan-2015", None], "z": [20141013, 20150101, 20160505]})
    out, _, log, lin = pl.apply_column_actions(df, [{"column": "d", "action": pl.A_YEAR_MONTH, "param": "%b-%Y"},
                                                    {"column": "z", "action": pl.A_YEAR_MONTH, "new_name": "tag"}])
    assert list(out.columns) == ["d_jahr", "d_monat", "tag_jahr", "tag_monat"]  # an der alten Position
    assert out["d_jahr"].tolist()[:2] == [2014.0, 2015.0] and out["d_monat"].tolist()[:2] == [10.0, 1.0]
    assert np.isnan(out["d_jahr"].iloc[2])
    assert out["tag_monat"].tolist() == [10.0, 1.0, 5.0] and lin["tag_jahr"] == "z"
    # neue Daten, die schon in der Form nach Schritt 2 vorliegen → Aktion wird übersprungen
    ready = out.copy()
    again, _, _, _ = pl.apply_column_actions(ready, [{"column": "d", "action": pl.A_YEAR_MONTH}], fit=False)
    assert again.equals(ready)


def test_year_month_rejects_plain_numbers():
    with pytest.raises(ValueError, match="kein Datum"):
        pl.apply_column_actions(pd.DataFrame({"x": [1.5, 2.0]}), [{"column": "x", "action": pl.A_YEAR_MONTH}])


# --- neue Spalten ------------------------------------------------------------------------------
def frame(n=200):
    return pd.DataFrame({
        "status": RNG.choice(["Fully Paid", "Charged Off", "Current"], n),
        "amount": RNG.uniform(1000, 5000, n),
        "income": np.where(np.arange(n) % 50 == 0, 0.0, RNG.uniform(10000, 90000, n)),
        "dti": np.where(np.arange(n) % 10 == 0, np.nan, RNG.uniform(0, 40, n)),
        "purpose": RNG.choice(["a", "b", "c", "d"], n, p=[0.5, 0.3, 0.15, 0.05]),
        "home ownership": RNG.choice(["RENT", "OWN"], n),
    })


def test_mapping_with_and_without_default():
    df = frame()
    out, specs, log, src = pl.apply_new_columns(df, [
        {"name": "repaid", "kind": pl.N_MAP, "source": "status", "definition": "Fully Paid=1; Charged Off=0"},
        {"name": "grob", "kind": pl.N_MAP, "source": "purpose", "definition": "a,b=häufig; sonst=selten"},
    ])
    assert out["repaid"].dtype == float
    assert out.loc[df["status"] == "Current", "repaid"].isna().all()  # nicht genannt → leer
    assert set(out["grob"]) == {"häufig", "selten"}
    assert src == {"repaid": ["status"], "grob": ["purpose"]}
    assert "ohne Zuordnung" in log[0]


def test_formula_and_condition():
    df = frame()
    out, specs, log, src = pl.apply_new_columns(df, [
        {"name": "quote", "kind": pl.N_FORMULA, "definition": "amount / income"},
        {"name": "hoch", "kind": pl.N_COND, "definition": "dti > 30 and `home ownership` == 'RENT'"},
    ])
    assert out["quote"].isna().sum() == (df["income"] == 0).sum()  # Division durch 0 → leer statt inf
    assert np.isfinite(out["quote"].dropna()).all()
    expect = ((df["dti"] > 30) & (df["home ownership"] == "RENT")).astype(float).where(df["dti"].notna())
    pd.testing.assert_series_equal(out["hoch"], expect, check_names=False)
    assert src["hoch"] == ["dti", "home ownership"]
    assert specs[0]["fitted"]["code"] == "df['amount'] / df['income']"


@pytest.mark.parametrize("formula", ["__import__('os').system('echo x')", "amount.__class__", "df['amount']",
                                     "open('x')", "[amount]", "lambda: 1", "amount if dti else 0"])
def test_formula_rejects_code(formula):
    with pytest.raises(ValueError, match="nicht erlaubt|Unbekannte Spalte"):
        pl.apply_new_columns(frame(), [{"name": "x", "kind": pl.N_FORMULA, "definition": formula}])


def test_formula_errors_are_readable():
    df = frame()
    with pytest.raises(ValueError, match="Unbekannte Spalte 'betrag'"):
        pl.apply_new_columns(df, [{"name": "x", "kind": pl.N_FORMULA, "definition": "betrag * 2"}])
    with pytest.raises(ValueError, match="keine Zahl"):
        pl.apply_new_columns(df, [{"name": "x", "kind": pl.N_FORMULA, "definition": "purpose + purpose"}])
    with pytest.raises(ValueError, match="wahr/falsch"):
        pl.apply_new_columns(df, [{"name": "x", "kind": pl.N_COND, "definition": "amount * 2"}])
    with pytest.raises(ValueError, match="gibt es schon"):
        pl.apply_new_columns(df, [{"name": "amount", "kind": pl.N_FORMULA, "definition": "dti * 2"}])


def test_copy_with_action_keeps_original_and_learned_values():
    df = frame()
    out, specs, _, _ = pl.apply_new_columns(df, [
        {"name": "top", "kind": pl.N_COPY, "source": "purpose", "action": pl.A_TOPN, "param": "2"},
        {"name": "kopie", "kind": pl.N_COPY, "source": "amount", "action": pl.A_KEEP},
    ])
    assert "purpose" in out and set(out["top"]) == {"a", "b", pl.OTHER}
    assert specs[0]["fitted"]["keep"] == ["a", "b"]
    new = pd.DataFrame({"purpose": ["d", "a"], "amount": [1.0, 2.0]})  # in neuen Daten: gelernte Liste gilt
    out2, _, _, _ = pl.apply_new_columns(new, specs, fit=False)
    assert out2["top"].tolist() == [pl.OTHER, "a"]


def test_new_columns_in_new_data_are_skipped_without_source():
    df = frame()
    _, specs, _, _ = pl.apply_new_columns(df, [{"name": "quote", "kind": pl.N_FORMULA, "definition": "amount / income"}])
    ready = pd.DataFrame({"quote": [0.1]})  # z. B. Formular in Schritt 10 – Feature liegt schon vor
    out, _, _, _ = pl.apply_new_columns(ready, specs, fit=False)
    assert out["quote"].tolist() == [0.1]


# --- Werte begrenzen (clip) – wie im Kurs: Grad.Rate 118 → 100 --------------------------------------------------
@pytest.mark.parametrize("param, expected", [
    (",100", [5.0, 100.0, 100.0, np.nan]), ("10,", [10.0, 100.0, 118.0, np.nan]), ("10,100", [10.0, 100.0, 100.0, np.nan]),
    ("0,5;99,5", [5.0, 99.5, 99.5, np.nan]),
])
def test_clip_values(param, expected):
    df = pd.DataFrame({"g": [5, 100, 118, None], "t": ["5 %", "100 %", "118 %", None]})
    out, fitted, log, _ = pl.apply_column_actions(df, [{"column": "g", "action": pl.A_CLIP, "param": param},
                                                       {"column": "t", "action": pl.A_CLIP, "param": param}])
    np.testing.assert_allclose(out["g"].to_numpy(dtype=float), expected)
    np.testing.assert_allclose(out["t"].to_numpy(dtype=float), expected)  # Text → erste Zahl → begrenzt
    assert any("begrenzt" in line for line in log)


@pytest.mark.parametrize("param, message", [
    ("", "Format min,max"), ("100", "Format min,max"), (",", "mindestens eine Grenze"), ("a,b", "keine Zahl"),
    ("100,0", "größer als max"), ("0,inf", "endliche"),
])
def test_clip_values_errors(param, message):
    with pytest.raises(ValueError, match=message):
        pl.apply_column_actions(pd.DataFrame({"g": [1.0, 2.0]}), [{"column": "g", "action": pl.A_CLIP, "param": param}])


# --- neue Spalten im ganzen Ablauf (Clustering → neue Daten) ----------------------------------------
@pytest.fixture(scope="module")
def trained():
    raw = lending_club_like(2000, seed=5)
    actions = [{"column": c, "action": pl.A_DROP} for c in ("emp_title", "title", "address", "grade", "emp_length",
                                                            "sub_grade", "purpose", "issue_d")]
    actions += [{"column": "term", "action": pl.A_NUM},
                {"column": "earliest_cr_line", "action": pl.A_YEAR_MONTH, "param": "%b-%Y", "new_name": "cr"},
                {"column": "revol_util", "action": pl.A_CLIP, "param": ",100"}]
    viz, fitted, _, lin = pl.apply_column_actions(raw, actions)
    specs = [{"name": "loan_repaid", "kind": pl.N_MAP, "source": "loan_status",
              "definition": "Fully Paid=1; Charged Off=0"},
             {"name": "kredit_einkommen", "kind": pl.N_FORMULA, "definition": "loan_amnt / annual_inc"}]
    viz, new_fitted, _, sources = pl.apply_new_columns(viz, specs)
    lin = pl.merge_lineage(lin, sources)
    feats = [c for c in viz.columns if c not in ("loan_status", "loan_repaid")]
    missing = {"mort_acc": {"strategy": pl.M_GROUP, "param": "total_acc"}, "revol_util": {"strategy": pl.M_MEDIAN},
               "pub_rec_bankruptcies": {"strategy": pl.M_MODE}}
    prep = pl.prepare_cluster_data(viz, feats, missing=missing, lineage=lin, column_actions=fitted,
                                   new_columns=new_fitted, group_col="loan_status")
    cfg = pl.ClusterConfig(n_clusters=4, n_init=3, curve=False)
    model, hist, info = pl.train_kmeans(cfg, prep)
    cmp = pl.compare_groups(model.labels_, prep.groups, 4)
    recipe = pl.finalize_recipe(prep.recipe, cfg, info, ["", "B", "", ""], cmp)
    return dict(raw=raw, viz=viz, prep=prep, model=model, hist=hist, recipe=recipe, lin=lin)


def test_group_from_the_same_source_as_a_feature_warns(trained):
    r = trained["recipe"]
    assert r["group_col"] == "loan_status" and r["group_source"] == ["loan_status"]
    assert pl.raw_sources(trained["lin"], "kredit_einkommen") == ["loan_amnt", "annual_inc"]
    prep = pl.prepare_cluster_data(trained["viz"], ["loan_amnt", "loan_repaid"], lineage=trained["lin"],
                                   group_col="loan_status")
    assert any("derselben CSV-Spalte wie die Vergleichsgruppe" in w for w in prep.warnings)


def test_new_data_get_new_columns_and_the_comparison(trained):
    r, model = trained["recipe"], trained["model"]
    assert {"loan_amnt", "annual_inc", "earliest_cr_line"} <= set(pl.required_raw_columns(r))
    assert "loan_status" not in pl.required_raw_columns(r)  # Vergleichsgruppe ist keine Pflichtspalte
    raw_new = lending_club_like(300, seed=8)
    res, rep, cmp = pl.assign_new(model, r, raw_new)
    assert cmp is not None and cmp["n"] == 300 and set(res["Cluster"]) <= {0, 1, 2, 3}
    assert "Cluster-Name" in res.columns and set(res.loc[res["Cluster"] == 1, "Cluster-Name"]) <= {"B"}
    res2, rep2, cmp2 = pl.assign_new(model, r, raw_new.drop(columns=["loan_status"]))
    assert cmp2 is None and len(res2) == 300 and "loan_status" not in rep2["extra_columns"]
    with pytest.raises(pl.MissingColumnsError) as err:
        pl.assign_new(model, r, raw_new.drop(columns=["annual_inc"]))
    assert "annual_inc" in err.value.raw


def test_same_rows_get_the_same_cluster_as_in_training(trained):
    r, model, raw, prep = trained["recipe"], trained["model"], trained["raw"], trained["prep"]
    res, rep, _ = pl.assign_new(model, r, raw)
    used = trained["viz"].index.get_indexer(prep.index)
    assert res["Cluster"].to_numpy()[used].tolist() == model.labels_.tolist()
    np.testing.assert_allclose(rep["distance"][used], trained["hist"]["dist"], rtol=1e-9, atol=1e-9)


# --- Fazit -----------------------------------------------------------------------------------------
def test_assessment_explains_filled_gaps(trained):
    r, model, raw = trained["recipe"], trained["model"], trained["raw"]
    x = raw.head(100).copy()
    x.loc[x.index[:40], "revol_util"] = np.nan  # 40 % Lücken
    res, rep, cmp = pl.assign_new(model, r, x)
    fz = pl.assess_assignment(r, res, rep, cmp)
    assert fz["level"] == pl.LEVEL_BAD
    gap = next(q for q in fz["quality"] if q["text"].startswith("Lücken"))
    # Spaltennamen im Fazit sind für Markdown escapt (revol\_util wird als revol_util angezeigt)
    assert any(d.startswith(pl.md_escape("revol_util") + ": 40") and "Trainingsdaten" in d for d in gap["details"])
    assert "Lücken ergänzt" in res["Hinweis"].iloc[0]
    assert any(m.startswith(f"Vergleich mit '{pl.md_escape('loan_status')}'") for m in fz["meaning"])


# --- Daten speichern -------------------------------------------------------------------------------
def test_cluster_ready_frame_and_csv_formats(trained):
    prep, viz = trained["prep"], trained["viz"]
    r = prep.recipe
    ml = pl.cluster_ready_frame(viz, prep)
    assert list(ml.columns) == r["feature_columns"] + ["loan_status"]  # Vergleichsgruppe als Zusatzspalte
    assert len(ml) == len(prep.X) and not ml[r["feature_columns"]].isna().any().any()
    scaled = pl.cluster_ready_frame(viz, prep, scaled=True)
    np.testing.assert_allclose(scaled[r["feature_columns"]].mean().to_numpy(), 0, atol=1e-5)  # StandardScaler
    de = pl.to_csv_bytes(ml.head(5), german=True)
    assert de.startswith("\ufeff".encode("utf-8")) and b";" in de
    back = pd.read_csv(io.BytesIO(de), sep=";", decimal=",", encoding="utf-8-sig")
    np.testing.assert_allclose(back[r["feature_columns"]].to_numpy(), ml.head(5)[r["feature_columns"]].to_numpy())
    out = pl.clusters_frame(viz, prep, trained["model"].labels_, trained["hist"]["dist"],
                            trained["recipe"]["cluster_names"],
                            {"Cluster": "Klaster", "Cluster-Name": "Nazwa klastra", "Abstand": "Odległość"})
    assert list(out.columns[-3:]) == ["Klaster", "Nazwa klastra", "Odległość"] and len(out) == len(prep.X)
