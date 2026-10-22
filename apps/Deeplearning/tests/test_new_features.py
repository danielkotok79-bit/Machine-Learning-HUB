"""
Tests für die Erweiterungen: Datum → Jahr + Monat, neue Spalten (Zuordnung, Formel, Bedingung, Kopie + Aktion),
Daten speichern (Schritt 5) und das Fazit in Schritt 10.

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


# --- neue Spalten im ganzen Ablauf (Training → neue Daten) ----------------------------------------
@pytest.fixture(scope="module")
def trained():
    raw = lending_club_like(2000, seed=5)
    actions = [{"column": c, "action": pl.A_DROP} for c in ("emp_title", "title", "address", "grade", "emp_length")]
    actions += [{"column": "term", "action": pl.A_NUM}, {"column": "issue_d", "action": pl.A_DROP},
                {"column": "earliest_cr_line", "action": pl.A_YEAR_MONTH, "param": "%b-%Y", "new_name": "cr"}]
    viz, fitted, _, lin = pl.apply_column_actions(raw, actions)
    specs = [{"name": "loan_repaid", "kind": pl.N_MAP, "source": "loan_status",
              "definition": "Fully Paid=1; Charged Off=0"},
             {"name": "kredit_einkommen", "kind": pl.N_FORMULA, "definition": "loan_amnt / annual_inc"}]
    viz, new_fitted, _, sources = pl.apply_new_columns(viz, specs)
    lin = pl.merge_lineage(lin, sources)
    feats = [c for c in viz.columns if c not in ("loan_status", "loan_repaid")]
    prep = pl.prepare_ml_data(viz, "loan_repaid", feats, pl.T_BIN, positive_class="1", lineage=lin,
                              column_actions=fitted, new_columns=new_fitted, test_size=0.25)
    cfg = pl.ModelConfig(epochs=2, layers=[{"units": 8, "activation": "relu", "dropout": 0}])
    model = pl.build_model(prep.X_train.shape[1], 1, cfg, pl.T_BIN)
    pl.train_model(model, prep, cfg)
    ev = pl.evaluate(prep.y_test, pl.predict_scores(model, prep.X_test, pl.T_BIN), pl.T_BIN, prep.recipe["classes"])
    recipe = pl.finalize_recipe(prep.recipe, cfg, None, pl.metrics_summary(ev), 0.5)
    return dict(raw=raw, viz=viz, prep=prep, model=model, recipe=recipe, lin=lin)


def test_label_from_new_column_and_leak_warning(trained):
    r = trained["recipe"]
    assert r["label_source"] == ["loan_status"] and r["classes"] == ["0", "1"]
    assert pl.raw_sources(trained["lin"], "kredit_einkommen") == ["loan_amnt", "annual_inc"]
    viz = trained["viz"]
    prep = pl.prepare_ml_data(viz, "loan_repaid", ["loan_amnt", "loan_status"], pl.T_BIN, lineage=trained["lin"])
    assert any("derselben CSV-Spalte" in w for w in prep.warnings)


def test_new_data_get_new_columns_and_label(trained):
    r, model = trained["recipe"], trained["model"]
    assert {"loan_amnt", "annual_inc", "earliest_cr_line"} <= set(pl.required_raw_columns(r))
    raw_new = lending_club_like(300, seed=8)
    res, rep, ev = pl.predict_new(model, r, raw_new)
    assert ev is not None and ev["n"] == 300  # loan_repaid aus loan_status der neuen Daten
    res2, _, ev2 = pl.predict_new(model, r, raw_new.drop(columns=["loan_status"]))
    assert ev2 is None and len(res2) == 300
    with pytest.raises(pl.MissingColumnsError) as err:
        pl.predict_new(model, r, raw_new.drop(columns=["annual_inc"]))
    assert "annual_inc" in err.value.raw


# --- Fazit -----------------------------------------------------------------------------------------
def test_assessment_detects_training_rows(trained):
    r, model, raw = trained["recipe"], trained["model"], trained["raw"]
    res, rep, ev = pl.predict_new(model, r, raw)  # dieselbe Datei wie im Training
    share = rep["train_overlap"]["share"]
    assert 0.6 < share <= 1.0  # 75 % waren Trainingszeilen
    fz = pl.assess_prediction(r, res, rep, ev, 0.5)
    assert any("kennt das Modell schon" in t for t in fz["meaning"])
    assert any("ehrliche Bewertung" in t for t in fz["advice"])
    res_new, rep_new, ev_new = pl.predict_new(model, r, lending_club_like(400, seed=77))
    assert rep_new["train_overlap"]["share"] < 0.02
    fz_new = pl.assess_prediction(r, res_new, rep_new, ev_new, 0.5)
    assert not any("kennt das Modell schon" in t for t in fz_new["meaning"])


def test_assessment_explains_filled_gaps(trained):
    r, model, raw = trained["recipe"], trained["model"], trained["raw"]
    x = raw.head(100).copy()
    x.loc[x.index[:40], "revol_util"] = np.nan  # 40 % Lücken
    res, rep, ev = pl.predict_new(model, r, x)
    fz = pl.assess_prediction(r, res, rep, ev, 0.5)
    assert fz["level"] == pl.LEVEL_BAD
    gap = next(q for q in fz["quality"] if q["text"].startswith("Lücken"))
    assert any(d.startswith("revol_util: 40") and "Trainingsdaten" in d for d in gap["details"])
    assert "Lücken ergänzt" in res["Hinweis"].iloc[0]


# --- Daten speichern -------------------------------------------------------------------------------
def test_ml_ready_frame_and_csv_formats(trained):
    prep, viz = trained["prep"], trained["viz"]
    ml = pl.ml_ready_frame(viz, prep)
    r = prep.recipe
    assert list(ml.columns) == r["feature_columns"] + [r["label"], "split"]
    assert (ml["split"] == "train").sum() == len(prep.y_train) and (ml["split"] == "test").sum() == len(prep.y_test)
    assert not ml[r["feature_columns"]].isna().any().any()
    scaled = pl.ml_ready_frame(viz, prep, scaled=True)
    train_rows = scaled[scaled["split"] == "train"][r["feature_columns"]]
    assert train_rows.min().min() >= -1e-6 and train_rows.max().max() <= 1 + 1e-6  # MinMax auf Train
    de = pl.to_csv_bytes(ml.head(5), german=True)
    assert de.startswith("﻿".encode("utf-8")) and b";" in de
    back = pd.read_csv(io.BytesIO(de), sep=";", decimal=",", encoding="utf-8-sig")
    np.testing.assert_allclose(back[r["feature_columns"]].to_numpy(), ml.head(5)[r["feature_columns"]].to_numpy())
