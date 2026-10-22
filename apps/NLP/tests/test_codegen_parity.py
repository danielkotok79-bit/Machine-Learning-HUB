"""
Regressionstests: Der angezeigte / exportierte Python-Code (codegen.py) rechnet exakt dasselbe wie die App
(pipeline.py) – Vokabular, Matrizen, Split, Wahrscheinlichkeiten, Validierungskurve, wichtigste Wörter und Exporte,
auch in Grenzfällen (Datei ohne Kopfzeile, Bool-/Zahlen-Label, leere Texte, Klassenauswahl, Stichprobe).

    pytest -q tests/test_codegen_parity.py
"""
import io
import os
import sys
import warnings
from contextlib import contextmanager
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
from synthetic import news_like, sms_like, write_sms_file, yelp_like  # noqa: E402

# Der erzeugte Code soll auch mit künftigen pandas-Versionen laufen → Deprecation-Warnungen sind Fehler
pytestmark = pytest.mark.filterwarnings("error::DeprecationWarning", "error::FutureWarning")


@contextmanager
def cwd(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_both(tmp_path, raw, actions, text, label, sample=None, new_columns=None, cfg=None, path=None, **prep_kw):
    """CSV schreiben → (a) App-Pipeline inkl. Training, (b) erzeugter Code (Schritt 1–8 und Exporte).
    Rückgabe: (prep, Namespace des Codes, extra) – extra: model, hist, viz, cfg, scores."""
    if path is None:
        path = tmp_path / "daten.csv"
        raw.to_csv(path, index=False)
    df, enc = pl.read_csv(path)
    meta = df.attrs["csv"]
    names = list(df.columns)
    df = pl.take_sample(df, sample)
    prof = pl.column_profile(df)
    for col, (act, param, new) in actions.items():
        i = prof.index[prof["Spalte"] == col][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [act, param, new]
    viz, fitted, _, lineage = pl.apply_column_actions(df, pl.actions_from_table(prof))
    viz, new_fitted, _, sources = pl.apply_new_columns(viz, new_columns or [])
    lineage = pl.merge_lineage(lineage, sources)
    prep = pl.prepare_text_data(viz, text, label, column_actions=fitted, lineage=lineage, new_columns=new_fitted,
                                **prep_kw)
    cfg = cfg or pl.TextModelConfig()
    model, hist, _ = pl.train_model(cfg, prep)
    r = prep.recipe
    scores = pl.predict_scores(model, prep.X_test, r["task"])
    calib = pl.calibration_cv(prep.y_train) if cfg.model == pl.MODEL_SVC else 5
    code = "\n".join([
        cg.IMPORTS,
        cg.load_code(str(path), meta["sep"], meta["decimal"], enc, sample, strip_columns=meta["strip_columns"],
                     header=meta["header"], names=names),
        cg.step2_code(fitted, {c for c in df.columns if pl.is_numeric(df[c])}, new_fitted),
        cg.step4_code(r), cg.model_code(cfg, calib),
        cg.curve_code(cfg, pl.vectorizer_config(r["vectorizer"]), (hist.get("curve") or {}).get("stratify", True)),
        cg.eval_code(r["task"], 0.5, cfg, r["classes"]),
        cg.export_code("split", r), cg.export_code("vocab", r), cg.predictions_export_code(r),
        "plt.close('all')",
    ])
    ns = {}
    with cwd(tmp_path):
        exec(compile(code, "generierter_code", "exec"), ns)  # noqa: S102 – eigener, erzeugter Code
    plt.close("all")
    return prep, ns, {"model": model, "hist": hist, "viz": viz, "cfg": cfg, "scores": scores, "dir": tmp_path}


def read_back(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(data), keep_default_na=False)


def assert_same(prep, ns, extra):
    r, model = prep.recipe, extra["model"]
    assert list(ns["vectorizer"].get_feature_names_out()) == list(prep.vectorizer.get_feature_names_out())
    assert (ns["X_train"] != prep.X_train).nnz == 0 and (ns["X_test"] != prep.X_test).nnz == 0
    assert np.array_equal(ns["y_train"], prep.y_train) and np.array_equal(ns["y_test"], prep.y_test)
    assert ns["classes"] == r["classes"]
    np.testing.assert_array_equal(ns["proba"], extra["scores"])  # exakt dieselben Wahrscheinlichkeiten
    curve = extra["hist"].get("curve")
    if curve:
        assert ns["values"] == curve["values"]
        assert ns["error_train"] == curve["train"] and ns["error_val"] == curve["val"]
    tw = pl.top_words(model, prep.vectorizer, r, 15)
    words, weights = ns["words"], ns["weights"]
    if r["task"] == pl.T_BIN:
        top = pd.Series(weights, index=words)
        assert top.nlargest(15).index.tolist() == tw["classes"][r["classes"][1]]["Wort"].tolist()
        assert (-top).nlargest(15).index.tolist() == tw["classes"][r["classes"][0]]["Wort"].tolist()
    else:
        for k, c in enumerate(r["classes"]):
            assert pd.Series(weights[k], index=words).nlargest(15).index.tolist() == tw["classes"][c]["Wort"].tolist()
    d = extra["dir"]
    pd.testing.assert_frame_equal(pd.read_csv(d / (i18n.tx("text_daten") + ".csv"), keep_default_na=False),
                                  read_back(pl.to_csv_bytes(pl.split_frame(prep, extra["viz"]))))
    voc = pl.vocabulary_table(prep.vectorizer, prep.X_train)
    pd.testing.assert_frame_equal(pd.read_csv(d / (i18n.tx("vokabular") + ".csv"), keep_default_na=False),
                                  read_back(pl.to_csv_bytes(voc)))
    pf = pl.predictions_frame(prep, extra["scores"], r)
    pd.testing.assert_frame_equal(pd.read_csv(d / (i18n.tx("vorhersagen_test") + ".csv"), keep_default_na=False),
                                  read_back(pl.to_csv_bytes(pf)))


# --- Szenarien ----------------------------------------------------------------------------------------------------
def test_yelp_like_course_with_step2_and_new_column(tmp_path):
    actions = {"review_id": (pl.A_DROP, "", ""), "user_id": (pl.A_DROP, "", ""), "date": (pl.A_YEAR, "", "jahr")}
    new_columns = [{"name": "text length", "kind": pl.N_COPY, "source": "text", "action": pl.A_LEN},
                   {"name": "words", "kind": pl.N_COPY, "source": "text", "action": pl.A_WORDS}]
    res = run_both(tmp_path, yelp_like(2000, seed=3), actions, "text", "stars", new_columns=new_columns,
                   classes=["1", "5"], stratify=False)
    assert res[0].recipe["classes_selected"] == ["1", "5"]
    assert_same(*res)
    assert {"text length", "words", "jahr"} <= set(res[1]["df"].columns)


def test_all_stars_tfidf_bigrams_logreg_with_validation_split(tmp_path):
    vec = pl.VectorizerConfig(kind="tfidf", ngram_max=2, min_df=2, max_df=0.9, stop_words="english",
                              sublinear_tf=True)
    cfg = pl.TextModelConfig(model=pl.MODEL_LOGREG, logreg_c=3.5, validation="split", validation_split=0.25)
    res = run_both(tmp_path, yelp_like(1800, seed=4), {}, "text", "stars", cfg=cfg, vec=vec)
    assert res[0].recipe["task"] == pl.T_MULTI
    assert_same(*res)


def test_sms_without_header_svc_calibrated(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    write_sms_file(path, 2000)
    actions = {"Spalte_1": (pl.A_KEEP, "", "label"), "Spalte_2": (pl.A_KEEP, "", "message")}
    vec = pl.VectorizerConfig(kind="tfidf", binary=True, strip_accents=True, lowercase=False, max_features=400)
    cfg = pl.TextModelConfig(model=pl.MODEL_SVC, svc_c=0.5, class_weight=True, validation="split")
    res = run_both(tmp_path, None, actions, "message", "label", cfg=cfg, path=path, vec=vec, positive_class="ham",
                   test_size=0.2)
    assert res[0].recipe["classes"] == ["spam", "ham"]
    assert "header=None" in cg.load_code(str(path), "\t", header=False, names=["Spalte_1", "Spalte_2"])
    assert_same(*res)


def test_selected_classes_blank_texts_kept_and_sample(tmp_path):
    news = news_like(1000, seed=8)
    news.loc[news.sample(40, random_state=1).index, "text"] = np.nan
    news.loc[news.sample(10, random_state=2).index, "text"] = "   "
    cfg = pl.TextModelConfig(alpha=0.3, fit_prior=False)
    res = run_both(tmp_path, news, {}, "text", "thema", sample=0.8, cfg=cfg, classes=["Sport", "Politik", "Technik"],
                   missing_text=pl.TEXT_EMPTY)
    assert res[0].recipe["text_stats"]["blank"] > 0
    assert_same(*res)


def test_multiclass_svc_without_curve_and_stratify(tmp_path):
    cfg = pl.TextModelConfig(model=pl.MODEL_SVC, curve=False)
    res = run_both(tmp_path, news_like(800, seed=9), {}, "text", "thema", cfg=cfg, stratify=False)
    assert res[0].recipe["split"]["stratify_used"] is False
    assert_same(*res)


@pytest.mark.parametrize("label_values", [[True, False], [1.0, 2.0], ["yes", "yes ", "no"], [2, 4, 8]])
def test_label_kinds(tmp_path, label_values):
    rng = np.random.default_rng(5)
    y = rng.choice(label_values, 400)
    words = {str(v).strip(): [f"w{i}{str(v).strip()}" for i in range(6)] for v in label_values}
    text = [" ".join(rng.choice(words[str(v).strip()], 4)) + " common filler" for v in y]
    raw = pd.DataFrame({"text": text, "y": y})
    res = run_both(tmp_path, raw, {}, "text", "y", cfg=pl.TextModelConfig(model=pl.MODEL_LOGREG))
    assert_same(*res)


def test_compare_code_prints_the_same_numbers(tmp_path, capsys):
    prep, ns, extra = run_both(tmp_path, yelp_like(1200, seed=6), {}, "text", "stars", classes=["1", "5"])
    cfg = pl.TextModelConfig(alpha=0.5, logreg_c=2.0, svc_c=0.3)
    capsys.readouterr()
    exec(cg.compare_code(cfg, pl.T_BIN, pl.calibration_cv(prep.y_train)), ns)  # noqa: S102
    printed = capsys.readouterr().out.strip().splitlines()
    table = pl.compare_models(cfg, prep)
    for line, (_, row) in zip(printed, table.iterrows()):
        assert line.startswith(pl.model_name(row["model"]))
        assert f"Accuracy {row['Accuracy']:.4f}" in line and f"F1 (macro) {row['F1 (macro)']:.4f}" in line
        assert f"ROC-AUC {row['ROC-AUC']:.4f}" in line


def test_full_script_runs_and_saves_the_same_pipeline(tmp_path):
    raw = yelp_like(1000, seed=12)
    path = tmp_path / "yelp.csv"
    raw.to_csv(path, index=False)
    df, enc = pl.read_csv(path)
    prof = pl.column_profile(df)
    viz, fitted, _, lineage = pl.apply_column_actions(df, pl.actions_from_table(prof))
    prep = pl.prepare_text_data(viz, "text", "stars", classes=["1", "5"], column_actions=fitted, lineage=lineage)
    cfg = pl.TextModelConfig(model=pl.MODEL_SVC, validation="split")
    model, hist, _ = pl.train_model(cfg, prep)
    src = {"path": str(path), "name": "yelp.csv", **{k: df.attrs["csv"][k] for k in ("sep", "decimal", "encoding")},
           "header": True}
    script = cg.full_script(src, fitted, {c for c in df.columns if pl.is_numeric(df[c])}, prep.recipe, cfg,
                            folder=str(tmp_path / "models" / "m"), curve_stratify=hist["curve"]["stratify"])
    ns = {}
    with cwd(tmp_path):
        exec(compile(script, "nlp_pipeline.py", "exec"), ns)  # noqa: S102
    plt.close("all")
    np.testing.assert_array_equal(ns["proba"], pl.predict_scores(model, prep.X_test, pl.T_BIN))
    later = ns["later_pipe"]
    new = yelp_like(50, seed=2)["text"]
    np.testing.assert_array_equal(later.predict_proba(new),
                                  pl.make_pipeline(prep.vectorizer, model).predict_proba(new))


def test_predict_code_snippet_runs(tmp_path):
    prep = pl.prepare_text_data(yelp_like(800, seed=13), "text", "stars", classes=["1", "5"])
    model, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep)
    folder = tmp_path / "models" / "nb"
    pl.save_bundle(pl.make_pipeline(prep.vectorizer, model), pl.finalize_recipe(prep.recipe), folder)
    yelp_like(30, seed=5).to_csv(tmp_path / i18n.tx("neue_texte.csv"), index=False)
    code = cg.predict_code(str(folder), prep.recipe)
    ns = {}
    with cwd(tmp_path):
        exec(compile(code, "predict", "exec"), ns)  # noqa: S102
    assert len(ns["result"]) == 30 and (tmp_path / i18n.tx("vorhersagen.csv")).exists()
    assert len(ns["proba"]) == 2


@pytest.mark.parametrize("kind, params", [
    ("overview", {}), ("count", {"x": "stars", "order_by": "Wert"}), ("hist", {"x": "useful", "hue": "stars"}),
    ("box", {"x": "stars", "y": "useful"}), ("heatmap", {"method": "spearman"}), ("groupby", {"by": "stars"}),
    ("length", {"text": "text", "group": "stars", "unit": "Zeichen"}), ("length", {"text": "text", "unit": "Wörter"}),
    ("terms", {"text": "text", "group": "stars", "stop_words": "english", "ngram_max": 2, "top": 5}),
    ("terms", {"text": "text", "stop_words": "", "top": 5}),
    ("examples", {"text": "text", "group": "stars", "n": 2}), ("examples", {"text": "text", "n": 2}),
])
def test_chart_code_runs(kind, params):
    ns = {"df": yelp_like(300, seed=1)}
    with warnings.catch_warnings():  # seaborn 0.13 ruft in sns.boxplot noch ax.boxplot(vert=…) auf (matplotlib 3.11)
        warnings.filterwarnings("ignore", message="vert: bool was deprecated")
        exec(cg.IMPORTS + "\n" + cg.chart_code(kind, **params), ns)  # noqa: S102
    plt.close("all")


def test_top_terms_code_matches_the_app():
    df = yelp_like(400, seed=2)
    ns = {"df": df}
    exec(cg.IMPORTS + "\n" + cg.chart_code("terms", text="text", stop_words="english", top=10), ns)  # noqa: S102
    plt.close("all")
    app = pl.top_terms(df["text"], None, 10, "english")[""]
    assert ns["top"].index.tolist() == app["Wort"].tolist() and ns["top"].tolist() == app["Anzahl"].tolist()


def test_exact_values_in_generated_code():
    cfg = pl.TextModelConfig(model=pl.MODEL_LOGREG, logreg_c=1.2345678, max_iter=250)
    assert "C=1.2345678" in cg.estimator_code(cfg) and "max_iter=250" in cg.estimator_code(cfg)
    assert cg.vectorizer_code(pl.VectorizerConfig(max_df=0.85, min_df=3)) == "CountVectorizer(min_df=3, max_df=0.85)"
    assert cg.vectorizer_code(pl.VectorizerConfig()) == "CountVectorizer()"
    assert cg.estimator_code(pl.TextModelConfig()) == "MultinomialNB()"
    svc = cg.estimator_code(pl.TextModelConfig(model=pl.MODEL_SVC), calib_cv=3)
    assert svc == "CalibratedClassifierCV(LinearSVC(random_state=101), ensemble=False, cv=3)"


def test_sms_like_frame_parity_without_file_header(tmp_path):
    """Auch eine 'normale' CSV mit Kopfzeile und Text-Label in anderer Reihenfolge der Spalten."""
    raw = sms_like(900, seed=4)[["message", "label"]]
    res = run_both(tmp_path, raw, {}, "message", "label", cfg=pl.TextModelConfig(model=pl.MODEL_LOGREG))
    assert_same(*res)
