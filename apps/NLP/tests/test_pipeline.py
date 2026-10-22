"""
Tests der Rechenlogik (pipeline.py): Laden, Schritt 2, Text → Zahlen, drei Modelle, Evaluation, Speichern/Laden,
neue Texte und Fazit.

    pytest -q tests/test_pipeline.py
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
from synthetic import news_like, sms_like, write_sms_file, yelp_like  # noqa: E402


@pytest.fixture(scope="module")
def yelp():
    return yelp_like(2500, seed=101)


@pytest.fixture(scope="module")
def prep15(yelp):
    return pl.prepare_text_data(yelp, "text", "stars", classes=["1", "5"])


# --- Laden ---------------------------------------------------------------------------------------
def test_sms_file_without_header_is_detected(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    write_sms_file(path, 300)
    df, enc = pl.read_csv(path)
    meta = df.attrs["csv"]
    assert df.shape == (300, 2) and list(df.columns) == ["Spalte_1", "Spalte_2"]
    assert meta["sep"] == "\t" and meta["header"] is False and enc == "utf-8"
    assert set(df["Spalte_1"]) == {"ham", "spam"}
    # erzwungen mit Kopfzeile: die erste Nachricht wird zum Spaltennamen
    df2, _ = pl.read_csv(path, header=True)
    assert len(df2) == 299 and df2.attrs["csv"]["header"] is True


def test_quoted_csv_with_commas_in_text_keeps_its_header():
    raw = yelp_like(40).to_csv(index=False).encode("utf-8")
    df, _ = pl.read_csv(raw)
    assert df.shape == (40, 10) and "text" in df.columns and df.attrs["csv"]["header"] is True


def test_semicolon_file_is_detected():
    raw = yelp_like(30)[["stars", "text"]].to_csv(index=False, sep=";").encode("utf-8")
    df, _ = pl.read_csv(raw)
    assert list(df.columns) == ["stars", "text"] and df.attrs["csv"]["sep"] == ";"


# --- Schritt 2 -------------------------------------------------------------------------------------
def test_free_text_is_kept_and_ids_dropped(yelp):
    assert pl.suggest_action(yelp["text"], "text") == (pl.A_KEEP, "")
    assert pl.suggest_action(yelp["review_id"], "review_id") == (pl.A_DROP, "")
    assert pl.is_free_text(yelp["text"]) and not pl.is_free_text(yelp["review_id"])
    assert pl.guess_text_column(yelp) == "text"


def test_text_length_and_word_count_actions():
    df = pd.DataFrame({"t": ["Great food and friendly staff", None, "ok", "  two  words "]})
    out, fitted, log, _ = pl.apply_column_actions(df, [{"column": "t", "action": pl.A_LEN, "new_name": "n"}])
    assert out["n"].tolist()[:1] == [29.0] and np.isnan(out["n"].iloc[1])
    out2, _, _, _ = pl.apply_new_columns(df, [{"name": "w", "kind": pl.N_COPY, "source": "t",
                                               "action": pl.A_WORDS}])
    assert out2["w"].tolist()[2:] == [1.0, 2.0] and np.isnan(out2["w"].iloc[1])


# --- Schritt 4 -------------------------------------------------------------------------------------
def test_prepare_selected_classes_like_the_course(prep15, yelp):
    r = prep15.recipe
    assert r["task"] == pl.T_BIN and r["classes"] == ["1", "5"] and r["class_values"] == [1, 5]
    assert r["classes_selected"] == ["1", "5"] and r["app"] == pl.APP_ID
    n = int(yelp["stars"].isin([1, 5]).sum())
    assert len(prep15.y_train) + len(prep15.y_test) == n
    assert set(prep15.idx_train) | set(prep15.idx_test) == set(yelp.index[yelp["stars"].isin([1, 5])])
    assert prep15.X_train.shape[1] == r["vocab_size"] == len(prep15.vectorizer.vocabulary_)
    assert any(line.startswith("Split zuerst") for line in prep15.log)


def test_vocabulary_is_learned_from_training_texts_only():
    df = pd.DataFrame({"text": [f"common word{i % 7}" for i in range(60)] + ["zzzonlytest"] * 0,
                       "y": ["a", "b"] * 30})
    prep = pl.prepare_text_data(df, "text", "y", test_size=0.3)
    test_only = {"uniquetestword"}
    texts = prep.text_test.copy()
    texts[0] = "uniquetestword common"
    X = prep.vectorizer.transform(texts)
    assert not test_only & set(prep.vectorizer.vocabulary_)
    assert X.shape[1] == prep.X_train.shape[1]


def test_positive_class_and_label_kinds():
    texts = [f"good {i}" if i % 2 else f"bad {i}" for i in range(60)]
    df = pd.DataFrame({"text": texts, "spam": ["spam ", "ham"] * 30, "flag": [True, False] * 30,
                       "num": [2.0, 1.0] * 30})
    p = pl.prepare_text_data(df, "text", "spam", positive_class="ham")
    assert p.recipe["classes"] == ["spam", "ham"] and p.recipe["label_kind"] == "text"
    p2 = pl.prepare_text_data(df, "text", "flag")
    assert p2.recipe["label_kind"] == "bool" and p2.recipe["class_values"] == [False, True]
    assert pl.encode_label(pd.Series(["True", "0", None]), p2.recipe)[:2].tolist() == [1.0, 0.0]
    p3 = pl.prepare_text_data(df, "text", "num")
    assert p3.recipe["classes"] == ["1", "2"]
    assert pl.encode_label(pd.Series(["2", 1, "1.0", 7]), p3.recipe)[:3].tolist() == [1.0, 0.0, 0.0]


def test_blank_texts_are_dropped_or_kept():
    df = pd.DataFrame({"text": ["nice place"] * 30 + ["awful place"] * 30, "y": ["p"] * 30 + ["n"] * 30})
    df.loc[[0, 1, 40], "text"] = [None, "   ", ""]
    drop = pl.prepare_text_data(df, "text", "y")
    keep = pl.prepare_text_data(df, "text", "y", missing_text=pl.TEXT_EMPTY)
    assert len(drop.y_train) + len(drop.y_test) == 57
    assert len(keep.y_train) + len(keep.y_test) == 60 and keep.recipe["text_stats"]["blank"] == 3


@pytest.mark.parametrize("kwargs, message", [
    (dict(text_col="n", label="y"), "enthält Zahlen"),
    (dict(text_col="text", label="text"), "verschiedene Spalten"),
    (dict(text_col="text", label="y", classes=["a"]), "mindestens 2 Klassen"),
    (dict(text_col="text", label="one"), "nur 1 Klasse"),
    (dict(text_col="text", label="y", test_size=1.5), "test_size"),
])
def test_clear_errors_in_step4(kwargs, message):
    df = pd.DataFrame({"text": [f"word{i} text" for i in range(40)], "n": range(40), "y": ["a", "b"] * 20,
                       "one": "x"})
    with pytest.raises(ValueError, match=message):
        pl.prepare_text_data(df, **kwargs)


def test_too_few_rows_and_too_many_classes():
    small = pd.DataFrame({"text": ["a b c"] * 10, "y": ["x", "y"] * 5})
    with pytest.raises(ValueError, match="zu wenig"):
        pl.prepare_text_data(small, "text", "y")
    many = pd.DataFrame({"text": [f"t {i}" for i in range(300)], "y": [f"k{i % 60}" for i in range(300)]})
    with pytest.raises(ValueError, match="höchstens 50"):
        pl.prepare_text_data(many, "text", "y")


def test_class_missing_in_training_gives_a_hint():
    df = pd.DataFrame({"text": [f"common text {i}" for i in range(41)], "y": ["a", "b"] * 20 + ["c"]})
    for seed in range(30):  # 'c' (1 Zeile) landet je nach Seed im Test → verständlicher Fehler
        try:
            p = pl.prepare_text_data(df, "text", "y", random_state=seed)
        except ValueError as exc:
            assert "kommen in den Trainingsdaten nicht vor" in str(exc)
            break
        assert "Stratify nicht möglich" in " ".join(p.warnings)
    else:
        pytest.fail("Klasse c war bei keinem Seed nur im Test")


def test_duplicates_and_same_source_warnings():
    base = sms_like(600, seed=3, dup_share=0.3)
    p = pl.prepare_text_data(base, "message", "label")
    assert any("wortgleich" in w for w in p.warnings) and p.recipe["text_stats"]["test_in_train"] >= 0.02
    lineage = {"message": ["msg"], "label": ["msg"]}
    p2 = pl.prepare_text_data(base, "message", "label", lineage=lineage)
    assert any("Datenleck" in w for w in p2.warnings)


def test_stop_words_only_gives_a_clear_message():
    df = pd.DataFrame({"text": ["the and is", "a an the"] * 15, "y": ["x", "y"] * 15})
    with pytest.raises(ValueError, match="Kein Wort bleibt im Vokabular"):
        pl.prepare_text_data(df, "text", "y", vec=pl.VectorizerConfig(stop_words="english"))


def test_vectorizer_options():
    vc = pl.VectorizerConfig(kind="tfidf", ngram_max=2, min_df=2, max_df=0.9, max_features=50, binary=True,
                             sublinear_tf=True, strip_accents=True, stop_words="english", lowercase=False)
    v = pl.make_vectorizer(vc)
    assert type(v).__name__ == "TfidfVectorizer" and v.ngram_range == (1, 2) and v.max_features == 50
    assert v.strip_accents == "unicode" and v.sublinear_tf and v.stop_words == "english" and not v.lowercase
    assert pl.vectorizer_config({"kind": "count", "unknown": 1}) == pl.VectorizerConfig()
    with pytest.raises(ValueError, match="Stoppwort"):
        pl.make_vectorizer(pl.VectorizerConfig(stop_words="german"))


def test_vocabulary_table_and_top_terms(prep15, yelp):
    voc = pl.vocabulary_table(prep15.vectorizer, prep15.X_train)
    assert list(voc.columns) == ["Wort", "Texte", "Anzahl"] and len(voc) == prep15.recipe["vocab_size"]
    assert voc["Texte"].is_monotonic_decreasing
    tf = pl.vocabulary_table(pl.fit_vectorizer(pl.VectorizerConfig(kind="tfidf"), prep15.text_train)[0])
    assert "IDF" in tf.columns and "Texte" not in tf.columns
    terms = pl.top_terms(yelp["text"], pl.label_display(yelp["stars"]), top=10)
    assert list(terms) == ["1", "2", "3", "4", "5"] and all(len(t) == 10 for t in terms.values())
    assert set(pl.top_terms(["the the the"], stop_words="english")) == set()  # nur Stoppwörter → leer


# --- Schritt 6–8 -----------------------------------------------------------------------------------
@pytest.mark.parametrize("model", pl.MODELS)
def test_three_models_learn_the_sentiment(prep15, model):
    cfg = pl.TextModelConfig(model=model)
    m, hist, info = pl.train_model(cfg, prep15)
    r = prep15.recipe
    scores = pl.predict_scores(m, prep15.X_test, r["task"])
    ev = pl.evaluate(prep15.y_test, scores, r["task"], r["classes"])
    assert ev["accuracy"] > 0.9 and ev["auc"] > 0.95 and info["test_error"] < info["baseline_error"]
    tw = pl.top_words(m, prep15.vectorizer, r, 10)
    assert {"great", "delicious", "excellent", "amazing", "friendly"} & set(tw["classes"]["5"]["Wort"])
    assert {"terrible", "rude", "worst", "awful", "horrible"} & set(tw["classes"]["1"]["Wort"])
    curve = hist["curve"]
    assert curve["param"] == ("alpha" if model == pl.MODEL_NB else "C") and len(curve["val"]) == len(curve["values"])
    assert curve["chosen"] in curve["values"]
    a = pl.curve_assessment(curve, info, cfg)
    assert a["verdict"] in {"ok", "memorize", "overfit", "underfit", "nosignal"} and a["notes"]


def test_explanation_matches_the_prediction(prep15):
    m, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep15)
    r = prep15.recipe
    text = "The food was great and friendly but cold"
    ex = pl.explain_text(m, prep15.vectorizer, r, text)
    p = pl.predict_scores(m, prep15.vectorizer.transform([text]), r["task"])[0]
    assert ex["proba"][1] == pytest.approx(p) and ex["label"] == r["classes"][int(p > 0.5)]
    # Naive Bayes binär: Summe der Beiträge + Prior = log-Odds der vorhergesagten Klasse
    lp = m.class_log_prior_
    odds = np.log(p / (1 - p))
    total = sum(ex["weights"].values()) + (lp[1] - lp[0]) * (1 if ex["pred"] == 1 else -1)
    assert total == pytest.approx(odds if ex["pred"] == 1 else -odds, rel=1e-6)
    spans = pl.highlight_spans("Great FOOD, cold!", prep15.vectorizer, ex["weights"], only={"cold"})
    assert "".join(s for s, _ in spans) == "Great FOOD, cold!"
    assert [s for s, w in spans if w is not None] == ["cold"]


def test_multiclass_and_compare(yelp):
    p = pl.prepare_text_data(news_like(900), "text", "thema", vec=pl.VectorizerConfig(kind="tfidf"))
    assert p.recipe["task"] == pl.T_MULTI and len(p.recipe["classes"]) == 4
    table = pl.compare_models(pl.TextModelConfig(), p)
    assert list(table["model"]) == pl.MODELS and (table["Accuracy"] > 0.85).all()
    m, _, _ = pl.train_model(pl.TextModelConfig(model=pl.MODEL_LOGREG, curve=False), p)
    scores = pl.predict_scores(m, p.X_test, pl.T_MULTI)
    assert scores.shape == (len(p.y_test), 4) and np.allclose(scores.sum(axis=1), 1)
    wrong = pl.misclassified(p, scores, p.recipe)
    assert list(wrong.columns) == ["Text", "tatsächlich", "vorhergesagt", "Sicherheit", "Zeile"]
    assert (wrong["tatsächlich"] != wrong["vorhergesagt"]).all()


def test_validation_split_uses_its_own_vocabulary(prep15):
    cfg = pl.TextModelConfig(model=pl.MODEL_LOGREG, validation="split", validation_split=0.25)
    X_fit, y_fit, X_val, y_val, strat = pl.curve_data(cfg, prep15)
    assert strat and X_fit.shape[0] + X_val.shape[0] == len(prep15.y_train)
    assert X_fit.shape[1] <= prep15.X_train.shape[1]


def test_exports(prep15, yelp):
    sf = pl.split_frame(prep15, yelp)
    assert list(sf.columns) == ["text", "stars", "stars_code", "split"]
    assert sf["split"].value_counts().to_dict() == {"train": len(prep15.y_train), "test": len(prep15.y_test)}
    assert sf["stars_code"].tolist() == [0 if s == 1 else 1 for s in sf["stars"]]
    m, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep15)
    scores = pl.predict_scores(m, prep15.X_test, pl.T_BIN)
    pf = pl.predictions_frame(prep15, scores, prep15.recipe, 0.5, {"richtig": "correct"})
    assert list(pf.columns) == ["text", "tatsächlich", "Vorhersage", "P(5)", "correct"]
    csv = pl.to_csv_bytes(pd.DataFrame({"t": ["=1+1", "ok"]}), excel="de").decode("utf-8-sig")
    assert "'=1+1" in csv


# --- Speichern, Laden, neue Texte ----------------------------------------------------------------
@pytest.fixture(scope="module")
def bundle(prep15, tmp_path_factory):
    m, _, info = pl.train_model(pl.TextModelConfig(model=pl.MODEL_SVC, curve=False), prep15)
    ev = pl.evaluate(prep15.y_test, pl.predict_scores(m, prep15.X_test, pl.T_BIN), pl.T_BIN, ["1", "5"])
    rec = pl.finalize_recipe(prep15.recipe, pl.TextModelConfig(model=pl.MODEL_SVC, curve=False), info,
                             pl.metrics_summary(ev), 0.5)
    folder = tmp_path_factory.mktemp("models") / "svc"
    pl.save_bundle(pl.make_pipeline(prep15.vectorizer, m), rec, folder)
    return folder


def test_save_load_and_predict(bundle, yelp):
    pipe, r = pl.load_bundle(bundle)
    assert r["model_file"] == "model.skops" and r["training"]["model"] == pl.MODEL_SVC
    assert pl.list_bundles(bundle.parent)[0]["name"] == "svc"
    new = yelp_like(300, seed=9)
    new.loc[[0, 1, 2], "text"] = ["", "zzzz qqqq xxxx", "great zzzz qqqq xxxx yyyy"]
    res, rep, ev = pl.predict_texts(pipe, r, new)
    assert res.loc[0, "Hinweis"] == "leerer Text" and res.loc[1, "Hinweis"] == "kein bekanntes Wort"
    assert res.loc[2, "Hinweis"] == "überwiegend unbekannte Wörter"
    assert ev is not None and ev["n"] == int(new["stars"].isin([1, 5]).sum())
    assert rep["label_unknown"] == int((~new["stars"].isin([1, 5])).sum())
    fz = pl.assess_prediction(r, res, rep, ev)
    assert fz["level"] in (pl.LEVEL_WARN, pl.LEVEL_BAD) and any("trainiert mit" in q["text"] for q in fz["quality"])


def test_overlap_with_training_texts_is_estimated(bundle, yelp, prep15):
    pipe, r = pl.load_bundle(bundle)
    train_rows = yelp.loc[prep15.idx_train[:200]]
    _, rep, _ = pl.predict_texts(pipe, r, train_rows)
    assert rep["train_overlap"]["share"] > 0.9
    fz = pl.assess_prediction(r, *pl.predict_texts(pipe, r, train_rows)[:1], rep, None)
    assert any("wortgleich aus dem Training" in t for t in fz["meaning"])


def test_missing_text_column_and_empty_file(bundle, yelp):
    pipe, r = pl.load_bundle(bundle)
    with pytest.raises(pl.MissingColumnsError):
        pl.predict_texts(pipe, r, yelp.drop(columns="text").head(5))
    with pytest.raises(ValueError, match="keine Datenzeilen"):
        pl.predict_texts(pipe, r, yelp.head(0))


def test_classify_single_text(bundle):
    pipe, r = pl.load_bundle(bundle)
    ex = pl.classify_text(pipe, r, "Terrible rude staff, the worst food")
    assert ex["label"] == "1" and ex["known"] >= 4
    assert pl.classify_text(pipe, r, "")["tokens"] == 0


def test_template_and_required_columns(bundle):
    _, r = pl.load_bundle(bundle)
    assert pl.template_csv(r).decode() == "text\n"
    assert pl.required_columns(r).iloc[0, 0] == "text"


def test_sms_like_end_to_end_with_rename(tmp_path):
    path = tmp_path / "SMSSpamCollection"
    write_sms_file(path, 1500)
    df, _ = pl.read_csv(path)
    out, fitted, _, lineage = pl.apply_column_actions(df, [
        {"column": "Spalte_1", "action": pl.A_KEEP, "new_name": "label"},
        {"column": "Spalte_2", "action": pl.A_KEEP, "new_name": "message"}])
    p = pl.prepare_text_data(out, "message", "label", column_actions=fitted, lineage=lineage,
                             vec=pl.VectorizerConfig(kind="tfidf", stop_words="english"))
    assert p.recipe["classes"] == ["ham", "spam"] and p.recipe["text_source"] == ["Spalte_2"]
    m, _, _ = pl.train_model(pl.TextModelConfig(curve=False), p)
    rec = pl.finalize_recipe(p.recipe, pl.TextModelConfig(curve=False))
    new, _ = pl.read_csv(path)  # neue Daten in derselben Form (ohne Kopfzeile) → Umbenennen wird wiederholt
    res, rep, ev = pl.predict_texts(pl.make_pipeline(p.vectorizer, m), rec, new.head(400))
    assert ev["accuracy"] > 0.9 and "P(spam)" in res.columns


def test_text_fingerprints_ignore_case_and_spacing():
    a = pl.text_fingerprints(["Hello  World", "hello world", "Hello World!"])
    assert a[0] == a[1] != a[2]


def test_md_escape_neutralises_markdown():
    assert pl.md_escape("![x](http://evil)\n**b**") == "\\!\\[x\\]\\(http://evil\\) \\*\\*b\\*\\*"
    assert pl.md_escape("$5 off") == "\\$5 off"


def test_library_versions_table():
    table, cpus = pl.library_versions()
    assert cpus >= 1 and "scikit-learn" in set(table.iloc[:, 0])


def test_read_csv_from_bytes_with_latin1():
    raw = "text;label\nCafé très bon;pos\nMauvais café;neg\n".encode("latin-1")
    df, enc = pl.read_csv(raw, sep=";")
    assert enc == "latin-1" and df["text"].iloc[0] == "Café très bon"


def test_to_csv_international_roundtrip():
    df = pd.DataFrame({"text": ["a, b", "c"], "x": [1.5, 2.0]})
    back = pd.read_csv(io.BytesIO(pl.to_csv_bytes(df)))
    pd.testing.assert_frame_equal(back, df)
