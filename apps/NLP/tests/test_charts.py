"""
Diagramme der NLP-Version: Textlänge je Klasse, häufigste Wörter, Modellvergleich, Validierungskurve, wichtigste
Wörter, Beitrag der Wörter eines Textes – auch mit ungewöhnlichen Daten (leer, viele Gruppen, '$' im Text).

    pytest -q tests/test_charts.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline as pl  # noqa: E402
import plots  # noqa: E402
from synthetic import news_like, yelp_like  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def texts(fig):
    return " | ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip())


@pytest.fixture(scope="module")
def yelp():
    return yelp_like(1500, seed=31)


def test_text_length_per_group_and_overall(yelp):
    lens = pl.text_length_frame(yelp, "text")
    fig = plots.fig_text_length(lens["Zeichen"], pl.label_display(yelp["stars"]), "Zeichen", 40, group_name="stars")
    assert isinstance(fig, Figure) and "stars = 1" in texts(fig) and "Median" in texts(fig)
    assert len([ax for ax in fig.axes if ax.get_visible() and ax.has_data()]) == 5
    fig2 = plots.fig_text_length(lens["Wörter"], None, "Wörter", 30, clip=False)
    assert len(fig2.axes) == 1 and "Perzentil" not in texts(fig2)


def test_text_length_with_many_groups_and_empty_values():
    n = 400
    groups = pd.Series([f"g{i % 20}" for i in range(n)])
    lengths = pd.Series(np.where(np.arange(n) % 7 == 0, np.nan, np.arange(n) % 50 + 1))
    fig = plots.fig_text_length(lengths, groups, "Zeichen")
    assert plots.tx(plots.OTHER) in texts(fig)  # höchstens 8 Gruppen, Rest → "Andere"
    empty = plots.fig_text_length(pd.Series([np.nan, np.nan]), None)
    assert "Keine Werte" in texts(empty)


def test_top_terms_grid_and_empty(yelp):
    terms = pl.top_terms(yelp["text"], pl.label_display(yelp["stars"]), top=8)
    fig = plots.fig_top_terms(terms, "Häufigste Wörter je stars", 8, group_name="stars")
    assert "stars = 5" in texts(fig) and len([a for a in fig.axes if a.has_data()]) == 5
    many = {f"k{i}": terms["1"] for i in range(20)}
    assert len([a for a in plots.fig_top_terms(many, "x", 5).axes if a.has_data()]) == plots.MAX_PANELS
    assert "Keine Wörter" in texts(plots.fig_top_terms({}, "leer"))


def test_compare_and_curve(yelp):
    prep = pl.prepare_text_data(yelp, "text", "stars", classes=["1", "5"])
    table = pl.compare_models(pl.TextModelConfig(), prep)
    fig = plots.fig_compare(table, [pl.model_name(m) for m in table["model"]])
    assert "Naive Bayes (MultinomialNB)" in texts(fig) and "ROC-AUC" in texts(fig)
    no_auc = table.assign(**{"ROC-AUC": None})
    assert "ROC-AUC" not in texts(plots.fig_compare(no_auc, ["a", "b", "c"]))
    _, hist, _ = pl.train_model(pl.TextModelConfig(model=pl.MODEL_SVC, svc_c=0.3), prep)
    curve = hist["curve"]
    fig = plots.fig_curve(curve, "Fehlerrate")
    assert fig.axes[0].get_xscale() == "log" and "eingestellt (0.3)" in texts(fig)
    assert "bester Wert" in texts(fig)


def test_top_words_binary_and_multiclass(yelp):
    prep = pl.prepare_text_data(yelp, "text", "stars", classes=["1", "5"])
    m, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep)
    tw = pl.top_words(m, prep.vectorizer, prep.recipe, 10)
    fig = plots.fig_top_words(tw, prep.recipe["classes"], 10)
    assert "spricht für '5'" in texts(fig) and "Log-Verhältnis (Naive Bayes)" in texts(fig)
    p3 = pl.prepare_text_data(news_like(600), "text", "thema")
    m3, _, _ = pl.train_model(pl.TextModelConfig(model=pl.MODEL_LOGREG, curve=False), p3)
    fig3 = plots.fig_top_words(pl.top_words(m3, p3.vectorizer, p3.recipe, 5), p3.recipe["classes"], 5)
    assert "Koeffizient" in texts(fig3) and len([a for a in fig3.axes if a.has_data()]) == 4
    long = {"kind": "coef", "classes": {f"Klasse {i}": tw["classes"]["1"] for i in range(15)}}
    assert "die ersten 12 von 15 Klassen" in texts(plots.fig_top_words(long, list(long["classes"]), 5))


def test_contribution_chart_and_dollar_signs(yelp):
    prep = pl.prepare_text_data(yelp, "text", "stars", classes=["1", "5"])
    m, _, _ = pl.train_model(pl.TextModelConfig(curve=False), prep)
    ex = pl.explain_text(m, prep.vectorizer, prep.recipe, "great food, $5 off")
    fig = plots.fig_contrib(ex["words"], ex["label"])
    assert "great" in texts(fig)
    empty = plots.fig_contrib(ex["words"].head(0), "5")
    assert "Kein Wort" in texts(empty)
    table = pd.DataFrame({"Wort": ["$5", "$$"], "Wert": [1, 1], "Beitrag": [0.5, -0.2]})
    fig = plots.fig_contrib(table, "$ a $")
    fig.canvas.draw()  # '$' wörtlich – keine Formel (text.parse_math=False)
    assert "$5" in texts(fig)


def test_confusion_proba_and_class_counts():
    cm = np.array([[50, 3], [7, 40]])
    assert "Confusion Matrix" in texts(plots.fig_confusion(cm, ["ham", "spam"]))
    scores = np.random.default_rng(1).random(200)
    fig = plots.fig_proba(scores, ["ham", "spam"], (scores > 0.5).astype(int), 0.6)
    assert "Schwelle 0.60" in texts(fig)
    fig = plots.fig_class_counts(pd.Series(["Sport", "Politik", "Sport"]), {"Sport": 0.3, "Politik": 0.7})
    assert "Anteil im Training" in texts(fig)


def test_short_titles_for_long_names():
    assert plots._short("x" * 100, 10) == "x" * 9 + "…" and plots._short("a\nb") == "a b"
