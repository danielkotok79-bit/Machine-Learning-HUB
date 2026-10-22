"""
Neue Diagramme in Schritt 3: Jointplot (scatter/kde/hex/reg), Dichte 2D (KDE), Regression (lmplot) und
Histogramm je Gruppe – Zeichnen, ungewöhnliche Daten, angezeigter Python-Code und Klick-Test der Oberfläche.

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
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import plots  # noqa: E402
from synthetic import lending_club_like  # noqa: E402

APP = str(ROOT / "app.py")
NEW_CHARTS = ["Jointplot", "Dichte 2D (KDE)", "Regression (lmplot)", "Histogramm je Gruppe"]


@pytest.fixture(scope="module")
def df():
    d = lending_club_like(3000, seed=11)
    d["credit.policy"] = (d["int_rate"] < 14).astype(int)  # wie im LendingClub-Beispiel des Kurses (0/1)
    d["bezahlt"] = d["loan_status"] == "Fully Paid"
    d["leer"] = np.nan
    d["konst"] = 5.0
    d["doppelt"] = d["loan_amnt"] * 2  # exakt linear → keine 2D-Dichte möglich
    d["mit_inf"] = d["dti"].where(d.index % 40 != 0, np.inf)
    d["ganz"] = d["open_acc"].astype("Int64").where(d.index % 9 != 0)  # Int64 mit Lücken
    return d


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def texts(fig):
    return " | ".join(t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip())


# --- Zeichnen ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kind", plots.JOINT_KINDS)
def test_jointplot_every_kind(df, kind):
    fig = plots.fig_joint(df, "loan_amnt", "int_rate", kind, "loan_status", max_points=800)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 3  # Mitte + Verteilungen am oberen und rechten Rand
    assert "int_rate gegen loan_amnt (800 Punkte, Stichprobe)" in texts(fig)


def test_kde2d_with_and_without_filter(df):
    fig = plots.fig_kde2d(df, "dti", "int_rate", max_points=600)
    assert "Dichte: int_rate gegen dti" in texts(fig)
    fig = plots.fig_kde2d(df, "dti", "int_rate", "home_ownership", "RENT", fill=False, max_points=600)
    assert "home_ownership = RENT" in texts(fig)
    # Filter auf Zahlen- und Bool-Spalten: Vergleich als Text wie in der Auswahlliste
    assert len(plots.filter_rows(df, "credit.policy", "1")) == int((df["credit.policy"] == 1).sum())
    assert len(plots.filter_rows(df, "bezahlt", "True")) == int(df["bezahlt"].sum())
    assert plots.category_values(df["credit.policy"]) == ["0", "1"]


def test_lmplot_hue_and_columns(df):
    fig = plots.fig_lm(df, "dti", "int_rate", "credit.policy", "term", max_points=800)
    assert len([ax for ax in fig.axes if ax.has_data()]) == 2  # eine Teilgrafik je Laufzeit
    assert "Regression: int_rate gegen dti" in texts(fig)
    fig = plots.fig_lm(df, "dti", "int_rate", None, "home_ownership", max_points=5000)  # > 3 Gruppen → 3 je Zeile
    assert len([ax for ax in fig.axes if ax.has_data()]) == df["home_ownership"].nunique()


def test_histogram_per_group_uses_the_same_bins(df):
    fig = plots.fig_hist_groups(df, "int_rate", "credit.policy", 25)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["credit.policy=0", "credit.policy=1"]
    left = [round(p.get_x(), 9) for p in ax.patches]
    assert len(left) == 50 and left[:25] == left[25:]  # beide Gruppen mit denselben 25 Klassen
    assert sum(p.get_height() for p in ax.patches) == df["int_rate"].notna().sum()


@pytest.mark.parametrize("draw, message", [
    (lambda d: plots.fig_joint(d, "leer", "int_rate"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_joint(d, "konst", "int_rate", "kde"), "Dichte-Schätzung"),
    (lambda d: plots.fig_joint(d.head(1), "dti", "int_rate", "reg"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_joint(d, "mit_inf", "ganz", "hex", "bezahlt"), None),
    (lambda d: plots.fig_joint(d, "loan_amnt", "doppelt", "kde"), "Dichte-Schätzung"),
    (lambda d: plots.fig_kde2d(d, "leer", "int_rate"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_kde2d(d, "konst", "int_rate"), "Dichte-Schätzung"),
    (lambda d: plots.fig_kde2d(d, "loan_amnt", "doppelt"), "Dichte-Schätzung"),
    (lambda d: plots.fig_kde2d(d, "dti", "int_rate", "home_ownership", "GIBT_ES_NICHT"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_kde2d(d, "mit_inf", "ganz", max_points=500), None),
    (lambda d: plots.fig_lm(d, "leer", "int_rate", "loan_status"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_lm(d, "konst", "int_rate", "term", "term", max_points=500), None),
    (lambda d: plots.fig_lm(d, "mit_inf", "ganz", "bezahlt", max_points=500), None),
    (lambda d: plots.fig_hist_groups(d, "leer", "loan_status"), "Keine Zeilen mit Werten"),
    (lambda d: plots.fig_hist_groups(d, "mit_inf", "bezahlt"), None),
    (lambda d: plots.fig_hist_groups(d, "ganz", "purpose"), None),  # 14 Gruppen → 7 + "Andere"
])
def test_unusual_data_gives_a_figure_instead_of_an_error(df, draw, message):
    fig = draw(df)
    assert isinstance(fig, Figure)
    if message:
        assert message in texts(fig)


@pytest.mark.parametrize("draw", [
    lambda d: plots.fig_joint(d, "dti", "dti"), lambda d: plots.fig_kde2d(d, "dti", "dti"),
    lambda d: plots.fig_lm(d, "dti", "dti"), lambda d: plots.fig_hist_groups(d, "term", "term"),
])
def test_the_same_column_twice_is_rejected(df, draw):
    with pytest.raises(ValueError, match="zwei verschiedene Spalten"):
        draw(df)


# --- angezeigter Python-Code -----------------------------------------------------------------------------------------
CODE = [
    ("joint", dict(x="loan_amnt", y="int_rate", joint_kind="scatter", hue="loan_status", n=500), "hue='loan_status'"),
    ("joint", dict(x="loan_amnt", y="int_rate", joint_kind="kde", n=300), "kind='kde'"),
    ("joint", dict(x="loan_amnt", y="int_rate", joint_kind="hex", hue="loan_status", n=500), "kind='hex')"),
    ("joint", dict(x="loan_amnt", y="int_rate", joint_kind="reg", n=500), "kind='reg'"),
    ("kde2d", dict(x="dti", y="int_rate", n=300), "fill=True, thresh=0.05"),
    ("kde2d", dict(x="dti", y="int_rate", filter_col="home_ownership", filter_value="RENT", fill=False, n=300),
     "df[df['home_ownership'] == 'RENT']"),
    ("kde2d", dict(x="dti", y="int_rate", filter_col="credit.policy", filter_value="1", filter_numeric=True, n=300),
     "df[df['credit.policy'] == 1]"),
    ("kde2d", dict(x="dti", y="int_rate", filter_col="bezahlt", filter_value="True", filter_numeric=True, n=300),
     "df[df['bezahlt'] == True]"),
    ("lm", dict(x="dti", y="int_rate", hue="credit.policy", col="term", n=500), "col='term'"),
    ("lm", dict(x="dti", y="int_rate", col="home_ownership", wrap=True, n=500), "col_wrap=3"),
    ("hist_groups", dict(x="int_rate", group="credit.policy", bins=30), "label='credit.policy=' + str(value)"),
]


@pytest.mark.parametrize("kind, p, expected", CODE)
def test_generated_chart_code_runs(df, kind, p, expected):
    code = cg.chart_code(kind, **p)
    assert expected in code
    if kind == "joint" and p["joint_kind"] in ("hex", "reg"):
        assert "hue=" not in code  # Farbe gibt es bei hex und reg nicht
    ns = {"df": df.drop(columns=["leer", "konst", "doppelt", "mit_inf", "ganz"]), "pd": pd, "np": np, "sns": sns,
          "plt": plt}
    exec(code, ns)  # noqa: S102
    assert plt.get_fignums()
    if p.get("filter_col"):  # der Code filtert dieselben Zeilen wie die App
        assert len(ns["df_f"]) == len(plots.filter_rows(df, p["filter_col"], p["filter_value"]))
    if kind == "hist_groups":
        legend = plt.gca().get_legend_handles_labels()[1]
        assert legend == ["credit.policy=0", "credit.policy=1"]


# --- Oberfläche --------------------------------------------------------------------------------------------------
def test_new_charts_in_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    lending_club_like(1500, seed=101).to_csv(tmp_path / "train.csv", index=False)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()

    def button(label):
        return next(b for b in at.button if b.label == label)

    def code():
        return "\n".join(c.value for c in at.code)

    def ok(where):
        assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
        assert not at.error, f"{where}: {[e.value for e in at.error]}"
        assert not any("Diagramm nicht möglich" in w.value for w in at.warning), (where, [w.value for w in at.warning])

    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "train.csv")).run()
    button("Daten laden").click().run()
    at.sidebar.radio(key="nav").set_value("2 · Aufbereitung 1 – Spalten").run()
    button("Beispiel-Konfiguration laden").click().run()
    button("Übernehmen").click().run()
    at.sidebar.radio(key="nav").set_value("3 · Analyse & Visualisierung").run()
    assert at.radio(key="w_chart").options[-4:] == NEW_CHARTS  # angehängt, die bisherigen bleiben, wo sie waren
    for chart in NEW_CHARTS:  # Standardauswahl
        at.radio(key="w_chart").set_value(chart).run()
        ok(chart)

    at.radio(key="w_chart").set_value("Jointplot").run()
    at.selectbox(key="w_jp_hue").set_value("loan_status").run()
    for kind in plots.JOINT_KINDS:
        at.radio(key="w_jp_kind").set_value(kind).run()
        ok(f"Jointplot {kind}")
        assert f"kind='{kind}'" in code() or kind == "scatter"
        assert ("hue='loan_status'" in code()) == (kind in ("scatter", "kde"))
        assert at.selectbox(key="w_jp_hue").disabled == (kind in ("hex", "reg"))

    at.radio(key="w_chart").set_value("Dichte 2D (KDE)").run()
    at.selectbox(key="w_kd_fcol").set_value("home_ownership").run()
    at.selectbox(key="w_kd_fval").set_value("RENT").run()
    at.checkbox(key="w_kd_fill").uncheck().run()
    ok("Dichte 2D mit Filter")
    assert "df[df['home_ownership'] == 'RENT']" in code() and "fill=True" not in code()

    at.radio(key="w_chart").set_value("Regression (lmplot)").run()
    at.selectbox(key="w_lm_hue").set_value("loan_status").run()
    at.selectbox(key="w_lm_col").set_value("term").run()
    ok("lmplot mit hue und col")
    assert "hue='loan_status', col='term'" in code()

    at.radio(key="w_chart").set_value("Histogramm je Gruppe").run()
    at.selectbox(key="w_hg_x").set_value("int_rate").run()
    at.selectbox(key="w_hg_group").set_value("term").run()
    at.slider(key="w_hg_bins").set_value(20).run()
    ok("Histogramm je Gruppe")
    assert "bins=20" in code() and "label='term=' + str(value)" in code()
