"""
Audit 3 (Version 8): je ein Test für die Fehler, die nach den Sprachversionen und den neuen Diagrammen gefunden
wurden – siehe AUDIT.md, Abschnitt „Audyt 3“.

    pytest -q tests/test_audit3_regressions.py
"""
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import i18n  # noqa: E402
import pipeline as pl  # noqa: E402
import plots  # noqa: E402

APP = str(ROOT / "app.py")
PAYLOAD = "x\nraise SystemExit('EINGESCHLEUST')  #"  # Zeilenumbruch in einem Namen oder Wert aus der CSV


@pytest.fixture()
def lang():
    def use(code):
        i18n.set_lang(code)
        return code
    yield use
    i18n.set_lang("de")


def injected(code: str) -> list[str]:
    """Code-Zeilen, die aus einem Kommentar „ausgebrochen“ sind (der Payload wird zu einem raise)."""
    tree = ast.parse(code)  # SyntaxError wäre ebenfalls ein Fehler
    return [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Raise)]


# --- A: Zeilenumbrüche aus den Daten dürfen im angezeigten Code keine neue Zeile beginnen -------------------------
@pytest.mark.parametrize("code_lang", ["de", "en", "pl"])
def test_names_and_values_cannot_inject_code_lines(lang, code_lang):
    lang(code_lang)
    raw = pd.DataFrame({PAYLOAD: [1.0, 2.0, 3.0, 4.0], "jahr_zahl": [2001.0, 2002.0, 2003.0, 2004.0],
                        "datum": ["Jan-2001", "Feb-2002", "Mar-2003", "Apr-2004"], "y": [0, 1, 0, 1]})
    prof = pl.column_profile(raw)
    for col, act, new in [(PAYLOAD, pl.A_NUM, ""), ("jahr_zahl", pl.A_YEAR, ""),
                          ("datum", pl.A_YEAR_MONTH, PAYLOAD)]:
        i = prof.index[prof["Spalte"] == col][0]
        prof.loc[i, ["Aktion", "Parameter", "Neuer Name"]] = [act, "%b-%Y" if act == pl.A_YEAR_MONTH else "", new]
    viz, fitted, _, _ = pl.apply_column_actions(raw, pl.actions_from_table(prof))
    new_columns = [{"name": "kopie", "kind": pl.N_COPY, "source": "y", "action": pl.A_KEEP},
                   {"name": "formel", "kind": pl.N_FORMULA, "definition": "y * 2"}]
    viz, new_fitted, _, _ = pl.apply_new_columns(viz, new_columns)
    new_fitted[0]["name"] = PAYLOAD  # Name und Definition stehen im Kommentar über der neuen Spalte
    new_fitted[1]["definition"] = PAYLOAD  # (Zuordnungen mit Zeilenumbruch lehnt schon die Prüfung ab)
    codes = {
        "Schritt 2": cg.step2_code(fitted, {PAYLOAD, "jahr_zahl"}, new_fitted),
        "unbekannte Aktion": "\n".join(cg._action_lines(PAYLOAD, PAYLOAD, "gibt_es_nicht", "", {}, "num", {})),
        "Speichern": cg.save_code("models/" + PAYLOAD),
        "Dichte 2D (Filterwert)": cg.chart_code("kde2d", x="a", y="b", filter_col="g", filter_value=PAYLOAD),
        "Dichte 2D (Filterspalte)": cg.chart_code("kde2d", x="a", y="b", filter_col=PAYLOAD, filter_value="v"),
    }
    for kind, p in [("joint", dict(x=PAYLOAD, y="b", hue=PAYLOAD)), ("lm", dict(x=PAYLOAD, y="b", col=PAYLOAD)),
                    ("hist_groups", dict(x=PAYLOAD, group=PAYLOAD)), ("rate", dict(cat=PAYLOAD, target="t",
                                                                                     positive=PAYLOAD))]:
        codes[kind] = cg.chart_code(kind, **p)
    for where, code in codes.items():
        assert not injected(code), (where, code)


def test_one_line_comments_do_not_change_normal_code(lang):
    lang("de")
    assert cg._cm("ganz normal") == "ganz normal"
    assert cg._cm("a\r\nb c") == "a b c"
    assert cg.save_code("models/m").endswith("# 'models/m/preprocessing.json' (Füllwerte, Dummy-Spalten, Scaler, "
                                             "Label-Kodierung)")


# --- C1/C2: Diagramm-Titel ------------------------------------------------------------------------------------------
@pytest.fixture()
def money():
    rng = np.random.default_rng(3)
    return pd.DataFrame({"Preis ($)": rng.normal(100, 10, 400), "Kosten ($)": rng.normal(50, 5, 400),
                         "g": rng.choice(["a", "b"], 400), "c": rng.choice(["k1", "k2", "k3", "k4"], 400)})


@pytest.mark.parametrize("draw", [
    lambda d: plots.fig_joint(d, "Preis ($)", "Kosten ($)"),
    lambda d: plots.fig_joint(d, "Preis ($)", "Kosten ($)", "kde", "g"),
    lambda d: plots.fig_lm(d, "Preis ($)", "Kosten ($)", "g"),
    lambda d: plots.fig_lm(d, "Preis ($)", "Kosten ($)", "g", "c"),
])
def test_figure_title_stays_inside_without_tight_bbox(money, draw):
    fig = draw(money)
    fig.canvas.draw()  # wie fig.savefig('bild.png') im Notebook – ohne bbox_inches='tight'
    assert fig._suptitle.get_window_extent().y1 <= fig.bbox.height
    plt.close(fig)


def test_dollar_signs_in_names_are_shown_literally(money):
    fig = plots.fig_scatter(money, "Preis ($)", "Kosten ($)")
    titles = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text() == "Kosten ($) gegen Preis ($)"]
    assert titles and all(t.get_parse_math() is False for t in titles)  # sonst: ') gegen Preis (' als Formel, $ weg
    plt.close(fig)


# --- B, C3, C4, C5: Oberfläche ------------------------------------------------------------------------------------
def test_language_switch_tables_export_and_step10(tmp_path, monkeypatch):
    monkeypatch.setenv("ML_TEMPLATE_MODELS", str(tmp_path / "models"))
    rng = np.random.default_rng(8)
    n = 500
    x = rng.normal(size=n)
    data = pd.DataFrame({"x": x.round(3), "z": rng.normal(size=n).round(3), "Typ": rng.choice(["A", "B"], n),
                         "Note": rng.choice(["gut", "schlecht"], n), "Prediction": rng.normal(size=n).round(3),
                         "Vorhersage": "alt"})
    data["Klasse"] = np.where(x + rng.normal(0, 0.5, n) > 0, "ja", "nein")
    data.to_csv(tmp_path / "d.csv", index=False)
    i18n.set_lang("de")
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()

    def button(label):
        return next(b for b in at.button if b.label == label)

    def go(page):
        at.sidebar.radio(key="nav").set_value(page).run()

    def ok(where):
        assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
        assert not at.error, f"{where}: {[e.value for e in at.error]}"

    def switch(code):
        next(e for e in at.sidebar if getattr(e, "key", None) == "w_lang").set_value(code).run()
        i18n.set_lang(code)  # AppTest formatiert Auswahllisten mit der Sprache des Test-Threads

    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "d.csv")).run()
    button("Daten laden").click().run()
    go("2 · Aufbereitung 1 – Spalten")
    button("Übernehmen").click().run()
    go("5 · Daten speichern")
    button("CSV erstellen").click().run()
    assert at.session_state["_export"][2] == "d_daten_aufbereitet.csv"
    assert "lang" not in at.query_params  # Deutsch ohne ?lang bleibt ohne

    # C5 + C4: Sprache wechseln → Adresse merkt sie sich; die alte CSV (deutscher Dateiname) wird neu erstellt
    switch("en")
    ok("Wechsel auf Englisch")
    assert at.query_params["lang"] == "en"
    old = [b for b in at.get("download_button") if b.label == i18n.tx("CSV herunterladen")]
    assert not old, "CSV aus der alten Sprache (deutscher Dateiname) noch angeboten"
    button(i18n.tx("CSV erstellen")).click().run()
    assert at.session_state["_export"][2] == "d_prepared_data.csv"

    # C3: Index-Name = Spaltenname aus den Daten ('Typ') bleibt unübersetzt
    go("2 · Aufbereitung 1 – Spalten")
    at.selectbox(key="w_inspect_col").set_value("Typ").run()
    names = [d.value.index.name for d in at.dataframe if list(d.value.columns[:1]) == ["Count"]]
    assert names == ["Typ"], names

    # B: Schritt 10 mit Spalten, die wie die Ergebnisspalten heißen (Englisch: Prediction/Note, intern: Vorhersage)
    go("4 · Aufbereitung 2 – ML-Daten")
    button(i18n.tx("Daten für ML vorbereiten")).click().run()
    go("6 · Modell & Training")
    at.number_input(key="w_rf_trees").set_value(10).run()
    button(i18n.tx("Training starten")).click().run()
    go("9 · Modell speichern & laden")
    at.text_input(key="w_save_name").input("m").run()
    button(i18n.tx("Speichern")).click().run()
    button(i18n.tx("Laden")).click().run()
    go("10 · Vorhersage neue Daten")
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "d.csv")).run()
    button(i18n.tx("Vorhersage berechnen")).click().run()
    ok("Schritt 10 auf Englisch")
    result = next(d.value for d in at.dataframe if "Prediction" in d.value.columns
                  and any(str(c).startswith("P(") for c in d.value.columns))
    assert result.columns.is_unique  # vorher: „Duplicate column names found“ → Absturz der Seite
    assert set(result.columns[:3]) == {"Prediction", "Note"} | {c for c in result.columns if str(c).startswith("P(")}
    assert any("Note, Prediction, Vorhersage" in i.value for i in at.info)

    switch("de")
    ok("Schritt 10 auf Deutsch")
    result = next(d.value for d in at.dataframe if "Vorhersage" in d.value.columns
                  and any(str(c).startswith("P(") for c in d.value.columns))
    assert result.columns.is_unique and {"Note", "Prediction"} <= set(result.columns)  # eigene Spalten bleiben
    assert any(i.value.endswith(": Vorhersage") for i in at.info)
    i18n.set_lang("de")
