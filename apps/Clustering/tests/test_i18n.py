"""
Sprachversionen (Deutsch = Original, Englisch, Polnisch): Katalog vollständig, Platzhalter stimmen, die App läuft
in allen Sprachen durch, keine deutschen Reste in der englischen/polnischen Oberfläche, Excel-CSV je Sprache.

    pytest -q tests/test_i18n.py
"""
import ast
import json
import re
import string
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import codegen as cg  # noqa: E402
import i18n  # noqa: E402
import pipeline as pl  # noqa: E402
import test_codegen_parity as parity  # noqa: E402
from locales import CATALOG  # noqa: E402
from synthetic import anomaly_like, college_like, mall_like  # noqa: E402

APP = str(ROOT / "app.py")
MODULES = ["pipeline", "plots", "codegen", "app"]


@pytest.fixture()
def lang():
    """Sprache für einen Test setzen und danach wieder auf Deutsch stellen."""
    def use(code):
        i18n.set_lang(code)
        return code
    yield use
    i18n.set_lang("de")


def keys_in_code() -> dict:
    keys = {}
    for mod in MODULES:
        src = (ROOT / f"{mod}.py").read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("tx", "N_") \
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.setdefault(node.args[0].value, mod)
    return keys


def fields(text):
    return sorted((name, conv, spec) for _, name, spec, conv in string.Formatter().parse(text) if name is not None)


# --- Katalog ----------------------------------------------------------------------------------------------------
def test_every_text_has_english_and_polish():
    keys = keys_in_code()
    missing = [k for k in keys if k not in CATALOG]
    empty = [k for k, (en, pl_) in CATALOG.items() if not en.strip() or not pl_.strip()]
    stale = [k for k in CATALOG if k not in keys]
    assert not missing, f"ohne Übersetzung: {missing[:5]}"
    assert not empty, f"leere Übersetzung: {empty[:5]}"
    assert not stale, f"im Katalog, aber nicht im Code: {stale[:5]}"


def test_placeholders_are_the_same_in_all_languages():
    bad = [(k, t) for k, pair in CATALOG.items() for t in pair if fields(k) != fields(t)]
    assert not bad, bad[:5]


def test_german_is_the_original(lang):
    lang("de")
    assert i18n.tx("Daten laden") == "Daten laden"
    assert i18n.tx("{n} Zeilen; alle {v} benötigten Spalten vorhanden", n=3, v=4).startswith("3 Zeilen")
    lang("en")
    assert i18n.tx("Daten laden") == "Load data"
    lang("pl")
    assert i18n.tx("Daten laden") == "Wczytaj dane"
    assert i18n.tx("unbekannter Text bleibt deutsch") == "unbekannter Text bleibt deutsch"


@pytest.mark.parametrize("arg, env, expected", [
    (["--lang", "pl"], None, "pl"), (["--lang=en"], None, "en"), ([], "en", "en"), ([], "xx", "de"), ([], None, "de"),
])
def test_start_language(monkeypatch, arg, env, expected):
    monkeypatch.setattr(sys, "argv", ["app.py", *arg])
    if env is None:
        monkeypatch.delenv("CL_TEMPLATE_LANG", raising=False)
    else:
        monkeypatch.setenv("CL_TEMPLATE_LANG", env)
    assert i18n.start_lang() == expected
    assert i18n.start_lang("pl") == "pl"  # ?lang=pl in der Adresse hat Vorrang


# --- Excel-CSV je Sprache ----------------------------------------------------------------------------------------
def test_excel_csv_matches_the_language():
    df = pd.DataFrame({"name": ["=1+1", "Łódź"], "wert": [1.5, -2.25]})
    de, plk, en = (pl.to_csv_bytes(df, excel=x) for x in ("de", "pl", "en"))
    assert de == pl.to_csv_bytes(df, german=True)  # Deutsch unverändert
    assert de.startswith(b"\xef\xbb\xbf") and en.startswith(b"\xef\xbb\xbf")  # BOM: Excel erkennt UTF-8
    assert plk.decode("utf-8-sig").splitlines() == ["name;wert", "'=1+1;1,5", "Łódź;-2,25"]
    assert en.decode("utf-8-sig").splitlines() == ["name,wert", "'=1+1,1.5", "Łódź,-2.25"]
    assert pl.to_csv_bytes(df).decode("utf-8").splitlines()[1] == "=1+1,1.5"  # international: Daten unverändert


@pytest.mark.parametrize("excel, sep, dec", [("de", ";", ","), ("pl", ";", ","), ("en", ",", ".")])
def test_export_code_writes_the_same_format(tmp_path, lang, excel, sep, dec):
    lang(excel)
    code = cg.export_code("step2", None, excel)
    df = pd.DataFrame({"a": [1.5, 2.0], "b": ["x", "=1+1"]})
    ns = {"df": df, "pd": pd}
    exec(code.replace(repr(i18n.tx("daten_aufbereitet") + ".csv"), repr(str(tmp_path / "out.csv"))), ns)  # noqa: S102
    back = pd.read_csv(tmp_path / "out.csv", sep=sep, decimal=dec, encoding="utf-8-sig")
    assert back["a"].tolist() == [1.5, 2.0] and back["b"].tolist() == ["x", "'=1+1"]
    assert (tmp_path / "out.csv").read_bytes() == pl.to_csv_bytes(df, excel=excel)  # Code = App
    assert "#" in code and ("Excel" in code)


# --- erzeugter Code: andere Sprache, gleiche Rechnung --------------------------------------------------------------
@pytest.mark.parametrize("code_lang", ["en", "pl"])
def test_generated_code_is_identical_apart_from_comments(tmp_path, lang, code_lang):
    raw = mall_like(400)
    missing = {"Annual Income (k$)": {"strategy": pl.M_GROUP, "param": "Age"}, "Region": {"strategy": pl.M_MODE}}
    lang(code_lang)
    r = parity.run_both(tmp_path, raw, {"CustomerID": (pl.A_DROP, "", "")}, group="Segment", missing=missing,
                        cfg=pl.ClusterConfig(n_clusters=4, k_max=5))
    parity.assert_same(r)
    assert "Füllwerte aus allen Zeilen" not in r["code"] and "Kreuztabelle" not in r["code"]  # Kommentare übersetzt
    assert i18n.tx("Fehlende Werte: Füllwerte aus allen Zeilen (Clustering: kein Train/Test-Split)") in r["code"]


# --- Oberfläche in Englisch und Polnisch ---------------------------------------------------------------------------
GERMAN = re.compile(r"[äöüÄÖÜß]|\b(und|der|die|das|nicht|mit|für|wird|werden|Zeilen|Spalte|Spalten|bitte|oder|"
                    r"zuerst|Daten|auf|ist|sind|eine|einen|Schritt|Vorhersage|speichern|laden)\b")
ALLOWED = re.compile(r"Sonstige|fehlend|_jahr|_monat|Deutsch|Sprache · Language · Język")  # Datenwerte / Sprachwahl


def visible_texts(at):
    out = []
    for kind in ("title", "header", "subheader", "markdown", "caption", "info", "warning", "error", "success",
                 "toast"):
        out += [e.value for e in getattr(at, kind)]
    for kind in ("button", "checkbox", "text_input", "number_input", "slider", "selectbox", "multiselect",
                 "radio", "expander", "tabs", "metric"):
        for e in getattr(at, kind):
            out += [getattr(e, attr, "") or "" for attr in ("label", "help", "placeholder")]
            if kind == "radio":
                out += list(e.options)
    for df in at.dataframe:  # Spaltennamen: Tabellen direkt, Editoren über die Beschriftung (column_config)
        if df.proto.editing_mode == 0:
            out += [str(c) for c in df.value.columns]
        out += [c.get("label") or "" for c in json.loads(df.proto.columns or "{}").values()]
    return [str(t) for t in out if t]


def german_leftovers(at):
    return sorted({t[:120] for t in visible_texts(at) if GERMAN.search(ALLOWED.sub("", t))})


def button(at, label):
    for b in at.button:
        if b.label == label:
            return b
    raise KeyError(f"Button '{label}' fehlt: {[b.label for b in at.button]}")


@pytest.mark.parametrize("ui_lang", ["en", "pl"])
def test_app_runs_in_english_and_polish(tmp_path, monkeypatch, ui_lang):
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    college_like(500, seed=101).to_csv(tmp_path / "train.csv")
    college_like(120, seed=7).to_csv(tmp_path / "neu.csv")
    i18n.set_lang(ui_lang)
    t = i18n.tx  # Beschriftungen in der Testsprache
    leftovers = {}

    def ok(where):
        assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
        assert not at.error, f"{where}: {[e.value for e in at.error]}"
        found = german_leftovers(at)
        if found:
            leftovers[where] = found

    def go(i):
        nav = at.sidebar.radio(key="nav")
        nav.set_value(nav.options[i]).run()

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["w_lang"] = ui_lang
    at.run()
    ok("Start")
    assert at.header[0].value == t("1 · Daten laden & Bibliotheken")
    for i in range(10):  # alle Seiten ohne Daten
        go(i)
        ok(f"leere Seite {i + 1}")

    go(0)
    # set_value bekommt den internen Wert – AppTest formatiert ihn mit format_func der App (hier: Testsprache)
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "train.csv")).run()
    button(at, t("Daten laden")).click().run()
    ok("Laden")

    go(1)
    button(at, t("Beispiel-Konfiguration laden")).click().run()
    button(at, t("Übernehmen")).click().run()
    ok("Schritt 2")
    at.radio(key="w_new_kind").set_value(pl.N_FORMULA).run()
    at.text_input(key="w_new_name").input("quote").run()
    at.text_input(key="w_new_formula").input("Accept / Apps").run()
    button(at, t("Spalte hinzufügen")).click().run()
    ok("neue Spalte")
    at.text_input(key="w_new_name").input("kaputt").run()
    at.text_input(key="w_new_formula").input("gibts_nicht * 2").run()
    button(at, t("Spalte hinzufügen")).click().run()
    assert any(t("Unbekannte Spalte '{name}' – Namen mit Leer-/Sonderzeichen in `Backticks`.", name="gibts_nicht")
               in e.value for e in at.error)  # Fehlermeldung aus der Rechenlogik – übersetzt

    go(2)
    for chart in at.radio(key="w_chart").options:
        at.radio(key="w_chart").set_value(chart).run()
        ok(f"Diagramm {chart}")
        assert not any(t("Diagramm nicht möglich mit dieser Auswahl: {err}: {exc}", err="", exc="")[:15] in w.value
                       for w in at.warning)

    go(3)
    button(at, t("Daten fürs Modell vorbereiten")).click().run()
    ok("Schritt 4")
    assert at.session_state["prep"].recipe["group_col"] == "Private"

    go(4)
    at.radio(key="w_exp_what").set_value("ml").run()
    at.radio(key="w_exp_fmt").set_value("excel").run()
    assert t("CSV für deutsches Excel (Semikolon, Dezimalkomma)") in at.radio(key="w_exp_fmt").options
    button(at, t("CSV erstellen")).click().run()
    ok("Schritt 5")
    head = at.session_state["_export"][1][:2000].decode("utf-8-sig").splitlines()[0]
    assert head.count(";" if ui_lang == "pl" else ",") > 5  # Excel: pl Semikolon · en Komma

    go(5)
    button(at, t("Training starten")).click().run()
    ok("Training")
    go(6)
    ok("Schritt 7")
    go(7)
    at.radio(key="w_pca_color").set_value("group").run()
    button(at, t("Zufällige Zeile ziehen")).click().run()
    ok("Schritt 8")
    go(4)
    names = at.session_state["ed_names__base"].copy()
    names["Name"] = ["klein & privat", ""]
    at.session_state["ed_names__draft"] = names  # wie eine Eingabe im Editor von Schritt 8 (dann Seitenwechsel)
    at.radio(key="w_exp_what").set_value("clusters").run()
    button(at, t("CSV erstellen")).click().run()
    ok("Export Cluster")
    head = at.session_state["_export"][1][:3000].decode("utf-8-sig").splitlines()[0]
    assert t("Cluster-Name") in head and t("Abstand") in head

    go(8)
    at.text_input(key="w_save_name").input("modell_i18n").run()
    button(at, t("Speichern")).click().run()
    ok("Speichern")
    button(at, t("Laden")).click().run()
    ok("Laden Modell")

    go(9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "neu.csv")).run()
    button(at, t("Cluster zuordnen")).click().run()
    ok("Zuordnung")
    assert any(h.value == t("Fazit") for h in at.subheader)
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    ok("Formular")

    assert not leftovers, f"deutsche Texte in der Oberfläche ({ui_lang}): {leftovers}"
    i18n.set_lang("de")


@pytest.mark.parametrize("ui_lang, model", [("en", "iforest"), ("pl", "lof"), ("en", "dbscan"), ("pl", "dbscan")])
def test_anomaly_pages_in_english_and_polish(tmp_path, monkeypatch, ui_lang, model):
    """Isolation Forest, LOF und DBSCAN durch die Schritte 4–10 – ohne deutsche Reste in der Oberfläche."""
    monkeypatch.setenv("CL_TEMPLATE_MODELS", str(tmp_path / "models"))
    anomaly_like(800, seed=3).to_csv(tmp_path / "train.csv", index=False)
    anomaly_like(120, share=0.05, seed=9).to_csv(tmp_path / "neu.csv", index=False)
    i18n.set_lang(ui_lang)
    t = i18n.tx
    leftovers = {}

    def ok(where):
        assert not at.exception, f"{where}: {[e.message for e in at.exception]}"
        assert not at.error, f"{where}: {[e.value for e in at.error]}"
        found = german_leftovers(at)
        if found:
            leftovers[where] = found

    def go(i):
        nav = at.sidebar.radio(key="nav")
        nav.set_value(nav.options[i]).run()

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["w_lang"] = ui_lang
    at.run()
    at.radio(key="w_data_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_data_path").input(str(tmp_path / "train.csv")).run()
    button(at, t("Daten laden")).click().run()
    go(1)
    button(at, t("Übernehmen")).click().run()
    ok("Schritt 2")
    go(3)
    at.selectbox(key="w_group").set_value("Class").run()
    button(at, t("Daten fürs Modell vorbereiten")).click().run()
    ok("Schritt 4")

    go(5)
    at.radio(key="w_model").set_value(model).run()
    ok(f"Schritt 6 – {model}")
    if model == "lof":
        at.number_input(key="w_n_neighbors").set_value(60).run()
    button(at, t("Training starten")).click().run()
    ok("Training")
    go(6)
    ok("Schritt 7")
    offers = [b for b in at.button if str(b.key or "").startswith(("apply_c_", "apply_eps_"))]
    if offers:
        offers[0].click().run()
        ok("Vorschlag übernommen")
    go(7)
    ok("Schritt 8")
    assert any(s.value == t("Die auffälligsten Zeilen") for s in at.subheader)
    button(at, t("Zufällige Zeile ziehen")).click().run()
    ok("Einzelfall")

    go(4)
    at.radio(key="w_exp_what").set_value("anomalies").run()
    button(at, t("CSV erstellen")).click().run()
    ok("Export Anomalien")
    head = at.session_state["_export"][1][:3000].decode("utf-8-sig").splitlines()[0]
    assert head.endswith(t("Anomalie") + "," + t("Anomalie-Score"))

    go(8)
    at.text_input(key="w_save_name").input(f"anomalie_{model}").run()
    button(at, t("Speichern")).click().run()
    ok("Speichern")
    button(at, t("Laden")).click().run()
    ok("Laden Modell")

    go(9)
    at.radio(key="w_pred_model").set_value("bundle").run()
    at.radio(key="w_pred_src").set_value("Pfad auf diesem Rechner").run()
    at.text_input(key="w_pred_path").input(str(tmp_path / "neu.csv")).run()
    button(at, t("Neue Daten bewerten")).click().run()
    ok("Bewertung")
    assert any(h.value == t("Fazit") for h in at.subheader)
    assert {"Anomalie", "Anomalie-Score"} <= set(at.session_state["pred_result"]["res"].columns)  # intern deutsch …
    assert any(t("Anomalie-Score") in d.value.columns for d in at.dataframe)  # … angezeigt in der Sprache
    at.radio(key="w_pred_mode").set_value("Einzelner Datensatz (Formular)").run()
    button(at, t("Neue Daten bewerten")).click().run()
    ok("Formular")

    assert not leftovers, f"deutsche Texte in der Oberfläche ({ui_lang}, {model}): {leftovers}"
    i18n.set_lang("de")
