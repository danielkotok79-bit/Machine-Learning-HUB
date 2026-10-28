"""
app.py – DATA-Analysis-Clustering: Cluster-Analyse mit K-Means und Anomalieerkennung mit Isolation Forest, LOF und
DBSCAN (scikit-learn) für CSV-Daten
===========================================================================================================

Start:   streamlit run app.py

Schritte (Navigation links)
    1  Daten laden & Bibliotheken           6  Modell & Training (K-Means, Isolation Forest, LOF, DBSCAN)
    2  Aufbereitung 1 – Spalten, neue Sp.   7  Wahl von K (Elbow & Silhouette) bzw. Schwelle / eps
    3  Analyse & Visualisierung             8  Auswertung (Cluster-Profile bzw. Anomalien, PCA, Vergleich)
    4  Aufbereitung 2 – Clustering-Daten    9  Modell speichern & laden
    5  Daten speichern (CSV)               10  Neue Daten zuordnen bzw. bewerten + Fazit
    Stichprobe für langsame Rechner: Option in der Seitenleiste

Unsupervised Learning: Die Modelle bekommen keine Antwort (kein Label). Eine bekannte Einteilung (im Kurs: die Spalte
Private der Hochschulen) geht nicht ins Modell – sie dient nur dem Vergleich in Schritt 8.

Die Rechenlogik steckt in pipeline.py, die Diagramme in plots.py, der angezeigte
Python-Code in codegen.py. Diese Datei enthält nur die Oberfläche.
"""
import matplotlib

matplotlib.use("Agg")  # Streamlit rechnet in einem Thread → kein GUI-Backend

import functools
import io
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DATA-Analysis-Clustering", page_icon=":material/bubble_chart:", layout="wide")

import i18n  # noqa: E402
from i18n import N_, tx  # noqa: E402

# Sprache vor allen Texten festlegen: Auswahl in der Seitenleiste, Start über ?lang=…, --lang oder CL_TEMPLATE_LANG
os.environ.setdefault("CL_TEMPLATE_LANG", "en")  # Startsprache Englisch
i18n.set_lang(st.session_state.get("w_lang") or i18n.start_lang(st.query_params.get("lang")))
# Gewählte Sprache in die Adresse übernehmen, damit Neuladen der Seite sie behält (Deutsch ohne ?lang bleibt ohne)
if st.query_params.get("lang") != i18n.get_lang() and (i18n.get_lang() != "de" or "lang" in st.query_params):
    st.query_params["lang"] = i18n.get_lang()

with st.spinner(tx("Lade Bibliotheken (pandas, scikit-learn) …")):
    import matplotlib.pyplot as plt

    import codegen as cg
    import pipeline as pl  # 1) Import der Bibliotheken – passiert hier automatisch
    import plots

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("CL_TEMPLATE_MODELS", APP_DIR / "models"))  # Speicherort für Schritt 9
CONFIG_DIR = APP_DIR / "configs"
NONE = N_("(keine)")  # interne Kennung – angezeigt in der gewählten Sprache
ss = st.session_state

# Widget-Werte über Seitenwechsel behalten (Streamlit löscht sonst Werte nicht angezeigter Widgets). Auch die
# Seitenauswahl ("nav"): so zeigt die Oberfläche nach einem Sprachwechsel die neue Beschriftung, und der nächste
# Klick springt nicht auf Schritt 1 zurück.
for _k in list(ss.keys()):
    if isinstance(_k, str) and (_k.startswith("w_") or _k == "nav"):
        ss[_k] = ss[_k]


# ===========================================================================
# Hilfsfunktionen für Widgets, Editoren und Zustand
# ===========================================================================
def init(key, value):
    if key not in ss:
        ss[key] = value


def _sanitize(key, options, default):
    if key in ss and ss[key] not in options:
        del ss[key]
    if key not in ss:
        ss[key] = default if default in options else options[0]


def sb(label, options, key, default=None, **kw):
    """selectbox mit gespeichertem Wert (ungültige Werte werden zurückgesetzt)."""
    options = list(options)
    if NONE in options and "format_func" not in kw:
        kw["format_func"] = lambda v: tx(NONE) if v == NONE else str(v)
    if not options:
        st.caption(tx("{label}: keine passenden Spalten", label=label))
        return None
    _sanitize(key, options, default)
    return st.selectbox(label, options, key=key, **kw)


def rd(label, options, key, default=None, **kw):
    options = list(options)
    _sanitize(key, options, default)
    return st.radio(label, options, key=key, **kw)


def ms(label, options, key, default=None, **kw):
    kw.setdefault("placeholder", tx("Auswählen …"))
    options = list(options)
    current = ss.get(key)
    if not isinstance(current, (list, tuple)):  # fehlt oder unbrauchbar (z. B. aus einer Konfigurationsdatei)
        current = default or []
    ss[key] = list(dict.fromkeys(v for v in current if v in options))  # ohne Duplikate
    return st.multiselect(label, options, key=key, **kw)


def editor(name, default_df=None, **kw):
    """st.data_editor, dessen Änderungen auch einen Seitenwechsel überleben."""
    base_k, ver_k, draft_k = f"{name}__base", f"{name}__ver", f"{name}__draft"
    if base_k not in ss:
        ss[base_k] = default_df.reset_index(drop=True)
        ss[ver_k] = 0
    wkey = f"{name}__w{ss[ver_k]}"
    if wkey not in ss and draft_k in ss:  # Seite gewechselt → letzten Entwurf übernehmen
        ss[base_k] = ss[draft_k]
        ss[ver_k] += 1
        wkey = f"{name}__w{ss[ver_k]}"
    edited = st.data_editor(ss[base_k], key=wkey, **kw)
    ss[draft_k] = edited
    return edited


def editor_reset(name, df):
    ss[f"{name}__base"] = df.reset_index(drop=True)
    ss.pop(f"{name}__draft", None)
    ss[f"{name}__ver"] = ss.get(f"{name}__ver", 0) + 1


def editor_value(name):
    return ss.get(f"{name}__draft", ss.get(f"{name}__base"))


RESULTS = {  # was in welchem Schritt entsteht – für das Verwerfen veralteter Ergebnisse
    "viz": ["df_viz", "actions_fitted", "new_cols_fitted", "lineage", "step2_log", "viz_token", "pairplot"],
    "prep": ["prep", "prep_token", "_export"],
    "model": ["model", "history", "train_info", "model_cfg", "model_token", "_names_for"],
    "eval": ["case_idx", "pred_result", "saved_folder"],
}
LEVELS = list(RESULTS)


def invalidate_from(level):
    for lv in LEVELS[LEVELS.index(level):]:
        for k in RESULTS[lv]:
            ss.pop(k, None)


def ready(key, msg):
    if key not in ss:
        st.info(msg)
        return False
    return True


def show(fig, **kw):
    if fig is not None:
        st.pyplot(fig, **kw)
        plt.close(fig)


@contextmanager
def chart_guard():
    """Fehler beim Zeichnen (ungewöhnliche Daten, z. B. nur Lücken) als Hinweis zeigen statt die Seite abzubrechen.
    st.rerun()/st.stop() sind keine Exception-Unterklassen und laufen normal durch."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        st.warning(tx("Diagramm nicht möglich mit dieser Auswahl: {err}: {exc}", err=type(exc).__name__, exc=exc),
                   icon=":material/warning:")


def code(txt, title=None, expanded=False):
    title = tx("Python-Code zu diesem Schritt") if title is None else title
    with st.expander(title, expanded=expanded, icon=":material/code:"):
        st.code(txt, language="python")


def opt(v):
    return None if v in (None, NONE) else v


def with_lang(fn):
    """Callbacks laufen vor dem Skript – dort zuerst die Sprache der Sitzung setzen (sonst Meldungen auf Deutsch)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        i18n.set_lang(ss.get("w_lang") or i18n.get_lang())
        return fn(*args, **kwargs)
    return wrapper


def trc(df, values=()):
    """Tabelle zur Anzeige: Spaltennamen – und die Werte der Spalten in `values` – in der gewählten Sprache.
    Der Index-Name bleibt: dort steht ein Spaltenname aus den Daten (z. B. bei value_counts), keine Überschrift."""
    if i18n.get_lang() == "de":
        return df
    out = df.copy()
    for col in values:
        if col in out.columns:
            out[col] = out[col].map(lambda v: tx(v) if isinstance(v, str) else v)
    return out.rename(columns=lambda c: tx(c) if isinstance(c, str) else c)


RESULT_COLUMNS = (N_("Cluster"), N_("Cluster-Name"), N_("Abstand"), N_("Anomalie"), N_("Anomalie-Score"),
                  N_("Hinweis"))  # = pl.RESULT_* (intern deutsch)
TABLE_HEADERS = (  # Spaltennamen der Anzeige-Tabellen – hier nur für den Übersetzungskatalog, übersetzt in trc()
    N_("Spalte"), N_("Typ"), N_("Eindeutig"), N_("Fehlend"), N_("Fehlend %"), N_("Beispiel"), N_("Neue Spalte"),
    N_("Art"), N_("Definition"), N_("Aktion"), N_("Parameter"), N_("Wirkung"), N_("Anzahl"), N_("Anteil"),
    N_("Rolle"), N_("Modell-Eingänge"), N_("Strategie"), N_("Kategorie-Spalte"), N_("Kategorien"),
    N_("weggelassen (drop_first)"), N_("neue 0/1-Spalten"), N_("Beispiele"), N_("Feature"), N_("Gruppe"),
    N_("Zeilen"), N_("Name"), N_("typische Vertreter"), N_("Randfälle (weit vom Zentrum)"), N_("Wert"),
    N_("Silhouette"), N_("Alle"), N_("Ladung"),
    # Anomalie-Modelle
    N_("Rang"), N_("Zeile"), N_("Bezeichnung"), N_("größte Abweichungen (z)"), N_("auffällig"), N_("Anteil auffällig"),
    N_("normal"), N_("Schwelle (Score)"), N_("Zeilen über der Schwelle"), N_("Rauschen"),
)


def export_columns() -> dict:
    """Namen der neuen Spalten (Cluster, Cluster-Name, Abstand, Anomalie, Anomalie-Score) im Export – in der gewählten
    Sprache."""
    return {c: tx(c) for c in (pl.RESULT_CLUSTER, pl.RESULT_NAME, pl.RESULT_DIST, pl.RESULT_ANOMALY, pl.RESULT_SCORE)}


def result_view(res):
    """Ergebnis aus Schritt 10 zur Anzeige und zum Download: Spaltennamen und Hinweise in der gewählten Sprache.
    Hat die neue CSV schon eine Spalte mit dem übersetzten Namen (z. B. 'Klaster' aus einem früheren Export),
    ersetzt das Ergebnis sie – wie auf Deutsch bei 'Cluster' (sonst doppelte Spaltennamen)."""
    if i18n.get_lang() == "de":
        return res
    out = res.copy()
    if pl.RESULT_NOTE in out.columns:
        out[pl.RESULT_NOTE] = out[pl.RESULT_NOTE].map(
            lambda h: "; ".join(tx(part) for part in h.split("; ")) if isinstance(h, str) and h else h)
    names = {c: tx(c) for c in RESULT_COLUMNS if c in out.columns}
    clash = [t for c, t in names.items() if t != c and t in out.columns]
    return out.drop(columns=clash).rename(columns=names)


def replaced_columns(columns, recipe):
    """Spalten der neuen Daten, die das Ergebnis aus Schritt 10 ersetzt (gleicher Name wie eine Ergebnisspalte –
    intern oder in der gewählten Sprache, z. B. aus einem früheren Export)."""
    key = pl.recipe_model(recipe)
    if pl.is_anomaly(key):
        made = [pl.RESULT_ANOMALY, pl.RESULT_SCORE, pl.RESULT_NOTE] + ([pl.RESULT_CLUSTER] if key == pl.MODEL_DBSCAN
                                                                        else [])
    else:
        made = [pl.RESULT_CLUSTER, pl.RESULT_DIST, pl.RESULT_NOTE]
    if any(recipe.get("cluster_names") or []):
        made.append(pl.RESULT_NAME)
    names = set(made) | {tx(c) for c in made}
    return [c for c in columns if c in names]


THIN = "\u202f"  # schmales Leerzeichen als Tausendertrennung (eindeutig neben dem Dezimalpunkt)


def md_text(text) -> str:
    """Text aus Dateien (Name eines hochgeladenen ZIP, Modellordner, Spalten- und Cluster-Namen eines geladenen
    Modells, Fehlermeldungen) in st.markdown wörtlich zeigen: Markdown-Zeichen escapen, Zeilenumbrüche entfernen –
    sonst könnte z. B. ![](http://…) ein fremdes Bild nachladen."""
    return pl.md_escape(text)


def fmt_int(n) -> str:
    return f"{int(n):,}".replace(",", THIN)


def fmt_num(v) -> str:
    """Zahl lesbar: große Werte ganzzahlig mit Tausendertrennung, kleine mit 4 signifikanten Stellen."""
    v = float(v)
    if not np.isfinite(v):
        return "–"
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", THIN)
    return f"{v:.4g}"


def cached(name, token, fn):
    """Kleines Ergebnis-Cache in der Sitzung (z. B. nunique), gebunden an einen Daten-Token."""
    hit = ss.get(name)
    if not hit or hit[0] != token:
        hit = (token, fn())
        ss[name] = hit
    return hit[1]


SEPS = [",", ";", "\t", "|"]
SEP_LABELS = {",": tx("Komma  ,"), ";": tx("Semikolon  ;"), "\t": tx("Tabulator"), "|": tx("Pipe  |")}


def csv_options():
    detected = ss.pop("_csv_detected", None)
    if detected:  # beim Laden automatisch erkannt → vor dem Zeichnen der Widgets übernehmen
        ss.w_sep, ss.w_dec, ss.w_enc = detected
    c1, c2, c3 = st.columns(3)
    with c1:
        sb(tx("Trennzeichen"), SEPS, "w_sep", format_func=SEP_LABELS.get)
    with c2:
        sb(tx("Dezimalzeichen"), [".", ","], "w_dec")
    with c3:
        sb(tx("Encoding"), ["utf-8", "latin-1"], "w_enc")


def csv_source(prefix):
    """Upload oder Pfad. Rückgabe: (Quelle, Name) – Upload-Bytes erst beim Laden lesen."""
    src = rd(tx("Quelle"), [N_("Datei hochladen"), N_("Pfad auf diesem Rechner")], f"w_{prefix}_src", horizontal=True,
             format_func=tx)
    if src == "Datei hochladen":
        up = st.file_uploader(tx("CSV-Datei"), type=["csv", "txt"], key=f"up_{prefix}")
        return (up, up.name) if up is not None else (None, None)
    init(f"w_{prefix}_path", "")
    path = pl.clean_path(st.text_input(tx("Pfad zur CSV-Datei"), key=f"w_{prefix}_path",
                                       placeholder=tx("z. B. C:\\Daten\\meine_daten.csv oder ../daten/meine_daten.csv")))
    return (path, Path(path).name) if path else (None, None)


def read_source(source):
    data = source.getvalue() if hasattr(source, "getvalue") else source
    return pl.read_csv(data, sep=ss.w_sep, decimal=ss.w_dec, encoding=ss.w_enc)


def guess_group(df, fallback=True):
    """Vorschlag für eine bekannte Einteilung – Farbe in Schritt 3, Vergleich in Schritt 8 (im Kurs: Private).
    fallback=False (Schritt 4): nur Spalten, die nach einer Einteilung heißen – sonst würde die letzte Spalte (z. B.
    TENURE mit 7 Werten) ungefragt zur Vergleichsgruppe und fiele still aus den Features."""
    names = {str(c).lower(): c for c in df.columns}
    for cand in ["private", "label", "target", "class", "klasse", "species", "segment", "group", "gruppe", "type",
                 "loan_status", "y"]:
        if cand in names:
            return names[cand]
    return df.columns[-1] if fallback else None


def with_descriptions(table):
    """Spalte 'Beschreibung' (feat_info) hinter 'Beispiel' einfügen bzw. aktualisieren."""
    table = table.copy()
    desc = [ss.get("feature_info", {}).get(c, "") for c in table["Spalte"]]
    if "Beschreibung" in table.columns:
        table["Beschreibung"] = desc
    else:
        table.insert(table.columns.get_loc("Beispiel") + 1, "Beschreibung", desc)
    return table


def describe_text(df):
    """describe() für Text-Spalten mit sauberen Ganzzahlen (count, unique, freq)."""
    out = df.describe().T
    for c in ("count", "unique", "freq"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def source_meta():
    s = ss.get("source", {})
    return {"source": s.get("name"), "rows_loaded": s.get("rows"),
            "sample_frac": ss.w_frac if ss.get("w_use_sample") else None}


# ===========================================================================
# Laden, Stichprobe, Konfiguration
# ===========================================================================
def sync_sample():
    """Stichprobe: df_raw = df_full.sample(frac) – bei Änderung werden Folgeschritte verworfen."""
    if "df_full" not in ss:
        return
    key = (bool(ss.get("w_use_sample")), round(float(ss.get("w_frac", 0.1)), 4))
    if not key[0]:
        key = (False, 1.0)
    if ss.get("sample_key") == key:
        return
    had_results = "df_viz" in ss
    ss.df_raw = pl.take_sample(ss.df_full, key[1] if key[0] else None)
    ss.sample_key = key
    invalidate_from("viz")
    if had_results:
        st.toast(tx("Stichprobe geändert – bitte Schritt 2 und 4 erneut übernehmen."), icon=":material/refresh:")


def load_data(source, name):
    try:
        with st.spinner(tx("Lese CSV …")):
            df, enc = read_source(source)
    except Exception as exc:  # noqa: BLE001 – Fehler in der Oberfläche anzeigen
        st.error(tx("Laden fehlgeschlagen: {exc}", exc=exc))
        return
    same_columns = "df_full" in ss and list(ss.df_full.columns) == list(df.columns)
    ss.df_full = df
    ss.data_token = uuid.uuid4().hex
    meta = df.attrs.get("csv", {})  # tatsächlich verwendete Optionen (evtl. automatisch erkannt)
    sep, dec = meta.get("sep", ss.w_sep), meta.get("decimal", ss.w_dec)
    if (sep, dec, enc) != (ss.w_sep, ss.w_dec, ss.w_enc):
        ss._csv_detected = (sep, dec, enc)
    detected = list(meta.get("detected", [])) + ([tx("Encoding")] if enc != ss.w_enc else [])
    ss.source = {"name": name, "path": source if isinstance(source, str) else None, "sep": sep, "decimal": dec,
                 "encoding": enc, "detected": detected, "strip_columns": meta.get("strip_columns"),
                 "rows": len(df), "loaded": datetime.now().strftime("%H:%M:%S")}
    ss.pop("sample_key", None)
    sync_sample()
    prof = pl.column_profile(ss.df_raw, ss.get("feature_info"))
    ss.profile = prof
    if same_columns and editor_value("ed_actions") is not None:  # gleiche Spalten → Einstellungen behalten
        old = editor_value("ed_actions").set_index("Spalte")
        for col in ("Aktion", "Parameter", "Neuer Name"):
            prof[col] = [old.at[c, col] if c in old.index else prof.at[i, col] for i, c in enumerate(prof["Spalte"])]
    else:
        for k in ["w_viz_cols", "w_group", "w_name_col", "w_features", "_features_for_group", "ed_missing__sig",
                  "new_cols"]:
            ss.pop(k, None)
    editor_reset("ed_actions", prof)
    st.rerun()


WIDGET_KEYS = [  # Einstellungen, die exportiert/importiert werden
    "w_viz_cols", "w_group", "w_name_col", "w_features", "w_scaler", "w_drop_first", "w_k", "w_init", "w_n_init",
    "w_max_iter", "w_algorithm", "w_seed", "w_curve", "w_k_max",
    "w_model", "w_contam_auto", "w_contam_pct", "w_n_estimators", "w_max_samples", "w_n_neighbors", "w_eps",
    "w_min_samples", "w_metric",
]


def export_config() -> bytes:
    def records(name, cols):
        df = editor_value(name)
        return None if df is None else df[[c for c in cols if c in df.columns]].to_dict("records")

    cfg = {
        "template_config": 1,
        "engine": "scikit-learn",
        "app": "clustering",
        "exported": datetime.now().isoformat(timespec="seconds"),
        "source": ss.get("source", {}).get("name"),
        "step2": records("ed_actions", ["Spalte", "Aktion", "Parameter", "Neuer Name"]),
        "missing": records("ed_missing", ["Spalte", "Strategie", "Parameter"]),
        "new_columns": ss.get("new_cols", []),
        "widgets": {k: ss[k] for k in WIDGET_KEYS if k in ss},
    }
    return json.dumps(pl._jsonable(cfg), indent=2, ensure_ascii=False).encode("utf-8")


def import_config(cfg):
    # Datei evtl. von Hand bearbeitet: Typen, Bereiche und Grenzen prüfen, Unbrauchbares verwerfen
    cfg, skipped = pl.sanitize_config(cfg)
    if cfg.get("step2") and "profile" in ss:
        prof = ss.profile.copy()
        amap = {a["Spalte"]: a for a in cfg["step2"]}
        for i, c in enumerate(prof["Spalte"]):
            if c in amap:
                for col in ("Aktion", "Parameter", "Neuer Name"):
                    prof.at[i, col] = amap[c][col]
        editor_reset("ed_actions", prof)
    if cfg.get("missing"):
        ss["ed_missing__draft"] = pd.DataFrame(cfg["missing"])
        ss.pop("ed_missing__sig", None)
    if "new_columns" in cfg:
        ss.new_cols = cfg["new_columns"]
    for k, v in cfg["widgets"].items():
        if k in WIDGET_KEYS:
            ss[k] = v
    if skipped:
        ss._cfg_skipped = skipped
    # Features aus der Datei behalten (sonst würde Schritt 4 sie beim ersten Anzeigen neu vorschlagen)
    ss._features_for_group = (opt(ss.get("w_group")), opt(ss.get("w_name_col")))


# Callbacks laufen vor dem nächsten Durchlauf – nur dort dürfen Widget-Werte gesetzt werden,
# auch wenn das Widget im aktuellen Durchlauf schon gezeichnet wurde.
@with_lang
def on_import_config():
    up = ss.get("up_cfg")
    if up is None:
        return
    if "profile" not in ss:
        ss._cfg_msg = ("error", tx("Bitte zuerst Daten laden (Schritt 1) – dann die Konfiguration importieren."))
        return
    try:  # ValueError umfasst auch JSONDecodeError und UnicodeDecodeError
        import_config(json.loads(up.getvalue().decode("utf-8-sig")))
    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
        ss._cfg_msg = ("error", tx("Ungültige Konfiguration: {exc}", exc=exc))
    else:
        msg = tx("Übernommen – jetzt Schritt 2 'Übernehmen' und Schritt 4 'Vorbereiten' klicken.")
        skipped = ss.pop("_cfg_skipped", None)
        if skipped:
            msg += tx(" Ungültige Werte ignoriert: {skipped}.", skipped=', '.join(skipped))
        ss._cfg_msg = ("success", msg)


@with_lang
def on_example_config(path):
    import_config(json.loads(Path(path).read_text(encoding="utf-8")))
    ss.pop("_cfg_skipped", None)
    ss._example_msg = True

# ===========================================================================
# Seite 1 – Daten laden & Bibliotheken
# ===========================================================================


def page_data():
    st.header(tx("1 · Daten laden & Bibliotheken"))
    with st.expander(tx("1) Bibliotheken – werden beim Start automatisch importiert")):
        c1, c2 = st.columns([3, 2])
        c1.code(cg.imports_code(), language="python")
        versions, cpus = cached("_versions", i18n.get_lang(), pl.library_versions)
        c2.dataframe(versions, hide_index=True)
        c2.caption(tx("scikit-learn rechnet auf der CPU – {n} Kerne; K-Means nutzt sie automatisch alle.", n=cpus))

    st.subheader(tx("CSV-Datei laden"))
    source, name = csv_source("data")
    csv_options()
    if st.button(tx("Daten laden"), type="primary", disabled=source is None):
        load_data(source, name)

    with st.expander(tx("Optional: Spalten-Beschreibungen laden")):
        st.caption(tx("CSV mit 2 Spalten: Spaltenname, Beschreibung. Die Texte erscheinen in Schritt 2 neben den Spalten."))
        info_src = rd(tx("Quelle"), [N_("Datei hochladen"), N_("Pfad")], "w_info_src", horizontal=True, format_func=tx)
        if info_src == "Datei hochladen":
            up = st.file_uploader(tx("Beschreibungs-CSV"), type=["csv", "txt"], key="up_info")
            info_source = up.getvalue() if up is not None else None
        else:
            init("w_info_path", "")
            p = pl.clean_path(st.text_input(tx("Pfad"), key="w_info_path",
                                            placeholder=tx("z. B. ../daten/beschreibungen.csv")))
            info_source = p or None
        if st.button(tx("Beschreibungen laden"), disabled=info_source is None):
            try:
                ss.feature_info = pl.read_feature_info(info_source)
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=exc))
            else:
                if "profile" in ss:
                    ss.profile = with_descriptions(ss.profile)
                    if editor_value("ed_actions") is not None:
                        editor_reset("ed_actions", with_descriptions(editor_value("ed_actions")))
                st.success(tx("{n} Beschreibungen geladen", n=len(ss.feature_info)))
        if "feature_info" in ss:
            f = sb(tx("Beschreibung nachschlagen"), sorted(ss.feature_info), "w_info_lookup")
            st.info(ss.feature_info.get(f, ""))

    if "df_raw" not in ss:
        st.info(tx("Noch keine Daten geladen – oben eine CSV-Datei hochladen oder den Pfad eintragen."))
        return
    df, src = ss.df_raw, ss.source
    st.subheader(tx("Überblick: {name}", name=src['name']))
    if src.get("detected"):
        st.info(tx("Automatisch erkannt: {v} → gelesen mit Trennzeichen „{v2}“, Dezimalzeichen „{decimal}“, "
                   "Encoding {encoding}.",
                   v=', '.join(src['detected']), v2=SEP_LABELS.get(src['sep'], src['sep']).split()[-1],
                   decimal=src['decimal'], encoding=src['encoding']))
    if df.shape[1] == 1:
        st.warning(tx("Nur 1 Spalte erkannt – stimmt das Trennzeichen?"))
    m = st.columns(4)
    m[0].metric(tx("Zeilen"), fmt_int(len(df)))
    m[1].metric(tx("Spalten"), df.shape[1])
    n_num = sum(pl.is_numeric(df[c]) for c in df.columns)
    m[2].metric(tx("Zahl / Text"), f"{n_num} / {df.shape[1] - n_num}")
    m[3].metric(tx("fehlende Zellen"), f"{df.isna().mean().mean():.2%}")
    if len(df) < src["rows"]:
        st.caption(tx("Stichprobe aktiv: {n} von {n2} Zeilen", n=fmt_int(len(df)), n2=fmt_int(src['rows'])))
    t1, t2, t3 = st.tabs(["df.head()", tx("Spalten (df.info)"), "df.describe()"])
    with t1:
        st.dataframe(df.head(20))
    with t2:
        st.dataframe(trc(ss.profile[["Spalte", "Typ", "Eindeutig", "Fehlend %", "Beispiel"]], values=["Typ"]),
                     hide_index=True)
    with t3:
        st.dataframe(cached("_describe", (ss.data_token, ss.sample_key), lambda: df.describe().T))
    code(cg.load_code(src.get("path") or src["name"], src["sep"], src["decimal"], src["encoding"],
                      ss.w_frac if ss.get("w_use_sample") else None, strip_columns=bool(src.get("strip_columns"))))

# ===========================================================================
# Seite 2 – Datenaufbereitung 1 (Spalten-Aktionen + neue Spalten)
# ===========================================================================


def shares_source(col, label):
    """Entsteht col aus derselben CSV-Spalte wie label (hier: die Vergleichsgruppe)? Als Feature würde K-Means die
    Antwort dann schon kennen – der Vergleich in Schritt 8 wäre geschönt."""
    lin = ss.get("lineage", {})
    return bool(label) and bool(set(pl.raw_sources(lin, col)) & set(pl.raw_sources(lin, label)))


def auto_select(old_cols, out):
    """Neu entstandene Spalten automatisch für Visualisierung und Features auswählen (nicht, wenn sie aus
    derselben Quelle wie die Vergleichsgruppe stammen)."""
    new_cols = [c for c in out.columns if c not in old_cols]
    if not old_cols or not new_cols:
        return
    if isinstance(ss.get("w_viz_cols"), list):
        ss.w_viz_cols = [c for c in out.columns if c in ss.w_viz_cols or c in new_cols]
    if isinstance(ss.get("w_features"), list):
        label = opt(ss.get("w_group"))
        ss.w_features = [c for c in out.columns if c in ss.w_features
                         or (c in new_cols and c != label and not shares_source(c, label))]


def apply_step2(table):
    """Spalten-Aktionen + neue Spalten auf die Rohdaten anwenden (wirft ValueError)."""
    out, fitted, log, lineage = pl.apply_column_actions(ss.df_raw, pl.actions_from_table(table), fit=True)
    out, new_fitted, log2, sources = pl.apply_new_columns(out, ss.get("new_cols", []), fit=True)
    old_cols = set(ss.df_viz.columns) if "df_viz" in ss else set()
    invalidate_from("viz")
    ss.df_viz, ss.actions_fitted, ss.new_cols_fitted = out, fitted, new_fitted
    ss.step2_log, ss.lineage = log + log2, pl.merge_lineage(lineage, sources)
    ss.viz_token = uuid.uuid4().hex
    auto_select(old_cols, out)


NEW_KEYS = {pl.N_MAP: ("w_new_src", "w_new_map"), pl.N_FORMULA: (None, "w_new_formula"),
            pl.N_COND: (None, "w_new_cond"), pl.N_COPY: ("w_new_copy_src", None)}


def new_spec_from_widgets():
    kind = ss.get("w_new_kind", pl.N_MAP)
    src_key, def_key = NEW_KEYS[kind]
    spec = {"name": pl._clean(ss.get("w_new_name")), "kind": kind,
            "source": pl._clean(ss.get(src_key)) if src_key else "",
            "definition": pl._clean(ss.get(def_key)) if def_key else "", "action": "", "param": ""}
    if kind == pl.N_COPY:
        spec.update(action=ss.get("w_new_copy_act", pl.A_KEEP), param=pl._clean(ss.get("w_new_copy_param")))
    return spec


# Callbacks: laufen vor dem nächsten Durchlauf – dort dürfen auch Eingabefelder geleert werden
@with_lang
def on_add_new_column():
    if "df_viz" not in ss:
        return
    spec = new_spec_from_widgets()
    try:
        out, fitted, log, sources = pl.apply_new_columns(ss.df_viz, [spec], fit=True)
    except ValueError as exc:
        ss._newcol_msg = ("error", str(exc))
        return
    had_model = "model" in ss
    old_cols = set(ss.df_viz.columns)
    invalidate_from("prep")
    ss.df_viz = out
    ss.new_cols = ss.get("new_cols", []) + [spec]
    ss.new_cols_fitted = ss.get("new_cols_fitted", []) + fitted
    ss.lineage = pl.merge_lineage(ss.get("lineage", {}), sources)
    ss.step2_log = ss.get("step2_log", []) + log
    ss.viz_token = uuid.uuid4().hex
    auto_select(old_cols, out)
    ss.w_new_name = ""
    ss._newcol_msg = ("success", tx("Hinzugefügt: {sources}", sources=', '.join(sources))
                      + (tx(" – das trainierte Modell wurde verworfen (Daten geändert).") if had_model else "."))


def new_column_dependents(remove: set) -> dict:
    """Welche verbleibenden neuen Spalten benutzen eine der zu entfernenden? → {abhängige Spalte: [benutzte]}"""
    specs = ss.get("new_cols", [])
    produced = set()
    for s in specs:
        if s["name"] in remove:
            produced.update(pl.result_names(s["name"], s.get("action") or pl.A_KEEP) if s["kind"] == pl.N_COPY
                            else [s["name"]])
    fitted = {s["name"]: s.get("fitted") or {} for s in ss.get("new_cols_fitted", [])}
    out = {}
    for s in specs:
        if s["name"] in remove:
            continue
        used = fitted.get(s["name"], {}).get("used")
        if used is None:
            if s["kind"] in (pl.N_FORMULA, pl.N_COND):
                try:
                    used = pl.translate_expression(s["definition"], list(ss.df_viz.columns))[1]
                except ValueError:
                    used = []
            else:
                used = [s.get("source")]
        hit = [u for u in used if u in produced]
        if hit:
            out[s["name"]] = hit
    return out


@with_lang
def on_remove_new_columns():
    names = set(ss.get("w_new_rm") or [])
    deps = new_column_dependents(names)
    if deps:  # sonst würde Schritt 2 mit einer unklaren Meldung scheitern
        ss._newcol_msg = ("error", tx("Nicht entfernt: ") + "; ".join(
            tx("'{d}' benutzt {v}", d=d, v=', '.join(repr(u) for u in used)) for d, used in deps.items())
            + tx(" – diese Spalte(n) zuerst oder gleichzeitig entfernen."))
        return
    before = list(ss.get("new_cols", []))
    ss.new_cols = [s for s in before if s["name"] not in names]
    ss.w_new_rm = []
    try:
        apply_step2(editor_value("ed_actions"))
    except ValueError as exc:
        ss.new_cols = before
        ss._newcol_msg = ("error", tx("Nicht entfernt: {exc}", exc=exc))
    else:
        ss._newcol_msg = ("success", tx("Entfernt: {v} – Schritt 2 neu übernommen.", v=', '.join(sorted(names))))


def new_columns_section():
    st.subheader(tx("Neue Spalten erstellen"))
    st.caption(tx("Neue Spalten entstehen aus dem Ergebnis oben (nach den Aktionen) und werden später automatisch "
               "auch für neue Daten berechnet (Schritt 10). Spätere neue Spalten dürfen frühere benutzen."))
    if "df_viz" not in ss:
        st.info(tx("Zuerst oben **Übernehmen** klicken – neue Spalten werden aus dem Ergebnis berechnet."))
        return
    v = ss.df_viz
    cols = list(v.columns)
    num = [c for c in cols if pl.is_numeric(v[c])]
    with st.container(border=True):
        kind = rd(tx("Art"), pl.NEW_KINDS, "w_new_kind", horizontal=True, format_func=tx)
        st.caption(tx(pl.NEW_HELP[kind][1]))
        init("w_new_name", "")
        if kind == pl.N_MAP:
            c1, c2, c3 = st.columns([2, 2, 4])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name", placeholder=tx("z. B. loan_repaid"))
            with c2:
                src = sb(tx("aus Spalte"), cols, "w_new_src")
            init("w_new_map", "")
            c3.text_input(tx("Zuordnung  alt=neu; …  (optional: sonst=…)"), key="w_new_map",
                          placeholder=tx("z. B. Fully Paid=1; Charged Off=0"))
            if src:
                vc = v[src].value_counts().head(12)
                st.caption(tx("Werte in {src}: ", src=src)
                           + " · ".join(f"{pl.format_value(k)} ({fmt_int(n)})" for k, n in vc.items())
                           + (" · …" if v[src].nunique() > 12 else ""))
        elif kind in (pl.N_FORMULA, pl.N_COND):
            key = "w_new_formula" if kind == pl.N_FORMULA else "w_new_cond"
            example = "loan_amnt / annual_inc" if kind == pl.N_FORMULA else "dti > 30"
            c1, c2 = st.columns([2, 6])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name",
                          placeholder=tx("z. B. kredit_zu_einkommen") if kind == pl.N_FORMULA else tx("z. B. hohe_dti"))
            init(key, "")
            c2.text_input(tx("Formel") if kind == pl.N_FORMULA else tx("Bedingung"), key=key,
                          placeholder=tx("z. B. {example}", example=example))
            shown = num if kind == pl.N_FORMULA else cols
            st.caption(tx("Spalten: ") + ", ".join(f"\\`{c}\\`" if not c.isidentifier() else c for c in shown[:40])
                       + (" …" if len(shown) > 40 else "") + tx(" · Namen mit Leer-/Sonderzeichen in \\`Backticks\\`"))
        else:
            c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name", placeholder=tx("z. B. earliest_cr_year"))
            with c2:
                sb(tx("aus Spalte"), cols, "w_new_copy_src")
            with c3:
                act = sb(tx("Aktion"), [a for a in pl.ACTIONS if a != pl.A_DROP], "w_new_copy_act", default=pl.A_YEAR,
                         format_func=tx)
            init("w_new_copy_param", "")
            c4.text_input(tx("Parameter"), key="w_new_copy_param", placeholder=tx(pl.ACTION_HELP[act][0]))
            if act == pl.A_YEAR_MONTH:
                st.caption(tx("Ergebnis: zwei Spalten <Name>_jahr und <Name>_monat."))
        st.button(tx("Spalte hinzufügen"), icon=":material/add:", on_click=on_add_new_column)
    msg = ss.pop("_newcol_msg", None)
    if msg:
        (st.error if msg[0] == "error" else st.success)(msg[1])
    if ss.get("new_cols"):
        def describe(s):
            if s["kind"] == pl.N_COPY:
                act = s.get("action") or pl.A_KEEP
                return tx("Kopie von {source}", source=s['source']) + ("" if act == pl.A_KEEP else f" + {tx(act)}") + \
                    (f" ({s['param']})" if s.get("param") else "")
            return (tx("aus {source}: ", source=s['source']) if s.get("source") else "") + s["definition"]

        st.dataframe(trc(pd.DataFrame([{"Neue Spalte": s["name"], "Art": s["kind"], "Definition": describe(s)}
                                       for s in ss.new_cols]), values=["Art"]), hide_index=True)
        c1, c2 = st.columns([4, 1], vertical_alignment="bottom")
        with c1:
            rm = ms(tx("Neue Spalten entfernen"), [s["name"] for s in ss.new_cols], "w_new_rm")
        c2.button(tx("Entfernen"), icon=":material/delete:", disabled=not rm, on_click=on_remove_new_columns)


EXAMPLES = [  # (Spalten, die der Datensatz haben muss, Beispiel-Konfiguration in configs/, Hinweis)
    ({"Private", "Apps", "Accept", "Enroll", "Outstate", "Grad.Rate"}, "college_beispiel.json",
     N_("College-Datensatz aus dem K-Means-Kurs erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und "
        "6 wie im Kurs ein – Name der Hochschule als Bezeichnung, Grad.Rate auf 100 begrenzt, 17 Features, "
        "Private nur zum Vergleich, K = 2. Anders als im Kurs mit StandardScaler; für den direkten Vergleich "
        "in Schritt 4 'keiner' wählen.")),
    ({"row", "symbol", "time", "ret_z", "range_z", "turnover_z"}, "bybit_zeitreihe_beispiel.json",
     N_("Bybit-Daten aus tools/bybit_features.py erkannt (Zeitreihe, eine Kerze je Zeile): Die Beispiel-Konfiguration "
        "stellt Schritt 2, 4 und 6 ein – row als Bezeichnung, alle Feature-Spalten (z-Werte, Volatilität, Spread, "
        "Funding, Open Interest …), StandardScaler und Isolation Forest mit 1 % auffälligen Zeilen.")),
    ({"symbol", "vola_ann_pct", "er", "max_dd_pct"}, "bybit_querschnitt_beispiel.json",
     N_("Bybit-Daten aus tools/bybit_features.py erkannt (Querschnitt, ein Token je Zeile): Die Beispiel-Konfiguration "
        "stellt Schritt 2, 4 und 6 ein – symbol als Bezeichnung, alle Kennzahlen als Features, StandardScaler und "
        "Isolation Forest mit 5 % auffälligen Tokens.")),
]


def page_prep1():
    st.header(tx("2 · Datenaufbereitung 1 – Spalten umwandeln (object → Zahl / Kategorie)"))
    if not ready("df_raw", tx("Bitte zuerst in Schritt 1 eine CSV laden.")):
        return
    df = ss.df_raw
    st.markdown(tx("Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann "
                "**Übernehmen**. Die Vorschläge sind automatisch – bitte prüfen. Für K-Means zählt nur, was Zeilen "
                "ähnlich oder verschieden macht: IDs und Freitext entfernen; eine Spalte mit Namen (z. B. der "
                "Hochschule) behalten und in Schritt 4 als Bezeichnung wählen. Unmögliche Werte und Ausreißer "
                "verschieben die Cluster-Zentren – z. B. Grad.Rate 118 % mit *Werte begrenzen* auf 100 setzen. "
                "Text-Spalten mit *behalten* werden in Schritt 4 zu Dummies."))
    example = next(((CONFIG_DIR / file, text) for cols, file, text in EXAMPLES
                    if cols <= set(df.columns) and (CONFIG_DIR / file).exists()), None)
    if example:
        c1, c2 = st.columns([4, 1])
        c1.info(tx(example[1]))
        c2.button(tx("Beispiel-Konfiguration laden"), on_click=on_example_config, args=(str(example[0]),))
        if ss.pop("_example_msg", None):
            st.success(tx("Beispiel-Konfiguration geladen (Schritt 2, 4 und 6) – jetzt unten **Übernehmen** klicken."))

    fixed = [c for c in ss.profile.columns if c not in ("Aktion", "Parameter", "Neuer Name")]
    col_cfg = {
        "Aktion": st.column_config.SelectboxColumn(tx("Aktion"), options=pl.ACTIONS, required=True, width="medium",
                                                   format_func=tx),
        "Spalte": st.column_config.TextColumn(tx("Spalte")),
        "Typ": st.column_config.SelectboxColumn(tx("Typ"), options=pl.DTYPE_LABELS, format_func=tx),
        "Eindeutig": st.column_config.NumberColumn(tx("Eindeutig")),
        "Beispiel": st.column_config.TextColumn(tx("Beispiel")),
        "Parameter": st.column_config.TextColumn(tx("Parameter"), help=tx("siehe 'Aktionen & Parameter erklärt'")),
        "Neuer Name": st.column_config.TextColumn(tx("Neuer Name"), help=tx("optional umbenennen, z. B. zip_code")),
        "Beschreibung": st.column_config.TextColumn(tx("Beschreibung"), width="large"),
        "Fehlend %": st.column_config.NumberColumn(tx("Fehlend %"), format="%.2f"),
    }
    edited = editor("ed_actions", ss.profile, column_config=col_cfg, disabled=fixed, hide_index=True,
                    height=min(35 * (len(ss.profile) + 1) + 3, 720))
    with st.expander(tx("Aktionen & Parameter erklärt")):
        st.dataframe(trc(pd.DataFrame([(a, p, d) for a, (p, d) in pl.ACTION_HELP.items()],
                                      columns=["Aktion", "Parameter", "Wirkung"]),
                         values=["Aktion", "Parameter", "Wirkung"]), hide_index=True)
    if "model" in ss:
        st.warning(tx("Übernehmen oder neue Spalten verwerfen das trainierte Modell – vorher in Schritt 9 speichern."))
    c1, c2, _ = st.columns([1, 1.4, 3])
    if c1.button(tx("Übernehmen"), type="primary"):
        try:
            with st.spinner(tx("Wende Aktionen an …")):
                apply_step2(edited)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.toast(tx("Schritt 2 übernommen"), icon=":material/check_circle:")
    if c2.button(tx("Vorschläge wiederherstellen")):
        editor_reset("ed_actions", pl.column_profile(df, ss.get("feature_info")))
        st.rerun()

    if "df_viz" in ss:
        v = ss.df_viz
        st.success(tx("Ergebnis: {n} Zeilen × {v} Spalten", n=fmt_int(v.shape[0]), v=v.shape[1]))
        if ss.step2_log:
            st.markdown("\n".join(f"- {line}" for line in ss.step2_log))
        st.dataframe(v.head(10))
        numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
        code(cg.step2_code(ss.actions_fitted, numeric_raw, ss.get("new_cols_fitted")))

    new_columns_section()

    st.subheader(tx("Spalte untersuchen"))
    base = ss.df_viz if "df_viz" in ss else df
    col = sb(tx("Spalte"), list(base.columns), "w_inspect_col")
    if col:
        s = base[col]
        if ss.get("feature_info", {}).get(col):
            st.info(ss.feature_info[col])
        c1, c2 = st.columns([2, 3])
        c1.dataframe(trc(pd.DataFrame({"Typ": [pl.dtype_label(s)], "Eindeutig": [s.nunique()],
                                       "Fehlend": [int(s.isna().sum())]}), values=["Typ"]), hide_index=True)
        if pl.is_numeric(s):
            c1.dataframe(s.describe().to_frame(), height=320)
        vc = s.value_counts(dropna=False).head(30).rename("Anzahl").to_frame()
        vc["Anteil"] = (vc["Anzahl"] / len(s)).map("{:.2%}".format)
        c2.dataframe(trc(vc), height=420)


# ===========================================================================
# Seite 3 – Analyse & Visualisierung
# ===========================================================================
CHARTS = [N_("Überblick"), N_("Countplot"), N_("Histogramm"), N_("Boxplot"), N_("Scatterplot"), N_("Korrelation (Heatmap)"),
          N_("Korrelation mit einer Spalte"), N_("Pairplot"), N_("Anteil je Kategorie"), N_("Gruppen-Statistik"),
          N_("Jointplot"), N_("Dichte 2D (KDE)"), N_("Regression (lmplot)"), N_("Histogramm je Gruppe")]


@with_lang
def on_viz_cols(which):
    full = ss.df_viz
    ss.w_viz_cols = (list(full.columns) if which == "alle"
                     else [c for c in full.columns if pl.is_numeric(full[c])] if which == "zahlen" else [])


def page_viz():
    st.header(tx("3 · Datenanalyse & Visualisierung"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    full = ss.df_viz
    with st.expander(tx("Spalten für die Visualisierung auswählen"), expanded=True, icon=":material/view_column:"):
        ms(tx("Spalten"), list(full.columns), "w_viz_cols", default=list(full.columns), label_visibility="collapsed")
        b1, b2, b3, _ = st.columns([1, 1.3, 1, 4])
        b1.button(tx("alle"), on_click=on_viz_cols, args=("alle",), width="stretch")
        b2.button(tx("nur Zahlenspalten"), on_click=on_viz_cols, args=("zahlen",), width="stretch")
        b3.button(tx("keine"), on_click=on_viz_cols, args=("keine",), width="stretch")
    cols = [c for c in ss.get("w_viz_cols", []) if c in full.columns] or list(full.columns)
    df = full[cols]
    nun = cached("_nunique", ss.viz_token, full.nunique)
    num = [c for c in cols if pl.is_numeric(df[c])]
    cat = [c for c in cols if c not in num]
    low = [c for c in cols if nun[c] <= 30]  # für Farbe (hue) geeignet
    binary = [c for c in cols if nun[c] == 2]
    few = [c for c in cols if 2 <= nun[c] <= 8]  # Gruppen für Jointplot, lmplot, Histogramm je Gruppe
    cont = [c for c in num if nun[c] > 10] or num  # Vorschlag für x/y: Spalten mit vielen verschiedenen Werten
    st.caption(tx("{n} von {v} Spalten ausgewählt · {n2} Zeilen", n=len(cols), v=full.shape[1], n2=fmt_int(len(df)))
               + ("" if ss.get("w_viz_cols") else tx(" (keine Auswahl → alle Spalten)")))
    chart = rd(tx("Diagramm"), CHARTS, "w_chart", horizontal=True, format_func=tx)

    with chart_guard():  # ungewöhnliche Daten dürfen die Seite nicht abbrechen
        if chart == "Überblick":
            if num:
                st.markdown(tx("**Zahlen** – `df.describe()`"))
                st.dataframe(cached("_describe_num", (ss.viz_token, tuple(cols)), lambda: df[num].describe().T))
            if cat:
                st.markdown(tx("**Kategorien / Text**"))
                st.dataframe(cached("_describe_cat", (ss.viz_token, tuple(cols)), lambda: describe_text(df[cat])))
            fig = plots.fig_missing(df)
            if fig is None:
                st.success(tx("Keine fehlenden Werte."))
            else:
                c1, _ = st.columns([2, 1])
                with c1:
                    show(fig)
            code(cg.chart_code("overview"))

        elif chart == "Countplot":
            c1, c2, c3 = st.columns(3)
            with c1:
                x = sb(tx("x (Kategorie)"), cat + [c for c in num if nun[c] <= 50], "w_cnt_x")
            with c2:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_cnt_hue")
            with c3:
                order = rd(tx("Reihenfolge"), [N_("Wert"), N_("Häufigkeit")], "w_cnt_order", horizontal=True,
                           format_func=tx)
            if x:
                show(plots.fig_count(df, x, opt(hue), order))
                code(cg.chart_code("count", x=x, hue=opt(hue), order_by=order))

        elif chart == "Histogramm":
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                x = sb(tx("Spalte (Zahl)"), num, "w_hist_x")
            with c2:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_hist_hue")
            with c3:
                init("w_hist_bins", 40)
                bins = st.slider("bins", 10, 120, key="w_hist_bins")
            with c4:
                init("w_hist_kde", False)
                init("w_hist_clip", False)
                kde = st.checkbox(tx("KDE-Linie"), key="w_hist_kde")
                clip = st.checkbox(tx("Ausreißer abschneiden (1–99 %)"), key="w_hist_clip")
            if x:
                show(plots.fig_hist(df, x, opt(hue), bins, kde, clip))
                code(cg.chart_code("hist", x=x, hue=opt(hue), bins=bins, kde=kde))

        elif chart == "Boxplot":
            c1, c2, c3 = st.columns(3)
            with c1:
                x = sb(tx("x (Kategorie)"), [c for c in cols if nun[c] <= 50], "w_box_x")
            with c2:
                y = sb(tx("y (Zahl)"), [c for c in num if c != x], "w_box_y")
            with c3:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_box_hue")
            if x and y:
                show(plots.fig_box(df, x, y, opt(hue)))
                code(cg.chart_code("box", x=x, y=y, hue=opt(hue)))

        elif chart == "Scatterplot":
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                x = sb("x", num, "w_sc_x")
            with c2:
                y = sb("y", num, "w_sc_y", default=num[1] if len(num) > 1 else None)
            with c3:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_sc_hue")
            with c4:
                init("w_sc_n", 5000)
                n = st.slider(tx("max. Punkte"), 500, 20000, step=500, key="w_sc_n")
            if x and y:
                show(plots.fig_scatter(df, x, y, opt(hue), n))
                code(cg.chart_code("scatter", x=x, y=y, hue=opt(hue)))

        elif chart == "Korrelation (Heatmap)":
            sel = ms(tx("Numerische Spalten"), num, "w_hm_cols", default=num[:25])
            c1, c2, c3 = st.columns(3)
            with c1:
                method = rd(tx("Methode"), ["pearson", "spearman"], "w_hm_method", horizontal=True)
            with c2:
                init("w_hm_annot", True)
                annot = st.checkbox(tx("Werte anzeigen (annot)"), key="w_hm_annot")
            with c3:
                init("w_hm_tri", False)
                tri = st.checkbox(tx("nur untere Hälfte"), key="w_hm_tri")
            if len(sel) >= 2:
                corr = plots.corr_matrix(df, sel, method)
                c1, _ = st.columns([4, 1] if len(sel) > 10 else [3, 2])
                with c1:
                    show(plots.fig_corr_heatmap(corr, annot, tri))
                with st.expander(tx("Tabelle (df.corr)")):
                    st.dataframe(corr.round(3))
                code(cg.chart_code("heatmap", method=method))
            else:
                st.info(tx("Mindestens 2 numerische Spalten wählen."))

        elif chart == "Korrelation mit einer Spalte":
            st.caption(tx("Wie stark hängt jede Zahlenspalte mit der gewählten Spalte zusammen? (+1 = gleichläufig, "
                       "−1 = gegenläufig, 0 = kein linearer Zusammenhang) – z. B. um eine Gruppenspalte für fehlende "
                       "Werte zu finden oder Features zu vergleichen."))
            targets = [c for c in cols if c in num or nun[c] == 2]
            c1, c2, c3 = st.columns(3)
            with c1:
                t = sb(tx("Spalte"), targets, "w_ct_t", default=guess_group(df) if guess_group(df) in targets else None)
            pos = None
            if t and t not in num:
                with c2:
                    vals = sorted(df[t].dropna().astype(str).unique())
                    pos = sb(tx("= 1 (positive Klasse)"), vals, "w_ct_pos", default=vals[-1])
            with c3:
                method = rd(tx("Methode"), ["pearson", "spearman"], "w_ct_method", horizontal=True)
            if t:
                corr = plots.corr_with_target(df, t, pos, method)
                if corr.empty:
                    st.info(tx("Keine numerischen Spalten zum Vergleichen."))
                else:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        show(plots.fig_corr_target(corr, t, pos))
                    c2.dataframe(corr.iloc[::-1].rename(tx("Korrelation")).round(3).to_frame(), height=500)
                    code(cg.chart_code("corr_target", target=t, positive=pos))

        elif chart == "Pairplot":
            sel = ms(tx("Numerische Spalten (2–6 empfohlen)"), num, "w_pp_cols", default=num[:4])
            c1, c2 = st.columns(2)
            with c1:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_pp_hue")
            with c2:
                init("w_pp_n", 1500)
                n = st.slider(tx("max. Zeilen (Stichprobe)"), 200, 10000, step=100, key="w_pp_n")
            if len(sel) > 8:
                st.warning(tx("Mehr als 8 Spalten → sehr langsam und unübersichtlich."))
            params = (ss.viz_token, tuple(sel), hue, n, i18n.get_lang())
            if st.button(tx("Pairplot erstellen"), type="primary", disabled=len(sel) < 2):
                with st.spinner(tx("Zeichne Pairplot …")):
                    fig = plots.fig_pairplot(df, sel, opt(hue), n)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                    plt.close(fig)
                    ss.pairplot = (params, buf.getvalue())
            if ss.get("pairplot") and ss.pairplot[0] == params:
                st.image(ss.pairplot[1], width=min(260 * len(sel), 1100))
            code(cg.chart_code("pairplot", cols=sel, hue=opt(hue), n=n))

        elif chart == "Anteil je Kategorie":
            st.caption(tx("Anteil einer Klasse je Kategorie – z. B. Ausfallquote je Beschäftigungsdauer."))
            c1, c2, c3 = st.columns(3)
            with c1:
                t = sb(tx("Zielspalte (2 Werte)"), binary, "w_rate_t",
                       default=guess_group(df) if guess_group(df) in binary else None)
            if t:
                with c2:
                    vals = sorted(df[t].dropna().astype(str).unique())
                    pos = sb(tx("Anteil von"), vals, "w_rate_pos", default=vals[0])
                with c3:
                    c = sb(tx("je Kategorie"), [c for c in cols if nun[c] <= 60 and c != t], "w_rate_cat")
                if c:
                    stats = plots.target_rate(df, c, t, pos)
                    overall = float(plots.binary_target(df[t], pos).mean())
                    show(plots.fig_target_rate(stats, c, t, pos, overall))
                    with st.expander(tx("Tabelle")):
                        st.dataframe(trc(stats.assign(Anteil=stats["Anteil"].map("{:.2%}".format))))
                    code(cg.chart_code("rate", cat=c, target=t, positive=pos))

        elif chart == "Gruppen-Statistik":
            c1, c2 = st.columns(2)
            with c1:
                by = sb(tx("gruppieren nach"), [c for c in cols if nun[c] <= 60], "w_gb_by")
            with c2:
                col = sb(tx("Statistik für"), num, "w_gb_col")
            if by and col and by != col:
                st.dataframe(df.groupby(by)[col].describe())
                code(cg.chart_code("groupby", by=by, col=col))

        elif chart == "Jointplot":
            st.caption(tx("Zwei Zahlenspalten gegeneinander, am Rand die Verteilung jeder Spalte (sns.jointplot)."))
            c1, c2, c3, c4, c5 = st.columns([2, 2, 3, 2, 2])
            with c1:
                x = sb("x", num, "w_jp_x", default=cont[0] if cont else None)
            with c2:
                y = sb("y", [c for c in num if c != x], "w_jp_y", default=next((c for c in cont if c != x), None))
            with c3:
                kind = rd(tx("Art (kind)"), plots.JOINT_KINDS, "w_jp_kind", horizontal=True,
                          help=tx("scatter = Punkte · kde = Dichte (Höhenlinien) · hex = Waben, gut bei sehr vielen "
                                  "Punkten · reg = Punkte mit Regressionsgerade"))
            with_hue = kind in ("scatter", "kde")
            with c4:
                hue = sb(tx("Farbe (hue)"), [NONE] + [c for c in few if c not in (x, y)], "w_jp_hue",
                         disabled=not with_hue, help=tx("Farbe nur bei scatter und kde."))
            with c5:
                init("w_jp_n", 3000)
                n = st.slider(tx("max. Punkte"), 500, 10000, step=500, key="w_jp_n")
            if x and y:
                hue = opt(hue) if with_hue else None
                c1, _ = st.columns([3, 2])
                with c1:
                    show(plots.fig_joint(df, x, y, kind, hue, n))
                code(cg.chart_code("joint", x=x, y=y, joint_kind=kind, hue=hue, n=n))

        elif chart == "Dichte 2D (KDE)":
            st.caption(tx("Wo liegen die meisten Punkte? Dichte zweier Zahlenspalten (sns.kdeplot) – je heller, desto "
                          "dichter. Mit Filter nur eine Gruppe, z. B. iris[iris['species'] == 'setosa']."))
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                x = sb("x", num, "w_kd_x", default=cont[0] if cont else None)
            with c2:
                y = sb("y", [c for c in num if c != x], "w_kd_y", default=next((c for c in cont if c != x), None))
            with c3:
                fcol = opt(sb(tx("Filter: Spalte"), [NONE] + [c for c in low if c not in (x, y)], "w_kd_fcol"))
            fval = None
            if fcol:
                with c4:
                    fval = sb(tx("Filter: Wert"), plots.category_values(df[fcol]), "w_kd_fval")
            with c5:
                init("w_kd_fill", True)
                fill = st.checkbox(tx("Flächen füllen (fill)"), key="w_kd_fill")
                init("w_kd_n", 3000)
                n = st.slider(tx("max. Punkte"), 500, 10000, step=500, key="w_kd_n")
            if x and y and (fval is not None or not fcol):
                c1, _ = st.columns([3, 2])
                with c1:
                    show(plots.fig_kde2d(df, x, y, fcol, fval, fill, n))
                code(cg.chart_code("kde2d", x=x, y=y, filter_col=fcol, filter_value=fval,
                                   filter_numeric=bool(fcol) and pl.is_numeric(df[fcol]), fill=fill, n=n))

        elif chart == "Regression (lmplot)":
            st.caption(tx("Punkte mit Regressionsgerade und 95-%-Band (sns.lmplot) – getrennt nach Farbe (hue) und "
                          "Teilgrafiken (col), z. B. int.rate gegen fico je credit.policy und not.fully.paid."))
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                x = sb("x", num, "w_lm_x", default=cont[0] if cont else None)
            with c2:
                y = sb("y", [c for c in num if c != x], "w_lm_y", default=next((c for c in cont if c != x), None))
            with c3:
                hue = opt(sb(tx("Farbe (hue)"), [NONE] + [c for c in few if c not in (x, y)], "w_lm_hue"))
            with c4:
                col = opt(sb(tx("Teilgrafiken (col)"), [NONE] + [c for c in few if c not in (x, y) and nun[c] <= 6],
                             "w_lm_col"))
            with c5:
                init("w_lm_n", 5000)
                n = st.slider(tx("max. Punkte"), 500, 20000, step=500, key="w_lm_n")
            if x and y:
                fig = plots.fig_lm(df, x, y, hue, col, n)
                if col:
                    show(fig)
                else:
                    c1, _ = st.columns([3, 2])
                    with c1:
                        show(fig)
                code(cg.chart_code("lm", x=x, y=y, hue=hue, col=col, wrap=bool(col) and nun[col] > 3, n=n))

        elif chart == "Histogramm je Gruppe":
            st.caption(tx("Histogramme einer Zahlenspalte für jede Gruppe übereinander – gleiche Klassen, halb "
                          "durchsichtig (z. B. fico je credit.policy)."))
            c1, c2, c3 = st.columns(3)
            with c1:
                x = sb(tx("Spalte (Zahl)"), num, "w_hg_x", default=cont[0] if cont else None)
            with c2:
                grp = sb(tx("Gruppe"), [c for c in few if c != x], "w_hg_group",
                         default=guess_group(df) if guess_group(df) in few else None)
            with c3:
                init("w_hg_bins", 30)
                bins = st.slider("bins", 10, 100, key="w_hg_bins")
            if x and grp:
                show(plots.fig_hist_groups(df, x, grp, bins))
                code(cg.chart_code("hist_groups", x=x, group=grp, bins=bins))

# ===========================================================================
# Seite 4 – Datenaufbereitung 2
# ===========================================================================


def missing_editor(df, miss_cols, features):
    prev = editor_value("ed_missing")
    prev_map = {r["Spalte"]: r for r in prev.to_dict("records")} if prev is not None else {}
    sig = (ss.get("viz_token"), tuple(miss_cols))  # neue Daten aus Schritt 2 → Zählungen neu berechnen
    if ss.get("ed_missing__sig") != sig:
        rows = []
        for c in miss_cols:
            if c in prev_map and pl._clean(prev_map[c].get("Strategie")):
                strat, param = prev_map[c]["Strategie"], pl._clean(prev_map[c].get("Parameter"))
            else:
                strat, param = pl.suggest_missing(df, c, features)
            rows.append({"Spalte": c, "Typ": N_("Zahl") if pl.is_numeric(df[c]) else N_("Kategorie"),
                         "Fehlend": int(df[c].isna().sum()), "Fehlend %": round(df[c].isna().mean() * 100, 3),
                         "Strategie": strat, "Parameter": param})
        editor_reset("ed_missing", pd.DataFrame(rows))
        ss["ed_missing__sig"] = sig
    col_cfg = {
        "Spalte": st.column_config.TextColumn(tx("Spalte")),
        "Typ": st.column_config.SelectboxColumn(tx("Typ"), options=["Zahl", "Kategorie"], format_func=tx),
        "Fehlend": st.column_config.NumberColumn(tx("Fehlend")),
        "Strategie": st.column_config.SelectboxColumn(tx("Strategie"), options=pl.MISSING_STRATEGIES, required=True,
                                                      format_func=tx,
                                                      width="medium"),
        "Parameter": st.column_config.TextColumn(tx("Parameter"),
                                                 help=tx("Konstante: Wert · Gruppen-Mittelwert: Gruppenspalte")),
        "Fehlend %": st.column_config.NumberColumn(tx("Fehlend %"), format="%.3f"),
    }
    return editor("ed_missing", None, column_config=col_cfg, disabled=["Spalte", "Typ", "Fehlend", "Fehlend %"],
                  hide_index=True)


def group_candidates(df, nun):
    """Spalten, die als bekannte Einteilung taugen: 2–30 verschiedene Werte (z. B. Private = Yes/No)."""
    return [c for c in df.columns if 2 <= int(nun[c]) <= pl.MAX_GROUPS]


def name_candidates(df, nun):
    """Spalten für die Bezeichnung der Zeilen: fast eindeutige Texte (Name der Hochschule) – vorgeschlagen – und
    eindeutige ganze Zahlen (IDs) – nur zur Auswahl. Rückgabe: (Texte, IDs)."""
    n = max(len(df), 1)
    text = [c for c in df.columns if not pl.is_numeric(df[c]) and int(nun[c]) >= 0.5 * n]
    ids = [c for c in df.columns if pl.is_numeric(df[c]) and not pd.api.types.is_bool_dtype(df[c])
           and int(nun[c]) == len(df) and pl._whole(df[c].dropna().to_numpy())]
    return text, ids


def page_prep2():
    st.header(tx("4 · Datenaufbereitung 2 – Clustering-Daten (Features, Lücken, Skalierung)"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    df = ss.df_viz
    cols = list(df.columns)
    lin = ss.get("lineage", {})
    nun = cached("_nunique", ss.viz_token, df.nunique)
    st.markdown(tx("**Unsupervised Learning:** Die Modelle bekommen keine Antwort – kein Label, kein y. K-Means sucht "
                   "Gruppen von Zeilen, die sich in den Features ähneln; Isolation Forest, LOF und DBSCAN suchen Zeilen, "
                   "die zu keiner Gruppe passen (Anomalien). Eine bekannte Einteilung (im Kurs die Spalte *Private*, bei "
                   "Betrugsdaten z. B. *Class*) geht nicht ins Modell; sie dient nur dem Vergleich in Schritt 8."))

    st.subheader(tx("Vergleichsgruppe & Bezeichnung (optional)"))
    groups = group_candidates(df, nun)
    guess = guess_group(df, fallback=False)  # nur Spalten, die nach einer Einteilung heißen (Private, species …)
    c1, c2 = st.columns(2)
    with c1:
        group = opt(sb(tx("Bekannte Einteilung – nur zum Vergleich"), [NONE] + groups, "w_group",
                       default=guess if guess in groups else NONE,
                       help=tx("Geht NICHT ins Clustering – erst danach wird verglichen: Finden die Cluster diese "
                               "Gruppen wieder? (im Kurs: Private = Yes/No). Spalten mit 2–30 verschiedenen Werten.")))
    texts, ids = name_candidates(df, nun)
    # IDs (CustomerID 1, 2, 3 …) sind als Feature schädlich, als Bezeichnung nützlich
    id_like = cached("_id_like", ss.viz_token, lambda: [c for c in cols if pl.looks_like_id(df[c], c)])
    name_opts = [c for c in texts + ids if c != group]
    with c2:
        name_col = opt(sb(tx("Bezeichnung der Zeilen – nur zur Anzeige"), [NONE] + name_opts, "w_name_col",
                          default=next((c for c in texts if c != group),
                                       next((c for c in id_like if c in name_opts), NONE)),
                          help=tx("Text, der eine Zeile benennt (z. B. Name der Hochschule): zeigt typische Vertreter "
                                  "je Cluster und steht im Export – nie ein Feature.")))
    if not name_opts:
        st.caption(tx("Tipp: Eine Spalte mit Namen (z. B. 'Unnamed: 0' – die erste Spalte von College_Data, im Kurs "
                      "index_col=0) in Schritt 2 behalten statt entfernen; dann lässt sie sich hier als Bezeichnung "
                      "wählen."))

    st.subheader(tx("Features (X) – daraus berechnet das Modell Ähnlichkeit und Auffälligkeit"))
    special = {c for c in (group, name_col) if c}
    candidates = [c for c in cols if c not in special]
    leak = [c for c in candidates if group and shares_source(c, group)]
    default = [c for c in candidates if c not in leak and c not in id_like
               and (pl.is_numeric(df[c]) or int(nun[c]) <= 30)]
    prev = ss.get("_features_for_group")
    if prev != (group, name_col):
        if isinstance(ss.get("w_features"), list):
            # eben noch Vergleichsgruppe/Bezeichnung → wieder Feature, wenn sie zum Vorschlag gehört (sonst bliebe
            # z. B. eine abgewählte Vergleichsgruppe still aus dem Clustering heraus)
            freed = [c for c in (prev or ()) if c and c not in special and c in default and c not in ss.w_features]
            ss.w_features = [c for c in ss.w_features if c not in special] + freed
        else:  # fehlt oder unbrauchbar → Vorschlag verwenden
            ss.pop("w_features", None)
        ss._features_for_group = (group, name_col)
    features = ms(tx("Features für das Modell"), candidates, "w_features", default=default)
    if set(leak) & set(features):
        st.warning(tx("{leak} stammt aus derselben CSV-Spalte wie die Vergleichsgruppe – K-Means kennt die Antwort dann "
                      "schon, der Vergleich in Schritt 8 wäre geschönt.",
                      leak=[c for c in leak if c in features]))
    id_feats = [c for c in features if c in id_like]
    if id_feats:
        st.warning(tx("{cols}: sieht aus wie eine ID (ganze Zahlen, jede Zeile ein anderer Wert) – als Feature würde "
                      "K-Means nach Zeilennummern gruppieren. Besser als Bezeichnung der Zeilen wählen und hier "
                      "abwählen.", cols=md_text(", ".join(id_feats))))
    init("w_drop_first", False)
    rows = []
    for c in cols:
        role = (N_("Feature (X)") if c in features else N_("Vergleichsgruppe") if c == group
                else N_("Bezeichnung") if c == name_col else "–")
        is_num = pl.is_numeric(df[c])
        inputs = 0
        if role == "Feature (X)":
            inputs = 1 if is_num else max(int(nun[c]) - (1 if ss.w_drop_first else 0), 0)
        kind = N_("Zahl") if is_num else (N_("Kategorie → Dummies") if role == "Feature (X)" else N_("Text"))
        rows.append({"Spalte": c, "Rolle": role, "Typ": kind,
                     "Eindeutig": int(nun[c]), "Fehlend": int(df[c].isna().sum()), "Modell-Eingänge": inputs})
    overview = pd.DataFrame(rows)
    n_inputs = int(overview["Modell-Eingänge"].sum())
    with st.expander(tx("Alle Spalten – {n} Features → ca. {n_inputs} Modell-Eingänge", n=len(features),
                        n_inputs=n_inputs), expanded=True):
        st.dataframe(trc(overview, values=["Rolle", "Typ"]), hide_index=True, height=min(35 * (len(rows) + 1) + 3, 600))
    many = [f"{r['Spalte']} ({r['Eindeutig']})" for r in rows
            if r["Typ"] != "Zahl" and r["Rolle"] == "Feature (X)" and r["Eindeutig"] > 50]
    if many:
        st.warning(tx("Viele Kategorien → viele Dummy-Spalten: {many}. Tipp: Schritt 2 'Top-N' oder entfernen.",
                      many=', '.join(many)))

    st.subheader(tx("Fehlende Werte (NULL)"))
    st.caption(tx("Beim Clustering gibt es keinen Train/Test-Split – K-Means bekommt keine Antwort, die man vor ihm "
                  "verstecken müsste. Füllwerte, Kategorien (Dummies) und Scaler werden deshalb aus allen Zeilen "
                  "gelernt und später unverändert auf neue Daten angewendet (Schritt 10)."))
    miss_cols = [c for c in features if df[c].isna().any()]
    miss_table = None
    if miss_cols:
        miss_table = missing_editor(df, miss_cols, features)
        with st.expander(tx("Strategien erklärt")):
            st.dataframe(trc(pd.DataFrame([(s, p, d) for s, (p, d) in pl.MISSING_HELP.items()],
                                          columns=["Strategie", "Parameter", "Wirkung"]),
                             values=["Strategie", "Parameter", "Wirkung"]), hide_index=True)
        num_miss = [c for c in miss_cols if pl.is_numeric(df[c])]
        if num_miss:
            with st.expander(tx("Hilfe: Welche Spalte passt als Gruppe? – Korrelation mit der lückenhaften Spalte"),
                             icon=":material/query_stats:"):
                c1, _ = st.columns([2, 3])
                with c1:
                    mc = sb(tx("Spalte mit Lücken"), num_miss, "w_mc_col",
                            default=max(num_miss, key=lambda c: int(df[c].isna().sum())))
                corr = plots.corr_with_target(df[[c for c in cols if pl.is_numeric(df[c])]], mc)
                corr = corr.drop([c for c in corr.index if c == group or (group and shares_source(c, group))],
                                 errors="ignore")
                if corr.empty:
                    st.info(tx("Keine weiteren Zahlenspalten zum Vergleichen."))
                else:
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        show(plots.fig_corr_target(corr, mc))
                    table = corr.iloc[::-1].rename(tx("Korrelation")).round(3).to_frame()
                    table[tx("verschiedene Werte")] = [int(nun[c]) for c in table.index]
                    c2.dataframe(table, height=480)
                    best = corr.abs().sort_values(ascending=False)
                    best = [c for c in best.index if nun[c] <= 200][:1]
                    if best:
                        st.caption(tx("Am stärksten mit {mc} zusammen hängt **{v}** (r = {v2:+.2f}). Strategie "
                                      "*{m_group}* mit Parameter `{v}` füllt jede Lücke mit dem Mittelwert von "
                                      "{mc} in derselben {v}-Gruppe.",
                                      mc=mc, v=best[0], v2=corr[best[0]], m_group=tx(pl.M_GROUP)))
                code(f"df.select_dtypes(include='number').corr()[{mc!r}].sort_values()")
    else:
        st.success(tx("Keine fehlenden Werte in den gewählten Features."))

    st.subheader(tx("Skalierung & Dummies"))
    c1, c2, _ = st.columns([1.5, 1.4, 2])
    with c1:
        sb(tx("Skalierung"), pl.SCALERS, "w_scaler", default="StandardScaler", format_func=tx,
           help=tx("K-Means rechnet mit Abständen: Ohne Skalierung bestimmt die Spalte mit den größten Zahlen (z. B. "
                   "Apps oder F.Undergrad) fast allein die Cluster. StandardScaler = je Spalte Mittelwert 0 und "
                   "Standardabweichung 1 (üblich für K-Means), MinMaxScaler = 0 … 1. Im Kurs wurde nicht skaliert."))
    with c2:
        st.checkbox(tx("drop_first (Dummies)"), key="w_drop_first",
                    help=tx("Für Abstände besser aus: Mit drop_first liegt die weggelassene Kategorie (alle Dummies 0) "
                            "näher an allen anderen als diese untereinander."))
    if ss.get("w_scaler") == "keiner":
        st.warning(tx("Ohne Skalierung zählen große Zahlen viel mehr als kleine – das Ergebnis hängt dann von den "
                      "Einheiten ab (Euro oder Tausend Euro). Für K-Means meist StandardScaler wählen."))
    elif ss.get("w_scaler") == "StandardScaler" and any(not pl.is_numeric(df[c]) for c in features):
        st.caption(tx("Hinweis: Dummies (0/1) werden mitskaliert – seltene Kategorien bekommen dadurch großes Gewicht. "
                      "Bei vielen seltenen Kategorien MinMaxScaler erwägen."))

    if "model" in ss:
        st.warning(tx("Neu vorbereiten verwirft das trainierte Modell – vorher in Schritt 9 speichern."))
    if st.button(tx("Daten fürs Modell vorbereiten"), type="primary", disabled=not features):
        missing_cfg = {}
        if miss_table is not None:
            missing_cfg = {r["Spalte"]: {"strategy": r["Strategie"], "param": pl._clean(r["Parameter"])}
                           for r in miss_table.to_dict("records")}
        try:
            with st.spinner(tx("Bereite Daten vor …")):
                prep = pl.prepare_cluster_data(
                    df, features, missing=missing_cfg, drop_first=bool(ss.w_drop_first), scaler=ss.w_scaler,
                    column_actions=ss.actions_fitted, lineage=lin, meta=source_meta(),
                    new_columns=ss.get("new_cols_fitted", []), group_col=group, name_col=name_col)
        except ValueError as exc:
            st.error(str(exc))
        else:
            invalidate_from("prep")
            ss.prep, ss.prep_token = prep, uuid.uuid4().hex
            st.toast(tx("Daten vorbereitet"), icon=":material/check_circle:")

    if "prep" not in ss:
        return
    p = ss.prep
    r = p.recipe
    used = set(r["features"]) | set(r["dropped_columns"]) | set(r.get("empty_columns", []))
    now_missing = ({row["Spalte"]: {"strategy": row["Strategie"], "param": pl._clean(row["Parameter"])}
                    for row in miss_table.to_dict("records")} if miss_table is not None else {})
    if (used != set(features) or (r.get("group_requested"), r.get("name_col")) != (group, name_col)
            or r["scaler"]["type"] != ss.get("w_scaler") or r["drop_first"] != bool(ss.get("w_drop_first"))
            or r.get("missing_config", {}) != now_missing):
        st.info(tx("Die Einstellungen oben wurden seit dem Vorbereiten geändert → erneut **Daten fürs Modell "
                   "vorbereiten**."))
    st.success(tx("Fertig: X {shape} – {n} Zeilen, {n2} Features → {n3} Spalten (Modell-Eingänge)",
                  shape=p.X.shape, n=fmt_int(p.X.shape[0]), n2=len(r["features"]), n3=len(r["feature_columns"])))
    for line in p.log:
        st.markdown(f"- {line}")
    for w in p.warnings:
        st.warning(w)
    if p.groups is not None:
        vc = p.groups.value_counts(dropna=False)
        dist = pd.DataFrame({"Gruppe": [tx("ohne Wert") if pd.isna(v) else v for v in vc.index],
                             "Anzahl": vc.to_numpy()})
        dist["Anteil"] = (dist["Anzahl"] / dist["Anzahl"].sum()).map("{:.1%}".format)
        st.markdown(tx("**Vergleichsgruppe {g}** – geht nicht ins Modell, wird erst in Schritt 8 verglichen:",
                       g=r["group_col"]))
        c1, _ = st.columns([2, 3])
        c1.dataframe(trc(dist), hide_index=True)
    cat_feats = r["categorical_features"]
    if cat_feats:
        rows = []
        for c in cat_feats:
            cats = r["categories"][c]
            used = cats[1:] if r["drop_first"] else cats
            rows.append({"Kategorie-Spalte": c, "Kategorien": len(cats),
                         "weggelassen (drop_first)": cats[0] if r["drop_first"] and cats else "–",
                         "neue 0/1-Spalten": len(used),
                         "Beispiele": ", ".join(f"{c}_{v}" for v in used[:5]) + (" …" if len(used) > 5 else "")})
        n_dummy = sum(row["neue 0/1-Spalten"] for row in rows)
        with st.expander(tx("get_dummies: {n} Kategorie-Spalten → {n_dummy} Dummy-Spalten (0/1)",
                            n=len(cat_feats), n_dummy=n_dummy),
                         icon=":material/view_column:"):
            st.dataframe(trc(pd.DataFrame(rows)), hide_index=True)
            st.caption(tx("Jede Kategorie wird zu einer eigenen 0/1-Spalte, z. B. Region → Region_North, Region_South … "
                          "Mit drop_first entfällt die erste Kategorie – sie ist daran erkennbar, dass alle anderen 0 "
                          "sind. Unbekannte Werte in neuen Daten → alle 0."))
    st.markdown(tx("**X nach Dummy-Kodierung (unskaliert, erste Zeilen)**"))
    st.dataframe(p.X_preview)
    st.markdown(tx("**X nach {type} (erste Zeilen) – so sieht das Modell die Daten**", type=tx(r['scaler']['type'])))
    st.dataframe(pd.DataFrame(p.X[:5], columns=r["feature_columns"]).round(4))
    code(cg.step4_code(r))


# ===========================================================================
# Seite 5 – Daten speichern
# ===========================================================================
EXPORTS = {
    "step2": N_("Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)"),
    "ml": N_("Clustering-Daten nach Schritt 4 – Lücken gefüllt, Dummies; Bezeichnung und Vergleichsgruppe als "
             "Zusatzspalten"),
    "ml_scaled": N_("wie zuvor, zusätzlich skaliert – genau diese Zahlen sieht das Modell"),
    "clusters": N_("Daten nach Schritt 2 mit Cluster-Nummer, Cluster-Name und Abstand zum Zentrum (nach Schritt 6)"),
    "anomalies": N_("Daten nach Schritt 2 mit Anomalie (1 = auffällig) und Anomalie-Score – bei DBSCAN auch der Cluster "
                    "(nach Schritt 6)"),
}
FORMATS = {"intl": N_("CSV international (Komma, Dezimalpunkt)"),  # Excel: Format der gewählten Sprache/Region
           "excel": N_("CSV für deutsches Excel (Semikolon, Dezimalkomma)")}


def trained_model() -> str | None:
    """Modell der Sitzung (kmeans, iforest, lof, dbscan) – None, solange nichts trainiert ist."""
    return ss.model_cfg.model if "model" in ss and "model_cfg" in ss else None


def export_frame(what):
    if what == "step2":
        return ss.df_viz
    if what == "anomalies":
        h = ss.history
        return pl.anomalies_frame(ss.df_viz, ss.prep, h["flag"], h["score"],
                                  h["labels"] if trained_model() == pl.MODEL_DBSCAN else None, export_columns())
    if what == "clusters":
        return pl.clusters_frame(ss.df_viz, ss.prep, ss.model.labels_, ss.history["dist"], current_names(),
                                 export_columns())
    return pl.cluster_ready_frame(ss.df_viz, ss.prep, scaled=what == "ml_scaled")


def page_export():
    st.header(tx("5 · Daten speichern"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    st.markdown(tx("Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen "
                "Programm, nach dem Training auch mit der Cluster-Nummer je Zeile."))
    model = trained_model()
    options = ["step2", "ml", "ml_scaled", "anomalies" if pl.is_anomaly(model) else "clusters"]
    what = rd(tx("Was speichern?"), options, "w_exp_what", format_func=lambda k: tx(EXPORTS[k]))
    fmt = rd(tx("Format"), list(FORMATS), "w_exp_fmt", format_func=lambda k: tx(FORMATS[k]), horizontal=True)
    if what in ("ml", "ml_scaled") and "prep" not in ss:
        st.info(tx("Für die Clustering-Daten zuerst Schritt 4 ausführen (**Daten fürs Modell vorbereiten**)."))
        return
    if what == "clusters" and "model" not in ss:
        st.info(tx("Für die Cluster-Nummer je Zeile zuerst in Schritt 6 trainieren."))
        return
    if ss.get("w_use_sample"):
        st.warning(tx("Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.",
                      n=fmt_int(len(ss.df_raw)), n2=fmt_int(ss.source['rows'])))
    if what == "clusters":
        st.caption(tx("Die Cluster-Nummern sind willkürlich (0 … K−1). Abstand = euklidischer Abstand zum eigenen "
                      "Zentrum in den skalierten Features – groß = untypisch für den Cluster."))
    if what == "anomalies":
        st.caption(tx("Anomalie = 1: Score über der Schwelle aus dem Training. Anomalie-Score: höher = auffälliger "
                      "(Isolation Forest 0 … 1, LOF ≈ 1 normal, DBSCAN: Abstand zum nächsten Kernpunkt in Vielfachen von "
                      "eps)."))
    excel = i18n.get_lang() if fmt == "excel" else None  # 'de'/'pl': ; und Dezimalkomma · 'en': , und Punkt
    names = current_names() if "model" in ss else []
    # Sprache im Schlüssel: nach einem Sprachwechsel neu erstellen – sonst bliebe der Dateiname in der alten Sprache
    key = (what, fmt, excel, i18n.get_lang(), ss.viz_token, ss.get("prep_token") if what != "step2" else None,
           (ss.get("model_token"), tuple(names)) if what in ("clusters", "anomalies") else None)
    name = f"{Path(ss.source['name']).stem}_{tx(cg.EXPORT_FILES[what])}.csv"
    if st.button(tx("CSV erstellen"), type="primary", icon=":material/table_view:"):
        with st.spinner(tx("Erstelle CSV …")):
            frame = export_frame(what)
            ss._export = (key, pl.to_csv_bytes(frame, excel=excel), name, frame.shape, frame.head(10))
    exp = ss.get("_export")
    if exp and exp[0] == key:
        _, data, name, shape, head = exp
        st.success(tx("{name}: {n} Zeilen × {v} Spalten · {v2:.1f} MB",
                      name=name, n=fmt_int(shape[0]), v=shape[1], v2=len(data) / 1e6))
        st.dataframe(head)
        c1, c2 = st.columns([1, 3], vertical_alignment="bottom")
        c1.download_button(tx("CSV herunterladen"), data, file_name=name, mime="text/csv", icon=":material/download:")
        with c2:
            init("w_exp_path", str(APP_DIR / "exports"))
            folder = pl.clean_path(st.text_input(tx("… oder direkt in diesen Ordner speichern"), key="w_exp_path"))
        target = Path(folder).expanduser() / name if folder else None
        allowed = target is not None
        # fremde Datei? (die in dieser Sitzung gespeicherte darf ohne Rückfrage ersetzt werden)
        if target is not None and target.exists() and ss.get("_exp_saved") != str(target):
            st.warning(tx("Die Datei **{target}** gibt es schon.", target=target))
            allowed = st.checkbox(tx("Ja, vorhandene Datei überschreiben"), key=f"exp_overwrite_{name}")
        if st.button(tx("In Ordner speichern"), icon=":material/save:", disabled=not allowed):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            except OSError as exc:
                st.error(tx("Speichern fehlgeschlagen: {exc}", exc=exc))
            else:
                ss._exp_saved = str(target)
                st.success(tx("Gespeichert: {target}", target=target))
        notes = [tx("Für die Zuordnung neuer Daten (Schritt 10) die **ursprüngliche** CSV-Datei verwenden, nicht diese "
                    "Datei – das gespeicherte Modell wiederholt die Aufbereitung selbst.")]
        if excel:
            notes.append(tx("Texte, die mit =, +, -, @ beginnen, bekommen ein ' davor, damit Excel sie nicht als "
                         "Formel ausführt."))
        st.caption(" ".join(notes))
    recipe = ss.prep.recipe if "prep" in ss else None
    code(cg.export_code(what, recipe, excel, names, export_columns(), list(ss.df_viz.columns),
                        model=model or pl.MODEL_KMEANS))


# ===========================================================================
# Seite 6 – Modell & Training (K-Means, Isolation Forest, LOF, DBSCAN)
# ===========================================================================
HOW_KMEANS = N_("""**So arbeitet K-Means**

1. **K Startzentren** wählen – mit *k-means++* gut verteilt, mit *random* zufällige Zeilen.
2. **Zuordnen:** Jede Zeile kommt zum nächsten Zentrum (euklidischer Abstand in den skalierten Features).
3. **Verschieben:** Jedes Zentrum wandert in den Mittelwert seiner Zeilen. Schritt 2 und 3 wiederholen, bis sich
   nichts mehr ändert (höchstens *max_iter*-mal).
4. Alles **n_init-mal** mit anderen Startzentren – es gewinnt die Lösung mit der kleinsten **Inertia** (Summe der
   quadrierten Abstände jeder Zeile zu ihrem Zentrum).

K-Means findet eher runde, ähnlich große Gruppen. K gibt man vor – welches K zu den Daten passt, zeigt Schritt 7.""")


HOW_ANOMALY = {
    pl.MODEL_IFOREST: N_("""**So arbeitet Isolation Forest**

1. **Viele Zufallsbäume** (*n_estimators*), jeder auf einer Stichprobe der Zeilen (*max_samples*, Standard 256).
2. Jeder Baum **schneidet** die Daten an zufälligen Stellen eines zufälligen Features – so lange, bis jede Zeile allein
   ist.
3. **Ungewöhnliche Zeilen sind schnell allein:** wenige Schnitte = kurzer Pfad = hoher Score (0 … 1, um 0.5 normal).
4. **Schwelle:** *contamination = auto* nimmt Score 0.5 aus dem Original-Paper; ein fester Anteil (z. B. 1 %) macht die
   auffälligsten 1 % der Trainingszeilen zu Anomalien.

Isolation Forest schneidet je Feature und braucht deshalb keine Skalierung – sie schadet aber nicht."""),
    pl.MODEL_LOF: N_("""**So arbeitet Local Outlier Factor (LOF)**

1. Für jede Zeile die **n_neighbors nächsten Nachbarn** (Abstand in den skalierten Features).
2. **Lokale Dichte:** wie nah liegen die Nachbarn?
3. **LOF = Dichte der Nachbarn / eigene Dichte:** ≈ 1 = so dicht wie die Umgebung (normal), deutlich über 1 = die Zeile
   liegt dünner als ihre Nachbarn – auch direkt neben einer dichten Gruppe (lokaler Ausreißer).
4. **Schwelle:** *contamination = auto* nimmt LOF 1.5; ein fester Anteil macht die auffälligsten Trainingszeilen zu
   Anomalien. *novelty=True*: Das Modell bewertet in Schritt 10 auch neue Daten."""),
    pl.MODEL_DBSCAN: N_("""**So arbeitet DBSCAN**

1. **Kernpunkt:** mindestens *min_samples* Zeilen (sie selbst eingeschlossen) im Umkreis *eps*.
2. Kernpunkte, die höchstens *eps* voneinander entfernt sind, bilden **einen Cluster**; Randpunkte gehören zum Cluster
   eines nahen Kernpunkts.
3. **Alles andere ist Rauschen** (Cluster −1) – das sind die Anomalien. Die Zahl der Cluster ergibt sich von selbst.
4. **Score** = Abstand zum nächsten Kernpunkt in Vielfachen von eps (über 1 = Rauschen). Neue Daten (Schritt 10)
   bekommen den Cluster des nächsten Kernpunkts – oder −1.

eps hängt von der Skalierung und der Zahl der Features ab – welches eps passt, zeigt Schritt 7 (k-Distanz)."""),
}


def model_config() -> pl.ClusterConfig:
    """Einstellungen aus den Widgets von Schritt 6 → ClusterConfig (nicht angezeigte Widgets = Standardwerte)."""
    d, g = pl.ClusterConfig(), ss.get
    model = g("w_model", d.model)
    metric = g("w_metric", d.metric)
    return pl.ClusterConfig(
        model=model if model in pl.MODEL_NAMES else d.model,
        n_clusters=int(g("w_k", d.n_clusters)), init=g("w_init", d.init), n_init=int(g("w_n_init", d.n_init)),
        max_iter=int(g("w_max_iter", d.max_iter)), algorithm=g("w_algorithm", d.algorithm), seed=int(g("w_seed", d.seed)),
        curve=bool(g("w_curve", d.curve)), k_max=int(g("w_k_max", d.k_max)),
        contamination=0.0 if g("w_contam_auto", True) else float(g("w_contam_pct", 1.0)) / 100,
        n_estimators=int(g("w_n_estimators", d.n_estimators)), max_samples=int(g("w_max_samples", d.max_samples)),
        n_neighbors=int(g("w_n_neighbors", d.n_neighbors)), eps=float(g("w_eps", d.eps)),
        min_samples=int(g("w_min_samples", d.min_samples)), metric=metric if metric in pl.METRICS else d.metric)


def fmt_score(v, digits=3) -> str:
    v = float(v)
    return f"{v:.{digits}f}" if np.isfinite(v) else "–"


def kmeans_params_ui(n):
    """Schritt 6: Parameter von K-Means und die Kurve für Schritt 7."""
    st.subheader(tx("Parameter"))
    c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1.4, 1, 1.1, 1.2, 1.3])
    with c1:
        init("w_k", 3)
        st.number_input(tx("K (n_clusters)"), 2, pl.MAX_K, step=1, key="w_k",
                        help=tx("Anzahl der Cluster – wie im Kurs KMeans(n_clusters=2). Welches K passt, zeigt Schritt 7 "
                                "(Elbow-Methode und Silhouette)."))
    with c2:
        sb("init", pl.INITS, "w_init", default="k-means++",
           help=tx("k-means++ = Startzentren gut verteilt (Standard, meist schneller und besser) · random = zufällige "
                   "Zeilen als Startzentren"))
    with c3:
        init("w_n_init", 10)
        st.number_input("n_init", 1, 100, step=1, key="w_n_init",
                        help=tx("Wie oft K-Means mit anderen Startzentren läuft – das beste Ergebnis (kleinste Inertia) "
                                "gewinnt. scikit-learn nimmt sonst 'auto' = 1 Lauf bei k-means++."))
    with c4:
        init("w_max_iter", 300)
        st.number_input("max_iter", 10, 100_000, step=50, key="w_max_iter",
                        help=tx("Höchstzahl der Schritte Zuordnen → Verschieben je Lauf (Standard 300)."))
    with c5:
        sb("algorithm", pl.ALGORITHMS, "w_algorithm", default="lloyd",
           help=tx("lloyd = klassisches K-Means (Standard) · elkan = gleiches Ergebnis, bei gut getrennten Clustern "
                   "manchmal schneller"))
    with c6:
        init("w_seed", pl.RANDOM_STATE)
        st.number_input(tx("Seed (random_state)"), 0, 1_000_000, step=1, key="w_seed",
                        help=tx("Gleicher Seed = gleiche Startzentren = gleiches Ergebnis."))

    st.subheader(tx("Wahl von K für Schritt 7"))
    c1, c2, _ = st.columns([3, 1.1, 2])
    with c1:
        init("w_curve", True)
        curve_on = st.checkbox(tx("Elbow- und Silhouette-Kurve berechnen (K = 1 … k_max)"), key="w_curve",
                               help=tx("Trainiert K-Means zusätzlich für jedes K von 1 bis k_max und misst Inertia und "
                                       "Silhouette – Schritt 7 zeigt damit, welches K passt. Dauert etwas länger."))
    with c2:
        init("w_k_max", pl.K_MAX_DEFAULT)
        st.number_input("k_max", 3, 30, step=1, key="w_k_max", disabled=not curve_on)


def contamination_ui():
    """Schritt 6 (Isolation Forest, LOF): Schwelle – 'auto' oder fester Anteil auffälliger Zeilen."""
    c1, c2, _ = st.columns([1.6, 1.4, 3])
    with c1:
        init("w_contam_auto", True)
        auto = st.checkbox(tx("contamination = auto"), key="w_contam_auto",
                           help=tx("auto: Schwelle aus dem Verfahren selbst (Isolation Forest: Score 0.5, LOF: 1.5). "
                                   "Aus: ein fester Anteil auffälliger Zeilen."))
    with c2:
        init("w_contam_pct", 1.0)
        st.number_input(tx("Anteil auffällig in %"), 0.1, 50.0, step=0.1, format="%.2f", key="w_contam_pct",
                        disabled=auto,
                        help=tx("contamination in Prozent: 1 = die auffälligsten 1 % der Trainingszeilen gelten als "
                                "Anomalien. Einen Vorschlag zeigt Schritt 7 (Knick der Score-Kurve)."))


def anomaly_params_ui(key, p, n, n_feat):
    """Schritt 6: Parameter von Isolation Forest, LOF oder DBSCAN."""
    with st.expander(tx("So arbeitet {name}", name=pl.model_name(key)), icon=":material/info:"):
        st.markdown(tx(HOW_ANOMALY[key]))
    st.subheader(tx("Parameter"))
    if key == pl.MODEL_IFOREST:
        c1, c2, c3, _ = st.columns([1.2, 1.4, 1.3, 2])
        with c1:
            init("w_n_estimators", 100)
            st.number_input("n_estimators", 10, 2000, step=50, key="w_n_estimators",
                            help=tx("Anzahl der Zufallsbäume – mehr = stabilerer Score, langsamer (Standard 100)."))
        with c2:
            init("w_max_samples", 0)
            st.number_input("max_samples", 0, 10_000_000, step=64, key="w_max_samples",
                            help=tx("Zeilen je Baum: 0 = auto = min(256, Zeilen). Kleine Stichproben isolieren "
                                    "Ausreißer am besten – größere Werte selten nötig."))
        with c3:
            init("w_seed", pl.RANDOM_STATE)
            st.number_input(tx("Seed (random_state)"), 0, 1_000_000, step=1, key="w_seed",
                            help=tx("Gleicher Seed = gleiche Bäume = gleiches Ergebnis."))
        contamination_ui()
    elif key == pl.MODEL_LOF:
        c1, c2, _ = st.columns([1.2, 1.4, 3])
        with c1:
            init("w_n_neighbors", 20)
            st.number_input("n_neighbors", 2, 500, step=1, key="w_n_neighbors",
                            help=tx("Anzahl der Nachbarn: klein = sehr lokale Ausreißer, groß = Vergleich mit einer "
                                    "größeren Umgebung (Standard 20). Bilden Ausreißer selbst eine kleine Gruppe, muss "
                                    "n_neighbors größer sein als diese Gruppe – sonst gelten sie untereinander als "
                                    "normal."))
        with c2:
            sb("metric", pl.METRICS, "w_metric", default="euclidean",
               help=tx("Abstandsmaß: euclidean = Luftlinie (Standard) · manhattan = Summe der Abstände je Feature"))
        contamination_ui()
    else:
        init("w_min_samples", int(min(max(5, 2 * n_feat), 50)))
        if "w_eps" not in ss:  # Startwert aus der k-Distanz einer Stichprobe – genauer misst Schritt 7
            ss.w_eps = cached("_eps_start", (ss.get("prep_token"), int(ss.w_min_samples), ss.get("w_metric")),
                              lambda: pl.suggest_eps(p.X, ss.w_min_samples, ss.get("w_metric", "euclidean")))
        c1, c2, c3, _ = st.columns([1.2, 1.2, 1.4, 2])
        with c1:
            st.number_input("eps", 1e-6, 1e9, step=0.05, format="%.4g", key="w_eps",
                            help=tx("Radius der Nachbarschaft in den skalierten Features. Startwert: Knick der "
                                    "k-Distanz einer Stichprobe – Schritt 7 zeigt, welches eps passt."))
        with c2:
            st.number_input("min_samples", 2, 1000, step=1, key="w_min_samples",
                            help=tx("So viele Zeilen (sie selbst eingeschlossen) müssen im Radius eps liegen, damit eine "
                                    "Zeile Kernpunkt ist. Faustregel: 2 × Zahl der Features (hier {v}).",
                                    v=int(min(max(5, 2 * n_feat), 50))))
        with c3:
            sb("metric", pl.METRICS, "w_metric", default="euclidean",
               help=tx("Abstandsmaß: euclidean = Luftlinie (Standard) · manhattan = Summe der Abstände je Feature"))
        init("w_curve", True)
        st.checkbox(tx("Rauschanteil je eps berechnen (für Schritt 7)"), key="w_curve",
                    help=tx("DBSCAN läuft zusätzlich für einige eps-Werte (Quantile der k-Distanz) – Schritt 7 zeigt "
                            "dann Rauschanteil und Zahl der Cluster je eps. Nur bis {m} Zeilen.", m=fmt_int(pl.CURVE_ROWS)))


def page_train():
    st.header(tx("6 · Modell & Training"))
    if not ready("prep", tx("Bitte zuerst Schritt 4 ausführen (Daten fürs Modell vorbereiten).")):
        return
    p = ss.prep
    r = p.recipe
    n, n_feat = p.X.shape
    key = rd(tx("Methode"), list(pl.MODEL_NAMES), "w_model", default=pl.MODEL_KMEANS, horizontal=True,
             format_func=pl.model_name)
    st.info(tx("**{name}** · {n} Zeilen · {p} Features (Modell-Eingänge) · Skalierung: {scaler}",
               name=pl.model_name(key), n=fmt_int(n), p=n_feat, scaler=tx(r["scaler"]["type"])))
    st.caption(tx(pl.MODEL_WHY[key]))
    if key == pl.MODEL_KMEANS:
        with st.expander(tx("So arbeitet K-Means"), icon=":material/info:"):
            st.markdown(tx(HOW_KMEANS))
        kmeans_params_ui(n)
    else:
        anomaly_params_ui(key, p, n, n_feat)
    cfg = model_config()
    too_big = key == pl.MODEL_KMEANS and cfg.n_clusters >= n
    if too_big:
        st.error(tx("K = {k} ist zu groß für {n} Zeilen.", k=cfg.n_clusters, n=n))
    if r["scaler"]["type"] == "keiner" and key == pl.MODEL_KMEANS:
        st.warning(tx("Ohne Skalierung (Schritt 4) bestimmen Features mit großen Zahlen fast allein die Abstände – und "
                      "damit die Cluster. In Schritt 4 StandardScaler wählen."))
    elif r["scaler"]["type"] == "keiner" and key in (pl.MODEL_LOF, pl.MODEL_DBSCAN):
        st.warning(tx("Ohne Skalierung (Schritt 4) bestimmen Features mit großen Zahlen fast allein die Abstände – LOF "
                      "und DBSCAN rechnen mit Abständen. In Schritt 4 StandardScaler wählen."))
    if key == pl.MODEL_KMEANS and n > 100_000 and (cfg.n_init >= 10 or cfg.curve):
        st.info(tx("{n} Zeilen: K-Means mit n_init = {i} kann einige Minuten dauern (die Kurve für Schritt 7 nutzt eine "
                   "Stichprobe von {s} Zeilen). Tipp: n_init senken oder Stichprobe in der Seitenleiste.",
                   n=fmt_int(n), i=cfg.n_init, s=fmt_int(pl.CURVE_ROWS)))
    if key in (pl.MODEL_LOF, pl.MODEL_DBSCAN) and n > 50_000:
        st.info(tx("{n} Zeilen: LOF und DBSCAN vergleichen jede Zeile mit ihren Nachbarn – das kann lange dauern und viel "
                   "Speicher brauchen. Tipp: Stichprobe in der Seitenleiste oder Isolation Forest.", n=fmt_int(n)))

    st.subheader(tx("scikit-learn-Code mit diesen Einstellungen"))
    st.code(cg.model_code(cfg, n), language="python")

    if "model" in ss:
        ti, mc = ss.train_info, ss.model_cfg
        if pl.is_anomaly(mc.model):
            st.info(tx("Aktuelles Modell: {name} · trainiert {v} · {seconds} s. Neues Training ersetzt es.",
                       name=pl.model_name(mc.model), v=ti["trained_at"][11:], seconds=ti["seconds"]))
        else:
            st.info(tx("Aktuelles Modell: K-Means mit K = {k} · trainiert {v} · {seconds} s. Neues Training ersetzt es.",
                       k=ti["k"], v=ti["trained_at"][11:], seconds=ti["seconds"]))
    st.caption(tx("Während des Trainings nichts anklicken – jede Eingabe bricht das Training ab."))
    if st.button(tx("Training starten"), type="primary", icon=":material/rocket_launch:", disabled=too_big):
        bar = st.progress(0.0, text=tx("Starte Training …"))
        live = st.empty()
        last = [0.0]
        total = len(pl.k_values(cfg, min(n, pl.CURVE_ROWS)))

        def progress(frac, text, curve):
            """Fortschrittsbalken + Live-Kurven (Inertia und Silhouette je K) während Schritt 7 berechnet wird."""
            bar.progress(min(max(float(frac), 0.0), 1.0), text=text)
            if curve and curve["values"] and (time.time() - last[0] >= 0.4 or len(curve["values"]) == total):
                last[0] = time.time()
                index = pd.Index(curve["values"], name="K")
                with live.container():
                    c1, c2 = st.columns(2)
                    c1.line_chart(pd.DataFrame({"Inertia": curve["inertia"]}, index=index), color=plots.SERIES[0],
                                  height=220)
                    sil = pd.DataFrame({tx("Silhouette"): curve["silhouette"]}, index=index).dropna()
                    if len(sil):
                        c2.line_chart(sil, color=plots.SERIES[2], height=220)

        try:
            if key == pl.MODEL_KMEANS:
                model, hist, info = pl.train_kmeans(cfg, p, progress)
            else:
                model, hist, info = pl.train_anomaly(cfg, p, progress)
        except Exception as exc:  # noqa: BLE001
            st.error(tx("Training fehlgeschlagen: {exc}", exc=exc))
            return
        bar.progress(1.0, text=tx("Fertig · {seconds} s", seconds=info["seconds"]))
        invalidate_from("model")
        ss.model, ss.history, ss.train_info, ss.model_cfg = model, hist, info, cfg
        ss.model_token = uuid.uuid4().hex  # eindeutig je Training (id() kann nach dem Löschen wiederverwendet werden)
        if key == pl.MODEL_KMEANS:
            st.success(tx("K-Means fertig: K = {k} · Silhouette {s} · Inertia {i} · {it} Iterationen → weiter mit "
                          "Schritt 7.",
                          k=info["k"], s=fmt_score(info["silhouette"]), i=fmt_num(info["inertia"]), it=info["n_iter"]))
        elif key == pl.MODEL_DBSCAN:
            st.success(tx("DBSCAN fertig: {c} Cluster, {k} Zeilen Rauschen ({pct}) → weiter mit Schritt 7 (eps) und 8 "
                          "(Auswertung).", c=info["k"], k=fmt_int(info["n_anomalies"]), pct=pl.share_text(info["share"])))
        else:
            st.success(tx("{name} fertig: {k} von {n} Zeilen auffällig ({pct}) · Schwelle {thr} → weiter mit Schritt 7 "
                          "(Schwelle) und 8 (Auswertung).",
                          name=pl.model_name(key), k=fmt_int(info["n_anomalies"]), n=fmt_int(info["n_rows"]),
                          pct=pl.share_text(info["share"]), thr=pl.score_text(info["threshold"])))
    if "model" in ss:
        info = ss.train_info
        if not info.get("converged", True):
            st.warning(tx("K-Means hat max_iter erreicht, ohne stabil zu werden – max_iter erhöhen."))
        for w in info.get("warnings", []):
            st.warning(tx("Hinweis von scikit-learn: {w}", w=w))
        with st.expander(tx("Modell (scikit-learn)")):
            st.code(repr(ss.model), language="text")


# ===========================================================================
# Seite 7 – Wahl von K: Elbow-Methode & Silhouette
# ===========================================================================
@with_lang
def on_apply_k(k):
    ss.w_k = int(k)
    ss._k_msg = tx("Übernommen: K = {k} – jetzt in Schritt 6 neu trainieren.", k=int(k))


@with_lang
def on_apply_contamination(c):
    ss.w_contam_auto = False
    ss.w_contam_pct = round(float(c) * 100, 2)
    ss._k_msg = tx("Übernommen: contamination = {pct} – jetzt in Schritt 6 neu trainieren.", pct=pl.share_text(c))


@with_lang
def on_apply_eps(e):
    ss.w_eps = float(e)
    ss._k_msg = tx("Übernommen: eps = {v} – jetzt in Schritt 6 neu trainieren.", v=f"{float(e):.4g}")


def page_threshold():
    """Schritt 7 für Isolation Forest und LOF (Schwelle des Scores) bzw. DBSCAN (eps aus der k-Distanz)."""
    info, cfg, hist = ss.train_info, ss.model_cfg, ss.history
    curve = hist.get("curve") or {}
    key = cfg.model
    if key == pl.MODEL_DBSCAN:
        st.header(tx("7 · Wahl von eps – DBSCAN (k-Distanz)"))
    else:
        st.header(tx("7 · Schwelle für Anomalien – {name}", name=pl.model_name(key)))
    a = pl.threshold_assessment(curve, info, cfg)
    m = st.columns(4)
    if key == pl.MODEL_DBSCAN:
        m[0].metric(tx("Cluster"), info["k"])
        m[1].metric(tx("Rauschen"), f"{fmt_int(info['n_anomalies'])} · {pl.share_text(info['share'])}",
                    help=tx("Zeilen ohne Kernpunkt im Umkreis eps – die Anomalien."))
        m[2].metric(tx("Kernpunkte"), fmt_int(info["n_core"]),
                    help=tx("Zeilen mit mindestens min_samples Zeilen im Umkreis eps."))
        m[3].metric("eps", f"{info['eps']:.4g}")
    else:
        m[0].metric(tx("auffällige Zeilen"), f"{fmt_int(info['n_anomalies'])} · {pl.share_text(info['share'])}")
        m[1].metric(tx("Schwelle (Score)"), pl.score_text(info["threshold"]),
                    help=tx("Auffällig ist ein Score über der Schwelle."))
        m[2].metric(tx("Median-Score"), pl.score_text(info["score_median"]),
                    help=tx("Typischer Score einer normalen Zeile."))
        m[3].metric(tx("höchster Score"), pl.score_text(info["score_max"]))
    with chart_guard():
        if key == pl.MODEL_DBSCAN:
            show(plots.fig_kdist(curve, float(cfg.eps)))
        else:
            show(plots.fig_scores(hist["score"], info["threshold"], curve))
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("\n\n".join(a["notes"]))
    with c2:
        if key == pl.MODEL_DBSCAN:
            st.markdown(tx("**So liest man die k-Distanz:** Für jede Zeile der Abstand zu ihrem min_samples-ten Nachbarn, "
                           "aufsteigend sortiert. Flach = dichte Bereiche, am Ende steil = Zeilen weit weg von allen "
                           "anderen. Ein eps am Knick trennt beides: darunter Kernpunkte, darüber meist Rauschen. Rechts: "
                           "Rauschanteil und Zahl der Cluster für andere eps-Werte."))
        else:
            st.markdown(tx("**So liest man die Bilder:** Links die Verteilung des Scores – rot = über der Schwelle. Rechts "
                           "die auffälligsten Zeilen nach Score sortiert: Wo die Kurve abflacht (Knick), unterscheiden sich "
                           "die Zeilen kaum noch – davor liegen die deutlichen Ausreißer. Eine Schwelle ist eine "
                           "Entscheidung, keine Wahrheit: lieber wenige klare Anomalien (z. B. 0.5–2 %) als eine lange "
                           "Liste."))
        for label, value in a["suggest"]:
            if key == pl.MODEL_DBSCAN:
                st.button(label, icon=":material/tune:", on_click=on_apply_eps, args=(value,), key=f"apply_eps_{value}")
            else:
                st.button(label, icon=":material/tune:", on_click=on_apply_contamination, args=(value,),
                          key=f"apply_c_{value}")
        msg = ss.pop("_k_msg", None)
        if msg:
            st.success(msg)
    with st.expander(tx("Tabelle")):
        if key == pl.MODEL_DBSCAN:
            grid = curve.get("grid")
            if grid:
                st.dataframe(trc(pd.DataFrame({"eps": grid["eps"], "Cluster": grid["clusters"],
                                               "Rauschen": [pl.share_text(v) for v in grid["noise"]]})),
                             hide_index=True)
            else:
                st.caption(tx("Rauschanteil je eps wurde nicht berechnet (Schritt 6, nur bis {m} Zeilen).",
                              m=fmt_int(pl.CURVE_ROWS)))
        elif curve:
            st.dataframe(trc(pd.DataFrame({"Anteil auffällig": [pl.share_text(c) for c in curve["levels"]],
                                           "Schwelle (Score)": [pl.score_text(t) for t in curve["thresholds"]],
                                           "Zeilen über der Schwelle": curve["counts"]})), hide_index=True)
            st.caption(tx("contamination = Anteil auffälliger Zeilen → die passende Schwelle auf dem Score."))
    code(cg.curve_code(cfg, info["n_rows"]))


def page_k():
    if "model" in ss and pl.is_anomaly(ss.model_cfg.model):
        page_threshold()
        return
    st.header(tx("7 · Wahl von K – Elbow-Methode & Silhouette"))
    if not ready("model", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    info, cfg, hist = ss.train_info, ss.model_cfg, ss.history
    curve = hist.get("curve")
    a = pl.k_assessment(curve, info, cfg)
    m = st.columns(5)
    m[0].metric("K", info["k"])
    m[1].metric(tx("Silhouette"), fmt_score(info["silhouette"]),
                help=tx("−1 … 1: Wie viel näher liegt jede Zeile an ihrem eigenen Cluster als am nächsten anderen? "
                        "Größer = besser getrennt."))
    m[2].metric("Inertia", fmt_num(info["inertia"]),
                help=tx("Summe der quadrierten Abstände zum eigenen Zentrum – kleiner = kompakter, sinkt aber mit "
                        "jedem zusätzlichen Cluster."))
    m[3].metric("Davies-Bouldin", fmt_score(info["davies_bouldin"]),
                help=tx("Kleiner = besser: Cluster kompakt und weit voneinander entfernt."))
    m[4].metric(tx("erklärte Streuung (R²)"), f"{info['r2']:.1%}" if np.isfinite(info["r2"]) else "–",
                help=tx("Anteil der Gesamtstreuung, den die Cluster erklären = 1 − Inertia / Gesamtstreuung."))
    if curve:
        show(plots.fig_k_curve(curve, cfg.n_clusters))
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("\n\n".join(a["notes"]))
        with c2:
            st.markdown(tx("**So liest man die Kurven:** Die Inertia sinkt mit jedem weiteren Cluster – gesucht ist "
                           "der *Knick* (Ellbogen), nach dem sie nur noch langsam sinkt. Die Silhouette misst, wie gut "
                           "die Cluster getrennt sind: Ihr Maximum ist ein guter Kandidat. Beide sind Hinweise – am "
                           "Ende zählt, ob sich die Cluster in Schritt 8 sinnvoll beschreiben lassen."))
            for label, k in a["suggest"]:
                st.button(label, icon=":material/tune:", on_click=on_apply_k, args=(k,), key=f"apply_k_{k}")
            msg = ss.pop("_k_msg", None)
            if msg:
                st.success(msg)
        with st.expander(tx("Tabelle")):
            table = pd.DataFrame({"K": curve["values"], "Inertia": curve["inertia"], "Silhouette": curve["silhouette"],
                                  "Davies-Bouldin": curve["davies_bouldin"],
                                  "Calinski-Harabasz": curve["calinski_harabasz"]})
            st.dataframe(table.round(4), hide_index=True)
            st.caption(tx("Davies-Bouldin: kleiner = besser · Calinski-Harabasz: größer = besser."))
    else:
        st.markdown("\n\n".join(a["notes"]))
        st.info(tx("Die Elbow-/Silhouette-Kurve ist in Schritt 6 ausgeschaltet – mit Kurve zeigt diese Seite, welches K "
                   "zu den Daten passt."))

    st.subheader(tx("Silhouetten-Diagramm"))
    sv = cached("_silvals", ss.model_token, lambda: pl.silhouette_values(ss.prep.X, ss.model.labels_, cfg.seed))
    if sv is None:
        st.info(tx("Silhouette nicht berechenbar (weniger als 2 Cluster in den Daten)."))
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_silhouette(sv, info["k"], current_names()))
        c2.markdown(tx("**Lesart:** Jede Zeile ist ein waagerechter Strich, je Cluster nach der Silhouette sortiert. "
                       "Nahe 1 = liegt mitten im eigenen Cluster, um 0 = auf der Grenze zum nächsten Cluster, unter 0 "
                       "= näher an einem anderen Cluster (vermutlich falsch zugeordnet). Ein guter Cluster ragt weit "
                       "über die gestrichelte Linie (Mittelwert) hinaus und hat kaum Werte unter 0."))
    code(cg.curve_code(cfg, info["n_rows"]))


# ===========================================================================
# Seite 8 – Cluster-Auswertung
# ===========================================================================
def current_names():
    """Namen der Cluster aus Schritt 8 ('' = nicht benannt) – nur für das aktuelle K-Means-Modell."""
    if "train_info" not in ss or pl.is_anomaly(ss.train_info.get("model")):
        return []
    k = ss.train_info["k"]
    table = editor_value("ed_names")
    if ss.get("_names_for") != ss.get("model_token") or table is None or "Name" not in table.columns:
        return [""] * k
    return pl.clean_names(table["Name"].tolist(), k)


def describe_line(d, sizes, names) -> str:
    """Eine Zeile Text je Cluster: was ihn vom Durchschnitt unterscheidet."""
    j = d["cluster"]
    total = max(int(np.sum(sizes)), 1)
    head = tx("**{label}** ({n} Zeilen · {pct})", label=md_text(pl.cluster_label(j, names)), n=fmt_int(sizes[j]),
              pct=f"{sizes[j] / total:.0%}")
    if not sizes[j]:
        return head + ": " + tx("leer – keine Zeile liegt diesem Zentrum am nächsten (doppelte Zeilen, K kleiner "
                                "wählen)")

    def z_text(z):
        return f"{z:+.1f}".replace("-", "−") + " σ"

    parts = []
    if d["high"]:
        parts.append(tx("über dem Durchschnitt: ") + ", ".join(f"{md_text(f)} ({z_text(z)})" for f, z in d["high"]))
    if d["low"]:
        parts.append(tx("unter dem Durchschnitt: ") + ", ".join(f"{md_text(f)} ({z_text(z)})" for f, z in d["low"]))
    return head + ": " + ("; ".join(parts) if parts else tx("liegt nahe am Durchschnitt aller Zeilen"))


def page_anomalies():
    """Schritt 8 für Isolation Forest, LOF und DBSCAN: die auffälligsten Zeilen, Profil, PCA-Bild, Vergleich."""
    p, model, info, cfg = ss.prep, ss.model, ss.train_info, ss.model_cfg
    r, key, token, h = p.recipe, cfg.model, ss.model_token, ss.history
    score, flag, labels = h["score"], h["flag"], h["labels"]
    st.header(tx("8 · Auswertung der Anomalien – {name}", name=pl.model_name(key)))
    st.caption(tx("{name} auf {n} Zeilen · {k} auffällig ({pct}) · Schwelle: Score über {thr}",
                  name=pl.model_name(key), n=fmt_int(info["n_rows"]), k=fmt_int(info["n_anomalies"]),
                  pct=pl.share_text(info["share"]), thr=pl.score_text(info["threshold"])))
    if key == pl.MODEL_DBSCAN:
        st.subheader(tx("Cluster und Rauschen"))
        c1, _ = st.columns([3, 2])
        with c1:
            show(plots.fig_dbscan_sizes(info["sizes"], info["n_anomalies"]))

    st.subheader(tx("Die auffälligsten Zeilen"))
    st.caption(tx("Nach Score sortiert. 'Zeile' = Nummer der Zeile in den Daten nach Schritt 2 (ab 0). Die Abweichungen "
                  "(z = Standardabweichungen vom Durchschnitt aller Zeilen) zeigen, was eine Zeile ungewöhnlich macht – "
                  "ein Blick in die Daten, nicht ins Modell."))
    c1, _ = st.columns([1, 4])
    with c1:
        init("w_top_n", 20)
        n_top = st.number_input(tx("Anzahl"), 5, 500, step=5, key="w_top_n")
    top = cached("_top", (token, int(n_top)), lambda: pl.top_anomalies(
        p, score, flag, int(n_top), labels if key == pl.MODEL_DBSCAN else None))
    st.dataframe(trc(top), hide_index=True)

    st.subheader(tx("Profil – was unterscheidet auffällige von normalen Zeilen?"))
    if not info["n_anomalies"]:
        st.info(tx("Keine auffällige Zeile – die Schwelle in Schritt 7 prüfen."))
    else:
        prof = cached("_aprof", token, lambda: pl.anomaly_profiles(p, flag))
        d = pl.describe_clusters(prof["z"][[1]])[0]
        parts = []
        if d["high"]:
            parts.append(tx("über dem Durchschnitt: ") + ", ".join(
                f"{md_text(f)} (" + f"{z:+.1f}".replace("-", "−") + " σ)" for f, z in d["high"]))
        if d["low"]:
            parts.append(tx("unter dem Durchschnitt: ") + ", ".join(
                f"{md_text(f)} (" + f"{z:+.1f}".replace("-", "−") + " σ)" for f, z in d["low"]))
        st.markdown(tx("**Auffällige Zeilen** im Mittel: ") + ("; ".join(parts) if parts
                                                               else tx("liegen nahe am Durchschnitt aller Zeilen")))
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_profile(prof["z"], labels=[tx("normal"), tx("auffällig")],
                                   title=tx("Profil: Abweichung vom Durchschnitt (z)")))
        with c2:
            means = prof["means"].copy()
            means.columns = [tx("Alle"), tx("normal"), tx("auffällig")]
            st.markdown(tx("**Mittelwerte in Originaleinheiten** (Dummies: Anteil 0 … 1)"))
            st.dataframe(means.round(3), height=min(35 * (len(means) + 1) + 3, 640))

    st.subheader(tx("Darstellung"))
    proj = cached("_apca", token, lambda: pl.pca_projection(
        p, model, seed=cfg.seed, labels=labels if key == pl.MODEL_DBSCAN else flag.astype(int), priority=flag))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_anomaly_pca(proj, flag[proj["rows"]], dbscan=key == pl.MODEL_DBSCAN))
    c2.markdown(tx("**PCA nur für das Bild:** Die {p} Spalten werden auf die 2 Richtungen mit der größten Streuung "
                   "projiziert ({pct} der Streuung). Das Modell rechnet mit allen Spalten – eine Anomalie kann im Bild "
                   "mitten in der Wolke liegen und trotzdem in einer anderen Richtung ungewöhnlich sein. Auffällige "
                   "Zeilen sind im Bild immer dabei.", p=len(r["feature_columns"]), pct=f"{sum(proj['explained']):.0%}"))

    if p.groups is not None:
        anomaly_group_comparison(p, flag, score)

    st.subheader(tx("Einzelfall prüfen"))
    st.caption(tx("Zufällige Zeile ziehen: Wie auffällig ist sie, und welche Features weichen am stärksten vom "
                  "Durchschnitt ab?"))
    if st.button(tx("Zufällige Zeile ziehen"), icon=":material/casino:"):
        ss.case_idx = int(np.random.default_rng().integers(len(p.index)))
    if "case_idx" in ss and ss.case_idx < len(p.index):
        i = ss.case_idx
        show_cols = [c for c in [r.get("name_col"), *r["features"], r.get("group_col")] if c]
        row = ss.df_viz.loc[p.index[i], list(dict.fromkeys(show_cols))]
        c1, c2 = st.columns([2, 3])
        c1.dataframe(row.map(lambda v: "–" if pd.isna(v) else pl.format_value(v)).rename(tx("Wert")).to_frame(),
                     height=420)
        with c2:
            m1, m2, m3 = st.columns(3)
            m1.metric(tx("Anomalie"), tx("ja") if flag[i] else tx("nein"))
            m2.metric(tx("Anomalie-Score"), pl.score_text(score[i]),
                      help=tx("Schwelle: {thr} – auffällig ist ein Score darüber.",
                              thr=pl.score_text(info["threshold"])))
            m3.metric(tx("Rang"), f"{int(np.sum(score > score[i])) + 1} / {fmt_int(len(score))}",
                      help=tx("1 = die auffälligste Zeile."))
            if key == pl.MODEL_DBSCAN:
                st.markdown(tx("Cluster: **{v}**", v=tx("Rauschen") if labels[i] < 0 else int(labels[i])))
            devs = pl.row_deviations(p, [i], top=10)[0]
            show(plots.fig_bars([f for f, _ in devs], [z for _, z in devs], tx("Stärkste Abweichungen dieser Zeile"),
                                tx("z = Abweichung vom Durchschnitt in Standardabweichungen"), signed=True))
    code(cg.eval_code(r, cfg))


def anomaly_group_comparison(p, flag, score):
    """Schritt 8 (Anomalie-Modelle): Vergleich mit der bekannten Einteilung – das Modell kannte sie nicht."""
    r = p.recipe
    g = md_text(r["group_col"])
    st.subheader(tx("Vergleich mit der bekannten Einteilung: {g}", g=g))
    cmp = cached("_acmp", ss.model_token, lambda: pl.compare_anomalies(flag, score, p.groups))
    if cmp is None:
        st.info(tx("Keine Zeile hat einen Wert in '{g}'.", g=g))
        return
    st.caption(tx("Das Modell hat '{g}' nie gesehen. Gezeigt wird, wie viele Zeilen jeder Gruppe als auffällig gelten – "
                  "bei zwei Gruppen zählt die seltenere als 'positiv' (Anomalien sind selten).", g=g))
    if cmp["binary"]:
        pos = md_text(cmp["positive"])
        m = st.columns(5)
        m[0].metric("Precision", "–" if np.isnan(cmp["precision"]) else pl.share_text(cmp["precision"]),
                    help=tx("Anteil der auffälligen Zeilen, die zu '{pos}' gehören.", pos=pos))
        m[1].metric("Recall", pl.share_text(cmp["recall"]),
                    help=tx("Anteil der '{pos}'-Zeilen, die als auffällig gemeldet werden.", pos=pos))
        m[2].metric(tx("Anteil '{pos}' insgesamt", pos=pos), pl.share_text(cmp["base_rate"]),
                    help=tx("So hoch wäre die Precision bei zufällig ausgewählten Zeilen."))
        m[3].metric("ROC-AUC", f"{cmp['roc_auc']:.3f}",
                    help=tx("Wie gut trennt der Score die Gruppen – unabhängig von der Schwelle: 0.5 = Zufall, 1 = "
                            "perfekt."))
        m[4].metric("Average Precision", f"{cmp['ap']:.3f}",
                    help=tx("Fläche unter der Precision-Recall-Kurve – bei seltenen Gruppen aussagekräftiger als ROC-AUC; "
                            "Zufall = Anteil '{pos}' insgesamt.", pos=pos))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_anomaly_groups(cmp["table"], cmp["share"], r["group_col"]))
    with c2:
        table = cmp["table"].copy()
        table["Anteil auffällig"] = table["Anteil auffällig"].map(pl.share_text)
        st.dataframe(trc(table), hide_index=True)
        if cmp["binary"]:
            st.markdown(tx("**Treffer:** {tp} auffällig und '{pos}' · **Fehlalarme:** {fp} auffällig, aber nicht '{pos}' "
                           "· **übersehen:** {fn} '{pos}'-Zeilen ohne Meldung.", pos=md_text(cmp["positive"]),
                           tp=fmt_int(cmp["tp"]), fp=fmt_int(cmp["fp"]), fn=fmt_int(cmp["fn"])))


def page_profiles():
    if "model" in ss and pl.is_anomaly(ss.model_cfg.model):
        page_anomalies()
        return
    st.header(tx("8 · Cluster-Auswertung – Profile, Darstellung, Vergleich"))
    if not ready("model", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    p, model, info, cfg = ss.prep, ss.model, ss.train_info, ss.model_cfg
    r = p.recipe
    k, labels = info["k"], model.labels_
    token = ss.model_token
    prof = cached("_profiles", token, lambda: pl.cluster_profiles(p, labels, k))
    sizes = prof["sizes"]
    if ss.get("_names_for") != token:  # neues Modell → Namen zurücksetzen
        editor_reset("ed_names", pd.DataFrame({"Cluster": range(k), "Zeilen": sizes, "Name": [""] * k}))
        ss._names_for = token
    st.caption(tx("K-Means mit K = {k} auf {n} Zeilen · Silhouette {s} · erklärte Streuung {r2}",
                  k=k, n=fmt_int(info["n_rows"]), s=fmt_score(info["silhouette"]),
                  r2=f"{info['r2']:.0%}" if np.isfinite(info["r2"]) else "–"))

    st.subheader(tx("Größe und Namen der Cluster"))
    c1, c2 = st.columns([2, 3])
    with c2:
        st.markdown(tx("**Cluster benennen:** K-Means vergibt nur Nummern (0 … K−1, die Reihenfolge ist willkürlich). Ein "
                       "Name aus dem Profil unten macht Ergebnisse lesbar – er wird mit dem Modell gespeichert und in "
                       "Schritt 10 ausgegeben."))
        edited = editor("ed_names", None, hide_index=True, disabled=["Cluster", "Zeilen"],
                        column_config={"Cluster": st.column_config.NumberColumn(tx("Cluster"), width="small"),
                                       "Zeilen": st.column_config.NumberColumn(tx("Zeilen"), width="small"),
                                       "Name": st.column_config.TextColumn(tx("Name"), max_chars=80, width="large")})
    names = pl.clean_names(edited["Name"].tolist() if "Name" in edited.columns else [], k)
    with c1:
        show(plots.fig_sizes(sizes, names))

    st.subheader(tx("Cluster-Profile – was macht jeden Cluster aus?"))
    desc = pl.describe_clusters(prof["z"])
    st.markdown("\n".join(f"- {describe_line(d, sizes, names)}" for d in desc))
    st.caption(tx("z = Abweichung des Cluster-Mittelwerts vom Durchschnitt aller Zeilen in Standardabweichungen (σ): +1 = "
                  "eine Standardabweichung darüber. Rot = über, Blau = unter dem Durchschnitt. Dummies: Anteil der "
                  "Kategorie im Cluster."))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_profile(prof["z"], names))
    with c2:
        means = prof["means"].copy()
        means.columns = [tx("Alle")] + [pl.cluster_label(j, names) for j in range(k)]
        st.markdown(tx("**Mittelwerte je Cluster in Originaleinheiten** (Dummies: Anteil 0 … 1)"))
        st.dataframe(means.round(3), height=min(35 * (len(means) + 1) + 3, 640))

    st.subheader(tx("Welche Features trennen die Cluster?"))
    sep = cached("_sep", token, lambda: pl.feature_separation(p, labels, k))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_bars(sep["Feature"], sep["R²"], tx("Trennkraft je Feature (R²)"),
                            tx("Anteil der Streuung des Features, den die Cluster erklären (0 … 1)")))
    with c2:
        st.dataframe(trc(sep.round(3)), hide_index=True, height=min(35 * (len(sep) + 1) + 3, 450))
        st.caption(tx("R² nahe 1: Das Feature unterscheidet sich fast nur zwischen den Clustern – es prägt die "
                      "Einteilung. Nahe 0: innerhalb der Cluster genauso verschieden wie insgesamt. Dummies einer "
                      "Kategorie zählen zusammen."))

    st.subheader(tx("Darstellung"))
    t1, t2 = st.tabs([tx("2D-Karte (PCA)"), tx("Zwei Features")])
    with t1:
        proj = cached("_pca", token, lambda: pl.pca_projection(p, model, seed=cfg.seed))
        options = ["cluster"] + (["group"] if p.groups is not None else [])
        color = rd(tx("Farbe"), options, "w_pca_color", horizontal=True,
                   format_func=lambda v: tx("Cluster") if v == "cluster"
                   else tx("bekannte Einteilung: {g}", g=r.get("group_col")))
        groups = p.groups.iloc[proj["rows"]].to_numpy() if color == "group" and p.groups is not None else None
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_pca(proj, names, groups, r.get("group_col") or ""))
        with c2:
            st.markdown(tx("**PCA nur für das Bild:** Die Daten haben {p} Dimensionen (Spalten) – für die Grafik werden "
                           "sie auf die 2 Richtungen mit der größten Streuung projiziert (Hauptkomponenten PC1, PC2). "
                           "K-Means selbst rechnet mit allen Spalten; Cluster, die sich hier überlappen, können im vollen "
                           "Raum getrennt sein. Zusammen zeigen PC1 und PC2 {pct} der Streuung.",
                           p=len(r["feature_columns"]), pct=f"{sum(proj['explained']):.0%}"))
            load = proj["loadings"]
            top = {pc: load[pc].abs().sort_values(ascending=False).index[:5] for pc in load.columns}
            rows = [{"PC": pc, "Feature": f, "Ladung": round(float(load.at[f, pc]), 3)} for pc, idx in top.items()
                    for f in idx]
            st.dataframe(trc(pd.DataFrame(rows)), hide_index=True)
            st.caption(tx("Ladung = Gewicht eines Features in der Hauptkomponente (die 5 größten je PC)."))
    with t2:
        num = [c for c in r["numeric_features"] if c in ss.df_viz.columns]
        if len(num) < 2:
            st.info(tx("Mindestens 2 Zahlen-Features nötig."))
        else:
            course = [c for c in ("Room.Board", "Grad.Rate") if c in num]
            best = [f for f in sep["Feature"] if f in num]
            dx = course[0] if len(course) == 2 else best[0]
            dy = course[1] if len(course) == 2 else next(f for f in best + num if f != dx)
            c1, c2, _ = st.columns([1, 1, 2])
            with c1:
                x = sb("x", num, "w_cs_x", default=dx)
            with c2:
                y = sb("y", [c for c in num if c != x], "w_cs_y", default=dy)
            if x and y:
                c1, _ = st.columns([3, 2])
                with c1:
                    with chart_guard():
                        show(plots.fig_cluster_scatter(ss.df_viz.loc[p.index], x, y, labels, names))
                st.caption(tx("Wie im Kurs (Grad.Rate gegen Room.Board, Farbe = Private) – nur mit den gefundenen "
                              "Clustern als Farbe. Zwei Features zeigen nur einen Ausschnitt: Die Cluster entstehen aus "
                              "allen Features."))

    if p.groups is not None:
        group_comparison(p, labels, k, names)
    if p.names is not None:
        st.subheader(tx("Typische Vertreter je Cluster"))
        st.caption(tx("Die Zeilen am nächsten am Zentrum ihres Clusters – daran lässt sich ein Cluster oft am schnellsten "
                      "benennen. Randfälle liegen am weitesten vom Zentrum entfernt."))
        dist = ss.history["dist"]
        rows = []
        for j in range(k):
            idx = np.where(labels == j)[0]
            order = idx[np.argsort(dist[idx], kind="stable")]
            nm = p.names.fillna("–").to_numpy()
            rows.append({"Cluster": pl.cluster_label(j, names), "Zeilen": len(idx),
                         "typische Vertreter": ", ".join(map(str, nm[order[:5]])),
                         "Randfälle (weit vom Zentrum)": ", ".join(map(str, nm[order[::-1][:3]]))})
        st.dataframe(trc(pd.DataFrame(rows)), hide_index=True)

    st.subheader(tx("Einzelfall prüfen"))
    st.caption(tx("Zufällige Zeile ziehen: Zu welchem Cluster gehört sie, und wie weit ist sie von allen Zentren "
                  "entfernt?"))
    if st.button(tx("Zufällige Zeile ziehen"), icon=":material/casino:"):
        ss.case_idx = int(np.random.default_rng().integers(len(p.index)))
    if "case_idx" in ss and ss.case_idx < len(p.index):
        i = ss.case_idx
        show_cols = [c for c in [r.get("name_col"), *r["features"], r.get("group_col")] if c]
        row = ss.df_viz.loc[p.index[i], list(dict.fromkeys(show_cols))]
        D = model.transform(p.X[i:i + 1])[0]
        j = int(labels[i])
        c1, c2 = st.columns([2, 3])
        c1.dataframe(row.map(lambda v: "–" if pd.isna(v) else pl.format_value(v)).rename(tx("Wert")).to_frame(),
                     height=420)
        with c2:
            m1, m2 = st.columns(2)
            m1.metric(tx("Cluster"), pl.cluster_label(j, names))
            m2.metric(tx("Abstand zum Zentrum"), f"{D[j]:.3f}",
                      help=tx("Euklidischer Abstand in den skalierten Features. Zum Vergleich: Median im Cluster {m}.",
                              m=f"{info['dist_median'][j]:.3f}"))
            if p.groups is not None:
                g = p.groups.iloc[i]
                st.markdown(tx("Bekannte Einteilung {g}: **{v}**", g=md_text(r["group_col"]),
                               v="–" if pd.isna(g) else md_text(g)))
            show(plots.fig_bars([pl.cluster_label(c, names) for c in range(k)], D, tx("Abstand zu allen Zentren"),
                                tx("Abstand (skalierte Features) – der kleinste bestimmt den Cluster"),
                                colors=plots.cluster_colors(k)))
    code(cg.eval_code(r, cfg))


def course_comparison(cmp) -> str:
    """Text zum Vergleich wie im Kurs: Cluster-Nummer = Gruppen-Nummer (sortiert) gegen die Zuordnung je Cluster."""
    classes = [str(v) for v in cmp["classes"]]
    text = tx("**Wie im Kurs** (Cluster-Nummer direkt mit der Gruppen-Nummer {codes} verglichen): {raw:.1%} – nach der "
              "Zuordnung {acc:.1%}.", codes=md_text(", ".join(f"{v} = {i}" for i, v in enumerate(classes))),
              raw=cmp["raw_accuracy"], acc=cmp["accuracy"])
    targets = [str(cmp["mapping"][j]) for j in range(len(classes))]
    if len(set(targets)) < len(targets):  # zwei Cluster → dieselbe Gruppe: mehr als eine Umnummerierung
        return text + " " + tx("Hier gehören mehrere Cluster zur selben Gruppe – die Zuordnung ist deshalb mehr als "
                               "eine Umnummerierung.")
    if cmp["accuracy"] - cmp["raw_accuracy"] < 0.0005:
        return text + " " + tx("Die Nummern passen hier schon zu den Gruppen.")
    return text + " " + tx("Der Unterschied liegt nur an den willkürlichen Nummern: K-Means weiß nicht, dass Cluster 1 "
                           "'{v}' heißen soll.", v=md_text(classes[1]))


def group_comparison(p, labels, k, names):
    """Schritt 8: Vergleich der Cluster mit der bekannten Einteilung (im Kurs: Private) – K-Means kannte sie nicht."""
    r = p.recipe
    g = md_text(r["group_col"])
    st.subheader(tx("Vergleich mit der bekannten Einteilung: {g}", g=g))
    cmp = cached("_cmp", ss.model_token, lambda: pl.compare_groups(labels, p.groups, k))
    if cmp is None:
        st.info(tx("Keine Zeile hat einen Wert in '{g}'.", g=g))
        return
    st.caption(tx("K-Means hat '{g}' nie gesehen. Weil die Cluster-Nummern willkürlich sind, bekommt jeder Cluster die "
                  "Gruppe, die in ihm am häufigsten ist – dann wird gezählt, wie oft das stimmt.", g=g))
    m = st.columns(4)
    m[0].metric(tx("Treffer nach Zuordnung"), f"{cmp['accuracy']:.1%}",
                help=tx("Anteil der Zeilen, deren Gruppe mit der Gruppe ihres Clusters übereinstimmt (Purity)."))
    m[1].metric(tx("naive Vorhersage"), f"{cmp['baseline']:.1%}",
                help=tx("Immer die häufigste Gruppe ('{v}') – so oft läge man auch ohne Clustering richtig.",
                        v=md_text(cmp["majority"])))
    m[2].metric("ARI", f"{cmp['ari']:.3f}",
                help=tx("Adjusted Rand Index: 0 = nicht besser als Zufall, 1 = identische Einteilung – unabhängig von "
                        "den Cluster-Nummern."))
    m[3].metric("NMI", f"{cmp['nmi']:.3f}",
                help=tx("Normalized Mutual Information: 0 = Cluster verraten nichts über die Gruppe, 1 = alles."))
    c1, c2 = st.columns([1, 1])
    with c1:
        show(plots.fig_crosstab(cmp["crosstab"], names, r["group_col"]))
    with c2:
        ct = cmp["crosstab"]
        lines = []
        for j in range(k):
            total = int(ct.loc[j].sum())
            v = cmp["mapping"][j]
            hit = int(ct.at[j, v]) if total else 0
            lines.append(tx("- {label} → **{v}** ({hit} von {total})", label=md_text(pl.cluster_label(j, names)),
                            v=md_text(v), hit=fmt_int(hit), total=fmt_int(total)))
        st.markdown(tx("**Zuordnung Cluster → Gruppe**") + "\n" + "\n".join(lines))
        if cmp["unmatched"]:
            st.warning(tx("Keine Mehrheit in einem Cluster: {v} – K-Means hat diese Gruppe(n) nicht als eigenen "
                          "Cluster gefunden.", v=md_text(", ".join(map(str, cmp["unmatched"])))))
        if cmp["accuracy"] <= cmp["baseline"] + 0.01:
            st.info(tx("Die Cluster treffen die Gruppen nicht besser als die naive Vorhersage – sie folgen einer "
                       "anderen Struktur in den Daten als '{g}'.", g=g))
        if cmp["raw_accuracy"] is not None:
            st.info(course_comparison(cmp))
    with st.expander(tx("Confusion Matrix und classification_report nach der Zuordnung")):
        c1, c2 = st.columns([2, 3])
        with c1:
            show(plots.fig_confusion(cmp["cm"], cmp["classes"]))
        c2.code(cmp["report_text"], language="text")


# ===========================================================================
# Seite 9 – Speichern & Laden
# ===========================================================================
def session_recipe():
    """Rezept des aktuellen Modells inkl. Einstellungen, Kennzahlen, Cluster-Namen und Zuordnung zur Vergleichsgruppe."""
    p = ss.prep
    if pl.is_anomaly(ss.model_cfg.model):
        h = ss.history
        cmp = (cached("_acmp", ss.model_token, lambda: pl.compare_anomalies(h["flag"], h["score"], p.groups))
               if p.groups is not None else None)
        return pl.finalize_recipe(p.recipe, ss.model_cfg, ss.train_info, None, cmp)
    k = ss.train_info["k"]
    cmp = (cached("_cmp", ss.model_token, lambda: pl.compare_groups(ss.model.labels_, p.groups, k))
           if p.groups is not None else None)
    return pl.finalize_recipe(p.recipe, ss.model_cfg, ss.train_info, current_names(), cmp)


def page_save():
    st.header(tx("9 · Modell speichern & laden"))
    st.markdown(tx("Gespeichert werden **zwei Dateien** in `models/<name>/`: das Modell (`model.skops` oder "
                   "`model.joblib` – K-Means mit den Cluster-Zentren, Isolation Forest, LOF oder DBSCAN) und "
                   "`preprocessing.json` – das Rezept aus Schritt 2 und 4 (Spalten-Aktionen, Füllwerte, "
                   "Kategorien/Dummy-Spalten, Scaler) mit den Einstellungen aus Schritt 6, den Cluster-Namen aus "
                   "Schritt 8, der Schwelle und den Kennzahlen. Ohne das Rezept lassen sich neue Daten nicht korrekt "
                   "zuordnen oder bewerten."))
    st.subheader(tx("Speichern"))
    if "model" not in ss:
        st.info(tx("Noch kein trainiertes Modell in dieser Sitzung (Schritt 6)."))
    else:
        r = ss.prep.recipe
        anomaly = pl.is_anomaly(ss.model_cfg.model)
        if ss.get("_save_name_for") != ss.get("model_token"):
            ss.w_save_name = (f"{ss.model_cfg.model}_{datetime.now():%Y%m%d_%H%M}" if anomaly else
                              f"kmeans_k{ss.train_info['k']}_{datetime.now():%Y%m%d_%H%M}")
            ss._save_name_for = ss.get("model_token")
        c1, c2 = st.columns([2, 2])
        with c1:
            name = st.text_input(tx("Name (Ordner unter models/)"), key="w_save_name")
        with c2:
            fmt = rd(tx("Format"), [".skops", ".joblib"], "w_save_fmt", horizontal=True,
                     format_func={".skops": tx(".skops (empfohlen, sicher)"),
                                  ".joblib": tx(".joblib (scikit-learn-Standard, pickle)")}.get,
                     help=tx(".skops lädt nur bekannte scikit-learn-Bausteine – kein fremder Code. .joblib (pickle) kann "
                             "beim Laden beliebigen Code ausführen: nur eigene oder vertrauenswürdige Dateien laden."))
        safe = re.sub(r"[^\w\-.]+", "_", name or "").strip("._") or "modell"
        target = MODELS_DIR / safe
        # fremdes Modell unter diesem Namen? (das eigene, gerade gespeicherte darf ohne Rückfrage ersetzt werden)
        exists = (target / "preprocessing.json").exists() and ss.get("saved_folder") != str(target)
        names = current_names()
        if anomaly:
            ti = ss.train_info
            st.caption(tx("{name} · Schwelle: Score über {thr} · im Training {pct} auffällig.",
                          name=pl.model_name(ss.model_cfg.model), thr=pl.score_text(ti["threshold"]),
                          pct=pl.share_text(ti["share"])))
        else:
            st.caption(tx("Cluster-Namen: {v}", v=md_text(", ".join(pl.cluster_label(j, names)
                                                                    for j in range(len(names)))))
                       + ("" if any(names) else tx(" – noch keine Namen (Schritt 8: Cluster benennen)")))
        overwrite = True
        if exists:
            st.warning(tx("Unter `models/{safe}` liegt schon ein Modell – Speichern ersetzt es.", safe=safe))
            overwrite = st.checkbox(tx("Ja, vorhandenes Modell überschreiben"), key=f"overwrite_{safe}")
        if st.button(tx("Speichern"), type="primary", icon=":material/save:", disabled=not overwrite):
            try:
                with st.spinner(tx("Speichere …")):
                    folder = pl.save_bundle(ss.model, session_recipe(), target, fmt)
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Speichern fehlgeschlagen: {exc}", exc=exc))
            else:
                ss.saved_folder = str(folder)
                ss.pop("_zip_cache", None)
                st.success(tx("Gespeichert: {folder}", folder=folder))
        if "saved_folder" in ss and Path(ss.saved_folder).exists():
            folder = Path(ss.saved_folder)
            stamp = (str(folder), max((f.stat().st_mtime for f in folder.iterdir()), default=0))
            zipped = cached("_zip_cache", stamp, lambda: pl.bundle_zip_bytes(folder))  # nicht bei jedem Klick neu packen
            st.download_button(tx("Modell + Rezept als ZIP"), zipped, file_name=f"{folder.name}.zip",
                               mime="application/zip", icon=":material/download:")
        src = dict(ss.get("source", {}))
        src["sample_frac"] = ss.w_frac if ss.get("w_use_sample") else None
        raw = ss.df_raw
        code_folder = f"models/{Path(ss.get('saved_folder', tx('mein_modell'))).name}"
        script = cg.full_script(src, ss.actions_fitted, {c for c in raw.columns if pl.is_numeric(raw[c])},
                                r, ss.model_cfg, code_folder, ss.w_save_fmt, new_columns=ss.get("new_cols_fitted"))
        st.download_button(tx("Komplette Pipeline als Python-Skript (für Jupyter)"), script.encode("utf-8"),
                           file_name="clustering_pipeline.py", mime="text/x-python", icon=":material/download:")
        code(cg.save_code(code_folder, ss.w_save_fmt, pl.MODEL_VARS.get(ss.model_cfg.model, "kmeans")))

    st.subheader(tx("Laden"))
    bundles = pl.list_bundles(MODELS_DIR)
    t1, t2 = st.tabs([tx("Aus dem Ordner models/"), tx("ZIP hochladen")])
    with t1:
        if not bundles:
            st.caption(tx("Noch keine gespeicherten Modelle."))
        else:
            labels = {b["name"]: (tx("{name} · {model} · {pct} auffällig · {n} Features · {v}",
                                     name=b["name"], model=pl.model_name(b["model"]), n=b["n_features"],
                                     pct=pl.share_text(b["share"]) if b.get("share") is not None else "?",
                                     v=b["saved_at"][:16].replace("T", " "))
                                  if pl.is_anomaly(b.get("model")) else
                                  tx("{name} · K-Means · K = {k} · {n} Features · {v}",
                                     name=b['name'], k=b.get("k") or "?", n=b["n_features"],
                                     v=b['saved_at'][:16].replace('T', ' ')))
                      for b in bundles}
            sel = sb(tx("Gespeicherte Modelle"), list(labels), "w_bundle_sel", format_func=labels.get)
            if st.button(tx("Laden"), key="btn_load_bundle"):
                try:
                    model, recipe = pl.load_bundle(MODELS_DIR / sel)
                except Exception as exc:  # noqa: BLE001
                    st.error(tx("Laden fehlgeschlagen: {exc}", exc=md_text(exc)))
                else:
                    ss.bundle = {"model": model, "recipe": recipe, "name": sel, "token": uuid.uuid4().hex}
                    ss.pop("pred_result", None)
                    st.success(tx("'{sel}' geladen → Schritt 10", sel=sel))
    with t2:
        up = st.file_uploader(tx("ZIP (model.skops/.joblib + preprocessing.json)"), type=["zip"], key="up_bundle")
        allow_pickle = False
        if up is not None and pl.zip_needs_pickle(up.getvalue()):
            st.warning(tx("Das ZIP enthält ein .joblib-Modell (pickle). Pickle kann beim Laden beliebigen Code ausführen "
                          "– nur Dateien aus vertrauenswürdiger Quelle laden (z. B. eigene)."))
            allow_pickle = st.checkbox(tx("Ich vertraue dieser Datei – .joblib laden"), key="w_zip_pickle")
        if st.button(tx("ZIP laden"), disabled=up is None):
            try:
                model, recipe = pl.load_bundle_zip(up.getvalue(), allow_pickle=allow_pickle)
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Laden fehlgeschlagen: {exc}", exc=md_text(exc)))
            else:
                ss.bundle = {"model": model, "recipe": recipe, "name": Path(up.name).stem, "token": uuid.uuid4().hex}
                ss.pop("pred_result", None)
                st.success(tx("ZIP geladen → Schritt 10"))
    if "bundle" in ss:
        b = ss.bundle
        r = b["recipe"]
        training = r.get("training") or {}
        key = pl.recipe_model(r)
        if pl.is_anomaly(key):
            share = (r.get("train_info") or {}).get("share")
            st.markdown(tx("**Geladen: {name}** · {model} · im Training {pct} auffällig · {n} Features → {n2} Eingänge · "
                           "gespeichert {v}", name=md_text(b["name"]), model=pl.model_name(key),
                           pct=pl.share_text(share) if share is not None else "?", n=len(r["features"]),
                           n2=len(r["feature_columns"]), v=md_text(str(r.get("saved_at", "?")).replace("T", " "))))
        else:
            k = int(training.get("n_clusters") or len(b["model"].cluster_centers_))
            names = pl.clean_names(r.get("cluster_names"), k)
            st.markdown(tx("**Geladen: {name}** · K-Means · K = {k} · {n} Features → {n2} Eingänge · gespeichert {v}",
                           name=md_text(b['name']), k=k, n=len(r['features']), n2=len(r['feature_columns']),
                           v=md_text(str(r.get('saved_at', '?')).replace('T', ' '))))
            if any(names):
                st.markdown(tx("Cluster: ") + " · ".join(md_text(pl.cluster_label(j, names)) for j in range(k)))
        c1, c2 = st.columns(2)
        c1.markdown(tx("**Kennzahlen (Training)**"))
        c1.json(r.get("metrics", {}))
        c2.markdown(tx("**Einstellungen**"))
        c2.json(training)


# ===========================================================================
# Seite 10 – Neue Daten den Clustern zuordnen
# ===========================================================================
HOW_TO_10 = N_("""
**So werden neue Daten zugeordnet**

1. **Nichts neu fitten.** Neue Daten werden mit dem *gespeicherten Rezept* aufbereitet – gleiche Spalten-Aktionen und
   neue Spalten, gleiche Füllwerte, gleiche Dummy-Spalten in gleicher Reihenfolge, gleicher Scaler. Danach kommt jede
   Zeile zum **nächsten Cluster-Zentrum** (`kmeans.predict`) – die Zentren bleiben, wie sie sind.
2. **Gleiche Rohspalten wie im Training.** Das Modell bestimmt die Features – die Tabelle unten zeigt, welche
   CSV-Spalten nötig sind (Vorlage zum Download). Fehlen Spalten → klare Fehlermeldung. Zusätzliche Spalten werden
   ignoriert.
3. **Abweichungen werden behandelt und gemeldet:** Lücken → Füllwerte aus dem Training; unbekannte Kategorien → alle
   Dummies 0; Werte außerhalb des Trainingsbereichs → Hinweis. Zeilen, die weiter vom nächsten Zentrum entfernt sind
   als 99 % der Trainingszeilen dieses Clusters, bekommen den Hinweis *weit weg von allen Clustern*.
4. **Interpretation und Fazit:** Verteilung auf die Cluster im Vergleich zum Training, Ausreißer und knappe
   Zuordnungen – und wenn die neuen Daten die bekannte Einteilung enthalten (z. B. Private), der Vergleich mit der
   Zuordnung Cluster → Gruppe aus dem Training.
""")
HOW_TO_10_ANOMALY = N_("""
**So werden neue Daten bewertet**

1. **Nichts neu fitten.** Neue Daten werden mit dem *gespeicherten Rezept* aufbereitet (gleiche Spalten-Aktionen und
   neue Spalten, gleiche Füllwerte, gleiche Dummy-Spalten, gleicher Scaler) und mit dem Modell aus dem Training bewertet
   – mit **derselben Schwelle** wie im Training.
2. **Gleiche Rohspalten wie im Training.** Die Tabelle unten zeigt, welche CSV-Spalten nötig sind (Vorlage zum
   Download). Fehlen Spalten → klare Fehlermeldung. Zusätzliche Spalten werden ignoriert.
3. **Ergebnis je Zeile:** *Anomalie* (1 = Score über der Schwelle), *Anomalie-Score* (höher = auffälliger) und bei
   DBSCAN der *Cluster* des nächsten Kernpunkts (−1 = Rauschen). *Hinweis* nennt ergänzte Lücken, unbekannte Kategorien
   und Werte außerhalb des Trainingsbereichs.
4. **Fazit:** Anteil auffälliger Zeilen im Vergleich zum Training, die auffälligsten Zeilen – und wenn die neuen Daten
   die bekannte Einteilung enthalten, wie gut die Anomalien sie treffen.

**Zeitreihen:** In Schritt 4–6 auf einem ruhigen Zeitraum trainieren und hier spätere Daten bewerten. Häufen sich die
Anomalien in einem Zeitraum, ist das eher ein neues Regime als lauter Einzelfälle.
""")
LEVEL_ICON = {pl.LEVEL_OK: ":green[:material/check_circle:]", pl.LEVEL_WARN: ":orange[:material/warning:]",
              pl.LEVEL_BAD: ":red[:material/error:]"}


def render_fazit(fz, meaning_title=None):
    st.subheader(tx("Fazit"))
    with st.container(border=True):
        st.markdown(tx("**Datenqualität: {v} {level}**", v=LEVEL_ICON[fz['level']], level=tx(fz['level'])))
        lines = []
        for q in fz["quality"]:
            lines.append(f"- {LEVEL_ICON[q['level']]} {q['text']}")
            lines += [f"    - {d}" for d in q["details"][:12]]
        st.markdown("\n".join(lines))
        if fz["meaning"]:
            st.markdown(meaning_title or tx("**Was die Zuordnung aussagt**"))
            st.markdown("\n".join(f"- {t}" for t in fz["meaning"]))
        if fz["advice"]:
            st.markdown(tx("**Empfehlung**"))
            st.markdown("\n".join(f"- {t}" for t in fz["advice"]))


def required_section(recipe):
    """Schritt 10: benötigte Rohspalten und die CSV-Vorlage."""
    st.subheader(tx("Benötigte Spalten"))
    st.dataframe(pl.required_columns(recipe), hide_index=True)
    st.download_button(tx("CSV-Vorlage (nur Kopfzeile)"), pl.template_csv(recipe), tx("vorlage_neue_daten.csv"),
                       mime="text/csv", icon=":material/download:")


def single_case_form(recipe, src, submit_label):
    """Schritt 10: ein Datensatz über ein Formular (vorbelegt mit Median bzw. häufigstem Wert aus dem Training).
    Rückgabe: (abgeschickt?, Werte)."""
    st.caption(tx("Werte eingeben (vorbelegt mit Median bzw. häufigstem Wert aus dem Training). Eingegeben wird "
               "die Form nach Schritt 2."))
    with st.form("single_case"):
        values = {}
        grid = st.columns(3)
        for i, f in enumerate(recipe["features"]):
            with grid[i % 3]:
                fb = recipe["fallback_fill"].get(f)
                if f in recipe["numeric_features"]:
                    values[f] = st.number_input(md_text(f), value=float(fb), format="%g", key=f"form_{src}_{f}")
                else:
                    cats = recipe["categories"][f]
                    values[f] = st.selectbox(md_text(f), cats, index=cats.index(fb) if fb in cats else 0,
                                             key=f"form_{src}_{f}")
        submitted = st.form_submit_button(submit_label, type="primary")
    return submitted, values


def assign_anomalies(model, recipe, model_id, key, src):
    """Schritt 10 für Isolation Forest, LOF und DBSCAN: neue Daten bewerten (score_new) – nichts wird neu gelernt."""
    with st.expander(tx("So funktioniert Schritt 10"), icon=":material/info:"):
        st.markdown(tx(HOW_TO_10_ANOMALY))
    required_section(recipe)
    ti = recipe.get("train_info") or {}
    thr = ti.get("threshold", 1.0 if key == pl.MODEL_DBSCAN else float("nan"))
    mode = rd(tx("Eingabe"), [N_("CSV-Datei"), N_("Einzelner Datensatz (Formular)")], "w_pred_mode", horizontal=True,
              format_func=tx)
    if mode == "CSV-Datei":
        source, name = csv_source("pred")
        csv_options()
        if st.button(tx("Neue Daten bewerten"), type="primary", disabled=source is None):
            try:
                with st.spinner(tx("Bereite neue Daten auf und bewerte …")):
                    df_new, _ = read_source(source)
                    res, rep, cmp = pl.score_new(model, recipe, df_new)
            except pl.MissingColumnsError as exc:
                st.error(tx("{exc}. Die CSV braucht dieselben Rohspalten wie die Trainingsdaten (siehe Tabelle "
                            "oben / CSV-Vorlage).",
                            exc=md_text(exc)))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=md_text(exc)))
            else:
                ss.pred_result = {"res": res, "rep": rep, "cmp": cmp, "name": name, "model": model_id,
                                  "cols": list(df_new.columns)}
        pr = ss.get("pred_result")
        if pr and pr["model"] == model_id:
            res, rep, cmp = pr["res"], pr["rep"], pr["cmp"]
            fz = pl.assess_scoring(recipe, res, rep, cmp)
            st.success(tx("{n} Zeilen bewertet ({name})", n=fmt_int(len(res)), name=md_text(pr["name"])))
            flag = np.asarray(rep["flag"], dtype=bool)
            m = st.columns(3)
            m[0].metric(tx("auffällige Zeilen"), f"{fmt_int(flag.sum())} · {pl.share_text(flag.mean())}")
            m[1].metric(tx("im Training auffällig"), pl.share_text(ti["share"]) if ti.get("share") is not None else "–")
            m[2].metric(tx("Schwelle (Score)"), pl.score_text(thr))
            c1, c2 = st.columns([3, 2])
            with c1:
                with chart_guard():
                    show(plots.fig_scores(rep["score"], thr, None, title=tx("Anomalie-Score der neuen Daten")))
            c2.markdown(tx("**Lesart:** `Anomalie` = 1, wenn der Score über der Schwelle aus dem Training liegt; "
                           "`Anomalie-Score` – höher = auffälliger; bei DBSCAN `Cluster` = Cluster des nächsten "
                           "Kernpunkts (−1 = Rauschen). `Hinweis` nennt ergänzte Lücken, unbekannte Kategorien und Werte "
                           "außerhalb des Trainingsbereichs."))
            if cmp is not None:
                st.subheader(tx("'{g}' ist enthalten → Vergleich mit den Anomalien ({n} Zeilen)",
                                g=md_text(recipe.get("group_col")), n=fmt_int(cmp["n"])))
                if cmp["binary"]:
                    m = st.columns(4)
                    m[0].metric("Precision", "–" if np.isnan(cmp["precision"]) else pl.share_text(cmp["precision"]))
                    m[1].metric("Recall", pl.share_text(cmp["recall"]))
                    m[2].metric("ROC-AUC", f"{cmp['roc_auc']:.3f}")
                    m[3].metric("Average Precision", f"{cmp['ap']:.3f}")
                c1, _ = st.columns([3, 2])
                with c1:
                    show(plots.fig_anomaly_groups(cmp["table"], cmp["share"], recipe.get("group_col") or ""))
            st.subheader(tx("Ergebnis"))
            replaced = replaced_columns(pr.get("cols", []), recipe)  # je Sprache neu (übersetzte Namen)
            if replaced:
                st.info(tx("Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis "
                           "werden sie ersetzt: {cols}", cols=md_text(", ".join(map(str, replaced)))))
            init("w_pred_sort", True)
            by_score = st.checkbox(tx("Nach Score sortieren (auffälligste zuerst)"), key="w_pred_sort")
            view = result_view(res)
            shown = view.iloc[np.argsort(-np.asarray(rep["score"], dtype=float), kind="stable")] if by_score else view
            lead = [c for c in view.columns if c in [tx(n) for n in RESULT_COLUMNS]]
            st.dataframe(shown[lead + [c for c in view.columns if c not in lead]].head(1000))
            st.download_button(tx("Ergebnis als CSV"), view.to_csv(index=False).encode("utf-8"),
                               file_name=tx("anomalien_{stem}.csv", stem=Path(pr['name']).stem), mime="text/csv",
                               icon=":material/download:")
            render_fazit(fz, tx("**Was die Bewertung aussagt**"))
    else:
        submitted, values = single_case_form(recipe, src, tx("Neue Daten bewerten"))
        if submitted:
            try:
                res, rep, _ = pl.score_new(model, recipe, pd.DataFrame([values]))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=md_text(exc)))
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric(tx("Anomalie"), tx("ja") if int(res[pl.RESULT_ANOMALY].iloc[0]) else tx("nein"))
                m2.metric(tx("Anomalie-Score"), pl.score_text(rep["score"][0]),
                          help=tx("Schwelle: {thr} – auffällig ist ein Score darüber.", thr=pl.score_text(thr)))
                if key == pl.MODEL_DBSCAN:
                    label = int(res[pl.RESULT_CLUSTER].iloc[0])
                    m3.metric(tx("Cluster"), tx("Rauschen") if label < 0 else str(label))
                note = res[pl.RESULT_NOTE].iloc[0]
                if note:
                    st.caption(tx("Hinweis: {v}", v="; ".join(tx(part) for part in note.split("; "))))
    code(cg.predict_code("models/<name>", key), tx("Python-Code (Bewertung ohne App)"))


def page_assign():
    head = st.empty()  # Überschrift je nach Modell: zuordnen (K-Means) oder bewerten (Anomalien)
    options = {}
    if "model" in ss:
        options["session"] = tx("Aktuelles Modell (diese Sitzung)")
    if "bundle" in ss:
        options["bundle"] = tx("Geladenes Modell: {name}", name=md_text(ss.bundle['name']))
    if not options:
        head.header(tx("10 · Neue Daten den Clustern zuordnen"))
        st.info(tx("Kein Modell vorhanden – in Schritt 6 trainieren oder in Schritt 9 laden."))
        st.markdown(tx(HOW_TO_10))
        return
    src = rd(tx("Modell"), list(options), "w_pred_model", format_func=options.get, horizontal=True)
    if src == "session":
        model, recipe = ss.model, session_recipe()
        model_id = f"session-{ss.get('model_token')}"
    else:
        model, recipe = ss.bundle["model"], ss.bundle["recipe"]
        model_id = f"bundle-{ss.bundle.get('token')}"
    key = pl.recipe_model(recipe)
    if pl.is_anomaly(key):
        head.header(tx("10 · Neue Daten bewerten – {name}", name=pl.model_name(key)))
        assign_anomalies(model, recipe, model_id, key, src)
        return
    head.header(tx("10 · Neue Daten den Clustern zuordnen"))
    k = len(model.cluster_centers_)
    names = pl.clean_names(recipe.get("cluster_names"), k)
    with st.expander(tx("So funktioniert Schritt 10"), icon=":material/info:"):
        st.markdown(tx(HOW_TO_10))
    required_section(recipe)

    mode = rd(tx("Eingabe"), [N_("CSV-Datei"), N_("Einzelner Datensatz (Formular)")], "w_pred_mode", horizontal=True,
              format_func=tx)
    if mode == "CSV-Datei":
        source, name = csv_source("pred")
        csv_options()
        if st.button(tx("Cluster zuordnen"), type="primary", disabled=source is None):
            try:
                with st.spinner(tx("Bereite neue Daten auf und ordne zu …")):
                    df_new, _ = read_source(source)
                    res, rep, cmp = pl.assign_new(model, recipe, df_new)
            except pl.MissingColumnsError as exc:
                st.error(tx("{exc}. Die CSV braucht dieselben Rohspalten wie die Trainingsdaten (siehe Tabelle "
                            "oben / CSV-Vorlage).",
                            exc=md_text(exc)))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=md_text(exc)))
            else:
                ss.pred_result = {"res": res, "rep": rep, "cmp": cmp, "name": name, "model": model_id,
                                  "cols": list(df_new.columns)}
        pr = ss.get("pred_result")
        if pr and pr["model"] == model_id:
            res, rep, cmp = pr["res"], pr["rep"], pr["cmp"]
            fz = pl.assess_assignment(recipe, res, rep, cmp)
            st.success(tx("{n} Zeilen zugeordnet ({name})", n=fmt_int(len(res)), name=md_text(pr['name'])))
            if fz["level"] != pl.LEVEL_OK or int(np.sum(rep.get("row_far", []))):
                st.caption(tx("{v} Hinweise zur Datenqualität und Einordnung → **Fazit** unten",
                              v=LEVEL_ICON[fz['level']]))
            share_new = np.bincount(rep["labels"], minlength=k) / max(len(res), 1)
            share_train = recipe.get("cluster_share") or None
            c1, c2 = st.columns([3, 2])
            with c1:
                show(plots.fig_cluster_shares(share_new, share_train, names))
            c2.markdown(tx("**Lesart:** `Cluster` = Nummer des nächsten Zentrums, `Abstand` = euklidischer Abstand dazu "
                           "(skalierte Features – klein = typisch für den Cluster). `Hinweis` nennt ergänzte Lücken, "
                           "unbekannte Kategorien, Werte außerhalb des Trainingsbereichs, Zeilen weit weg von allen "
                           "Clustern und knappe Zuordnungen zwischen zwei Clustern."))
            if cmp is not None:
                st.subheader(tx("'{g}' ist enthalten → Vergleich mit der Zuordnung aus dem Training ({n} Zeilen)",
                                g=md_text(recipe.get("group_col")), n=fmt_int(cmp["n"])))
                m = st.columns(4)
                m[0].metric(tx("Treffer nach Zuordnung"), f"{cmp['accuracy']:.1%}")
                m[1].metric(tx("naive Vorhersage"), f"{cmp['baseline']:.1%}")
                m[2].metric("ARI", f"{cmp['ari']:.3f}")
                m[3].metric("NMI", f"{cmp['nmi']:.3f}")
                c1, _ = st.columns([1, 1])
                with c1:
                    show(plots.fig_crosstab(cmp["crosstab"], names, recipe.get("group_col") or ""))
            st.subheader(tx("Ergebnis"))
            replaced = replaced_columns(pr.get("cols", []), recipe)  # je Sprache neu (übersetzte Namen)
            if replaced:
                st.info(tx("Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis "
                           "werden sie ersetzt: {cols}", cols=md_text(", ".join(map(str, replaced)))))
            view = result_view(res)
            lead = [c for c in view.columns if c in [tx(n) for n in RESULT_COLUMNS]]
            st.dataframe(view[lead + [c for c in view.columns if c not in lead]].head(1000))
            st.download_button(tx("Ergebnis als CSV"), view.to_csv(index=False).encode("utf-8"),
                               file_name=tx("cluster_{stem}.csv", stem=Path(pr['name']).stem), mime="text/csv",
                               icon=":material/download:")
            render_fazit(fz)
    else:
        submitted, values = single_case_form(recipe, src, tx("Cluster zuordnen"))
        if submitted:
            try:
                res, rep, _ = pl.assign_new(model, recipe, pd.DataFrame([values]))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=md_text(exc)))
            else:
                j = int(res[pl.RESULT_CLUSTER].iloc[0])
                X, _, _ = pl.prepare_new_data(pd.DataFrame([values]), recipe)
                D = model.transform(X)[0]
                m1, m2, _ = st.columns([1.4, 1, 2])
                m1.metric(tx("Cluster"), pl.cluster_label(j, names))
                m2.metric(tx("Abstand zum Zentrum"), f"{D[j]:.3f}")
                note = res[pl.RESULT_NOTE].iloc[0]
                if note:
                    st.caption(tx("Hinweis: {v}", v="; ".join(tx(part) for part in note.split("; "))))
                c1, _ = st.columns([3, 2])
                with c1:
                    show(plots.fig_bars([pl.cluster_label(c, names) for c in range(k)], D,
                                        tx("Abstand zu allen Zentren"),
                                        tx("Abstand (skalierte Features) – der kleinste bestimmt den Cluster"),
                                        colors=plots.cluster_colors(k)))
    code(cg.predict_code("models/<name>"), tx("Python-Code (Zuordnung ohne App)"))


# ===========================================================================
# Seitenleiste & Navigation
# ===========================================================================
PAGES = {
    N_("1 · Daten & Bibliotheken"): page_data,
    N_("2 · Aufbereitung 1 – Spalten"): page_prep1,
    N_("3 · Analyse & Visualisierung"): page_viz,
    N_("4 · Aufbereitung 2 – Clustering-Daten"): page_prep2,
    N_("5 · Daten speichern"): page_export,
    N_("6 · Modell & Training"): page_train,
    N_("7 · Wahl von K bzw. Schwelle"): page_k,
    N_("8 · Auswertung"): page_profiles,
    N_("9 · Modell speichern & laden"): page_save,
    N_("10 · Neue Daten zuordnen / bewerten"): page_assign,
}

with st.sidebar:
    st.markdown("### :material/bubble_chart: DATA-Analysis-Clustering")
    init("w_lang", i18n.get_lang())
    st.segmented_control("Sprache · Language · Język", list(i18n.LANGUAGES), key="w_lang", required=True,
                         format_func=i18n.LANGUAGES.get, label_visibility="collapsed")
    page = st.radio(tx("Schritt"), list(PAGES), key="nav", label_visibility="collapsed", format_func=tx)
    st.divider()
    st.markdown(tx("**Stichprobe** – für langsame Rechner"))
    init("w_use_sample", False)
    init("w_frac", 0.10)
    use_sample = st.checkbox(tx("Nur Stichprobe verwenden (df.sample)"), key="w_use_sample",
                             help=tx("Standard: alle Daten (100 %). Gilt ab dem Laden für alle Schritte."))
    st.number_input("frac", 0.01, 1.0, step=0.05, format="%.2f", key="w_frac", disabled=not use_sample)
    sync_sample()
    st.divider()
    status_box = st.container()  # wird nach der Seite gefüllt → zeigt den aktuellen Stand
    st.divider()

PAGES[page]()

with st.sidebar:
    with status_box:
        def status(ok, text):
            st.markdown(f"{':green[:material/check_circle:]' if ok else ':gray[:material/radio_button_unchecked:]'} {text}")

        s = ss.get("source")
        status("df_raw" in ss, tx("Daten: {name} ({n} Zeilen)", name=md_text(s['name']), n=fmt_int(len(ss.df_raw)))
               if s and "df_raw" in ss else tx("Daten laden"))
        status("df_viz" in ss, tx("Aufbereitung 1 übernommen"))
        status("prep" in ss, tx("Clustering-Daten: X {shape}", shape=ss.prep.X.shape) if "prep" in ss
               else tx("Clustering-Daten vorbereitet"))
        if "model" in ss and pl.is_anomaly(ss.model_cfg.model):
            status(True, tx("{name} trainiert ({pct} auffällig)", name=pl.model_name(ss.model_cfg.model),
                            pct=pl.share_text(ss.train_info["share"])))
        else:
            status("model" in ss, tx("K-Means trainiert (K = {k})", k=ss.train_info["k"]) if "model" in ss
                   else tx("Modell trainiert"))
        status("bundle" in ss, tx("Geladen: {name}", name=md_text(ss.bundle['name'])) if "bundle" in ss
               else tx("Modell geladen (Schritt 9)"))
    with st.expander(tx("Konfiguration speichern / laden"), icon=":material/settings:"):
        st.caption(tx("Alle Einstellungen (Schritt 2 inkl. neuer Spalten, 4, 6) als JSON – für geänderte Daten mit "
                   "gleichen Spalten."))
        st.download_button(tx("Exportieren (JSON)"), export_config(), file_name="clustering_config.json",
                           mime="application/json")
        up_cfg = st.file_uploader(tx("Importieren"), type=["json"], key="up_cfg")
        st.button(tx("Konfiguration übernehmen"), disabled=up_cfg is None, on_click=on_import_config)
        msg = ss.pop("_cfg_msg", None)
        if msg:
            (st.error if msg[0] == "error" else st.success)(msg[1])
