"""
app.py – DATA-Analysis-NLP: Datenanalyse und Textklassifikation (scikit-learn) für CSV-Daten mit Texten
======================================================================================================

Start:   streamlit run app.py

Schritte (Navigation links)
    1  Daten laden & Bibliotheken          6  Modell & Training (3 Modelle, Vergleich)
    2  Aufbereitung 1 – Spalten, neue Sp.  7  Training vs. Validierung
    3  Analyse & Visualisierung            8  Evaluation & wichtige Wörter
    4  Aufbereitung 2 – Text → Zahlen      9  Modell speichern & laden
    5  Daten speichern (CSV)              10  Neue Texte einordnen + Fazit
    Stichprobe für langsame Rechner: Option in der Seitenleiste

Modelle: Naive Bayes (MultinomialNB, wie im Kurs) · Logistische Regression · Lineare SVM (LinearSVC)
Text → Zahlen: CountVectorizer (Bag of Words, wie im Kurs) oder TfidfVectorizer

Die Rechenlogik steckt in pipeline.py, die Diagramme in plots.py, der angezeigte
Python-Code in codegen.py. Diese Datei enthält nur die Oberfläche.
"""
import matplotlib

matplotlib.use("Agg")  # Streamlit rechnet in einem Thread → kein GUI-Backend

import functools
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

st.set_page_config(page_title="DATA-Analysis-NLP", page_icon=":material/translate:", layout="wide")

import i18n  # noqa: E402
from i18n import N_, tx  # noqa: E402

# Sprache vor allen Texten festlegen: Auswahl in der Seitenleiste, Start über ?lang=…, --lang oder NLP_TEMPLATE_LANG
os.environ.setdefault("NLP_TEMPLATE_LANG", "en")  # Startsprache Englisch
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
MODELS_DIR = Path(os.environ.get("NLP_TEMPLATE_MODELS", APP_DIR / "models"))  # Speicherort für Schritt 9
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
    "viz": ["df_viz", "actions_fitted", "new_cols_fitted", "lineage", "step2_log", "viz_token"],
    "prep": ["prep", "prep_token", "prep_sig", "_export", "compare", "bow_idx"],
    "model": ["model", "history", "train_info", "model_cfg", "model_token"],
    "eval": ["eval_scores", "case_idx", "pred_result", "saved_folder", "_explain"],
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


def csv_downloads(make_frame, label, file_name, token):
    """Ergebnis-Tabelle als CSV international und als CSV für Excel (Format der gewählten Sprache). Texte aus fremden
    Quellen (Bewertungen, Nachrichten) können mit = + - @ beginnen – die Excel-Datei entschärft sie (CSV-Injection).
    Beide Dateien werden nur neu erstellt, wenn sich token (Modell, Schwelle, Ergebnis) oder die Sprache ändern."""
    def build():
        frame = make_frame()
        return pl.to_csv_bytes(frame), pl.to_csv_bytes(frame, excel=i18n.get_lang())

    intl, excel = cached("_csv_downloads", (token, file_name, i18n.get_lang()), build)
    with st.container(horizontal=True):
        st.download_button(label, intl, file_name=file_name, mime="text/csv", icon=":material/download:")
        st.download_button(tx("CSV für Excel"), excel,
                           file_name=file_name.removesuffix(".csv") + "_excel.csv", mime="text/csv",
                           icon=":material/table_view:",
                           help=tx("Trennzeichen und Dezimalzeichen wie Excel in der gewählten Sprache (Deutsch und "
                                   "Polnisch: ; und Komma, Englisch: , und Punkt), UTF-8 mit BOM. Texte, die mit "
                                   "= + - @ beginnen, bekommen ein ' davor – Excel führt sie sonst als Formel aus."))


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


RESULT_COLUMNS = (N_("Vorhersage"), N_("Sicherheit"), N_("Hinweis"))  # = pl.RESULT_* (intern deutsch)
TABLE_HEADERS = (  # Spaltennamen der Anzeige-Tabellen – hier nur für den Übersetzungskatalog, übersetzt in trc()
    N_("Spalte"), N_("Typ"), N_("Eindeutig"), N_("Fehlend"), N_("Fehlend %"), N_("Beispiel"), N_("Neue Spalte"),
    N_("Art"), N_("Definition"), N_("Aktion"), N_("Parameter"), N_("Wirkung"), N_("Anzahl"), N_("Anteil"),
    N_("Klasse"), N_("Kodiert"), N_("Train"), N_("Test"), N_("Anteil Train"), N_("Wort"), N_("Texte"), N_("IDF"),
    N_("Gewicht"), N_("Wert"), N_("Beitrag"), N_("Text"), N_("tatsächlich"), N_("vorhergesagt"), N_("Zeile"),
    N_("Training"), N_("Validierung"), N_("Modell"), N_("Warum?"), N_("Am besten für"), N_("Sekunden"),
    N_("Zeichen"), N_("Wörter"), N_("richtig"),
)


def export_columns() -> dict:
    """Namen der neuen Spalten im Export der Testvorhersagen (Schritt 8) – in der gewählten Sprache."""
    return {c: tx(c) for c in ("tatsächlich", pl.RESULT_PRED, "richtig")}


def vocab_columns() -> dict:
    """Spaltennamen des Vokabulars (Schritt 4, 5) in der gewählten Sprache."""
    return {c: tx(c) for c in ("Wort", "Texte", "Anzahl", "IDF")}


def result_view(res):
    """Ergebnis aus Schritt 10 zur Anzeige und zum Download: Spaltennamen und Hinweise in der gewählten Sprache.
    Hat die neue CSV schon eine Spalte mit dem übersetzten Namen (z. B. 'Prediction' aus einem früheren Export),
    ersetzt das Ergebnis sie – wie auf Deutsch bei 'Vorhersage' (sonst doppelte Spaltennamen)."""
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
    task, classes = recipe["task"], recipe.get("classes") or []
    made = [pl.RESULT_PRED, pl.RESULT_NOTE] + ([pl.RESULT_CONF] if task == pl.T_MULTI else [])
    made += [f"P({c})" for c in (classes[1:] if task == pl.T_BIN else classes)]
    names = set(made) | {tx(c) for c in made}
    return [c for c in columns if c in names]


THIN = "\u202f"  # schmales Leerzeichen als Tausendertrennung (eindeutig neben dem Dezimalpunkt)


def md_text(text) -> str:
    """Text aus Dateien (Texte, Spalten- und Klassennamen, Name eines hochgeladenen ZIP, Fehlermeldungen) in
    st.markdown wörtlich zeigen: Markdown-Zeichen escapen, Zeilenumbrüche entfernen – sonst könnte z. B.
    ![](http://…) ein fremdes Bild nachladen."""
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


def highlight_md(text, vectorizer, ex, limit: int = 4000) -> str:
    """Text als Markdown: die Wörter aus der Erklärung (ex = pipeline.explain_text) farbig – blau spricht für die
    vorhergesagte Klasse, orange dagegen. Alles andere wörtlich (escaped)."""
    shown = str(text or "")[:limit]
    out = []
    for piece, w in pl.highlight_spans(shown, vectorizer, ex["weights"], only=set(ex["words"]["Wort"].astype(str))):
        if not w:
            out.append(md_text(piece) if piece.strip() else " ")
        else:
            out.append(f":{'blue' if w > 0 else 'orange'}[**{md_text(piece)}**]")
    return "".join(out) + (" …" if len(str(text or "")) > limit else "")


SEPS = [",", ";", "\t", "|"]
SEP_LABELS = {",": tx("Komma  ,"), ";": tx("Semikolon  ;"), "\t": tx("Tabulator"), "|": tx("Pipe  |")}
HEADERS = {"auto": N_("automatisch erkennen"), "ja": N_("ja"), "nein": N_("nein – Spalten heißen Spalte_1, Spalte_2 …")}


def csv_options(header=True):
    detected = ss.pop("_csv_detected", None)
    if detected:  # beim Laden automatisch erkannt → vor dem Zeichnen der Widgets übernehmen
        ss.w_sep, ss.w_dec, ss.w_enc = detected
    cols = st.columns(4 if header else 3)
    with cols[0]:
        sb(tx("Trennzeichen"), SEPS, "w_sep", format_func=SEP_LABELS.get)
    with cols[1]:
        sb(tx("Dezimalzeichen"), [".", ","], "w_dec")
    with cols[2]:
        sb(tx("Encoding"), ["utf-8", "latin-1"], "w_enc")
    if header:
        with cols[3]:
            sb(tx("Kopfzeile"), list(HEADERS), "w_header", format_func=lambda v: tx(HEADERS[v]),
               help=tx("Hat die erste Zeile die Spaltennamen? SMSSpamCollection aus dem Kurs hat keine – dort heißen "
                       "die Spalten dann Spalte_1, Spalte_2 (in Schritt 2 umbenennen, z. B. label und message)."))


def csv_source(prefix):
    """Upload oder Pfad. Rückgabe: (Quelle, Name) – Upload-Bytes erst beim Laden lesen."""
    src = rd(tx("Quelle"), [N_("Datei hochladen"), N_("Pfad auf diesem Rechner")], f"w_{prefix}_src", horizontal=True,
             format_func=tx)
    if src == "Datei hochladen":
        up = st.file_uploader(tx("CSV-Datei"), type=["csv", "txt", "tsv"], key=f"up_{prefix}")
        return (up, up.name) if up is not None else (None, None)
    init(f"w_{prefix}_path", "")
    path = pl.clean_path(st.text_input(tx("Pfad zur CSV-Datei"), key=f"w_{prefix}_path",
                                       placeholder=tx("z. B. C:\\Daten\\yelp.csv oder ../daten/SMSSpamCollection")))
    return (path, Path(path).name) if path else (None, None)


def read_source(source, header=None):
    data = source.getvalue() if hasattr(source, "getvalue") else source
    return pl.read_csv(data, sep=ss.w_sep, decimal=ss.w_dec, encoding=ss.w_enc, header=header)


def header_choice():
    return {"ja": True, "nein": False}.get(ss.get("w_header", "auto"))


LABEL_NAMES = ["label", "stars", "sentiment", "class", "category", "kategorie", "klasse", "target", "spam", "thema",
               "topic", "rating", "y"]


def guess_label(df, text_col=None):
    """Vorschlag für das Label: übliche Namen (label, stars, sentiment …), sonst die Spalte mit den wenigsten
    verschiedenen Werten (mindestens 2), die nicht die Textspalte ist."""
    names = {str(c).lower(): c for c in df.columns if c != text_col}
    for cand in LABEL_NAMES:
        if cand in names:
            return names[cand]
    nun = df.nunique()
    cands = [c for c in df.columns if c != text_col and 2 <= nun[c] <= pl.MAX_CLASSES]
    return min(cands, key=lambda c: nun[c]) if cands else [c for c in df.columns if c != text_col][-1]


def describe_text(df):
    """describe() für Text-Spalten mit sauberen Ganzzahlen (count, unique, freq)."""
    out = df.describe().T
    for c in ("count", "unique", "freq"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int64")
    return out


def source_meta():
    s = ss.get("source", {})
    return {"source": s.get("name"), "rows_loaded": s.get("rows"), "header": s.get("header", True),
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
            df, enc = read_source(source, header_choice())
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
                 "header": meta.get("header", True), "names": list(df.columns),
                 "rows": len(df), "loaded": datetime.now().strftime("%H:%M:%S")}
    ss.pop("sample_key", None)
    sync_sample()
    prof = pl.column_profile(ss.df_raw)
    ss.profile = prof
    if same_columns and editor_value("ed_actions") is not None:  # gleiche Spalten → Einstellungen behalten
        old = editor_value("ed_actions").set_index("Spalte")
        for col in ("Aktion", "Parameter", "Neuer Name"):
            prof[col] = [old.at[c, col] if c in old.index else prof.at[i, col] for i, c in enumerate(prof["Spalte"])]
    else:
        for k in ["w_viz_cols", "w_text", "w_label", "w_classes", "w_pos", "_classes_for_label", "new_cols"]:
            ss.pop(k, None)
    editor_reset("ed_actions", prof)
    st.rerun()


WIDGET_KEYS = [  # Einstellungen, die exportiert/importiert werden
    "w_viz_cols", "w_text", "w_label", "w_classes", "w_pos", "w_missing_text", "w_test_size", "w_split_seed",
    "w_stratify", "w_vec", "w_lower", "w_stop", "w_ngram", "w_min_df", "w_max_df", "w_max_feat", "w_binary",
    "w_sublinear", "w_accents", "w_model", "w_alpha", "w_fit_prior", "w_log_c", "w_log_iter", "w_svc_c", "w_cw",
    "w_curve", "w_val", "w_val_split", "w_seed", "w_threshold",
]


def export_config() -> bytes:
    def records(name, cols):
        df = editor_value(name)
        return None if df is None else df[[c for c in cols if c in df.columns]].to_dict("records")

    cfg = {
        "template_config": 1,
        "engine": "scikit-learn",
        "app": pl.APP_ID,
        "exported": datetime.now().isoformat(timespec="seconds"),
        "source": ss.get("source", {}).get("name"),
        "step2": records("ed_actions", ["Spalte", "Aktion", "Parameter", "Neuer Name"]),
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
    if "new_columns" in cfg:
        ss.new_cols = cfg["new_columns"]
    for k, v in cfg["widgets"].items():
        if k in WIDGET_KEYS:
            ss[k] = v
    if skipped:
        ss._cfg_skipped = skipped
    if "w_label" in ss:  # Klassenauswahl aus der Datei behalten (sonst würde Schritt 4 sie zurücksetzen)
        ss._classes_for_label = ss.w_label


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
        msg = tx("Übernommen – jetzt Schritt 2 'Übernehmen' und Schritt 4 'Texte vektorisieren' klicken.")
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
        c2.caption(tx("scikit-learn rechnet auf der CPU – {n} Kerne. Naive Bayes, Logistische Regression und LinearSVC "
                      "sind auch bei vielen Texten schnell.", n=cpus))

    st.subheader(tx("CSV-Datei laden"))
    st.caption(tx("Eine Tabelle mit einer Textspalte (Bewertung, Nachricht, Beschreibung …) und einer Spalte mit der "
                  "Klasse (z. B. Sterne, spam/ham, Thema). Wie im Kurs: yelp.csv oder SMSSpamCollection (Tab, ohne "
                  "Kopfzeile)."))
    source, name = csv_source("data")
    csv_options()
    if st.button(tx("Daten laden"), type="primary", disabled=source is None):
        load_data(source, name)

    if "df_raw" not in ss:
        st.info(tx("Noch keine Daten geladen – oben eine CSV-Datei hochladen oder den Pfad eintragen."))
        return
    df, src = ss.df_raw, ss.source
    st.subheader(tx("Überblick: {name}", name=md_text(src['name'])))
    if src.get("detected"):
        st.info(tx("Automatisch erkannt: {v} → gelesen mit Trennzeichen „{v2}“, Dezimalzeichen „{decimal}“, "
                   "Encoding {encoding}.",
                   v=', '.join(src['detected']), v2=SEP_LABELS.get(src['sep'], src['sep']).split()[-1],
                   decimal=src['decimal'], encoding=src['encoding']))
    if src.get("header") is False:
        st.info(tx("Die Datei hat keine Kopfzeile – die Spalten heißen {cols}. In Schritt 2 unter **Neuer Name** "
                   "umbenennen (wie im Kurs: label, message).", cols=", ".join(map(str, df.columns[:6]))))
    if df.shape[1] == 1:
        st.warning(tx("Nur 1 Spalte erkannt – stimmt das Trennzeichen?"))
    m = st.columns(4)
    m[0].metric(tx("Zeilen"), fmt_int(len(df)))
    m[1].metric(tx("Spalten"), df.shape[1])
    texts = cached("_text_cols", (ss.data_token, ss.sample_key), lambda: pl.text_columns(df))
    free = [c for c in texts if pl.is_free_text(df[c])]
    m[2].metric(tx("Textspalten (Freitext)"), len(free))
    m[3].metric(tx("fehlende Zellen"), f"{df.isna().mean().mean():.2%}")
    short = [c for c in texts if c not in free and pl.mean_words(df[c]) > 1.1]  # Titel, Schlagzeilen …
    if free:
        words = {c: pl.mean_words(df[c]) for c in free[:3]}
        st.caption(tx("Freitext erkannt: ") + " · ".join(tx("{c} (Ø {n:.0f} Wörter)", c=md_text(c), n=w)
                                                         for c, w in words.items()))
    elif short:
        st.info(tx("Nur kurze Texte gefunden (im Mittel unter 3 Wörtern, z. B. Titel): {cols} – das geht auch; die "
                   "Textspalte wird in Schritt 4 gewählt.", cols=", ".join(md_text(c) for c in short[:3])))
    else:
        st.warning(tx("Keine Spalte mit Freitext gefunden (im Mittel mindestens 3 Wörter je Wert) – für die "
                      "Textklassifikation braucht es eine Spalte mit Texten."))
    if len(df) < src["rows"]:
        st.caption(tx("Stichprobe aktiv: {n} von {n2} Zeilen", n=fmt_int(len(df)), n2=fmt_int(src['rows'])))
    t1, t2, t3 = st.tabs(["df.head()", tx("Spalten (df.info)"), "df.describe()"])
    with t1:
        st.dataframe(df.head(20))
    with t2:
        st.dataframe(trc(ss.profile[["Spalte", "Typ", "Eindeutig", "Fehlend %", "Beispiel"]], values=["Typ"]),
                     hide_index=True)
    with t3:
        num = [c for c in df.columns if pl.is_numeric(df[c])]
        if num:
            st.dataframe(cached("_describe", (ss.data_token, ss.sample_key), lambda: df[num].describe().T))
        if len(num) < df.shape[1]:
            st.dataframe(cached("_describe_txt", (ss.data_token, ss.sample_key),
                                lambda: describe_text(df.drop(columns=num))))
    code(cg.load_code(src.get("path") or src["name"], src["sep"], src["decimal"], src["encoding"],
                      ss.w_frac if ss.get("w_use_sample") else None, strip_columns=bool(src.get("strip_columns")),
                      header=src.get("header", True) is not False, names=src.get("names")))


# ===========================================================================
# Seite 2 – Datenaufbereitung 1 (Spalten-Aktionen + neue Spalten)
# ===========================================================================
def shares_source(col, label):
    """Entsteht col aus derselben CSV-Spalte wie das Label? (dann verrät der Text die Antwort – Datenleck)"""
    lin = ss.get("lineage", {})
    return bool(label) and bool(set(pl.raw_sources(lin, col)) & set(pl.raw_sources(lin, label)))


def auto_select(old_cols, out):
    """Neu entstandene Spalten automatisch für die Visualisierung auswählen."""
    new_cols = [c for c in out.columns if c not in old_cols]
    if not old_cols or not new_cols:
        return
    if isinstance(ss.get("w_viz_cols"), list):
        ss.w_viz_cols = [c for c in out.columns if c in ss.w_viz_cols or c in new_cols]


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


def add_new_column(spec, clear_name=True):
    """Neue Spalte berechnen und an die Liste anhängen (Meldung in _newcol_msg)."""
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
    if clear_name:
        ss.w_new_name = ""
    ss._newcol_msg = ("success", tx("Hinzugefügt: {sources}", sources=', '.join(sources))
                      + (tx(" – das trainierte Modell wurde verworfen (Daten geändert).") if had_model else "."))


# Callbacks: laufen vor dem nächsten Durchlauf – dort dürfen auch Eingabefelder geleert werden
@with_lang
def on_add_new_column():
    if "df_viz" in ss:
        add_new_column(new_spec_from_widgets())


@with_lang
def on_add_length(col, act):
    """Schnell: Länge einer Textspalte als neue Spalte – wie im Kurs: yelp['text length'] = yelp['text'].apply(len)."""
    if "df_viz" not in ss or col not in ss.df_viz.columns:
        return
    name = pl.free_name(ss.df_viz.columns, f"{col} length" if act == pl.A_LEN else f"{col} words")
    add_new_column({"name": name, "kind": pl.N_COPY, "source": col, "definition": "", "action": act, "param": ""},
                   clear_name=False)


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
    texts = [c for c in pl.text_columns(v) if pl.is_free_text(v[c])]
    if texts:
        with st.container(border=True):
            st.markdown(tx("**Schnell: Textlänge als neue Spalte** – wie im Kurs: `yelp['text length'] = "
                           "yelp['text'].apply(len)`. Gut für Schritt 3 (Länge je Klasse)."))
            c1, c2, c3 = st.columns([2, 1.4, 1.4], vertical_alignment="bottom")
            with c1:
                tcol = sb(tx("Textspalte"), texts, "w_len_col")
            c2.button(tx("Länge in Zeichen"), icon=":material/straighten:", on_click=on_add_length,
                      args=(tcol, pl.A_LEN), width="stretch")
            c3.button(tx("Anzahl Wörter"), icon=":material/notes:", on_click=on_add_length, args=(tcol, pl.A_WORDS),
                      width="stretch")
    with st.container(border=True):
        kind = rd(tx("Art"), pl.NEW_KINDS, "w_new_kind", horizontal=True, format_func=tx)
        st.caption(tx(pl.NEW_HELP[kind][1]))
        init("w_new_name", "")
        if kind == pl.N_MAP:
            c1, c2, c3 = st.columns([2, 2, 4])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name", placeholder=tx("z. B. sentiment"))
            with c2:
                src = sb(tx("aus Spalte"), cols, "w_new_src")
            init("w_new_map", "")
            c3.text_input(tx("Zuordnung  alt=neu; …  (optional: sonst=…)"), key="w_new_map",
                          placeholder=tx("z. B. 1=negativ; 2=negativ; 4=positiv; 5=positiv"))
            if src:
                vc = v[src].value_counts().head(12)
                st.caption(tx("Werte in {src}: ", src=md_text(src))
                           + " · ".join(f"{md_text(pl.format_value(k))} ({fmt_int(n)})" for k, n in vc.items())
                           + (" · …" if v[src].nunique() > 12 else ""))
        elif kind in (pl.N_FORMULA, pl.N_COND):
            key = "w_new_formula" if kind == pl.N_FORMULA else "w_new_cond"
            example = "useful + funny" if kind == pl.N_FORMULA else "stars >= 4"
            c1, c2 = st.columns([2, 6])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name",
                          placeholder=tx("z. B. stimmen") if kind == pl.N_FORMULA else tx("z. B. positiv"))
            init(key, "")
            c2.text_input(tx("Formel") if kind == pl.N_FORMULA else tx("Bedingung"), key=key,
                          placeholder=tx("z. B. {example}", example=example))
            shown = num if kind == pl.N_FORMULA else cols
            st.caption(tx("Spalten: ") + ", ".join(f"\\`{md_text(c)}\\`" if not str(c).isidentifier() else md_text(c)
                                                    for c in shown[:40])
                       + (" …" if len(shown) > 40 else "") + tx(" · Namen mit Leer-/Sonderzeichen in \\`Backticks\\`"))
        else:
            c1, c2, c3, c4 = st.columns([2, 2, 3, 2])
            c1.text_input(tx("Name der neuen Spalte"), key="w_new_name", placeholder=tx("z. B. text length"))
            with c2:
                sb(tx("aus Spalte"), cols, "w_new_copy_src")
            with c3:
                act = sb(tx("Aktion"), [a for a in pl.ACTIONS if a != pl.A_DROP], "w_new_copy_act", default=pl.A_LEN,
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


EXAMPLES = [  # (Spalten, die vorhanden sein müssen, Datei, Hinweis)
    ({"business_id", "review_id", "stars", "text", "cool", "useful", "funny"}, "yelp_beispiel.json",
     N_("Yelp-Datensatz aus dem NLP-Projekt des Kurses erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und 6 "
        "wie im Kurs ein – IDs entfernen, neue Spalte text length, nur 1 und 5 Sterne, CountVectorizer und Naive Bayes "
        "(test_size 0.3, random_state 101).")),
    ({"Spalte_1", "Spalte_2"}, "sms_beispiel.json",
     N_("Zwei Spalten ohne Kopfzeile – wie SMSSpamCollection aus dem Kurs? Die Beispiel-Konfiguration benennt sie in "
        "label und message um, ergänzt die Länge und stellt TF-IDF + Naive Bayes ein (wie die Pipeline im Kurs).")),
]


def page_prep1():
    st.header(tx("2 · Datenaufbereitung 1 – Spalten umwandeln"))
    if not ready("df_raw", tx("Bitte zuerst in Schritt 1 eine CSV laden.")):
        return
    df = ss.df_raw
    st.markdown(tx("Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann "
                "**Übernehmen**. Die Vorschläge sind automatisch – bitte prüfen: Die **Textspalte** bleibt unverändert "
                "(*behalten*) – sie wird erst in Schritt 4 in Zahlen umgewandelt. IDs (z. B. review_id, user_id) "
                "entfernen; Spalten, die die Antwort schon verraten, sind ein Datenleck."))
    for cols, file, hint in EXAMPLES:
        example_cfg = CONFIG_DIR / file
        if cols <= set(df.columns) and example_cfg.exists() and (file != "sms_beispiel.json" or df.shape[1] == 2):
            c1, c2 = st.columns([4, 1])
            c1.info(tx(hint))
            c2.button(tx("Beispiel-Konfiguration laden"), on_click=on_example_config, args=(str(example_cfg),),
                      key=f"ex_{file}")
            break
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
        "Neuer Name": st.column_config.TextColumn(tx("Neuer Name"), help=tx("optional umbenennen, z. B. message")),
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
        editor_reset("ed_actions", pl.column_profile(df))
        st.rerun()

    if "df_viz" in ss:
        v = ss.df_viz
        st.success(tx("Ergebnis: {n} Zeilen × {v} Spalten", n=fmt_int(v.shape[0]), v=v.shape[1]))
        if ss.step2_log:
            st.markdown("\n".join(f"- {md_text(line)}" for line in ss.step2_log))
        st.dataframe(v.head(10))
        numeric_raw = {c for c in df.columns if pl.is_numeric(df[c])}
        code(cg.step2_code(ss.actions_fitted, numeric_raw, ss.get("new_cols_fitted")))

    new_columns_section()

    st.subheader(tx("Spalte untersuchen"))
    base = ss.df_viz if "df_viz" in ss else df
    col = sb(tx("Spalte"), list(base.columns), "w_inspect_col")
    if col:
        s = base[col]
        c1, c2 = st.columns([2, 3])
        c1.dataframe(trc(pd.DataFrame({"Typ": [pl.dtype_label(s)], "Eindeutig": [s.nunique()],
                                       "Fehlend": [int(s.isna().sum())]}), values=["Typ"]), hide_index=True)
        if pl.is_numeric(s):
            c1.dataframe(s.describe().to_frame(), height=320)
        elif pl.is_free_text(s):
            lens = pl.text_length_frame(base, col)
            c1.dataframe(trc(lens.describe().T.round(1)), height=150)
        vc = s.value_counts(dropna=False).head(30).rename("Anzahl").to_frame()
        vc["Anteil"] = (vc["Anzahl"] / len(s)).map("{:.2%}".format)
        c2.dataframe(trc(vc), height=420)


# ===========================================================================
# Seite 3 – Analyse & Visualisierung
# ===========================================================================
CHARTS = [N_("Überblick"), N_("Countplot"), N_("Textlänge"), N_("Häufigste Wörter"), N_("Beispieltexte"),
          N_("Histogramm"), N_("Boxplot"), N_("Korrelation (Heatmap)"), N_("Mittelwerte je Gruppe")]
UNITS = {"Zeichen": N_("Zeichen"), "Wörter": N_("Wörter")}


@with_lang
def on_viz_cols(which):
    full = ss.df_viz
    ss.w_viz_cols = (list(full.columns) if which == "alle"
                     else [c for c in full.columns if pl.is_numeric(full[c])] if which == "zahlen" else [])


@with_lang
def on_new_examples():
    ss.w_ex_seed = int(np.random.default_rng().integers(1_000_000))


def page_viz():
    st.header(tx("3 · Datenanalyse & Visualisierung"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    full = ss.df_viz
    with st.expander(tx("Spalten für die Visualisierung auswählen"), expanded=False, icon=":material/view_column:"):
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
    texts = cached("_viz_texts", (ss.viz_token, tuple(cols)), lambda: pl.text_columns(df))
    free = [c for c in texts if pl.is_free_text(df[c])] or texts
    low = [c for c in cols if nun[c] <= 30]  # für Farbe (hue) und Gruppen geeignet
    groups = [c for c in cols if 2 <= nun[c] <= pl.MAX_CLASSES and c not in free]
    label_guess = guess_label(df, free[0] if free else None)
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
                x = sb(tx("x (Kategorie)"), [c for c in cols if nun[c] <= 50], "w_cnt_x",
                       default=label_guess if nun.get(label_guess, 99) <= 50 else None)
            with c2:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_cnt_hue")
            with c3:
                order = rd(tx("Reihenfolge"), [N_("Wert"), N_("Häufigkeit")], "w_cnt_order", horizontal=True,
                           format_func=tx)
            if x:
                show(plots.fig_count(df, x, opt(hue), order))
                code(cg.chart_code("count", x=x, hue=opt(hue), order_by=order))

        elif chart == "Textlänge":
            st.caption(tx("Wie lang sind die Texte je Klasse? Wie im Kurs: Länge berechnen und je Klasse ein Histogramm "
                          "(sns.FacetGrid) – unterscheiden sich die Klassen schon in der Länge?"))
            c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
            with c1:
                tcol = sb(tx("Textspalte"), free, "w_len_text")
            with c2:
                grp = opt(sb(tx("je Gruppe"), [NONE] + [c for c in groups if c != tcol], "w_len_group",
                             default=label_guess if label_guess in groups else NONE))
            with c3:
                unit = rd(tx("Einheit"), list(UNITS), "w_len_unit", horizontal=True, format_func=lambda u: tx(UNITS[u]))
            with c4:
                init("w_len_bins", 50)
                bins = st.slider("bins", 10, 120, key="w_len_bins")
            if tcol:
                lens = cached("_lens", (ss.viz_token, tcol), lambda: pl.text_length_frame(df, tcol))
                values = lens[unit]
                gvals = pl.label_display(df[grp]) if grp else None
                show(plots.fig_text_length(values, gvals, unit, bins, group_name=grp or ""))
                if grp:
                    box = pd.DataFrame({grp: gvals, tx(UNITS[unit]): values})
                    c1, c2 = st.columns([3, 2])
                    with c1:
                        show(plots.fig_box(box, grp, tx(UNITS[unit])))
                    stats = box.groupby(grp)[tx(UNITS[unit])].describe()[["count", "mean", "50%", "max"]]
                    c2.dataframe(stats.round(1))
                code(cg.chart_code("length", text=tcol, group=grp, unit=unit, bins=bins))

        elif chart == "Häufigste Wörter":
            st.caption(tx("Welche Wörter kommen am häufigsten vor – insgesamt oder je Klasse? Ohne Stoppwörter (the, "
                          "and, is …) sind die Unterschiede zwischen den Klassen besser zu sehen."))
            c1, c2, c3, c4, c5 = st.columns([2, 2, 2, 1.2, 1.4])
            with c1:
                tcol = sb(tx("Textspalte"), free, "w_tt_text")
            with c2:
                grp = opt(sb(tx("je Gruppe"), [NONE] + [c for c in groups if c != tcol], "w_tt_group",
                             default=label_guess if label_guess in groups else NONE))
            with c3:
                stop = sb(tx("Stoppwörter entfernen"), list(pl.STOP_WORDS), "w_tt_stop", default="english",
                          format_func=lambda v: tx(pl.STOP_WORDS[v]))
            with c4:
                ngram = sb("n-gram", [1, 2], "w_tt_ngram", format_func=lambda v: tx("Wörter") if v == 1
                           else tx("Wortpaare"))
            with c5:
                init("w_tt_top", 20)
                top = st.slider(tx("Anzahl"), 5, 40, key="w_tt_top")
            if tcol:
                labels = pl.label_display(df[grp]) if grp else None
                terms = cached("_terms", (ss.viz_token, tcol, grp, stop, ngram, top),
                               lambda: pl.top_terms(df[tcol], labels, top, stop, ngram))
                if grp and len(terms) > plots.MAX_PANELS:
                    st.caption(tx("Gezeigt: die ersten {n} von {m} Gruppen.", n=plots.MAX_PANELS, m=len(terms)))
                title = (tx("Häufigste Wörter je {g}", g=grp) if grp else tx("Häufigste Wörter in {c}", c=tcol))
                show(plots.fig_top_terms(terms, title, top, group_name=grp or ""))
                with st.expander(tx("Tabelle")):
                    st.dataframe(trc(pd.concat({k or tx("alle Texte"): v for k, v in terms.items()},
                                               names=[grp or "", ""]).reset_index(level=0)), hide_index=True)
                if len(df) > 20_000:
                    st.caption(tx("Gezählt auf einer Stichprobe von 20 000 Texten."))
                code(cg.chart_code("terms", text=tcol, group=grp, stop_words=stop, ngram_max=ngram, top=top))

        elif chart == "Beispieltexte":
            c1, c2, c3, c4 = st.columns([2, 2, 1.2, 1.2], vertical_alignment="bottom")
            with c1:
                tcol = sb(tx("Textspalte"), free, "w_ex_text")
            with c2:
                grp = opt(sb(tx("je Gruppe"), [NONE] + [c for c in groups if c != tcol], "w_ex_group",
                             default=label_guess if label_guess in groups else NONE))
            with c3:
                init("w_ex_n", 3)
                n = st.number_input(tx("Texte je Gruppe"), 1, 20, key="w_ex_n")
            init("w_ex_seed", 101)
            c4.button(tx("Andere Texte"), icon=":material/casino:", on_click=on_new_examples)
            if tcol:
                if grp:
                    disp = pl.label_display(df[grp])
                    for value in sorted(disp.dropna().unique(), key=pl._natural_key)[:plots.MAX_PANELS]:
                        part = df.loc[disp == value, tcol].dropna()
                        st.markdown(tx("**{g} = {v}** · {n} Texte", g=md_text(grp), v=md_text(value),
                                       n=fmt_int(len(part))))
                        for t in part.sample(min(int(n), len(part)), random_state=int(ss.w_ex_seed)):
                            st.markdown(f"> {md_text(str(t)[:1200])}" + (" …" if len(str(t)) > 1200 else ""))
                else:
                    part = df[tcol].dropna()
                    for t in part.sample(min(int(n), len(part)), random_state=int(ss.w_ex_seed)):
                        st.markdown(f"> {md_text(str(t)[:1200])}" + (" …" if len(str(t)) > 1200 else ""))
                code(cg.chart_code("examples", text=tcol, group=grp, n=int(n), seed=int(ss.w_ex_seed)))

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
            elif not num:
                st.info(tx("Keine Zahlenspalten – in Schritt 2 z. B. die Textlänge als neue Spalte anlegen."))

        elif chart == "Boxplot":
            c1, c2, c3 = st.columns(3)
            with c1:
                x = sb(tx("x (Kategorie)"), [c for c in cols if nun[c] <= 50], "w_box_x",
                       default=label_guess if nun.get(label_guess, 99) <= 50 else None)
            with c2:
                y = sb(tx("y (Zahl)"), [c for c in num if c != x], "w_box_y")
            with c3:
                hue = sb(tx("Farbe (hue)"), [NONE] + low, "w_box_hue")
            if x and y:
                show(plots.fig_box(df, x, y, opt(hue)))
                code(cg.chart_code("box", x=x, y=y, hue=opt(hue)))
            elif not num:
                st.info(tx("Keine Zahlenspalten – in Schritt 2 z. B. die Textlänge als neue Spalte anlegen."))

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

        elif chart == "Mittelwerte je Gruppe":
            st.caption(tx("Wie im Kurs: stars = yelp.groupby('stars').mean() – Mittelwert jeder Zahlenspalte je Gruppe "
                          "und die Korrelation dieser Mittelwerte (z. B. cool, useful, funny, text length)."))
            c1, _ = st.columns([2, 3])
            with c1:
                by = sb(tx("gruppieren nach"), groups, "w_gb_by", default=label_guess if label_guess in groups else None)
            if by:
                means = df.groupby(by)[[c for c in num if c != by]].mean()
                if means.shape[1] == 0:
                    st.info(tx("Keine Zahlenspalten – in Schritt 2 z. B. die Textlänge als neue Spalte anlegen."))
                else:
                    st.dataframe(means.round(3))
                    if means.shape[1] >= 2 and len(means) >= 3:
                        c1, _ = st.columns([3, 2])
                        with c1:
                            show(plots.fig_corr_heatmap(means.corr(), True, False))
                    code(cg.chart_code("groupby", by=by))


# ===========================================================================
# Seite 4 – Datenaufbereitung 2 (Text, Label, Klassen, Split, Vektorisierung)
# ===========================================================================
HOW_VECTORIZE = N_("""**So wird Text zu Zahlen (Bag of Words)**

1. **Wörter zerlegen** – Kleinschreibung, Satzzeichen weg, optional Stoppwörter (the, and, is …) entfernen.
2. **Vokabular** – jedes Wort aus den **Trainingstexten** bekommt eine Spalte (oft Tausende).
3. **Zählen** – jeder Text wird zu einer Zeile: wie oft kommt jedes Wort vor (CountVectorizer). Die meisten Werte
   sind 0 – daher eine *sparse* Matrix, die nur die Nicht-Nullen speichert.
4. **TF-IDF** (optional) – Wörter, die in fast allen Texten vorkommen, zählen weniger; seltene, typische Wörter mehr.

Wie im Kurs mit `CountVectorizer` (und `TfidfTransformer`) – nur in der richtigen Reihenfolge: **erst aufteilen**,
dann das Vokabular nur aus den Trainingstexten lernen. Die Testtexte werden damit umgewandelt wie später neue Texte.""")


def vec_config() -> pl.VectorizerConfig:
    """Einstellungen aus den Widgets von Schritt 4 → VectorizerConfig."""
    d, g = pl.VectorizerConfig(), ss.get
    return pl.VectorizerConfig(
        kind=g("w_vec", d.kind), lowercase=bool(g("w_lower", d.lowercase)), stop_words=g("w_stop", d.stop_words),
        ngram_max=int(g("w_ngram", d.ngram_max)), min_df=int(g("w_min_df", d.min_df)),
        max_df=float(g("w_max_df", d.max_df)), max_features=int(g("w_max_feat", d.max_features)),
        binary=bool(g("w_binary", d.binary)), sublinear_tf=bool(g("w_sublinear", d.sublinear_tf)),
        strip_accents=bool(g("w_accents", d.strip_accents)))


def prep_signature(text, label, classes, pos):
    """Alle Einstellungen von Schritt 4 – geändert seit dem Vektorisieren? Dann Hinweis."""
    return (ss.get("viz_token"), text, label, tuple(classes or ()), pos, ss.get("w_missing_text"),
            float(ss.get("w_test_size", 0.3)), int(ss.get("w_split_seed", pl.RANDOM_STATE)),
            bool(ss.get("w_stratify", True)), tuple(sorted(vars(vec_config()).items())))


@with_lang
def on_bow_example():
    p = ss.get("prep")
    if p is not None:
        ss.bow_idx = int(np.random.default_rng().integers(len(p.text_train)))


def page_prep2():
    st.header(tx("4 · Datenaufbereitung 2 – Text, Label & Vektorisierung"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    df = ss.df_viz
    lin = ss.get("lineage", {})
    texts = cached("_prep_texts", ss.viz_token, lambda: pl.text_columns(df))
    if not texts:
        st.error(tx("Keine Textspalte in den Daten – für die Textklassifikation braucht es eine Spalte mit Texten "
                    "(in Schritt 2 behalten bzw. 'als Kategorie (Text)')."))
        return

    st.subheader(tx("Text (X) & Label (y)"))
    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        text = sb(tx("Textspalte – was gelesen wird"), texts, "w_text", default=texts[0],
                  help=tx("Die Spalte mit den Texten, z. B. text (Yelp) oder message (SMS)."))
    others = [c for c in df.columns if c != text]
    with c2:
        label = sb(tx("Label – was vorhergesagt werden soll"), others, "w_label", default=guess_label(df, text))
    if not label:
        st.error(tx("Es gibt keine weitere Spalte für das Label."))
        return
    nun_label = int(df[label].nunique())
    values = pl.label_values(df[label]) if nun_label <= 200 else []
    if ss.get("_classes_for_label") != label:  # anderes Label → Klassenauswahl zurücksetzen
        ss.w_classes = []
        ss._classes_for_label = label
    with c3:
        if values:
            chosen = ms(tx("Klassen (leer = alle)"), values, "w_classes",
                        help=tx("Nur einige Klassen verwenden – wie im Kurs nur 1 und 5 Sterne (deutlich negativ / "
                                "positiv). Zeilen mit anderen Klassen werden entfernt."))
        else:
            chosen = []
            st.error(tx("Das Label hat {n} verschiedene Werte – das ist kein Label (eher eine ID oder ein Text).",
                        n=fmt_int(nun_label)))
    effective = chosen if len(chosen) >= 2 else values
    pos = None
    c1, c2, c3 = st.columns([2, 2, 3])
    if len(effective) == 2:
        with c1:
            pos = sb(tx("Positive Klasse (= 1)"), effective, "w_pos", default=effective[-1],
                     help=tx("Diese Klasse wird zu 1 kodiert, die andere zu 0 (z. B. spam → 1, ham → 0). Die "
                             "Schwelle in Schritt 8 bezieht sich auf sie."))
    with c2:
        rd(tx("Zeilen ohne Text"), list(pl.MISSING_TEXT), "w_missing_text", format_func=lambda v: tx(pl.MISSING_TEXT[v]),
           help=tx("Leere oder fehlende Texte enthalten kein Wort – entfernen ist meist richtig."))
    if values and len(chosen) == 1:
        st.warning(tx("Bitte mindestens 2 Klassen auswählen – oder keine (dann alle)."))
    if values and len(effective) > pl.MAX_CLASSES:
        st.warning(tx("{n} Klassen – höchstens {m} sind möglich. Bitte einige Klassen auswählen.",
                      n=len(effective), m=pl.MAX_CLASSES))
    if shares_source(text, label):
        st.warning(tx("Textspalte und Label stammen aus derselben CSV-Spalte → Datenleck!"))
    if values:
        with st.expander(tx("Verteilung von {label}", label=md_text(label))):
            counts = pl.label_counts(df[label])
            vc = counts.rename("Anzahl").to_frame()
            vc["Anteil"] = (vc["Anzahl"] / max(int(counts.sum()), 1)).map("{:.2%}".format)
            if chosen:
                vc[tx("verwendet")] = [v in chosen for v in vc.index]
            st.dataframe(trc(vc))
            no_label = int(pl.missing_label(df[label]).sum())  # fehlend oder nur Leerzeichen
            if no_label:
                st.caption(tx("{n} Zeilen ohne Label werden entfernt.", n=fmt_int(no_label)))

    st.subheader(tx("Train/Test-Split"))
    c1, c2, c3 = st.columns([1.4, 1.1, 1])
    with c1:
        init("w_test_size", 0.3)
        st.slider("test_size", 0.05, 0.5, step=0.01, key="w_test_size",
                  help=tx("Anteil der Texte für den Test – im Kurs 0.3 (Yelp) bzw. 0.2 (SMS)."))
    with c2:
        init("w_split_seed", pl.RANDOM_STATE)
        st.number_input("random_state", 0, 1_000_000, step=1, key="w_split_seed")
    with c3:
        init("w_stratify", True)
        st.checkbox("stratify=y", key="w_stratify",
                    help=tx("Klassenverhältnis in Train und Test gleich halten (empfohlen, z. B. nur 13 % spam)"))

    st.subheader(tx("Vektorisierung – Text → Zahlen"))
    with st.expander(tx("So wird Text zu Zahlen"), icon=":material/info:"):
        st.markdown(tx(HOW_VECTORIZE))
    c1, c2, c3, c4 = st.columns([3, 1.6, 1.4, 1.4])
    with c1:
        kind = rd(tx("Vektorisierer"), list(pl.VECTORIZERS), "w_vec", format_func=lambda v: tx(pl.VECTORIZERS[v]))
    with c2:
        sb(tx("Stoppwörter"), list(pl.STOP_WORDS), "w_stop", format_func=lambda v: tx(pl.STOP_WORDS[v]),
           help=tx("Sehr häufige Wörter ohne Inhalt (the, and, is …) entfernen – die Liste von scikit-learn ist "
                   "englisch. Im Kurs: stopwords.words('english') aus nltk."))
    with c3:
        sb("ngram_range", [1, 2, 3], "w_ngram", format_func=lambda v: f"(1, {v})",
           help=tx("(1, 2) = auch Wortpaare wie 'not good' – erkennt Verneinungen, aber das Vokabular wird viel "
                   "größer."))
    with c4:
        init("w_lower", True)
        st.checkbox(tx("Kleinschreibung (lowercase)"), key="w_lower")
        init("w_accents", False)
        st.checkbox(tx("Akzente entfernen"), key="w_accents", help=tx("café = cafe, Müller = Muller (strip_accents)"))
    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.4, 1.3, 1.5])
    with c1:
        init("w_min_df", 1)
        st.number_input("min_df", 1, 100_000, step=1, key="w_min_df",
                        help=tx("Wort muss in mindestens so vielen Trainingstexten vorkommen – 2 oder 5 entfernt "
                                "Tippfehler und Einzelfälle, das Vokabular wird kleiner."))
    with c2:
        init("w_max_df", 1.0)
        st.number_input("max_df", 0.05, 1.0, step=0.05, format="%.2f", key="w_max_df",
                        help=tx("Wörter, die in mehr als diesem Anteil der Texte vorkommen, weglassen (z. B. 0.9) – "
                                "eine Art automatischer Stoppwörter."))
    with c3:
        init("w_max_feat", 0)
        st.number_input("max_features", 0, 10_000_000, step=1000, key="w_max_feat",
                        help=tx("Nur die häufigsten N Wörter behalten – 0 = alle."))
    with c4:
        init("w_binary", False)
        st.checkbox("binary", key="w_binary", help=tx("Nur vorhanden (1) oder nicht (0) statt Anzahl"))
    with c5:
        init("w_sublinear", False)
        st.checkbox("sublinear_tf", key="w_sublinear", disabled=kind != pl.VEC_TFIDF,
                    help=tx("Nur TF-IDF: 1 + log(Anzahl) statt Anzahl – ein Wort, das 10-mal vorkommt, zählt nicht "
                            "10-mal so viel."))

    sig = prep_signature(text, label, chosen, pos)
    if "model" in ss:
        st.warning(tx("Neu vektorisieren verwirft das trainierte Modell – vorher in Schritt 9 speichern."))
    disabled = not values or len(chosen) == 1
    if st.button(tx("Texte vektorisieren"), type="primary", icon=":material/text_fields:", disabled=disabled):
        try:
            with st.spinner(tx("Teile auf und lerne das Vokabular …")):
                prep = pl.prepare_text_data(
                    df, text, label, classes=chosen or None, positive_class=pos,
                    missing_text=ss.get("w_missing_text", pl.TEXT_DROP), test_size=float(ss.w_test_size),
                    random_state=int(ss.w_split_seed), stratify=bool(ss.w_stratify), vec=vec_config(),
                    column_actions=ss.actions_fitted, lineage=lin, meta=source_meta(),
                    new_columns=ss.get("new_cols_fitted", []))
        except ValueError as exc:
            st.error(md_text(exc))  # Meldungen nennen Klassen und Spalten aus den Daten → wörtlich zeigen
        else:
            invalidate_from("prep")
            ss.prep, ss.prep_token, ss.prep_sig = prep, uuid.uuid4().hex, sig
            st.toast(tx("Texte vektorisiert"), icon=":material/check_circle:")

    if "prep" not in ss:
        return
    p = ss.prep
    r = p.recipe
    if ss.get("prep_sig") != sig:
        st.info(tx("Die Einstellungen oben wurden seit dem Vektorisieren geändert → erneut **Texte vektorisieren**."))
    nnz = p.X_train.nnz / max(p.X_train.shape[0] * p.X_train.shape[1], 1)
    st.success(tx("Fertig: X_train {shape} · X_test {shape2} · Vokabular {v} Wörter · {pct} der Werte sind nicht 0 "
                  "(sparse Matrix)", shape=p.X_train.shape, shape2=p.X_test.shape, v=fmt_int(r["vocab_size"]),
                  pct=f"{nnz:.3%}"))
    for line in p.log:
        st.markdown(f"- {md_text(line)}")
    for w in p.warnings:
        st.warning(md_text(w))
    dist = pd.DataFrame({
        "Klasse": r["classes"],
        "Kodiert": range(len(r["classes"])),
        "Train": np.bincount(p.y_train, minlength=len(r["classes"])),
        "Test": np.bincount(p.y_test, minlength=len(r["classes"])),
    })
    dist["Anteil Train"] = (dist["Train"] / dist["Train"].sum()).map("{:.1%}".format)
    c1, c2 = st.columns([2, 3])
    c1.dataframe(trc(dist), hide_index=True)
    stats = r["text_stats"]
    c2.markdown(tx("**Trainingstexte:** Ø {c:.0f} Zeichen · Ø {w:.0f} Wörter", c=stats["mean_chars"],
                   w=stats["mean_words"]))
    if stats.get("test_coverage") is not None:
        c2.markdown(tx("**Testtexte:** {pct} ihrer Wörter stehen im Vokabular – der Rest war im Training nie zu sehen "
                       "und wird ignoriert.", pct=f"{stats['test_coverage']:.1%}"))

    st.markdown(tx("**Beispiel: ein Trainingstext als Zahlen**"))
    i = min(ss.get("bow_idx", 0), len(p.text_train) - 1)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.caption(tx("Text {i} von {n} · Klasse '{c}'", i=i + 1, n=fmt_int(len(p.text_train)),
                      c=md_text(r["classes"][p.y_train[i]])))
        st.markdown(f"> {md_text(str(p.text_train[i])[:1500])}" + (" …" if len(str(p.text_train[i])) > 1500 else ""))
        st.button(tx("Anderer Text"), icon=":material/casino:", on_click=on_bow_example)
    with c2:
        row = p.X_train[i]
        words = p.vectorizer.get_feature_names_out()[row.indices]
        vec_df = pd.DataFrame({"Wort": words, "Wert": np.round(row.data, 4)}).sort_values(
            ["Wert", "Wort"], ascending=[False, True])
        st.dataframe(trc(vec_df), hide_index=True, height=260)
        st.caption(tx("{n} von {v} Spalten sind hier nicht 0 – wie im Kurs: bow_transformer.transform([text])",
                      n=len(row.indices), v=fmt_int(r["vocab_size"])))
    with st.expander(tx("Vokabular (häufigste Wörter in den Trainingstexten)"), icon=":material/dictionary:"):
        voc = cached("_vocab", ss.prep_token, lambda: pl.vocabulary_table(p.vectorizer, p.X_train))
        st.dataframe(trc(voc.head(500)), hide_index=True, height=360)
        st.caption(tx("{n} Wörter – die ersten 500 nach Anzahl Texte. Komplett als CSV in Schritt 5.",
                      n=fmt_int(len(voc))))
    code(cg.step4_code(r))


# ===========================================================================
# Seite 5 – Daten speichern
# ===========================================================================
EXPORTS = {
    "step2": N_("Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)"),
    "split": N_("Texte nach Schritt 4 – Text, Label, Label als Zahl und Spalte 'split' (train/test)"),
    "vocab": N_("Vokabular nach Schritt 4 – jedes Wort mit Anzahl Texte (und IDF bei TF-IDF)"),
}
FORMATS = {"intl": N_("CSV international (Komma, Dezimalpunkt)"),  # Excel: Format der gewählten Sprache/Region
           "excel": N_("CSV für Excel (Format der gewählten Sprache)")}


def export_frame(what):
    if what == "step2":
        return ss.df_viz
    if what == "split":
        return pl.split_frame(ss.prep, ss.df_viz)
    return pl.vocabulary_table(ss.prep.vectorizer, ss.prep.X_train).rename(columns=vocab_columns())


def page_export():
    st.header(tx("5 · Daten speichern"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    st.markdown(tx("Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen "
                "Programm."))
    what = rd(tx("Was speichern?"), list(EXPORTS), "w_exp_what", format_func=lambda v: tx(EXPORTS[v]))
    fmt = rd(tx("Format"), list(FORMATS), "w_exp_fmt", format_func=lambda v: tx(FORMATS[v]), horizontal=True)
    if what != "step2" and "prep" not in ss:
        st.info(tx("Dafür zuerst Schritt 4 ausführen (**Texte vektorisieren**)."))
        return
    if ss.get("w_use_sample"):
        st.warning(tx("Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.",
                      n=fmt_int(len(ss.df_raw)), n2=fmt_int(ss.source['rows'])))
    if what == "split":
        r = ss.prep.recipe
        st.caption(tx("Label kodiert: ") + ", ".join(f"{md_text(c)} → {i}" for i, c in enumerate(r["classes"])))
    excel = i18n.get_lang() if fmt == "excel" else None  # 'de'/'pl': ; und Dezimalkomma · 'en': , und Punkt
    # Sprache im Schlüssel: nach einem Sprachwechsel neu erstellen – sonst bliebe der Dateiname in der alten Sprache
    key = (what, fmt, excel, i18n.get_lang(), ss.viz_token, ss.get("prep_token") if what != "step2" else None)
    stem = {"step2": tx("daten_aufbereitet"), "split": tx("text_daten"), "vocab": tx("vokabular")}[what]
    name = f"{Path(ss.source['name']).stem}_{stem}.csv"
    if st.button(tx("CSV erstellen"), type="primary", icon=":material/table_view:"):
        with st.spinner(tx("Erstelle CSV …")):
            frame = export_frame(what)
            ss._export = (key, pl.to_csv_bytes(frame, excel=excel), name, frame.shape, frame.head(10))
    exp = ss.get("_export")
    if exp and exp[0] == key:
        _, data, name, shape, head = exp
        st.success(tx("{name}: {n} Zeilen × {v} Spalten · {v2:.1f} MB",
                      name=md_text(name), n=fmt_int(shape[0]), v=shape[1], v2=len(data) / 1e6))
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
            st.warning(tx("Die Datei **{target}** gibt es schon.", target=md_text(target)))
            allowed = st.checkbox(tx("Ja, vorhandene Datei überschreiben"), key=f"exp_overwrite_{name}")
        if st.button(tx("In Ordner speichern"), icon=":material/save:", disabled=not allowed):
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            except OSError as exc:
                st.error(tx("Speichern fehlgeschlagen: {exc}", exc=exc))
            else:
                ss._exp_saved = str(target)
                st.success(tx("Gespeichert: {target}", target=md_text(target)))
        notes = [tx("Für Vorhersagen (Schritt 10) die **ursprüngliche** CSV-Datei verwenden, nicht diese Datei – "
                 "das gespeicherte Modell wiederholt die Aufbereitung selbst.")]
        if excel:
            notes.append(tx("Texte, die mit =, +, -, @ beginnen, bekommen ein ' davor, damit Excel sie nicht als "
                         "Formel ausführt."))
        st.caption(" ".join(notes))
    r = ss.prep.recipe if "prep" in ss else None
    code(cg.export_code(what, r, excel=excel, columns=vocab_columns()))


# ===========================================================================
# Seite 6 – Modell & Training
# ===========================================================================
CURVE_WHAT = {  # was die Validierungskurve in Schritt 7 zeigt
    pl.MODEL_NB: N_("Fehlerrate je alpha (0.01 … 5) – wie stark geglättet wird"),
    pl.MODEL_LOGREG: N_("Fehlerrate je C (0.01 … 100) – wie stark regularisiert wird"),
    pl.MODEL_SVC: N_("Fehlerrate je C (0.001 … 10) – wie stark regularisiert wird"),
}


def model_label(m):
    """'1 · Naive Bayes (MultinomialNB)' – Nummer + Name."""
    return f"{pl.MODELS.index(m) + 1} · {pl.model_name(m)}"


def model_captions():
    return [f"{tx(pl.MODEL_WHY[m])}  \n*{tx('Am besten für')}: {tx(pl.MODEL_BEST[m])}*" for m in pl.MODELS]


def metric_name():
    return tx("Fehlerrate")


def model_config() -> pl.TextModelConfig:
    """Einstellungen aus den Widgets von Schritt 6 → TextModelConfig (nicht angezeigte Widgets = Standardwerte)."""
    d, g = pl.TextModelConfig(), ss.get
    return pl.TextModelConfig(
        model=g("w_model", d.model), alpha=float(g("w_alpha", d.alpha)), fit_prior=bool(g("w_fit_prior", d.fit_prior)),
        logreg_c=float(g("w_log_c", d.logreg_c)), max_iter=int(g("w_log_iter", d.max_iter)),
        svc_c=float(g("w_svc_c", d.svc_c)), class_weight=bool(g("w_cw", d.class_weight)),
        curve=bool(g("w_curve", d.curve)), validation=g("w_val", d.validation),
        validation_split=float(g("w_val_split", d.validation_split)), seed=int(g("w_seed", d.seed)))


def model_params_ui(m):
    """Parameter des gewählten Modells (Standardwerte wie in scikit-learn)."""
    if m == pl.MODEL_NB:
        c1, c2, _ = st.columns([1, 1.6, 2])
        with c1:
            init("w_alpha", 1.0)
            st.number_input("alpha", 0.0001, 10_000.0, step=0.1, format="%g", key="w_alpha",
                            help=tx("Glättung (Laplace): auch Wörter, die in einer Klasse nie vorkamen, bekommen eine "
                                    "kleine Wahrscheinlichkeit. Klein = das Modell vertraut einzelnen Wörtern mehr "
                                    "(Gefahr von Overfitting), groß = glatter. Standard 1.0."))
        with c2:
            init("w_fit_prior", True)
            st.checkbox(tx("Klassenanteile als Vorwissen (fit_prior)"), key="w_fit_prior",
                        help=tx("Ein: die Häufigkeit der Klassen im Training zählt mit (Standard). Aus: alle Klassen "
                                "gleich wahrscheinlich – hilft manchmal der seltenen Klasse."))
    elif m == pl.MODEL_LOGREG:
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            init("w_log_c", 1.0)
            st.number_input("C", 0.0001, 1_000_000.0, step=0.5, format="%g", key="w_log_c",
                            help=tx("Kehrwert der Regularisierung: klein = stärkere Regularisierung (einfacheres "
                                    "Modell), groß = schwächere. Bei Texten oft 1–10."))
        with c2:
            init("w_log_iter", 1000)
            st.number_input("max_iter", 10, 100_000, step=100, key="w_log_iter",
                            help=tx("Maximale Iterationen des Lösers – erhöhen, wenn eine ConvergenceWarning erscheint "
                                    "(Standard in scikit-learn: 100)."))
    else:
        c1, _ = st.columns([1, 3])
        with c1:
            init("w_svc_c", 1.0)
            st.number_input("C", 0.0001, 1_000_000.0, step=0.5, format="%g", key="w_svc_c",
                            help=tx("Strafe für Fehler im Training: groß = genaue Trennung (Gefahr von Overfitting), "
                                    "klein = breiterer Rand, einfacheres Modell. Bei Texten oft 0.1–1."))
        st.caption(tx("LinearSVC liefert keine Wahrscheinlichkeiten – CalibratedClassifierCV ergänzt sie (für Schwelle "
                      "und ROC-AUC), das Training dauert dadurch etwas länger."))


def page_train():
    st.header(tx("6 · Modell & Training"))
    if not ready("prep", tx("Bitte zuerst Schritt 4 ausführen (Texte vektorisieren).")):
        return
    p = ss.prep
    r = p.recipe
    task = r["task"]
    n_train = len(p.y_train)
    st.info(tx("**{v}** · {k} Klassen · {n_feat} Wörter im Vokabular · {n} Trainingstexte · {n2} Testtexte",
               v=tx(pl.TASK_LABELS[task]), k=len(r["classes"]), n_feat=fmt_int(p.X_train.shape[1]),
               n=fmt_int(n_train), n2=fmt_int(len(p.y_test))))

    st.subheader(tx("Modell wählen"))
    m = rd(tx("Modell"), pl.MODELS, "w_model", default=pl.MODEL_NB, format_func=model_label, captions=model_captions())
    with st.expander(tx("Die drei Modelle als Tabelle"), icon=":material/table:"):
        table = pd.DataFrame({"Modell": [pl.model_name(k) for k in pl.MODELS],
                              "Warum?": [tx(pl.MODEL_WHY[k]) for k in pl.MODELS],
                              "Am besten für": [tx(pl.MODEL_BEST[k]) for k in pl.MODELS]})
        st.table(trc(table).set_index(tx("Modell")))

    st.subheader(tx("Parameter: {name}", name=pl.model_name(m)))
    model_params_ui(m)
    vec_kind = (r.get("vectorizer") or {}).get("kind")
    if m == pl.MODEL_SVC and vec_kind == pl.VEC_COUNT:
        st.caption(tx("Tipp: LinearSVC arbeitet am besten mit TF-IDF (Schritt 4) – mit rohen Wortzählungen langer Texte "
                      "dauert das Training länger und konvergiert oft nicht vollständig."))
    elif m == pl.MODEL_NB and vec_kind == pl.VEC_TFIDF:
        st.caption(tx("Tipp: Mit TF-IDF sind die Werte klein (meist unter 1) – alpha = 1 glättet dann stark, besonders "
                      "bei großem Vokabular (Wortpaare). Oft passt ein kleineres alpha, z. B. 0.1; die Validierungskurve "
                      "in Schritt 7 zeigt den besten Wert."))
    elif m == pl.MODEL_LOGREG and vec_kind == pl.VEC_TFIDF:
        st.caption(tx("Tipp: Mit TF-IDF braucht die Logistische Regression meist ein größeres C, z. B. 10 (die Werte "
                      "sind normiert und klein); die Validierungskurve in Schritt 7 zeigt den besten Wert."))

    st.subheader(tx("Training & Validierung"))
    c1, c2, c3, c4, c5 = st.columns([2.5, 2.5, 1.5, 2.1, 1])
    with c1:
        init("w_curve", True)
        curve_on = st.checkbox(tx("Validierungskurve für Schritt 7"), key="w_curve",
                               help=tx("Trainiert das Modell zusätzlich für mehrere Werte von alpha bzw. C und misst die "
                                       "Fehlerrate auf Trainings- und Validierungsdaten – dauert etwas länger."))
    with c2:
        val = rd(tx("Validierung"), ["test", "split"], "w_val", horizontal=True, disabled=not curve_on,
                 format_func={"test": tx("Testdaten"), "split": tx("Anteil aus Training (sauberer)")}.get,
                 help=tx("Testdaten: Die Kurve wird auf X_test berechnet (wie die Elbow-Methode im Kurs) – einfach, "
                         "aber wer danach den besten Wert wählt, hat die Testdaten mitbenutzt. Sauberer: eigener "
                         "Validierungsanteil aus den Trainingsdaten (mit eigenem Vokabular), damit das Testset bis "
                         "Schritt 8 unberührt bleibt."))
    with c3:
        init("w_val_split", 0.2)
        st.slider("validation_split", 0.05, 0.4, step=0.05, key="w_val_split", disabled=not curve_on or val != "split")
    with c4:
        init("w_cw", False)
        st.checkbox("class_weight='balanced'", key="w_cw", disabled=m == pl.MODEL_NB,
                    help=tx("class_weight='balanced' – hilft der seltenen Klasse (z. B. spam). Naive Bayes kennt das "
                            "nicht – dort fit_prior ausschalten."))
    with c5:
        init("w_seed", pl.RANDOM_STATE)
        st.number_input(tx("Seed"), 0, 1_000_000, step=1, key="w_seed")
    cfg = model_config()
    st.caption(tx("Schritt 7 zeigt: {what}.", what=tx(CURVE_WHAT[m])) if cfg.curve
               else tx("Ohne Validierungskurve vergleicht Schritt 7 nur Training und Test."))
    if p.X_train.shape[1] > 100_000 and m != pl.MODEL_NB:
        st.info(tx("Sehr großes Vokabular ({v} Wörter): Logistische Regression und LinearSVC brauchen länger. Tipp: in "
                   "Schritt 4 min_df erhöhen oder max_features begrenzen.", v=fmt_int(p.X_train.shape[1])))

    st.subheader(tx("scikit-learn-Code mit diesen Einstellungen"))
    calib = pl.calibration_cv(p.y_train) if m == pl.MODEL_SVC else 5
    st.code(cg.model_code(cfg, calib), language="python")

    if "model" in ss:
        ti = ss.train_info
        st.info(tx("Aktuelles Modell: {name} · trainiert {v} · {seconds} s. Neues Training ersetzt es.",
                   name=pl.model_name(ti["model"]), v=ti["trained_at"][11:], seconds=ti["seconds"]))
    st.caption(tx("Während des Trainings nichts anklicken – jede Eingabe bricht das Training ab."))
    if st.button(tx("Training starten"), type="primary", icon=":material/rocket_launch:"):
        bar = st.progress(0.0, text=tx("Starte Training …"))
        chart = st.empty()
        last = [0.0]

        def progress(frac, text, curve):
            """Fortschrittsbalken + Live-Kurve (Training/Validierung)."""
            bar.progress(min(max(float(frac), 0.0), 1.0), text=text)
            if curve and curve["train"] and (time.time() - last[0] >= 0.4
                                             or len(curve["train"]) == len(curve["values"])):
                last[0] = time.time()
                k = len(curve["train"])
                live = pd.DataFrame({tx("Training"): curve["train"], tx("Validierung"): curve["val"]},
                                    index=pd.Index([f"{v:g}" for v in curve["values"][:k]], name=curve["param"]))
                chart.line_chart(live, color=[plots.SERIES[0], plots.SERIES[1]], height=260)

        try:
            model, hist, info = pl.train_model(cfg, p, progress)
        except Exception as exc:  # noqa: BLE001 – Meldung zeigen; der Vergleich unten bleibt erreichbar
            bar.empty()
            chart.empty()
            st.error(tx("Training fehlgeschlagen: {exc}", exc=md_text(exc)))
        else:
            bar.progress(1.0, text=tx("Fertig · {seconds} s", seconds=info["seconds"]))
            invalidate_from("model")
            ss.model, ss.history, ss.train_info, ss.model_cfg = model, hist, info, cfg
            ss.model_token = uuid.uuid4().hex  # eindeutig je Training (id() kann nach dem Löschen wiederverwendet werden)
            st.success(tx("Training abgeschlossen ({metric}): Training {t} · Test {v} → weiter mit Schritt 7.",
                          metric=metric_name(), t=pl._pct(info["train_error"]), v=pl._pct(info["test_error"])))
    if "model" in ss:
        if not ss.train_info.get("converged", True):
            st.warning(tx("LinearSVC ist nicht vollständig konvergiert (ConvergenceWarning) – C verkleinern oder in "
                          "Schritt 4 TF-IDF wählen (normierte Werte statt roher Wortzählungen).")
                       if ss.train_info.get("model") == pl.MODEL_SVC else
                       tx("Das Modell ist nicht vollständig konvergiert (ConvergenceWarning) – max_iter erhöhen oder C "
                          "verkleinern."))
        with st.expander(tx("Modell (scikit-learn)")):
            st.code(repr(pl.make_pipeline(p.vectorizer, ss.model)), language="text")

    st.subheader(tx("Vergleich: alle drei Modelle"))
    st.caption(tx("Trainiert Naive Bayes, Logistische Regression und LinearSVC mit den aktuellen Einstellungen (alpha, "
                  "C, class_weight) auf denselben Trainingstexten und misst sie auf den Testdaten – Schwelle 0.5."))
    if st.button(tx("Modelle vergleichen"), icon=":material/compare_arrows:"):
        bar = st.progress(0.0, text=tx("Vergleiche …"))
        try:
            table = pl.compare_models(cfg, p, lambda f, t: bar.progress(min(max(float(f), 0.0), 1.0), text=t))
        except Exception as exc:  # noqa: BLE001
            st.error(tx("Vergleich fehlgeschlagen: {exc}", exc=md_text(exc)))
        else:
            ss.compare = (table, cfg)
        bar.empty()
    if ss.get("compare"):
        table, ccfg = ss.compare
        names = [pl.model_name(k) for k in table["model"]]
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_compare(table, names))
        view = table.drop(columns="model").assign(Modell=names).set_index("Modell")
        c2.dataframe(trc(view.reset_index()).set_index(tx("Modell")))
        best = names[int(table["Accuracy"].to_numpy().argmax())]
        c2.markdown(tx("Höchste Accuracy: **{m}**. Mit den Einstellungen alpha = {a:g}, C = {c:g} (Logistische "
                       "Regression), C = {s:g} (LinearSVC){cw}.", m=best, a=ccfg.alpha, c=ccfg.logreg_c, s=ccfg.svc_c,
                       cw=", class_weight='balanced'" if ccfg.class_weight else ""))
        for m, why in (table.attrs.get("skipped") or {}).items():
            c2.info(tx("{m} fehlt im Vergleich: {why}", m=pl.model_name(m), why=md_text(why)))
        code(cg.compare_code(ccfg, task, pl.calibration_cv(p.y_train), list(table["model"])),
             tx("Python-Code zum Vergleich"))


# ===========================================================================
# Seite 7 – Training vs. Validierung
# ===========================================================================
@with_lang
def on_apply_best(param, value):
    """Besten Wert der Validierungskurve in die Widgets von Schritt 6 übernehmen – für das TRAINIERTE Modell (die
    Kurve gehört zu ihm), auch wenn in Schritt 6 inzwischen ein anderes Modell ausgewählt ist."""
    cfg = ss.get("model_cfg")
    if cfg is None:
        return
    ss.w_model = cfg.model
    if param == "alpha":
        ss.w_alpha = float(value)
    elif cfg.model == pl.MODEL_LOGREG:
        ss.w_log_c = float(value)
    else:
        ss.w_svc_c = float(value)
    ss._curve_msg = tx("Übernommen: {param} = {v} – jetzt in Schritt 6 neu trainieren.", param=param, v=f"{value:g}")


def page_curve():
    st.header(tx("7 · Auswertung Training vs. Validierung"))
    if not ready("history", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    hist, info, cfg = ss.history, ss.train_info, ss.model_cfg
    curve = hist.get("curve")
    a = pl.curve_assessment(curve, info, cfg)
    st.caption(tx("{name} · {metric} – kleiner = besser. Naive Vorhersage: man sagt immer die häufigste Klasse "
                  "vorher.", name=pl.model_name(cfg.model), metric=metric_name()))
    where = tx("Anteil aus Training") if curve and curve["validation"] == "split" else tx("Testdaten")
    m = st.columns(3)
    m[0].metric(tx("Training"), pl._pct(a["train"]))
    m[1].metric(tx("Validierung ({where})", where=where), pl._pct(a["val"]))
    m[2].metric(tx("naive Vorhersage"), pl._pct(a["baseline"]))
    if curve:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_curve(curve, metric_name()))
        with c2:
            st.markdown("\n\n".join(a["notes"]))
            if a["suggest"] is not None:
                st.button(tx("Besten Wert in Schritt 6 übernehmen"), icon=":material/tune:", on_click=on_apply_best,
                          args=(a["suggest"]["param"], a["suggest"]["value"]))
            msg = ss.pop("_curve_msg", None)
            if msg:
                st.success(msg)
            if cfg.model == pl.MODEL_SVC:
                st.caption(tx("Die Kurve nutzt die reine LinearSVC (ohne Kalibrierung, schneller) – die Kennzahlen in "
                              "Schritt 8 können leicht abweichen."))
            if curve["validation"] == "split":
                st.caption(tx("Validierung: {n} der {n2} Trainingstexte; das Vokabular dafür wurde nur aus den übrigen "
                              "{n3} Texten gelernt.", n=fmt_int(curve["n_val"]),
                              n2=fmt_int(curve["n_val"] + curve["n_fit"]), n3=fmt_int(curve["n_fit"])))
        with st.expander(tx("Tabelle")):
            table = pd.DataFrame({curve["param"]: [f"{v:g}" for v in curve["values"]],
                                  "Training": curve["train"], "Validierung": curve["val"]})
            st.dataframe(trc(table.round(5)), hide_index=True)
    else:
        st.markdown("\n\n".join(a["notes"]))
        st.info(tx("Die Validierungskurve ist in Schritt 6 ausgeschaltet – mit Kurve zeigt diese Seite den Verlauf über "
                   "alpha bzw. C."))
    code(cg.curve_code(cfg, pl.vectorizer_config(ss.prep.recipe.get("vectorizer")),
                       (curve or {}).get("stratify", True)))


# ===========================================================================
# Seite 8 – Evaluation & wichtige Wörter
# ===========================================================================
def test_scores():
    token = ss.get("model_token")
    if ss.get("eval_scores", (None,))[0] != token:
        ss.eval_scores = (token, pl.predict_scores(ss.model, ss.prep.X_test, ss.prep.recipe["task"]))
    return ss.eval_scores[1]


def metric_row(ev):
    m = st.columns(3)
    m[0].metric(tx("Accuracy"), f"{ev['accuracy']:.3f}",
                help=tx("Anteil richtig eingeordneter Texte – naive Vorhersage (immer die häufigste Klasse der "
                        "Trainingsdaten): {b:.3f}", b=ev["baseline"]))
    m[1].metric("F1 (macro)", f"{ev['f1_macro']:.3f}", help=tx("Mittel über alle Klassen – fair bei Ungleichgewicht"))
    m[2].metric("ROC-AUC", f"{ev['auc']:.3f}" if ev.get("auc") is not None else "–",
                help=tx("0.5 = Zufall, 1.0 = perfekt; unabhängig von der Schwelle"))


def classification_block(ev, classes):
    c1, c2 = st.columns([3, 2])
    rep = pd.DataFrame({k: v for k, v in ev["report"].items() if isinstance(v, dict)}).T
    rep["support"] = rep["support"].astype(int)
    c1.dataframe(rep.round(3))
    with c1.expander(tx("classification_report (Text)")):
        st.code(ev["report_text"], language="text")
    with c2:
        show(plots.fig_confusion(ev["cm"], classes))


RIGHT, WRONG = N_(":green[:material/check_circle:] richtig"), N_(":red[:material/cancel:] falsch")


def explanation_block(ex, text, vectorizer, recipe, truth=None):
    """Ein Text mit Vorhersage, Wahrscheinlichkeiten und den Wörtern, die entschieden haben."""
    classes = recipe["classes"]
    m = st.columns(3 if truth is None else 4)
    m[0].metric(tx("Vorhersage"), ex["label"])
    if recipe["task"] == pl.T_BIN:
        m[1].metric(f"P({md_text(classes[1])})", f"{ex['proba'][1]:.1%}")
    else:
        m[1].metric(tx("Sicherheit"), f"{ex['proba'][ex['pred']]:.1%}")
    m[2].metric(tx("bekannte Wörter"), f"{ex['known']} / {ex['tokens']}",
                help=tx("Wörter (nach Kleinschreibung und Stoppwörtern), die im Vokabular stehen – nur sie zählen."))
    if truth is not None:
        m[3].metric(tx("tatsächlich"), classes[truth])
        st.markdown(tx(RIGHT) if truth == ex["pred"] else tx(WRONG))
    st.markdown(highlight_md(text, vectorizer, ex))
    st.caption(tx(":blue[**blau**] = spricht für '{c}', :orange[**orange**] = spricht dagegen (die wichtigsten Wörter "
                  "dieses Textes).", c=md_text(ex["label"])))
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_contrib(ex["words"], ex["label"]))
    with c2:
        if recipe["task"] == pl.T_MULTI:
            st.dataframe(pd.DataFrame({tx("Klasse"): classes, "P": np.round(ex["proba"], 4)}).sort_values(
                "P", ascending=False), hide_index=True, height=min(35 * (len(classes) + 1) + 3, 300))
        st.dataframe(trc(ex["words"]), hide_index=True)


@with_lang
def on_test_example():
    p = ss.get("prep")
    if p is not None:
        i = int(np.random.default_rng().integers(len(p.text_test)))
        ss.w_try_text = str(p.text_test[i])[:5000]


def page_eval():
    st.header(tx("8 · Modellevaluation & wichtige Wörter"))
    if not ready("model", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    p = ss.prep
    r = p.recipe
    task = r["task"]
    scores = test_scores()
    thr = 0.5
    if task == pl.T_BIN:
        init("w_threshold", 0.5)
        thr = st.slider(tx("Entscheidungsschwelle: P({v}) > Schwelle → '{v}'", v=md_text(r['classes'][1])),
                        pl.THRESHOLD_MIN, pl.THRESHOLD_MAX, step=0.01, key="w_threshold",
                        help=tx("Standard 0.5. Höher = strenger für Klasse 1. Wird beim Speichern mitgespeichert."))
    ev = pl.evaluate(p.y_test, scores, task, r["classes"], thr, majority=int(np.bincount(p.y_train).argmax()))
    st.caption(tx("Bewertet auf {n} Testtexten (X_test, y_test) · {name}", n=fmt_int(len(p.y_test)),
                  name=pl.model_name(ss.model_cfg.model)))
    metric_row(ev)
    classification_block(ev, r["classes"])
    if task == pl.T_BIN:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_proba(scores, r["classes"], p.y_test, thr))
        c2.markdown(tx("**Schwelle verstehen:** Das Modell liefert für jeden Text eine Wahrscheinlichkeit für '{v}'. "
                       "Liegt sie über der Schwelle, lautet die Vorhersage '{v}'. Eine höhere Schwelle macht '{v}' "
                       "seltener (weniger Fehlalarme, aber mehr übersehene '{v}'); eine niedrigere erkennt mehr '{v}'.",
                       v=md_text(r['classes'][1])))
    csv_downloads(lambda: pl.predictions_frame(p, scores, r, thr, export_columns()),
                  tx("Testtexte mit Vorhersage als CSV"), tx("vorhersagen_test") + ".csv", (ss.model_token, thr))

    st.subheader(tx("Welche Wörter sprechen für welche Klasse?"))
    kind = "nb" if ss.model_cfg.model == pl.MODEL_NB else "coef"
    st.caption(tx("Naive Bayes: wie viel wahrscheinlicher ein Wort in dieser Klasse ist als in den anderen "
                  "(log P(Wort | Klasse) minus Mittel der anderen). Je größer, desto typischer.") if kind == "nb"
               else tx("Koeffizient des Worts für die Klasse – positiv: das Wort macht die Klasse wahrscheinlicher. Ein "
                       "Wort zählt so oft, wie es im Text vorkommt (bzw. mit seinem TF-IDF-Wert)."))
    tw = cached("_topwords", (ss.model_token, i18n.get_lang()), lambda: pl.top_words(ss.model, p.vectorizer, r, 15))
    show(plots.fig_top_words(tw, r["classes"], 15))
    with st.expander(tx("Tabelle")):
        st.dataframe(trc(pd.concat(dict(tw["classes"]), names=[tx("Klasse"), ""])
                         .reset_index(level=0)), hide_index=True)

    st.subheader(tx("Falsch eingeordnete Texte"))
    st.caption(tx("Die sichersten Fehler zuerst – oft Ironie, gemischte Bewertungen oder ein falsches Label in den "
                  "Daten. Ein Blick darauf zeigt, wo das Modell an seine Grenzen kommt."))
    wrong = pl.misclassified(p, scores, r, thr, 50)
    if wrong.empty:
        st.success(tx("Kein Testtext falsch eingeordnet."))
    else:
        st.dataframe(trc(wrong), hide_index=True, height=320,
                     column_config={"Text": st.column_config.TextColumn(tx("Text"), width="large")})

    st.subheader(tx("Einzelfall: zufälliger Testtext"))
    if st.button(tx("Zufälligen Testtext ziehen"), icon=":material/casino:"):
        ss.case_idx = int(np.random.default_rng().integers(len(p.idx_test)))
    if "case_idx" in ss and ss.case_idx < len(p.idx_test):
        i = ss.case_idx
        ex = pl.explain_text(ss.model, p.vectorizer, r, p.text_test[i], threshold=thr)
        explanation_block(ex, p.text_test[i], p.vectorizer, r, truth=int(p.y_test[i]))

    st.subheader(tx("Eigenen Text testen"))
    init("w_try_text", "")
    with st.form("try_text"):
        st.text_area(tx("Text"), key="w_try_text", height=110,
                     placeholder=tx("z. B. The food was great, but the service was terribly slow."))
        submitted = st.form_submit_button(tx("Einordnen"), type="primary")
    st.button(tx("Beispiel aus den Testdaten einsetzen"), on_click=on_test_example)
    if (submitted or ss.get("_explain")) and ss.w_try_text.strip():
        ex = pl.explain_text(ss.model, p.vectorizer, r, ss.w_try_text, threshold=thr)
        ss._explain = True
        explanation_block(ex, ss.w_try_text, p.vectorizer, r)
    code(cg.eval_code(task, thr, ss.model_cfg, r["classes"]))


# ===========================================================================
# Seite 9 – Speichern & Laden
# ===========================================================================
def session_recipe(threshold=None):
    """Rezept des aktuellen Modells inkl. Training, Kennzahlen und Schwelle."""
    p = ss.prep
    r = p.recipe
    thr = ss.get("w_threshold", 0.5) if threshold is None else threshold
    ev = pl.evaluate(p.y_test, test_scores(), r["task"], r["classes"], thr)
    return pl.finalize_recipe(r, ss.model_cfg, ss.train_info, pl.metrics_summary(ev),
                              thr if r["task"] == pl.T_BIN else 0.5)


def session_pipe():
    return pl.make_pipeline(ss.prep.vectorizer, ss.model)


def page_save():
    st.header(tx("9 · Modell speichern & laden"))
    st.markdown(tx("Gespeichert werden **zwei Dateien** in `models/<name>/`: die scikit-learn-Pipeline aus "
                   "Vektorisierer und Modell (`model.skops` oder `model.joblib`, wie im Kurs: Pipeline([('bow', …), "
                   "('classifier', …)])) und `preprocessing.json` – das Rezept aus Schritt 2 und 4 (Spalten-Aktionen, "
                   "Textspalte, Klassen, Schwelle) mit den Einstellungen aus Schritt 6. Neue Texte werden damit genau so "
                   "umgewandelt wie die Trainingstexte."))
    st.subheader(tx("Speichern"))
    if "model" not in ss:
        st.info(tx("Noch kein trainiertes Modell in dieser Sitzung (Schritt 6)."))
    else:
        r = ss.prep.recipe
        if ss.get("_save_name_for") != ss.get("model_token"):
            ss.w_save_name = (f"{re.sub(r'[^0-9A-Za-z_-]+', '_', r['label'])}_{ss.model_cfg.model}_"
                              f"{datetime.now():%Y%m%d_%H%M}")
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
        init("w_save_fp", True)
        keep_fp = st.checkbox(
            tx("Prüfsummen von Trainingstexten mitspeichern (Schritt 10 erkennt damit Trainingsdaten)"), key="w_save_fp",
            help=tx("Bis zu 5000 Prüfsummen (Hashes) von Trainingstexten. Aus einer Prüfsumme lässt sich der Text "
                    "nicht zurückrechnen – wer einen Text hat, kann aber prüfen, ob er im Training war. Bei "
                    "vertraulichen Texten ausschalten, bevor das Modell weitergegeben wird."))
        st.caption(tx("Hinweis: Das Vokabular (alle Wörter der Trainingstexte) steckt im Modell – bei vertraulichen "
                      "Texten das Modell nur an Berechtigte weitergeben."))
        overwrite = True
        if exists:
            st.warning(tx("Unter `models/{safe}` liegt schon ein Modell – Speichern ersetzt es.", safe=safe))
            overwrite = st.checkbox(tx("Ja, vorhandenes Modell überschreiben"), key=f"overwrite_{safe}")
        if st.button(tx("Speichern"), type="primary", icon=":material/save:", disabled=not overwrite):
            with st.spinner(tx("Speichere …")):
                recipe = session_recipe()
                if not keep_fp:
                    recipe.pop("train_fingerprints", None)
                folder = pl.save_bundle(session_pipe(), recipe, target, fmt)
            ss.saved_folder = str(folder)
            ss.pop("_zip_cache", None)
            st.success(tx("Gespeichert: {folder}", folder=md_text(folder)))
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
        curve = ss.history.get("curve") or {}
        script = cg.full_script(src, ss.actions_fitted, {c for c in raw.columns if pl.is_numeric(raw[c])},
                                r, ss.model_cfg, ss.get("w_threshold", 0.5), code_folder, ss.w_save_fmt,
                                new_columns=ss.get("new_cols_fitted"), curve_stratify=curve.get("stratify", True))
        st.download_button(tx("Komplette Pipeline als Python-Skript (für Jupyter)"), script.encode("utf-8"),
                           file_name="nlp_pipeline.py", mime="text/x-python", icon=":material/download:")
        code(cg.save_code(code_folder, ss.w_save_fmt))

    st.subheader(tx("Laden"))
    bundles = pl.list_bundles(MODELS_DIR)
    t1, t2 = st.tabs([tx("Aus dem Ordner models/"), tx("ZIP hochladen")])
    with t1:
        if not bundles:
            st.caption(tx("Noch keine gespeicherten Modelle."))
        else:
            labels = {b["name"]: tx("{name} · {model} · Label {label} · {v}",
                                    name=b['name'], model=pl.model_name(b.get("model") or "?"), label=b['label'],
                                    v=b['saved_at'][:16].replace('T', ' '))
                      for b in bundles}
            sel = sb(tx("Gespeicherte Modelle"), list(labels), "w_bundle_sel", format_func=labels.get)
            if st.button(tx("Laden"), key="btn_load_bundle"):
                try:
                    pipe, recipe = pl.load_bundle(MODELS_DIR / sel)
                except Exception as exc:  # noqa: BLE001
                    st.error(tx("Laden fehlgeschlagen: {exc}", exc=md_text(exc)))
                else:
                    ss.bundle = {"pipe": pipe, "recipe": recipe, "name": sel, "token": uuid.uuid4().hex}
                    ss.pop("pred_result", None)
                    st.success(tx("'{sel}' geladen → Schritt 10", sel=md_text(sel)))
    with t2:
        up = st.file_uploader(tx("ZIP (model.skops/.joblib + preprocessing.json)"), type=["zip"], key="up_bundle")
        allow_pickle = False
        if up is not None and pl.zip_needs_pickle(up.getvalue()):
            st.warning(tx("Das ZIP enthält ein .joblib-Modell (pickle). Pickle kann beim Laden beliebigen Code ausführen "
                          "– nur Dateien aus vertrauenswürdiger Quelle laden (z. B. eigene)."))
            allow_pickle = st.checkbox(tx("Ich vertraue dieser Datei – .joblib laden"), key="w_zip_pickle")
        if st.button(tx("ZIP laden"), disabled=up is None):
            try:
                pipe, recipe = pl.load_bundle_zip(up.getvalue(), allow_pickle=allow_pickle)
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Laden fehlgeschlagen: {exc}", exc=md_text(exc)))
            else:
                ss.bundle = {"pipe": pipe, "recipe": recipe, "name": Path(up.name).stem, "token": uuid.uuid4().hex}
                ss.pop("pred_result", None)
                st.success(tx("ZIP geladen → Schritt 10"))
    if "bundle" in ss:
        b = ss.bundle
        r = b["recipe"]
        training = r.get("training") or {}
        st.markdown(tx("**Geladen: {name}** · {model} · Text `{text}` → Label `{label}` · {v} · Klassen: {classes} · "
                       "gespeichert {v2}",
                       name=md_text(b['name']), model=pl.model_name(training.get("model") or "?"),
                       text=" ".join(str(r['text_col']).replace("`", "'").splitlines()),
                       label=" ".join(str(r['label']).replace("`", "'").splitlines()),
                       v=tx(pl.TASK_LABELS[r['task']]), classes=", ".join(md_text(c) for c in r["classes"][:12])
                       + (" …" if len(r["classes"]) > 12 else ""),
                       v2=md_text(str(r.get('saved_at', '?')).replace('T', ' '))))
        c1, c2, c3 = st.columns(3)
        c1.markdown(tx("**Kennzahlen (Testdaten)**"))
        c1.json(r.get("metrics", {}))
        c2.markdown(tx("**Training**"))
        c2.json(pl.model_params(training))
        c3.markdown(tx("**Vektorisierer**"))
        c3.json(r.get("vectorizer", {}))


# ===========================================================================
# Seite 10 – Neue Texte einordnen
# ===========================================================================
HOW_TO_10 = N_("""
**So werden neue Texte eingeordnet**

1. **Nichts neu fitten.** Die gespeicherte Pipeline wandelt neue Texte mit dem **Vokabular aus dem Training** in
   Zahlen um – würde man den Vektorisierer neu fitten, passten die Spalten nicht mehr zum Modell. Wörter, die das
   Modell nie gesehen hat, werden ignoriert.
2. **Gleiche Rohspalte wie im Training.** Nötig ist nur die Textspalte (Tabelle unten, Vorlage zum Download);
   Schritt 2 (z. B. Umbenennen) wird automatisch wiederholt. Fehlt sie → klare Fehlermeldung. Weitere Spalten
   werden ignoriert.
3. **Abweichungen werden gemeldet:** leere Texte, Texte ohne ein einziges bekanntes Wort (andere Sprache?) und
   Texte aus überwiegend unbekannten Wörtern – Spalte *Hinweis*.
4. **Interpretation und Fazit:** Bei binär = Wahrscheinlichkeit für die positive Klasse + Schwelle. Ist das Label
   in den neuen Daten enthalten, wird automatisch evaluiert. Das Fazit bewertet die Texte und erklärt, was das
   Ergebnis aussagt – auch, ob Texte schon im Training waren.
""")


def render_interpretation(res, recipe, thr):
    task = recipe["task"]
    n = len(res)
    if task == pl.T_BIN:
        neg, pos = recipe["classes"]
        p = res[f"P({pos})"].to_numpy()
        n_pos = int((res[pl.RESULT_PRED] == pos).sum())
        train_share = recipe.get("label_stats", {}).get("class_share", {}).get(pos)
        txt = (tx("**Lesart:** `P({pos})` ist die vom Modell geschätzte Wahrscheinlichkeit für '{pos}'. Über der "
                  "Schwelle {thr:.2f} → Vorhersage '{pos}', sonst '{neg}'. Werte nahe 0 oder 1 = sicher, um "
                  "{thr:.2f} = unsicher.",
                  pos=md_text(pos), thr=thr, neg=md_text(neg)))
        txt += (tx("\n\n**Ergebnis:** {n} von {n2} Texten ({v:.1%}) werden als '{pos}' eingeordnet, {n3} als "
                   "'{neg}'.",
                   n=fmt_int(n_pos), n2=fmt_int(n), v=n_pos / n, pos=md_text(pos), n3=fmt_int(n - n_pos),
                   neg=md_text(neg)))
        if train_share is not None:
            txt += (tx(" Zum Vergleich: In den Trainingsdaten waren {train_share:.1%} '{pos}'. Einordnung im Fazit "
                       "unten.", train_share=train_share, pos=md_text(pos)))
        st.markdown(txt)
        c1, _ = st.columns([3, 2])
        with c1:
            show(plots.fig_proba(p, recipe["classes"], threshold=thr))
    else:
        st.markdown(tx("**Lesart:** `Vorhersage` = Klasse mit der höchsten Wahrscheinlichkeit, `Sicherheit` = diese "
                    "Wahrscheinlichkeit; `P(…)` je Klasse summieren sich zu 1."))
        if n > 1:
            c1, _ = st.columns([3, 2])
            with c1:
                show(plots.fig_class_counts(res[pl.RESULT_PRED], recipe.get("label_stats", {}).get("class_share")))


LEVEL_ICON = {pl.LEVEL_OK: ":green[:material/check_circle:]", pl.LEVEL_WARN: ":orange[:material/warning:]",
              pl.LEVEL_BAD: ":red[:material/error:]"}


def render_fazit(fz):
    st.subheader(tx("Fazit"))
    with st.container(border=True):
        st.markdown(tx("**Qualität der Texte: {v} {level}**", v=LEVEL_ICON[fz['level']], level=tx(fz['level'])))
        lines = []
        for q in fz["quality"]:
            lines.append(f"- {LEVEL_ICON[q['level']]} {q['text']}")
            lines += [f"    - {d}" for d in q["details"][:12]]
        st.markdown("\n".join(lines))
        if fz["meaning"]:
            st.markdown(tx("**Was das Ergebnis aussagt**"))
            st.markdown("\n".join(f"- {t}" for t in fz["meaning"]))
        if fz["advice"]:
            st.markdown(tx("**Empfehlung**"))
            st.markdown("\n".join(f"- {t}" for t in fz["advice"]))


def page_predict():
    st.header(tx("10 · Neue Texte einordnen"))
    options = {}
    if "model" in ss:
        options["session"] = tx("Aktuelles Modell (diese Sitzung)")
    if "bundle" in ss:
        options["bundle"] = tx("Geladenes Modell: {name}", name=md_text(ss.bundle['name']))
    if not options:
        st.info(tx("Kein Modell vorhanden – in Schritt 6 trainieren oder in Schritt 9 laden."))
        st.markdown(tx(HOW_TO_10))
        return
    src = rd(tx("Modell"), list(options), "w_pred_model", format_func=options.get, horizontal=True)
    if src == "session":
        pipe, recipe = session_pipe(), session_recipe()
    else:
        pipe, recipe = ss.bundle["pipe"], ss.bundle["recipe"]
    task = recipe["task"]
    with st.expander(tx("So funktioniert Schritt 10"), icon=":material/info:"):
        st.markdown(tx(HOW_TO_10))

    thr = recipe.get("threshold", 0.5)
    if task == pl.T_BIN:
        model_id = f"{src}-{ss.get('model_token') if src == 'session' else ss.bundle.get('token')}"
        if ss.get("_pred_thr_for") != model_id:
            ss.w_pred_thr = float(thr)
            ss._pred_thr_for = model_id
        thr = st.slider(tx("Schwelle: P({v}) > Schwelle → '{v}'", v=md_text(recipe["classes"][1])),
                        pl.THRESHOLD_MIN, pl.THRESHOLD_MAX,
                        step=0.01, key="w_pred_thr")

    mode = rd(tx("Eingabe"), [N_("Einzelner Text"), N_("CSV-Datei")], "w_pred_mode", horizontal=True, format_func=tx)
    if mode == "Einzelner Text":
        init("w_pred_text", "")
        with st.form("single_text"):
            st.text_area(tx("Text"), key="w_pred_text", height=130,
                         placeholder=tx("Neuen Text eingeben, z. B. eine Bewertung oder eine Nachricht …"))
            submitted = st.form_submit_button(tx("Einordnen"), type="primary")
        if submitted and not ss.w_pred_text.strip():
            st.warning(tx("Bitte einen Text eingeben."))
        elif submitted or (ss.get("_pred_text_for") == (src, ss.w_pred_text) and ss.w_pred_text.strip()):
            ss._pred_text_for = (src, ss.w_pred_text)
            ex = pl.classify_text(pipe, recipe, ss.w_pred_text, threshold=thr)
            explanation_block(ex, ss.w_pred_text, pipe.steps[0][1], recipe)
            if ex["known"] == 0:
                st.warning(tx("Kein Wort dieses Textes steht im Vokabular des Modells – die Vorhersage folgt nur der "
                              "Häufigkeit der Klassen im Training (andere Sprache?)."))
    else:
        st.subheader(tx("Benötigte Spalten"))
        st.dataframe(pl.required_columns(recipe), hide_index=True)
        st.download_button(tx("CSV-Vorlage (nur Kopfzeile)"), pl.template_csv(recipe), tx("vorlage_neue_texte.csv"),
                           mime="text/csv", icon=":material/download:")
        meta_header = (recipe.get("meta") or {}).get("header")
        source, name = csv_source("pred")
        csv_options()
        if meta_header is False:
            st.caption(tx("Das Modell wurde mit einer Datei ohne Kopfzeile trainiert – für neue Daten dieselbe Form "
                          "verwenden (Kopfzeile: nein) oder eine CSV mit den Spaltennamen aus der Tabelle oben."))
        if st.button(tx("Vorhersage berechnen"), type="primary", disabled=source is None):
            try:
                with st.spinner(tx("Wandle die neuen Texte um und ordne sie ein …")):
                    df_new, _ = read_source(source, header_choice())
                    res, rep, ev = pl.predict_texts(pipe, recipe, df_new, thr)
            except pl.MissingColumnsError as exc:
                st.error(tx("{exc}. Die CSV braucht dieselbe Textspalte wie die Trainingsdaten (siehe Tabelle oben / "
                            "CSV-Vorlage).", exc=md_text(exc)))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=md_text(exc)))
            else:
                ss.pred_result = {"res": res, "rep": rep, "ev": ev, "name": name, "thr": thr, "src": src,
                                  "cols": list(df_new.columns), "token": uuid.uuid4().hex}
        pr = ss.get("pred_result")
        if pr and pr["src"] == src:
            res, rep, ev = pr["res"], pr["rep"], pr["ev"]
            fz = pl.assess_prediction(recipe, res, rep, ev, pr["thr"])
            st.success(tx("{n} Texte eingeordnet ({name})", n=fmt_int(len(res)), name=md_text(pr['name'])))
            if fz["level"] != pl.LEVEL_OK or (rep.get("train_overlap") or {}).get("share", 0) >= 0.05:
                st.caption(tx("{v} Hinweise zur Qualität der Texte und Einordnung → **Fazit** unten",
                              v=LEVEL_ICON[fz['level']]))
            render_interpretation(res, recipe, pr["thr"])
            if ev is not None:
                st.subheader(tx("Label '{label}' ist enthalten → Evaluation auf {n} Texten",
                                label=md_text(recipe['label']), n=fmt_int(ev['n'])))
                metric_row(ev)
                classification_block(ev, recipe["classes"])
            st.subheader(tx("Ergebnis"))
            replaced = replaced_columns(pr.get("cols", []), recipe)  # je Sprache neu (übersetzte Namen)
            if replaced:
                st.info(tx("Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis "
                           "werden sie ersetzt: {cols}", cols=", ".join(md_text(c) for c in replaced)))
            view = result_view(res)
            lead = [c for c in view.columns if str(c).startswith("P(") or c in [tx(n) for n in RESULT_COLUMNS]]
            st.dataframe(view[lead + [c for c in view.columns if c not in lead]].head(1000))
            csv_downloads(lambda: view, tx("Ergebnis als CSV"), tx("vorhersagen_{stem}.csv", stem=Path(pr['name']).stem),
                          pr.get("token"))
            render_fazit(fz)
    code(cg.predict_code("models/<name>", recipe), tx("Python-Code (Vorhersage ohne App)"))


# ===========================================================================
# Seitenleiste & Navigation
# ===========================================================================
PAGES = {
    N_("1 · Daten & Bibliotheken"): page_data,
    N_("2 · Aufbereitung 1 – Spalten"): page_prep1,
    N_("3 · Analyse & Visualisierung"): page_viz,
    N_("4 · Aufbereitung 2 – Text → Zahlen"): page_prep2,
    N_("5 · Daten speichern"): page_export,
    N_("6 · Modell & Training"): page_train,
    N_("7 · Training vs. Validierung"): page_curve,
    N_("8 · Evaluation & Wörter"): page_eval,
    N_("9 · Modell speichern & laden"): page_save,
    N_("10 · Neue Texte einordnen"): page_predict,
}

with st.sidebar:
    st.markdown("### :material/translate: DATA-Analysis-NLP")
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
        status("prep" in ss, tx("Texte vektorisiert: X_train {shape}", shape=ss.prep.X_train.shape) if "prep" in ss
               else tx("Texte vektorisiert"))
        status("model" in ss, tx("Modell trainiert: {name}", name=pl.model_name(ss.model_cfg.model)) if "model" in ss
               else tx("Modell trainiert"))
        status("bundle" in ss, tx("Geladen: {name}", name=md_text(ss.bundle['name'])) if "bundle" in ss
               else tx("Modell geladen (Schritt 9)"))
    with st.expander(tx("Konfiguration speichern / laden"), icon=":material/settings:"):
        st.caption(tx("Alle Einstellungen (Schritt 2 inkl. neuer Spalten, 4, 6) als JSON – für geänderte Daten mit "
                   "gleichen Spalten."))
        st.download_button(tx("Exportieren (JSON)"), export_config(), file_name="nlp_template_config.json",
                           mime="application/json")
        up_cfg = st.file_uploader(tx("Importieren"), type=["json"], key="up_cfg")
        st.button(tx("Konfiguration übernehmen"), disabled=up_cfg is None, on_click=on_import_config)
        msg = ss.pop("_cfg_msg", None)
        if msg:
            (st.error if msg[0] == "error" else st.success)(msg[1])
