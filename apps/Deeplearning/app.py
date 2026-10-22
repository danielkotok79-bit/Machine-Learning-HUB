"""
app.py – DATA-Analysis-Deeplearning: Datenanalyse und Deep Learning (Keras) für tabellarische CSV-Daten
==================================================================================================

Start:   streamlit run app.py

Schritte (Navigation links)
    1  Daten laden & Bibliotheken          6  Modell & Training
    2  Aufbereitung 1 – Spalten, neue Sp.  7  loss vs. val_loss
    3  Analyse & Visualisierung            8  Evaluation
    4  Aufbereitung 2 – ML-Daten           9  Modell speichern & laden
    5  Daten speichern (CSV)              10  Vorhersage auf neuen Daten + Fazit
    Stichprobe für langsame Rechner: Option in der Seitenleiste

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

st.set_page_config(page_title="DATA-Analysis-Deeplearning", page_icon=":material/analytics:", layout="wide")

import i18n  # noqa: E402
from i18n import N_, tx  # noqa: E402

# Sprache vor allen Texten festlegen: Auswahl in der Seitenleiste, Start über ?lang=…, --lang oder DL_TEMPLATE_LANG
os.environ.setdefault("DL_TEMPLATE_LANG", "en")  # Startsprache Englisch
i18n.set_lang(st.session_state.get("w_lang") or i18n.start_lang(st.query_params.get("lang")))
# Gewählte Sprache in die Adresse übernehmen, damit Neuladen der Seite sie behält (Deutsch ohne ?lang bleibt ohne)
if st.query_params.get("lang") != i18n.get_lang() and (i18n.get_lang() != "de" or "lang" in st.query_params):
    st.query_params["lang"] = i18n.get_lang()

with st.spinner(tx("Lade Bibliotheken (pandas, scikit-learn, TensorFlow) …")):
    import matplotlib.pyplot as plt
    import tensorflow as tf

    import codegen as cg
    import pipeline as pl  # 1) Import der Bibliotheken – passiert hier automatisch
    import plots

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = Path(os.environ.get("DL_TEMPLATE_MODELS", APP_DIR / "models"))  # Speicherort für Schritt 9
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
    "model": ["model", "history", "train_info", "model_cfg", "model_token"],
    "eval": ["eval_scores", "importance", "case_idx", "pred_result", "saved_folder"],
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


RESULT_COLUMNS = (N_("Vorhersage"), N_("Sicherheit"), N_("Hinweis"))
TABLE_HEADERS = (  # Spaltennamen der Anzeige-Tabellen – hier nur für den Übersetzungskatalog, übersetzt in trc()
    N_("Spalte"), N_("Typ"), N_("Eindeutig"), N_("Fehlend"), N_("Fehlend %"), N_("Beispiel"), N_("Neue Spalte"),
    N_("Art"), N_("Definition"), N_("Aktion"), N_("Parameter"), N_("Wirkung"), N_("Anzahl"), N_("Anteil"),
    N_("Rolle"), N_("Modell-Eingänge"), N_("Strategie"), N_("Klasse"), N_("Kodiert"), N_("Train"), N_("Test"),
    N_("Anteil Train"), N_("Kategorie-Spalte"), N_("Kategorien"), N_("weggelassen (drop_first)"),
    N_("neue 0/1-Spalten"), N_("Beispiele"), N_("Feature"), N_("Wichtigkeit"), N_("Streuung"),
)


def result_view(res):
    """Ergebnis aus Schritt 10 zur Anzeige und zum Download: Spaltennamen und Hinweise in der gewählten Sprache.
    Hat die neue CSV schon eine Spalte mit dem übersetzten Namen (z. B. 'Prediction' aus einem früheren Export),
    ersetzt das Ergebnis sie – wie auf Deutsch bei 'Vorhersage' (sonst doppelte Spaltennamen)."""
    if i18n.get_lang() == "de":
        return res
    out = res.copy()
    if "Hinweis" in out.columns:
        out["Hinweis"] = out["Hinweis"].map(
            lambda h: "; ".join(tx(part) for part in h.split("; ")) if isinstance(h, str) and h else h)
    names = {c: tx(c) for c in RESULT_COLUMNS if c in out.columns}
    clash = [t for c, t in names.items() if t != c and t in out.columns]
    return out.drop(columns=clash).rename(columns=names)


def replaced_columns(columns, recipe):
    """Spalten der neuen Daten, die das Ergebnis aus Schritt 10 ersetzt (gleicher Name wie eine Ergebnisspalte –
    intern oder in der gewählten Sprache, z. B. aus einem früheren Export)."""
    task, classes = recipe["task"], recipe.get("classes") or []
    made = [N_("Vorhersage"), N_("Hinweis")] + ([N_("Sicherheit")] if task == pl.T_MULTI else [])
    made += [] if task == pl.T_REG else [f"P({c})" for c in (classes[1:] if task == pl.T_BIN else classes)]
    names = set(made) | {tx(c) for c in made}
    return [c for c in columns if c in names]


THIN = "\u202f"  # schmales Leerzeichen als Tausendertrennung (eindeutig neben dem Dezimalpunkt)


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


def guess_label(df):
    names = {c.lower(): c for c in df.columns}
    for cand in ["label", "target", "y", "class", "klasse", "loan_status", "loan_repaid", "price", "preis",
                 "species"]:
        if cand in names:
            return names[cand]
    return df.columns[-1]


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
        for k in ["w_viz_cols", "w_label", "w_features", "_task_for_label", "_features_for_label", "ed_missing__sig",
                  "new_cols"]:
            ss.pop(k, None)
    editor_reset("ed_actions", prof)
    st.rerun()


WIDGET_KEYS = [  # Einstellungen, die exportiert/importiert werden
    "w_viz_cols", "w_label", "w_task", "w_pos", "w_features", "w_test_size", "w_split_seed", "w_stratify",
    "w_scaler", "w_drop_first", "w_scale_y", "w_preset", "w_out_act", "w_loss", "w_opt", "w_lr", "w_epochs",
    "w_batch", "w_es", "w_patience", "w_restore", "w_val", "w_val_split", "w_cw", "w_seed", "w_maxnorm", "w_tb",
    "w_threshold",
]


def export_config() -> bytes:
    def records(name, cols):
        df = editor_value(name)
        return None if df is None else df[[c for c in cols if c in df.columns]].to_dict("records")

    cfg = {
        "template_config": 1,
        "exported": datetime.now().isoformat(timespec="seconds"),
        "source": ss.get("source", {}).get("name"),
        "step2": records("ed_actions", ["Spalte", "Aktion", "Parameter", "Neuer Name"]),
        "missing": records("ed_missing", ["Spalte", "Strategie", "Parameter"]),
        "layers": records("ed_layers", ["Neuronen", "Aktivierung", "Dropout"]),
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
    if cfg.get("layers"):
        editor_reset("ed_layers", pd.DataFrame(cfg["layers"]))
    if "new_columns" in cfg:
        ss.new_cols = cfg["new_columns"]
    for k, v in cfg["widgets"].items():
        if k in WIDGET_KEYS:
            ss[k] = v
    if skipped:
        ss._cfg_skipped = skipped
    if "w_label" in ss:
        ss._task_for_label = ss._features_for_label = ss.w_label


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
        versions, gpus = cached("_versions", i18n.get_lang(), pl.library_versions)
        c2.dataframe(versions, hide_index=True)
        c2.caption(tx("GPU: {gpus} gefunden", gpus=gpus) if gpus
                   else tx("GPU: keine gefunden – Training läuft auf der CPU"))

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
    """Entsteht col aus derselben CSV-Spalte wie das Label? (dann wäre es als Feature ein Datenleck)"""
    lin = ss.get("lineage", {})
    return bool(label) and bool(set(pl.raw_sources(lin, col)) & set(pl.raw_sources(lin, label)))


def auto_select(old_cols, out):
    """Neu entstandene Spalten automatisch für Visualisierung und Features auswählen (nicht, wenn sie aus
    derselben Quelle wie das Label stammen)."""
    new_cols = [c for c in out.columns if c not in old_cols]
    if not old_cols or not new_cols:
        return
    if isinstance(ss.get("w_viz_cols"), list):
        ss.w_viz_cols = [c for c in out.columns if c in ss.w_viz_cols or c in new_cols]
    if isinstance(ss.get("w_features"), list):
        label = ss.get("w_label")
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


def page_prep1():
    st.header(tx("2 · Datenaufbereitung 1 – Spalten umwandeln (object → Zahl / Kategorie)"))
    if not ready("df_raw", tx("Bitte zuerst in Schritt 1 eine CSV laden.")):
        return
    df = ss.df_raw
    st.markdown(tx("Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann "
                "**Übernehmen**. Die Vorschläge sind automatisch – bitte prüfen: Spalten, die das Ergebnis schon "
                "verraten (z. B. erst später erfasste Daten), sind ein Datenleck und sollten entfernt werden. "
                "Text-Spalten mit *behalten* werden in Schritt 4 zu Dummies."))
    lc_cols = {"loan_status", "sub_grade", "address", "earliest_cr_line"}
    example_cfg = CONFIG_DIR / "lending_club_beispiel.json"
    if lc_cols <= set(df.columns) and example_cfg.exists():
        c1, c2 = st.columns([4, 1])
        c1.info(tx("LendingClub-Datensatz erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und 6 passend ein "
                "(78 Features, 78 → 39 → 19, Dropout 0.2, 25 Epochen)."))
        c2.button(tx("Beispiel-Konfiguration laden"), on_click=on_example_config, args=(str(example_cfg),))
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
                t = sb(tx("Spalte"), targets, "w_ct_t", default=guess_label(df) if guess_label(df) in targets else None)
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
                       default=guess_label(df) if guess_label(df) in binary else None)
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
                         default=guess_label(df) if guess_label(df) in few else None)
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


def page_prep2():
    st.header(tx("4 · Datenaufbereitung 2 – fehlende Werte, Label & Features"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    df = ss.df_viz
    cols = list(df.columns)
    lin = ss.get("lineage", {})
    nun = cached("_nunique", ss.viz_token, df.nunique)

    st.subheader(tx("Label (y) & Problemtyp"))
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        label = sb(tx("Label – was soll vorhergesagt werden?"), cols, "w_label", default=guess_label(df))
    if ss.get("_task_for_label") != label:
        ss.w_task = pl.suggest_task(df[label])
        ss._task_for_label = label
    with c2:
        task = rd(tx("Problemtyp"), [pl.T_BIN, pl.T_MULTI, pl.T_REG], "w_task",
                  format_func=lambda t: tx(pl.TASK_LABELS[t]))
    err = pl.validate_task(df[label], task)
    pos = None
    if task == pl.T_BIN and not err:
        vals = pl.label_classes(df[label], pl.T_BIN)
        with c3:
            pos = sb(tx("Positive Klasse (= 1)"), vals, "w_pos", default=vals[-1],
                     help=tx("Diese Klasse wird zu 1 kodiert, die andere zu 0 "
                             "(z. B. 'Fully Paid' → 1, 'Charged Off' → 0)."))
    if err:
        st.error(err)
    with st.expander(tx("Verteilung von {label}", label=label)):
        if task == pl.T_REG:
            st.dataframe(df[label].describe().to_frame().T)
        else:
            vc = df[label].value_counts(dropna=False).head(30).rename("Anzahl").to_frame()
            vc["Anteil"] = (vc["Anzahl"] / len(df)).map("{:.2%}".format)
            st.dataframe(trc(vc))

    st.subheader(tx("Features (X)"))
    candidates = [c for c in cols if c != label]
    leak = [c for c in candidates if shares_source(c, label)]
    if ss.get("_features_for_label") != label:
        if isinstance(ss.get("w_features"), list):
            ss.w_features = [c for c in ss.w_features if c != label]
        else:  # fehlt oder unbrauchbar → Vorschlag verwenden
            ss.pop("w_features", None)
        ss._features_for_label = label
    features = ms(tx("Features für das Modell"), candidates, "w_features",
                  default=[c for c in candidates if c not in leak])
    if leak:
        st.warning(tx("{leak} stammt aus derselben CSV-Spalte wie das Label → nicht als Feature verwenden "
                      "(Datenleck).",
                      leak=leak))
    init("w_drop_first", True)
    rows = []
    for c in cols:
        role = N_("Label (y)") if c == label else (N_("Feature (X)") if c in features else "–")
        is_num = pl.is_numeric(df[c])
        inputs = 0
        if role == "Feature (X)":
            inputs = 1 if is_num else max(int(nun[c]) - (1 if ss.w_drop_first else 0), 0)
        kind = N_("Zahl") if is_num else (N_("Kategorie") if role == "Label (y)" else N_("Kategorie → Dummies"))
        rows.append({"Spalte": c, "Rolle": role, "Typ": kind,
                     "Eindeutig": int(nun[c]), "Fehlend": int(df[c].isna().sum()), "Modell-Eingänge": inputs})
    overview = pd.DataFrame(rows)
    n_inputs = int(overview["Modell-Eingänge"].sum())
    with st.expander(tx("Alle Parameter – {n} Features → ca. {n_inputs} Modell-Eingänge",
                        n=len(features), n_inputs=n_inputs), expanded=True):
        st.dataframe(trc(overview, values=["Rolle", "Typ"]), hide_index=True, height=min(35 * (len(rows) + 1) + 3, 600))
    many = [f"{r['Spalte']} ({r['Eindeutig']})" for r in rows
            if r["Typ"] != "Zahl" and r["Rolle"] == "Feature (X)" and r["Eindeutig"] > 50]
    if many:
        st.warning(tx("Viele Kategorien → viele Dummy-Spalten: {many}. Tipp: Schritt 2 'Top-N' oder entfernen.",
                      many=', '.join(many)))

    st.subheader(tx("Fehlende Werte (NULL)"))
    st.caption(tx("Die Füllwerte werden erst **nach** dem Train/Test-Split berechnet – nur aus den Trainingsdaten – "
               "und dann auf Train, Test und später neue Daten angewendet. So fließt keine Information aus den "
               "Testdaten ins Modell (kein Datenleck). Dasselbe gilt für die Kategorien (Dummies) und den Scaler."))
    if df[label].isna().any():
        st.info(tx("{n} Zeilen ohne Label werden automatisch entfernt.", n=int(df[label].isna().sum())))
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
                corr = corr.drop([c for c in corr.index if shares_source(c, label) or c == label], errors="ignore")
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

    st.subheader(tx("Train/Test-Split & Skalierung"))
    init("w_scale_y", True)
    split_cols = st.columns([1.3, 1.1, 1, 1.2, 1.1, 1.1] if task == pl.T_REG else [1.3, 1.1, 1, 1.2, 1.1])
    c1, c2, c3, c4, c5 = split_cols[:5]
    with c1:
        init("w_test_size", 0.33)
        st.slider("test_size", 0.05, 0.5, step=0.01, key="w_test_size")
    with c2:
        init("w_split_seed", pl.RANDOM_STATE)
        st.number_input("random_state", 0, 1_000_000, step=1, key="w_split_seed")
    with c3:
        init("w_stratify", True)
        st.checkbox("stratify=y", key="w_stratify", disabled=task == pl.T_REG,
                    help=tx("Klassenverhältnis in Train und Test gleich halten (empfohlen bei unausgewogenen Klassen)"))
    with c4:
        sb(tx("Skalierung"), pl.SCALERS, "w_scaler", default="MinMaxScaler", format_func=tx)
    with c5:
        st.checkbox(tx("drop_first (Dummies)"), key="w_drop_first")
    if task == pl.T_REG:
        with split_cols[5]:
            st.checkbox(tx("Label skalieren"), key="w_scale_y",
                        help=tx("y wird standardisiert, Vorhersagen automatisch zurückgerechnet. Große Werte "
                             "(z. B. Preise) lernen sonst sehr langsam, und loss wird lesbar."))

    if "model" in ss:
        st.warning(tx("Neu vorbereiten verwirft das trainierte Modell – vorher in Schritt 9 speichern."))
    if st.button(tx("Daten für ML vorbereiten"), type="primary", disabled=bool(err) or not features):
        missing_cfg = {}
        if miss_table is not None:
            missing_cfg = {r["Spalte"]: {"strategy": r["Strategie"], "param": pl._clean(r["Parameter"])}
                           for r in miss_table.to_dict("records")}
        try:
            with st.spinner(tx("Bereite Daten vor …")):
                prep = pl.prepare_ml_data(
                    df, label, features, task, positive_class=pos, missing=missing_cfg,
                    drop_first=ss.w_drop_first, scaler=ss.w_scaler, test_size=float(ss.w_test_size),
                    random_state=int(ss.w_split_seed), stratify=bool(ss.w_stratify),
                    column_actions=ss.actions_fitted, lineage=lin, meta=source_meta(),
                    scale_label=bool(ss.get("w_scale_y", True)) and task == pl.T_REG,
                    new_columns=ss.get("new_cols_fitted", []))
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
    if (r["label"], r["task"]) != (label, task) or used != set(features):
        st.info(tx("Die Einstellungen oben wurden seit dem Vorbereiten geändert → erneut **Daten für ML vorbereiten**."))
    st.success(tx("Fertig: X_train {shape} · X_test {shape2} · {n} Features → {n2} Modell-Eingänge",
                  shape=p.X_train.shape, shape2=p.X_test.shape, n=len(r['features']), n2=len(r['feature_columns'])))
    for line in p.log:
        st.markdown(f"- {line}")
    for w in p.warnings:
        st.warning(w)
    if r["task"] != pl.T_REG:
        dist = pd.DataFrame({
            "Klasse": r["classes"],
            "Kodiert": range(len(r["classes"])),
            "Train": np.bincount(p.y_train.astype(int), minlength=len(r["classes"])),
            "Test": np.bincount(p.y_test.astype(int), minlength=len(r["classes"])),
        })
        dist["Anteil Train"] = (dist["Train"] / dist["Train"].sum()).map("{:.1%}".format)
        st.dataframe(trc(dist), hide_index=True)
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
            st.caption(tx("Jede Kategorie wird zu einer eigenen 0/1-Spalte, z. B. sub_grade → sub_grade_A2, "
                          "sub_grade_A3 … Mit drop_first entfällt die erste Kategorie – sie ist daran erkennbar, dass "
                          "alle anderen 0 sind. Die Kategorien stammen aus den Trainingsdaten; unbekannte Werte → "
                          "alle 0."))
    st.markdown(tx("**X nach Dummy-Kodierung (unskaliert, erste Zeilen)**"))
    st.dataframe(p.X_preview)
    st.markdown(tx("**X_train nach {type} (erste Zeilen)**", type=r['scaler']['type']))
    st.dataframe(pd.DataFrame(p.X_train[:5], columns=r["feature_columns"]).round(4))
    code(cg.step4_code(r))


# ===========================================================================
# Seite 5 – Daten speichern
# ===========================================================================
EXPORTS = {
    "step2": tx("Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)"),
    "ml": tx("ML-fertige Daten nach Schritt 4 – Lücken gefüllt, Dummies, Label kodiert, Spalte 'split' (train/test)"),
    "ml_scaled": tx("wie zuvor, zusätzlich skaliert – so sieht das Modell die Daten"),
}
FORMATS = {"intl": tx("CSV international (Komma, Dezimalpunkt)"),  # Excel: Format der gewählten Sprache/Region
           "excel": tx("CSV für deutsches Excel (Semikolon, Dezimalkomma)")}


def export_frame(what):
    if what == "step2":
        return ss.df_viz
    return pl.ml_ready_frame(ss.df_viz, ss.prep, scaled=what == "ml_scaled")


def page_export():
    st.header(tx("5 · Daten speichern"))
    if not ready("df_viz", tx("Bitte zuerst Schritt 2 übernehmen.")):
        return
    st.markdown(tx("Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen "
                "Programm."))
    what = rd(tx("Was speichern?"), list(EXPORTS), "w_exp_what", format_func=EXPORTS.get)
    fmt = rd(tx("Format"), list(FORMATS), "w_exp_fmt", format_func=FORMATS.get, horizontal=True)
    if what != "step2" and "prep" not in ss:
        st.info(tx("Für ML-fertige Daten zuerst Schritt 4 ausführen (**Daten für ML vorbereiten**)."))
        return
    if ss.get("w_use_sample"):
        st.warning(tx("Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.",
                      n=fmt_int(len(ss.df_raw)), n2=fmt_int(ss.source['rows'])))
    if what != "step2":
        r = ss.prep.recipe
        if r["task"] != pl.T_REG:
            st.caption(tx("Label kodiert: ") + ", ".join(f"{c} → {i}" for i, c in enumerate(r["classes"])))
    excel = i18n.get_lang() if fmt == "excel" else None  # 'de'/'pl': ; und Dezimalkomma · 'en': , und Punkt
    # Sprache im Schlüssel: nach einem Sprachwechsel neu erstellen – sonst bliebe der Dateiname in der alten Sprache
    key = (what, fmt, excel, i18n.get_lang(), ss.viz_token, ss.get("prep_token") if what != "step2" else None)
    stem = {"step2": tx("daten_aufbereitet"), "ml": tx("ml_daten"), "ml_scaled": tx("ml_daten_skaliert")}[what]
    name = f"{Path(ss.source['name']).stem}_{stem}.csv"
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
        notes = [tx("Für Vorhersagen (Schritt 10) die **ursprüngliche** CSV-Datei verwenden, nicht diese Datei – "
                 "das gespeicherte Modell wiederholt die Aufbereitung selbst.")]
        if excel:
            notes.append(tx("Texte, die mit =, +, -, @ beginnen, bekommen ein ' davor, damit Excel sie nicht als "
                         "Formel ausführt."))
        st.caption(" ".join(notes))
    label = ss.prep.recipe["label"] if "prep" in ss else "label"
    split_col = pl.split_column_name(ss.prep.recipe["feature_columns"] + [label]) if "prep" in ss else "split"
    code(cg.export_code("step2" if what == "step2" else "ml", False, label, scaled=what == "ml_scaled",
                        split_col=split_col, excel=excel))


# ===========================================================================
# Seite 6 – Modell & Training
# ===========================================================================
class StreamlitProgress(tf.keras.callbacks.Callback):
    """Fortschrittsbalken + Live-Kurve während model.fit."""

    def __init__(self, epochs, bar, chart, every=0.4):
        super().__init__()
        self.epochs, self.bar, self.chart, self.every = epochs, bar, chart, every
        self.rows, self.last = [], 0.0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.rows.append({"loss": logs.get("loss"), "val_loss": logs.get("val_loss")})
        now = time.time()
        if now - self.last >= self.every or epoch + 1 == self.epochs:
            self.last = now
            self.bar.progress(min((epoch + 1) / self.epochs, 1.0),
                              text=tx("Epoche {v}/{epochs} · loss {v2:.4g} · val_loss {v3:.4g}",
                                      v=epoch + 1, epochs=self.epochs, v2=logs.get('loss', np.nan),
                                      v3=logs.get('val_loss', np.nan)))
            hist = pd.DataFrame(self.rows, index=pd.RangeIndex(1, len(self.rows) + 1, name=tx("Epoche")))
            self.chart.line_chart(hist, color=[plots.SERIES[0], plots.SERIES[1]], height=260)


def page_train():
    st.header(tx("6 · Modell & Training"))
    if not ready("prep", tx("Bitte zuerst Schritt 4 ausführen (Daten für ML vorbereiten).")):
        return
    p = ss.prep
    r = p.recipe
    task = r["task"]
    n_feat, n_out = p.X_train.shape[1], pl.n_outputs(r)
    st.info(tx("**{v}** · {n_feat} Eingänge · Ausgabeschicht: {n_out} Neuron(en) · {n} Trainingszeilen",
               v=tx(pl.TASK_LABELS[task]), n_feat=n_feat, n_out=n_out, n=fmt_int(len(p.y_train))))

    st.subheader(tx("Hidden Layers"))
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        preset = sb(tx("Vorlage"), list(pl.PRESETS), "w_preset", default=pl.default_preset(task), format_func=tx)
    with c2:
        load_preset = st.button(tx("Vorlage laden"))
    if load_preset or "ed_layers__base" not in ss:
        editor_reset("ed_layers", pd.DataFrame(pl.PRESETS[preset](n_feat)))
    layer_cfg = {
        "Neuronen": st.column_config.NumberColumn(tx("Neuronen (units)"), min_value=1, max_value=pl.MAX_UNITS, step=1,
                                                  required=True),
        "Aktivierung": st.column_config.SelectboxColumn(tx("Aktivierung"), options=pl.HIDDEN_ACTIVATIONS, required=True),
        "Dropout": st.column_config.NumberColumn(tx("Dropout danach"), min_value=0.0, max_value=0.9, step=0.05,
                                                 format="%.2f"),
    }
    layers_df = editor("ed_layers", None, column_config=layer_cfg, num_rows="dynamic", hide_index=True)
    st.caption(tx("Zeilen hinzufügen/löschen = Anzahl Hidden Layers · Dropout 0 = kein Dropout-Layer"))
    layer_error = None
    try:
        layers = pl.layers_from_table(layers_df)
    except ValueError as exc:
        layers, layer_error = [], str(exc)
        st.error(layer_error)

    st.subheader(tx("Ausgabeschicht & model.compile"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        out_act = sb(tx("Aktivierung letzte Schicht"), pl.OUTPUT_ACTIVATIONS[task], "w_out_act",
                     format_func=lambda a: tx("keine (linear)") if a == "linear" else a,
                     help=tx("binär: sigmoid · mehrere Klassen: softmax · Regression: keine"))
    with c2:
        loss = sb("loss", pl.LOSSES[task], "w_loss",
                  format_func=lambda x: f"{pl.LOSS_SHORT[x]} ({x})" if x in pl.LOSS_SHORT else x)
    with c3:
        optimizer = sb("optimizer", pl.OPTIMIZERS, "w_opt")
    with c4:
        init("w_lr", 0.001)
        lr = st.number_input("learning_rate", 0.00001, 1.0, step=0.0005, format="%.5f", key="w_lr")

    st.subheader(tx("model.fit & EarlyStopping"))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        init("w_epochs", 600)
        epochs = st.number_input("epochs", 1, 10000, step=10, key="w_epochs")
    with c2:
        init("w_batch", 256)
        batch = st.number_input("batch_size", 1, 65536, step=32, key="w_batch")
    with c3:
        init("w_es", True)
        es = st.checkbox("EarlyStopping", key="w_es")
    with c4:
        init("w_patience", 25)
        patience = st.number_input("patience", 1, 1000, step=1, key="w_patience", disabled=not es)
    with c5:
        init("w_restore", True)
        restore = st.checkbox("restore_best_weights", key="w_restore", disabled=not es,
                              help=tx("Am Ende die Gewichte der besten Epoche zurückholen (empfohlen)"))
    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
    with c1:
        val = rd(tx("Validierung"), ["test", "split"], "w_val", horizontal=True,
                 format_func={"test": tx("Testdaten"), "split": tx("Anteil aus Training (sauberer)")}.get,
                 help=tx("Testdaten: validation_data=(X_test, y_test) – einfach, aber die Testdaten steuern dann "
                      "EarlyStopping mit. Sauberer: eigener Validierungsanteil aus den Trainingsdaten, damit das "
                      "Testset bis Schritt 8 unberührt bleibt."))
    with c2:
        init("w_val_split", 0.2)
        val_split = st.slider("validation_split", 0.05, 0.4, step=0.05, key="w_val_split", disabled=val != "split")
    with c3:
        init("w_cw", False)
        cw = st.checkbox("class_weight='balanced'", key="w_cw", disabled=task == pl.T_REG,
                         help=tx("class_weight='balanced' – hilft der seltenen Klasse (z. B. Charged Off)"))
    with c4:
        init("w_seed", pl.RANDOM_STATE)
        seed = st.number_input(tx("Seed"), 0, 1_000_000, step=1, key="w_seed")
    with st.expander(tx("Erweitert")):
        c1, c2 = st.columns(2)
        with c1:
            init("w_maxnorm", 0.0)
            maxnorm = st.number_input(tx("kernel_constraint=max_norm(…) – 0 = aus"), 0.0, 10.0, step=0.5,
                                      key="w_maxnorm")
        with c2:
            init("w_tb", False)
            tb = st.checkbox(tx("TensorBoard-Logs schreiben (logs/fit)"), key="w_tb")
            if tb:
                st.code("tensorboard --logdir logs/fit", language="bash")

    cfg = pl.ModelConfig(layers=layers, output_activation=out_act, loss=loss, optimizer=optimizer,
                         learning_rate=float(lr), max_norm=float(maxnorm), epochs=int(epochs),
                         batch_size=int(batch), early_stopping=bool(es), patience=int(patience),
                         restore_best_weights=bool(restore), validation=val, validation_split=float(val_split),
                         class_weight=bool(cw), seed=int(seed), tensorboard=bool(tb))
    st.subheader(tx("Keras-Code mit diesen Einstellungen"))
    st.code(cg.model_code(cfg, task, n_feat, n_out) if layers else tx("# Layer-Tabelle prüfen"), language="python")

    if "model" in ss:
        ti = ss.train_info
        st.info(tx("Aktuelles Modell: trainiert {v} · {epochs_run} Epochen · bestes val_loss {best_val_loss:.4g} "
                   "(Epoche {best_epoch}). Neues Training ersetzt es.",
                   v=ti['trained_at'][11:], epochs_run=ti['epochs_run'], best_val_loss=ti['best_val_loss'],
                   best_epoch=ti['best_epoch']))
    st.caption(tx("Während des Trainings nichts anklicken – jede Eingabe bricht das Training ab."))
    if st.button(tx("Training starten"), type="primary", icon=":material/rocket_launch:",
                 disabled=layer_error is not None):
        model = pl.build_model(n_feat, n_out, cfg, task)
        bar = st.progress(0.0, text=tx("Starte Training …"))
        chart = st.empty()
        try:
            hist, info = pl.train_model(model, p, cfg, callbacks=[StreamlitProgress(cfg.epochs, bar, chart)])
        except Exception as exc:  # noqa: BLE001
            st.error(tx("Training fehlgeschlagen: {exc}", exc=exc))
            return
        stop = tx(" (EarlyStopping)") if info["stopped_early"] else ""
        bar.progress(1.0, text=tx("Fertig nach {epochs_run} Epochen{stop} · {seconds} s",
                                  epochs_run=info['epochs_run'], stop=stop, seconds=info['seconds']))
        invalidate_from("model")
        ss.model, ss.history, ss.train_info, ss.model_cfg = model, hist, info, cfg
        ss.model_token = uuid.uuid4().hex  # eindeutig je Training (id() kann nach dem Löschen wiederverwendet werden)
        st.success(tx("Training abgeschlossen: bestes val_loss {best_val_loss:.4g} in Epoche {best_epoch} → weiter "
                      "mit Schritt 7.",
                      best_val_loss=info['best_val_loss'], best_epoch=info['best_epoch']))
    if "model" in ss:
        with st.expander("model.summary()"):
            st.code(pl.model_summary_text(ss.model), language="text")


# ===========================================================================
# Seite 7 – loss vs. val_loss
# ===========================================================================
def page_loss():
    st.header(tx("7 · Auswertung loss vs. val_loss"))
    if not ready("history", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    h, info, cfg = ss.history, ss.train_info, ss.model_cfg
    task = ss.prep.recipe["task"]
    c1, c2 = st.columns([3, 2])
    with c1:
        show(plots.fig_history(h, "loss", "loss vs. val_loss"))
    loss, val = np.array(h["loss"]), np.array(h.get("val_loss", h["loss"]))
    best = info["best_epoch"]
    notes = [tx("**{epochs_run} Epochen** trainiert, bestes val_loss **{v:.4g}** in Epoche **{best}**.",
                epochs_run=info['epochs_run'], v=val.min(), best=best)]
    if ss.prep.recipe.get("label_scaler"):
        notes.append(tx("loss bezieht sich auf das *skalierte* Label (Schritt 4) – die Kennzahlen in Schritt 8 sind "
                     "in Originaleinheiten."))
    if info["stopped_early"]:
        notes.append(tx("EarlyStopping hat gestoppt ({patience} Epochen ohne Verbesserung)", patience=cfg.patience)
                     + (tx(" und die Gewichte der besten Epoche wiederhergestellt.") if cfg.restore_best_weights else "."))
    elif best == info["epochs_run"]:
        notes.append(tx("val_loss fällt bis zur letzten Epoche → evtl. noch nicht fertig: mehr Epochen versuchen."))
    if val[-1] > val.min() * 1.05 and len(val) > best:
        notes.append(tx("val_loss steigt nach dem Minimum wieder, während loss weiter fällt → **Overfitting**. "
                     "Gegenmittel: EarlyStopping/restore_best_weights, mehr Dropout, kleineres Netz."))
    if val[best - 1] < loss[best - 1]:
        notes.append(tx("val_loss < loss ist bei Dropout normal: Dropout ist nur im Training aktiv."))
    with c2:
        st.markdown("\n\n".join(notes))
    metric = "accuracy" if "accuracy" in h else ("mean_absolute_error" if "mean_absolute_error" in h else None)
    if metric:
        c1, _ = st.columns([3, 2])
        with c1:
            show(plots.fig_history(h, metric))
    with st.expander(tx("Tabelle: pd.DataFrame(model.history.history)")):
        table = pd.DataFrame(h)
        table.index = pd.RangeIndex(1, len(table) + 1, name=tx("Epoche"))
        st.dataframe(table)
    code(cg.loss_code(task))


# ===========================================================================
# Seite 8 – Evaluation
# ===========================================================================
def test_scores():
    token = ss.get("model_token")
    if ss.get("eval_scores", (None,))[0] != token:
        ss.eval_scores = (token, pl.predict_scores(ss.model, ss.prep.X_test, ss.prep.recipe["task"], ss.prep.recipe))
    return ss.eval_scores[1]


def metric_row(ev):
    if ev["task"] == pl.T_REG:
        m = st.columns(4)
        m[0].metric("MAE", fmt_num(ev["MAE"]))
        m[1].metric("RMSE", fmt_num(ev["RMSE"]))
        m[2].metric("R²", f"{ev['R2']:.3f}")
        m[3].metric(tx("Explained Variance"), f"{ev['Explained Variance']:.3f}")
        st.caption(tx("MAE = {x:.1%} vom Mittelwert des Labels ({x2})",
                      x=abs(ev['MAE'] / ev['Mittelwert y']), x2=fmt_num(ev['Mittelwert y'])) if ev["Mittelwert y"] else "")
    else:
        m = st.columns(3)
        m[0].metric(tx("Accuracy"), f"{ev['accuracy']:.3f}")
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


RIGHT, WRONG = tx(":green[:material/check_circle:] richtig"), tx(":red[:material/cancel:] falsch")


def page_eval():
    st.header(tx("8 · Modellevaluation"))
    if not ready("model", tx("Bitte zuerst in Schritt 6 trainieren.")):
        return
    p = ss.prep
    r = p.recipe
    task = r["task"]
    scores = test_scores()
    thr = 0.5
    if task == pl.T_BIN:
        init("w_threshold", 0.5)
        thr = st.slider(tx("Entscheidungsschwelle: P({v}) > Schwelle → '{v}'", v=r['classes'][1]),
                        0.05, 0.95, step=0.01, key="w_threshold",
                        help=tx("Standard 0.5. Höher = strenger für Klasse 1. Wird beim Speichern mitgespeichert."))
    ev = pl.evaluate(p.y_test, scores, task, r.get("classes"), thr)
    st.caption(tx("Bewertet auf {n} Testzeilen (X_test, y_test)", n=fmt_int(len(p.y_test))))
    metric_row(ev)
    if task == pl.T_REG:
        show(plots.fig_regression(p.y_test, scores))
    else:
        classification_block(ev, r["classes"])
        if task == pl.T_BIN:
            c1, c2 = st.columns([3, 2])
            with c1:
                show(plots.fig_proba(scores, r["classes"], p.y_test, thr))
            c2.markdown(tx("**Schwelle verstehen:** Das Modell liefert für jede Zeile eine Wahrscheinlichkeit für "
                           "'{v}'. Liegt sie über der Schwelle, lautet die Vorhersage '{v}'. Eine höhere Schwelle "
                           "erkennt mehr '{v2}' (Recall Klasse 0 steigt), auf Kosten von Fehlalarmen. Bei "
                           "unausgewogenen Klassen lohnt es sich, die Schwelle zu verschieben oder in Schritt 6 "
                           "Klassengewichte zu nutzen.",
                           v=r['classes'][1], v2=r['classes'][0]))

    st.subheader(tx("Einzelfall prüfen"))
    st.caption(tx("Zufälligen Datensatz aus den Testdaten ziehen und die Vorhersage mit der Wahrheit vergleichen."))
    if st.button(tx("Zufälligen Datensatz ziehen"), icon=":material/casino:"):
        ss.case_idx = int(np.random.default_rng().integers(len(p.idx_test)))
    if "case_idx" in ss and ss.case_idx < len(p.idx_test):
        i = ss.case_idx
        row = ss.df_viz.loc[p.idx_test[i], r["features"] + [r["label"]]]
        c1, c2 = st.columns([2, 3])
        shown = row.map(lambda v: "–" if pd.isna(v) else pl.format_value(v))
        c1.dataframe(shown.rename(tx("Wert")).to_frame(), height=420)
        with c2:
            if task == pl.T_REG:
                m1, m2 = st.columns(2)
                m1.metric(tx("Vorhersage"), fmt_num(scores[i]))
                m2.metric(tx("tatsächlich"), fmt_num(p.y_test[i]))
                diff = scores[i] - p.y_test[i]
                st.markdown(tx("Abweichung: **{v}{x}**", v='+' if diff >= 0 else '−', x=fmt_num(abs(diff))))
            else:
                pred = pl.to_codes(scores[i:i + 1], task, thr)[0]
                truth = int(p.y_test[i])
                m1, m2 = st.columns(2)
                m1.metric(tx("Vorhersage"), r["classes"][pred])
                m2.metric(tx("tatsächlich"), r["classes"][truth])
                if task == pl.T_BIN:
                    st.markdown(tx("P({v}) = **{x:.1%}** · Schwelle {thr:.2f} · ",
                                   v=r['classes'][1], x=float(scores[i]), thr=thr)
                                + (RIGHT if pred == truth else WRONG))
                else:
                    st.markdown(tx("Sicherheit **{x:.1%}** · ", x=float(scores[i][pred]))
                                + (RIGHT if pred == truth else WRONG))

    st.subheader(tx("Welche Features sind wichtig?"))
    st.caption(tx("Permutation Importance: Werte eines Features werden gemischt – je stärker das Modell dadurch "
               "schlechter wird, desto wichtiger ist das Feature. Dummies einer Kategorie werden gemeinsam gemischt."))
    if st.button(tx("Permutation Importance berechnen (Stichprobe 2000 Zeilen)")):
        with st.spinner(tx("Berechne …")):
            ss.importance = pl.permutation_importance(ss.model, p.X_test, p.y_test, r, threshold=thr)
    if "importance" in ss:
        c1, c2 = st.columns([3, 2])
        with c1:
            show(plots.fig_importance(ss.importance))
        c2.dataframe(trc(ss.importance.round(5)), hide_index=True, height=450)
    code(cg.eval_code(task, thr, bool(r.get("label_scaler"))))


# ===========================================================================
# Seite 9 – Speichern & Laden
# ===========================================================================
def session_recipe(threshold=None):
    """Rezept des aktuellen Modells inkl. Training, Kennzahlen und Schwelle."""
    p = ss.prep
    r = p.recipe
    thr = ss.get("w_threshold", 0.5) if threshold is None else threshold
    ev = pl.evaluate(p.y_test, test_scores(), r["task"], r.get("classes"), thr)
    return pl.finalize_recipe(r, ss.model_cfg, ss.train_info, pl.metrics_summary(ev),
                              thr if r["task"] == pl.T_BIN else 0.5)


def page_save():
    st.header(tx("9 · Modell speichern & laden"))
    st.markdown(tx("Gespeichert werden **zwei Dateien** in `models/<name>/`: das Keras-Modell (`model.keras` oder "
                "`model.h5`) und `preprocessing.json` – das Rezept aus Schritt 2 und 4 (Spalten-Aktionen, "
                "Füllwerte, Kategorien/Dummy-Spalten, Scaler, Label-Kodierung, Schwelle). Ohne das Rezept "
                "lassen sich neue Daten nicht korrekt aufbereiten."))
    st.subheader(tx("Speichern"))
    if "model" not in ss:
        st.info(tx("Noch kein trainiertes Modell in dieser Sitzung (Schritt 6)."))
    else:
        r = ss.prep.recipe
        if ss.get("_save_name_for") != ss.get("model_token"):
            ss.w_save_name = f"{re.sub(r'[^0-9A-Za-z_-]+', '_', r['label'])}_{datetime.now():%Y%m%d_%H%M}"
            ss._save_name_for = ss.get("model_token")
        c1, c2 = st.columns([2, 2])
        with c1:
            name = st.text_input(tx("Name (Ordner unter models/)"), key="w_save_name")
        with c2:
            fmt = rd(tx("Format"), [".keras", ".h5"], "w_save_fmt", horizontal=True,
                     format_func={".keras": tx(".keras (empfohlen, Keras 3)"), ".h5": tx(".h5 (HDF5, älteres Format)")}.get)
        safe = re.sub(r"[^\w\-.]+", "_", name or "").strip("._") or "modell"
        target = MODELS_DIR / safe
        # fremdes Modell unter diesem Namen? (das eigene, gerade gespeicherte darf ohne Rückfrage ersetzt werden)
        exists = (target / "preprocessing.json").exists() and ss.get("saved_folder") != str(target)
        init("w_save_fp", True)
        keep_fp = st.checkbox(
            tx("Prüfsummen von Trainingszeilen mitspeichern (Schritt 10 erkennt damit Trainingsdaten)"), key="w_save_fp",
            help=tx("Bis zu 3000 Prüfsummen (Hashes) zufälliger Trainingszeilen. Aus einer Prüfsumme lässt sich die "
                 "Zeile nicht zurückrechnen – wer die Originaldaten hat, kann aber prüfen, ob eine bestimmte Zeile "
                 "im Training war. Bei vertraulichen Daten ausschalten, bevor das Modell weitergegeben wird."))
        overwrite = True
        if exists:
            st.warning(tx("Unter `models/{safe}` liegt schon ein Modell – Speichern ersetzt es.", safe=safe))
            overwrite = st.checkbox(tx("Ja, vorhandenes Modell überschreiben"), key=f"overwrite_{safe}")
        if st.button(tx("Speichern"), type="primary", icon=":material/save:", disabled=not overwrite):
            with st.spinner(tx("Speichere …")):
                recipe = session_recipe()
                if not keep_fp:
                    recipe.pop("train_fingerprints", None)
                folder = pl.save_bundle(ss.model, recipe, target, fmt)
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
                                r, ss.model_cfg, ss.get("w_threshold", 0.5), code_folder, ss.w_save_fmt,
                                new_columns=ss.get("new_cols_fitted"))
        st.download_button(tx("Komplette Pipeline als Python-Skript (für Jupyter)"), script.encode("utf-8"),
                           file_name="dl_pipeline.py", mime="text/x-python", icon=":material/download:")
        code(cg.save_code(code_folder, ss.w_save_fmt))

    st.subheader(tx("Laden"))
    bundles = pl.list_bundles(MODELS_DIR)
    t1, t2 = st.tabs([tx("Aus dem Ordner models/"), tx("ZIP hochladen")])
    with t1:
        if not bundles:
            st.caption(tx("Noch keine gespeicherten Modelle."))
        else:
            labels = {b["name"]: tx("{name} · Label {label} · {v}",
                                    name=b['name'], label=b['label'], v=b['saved_at'][:16].replace('T', ' '))
                      for b in bundles}
            sel = sb(tx("Gespeicherte Modelle"), list(labels), "w_bundle_sel", format_func=labels.get)
            if st.button(tx("Laden"), key="btn_load_bundle"):
                try:
                    model, recipe = pl.load_bundle(MODELS_DIR / sel)
                except Exception as exc:  # noqa: BLE001
                    st.error(tx("Laden fehlgeschlagen: {exc}", exc=exc))
                else:
                    ss.bundle = {"model": model, "recipe": recipe, "name": sel, "token": uuid.uuid4().hex}
                    ss.pop("pred_result", None)
                    st.success(tx("'{sel}' geladen → Schritt 10", sel=sel))
    with t2:
        up = st.file_uploader(tx("ZIP (model.keras/.h5 + preprocessing.json)"), type=["zip"], key="up_bundle")
        if st.button(tx("ZIP laden"), disabled=up is None):
            try:
                model, recipe = pl.load_bundle_zip(up.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Laden fehlgeschlagen: {exc}", exc=exc))
            else:
                ss.bundle = {"model": model, "recipe": recipe, "name": Path(up.name).stem, "token": uuid.uuid4().hex}
                ss.pop("pred_result", None)
                st.success(tx("ZIP geladen → Schritt 10"))
    if "bundle" in ss:
        b = ss.bundle
        r = b["recipe"]
        st.markdown(tx("**Geladen: {name}** · Label `{label}` · {v} · {n} Features → {n2} Eingänge · gespeichert "
                       "{v2}",
                       name=b['name'], label=r['label'], v=tx(pl.TASK_LABELS[r['task']]), n=len(r['features']),
                       n2=len(r['feature_columns']), v2=r.get('saved_at', '?').replace('T', ' ')))
        c1, c2 = st.columns(2)
        c1.markdown(tx("**Kennzahlen (Testdaten)**"))
        c1.json(r.get("metrics", {}))
        c2.markdown(tx("**Training**"))
        c2.json({k: v for k, v in (r.get("training") or {}).items()
                 if k in ("layers", "loss", "optimizer", "epochs", "batch_size", "patience")})


# ===========================================================================
# Seite 10 – Vorhersage auf neuen Daten
# ===========================================================================
HOW_TO_10 = tx("""
**So werden neue Daten verarbeitet**

1. **Nichts neu fitten.** Neue Daten werden mit dem *gespeicherten Rezept* aufbereitet – gleiche
   Spalten-Aktionen und neue Spalten, gleiche Füllwerte, gleiche Dummy-Spalten in gleicher Reihenfolge,
   gleicher Scaler. Würde man `get_dummies` oder `MinMaxScaler` auf den neuen Daten neu rechnen, passen
   Spalten und Skalen nicht mehr zum Modell.
2. **Gleiche Rohspalten wie im Training.** Das Modell bestimmt die Features – eine Auswahl ist hier nicht
   mehr möglich. Die Tabelle unten zeigt, welche CSV-Spalten nötig sind (Vorlage zum Download).
   Fehlen Spalten → klare Fehlermeldung. Zusätzliche Spalten (auch das Label) werden ignoriert.
3. **Abweichungen werden behandelt und gemeldet:** Lücken → Füllwerte aus dem Training (jede Zeile braucht
   für eine Vorhersage einen Wert – auch dort, wo im Training Zeilen mit Lücken entfernt wurden);
   unbekannte Kategorien → alle Dummies 0; Werte außerhalb des Trainingsbereichs → Hinweis.
4. **Interpretation und Fazit:** Bei binär = Wahrscheinlichkeit für die positive Klasse + Schwelle. Ist das
   Label in den neuen Daten enthalten, wird automatisch evaluiert. Das Fazit am Ende bewertet die
   Datenqualität und erklärt, was das Ergebnis aussagt – auch, ob die Daten schon im Training waren.
""")


def render_interpretation(res, recipe, thr):
    task = recipe["task"]
    n = len(res)
    if task == pl.T_BIN:
        neg, pos = recipe["classes"]
        p = res[f"P({pos})"].to_numpy()
        n_pos = int((res["Vorhersage"] == pos).sum())
        train_share = recipe.get("label_stats", {}).get("class_share", {}).get(pos)
        txt = (tx("**Lesart:** `P({pos})` ist die vom Modell geschätzte Wahrscheinlichkeit für '{pos}'. Über der "
                  "Schwelle {thr:.2f} → Vorhersage '{pos}', sonst '{neg}'. Werte nahe 0 oder 1 = sicher, um "
                  "{thr:.2f} = unsicher.",
                  pos=pos, thr=thr, neg=neg))
        if n == 1:
            st.markdown(txt)
            return
        txt += (tx("\n\n**Ergebnis:** {n} von {n2} Zeilen ({v:.1%}) werden als '{pos}' vorhergesagt, {n3} als "
                   "'{neg}'.",
                   n=fmt_int(n_pos), n2=fmt_int(n), v=n_pos / n, pos=pos, n3=fmt_int(n - n_pos), neg=neg))
        if train_share is not None:
            txt += (tx(" Zum Vergleich: In den Trainingsdaten waren tatsächlich {train_share:.1%} '{pos}'. Ein "
                       "Unterschied zeigt, wie oft das Modell danebenliegt oder dass die Daten anders sind – "
                       "Einordnung im Fazit unten.",
                       train_share=train_share, pos=pos))
        st.markdown(txt)
        c1, _ = st.columns([3, 2])
        with c1:
            show(plots.fig_proba(p, recipe["classes"], threshold=thr))
    elif task == pl.T_MULTI:
        st.markdown(tx("**Lesart:** `Vorhersage` = Klasse mit der höchsten Wahrscheinlichkeit, `Sicherheit` = diese "
                    "Wahrscheinlichkeit; `P(…)` je Klasse summieren sich zu 1."))
        if n > 1:
            c1, _ = st.columns([3, 2])
            with c1:
                show(plots.fig_class_counts(res["Vorhersage"], recipe.get("label_stats", {}).get("class_share")))
    else:
        ls = recipe.get("label_stats", {})
        mae = (recipe.get("metrics") or {}).get("MAE")
        pred = res["Vorhersage"].to_numpy()
        txt = tx("**Lesart:** `Vorhersage` ist der geschätzte Wert von `{label}` (gleiche Einheit).",
                 label=recipe['label'])
        if n > 1:
            txt += (tx(" Mittelwert der Vorhersagen: **{x}** (Training: {x2}).",
                       x=fmt_num(pred.mean()), x2=fmt_num(ls.get('mean', float('nan')))))
        if mae:
            txt += tx(" Typische Abweichung laut Test: ± {x} (MAE).", x=fmt_num(mae))
        st.markdown(txt)
        if n > 1:
            c1, _ = st.columns([3, 2])
            with c1:
                show(plots.fig_values(pred, tx("Vorhersage {label}", label=recipe['label']),
                                      tx("Verteilung der Vorhersagen"), ls.get("mean"), tx("Mittelwert Training")))


LEVEL_ICON = {pl.LEVEL_OK: ":green[:material/check_circle:]", pl.LEVEL_WARN: ":orange[:material/warning:]",
              pl.LEVEL_BAD: ":red[:material/error:]"}


def render_fazit(fz):
    st.subheader(tx("Fazit"))
    with st.container(border=True):
        st.markdown(tx("**Datenqualität: {v} {level}**", v=LEVEL_ICON[fz['level']], level=tx(fz['level'])))
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
    st.header(tx("10 · Vorhersagen auf neuen Daten"))
    options = {}
    if "model" in ss:
        options["session"] = tx("Aktuelles Modell (diese Sitzung)")
    if "bundle" in ss:
        options["bundle"] = tx("Geladenes Modell: {name}", name=ss.bundle['name'])
    if not options:
        st.info(tx("Kein Modell vorhanden – in Schritt 6 trainieren oder in Schritt 9 laden."))
        st.markdown(HOW_TO_10)
        return
    src = rd(tx("Modell"), list(options), "w_pred_model", format_func=options.get, horizontal=True)
    if src == "session":
        model, recipe = ss.model, session_recipe()
    else:
        model, recipe = ss.bundle["model"], ss.bundle["recipe"]
    task = recipe["task"]
    with st.expander(tx("So funktioniert Schritt 10"), icon=":material/info:"):
        st.markdown(HOW_TO_10)

    st.subheader(tx("Benötigte Spalten"))
    st.dataframe(pl.required_columns(recipe), hide_index=True)
    st.download_button(tx("CSV-Vorlage (nur Kopfzeile)"), pl.template_csv(recipe), tx("vorlage_neue_daten.csv"),
                       mime="text/csv", icon=":material/download:")
    thr = recipe.get("threshold", 0.5)
    if task == pl.T_BIN:
        model_id = f"{src}-{ss.get('model_token') if src == 'session' else ss.bundle.get('token')}"
        if ss.get("_pred_thr_for") != model_id:
            ss.w_pred_thr = float(thr)
            ss._pred_thr_for = model_id
        thr = st.slider(tx("Schwelle"), 0.05, 0.95, step=0.01, key="w_pred_thr")

    mode = rd(tx("Eingabe"), [N_("CSV-Datei"), N_("Einzelner Datensatz (Formular)")], "w_pred_mode", horizontal=True,
              format_func=tx)
    if mode == "CSV-Datei":
        source, name = csv_source("pred")
        csv_options()
        if st.button(tx("Vorhersage berechnen"), type="primary", disabled=source is None):
            try:
                with st.spinner(tx("Bereite neue Daten auf und sage vorher …")):
                    df_new, _ = read_source(source)
                    res, rep, ev = pl.predict_new(model, recipe, df_new, thr)
            except pl.MissingColumnsError as exc:
                st.error(tx("{exc}. Die CSV braucht dieselben Rohspalten wie die Trainingsdaten (siehe Tabelle "
                            "oben / CSV-Vorlage).",
                            exc=exc))
            except Exception as exc:  # noqa: BLE001
                st.error(tx("Fehler: {exc}", exc=exc))
            else:
                ss.pred_result = {"res": res, "rep": rep, "ev": ev, "name": name, "thr": thr, "src": src,
                                  "cols": list(df_new.columns)}
        pr = ss.get("pred_result")
        if pr and pr["src"] == src:
            res, rep, ev = pr["res"], pr["rep"], pr["ev"]
            fz = pl.assess_prediction(recipe, res, rep, ev, pr["thr"])
            st.success(tx("{n} Vorhersagen für {name}", n=fmt_int(len(res)), name=pr['name']))
            if fz["level"] != pl.LEVEL_OK or (rep.get("train_overlap") or {}).get("share", 0) >= 0.05:
                st.caption(tx("{v} Hinweise zur Datenqualität und Einordnung → **Fazit** unten",
                              v=LEVEL_ICON[fz['level']]))
            render_interpretation(res, recipe, pr["thr"])
            if ev is not None:
                st.subheader(tx("Label '{label}' ist enthalten → Evaluation auf {n} Zeilen",
                                label=recipe['label'], n=fmt_int(ev['n'])))
                metric_row(ev)
                if task != pl.T_REG:
                    classification_block(ev, recipe["classes"])
            st.subheader(tx("Ergebnis"))
            replaced = replaced_columns(pr.get("cols", []), recipe)  # je Sprache neu (übersetzte Namen)
            if replaced:
                st.info(tx("Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis "
                           "werden sie ersetzt: {cols}", cols=", ".join(map(str, replaced))))
            view = result_view(res)
            lead = [c for c in view.columns if c.startswith("P(") or c in [tx(n) for n in RESULT_COLUMNS]]
            st.dataframe(view[lead + [c for c in view.columns if c not in lead]].head(1000))
            st.download_button(tx("Ergebnis als CSV"), view.to_csv(index=False).encode("utf-8"),
                               file_name=tx("vorhersagen_{stem}.csv", stem=Path(pr['name']).stem), mime="text/csv",
                               icon=":material/download:")
            render_fazit(fz)
    else:
        st.caption(tx("Werte eingeben (vorbelegt mit Median bzw. häufigstem Wert aus dem Training). Eingegeben wird "
                   "die Form nach Schritt 2, z. B. term als Zahl."))
        with st.form("single_case"):
            values = {}
            grid = st.columns(3)
            for i, f in enumerate(recipe["features"]):
                with grid[i % 3]:
                    fb = recipe["fallback_fill"].get(f)
                    if f in recipe["numeric_features"]:
                        values[f] = st.number_input(f, value=float(fb), format="%g", key=f"form_{src}_{f}")
                    else:
                        cats = recipe["categories"][f]
                        values[f] = st.selectbox(f, cats, index=cats.index(fb) if fb in cats else 0,
                                                 key=f"form_{src}_{f}")
            submitted = st.form_submit_button(tx("Vorhersage"), type="primary")
        if submitted:
            res, rep, _ = pl.predict_new(model, recipe, pd.DataFrame([values]), thr)
            row = res.iloc[0]
            m1, m2, _ = st.columns([1, 1, 2])
            if task == pl.T_BIN:
                pos = recipe["classes"][1]
                m1.metric(tx("Vorhersage"), row["Vorhersage"])
                m2.metric(f"P({pos})", f"{row[f'P({pos})']:.1%}")
            elif task == pl.T_MULTI:
                m1.metric(tx("Vorhersage"), row["Vorhersage"])
                m2.metric(tx("Sicherheit"), f"{row['Sicherheit']:.1%}")
                st.dataframe(res[[c for c in res.columns if c.startswith("P(")]], hide_index=True)
            else:
                m1.metric(tx("Vorhersage"), fmt_num(row["Vorhersage"]))
            render_interpretation(res, recipe, thr)
    code(cg.predict_code("models/<name>"), tx("Python-Code (Vorhersage ohne App)"))


# ===========================================================================
# Seitenleiste & Navigation
# ===========================================================================
PAGES = {
    N_("1 · Daten & Bibliotheken"): page_data,
    N_("2 · Aufbereitung 1 – Spalten"): page_prep1,
    N_("3 · Analyse & Visualisierung"): page_viz,
    N_("4 · Aufbereitung 2 – ML-Daten"): page_prep2,
    N_("5 · Daten speichern"): page_export,
    N_("6 · Modell & Training"): page_train,
    N_("7 · loss vs. val_loss"): page_loss,
    N_("8 · Evaluation"): page_eval,
    N_("9 · Modell speichern & laden"): page_save,
    N_("10 · Vorhersage neue Daten"): page_predict,
}

with st.sidebar:
    st.markdown("### :material/analytics: DATA-Analysis-Deeplearning")
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
        status("df_raw" in ss, tx("Daten: {name} ({n} Zeilen)", name=s['name'], n=fmt_int(len(ss.df_raw)))
               if s and "df_raw" in ss else tx("Daten laden"))
        status("df_viz" in ss, tx("Aufbereitung 1 übernommen"))
        status("prep" in ss, tx("ML-Daten: X_train {shape}", shape=ss.prep.X_train.shape) if "prep" in ss
               else tx("ML-Daten vorbereitet"))
        status("model" in ss, tx("Modell trainiert"))
        status("bundle" in ss, tx("Geladen: {name}", name=ss.bundle['name']) if "bundle" in ss
               else tx("Modell geladen (Schritt 9)"))
    with st.expander(tx("Konfiguration speichern / laden"), icon=":material/settings:"):
        st.caption(tx("Alle Einstellungen (Schritt 2 inkl. neuer Spalten, 4, 6) als JSON – für geänderte Daten mit "
                   "gleichen Spalten."))
        st.download_button(tx("Exportieren (JSON)"), export_config(), file_name="dl_template_config.json",
                           mime="application/json")
        up_cfg = st.file_uploader(tx("Importieren"), type=["json"], key="up_cfg")
        st.button(tx("Konfiguration übernehmen"), disabled=up_cfg is None, on_click=on_import_config)
        msg = ss.pop("_cfg_msg", None)
        if msg:
            (st.error if msg[0] == "error" else st.success)(msg[1])
