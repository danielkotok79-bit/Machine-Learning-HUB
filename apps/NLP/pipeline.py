"""
pipeline.py – Rechenlogik des DATA-Analysis-NLP (Textklassifikation, ohne Streamlit)
=====================================================================================

Alle Schritte einer Textklassifikation (NLP mit scikit-learn) für Tabellen (CSV) mit einer Textspalte als
wiederverwendbare Funktionen. Die Streamlit-App (app.py) ruft nur diese Funktionen auf – das Modul lässt sich
genauso im Jupyter Notebook verwenden:

    import pipeline as pl
    df, _ = pl.read_csv("yelp.csv")

Schritte
    1) Bibliotheken ............ Import-Block direkt unten
    2) Aufbereitung 1 .......... suggest_action(), apply_column_actions(), apply_new_columns()
    4) Aufbereitung 2 .......... prepare_text_data()  (Text, Label, Klassen, Split, Vektorisierung)
    5) Daten speichern ......... split_frame(), vocabulary_table(), predictions_frame(), to_csv_bytes()
    6) Modell & Training ....... MODELS, TextModelConfig, make_model(), train_model()
    7) Training vs. Validierung  validation_curve(), curve_assessment()
    8) Evaluation .............. predict_scores(), evaluate(), top_words(), explain_text(), misclassified()
    9) Speichern & Laden ....... make_pipeline(), save_bundle(), load_bundle()  (model.skops oder model.joblib)
   10) Neue Texte .............. predict_texts(), assess_prediction()

Modelle: MultinomialNB (Naive Bayes, wie im Kurs) · Logistische Regression · LinearSVC (lineare SVM)

Kernidee: Text → Zahlen mit CountVectorizer (Wörter zählen, „Bag of Words“) oder TfidfVectorizer. Der
Vektorisierer lernt sein Vokabular nur aus den Trainingstexten; zusammen mit dem Modell bildet er eine
scikit-learn-Pipeline, die gespeichert wird und neue Texte unverändert umwandelt – nichts wird neu gelernt.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 1) Import der Bibliotheken
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Ergänzungen für das Template
import io
import json
import re
import tempfile
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from i18n import N_, tx  # Texte in der gewählten Sprache (Deutsch = Original); N_ markiert nur

TEMPLATE_VERSION = 1  # Rezept der NLP-Version
RANDOM_STATE = 101  # Standard-Seed für Split und Training

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
T_BIN, T_MULTI = "binary", "multiclass"
TASK_LABELS = {
    T_BIN: N_("Binär (2 Klassen)"),
    T_MULTI: N_("Mehrklassen"),
}
APP_ID = "nlp"  # Rezepte dieser App – Modelle der Supervised-/Clustering-Version passen nicht
THRESHOLD_MIN, THRESHOLD_MAX = 0.05, 0.95  # Schwelle binärer Modelle: Grenzen der Schieberegler (Schritt 8 und 10)

# Schritt 2 – Spalten-Aktionen
A_KEEP = N_("behalten")
A_DROP = N_("entfernen")
A_NUM = N_("Zahl extrahieren")
A_YEAR = N_("Datum → Jahr")
A_MONTH = N_("Datum → Monat")
A_YEAR_MONTH = N_("Datum → Jahr + Monat")
A_FIRST = N_("Text: erste N Zeichen")
A_LAST = N_("Text: letzte N Zeichen")
A_REPLACE = N_("Werte ersetzen")
A_TOPN = N_("Top-N behalten, Rest → Sonstige")
A_CAT = N_("als Kategorie (Text)")
A_LEN = N_("Text → Länge (Zeichen)")
A_WORDS = N_("Text → Anzahl Wörter")
ACTIONS = [A_KEEP, A_DROP, A_NUM, A_YEAR, A_MONTH, A_YEAR_MONTH, A_FIRST, A_LAST, A_REPLACE, A_TOPN, A_CAT,
           A_LEN, A_WORDS]
DATE_ACTIONS = (A_YEAR, A_MONTH, A_YEAR_MONTH)
ACTION_HELP = {
    A_KEEP: ("–", N_("unverändert – eine Textspalte (z. B. die Bewertung) wird in Schritt 4 die Textspalte")),
    A_DROP: ("–", N_("Spalte löschen (z. B. IDs wie review_id oder user_id)")),
    A_NUM: ("–", N_("erste Zahl aus dem Text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, "
                 "'1.234,5 €' → 1234.5")),
    A_YEAR: (N_("optional Format, z. B. %b-%Y"), N_("Datum → Jahr: 'Jun-1990' → 1990, auch Zahl 20141013 → 2014")),
    A_MONTH: (N_("optional Format, z. B. %d.%m.%Y"), N_("Datum → Monat (1–12)")),
    A_YEAR_MONTH: (N_("optional Format, z. B. %b-%Y"), N_("zwei neue Zahlenspalten: <Name>_jahr und <Name>_monat "
                                                   "('Oct-2014' → 2014 und 10)")),
    A_FIRST: (N_("N (Standard 1)"), N_("erste N Zeichen: 'B4' → 'B'")),
    A_LAST: (N_("N (Standard 5)"), N_("letzte N Zeichen: Adresse → PLZ '22690'")),
    A_REPLACE: (N_("alt1,alt2=neu; alt3=neu2"), N_("Werte ersetzen: NONE,ANY=OTHER")),
    A_TOPN: (N_("N (Standard 10)"), N_("nur die N häufigsten Werte behalten, Rest → 'Sonstige'")),
    A_CAT: ("–", N_("Zahlen als Kategorie behandeln (Codes, PLZ als Zahl …)")),
    A_LEN: ("–", N_("Anzahl Zeichen je Text – wie im Kurs: text.apply(len); meist als neue Spalte (Kopie + Aktion)")),
    A_WORDS: ("–", N_("Anzahl Wörter je Text (durch Leerzeichen getrennt); meist als neue Spalte (Kopie + Aktion)")),
}
TEXT_ACTIONS = (A_LEN, A_WORDS)
OTHER = "Sonstige"

_NUM_TOKEN = r"(-?\d(?:[\d.,]*\d)?)"  # erste Zahl inkl. Dezimal-/Tausendertrennzeichen
_NUM_UNIT_RE = r"\s*[$€£]?\s*-?\d(?:[\d.,]*\d)?\s*[A-Za-zÄÖÜäöüß%€$£]*\s*"


class MissingColumnsError(ValueError):
    """Neue Daten enthalten nicht alle Spalten, mit denen das Modell trainiert wurde."""

    def __init__(self, missing: list[str], lineage: dict | None = None):
        self.missing = missing
        self.raw = list(dict.fromkeys(r for c in missing for r in raw_sources(lineage, c)))
        parts = [c if raw_sources(lineage, c) == [c] else f"{', '.join(raw_sources(lineage, c))} (→ {c})"
                 for c in missing]
        super().__init__(tx("Es fehlen Spalten: ") + ", ".join(parts))


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def is_numeric(s: pd.Series) -> bool:
    """True für Zahlen- und Bool-Spalten (pandas 3: Text ist 'str', nicht mehr 'object')."""
    return bool(pd.api.types.is_numeric_dtype(s))


def _clean(v) -> str:
    """Zelle aus dem Daten-Editor → sauberer String ('' für None/NaN)."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")


def md_escape(text) -> str:
    """Text aus Dateien (Spalten-, Klassen-Namen, Werte aus einem fremden Modell) wörtlich in Markdown zeigen:
    Sonderzeichen escapen, Zeilenumbrüche entfernen – sonst könnte z. B. ![](http://…) ein fremdes Bild nachladen."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).splitlines()))


def _py(v):
    """numpy-Skalar → Python-Typ (für JSON)."""
    return v.item() if hasattr(v, "item") else v


def _fmt_value(v) -> str:
    """Wert → Text; ganze Zahlen ohne '.0' (1.0 → '1')."""
    v = _py(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


format_value = _fmt_value  # öffentlich, z. B. für Tabellen in der App


def _nice(v: float) -> str:
    """Kurz und lesbar für Log-Texte: 440312.7 → '440 313', 0.01234 → '0.01234'."""
    return f"{v:,.0f}".replace(",", "\u202f") if abs(v) >= 1000 else f"{v:.4g}"


def _whole(values) -> bool:
    """Alle Werte ganzzahlig (22690.0)? inf/NaN zählen nicht als ganzzahlig – ohne RuntimeWarning."""
    v = np.asarray(values, dtype="float64")
    return (bool(len(v)) and bool(np.isfinite(v).all()) and bool((np.abs(v) < 2**53).all())
            and bool((np.mod(v, 1) == 0).all()))


def _to_str(s: pd.Series) -> pd.Series:
    """Spalte als Text, NaN bleibt NaN. Ganzzahlige Floats ohne '.0' (22690.0 → '22690')."""
    if pd.api.types.is_float_dtype(s):
        nn = s.dropna()
        if _whole(nn.to_numpy()):
            return s.astype("Int64").astype(str).where(s.notna(), np.nan)
    return s.astype(str).where(s.notna(), np.nan)


def _int_param(param: str, default: int) -> int:
    if not param:
        return default
    try:
        n = int(float(param))
    except ValueError as exc:
        raise ValueError(tx("Parameter '{param}' ist keine ganze Zahl.", param=param)) from exc
    if n < 1:
        raise ValueError(tx("Parameter N muss ≥ 1 sein."))
    return n


_DATE_FORMATS = ["%b-%Y", "%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y%m%dT%H%M%S",
                 "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y", "%b %Y", "%Y-%m", "%d.%m.%y"]


def guess_date_format(s: pd.Series) -> str:
    """Häufige Datumsformate durchprobieren (schneller und eindeutiger als automatisches Raten)."""
    txt = _to_str(s.dropna())
    if txt.empty:
        return ""
    for fmt in _DATE_FORMATS:
        if pd.to_datetime(txt, format=fmt, errors="coerce").notna().mean() >= 0.95:
            return fmt
    return ""


def _parse_dates(s: pd.Series, fmt: str | None = None) -> pd.Series:
    txt = _to_str(s)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if fmt:
            return pd.to_datetime(txt, format=fmt, errors="coerce")
        return pd.to_datetime(txt, errors="coerce")


def parse_replace(param: str) -> dict:
    """'NONE,ANY=OTHER; X=Y' → {'NONE': 'OTHER', 'ANY': 'OTHER', 'X': 'Y'}"""
    mapping = {}
    for part in [p for p in re.split(r"[;\n]", param or "") if p.strip()]:
        if "=" not in part:
            raise ValueError(tx("Werte ersetzen: '{part}' hat kein '=' (Format: alt1,alt2=neu)", part=part))
        old, new = part.split("=", 1)
        for o in old.split(","):
            if o.strip():
                mapping[o.strip()] = new.strip()
    if not mapping:
        raise ValueError(tx("Werte ersetzen: Parameter fehlt (Format: alt1,alt2=neu)"))
    return mapping


def _to_number(v: str) -> float:
    try:
        return float(str(v).replace(",", "."))
    except ValueError as exc:
        raise ValueError(tx("'{v}' ist keine Zahl (Spalte ist numerisch).", v=v)) from exc


def _normalize_number(tok) -> float:
    """'1,200' → 1200 · '1.234,5' → 1234.5 · '13,5' → 13.5 · '1.234.567' → 1234567 · '2.5' → 2.5"""
    if tok is None or (isinstance(tok, float) and np.isnan(tok)):
        return np.nan
    t = str(tok)
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):  # deutsch: 1.234,5
            t = t.replace(".", "").replace(",", ".")
        else:  # englisch: 1,234.5
            t = t.replace(",", "")
    elif "," in t:
        # 1,200 / 1,234,567 = Tausendertrennung · 13,5 = Dezimalkomma
        t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", t) else t.replace(",", ".")
    elif t.count(".") > 1:  # 1.234.567
        t = t.replace(".", "")
    try:
        return float(t)
    except ValueError:
        return np.nan


def parse_numbers(s: pd.Series) -> pd.Series:
    """Erste Zahl aus Text holen (Einheiten, Währungen, Dezimalkomma und Tausendertrennzeichen).

    'True'/'False' als Text → 1/0 (z. B. Bool-Spalten mit Lücken in neuen Daten).
    """
    if is_numeric(s):
        return s.astype(float)
    txt = _to_str(s)
    tok = txt.str.extract(_NUM_TOKEN, expand=False)
    uniq = tok.dropna().unique()
    out = tok.map({u: _normalize_number(u) for u in uniq}).astype(float)
    low = txt.str.strip().str.lower()
    is_bool_text = low.isin(["true", "false"]).fillna(False).to_numpy(dtype=bool)
    if is_bool_text.any():
        out = out.where(~is_bool_text, low.map({"true": 1.0, "false": 0.0}).astype(float))
    return out


def _cat(s: pd.Series) -> pd.Series:
    """Kategorie-Werte als Text ohne Leerzeichen am Rand ('RENT ' = 'RENT')."""
    return _to_str(s).str.strip()


def src_kind(s: pd.Series) -> str:
    """Art der Rohspalte – für den generierten Code: bool, int, float_int, float, text."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    if pd.api.types.is_integer_dtype(s):
        return "int"
    if pd.api.types.is_float_dtype(s):
        nn = s.dropna()
        return "float_int" if _whole(nn.to_numpy()) else "float"
    return "text"


def _is_yyyymmdd(s: pd.Series) -> bool:
    """Zahlen wie 20141013 (Datum als Zahl)?"""
    nn = pd.to_numeric(s, errors="coerce").dropna()
    return bool(len(nn)) and bool(nn.between(1e7, 1e8).mean() >= 0.95) and _whole(nn.to_numpy())


def clean_path(p: str) -> str:
    """Pfad aus dem Eingabefeld: Leerzeichen und Anführungszeichen ('Als Pfad kopieren' unter Windows) entfernen."""
    return str(p or "").strip().strip('"').strip("'").strip()


# Einstellungen aus einer importierten Konfiguration: Typ + erlaubter Bereich
SETTING_TYPES = {
    # Schritt 4 – Text, Label, Split, Vektorisierung
    "w_test_size": (float, 0.05, 0.5), "w_split_seed": (int, 0, 1_000_000), "w_stratify": (bool, None, None),
    "w_ngram": (int, 1, 3), "w_min_df": (int, 1, 100_000), "w_max_df": (float, 0.05, 1.0),
    "w_max_feat": (int, 0, 10_000_000), "w_lower": (bool, None, None), "w_binary": (bool, None, None),
    "w_sublinear": (bool, None, None), "w_accents": (bool, None, None),
    # Schritt 6 – Modell
    "w_alpha": (float, 1e-6, 1e4), "w_log_c": (float, 1e-6, 1e6), "w_svc_c": (float, 1e-6, 1e6),
    "w_log_iter": (int, 10, 100_000), "w_val_split": (float, 0.05, 0.4), "w_seed": (int, 0, 1_000_000),
    "w_threshold": (float, THRESHOLD_MIN, THRESHOLD_MAX), "w_fit_prior": (bool, None, None), "w_cw": (bool, None, None),
    "w_curve": (bool, None, None),
    "w_viz_cols": (list, None, None), "w_classes": (list, None, None),
    **{k: (str, None, None) for k in ("w_text", "w_label", "w_pos", "w_missing_text", "w_vec", "w_stop", "w_model",
                                      "w_val")},
}


def coerce_setting(key: str, value):
    """Wert aus einer (evtl. von Hand bearbeiteten) Konfiguration in Typ und Bereich zwingen.
    Rückgabe None = Wert unbrauchbar → ignorieren."""
    spec = SETTING_TYPES.get(key)
    if spec is None:
        return value
    typ, lo, hi = spec
    if typ is list:  # Spaltenlisten: nur Texte/Zahlen übernehmen
        if not isinstance(value, list):
            return None
        return [str(v) for v in value if isinstance(v, (str, int, float)) and not isinstance(v, bool)]
    if typ is str:
        return str(value) if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None
    try:
        if typ is bool:
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "ja", "yes")
            return bool(value)
        f = float(str(value).replace(",", "."))
        if not np.isfinite(f):  # JSON erlaubt in Python NaN/Infinity
            return None
        v = typ(f)
    except (TypeError, ValueError, OverflowError):
        return None
    return min(max(v, lo), hi)


MAX_NEW_COLUMNS = 200


def sanitize_config(cfg) -> tuple[dict, list[str]]:
    """Importierte Konfiguration (JSON) prüfen: falsche Typen und Werte werden verworfen statt die App
    abstürzen zu lassen. Rückgabe: (bereinigte Konfiguration, Liste verworfener Teile).
    Einträge anderer Versionen (z. B. "missing" oder w_features der Supervised-Version) werden still übergangen –
    Schritt 2 (Spalten-Aktionen und neue Spalten) wird übernommen."""
    if not isinstance(cfg, dict):
        raise ValueError(tx("Die Datei enthält keine Konfiguration (JSON-Objekt erwartet)."))
    out, skipped = {}, []

    def rows(key):
        value = cfg.get(key)
        if value is None:
            return None
        if not isinstance(value, list):
            skipped.append(key)
            return None
        good = [r for r in value if isinstance(r, dict)]
        if len(good) < len(value):
            skipped.append(tx("{key} ({v} Einträge)", key=key, v=len(value) - len(good)))
        return good

    step2 = rows("step2")
    if step2 is not None:
        out["step2"] = [{"Spalte": str(r["Spalte"]),
                         "Aktion": _clean(r.get("Aktion")) if _clean(r.get("Aktion")) in ACTIONS else A_KEEP,
                         "Parameter": _clean(r.get("Parameter"))[:200], "Neuer Name": _clean(r.get("Neuer Name"))[:200]}
                        for r in step2 if isinstance(r.get("Spalte"), str)]
    new_cols = rows("new_columns")
    if new_cols is not None:
        keys = ("name", "kind", "source", "definition", "action", "param")
        out["new_columns"] = [{k: _clean(s.get(k))[:MAX_FORMULA_LEN] for k in keys} for s in new_cols[:MAX_NEW_COLUMNS]
                              if s.get("kind") in NEW_KINDS and _clean(s.get("name"))]
    widgets = cfg.get("widgets")
    if widgets is not None and not isinstance(widgets, dict):
        skipped.append("widgets")
        widgets = None
    out["widgets"] = {}
    for k, v in (widgets or {}).items():
        if k not in SETTING_TYPES:  # z. B. w_features der Supervised-Version – hier ohne Bedeutung
            continue
        val = coerce_setting(k, v)
        if val is None:
            skipped.append(k)
        else:
            out["widgets"][k] = val
    return out, skipped


def _jsonable(obj):
    """Rekursiv in JSON-taugliche Python-Typen umwandeln."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj


def library_versions() -> tuple[pd.DataFrame, int]:
    """Versionen der wichtigsten Bibliotheken + Anzahl CPU-Kerne (scikit-learn rechnet auf der CPU)."""
    from importlib.metadata import PackageNotFoundError, version

    import matplotlib
    import scipy
    import sklearn

    try:
        skops_version = version("skops")
    except PackageNotFoundError:
        skops_version = "–"
    rows = [
        ("pandas", pd.__version__),
        ("numpy", np.__version__),
        ("scikit-learn", sklearn.__version__),
        ("scipy", scipy.__version__),
        ("joblib", joblib.__version__),
        ("skops", skops_version),
        ("matplotlib", matplotlib.__version__),
        ("seaborn", sns.__version__),
    ]
    return pd.DataFrame(rows, columns=[tx("Bibliothek"), tx("Version")]), os.cpu_count() or 1


# ---------------------------------------------------------------------------
# Daten laden
# ---------------------------------------------------------------------------
NO_HEADER_PREFIX = "Spalte_"  # Spaltennamen, wenn die Datei keine Kopfzeile hat (wie im Kurs: SMSSpamCollection)


def _header_is_data(df: pd.DataFrame) -> bool:
    """Sieht die erste Zeile (die pandas als Spaltennamen nimmt) nach Daten aus? Das ist so, wenn ein
    „Spaltenname“ einer Textspalte in derselben Spalte als häufiger Wert vorkommt (SMSSpamCollection: Spalte 'ham'
    enthält 'ham' über 4000-mal). Mindestens 2-mal und in 1 % der Zeilen: Ein einzelner Wert wie der Spaltenname
    (Kategorie 'typ' in Spalte 'typ') oder eine wiederholte Kopfzeile (zwei aneinandergehängte CSV-Exporte) machen
    aus einer echten Kopfzeile noch keine Datenzeile."""
    if df.shape[1] > 10 or len(df) < 5:
        return False
    for c in df.columns:
        s = df[c]
        if is_numeric(s):
            continue
        values = _to_str(s.head(5000)).dropna().str.strip()
        if int((values == str(c).strip()).sum()) >= max(2, 0.01 * len(values)):
            return True
    return False


def _sniff_sep(sample: str, sep: str) -> str:
    """Trennzeichen aus den ersten Zeilen: Ist das gewählte nicht in jeder Zeile gleich oft vorhanden, ein anderes
    aber schon (z. B. genau ein Tab je Zeile wie in SMSSpamCollection, Kommas nur im Text), gilt das andere."""
    lines = [ln for ln in sample.splitlines()[:50] if ln.strip()]
    if len(lines) < 3:
        return sep

    def steady(c):
        counts = {ln.count(c) for ln in lines}
        return len(counts) == 1 and counts.pop() >= 1

    if steady(sep):
        return sep
    other = [c for c in ["\t", ";", "|", ","] if c != sep and steady(c)]
    return other[0] if len(other) == 1 else sep


def read_csv(source, sep: str = ",", decimal: str = ".", encoding: str = "utf-8", auto: bool = True,
             header: bool | None = None):
    """CSV laden – Pfad (str/Path) oder Bytes (Upload). Rückgabe: (DataFrame, Encoding).

    auto=True: Lässt sich die Datei mit dem Trennzeichen nicht lesen oder wird nur 1 Spalte erkannt, werden Tab ;
    | , probiert; sieht eine Textspalte nach Dezimalkomma aus (1,5), wird mit decimal=',' neu gelesen; sieht die erste
    Zeile nach Daten aus (keine Kopfzeile), heißen die Spalten Spalte_1, Spalte_2 … header=True/False erzwingt eine
    Kopfzeile bzw. keine. Die tatsächlich verwendeten Optionen stehen in df.attrs['csv'].
    """
    if isinstance(source, (str, Path)):
        source = clean_path(source)
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(tx("Datei nicht gefunden: {path}", path=path))

    def _read(enc, sep_, dec_, head=True):
        buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else Path(source).expanduser()
        if head:
            return pd.read_csv(buf, sep=sep_, decimal=dec_, encoding=enc, low_memory=False)
        first = pd.read_csv(buf, sep=sep_, decimal=dec_, encoding=enc, low_memory=False, header=None, nrows=1)
        names = [f"{NO_HEADER_PREFIX}{i + 1}" for i in range(first.shape[1])]
        buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else Path(source).expanduser()
        return pd.read_csv(buf, sep=sep_, decimal=dec_, encoding=enc, low_memory=False, header=None, names=names)

    detected = []
    use_header = header is not False
    if auto:
        if isinstance(source, (bytes, bytearray)):
            raw = bytes(source[:65536])
        else:
            with open(Path(source).expanduser(), "rb") as fh:
                raw = fh.read(65536)
        sniffed = _sniff_sep(raw.decode(encoding, errors="replace"), sep)
        if sniffed != sep:
            sep = sniffed
            detected.append(tx("Trennzeichen"))

    def _read_any(enc):
        nonlocal sep
        try:
            return _read(enc, sep, decimal, use_header)
        except pd.errors.ParserError:
            if not auto:
                raise
            for cand in ["\t", ";", "|", ","]:  # z. B. Tab-getrennte Datei, gelesen mit ','
                if cand == sep:
                    continue
                try:
                    df_ = _read(enc, cand, decimal, use_header)
                except pd.errors.ParserError:
                    continue
                if df_.shape[1] > 1:
                    sep = cand
                    detected.append(tx("Trennzeichen"))
                    return df_
            raise

    try:
        df = _read_any(encoding)
    except UnicodeDecodeError:
        encoding = "latin-1" if encoding.lower().replace("-", "") == "utf8" else "utf-8"
        df = _read_any(encoding)
    if auto and df.shape[1] == 1 and not detected:
        first = str(df.columns[0])
        for cand in [";", "\t", "|", ","]:
            if cand != sep and cand in first:
                df2 = _read(encoding, cand, decimal, use_header)
                if df2.shape[1] > 1:
                    df, sep = df2, cand
                    detected.append(tx("Trennzeichen"))
                    break
    if auto and decimal == "." and sep != ",":
        comma_dec = [c for c in df.columns if not is_numeric(df[c])
                     and _to_str(df[c].dropna().head(200)).str.fullmatch(r"\s*-?\d+(?:\.\d{3})*,\d+\s*").mean() > 0.9]
        if comma_dec:
            df = _read(encoding, sep, ",", use_header)
            decimal = ","
            detected.append(tx("Dezimalkomma"))
    if header is None and auto and use_header and _header_is_data(df):
        df = _read(encoding, sep, decimal, False)
        use_header = False
        detected.append(tx("keine Kopfzeile"))
    strip_columns = use_header and any(str(c) != str(c).strip() for c in df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["csv"] = {"sep": sep, "decimal": decimal, "encoding": encoding, "detected": detected,
                       "strip_columns": strip_columns, "header": use_header}
    return df, encoding


def read_feature_info(source) -> dict:
    """Optionale Feature-Beschreibung (wie lending_club_info.csv): 1. Spalte Name, 2. Spalte Text."""
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else Path(clean_path(source)).expanduser()
    info = pd.read_csv(buf, sep=None, engine="python")
    if info.shape[1] < 2:
        raise ValueError(tx("Die Beschreibungs-CSV braucht mindestens 2 Spalten (Name, Beschreibung)."))
    return {str(k).strip(): str(v) for k, v in zip(info.iloc[:, 0], info.iloc[:, 1])}


def take_sample(df: pd.DataFrame, frac: float | None, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Optionale Stichprobe (Seitenleiste): df.sample(frac=…) – None oder 1.0 = alle Daten."""
    if not frac or frac >= 1:
        return df
    return df.sample(frac=frac, random_state=seed).reset_index(drop=True)


DTYPE_LABELS = [N_("Bool"), N_("Zahl (int)"), N_("Zahl (float)"), N_("Kategorie"), N_("Text")]  # Werte von dtype_label


def dtype_label(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return N_("Bool")
    if pd.api.types.is_integer_dtype(s):
        return N_("Zahl (int)")
    if pd.api.types.is_float_dtype(s):
        return N_("Zahl (float)")
    if isinstance(s.dtype, pd.CategoricalDtype):
        return N_("Kategorie")
    return N_("Text")


# ---------------------------------------------------------------------------
# 2) Datenaufbereitung 1 – Spalten umwandeln
# ---------------------------------------------------------------------------
def mean_words(s: pd.Series) -> float:
    """Durchschnittliche Anzahl Wörter je Wert (Stichprobe) – 0 für Zahlenspalten."""
    if is_numeric(s):
        return 0.0
    nn = s.dropna()
    nn = nn.sample(1000, random_state=0) if len(nn) > 1000 else nn
    return float(_to_str(nn).str.split().str.len().mean()) if len(nn) else 0.0


def is_free_text(s: pd.Series) -> bool:
    """Freitext (Bewertung, Nachricht, Beschreibung)? Textspalte mit im Mittel mindestens 3 Wörtern je Wert."""
    return mean_words(s) >= 3


def suggest_action(s: pd.Series, name: str = "") -> tuple[str, str]:
    """Vorschlag für eine Spalte: (Aktion, Parameter). Nur ein Vorschlag – bitte prüfen."""
    if re.match(r"^Unnamed:\s*\d+$", str(name)):
        return A_DROP, ""
    nn = s.dropna()
    if nn.empty:
        return A_DROP, ""
    if is_numeric(s):
        if str(name).lower() in ("id", "index") and nn.nunique() == len(s):
            return A_DROP, ""
        return A_KEEP, ""
    sample = nn.sample(500, random_state=0) if len(nn) > 500 else nn
    txt = _to_str(sample)
    if is_free_text(nn):  # Bewertungen, Nachrichten … – das ist der Text für Schritt 4
        return A_KEEP, ""
    if txt.str.fullmatch(_NUM_UNIT_RE).mean() >= 0.98:
        return A_NUM, ""
    looks_like_date = txt.str.contains(r"[-/.:]|[A-Za-z]{3}", regex=True) | txt.str.fullmatch(r"\d{8}T\d{6}")
    if looks_like_date.mean() >= 0.9 and _parse_dates(txt).notna().mean() >= 0.95:
        return A_YEAR, guess_date_format(txt)
    if nn.nunique() > 50 and mean_words(nn) <= 1.1:  # IDs wie review_id: viele verschiedene Werte ohne Leerzeichen
        return A_DROP, ""
    return A_KEEP, ""  # auch kurze Texte (Titel, Schlagzeilen, Ø unter 3 Wörtern) bleiben – sie können der Text sein


def column_profile(df: pd.DataFrame, info: dict | None = None) -> pd.DataFrame:
    """Tabelle für Schritt 2: Typ, Eindeutige, Fehlend %, Beispiel + vorgeschlagene Aktion."""
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        act, param = suggest_action(s, c)
        example = _fmt_value(nn.iloc[0]) if len(nn) else ""
        example = example.replace("\n", " ⏎ ")[:40]
        row = {
            "Spalte": c,
            "Typ": dtype_label(s),
            "Eindeutig": int(s.nunique()),
            "Fehlend %": round(float(s.isna().mean() * 100), 2),
            "Beispiel": example,
        }
        if info:
            row["Beschreibung"] = info.get(c, "")
        row.update({"Aktion": act, "Parameter": param, "Neuer Name": ""})
        rows.append(row)
    return pd.DataFrame(rows)


def actions_from_table(table: pd.DataFrame) -> list[dict]:
    """Editor-Tabelle aus Schritt 2 → Liste von Aktionen."""
    return [
        {
            "column": r["Spalte"],
            "action": _clean(r.get("Aktion")) or A_KEEP,
            "param": _clean(r.get("Parameter")),
            "new_name": _clean(r.get("Neuer Name")),
        }
        for r in table.to_dict("records")
    ]


def _transform_column(s: pd.Series, act: str, param: str, action: dict, fit: bool):
    """Eine Spalte umwandeln. Rückgabe: (neue Spalte, Log-Text).

    Bei "Datum → Jahr + Monat" ist das Ergebnis ein dict {"jahr": Serie, "monat": Serie}."""
    if act == A_KEEP:
        return s, tx("unverändert")
    if act == A_NUM:
        if is_numeric(s):
            return s.astype(float), tx("ist bereits numerisch")
        res = parse_numbers(s)
        return res, tx("Zahl extrahiert ({v:.0%} gültig)", v=res.notna().mean())
    if act in DATE_ACTIONS:
        if is_numeric(s) and not _is_yyyymmdd(s):
            if act == A_YEAR_MONTH:
                raise ValueError(tx("Zahl, aber kein Datum – erwartet Text wie 'Oct-2014' oder eine Zahl wie 20141013."))
            return s.astype(float), tx("ist bereits numerisch (unverändert)")  # z. B. neue Daten mit Jahreszahl
        if is_numeric(s):
            dt = _parse_dates(s.astype("Int64").astype(str), param or "%Y%m%d")  # Datum als Zahl 20141013
        else:
            dt = _parse_dates(s, param or None)
        found = tx("({v:.0%} erkannt)", v=dt.notna().mean())
        if act == A_YEAR_MONTH:
            return {"jahr": dt.dt.year.astype(float), "monat": dt.dt.month.astype(float)}, tx("Jahr + Monat {found}",
                                                                                              found=found)
        res = (dt.dt.year if act == A_YEAR else dt.dt.month).astype(float)
        return res, (tx("Jahr aus Datum {found}", found=found) if act == A_YEAR
                     else tx("Monat aus Datum {found}", found=found))
    if act in (A_FIRST, A_LAST):
        n = _int_param(param, 1 if act == A_FIRST else 5)
        txt = _to_str(s)
        res = txt.str[:n] if act == A_FIRST else txt.str[-n:]
        return res, tx("erste {n} Zeichen", n=n) if act == A_FIRST else tx("letzte {n} Zeichen", n=n)
    if act == A_REPLACE:
        mapping = parse_replace(param)
        if is_numeric(s):
            mapping = {_to_number(k): _to_number(v) for k, v in mapping.items()}
        return s.replace(mapping), tx("Werte ersetzt {mapping}", mapping=mapping)
    if act == A_TOPN:
        n = _int_param(param, 10)
        txt = _to_str(s)
        if fit:
            action["fitted"] = {"keep": [str(v) for v in txt.value_counts().index[:n]]}
        keep = (action.get("fitted") or {}).get("keep", [])
        return txt.where(txt.isin(keep) | txt.isna(), OTHER), tx("Top-{n} behalten, Rest → '{other}'",
                                                                 n=n, other=OTHER)
    if act == A_CAT:
        return _to_str(s), tx("als Kategorie (Text)")
    if act == A_LEN:
        return _to_str(s).str.len().astype(float), tx("Länge in Zeichen")
    if act == A_WORDS:
        return _to_str(s).str.split().str.len().astype(float), tx("Anzahl Wörter")
    raise ValueError(tx("Unbekannte Aktion: {act}", act=act))


def result_names(target: str, act: str) -> list[str]:
    """Namen der Ergebnisspalte(n) einer Aktion: meist eine, bei Jahr + Monat zwei."""
    return [f"{target}_jahr", f"{target}_monat"] if act == A_YEAR_MONTH else [target]


def apply_column_actions(df: pd.DataFrame, actions: list[dict], fit: bool = True):
    """2) Spalten-Aktionen anwenden.

    fit=True  (Trainingsdaten): gelernte Werte (z. B. Top-N-Liste) werden in action['fitted'] gemerkt.
    fit=False (neue Daten): gemerkte Werte werden genutzt; fehlt eine Quellspalte, wird die Aktion
              übersprungen (die Spalte liegt evtl. schon umgewandelt vor).
    Rückgabe: (df_neu, actions_mit_fitted, log, lineage)  –  lineage = {Ergebnisspalte: CSV-Spalte}
    """
    out = df.copy()
    done, log, lineage = [], [], {}
    for raw in actions:
        a = dict(raw)
        col = a["column"]
        act = _clean(a.get("action")) or A_KEEP
        param, new = _clean(a.get("param")), _clean(a.get("new_name"))
        a.update(action=act, param=param, new_name=new)
        done.append(a)
        target = new or col
        names = result_names(target, act)
        if act == A_DROP:
            if col in out.columns:
                out = out.drop(columns=col)
                log.append(tx("{col}: entfernt", col=col))
            continue
        if col not in out.columns:
            if fit:
                raise ValueError(tx("Spalte '{col}' ist nicht in den Daten.", col=col))
            continue
        if not fit and names != [col] and all(n in out.columns for n in names):
            continue  # neue Daten enthalten die Zielspalte(n) schon
        kind = src_kind(out[col])
        numeric_date = act in DATE_ACTIONS and kind in ("int", "float_int") and _is_yyyymmdd(out[col])
        try:
            res, msg = _transform_column(out[col], act, param, a, fit)
        except ValueError as exc:
            raise ValueError(tx("Spalte '{col}' ({act}): {exc}", col=col, act=tx(act), exc=exc)) from exc
        if fit:  # Infos für den generierten Python-Code (codegen.step2_code)
            a["fitted"] = {**(a.get("fitted") or {}), "src_kind": kind, "numeric_date": numeric_date}
        parts = res if isinstance(res, dict) else {None: res}
        clash = [n for n in names if n != col and n in out.columns]
        if clash:
            raise ValueError(tx("Neuer Name {clash} (für '{col}') existiert bereits als Spalte.",
                                clash=clash, col=col))
        if names == [col]:
            out[col] = res
        else:
            pos = out.columns.get_loc(col)
            out = out.drop(columns=col)
            for k, (n, values) in enumerate(zip(names, parts.values())):
                out.insert(pos + k, n, values)
            msg += f" → {', '.join(names)}" if len(names) > 1 else tx(", umbenannt in '{target}'", target=target)
        for n in names:
            lineage[n] = col
        if act != A_KEEP or target != col:
            log.append(f"{col}: {msg}")
    for c in out.columns:
        lineage.setdefault(c, c)
    return out, done, log, lineage


# ---------------------------------------------------------------------------
# 2b) Neue Spalten – nach den Spalten-Aktionen, auch für neue Daten (Rezept)
# ---------------------------------------------------------------------------
N_MAP = N_("Werte zuordnen")
N_FORMULA = N_("Formel")
N_COND = N_("Bedingung → 0/1")
N_COPY = N_("Kopie + Aktion")
NEW_KINDS = [N_MAP, N_FORMULA, N_COND, N_COPY]
NEW_HELP = {
    N_MAP: (N_("Quellspalte · alt=neu; …; sonst=…"),
            N_("z. B. loan_repaid aus loan_status: Fully Paid=1; Charged Off=0 – nicht genannte Werte bleiben "
               "leer (oder 'sonst=…'); sind alle neuen Werte Zahlen, wird die Spalte eine Zahlenspalte")),
    N_FORMULA: (N_("Formel"),
                N_("Rechnen mit Zahlenspalten: + − * / ** // % und Klammern; Funktionen abs, log, log10, "
                   "sqrt, exp, round – z. B. loan_amnt / annual_inc")),
    N_COND: (N_("Bedingung"),
             N_("1 = erfüllt, 0 = nicht erfüllt, leer bei Lücken: dti > 30 · home_ownership == 'RENT' · "
                "(dti > 30) and (term == 60)")),
    N_COPY: (N_("Quellspalte · Aktion · Parameter"),
             N_("wie eine Aktion aus der Tabelle oben – das Original bleibt erhalten")),
}
FORMULA_FUNCS = {"abs": "np.abs", "log": "np.log", "log10": "np.log10", "sqrt": "np.sqrt", "exp": "np.exp",
                 "round": "np.round"}
MAX_FORMULA_LEN = 1000
MAX_FORMULA_DEPTH = 100  # Verschachtelung (Knoten im Syntaxbaum) – schützt vor Rekursionsfehlern


def _ast_depth(tree) -> int:
    """Tiefe des Syntaxbaums – ohne Rekursion (sonst scheitert schon das Messen an z. B. '------…a')."""
    import ast

    best, stack = 0, [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        best = max(best, depth)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return best


class UnknownColumnError(ValueError):
    """Formel benutzt eine Spalte, die es (in diesen Daten) nicht gibt."""


def raw_sources(lineage: dict | None, col: str) -> list[str]:
    """CSV-Rohspalte(n), aus denen eine Spalte entsteht (neue Spalten können mehrere haben)."""
    v = (lineage or {}).get(col, col)
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else [str(v)]


def merge_lineage(lineage: dict, sources: dict) -> dict:
    """Herkunft der neuen Spalten ergänzen: {neue Spalte: [benutzte Spalten]} → CSV-Rohspalten."""
    lin = dict(lineage or {})
    for new, used in sources.items():
        raw = []
        for c in used:
            raw += [r for r in raw_sources(lin, c) if r not in raw]
        lin[new] = raw[0] if len(raw) == 1 else raw
    return lin


def parse_mapping(text: str):
    """'Fully Paid=1; Charged Off=0; sonst=0' → ({'Fully Paid': 1.0, 'Charged Off': 0.0}, 0.0)."""
    try:
        mapping = parse_replace(text)
    except ValueError as exc:
        raise ValueError(str(exc).replace(tx("Werte ersetzen"), tx("Zuordnung"))) from exc
    default = None
    for key in [k for k in mapping if k.lower() in ("sonst", "rest")]:
        default = mapping.pop(key)
    if not mapping:
        raise ValueError(tx("Zuordnung: mindestens ein Wert nötig, z. B. Fully Paid=1"))

    def num(v):
        try:
            return float(str(v).replace(",", "."))
        except ValueError:
            return None

    values = list(mapping.values()) + ([default] if default is not None else [])
    if all(num(v) is not None for v in values):  # alles Zahlen → Zahlenspalte
        mapping = {k: num(v) for k, v in mapping.items()}
        default = None if default is None else num(default)
    return mapping, default


def translate_expression(expr: str, columns) -> tuple[str, list[str]]:
    """Formel/Bedingung → pandas-Ausdruck über df['…'] (nur Rechnen, Vergleiche und wenige Funktionen –
    kein beliebiger Python-Code). Spaltennamen direkt oder in `Backticks` (bei Leer-/Sonderzeichen).

    'dti > 30 and term == 60' → "(df['dti'] > 30) & (df['term'] == 60)".  Rückgabe: (Ausdruck, Spalten)."""
    import ast

    columns = [str(c) for c in columns]
    text = str(expr or "").strip()
    if not text:
        raise ValueError(tx("Formel fehlt."))
    if len(text) > MAX_FORMULA_LEN:
        raise ValueError(tx("Formel zu lang (höchstens {max_formula_len} Zeichen).", max_formula_len=MAX_FORMULA_LEN))
    ticks = {}

    def tick(m):
        key = f"_spalte_{len(ticks)}_"
        ticks[key] = m.group(1)
        return key

    text = re.sub(r"`([^`]+)`", tick, text)
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(tx("Formel ungültig ({msg}).", msg=exc.msg)) from exc
    except (RecursionError, MemoryError) as exc:
        raise ValueError(tx("Formel zu stark verschachtelt.")) from exc
    if _ast_depth(tree) > MAX_FORMULA_DEPTH:
        raise ValueError(tx("Formel zu stark verschachtelt – bitte in mehrere neue Spalten aufteilen "
                         "(z. B. erst teil1 = …, dann gesamt = teil1 + …)."))
    used: list[str] = []

    def has_col(node):
        return any(isinstance(x, ast.Subscript) for x in ast.walk(node))

    def number(node):
        """Zahl-Literal (auch mit Vorzeichen) → Wert, sonst None."""
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = number(node.operand)
            return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
        return None

    def as_float(node):
        """Ganze Zahlen direkt an einer Potenz als Kommazahl rechnen: 9 ** 9 ** 9 wird so zu einem
        schnellen Überlauf statt einer riesigen Ganzzahl, die das Programm minutenlang blockiert."""
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            try:
                return ast.Constant(float(node.value))
            except OverflowError as exc:
                raise ValueError(tx("Zahl in der Formel ist zu groß.")) from exc
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(node.op, as_float(node.operand))
        return node

    def col(name):
        if name not in columns:
            raise UnknownColumnError(tx("Unbekannte Spalte '{name}' – Namen mit Leer-/Sonderzeichen in `Backticks`.",
                                        name=name))
        if name not in used:
            used.append(name)
        return ast.Subscript(value=ast.Name("df", ast.Load()), slice=ast.Constant(name), ctx=ast.Load())

    def conv(n):
        if isinstance(n, ast.Name):
            return col(ticks.get(n.id, n.id))
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, str, bool)):
            return n
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                                                          ast.Mod, ast.Pow)):
            left, right = conv(n.left), conv(n.right)
            if any(isinstance(x, ast.Constant) and isinstance(x.value, str) for x in (left, right)):
                raise ValueError(tx("Text in Anführungszeichen ist nur in Vergleichen erlaubt, z. B. "
                                 "home_ownership == 'RENT'."))
            if isinstance(n.op, ast.Pow):
                exp = number(right)
                if exp is None and not has_col(right) and not any(isinstance(x, ast.Pow) for x in ast.walk(right)):
                    try:  # konstanter Ausdruck ohne Potenz, z. B. (1/2) – ungefährlich auszurechnen
                        exp = eval(compile(ast.fix_missing_locations(ast.Expression(right)), "<exp>", "eval"),
                                   {"__builtins__": {}}, {})  # noqa: S307 – nur Zahlen und + - * / // %
                    except Exception:  # noqa: BLE001
                        exp = None
                    exp = exp if isinstance(exp, (int, float)) and not isinstance(exp, bool) else None
                if not has_col(right) and (exp is None or abs(exp) > 100):
                    raise ValueError(tx("Potenz: Der Exponent muss eine Zahl zwischen -100 und 100 sein "
                                     "(oder eine Spalte enthalten)."))
                left, right = as_float(left), as_float(right)
            return ast.BinOp(left, n.op, right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            return ast.UnaryOp(n.op, conv(n.operand))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return ast.UnaryOp(ast.Invert(), conv(n.operand))
        if isinstance(n, ast.BoolOp):
            op = ast.BitAnd() if isinstance(n.op, ast.And) else ast.BitOr()
            out = conv(n.values[0])
            for v in n.values[1:]:
                out = ast.BinOp(out, op, conv(v))
            return out
        if isinstance(n, ast.Compare):
            if not all(isinstance(o, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for o in n.ops):
                raise ValueError(tx("Vergleich nicht erlaubt – nur == != < <= > >=."))
            parts, left = [], n.left
            for o, right in zip(n.ops, n.comparators):  # a < b < c → (a < b) & (b < c)
                parts.append(ast.Compare(conv(left), [o], [conv(right)]))
                left = right
            out = parts[0]
            for q in parts[1:]:
                out = ast.BinOp(out, ast.BitAnd(), q)
            return out
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FORMULA_FUNCS
                and not n.keywords and 1 <= len(n.args) <= (2 if n.func.id == "round" else 1)):
            mod, fn = FORMULA_FUNCS[n.func.id].split(".")
            args = [conv(n.args[0])]
            if len(n.args) == 2:  # round(x, Stellen): Stellen als kleine ganze Zahl
                digits = number(n.args[1])
                if not isinstance(digits, int) or not 0 <= digits <= 10:
                    raise ValueError(tx("round(x, n): n muss eine ganze Zahl von 0 bis 10 sein."))
                args.append(ast.Constant(digits))
            return ast.Call(ast.Attribute(ast.Name(mod, ast.Load()), fn, ast.Load()), args, [])
        what = n.func.id if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else type(n).__name__
        raise ValueError(tx("In Formeln nicht erlaubt: {what}. Erlaubt: + - * / ** // %, Vergleiche, and/or/not, "
                            "{formula_funcs}.",
                            what=what, formula_funcs=', '.join(FORMULA_FUNCS)))

    try:
        code = ast.unparse(ast.fix_missing_locations(conv(tree.body)))
    except RecursionError as exc:
        raise ValueError(tx("Formel zu stark verschachtelt.")) from exc
    if not used:
        raise ValueError(tx("Die Formel muss mindestens eine Spalte enthalten."))
    return code, used


def _eval_expression(code: str, df: pd.DataFrame) -> pd.Series:
    """Übersetzten Ausdruck auswerten – nur df und np sind sichtbar (siehe translate_expression)."""
    try:
        with np.errstate(all="ignore"):
            res = eval(code, {"__builtins__": {}, "np": np}, {"df": df})  # noqa: S307 – geprüfter Ausdruck
    except Exception as exc:  # noqa: BLE001
        raise ValueError(tx("Formel lässt sich nicht berechnen: {exc}", exc=exc)) from exc
    if not isinstance(res, pd.Series):
        raise ValueError(tx("Die Formel ergibt keinen Wert je Zeile."))
    return res


def apply_new_columns(df: pd.DataFrame, specs: list[dict] | None, fit: bool = True):
    """2b) Neue Spalten berechnen – nach den Spalten-Aktionen, in der Reihenfolge der Liste (spätere dürfen
    frühere benutzen). Jede Definition: name, kind (NEW_KINDS), source, definition, action, param.

    fit=False (neue Daten): fehlt eine benötigte Spalte, wird die neue Spalte übersprungen.
    Rückgabe: (df_neu, Definitionen mit 'fitted', log, sources) – sources = {neue Spalte: [benutzte Spalten]}
    """
    out = df.copy()
    done, log, sources = [], [], {}
    for raw in specs or []:
        s = dict(raw)
        name, kind = _clean(s.get("name")), _clean(s.get("kind"))
        src, definition = _clean(s.get("source")), _clean(s.get("definition"))
        act, param = _clean(s.get("action")) or A_KEEP, _clean(s.get("param"))
        s.update(name=name, kind=kind, source=src, definition=definition, action=act, param=param)
        label = tx("Neue Spalte '{name}'", name=name)
        try:
            if not name:
                raise ValueError(tx("Name fehlt."))
            if kind not in NEW_KINDS:
                raise ValueError(tx("unbekannte Art '{kind}'.", kind=kind))
            if kind in (N_FORMULA, N_COND):
                code, used = translate_expression(definition, out.columns)
            else:
                if not src:
                    raise ValueError(tx("Quellspalte fehlt."))
                if src not in out.columns:
                    raise UnknownColumnError(tx("Spalte '{src}' gibt es nicht.", src=src))
                used = [src]
        except UnknownColumnError as exc:
            if not fit:
                continue  # neue Daten ohne diese Quelle – die neue Spalte wird nicht gebraucht oder liegt schon vor
            raise ValueError(f"{label}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        names = result_names(name, act) if kind == N_COPY else [name]
        if fit:
            clash = [n for n in names if n in out.columns]
            if clash:
                raise ValueError(tx("{label}: {clash} gibt es schon – bitte anderen Namen wählen.",
                                    label=label, clash=clash))
        fitted = dict(s.get("fitted") or {})
        try:
            if kind == N_MAP:
                mapping, default = parse_mapping(definition)
                txt = _cat(out[src])
                res = txt.map(mapping)
                if default is not None:
                    res = res.where(txt.isna() | txt.isin(list(mapping)), default)
                numeric = all(isinstance(v, float) for v in mapping.values())
                res = res.astype(float) if numeric else res
                if fit and txt.notna().any() and res.notna().sum() == 0:
                    seen = ", ".join(map(str, txt.dropna().unique()[:6]))
                    raise ValueError(tx("Kein Wert von '{src}' passt zur Zuordnung – Schreibweise prüfen "
                                        "(Groß-/Kleinschreibung). Vorhandene Werte z. B.: {seen}",
                                        src=src, seen=seen))
                fitted.update(src_kind=src_kind(out[src]), numeric=numeric)
                unmapped = int((txt.notna() & res.isna()).sum())
                msg = tx("aus {src} zugeordnet", src=src) + (
                    tx(" ({unmapped} Werte ohne Zuordnung → leer)", unmapped=unmapped) if unmapped else "")
            elif kind in (N_FORMULA, N_COND):
                res = _eval_expression(code, out)
                if kind == N_COND:
                    if not pd.api.types.is_bool_dtype(res):
                        raise ValueError(tx("Eine Bedingung muss wahr/falsch ergeben, z. B. dti > 30."))
                    res = res.astype(float).where(out[used].notna().all(axis=1))
                    msg = tx("Bedingung {definition} → 1 bei {n} Zeilen",
                             definition=definition, n=int((res == 1).sum()))
                else:
                    if pd.api.types.is_bool_dtype(res):
                        res = res.astype(float)
                    if not pd.api.types.is_numeric_dtype(res):
                        raise ValueError(tx("Die Formel ergibt keine Zahl – nur mit Zahlenspalten rechnen."))
                    res = res.astype(float)
                    n_inf = int(np.isinf(res.to_numpy()).sum())
                    res = res.replace([np.inf, -np.inf], np.nan)
                    msg = f"= {definition}" + (tx(" ({n_inf} × Division durch 0 → leer)", n_inf=n_inf) if n_inf else "")
                fitted.update(code=code, used=used)
            else:  # Kopie + Aktion
                if act == A_DROP:
                    raise ValueError(tx("'entfernen' ist hier nicht sinnvoll – Aktion wählen oder 'behalten' "
                                        "für eine Kopie."))
                kind_src = src_kind(out[src])
                numeric_date = act in DATE_ACTIONS and kind_src in ("int", "float_int") and _is_yyyymmdd(out[src])
                res, msg = _transform_column(out[src], act, param, s, fit)
                fitted.update((s.get("fitted") or {}), src_kind=kind_src, numeric_date=numeric_date)
                msg = tx("Kopie von {src}", src=src) + ("" if act == A_KEEP else f" + {tx(act)}: {msg}")
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        if fit:
            s["fitted"] = fitted
        parts = list(res.values()) if isinstance(res, dict) else [res]
        for n, values in zip(names, parts):
            out[n] = values
            sources[n] = list(used)
        done.append(s)
        log.append(tx("{names} (neu): {msg}", names=', '.join(names), msg=msg))
    return out, done, log, sources


# ---------------------------------------------------------------------------
# 4) Datenaufbereitung 2 – Text, Label, Klassen, Split, Vektorisierung
# ---------------------------------------------------------------------------
MIN_ROWS = 20  # weniger Zeilen reichen nicht für Training und Test
MAX_CLASSES = 50
TEXT_DROP, TEXT_EMPTY = "drop", "empty"  # fehlender/leerer Text: Zeile entfernen | als leeren Text behandeln
MISSING_TEXT = {TEXT_DROP: N_("Zeilen ohne Text entfernen"), TEXT_EMPTY: N_("als leeren Text behandeln")}
VEC_COUNT, VEC_TFIDF = "count", "tfidf"
VECTORIZERS = {VEC_COUNT: N_("CountVectorizer – Wörter zählen (Bag of Words, wie im Kurs)"),
               VEC_TFIDF: N_("TfidfVectorizer – TF-IDF (Wörter, die in vielen Texten vorkommen, zählen weniger)")}
STOP_WORDS = {"": N_("keine"), "english": N_("Englisch (Liste von scikit-learn)")}
BIG_VOCAB = 200_000  # ab hier Hinweis: max_features oder min_df
FINGERPRINTS = 5000  # so viele Trainingstexte (Prüfsummen) merkt sich das Rezept für Schritt 10
COVERAGE_SAMPLE = 5000  # Stichprobe für den Anteil bekannter Wörter
MIN_CLASS_TEXTS = 5  # weniger Trainingstexte je Klasse → Hinweis in Schritt 4


def _n_label_values(nn: pd.Series) -> int:
    """Anzahl verschiedener Label-Werte – Text ohne Leerzeichen am Rand ('ham ' = 'ham', wie bei der Kodierung)."""
    return int(_cat(nn).nunique()) if not is_numeric(nn) else int(nn.nunique())


def label_kind(s: pd.Series) -> str:
    """bool, numeric oder text – bestimmt, wie das Label kodiert wird."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    return "numeric" if is_numeric(s) else "text"


def missing_label(s: pd.Series) -> pd.Series:
    """Zeilen ohne Label: fehlend (NaN) oder – bei Text – nur Leerzeichen (' '). Ein leeres Label ist keine Klasse;
    sonst würde aus 'ham'/'spam' mit ein paar leeren Zellen still eine Aufgabe mit drei Klassen."""
    if label_kind(s) != "text":
        return s.isna()
    return s.isna() | _cat(s).eq("").fillna(False).astype(bool)


def label_display(s: pd.Series) -> pd.Series:
    """Label je Zeile als Anzeigetext ('5', 'spam', 'True'); NaN und leere Texte → NaN. Zahlen ohne '.0'."""
    if label_kind(s) == "text":
        t = _cat(s)
        return t.where(t != "")
    return s.map(lambda v: _fmt_value(v) if pd.notna(v) else np.nan)


def _label_display(s: pd.Series) -> dict:
    """{Anzeigetext: Originalwert} für alle Label-Werte (1.0 → '1', True → 'True'); leere Texte zählen nicht."""
    nn = s.dropna()
    if label_kind(s) == "text":
        return {v: v for v in _cat(nn).unique() if v != ""}
    return {_fmt_value(v): _py(v) for v in nn.unique()}


def _natural_key(v) -> list:
    """'2 years' < '10+ years', 'A2' < 'A10' – Reihenfolge von Klassen in Tabellen."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", str(v))]


def label_values(s: pd.Series) -> list[str]:
    """Alle Klassen einer Label-Spalte als Text, sortiert (Zahlen nach Wert: 1 < 2 < 10) – für die Auswahl."""
    disp = _label_display(s)
    if label_kind(s) == "text":
        return sorted(disp, key=_natural_key)
    return sorted(disp, key=lambda k: disp[k])


def label_counts(s: pd.Series) -> pd.Series:
    """Anzahl Zeilen je Klasse (Anzeigetext), in der Reihenfolge von label_values()."""
    counts = label_display(s.dropna()).value_counts()
    return counts.reindex(label_values(s)).fillna(0).astype(int)


def label_classes(s: pd.Series, positive_class=None) -> list[str]:
    """Klassen als Text in Kodier-Reihenfolge (Index = Zahl im Modell). Binär: [negativ, positiv]."""
    classes = label_values(s)
    if len(classes) == 2 and positive_class is not None and str(positive_class) in classes:
        pos = str(positive_class)
        classes = [c for c in classes if c != pos] + [pos]
    return classes


def encode_label(s: pd.Series, recipe: dict) -> np.ndarray:
    """Label → Zahlen (0/1 bzw. 0..K-1). Unbekannte Klassen → NaN.

    Zahlen- und Bool-Labels werden über den Wert kodiert (nicht über den Text), damit 5.0, 5 und '5' in neuen Daten
    dieselbe Klasse sind."""
    kind = recipe.get("label_kind")
    if kind == "bool":
        mapping = {str(v).lower(): i for i, v in enumerate(recipe["class_values"])}  # 'true'/'false'
        txt = _to_str(s).str.strip().str.lower().replace({"1": "true", "0": "false", "1.0": "true", "0.0": "false"})
        return txt.map(mapping).to_numpy(dtype="float64")
    if kind == "numeric":
        mapping = {float(v): i for i, v in enumerate(recipe["class_values"])}
        return parse_numbers(s).map(mapping).to_numpy(dtype="float64")
    mapping = {c: i for i, c in enumerate(recipe["classes"])}
    return _cat(s).map(mapping).to_numpy(dtype="float64")


def text_values(s: pd.Series) -> pd.Series:
    """Textspalte als Text: fehlende Werte → '' (leerer Text)."""
    return _to_str(s).fillna("")


def text_columns(df: pd.DataFrame) -> list[str]:
    """Spalten, die als Text in Frage kommen (keine Zahlen), die längsten Texte zuerst."""
    cols = [c for c in df.columns if not is_numeric(df[c])]
    return sorted(cols, key=lambda c: -mean_words(df[c]))


def guess_text_column(df: pd.DataFrame) -> str | None:
    """Vorschlag für die Textspalte: die Spalte mit den längsten Texten."""
    cols = text_columns(df)
    return cols[0] if cols else None


def text_fingerprints(texts) -> np.ndarray:
    """Prüfsumme je Text (klein geschrieben, Leerraum vereinheitlicht) – erkennt wortgleiche Texte im Training."""
    norm = pd.Series(list(texts), dtype=object).astype(str).str.lower().str.split().str.join(" ")
    return pd.util.hash_pandas_object(norm, index=False).to_numpy()


def _non_blank(texts) -> np.ndarray:
    return np.array([bool(str(t).strip()) for t in texts], dtype=bool)


def rare_classes(y_train, classes, below: int) -> list[str]:
    """Klassen mit weniger als `below` Trainingstexten (in der Reihenfolge der Klassen)."""
    counts = np.bincount(np.asarray(y_train).astype(int), minlength=len(classes))
    return [cl for cl, n in zip(classes, counts) if n < below]


@dataclass
class VectorizerConfig:
    kind: str = VEC_COUNT  # count (wie im Kurs) | tfidf
    lowercase: bool = True
    stop_words: str = ""  # "" = keine | "english"
    ngram_max: int = 1  # ngram_range = (1, ngram_max): 2 = auch Wortpaare
    min_df: int = 1  # Wort muss in mindestens so vielen Trainingstexten vorkommen
    max_df: float = 1.0  # … und in höchstens diesem Anteil der Texte
    max_features: int = 0  # 0 = alle Wörter
    binary: bool = False  # nur vorhanden/nicht vorhanden statt Anzahl
    sublinear_tf: bool = False  # nur TF-IDF: 1 + log(Anzahl)
    strip_accents: bool = False  # 'unicode': café = cafe


def vectorizer_kwargs(vc: VectorizerConfig) -> dict:
    """Parameter des Vektorisierers – dieselben in App und erzeugtem Code."""
    kw = dict(lowercase=bool(vc.lowercase), stop_words=vc.stop_words or None, ngram_range=(1, int(vc.ngram_max)),
              min_df=int(vc.min_df), max_df=float(vc.max_df), max_features=int(vc.max_features) or None,
              binary=bool(vc.binary), strip_accents="unicode" if vc.strip_accents else None)
    if vc.kind == VEC_TFIDF:
        kw["sublinear_tf"] = bool(vc.sublinear_tf)
    return kw


def make_vectorizer(vc: VectorizerConfig):
    """CountVectorizer (Bag of Words wie im Kurs) oder TfidfVectorizer – noch nicht gefittet."""
    if vc.kind not in VECTORIZERS:
        raise ValueError(tx("Unbekannter Vektorisierer: {v}", v=vc.kind))
    if vc.stop_words not in STOP_WORDS:
        raise ValueError(tx("Unbekannte Stoppwort-Liste: {v}", v=vc.stop_words))
    kw = vectorizer_kwargs(vc)
    return TfidfVectorizer(**kw) if vc.kind == VEC_TFIDF else CountVectorizer(**kw)


def vectorizer_config(d: dict | None) -> VectorizerConfig:
    """VectorizerConfig aus einem Rezept (unbekannte Einträge werden ignoriert)."""
    known = set(VectorizerConfig.__dataclass_fields__)
    return VectorizerConfig(**{k: v for k, v in (d or {}).items() if k in known})


def fit_vectorizer(vc: VectorizerConfig, texts):
    """Vektorisierer auf den Trainingstexten fitten. Rückgabe: (Vektorisierer, Matrix). Verständliche Meldung, wenn
    kein Wort übrig bleibt (nur Stoppwörter, min_df/max_df zu streng)."""
    vec = make_vectorizer(vc)
    try:
        X = vec.fit_transform(texts)
    except ValueError as exc:
        msg = str(exc)
        if "empty vocabulary" in msg or "no terms remain" in msg or "max_df corresponds" in msg:
            raise ValueError(tx("Kein Wort bleibt im Vokabular – die Texte bestehen nur aus Stoppwörtern oder "
                                "einzelnen Zeichen, oder min_df / max_df sind zu streng ({exc}).", exc=msg)) from exc
        raise
    return vec, X.tocsr()


def vocabulary_coverage(vectorizer, texts) -> tuple[np.ndarray, np.ndarray]:
    """Je Text: Anzahl Wörter (nach Kleinschreibung, Stoppwörtern, Wortpaaren – wie im Modell) und davon bekannte."""
    analyze, vocab = vectorizer.build_analyzer(), vectorizer.vocabulary_
    total, known = np.zeros(len(texts), dtype=int), np.zeros(len(texts), dtype=int)
    for i, t in enumerate(texts):
        tokens = analyze(str(t))
        total[i] = len(tokens)
        known[i] = sum(1 for w in tokens if w in vocab)
    return total, known


@dataclass
class TextPrep:
    recipe: dict
    vectorizer: object  # nur auf den Trainingstexten gefittet
    X_train: sparse.csr_matrix
    X_test: sparse.csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray  # Zeilen-Index (df aus Schritt 2) – für Einzelfall und Export
    idx_test: np.ndarray
    text_train: np.ndarray
    text_test: np.ndarray
    log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def prepare_text_data(
    df: pd.DataFrame,
    text_col: str,
    label: str,
    classes: list | None = None,
    positive_class=None,
    missing_text: str = TEXT_DROP,
    test_size: float = 0.3,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    vec: VectorizerConfig | None = None,
    column_actions: list | None = None,
    lineage: dict | None = None,
    meta: dict | None = None,
    new_columns: list | None = None,
) -> TextPrep:
    """4) Label kodieren (optional nur ausgewählte Klassen – wie im Kurs nur 1 und 5 Sterne) → train_test_split →
    Vektorisierer NUR auf den Trainingstexten fitten → Trainings- und Testtexte in Zahlen umwandeln.

    Anders als im Kurs (cv.fit_transform auf allen Texten vor dem Split) kennt der Vektorisierer die Testtexte nicht –
    sie werden umgewandelt wie später neue Texte (Wörter, die nur im Test vorkommen, werden ignoriert)."""
    vec = vec or VectorizerConfig()
    lineage = dict(lineage or {})
    log, warn = [], []
    for c, what in ((text_col, tx("Textspalte")), (label, tx("Label"))):
        if not c or c not in df.columns:
            raise ValueError(tx("{what} '{c}' ist nicht in den Daten.", what=what, c=c))
    if text_col == label:
        raise ValueError(tx("Textspalte und Label müssen verschiedene Spalten sein."))
    if is_numeric(df[text_col]):
        raise ValueError(tx("Die Spalte '{c}' enthält Zahlen, keinen Text – bitte die Spalte mit den Texten wählen.",
                            c=text_col))
    if not 0 < float(test_size) < 1:
        raise ValueError(tx("test_size muss zwischen 0 und 1 liegen."))
    d = df[[text_col, label]].copy()
    n0 = len(d)
    d = d[~missing_label(d[label]).to_numpy()]  # fehlend oder nur Leerzeichen
    if len(d) < n0:
        log.append(tx("{v} Zeilen ohne Label entfernt", v=_n(n0 - len(d))))
    all_values = label_values(d[label])
    selected = None
    if classes:
        wanted = {str(c) for c in classes}
        keep = [v for v in all_values if v in wanted]
        if len(keep) < 2:
            raise ValueError(tx("Bitte mindestens 2 Klassen auswählen (vorhanden: {v}).",
                                v=", ".join(all_values[:20])))
        if len(keep) < len(all_values):
            selected = keep
            n1 = len(d)
            d = d[label_display(d[label]).isin(keep)]
            log.append(tx("Nur die Klassen {keep}: {v} Zeilen mit anderen Klassen entfernt (wie im Kurs: nur 1 und 5 "
                          "Sterne)", keep=", ".join(keep), v=_n(n1 - len(d))))

    # Text: fehlend/leer → Zeile entfernen (Standard) oder als leeren Text behandeln
    texts = text_values(d[text_col])
    blank = (texts.str.strip() == "").to_numpy(dtype=bool)
    n_blank = int(blank.sum())
    if n_blank and missing_text != TEXT_EMPTY:
        d, texts = d[~blank], texts[~blank]
        log.append(tx("{v} Zeilen ohne Text entfernt", v=_n(n_blank)))
    elif n_blank:
        log.append(tx("{v} Zeilen ohne Text – als leerer Text behandelt (das Modell sagt dort nur nach der "
                      "Häufigkeit der Klassen voraus)", v=_n(n_blank)))
    k = _n_label_values(d[label]) if len(d) else 0
    if k < 2:
        raise ValueError(tx("Das Label hat nur {k} Klasse – für eine Klassifikation sind mindestens 2 nötig.", k=k))
    if k > MAX_CLASSES:
        raise ValueError(tx("Das Label hat {k} Klassen – höchstens {m} sind möglich. Ist das wirklich das Label (oder "
                            "eine ID / ein Text)? Sonst in Schritt 4 nur einige Klassen auswählen.", k=k, m=MAX_CLASSES))
    if len(d) < MIN_ROWS:
        raise ValueError(tx("Nur noch {n} Zeilen übrig – zu wenig für Training und Test (mindestens {m}).",
                            n=len(d), m=MIN_ROWS))
    task = T_BIN if k == 2 else T_MULTI

    # Label kodieren – die Klassen beschreiben die Aufgabe, sie werden nicht "gelernt"
    classes = label_classes(d[label], positive_class)
    kind = label_kind(d[label])
    disp = _label_display(d[label])
    class_values = [disp[c] for c in classes]
    y = encode_label(d[label], {"label_kind": kind, "classes": classes, "class_values": class_values})
    if np.isnan(y).any():
        raise ValueError(tx("Label konnte nicht vollständig kodiert werden."))
    counts = np.bincount(y.astype(int), minlength=len(classes))
    log.append(tx("Klassen: ") + ", ".join(f"{cl} → {i} ({_n(n)})" for i, (cl, n) in enumerate(zip(classes, counts))))

    # train_test_split ZUERST – das Vokabular wird danach nur aus den Trainingstexten gelernt
    strat_y = None
    if stratify:
        n_test = int(np.ceil(float(test_size) * len(y)))
        if counts.min() < 2:
            warn.append(tx("Stratify nicht möglich (eine Klasse hat < 2 Zeilen) – Split ohne stratify."))
        elif n_test < len(classes) or len(y) - n_test < len(classes):
            warn.append(tx("Stratify nicht möglich: Testmenge ({n_test} Zeilen) kleiner als Anzahl Klassen ({k}) – "
                           "Split ohne stratify. Tipp: test_size erhöhen.", n_test=n_test, k=len(classes)))
        else:
            strat_y = y
    # gleiche Aufteilung wie train_test_split(texts, y, …): sie hängt nur von Zeilenzahl, Seed und y ab
    pos_tr, pos_te = train_test_split(np.arange(len(d)), test_size=float(test_size), random_state=int(random_state),
                                      stratify=strat_y)
    t_all = texts.to_numpy(dtype=object)
    t_tr, t_te = t_all[pos_tr], t_all[pos_te]
    y_tr, y_te = y[pos_tr].astype(int), y[pos_te].astype(int)
    missing = [cl for i, cl in enumerate(classes) if i not in set(y_tr.tolist())]
    if missing:
        raise ValueError(tx("Die Klasse(n) {c} kommen in den Trainingsdaten nicht vor – stratify einschalten, test_size "
                            "verkleinern oder die Klasse in Schritt 4 abwählen.", c=", ".join(missing)))
    rare = rare_classes(y_tr, classes, MIN_CLASS_TEXTS)
    if rare:  # lernt das Modell kaum; LinearSVC (Kalibrierung mit Kreuzvalidierung) braucht mindestens 2 je Klasse
        warn.append(tx("Sehr wenige Trainingstexte für {c} (jeweils weniger als {m}) – solche Klassen lernt das Modell "
                       "kaum, ihre Kennzahlen sind Zufall. Tipp: diese Klassen in Schritt 4 abwählen oder mehr Texte "
                       "sammeln.", c=", ".join(rare[:10]) + (" …" if len(rare) > 10 else ""), m=MIN_CLASS_TEXTS)
                    + (tx(" Hat eine Klasse nur einen Trainingstext, ist LinearSVC nicht möglich.")
                       if rare_classes(y_tr, classes, 2) else ""))
    vectorizer, X_tr = fit_vectorizer(vec, t_tr)
    X_te = vectorizer.transform(t_te).tocsr()
    vocab = len(vectorizer.vocabulary_)
    log.append(tx("Split zuerst: Vokabular nur aus den {n} Trainingstexten gelernt ({v} Wörter{grams}) – die {n2} "
                  "Testtexte werden damit nur umgewandelt (kein Datenleck)",
                  n=_n(len(pos_tr)), v=_n(vocab), grams=tx(" und Wortpaare") if int(vec.ngram_max) > 1 else "",
                  n2=_n(len(pos_te))))
    empty_tr = int((X_tr.getnnz(axis=1) == 0).sum())
    empty_te = int((X_te.getnnz(axis=1) == 0).sum())
    if empty_tr or empty_te:
        log.append(tx("Texte ohne ein einziges Wort aus dem Vokabular: {a} im Training, {b} im Test (nur Stoppwörter, "
                      "Zahlen, einzelne Zeichen oder unbekannte Wörter)", a=_n(empty_tr), b=_n(empty_te)))
    if vocab > BIG_VOCAB:
        warn.append(tx("Sehr großes Vokabular ({v} Wörter) – Training und Speichern dauern länger. Tipp: min_df "
                       "erhöhen (z. B. 2) oder max_features begrenzen.", v=_n(vocab)))
    # wortgleiche Texte in Training und Test machen die Testgenauigkeit zu optimistisch (z. B. SMS-Vorlagen)
    fp_tr = text_fingerprints(t_tr)[_non_blank(t_tr)]
    fp_te = text_fingerprints(t_te)[_non_blank(t_te)]
    dup = np.isin(fp_te, fp_tr)
    dup_share = float(dup.mean()) if len(dup) else 0.0
    if dup_share >= 0.02:
        warn.append(tx("{pct} der Testtexte ({n}) kommen wortgleich auch im Training vor – für sie kennt das Modell "
                       "die Antwort schon; die Testgenauigkeit ist dadurch etwas zu optimistisch. Tipp: doppelte Texte "
                       "in den Daten prüfen.", pct=_pct(dup_share), n=_n(int(dup.sum()))))
    same_source = set(raw_sources(lineage, text_col)) & set(raw_sources(lineage, label))
    if same_source:
        warn.append(tx("Textspalte und Label stammen aus derselben CSV-Spalte ({c}) → Datenleck!",
                       c=", ".join(sorted(same_source))))
    rng = np.random.default_rng(0)
    unique_fp = np.unique(fp_tr)
    stored_fp = (np.sort(rng.choice(unique_fp, FINGERPRINTS, replace=False)) if len(unique_fp) > FINGERPRINTS
                 else unique_fp)
    sample = t_te if len(t_te) <= COVERAGE_SAMPLE else t_te[rng.choice(len(t_te), COVERAGE_SAMPLE, replace=False)]
    tot, kn = vocabulary_coverage(vectorizer, sample)
    words_tr = pd.Series(t_tr, dtype=object).astype(str).str.split().str.len()
    share = np.bincount(y_tr, minlength=len(classes)) / len(y_tr)
    recipe = {
        "app": APP_ID,
        "template_version": TEMPLATE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "column_actions": column_actions or [],
        "new_columns": new_columns or [],
        "lineage": lineage,
        "text_col": text_col,
        "text_source": raw_sources(lineage, text_col),
        "label": label,
        "label_source": raw_sources(lineage, label),
        "label_kind": kind,
        "task": task,
        "classes": classes,
        "class_values": class_values,
        "classes_selected": selected,
        "missing_text": missing_text if missing_text in MISSING_TEXT else TEXT_DROP,
        "split": {"test_size": float(test_size), "random_state": int(random_state), "stratify": bool(stratify),
                  "stratify_used": strat_y is not None},
        "vectorizer": asdict(vec),
        "vocab_size": int(vocab),
        "threshold": 0.5,
        "label_stats": {"class_share": {cl: float(v) for cl, v in zip(classes, share)}},
        "text_stats": {"mean_chars": float(pd.Series(t_tr, dtype=object).astype(str).str.len().mean()),
                       "mean_words": float(words_tr.mean()), "blank": n_blank, "empty_train": empty_tr,
                       "empty_test": empty_te, "test_in_train": dup_share, "unique_train": int(len(unique_fp)),
                       "test_coverage": float(kn.sum() / tot.sum()) if tot.sum() else None},
        "n_rows": {"train": int(len(y_tr)), "test": int(len(y_te))},
        "train_fingerprints": [int(h) for h in stored_fp],
    }
    index = d.index.to_numpy()
    return TextPrep(recipe=recipe, vectorizer=vectorizer, X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te,
                    idx_train=index[pos_tr], idx_test=index[pos_te], text_train=t_tr, text_test=t_te,
                    log=log, warnings=warn)


def vocabulary_table(vectorizer, X_train=None) -> pd.DataFrame:
    """Vokabular: Wort (bzw. Wortpaar), in wie vielen Trainingstexten es vorkommt, wie oft insgesamt (nur
    CountVectorizer ohne binary) und – bei TF-IDF – der IDF-Wert. Sortiert nach Anzahl Texte."""
    out = pd.DataFrame({"Wort": vectorizer.get_feature_names_out()})
    if X_train is not None:
        out["Texte"] = np.asarray((X_train > 0).sum(axis=0)).ravel().astype(int)
        if not isinstance(vectorizer, TfidfVectorizer) and not vectorizer.binary:
            out["Anzahl"] = np.asarray(X_train.sum(axis=0)).ravel().astype(int)
    if hasattr(vectorizer, "idf_"):
        out["IDF"] = np.round(vectorizer.idf_, 4)
    if "Texte" in out:
        out = out.sort_values(["Texte", "Wort"], ascending=[False, True], kind="stable")
    return out.reset_index(drop=True)


def top_terms(texts, labels=None, top: int = 20, stop_words: str = "english", ngram_max: int = 1,
              max_rows: int = 20_000, seed: int = RANDOM_STATE) -> dict:
    """Häufigste Wörter (Schritt 3) – insgesamt oder je Gruppe. Rückgabe: {Gruppe: DataFrame[Wort, Anzahl, Anteil]}
    (Anteil = in wie vielen Texten der Gruppe das Wort vorkommt). Bei vielen Texten auf einer Stichprobe."""
    t = pd.Series(list(texts), dtype=object).fillna("").astype(str)
    g = None if labels is None else pd.Series(list(labels), dtype=object)
    if len(t) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(len(t), max_rows, replace=False))
        t = t.iloc[rows].reset_index(drop=True)
        g = None if g is None else g.iloc[rows].reset_index(drop=True)
    vec = CountVectorizer(stop_words=stop_words or None, ngram_range=(1, int(ngram_max)))
    try:
        X = vec.fit_transform(t)
    except ValueError:
        return {}
    words = vec.get_feature_names_out()
    groups = {"": np.ones(len(t), dtype=bool)} if g is None else {
        str(v): (g == v).to_numpy() for v in sorted(g.dropna().unique(), key=_natural_key)}
    out = {}
    for name, m in groups.items():
        if not m.any():
            continue
        Xg = X[m]
        counts = np.asarray(Xg.sum(axis=0)).ravel()
        docs = np.asarray((Xg > 0).sum(axis=0)).ravel() / max(int(m.sum()), 1)
        order = np.lexsort((words, -counts))[:top]
        order = order[counts[order] > 0]
        out[name] = pd.DataFrame({"Wort": words[order], "Anzahl": counts[order].astype(int),
                                  "Anteil": np.round(docs[order], 4)})
    return out


def text_length_frame(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Länge der Texte (Zeichen, Wörter) – wie im Kurs: yelp['text length'] = yelp['text'].apply(len)."""
    t = text_values(df[text_col])
    return pd.DataFrame({"Zeichen": t.str.len().astype(int), "Wörter": t.str.split().str.len().astype(int)},
                        index=df.index)


# ---------------------------------------------------------------------------
# 6) Modell & Training – MultinomialNB, Logistische Regression, LinearSVC
# ---------------------------------------------------------------------------
MODEL_NB, MODEL_LOGREG, MODEL_SVC = "nb", "logreg", "linsvc"
MODELS = [MODEL_NB, MODEL_LOGREG, MODEL_SVC]
MODEL_NAMES = {
    MODEL_NB: N_("Naive Bayes (MultinomialNB)"),
    MODEL_LOGREG: N_("Logistische Regression"),
    MODEL_SVC: N_("Lineare SVM (LinearSVC)"),
}
MODEL_WHY = {
    MODEL_NB: N_("Der Klassiker für Texte und das Modell aus dem Kurs: zählt, wie typisch jedes Wort für eine Klasse "
                 "ist. Sehr schnell, braucht wenig Daten, gute Baseline – nimmt aber an, dass Wörter unabhängig "
                 "voneinander sind."),
    MODEL_LOGREG: N_("Gewichtet jedes Wort und liefert gut kalibrierte Wahrscheinlichkeiten. Meist genauer als Naive "
                     "Bayes, wenn genug Texte da sind; die Gewichte sind leicht zu lesen."),
    MODEL_SVC: N_("Sucht die Trennlinie mit dem größten Abstand zwischen den Klassen – bei Texten mit TF-IDF oft das "
                  "genaueste der drei. Wahrscheinlichkeiten kommen über eine Kalibrierung (CalibratedClassifierCV)."),
}
MODEL_BEST = {
    MODEL_NB: N_("schneller Start, wenige Texte, Baseline"),
    MODEL_LOGREG: N_("Wahrscheinlichkeiten und Schwelle, lesbare Gewichte"),
    MODEL_SVC: N_("viele Texte, TF-IDF, höchste Genauigkeit"),
}
CURVE_VALUES = {MODEL_NB: (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0),
                MODEL_LOGREG: (0.01, 0.1, 1.0, 10.0, 100.0),
                MODEL_SVC: (0.001, 0.01, 0.1, 1.0, 10.0)}


@dataclass
class TextModelConfig:
    model: str = MODEL_NB
    # Naive Bayes
    alpha: float = 1.0  # Glättung (Laplace): nie gesehene Wort-Klasse-Paare bekommen eine kleine Wahrscheinlichkeit
    fit_prior: bool = True  # Anteil der Klassen im Training als Vorwissen
    # Logistische Regression
    logreg_c: float = 1.0  # klein = stärkere Regularisierung
    max_iter: int = 1000
    # LinearSVC
    svc_c: float = 1.0
    # allgemein
    class_weight: bool = False  # 'balanced' (nur Logistische Regression und LinearSVC)
    curve: bool = True  # Validierungskurve für Schritt 7
    validation: str = "test"  # "test" = Testdaten (wie im Kurs) | "split" = Anteil aus den Trainingsdaten
    validation_split: float = 0.2
    seed: int = RANDOM_STATE


def model_name(key: str) -> str:
    """Anzeigename des Modells – unbekannte Werte (z. B. aus einem fremden Rezept) als „?“, nie als Markdown."""
    return tx(MODEL_NAMES[key]) if key in MODEL_NAMES else "?"


def model_config(d: dict | None) -> TextModelConfig:
    """TextModelConfig aus einem Rezept (unbekannte Einträge werden ignoriert)."""
    known = set(TextModelConfig.__dataclass_fields__)
    return TextModelConfig(**{k: v for k, v in (d or {}).items() if k in known})


def param_of(cfg: TextModelConfig) -> tuple[str, float]:
    """Wichtigster Parameter des Modells und sein Wert: alpha (Naive Bayes) bzw. C."""
    if cfg.model == MODEL_NB:
        return "alpha", float(cfg.alpha)
    return "C", float(cfg.logreg_c if cfg.model == MODEL_LOGREG else cfg.svc_c)


def with_param(cfg: TextModelConfig, value: float) -> TextModelConfig:
    if cfg.model == MODEL_NB:
        return replace(cfg, alpha=float(value))
    return replace(cfg, **({"logreg_c": float(value)} if cfg.model == MODEL_LOGREG else {"svc_c": float(value)}))


def calibration_cv(y_train=None, recipe: dict | None = None) -> int:
    """Folds für CalibratedClassifierCV (Wahrscheinlichkeiten der LinearSVC): 5 wie scikit-learn, weniger bei sehr
    seltenen Klassen. Aus den Trainings-Labels oder – für das exportierte Skript – aus dem Rezept."""
    if y_train is not None:
        counts = np.bincount(np.asarray(y_train).astype(int))
    else:
        n = (recipe.get("n_rows") or {}).get("train", 0)
        counts = np.array([round(s * n) for s in (recipe.get("label_stats") or {}).get("class_share", {}).values()])
    counts = counts[counts > 0]
    return int(max(2, min(5, counts.min()))) if len(counts) else 5


def svc_problem(y_train, classes) -> str | None:
    """LinearSVC liefert Wahrscheinlichkeiten über eine Kalibrierung mit Kreuzvalidierung – dafür braucht jede Klasse
    mindestens 2 Trainingstexte (sonst bricht scikit-learn mit einer englischen Meldung ab). Rückgabe: verständliche
    Meldung oder None."""
    rare = rare_classes(y_train, classes, 2)
    if not rare:
        return None
    return tx("LinearSVC braucht für die Wahrscheinlichkeiten (Kalibrierung mit Kreuzvalidierung) mindestens 2 "
              "Trainingstexte je Klasse – {c}: nur einer. Die Klasse in Schritt 4 abwählen oder Naive Bayes bzw. "
              "Logistische Regression wählen.", c=", ".join(rare[:10]) + (" …" if len(rare) > 10 else ""))


def make_model(cfg: TextModelConfig, probability: bool = True, calib_cv: int = 5):
    """Noch nicht trainiertes Modell. LinearSVC: CalibratedClassifierCV(LinearSVC(…), ensemble=False) liefert
    Wahrscheinlichkeiten für Schwelle und ROC-AUC; probability=False = reine LinearSVC (z. B. für die Kurve)."""
    cw = "balanced" if cfg.class_weight and cfg.model in (MODEL_LOGREG, MODEL_SVC) else None
    if cfg.model == MODEL_NB:
        return MultinomialNB(alpha=float(cfg.alpha), fit_prior=bool(cfg.fit_prior))
    if cfg.model == MODEL_LOGREG:
        return LogisticRegression(C=float(cfg.logreg_c), max_iter=int(cfg.max_iter), class_weight=cw,
                                  random_state=int(cfg.seed))
    if cfg.model == MODEL_SVC:
        svc = LinearSVC(C=float(cfg.svc_c), class_weight=cw, random_state=int(cfg.seed))
        return CalibratedClassifierCV(svc, ensemble=False, cv=int(calib_cv)) if probability else svc
    raise ValueError(tx("Unbekanntes Modell: {m}", m=cfg.model))


def base_estimator(model):
    """Das eigentliche Modell – bei der LinearSVC steckt es in CalibratedClassifierCV."""
    if isinstance(model, CalibratedClassifierCV):
        cc = getattr(model, "calibrated_classifiers_", None)
        return cc[0].estimator if cc else model.estimator
    return model


MODEL_PARAMS = {MODEL_NB: ("alpha", "fit_prior"), MODEL_LOGREG: ("logreg_c", "max_iter", "class_weight"),
                MODEL_SVC: ("svc_c", "class_weight")}


def model_params(training: dict) -> dict:
    """Nur die Einstellungen, die für das gewählte Modell eine Rolle spielen (z. B. für Schritt 9)."""
    t = dict(training or {})
    m = t.get("model")
    out = {"model": m}
    out.update({k: t[k] for k in MODEL_PARAMS.get(m, ()) if k in t})
    return out


def class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y.astype(int))
    weights = compute_class_weight("balanced", classes=classes, y=y.astype(int))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _error(y_true, y_pred) -> float:
    """Fehlerrate = 1 − Accuracy (kleiner = besser)."""
    return float(np.mean(np.asarray(y_true).astype(int) != np.asarray(y_pred).astype(int)))


def _baseline_error(y_fit, y_val) -> float:
    """Fehlerrate einer naiven Vorhersage: immer die häufigste Klasse der Trainingsdaten."""
    majority = int(np.bincount(np.asarray(y_fit).astype(int)).argmax())
    return float(np.mean(np.asarray(y_val).astype(int) != majority))


def _holdout(y_codes, share: float, seed: int):
    """Validierungsanteil aus den Trainingsdaten (Zeilennummern) – möglichst stratifiziert.
    Rückgabe: (Zeilen fürs Training, Zeilen für die Validierung, stratifiziert?)."""
    n = len(y_codes)
    counts = np.bincount(np.asarray(y_codes).astype(int))
    counts = counts[counts > 0]
    n_val = int(np.ceil(share * n))
    strat = y_codes if counts.min() >= 2 and n_val >= len(counts) and n - n_val >= len(counts) else None
    tr, va = train_test_split(np.arange(n), test_size=share, random_state=seed, stratify=strat)
    return tr, va, strat is not None


def curve_data(cfg: TextModelConfig, prep: TextPrep):
    """Daten der Validierungskurve: Testdaten (wie im Kurs) oder ein Anteil der Trainingsdaten – dann lernt ein
    eigener Vektorisierer sein Vokabular nur aus dem verbleibenden Trainingsteil (sonst kennte er die
    Validierungstexte). Rückgabe: (X_fit, y_fit, X_val, y_val, stratifiziert?)."""
    if cfg.validation != "split":
        return prep.X_train, prep.y_train, prep.X_test, prep.y_test, False
    tr, va, strat = _holdout(prep.y_train, float(cfg.validation_split), int(cfg.seed))
    vec, X_fit = fit_vectorizer(vectorizer_config(prep.recipe["vectorizer"]), prep.text_train[tr])
    return X_fit, prep.y_train[tr], vec.transform(prep.text_train[va]).tocsr(), prep.y_train[va], strat


def validation_curve(cfg: TextModelConfig, prep: TextPrep, progress=None) -> dict:
    """Schritt 7: Fehlerrate auf Trainings- und Validierungsdaten für verschiedene Werte des wichtigsten Parameters –
    alpha (Naive Bayes) bzw. C (Logistische Regression, LinearSVC; ohne Kalibrierung, schneller)."""
    report = progress or (lambda *_: None)
    X_fit, y_fit, X_val, y_val, stratified = curve_data(cfg, prep)
    if len(np.unique(y_fit)) < 2:  # nur bei „Anteil aus Training“ mit sehr wenigen Texten möglich
        raise ValueError(tx("Validierungskurve nicht möglich: Im Trainingsteil bleibt nur eine Klasse übrig (zu wenige "
                            "Texte für „Anteil aus Training“). In Schritt 6 die Testdaten zur Validierung wählen oder die "
                            "Kurve ausschalten."))
    param, chosen = param_of(cfg)
    values = sorted(set(CURVE_VALUES[cfg.model]) | {chosen})
    curve = {"validation": cfg.validation, "n_fit": int(X_fit.shape[0]), "n_val": int(X_val.shape[0]),
             "metric": "error", "param": param, "values": values, "chosen": chosen, "train": [], "val": [],
             "baseline": _baseline_error(y_fit, y_val), "stratify": stratified}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for i, v in enumerate(values):
            m = make_model(with_param(cfg, v), probability=False).fit(X_fit, y_fit)
            curve["train"].append(_error(y_fit, m.predict(X_fit)))
            curve["val"].append(_error(y_val, m.predict(X_val)))
            report((i + 1) / len(values), tx("Validierungskurve: {param} = {v} ({i}/{n})", param=param, v=f"{v:g}",
                                             i=i + 1, n=len(values)), curve)
    best = min(curve["val"])  # bei Gleichstand der eingestellte Wert, sonst der erste
    i_chosen = values.index(chosen)
    curve["best_index"] = i_chosen if curve["val"][i_chosen] <= best else int(np.argmin(curve["val"]))
    return curve


def fit_model(cfg: TextModelConfig, X, y):
    """Modell erzeugen und trainieren: model.fit(X_train, y_train). Rückgabe: (model, converged)."""
    calib = calibration_cv(y) if cfg.model == MODEL_SVC else 5
    model = make_model(cfg, calib_cv=calib)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    return model, not any(issubclass(w.category, ConvergenceWarning) for w in caught)


def train_model(cfg: TextModelConfig, prep: TextPrep, progress=None):
    """6) Modell trainieren: model.fit(X_train, y_train) – vorher optional die Validierungskurve für Schritt 7.

    progress(anteil 0–1, text, kurve) meldet den Fortschritt (Balken und Live-Kurve in der App).
    Rückgabe: (model, hist, info) – hist = {"curve": …} für Schritt 7."""
    report = progress or (lambda *_: None)
    if cfg.model not in MODELS:
        raise ValueError(tx("Unbekanntes Modell: {m}", m=cfg.model))
    problem = svc_problem(prep.y_train, prep.recipe["classes"]) if cfg.model == MODEL_SVC else None
    if problem:
        raise ValueError(problem)
    y = prep.y_train
    t0 = time.time()
    hist = {}
    if cfg.curve:
        hist["curve"] = validation_curve(cfg, prep, lambda f, t, c: report(0.85 * f, t, c))
    report(0.9, tx("Trainiere das Modell auf allen Trainingsdaten ({n} Texte) …", n=_n(len(y))), hist.get("curve"))
    model, converged = fit_model(cfg, prep.X_train, y)
    info = {
        "model": cfg.model,
        "seconds": round(time.time() - t0, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(y)),
        "metric": "error",
        "train_error": _error(y, model.predict(prep.X_train)),
        "test_error": _error(prep.y_test, model.predict(prep.X_test)),
        "baseline_error": _baseline_error(y, prep.y_test),
        "converged": converged,
        "calib_cv": calibration_cv(y) if cfg.model == MODEL_SVC else None,
        "class_weight": class_weights(y) if cfg.class_weight and cfg.model != MODEL_NB else None,
    }
    report(1.0, tx("Fertig."), hist.get("curve"))
    return model, hist, info


def compare_models(cfg: TextModelConfig, prep: TextPrep, progress=None) -> pd.DataFrame:
    """Alle drei Modelle mit den aktuellen Einstellungen (alpha, C, class_weight) auf denselben Daten trainieren und
    auf den Testdaten vergleichen – Schwelle 0.5. Ist LinearSVC nicht möglich (eine Klasse mit nur einem
    Trainingstext), fehlt sie in der Tabelle; der Grund steht in table.attrs["skipped"]."""
    report = progress or (lambda *_: None)
    task, classes = prep.recipe["task"], prep.recipe["classes"]
    rows, skipped = [], {}
    for i, m in enumerate(MODELS):
        problem = svc_problem(prep.y_train, classes) if m == MODEL_SVC else None
        if problem:
            skipped[m] = problem
            continue
        report(i / len(MODELS), tx("Vergleich: {m} ({i}/{n})", m=model_name(m), i=i + 1, n=len(MODELS)))
        t0 = time.time()
        model, _ = fit_model(replace(cfg, model=m), prep.X_train, prep.y_train)
        ev = evaluate(prep.y_test, predict_scores(model, prep.X_test, task), task, classes)
        rows.append({"model": m, "Accuracy": round(ev["accuracy"], 4), "F1 (macro)": round(ev["f1_macro"], 4),
                     "ROC-AUC": None if ev["auc"] is None else round(ev["auc"], 4),
                     "Sekunden": round(time.time() - t0, 2)})
    report(1.0, tx("Fertig."))
    table = pd.DataFrame(rows)
    table.attrs["skipped"] = skipped
    return table


# --- Schritt 7: Einordnung Training vs. Validierung ------------------------------------------------------------
REMEDY_OVER = {
    MODEL_NB: N_("alpha vergrößern (stärkere Glättung) oder in Schritt 4 min_df erhöhen"),
    MODEL_LOGREG: N_("C verkleinern (stärkere Regularisierung) oder in Schritt 4 min_df erhöhen"),
    MODEL_SVC: N_("C verkleinern (stärkere Regularisierung) oder in Schritt 4 min_df erhöhen"),
}
REMEDY_UNDER = {
    MODEL_NB: N_("alpha verkleinern, in Schritt 4 Wortpaare (ngram 2) oder TF-IDF probieren"),
    MODEL_LOGREG: N_("C vergrößern, in Schritt 4 Wortpaare (ngram 2) probieren"),
    MODEL_SVC: N_("C vergrößern, in Schritt 4 Wortpaare (ngram 2) probieren"),
}


def _noise(e: float, n: int) -> float:
    """Kleinster Unterschied, der mehr als Zufall ist: Standardfehler der Fehlerrate."""
    return max(0.003, float(np.sqrt(max(e * (1 - e), 1e-6) / max(n, 1))))


def curve_assessment(curve: dict | None, info: dict, cfg: TextModelConfig) -> dict:
    """Schritt 7: Training vs. Validierung einordnen – Overfitting, Underfitting, Vergleich mit einer naiven
    Vorhersage und bester Parameterwert der Kurve. Rückgabe: {"train", "val", "baseline", "verdict", "notes",
    "suggest"} – suggest = {"param", "value"} oder None."""
    if curve:
        i = curve["values"].index(curve["chosen"])
        e_tr, e_va, base, n_val = curve["train"][i], curve["val"][i], curve["baseline"], curve["n_val"]
    else:
        e_tr, e_va, base, n_val = info["train_error"], info["test_error"], info["baseline_error"], 0
    f = _pct
    notes, suggest = [], None
    gap = e_va - e_tr
    if base > 0 and e_va >= 0.98 * base:
        verdict = "nosignal"
        if curve and min(curve["val"]) < 0.9 * base:  # die Daten tragen ein Signal – nur der Wert passt nicht
            notes.append(tx("**Mit {param} = {chosen} nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber "
                            "{b}, wenn man immer die häufigste Klasse vorhersagt. Andere Werte der Kurve sind deutlich "
                            "besser – der eingestellte Wert passt nicht zu diesen Daten (z. B. alpha zu groß bei TF-IDF).",
                            param=curve["param"], chosen=f"{curve['chosen']:g}", v=f(e_va), b=f(base)))
        else:
            notes.append(tx("**Nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber {b}, wenn man immer "
                            "die häufigste Klasse vorhersagt. In diesen Texten steckt für das Modell kaum ein Signal – "
                            "Textspalte, Label und Vektorisierung prüfen (Schritt 4).", v=f(e_va), b=f(base)))
    elif gap > 0.05 and e_tr <= 0.02:
        verdict = "memorize"
        notes.append(tx("**Trainingsfehler {t} ist bei Texten normal:** Es gibt viel mehr Wörter als Texte, das Modell "
                        "kann die Trainingstexte fast immer trennen. Aussagekräftig ist der Validierungsfehler {v} – zum "
                        "Vergleich: naive Vorhersage {b}. Liegt er deutlich über dem Minimum der Kurve, hilft: {fix}.",
                        t=f(e_tr), v=f(e_va), b=f(base), fix=tx(REMEDY_OVER[cfg.model])))
    elif gap > 0.05:
        verdict = "overfit"
        notes.append(tx("**Overfitting:** Training {t}, Validierung {v} – das Modell lernt die Trainingstexte deutlich "
                        "besser, als es neue Texte einordnet. Gegenmittel: {fix}.",
                        t=f(e_tr), v=f(e_va), fix=tx(REMEDY_OVER[cfg.model])))
    elif e_va > 0.85 * base and gap < 0.03:
        verdict = "underfit"
        notes.append(tx("**Underfitting:** Training {t} und Validierung {v} sind ähnlich hoch und kaum besser als die "
                        "naive Vorhersage ({b}) – das Modell ist zu einfach. Gegenmittel: {fix}.",
                        t=f(e_tr), v=f(e_va), b=f(base), fix=tx(REMEDY_UNDER[cfg.model])))
    else:
        verdict = "ok"
        notes.append(tx("Training {t} und Validierung {v} liegen nah beieinander und deutlich unter der naiven "
                        "Vorhersage ({b}) – keine Anzeichen für Overfitting.", t=f(e_tr), v=f(e_va), b=f(base)))
    if curve:
        j = curve.get("best_index", int(np.argmin(curve["val"])))
        best, e_best = curve["values"][j], curve["val"][j]
        if best != curve["chosen"] and e_best < e_va - _noise(e_va, n_val):
            notes.append(tx("**Bester Wert laut Validierungskurve:** {param} = {best} (Fehler {e}). Eingestellt ist "
                            "{chosen} (Fehler {e2}).", param=curve["param"], best=f"{best:g}", e=f(e_best),
                            chosen=f"{curve['chosen']:g}", e2=f(e_va)))
            suggest = {"param": curve["param"], "value": best}
        else:
            notes.append(tx("Der eingestellte Wert {param} = {chosen} liegt am Minimum der Validierungskurve (oder "
                            "nur knapp daneben).", param=curve["param"], chosen=f"{curve['chosen']:g}"))
        if curve["validation"] == "test":
            notes.append(tx("Die Kurve nutzt die Testdaten (wie im Kurs). Wer danach einen Wert auswählt, hat die "
                            "Testdaten schon zur Wahl benutzt – die Testgenauigkeit ist dann leicht zu optimistisch. "
                            "Sauberer: in Schritt 6 „Anteil aus Training“ wählen."))
    return {"train": e_tr, "val": e_va, "baseline": base, "verdict": verdict, "notes": notes, "suggest": suggest}


# ---------------------------------------------------------------------------
# 8) Evaluation und Modell verstehen
# ---------------------------------------------------------------------------
def predict_scores(model, X, task: str) -> np.ndarray:
    """Binär: Wahrscheinlichkeit für Klasse 1 (positive Klasse) · Mehrklassen: (n, K)."""
    proba = np.asarray(model.predict_proba(X), dtype="float64")
    return proba[:, 1] if task == T_BIN else proba


def to_codes(scores: np.ndarray, task: str, threshold: float = 0.5) -> np.ndarray:
    """Wahrscheinlichkeiten → Klassen: binär über die Schwelle, Mehrklassen über die größte Wahrscheinlichkeit."""
    if task == T_BIN:
        return (scores > threshold).astype(int)
    return scores.argmax(axis=1)


def training_majority(recipe: dict) -> int | None:
    """Häufigste Klasse der Trainingsdaten (Kodierung) laut Rezept – die naive Vorhersage."""
    share = (recipe.get("label_stats") or {}).get("class_share") or {}
    values = [share.get(c) for c in recipe.get("classes", [])]
    if not values or any(v is None for v in values):
        return None
    return int(np.argmax(values))


def evaluate(y_true: np.ndarray, scores: np.ndarray, task: str, classes: list, threshold: float = 0.5,
             majority: int | None = None) -> dict:
    """Kennzahlen. majority = häufigste Klasse der TRAININGSDATEN für die naive Vorhersage (wie in Schritt 7);
    ohne Angabe die häufigste Klasse in y_true."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = to_codes(scores, task, threshold)
    labels = list(range(len(classes)))
    naive = int(np.bincount(y_true, minlength=len(classes)).argmax()) if majority is None else int(majority)
    res = {
        "task": task,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=classes, output_dict=True,
                                        zero_division=0),
        "report_text": classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
        "y_pred": y_pred,
        "auc": None,
        "baseline": float(np.mean(y_true == naive)),  # Accuracy der naiven Vorhersage
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            auc = (roc_auc_score(y_true, scores) if task == T_BIN
                   else roc_auc_score(y_true, scores, multi_class="ovr", labels=labels))
        res["auc"] = float(auc) if np.isfinite(auc) else None
    except ValueError:
        pass  # z. B. nur eine Klasse vorhanden
    return res


def metrics_summary(ev: dict) -> dict:
    """Kompakte Kennzahlen für preprocessing.json / Anzeige."""
    out = {"accuracy": round(ev["accuracy"], 4), "f1_macro": round(ev["f1_macro"], 4)}
    if ev.get("auc") is not None:
        out["roc_auc"] = round(ev["auc"], 4)
    return out


def word_weights(model, n_words: int) -> tuple[str, np.ndarray]:
    """Gewicht jedes Worts je Klasse (K × Wörter; binär: Zeile 1 = für die positive Klasse, Zeile 0 = das Negative).
    Naive Bayes: log P(Wort | Klasse) minus Mittel der anderen Klassen; Logistische Regression und LinearSVC:
    Koeffizienten. Rückgabe: (Art "nb" | "coef", Matrix)."""
    est = base_estimator(model)
    if isinstance(est, MultinomialNB):
        flp = np.asarray(est.feature_log_prob_, dtype="float64")
        k = flp.shape[0]
        if k == 2:
            diff = flp[1] - flp[0]
            return "nb", np.vstack([-diff, diff])
        others = (flp.sum(axis=0, keepdims=True) - flp) / (k - 1)
        return "nb", flp - others
    coef = np.asarray(est.coef_, dtype="float64").reshape(-1, n_words)
    if coef.shape[0] == 1:
        return "coef", np.vstack([-coef[0], coef[0]])
    return "coef", coef


def top_words(model, vectorizer, recipe: dict, top: int = 15) -> dict:
    """Wörter, die am stärksten für jede Klasse sprechen. Rückgabe: {"kind", "classes": {Klasse: DataFrame[Wort,
    Gewicht]}} – bei Naive Bayes ist das Gewicht ein Log-Verhältnis, bei den linearen Modellen der Koeffizient."""
    words = vectorizer.get_feature_names_out()
    kind, W = word_weights(model, len(words))
    out = {}
    for j, cl in enumerate(recipe["classes"]):
        order = np.lexsort((words, -W[j]))[:top]
        out[cl] = pd.DataFrame({"Wort": words[order], "Gewicht": np.round(W[j][order], 4)})
    return {"kind": kind, "classes": out}


def explain_text(model, vectorizer, recipe: dict, text: str, top: int = 12, threshold: float | None = None) -> dict:
    """Welche Wörter haben einen Text in eine Klasse geschoben? Beitrag = Wert des Worts im Text (Anzahl bzw.
    TF-IDF) × Gewicht des Worts für die vorhergesagte Klasse (positiv = spricht dafür). Rückgabe: {"proba" (je
    Klasse), "pred", "label", "words": DataFrame[Wort, Wert, Beitrag], "weights": {Wort: Beitrag}, "known",
    "tokens"}."""
    text = str(text or "")
    X = vectorizer.transform([text]).tocsr()
    task = recipe["task"]
    scores = predict_scores(model, X, task)
    proba = np.array([1 - scores[0], scores[0]]) if task == T_BIN else scores[0]
    thr = recipe.get("threshold", 0.5) if threshold is None else threshold
    pred = int(scores[0] > thr) if task == T_BIN else int(np.argmax(proba))
    words = vectorizer.get_feature_names_out()
    _, W = word_weights(model, len(words))
    idx, vals = X.indices, X.data
    contrib = vals * W[pred, idx]
    table = pd.DataFrame({"Wort": words[idx], "Wert": np.round(vals, 4), "Beitrag": np.round(contrib, 4)})
    table = table.iloc[np.lexsort((table["Wort"].to_numpy(), -np.abs(contrib)))].reset_index(drop=True)
    tokens = vectorizer.build_analyzer()(text)
    known = sum(1 for t in tokens if t in vectorizer.vocabulary_)
    return {"proba": proba, "pred": pred, "label": recipe["classes"][pred], "words": table.head(top),
            "weights": {str(w): float(c) for w, c in zip(words[idx], contrib)}, "known": int(known),
            "tokens": int(len(tokens))}


def highlight_spans(text: str, vectorizer, weights: dict, limit: int = 20_000,
                    only=None) -> list[tuple[str, float | None]]:
    """Text in Stücke zerlegen: (Stück, Beitrag) für Wörter aus dem Vokabular, (Stück, None) für alles andere – zum
    Einfärben in der App. Nur einzelne Wörter (Wortpaare stehen in der Tabelle). only = nur diese Wörter (wie im
    Vokabular geschrieben) einfärben, z. B. die wichtigsten aus explain_text."""
    text = str(text or "")[:limit]
    token_pattern = getattr(vectorizer, "token_pattern", None)
    if not token_pattern:
        return [(text, None)]
    pattern = re.compile(token_pattern)
    out, pos = [], 0
    for m in pattern.finditer(text):
        tok = m.group(0)
        key = tok.lower() if vectorizer.lowercase else tok
        if vectorizer.strip_accents == "unicode":
            key = strip_accents_unicode(key)
        w = weights.get(key)
        if w is None or (only is not None and key not in only):
            continue
        if m.start() > pos:
            out.append((text[pos:m.start()], None))
        out.append((tok, float(w)))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], None))
    return out


def misclassified(prep: TextPrep, scores: np.ndarray, recipe: dict, threshold: float = 0.5,
                  n: int = 30) -> pd.DataFrame:
    """Falsch eingeordnete Testtexte – die sichersten Fehler zuerst (oft Ironie, gemischte Bewertungen oder ein
    falsches Label in den Daten)."""
    task, classes = recipe["task"], np.asarray(recipe["classes"])
    pred = to_codes(scores, task, threshold)
    conf = np.where(pred == 1, scores, 1 - scores) if task == T_BIN else scores.max(axis=1)
    wrong = np.where(pred != prep.y_test)[0]
    order = wrong[np.argsort(-conf[wrong], kind="stable")][:n]
    return pd.DataFrame({"Text": [str(t) for t in prep.text_test[order]], "tatsächlich": classes[prep.y_test[order]],
                         "vorhergesagt": classes[pred[order]], "Sicherheit": np.round(conf[order], 4),
                         "Zeile": prep.idx_test[order]})


# ---------------------------------------------------------------------------
# 9) Speichern & Laden
# ---------------------------------------------------------------------------
MODEL_FILES = ("model.skops", "model.joblib")
BUNDLE_FILES = (*MODEL_FILES, "preprocessing.json")
MAX_BUNDLE_BYTES = 2 * 10**9  # entpackte Größe eines Modell-ZIPs
MAX_RECIPE_BYTES = 50 * 10**6  # preprocessing.json – ein Rezept dieser App ist viel kleiner (schützt den Speicher)
# skops lädt nur Typen, denen man ausdrücklich vertraut: die Kalibrierung der LinearSVC kennt skops nicht von sich aus –
# alles andere (z. B. fremder Code) wird abgelehnt.
SKOPS_TRUSTED = {"sklearn.calibration._CalibratedClassifier", "sklearn.calibration._SigmoidCalibration"}
SKOPS_TRUSTED_PREFIX: tuple = ()
MODEL_TYPES = {MODEL_NB: MultinomialNB, MODEL_LOGREG: LogisticRegression, MODEL_SVC: CalibratedClassifierCV}


def make_pipeline(vectorizer, model) -> Pipeline:
    """Vektorisierer + Modell als eine scikit-learn-Pipeline (wie im Kurs) – beide schon gefittet."""
    return Pipeline([("vect", vectorizer), ("clf", model)])


def finalize_recipe(recipe: dict, cfg: TextModelConfig | None = None, info: dict | None = None,
                    metrics: dict | None = None, threshold: float | None = None) -> dict:
    """Trainingsparameter, Kennzahlen und Schwelle ins Rezept schreiben."""
    r = dict(recipe)
    if cfg is not None:
        r["training"] = asdict(cfg)
    if info is not None:
        r["train_info"] = info
    if metrics is not None:
        r["metrics"] = metrics
    if threshold is not None:
        r["threshold"] = float(threshold)
    return r


def save_bundle(pipe, recipe: dict, folder, fmt: str = ".skops") -> Path:
    """Pipeline (Vektorisierer + Modell) + Rezept speichern: <folder>/model.skops (oder .joblib) + preprocessing.json

    .skops: sicheres Format – beim Laden wird kein Code ausgeführt. .joblib: Standard von scikit-learn (pickle)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / f"model{fmt if fmt in ('.skops', '.joblib') else '.skops'}"
    r = dict(recipe)
    r["engine"] = "scikit-learn"
    r["model_file"] = model_path.name
    r["saved_at"] = datetime.now().isoformat(timespec="seconds")
    # Erst beide Dateien vollständig als .tmp schreiben, dann ersetzen: Scheitert das Speichern (Speicherplatz,
    # nicht speicherbares Objekt), bleibt ein vorhandenes Modell unter diesem Namen unverändert ladbar.
    tmp_model, tmp_json = folder / f".{model_path.name}.tmp", folder / ".preprocessing.json.tmp"
    try:
        if model_path.suffix == ".joblib":
            joblib.dump(pipe, tmp_model, compress=3)
        else:
            import skops.io as sio

            sio.dump(pipe, tmp_model)
        tmp_json.write_text(json.dumps(_jsonable(r), indent=2, ensure_ascii=False), encoding="utf-8")
    except BaseException:
        tmp_model.unlink(missing_ok=True)
        tmp_json.unlink(missing_ok=True)
        raise
    os.replace(tmp_model, model_path)
    for name in MODEL_FILES:  # Modell im anderen Format (frühere Speicherung) passt nicht mehr zum Rezept
        if name != model_path.name:
            (folder / name).unlink(missing_ok=True)
    os.replace(tmp_json, folder / "preprocessing.json")
    return folder


def _load_model_file(path: Path, allow_pickle: bool):
    if path.suffix == ".joblib":
        if not allow_pickle:
            raise ValueError(tx("Das Modell ist eine .joblib-Datei (pickle) – sie kann beim Laden Programmcode "
                                "ausführen. Nur laden, wenn die Datei aus einer vertrauenswürdigen Quelle stammt "
                                "(Häkchen setzen), oder das Modell als .skops speichern."))
        return joblib.load(path)
    import skops.io as sio

    try:  # .skops ist selbst ein ZIP – entpackte Größe prüfen, bevor skops die Arrays in den Speicher lädt
        with zipfile.ZipFile(path) as zf:
            size = sum(i.file_size for i in zf.infolist())
    except zipfile.BadZipFile:
        size = 0  # kein ZIP – skops meldet den Fehler selbst
    if size > MAX_BUNDLE_BYTES:
        raise ValueError(tx("ZIP zu groß: entpackt {v:.1f} GB (höchstens {v2:.0f} GB).",
                            v=size / 1e9, v2=MAX_BUNDLE_BYTES / 1e9))
    untrusted = sio.get_untrusted_types(file=path)
    bad = [t for t in untrusted if t not in SKOPS_TRUSTED and not t.startswith(SKOPS_TRUSTED_PREFIX)]
    if bad:
        raise ValueError(tx("Das Modell enthält Typen, die nicht als sicher gelten – Laden abgelehnt: {types}",
                            types=", ".join(bad)))
    return sio.load(path, trusted=untrusted)


def _bundle_model_name(recipe: dict, present) -> str | None:
    """Welche Modelldatei geladen wird: die im Rezept genannte (nur der Dateiname, nur model.skops/.joblib), sonst
    die erste vorhandene. present = vorhandene Dateinamen (Ordner oder ZIP)."""
    named = Path(str(recipe.get("model_file", MODEL_FILES[0]))).name if isinstance(recipe, dict) else ""
    if named in MODEL_FILES and named in present:
        return named
    return next((n for n in MODEL_FILES if n in present), None)


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _numbers(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _opt_str(v) -> bool:
    return v is None or isinstance(v, str)


def _fitted_problem(fitted) -> bool:
    """Gemerkte Werte einer Aktion (z. B. die Top-N-Liste) unbrauchbar?"""
    return fitted is not None and (not isinstance(fitted, dict)
                                   or ("keep" in fitted and not _strings(fitted["keep"]))
                                   or ("used" in fitted and not _strings(fitted["used"])))


def _action_problem(a: dict) -> bool:
    """Spalten-Aktion aus einem Rezept unbrauchbar? Schritt 10 wendet sie auf neue Daten an – ein Fehler soll beim
    Laden auffallen, nicht erst als KeyError/AttributeError bei der Vorhersage."""
    return (not isinstance(a.get("column"), str) or not a["column"]
            or a.get("action") not in (None, "", *ACTIONS)
            or not all(_opt_str(a.get(k)) for k in ("param", "new_name")) or _fitted_problem(a.get("fitted")))


def _new_column_problem(s: dict) -> bool:
    """Definition einer neuen Spalte aus einem Rezept unbrauchbar? (Formeln prüft translate_expression beim
    Berechnen – dort wird nie beliebiger Code ausgeführt.)"""
    return (not isinstance(s.get("name"), str) or not s["name"].strip() or s.get("kind") not in NEW_KINDS
            or not all(_opt_str(s.get(k)) for k in ("source", "definition", "action", "param"))
            or s.get("action") not in (None, "", *ACTIONS) or _fitted_problem(s.get("fitted")))


def recipe_problem(recipe: dict) -> str | None:
    """Ist ein NLP-Rezept vollständig und in sich stimmig? Ein beschädigtes oder von Hand verändertes
    preprocessing.json soll beim Laden scheitern – nicht erst in Schritt 10. Geprüft wird jeder Eintrag, den
    Schritt 9 und 10 benutzen. Rückgabe: betroffener Eintrag oder None."""
    for key in ("text_col", "label"):
        if not isinstance(recipe.get(key), str) or not recipe[key]:
            return key
    classes = recipe.get("classes")
    if not _strings(classes) or not 2 <= len(classes) <= MAX_CLASSES or len(set(classes)) != len(classes):
        return "classes"
    if recipe.get("task") != (T_BIN if len(classes) == 2 else T_MULTI):
        return "task"
    kind = recipe.get("label_kind")
    if kind not in ("bool", "numeric", "text"):
        return "label_kind"
    values = recipe.get("class_values")
    if not isinstance(values, list) or len(values) != len(classes):
        return "class_values"
    if (kind == "numeric" and not _numbers(values)) or (kind == "bool" and not all(isinstance(v, bool) for v in values)):
        return "class_values"
    thr = recipe.get("threshold", 0.5)  # dieselben Grenzen wie die Schieberegler in Schritt 8 und 10
    if not isinstance(thr, (int, float)) or isinstance(thr, bool) or not THRESHOLD_MIN <= thr <= THRESHOLD_MAX:
        return "threshold"
    selected = recipe.get("classes_selected")  # Klassenauswahl aus Schritt 4 – das Fazit nennt sie
    if selected is not None and (not _strings(selected) or not set(selected) <= set(classes)):
        return "classes_selected"
    if recipe.get("missing_text") not in (None, *MISSING_TEXT):
        return "missing_text"
    for key, bad in (("column_actions", _action_problem), ("new_columns", _new_column_problem)):
        items = recipe.get(key, [])
        if not isinstance(items, list) or not all(isinstance(a, dict) and not bad(a) for a in items):
            return key
    for key in ("lineage", "training", "train_info", "metrics", "label_stats", "text_stats", "vectorizer", "meta",
                "n_rows", "split"):
        if recipe.get(key) is not None and not isinstance(recipe[key], dict):
            return key
    if not all(isinstance(v, str) or _strings(v) for v in (recipe.get("lineage") or {}).values()):
        return "lineage"
    if not _numbers(list((recipe.get("metrics") or {}).values())):  # das Fazit rechnet mit den Testkennzahlen
        return "metrics"
    share = (recipe.get("label_stats") or {}).get("class_share")
    if share is not None and (not isinstance(share, dict) or not _numbers(list(share.values()))):
        return "label_stats"
    stats = recipe.get("text_stats") or {}
    if not all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in stats.values()):
        return "text_stats"
    if not _numbers(list((recipe.get("n_rows") or {}).values())):
        return "n_rows"
    fps = recipe.get("train_fingerprints")
    if fps is not None and not _numbers(fps):
        return "train_fingerprints"
    training = recipe.get("training") or {}
    if training and training.get("model") not in MODELS:
        return "training"
    return None


def _n_features(clf) -> int | None:
    est = base_estimator(clf)
    for attr in ("feature_log_prob_", "coef_"):
        if hasattr(est, attr):
            return int(np.asarray(getattr(est, attr)).shape[-1])
    return None


def check_recipe(recipe, pipe=None) -> None:
    """Nur Modelle der NLP-Version laden – Modelle der Supervised- oder Clustering-Version haben ein anderes Rezept.
    Prüft außerdem, ob das Rezept vollständig ist und (mit pipe) zur Pipeline passt."""
    if not isinstance(recipe, dict) or recipe.get("app") != APP_ID:
        app = (recipe.get("app") or recipe.get("task")) if isinstance(recipe, dict) else None
        raise ValueError(tx("Dieses Modell stammt nicht aus der NLP-Version ({app}) – hier lassen sich nur "
                            "Textklassifikations-Modelle laden, die mit dieser App gespeichert wurden.",
                            app=md_escape(str(app or "?")[:40])))
    problem = recipe_problem(recipe)
    if problem:
        raise ValueError(tx("preprocessing.json ist unvollständig oder beschädigt (Eintrag: {part}) – das Modell bitte "
                            "neu speichern (Schritt 9).", part=problem))
    if pipe is None:
        return
    steps = getattr(pipe, "steps", None)
    vect = steps[0][1] if isinstance(pipe, Pipeline) and len(steps) == 2 else None
    clf = steps[1][1] if vect is not None else None
    if not (isinstance(vect, (CountVectorizer, TfidfVectorizer)) and isinstance(clf, tuple(MODEL_TYPES.values()))
            and (not isinstance(clf, CalibratedClassifierCV) or isinstance(base_estimator(clf), LinearSVC))):
        raise ValueError(tx("Die Modelldatei enthält keine Pipeline aus Vektorisierer und Textmodell ({type}).",
                            type=type(pipe).__name__))
    if not hasattr(vect, "vocabulary_") or not hasattr(clf, "classes_"):
        raise ValueError(tx("Die Pipeline in der Modelldatei ist nicht trainiert."))
    try:
        same = [int(c) for c in clf.classes_] == list(range(len(recipe["classes"])))
    except (TypeError, ValueError):
        same = False
    if not same:
        raise ValueError(tx("Modell und preprocessing.json passen nicht zusammen: das Modell kennt {k} Klassen, das "
                            "Rezept {m}.", k=len(clf.classes_), m=len(recipe["classes"])))
    if _n_features(clf) != len(vect.vocabulary_):
        raise ValueError(tx("Vektorisierer und Modell passen nicht zusammen ({a} Wörter im Vokabular, das Modell "
                            "erwartet {b}).", a=len(vect.vocabulary_), b=_n_features(clf)))
    m = (recipe.get("training") or {}).get("model")
    if m in MODEL_TYPES and not isinstance(clf, MODEL_TYPES[m]):
        raise ValueError(tx("Modell und preprocessing.json passen nicht zusammen: laut Rezept {m}, in der Datei "
                            "{type}.", m=model_name(m), type=type(clf).__name__))
    kind = (recipe.get("vectorizer") or {}).get("kind")  # Schritt 9 zeigt die Einstellungen aus dem Rezept
    if kind in VECTORIZERS and isinstance(vect, TfidfVectorizer) != (kind == VEC_TFIDF):
        raise ValueError(tx("Vektorisierer und preprocessing.json passen nicht zusammen: laut Rezept {kind}, in der "
                            "Datei {type}.", kind=kind, type=type(vect).__name__))


def load_bundle(folder, allow_pickle: bool = True):
    """Gespeicherte Pipeline + Rezept laden. allow_pickle=False lehnt .joblib-Dateien ab (z. B. aus fremden ZIPs)."""
    folder = Path(folder)
    meta_file = folder / "preprocessing.json"
    if meta_file.is_file() and meta_file.stat().st_size > MAX_RECIPE_BYTES:
        raise ValueError(tx("preprocessing.json ist zu groß ({v:.0f} MB) – das ist kein Rezept dieser App.",
                            v=meta_file.stat().st_size / 1e6))
    try:
        recipe = json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(tx("preprocessing.json ist keine gültige JSON-Datei ({exc}).", exc=exc)) from exc
    check_recipe(recipe)
    name = _bundle_model_name(recipe, {n for n in MODEL_FILES if (folder / n).is_file()})
    if name is None:
        raise ValueError(tx("Kein scikit-learn-Modell gefunden (model.skops oder model.joblib)."))
    pipe = _load_model_file(folder / name, allow_pickle)
    check_recipe(recipe, pipe)
    return pipe, recipe


def bundle_zip_bytes(folder) -> bytes:
    folder = Path(folder)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BUNDLE_FILES:
            if (folder / name).exists():
                zf.write(folder / name, arcname=name)
    return buf.getvalue()


def zip_needs_pickle(data: bytes) -> bool:
    """Würde aus dem ZIP ein .joblib-Modell (pickle) geladen? Dann fragt die App vor dem Laden nach – dieselbe Wahl
    wie in load_bundle (Rezept nennt die Datei; sonst die erste vorhandene)."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = {Path(n).name: n for n in zf.namelist()}
            info = zf.getinfo(names["preprocessing.json"]) if "preprocessing.json" in names else None
            recipe = (json.loads(zf.read(info).decode("utf-8"))
                      if info is not None and info.file_size <= MAX_RECIPE_BYTES else {})
    except (zipfile.BadZipFile, ValueError, KeyError, UnicodeDecodeError):
        return False  # unlesbar – load_bundle_zip meldet den Fehler
    return _bundle_model_name(recipe, set(names)) == "model.joblib"


def load_bundle_zip(data: bytes, allow_pickle: bool = False):
    """Modell aus einer ZIP-Datei laden (nur die erwarteten Dateinamen werden entpackt)."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        if "preprocessing.json" not in names:
            raise ValueError(tx("ZIP enthält keine preprocessing.json."))
        sizes = {name: zf.getinfo(names[name]).file_size for name in BUNDLE_FILES if name in names}
        if sum(sizes.values()) > MAX_BUNDLE_BYTES:  # Schutz vor "ZIP-Bomben" (winzig gepackt, riesig entpackt)
            raise ValueError(tx("ZIP zu groß: entpackt {v:.1f} GB (höchstens {v2:.0f} GB).",
                                v=sum(sizes.values()) / 1e9, v2=MAX_BUNDLE_BYTES / 1e9))
        for name in sizes:
            (Path(tmp) / name).write_bytes(zf.read(names[name]))
        return load_bundle(tmp, allow_pickle=allow_pickle)


def list_bundles(models_dir) -> list[dict]:
    """Gespeicherte Textmodelle im Ordner models/ auflisten (neueste zuerst) – Modelle anderer Versionen nicht."""
    out = []
    root = Path(models_dir)
    if not root.is_dir():
        return out
    for sub in root.iterdir():
        meta_file = sub / "preprocessing.json"
        if not meta_file.is_file() or not any((sub / n).exists() for n in MODEL_FILES):
            continue
        try:
            r = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if not isinstance(r, dict) or r.get("app") != APP_ID:
            continue
        out.append({"name": sub.name, "path": str(sub), "label": r.get("label"), "text_col": r.get("text_col"),
                    "model": (r.get("training") or {}).get("model"), "saved_at": str(r.get("saved_at", "")),
                    "metrics": r.get("metrics") if isinstance(r.get("metrics"), dict) else {}})
    return sorted(out, key=lambda b: b["saved_at"], reverse=True)


# ---------------------------------------------------------------------------
# 10) Neue Texte einordnen
# ---------------------------------------------------------------------------
def required_raw_columns(recipe: dict) -> list[str]:
    """CSV-Rohspalte(n), aus denen die Textspalte entsteht."""
    return list(dict.fromkeys(raw_sources(recipe.get("lineage", {}), recipe["text_col"])))


def required_columns(recipe: dict) -> pd.DataFrame:
    """Welche Spalten brauchen neue Daten? Die Textspalte – das Label nur, wenn man die Vorhersage prüfen möchte."""
    lin = recipe.get("lineage", {})
    return pd.DataFrame([
        {tx("CSV-Spalte"): ", ".join(required_raw_columns(recipe)), tx("Rolle"): tx("Text (nötig)"),
         tx("Spalte im Modell"): recipe["text_col"]},
        {tx("CSV-Spalte"): ", ".join(raw_sources(lin, recipe["label"])),
         tx("Rolle"): tx("Label (optional – dann wird die Vorhersage geprüft)"), tx("Spalte im Modell"): recipe["label"]},
    ])


def template_csv(recipe: dict) -> bytes:
    """Leere CSV-Vorlage mit der benötigten Textspalte."""
    return (",".join(required_raw_columns(recipe)) + "\n").encode("utf-8")


RESULT_PRED, RESULT_CONF, RESULT_NOTE = "Vorhersage", "Sicherheit", "Hinweis"
NOTE_EMPTY, NOTE_NO_KNOWN, NOTE_MOSTLY_UNKNOWN = (N_("leerer Text"), N_("kein bekanntes Wort"),
                                                  N_("überwiegend unbekannte Wörter"))


def prepare_new_texts(df_new: pd.DataFrame, recipe: dict):
    """Neue Rohdaten wie die Trainingsdaten aufbereiten (Schritt 2 + Textspalte). Rückgabe: (Texte, df nach Schritt 2,
    Bericht, y oder None). Wirft MissingColumnsError, wenn die Textspalte fehlt."""
    d, _, _, _ = apply_column_actions(df_new, recipe.get("column_actions", []), fit=False)
    d, _, _, _ = apply_new_columns(d, recipe.get("new_columns", []), fit=False)
    tcol = recipe["text_col"]
    if tcol not in d.columns:
        raise MissingColumnsError([tcol], recipe.get("lineage"))
    texts = text_values(d[tcol]).to_numpy(dtype=object)
    lin = recipe.get("lineage", {})
    used_raw = set(required_raw_columns(recipe)) | set(raw_sources(lin, recipe["label"])) | {tcol, recipe["label"]}
    rep = {"extra_columns": [c for c in df_new.columns if c not in used_raw]}
    # Kennt das Modell diese Texte schon? (Vergleich mit den Prüfsummen einer Stichprobe der Trainingstexte)
    fps = set(recipe.get("train_fingerprints") or [])
    if fps:
        hits = np.array([h in fps for h in text_fingerprints(texts).tolist()], dtype=bool) & _non_blank(texts)
        n_unique = (recipe.get("text_stats") or {}).get("unique_train") or len(fps)
        est = float(hits.sum()) * max(n_unique, len(fps)) / len(fps)
        rows = int(round(min(est, len(texts))))
        rep["train_overlap"] = {"hits": int(hits.sum()), "sample": len(fps), "rows": rows,
                                "share": rows / max(len(texts), 1)}
    y = None
    if recipe["label"] in d.columns:
        y = encode_label(d[recipe["label"]], recipe)
        # unbekannte Klassen – Zeilen ohne Label (auch nur Leerzeichen) zählen nicht dazu
        rep["label_unknown"] = int(np.isnan(y).sum() - missing_label(d[recipe["label"]]).sum())
    return texts, d, rep, y


def predict_texts(pipe, recipe: dict, df_new: pd.DataFrame, threshold: float | None = None):
    """Vorhersage für neue Texte. Rückgabe: (Ergebnis-DataFrame, Bericht, Evaluation oder None)."""
    check_recipe(recipe, pipe)
    task = recipe["task"]
    thr = recipe.get("threshold", 0.5) if threshold is None else threshold
    if len(df_new) == 0:
        raise ValueError(tx("Die Datei enthält keine Datenzeilen (nur Kopfzeile?)."))
    texts, d, rep, y = prepare_new_texts(df_new, recipe)
    vect, model = pipe.steps[0][1], pipe.steps[1][1]
    scores = predict_scores(model, vect.transform(texts), task)
    total, known = vocabulary_coverage(vect, texts)
    empty = ~_non_blank(texts)
    no_known = (total > 0) & (known == 0)
    mostly = (known > 0) & (known < 0.5 * total)
    words = pd.Series(texts, dtype=object).astype(str).str.split().str.len().to_numpy()
    rep.update(y=y, total=total, known=known, row_empty=empty, row_no_known=no_known, row_mostly_unknown=mostly,
               mean_words=float(words.mean()) if len(words) else 0.0)
    res = df_new.copy()
    classes = np.asarray(recipe["classes"])
    if task == T_BIN:
        neg, pos = recipe["classes"]
        res[f"P({pos})"] = np.round(scores, 4)
        res[RESULT_PRED] = np.where(scores > thr, pos, neg)
    else:
        res[RESULT_PRED] = classes[scores.argmax(axis=1)]
        res[RESULT_CONF] = np.round(scores.max(axis=1), 4)
        for k, cl in enumerate(recipe["classes"]):
            res[f"P({cl})"] = np.round(scores[:, k], 4)
    notes = pd.Series("", index=range(len(res)), dtype=object)
    for mask, text in ((empty, NOTE_EMPTY), (no_known, NOTE_NO_KNOWN), (mostly, NOTE_MOSTLY_UNKNOWN)):
        if mask.any():
            notes[mask] = notes[mask].where(notes[mask] == "", notes[mask] + "; ") + text
    res[RESULT_NOTE] = notes.to_numpy()
    ev = None
    if y is not None:
        mask = np.isfinite(y)
        if mask.sum() > 0:
            ev = evaluate(y[mask], scores[mask], task, recipe["classes"], thr, majority=training_majority(recipe))
            ev["n"] = int(mask.sum())
    return res, rep, ev


def classify_text(pipe, recipe: dict, text: str, top: int = 12, threshold: float | None = None) -> dict:
    """Einen einzelnen Text einordnen und erklären (Schritt 8 und 10)."""
    check_recipe(recipe, pipe)
    return explain_text(pipe.steps[1][1], pipe.steps[0][1], recipe, text, top, threshold)


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
def free_name(columns, name: str) -> str:
    """Spaltenname, der noch nicht vergeben ist: 'split' – sonst 'split_2', 'split_3' …"""
    taken = {str(c) for c in columns}
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def split_frame(prep: TextPrep, df_viz: pd.DataFrame) -> pd.DataFrame:
    """Daten nach Schritt 4: Text (wie ihn das Modell sieht), Label, Label als Zahl und die Spalte 'split' –
    dieselben Zeilen wie im Training (ohne entfernte Zeilen), in der ursprünglichen Reihenfolge."""
    r = prep.recipe
    pos = np.concatenate([prep.idx_train, prep.idx_test])
    split = np.array(["train"] * len(prep.idx_train) + ["test"] * len(prep.idx_test), dtype=object)
    code = np.concatenate([prep.y_train, prep.y_test])
    text = np.concatenate([prep.text_train, prep.text_test])
    order = np.argsort(df_viz.index.get_indexer(pos), kind="stable")
    labels = df_viz.loc[pos[order], r["label"]].to_numpy()
    out = pd.DataFrame({r["text_col"]: text[order]})
    out[free_name(out.columns, r["label"])] = labels
    out[free_name(out.columns, f"{r['label']}_code")] = code[order].astype(int)
    out[free_name(out.columns, "split")] = split[order]
    return out


def predictions_frame(prep: TextPrep, scores: np.ndarray, recipe: dict, threshold: float = 0.5,
                      columns: dict | None = None) -> pd.DataFrame:
    """Testdaten mit Vorhersage: Text, tatsächliche und vorhergesagte Klasse, Wahrscheinlichkeit(en), richtig?
    columns = Namen der neuen Spalten je Sprache."""
    names = {"tatsächlich": "tatsächlich", RESULT_PRED: RESULT_PRED, "richtig": "richtig", **(columns or {})}
    task, classes = recipe["task"], np.asarray(recipe["classes"])
    pred = to_codes(scores, task, threshold)
    out = pd.DataFrame({recipe["text_col"]: [str(t) for t in prep.text_test]})
    out[free_name(out.columns, names["tatsächlich"])] = classes[prep.y_test]
    out[free_name(out.columns, names[RESULT_PRED])] = classes[pred]
    if task == T_BIN:
        out[free_name(out.columns, f"P({classes[1]})")] = np.round(scores, 4)
    else:
        for k, cl in enumerate(classes):
            out[free_name(out.columns, f"P({cl})")] = np.round(scores[:, k], 4)
    out[free_name(out.columns, names["richtig"])] = pred == prep.y_test
    return out


def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Text, der mit = + - @ beginnt, würde Excel als Formel ausführen (CSV-Injection) → mit ' entschärfen."""
    out = df
    for c in df.columns:
        if is_numeric(df[c]):
            continue
        s = df[c]
        risky = s.notna() & s.astype(str).str.match(r"^[=+\-@\t\r]")
        if risky.any():
            if out is df:
                out = df.copy()
            out[c] = s.astype(object).where(~risky, "'" + s.astype(str))
    return out


EXCEL_CSV = {  # Standard von Excel je Sprache/Region: (Trennzeichen, Dezimalzeichen)
    "de": (";", ","),  # Deutschland, Österreich
    "pl": (";", ","),  # Polen
    "en": (",", "."),  # Großbritannien, USA, Irland, Kanada, Australien, Neuseeland
}


def to_csv_bytes(df: pd.DataFrame, german: bool = False, excel: str | None = None) -> bytes:
    """CSV als Bytes – international (Komma, Dezimalpunkt) oder für Excel einer Sprache/Region
    (excel = 'de', 'pl', 'en' – Trennzeichen und Dezimalzeichen wie Excel dort, UTF-8 mit BOM; Text, der wie
    eine Excel-Formel beginnt, wird mit ' entschärft). german=True entspricht excel='de'."""
    excel = excel or ("de" if german else None)
    if excel:
        sep, dec = EXCEL_CSV[excel]
        return excel_safe(df).to_csv(index=False, sep=sep, decimal=dec).encode("utf-8-sig")  # BOM: Excel-UTF-8
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# 10) Fazit: Qualität der neuen Texte und Aussage des Ergebnisses
# ---------------------------------------------------------------------------
LEVEL_OK, LEVEL_WARN, LEVEL_BAD = N_("gut"), N_("mit Einschränkungen"), N_("kritisch")
_LEVEL_RANK = {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_BAD: 2}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _n(x: int) -> str:
    """Ganze Zahl mit schmalem Leerzeichen als Tausendertrennung: 396030 → '396 030'."""
    return f"{int(x):,}".replace(",", " ")


def _level(share: float, warn: float, bad: float) -> str:
    return LEVEL_BAD if share > bad else LEVEL_WARN if share > warn else LEVEL_OK


def assess_prediction(recipe: dict, res: pd.DataFrame, rep: dict, ev: dict | None, threshold: float = 0.5) -> dict:
    """Fazit zu Schritt 10.

    Rückgabe: {"level": Gesamtstufe der Datenqualität,
               "quality": [{"level", "text", "details": [Text]}], "meaning": [Text], "advice": [Text]}"""
    task, n = recipe["task"], len(res)
    y = rep.get("y")
    stats = recipe.get("text_stats") or {}
    quality, meaning, advice = [], [], []

    def add(level, text, details=None):
        quality.append({"level": level, "text": text, "details": details or []})

    # --- Datenqualität ------------------------------------------------------------------------------
    extra = rep.get("extra_columns") or []
    add(LEVEL_OK, tx("{n} Texte aus der Spalte '{c}'", n=_n(n), c=md_escape(recipe["text_col"]))
        + (tx(", {n} weitere Spalten werden nicht gebraucht (ignoriert).", n=len(extra)) if extra else "."))
    n_empty = int(np.sum(rep.get("row_empty", [])))
    if n_empty:
        add(_level(n_empty / n, 0.01, 0.1),
            tx("{n} leere Texte ({pct}) – dort sagt das Modell nur nach der Häufigkeit der Klassen im Training voraus.",
               n=_n(n_empty), pct=_pct(n_empty / n)))
    n_none = int(np.sum(rep.get("row_no_known", [])))
    if n_none:
        add(_level(n_none / n, 0.02, 0.1),
            tx("{n} Texte ({pct}) enthalten kein einziges Wort aus dem Vokabular des Trainings (andere Sprache, nur "
               "Zahlen, Tippfehler?) – ihre Vorhersage ist geraten.", n=_n(n_none), pct=_pct(n_none / n)))
    n_most = int(np.sum(rep.get("row_mostly_unknown", [])))
    if n_most:
        add(_level(n_most / n, 0.1, 0.3),
            tx("{n} Texte ({pct}) bestehen überwiegend aus Wörtern, die das Modell nicht kennt – die Vorhersage stützt "
               "sich auf wenige Wörter.", n=_n(n_most), pct=_pct(n_most / n)))
    total, known = np.asarray(rep.get("total", [])), np.asarray(rep.get("known", []))
    if total.sum():
        cover, test_cover = float(known.sum() / total.sum()), stats.get("test_coverage")
        if test_cover is not None and cover < test_cover - 0.15:
            add(LEVEL_WARN, tx("Das Modell kennt nur {pct} der Wörter in diesen Texten – in den Testdaten waren es {pct2}. "
                               "Die neuen Texte unterscheiden sich deutlich vom Training (andere Sprache, anderes Thema, "
                               "andere Schreibweise?).", pct=_pct(cover), pct2=_pct(test_cover)))
        else:
            add(LEVEL_OK, tx("Das Modell kennt {pct} der Wörter in diesen Texten", pct=_pct(cover))
                + (tx(" (Testdaten: {pct}).", pct=_pct(test_cover)) if test_cover is not None else "."))
    mean_new, mean_train = rep.get("mean_words"), stats.get("mean_words")
    if mean_new and mean_train and not 1 / 3 <= mean_new / mean_train <= 3:
        add(LEVEL_WARN, tx("Die neuen Texte sind im Mittel {a} Wörter lang, im Training {b} – sehr andere Texte als im "
                           "Training.", a=f"{mean_new:.0f}", b=f"{mean_train:.0f}"))
    text_issue = any(q["level"] != LEVEL_OK for q in quality)  # bis hier: nur Hinweise zu den Texten selbst
    unknown = rep.get("label_unknown") or 0
    if unknown:
        selected = recipe.get("classes_selected")
        add(LEVEL_WARN, tx("{n} Zeilen haben im Label eine Klasse, die das Modell nicht kennt (trainiert mit: {c}) – sie "
                           "fehlen in der Evaluation; vorhergesagt wird für sie trotzdem eine der bekannten Klassen.",
                           n=_n(unknown), c=", ".join(md_escape(c) for c in recipe["classes"])))
        if selected:
            advice.append(tx("Das Modell wurde nur mit den Klassen {c} trainiert (Schritt 4). Für alle Klassen in Schritt 4 "
                             "keine Auswahl treffen und neu trainieren.", c=", ".join(md_escape(c) for c in selected)))
    level = max((q["level"] for q in quality), key=_LEVEL_RANK.get)

    # --- Überschneidung mit den Trainingsdaten -------------------------------------------------------
    ov = rep.get("train_overlap") or {}
    overlap = ov.get("share", 0.0)
    if overlap >= 0.05:
        meaning.append(tx("**Ca. {pct} der Texte ({n} von {n2}) kennt das Modell wortgleich aus dem Training.** Für diese "
                          "Texte hat es die richtige Antwort beim Lernen gesehen – Kennzahlen auf solchen Daten sind zu "
                          "optimistisch.", pct=_pct(overlap), n=_n(ov['rows']), n2=_n(n)))
        advice.append(tx("Für eine ehrliche Bewertung nur Texte verwenden, die nicht im Training waren – oder die "
                         "Kennzahlen aus Schritt 8 (Testdaten) heranziehen."))

    # --- Was sagt das Ergebnis? ----------------------------------------------------------------------
    test = recipe.get("metrics") or {}
    train_share = (recipe.get("label_stats") or {}).get("class_share", {})
    if task == T_BIN:
        neg, pos = recipe["classes"]
        p = res[f"P({pos})"].to_numpy(dtype=float)
        is_pos = (res[RESULT_PRED] == pos).to_numpy()
        pred_share = float(is_pos.mean())
        txt = tx("Vorhergesagt '{pos}': {pct} der Texte", pos=md_escape(pos), pct=_pct(pred_share))
        true_share = None
        known_rows = np.isfinite(y) if y is not None else np.zeros(n, dtype=bool)
        if known_rows.any():
            true_share = float(np.mean(y[known_rows]))
            if known_rows.all():
                txt += tx(" · tatsächlich in diesen Daten: {pct}", pct=_pct(true_share))
            else:
                txt += tx(" · in den {n} Zeilen mit bekanntem Label: vorhergesagt {pct}, tatsächlich {pct2}",
                          n=_n(int(known_rows.sum())), pct=_pct(float(is_pos[known_rows].mean())),
                          pct2=_pct(true_share))
        if pos in train_share:
            txt += tx(" · in den Trainingsdaten: {pct}", pct=_pct(train_share[pos]))
        meaning.append(txt + ".")
        if ev is not None and true_share is not None:
            rep_pos = ev["report"].get(pos, {})
            recall, precision = rep_pos.get("recall", 0.0), rep_pos.get("precision", 0.0)
            gap = float(is_pos[known_rows].mean()) - true_share
            if abs(gap) >= 0.02:
                meaning.append(tx("Das Modell sagt '{pos}' {direction} voraus, als es tatsächlich vorkommt: Es erkennt "
                                  "{pct} der echten '{pos}' (Recall), und {pct2} seiner '{pos}'-Vorhersagen stimmen "
                                  "(Precision) – bei der Schwelle {threshold:.2f}.",
                                  pos=md_escape(pos), direction=tx("seltener") if gap < 0 else tx("häufiger"),
                                  pct=_pct(recall), pct2=_pct(precision), threshold=threshold))
            if gap <= -0.02:
                advice.append(tx("Mehr '{pos}' erkennen: Schwelle senken (z. B. {v:.2f}) oder in Schritt 6 "
                                 "class_weight='balanced' – kostet mehr Fehlalarme.",
                                 pos=md_escape(pos), v=max(threshold - 0.15, 0.05)))
            elif gap >= 0.02:
                advice.append(tx("Weniger Fehlalarme für '{pos}': Schwelle erhöhen (z. B. {v:.2f}).",
                                 pos=md_escape(pos), v=min(threshold + 0.15, 0.95)))
        elif pos in train_share and abs(pred_share - train_share[pos]) > 0.15:
            meaning.append(tx("Der vorhergesagte Anteil weicht deutlich vom Training ab – die neuen Texte unterscheiden "
                              "sich womöglich vom Training (anderes Thema, andere Quelle, andere Zeit)."))
        unsure = float(np.mean(np.abs(p - threshold) < 0.1))
        meaning.append(tx("{pct} der Vorhersagen liegen nahe an der Schwelle (±0.1) und sind unsicher; Werte nahe 0 oder "
                          "1 sind sicher.", pct=_pct(unsure)))
    else:
        share_pred = res[RESULT_PRED].value_counts(normalize=True)
        diff = max((abs(share_pred.get(c, 0.0) - train_share.get(c, 0.0)) for c in recipe["classes"]), default=0.0)
        meaning.append(tx("Durchschnittliche Sicherheit der Vorhersagen: {pct}; {pct2} der Texte unter 50 % (unsicher).",
                          pct=_pct(float(res[RESULT_CONF].mean())),
                          pct2=_pct(float((res[RESULT_CONF] < 0.5).mean()))))
        if diff > 0.15:
            meaning.append(tx("Die Verteilung der vorhergesagten Klassen weicht deutlich vom Training ab."))

    # --- Vergleich mit dem Test aus dem Training ------------------------------------------------------
    if ev is not None and "accuracy" in test:
        key, name = ("roc_auc", "ROC-AUC") if ev.get("auc") is not None and "roc_auc" in test else ("accuracy",
                                                                                                    "Accuracy")
        new = ev["auc"] if key == "roc_auc" else ev["accuracy"]
        old = test[key]
        if new - old > 0.02:
            why = (tx(" – vermutlich, weil das Modell einen Teil der Texte aus dem Training kennt") if overlap >= 0.05
                   else "")
            meaning.append(tx("{name} hier {new:.3f} – besser als im Test ({old:.3f}){why}.",
                              name=name, new=new, old=old, why=why))
        elif new - old < -0.03 and overlap < 0.05:
            meaning.append(tx("{name} hier {new:.3f} – schlechter als im Test ({old:.3f}): Die neuen Texte unterscheiden "
                              "sich vom Training (Thema, Sprache, Quelle, Zeit).", name=name, new=new, old=old))
            advice.append(tx("Prüfen, ob sich die neuen Texte vom Training unterscheiden; ggf. mit aktuelleren Texten neu "
                             "trainieren."))
        elif new - old < -0.03:
            meaning.append(tx("{name} hier {new:.3f}, im Test {old:.3f}.", name=name, new=new, old=old))
        else:
            meaning.append(tx("{name} hier {new:.3f} – vergleichbar mit dem Test ({old:.3f}): Das Modell arbeitet auf "
                              "diesen Texten so gut wie erwartet.", name=name, new=new, old=old))
    if text_issue:
        advice.append(tx("Texte prüfen: leere Texte, andere Sprache oder ganz andere Themen als im Training – betroffene "
                         "Zeilen stehen in der Spalte 'Hinweis'."))
    return {"level": level, "quality": quality, "meaning": meaning, "advice": advice}
