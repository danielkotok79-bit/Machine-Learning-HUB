"""
pipeline.py – Rechenlogik des Data-Analysis-SupervisedLearning (ohne Streamlit)
===============================================================================

Alle Schritte einer Supervised-Learning-Analyse (scikit-learn) für Tabellen (CSV) als
wiederverwendbare Funktionen. Die Streamlit-App (app.py) ruft nur diese Funktionen auf – das
Modul lässt sich genauso im Jupyter Notebook verwenden:

    import pipeline as pl
    df, _ = pl.read_csv("daten.csv")

Schritte
    1) Bibliotheken ............ Import-Block direkt unten
    2) Aufbereitung 1 .......... suggest_action(), apply_column_actions(), apply_new_columns()
    4) Aufbereitung 2 .......... suggest_missing(), prepare_ml_data()  (Split zufällig oder zeitlich)
    5) Daten speichern ......... ml_ready_frame(), to_csv_bytes()
    6) Modell & Training ....... MODELS, ModelConfig, make_model(), train_model()
    7) Training vs. Validierung  validation_curve(), grid_search(), curve_assessment()
    8) Evaluation .............. predict_scores(), evaluate(), compare_probabilities(), permutation_importance(),
                                 model_details()
    9) Speichern & Laden ....... save_bundle(), load_bundle()  (model.skops oder model.joblib)
   10) Neue Daten .............. predict_new(), assess_prediction()

Modelle (Reihenfolge = Rangliste): 1 Gradient Boosting (HistGradientBoosting) · 2 Decision Trees + Random Forests ·
3 Support Vector Machines · 4 Logistische Regression · 5 K-Nearest Neighbors · 6 Lineare Regression

Kernidee für Schritt 10: Alles, was in Schritt 2 und 4 aus den Trainingsdaten
gelernt wird (Füllwerte, Kategorien bzw. Dummy-Spalten, Skalierung,
Label-Kodierung), steht in einem "Rezept" (dict). Das Rezept wird zusammen mit
dem Modell gespeichert und auf neue Daten unverändert angewendet – auf neuen
Daten wird nichts neu gefittet.
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
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.class_weight import compute_class_weight

from i18n import N_, tx  # Texte in der gewählten Sprache (Deutsch = Original); N_ markiert nur

TEMPLATE_VERSION = 3  # 3 = Füllwerte/Kategorien nur aus Trainingszeilen · 2 = nach dem Audit · ältere bleiben ladbar
RANDOM_STATE = 101  # Standard-Seed für Split und Training

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
T_BIN, T_MULTI, T_REG = "binary", "multiclass", "regression"
TASK_LABELS = {
    T_BIN: N_("Binär (2 Klassen)"),
    T_MULTI: N_("Mehrklassen"),
    T_REG: N_("Regression (Zahl vorhersagen)"),
}
SCALERS = ["MinMaxScaler", "StandardScaler", N_("keiner")]

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
ACTIONS = [A_KEEP, A_DROP, A_NUM, A_YEAR, A_MONTH, A_YEAR_MONTH, A_FIRST, A_LAST, A_REPLACE, A_TOPN, A_CAT]
DATE_ACTIONS = (A_YEAR, A_MONTH, A_YEAR_MONTH)
ACTION_HELP = {
    A_KEEP: ("–", N_("unverändert; Text bleibt Kategorie und wird in Schritt 4 zu Dummies")),
    A_DROP: ("–", N_("Spalte löschen (z. B. emp_title, title oder Datenleck wie issue_d)")),
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
}
OTHER = "Sonstige"

# Schritt 4 – fehlende Werte
M_DROP_ROWS = N_("Zeilen entfernen")
M_DROP_COL = N_("Spalte entfernen")
M_MEAN = N_("Mittelwert")
M_MEDIAN = N_("Median")
M_MODE = N_("häufigster Wert")
M_CONST = N_("Konstante")
M_GROUP = N_("Gruppen-Mittelwert")
MISSING_STRATEGIES = [M_DROP_ROWS, M_DROP_COL, M_MEDIAN, M_MEAN, M_MODE, M_CONST, M_GROUP]
MISSING_HELP = {
    M_DROP_ROWS: ("–", N_("df.dropna(subset=[…]) – sinnvoll bei < 1 % fehlend")),
    M_DROP_COL: ("–", N_("Spalte komplett weglassen – bei sehr vielen Lücken")),
    M_MEDIAN: ("–", N_("Median der Spalte in den Trainingsdaten (robust gegen Ausreißer)")),
    M_MEAN: ("–", N_("Mittelwert der Spalte in den Trainingsdaten")),
    M_MODE: ("–", N_("häufigster Wert in den Trainingsdaten – auch für Kategorien")),
    M_CONST: (N_("Wert, z. B. 0 oder fehlend"), N_("fester Wert; bei Kategorien wird 'fehlend' eine eigene Kategorie")),
    M_GROUP: (N_("Gruppenspalte, z. B. total_acc"), N_("Mittelwert je Gruppe in den Trainingsdaten, z. B. mort_acc je "
                                                "total_acc – passende Gruppe: Korrelation (Hilfe unten)")),
}

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
    "w_test_size": (float, 0.05, 0.5), "w_val_split": (float, 0.05, 0.4), "w_threshold": (float, 0.05, 0.95),
    "w_svm_c": (float, 1e-4, 1e6), "w_svm_eps": (float, 0.0, 1e6), "w_log_c": (float, 1e-4, 1e6),
    "w_split_seed": (int, 0, 1_000_000), "w_seed": (int, 0, 1_000_000),
    "w_rf_trees": (int, 1, 5000), "w_rf_depth": (int, 0, 1000), "w_rf_leaf": (int, 1, 100_000),
    "w_gb_iter": (int, 1, 5000), "w_gb_lr": (float, 0.001, 1.0), "w_gb_leaves": (int, 2, 1000),
    "w_gb_depth": (int, 0, 1000), "w_gb_leaf": (int, 1, 100_000),
    "w_svm_cv": (int, 2, 10), "w_log_iter": (int, 10, 100_000), "w_knn_k": (int, 1, 1000), "w_knn_p": (int, 1, 2),
    "w_stratify": (bool, None, None), "w_drop_first": (bool, None, None), "w_scale_y": (bool, None, None),
    "w_cw": (bool, None, None), "w_curve": (bool, None, None), "w_svm_grid": (bool, None, None),
    "w_lin_intercept": (bool, None, None),
    "w_viz_cols": (list, None, None), "w_features": (list, None, None),
    **{k: (str, None, None) for k in ("w_label", "w_task", "w_pos", "w_scaler", "w_val", "w_model", "w_rf_variant",
                                      "w_rf_features", "w_svm_kernel", "w_svm_gamma", "w_knn_weights",
                                      "w_split_mode", "w_time_col", "w_base_col")},
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
    Einträge der Deep-Learning-Version (z. B. "layers", w_epochs) werden still übergangen."""
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
    missing = rows("missing")
    if missing is not None:
        out["missing"] = [{"Spalte": str(r["Spalte"]), "Strategie": _clean(r.get("Strategie")),
                           "Parameter": _clean(r.get("Parameter"))[:200]}
                          for r in missing if isinstance(r.get("Spalte"), str)
                          and _clean(r.get("Strategie")) in MISSING_STRATEGIES]
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
        if k not in SETTING_TYPES:  # z. B. w_epochs aus der Deep-Learning-Version – hier ohne Bedeutung
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
def read_csv(source, sep: str = ",", decimal: str = ".", encoding: str = "utf-8", auto: bool = True):
    """CSV laden – Pfad (str/Path) oder Bytes (Upload). Rückgabe: (DataFrame, Encoding).

    auto=True: Wird nur 1 Spalte erkannt, werden ; Tab | als Trennzeichen probiert; sieht eine
    Textspalte nach Dezimalkomma aus (1,5), wird mit decimal=',' neu gelesen. Die tatsächlich
    verwendeten Optionen stehen in df.attrs['csv'].
    """
    if isinstance(source, (str, Path)):
        source = clean_path(source)
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(tx("Datei nicht gefunden: {path}", path=path))

    def _read(enc, sep_, dec_):
        buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else Path(source).expanduser()
        return pd.read_csv(buf, sep=sep_, decimal=dec_, encoding=enc, low_memory=False)

    try:
        df = _read(encoding, sep, decimal)
    except UnicodeDecodeError:
        encoding = "latin-1" if encoding.lower().replace("-", "") == "utf8" else "utf-8"
        df = _read(encoding, sep, decimal)
    detected = []
    if auto and df.shape[1] == 1:
        header = str(df.columns[0])
        for cand in [";", "\t", "|", ","]:
            if cand != sep and cand in header:
                df2 = _read(encoding, cand, decimal)
                if df2.shape[1] > 1:
                    df, sep = df2, cand
                    detected.append(tx("Trennzeichen"))
                    break
    if auto and decimal == "." and sep != ",":
        comma_dec = [c for c in df.columns if not is_numeric(df[c])
                     and _to_str(df[c].dropna().head(200)).str.fullmatch(r"\s*-?\d+(?:\.\d{3})*,\d+\s*").mean() > 0.9]
        if comma_dec:
            df = _read(encoding, sep, ",")
            decimal = ","
            detected.append(tx("Dezimalkomma"))
    strip_columns = any(str(c) != str(c).strip() for c in df.columns)
    df.columns = [str(c).strip() for c in df.columns]
    df.attrs["csv"] = {"sep": sep, "decimal": decimal, "encoding": encoding, "detected": detected,
                       "strip_columns": strip_columns}
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
    if txt.str.fullmatch(_NUM_UNIT_RE).mean() >= 0.98:
        return A_NUM, ""
    looks_like_date = txt.str.contains(r"[-/.:]|[A-Za-z]{3}", regex=True) | txt.str.fullmatch(r"\d{8}T\d{6}")
    if looks_like_date.mean() >= 0.9 and _parse_dates(txt).notna().mean() >= 0.95:
        return A_YEAR, guess_date_format(txt)
    if nn.nunique() > 50:
        return A_DROP, ""
    return A_KEEP, ""


def column_profile(df: pd.DataFrame, info: dict | None = None) -> pd.DataFrame:
    """Tabelle für Schritt 2: Typ, Eindeutige, Fehlend %, Beispiel + vorgeschlagene Aktion.
    Datensatz des Sportdaten-Tools (match_data_layout): die Datumsspalte bleibt unverändert – Schritt 4 braucht sie
    für den zeitlichen Split (aus 'Datum → Jahr' würde nur das Jahr)."""
    rows = []
    layout = match_data_layout(df.columns)
    keep = {layout["time"]} if layout and layout["time"] else set()
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        act, param = (A_KEEP, "") if c in keep else suggest_action(s, c)
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
# 4) Datenaufbereitung 2 – fehlende Werte, Label, Features, Split, Skalierung
# ---------------------------------------------------------------------------
SPLIT_RANDOM, SPLIT_TIME = "random", "time"
SPLIT_MODES = {SPLIT_RANDOM: N_("zufällig (train_test_split)"),
               SPLIT_TIME: N_("zeitlich – Test = die jüngsten Zeilen")}
TIME_NAMES = ("date", "datum", "match_date_est", "tourney_date", "game_date", "season", "saison", "year", "jahr")


def suggest_time_column(columns) -> str | None:
    """Spalte, nach der ein zeitlicher Split naheliegt (date, season, year …) – sonst None."""
    names = {str(c).lower(): c for c in columns}
    for n in TIME_NAMES:
        if n in names:
            return names[n]
    return next((c for c in columns if re.search(r"date|datum|time|zeit", str(c), re.I)), None)


def time_key(s: pd.Series) -> tuple[pd.Series, str, str | None]:
    """Zeitspalte → sortierbare Werte: Zahl (Jahr, Saison, 20141013) oder Datum (Format wird erkannt).
    Rückgabe: (Werte, Art 'number' oder 'date', Datumsformat oder None)."""
    if is_numeric(s) and not pd.api.types.is_bool_dtype(s):
        return s.astype("float64").replace([np.inf, -np.inf], np.nan), "number", None
    fmt = guess_date_format(s) if not pd.api.types.is_bool_dtype(s) else ""
    if not fmt:
        raise ValueError(tx("Zeitspalte '{col}': weder eine Zahl (z. B. Jahr, Saison) noch ein erkennbares Datum "
                            "(z. B. 2024-01-31).", col=s.name))
    return _parse_dates(s, fmt), "date", fmt


def time_text(v) -> str:
    """Zeitwert lesbar: Datum als 2024-01-31 (mit Uhrzeit nur, wenn es eine gibt), Zahl ohne '.0'."""
    if isinstance(v, (np.datetime64, pd.Timestamp)):
        ts = pd.Timestamp(v)
        return ts.strftime("%Y-%m-%d" if ts == ts.normalize() else "%Y-%m-%d %H:%M")
    return _fmt_value(v)


def time_split(values: np.ndarray, test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Zeitlicher Split: Test = die jüngsten ceil(test_size · n) Zeilen. Zeilen mit demselben Zeitwert wie die erste
    Testzeile kommen alle in den Test – das Training liegt damit vollständig vor dem Test (kein Blick in die Zukunft).
    Rückgabe: Positionen (Training, Test), jeweils nach der Zeit sortiert (gleiche Zeit: ursprüngliche Reihenfolge)."""
    n = len(values)
    n_test = int(np.ceil(test_size * n))
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    first = ordered[n - n_test]
    pos_tr, pos_te = order[ordered < first], order[ordered >= first]
    if not len(pos_tr):
        newer = float(np.mean(ordered > ordered[0]))  # Anteil der Zeilen nach dem ältesten Zeitwert
        if newer == 0:
            raise ValueError(tx("Zeitlicher Split nicht möglich: Alle Zeilen haben denselben Zeitwert ({v}) – die "
                                "Spalte trennt keine Zeiträume. Andere Zeitspalte wählen.", v=time_text(ordered[0])))
        mx = np.floor(newer * 100) / 100
        if mx < 0.05:  # kleinster Wert des Reglers in Schritt 4 – ein kleineres test_size ist dort nicht möglich
            raise ValueError(tx("Zeitlicher Split nicht möglich: Nur {pct} der Zeilen liegen nach dem ältesten Zeitwert "
                                "({v}) – die Spalte trennt kaum Zeiträume. Eine feinere Zeitspalte wählen (z. B. Datum "
                                "statt Saison).", pct=_pct(newer), v=time_text(ordered[0])))
        raise ValueError(tx("Zeitlicher Split nicht möglich: Bei test_size = {ts} fällt die Grenze auf den ältesten "
                            "Zeitwert ({v}) – weil Zeilen mit demselben Zeitwert zusammenbleiben, kämen alle Zeilen in "
                            "den Test. test_size höchstens {mx} wählen.",
                            ts=_fmt_value(test_size), v=time_text(ordered[0]), mx=f"{mx:g}"))
    return pos_tr, pos_te


SPORT_LABELS = ("home_win", "p1_win")  # Label der Datensätze des Sportdaten-Tools (NBA bzw. Tennis)


def match_data_layout(columns) -> dict | None:
    """Datensatz im Format des Sportdaten-Tools (Label home_win bzw. p1_win und Spalten pre_* / lineup_*, dazu oft
    inplay_* und post_*)? Dann Vorschläge für Schritt 4: als Features nur Werte, die vor dem Spiel bekannt sind;
    post_* (nach dem Spiel) und inplay_* (während des Spiels) wären ein Datenleck. Erkannt wird am Label und an pre_* –
    nicht an post_*/inplay_*, die sich im Tool abwählen lassen (und Namen wie post_code in fremden Daten nicht
    verwechseln). Rückgabe: {"features", "leaky", "label", "time", "baseline"} oder None."""
    cols = [str(c) for c in columns]

    def pick(names):
        return next((n for n in names if n in cols), None)

    label = pick(SPORT_LABELS)
    pre = [c for c in cols if c.startswith(("pre_", "lineup_"))]
    if label is None or not pre:
        return None
    return {"features": pre, "leaky": [c for c in cols if c.startswith(("post_", "inplay_"))], "label": label,
            "time": pick(("date", "match_date_est", "tourney_date", "season", "year")),
            "baseline": next((c for c in pre if "elo_prob" in c), None)}


def probability_columns(df: pd.DataFrame, exclude=()) -> list[str]:
    """Zahlenspalten mit Werten zwischen 0 und 1 und mehr als zwei verschiedenen Werten (keine 0/1-Flags) – mögliche
    Vergleichs-Wahrscheinlichkeiten für Schritt 8, z. B. die Siegchance nach Elo."""
    out = []
    for c in df.columns:
        s = df[c]
        if c in exclude or not is_numeric(s) or pd.api.types.is_bool_dtype(s):
            continue
        v = s.to_numpy(dtype="float64")
        v = v[np.isfinite(v)]
        if len(v) and v.min() >= 0 and v.max() <= 1 and len(np.unique(v[:5000])) > 2:
            out.append(c)
    return out


def _n_label_values(nn: pd.Series) -> int:
    """Anzahl verschiedener Label-Werte – Text ohne Leerzeichen am Rand ('yes ' = 'yes', wie bei der Kodierung)."""
    return int(_cat(nn).nunique()) if not is_numeric(nn) else int(nn.nunique())


def suggest_task(s: pd.Series) -> str:
    nn = s.dropna()
    k = _n_label_values(nn)
    if k == 2:
        return T_BIN
    if is_numeric(nn):
        integer = _whole(nn.to_numpy(dtype=float))
        return T_MULTI if integer and k <= 10 else T_REG
    return T_MULTI


def validate_task(s: pd.Series, task: str) -> str | None:
    """Fehlermeldung, wenn Label und Problemtyp nicht zusammenpassen – sonst None."""
    nn = s.dropna()
    k = _n_label_values(nn)
    if task == T_BIN and k != 2:
        return tx("Binär braucht genau 2 verschiedene Label-Werte – gefunden: {k}.", k=k)
    if task == T_MULTI and not 2 <= k <= 200:
        return tx("Mehrklassen braucht 2–200 Klassen – gefunden: {k}.", k=k)
    if task == T_REG and not is_numeric(nn):
        return tx("Regression braucht ein numerisches Label (Text-Label → Binär oder Mehrklassen).")
    return None


def label_kind(s: pd.Series) -> str:
    """bool, numeric oder text – bestimmt, wie das Label kodiert wird."""
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    return "numeric" if is_numeric(s) else "text"


def _label_display(s: pd.Series) -> dict:
    """{Anzeigetext: Originalwert} für alle Label-Werte (1.0 → '1', True → 'True')."""
    nn = s.dropna()
    if label_kind(nn) == "text":
        return {v: v for v in _cat(nn).unique()}
    return {_fmt_value(v): _py(v) for v in nn.unique()}


def label_classes(s: pd.Series, task: str, positive_class=None) -> list[str] | None:
    """Klassen als Text in Kodier-Reihenfolge (Index = Zahl im Modell). Binär: [negativ, positiv]."""
    if task == T_REG:
        return None
    disp = _label_display(s)
    if label_kind(s.dropna()) == "text":
        classes = sorted(disp)
    else:
        classes = sorted(disp, key=lambda k: disp[k])  # nach Zahlenwert: 0.5 < 1 < 1.5
    if task == T_BIN and positive_class is not None and str(positive_class) in classes:
        pos = str(positive_class)
        classes = [c for c in classes if c != pos] + [pos]
    return classes


def encode_label(s: pd.Series, recipe: dict) -> np.ndarray:
    """Label → Zahlen (0/1, 0..K-1 oder Wert). Unbekannte Klassen → NaN.

    Zahlen- und Bool-Labels werden über den Wert kodiert (nicht über den Text), damit 1.0, 1 und '1'
    in neuen Daten dieselbe Klasse sind."""
    if recipe["task"] == T_REG:
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")
    kind = recipe.get("label_kind")
    if kind == "bool":
        mapping = {str(v).lower(): i for i, v in enumerate(recipe["class_values"])}  # 'true'/'false'
        txt = _to_str(s).str.strip().str.lower().replace({"1": "true", "0": "false", "1.0": "true", "0.0": "false"})
        return txt.map(mapping).to_numpy(dtype="float64")
    if kind == "numeric":
        mapping = {float(v): i for i, v in enumerate(recipe["class_values"])}
        return parse_numbers(s).map(mapping).to_numpy(dtype="float64")
    mapping = {c: i for i, c in enumerate(recipe["classes"])}
    values = _cat(s) if kind == "text" else _to_str(s)  # ältere Rezepte ohne label_kind
    return values.map(mapping).to_numpy(dtype="float64")


def group_candidates(df: pd.DataFrame, candidates) -> list[str]:
    """Spalten, die als Gruppe für den Gruppen-Mittelwert taugen: Zahl, höchstens 1 % Lücken, höchstens 200
    verschiedene Werte (Reihenfolge wie übergeben)."""
    return [c for c in dict.fromkeys(candidates or []) if c in df.columns and is_numeric(df[c])
            and df[c].isna().mean() <= 0.01 and df[c].nunique() <= 200]


def suggest_missing(df: pd.DataFrame, col: str, candidates: list[str] | None = None,
                    corr: pd.DataFrame | None = None) -> tuple[str, str]:
    """Vorschlag für fehlende Werte: bei Zahlen Gruppen-Mittelwert über die am stärksten korrelierte,
    diskrete Spalte (z. B. mort_acc ↔ total_acc), sonst Median bzw. eigene Kategorie.
    corr (von suggest_missing_all): Korrelationen schon berechnet – candidates sind dann bereits geprüft."""
    s = df[col]
    pct = s.isna().mean() * 100
    if pct > 40:
        return M_DROP_COL, ""
    if pct <= 1:
        return M_DROP_ROWS, ""
    if not is_numeric(s):
        return M_CONST, "fehlend"
    best, best_r = None, 0.0
    for c in (candidates or []) if corr is not None else group_candidates(df, candidates):
        if c == col:
            continue
        r = corr.at[col, c] if corr is not None else s.corr(df[c].astype(float))
        if pd.notna(r) and abs(r) > best_r:
            best, best_r = c, abs(r)
    if best is not None and best_r >= 0.3:
        return M_GROUP, best
    return M_MEDIAN, ""


def suggest_missing_all(df: pd.DataFrame, cols, candidates: list[str] | None = None) -> dict:
    """Vorschläge für mehrere Spalten – dieselben Regeln wie suggest_missing, aber die Kandidaten werden nur einmal
    geprüft und alle Korrelationen in einem Schritt berechnet (DataFrame.corr, paarweise vollständige Zeilen wie
    Series.corr). Bei breiten Daten wie denen des Sportdaten-Tools (ATP: 67 Spalten mit Lücken × 49 Kandidaten,
    79 000 Zeilen) dauert das etwa 3 statt 20 Sekunden. Rückgabe: {Spalte: (Strategie, Parameter)}."""
    cols = list(dict.fromkeys(cols))
    eligible = group_candidates(df, candidates)
    need = [c for c in cols if is_numeric(df[c]) and 1 < df[c].isna().mean() * 100 <= 40]
    corr = df[list(dict.fromkeys(need + eligible))].astype(float).corr() if need and eligible else None
    return {c: suggest_missing(df, c, eligible, corr=corr) if corr is not None else suggest_missing(df, c, [])
            for c in cols}


def _apply_fills(X: pd.DataFrame, fills: dict) -> pd.DataFrame:
    """Gespeicherte Füllwerte anwenden (Training und neue Daten – identischer Code)."""
    group_fills = []
    for c, f in fills.items():
        if c not in X.columns:
            continue
        if f["strategy"] == M_GROUP:
            group_fills.append((c, f))
        else:
            X[c] = X[c].fillna(f["value"])
    for c, f in group_fills:  # Gruppen zuletzt: Gruppenspalte ist dann schon gefüllt
        means = {k: v for k, v in f["group_means"]}
        X[c] = X[c].fillna(X[f["group_col"]].map(means)).fillna(f["overall"])
    return X


def transform_features(df: pd.DataFrame, recipe: dict):
    """Features → Modell-Matrix (vor der Skalierung): Typen, Füllwerte, Dummies, Spaltenreihenfolge.

    Wird für Training UND neue Daten benutzt. Unbekannte Kategorien werden zu 0 in allen
    Dummy-Spalten, fehlende Werte mit den Trainingswerten gefüllt.
    Rückgabe: (DataFrame float64 – wie pandas/scikit-learn im Notebook, Bericht)
    """
    feats = recipe["features"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing, recipe.get("lineage"))
    X = df[feats].copy()
    report = {"imputed": {}, "unknown": {}, "invalid_numeric": {}}
    for c in recipe["numeric_features"]:
        if not is_numeric(X[c]):
            had_value = X[c].notna()
            X[c] = parse_numbers(X[c])  # Text in Zahlenspalte, z. B. '1,5', 'True' oder '12 kg'
            bad = int((had_value & X[c].isna()).sum())
            if bad:
                report["invalid_numeric"][c] = bad
        X[c] = X[c].astype(float).replace([np.inf, -np.inf], np.nan)
    for c in recipe["categorical_features"]:
        X[c] = _cat(X[c])

    na = X.isna()
    report["imputed"] = {c: int(n) for c, n in na.sum().items() if n}
    row_flag = na.any(axis=1).to_numpy()
    X = _apply_fills(X, recipe["fills"])
    for c in feats:  # Rest, z. B. Spalten mit "Zeilen entfernen" im Training
        if X[c].isna().any():
            X[c] = X[c].fillna(recipe["fallback_fill"][c])

    unknown_flag = np.zeros(len(X), dtype=bool)
    for c in recipe["categorical_features"]:
        cats = recipe["categories"][c]
        known = X[c].isin(cats)
        if not known.all():
            report["unknown"][c] = {str(k): int(v) for k, v in X.loc[~known, c].value_counts().head(10).items()}
            unknown_flag |= (~known).to_numpy()
        X[c] = pd.Categorical(X[c].where(known), categories=cats)
    if recipe["categorical_features"]:
        X = pd.get_dummies(
            X, columns=recipe["categorical_features"], drop_first=recipe["drop_first"], dtype="float64"
        )
    if recipe.get("feature_columns"):
        X = X.reindex(columns=recipe["feature_columns"], fill_value=0.0)
    report["row_imputed"] = row_flag
    report["row_unknown"] = unknown_flag
    return X.astype("float64"), report


def fit_scaler(X: np.ndarray, kind: str) -> dict:
    """Skalierer nur auf X_train fitten; Parameter als Liste speichern (JSON)."""
    if kind == "MinMaxScaler":
        sc = MinMaxScaler().fit(X)
        return {"type": kind, "scale": sc.scale_.tolist(), "offset": sc.min_.tolist()}
    if kind == "StandardScaler":
        sc = StandardScaler().fit(X)
        return {"type": kind, "mean": sc.mean_.tolist(), "std": sc.scale_.tolist()}
    return {"type": "keiner"}


def apply_scaler(X, params: dict) -> np.ndarray:
    """Gleiche Rechnung wie scaler.transform() von scikit-learn."""
    X = np.asarray(X, dtype="float64")
    if params["type"] == "MinMaxScaler":
        X = X * np.asarray(params["scale"]) + np.asarray(params["offset"])
    elif params["type"] == "StandardScaler":
        X = (X - np.asarray(params["mean"])) / np.asarray(params["std"])
    return X


@dataclass
class PrepResult:
    recipe: dict
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    idx_train: np.ndarray  # Zeilen-Index (df aus Schritt 2) – für die Einzelfallprüfung
    idx_test: np.ndarray
    X_preview: pd.DataFrame  # erste Zeilen nach Dummies, unskaliert
    log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    y_train_fit: np.ndarray | None = None  # skaliertes Label (nur Regression mit Label-Skalierung)
    y_test_fit: np.ndarray | None = None
    base_test: np.ndarray | None = None  # Vergleichs-Wahrscheinlichkeit aus den Daten (z. B. Elo) je Testzeile

    def fit_targets(self):
        """Label so, wie das Modell es beim Training sieht (ggf. skaliert)."""
        if self.y_train_fit is None:
            return self.y_train, self.y_test
        return self.y_train_fit, self.y_test_fit


def prepare_ml_data(
    df: pd.DataFrame,
    label: str,
    features: list[str],
    task: str,
    positive_class=None,
    missing: dict | None = None,
    drop_first: bool = True,
    scaler: str = "MinMaxScaler",
    test_size: float = 0.33,
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
    column_actions: list | None = None,
    lineage: dict | None = None,
    meta: dict | None = None,
    scale_label: bool = False,
    new_columns: list | None = None,
    split_mode: str = SPLIT_RANDOM,
    time_column: str | None = None,
    baseline_column: str | None = None,
) -> PrepResult:
    """4) Label kodieren → Split (zufällig wie train_test_split oder zeitlich: Test = die jüngsten Zeilen) →
    Füllwerte, Kategorien/Dummies, Skalierung.

    Alles, was aus Daten gelernt wird (Füllwerte, Kategorien, Scaler), stammt nur aus den Trainingszeilen;
    die Testzeilen laufen durch dasselbe Rezept wie später neue Daten (transform_features).
    baseline_column (nur binär): Spalte mit einer Wahrscheinlichkeit für die positive Klasse aus den Daten (z. B. die
    Siegchance nach Elo) – Schritt 8 vergleicht das Modell damit auf denselben Testzeilen.
    """
    if label not in df.columns:
        raise ValueError(tx("Label '{label}' ist nicht in den Daten.", label=label))
    if split_mode not in SPLIT_MODES:
        raise ValueError(tx("Unbekannte Art des Splits: {v}", v=split_mode))
    if not 0 < float(test_size) < 1:
        raise ValueError(tx("test_size muss zwischen 0 und 1 liegen (z. B. 0.2) – angegeben: {v}.", v=test_size))
    if split_mode == SPLIT_TIME:
        if not time_column or time_column not in df.columns:
            raise ValueError(tx("Zeitlicher Split: bitte eine Zeitspalte wählen (z. B. date oder season)."))
        if time_column == label:
            raise ValueError(tx("Die Zeitspalte darf nicht das Label sein."))
    if baseline_column is not None and baseline_column not in df.columns:
        raise ValueError(tx("Vergleichsspalte '{col}' ist nicht in den Daten.", col=baseline_column))
    if baseline_column is not None and baseline_column == label:
        raise ValueError(tx("Die Vergleichsspalte darf nicht das Label sein."))
    features = [f for f in dict.fromkeys(features) if f != label]
    unknown = [f for f in features if f not in df.columns]
    if unknown:
        raise ValueError(tx("Unbekannte Features: {unknown}", unknown=unknown))
    if not features:
        raise ValueError(tx("Bitte mindestens ein Feature auswählen."))
    missing = dict(missing or {})
    lineage = dict(lineage or {})
    log, warn = [], []

    d = df[features + [label]].copy()
    n0 = len(d)
    d = d[d[label].notna()].copy()
    if len(d) < n0:
        log.append(tx("{v} Zeilen ohne Label entfernt", v=n0 - len(d)))
    label_inf = 0
    if is_numeric(d[label]) and not pd.api.types.is_bool_dtype(d[label]):
        finite = np.isfinite(d[label].to_numpy(dtype=float))
        label_inf = int((~finite).sum())
        if label_inf:
            d = d[finite].copy()
            log.append(tx("{label_inf} Zeilen mit unendlichem Label (inf) entfernt", label_inf=label_inf))
    err = validate_task(d[label], task)
    if err:
        raise ValueError(err)

    # ±inf in Zahlenspalten wie fehlende Werte behandeln (sonst Fehler beim Skalieren / NaN-Loss)
    inf_cols = []
    for c in features:
        if is_numeric(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            n_inf = int(np.isinf(d[c].to_numpy(dtype=float)).sum())
            if n_inf:
                d[c] = d[c].astype(float).replace([np.inf, -np.inf], np.nan)
                inf_cols.append(c)
                log.append(tx("{c}: {n_inf} unendliche Werte (inf) als fehlend behandelt", c=c, n_inf=n_inf))
    # Spalten ohne einen einzigen Wert tragen keine Information
    empty = [c for c in features if d[c].isna().all()]
    if empty:
        d = d.drop(columns=empty)
        features = [f for f in features if f not in empty]
        warn.append(tx("Spalten ohne Werte entfernt: {empty}", empty=empty))
    if not features:
        raise ValueError(tx("Keine Features mit Werten übrig."))

    def strategy(c):
        return (missing.get(c) or {}).get("strategy")

    # a) Spalten entfernen
    drop_cols = [c for c in features if d[c].isna().any() and strategy(c) == M_DROP_COL]
    if drop_cols:
        d = d.drop(columns=drop_cols)
        features = [f for f in features if f not in drop_cols]
        log.append(tx("Spalten wegen fehlender Werte entfernt: {drop_cols}", drop_cols=drop_cols))
    if not features:
        raise ValueError(tx("Nach dem Entfernen der Spalten bleiben keine Features übrig."))
    # b) Zeilen entfernen
    dropna_cols = [c for c in features if d[c].isna().any() and strategy(c) == M_DROP_ROWS]
    if dropna_cols:
        n1 = len(d)
        d = d.dropna(subset=dropna_cols)
        log.append(tx("{v} Zeilen mit fehlenden Werten in {dropna_cols} entfernt",
                      v=n1 - len(d), dropna_cols=dropna_cols))
    if len(d) < 20:
        raise ValueError(tx("Nur noch {n} Zeilen übrig – zu wenig für Training und Test.", n=len(d)))
    tkey = time_kind = time_fmt = None
    time_removed = 0
    if split_mode == SPLIT_TIME:  # Zeitwerte der verbleibenden Zeilen; Zeilen ohne Zeitwert lassen sich nicht einordnen
        tkey, time_kind, time_fmt = time_key(df.loc[d.index, time_column])
        known = tkey.notna().to_numpy()
        time_removed = int((~known).sum())
        if time_removed:
            d, tkey = d[known], tkey[known]
            log.append(tx("{n} Zeilen ohne Wert in der Zeitspalte '{col}' entfernt", n=time_removed, col=time_column))
            if len(d) < 20:
                raise ValueError(tx("Nur noch {n} Zeilen übrig – zu wenig für Training und Test.", n=len(d)))

    numeric = [c for c in features if is_numeric(d[c])]
    categorical = [c for c in features if c not in numeric]
    for c in numeric:
        d[c] = d[c].astype(float)
    for c in categorical:
        d[c] = _cat(d[c])

    # c) Label kodieren – die Klassen (aus allen Zeilen) beschreiben die Aufgabe, sie werden nicht "gelernt"
    classes = label_classes(d[label], task, positive_class)
    kind = label_kind(d[label])
    class_values = None
    if classes is not None:
        disp = _label_display(d[label])
        class_values = [disp[c] for c in classes]
    y = encode_label(d[label], {"task": task, "label_kind": kind, "classes": classes, "class_values": class_values})
    if np.isnan(y).any():
        raise ValueError(tx("Label konnte nicht vollständig kodiert werden."))
    if task != T_REG:
        counts = np.bincount(y.astype(int), minlength=len(classes))
        log.append(tx("Klassen: ") + ", ".join(f"{cl} → {i} ({n})" for i, (cl, n) in enumerate(zip(classes, counts))))

    # d) Split ZUERST – Füllwerte, Kategorien und Scaler werden danach nur aus den
    #    Trainingszeilen gelernt; die Testzeilen werden damit nur transformiert (wie später neue Daten).
    strat_y = None
    split_info = {}
    if split_mode == SPLIT_TIME:
        # Test = die jüngsten Zeilen, Training = alles davor – wie später im Einsatz: Das Modell kennt nur die
        # Vergangenheit. Ein zufälliger Split würde Spiele/Kurse aus der Zukunft ins Training mischen.
        values = tkey.to_numpy()
        pos_tr, pos_te = time_split(values, test_size)
        t_tr, t_te = values[pos_tr], values[pos_te]
        split_info = {"time_column": time_column, "time_kind": time_kind, "time_format": time_fmt,
                      "time_removed": time_removed, "train_from": time_text(t_tr[0]), "train_until": time_text(t_tr[-1]),
                      "test_from": time_text(t_te[0]), "test_until": time_text(t_te[-1]),
                      "test_share": float(len(pos_te) / len(values))}
        log.append(tx("Zeitlicher Split nach '{col}': Training {n} Zeilen ({a} … {b}), Test {n2} Zeilen ({c} … {d}) – "
                      "das Modell lernt nur aus der Vergangenheit",
                      col=time_column, n=_n(len(pos_tr)), a=split_info["train_from"], b=split_info["train_until"],
                      n2=_n(len(pos_te)), c=split_info["test_from"], d=split_info["test_until"]))
        if abs(split_info["test_share"] - test_size) > 0.05:
            warn.append(tx("Testanteil {pct} statt {pct2}: Zeilen mit demselben Zeitwert ('{col}') bleiben zusammen – "
                           "eine Saison oder ein Tag wird nicht zwischen Training und Test geteilt.",
                           pct=_pct(split_info["test_share"]), pct2=_pct(test_size), col=time_column))
    else:
        if task != T_REG and stratify:
            k = len(classes)
            n_test = int(np.ceil(test_size * len(y)))
            if np.bincount(y.astype(int)).min() < 2:
                warn.append(tx("Stratify nicht möglich (eine Klasse hat < 2 Zeilen) – Split ohne stratify."))
            elif n_test < k or len(y) - n_test < k:
                warn.append(tx("Stratify nicht möglich: Testmenge ({n_test} Zeilen) kleiner als Anzahl Klassen ({k}) – "
                               "Split ohne stratify. Tipp: test_size erhöhen.",
                               n_test=n_test, k=k))
            else:
                strat_y = y
        # gleiche Aufteilung wie train_test_split(X, y, …): sie hängt nur von Zeilenzahl, Seed und y ab
        pos_tr, pos_te = train_test_split(np.arange(len(d)), test_size=test_size, random_state=random_state,
                                          stratify=strat_y)
    # Genug zum Lernen? Beim zeitlichen Split kann das Training klein werden (Zeilen mit demselben Zeitwert bleiben
    # zusammen) oder eine Klasse fehlen (z. B. im frühen Zeitraum nur Siege) – dann lernt kein Modell etwas Sinnvolles.
    period = (tx(" (zeitlicher Split: Training {a} … {b})", a=split_info["train_from"], b=split_info["train_until"])
              if split_mode == SPLIT_TIME else "")
    if len(pos_tr) < 10:
        raise ValueError(tx("Nur {n} Trainingszeilen{period} – zu wenig zum Lernen (mindestens 10). test_size "
                            "verkleinern oder eine feinere Zeitspalte wählen.", n=len(pos_tr), period=period))
    if task != T_REG:
        present = sorted(set(np.unique(y[pos_tr].astype(int)).tolist()))
        absent = [classes[i] for i in range(len(classes)) if i not in present]
        if len(present) < 2:
            raise ValueError(tx("In den Trainingsdaten{period} kommt nur die Klasse '{cls}' vor – das Modell könnte "
                                "nichts anderes vorhersagen. test_size ändern, eine andere Zeitspalte oder den "
                                "zufälligen Split (mit stratify) wählen.", period=period, cls=classes[present[0]]))
        if absent:
            warn.append(tx("Klassen ohne eine einzige Trainingszeile{period}: {cls} – das Modell kann sie nie "
                           "vorhersagen.", period=period, cls=", ".join(f"'{c}'" for c in absent)))
    d_tr = d.iloc[pos_tr]
    log.append(tx("Split zuerst: Füllwerte, Kategorien und Scaler nur aus den {n} Trainingszeilen gelernt – die "
                  "{n2} Testzeilen werden damit nur umgewandelt (kein Datenleck)",
                  n=len(pos_tr), n2=len(pos_te)))

    # Spalten ohne einen einzigen Wert in den Trainingszeilen: daraus lässt sich nichts lernen
    empty_tr = [c for c in features if d_tr[c].isna().all()]
    if empty_tr:
        features = [f for f in features if f not in empty_tr]
        numeric = [c for c in numeric if c not in empty_tr]
        categorical = [c for c in categorical if c not in empty_tr]
        empty = empty + empty_tr
        warn.append(tx("Spalten ohne Werte in den Trainingsdaten entfernt: {empty_tr}", empty_tr=empty_tr))
    if not features:
        raise ValueError(tx("Keine Features mit Werten in den Trainingsdaten übrig."))

    # e) Füllwerte NUR aus den Trainingszeilen lernen (gespeichert im Rezept; gelten für Test und neue Daten)
    fills = {}
    for c in features:
        if not d[c].isna().any():
            continue
        cfg = missing.get(c) or {}
        strat = cfg.get("strategy") or (M_MEDIAN if c in numeric else M_MODE)
        param = _clean(cfg.get("param"))
        if strat in (M_MEAN, M_MEDIAN, M_GROUP) and c in categorical:
            raise ValueError(tx("'{c}' ist eine Kategorie – '{strat}' geht nur bei Zahlen. Bitte '{m_mode}' oder "
                                "'{m_const}' wählen.",
                                c=c, strat=tx(strat), m_mode=tx(M_MODE), m_const=tx(M_CONST)))
        if strat in (M_DROP_ROWS, M_DROP_COL):
            continue
        n_na = int(d[c].isna().sum())
        tr = d_tr[c]
        if strat == M_MEAN:
            fills[c] = {"strategy": strat, "value": float(tr.mean())}
        elif strat == M_MEDIAN:
            fills[c] = {"strategy": strat, "value": float(tr.median())}
        elif strat == M_MODE:
            fills[c] = {"strategy": strat, "value": _py(tr.mode().iloc[0])}
        elif strat == M_CONST:
            value = (_to_number(param) if param else 0.0) if c in numeric else (param or "fehlend")
            fills[c] = {"strategy": strat, "value": value}
        elif strat == M_GROUP:
            g = param
            if g not in features or g == c:
                raise ValueError(tx("Gruppen-Mittelwert für '{c}': Parameter muss eine andere Feature-Spalte sein "
                                    "(z. B. total_acc) – angegeben: '{g}'.",
                                    c=c, g=g))
            means = d_tr.groupby(g)[c].mean().dropna()
            fills[c] = {
                "strategy": strat,
                "group_col": g,
                "group_means": [[_py(k), float(v)] for k, v in means.items()],
                "overall": float(tr.mean()),
            }
        else:
            raise ValueError(tx("Unbekannte Strategie '{strat}' für '{c}'.", strat=strat, c=c))
        shown = fills[c].get("value", tx("Mittelwert je {v}", v=fills[c].get('group_col')))
        if isinstance(shown, float):
            shown = round(shown, 4)
        log.append(tx("{c}: {n_na} fehlende Werte → {strat} ({shown})", c=c, n_na=n_na, strat=tx(strat), shown=shown))

    # f) Fallback-Füllwerte für neue Daten (Median / häufigster Wert der Trainingszeilen)
    fallback = {}
    for c in numeric:
        fallback[c] = float(d_tr[c].median()) if d_tr[c].notna().any() else 0.0
    for c in categorical:
        mode = d_tr[c].mode()
        fallback[c] = str(mode.iloc[0]) if len(mode) else "fehlend"

    # g) Kategorien aus den gefüllten Trainingszeilen ('fehlend' wird ggf. eigene Kategorie).
    #    Kategorien, die nur im Test vorkommen, werden wie unbekannte Werte in neuen Daten behandelt.
    fingerprints = row_fingerprints(d_tr, features, numeric)
    if len(fingerprints) > 3000:  # Stichprobe reicht für eine Schätzung (±2 %)
        fingerprints = np.random.default_rng(0).choice(fingerprints, 3000, replace=False)
    tr_cat = _apply_fills(d_tr[categorical].copy(), fills)
    categories = {c: sorted(tr_cat[c].dropna().astype(str).unique().tolist()) for c in categorical}
    del tr_cat, d_tr
    for c in categorical:
        if len(categories[c]) > 100:
            warn.append(tx("'{c}' hat {n} Kategorien → sehr viele Dummy-Spalten. Tipp: Schritt 2 'Top-N behalten' "
                           "oder entfernen.",
                           c=c, n=len(categories[c])))

    recipe = {
        "template_version": TEMPLATE_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "column_actions": column_actions or [],
        "new_columns": new_columns or [],
        "lineage": lineage,
        "label": label,
        "label_source": raw_sources(lineage, label),
        "label_is_numeric": bool(is_numeric(d[label])),
        "label_kind": kind,
        "task": task,
        "classes": classes,
        "class_values": class_values,
        "features": features,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "categories": categories,
        "missing_config": {c: {"strategy": (missing.get(c) or {}).get("strategy"),
                               "param": _clean((missing.get(c) or {}).get("param"))}
                           for c in missing},
        "dropped_columns": drop_cols,
        "empty_columns": empty,
        "inf_columns": [c for c in inf_cols if c in features],
        "dropna_columns": dropna_cols,
        "label_inf_removed": label_inf > 0,
        "fit_on": "train",  # Füllwerte, Kategorien, Scaler: nur aus den Trainingszeilen
        "fills": fills,
        "fallback_fill": fallback,
        "drop_first": bool(drop_first),
        "split": {"mode": split_mode, "test_size": float(test_size), "random_state": int(random_state),
                  "stratify": bool(stratify) and split_mode == SPLIT_RANDOM, "stratify_used": strat_y is not None,
                  **split_info},
        "threshold": 0.5,
        "label_scaler": None,
        "train_fingerprints": sorted(int(h) for h in np.unique(fingerprints)),
    }

    # h) Dummies für ALLE Zeilen mit dem Trainings-Rezept – exakt derselbe Code wie später für neue Daten
    X_df, rep = transform_features(d, recipe)
    if X_df.columns.duplicated().any():
        dupes = X_df.columns[X_df.columns.duplicated()].tolist()
        raise ValueError(tx("Doppelte Spaltennamen nach Dummy-Kodierung: {dupes} – bitte in Schritt 2 umbenennen.",
                            dupes=dupes))
    cols = list(X_df.columns)
    pos = {c: i for i, c in enumerate(cols)}
    groups = {c: [pos[c]] for c in numeric}
    for c in categorical:
        used = categories[c][1:] if drop_first else categories[c]
        groups[c] = [pos[f"{c}_{v}"] for v in used if f"{c}_{v}" in pos]
    recipe["feature_columns"] = cols
    recipe["feature_groups"] = groups
    unknown_test = int(rep["row_unknown"][pos_te].sum())
    if unknown_test:
        log.append(tx("Testdaten: {unknown_test} Zeilen mit Kategorien, die in den Trainingsdaten nicht vorkommen "
                      "→ dort alle Dummies 0 (wie bei neuen Daten)",
                      unknown_test=unknown_test))
    X_all = X_df.to_numpy(dtype="float64")  # float64 wie im Notebook – scikit-learn rechnet damit genau gleich
    X_tr, X_te = X_all[pos_tr], X_all[pos_te]
    del X_all
    y_tr, y_te = y[pos_tr], y[pos_te]
    index = d.index.to_numpy()
    i_tr, i_te = index[pos_tr], index[pos_te]

    # Vergleichs-Wahrscheinlichkeit aus den Daten (z. B. Elo) – nur gelesen, nie gelernt; gilt für die positive Klasse
    base_te = None
    recipe["baseline_column"] = None
    if baseline_column is not None:
        if task != T_BIN:
            warn.append(tx("Vergleichsspalte '{col}' wird ignoriert – den Vergleich gibt es nur bei binären Labels.",
                           col=baseline_column))
        else:
            def probs(rows):
                b = df.loc[rows, baseline_column]
                return (b if is_numeric(b) else parse_numbers(b)).to_numpy(dtype="float64", na_value=np.nan)

            b = probs(i_te)
            ok = b[np.isfinite(b)]
            if not len(ok):
                raise ValueError(tx("Vergleichsspalte '{col}' hat in den Testzeilen keinen einzigen Wert.",
                                    col=baseline_column))
            if ok.min() < 0 or ok.max() > 1:
                raise ValueError(tx("Vergleichsspalte '{col}' muss Wahrscheinlichkeiten zwischen 0 und 1 enthalten.",
                                    col=baseline_column))
            base_te = b
            recipe["baseline_column"] = baseline_column
            if len(ok) < len(b):
                log.append(tx("Vergleichsspalte '{col}': {n} Testzeilen ohne Wert – der Vergleich nutzt nur Zeilen mit "
                              "Wert", col=baseline_column, n=_n(len(b) - len(ok))))
            # Passt die Spalte zur positiven Klasse? Auf den Trainingszeilen (der Test bleibt unberührt) muss sie
            # „positive Klasse“ besser als Zufall vorhersagen – sonst ist es meist die Wahrscheinlichkeit der anderen
            # Klasse (z. B. Siegchance der Heimmannschaft, aber positive Klasse „0“ = Auswärtssieg).
            b_tr = probs(i_tr)
            fin = np.isfinite(b_tr)
            recipe["baseline_auc_train"] = None
            if fin.sum() >= 10 and len(np.unique(y_tr[fin])) == 2:
                auc = float(roc_auc_score(y_tr[fin].astype(int), b_tr[fin]))
                recipe["baseline_auc_train"] = auc
                if auc < 0.5:
                    warn.append(tx("Vergleichsspalte '{col}' sagt '{pos}' auf den Trainingszeilen schlechter vorher "
                                   "als Zufall (ROC-AUC {auc:.2f}) – vermutlich ist sie die Wahrscheinlichkeit für "
                                   "'{neg}'. Dann in Schritt 4 '{neg}' als positive Klasse wählen, sonst ist der "
                                   "Vergleich in Schritt 8 wertlos.",
                                   col=baseline_column, pos=classes[1], neg=classes[0], auc=auc))

    # i) Datenleck-Check (Trainingszeilen): Feature korreliert fast perfekt mit dem Label?
    if task in (T_BIN, T_REG) and np.std(y_tr) > 0:
        rows = np.arange(len(y_tr))
        if len(rows) > 50_000:  # Stichprobe reicht für den Check
            rows = np.random.default_rng(0).choice(rows, 50_000, replace=False)
        Xv = X_tr[rows].astype("float64")
        yv = y_tr[rows].astype("float64")
        Xc = Xv - Xv.mean(axis=0)
        yc = yv - yv.mean()
        denom = np.sqrt(np.einsum("ij,ij->j", Xc, Xc) * (yc @ yc))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(denom > 0, (yc @ Xc) / denom, 0.0)
        suspects = [f"{cols[i]} (r={r[i]:+.2f})" for i in np.where(np.abs(r) >= 0.95)[0]]
        if suspects:
            warn.append(tx("Verdacht auf Datenleck – fast perfekte Korrelation mit dem Label: ")
                        + ", ".join(suspects) + tx(". Gehört diese Spalte wirklich zu den Features?"))
        del Xv, Xc
    same_source = [f for f in features if set(raw_sources(lineage, f)) & set(recipe["label_source"])]
    if same_source:
        warn.append(tx("Feature(s) {same_source} stammen aus derselben CSV-Spalte wie das Label → Datenleck!",
                       same_source=same_source))

    # j) Skalierung – fit nur auf X_train (kein Datenleck aus dem Testset)
    recipe["scaler"] = fit_scaler(X_tr, scaler)
    recipe["train_min"] = X_tr.min(axis=0).tolist()
    recipe["train_max"] = X_tr.max(axis=0).tolist()
    if task == T_REG:
        recipe["label_stats"] = {"mean": float(np.mean(y_tr)), "std": float(np.std(y_tr)),
                                 "min": float(np.min(y_tr)), "max": float(np.max(y_tr))}
    else:
        share = np.bincount(y_tr.astype(int), minlength=len(classes)) / len(y_tr)
        recipe["label_stats"] = {"class_share": {cl: float(s) for cl, s in zip(classes, share)}}
    recipe["n_rows"] = {"train": int(len(y_tr)), "test": int(len(y_te))}

    # j) Optional (Regression): Label standardisieren – große Werte wie Preise lernen sonst sehr langsam
    y_tr_fit = y_te_fit = None
    if task == T_REG and scale_label:
        mu, sd = float(np.mean(y_tr)), float(np.std(y_tr)) or 1.0
        recipe["label_scaler"] = {"mean": mu, "std": sd}
        y_tr_fit = (y_tr - mu) / sd
        y_te_fit = (y_te - mu) / sd
        log.append(tx("Label skaliert: (y − {x}) / {x2} – Vorhersagen werden automatisch zurückgerechnet",
                      x=_nice(mu), x2=_nice(sd)))

    return PrepResult(
        recipe=recipe,
        X_train=apply_scaler(X_tr, recipe["scaler"]),
        X_test=apply_scaler(X_te, recipe["scaler"]),
        y_train=y_tr,
        y_test=y_te,
        idx_train=i_tr,
        idx_test=i_te,
        X_preview=X_df.head(10),
        log=log,
        warnings=warn,
        y_train_fit=y_tr_fit,
        y_test_fit=y_te_fit,
        base_test=base_te,
    )


# ---------------------------------------------------------------------------
# 6) Modell & Training – Supervised Learning mit scikit-learn
# ---------------------------------------------------------------------------
MODEL_GB, MODEL_TREES, MODEL_SVM, MODEL_LOGREG, MODEL_KNN, MODEL_LINREG = "gb", "trees", "svm", "logreg", "knn", "linreg"
MODELS = [MODEL_GB, MODEL_TREES, MODEL_SVM, MODEL_LOGREG, MODEL_KNN, MODEL_LINREG]  # Reihenfolge = Rangliste (Platz 1 … 6)
MODEL_NAMES = {
    MODEL_GB: N_("Gradient Boosting (HistGradientBoosting)"),
    MODEL_TREES: N_("Decision Trees + Random Forests"),
    MODEL_SVM: N_("Support Vector Machines"),
    MODEL_LOGREG: N_("Logistische Regression"),
    MODEL_KNN: N_("K-Nearest Neighbors"),
    MODEL_LINREG: N_("Lineare Regression"),
}
MODEL_WHY = {  # Rangliste – Spalte „Warum?“
    MODEL_GB: N_("Der beste dieser Liste für Tabellen. Baut Bäume nacheinander – jeder korrigiert die Fehler der "
                 "vorigen. Meist genauer als Random Forest und auch bei vielen Zeilen schnell; braucht aber eine "
                 "passende Anzahl Runden (max_iter), sonst lernt er Rauschen mit – Schritt 7 zeigt die beste."),
    MODEL_TREES: N_("Sehr robust und kaum zu verderben. Kommt mit Nichtlinearitäten, Wechselwirkungen zwischen Features "
                    "und Rauschen zurecht. Random Forest begrenzt das Overfitting eines einzelnen Baums stark."),
    MODEL_SVM: N_("Gut, besonders mit RBF-Kernel. Trennt Klassen gut, ist aber empfindlich gegenüber der Skalierung "
                  "der Daten und der Wahl der Hyperparameter."),
    MODEL_LOGREG: N_("Solide Baseline. Schnell, interpretierbar, stabil. Verliert bei komplexen Zusammenhängen oft "
                     "gegen Bäume."),
    MODEL_KNN: N_("Einfach, aber meist schwächer. Verträgt Rauschen, irrelevante Features und hohe Dimensionalität "
                  "(typisch für Börsendaten) schlecht."),
    MODEL_LINREG: N_("Am schwächsten für echte Vorhersagen. Setzt einen linearen Zusammenhang voraus, Märkte sind aber "
                     "stark nichtlinear und verrauscht."),
}
MODEL_BEST = {  # Rangliste – Spalte „Am besten für“
    MODEL_GB: N_("Erste Wahl für Tabellen: Richtung, Sieg/Niederlage (Klassifikation) und Zahlen (Regression)"),
    MODEL_TREES: N_("Vorhersage der Richtung (Klassifikation) und der Renditen"),
    MODEL_SVM: N_("Klassifikation der Richtung (buy/sell)"),
    MODEL_LOGREG: N_("Einfache Klassifikation der Richtung"),
    MODEL_KNN: N_("Eher Experimente / Baseline"),
    MODEL_LINREG: N_("Nur als sehr einfacher Referenzpunkt"),
}
MODEL_TASKS = {MODEL_GB: (T_BIN, T_MULTI, T_REG), MODEL_TREES: (T_BIN, T_MULTI, T_REG),
               MODEL_SVM: (T_BIN, T_MULTI, T_REG), MODEL_LOGREG: (T_BIN, T_MULTI), MODEL_KNN: (T_BIN, T_MULTI, T_REG),
               MODEL_LINREG: (T_REG,)}
GB_DEFAULTS = {"gb_iter": 100, "gb_lr": 0.1, "gb_leaves": 31, "gb_depth": 0, "gb_leaf": 20}  # wie in scikit-learn
TREE_FOREST, TREE_SINGLE = "forest", "tree"
TREE_VARIANTS = {TREE_FOREST: N_("Random Forest (viele Bäume)"), TREE_SINGLE: N_("Decision Tree (ein Baum)")}
MAX_FEATURES = ["sqrt", "log2", "1.0"]  # Random Forest: Features je Split – 1.0 = alle
KERNELS = ["rbf", "linear", "poly", "sigmoid"]
GAMMAS = ["scale", "auto", "1", "0.1", "0.01", "0.001"]
KNN_WEIGHTS = ["uniform", "distance"]
GRID_C = [0.1, 1, 10, 100]  # param_grid wie im Kurs: GridSearchCV(SVC(), param_grid, …)
GRID_GAMMA = [1, 0.1, 0.01, 0.001]
KNN_MAX_K = 40  # Elbow-Methode: K = 1 … 40
CURVE_ROWS, KNN_CURVE_ROWS = 5000, 2000  # Stichprobe je Auswertung der Validierungskurve – hält Schritt 6 schnell
MAX_TREES = 5000


@dataclass
class ModelConfig:
    model: str = MODEL_TREES
    # 1 · Gradient Boosting (HistGradientBoosting) – Standardwerte wie in scikit-learn, ohne early_stopping
    gb_iter: int = GB_DEFAULTS["gb_iter"]  # max_iter: Anzahl Boosting-Runden (Bäume nacheinander)
    gb_lr: float = GB_DEFAULTS["gb_lr"]  # learning_rate: wie stark jeder Baum korrigiert
    gb_leaves: int = GB_DEFAULTS["gb_leaves"]  # max_leaf_nodes: Größe jedes Baums
    gb_depth: int = GB_DEFAULTS["gb_depth"]  # max_depth – 0 = unbegrenzt (None)
    gb_leaf: int = GB_DEFAULTS["gb_leaf"]  # min_samples_leaf
    # 2 · Decision Trees + Random Forests
    tree_variant: str = TREE_FOREST
    n_estimators: int = 100
    max_depth: int = 0  # 0 = unbegrenzt (None)
    min_samples_leaf: int = 1
    max_features: str = "sqrt"
    # 3 · Support Vector Machines
    kernel: str = "rbf"
    svm_c: float = 1.0
    gamma: str = "scale"
    epsilon: float = 0.1  # nur SVR (Regression)
    grid_search: bool = False
    cv: int = 5
    # 4 · Logistische Regression
    logreg_c: float = 1.0
    max_iter: int = 1000
    # 5 · K-Nearest Neighbors
    n_neighbors: int = 5
    weights: str = "uniform"
    p: int = 2  # 2 = euklidisch, 1 = Manhattan
    # 6 · Lineare Regression
    fit_intercept: bool = True
    # allgemein
    class_weight: bool = False
    curve: bool = True  # Validierungskurve für Schritt 7 berechnen
    validation: str = "test"  # "test" = X_test (wie im Kurs) | "split" = Anteil aus den Trainingsdaten
    validation_split: float = 0.2
    seed: int = RANDOM_STATE


def model_name(key: str) -> str:
    """Anzeigename des Modells – unbekannte Werte (z. B. aus einem fremden Rezept) als „?“, nie als Markdown."""
    return tx(MODEL_NAMES[key]) if key in MODEL_NAMES else "?"


def supports_class_weight(cfg: ModelConfig, task: str) -> bool:
    """class_weight='balanced' gibt es für Boosting, Bäume, SVM und Logistische Regression (nicht für KNN)."""
    return task != T_REG and cfg.model in (MODEL_GB, MODEL_TREES, MODEL_SVM, MODEL_LOGREG)


def model_problem(cfg: ModelConfig, task: str) -> str | None:
    """Fehlermeldung, wenn das Modell nicht zum Problemtyp passt – sonst None."""
    if cfg.model not in MODEL_TASKS:
        return tx("Unbekanntes Modell: {m}", m=cfg.model)
    if task in MODEL_TASKS[cfg.model]:
        return None
    if cfg.model == MODEL_LOGREG:
        return tx("Logistische Regression sagt Klassen vorher. Für eine Zahl als Label (Regression) z. B. "
                  "Gradient Boosting (Platz 1) oder Lineare Regression (Platz 6) wählen – oder in Schritt 4 den "
                  "Problemtyp ändern.")
    return tx("Lineare Regression sagt eine Zahl vorher. Für Klassen z. B. Gradient Boosting (Platz 1) oder "
              "Logistische Regression (Platz 4) wählen – oder in Schritt 4 den Problemtyp Regression wählen.")


def _depth(v) -> int | None:
    v = int(v or 0)
    return None if v <= 0 else v


def _max_features(v):
    v = str(v or "sqrt")
    if v in ("sqrt", "log2"):
        return v
    try:
        f = float(v)
    except ValueError:
        return "sqrt"
    return f if 0 < f <= 1 else "sqrt"


def gamma_value(v):
    """'scale' / 'auto' oder eine Zahl (als Text aus der Auswahlliste)."""
    v = str(v or "scale").strip()
    if v in ("scale", "auto"):
        return v
    try:
        f = float(v)
    except ValueError:
        return "scale"
    return f if f > 0 else "scale"


def calibration_cv(y_train=None, recipe: dict | None = None) -> int:
    """Folds für CalibratedClassifierCV (Wahrscheinlichkeiten der SVM): 5 wie scikit-learn, weniger nur bei sehr
    seltenen Klassen. Aus den Trainings-Labels oder – für das exportierte Skript – aus dem Rezept."""
    counts = _train_counts(T_BIN, y_train, recipe)
    counts = counts[counts > 0]
    return int(max(2, min(5, counts.min()))) if len(counts) else 5


def make_model(cfg: ModelConfig, task: str, probability: bool = True, calib_cv: int = 5):
    """Noch nicht trainiertes scikit-learn-Modell aus der Konfiguration (Klassifikation oder Regression).

    SVM-Klassifikation: CalibratedClassifierCV(SVC(…), ensemble=False) liefert Wahrscheinlichkeiten für Schwelle und
    ROC-AUC – SVC(probability=True) ist seit scikit-learn 1.9 veraltet. probability=False: reines SVC (schneller,
    z. B. für die Validierungskurve)."""
    clf = task != T_REG
    seed = int(cfg.seed)
    cw = "balanced" if cfg.class_weight and supports_class_weight(cfg, task) else None
    if cfg.model == MODEL_GB:
        # early_stopping=False: genau max_iter Runden – die passende Zahl zeigt die Validierungskurve in Schritt 7
        # (scikit-learn würde sonst ab 10 000 Zeilen selbst zufällig 10 % der Trainingsdaten zur Validierung abzweigen)
        kw = dict(max_iter=min(max(int(cfg.gb_iter), 1), MAX_TREES), learning_rate=float(cfg.gb_lr),
                  max_leaf_nodes=max(int(cfg.gb_leaves), 2), max_depth=_depth(cfg.gb_depth),
                  min_samples_leaf=max(int(cfg.gb_leaf), 1), early_stopping=False, random_state=seed)
        if clf:
            kw["class_weight"] = cw
        return (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(**kw)
    if cfg.model == MODEL_TREES:
        kw = dict(max_depth=_depth(cfg.max_depth), min_samples_leaf=max(int(cfg.min_samples_leaf), 1),
                  random_state=seed)
        if clf:
            kw["class_weight"] = cw
        if cfg.tree_variant == TREE_SINGLE:
            return (DecisionTreeClassifier if clf else DecisionTreeRegressor)(**kw)
        kw.update(n_estimators=min(max(int(cfg.n_estimators), 1), MAX_TREES), max_features=_max_features(cfg.max_features),
                  n_jobs=-1)
        return (RandomForestClassifier if clf else RandomForestRegressor)(**kw)
    if cfg.model == MODEL_SVM:
        if not clf:
            return SVR(kernel=cfg.kernel, C=float(cfg.svm_c), gamma=gamma_value(cfg.gamma), epsilon=float(cfg.epsilon))
        svc = SVC(kernel=cfg.kernel, C=float(cfg.svm_c), gamma=gamma_value(cfg.gamma), class_weight=cw)
        return CalibratedClassifierCV(svc, ensemble=False, cv=int(calib_cv)) if probability else svc
    if cfg.model == MODEL_LOGREG and clf:
        return LogisticRegression(C=float(cfg.logreg_c), max_iter=int(cfg.max_iter), class_weight=cw, random_state=seed)
    if cfg.model == MODEL_KNN:
        return (KNeighborsClassifier if clf else KNeighborsRegressor)(n_neighbors=int(cfg.n_neighbors),
                                                                     weights=cfg.weights, p=int(cfg.p), n_jobs=-1)
    if cfg.model == MODEL_LINREG and not clf:
        return LinearRegression(fit_intercept=bool(cfg.fit_intercept))
    raise ValueError(model_problem(cfg, task) or tx("Unbekanntes Modell: {m}", m=cfg.model))


def base_estimator(model):
    """Das eigentliche Modell – bei der SVM-Klassifikation steckt das SVC in CalibratedClassifierCV."""
    if isinstance(model, CalibratedClassifierCV):
        return model.calibrated_classifiers_[0].estimator
    return model


MODEL_PARAMS = {  # Parameter je Modell – für Anzeige und Rezept
    MODEL_GB: ("gb_iter", "gb_lr", "gb_leaves", "gb_depth", "gb_leaf"),
    MODEL_TREES: ("tree_variant", "n_estimators", "max_depth", "min_samples_leaf", "max_features"),
    MODEL_SVM: ("kernel", "svm_c", "gamma", "epsilon", "grid_search", "cv"),
    MODEL_LOGREG: ("logreg_c", "max_iter"),
    MODEL_KNN: ("n_neighbors", "weights", "p"),
    MODEL_LINREG: ("fit_intercept",),
}


def model_params(training: dict, task: str | None = None) -> dict:
    """Nur die Einstellungen, die für das gewählte Modell eine Rolle spielen (z. B. für Schritt 9)."""
    t = dict(training or {})
    m = t.get("model")
    keys = list(MODEL_PARAMS.get(m, ()))
    if m == MODEL_TREES and t.get("tree_variant") == TREE_SINGLE:
        keys = [k for k in keys if k not in ("n_estimators", "max_features")]
    if m == MODEL_SVM:
        if task is not None and task != T_REG:
            keys.remove("epsilon")
        if not t.get("grid_search"):
            keys.remove("cv")
    if task != T_REG and m in (MODEL_GB, MODEL_TREES, MODEL_SVM, MODEL_LOGREG):
        keys.append("class_weight")
    out = {"model": m}
    out.update({k: t[k] for k in keys if k in t})
    return out


def class_weights(y: np.ndarray) -> dict:
    classes = np.unique(y.astype(int))
    weights = compute_class_weight("balanced", classes=classes, y=y.astype(int))
    return {int(c): float(w) for c, w in zip(classes, weights)}


def _target(y, task: str) -> np.ndarray:
    """Label so, wie das Modell es beim Fit bekommt: Klassen als ganze Zahlen, Regression als float."""
    y = np.asarray(y)
    return y.astype("float64") if task == T_REG else y.astype(int)


def _to_original(p, task: str, recipe: dict) -> np.ndarray:
    """Regression mit skaliertem Label → zurück in Originaleinheiten; Klassen bleiben Codes."""
    p = np.asarray(p)
    if task != T_REG:
        return p.astype(int)
    p = p.astype("float64").reshape(-1)
    ls = recipe.get("label_scaler")
    return p * ls["std"] + ls["mean"] if ls else p


def _plain_predict(model, X, task: str, recipe: dict) -> np.ndarray:
    """model.predict – Klassen-Codes bzw. Werte in Originaleinheiten (für Fehlerrate und MAE in Schritt 7)."""
    return _to_original(model.predict(X), task, recipe)


def _error(task: str, y_true, y_pred) -> float:
    """Fehler für Schritt 7 (kleiner = besser): Fehlerrate = 1 − Accuracy bei Klassen, MAE bei Regression."""
    if task == T_REG:
        return float(mean_absolute_error(np.asarray(y_true, dtype="float64"), np.asarray(y_pred, dtype="float64")))
    return float(np.mean(np.asarray(y_true).astype(int) != np.asarray(y_pred).astype(int)))


def _baseline_error(task: str, y_fit, y_val) -> float:
    """Fehler einer naiven Vorhersage: immer die häufigste Klasse bzw. immer der Mittelwert der Trainingsdaten."""
    if task == T_REG:
        return float(np.mean(np.abs(np.asarray(y_val, dtype="float64") - float(np.mean(y_fit)))))
    majority = int(np.bincount(np.asarray(y_fit).astype(int)).argmax())
    return float(np.mean(np.asarray(y_val).astype(int) != majority))


def _sample_rows(n: int, cap: int, seed: int) -> np.ndarray:
    """Zeilennummern für eine Auswertung: alle oder eine zufällige Stichprobe (sortiert)."""
    if n <= cap:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def is_time_split(recipe: dict | None) -> bool:
    """Wurde in Schritt 4 zeitlich gesplittet? (Ältere Rezepte: immer zufällig.)"""
    return ((recipe or {}).get("split") or {}).get("mode") == SPLIT_TIME


def _holdout(y_codes, task: str, share: float, seed: int, time_order: bool = False):
    """Validierungsanteil aus den Trainingsdaten (Zeilennummern) – bei Klassen möglichst stratifiziert.
    time_order=True (zeitlicher Split): die Trainingszeilen sind nach der Zeit sortiert → Validierung = der jüngste
    Anteil, wie train_test_split(…, shuffle=False)."""
    n = len(y_codes)
    if time_order:
        return train_test_split(np.arange(n), test_size=share, shuffle=False)
    strat = None
    if task != T_REG:
        counts = np.bincount(np.asarray(y_codes).astype(int))
        counts = counts[counts > 0]
        n_val = int(np.ceil(share * n))
        if counts.min() >= 2 and n_val >= len(counts) and n - n_val >= len(counts):
            strat = y_codes
    return train_test_split(np.arange(n), test_size=share, random_state=seed, stratify=strat)


def forest_steps(n: int) -> list[int]:
    """Anzahl Bäume für die Kurve (Random Forest wächst mit warm_start Baum für Baum – wie Epochen; Gradient Boosting
    liefert mit staged_predict die Vorhersage nach jeder Runde)."""
    base = [1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000]
    n = min(max(int(n), 1), MAX_TREES)
    return [v for v in base if v < n] + [n]


def _knn_predict_all(dist, ind, y_fit, values, weights: str, task: str) -> dict:
    """Vorhersagen für alle K aus EINER Nachbarsuche – dieselben Regeln wie KNeighborsClassifier/-Regressor.predict
    (weights='distance': Nachbarn mit Abstand 0 zählen allein, wie in scikit-learn)."""
    n, kmax = ind.shape
    yn = np.asarray(y_fit)[ind]
    want, out = set(values), {}
    distance = weights == "distance"
    if distance:
        zero = dist == 0
        w = np.where(zero, 0.0, 1.0 / np.where(zero, 1.0, dist))
    if task == T_REG:
        s, sw, sz, cz = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for j in range(kmax):
            if distance:
                s += w[:, j] * yn[:, j]
                sw += w[:, j]
                sz += zero[:, j] * yn[:, j]
                cz += zero[:, j]
                if j + 1 in want:
                    out[j + 1] = np.where(cz > 0, sz / np.maximum(cz, 1), s / np.where(sw > 0, sw, 1))
            else:
                s += yn[:, j]
                if j + 1 in want:
                    out[j + 1] = s / (j + 1)
        return out
    rows = np.arange(n)
    labels = yn.astype(int)
    votes = np.zeros((n, int(np.max(y_fit)) + 1))
    zvotes = np.zeros_like(votes)
    for j in range(kmax):
        votes[rows, labels[:, j]] += w[:, j] if distance else 1.0
        if distance:
            zvotes[rows, labels[:, j]] += zero[:, j]
        if j + 1 in want:
            pred = votes.argmax(axis=1)
            if distance:
                pred = np.where(zvotes.sum(axis=1) > 0, zvotes.argmax(axis=1), pred)
            out[j + 1] = pred
    return out


def curve_value_text(curve: dict, v) -> str:
    """Wert der Kurve als Text – max_depth None = unbegrenzt (mit der tatsächlichen Tiefe)."""
    if v is None:
        return tx("None (unbegrenzt, Tiefe {d})", d=curve.get("depth", "?"))
    return _fmt_value(v) if not isinstance(v, float) else f"{v:g}"


def validation_curve(cfg: ModelConfig, prep: "PrepResult", progress=None):
    """Schritt 7: Fehler auf Trainings- und Validierungsdaten in Abhängigkeit vom wichtigsten Parameter – Anzahl
    Runden (Gradient Boosting), Anzahl Bäume (Random Forest), max_depth (Einzelbaum), C (SVM, Logistische
    Regression), K (KNN, Elbow-Methode wie im Kurs) bzw. Anzahl Trainingszeilen (Lernkurve der Linearen Regression).

    cfg.validation: "test" → Validierung mit X_test (wie im Kurs) · "split" → Anteil aus den Trainingsdaten (beim
    zeitlichen Split der jüngste Anteil).
    Rückgabe: (kurve, modell) – modell nur, wenn es bereits das fertige Modell auf allen Trainingsdaten ist."""
    r = prep.recipe
    task = r["task"]
    report = progress or (lambda *_: None)
    y_fit_all, _ = prep.fit_targets()
    yf_all = _target(y_fit_all, task)
    if cfg.validation == "split":
        tr, va = _holdout(yf_all, task, float(cfg.validation_split), int(cfg.seed), is_time_split(r))
        X_fit, yf, yo_fit = prep.X_train[tr], yf_all[tr], prep.y_train[tr]
        X_val, yo_val = prep.X_train[va], prep.y_train[va]
    else:
        X_fit, yf, yo_fit = prep.X_train, yf_all, prep.y_train
        X_val, yo_val = prep.X_test, prep.y_test
    seed = int(cfg.seed)
    cap = KNN_CURVE_ROWS if cfg.model == MODEL_KNN else CURVE_ROWS
    ev_fit = _sample_rows(len(X_fit), cap, seed)
    ev_val = _sample_rows(len(X_val), cap, seed + 1)
    curve = {"validation": cfg.validation, "n_fit": int(len(X_fit)), "n_val": int(len(X_val)),
             "metric": "mae" if task == T_REG else "error", "log_x": False, "train": [], "val": [],
             "baseline": _baseline_error(task, yo_fit, yo_val[ev_val])}

    def score(model, rows=None):
        rows = ev_fit if rows is None else rows
        return (_error(task, yo_fit[rows], _plain_predict(model, X_fit[rows], task, r)),
                _error(task, yo_val[ev_val], _plain_predict(model, X_val[ev_val], task, r)))

    def add(i, errors):
        curve["train"].append(errors[0])
        curve["val"].append(errors[1])
        v = curve_value_text(curve, curve["values"][i])
        report((i + 1) / len(curve["values"]), tx("Validierungskurve: {param} = {v} ({i}/{n})", param=curve["param"],
                                                  v=v, i=i + 1, n=len(curve["values"])), curve)

    finished = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*warm_start.*")  # class_weight + warm_start, gleiche Daten
        if cfg.model == MODEL_GB:
            # EIN Training; staged_predict liefert die Vorhersage nach jeder Runde – mehr Runden können hier (anders
            # als beim Random Forest) schaden: Ab einem Punkt lernt das Modell Rauschen, der Validierungsfehler steigt.
            values = forest_steps(cfg.gb_iter)
            curve.update(param="max_iter", label=N_("Anzahl Runden (max_iter)"), values=values, x=values,
                         chosen=values[-1])
            report(0.05, tx("Gradient Boosting: {n} Runden trainieren …", n=values[-1]), None)
            model = make_model(cfg, task).fit(X_fit, yf)
            want = {v: i for i, v in enumerate(values)}
            stages = zip(model.staged_predict(X_fit[ev_fit]), model.staged_predict(X_val[ev_val]))
            for k, (p_fit, p_val) in enumerate(stages, start=1):
                if k in want:
                    add(want[k], (_error(task, yo_fit[ev_fit], _to_original(p_fit, task, r)),
                                  _error(task, yo_val[ev_val], _to_original(p_val, task, r))))
            if cfg.validation != "split":
                finished = model  # schon das fertige Modell: alle Trainingsdaten, gleiche Einstellungen
        elif cfg.model == MODEL_TREES and cfg.tree_variant != TREE_SINGLE:
            values = forest_steps(cfg.n_estimators)
            curve.update(param="n_estimators", label=N_("Anzahl Bäume (n_estimators)"), values=values, x=values,
                         chosen=values[-1])
            model = make_model(cfg, task)
            model.set_params(warm_start=True)  # jeder Schritt fügt nur neue Bäume hinzu
            for i, n in enumerate(values):
                model.set_params(n_estimators=n)
                model.fit(X_fit, yf)
                add(i, score(model))
            model.set_params(warm_start=False)
            if cfg.validation != "split":
                finished = model  # schon das fertige Modell: alle Trainingsdaten, gleiche Bäume wie ein direktes fit
        elif cfg.model == MODEL_TREES:
            full = make_model(replace(cfg, max_depth=0), task).fit(X_fit, yf)
            depth = max(int(full.get_depth()), 1)
            chosen = _depth(cfg.max_depth)
            chosen = chosen if chosen is not None and chosen < depth else None
            values = [d for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100) if d < depth]
            if chosen is not None and chosen not in values:
                values = sorted(values + [chosen])
            values.append(None)
            curve.update(param="max_depth", label="max_depth", values=values, depth=depth, chosen=chosen,
                         x=[depth if d is None else d for d in values])
            for i, d in enumerate(values):
                add(i, score(full if d is None else make_model(replace(cfg, max_depth=d), task).fit(X_fit, yf)))
        elif cfg.model in (MODEL_SVM, MODEL_LOGREG):
            svm = cfg.model == MODEL_SVM
            chosen = float(cfg.svm_c if svm else cfg.logreg_c)
            base = (0.01, 0.1, 1.0, 10.0, 100.0) if svm else (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
            values = sorted(set(base) | {chosen})
            curve.update(param="C", label="C", values=values, x=values, log_x=True, chosen=chosen)
            for i, c in enumerate(values):
                m = make_model(replace(cfg, **({"svm_c": c} if svm else {"logreg_c": c})), task, probability=False)
                add(i, score(m.fit(X_fit, yf)))
        elif cfg.model == MODEL_KNN:
            n = len(X_fit)
            chosen = min(int(cfg.n_neighbors), n)  # Validierungsanteil: weniger Zeilen als im fertigen Modell
            values = sorted(set(range(1, min(KNN_MAX_K, n) + 1)) | {chosen})
            curve.update(param="n_neighbors", label="K (n_neighbors)", values=values, x=values, chosen=chosen)
            report(0.02, tx("Elbow-Methode: nächste Nachbarn suchen …"), None)
            nn = NearestNeighbors(n_neighbors=max(values), p=int(cfg.p), n_jobs=-1).fit(X_fit)
            pred_fit = _knn_predict_all(*nn.kneighbors(X_fit[ev_fit]), yf, values, cfg.weights, task)
            pred_val = _knn_predict_all(*nn.kneighbors(X_val[ev_val]), yf, values, cfg.weights, task)
            for i, k in enumerate(values):
                add(i, (_error(task, yo_fit[ev_fit], _to_original(pred_fit[k], task, r)),
                        _error(task, yo_val[ev_val], _to_original(pred_val[k], task, r))))
        else:  # Lineare Regression: kein Hyperparameter → Lernkurve über die Anzahl Trainingszeilen
            n = len(X_fit)
            order = np.random.default_rng(seed).permutation(n)
            values = sorted({max(2, int(round(f * n))) for f in np.linspace(0.1, 1.0, 10)})
            curve.update(param="n_rows", label=N_("Trainingszeilen"), values=values, x=values, chosen=values[-1])
            for i, s in enumerate(values):
                rows = order[:s]
                add(i, score(make_model(cfg, task).fit(X_fit[rows], yf[rows]), np.sort(rows[:cap])))
    # Anzahl Bäume und Lernkurve: kein „bester Wert“ (mehr Bäume / mehr Daten schaden nicht) – nur die Kurve zählt
    curve["best_index"] = None if curve["param"] in ("n_estimators", "n_rows") else int(np.argmin(curve["val"]))
    curve["chosen_x"] = curve["x"][curve["values"].index(curve["chosen"])]
    return curve, finished


def _train_counts(task: str, y_train=None, recipe: dict | None = None) -> np.ndarray:
    """Zeilen je Klasse (Klassifikation) bzw. [Anzahl Zeilen] (Regression) – aus den Trainings-Labels oder, für das
    exportierte Skript, aus dem Rezept."""
    if y_train is not None:
        y = np.asarray(y_train)
        return np.array([len(y)]) if task == T_REG else np.bincount(y.astype(int))
    n = (recipe.get("n_rows") or {}).get("train", 0)
    if task == T_REG:
        return np.array([n])
    shares = (recipe.get("label_stats") or {}).get("class_share", {})
    return np.array([round(s * n) for s in shares.values()])


def grid_cv(cfg: ModelConfig, task: str, y_train=None, recipe: dict | None = None) -> int:
    """Folds für GridSearchCV: wie eingestellt (2–10), aber bei Klassen höchstens so viele wie Zeilen der seltensten
    Klasse und bei Regression mindestens 2 Zeilen je Fold – R² ist für eine einzelne Zeile nicht definiert (sonst
    wären alle Bewertungen NaN und die „beste“ Kombination zufällig)."""
    cv = min(max(int(cfg.cv), 2), 10)
    counts = _train_counts(task, y_train, recipe)
    counts = counts[counts > 0]
    if len(counts):
        cv = min(cv, int(counts.min()) if task != T_REG else int(counts[0]) // 2)
    return max(2, cv)


def grid_search(cfg: ModelConfig, task: str, X: np.ndarray, y: np.ndarray, time_order: bool = False) -> dict:
    """GridSearchCV wie im Kurs: C × gamma (beim linearen Kernel nur C), Kreuzvalidierung auf den Trainingsdaten.
    time_order=True (zeitlicher Split): TimeSeriesSplit – jeder Fold validiert auf Zeilen, die nach seinen
    Trainingszeilen liegen."""
    clf = task != T_REG
    cw = "balanced" if cfg.class_weight and clf else None
    est = SVC(kernel=cfg.kernel, class_weight=cw) if clf else SVR(kernel=cfg.kernel, epsilon=float(cfg.epsilon))
    grid = {"C": list(GRID_C)} if cfg.kernel == "linear" else {"C": list(GRID_C), "gamma": list(GRID_GAMMA)}
    cv = grid_cv(cfg, task, y)
    gs = GridSearchCV(est, grid, cv=TimeSeriesSplit(n_splits=cv) if time_order else cv, refit=False,
                      return_train_score=True, n_jobs=-1)
    gs.fit(X, y)
    if not np.isfinite(gs.best_score_):
        if time_order:  # TimeSeriesSplit: der erste Fold lernt nur aus dem ältesten Teil – oft nur eine Klasse
            raise ValueError(tx("GridSearchCV: keine gültige Bewertung – beim zeitlichen Split lernen die ersten Folds "
                                "(TimeSeriesSplit) nur aus den ältesten Zeilen; dort fehlt vermutlich eine Klasse oder "
                                "es sind zu wenige Zeilen. cv verkleinern oder ohne GridSearchCV trainieren."))
        raise ValueError(tx("GridSearchCV: keine gültige Bewertung (zu wenige Zeilen je Fold) – cv verkleinern oder "
                            "ohne GridSearchCV trainieren."))
    res = gs.cv_results_
    rows = [{"C": float(p["C"]), "gamma": _py(p.get("gamma")), "val": float(v), "train": float(t)}
            for p, v, t in zip(res["params"], res["mean_test_score"], res["mean_train_score"])]
    return {"rows": rows, "best": {k: _py(v) for k, v in gs.best_params_.items()}, "best_score": float(gs.best_score_),
            "cv": int(cv), "scoring": "R²" if task == T_REG else "Accuracy", "grid": grid, "kernel": cfg.kernel,
            "time_order": bool(time_order)}


def _check_training(cfg: ModelConfig, task: str, y: np.ndarray) -> None:
    problem = model_problem(cfg, task)
    if problem:
        raise ValueError(problem)
    if cfg.model == MODEL_KNN and int(cfg.n_neighbors) > len(y):
        raise ValueError(tx("K = {k} ist größer als die Anzahl der Trainingszeilen ({n}).",
                            k=int(cfg.n_neighbors), n=len(y)))
    if cfg.model == MODEL_SVM and task != T_REG:
        counts = np.bincount(y.astype(int))
        if counts[counts > 0].min() < 2:
            raise ValueError(tx("SVM: Jede Klasse braucht mindestens 2 Trainingszeilen (für die Wahrscheinlichkeiten)."))


def train_model(cfg: ModelConfig, prep: "PrepResult", progress=None):
    """6) Modell trainieren: model.fit(X_train, y_train) – vorher optional GridSearchCV (SVM) und die
    Validierungskurve für Schritt 7.

    progress(anteil 0–1, text, kurve) meldet den Fortschritt (Balken und Live-Kurve in der App).
    Rückgabe: (model, hist, info) – hist = {"curve": …, "grid": …} für Schritt 7."""
    r = prep.recipe
    task = r["task"]
    report = progress or (lambda *_: None)
    y_fit, _ = prep.fit_targets()
    y = _target(y_fit, task)
    _check_training(cfg, task, y)
    t0 = time.time()
    hist, used = {}, cfg
    svm_clf = cfg.model == MODEL_SVM and task != T_REG
    calib = calibration_cv(y) if svm_clf else 5
    if cfg.model == MODEL_SVM and cfg.grid_search:
        n_comb = len(GRID_C) * (1 if cfg.kernel == "linear" else len(GRID_GAMMA))
        report(0.02, tx("GridSearchCV: {n} Kombinationen × {cv} Folds …", n=n_comb, cv=grid_cv(cfg, task, y)), None)
        hist["grid"] = grid_search(cfg, task, prep.X_train, y, is_time_split(r))
        best = hist["grid"]["best"]
        used = replace(cfg, svm_c=float(best["C"]), gamma=str(best.get("gamma", cfg.gamma)))
    model = None
    if cfg.curve:
        start = 0.35 if "grid" in hist else 0.0
        hist["curve"], model = validation_curve(used, prep, lambda f, t, c: report(start + (0.9 - start) * f, t, c))
    converged = True
    if model is None:
        report(0.92, tx("Trainiere das Modell auf allen Trainingsdaten ({n} Zeilen) …", n=_n(len(y))),
               hist.get("curve"))
        model = make_model(used, task, calib_cv=calib)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(prep.X_train, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    seed = int(cfg.seed)
    cap = KNN_CURVE_ROWS if cfg.model == MODEL_KNN else CURVE_ROWS
    rows_tr = _sample_rows(len(y), cap, seed)
    rows_te = _sample_rows(len(prep.y_test), cap, seed + 1)
    info = {
        "model": cfg.model,
        "seconds": round(time.time() - t0, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(y)),
        "metric": "mae" if task == T_REG else "error",
        "train_error": _error(task, prep.y_train[rows_tr], _plain_predict(model, prep.X_train[rows_tr], task, r)),
        "test_error": _error(task, prep.y_test[rows_te], _plain_predict(model, prep.X_test[rows_te], task, r)),
        "baseline_error": _baseline_error(task, prep.y_train, prep.y_test[rows_te]),
        "converged": converged,
        "calib_cv": calib if svm_clf else None,
        "best_params": (hist.get("grid") or {}).get("best"),
        "class_weight": class_weights(y) if cfg.class_weight and supports_class_weight(cfg, task) else None,
    }
    return model, hist, info


# --- Schritt 7: Einordnung Training vs. Validierung ------------------------------------------------------------
REMEDY_OVER = {
    MODEL_GB: N_("max_iter senken (bester Wert: Validierungskurve), learning_rate verkleinern oder min_samples_leaf "
                 "erhöhen"),
    TREE_FOREST: N_("max_depth begrenzen oder min_samples_leaf erhöhen"),
    TREE_SINGLE: N_("max_depth begrenzen, min_samples_leaf erhöhen oder Random Forest wählen"),
    MODEL_SVM: N_("C oder gamma verkleinern"),
    MODEL_LOGREG: N_("C verkleinern (stärkere Regularisierung)"),
    MODEL_KNN: N_("K erhöhen"),
    MODEL_LINREG: N_("weniger Features oder mehr Trainingsdaten"),
}
REMEDY_UNDER = {
    MODEL_GB: N_("max_iter oder learning_rate erhöhen, größere Bäume (max_leaf_nodes)"),
    TREE_FOREST: N_("max_depth erhöhen (0 = unbegrenzt) oder mehr Features je Split (max_features)"),
    TREE_SINGLE: N_("max_depth erhöhen oder Random Forest wählen"),
    MODEL_SVM: N_("C oder gamma vergrößern, RBF-Kernel versuchen"),
    MODEL_LOGREG: N_("C vergrößern oder ein nichtlineares Modell wählen (Random Forest)"),
    MODEL_KNN: N_("K verkleinern"),
    MODEL_LINREG: N_("ein nichtlineares Modell wählen (Random Forest) oder bessere Features bilden"),
}
TUNABLE = {"max_iter", "n_estimators", "max_depth", "C", "n_neighbors"}  # Schritt 7 → Schritt 6 übernehmen


def _remedy_key(cfg: ModelConfig) -> str:
    return cfg.tree_variant if cfg.model == MODEL_TREES else cfg.model


def format_error(task: str, e: float) -> str:
    """Fehlerrate in Prozent, MAE als Zahl."""
    return _nice(e) if task == T_REG else _pct(e)


def _memorizes(cfg: ModelConfig) -> str | None:
    """Modelle, deren Trainingsfehler konstruktionsbedingt nahe 0 liegt – dort ist der Abstand Training ↔ Validierung
    kein Hinweis auf Overfitting. Rückgabe: Begründung oder None."""
    if cfg.model == MODEL_TREES and cfg.tree_variant != TREE_SINGLE:
        return tx("Beim Random Forest lernt jeder Baum seine Stichprobe fast auswendig")
    if cfg.model == MODEL_KNN and cfg.weights == "distance":
        return tx("Mit weights='distance' ist jeder Trainingspunkt sein eigener nächster Nachbar")
    if cfg.model == MODEL_KNN and int(cfg.n_neighbors) == 1:
        return tx("Mit K = 1 ist jeder Trainingspunkt sein eigener nächster Nachbar")
    return None


def _noise(task: str, e: float, n: int) -> float:
    """Kleinster Unterschied, der mehr als Zufall ist: Standardfehler der Fehlerrate bzw. 2 % des MAE."""
    if task == T_REG:
        return 0.02 * e
    return max(0.003, float(np.sqrt(max(e * (1 - e), 1e-6) / max(n, 1))))


def curve_assessment(curve: dict | None, info: dict, cfg: ModelConfig, task: str) -> dict:
    """Schritt 7: Training vs. Validierung einordnen – Overfitting, Underfitting, Vergleich mit einer naiven Vorhersage
    und bester Parameterwert der Kurve. Rückgabe: {"train", "val", "baseline", "verdict", "notes", "suggest"} –
    suggest = {"param", "value"} oder None (value kann None sein: max_depth unbegrenzt)."""
    reg = task == T_REG
    if curve:
        i = curve["values"].index(curve["chosen"])
        e_tr, e_va, base = curve["train"][i], curve["val"][i], curve["baseline"]
        n_val = min(curve["n_val"], KNN_CURVE_ROWS if cfg.model == MODEL_KNN else CURVE_ROWS)
    else:
        e_tr, e_va, base = info["train_error"], info["test_error"], info["baseline_error"]
        n_val = 0
    f = lambda e: format_error(task, e)  # noqa: E731
    key = _remedy_key(cfg)
    notes, suggest = [], None
    gap = e_va - e_tr
    memo = _memorizes(cfg)
    if base > 0 and e_va >= 0.98 * base:
        verdict = "nosignal"
        notes.append(tx("**Nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber {b}, wenn man immer "
                        "{how} vorhersagt. Das Modell findet in diesen Features kaum ein Signal – Features prüfen "
                        "(Schritt 2–4) oder ein anderes Modell testen.",
                        v=f(e_va), b=f(base), how=tx("den Mittelwert") if reg else tx("die häufigste Klasse")))
    elif memo and e_tr <= 0.5 * e_va:  # nur, wenn der Trainingsfehler wirklich klein ist (z. B. nicht bei min_samples_leaf=50)
        verdict = "memorize"
        notes.append(tx("**Trainingsfehler {t} ist hier normal:** {why}. Aussagekräftig ist nur der Validierungsfehler "
                        "{v} – zum Vergleich: naive Vorhersage {b}.", t=f(e_tr), why=memo, v=f(e_va), b=f(base)))
    elif (gap > 0.05) if not reg else (e_tr < 0.7 * e_va):
        verdict = "overfit"
        notes.append(tx("**Overfitting:** Training {t}, Validierung {v} – das Modell lernt die Trainingsdaten "
                        "deutlich besser, als es auf neuen Daten vorhersagt. Gegenmittel: {fix}.",
                        t=f(e_tr), v=f(e_va), fix=tx(REMEDY_OVER[key])))
    elif ((e_va > 0.85 * base and gap < 0.03) if not reg else (e_va > 0.8 * base and e_tr > 0.8 * e_va)):
        verdict = "underfit"
        notes.append(tx("**Underfitting:** Training {t} und Validierung {v} sind ähnlich hoch und kaum besser als die "
                        "naive Vorhersage ({b}) – das Modell ist zu einfach für die Daten. Gegenmittel: {fix}.",
                        t=f(e_tr), v=f(e_va), b=f(base), fix=tx(REMEDY_UNDER[key])))
    else:
        verdict = "ok"
        notes.append(tx("Training {t} und Validierung {v} liegen nah beieinander und deutlich unter der naiven "
                        "Vorhersage ({b}) – keine Anzeichen für Overfitting.", t=f(e_tr), v=f(e_va), b=f(base)))
    if curve and curve["param"] == "n_rows":
        v = curve["val"]
        if len(v) >= 3 and v[-1] < v[-3] - (0.02 * v[-3] if reg else 0.005):
            notes.append(tx("Der Validierungsfehler sinkt bis zur letzten Datenmenge – mehr Trainingsdaten würden "
                            "vermutlich helfen."))
        else:
            notes.append(tx("Die Lernkurve ist am Ende flach – mehr Daten derselben Art bringen kaum noch etwas; "
                            "bessere Features oder ein anderes Modell schon eher."))
    elif curve and curve["param"] == "n_estimators":
        tol = _noise(task, curve["val"][-1], n_val)
        stable = next(v for v, e in zip(curve["values"], curve["val"]) if e <= curve["val"][-1] + tol)
        notes.append(tx("Ab etwa {n} Bäumen ändert sich der Validierungsfehler kaum noch ({e} bei {n2} Bäumen). Mehr "
                        "Bäume schaden nicht, kosten aber Rechenzeit und Speicher.",
                        n=stable, e=f(curve["val"][-1]), n2=curve["values"][-1]))
    elif curve:
        j = int(np.argmin(curve["val"]))
        best, e_best = curve["values"][j], curve["val"][j]
        tol = _noise(task, e_va, n_val)
        if best != curve["chosen"] and e_best < e_va - tol:
            notes.append(tx("**Bester Wert laut Validierungskurve:** {param} = {best} (Fehler {e}). Eingestellt ist "
                            "{chosen} (Fehler {e2}).",
                            param=curve["param"], best=curve_value_text(curve, best), e=f(e_best),
                            chosen=curve_value_text(curve, curve["chosen"]), e2=f(e_va)))
            suggest = {"param": curve["param"], "value": best} if curve["param"] in TUNABLE else None
        else:
            notes.append(tx("Der eingestellte Wert {param} = {chosen} liegt am Minimum der Validierungskurve (oder "
                            "nur knapp daneben).",
                            param=curve["param"], chosen=curve_value_text(curve, curve["chosen"])))
        v = curve["val"]
        if curve["param"] == "max_iter" and len(v) >= 3 and j == len(v) - 1 and v[-1] < v[-3] - tol:
            notes.append(tx("Der Validierungsfehler sinkt bis zur letzten Runde – mehr Runden (max_iter) oder eine "
                            "größere learning_rate könnten noch helfen."))
    return {"train": e_tr, "val": e_va, "baseline": base, "verdict": verdict, "notes": notes, "suggest": suggest}


# --- Schritt 8: Modell-Details (Koeffizienten, Wichtigkeit, Kennzahlen) ----------------------------------------
def unscaled_coefficients(coef, intercept: float, recipe: dict):
    """Koeffizienten eines linearen Modells auf skalierten Features → je 1 Einheit des ursprünglichen Features (und
    ggf. zurück in die Einheit des Labels). Rückgabe: (Koeffizienten, Achsenabschnitt)."""
    w = np.asarray(coef, dtype="float64").reshape(-1)
    b = float(intercept)
    sc = recipe["scaler"]
    if sc["type"] == "MinMaxScaler":  # x_s = x · scale + offset
        b += float(w @ np.asarray(sc["offset"]))
        w = w * np.asarray(sc["scale"])
    elif sc["type"] == "StandardScaler":  # x_s = (x − mean) / std
        w = w / np.asarray(sc["std"])
        b -= float(w @ np.asarray(sc["mean"]))
    ls = recipe.get("label_scaler")
    if ls:  # y = mean + std · y_s
        w, b = w * ls["std"], b * ls["std"] + ls["mean"]
    return w, b


def model_details(model, recipe: dict) -> dict:
    """Modell-spezifische Auswertung für Schritt 8.

    Rückgabe: {"kind": "importance" | "coef" | None, "table": DataFrame | None, "facts": {Name: Wert}}
    Bäume: feature_importances_ (Dummies einer Kategorie zusammengezählt) · Lineare/Logistische Regression:
    Koeffizienten · SVM und KNN: Kennzahlen des Modells."""
    est = base_estimator(model)
    cols = recipe["feature_columns"]
    out = {"kind": None, "table": None, "facts": {}}
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype="float64")
        rows = [{"Feature": f, "Wichtigkeit": float(imp[c].sum())} for f, c in recipe["feature_groups"].items() if c]
        out["kind"] = "importance"
        out["table"] = pd.DataFrame(rows).sort_values("Wichtigkeit", ascending=False).reset_index(drop=True)
        if hasattr(est, "estimators_"):
            depths = [t.get_depth() for t in est.estimators_]
            out["facts"] = {tx("Bäume"): len(est.estimators_), tx("Tiefe (Mittel)"): round(float(np.mean(depths)), 1),
                            tx("Blätter (Mittel)"): round(float(np.mean([t.get_n_leaves() for t in est.estimators_])))}
        else:
            out["facts"] = {tx("Tiefe"): int(est.get_depth()), tx("Blätter"): int(est.get_n_leaves())}
    elif isinstance(est, LinearRegression):
        w, b = unscaled_coefficients(est.coef_, est.intercept_, recipe)
        out["kind"] = "coef"
        out["table"] = pd.DataFrame({"Feature": cols, "Koeffizient": w,
                                     "Koeffizient (skaliert)": np.asarray(est.coef_, dtype="float64").reshape(-1)})
        out["table"] = out["table"].reindex(out["table"]["Koeffizient (skaliert)"].abs().sort_values(
            ascending=False).index).reset_index(drop=True)
        out["facts"] = {tx("Achsenabschnitt (intercept)"): _nice(b)}
    elif isinstance(est, LogisticRegression):
        coef = np.asarray(est.coef_, dtype="float64")
        out["kind"] = "coef"
        if coef.shape[0] == 1:  # binär: ein Koeffizient je Feature (für Klasse 1)
            table = pd.DataFrame({"Feature": cols, "Koeffizient (skaliert)": coef[0], "Odds-Faktor": np.exp(coef[0])})
            out["table"] = table.reindex(table["Koeffizient (skaliert)"].abs().sort_values(ascending=False).index
                                         ).reset_index(drop=True)
        else:
            table = pd.DataFrame(coef.T, columns=[str(c) for c in recipe["classes"]])
            table.insert(0, "Feature", cols)
            out["table"] = table.reindex(table.iloc[:, 1:].abs().max(axis=1).sort_values(ascending=False).index
                                         ).reset_index(drop=True)
        out["facts"] = {tx("Iterationen"): int(np.max(est.n_iter_))}
    elif isinstance(est, (SVC, SVR)):
        gamma = getattr(est, "_gamma", est.gamma)
        out["facts"] = {"kernel": est.kernel, "C": f"{est.C:g}",
                        "gamma": f"{gamma:.4g}" if isinstance(gamma, float) else str(gamma),
                        tx("Stützvektoren"): int(len(est.support_))}
        if isinstance(est, SVC):
            out["facts"][tx("Stützvektoren je Klasse")] = ", ".join(
                f"{recipe['classes'][int(c)]}: {int(k)}" for c, k in zip(est.classes_, est.n_support_))
    elif isinstance(est, (KNeighborsClassifier, KNeighborsRegressor)):
        out["facts"] = {"K (n_neighbors)": int(est.n_neighbors), "weights": est.weights,
                        tx("Abstand"): tx("euklidisch (p=2)") if int(est.p) == 2 else tx("Manhattan (p=1)") if int(est.p) == 1
                        else f"p={est.p}", tx("gespeicherte Trainingszeilen"): int(est.n_samples_fit_)}
    elif isinstance(est, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        # kein feature_importances_ – die Wichtigkeit zeigt die Permutation Importance in Schritt 8
        out["facts"] = {tx("Runden (n_iter_)"): int(est.n_iter_),
                        tx("Bäume"): int(est.n_iter_) * int(est.n_trees_per_iteration_),
                        "learning_rate": f"{est.learning_rate:g}", "max_leaf_nodes": str(est.max_leaf_nodes),
                        "max_depth": str(est.max_depth), "min_samples_leaf": int(est.min_samples_leaf)}
    return out


# ---------------------------------------------------------------------------
# 8) Evaluation
# ---------------------------------------------------------------------------
def predict_scores(model, X: np.ndarray, task: str, recipe: dict | None = None) -> np.ndarray:
    """Binär: Wahrscheinlichkeit für Klasse 1 · Mehrklassen: (n, K) · Regression: Werte
    (bei skaliertem Label automatisch in Originaleinheiten zurückgerechnet)."""
    if task == T_REG:
        return _to_original(model.predict(X), task, recipe or {})
    proba = np.asarray(model.predict_proba(X), dtype="float64")
    seen = np.asarray(model.classes_).astype(int)  # Klassen, die das Modell im Training gesehen hat
    k = len((recipe or {}).get("classes") or []) or int(seen.max()) + 1
    full = np.zeros((len(proba), max(k, int(seen.max()) + 1)))
    full[:, seen] = proba
    return full[:, 1] if task == T_BIN else full


def to_codes(scores: np.ndarray, task: str, threshold: float = 0.5) -> np.ndarray:
    """Wahrscheinlichkeiten → Klassen: binär über die Schwelle, Mehrklassen über die größte Wahrscheinlichkeit."""
    if task == T_BIN:
        return (scores > threshold).astype(int)
    if task == T_MULTI:
        return scores.argmax(axis=1)
    return scores


def evaluate(y_true: np.ndarray, scores: np.ndarray, task: str, classes: list | None = None,
             threshold: float = 0.5) -> dict:
    if task == T_REG:
        mse = mean_squared_error(y_true, scores)
        return {
            "task": task,
            "MAE": float(mean_absolute_error(y_true, scores)),
            "RMSE": float(np.sqrt(mse)),
            "MSE": float(mse),
            "R2": float(r2_score(y_true, scores)),
            "Explained Variance": float(explained_variance_score(y_true, scores)),
            "Mittelwert y": float(np.mean(y_true)),
        }
    y_true = y_true.astype(int)
    y_pred = to_codes(scores, task, threshold)
    labels = list(range(len(classes)))
    res = {
        "task": task,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "report": classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                        output_dict=True, zero_division=0),
        "report_text": classification_report(y_true, y_pred, labels=labels, target_names=classes, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
        "y_pred": y_pred,
        "auc": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if task == T_BIN:
                auc = roc_auc_score(y_true, scores)
            else:
                auc = roc_auc_score(y_true, scores, multi_class="ovr", labels=labels)
        res["auc"] = float(auc) if np.isfinite(auc) else None
    except ValueError:
        pass  # z. B. nur eine Klasse vorhanden
    # Güte der Wahrscheinlichkeiten (kleiner = besser) – entscheidend, wenn man mit ihnen rechnet (z. B. Quoten)
    res["log_loss"] = res["brier"] = None
    try:
        res["log_loss"] = float(log_loss(y_true, scores, labels=labels))
        if task == T_BIN:
            res["brier"] = float(brier_score_loss(y_true, scores, pos_label=1))
    except ValueError:
        pass
    return res


def metrics_summary(ev: dict) -> dict:
    """Kompakte Kennzahlen für preprocessing.json / Anzeige."""
    if ev["task"] == T_REG:
        return {k: round(ev[k], 4) for k in ("MAE", "RMSE", "R2", "Explained Variance")}
    out = {"accuracy": round(ev["accuracy"], 4), "f1_macro": round(ev["f1_macro"], 4)}
    if ev.get("auc") is not None:
        out["roc_auc"] = round(ev["auc"], 4)
    for k in ("log_loss", "brier"):
        if ev.get(k) is not None:
            out[k] = round(ev[k], 4)
    return out


def _point_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Log Loss je Zeile (binär) – mit derselben Begrenzung wie sklearn.metrics.log_loss."""
    eps = np.finfo(np.float64).eps
    p = np.clip(np.asarray(p, dtype="float64"), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare_probabilities(y_true: np.ndarray, p_model: np.ndarray, p_base: np.ndarray, prior: float) -> dict:
    """Schritt 8 (binär): Modell gegen eine Wahrscheinlichkeit aus den Daten (z. B. Elo) und gegen die naive
    Vorhersage (immer der Anteil der positiven Klasse im Training) – auf denselben Testzeilen (nur Zeilen, in denen die
    Vergleichsspalte einen Wert hat).

    Rückgabe: {"n", "rows": [{name, accuracy, log_loss, brier, auc}], "delta", "se", "verdict"} – delta = mittlerer
    Unterschied im Log Loss je Zeile (Modell − Vergleich, negativ = Modell besser), se = sein Standardfehler (gepaart:
    dieselben Spiele), verdict "better" | "worse" | "same" (Unterschied kleiner als 2 Standardfehler)."""
    y = np.asarray(y_true).astype(int)
    pm = np.asarray(p_model, dtype="float64")
    pb = np.asarray(p_base, dtype="float64")
    ok = np.isfinite(pb)
    y, pm, pb = y[ok], pm[ok], pb[ok]
    rows = []
    for name, p in ((N_("Modell"), pm), (N_("Vergleich"), pb), (N_("naive Vorhersage"), np.full(len(y), prior))):
        row = {"name": name, "accuracy": float(np.mean((p > 0.5).astype(int) == y)),
               "log_loss": float(np.mean(_point_log_loss(y, p))), "brier": float(np.mean((p - y) ** 2)), "auc": None}
        if len(np.unique(y)) == 2 and len(np.unique(p)) > 1:
            row["auc"] = float(roc_auc_score(y, p))
        rows.append(row)
    diff = _point_log_loss(y, pm) - _point_log_loss(y, pb)
    delta = float(np.mean(diff)) if len(diff) else float("nan")
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    verdict = "same"
    if np.isfinite(se) and abs(delta) > 2 * se:
        verdict = "better" if delta < 0 else "worse"
    return {"n": int(len(y)), "rows": rows, "delta": delta, "se": se, "verdict": verdict}


def permutation_importance(model, X: np.ndarray, y: np.ndarray, recipe: dict, n_rows: int = 2000,
                           n_repeats: int = 3, seed: int = RANDOM_STATE, threshold: float = 0.5) -> pd.DataFrame:
    """Wie wichtig ist jedes Feature? Werte eines Features mischen und messen, wie stark sich das
    Modell verschlechtert. Dummy-Spalten einer Kategorie werden gemeinsam gemischt."""
    task = recipe["task"]
    rng = np.random.default_rng(seed)
    if len(X) > n_rows:
        idx = rng.choice(len(X), n_rows, replace=False)
        X, y = X[idx], y[idx]

    def score(Xs):
        s = predict_scores(model, Xs, task, recipe)
        if task == T_BIN:
            try:
                return roc_auc_score(y.astype(int), s)
            except ValueError:
                return accuracy_score(y.astype(int), to_codes(s, task, threshold))
        if task == T_MULTI:
            return accuracy_score(y.astype(int), s.argmax(axis=1))
        return -mean_absolute_error(y, s)

    base = score(X)
    rows = []
    for feat, cols in recipe["feature_groups"].items():
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, cols] = X[rng.permutation(len(X))][:, cols]
            drops.append(base - score(Xp))
        rows.append({"Feature": feat, "Wichtigkeit": float(np.mean(drops)), "Streuung": float(np.std(drops))})
    metric = {T_BIN: "ROC-AUC", T_MULTI: "Accuracy", T_REG: "MAE"}[task]
    out = pd.DataFrame(rows).sort_values("Wichtigkeit", ascending=False).reset_index(drop=True)
    out.attrs["metric"] = metric
    return out


# ---------------------------------------------------------------------------
# 9) Speichern & Laden
# ---------------------------------------------------------------------------
MODEL_FILES = ("model.skops", "model.joblib")
BUNDLE_FILES = (*MODEL_FILES, "preprocessing.json")
MAX_BUNDLE_BYTES = 2 * 10**9  # entpackte Größe eines Modell-ZIPs
# skops lädt nur Typen, denen man ausdrücklich vertraut. scikit-learn-Bausteine unserer Modelle, die skops nicht
# von sich aus kennt – alles andere (z. B. fremder Code) wird abgelehnt.
SKOPS_TRUSTED = {"sklearn.tree._tree.Tree", "sklearn.neighbors._kd_tree.KDTree",
                 "sklearn.neighbors._ball_tree.BallTree", "sklearn.calibration._CalibratedClassifier",
                 "sklearn.calibration._SigmoidCalibration",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"}
SKOPS_TRUSTED_PREFIX = ("sklearn.metrics._dist_metrics.",)


def finalize_recipe(recipe: dict, cfg: ModelConfig | None = None, info: dict | None = None,
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


def save_bundle(model, recipe: dict, folder, fmt: str = ".skops") -> Path:
    """Modell + Vorverarbeitung speichern: <folder>/model.skops (oder .joblib) + preprocessing.json

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
            joblib.dump(model, tmp_model, compress=3)
        else:
            import skops.io as sio

            sio.dump(model, tmp_model)
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


def load_bundle(folder, allow_pickle: bool = True):
    """Gespeichertes Modell + Rezept laden. allow_pickle=False lehnt .joblib-Dateien ab (z. B. aus fremden ZIPs)."""
    folder = Path(folder)
    recipe = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    name = _bundle_model_name(recipe, {n for n in MODEL_FILES if (folder / n).is_file()})
    model_file = folder / name if name else None
    if model_file is None:
        raise ValueError(tx("Kein scikit-learn-Modell gefunden (model.skops oder model.joblib) – Modelle der "
                            "Deep-Learning-Version (.keras/.h5) lassen sich hier nicht laden."))
    return _load_model_file(model_file, allow_pickle), recipe


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
                      if info is not None and info.file_size <= 50 * 10**6 else {})
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
    """Gespeicherte Modelle im Ordner models/ auflisten (neueste zuerst) – nur scikit-learn-Modelle."""
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
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"name": sub.name, "path": str(sub), "label": r.get("label"), "task": r.get("task"),
                    "model": (r.get("training") or {}).get("model"), "saved_at": r.get("saved_at", ""),
                    "metrics": r.get("metrics", {})})
    return sorted(out, key=lambda b: b["saved_at"], reverse=True)


# ---------------------------------------------------------------------------
# 10) Vorhersagen auf neuen Daten
# ---------------------------------------------------------------------------
def required_columns(recipe: dict) -> pd.DataFrame:
    """Welche Spalten müssen neue Daten haben? (Rohspalte(n) der CSV → Feature im Modell)"""
    lin = recipe.get("lineage", {})
    rows = []
    for f in recipe["features"]:
        if f in recipe["numeric_features"]:
            i = recipe["feature_columns"].index(f)
            kind, values = tx("Zahl"), f"{recipe['train_min'][i]:g} … {recipe['train_max'][i]:g}"
        else:
            cats = recipe["categories"][f]
            kind = tx("Kategorie")
            values = ", ".join(cats[:8]) + (f" … (+{len(cats) - 8})" if len(cats) > 8 else "")
        fill = recipe["fills"].get(f, {}).get("value", recipe["fallback_fill"].get(f))
        if recipe["fills"].get(f, {}).get("strategy") == M_GROUP:
            fill = tx("Mittelwert je {group_col}", group_col=recipe['fills'][f]['group_col'])
        rows.append({tx("CSV-Spalte"): ", ".join(raw_sources(lin, f)), tx("Feature im Modell"): f, tx("Typ"): kind,
                     tx("Werte im Training"): values, tx("Füllwert bei Lücke"): _fmt_value(fill)})
    return pd.DataFrame(rows)


def required_raw_columns(recipe: dict) -> list[str]:
    """Alle CSV-Rohspalten, die neue Daten brauchen (in Reihenfolge der Features)."""
    lin = recipe.get("lineage", {})
    return list(dict.fromkeys(r for f in recipe["features"] for r in raw_sources(lin, f)))


def template_csv(recipe: dict) -> bytes:
    """Leere CSV-Vorlage mit genau den benötigten Rohspalten."""
    return (",".join(required_raw_columns(recipe)) + "\n").encode("utf-8")


def _normalized(d: pd.DataFrame, features: list[str], numeric: list[str]) -> pd.DataFrame:
    """Feature-Werte vor dem Füllen in einheitlicher Form (Zahl → float, Kategorie → Text ohne Rand-Leerzeichen)."""
    out = {}
    for c in features:
        s = d[c]
        if c in numeric:
            s = (s if is_numeric(s) else parse_numbers(s)).astype(float).replace([np.inf, -np.inf], np.nan)
        else:
            s = _cat(s).astype(object)
        out[c] = s
    return pd.DataFrame(out, index=d.index)


def row_fingerprints(d: pd.DataFrame, features: list[str], numeric: list[str]) -> np.ndarray:
    """Prüfsumme je Zeile über die Feature-Werte – erkennt Zeilen, die das Modell schon aus dem Training kennt."""
    return pd.util.hash_pandas_object(_normalized(d, features, numeric), index=False).to_numpy()


def prepare_new_data(df_new: pd.DataFrame, recipe: dict):
    """Neue Rohdaten exakt wie die Trainingsdaten aufbereiten (Schritt 2 + 4 + Skalierung).

    Rückgabe: (X skaliert, df nach Schritt 2, Bericht, y oder None)
    Wirft MissingColumnsError, wenn benötigte Spalten fehlen.
    """
    d, _, _, _ = apply_column_actions(df_new, recipe.get("column_actions", []), fit=False)
    d, _, _, _ = apply_new_columns(d, recipe.get("new_columns", []), fit=False)
    X_df, rep = transform_features(d, recipe)
    num_idx = [recipe["feature_columns"].index(c) for c in recipe["numeric_features"]]
    rep["row_out_of_range"] = np.zeros(len(X_df), dtype=bool)
    rep["out_of_range"] = {}
    if num_idx:
        vals = X_df.to_numpy()[:, num_idx]
        lo = np.asarray(recipe["train_min"])[num_idx]
        hi = np.asarray(recipe["train_max"])[num_idx]
        outside = (vals < lo) | (vals > hi)
        rep["out_of_range"] = {recipe["numeric_features"][j]: int(n) for j, n in enumerate(outside.sum(axis=0)) if n}
        rep["row_out_of_range"] = outside.any(axis=1)
    label_raw = recipe.get("label_source") or [recipe["label"]]
    label_raw = label_raw if isinstance(label_raw, list) else [label_raw]
    used_raw = set(required_raw_columns(recipe)) | set(label_raw) | set(recipe["features"]) | {recipe["label"]}
    rep["extra_columns"] = [c for c in df_new.columns if c not in used_raw]
    rep["n_required"] = len(required_raw_columns(recipe))
    # Kennt das Modell diese Zeilen schon? (Vergleich mit einer Stichprobe der Trainingszeilen)
    fps = recipe.get("train_fingerprints")
    if fps:
        new_fp = set(row_fingerprints(d, recipe["features"], recipe["numeric_features"]).tolist())
        hits = sum(1 for h in fps if h in new_fp)
        n_train = (recipe.get("n_rows") or {}).get("train", len(fps))
        est = hits / len(fps) * n_train
        rep["train_overlap"] = {"hits": hits, "sample": len(fps), "rows": int(round(min(est, len(d)))),
                                "share": float(min(est / max(len(d), 1), 1.0))}
    y = None
    if recipe["label"] in d.columns:
        y = encode_label(d[recipe["label"]], recipe)
        rep["label_unknown"] = int(np.isnan(y).sum() - d[recipe["label"]].isna().sum())
    return apply_scaler(X_df.to_numpy(), recipe["scaler"]), d, rep, y


def predict_new(model, recipe: dict, df_new: pd.DataFrame, threshold: float | None = None):
    """Vorhersage für neue Daten. Rückgabe: (Ergebnis-DataFrame, Bericht, Evaluation oder None)."""
    task = recipe["task"]
    thr = recipe.get("threshold", 0.5) if threshold is None else threshold
    if len(df_new) == 0:
        raise ValueError(tx("Die Datei enthält keine Datenzeilen (nur Kopfzeile?)."))
    X, d, rep, y = prepare_new_data(df_new, recipe)
    rep["y"] = y
    scores = predict_scores(model, X, task, recipe)
    res = df_new.copy()
    if task == T_BIN:
        neg, pos = recipe["classes"]
        res[f"P({pos})"] = np.round(scores, 4)
        res["Vorhersage"] = np.where(scores > thr, pos, neg)
    elif task == T_MULTI:
        classes = np.asarray(recipe["classes"])
        res["Vorhersage"] = classes[scores.argmax(axis=1)]
        res["Sicherheit"] = np.round(scores.max(axis=1), 4)
        for k, cl in enumerate(recipe["classes"]):
            res[f"P({cl})"] = np.round(scores[:, k], 4)
    else:
        res["Vorhersage"] = scores
    flags = [(rep["row_imputed"], N_("Lücken ergänzt")), (rep["row_unknown"], N_("unbekannte Kategorie")),
             (rep["row_out_of_range"], N_("außerhalb Trainingsbereich"))]
    notes = pd.Series("", index=range(len(res)), dtype=object)
    for mask, text in flags:
        if mask.any():
            notes[mask] = notes[mask].where(notes[mask] == "", notes[mask] + "; ") + text
    res["Hinweis"] = notes.to_numpy()
    ev = None
    if y is not None:
        mask = np.isfinite(y)
        if mask.sum() > 0:
            sub = scores[mask]
            ev = evaluate(y[mask], sub, task, recipe.get("classes"), thr)
            ev["n"] = int(mask.sum())
    return res, rep, ev


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
def split_column_name(columns) -> str:
    """Name der train/test-Spalte im ML-Export: 'split' – außer eine Spalte (z. B. ein Feature) heißt schon so."""
    taken = {str(c) for c in columns}
    for name in ("split", "split_train_test"):
        if name not in taken:
            return name
    i = 2
    while f"split_{i}" in taken:
        i += 1
    return f"split_{i}"


def ml_ready_frame(df_viz: pd.DataFrame, prep: "PrepResult", scaled: bool = False) -> pd.DataFrame:
    """Daten nach Schritt 4 als Tabelle: Füllwerte + Dummies (optional skaliert), Label kodiert, Spalte 'split'.
    Gleiche Zeilen wie im Training (ohne entfernte Zeilen), in der ursprünglichen Reihenfolge."""
    r = prep.recipe
    keep = df_viz.index.isin(prep.idx_train) | df_viz.index.isin(prep.idx_test)
    d = df_viz.loc[keep]
    X_df, _ = transform_features(d, r)
    if scaled:
        X_df = pd.DataFrame(apply_scaler(X_df.to_numpy(), r["scaler"]), columns=X_df.columns, index=X_df.index)
    out = X_df.astype("float64").round(6)
    label = r["label"]
    if r["task"] == T_REG:  # Originalwerte (float64), ungerundet
        out[label] = pd.to_numeric(d[label], errors="coerce").to_numpy(dtype="float64")
    else:  # Klassen als ganze Zahlen 0, 1, …
        out[label] = pd.Series(encode_label(d[label], r), index=d.index).round().astype("Int64")
    out[split_column_name(out.columns)] = np.where(d.index.isin(prep.idx_train), "train", "test")
    return out.reset_index(drop=True)


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
# 10) Fazit: Qualität der neuen Daten und Aussage des Ergebnisses
# ---------------------------------------------------------------------------
LEVEL_OK, LEVEL_WARN, LEVEL_BAD = N_("gut"), N_("mit Einschränkungen"), N_("kritisch")
_LEVEL_RANK = {LEVEL_OK: 0, LEVEL_WARN: 1, LEVEL_BAD: 2}


def _pct(x: float) -> str:
    return f"{x:.1%}"


def _n(x: int) -> str:
    """Ganze Zahl mit schmalem Leerzeichen als Tausendertrennung: 396030 → '396 030'."""
    return f"{int(x):,}".replace(",", "\u202f")


def _level(share: float, warn: float, bad: float) -> str:
    return LEVEL_BAD if share > bad else LEVEL_WARN if share > warn else LEVEL_OK


# ganze Sätze statt "{Strategie} der Trainingsdaten": in Englisch/Polnisch steht die Strategie dann klein
FILL_TEXT = {M_MEAN: N_("Mittelwert der Trainingsdaten ({v})"), M_MEDIAN: N_("Median der Trainingsdaten ({v})"),
             M_MODE: N_("häufigster Wert der Trainingsdaten ({v})")}


def _fill_text(recipe: dict, col: str) -> str:
    """Wie wird eine Lücke in dieser Spalte ergänzt? (für Menschen formuliert)"""
    f = recipe["fills"].get(col)
    fb = _fmt_value(recipe["fallback_fill"].get(col))
    if f and f["strategy"] == M_GROUP:
        return tx("Mittelwert je {group_col} aus den Trainingsdaten", group_col=f['group_col'])
    if f and f["strategy"] == M_CONST:
        return tx("fester Wert {v}", v=_fmt_value(f['value']))
    if f:
        v = _fmt_value(round(f['value'], 4) if isinstance(f['value'], float) else f['value'])
        if f["strategy"] in FILL_TEXT:
            return tx(FILL_TEXT[f["strategy"]], v=v)
        return tx("{what} der Trainingsdaten ({v})", what=f["strategy"], v=v)
    num = col in recipe["numeric_features"]
    if col in recipe.get("dropna_columns", []):
        return tx(N_("Median der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für "
                     "eine Vorhersage braucht aber jede Zeile einen Wert") if num else
                  N_("häufigster Wert der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier "
                     "entfernt; für eine Vorhersage braucht aber jede Zeile einen Wert"), fb=fb)
    return tx(N_("Median der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken") if num else
              N_("häufigster Wert der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken"), fb=fb)


def assess_prediction(recipe: dict, res: pd.DataFrame, rep: dict, ev: dict | None, threshold: float = 0.5,
                      y: np.ndarray | None = None) -> dict:
    """Fazit zu Schritt 10.

    Rückgabe: {"level": Gesamtstufe der Datenqualität,
               "quality": [{"level", "text", "details": [Text]}], "meaning": [Text], "advice": [Text]}"""
    task, n = recipe["task"], len(res)
    y = rep.get("y") if y is None else y
    quality, meaning, advice = [], [], []

    def add(level, text, details=None):
        quality.append({"level": level, "text": text, "details": details or []})

    # --- Datenqualität ------------------------------------------------------------------------------
    extra = rep.get("extra_columns") or []
    add(LEVEL_OK, tx("{n} Zeilen; alle {v} benötigten Spalten vorhanden",
                     n=_n(n), v=rep.get('n_required', len(recipe['features'])))
        + (tx(", {n} weitere Spalten werden nicht gebraucht (ignoriert).", n=len(extra)) if extra else "."))
    n_imp = int(np.sum(rep.get("row_imputed", [])))
    if n_imp:
        share = n_imp / n
        add(_level(share, 0.05, 0.2),
            tx("Lücken in {n} Zeilen ({pct}) mit Werten aus dem Training ergänzt. Ergänzte Werte sind Schätzungen "
               "– die Vorhersage dieser Zeilen ist etwas unsicherer.",
               n=_n(n_imp), pct=_pct(share)),
            [f"{c}: {_n(k)} × → {_fill_text(recipe, c)}" for c, k in rep["imputed"].items()])
    else:
        add(LEVEL_OK, tx("Keine Lücken in den benötigten Spalten."))
    n_unk = int(np.sum(rep.get("row_unknown", [])))
    if n_unk:
        share = n_unk / n
        add(_level(share, 0.01, 0.05),
            tx("{n} Zeilen ({pct}) mit Kategorien, die es im Training nicht gab – dort zählen alle Dummies dieser "
               "Spalte als 0 (das Modell kennt die Kategorie nicht).",
               n=_n(n_unk), pct=_pct(share)),
            [f"{c}: " + ", ".join(f"{k} ({v})" for k, v in list(vals.items())[:5]) for c, vals in rep["unknown"].items()])
    n_out = int(np.sum(rep.get("row_out_of_range", [])))
    if n_out:
        share = n_out / n
        add(_level(share, 0.01, 0.05),
            tx("{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – dort muss das Modell "
               "extrapolieren, die Vorhersage ist unsicherer.",
               n=_n(n_out), pct=_pct(share)),
            [tx("{c}: {n} Zeile", c=c, n=_n(k)) if k == 1 else tx("{c}: {n} Zeilen", c=c, n=_n(k))
             for c, k in rep["out_of_range"].items()])
    if rep.get("invalid_numeric"):
        k = sum(rep["invalid_numeric"].values())
        add(_level(k / n, 0.01, 0.05), tx("{n} Werte in Zahlenspalten waren keine Zahl und wurden wie Lücken "
                                          "behandelt.",
                                          n=_n(k)), [f"{c}: {v}" for c, v in rep["invalid_numeric"].items()])
    if rep.get("label_unknown"):
        add(LEVEL_WARN, tx("{n} Zeilen haben im Label einen Wert, den es im Training nicht gab – sie fehlen in der "
                           "Evaluation.",
                           n=_n(rep['label_unknown'])))
    level = max((q["level"] for q in quality), key=_LEVEL_RANK.get)

    # --- Überschneidung mit den Trainingsdaten -------------------------------------------------------
    ov = rep.get("train_overlap") or {}
    overlap = ov.get("share", 0.0)
    if overlap >= 0.05:
        meaning.append(tx("**Ca. {pct} der Zeilen ({n} von {n2}) kennt das Modell schon aus dem Training** "
                          "(gleiche Werte wie Trainingszeilen). Für diese Zeilen hat es die richtige Antwort beim "
                          "Lernen gesehen – Kennzahlen auf solchen Daten sind zu optimistisch.",
                          pct=_pct(overlap), n=_n(ov['rows']), n2=_n(n)))
        advice.append(tx("Für eine ehrliche Bewertung nur Daten verwenden, die nicht im Training waren – oder die "
                      "Kennzahlen aus Schritt 8 (Testdaten) heranziehen."))

    # --- Was sagt das Ergebnis? ----------------------------------------------------------------------
    test = recipe.get("metrics") or {}
    if task == T_BIN:
        neg, pos = recipe["classes"]
        p = res[f"P({pos})"].to_numpy(dtype=float)
        pred_share = float((res["Vorhersage"] == pos).mean())
        train_share = (recipe.get("label_stats") or {}).get("class_share", {}).get(pos)
        txt = tx("Vorhergesagt '{pos}': {pct} der Zeilen", pos=pos, pct=_pct(pred_share))
        true_share = None
        if y is not None and np.isfinite(y).any():
            true_share = float(np.mean(y[np.isfinite(y)]))
            txt += tx(" · tatsächlich in diesen Daten: {pct}", pct=_pct(true_share))
        if train_share is not None:
            txt += tx(" · in den Trainingsdaten: {pct}", pct=_pct(train_share))
        meaning.append(txt + ".")
        if ev is not None and true_share is not None:
            rep_pos = ev["report"].get(pos, {})
            recall, precision = rep_pos.get("recall", 0.0), rep_pos.get("precision", 0.0)
            gap = pred_share - true_share
            direction = tx("seltener") if gap < 0 else tx("häufiger")
            if abs(gap) >= 0.02:
                meaning.append(tx("Das Modell sagt '{pos}' {direction} voraus, als es tatsächlich vorkommt: Es "
                                  "erkennt {pct} der echten '{pos}' (Recall), und {pct2} seiner "
                                  "'{pos}'-Vorhersagen stimmen (Precision). Der Unterschied ist kein Fehler der "
                                  "App, sondern die Trefferquote des Modells bei der Schwelle {threshold:.2f}.",
                                  pos=pos, direction=direction, pct=_pct(recall), pct2=_pct(precision), threshold=threshold))
            if gap <= -0.02:
                advice.append(tx("Mehr '{pos}' erkennen: Schwelle senken (z. B. {v:.2f}) oder in Schritt 6 "
                                 "class_weight='balanced' – kostet mehr Fehlalarme.",
                                 pos=pos, v=max(threshold - 0.15, 0.05)))
            elif gap >= 0.02:
                advice.append(tx("Weniger Fehlalarme für '{pos}': Schwelle erhöhen (z. B. {v:.2f}).",
                                 pos=pos, v=min(threshold + 0.15, 0.95)))
        elif train_share is not None and abs(pred_share - train_share) > 0.15:
            meaning.append(tx("Der vorhergesagte Anteil weicht deutlich vom Training ab – die neuen Daten unterscheiden "
                           "sich womöglich vom Training (andere Zeit, andere Gruppe)."))
        unsure = float(np.mean(np.abs(p - threshold) < 0.1))
        meaning.append(tx("{pct} der Vorhersagen liegen nahe an der Schwelle (±0.1) und sind unsicher; Werte nahe "
                          "0 oder 1 sind sicher.",
                          pct=_pct(unsure)))
    elif task == T_MULTI:
        share_pred = res["Vorhersage"].value_counts(normalize=True)
        train = (recipe.get("label_stats") or {}).get("class_share", {})
        diff = max((abs(share_pred.get(c, 0.0) - train.get(c, 0.0)) for c in recipe["classes"]), default=0.0)
        meaning.append(tx("Durchschnittliche Sicherheit der Vorhersagen: {pct}; {pct2} der Zeilen unter 50 % "
                          "(unsicher).",
                          pct=_pct(float(res['Sicherheit'].mean())), pct2=_pct(float((res['Sicherheit'] < 0.5).mean()))))
        if diff > 0.15:
            meaning.append(tx("Die Verteilung der vorhergesagten Klassen weicht deutlich vom Training ab."))
    else:
        stats = recipe.get("label_stats") or {}
        pred = res["Vorhersage"].to_numpy(dtype=float)
        meaning.append(tx("Mittelwert der Vorhersagen: {x} (Training: {x2}).",
                          x=_nice(float(np.mean(pred))), x2=_nice(float(stats.get('mean', np.nan)))))

    # --- Vergleich mit dem Test aus dem Training ------------------------------------------------------
    if ev is not None and test:
        if task == T_REG and "MAE" in test:
            new, old = ev["MAE"], test["MAE"]
            rel = (new - old) / old if old else 0.0
            verdict = (tx("besser als") if rel < -0.1 else tx("schlechter als") if rel > 0.1 else tx("vergleichbar mit"))
            meaning.append(tx("Abweichung (MAE) hier {x} – {verdict} dem Test aus dem Training ({x2}).",
                              x=_nice(new), verdict=verdict, x2=_nice(old)))
        elif task != T_REG and "accuracy" in test:
            key, name = ("roc_auc", "ROC-AUC") if ev.get("auc") is not None and "roc_auc" in test else ("accuracy", "Accuracy")
            new = ev["auc"] if key == "roc_auc" else ev["accuracy"]
            old = test[key]
            if new - old > 0.02:
                why = (tx(" – vermutlich, weil das Modell einen Teil der Zeilen aus dem Training kennt") if overlap >= 0.05
                       else "")
                meaning.append(tx("{name} hier {new:.3f} – besser als im Test ({old:.3f}){why}.",
                                  name=name, new=new, old=old, why=why))
            elif new - old < -0.03 and overlap < 0.05:
                meaning.append(tx("{name} hier {new:.3f} – schlechter als im Test ({old:.3f}): Die neuen Daten "
                                  "unterscheiden sich vom Training (Datenqualität, andere Zeit/Gruppe).",
                                  name=name, new=new, old=old))
                advice.append(tx("Prüfen, ob sich die neuen Daten vom Training unterscheiden; ggf. mit aktuelleren "
                              "Daten neu trainieren."))
            elif new - old < -0.03:
                meaning.append(tx("{name} hier {new:.3f}, im Test {old:.3f}.", name=name, new=new, old=old))
            else:
                meaning.append(tx("{name} hier {new:.3f} – vergleichbar mit dem Test ({old:.3f}): Das Modell "
                                  "arbeitet auf diesen Daten so gut wie erwartet.",
                                  name=name, new=new, old=old))
    if level != LEVEL_OK:
        advice.append(tx("Datenqualität verbessern: Lücken, unbekannte Kategorien und Ausreißer in der Quelle "
                         "prüfen – betroffene Zeilen stehen in der Spalte 'Hinweis'."))
    return {"level": level, "quality": quality, "meaning": meaning, "advice": advice}

