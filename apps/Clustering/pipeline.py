"""
pipeline.py – Rechenlogik des DATA-Analysis-Clustering (ohne Streamlit)
=====================================================================

Alle Schritte einer Cluster-Analyse mit K-Means und der Anomalieerkennung mit Isolation Forest, Local Outlier
Factor (LOF), DBSCAN und einem Autoencoder (neuronales Netz, MLPRegressor) – scikit-learn, Unsupervised Learning – für
Tabellen (CSV) als wiederverwendbare Funktionen.
Die Streamlit-App (app.py) ruft nur diese Funktionen auf – das Modul lässt sich genauso im Jupyter Notebook
verwenden:

    import pipeline as pl
    df, _ = pl.read_csv("College_Data.csv")

Schritte
    1) Bibliotheken ............ Import-Block direkt unten
    2) Aufbereitung 1 .......... suggest_action(), apply_column_actions(), apply_new_columns()
    4) Aufbereitung 2 .......... suggest_missing(), prepare_cluster_data()   (kein Label, kein Split)
    5) Daten speichern ......... cluster_ready_frame(), clusters_frame(), anomalies_frame(), to_csv_bytes()
    6) Modell & Training ....... ClusterConfig, make_kmeans(), train_kmeans()
                                 Anomalien: make_anomaly_model(), train_anomaly(), anomaly_scores()
    7) Wahl von K .............. k_curve() (Elbow + Silhouette), k_assessment(), silhouette_values()
       Schwelle / eps .......... anomaly_curve() (Score-Kurve, k-Distanz), threshold_assessment()
    8) Cluster-Auswertung ...... cluster_profiles(), feature_separation(), pca_projection(), compare_groups()
       Anomalien ............... anomaly_profiles(), top_anomalies(), compare_anomalies()
    9) Speichern & Laden ....... save_bundle(), load_bundle()  (model.skops oder model.joblib)
   10) Neue Daten zuordnen ..... assign_new(), assess_assignment()  ·  Anomalien: score_new(), assess_scoring()

Unsupervised: Die Modelle bekommen keine Antwort (kein Label). Gibt es eine bekannte Einteilung (im Kurs: die
Spalte Private bei den Hochschulen, bei Betrugsdaten z. B. Class), wird sie nur NACH dem Training zum Vergleich
benutzt – nie als Feature.

Kernidee für Schritt 10: Alles, was in Schritt 2 und 4 aus den Daten gelernt wird (Füllwerte, Kategorien
bzw. Dummy-Spalten, Skalierung), steht in einem "Rezept" (dict). Das Rezept wird zusammen mit dem Modell
gespeichert und auf neue Daten unverändert angewendet – auf neuen Daten wird nichts neu gefittet, sie werden
nur dem nächsten Cluster-Zentrum zugeordnet bzw. mit dem Anomalie-Modell aus dem Training bewertet.
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

# Ergänzungen für das Template
import io
import json
import re
import tempfile
import time
import warnings
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    pairwise_distances,
    silhouette_samples,
)

from i18n import N_, tx  # Texte in der gewählten Sprache (Deutsch = Original); N_ markiert nur

TEMPLATE_VERSION = 1  # Rezept-Version der Clustering-App
RANDOM_STATE = 101  # Standard-Seed für K-Means, Stichproben und PCA

# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------
T_CLUSTER = "clustering"  # Aufgabe im Rezept – unterscheidet die Modelle von denen der Supervised-Version
SCALERS = ["StandardScaler", "MinMaxScaler", N_("keiner")]

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
A_CLIP = N_("Werte begrenzen (min, max)")
ACTIONS = [A_KEEP, A_DROP, A_NUM, A_YEAR, A_MONTH, A_YEAR_MONTH, A_FIRST, A_LAST, A_REPLACE, A_TOPN, A_CAT, A_CLIP]
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
    A_CLIP: (N_("min,max – z. B. 0,100 oder ,100"), N_("Zahlen auf einen Bereich begrenzen (clip): ',100' macht aus "
                                                    "Grad.Rate 118 den Wert 100 – unmögliche Werte und Ausreißer "
                                                    "verschieben die Cluster-Zentren")),
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
    """Wert → Text; ganze Zahlen ohne '.0' (1.0 → '1'), Kommazahlen ohne Rundungsrauschen
    (0.17886999999999997 → '0.17887')."""
    v = _py(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.12g}"
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


def parse_clip(param: str) -> tuple[float | None, float | None]:
    """'0,100' → (0, 100) · ',100' → (None, 100) · '0,' → (0, None) · mit Semikolon auch Dezimalkomma: '0,5;9,5'."""
    p = _clean(param)
    parts = p.split(";") if ";" in p else p.split(",")
    if len(parts) != 2:
        raise ValueError(tx("Werte begrenzen: Format min,max – z. B. 0,100 oder ,100 (nur Obergrenze)."))
    lo, hi = (None if not t.strip() else _to_number(t.strip()) for t in parts)
    if lo is None and hi is None:
        raise ValueError(tx("Werte begrenzen: mindestens eine Grenze angeben – z. B. ,100."))
    if any(v is not None and not np.isfinite(v) for v in (lo, hi)):
        raise ValueError(tx("Werte begrenzen: Grenzen müssen endliche Zahlen sein."))
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(tx("Werte begrenzen: min ({lo}) ist größer als max ({hi}).", lo=_fmt_value(lo), hi=_fmt_value(hi)))
    return lo, hi


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


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")


def md_escape(text) -> str:
    """Text aus Dateien (Spalten- und Cluster-Namen, Werte aus einem fremden Modell) wörtlich in Markdown zeigen:
    Sonderzeichen escapen, Zeilenumbrüche entfernen – sonst könnte z. B. ![](http://…) ein fremdes Bild nachladen."""
    return _MD_SPECIAL.sub(r"\\\1", " ".join(str(text).splitlines()))


_ID_NAME = re.compile(r"(?i)(id|nr|nummer|(^|[^a-z])(no|number|index))$")


def looks_like_id(s: pd.Series, name: str = "") -> bool:
    """Ganzzahlige Spalte, in der jede Zeile einen anderen Wert hat, und die nach einer ID aussieht (Name endet auf
    ID/Nr/Number/Index … oder Werte 1, 2, 3 … bzw. 0, 1, 2 …). Als Feature würde K-Means nach Zeilennummern
    gruppieren – als Bezeichnung der Zeilen ist sie dagegen nützlich."""
    if not is_numeric(s) or pd.api.types.is_bool_dtype(s) or len(s) < 3:
        return False
    nn = s.dropna()
    if len(nn) != len(s) or nn.nunique() != len(nn) or not _whole(nn.to_numpy()):
        return False
    if _ID_NAME.search(str(name).strip()):
        return True
    lo, hi = float(nn.min()), float(nn.max())
    return lo in (0.0, 1.0) and hi - lo + 1 == len(nn)


# Einstellungen aus einer importierten Konfiguration: Typ + erlaubter Bereich
MAX_K = 50  # höchstens so viele Cluster
SETTING_TYPES = {
    "w_k": (int, 2, MAX_K), "w_k_max": (int, 3, 30), "w_n_init": (int, 1, 100), "w_max_iter": (int, 10, 100_000),
    "w_seed": (int, 0, 1_000_000),
    # Anomalieerkennung (Isolation Forest, LOF, DBSCAN, Autoencoder) – contamination in Prozent
    "w_n_estimators": (int, 10, 2000), "w_max_samples": (int, 0, 10_000_000), "w_n_neighbors": (int, 2, 500),
    "w_min_samples": (int, 2, 1000), "w_eps": (float, 1e-6, 1e9), "w_contam_pct": (float, 0.1, 50.0),
    "w_drop_first": (bool, None, None), "w_curve": (bool, None, None), "w_contam_auto": (bool, None, None),
    # Autoencoder
    "w_ae_epochs": (int, 10, 5000), "w_ae_alpha": (float, 0.0, 10.0), "w_ae_lr": (float, 1e-5, 1.0),
    "w_ae_early": (bool, None, None),
    "w_viz_cols": (list, None, None), "w_features": (list, None, None),
    **{k: (str, None, None) for k in ("w_group", "w_name_col", "w_scaler", "w_init", "w_algorithm", "w_model",
                                      "w_metric", "w_ae_layers", "w_ae_activation")},
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
    Einträge anderer Versionen (z. B. w_label der Supervised-Version, w_epochs der Deep-Learning-Version) werden
    still übergangen – gemeinsame Teile (Schritt 2, fehlende Werte, w_features, w_scaler) werden übernommen."""
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
    if act == A_CLIP:
        lo, hi = parse_clip(param)
        x = s.astype(float) if is_numeric(s) else parse_numbers(s)
        changed = (int((x < lo).sum()) if lo is not None else 0) + (int((x > hi).sum()) if hi is not None else 0)
        span = f"[{'' if lo is None else _fmt_value(lo)}, {'' if hi is None else _fmt_value(hi)}]"
        return x.clip(lower=lo, upper=hi), tx("auf {span} begrenzt ({n} Werte geändert)", span=span, n=changed)
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
# 4) Datenaufbereitung 2 – Features, fehlende Werte, Dummies, Skalierung (kein Label, kein Split)
# ---------------------------------------------------------------------------
MIN_ROWS = 10  # weniger Zeilen ergeben keine sinnvollen Cluster
MAX_GROUPS = 30  # Vergleichsgruppe: höchstens so viele verschiedene Werte


def _natural_key(v) -> list:
    """'2 years' < '10+ years', 'A2' < 'A10' – Reihenfolge von Gruppen und Kategorien in Tabellen."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", str(v))]


def suggest_missing(df: pd.DataFrame, col: str, candidates: list[str] | None = None) -> tuple[str, str]:
    """Vorschlag für fehlende Werte: bei Zahlen Gruppen-Mittelwert über die am stärksten korrelierte,
    diskrete Spalte (z. B. mort_acc ↔ total_acc), sonst Median bzw. eigene Kategorie."""
    s = df[col]
    pct = s.isna().mean() * 100
    if pct > 40:
        return M_DROP_COL, ""
    if pct <= 1:
        return M_DROP_ROWS, ""
    if not is_numeric(s):
        return M_CONST, "fehlend"
    best, best_r = None, 0.0
    for c in candidates or []:
        if c == col or c not in df.columns or not is_numeric(df[c]):
            continue
        if df[c].isna().mean() > 0.01 or df[c].nunique() > 200:
            continue
        r = s.corr(df[c].astype(float))
        if pd.notna(r) and abs(r) > best_r:
            best, best_r = c, abs(r)
    if best is not None and best_r >= 0.3:
        return M_GROUP, best
    return M_MEDIAN, ""


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

    Wird für das Training UND für neue Daten benutzt. Unbekannte Kategorien werden zu 0 in allen
    Dummy-Spalten, fehlende Werte mit den Werten aus dem Training gefüllt.
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
    """Skalierer auf den Clustering-Daten fitten; Parameter als Liste speichern (JSON)."""
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


def invert_scaler(X, params: dict) -> np.ndarray:
    """Zurück in Originaleinheiten – gleiche Rechnung wie scaler.inverse_transform(), z. B. für Cluster-Zentren."""
    X = np.asarray(X, dtype="float64")
    if params["type"] == "MinMaxScaler":
        X = (X - np.asarray(params["offset"])) / np.asarray(params["scale"])
    elif params["type"] == "StandardScaler":
        X = X * np.asarray(params["std"]) + np.asarray(params["mean"])
    return X


def group_values(s: pd.Series) -> pd.Series:
    """Bekannte Einteilung (z. B. Private) als Text: Zahlen ohne '.0', Text ohne Rand-Leerzeichen, Lücke bleibt NaN."""
    return _cat(s)


@dataclass
class PrepResult:
    recipe: dict
    X: np.ndarray  # skaliert (float64) – darauf rechnet K-Means
    index: np.ndarray  # Zeilen-Index im df aus Schritt 2 – welche Zeilen ins Clustering gehen
    X_preview: pd.DataFrame  # erste Zeilen nach den Dummies, unskaliert
    groups: pd.Series | None = None  # bekannte Einteilung (nur zum Vergleich in Schritt 8), gleiche Reihenfolge wie X
    names: pd.Series | None = None  # Bezeichnung je Zeile (z. B. Name der Hochschule) – nur zur Anzeige
    log: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def prepare_cluster_data(
    df: pd.DataFrame,
    features: list[str],
    missing: dict | None = None,
    drop_first: bool = False,
    scaler: str = "StandardScaler",
    column_actions: list | None = None,
    lineage: dict | None = None,
    meta: dict | None = None,
    new_columns: list | None = None,
    group_col: str | None = None,
    name_col: str | None = None,
) -> PrepResult:
    """4) Features → Füllwerte, Kategorien/Dummies, Skalierung.

    Unsupervised: kein Label und kein Train/Test-Split – K-Means bekommt keine Antwort, die man vor ihm verstecken
    müsste, und lernt (wie im Kurs: kmeans.fit auf alle Daten) aus allen Zeilen. Füllwerte, Kategorien und Scaler
    stammen deshalb ebenfalls aus allen Zeilen und gelten unverändert für neue Daten (Schritt 10).

    group_col: bekannte Einteilung (im Kurs: Private) – nur für den Vergleich in Schritt 8, nie ein Feature.
    name_col:  Bezeichnung je Zeile (z. B. Name der Hochschule) – nur zur Anzeige.
    """
    missing = dict(missing or {})
    lineage = dict(lineage or {})
    log, warn = [], []
    group_col, name_col = group_col or None, name_col or None
    for c in (group_col, name_col):
        if c is not None and c not in df.columns:
            raise ValueError(tx("Spalte '{c}' ist nicht in den Daten.", c=c))
    special = {c for c in (group_col, name_col) if c}
    features = list(dict.fromkeys(features))
    taken = [f for f in features if f in special]
    if taken:
        features = [f for f in features if f not in special]
        warn.append(tx("{cols}: als Vergleichsgruppe bzw. Bezeichnung gewählt – nicht als Feature verwendet.",
                       cols=taken))
    unknown = [f for f in features if f not in df.columns]
    if unknown:
        raise ValueError(tx("Unbekannte Features: {unknown}", unknown=unknown))
    if not features:
        raise ValueError(tx("Bitte mindestens ein Feature auswählen."))

    d = df[features].copy()
    # ±inf in Zahlenspalten wie fehlende Werte behandeln (sonst Fehler beim Skalieren)
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
    if len(d) < MIN_ROWS:
        raise ValueError(tx("Nur noch {n} Zeilen übrig – zu wenig für ein Clustering (mindestens {m}).",
                            n=len(d), m=MIN_ROWS))

    numeric = [c for c in features if is_numeric(d[c])]
    categorical = [c for c in features if c not in numeric]
    for c in numeric:
        d[c] = d[c].astype(float)
    for c in categorical:
        d[c] = _cat(d[c])

    # c) Füllwerte aus allen Zeilen lernen (gespeichert im Rezept; gelten unverändert für neue Daten)
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
        col = d[c]
        if strat == M_MEAN:
            fills[c] = {"strategy": strat, "value": float(col.mean())}
        elif strat == M_MEDIAN:
            fills[c] = {"strategy": strat, "value": float(col.median())}
        elif strat == M_MODE:
            fills[c] = {"strategy": strat, "value": _py(col.mode().iloc[0])}
        elif strat == M_CONST:
            value = (_to_number(param) if param else 0.0) if c in numeric else (param or "fehlend")
            fills[c] = {"strategy": strat, "value": value}
        elif strat == M_GROUP:
            g = param
            if g not in features or g == c:
                raise ValueError(tx("Gruppen-Mittelwert für '{c}': Parameter muss eine andere Feature-Spalte sein "
                                    "(z. B. total_acc) – angegeben: '{g}'.",
                                    c=c, g=g))
            means = d.groupby(g)[c].mean().dropna()
            fills[c] = {
                "strategy": strat,
                "group_col": g,
                "group_means": [[_py(k), float(v)] for k, v in means.items()],
                "overall": float(col.mean()),
            }
        else:
            raise ValueError(tx("Unbekannte Strategie '{strat}' für '{c}'.", strat=strat, c=c))
        shown = fills[c].get("value", tx("Mittelwert je {v}", v=fills[c].get('group_col')))
        if isinstance(shown, float):
            shown = round(shown, 4)
        log.append(tx("{c}: {n_na} fehlende Werte → {strat} ({shown})", c=c, n_na=n_na, strat=tx(strat), shown=shown))

    # d) Fallback-Füllwerte für neue Daten (Median / häufigster Wert)
    fallback = {}
    for c in numeric:
        fallback[c] = float(d[c].median()) if d[c].notna().any() else 0.0
    for c in categorical:
        mode = d[c].mode()
        fallback[c] = str(mode.iloc[0]) if len(mode) else "fehlend"

    # e) Kategorien aus den gefüllten Zeilen ('fehlend' wird ggf. eigene Kategorie)
    filled_cat = _apply_fills(d[categorical].copy(), fills)
    categories = {c: sorted(filled_cat[c].dropna().astype(str).unique().tolist()) for c in categorical}
    del filled_cat
    for c in categorical:
        if len(categories[c]) > 100:
            warn.append(tx("'{c}' hat {n} Kategorien → sehr viele Dummy-Spalten. Tipp: Schritt 2 'Top-N behalten' "
                           "oder entfernen.",
                           c=c, n=len(categories[c])))

    recipe = {
        "template_version": TEMPLATE_VERSION,
        "app": "clustering",
        "created": datetime.now().isoformat(timespec="seconds"),
        "meta": meta or {},
        "column_actions": column_actions or [],
        "new_columns": new_columns or [],
        "lineage": lineage,
        "task": T_CLUSTER,
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
        "fit_on": "all",  # Füllwerte, Kategorien, Scaler: aus allen Zeilen (kein Split beim Clustering)
        "fills": fills,
        "fallback_fill": fallback,
        "drop_first": bool(drop_first),
        "group_col": None,  # gesetzt, wenn die Vergleichsgruppe brauchbar ist (2–30 Werte)
        "group_requested": group_col,
        "name_col": name_col,
    }

    # f) Dummies – exakt derselbe Code wie später für neue Daten
    X_df, _ = transform_features(d, recipe)
    if X_df.columns.duplicated().any():
        dupes = X_df.columns[X_df.columns.duplicated()].tolist()
        raise ValueError(tx("Doppelte Spaltennamen nach Dummy-Kodierung: {dupes} – bitte in Schritt 2 umbenennen.",
                            dupes=dupes))
    cols = list(X_df.columns)
    pos = {c: i for i, c in enumerate(cols)}
    groups_idx = {c: [pos[c]] for c in numeric}
    for c in categorical:
        used = categories[c][1:] if drop_first else categories[c]
        groups_idx[c] = [pos[f"{c}_{v}"] for v in used if f"{c}_{v}" in pos]
    recipe["feature_columns"] = cols
    recipe["feature_groups"] = groups_idx
    X_raw = X_df.to_numpy(dtype="float64")  # float64 wie im Notebook – scikit-learn rechnet damit genau gleich

    # g) Hinweise für K-Means: konstante Spalten und stark korrelierte Paare (Spannweite 0 ist exakt – die
    #    Standardabweichung wäre bei z. B. lauter 0.1 durch Rundung 2.8e-17 statt 0)
    const = [cols[j] for j in np.where(np.ptp(X_raw, axis=0) == 0)[0]]
    if const:
        warn.append(tx("Ohne Streuung (überall derselbe Wert): {const} – tragen nichts zum Clustering bei.",
                       const=const[:10]))
    if 2 <= len(numeric) <= 200:
        sample = d[numeric] if len(d) <= 50_000 else d[numeric].sample(50_000, random_state=0)
        corr = sample.corr().to_numpy()
        pairs = [(numeric[i], numeric[j], corr[i, j]) for i in range(len(numeric)) for j in range(i + 1, len(numeric))
                 if np.isfinite(corr[i, j]) and abs(corr[i, j]) >= 0.9]
        if pairs:
            pairs.sort(key=lambda t: -abs(t[2]))
            log.append(tx("Stark korreliert (|r| ≥ 0.9): {pairs} – solche Paare zählen im Abstand fast doppelt; wer "
                          "das nicht möchte, lässt eine Spalte des Paares weg.",
                          pairs=", ".join(f"{a} ~ {b} (r = {r:+.2f})" for a, b, r in pairs[:6])
                          + (" …" if len(pairs) > 6 else "")))

    # h) Skalierung – K-Means rechnet mit Abständen: ohne Skalierung bestimmen große Zahlen das Ergebnis
    recipe["scaler"] = fit_scaler(X_raw, scaler)
    recipe["train_min"] = X_raw.min(axis=0).tolist()
    recipe["train_max"] = X_raw.max(axis=0).tolist()
    recipe["n_rows"] = {"train": int(len(X_raw))}
    log.append(tx("Clustering-Daten: {n} Zeilen × {p} Spalten – Füllwerte, Kategorien und {scaler} aus allen Zeilen "
                  "gelernt (kein Train/Test-Split: K-Means bekommt keine Antwort)",
                  n=_n(len(X_raw)), p=len(cols), scaler=tx(recipe["scaler"]["type"])))

    # i) Bekannte Einteilung (nur Vergleich) und Bezeichnung (nur Anzeige)
    groups = None
    if group_col:
        g = group_values(df.loc[d.index, group_col])
        k = int(g.nunique())
        if not 2 <= k <= MAX_GROUPS:
            warn.append(tx("Vergleichsgruppe '{c}' hat {k} verschiedene Werte – für den Vergleich sind 2–{m} nötig; "
                           "sie wird nicht verwendet.",
                           c=group_col, k=k, m=MAX_GROUPS))
        else:
            groups = g.reset_index(drop=True)
            recipe["group_col"] = group_col
            recipe["group_kind"] = src_kind(df[group_col])  # für den erzeugten Code (Zahl als Text ohne '.0')
            recipe["group_source"] = raw_sources(lineage, group_col)
            recipe["group_values"] = sorted(g.dropna().unique().tolist(), key=_natural_key)
            if g.isna().any():
                log.append(tx("{n} Zeilen ohne Wert in '{c}' – sie werden geclustert, fehlen aber im Vergleich.",
                              n=_n(int(g.isna().sum())), c=group_col))
            same = [f for f in features if set(raw_sources(lineage, f)) & set(recipe["group_source"])]
            if same:
                warn.append(tx("Feature(s) {same} stammen aus derselben CSV-Spalte wie die Vergleichsgruppe "
                               "'{c}' – K-Means kennt die Antwort dann schon, der Vergleich in Schritt 8 wäre geschönt.",
                               same=same, c=group_col))
    names = None
    if name_col:
        names = _to_str(df.loc[d.index, name_col]).reset_index(drop=True)

    return PrepResult(
        recipe=recipe,
        X=apply_scaler(X_raw, recipe["scaler"]),
        index=d.index.to_numpy(),
        X_preview=X_df.head(10),
        groups=groups,
        names=names,
        log=log,
        warnings=warn,
    )


# ---------------------------------------------------------------------------
# 6) Modell & Training – K-Means (scikit-learn)
# ---------------------------------------------------------------------------
MODEL_KMEANS = "kmeans"
MODEL_IFOREST = "iforest"
MODEL_LOF = "lof"
MODEL_DBSCAN = "dbscan"
MODEL_AUTOENCODER = "autoencoder"
MODEL_NAMES = {MODEL_KMEANS: "K-Means", MODEL_IFOREST: "Isolation Forest", MODEL_LOF: "Local Outlier Factor (LOF)",
               MODEL_DBSCAN: "DBSCAN", MODEL_AUTOENCODER: "Autoencoder"}
ANOMALY_MODELS = (MODEL_IFOREST, MODEL_LOF, MODEL_DBSCAN, MODEL_AUTOENCODER)  # liefern je Zeile einen Anomalie-Score
MODEL_CLASSES = {MODEL_KMEANS: KMeans, MODEL_IFOREST: IsolationForest, MODEL_LOF: LocalOutlierFactor,
                 MODEL_DBSCAN: DBSCAN, MODEL_AUTOENCODER: MLPRegressor}
MODEL_VARS = {MODEL_KMEANS: "kmeans", MODEL_IFOREST: "iforest", MODEL_LOF: "lof", MODEL_DBSCAN: "dbscan",
              MODEL_AUTOENCODER: "autoencoder"}  # Variable im erzeugten Code
MODEL_WHY = {
    MODEL_KMEANS: N_("Gruppen (Cluster) ähnlicher Zeilen finden – K gibt man vor, Schritt 7 hilft bei der Wahl. "
                     "Ausreißer zeigt Schritt 10: Zeilen weit weg von allen Zentren."),
    MODEL_IFOREST: N_("Anomalien finden: Zufällige Schnitte isolieren jede Zeile – ungewöhnliche Zeilen sind nach "
                      "wenigen Schnitten allein. Schnell, auch bei vielen Zeilen und Features; der Standard für "
                      "Tabellen."),
    MODEL_LOF: N_("Anomalien finden: vergleicht die Dichte um eine Zeile mit der Dichte um ihre Nachbarn – findet "
                  "auch lokale Ausreißer neben dichten Gruppen. Bei sehr vielen Zeilen langsamer."),
    MODEL_DBSCAN: N_("Cluster beliebiger Form plus Rauschen: Zeilen in dünn besetzten Bereichen sind Rauschen = "
                     "Anomalien. Kein K nötig, dafür eps (Radius) und min_samples – eps zeigt Schritt 7 (k-Distanz)."),
    MODEL_AUTOENCODER: N_("Anomalien finden mit einem neuronalen Netz (Deep Learning, unsupervised): Es lernt, jede "
                          "Zeile durch einen Engpass hindurch nachzubauen – normale Muster gelingen, ungewöhnliche "
                          "Zeilen nicht. Score = Rekonstruktionsfehler; findet auch ungewöhnliche Kombinationen von "
                          "Features, die einzeln normal aussehen."),
}
METRICS = ["euclidean", "manhattan"]  # Abstandsmaß für LOF und DBSCAN
AE_ACTIVATIONS = ["tanh", "relu"]  # Autoencoder: Aktivierung der verborgenen Schichten (tanh: bei wenigen Neuronen meist besser)
AE_DEFAULT_CONTAMINATION = 0.01  # Autoencoder: keine eingebaute Schwelle – ohne Angabe die auffälligsten 1 %
AE_MAX_LAYERS, AE_MAX_UNITS = 9, 1024
# Autoencoder: Zeilen je Lernschritt (batch_size) = Zeilen/20, zwischen 8 und 200 → etwa 20 Lernschritte je Epoche.
# scikit-learn nimmt sonst min(200, Zeilen): bei 1 000 Zeilen 5 Schritte je Epoche, unter 200 Zeilen nur einen – in 300
# Epochen lernt das Netz dann kaum etwas (ae_batch_size). Ab 4 000 Zeilen gilt der Standard 200.
AE_STEPS_PER_EPOCH, AE_BATCH_MIN, AE_BATCH_MAX = 20, 8, 200
AE_EARLY_MIN_ROWS = 200  # Early Stopping erst ab 200 Zeilen: 10 % zur Validierung = mindestens 20 Zeilen
INITS = ["k-means++", "random"]
ALGORITHMS = ["lloyd", "elkan"]
K_MAX_DEFAULT = 10  # Elbow-/Silhouette-Kurve: K = 1 … 10
CURVE_ROWS = 20_000  # Kurve auf einer Stichprobe – K-Means wird dafür k_max-mal trainiert
SIL_ROWS = 5000  # Silhouette vergleicht alle Zeilen paarweise (n²) → Stichprobe, wie silhouette_score(sample_size=…)
MAX_CONTAMINATION = 0.5  # scikit-learn: contamination höchstens 0.5
CONTAM_LEVELS = [0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1]  # Schritt 7: Schwelle für diese Anteile auffälliger Zeilen
EPS_LEVELS = [0.5, 0.75, 0.9, 0.95, 0.975, 0.99]  # Schritt 7 (DBSCAN): eps = diese Quantile der k-Distanz
KNEE_TOP = 0.2  # Knick der Score-Kurve: in den auffälligsten 20 % der Zeilen gesucht
EPS_MIN = 1e-6  # kleinstes eps (Eingabe in Schritt 6): bei vielen genau gleichen Zeilen ist die k-Distanz oft 0
# DBSCAN hält alle Nachbarschaften (Abstand ≤ eps) gleichzeitig im Speicher – je Nachbarpaar etwa 15–35 Byte –, LOF je
# Zeile n_neighbors Abstände (etwa 50 Byte). Über dieser Zahl von Paaren droht ein Absturz (Speicher voll): Training
# mit Hinweis abgelehnt, in der eps-Tabelle von Schritt 7 übersprungen.
MAX_NEIGHBOR_PAIRS = 40_000_000
PAIR_SAMPLE = 400  # an so vielen zufälligen Zeilen wird die Zahl der Nachbarpaare für DBSCAN geschätzt


@dataclass
class ClusterConfig:
    model: str = MODEL_KMEANS
    n_clusters: int = 3
    init: str = "k-means++"
    n_init: int = 10  # Anzahl Starts mit verschiedenen Anfangszentren – der beste (kleinste Inertia) gewinnt
    max_iter: int = 300
    algorithm: str = "lloyd"
    seed: int = RANDOM_STATE
    curve: bool = True  # Kurve für Schritt 7: K-Means Elbow + Silhouette, DBSCAN Rauschanteil je eps
    k_max: int = K_MAX_DEFAULT
    # Anomalieerkennung
    contamination: float = 0.0  # Anteil auffälliger Zeilen (IF, LOF, Autoencoder): 0 = 'auto' (Autoencoder: 1 %)
    n_estimators: int = 100  # Isolation Forest: Anzahl Bäume
    max_samples: int = 0  # Isolation Forest: Zeilen je Baum, 0 = 'auto' = min(256, Zeilen)
    n_neighbors: int = 20  # LOF: Anzahl Nachbarn
    eps: float = 0.5  # DBSCAN: Radius der Nachbarschaft (in den skalierten Features)
    min_samples: int = 5  # DBSCAN: so viele Zeilen (sie selbst eingeschlossen) im Radius eps → Kernpunkt
    metric: str = "euclidean"  # LOF, DBSCAN
    # Autoencoder (MLPRegressor, Eingabe = Ziel)
    ae_layers: str = ""  # Neuronen je verborgener Schicht, z. B. "16,3,16" – leer = aus der Zahl der Features
    ae_activation: str = "tanh"
    ae_epochs: int = 300  # max_iter: höchstens so viele Durchläufe durch die Trainingsdaten
    ae_alpha: float = 1e-4  # L2-Strafe auf die Gewichte (wie MLPRegressor)
    ae_learning_rate: float = 1e-3  # learning_rate_init (Adam)
    ae_early_stopping: bool = True  # 10 % der Zeilen zur Validierung, Ende ohne Verbesserung


def model_name(key) -> str:
    """Anzeigename eines Modells – '?' für unbekannte Einträge (z. B. Modelle anderer Versionen)."""
    return MODEL_NAMES.get(key, "?")


def is_anomaly(key) -> bool:
    """Isolation Forest, LOF, DBSCAN und Autoencoder liefern je Zeile einen Anomalie-Score (K-Means: Cluster und
    Abstand)."""
    return key in ANOMALY_MODELS


def recipe_model(recipe: dict | None) -> str:
    """Modell eines Rezepts – ältere Rezepte ohne Eintrag stammen von K-Means."""
    return ((recipe or {}).get("training") or {}).get("model") or MODEL_KMEANS


def make_kmeans(cfg: ClusterConfig, n_clusters: int | None = None) -> KMeans:
    """K-Means wie im Kurs: KMeans(n_clusters=…) – n_init ausdrücklich (Standard 'auto' wäre 1 Start bei k-means++)."""
    return KMeans(n_clusters=int(n_clusters or cfg.n_clusters), init=cfg.init if cfg.init in INITS else "k-means++",
                  n_init=int(cfg.n_init), max_iter=int(cfg.max_iter),
                  algorithm=cfg.algorithm if cfg.algorithm in ALGORITHMS else "lloyd", random_state=int(cfg.seed))


def _safe_score(fn, X, labels) -> float:
    """Kennzahl, die mindestens 2 Cluster (und weniger Cluster als Zeilen) braucht – sonst NaN statt Fehler."""
    k = len(np.unique(labels))
    if not 2 <= k < len(labels):
        return float("nan")
    try:
        return float(fn(X, labels))
    except ValueError:
        return float("nan")


def silhouette(X: np.ndarray, labels: np.ndarray, seed: int = RANDOM_STATE) -> float:
    """Mittlerer Silhouette-Koeffizient (−1 … 1) – ab SIL_ROWS Zeilen auf einer festen Stichprobe, genau wie
    silhouette_score(X, labels, sample_size=5000, random_state=seed) im erzeugten Code."""
    big = len(X) > SIL_ROWS
    return _safe_score(lambda a, b: silhouette_score(a, b, sample_size=SIL_ROWS if big else None,
                                                     random_state=int(seed)), X, labels)


def silhouette_values(X: np.ndarray, labels: np.ndarray, seed: int = RANDOM_STATE) -> dict | None:
    """Silhouette je Zeile – dieselbe Stichprobe wie silhouette(): Grundlage für das Silhouetten-Diagramm."""
    n = len(X)
    idx = np.random.RandomState(int(seed)).permutation(n)[:SIL_ROWS] if n > SIL_ROWS else np.arange(n)
    lab = np.asarray(labels)[idx]
    if not 2 <= len(np.unique(lab)) < len(idx):
        return None
    vals = silhouette_samples(X[idx], lab)
    return {"values": vals, "labels": lab, "mean": float(vals.mean()), "n": int(len(idx)), "sampled": n > SIL_ROWS}


def curve_rows(n: int, seed: int = RANDOM_STATE) -> np.ndarray | None:
    """Zeilen für die Elbow-/Silhouette-Kurve: None = alle; sonst eine feste Stichprobe (gleich im erzeugten Code)."""
    return None if n <= CURVE_ROWS else np.random.default_rng(int(seed)).choice(n, CURVE_ROWS, replace=False)


def k_values(cfg: ClusterConfig, n: int) -> list[int]:
    """K-Werte der Kurve: 1 … k_max (höchstens n − 1) – das eingestellte K immer dabei."""
    hi = max(min(int(cfg.k_max), n - 1, MAX_K), 2)
    values = set(range(1, hi + 1))
    if int(cfg.n_clusters) < n:
        values.add(int(cfg.n_clusters))
    return sorted(values)


def elbow_k(values, inertia, min_ratio: float = 2.5) -> int | None:
    """Knick der Elbow-Kurve: das K, nach dem die Inertia am deutlichsten langsamer sinkt – Abnahme je K bis dahin
    geteilt durch die Abnahme je K danach, mindestens min_ratio-fach. Nur Stellen mit einer nennenswerten Abnahme
    davor (≥ 5 % der gesamten Abnahme) zählen; None, wenn die Kurve gleichmäßig sinkt (kein deutlicher Knick)."""
    x, y = np.asarray(values, dtype="float64"), np.asarray(inertia, dtype="float64")
    if len(x) < 3 or not np.isfinite(y).all() or y[0] <= y[-1]:
        return None
    total = y[0] - y[-1]
    best, best_ratio = None, min_ratio
    for i in range(1, len(x) - 1):
        before = (y[i - 1] - y[i]) / (x[i] - x[i - 1])
        after = (y[i] - y[i + 1]) / (x[i + 1] - x[i])
        if before < 0.05 * total:
            continue
        ratio = before / max(after, 0.01 * total)
        if ratio >= best_ratio:
            best, best_ratio = int(x[i]), ratio
    return best


def k_curve(cfg: ClusterConfig, X: np.ndarray, progress=None, main: KMeans | None = None) -> dict:
    """7) K-Means für K = 1 … k_max: Inertia (Elbow-Methode) und Silhouette, dazu Davies-Bouldin und
    Calinski-Harabasz. main = das bereits trainierte Modell (wird für das eingestellte K wiederverwendet)."""
    rows = curve_rows(len(X), cfg.seed)
    Xc = X if rows is None else X[rows]
    values = k_values(cfg, len(Xc))
    curve = {"values": [], "inertia": [], "silhouette": [], "davies_bouldin": [], "calinski_harabasz": [],
             "n_rows": int(len(Xc)), "sampled": rows is not None, "chosen": int(cfg.n_clusters)}
    for i, k in enumerate(values):
        if main is not None and rows is None and k == int(cfg.n_clusters):
            km = main  # gleiche Daten, gleiche Einstellungen → dasselbe Modell
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                km = make_kmeans(cfg, k).fit(Xc)
        labels = km.labels_
        curve["values"].append(int(k))
        curve["inertia"].append(float(km.inertia_))
        curve["silhouette"].append(silhouette(Xc, labels, cfg.seed) if k >= 2 else float("nan"))
        curve["davies_bouldin"].append(_safe_score(davies_bouldin_score, Xc, labels) if k >= 2 else float("nan"))
        curve["calinski_harabasz"].append(_safe_score(calinski_harabasz_score, Xc, labels) if k >= 2 else float("nan"))
        if progress:
            progress(0.15 + 0.85 * (i + 1) / len(values),
                     tx("Elbow- und Silhouette-Kurve: K = {k} ({i} von {n})", k=k, i=i + 1, n=len(values)), curve)
    curve["elbow"] = elbow_k(curve["values"], curve["inertia"])
    sil = [(s, k) for k, s in zip(curve["values"], curve["silhouette"]) if np.isfinite(s)]
    curve["best_silhouette"] = max(sil, key=lambda t: (t[0], -t[1]))[1] if sil else None
    return curve


def train_kmeans(cfg: ClusterConfig, prep: PrepResult, progress=None):
    """6) K-Means auf allen Zeilen trainieren (wie im Kurs: kmeans.fit(df.drop('Private', axis=1))).

    Rückgabe: (Modell, history mit Elbow-/Silhouette-Kurve und Abständen, info mit Kennzahlen)."""
    X = prep.X
    n, k = len(X), int(cfg.n_clusters)
    if not 2 <= k <= MAX_K:
        raise ValueError(tx("K muss zwischen 2 und {m} liegen.", m=MAX_K))
    if k >= n:
        raise ValueError(tx("K = {k} ist zu groß für {n} Zeilen.", k=k, n=n))
    t0 = time.time()
    if progress:
        progress(0.02, tx("Trainiere K-Means (K = {k}, n_init = {n}) …", k=k, n=int(cfg.n_init)), None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = make_kmeans(cfg).fit(X)
    notes = list(dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)))
    labels = model.labels_
    sizes = np.bincount(labels, minlength=k)
    dist = np.sqrt(((X - model.cluster_centers_[labels]) ** 2).sum(axis=1))  # Abstand zum eigenen Zentrum
    tss = float(((X - X.mean(axis=0)) ** 2).sum())
    info = {
        "model": MODEL_KMEANS, "k": k, "n_rows": int(n), "n_features": int(X.shape[1]),
        "inertia": float(model.inertia_), "n_iter": int(model.n_iter_),
        "converged": int(model.n_iter_) < int(cfg.max_iter),
        "r2": float(1 - model.inertia_ / tss) if tss > 0 else float("nan"),
        "silhouette": silhouette(X, labels, cfg.seed),
        "davies_bouldin": _safe_score(davies_bouldin_score, X, labels),
        "calinski_harabasz": _safe_score(calinski_harabasz_score, X, labels),
        "sizes": sizes.tolist(),
        "dist_median": [float(np.median(dist[labels == j])) if sizes[j] else 0.0 for j in range(k)],
        "dist_p99": [float(np.percentile(dist[labels == j], 99)) if sizes[j] else 0.0 for j in range(k)],
        "warnings": notes,
    }
    if progress:
        progress(0.15, tx("K-Means fertig – {n} Iterationen", n=info["n_iter"]), None)
    curve = k_curve(cfg, X, progress, main=model) if cfg.curve else None
    info["seconds"] = round(time.time() - t0, 1)
    info["trained_at"] = datetime.now().isoformat(timespec="seconds")
    return model, {"curve": curve, "dist": dist}, info


# ---------------------------------------------------------------------------
# 6b) Anomalieerkennung – Isolation Forest, Local Outlier Factor (LOF), DBSCAN, Autoencoder
# ---------------------------------------------------------------------------
def default_ae_layers(p: int) -> tuple[int, ...]:
    """Startwert für die Schichten des Autoencoders aus der Zahl der Features p: außen die nächste Zweierpotenz ≥ p
    (8 … 64 Neuronen), in der Mitte ein Engpass mit etwa p/4 Neuronen (2 … 8, immer weniger als p) – z. B.
    11 Features → (16, 3, 16), 30 Features → (32, 8, 32)."""
    p = max(int(p), 1)
    width = int(min(64, max(8, 2 ** int(np.ceil(np.log2(p))))))
    code = int(min(8, max(2, round(p / 4))))
    return width, (code if code < p else max(1, p - 1)), width


def layers_text(layers) -> str:
    """(16, 3, 16) → '16,3,16' (so steht es im Eingabefeld und in der Konfiguration)."""
    return ",".join(str(int(v)) for v in layers)


def parse_layers(text, p: int | None = None) -> tuple[int, ...]:
    """'16, 3, 16' → (16, 3, 16) – Neuronen je verborgener Schicht. Leer → default_ae_layers(p).
    Wirft ValueError mit verständlichem Text, wenn die Angabe nicht passt."""
    text = str(text or "").strip()
    if not text:
        return default_ae_layers(p or 8)
    parts = [t for t in re.split(r"[,;\s]+", text) if t]
    try:
        layers = tuple(int(t) for t in parts)
    except ValueError:
        raise ValueError(tx("Schichten: ganze Zahlen mit Komma getrennt, z. B. 16,3,16.")) from None
    if not 1 <= len(layers) <= AE_MAX_LAYERS or not all(1 <= v <= AE_MAX_UNITS for v in layers):
        raise ValueError(tx("Schichten: 1 bis {n} Schichten mit je 1 bis {m} Neuronen, z. B. 16,3,16.",
                            n=AE_MAX_LAYERS, m=AE_MAX_UNITS))
    return layers


def ae_batch_size(n: int) -> int:
    """Autoencoder: Zeilen je Lernschritt – Zeilen/20, zwischen 8 und 200, also etwa 20 Lernschritte je Epoche. Mit dem
    scikit-learn-Standard min(200, Zeilen) bekäme das Netz bei wenigen hundert Zeilen nur einen Schritt je Epoche und
    fände ungewöhnliche Kombinationen kaum (gemessen: Average Precision 0.1–0.7 statt 0.85–0.9). Ab 4 000 Zeilen = 200."""
    n = max(int(n), 1)
    return int(min(AE_BATCH_MAX, n, max(AE_BATCH_MIN, n // AE_STEPS_PER_EPOCH)))


def ae_early_stopping(cfg: ClusterConfig, n: int) -> bool:
    """Early Stopping nur, wenn gewählt und mindestens AE_EARLY_MIN_ROWS Zeilen da sind: Mit weniger wären die 10 %
    Validierungszeilen zu wenige – das R² darauf schwankt stark und beendet das Training zu früh, bei 10 Zeilen bricht
    scikit-learn ab ('validation set is too small')."""
    return bool(cfg.ae_early_stopping) and int(n) >= AE_EARLY_MIN_ROWS


def anomaly_params(cfg: ClusterConfig, n: int, p: int | None = None) -> dict:
    """Parameter des Anomalie-Modells – genau diese Werte stehen auch im erzeugten Code (codegen.anomaly_code).
    contamination 0 = 'auto'. Werte über der Zeilenzahl begrenzt scikit-learn selbst (mit Warnung) – hier ausdrücklich.
    p (Zahl der Features) braucht nur der Autoencoder, wenn keine Schichten angegeben sind."""
    if cfg.model == MODEL_AUTOENCODER:  # MLPRegressor mit Eingabe = Ziel; die Schwelle setzt train_anomaly
        act = cfg.ae_activation if cfg.ae_activation in AE_ACTIVATIONS else "tanh"
        q = {"hidden_layer_sizes": parse_layers(cfg.ae_layers, p), "activation": act}
        if float(cfg.ae_alpha) != 1e-4:
            q["alpha"] = float(cfg.ae_alpha)
        if float(cfg.ae_learning_rate) != 1e-3:
            q["learning_rate_init"] = float(cfg.ae_learning_rate)
        if ae_batch_size(n) < AE_BATCH_MAX:  # 200 = Standard ('auto' = min(200, Zeilen)) – dann nicht nötig
            q["batch_size"] = ae_batch_size(n)
        q.update(max_iter=int(cfg.ae_epochs), early_stopping=ae_early_stopping(cfg, n), random_state=int(cfg.seed))
        return q
    contamination = float(cfg.contamination) if cfg.contamination and cfg.contamination > 0 else "auto"
    metric = cfg.metric if cfg.metric in METRICS else "euclidean"
    if cfg.model == MODEL_IFOREST:
        p = {}
        if int(cfg.n_estimators) != 100:
            p["n_estimators"] = int(cfg.n_estimators)
        if int(cfg.max_samples) > 0:
            p["max_samples"] = min(int(cfg.max_samples), int(n))
        p["contamination"] = contamination
        p["random_state"] = int(cfg.seed)
        return p
    if cfg.model == MODEL_LOF:
        p = {"n_neighbors": max(1, min(int(cfg.n_neighbors), int(n) - 1)), "contamination": contamination}
        if metric != "euclidean":
            p["metric"] = metric
        p["novelty"] = True  # das Modell bewertet später auch neue Daten (Schritt 10)
        return p
    if cfg.model == MODEL_DBSCAN:
        p = {"eps": float(cfg.eps), "min_samples": int(cfg.min_samples)}
        if metric != "euclidean":
            p["metric"] = metric
        return p
    raise ValueError(tx("Unbekanntes Modell: {model}", model=cfg.model))


def make_anomaly_model(cfg: ClusterConfig, n: int, p: int | None = None):
    """IsolationForest(…), LocalOutlierFactor(…, novelty=True), DBSCAN(…) oder MLPRegressor(…) (Autoencoder) mit den
    Einstellungen aus Schritt 6."""
    if not is_anomaly(cfg.model):
        raise ValueError(tx("Unbekanntes Modell: {model}", model=cfg.model))
    return MODEL_CLASSES[cfg.model](**anomaly_params(cfg, n, p))


def reconstruction(model, X: np.ndarray) -> np.ndarray:
    """Autoencoder: Eingabe minus Nachbau je Zeile und Feature (in den skalierten Features) – was das Netz nicht
    nachbauen konnte."""
    X = np.asarray(X, dtype="float64")
    return X - model.predict(X)


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """Autoencoder-Score: Rekonstruktionsfehler je Zeile als RMSE über die Features – bei StandardScaler in
    Standardabweichungen (um 0.3 gut nachgebaut, über 1 deutlich daneben)."""
    return np.sqrt(np.mean(reconstruction(model, X) ** 2, axis=1))


AE_ACT_FUNCS = {"tanh": np.tanh, "relu": lambda v: np.maximum(v, 0.0), "identity": lambda v: v,
                "logistic": lambda v: 1.0 / (1.0 + np.exp(-v))}


def code_layer(model) -> int:
    """Index der engsten verborgenen Schicht (Engpass) des Autoencoders."""
    sizes = model.hidden_layer_sizes
    sizes = [int(sizes)] if np.isscalar(sizes) else [int(v) for v in sizes]
    return int(np.argmin(sizes))


def autoencoder_codes(model, X: np.ndarray) -> np.ndarray:
    """Ausgabe des Engpasses je Zeile – die gelernte Verdichtung der Features (Encoder-Teil des Netzes): dieselbe
    Rechnung wie in MLPRegressor, Schicht für Schicht Gewichte · Eingabe + Bias und die Aktivierung."""
    act = AE_ACT_FUNCS[model.activation]
    a = np.asarray(X, dtype="float64")
    k = code_layer(model)
    for w, b in zip(model.coefs_[:k + 1], model.intercepts_[:k + 1]):
        a = act(a @ w + b)
    return a


def core_distance(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """DBSCAN: Abstand jeder Zeile zum nächsten Kernpunkt (components_) und dessen Nummer. Ohne Kernpunkt (eps zu
    klein) ist der Abstand unendlich."""
    core = np.asarray(model.components_)
    if not len(core):
        return np.full(len(X), np.inf), np.zeros(len(X), dtype=int)
    dist, idx = NearestNeighbors(n_neighbors=1, metric=model.metric).fit(core).kneighbors(X)
    return dist[:, 0], idx[:, 0]


def anomaly_scores(model, X: np.ndarray, key: str, training: bool = False):
    """Anomalie-Score je Zeile (höher = auffälliger), Anomalie ja/nein und – nur bei DBSCAN – der Cluster (−1 = Rauschen).

    Isolation Forest: Score = −score_samples (0 … 1; um 0,5 normal, nahe 1 sehr auffällig), Anomalie = predict == −1.
    LOF: Score = LOF-Wert (≈ 1 normal, deutlich über 1 auffällig). Trainingszeilen: negative_outlier_factor_ aus dem
         Training (jede Zeile ohne sich selbst als Nachbarn); neue Daten: score_samples und predict (novelty=True).
    DBSCAN: Score = Abstand zum nächsten Kernpunkt in Vielfachen von eps (über 1 = Rauschen). Trainingszeilen
         behalten die Cluster aus dem Training (labels_); eine neue Zeile bekommt den Cluster des nächsten
         Kernpunkts, wenn er höchstens eps entfernt ist – sonst −1 (Rauschen = Anomalie).
    Autoencoder: Score = Rekonstruktionsfehler (RMSE je Zeile), Anomalie = Score über threshold_ aus dem Training."""
    X = np.asarray(X, dtype="float64")
    if key == MODEL_AUTOENCODER:
        score = reconstruction_error(model, X)
        return score, score > float(model.threshold_), None
    if key == MODEL_IFOREST:
        return -model.score_samples(X), model.predict(X) == -1, None
    if key == MODEL_LOF:
        if training:
            nof = np.asarray(model.negative_outlier_factor_)
            return -nof, nof < model.offset_, None
        return -model.score_samples(X), model.predict(X) == -1, None
    if key == MODEL_DBSCAN:
        dist, nearest = core_distance(model, X)
        if training:
            labels = np.asarray(model.labels_).astype(int)
        else:
            core_labels = np.asarray(model.labels_)[model.core_sample_indices_].astype(int)
            labels = (np.where(dist <= model.eps, core_labels[nearest], -1) if len(core_labels)
                      else np.full(len(X), -1))
        return dist / float(model.eps), labels == -1, labels
    raise ValueError(tx("Unbekanntes Modell: {model}", model=key))


def anomaly_threshold(model, key: str) -> float:
    """Schwelle auf der Skala des Scores – auffällig ist ein Score darüber. IF/LOF: −offset_ (contamination = auto:
    0,5 bzw. 1,5; sonst das passende Quantil der Trainings-Scores), DBSCAN: 1 (= eps), Autoencoder: threshold_
    (Quantil der Rekonstruktionsfehler im Training)."""
    if key == MODEL_AUTOENCODER:
        return float(model.threshold_)
    return 1.0 if key == MODEL_DBSCAN else float(-model.offset_)


def knee_index(y) -> int | None:
    """Knick einer sortierten Kurve (vereinfachtes Kneedle-Verfahren): beide Achsen auf 0 … 1 normiert, dann die Stelle
    mit dem größten Abstand zur Geraden vom ersten zum letzten Punkt. None, wenn die Kurve fast gerade verläuft."""
    y = np.asarray(y, dtype="float64")
    if len(y) < 5 or not np.isfinite(y).all() or y[-1] == y[0]:
        return None
    x = np.linspace(0.0, 1.0, len(y))
    gap = np.abs((y - y[0]) / (y[-1] - y[0]) - x)
    i = int(np.argmax(gap))
    return i if gap[i] >= 0.1 else None


def suggest_share(n_top: int, n: int) -> float:
    """Vorschlag für contamination aus der Anzahl auffälliger Zeilen: 2 signifikante Stellen, 0,1 % … 50 %."""
    share = float(f"{n_top / max(int(n), 1):.2g}")
    return min(max(share, 0.001), MAX_CONTAMINATION)


def anomaly_curve(cfg: ClusterConfig, X: np.ndarray, score: np.ndarray, progress=None) -> dict:
    """7) Daten für die Wahl der Schwelle bzw. von eps.

    Isolation Forest / LOF: Schwelle je Anteil auffälliger Zeilen (CONTAM_LEVELS) und der Knick der absteigend
    sortierten Score-Kurve in den auffälligsten 20 % – dahinter flacht sie ab (Vorschlag für contamination).
    DBSCAN: k-Distanz = Abstand jeder Zeile zu ihrem min_samples-ten Nachbarn (sie selbst zählt mit, wie bei DBSCAN):
    Kernpunkt ⇔ k-Distanz ≤ eps. Der Knick der aufsteigend sortierten k-Distanz ist ein guter Kandidat für eps. Die
    Nachbarn kommen aus allen Zeilen, ab CURVE_ROWS Zeilen wird nur für eine Stichprobe gemessen. Mit cfg.curve (und
    höchstens CURVE_ROWS Zeilen) zusätzlich Cluster und Rauschanteil für eps = Quantile der k-Distanz."""
    n = len(X)
    score = np.asarray(score, dtype="float64")
    if cfg.model in (MODEL_IFOREST, MODEL_LOF, MODEL_AUTOENCODER):
        thresholds = np.quantile(score, [1 - c for c in CONTAM_LEVELS])
        top = max(int(np.ceil(KNEE_TOP * n)), min(n, 20))
        desc = np.sort(score)[::-1][:top]
        knee = knee_index(desc)
        if knee is not None and knee + 1 > int(np.ceil(KNEE_TOP * n)):  # unter 100 Zeilen ist die Kurve länger als
            knee = None  # 20 % – ein Knick dahinter wäre kein Vorschlag für Anomalien (bis zu 50 % der Zeilen)
        return {"kind": "threshold", "levels": list(CONTAM_LEVELS), "thresholds": thresholds.tolist(),
                "counts": [int(np.sum(score > t)) for t in thresholds], "sorted": desc, "knee": knee,
                "suggest": None if knee is None else suggest_share(knee + 1, n)}
    if cfg.model != MODEL_DBSCAN:
        raise ValueError(tx("Unbekanntes Modell: {model}", model=cfg.model))
    params = anomaly_params(cfg, n)
    k = max(1, min(int(cfg.min_samples), n))
    rows = curve_rows(n, cfg.seed)
    query = X if rows is None else X[rows]
    neighbors = NearestNeighbors(n_neighbors=k, **{key: v for key, v in params.items() if key == "metric"}).fit(X)
    kdist = np.sort(neighbors.kneighbors(query)[0][:, -1])
    knee = knee_index(kdist)
    out = {"kind": "kdist", "kdist": kdist, "k": k, "n_rows": int(len(query)), "sampled": rows is not None,
           "knee": knee,  # Vorschlag mindestens EPS_MIN: bei vielen gleichen Zeilen liegt der Knick oft bei 0
           "suggest": None if knee is None else float(f"{max(float(kdist[knee]), EPS_MIN):.4g}"),
           "core_share": float(np.mean(kdist <= float(cfg.eps))),
           "zero_share": float(np.mean(kdist <= EPS_MIN)),  # Zeilen mit min_samples − 1 genau gleichen Zeilen
           "grid": None}
    if cfg.curve and n <= CURVE_ROWS:
        values = sorted({float(f"{v:.4g}") for v in np.quantile(kdist, EPS_LEVELS) if v > 0} | {float(cfg.eps)})
        pairs = neighbor_pairs(X, values, params.get("metric", "euclidean"), cfg.seed)
        grid = {"eps": [], "clusters": [], "noise": [],  # zu große eps (Speicher) übersprungen – immer die größten
                "skipped": [e for e, m in zip(values, pairs) if m > MAX_NEIGHBOR_PAIRS]}
        values = [e for e, m in zip(values, pairs) if m <= MAX_NEIGHBOR_PAIRS]
        for i, e in enumerate(values):
            labels = DBSCAN(**{**params, "eps": e}).fit(X).labels_
            grid["eps"].append(e)
            grid["clusters"].append(int(labels.max()) + 1 if (labels >= 0).any() else 0)
            grid["noise"].append(float(np.mean(labels == -1)))
            if progress:
                progress(0.4 + 0.6 * (i + 1) / len(values),
                         tx("Rauschanteil je eps: eps = {e} ({i} von {n})", e=f"{e:.4g}", i=i + 1, n=len(values)),
                         None)
        out["grid"] = grid
    return out


def suggest_eps(X: np.ndarray, min_samples: int = 5, metric: str = "euclidean", seed: int = RANDOM_STATE,
                max_rows: int = 5000) -> float:
    """Startwert für eps (Schritt 6): Knick der k-Distanz auf einer Stichprobe von höchstens max_rows Zeilen – ohne
    Knick das 90-%-Quantil. Auf einer Stichprobe liegen die Nachbarn weiter auseinander als in allen Daten, der
    Startwert ist also eher groß; Schritt 7 misst genauer (Nachbarn aus allen Zeilen)."""
    X = np.asarray(X, dtype="float64")
    n = len(X)
    sample = X if n <= max_rows else X[np.random.default_rng(int(seed)).choice(n, max_rows, replace=False)]
    k = max(1, min(int(min_samples), len(sample)))
    kw = {"metric": metric} if metric in METRICS and metric != "euclidean" else {}
    kdist = np.sort(NearestNeighbors(n_neighbors=k, **kw).fit(sample).kneighbors(sample)[0][:, -1])
    i = knee_index(kdist)
    value = kdist[i] if i is not None else np.quantile(kdist, 0.9)
    return float(f"{max(float(value), EPS_MIN):.4g}")


def neighbor_pairs(X: np.ndarray, eps, metric: str = "euclidean", seed: int = RANDOM_STATE):
    """DBSCAN: Wie viele Nachbarpaare (Abstand ≤ eps, jede Zeile zählt sich selbst mit) hält DBSCAN.fit gleichzeitig
    im Speicher? Geschätzt aus PAIR_SAMPLE zufälligen Zeilen: mittlere Zahl der Nachbarn × Zeilen. Solange selbst
    'jede Zeile mit jeder' (n²) unter MAX_NEIGHBOR_PAIRS bleibt, wird nichts gemessen – Rückgabe dann n².
    eps: eine Zahl oder eine Liste (gleiche Stichprobe für alle → wächst mit eps, nie umgekehrt)."""
    X = np.asarray(X, dtype="float64")
    values = np.atleast_1d(np.asarray(eps, dtype="float64"))
    n = len(X)
    if n * n <= MAX_NEIGHBOR_PAIRS:
        counts = np.full(len(values), float(n * n))
    else:
        rows = np.random.default_rng(int(seed)).choice(n, min(PAIR_SAMPLE, n), replace=False)
        step = max(1, 2_000_000 // n)  # höchstens etwa 2 Mio. Abstände (16 MB) auf einmal
        hits = np.zeros(len(values))
        for i in range(0, len(rows), step):
            d = pairwise_distances(X[rows[i:i + step]], X, metric=metric if metric in METRICS else "euclidean")
            hits += [np.count_nonzero(d <= e) for e in values]
        counts = hits / len(rows) * n
    return float(counts[0]) if np.ndim(eps) == 0 else counts.tolist()


def size_problem(cfg: ClusterConfig, X: np.ndarray) -> str | None:
    """Sprengt das Training den Speicher? DBSCAN: geschätzte Nachbarpaare bei eps (neighbor_pairs), LOF: Zeilen ×
    n_neighbors. Über MAX_NEIGHBOR_PAIRS eine Meldung mit Ausweg statt eines Absturzes – sonst None."""
    n = len(X)
    if cfg.model == MODEL_DBSCAN:
        pairs = neighbor_pairs(X, float(cfg.eps), cfg.metric, cfg.seed)
        if pairs > MAX_NEIGHBOR_PAIRS:
            return tx("DBSCAN: Bei eps = {eps} liegen im Mittel {m} Zeilen im Umkreis jeder Zeile – zusammen etwa {p} "
                      "Nachbarpaare, die DBSCAN gleichzeitig im Speicher hält (Grenze: {lim}). eps verkleinern (Knick "
                      "der k-Distanz in Schritt 7) oder eine Stichprobe nehmen (Seitenleiste). Viele genau gleiche "
                      "Zeilen (z. B. nur kategoriale Features) wirken wie ein großes eps.",
                      eps=f"{float(cfg.eps):.4g}", m=_n(round(pairs / max(n, 1))), p=_n(pairs),
                      lim=_n(MAX_NEIGHBOR_PAIRS))
    elif cfg.model == MODEL_LOF:
        k = anomaly_params(cfg, n)["n_neighbors"]
        if n * k > MAX_NEIGHBOR_PAIRS:
            return tx("LOF: {n} Zeilen × n_neighbors {k} = {p} Abstände im Speicher (Grenze: {lim}). n_neighbors "
                      "senken oder eine Stichprobe nehmen (Seitenleiste).",
                      n=_n(n), k=k, p=_n(n * k), lim=_n(MAX_NEIGHBOR_PAIRS))
    return None


def train_anomaly(cfg: ClusterConfig, prep: PrepResult, progress=None):
    """6) Isolation Forest, LOF, DBSCAN oder Autoencoder auf allen Zeilen trainieren – unsupervised, ohne Label.

    Autoencoder: MLPRegressor mit Eingabe = Ziel (fit(X, X)); Schwelle = Quantil der Rekonstruktionsfehler für
    contamination (ohne Angabe 1 %), gespeichert als model.threshold_. Leere Schichten und contamination 0 werden dabei
    in cfg eingetragen – so stehen sie im Rezept und im angezeigten Code.

    Rückgabe: (Modell, history mit Score, Anomalie ja/nein, DBSCAN-Clustern und den Daten für Schritt 7, info)."""
    X = prep.X
    n, p = X.shape
    if not is_anomaly(cfg.model):
        raise ValueError(tx("Unbekanntes Modell: {model}", model=cfg.model))
    if cfg.contamination and not 0 < float(cfg.contamination) <= MAX_CONTAMINATION:
        raise ValueError(tx("contamination muss zwischen 0 und 0.5 liegen (Anteil auffälliger Zeilen, z. B. 0.01 = 1 %)."))
    if cfg.model == MODEL_AUTOENCODER:
        cfg.ae_layers = layers_text(parse_layers(cfg.ae_layers, p))
        cfg.contamination = float(cfg.contamination or AE_DEFAULT_CONTAMINATION)
    if cfg.model == MODEL_DBSCAN and not (np.isfinite(cfg.eps) and cfg.eps > 0):
        raise ValueError(tx("eps muss größer als 0 sein."))
    if not np.ptp(X, axis=0).any():  # Isolation Forest meldete hier sonst ALLE Zeilen als auffällig
        raise ValueError(tx("Alle Zeilen sind nach der Vorbereitung gleich – es gibt nichts Auffälliges zu finden. In "
                            "Schritt 4 Features wählen, die sich zwischen den Zeilen unterscheiden."))
    problem = size_problem(cfg, X)
    if problem:
        raise ValueError(problem)
    t0 = time.time()
    if progress:
        progress(0.05, tx("Trainiere {name} …", name=model_name(cfg.model)), None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if cfg.model == MODEL_AUTOENCODER:  # Eingabe = Ziel: das Netz lernt, die Zeilen nachzubauen
            model = make_anomaly_model(cfg, n, p).fit(X, X)
            model.threshold_ = float(np.percentile(reconstruction_error(model, X), 100 * (1 - cfg.contamination)))
        else:
            model = make_anomaly_model(cfg, n, p).fit(X)
        score, flag, labels = anomaly_scores(model, X, cfg.model, training=True)
    notes = list(dict.fromkeys(str(w.message) for w in caught
                               if issubclass(w.category, (UserWarning, RuntimeWarning, ConvergenceWarning))))
    if cfg.model == MODEL_AUTOENCODER:  # „Maximum iterations reached“ erklärt autoencoder_notes verständlicher
        notes = [w for w in notes if not w.startswith("Stochastic Optimizer: Maximum iterations")]
    finite = score[np.isfinite(score)]
    info = {
        "model": cfg.model, "n_rows": int(n), "n_features": int(p), "n_anomalies": int(flag.sum()),
        "share": float(flag.mean()), "threshold": anomaly_threshold(model, cfg.model),
        "contamination": float(cfg.contamination or 0.0),
        "score_median": float(np.median(finite)) if len(finite) else float("nan"),
        "score_p99": float(np.percentile(finite, 99)) if len(finite) else float("nan"),
        "score_max": float(finite.max()) if len(finite) else float("nan"),
        "warnings": notes,
    }
    if cfg.model == MODEL_DBSCAN:
        k = int(labels.max()) + 1 if (labels >= 0).any() else 0
        info.update(k=k, sizes=np.bincount(labels[labels >= 0], minlength=k).tolist() if k else [],
                    n_core=int(len(model.core_sample_indices_)), eps=float(model.eps),
                    min_samples=int(model.min_samples))
    if cfg.model == MODEL_AUTOENCODER:
        best = getattr(model, "best_validation_score_", None)
        info.update(layers=[int(v) for v in model.hidden_layer_sizes], code_size=int(min(model.hidden_layer_sizes)),
                    n_iter=int(model.n_iter_), loss=float(model.loss_), epochs=int(cfg.ae_epochs),
                    batch_size=ae_batch_size(n), early_stopping=bool(model.early_stopping),
                    early_requested=bool(cfg.ae_early_stopping),
                    best_validation=float(best) if best is not None and np.isfinite(best) else None)
    if progress:
        progress(0.4, tx("{name} fertig – {n} auffällige Zeilen", name=model_name(cfg.model), n=info["n_anomalies"]),
                 None)
    curve = anomaly_curve(cfg, X, score, progress)
    info["seconds"] = round(time.time() - t0, 1)
    info["trained_at"] = datetime.now().isoformat(timespec="seconds")
    return model, {"score": score, "flag": flag, "labels": labels, "curve": curve}, info


# ---------------------------------------------------------------------------
# 7) Wahl von K – Elbow-Methode und Silhouette
# ---------------------------------------------------------------------------
def silhouette_verdict(s: float) -> str:
    """Faustregel nach Kaufman & Rousseeuw für den mittleren Silhouette-Koeffizienten."""
    if not np.isfinite(s):
        return tx("nicht berechenbar")
    if s > 0.70:
        return tx("starke Struktur – die Cluster sind klar getrennt")
    if s > 0.50:
        return tx("vernünftige Struktur")
    if s > 0.25:
        return tx("schwache Struktur – die Cluster überlappen; Ergebnis mit Vorsicht deuten")
    return tx("kaum Struktur – die Cluster überlappen stark (bei vielen Features ist die Silhouette oft niedrig; ob "
              "die Cluster trotzdem sinnvoll sind, zeigen die Profile in Schritt 8)")


def k_assessment(curve: dict | None, info: dict, cfg: ClusterConfig) -> dict:
    """Fazit zu Schritt 7: Was sagen Elbow-Kurve, Silhouette und Cluster-Größen über das gewählte K?

    Rückgabe: {"notes": [Markdown], "suggest": [(Text, K)] – Vorschläge für den Knopf 'übernehmen'}"""
    k, n = info["k"], info["n_rows"]
    notes, suggest = [], []
    s = info.get("silhouette", float("nan"))
    notes.append(tx("**Silhouette für K = {k}: {s}** – {verdict}.", k=k,
                    s=f"{s:.3f}" if np.isfinite(s) else "–", verdict=silhouette_verdict(s)))
    if curve:
        e, b = curve.get("elbow"), curve.get("best_silhouette")
        sil = dict(zip(curve["values"], curve["silhouette"]))
        if e is not None:
            notes.append(tx("**Elbow-Methode:** Knick bei **K = {e}** – danach sinkt die Inertia deutlich langsamer.",
                            e=e))
        else:
            notes.append(tx("**Elbow-Methode:** kein deutlicher Knick – die Inertia sinkt gleichmäßig."))
        if b is not None:
            notes.append(tx("**Silhouette** am höchsten bei **K = {b}** ({s:.3f}).", b=b, s=sil[b])
                         + (tx(" Das ist der Rand des geprüften Bereichs – k_max in Schritt 6 erhöhen, um größere K zu "
                               "sehen.") if b == max(curve["values"]) else ""))
            if sil[b] <= 0.25:  # kaum Struktur: das Maximum ist dann eher Zufall als ein gutes K
                notes.append(tx("Die Silhouette bleibt für alle K niedrig – die Daten zerfallen kaum in getrennte "
                                "Gruppen; die Silhouette liefert hier keinen Vorschlag."))
                b = None
        if e is not None and e == b:
            notes.append(tx("Beide Methoden zeigen **K = {k2}**.", k2=e)
                         + ("" if e == k else tx(" Eingestellt ist K = {k}.", k=k)))
            if e != k:
                suggest.append((tx("K = {v} übernehmen", v=e), e))
        else:
            for value, what in ((e, tx("Elbow")), (b, tx("Silhouette"))):
                if value is not None and value != k:
                    suggest.append((tx("K = {v} übernehmen ({what})", v=value, what=what), value))
            if e is not None and b is not None:
                notes.append(tx("Elbow und Silhouette zeigen verschiedene K – beide ausprobieren: Am Ende zählt, ob sich "
                                "die Cluster in Schritt 8 sinnvoll beschreiben lassen."))
        if curve.get("sampled"):
            notes.append(tx("Kurve auf einer Stichprobe von {n} Zeilen – das Modell selbst nutzt alle Zeilen.",
                            n=_n(curve["n_rows"])))
    sizes = np.asarray(info.get("sizes", []), dtype=float)
    if len(sizes) and n:
        empty = [int(j) for j in np.where(sizes == 0)[0]]
        if empty:  # scikit-learn verlegt leere Cluster – leer bleiben sie nur bei weniger verschiedenen Zeilen als K
            notes.append(tx(":orange[**Leerer Cluster:**] {clusters} – keine Zeile liegt diesem Zentrum am nächsten. Die "
                            "Daten haben weniger verschiedene Zeilen als K (doppelte Zeilen) – K verkleinern.",
                            clusters=", ".join(cluster_label(j) for j in empty)))
        filled = np.where(sizes > 0)[0]
        j = int(filled[np.argmin(sizes[filled])]) if len(filled) else 0
        if len(filled) and sizes[j] / n < 0.02:
            notes.append(tx(":orange[**Sehr kleiner Cluster:**] Cluster {j} hat nur {v} Zeilen ({pct}) – oft Ausreißer. "
                            "Tipp: Ausreißer in Schritt 2 begrenzen (Werte begrenzen) oder K verkleinern.",
                            j=j, v=_n(int(sizes[j])), pct=_pct(sizes[j] / n)))
        j = int(np.argmax(sizes))
        if sizes[j] / n > 0.8:
            notes.append(tx(":orange[**Ein Cluster dominiert:**] Cluster {j} enthält {pct} der Zeilen – die Daten trennen "
                            "sich kaum, oder wenige Ausreißer bilden die übrigen Cluster.",
                            j=j, pct=_pct(sizes[j] / n)))
    if not info.get("converged", True):
        notes.append(tx(":orange[**Nicht konvergiert:**] K-Means hat max_iter erreicht – max_iter in Schritt 6 erhöhen."))
    for w in info.get("warnings", []):
        notes.append(tx(":orange[**Hinweis von scikit-learn:**] {w}", w=w))
    return {"notes": notes, "suggest": suggest}


def share_text(x: float) -> str:
    """Anteil lesbar: unter 1 % mit zwei Nachkommastellen (0.25 %), sonst eine (1.5 %)."""
    x = float(x)
    return f"{x:.2%}" if 0 < x < 0.01 else f"{x:.1%}"


def score_text(v) -> str:
    """Score mit 4 signifikanten Stellen; ∞ (DBSCAN ohne Kernpunkt), '–' wenn unbekannt."""
    v = float(v)
    if np.isnan(v):
        return "–"
    return "∞" if np.isinf(v) else f"{v:.4g}"


def autoencoder_notes(info: dict) -> list[str]:
    """Schritt 6/7, Autoencoder: Aufbau des Netzes, Training (Epochen, Loss, Validierung) und Warnung, wenn der Engpass
    nicht enger ist als die Zahl der Features – dann kann das Netz die Zeilen einfach durchreichen."""
    layers, b, p = info.get("layers") or [], int(info.get("code_size") or 0), int(info["n_features"])
    text = tx("**Netz:** {p} Features → Schichten {layers} → {p} Features, Engpass {b} Neuronen · {e} Epochen mit "
              "batch_size {bs}, Loss {loss}.", p=p, layers=" – ".join(str(v) for v in layers), b=b,
              e=info.get("n_iter", "?"), bs=info.get("batch_size", "?"), loss=score_text(info.get("loss", float("nan"))))
    best = info.get("best_validation")
    if best is not None:
        text += tx(" Early Stopping: bestes R² auf den Validierungszeilen {r}.", r=f"{best:.3f}")
    notes = [text]
    if info.get("early_requested") and not info.get("early_stopping", True):
        notes.append(tx("Early Stopping aus: erst ab {m} Zeilen sinnvoll (10 % = mindestens {v} Zeilen zur Validierung) – "
                        "bei {n} Zeilen lernt das Netz mit allen Zeilen und endet, wenn der Trainingsfehler nicht mehr "
                        "sinkt.", m=_n(AE_EARLY_MIN_ROWS), v=AE_EARLY_MIN_ROWS // 10, n=_n(info["n_rows"])))
    if best is not None and best < 0:
        notes.append(tx(":orange[**Schwacher Nachbau:**] Das beste R² auf den Validierungszeilen ist negativ – das Netz "
                        "baut sie schlechter nach als der Mittelwert jeder Spalte. Mehr Epochen, eine größere Lernrate "
                        "oder mehr Neuronen probieren (Schritt 6)."))
    if info.get("epochs") and info.get("n_iter", 0) >= info["epochs"]:
        notes.append(tx("Das Training hat die Höchstzahl von {e} Epochen erreicht, der Fehler sank zuletzt noch leicht – "
                        "mehr Epochen (Schritt 6) können den Nachbau etwas verbessern.", e=info["epochs"]))
    if layers and b >= p > 1:
        notes.append(tx(":orange[**Kein echter Engpass:**] Die engste Schicht hat {b} Neuronen bei {p} Features – das "
                        "Netz kann die Zeilen fast unverändert durchreichen, ungewöhnliche Zeilen fallen dann kaum auf. "
                        "In der Mitte weniger Neuronen wählen (z. B. {s}).",
                        b=b, p=p, s=layers_text(default_ae_layers(p))))
    return notes


def threshold_assessment(curve: dict | None, info: dict, cfg: ClusterConfig) -> dict:
    """Fazit zu Schritt 7 für die Anomalie-Modelle: Wie viele Zeilen sind auffällig, passt die Schwelle (IF, LOF) bzw.
    eps (DBSCAN)? Rückgabe wie k_assessment: {"notes": [Markdown], "suggest": [(Text, Wert)]} – Wert = contamination
    (Anteil) bzw. eps für den Knopf 'übernehmen'."""
    n, k, share = info["n_rows"], info["n_anomalies"], info["share"]
    notes, suggest = [], []
    curve = curve or {}
    if cfg.model in (MODEL_IFOREST, MODEL_LOF, MODEL_AUTOENCODER):
        notes.append(tx("**{k} von {n} Zeilen auffällig ({pct})** – Schwelle: Score über {thr}.",
                        k=_n(k), n=_n(n), pct=share_text(share), thr=score_text(info["threshold"])))
        if cfg.model == MODEL_AUTOENCODER:
            notes.extend(autoencoder_notes(info))
        if not cfg.contamination:
            notes.append(tx(N_("contamination = auto: Die Schwelle kommt aus dem Verfahren selbst – Score 0.5 wie im "
                               "Original-Paper zu Isolation Forest.") if cfg.model == MODEL_IFOREST else
                            N_("contamination = auto: Die Schwelle kommt aus dem Verfahren selbst – LOF-Wert 1.5.")))
            if share > 0.1:
                notes.append(tx(":orange[**Viele auffällige Zeilen ({pct}):**] Für eine Liste der stärksten Ausreißer "
                                "mit contamination einen festen, kleineren Anteil wählen (z. B. 1 %).",
                                pct=share_text(share)))
        else:
            notes.append(tx("contamination = {pct}: Die Schwelle ist das passende Quantil der Trainings-Scores – etwa "
                            "dieser Anteil der Zeilen gilt als auffällig.", pct=share_text(cfg.contamination)))
        if k == 0:
            notes.append(tx(":orange[**Keine auffällige Zeile:**] Kein Score liegt über der Schwelle – mit "
                            "contamination einen festen Anteil wählen."))
        if curve.get("knee") is not None:
            c = curve["suggest"]
            notes.append(tx("**Knick der Score-Kurve:** Nach den auffälligsten {n} Zeilen ({pct}) flacht die Kurve ab – "
                            "diese Zeilen heben sich deutlich vom Rest ab. Ein Kandidat für contamination.",
                            n=_n(curve["knee"] + 1), pct=share_text(c)))
            if not cfg.contamination or abs(c - cfg.contamination) > 0.2 * c:
                suggest.append((tx("contamination = {pct} übernehmen", pct=share_text(c)), c))
        elif curve:
            notes.append(tx("**Score-Kurve ohne deutlichen Knick:** Die Scores sinken gleichmäßig – die Schwelle ist dann "
                            "eine Frage des gewünschten Anteils (Tabelle unten)."))
    else:
        notes.append(tx("**{c} Cluster, {k} Zeilen Rauschen ({pct})** bei eps = {eps} und min_samples = {m}.",
                        c=info.get("k", 0), k=_n(k), pct=share_text(share), eps=f"{info['eps']:.4g}",
                        m=info["min_samples"]))
        notes.append(tx("Kernpunkte: {c} von {n} Zeilen ({pct}).", c=_n(info["n_core"]), n=_n(n),
                        pct=share_text(info["n_core"] / max(n, 1))))
        if info["n_core"] == 0:
            notes.append(tx(":red[**Kein Kernpunkt:**] eps ist zu klein – alle Zeilen sind Rauschen. eps vergrößern "
                            "(Knick der k-Distanz)."))
        elif share > 0.5:
            notes.append(tx(":orange[**Mehr als die Hälfte ist Rauschen:**] eps ist zu klein oder min_samples zu groß."))
        elif info.get("k", 0) == 1 and share < 0.005:
            notes.append(tx(":orange[**Ein Cluster, fast kein Rauschen:**] eps ist eher zu groß – kaum eine Zeile gilt als "
                            "Anomalie."))
        if curve.get("knee") is not None:
            e = curve["suggest"]
            notes.append(tx("**Knick der k-Distanz bei eps ≈ {eps}:** Rechts davon liegen die Zeilen deutlich weiter von "
                            "ihren Nachbarn entfernt – Kandidaten für Rauschen.", eps=f"{e:.4g}"))
            if e <= EPS_MIN and curve.get("zero_share"):
                notes.append(tx(":orange[**Viele genau gleiche Zeilen:**] Bei {pct} der Zeilen ist die k-Distanz 0 – sie "
                                "haben mindestens min_samples − 1 genau gleiche Zeilen (typisch, wenn alle Features "
                                "kategorial sind). Mit dem kleinsten eps ({e}) bilden genau gleiche Zeilen die Cluster, "
                                "seltene Kombinationen sind Rauschen.",
                                pct=share_text(curve["zero_share"]), e=f"{EPS_MIN:g}"))
            if abs(e - float(cfg.eps)) > 0.05 * e:
                suggest.append((tx("eps = {eps} übernehmen", eps=f"{e:.4g}"), e))
        elif curve:
            notes.append(tx("**k-Distanz ohne deutlichen Knick:** eps nach dem gewünschten Rauschanteil wählen (Tabelle "
                            "unten)."))
        if curve.get("sampled"):
            notes.append(tx("k-Distanz für {n} zufällige Zeilen gemessen (Nachbarn aus allen Zeilen).",
                            n=_n(curve["n_rows"])))
        if curve and curve.get("grid") is None and cfg.curve and n > CURVE_ROWS:
            notes.append(tx("Rauschanteil je eps nur bis {m} Zeilen – DBSCAN müsste für jedes eps neu laufen. Tipp: "
                            "Stichprobe in der Seitenleiste.", m=_n(CURVE_ROWS)))
        skipped = (curve.get("grid") or {}).get("skipped")
        if skipped:
            notes.append(tx("Rauschanteil je eps: eps ab {e} übersprungen – DBSCAN hielte dafür zu viele Nachbarpaare "
                            "gleichzeitig im Speicher.", e=f"{min(skipped):.4g}"))
    for w in info.get("warnings", []):
        notes.append(tx(":orange[**Hinweis von scikit-learn:**] {w}", w=w))
    return {"notes": notes, "suggest": suggest}


# ---------------------------------------------------------------------------
# 8) Cluster-Auswertung – Profile, Trennkraft, PCA, Vergleich mit bekannten Gruppen
# ---------------------------------------------------------------------------
def _cluster_means(X: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.bincount(labels, minlength=k)
    means = np.full((k, X.shape[1]), np.nan)
    for j in range(k):
        if sizes[j]:
            means[j] = X[labels == j].mean(axis=0)
    return means, sizes


def cluster_profiles(prep: PrepResult, labels: np.ndarray, k: int) -> dict:
    """Mittelwert jedes Features je Cluster in Originaleinheiten (Dummies: Anteil 0 … 1) und die Abweichung vom
    Gesamtmittel in Standardabweichungen (z) – daran lässt sich ablesen, was einen Cluster ausmacht."""
    r = prep.recipe
    X = prep.X
    means, sizes = _cluster_means(X, labels, k)
    overall, std = X.mean(axis=0), X.std(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(std > 0, (means - overall) / np.where(std > 0, std, 1.0), 0.0)
    raw = invert_scaler(np.vstack([overall, means]), r["scaler"])  # affin → Mittelwerte bleiben Mittelwerte
    cols = r["feature_columns"]
    table = pd.DataFrame(raw.T, index=cols, columns=["Alle"] + list(range(k)))
    return {"sizes": sizes, "means": table, "z": pd.DataFrame(z.T, index=cols, columns=list(range(k)))}


def describe_clusters(z: pd.DataFrame, top: int = 3, threshold: float = 0.3) -> list[dict]:
    """Je Cluster die Features mit der stärksten Abweichung vom Durchschnitt (|z| ≥ threshold)."""
    out = []
    for j in z.columns:
        s = z[j].dropna()
        high = s[s >= threshold].sort_values(ascending=False).head(top)
        low = s[s <= -threshold].sort_values().head(top)
        out.append({"cluster": int(j), "high": list(high.items()), "low": list(low.items())})
    return out


def feature_separation(prep: PrepResult, labels: np.ndarray, k: int) -> pd.DataFrame:
    """Trennkraft je Feature: Anteil seiner Streuung, den die Cluster erklären (R² = Streuung zwischen den Clustern /
    Gesamtstreuung, 0 … 1). Dummies einer Kategorie zählen zusammen."""
    X = prep.X
    means, sizes = _cluster_means(X, labels, k)
    overall = X.mean(axis=0)
    ok = sizes > 0
    between = (sizes[ok, None] * (means[ok] - overall) ** 2).sum(axis=0)
    total = ((X - overall) ** 2).sum(axis=0)
    rows = []
    for f, idx in prep.recipe["feature_groups"].items():
        t = float(total[idx].sum())
        rows.append({"Feature": f, "R²": float(between[idx].sum()) / t if t > 0 else 0.0})
    return pd.DataFrame(rows).sort_values("R²", ascending=False, ignore_index=True)


def plot_rows(n: int, max_points: int = 5000, seed: int = RANDOM_STATE, priority=None) -> np.ndarray:
    """Zeilen für ein Streudiagramm: alle – oder eine feste Stichprobe von max_points Zeilen, in der die Zeilen aus
    priority (z. B. die Anomalien, höchstens die Hälfte des Bildes) immer dabei sind; sonst fehlten seltene Ausreißer."""
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(int(seed))
    rows = rng.choice(n, max_points, replace=False)
    if priority is not None:  # Anomalien zuerst, der Rest zufällig
        must = np.flatnonzero(np.asarray(priority, dtype=bool))
        if len(must) > max_points // 2:
            must = rng.choice(must, max_points // 2, replace=False)
        rows = np.concatenate([must, rows[~np.isin(rows, must)][: max_points - len(must)]])
    return np.sort(rows)


def code_projection(prep: PrepResult, model, max_points: int = 5000, seed: int = RANDOM_STATE,
                    priority: np.ndarray | None = None) -> dict | None:
    """Autoencoder, Schritt 8: die Zeilen im Engpass des Netzes – seine eigene, nichtlineare Verdichtung der Features.
    Engpass mit 2 Neuronen: die Ausgaben direkt; mit mehr: PCA der Ausgaben auf 2 Achsen; mit 1 Neuron: None (keine
    2D-Karte). Rückgabe wie pca_projection (coords, rows) plus die Achsentitel."""
    codes = autoencoder_codes(model, prep.X)
    b = codes.shape[1]
    if b < 2:
        return None
    rows = plot_rows(len(codes), max_points, seed, priority)
    if b == 2:
        xy, axes, ratio = codes[rows], (tx("Neuron 1 im Engpass"), tx("Neuron 2 im Engpass")), None
    else:
        pca = PCA(n_components=2, random_state=int(seed)).fit(codes)
        ratio = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0).tolist()
        xy = pca.transform(codes[rows])
        axes = tuple(tx("Code-PC{i} ({pct} der Streuung im Engpass)", i=i + 1, pct=f"{ratio[i]:.0%}") for i in range(2))
    return {"coords": xy, "rows": rows, "labels": np.zeros(len(rows), dtype=int), "centers": np.empty((0, 2)),
            "explained": ratio or [0.0, 0.0], "axes": axes, "code_size": int(b), "n": int(len(codes))}


def pca_projection(prep: PrepResult, model, max_points: int = 5000, seed: int = RANDOM_STATE,
                   labels: np.ndarray | None = None, priority: np.ndarray | None = None) -> dict:
    """PCA auf 2 Dimensionen – nur für die Darstellung. Die Modelle rechnen weiter mit allen Features.

    labels: Farbe je Zeile (Standard: model.labels_ – bei den Anomalie-Modellen 0/1 bzw. die DBSCAN-Cluster).
    priority: Zeilen, die in der Stichprobe für das Bild immer dabei sind (z. B. die Anomalien – sonst fehlten
    seltene Ausreißer im Bild). Zentren gibt es nur bei K-Means."""
    X = prep.X
    n, p = X.shape
    n_comp = min(2, p)
    pca = PCA(n_components=n_comp, random_state=int(seed)).fit(X)
    rows = plot_rows(n, max_points, seed, priority)
    centers_in = getattr(model, "cluster_centers_", None)
    coords = pca.transform(X[rows])
    centers = pca.transform(centers_in) if centers_in is not None else np.empty((0, n_comp))
    if n_comp == 1:  # nur ein Feature: zweite Achse = 0
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords))])
        centers = np.column_stack([centers[:, 0], np.zeros(len(centers))])
    ratio = np.nan_to_num(pca.explained_variance_ratio_, nan=0.0).tolist() + [0.0] * (2 - n_comp)
    load = pd.DataFrame(pca.components_.T, index=prep.recipe["feature_columns"],
                        columns=[f"PC{i + 1}" for i in range(n_comp)])
    if labels is None:  # Isolation Forest und LOF haben keine labels_ → alle 0
        labels = getattr(model, "labels_", np.zeros(n, dtype=int))
    lab = np.asarray(labels)
    return {"coords": coords, "rows": rows, "labels": lab[rows], "centers": centers,
            "explained": ratio, "loadings": load, "n": int(n)}


def compare_groups(labels: np.ndarray, groups: pd.Series, k: int, mapping: dict | None = None) -> dict | None:
    """Vergleich mit einer bekannten Einteilung (im Kurs: Private). Die Nummern der Cluster sind willkürlich –
    deshalb bekommt jeder Cluster die Gruppe, die in ihm am häufigsten ist (Mehrheit). mapping = Zuordnung aus dem
    Training (für neue Daten); sonst wird sie hier aus den Daten bestimmt.

    Rückgabe: Kreuztabelle, Zuordnung, Treffer nach Zuordnung, naive Vorhersage, ARI, NMI, Confusion Matrix …"""
    g = pd.Series(groups).reset_index(drop=True).astype(object)
    lab = pd.Series(np.asarray(labels)).reset_index(drop=True)
    mask = g.notna()
    if not mask.any():
        return None
    g, lab = g[mask], lab[mask]
    values = sorted(set(g.unique()) | set((mapping or {}).values()), key=_natural_key)
    ct = pd.crosstab(lab, g).reindex(index=range(k), columns=values, fill_value=0)
    majority = g.value_counts().idxmax()
    if mapping is None:
        mapping = {j: (ct.columns[int(np.argmax(ct.loc[j].to_numpy()))] if ct.loc[j].sum() else majority)
                   for j in range(k)}
    else:
        mapping = {int(j): v for j, v in mapping.items()}
    pred = lab.map(lambda j: mapping.get(int(j), majority))
    cm = confusion_matrix(g, pred, labels=values)
    raw = None
    if len(values) == k:  # wie im Kurs: Gruppen-Nummer (sortiert) direkt mit der Cluster-Nummer verglichen
        code = {v: i for i, v in enumerate(values)}
        raw = float((g.map(code).to_numpy() == lab.to_numpy()).mean())
    return {
        "crosstab": ct, "mapping": mapping, "classes": values, "cm": cm,
        "accuracy": float((pred == g).mean()), "baseline": float(g.value_counts(normalize=True).max()),
        "majority": majority, "raw_accuracy": raw,
        "ari": float(adjusted_rand_score(g, lab)), "nmi": float(normalized_mutual_info_score(g, lab)),
        "report": classification_report(g, pred, labels=values, output_dict=True, zero_division=0),
        "report_text": classification_report(g, pred, labels=values, zero_division=0),
        "n": int(mask.sum()), "missing": int((~mask).sum()),
        "unmatched": [v for v in values if v not in set(mapping.values())],
    }


def cluster_distances(model, X: np.ndarray, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Abstand jeder Zeile zu allen Zentren (wie kmeans.transform) und Cluster (wie kmeans.predict)."""
    D = model.transform(X)
    lab = model.predict(X) if labels is None else np.asarray(labels)
    return D, lab


# --- 8) Auswertung der Anomalie-Modelle ------------------------------------------------------------------------------
def anomaly_profiles(prep: PrepResult, flag: np.ndarray) -> dict:
    """Mittelwerte der normalen und der auffälligen Zeilen in Originaleinheiten und ihre Abweichung vom Durchschnitt
    (z) – wie die Cluster-Profile mit zwei Gruppen: 0 = normal, 1 = Anomalie."""
    return cluster_profiles(prep, np.asarray(flag, dtype=bool).astype(int), 2)


def row_deviations(prep: PrepResult, rows, top: int = 3) -> list[list[tuple[str, float]]]:
    """Je Zeile die Features mit der größten Abweichung vom Durchschnitt aller Zeilen (z in Standardabweichungen) –
    ein Blick in die Daten, was eine auffällige Zeile ausmacht (unabhängig vom Modell)."""
    X = prep.X
    mean, std = X.mean(axis=0), X.std(axis=0)
    rows = np.asarray(rows, dtype=int)
    z = np.where(std > 0, (X[rows] - mean) / np.where(std > 0, std, 1.0), 0.0)
    cols = prep.recipe["feature_columns"]
    out = []
    for zr in z.reshape(len(rows), -1):
        idx = np.argsort(-np.abs(zr), kind="stable")[:top]
        out.append([(cols[j], float(zr[j])) for j in idx])
    return out


def deviation_text(devs, unit: str = " σ") -> str:
    """[('vol_z', 3.2), ('ret_z', -2.9)] → 'vol_z +3.2 σ · ret_z −2.9 σ'."""
    return " · ".join(f"{c} " + f"{z:+.1f}".replace("-", "−") + unit for c, z in devs)


def recon_deviations(prep: PrepResult, recon: np.ndarray, rows, top: int = 3) -> list[list[tuple[str, float]]]:
    """Autoencoder: je Zeile die Features, die das Netz am schlechtesten nachgebaut hat – Eingabe minus Nachbau in
    den skalierten Features (StandardScaler: Standardabweichungen). Anders als row_deviations ein Blick ins Modell:
    auch ein Feature mit normalem Wert kann hier stehen, wenn es nicht zu den übrigen Features der Zeile passt."""
    cols = prep.recipe["feature_columns"]
    out = []
    for r in np.asarray(recon, dtype="float64")[np.asarray(rows, dtype=int)].reshape(len(rows), -1):
        idx = np.argsort(-np.abs(r), kind="stable")[:top]
        out.append([(cols[j], float(r[j])) for j in idx])
    return out


def top_anomalies(prep: PrepResult, score, flag, n: int = 15, labels=None, recon=None) -> pd.DataFrame:
    """Die n auffälligsten Zeilen (höchster Score): Zeile im Datensatz nach Schritt 2, Bezeichnung, Score und die
    Features mit der größten Abweichung vom Durchschnitt – beim Autoencoder (recon = reconstruction()) zusätzlich die
    am schlechtesten nachgebauten Features."""
    score = np.asarray(score, dtype="float64")
    order = np.argsort(-score, kind="stable")[:n]
    worst = recon_deviations(prep, recon, order) if recon is not None else [None] * len(order)
    rows = []
    for rank, (i, devs, bad) in enumerate(zip(order, row_deviations(prep, order), worst), 1):
        row = {"Rang": rank, "Zeile": int(prep.index[i])}
        if prep.names is not None:
            name = prep.names.iloc[i]
            row["Bezeichnung"] = "–" if pd.isna(name) else str(name)
        if labels is not None:
            row["Cluster"] = int(labels[i])
        row.update({"Anomalie-Score": float(np.round(score[i], 4)), "Anomalie": bool(flag[i]),
                    "größte Abweichungen (z)": deviation_text(devs)})
        if bad is not None:
            row["am schlechtesten nachgebaut"] = deviation_text(bad, unit="")
        rows.append(row)
    return pd.DataFrame(rows)


def compare_anomalies(flag, score, groups, positive=None) -> dict | None:
    """Vergleich mit einer bekannten Einteilung (z. B. Class bei Betrugsdaten) – das Modell kannte sie nicht.

    Je Gruppe der Anteil auffälliger Zeilen. Bei genau zwei Gruppen gilt die seltenere als 'positiv' (Anomalien sind
    selten): Precision = Anteil der auffälligen Zeilen, die zu ihr gehören; Recall = Anteil ihrer Zeilen, die als
    auffällig gemeldet werden; ROC-AUC und Average Precision bewerten den Score unabhängig von der Schwelle.
    positive: die positive Gruppe aus dem Training (Schritt 10) – sonst kippte die Bewertung, wenn in den neuen Daten
    zufällig die andere Gruppe seltener ist (z. B. eine Stichprobe mit vielen Betrugsfällen)."""
    g = pd.Series(groups).reset_index(drop=True).astype(object)
    f = np.asarray(flag, dtype=bool)
    s = np.asarray(score, dtype="float64")
    mask = g.notna().to_numpy()
    if not mask.any():
        return None
    g, f, s = g[mask].reset_index(drop=True), f[mask], s[mask]
    values = sorted(g.unique(), key=_natural_key)
    counts = g.value_counts()
    table = pd.DataFrame({"Gruppe": values, "Zeilen": [int(counts[v]) for v in values],
                          "auffällig": [int(f[(g == v).to_numpy()].sum()) for v in values]})
    table["Anteil auffällig"] = table["auffällig"] / table["Zeilen"]
    out = {"table": table, "values": values, "n": int(mask.sum()), "missing": int((~mask).sum()),
           "share": float(f.mean()), "binary": len(values) == 2}
    if len(values) == 2:
        want = None if positive is None else group_values(pd.Series([positive])).iloc[0]  # wie die Gruppen: 1.0 → '1'
        known = [v for v in values if want is not None and not pd.isna(want) and str(v) == str(want)]
        pos = known[0] if known else min(values, key=lambda v: (int(counts[v]), _natural_key(v)))
        out["positive_source"] = "training" if known else "rarer"
        y = (g == pos).to_numpy()
        finite = np.isfinite(s)
        s_rank = np.where(np.isposinf(s), (s[finite].max() + 1.0) if finite.any() else 1.0, s)  # ∞ (DBSCAN) = höchster
        precision = float(y[f].mean()) if f.any() else float("nan")
        out.update(positive=pos, base_rate=float(y.mean()), precision=precision, recall=float(f[y].mean()),
                   lift=precision / float(y.mean()) if f.any() else float("nan"),
                   roc_auc=float(roc_auc_score(y, s_rank)), ap=float(average_precision_score(y, s_rank)),
                   tp=int((f & y).sum()), fp=int((f & ~y).sum()), fn=int((~f & y).sum()), tn=int((~f & ~y).sum()))
    return out


# ---------------------------------------------------------------------------
# 9) Speichern & Laden
# ---------------------------------------------------------------------------
MODEL_FILES = ("model.skops", "model.joblib")
BUNDLE_FILES = (*MODEL_FILES, "preprocessing.json")
MAX_BUNDLE_BYTES = 2 * 10**9  # entpackte Größe eines Modell-ZIPs
# skops lädt nur Typen, denen man ausdrücklich vertraut: K-Means braucht keine zusätzlichen, Isolation Forest die Bäume
# (Tree), LOF den Suchbaum der Nachbarn (KDTree/BallTree, wie in der Supervised-Version). Alles andere, was skops nicht
# von sich aus kennt (z. B. fremder Code), wird abgelehnt.
SKOPS_TRUSTED: set = {"sklearn.tree._tree.Tree", "sklearn.neighbors._kd_tree.KDTree",
                      "sklearn.neighbors._ball_tree.BallTree",
                      "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"}  # Autoencoder: Zustand von Adam
SKOPS_TRUSTED_PREFIX = ("sklearn.metrics._dist_metrics.",)


def clean_names(names, k: int) -> list[str]:
    """Namen der Cluster aus Schritt 8: genau k Texte, '' = nicht benannt (Anzeige dann 'Cluster 2')."""
    names = list(names or [])
    return [" ".join(str(names[j]).split())[:80] if j < len(names) and names[j] is not None
            and str(names[j]).strip() not in ("", "nan", "None") else "" for j in range(k)]


def cluster_label(j: int, names=None) -> str:
    """'Cluster 2' (in der gewählten Sprache) oder '2 · Name', wenn der Cluster in Schritt 8 benannt wurde."""
    name = names[j] if names is not None and j < len(names) and names[j] else ""
    return f"{j} · {name}" if name else tx("Cluster {j}", j=j)


def finalize_recipe(recipe: dict, cfg: ClusterConfig | None = None, info: dict | None = None,
                    cluster_names: list | None = None, comparison: dict | None = None) -> dict:
    """Einstellungen, Kennzahlen, Cluster-Namen und die Zuordnung zur Vergleichsgruppe ins Rezept schreiben –
    bei den Anomalie-Modellen Schwelle, Anteil auffälliger Zeilen und der Vergleich mit der bekannten Einteilung."""
    r = dict(recipe)
    if cfg is not None:
        r["training"] = asdict(cfg)
    if info is not None and is_anomaly(info.get("model")):
        r["train_info"] = info
        sizes = np.asarray(info.get("sizes", []), dtype=float)  # nur DBSCAN: Zeilen je Cluster (ohne Rauschen)
        r["cluster_share"] = (sizes / max(int(info.get("n_rows", 0)), 1)).tolist() if len(sizes) else []
        r["metrics"] = {key: info.get(key) for key in ("n_anomalies", "share", "threshold", "score_median",
                                                       "score_p99")}
        r["cluster_names"] = []
    elif info is not None:
        r["train_info"] = info
        sizes = np.asarray(info.get("sizes", []), dtype=float)
        r["cluster_share"] = (sizes / sizes.sum()).tolist() if sizes.sum() else []
        r["metrics"] = {key: info.get(key) for key in ("inertia", "silhouette", "davies_bouldin",
                                                       "calinski_harabasz", "r2")}
        r["cluster_names"] = clean_names(cluster_names, int(info.get("k", len(sizes))))
    if comparison is not None and "mapping" in comparison:
        r["group_mapping"] = {str(j): v for j, v in comparison["mapping"].items()}
        r["group_metrics"] = {"accuracy": comparison["accuracy"], "baseline": comparison["baseline"],
                              "ari": comparison["ari"], "nmi": comparison["nmi"], "n": comparison["n"]}
    elif comparison is not None:  # Anomalie-Modell: Vergleich mit der bekannten Einteilung (compare_anomalies)
        r["anomaly_metrics"] = {key: comparison.get(key) for key in ("positive", "base_rate", "precision", "recall",
                                                                     "roc_auc", "ap", "share", "n")}
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


SCALER_PARAMS = {"StandardScaler": ("mean", "std"), "MinMaxScaler": ("scale", "offset"), "keiner": ()}


def _strings(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _numbers(v, n: int | None = None) -> bool:
    return (isinstance(v, list) and (n is None or len(v) == n)
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def recipe_problem(recipe: dict) -> str | None:
    """Ist ein Clustering-Rezept vollständig und in sich stimmig? Ein beschädigtes oder von Hand verändertes
    preprocessing.json soll beim Laden scheitern – nicht erst in Schritt 10 oder gar still falsch rechnen (z. B. ein
    unbekannter Scaler, der sonst wie 'keiner' wirkt). Rückgabe: betroffener Eintrag oder None."""
    feats = recipe.get("features")
    if not _strings(feats) or not feats:
        return "features"
    num, cat = recipe.get("numeric_features"), recipe.get("categorical_features")
    if not _strings(num) or not _strings(cat) or set(num) & set(cat) or set(num) | set(cat) != set(feats):
        return "numeric_features / categorical_features"
    cats = recipe.get("categories")
    if not isinstance(cats, dict) or any(not _strings(cats.get(c)) for c in cat):
        return "categories"
    cols = recipe.get("feature_columns")
    if not _strings(cols) or not cols or any(c not in cols for c in num):
        return "feature_columns"
    sc = recipe.get("scaler")
    if (not isinstance(sc, dict) or sc.get("type") not in SCALER_PARAMS
            or any(not _numbers(sc.get(k), len(cols)) for k in SCALER_PARAMS[sc.get("type")])):
        return "scaler"
    if not _numbers(recipe.get("train_min"), len(cols)) or not _numbers(recipe.get("train_max"), len(cols)):
        return "train_min / train_max"
    fills, fallback = recipe.get("fills"), recipe.get("fallback_fill")
    if (not isinstance(fills, dict) or not isinstance(fallback, dict) or any(c not in fallback for c in feats)
            or not _numbers([fallback[c] for c in num]) or not _strings([fallback[c] for c in cat])):
        return "fills / fallback_fill"
    for f in fills.values():
        if not isinstance(f, dict) or not isinstance(f.get("strategy"), str):
            return "fills"
        if f["strategy"] == M_GROUP:
            if (not isinstance(f.get("group_col"), str) or not isinstance(f.get("group_means"), list)
                    or not _numbers([f.get("overall")])):
                return "fills"
        elif "value" not in f:
            return "fills"
    if not isinstance(recipe.get("drop_first", False), bool):
        return "drop_first"
    for key in ("column_actions", "new_columns"):
        if not isinstance(recipe.get(key, []), list):
            return key
    for key in ("lineage", "training", "train_info", "metrics"):
        if not isinstance(recipe.get(key) or {}, dict):
            return key
    for key in ("group_col", "name_col"):
        if not isinstance(recipe.get(key) or "", str):
            return key
    ti = recipe.get("train_info") or {}
    for key in ("dist_p99", "sizes"):
        if ti.get(key) is not None and not _numbers(ti[key]):
            return f"train_info.{key}"
    if recipe.get("cluster_share") is not None and not _numbers(recipe["cluster_share"]):
        return "cluster_share"
    if recipe.get("cluster_names") is not None and not isinstance(recipe["cluster_names"], list):
        return "cluster_names"
    gm = recipe.get("group_mapping")
    if gm is not None and (not isinstance(gm, dict) or any(not str(j).isdigit() for j in gm)):
        return "group_mapping"
    k = (recipe.get("training") or {}).get("n_clusters")
    if k is not None and (not isinstance(k, int) or isinstance(k, bool)):
        return "training.n_clusters"
    if recipe_model(recipe) not in MODEL_NAMES:
        return "training.model"
    for key in ("threshold", "share"):  # Anomalie-Modelle: Schwelle endlich, Anteil auffälliger Zeilen 0 … 1
        v = ti.get(key)
        if v is not None and (not _numbers([v]) or not np.isfinite(v) or (key == "share" and not 0 <= v <= 1)):
            return f"train_info.{key}"
    return None


def check_recipe(recipe, model=None) -> None:
    """Nur Modelle der Clustering-Version laden – ein Modell der Supervised-Version hätte ein anderes Rezept. Prüft
    außerdem, ob das Rezept vollständig ist und (mit model) zum Modell passt: Modelltyp laut Rezept (K-Means, Isolation
    Forest, LOF, DBSCAN, Autoencoder), Zahl der Eingänge, bei K-Means die Zahl der Cluster."""
    if not isinstance(recipe, dict) or recipe.get("task") != T_CLUSTER:
        task = recipe.get("task") if isinstance(recipe, dict) else None
        raise ValueError(tx("Dieses Modell stammt nicht aus der Clustering-Version (Aufgabe: {task}) – hier lassen "
                            "sich nur Modelle laden, die mit dieser App gespeichert wurden (K-Means, Isolation Forest, "
                            "LOF, DBSCAN, Autoencoder).",
                            task=str(task or "?")[:40]))
    problem = recipe_problem(recipe)
    if problem:
        raise ValueError(tx("preprocessing.json ist unvollständig oder beschädigt (Eintrag: {part}) – das Modell bitte "
                            "neu speichern (Schritt 9).", part=problem))
    if model is None:
        return
    key = recipe_model(recipe)
    if not isinstance(model, MODEL_CLASSES[key]):
        raise ValueError(tx("Die Modelldatei enthält kein {name}-Modell ({type}).", name=model_name(key),
                            type=type(model).__name__))
    if key == MODEL_KMEANS:
        k = len(model.cluster_centers_)
        n_in = int(getattr(model, "n_features_in_", model.cluster_centers_.shape[1]))
        sizes = (recipe.get("train_info") or {}).get("sizes")
        if (n_in != len(recipe["feature_columns"]) or (sizes is not None and len(sizes) != k)
                or (recipe.get("training") or {}).get("n_clusters") not in (None, k)):
            raise ValueError(tx("Modell und preprocessing.json passen nicht zusammen: das Modell hat {k} Cluster und "
                                "erwartet {n} Eingänge, das Rezept liefert {m}.",
                                k=k, n=n_in, m=len(recipe["feature_columns"])))
        return
    n_in = int(getattr(model, "n_features_in_", -1))
    if n_in != len(recipe["feature_columns"]):
        raise ValueError(tx("Modell und preprocessing.json passen nicht zusammen: das {name}-Modell erwartet {n} "
                            "Eingänge, das Rezept liefert {m}.",
                            name=model_name(key), n=n_in, m=len(recipe["feature_columns"])))
    if key == MODEL_LOF and not getattr(model, "novelty", False):
        raise ValueError(tx("Das LOF-Modell wurde ohne novelty=True trainiert – damit lassen sich keine neuen Daten "
                            "bewerten."))
    if key == MODEL_AUTOENCODER:
        t = getattr(model, "threshold_", None)
        if int(getattr(model, "n_outputs_", -1)) != n_in or not _numbers([t]) or not np.isfinite(t):
            raise ValueError(tx("Das Autoencoder-Modell ist unvollständig (Ausgänge oder Schwelle threshold_ fehlen) – "
                                "bitte in der App neu trainieren und speichern."))


def load_bundle(folder, allow_pickle: bool = True):
    """Gespeichertes Modell + Rezept laden. allow_pickle=False lehnt .joblib-Dateien ab (z. B. aus fremden ZIPs)."""
    folder = Path(folder)
    recipe = json.loads((folder / "preprocessing.json").read_text(encoding="utf-8"))
    check_recipe(recipe)
    name = _bundle_model_name(recipe, {n for n in MODEL_FILES if (folder / n).is_file()})
    model_file = folder / name if name else None
    if model_file is None:
        raise ValueError(tx("Kein scikit-learn-Modell gefunden (model.skops oder model.joblib)."))
    model = _load_model_file(model_file, allow_pickle)
    check_recipe(recipe, model)  # Modelltyp laut Rezept, Eingänge, Cluster
    return model, recipe


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
    """Gespeicherte Modelle der Clustering-Version im Ordner models/ auflisten (neueste zuerst) – Modelle anderer
    Versionen nicht. model: kmeans, iforest, lof oder dbscan (ältere Rezepte ohne Eintrag: K-Means)."""
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
        if not isinstance(r, dict) or r.get("task") != T_CLUSTER:
            continue
        out.append({"name": sub.name, "path": str(sub), "k": (r.get("training") or {}).get("n_clusters"),
                    "n_features": len(r.get("features") or []), "model": (r.get("training") or {}).get("model"),
                    "share": (r.get("train_info") or {}).get("share"),
                    "saved_at": str(r.get("saved_at", "")), "metrics": r.get("metrics", {})})
    return sorted(out, key=lambda b: b["saved_at"], reverse=True)


# ---------------------------------------------------------------------------
# 10) Neue Daten den Clustern zuordnen
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


def prepare_new_data(df_new: pd.DataFrame, recipe: dict):
    """Neue Rohdaten exakt wie die Trainingsdaten aufbereiten (Schritt 2 + 4 + Skalierung).

    Rückgabe: (X skaliert, df nach Schritt 2, Bericht). Wirft MissingColumnsError, wenn benötigte Spalten fehlen.
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
    lin = recipe.get("lineage", {})
    info_cols = [c for c in (recipe.get("group_col"), recipe.get("name_col")) if c]
    used_raw = (set(required_raw_columns(recipe)) | set(recipe["features"]) | set(info_cols)
                | {r for c in info_cols for r in raw_sources(lin, c)})
    rep["extra_columns"] = [c for c in df_new.columns if c not in used_raw]
    rep["n_required"] = len(required_raw_columns(recipe))
    return apply_scaler(X_df.to_numpy(), recipe["scaler"]), d, rep


RESULT_CLUSTER, RESULT_NAME, RESULT_DIST, RESULT_NOTE = "Cluster", "Cluster-Name", "Abstand", "Hinweis"
RESULT_ANOMALY, RESULT_SCORE = "Anomalie", "Anomalie-Score"  # Ergebnisspalten der Anomalie-Modelle (1 = auffällig)
AMBIGUOUS = 0.95  # Abstand zum nächsten / zum zweitnächsten Zentrum ≥ 0.95 → liegt fast genau zwischen zwei Clustern


def assign_new(model, recipe: dict, df_new: pd.DataFrame):
    """Neue Daten dem nächsten Cluster-Zentrum zuordnen (kmeans.predict) – nichts wird neu gelernt.

    Rückgabe: (Ergebnis-DataFrame, Bericht, Vergleich mit der bekannten Einteilung oder None)."""
    if is_anomaly(recipe_model(recipe)):
        raise ValueError(tx("Anomalie-Modell: neue Daten mit score_new() bewerten (assign_new ist für K-Means)."))
    check_recipe(recipe, model)
    if len(df_new) == 0:
        raise ValueError(tx("Die Datei enthält keine Datenzeilen (nur Kopfzeile?)."))
    X, d, rep = prepare_new_data(df_new, recipe)
    D, lab = cluster_distances(model, X)
    k = D.shape[1]
    d1 = D[np.arange(len(D)), lab]
    d2 = np.partition(D, 1, axis=1)[:, 1] if k > 1 else np.full(len(D), np.inf)
    p99 = np.asarray((recipe.get("train_info") or {}).get("dist_p99") or [np.inf] * k, dtype=float)
    far = d1 > p99[lab] if len(p99) == k else np.zeros(len(D), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        ambiguous = (d2 > 0) & (d1 / d2 >= AMBIGUOUS) & ~far
    rep.update(labels=lab, distance=d1, row_far=far, row_ambiguous=ambiguous)
    res = df_new.copy()
    res[RESULT_CLUSTER] = lab.astype(int)
    names = clean_names(recipe.get("cluster_names"), k)
    if any(names):
        res[RESULT_NAME] = [names[j] for j in lab]
    res[RESULT_DIST] = np.round(d1, 4)
    flags = [(rep["row_imputed"], N_("Lücken ergänzt")), (rep["row_unknown"], N_("unbekannte Kategorie")),
             (rep["row_out_of_range"], N_("außerhalb Trainingsbereich")), (far, N_("weit weg von allen Clustern")),
             (ambiguous, N_("zwischen zwei Clustern"))]
    notes = pd.Series("", index=range(len(res)), dtype=object)
    for mask, text in flags:
        if mask.any():
            notes[mask] = notes[mask].where(notes[mask] == "", notes[mask] + "; ") + text
    res[RESULT_NOTE] = notes.to_numpy()
    cmp = None
    gcol = recipe.get("group_col")
    if gcol and gcol in d.columns:
        cmp = compare_groups(lab, group_values(d[gcol]), k, recipe.get("group_mapping"))
    return res, rep, cmp


def score_new(model, recipe: dict, df_new: pd.DataFrame):
    """Neue Daten mit dem Anomalie-Modell aus dem Training bewerten (Isolation Forest, LOF, DBSCAN, Autoencoder) –
    nichts wird neu gelernt: gleiche Aufbereitung (Rezept), gleiche Schwelle wie im Training.

    Ergebnis: Daten + Anomalie (1 = auffällig), Anomalie-Score (höher = auffälliger), bei DBSCAN zusätzlich Cluster
    (−1 = Rauschen) und Hinweis (Datenqualität). Rückgabe: (Ergebnis-DataFrame, Bericht, Vergleich mit der bekannten
    Einteilung oder None)."""
    key = recipe_model(recipe)
    if not is_anomaly(key):
        raise ValueError(tx("K-Means-Modell: neue Daten mit assign_new() zuordnen (score_new ist für die "
                            "Anomalie-Modelle)."))
    check_recipe(recipe, model)
    if len(df_new) == 0:
        raise ValueError(tx("Die Datei enthält keine Datenzeilen (nur Kopfzeile?)."))
    X, d, rep = prepare_new_data(df_new, recipe)
    score, flag, labels = anomaly_scores(model, X, key)
    name_col = recipe.get("name_col")
    rep.update(score=score, flag=flag, labels=labels,
               names=_to_str(d[name_col]).to_numpy() if name_col and name_col in d.columns else None)
    res = df_new.copy()
    if labels is not None:
        res[RESULT_CLUSTER] = labels.astype(int)
    res[RESULT_ANOMALY] = flag.astype(int)
    res[RESULT_SCORE] = np.round(score, 4)
    flags = [(rep["row_imputed"], N_("Lücken ergänzt")), (rep["row_unknown"], N_("unbekannte Kategorie")),
             (rep["row_out_of_range"], N_("außerhalb Trainingsbereich"))]
    notes = pd.Series("", index=range(len(res)), dtype=object)
    for mask, text in flags:
        if mask.any():
            notes[mask] = notes[mask].where(notes[mask] == "", notes[mask] + "; ") + text
    res[RESULT_NOTE] = notes.to_numpy()
    cmp = None
    gcol = recipe.get("group_col")
    if gcol and gcol in d.columns:  # positive Gruppe wie im Training (nicht die in den neuen Daten seltenere)
        cmp = compare_anomalies(flag, score, group_values(d[gcol]),
                                positive=(recipe.get("anomaly_metrics") or {}).get("positive"))
    return res, rep, cmp


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
def free_name(columns, name: str) -> str:
    """Spaltenname, der noch nicht vergeben ist: 'Cluster' – sonst 'Cluster_2', 'Cluster_3' …"""
    taken = {str(c) for c in columns}
    if name not in taken:
        return name
    i = 2
    while f"{name}_{i}" in taken:
        i += 1
    return f"{name}_{i}"


def cluster_ready_frame(df_viz: pd.DataFrame, prep: PrepResult, scaled: bool = False) -> pd.DataFrame:
    """Daten nach Schritt 4 als Tabelle: Füllwerte + Dummies (optional skaliert) – so sieht K-Means die Daten.
    Bezeichnung und Vergleichsgruppe stehen als zusätzliche Spalten dabei (sie gehen nicht ins Modell)."""
    r = prep.recipe
    d = df_viz.loc[prep.index]
    X_df, _ = transform_features(d, r)
    if scaled:
        X_df = pd.DataFrame(apply_scaler(X_df.to_numpy(), r["scaler"]), columns=X_df.columns, index=X_df.index)
    out = X_df.astype("float64").round(6)
    if r.get("name_col"):
        out.insert(0, free_name(out.columns, r["name_col"]), d[r["name_col"]].to_numpy())
    if r.get("group_col"):
        out[free_name(out.columns, r["group_col"])] = d[r["group_col"]].to_numpy()
    return out.reset_index(drop=True)


def clusters_frame(df_viz: pd.DataFrame, prep: PrepResult, labels: np.ndarray, distance: np.ndarray | None = None,
                   names: list | None = None, columns: dict | None = None) -> pd.DataFrame:
    """Daten nach Schritt 2 (verwendete Zeilen) + Cluster-Nummer, optional Cluster-Name und Abstand zum Zentrum.
    columns = Namen der neuen Spalten je Sprache (Standard: Cluster, Cluster-Name, Abstand)."""
    columns = {RESULT_CLUSTER: RESULT_CLUSTER, RESULT_NAME: RESULT_NAME, RESULT_DIST: RESULT_DIST, **(columns or {})}
    out = df_viz.loc[prep.index].copy()
    out[free_name(out.columns, columns[RESULT_CLUSTER])] = np.asarray(labels).astype(int)
    k = int(np.max(labels)) + 1 if len(labels) else 0
    names = clean_names(names, max(k, len(names or [])))
    if any(names):
        out[free_name(out.columns, columns[RESULT_NAME])] = [names[j] for j in labels]
    if distance is not None:
        out[free_name(out.columns, columns[RESULT_DIST])] = np.round(np.asarray(distance, dtype=float), 4)
    return out.reset_index(drop=True)


def anomalies_frame(df_viz: pd.DataFrame, prep: PrepResult, flag: np.ndarray, score: np.ndarray,
                    labels: np.ndarray | None = None, columns: dict | None = None) -> pd.DataFrame:
    """Daten nach Schritt 2 (verwendete Zeilen) + bei DBSCAN der Cluster (−1 = Rauschen), dazu Anomalie (1 = auffällig)
    und Anomalie-Score. columns = Namen der neuen Spalten je Sprache (Standard: Cluster, Anomalie, Anomalie-Score)."""
    columns = {RESULT_CLUSTER: RESULT_CLUSTER, RESULT_ANOMALY: RESULT_ANOMALY, RESULT_SCORE: RESULT_SCORE,
               **(columns or {})}
    out = df_viz.loc[prep.index].copy()
    if labels is not None:
        out[free_name(out.columns, columns[RESULT_CLUSTER])] = np.asarray(labels).astype(int)
    out[free_name(out.columns, columns[RESULT_ANOMALY])] = np.asarray(flag, dtype=bool).astype(int)
    out[free_name(out.columns, columns[RESULT_SCORE])] = np.round(np.asarray(score, dtype=float), 4)
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
# 10) Fazit: Qualität der neuen Daten und Aussage der Zuordnung
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


# ganze Sätze statt "{Strategie} der Trainingsdaten": in Englisch/Polnisch steht die Strategie dann klein
FILL_TEXT = {M_MEAN: N_("Mittelwert der Trainingsdaten ({v})"), M_MEDIAN: N_("Median der Trainingsdaten ({v})"),
             M_MODE: N_("häufigster Wert der Trainingsdaten ({v})")}


def _fill_text(recipe: dict, col: str) -> str:
    """Wie wird eine Lücke in dieser Spalte ergänzt? (für Menschen formuliert)"""
    f = recipe["fills"].get(col)
    fb = md_escape(_fmt_value(recipe["fallback_fill"].get(col)))
    if f and f["strategy"] == M_GROUP:
        return tx("Mittelwert je {group_col} aus den Trainingsdaten", group_col=md_escape(f['group_col']))
    if f and f["strategy"] == M_CONST:
        return tx("fester Wert {v}", v=md_escape(_fmt_value(f['value'])))
    if f:
        v = md_escape(_fmt_value(round(f['value'], 4) if isinstance(f['value'], float) else f['value']))
        if f["strategy"] in FILL_TEXT:
            return tx(FILL_TEXT[f["strategy"]], v=v)
        return tx("{what} der Trainingsdaten ({v})", what=md_escape(f["strategy"]), v=v)
    num = col in recipe["numeric_features"]
    if col in recipe.get("dropna_columns", []):
        return tx(N_("Median der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für "
                     "eine Zuordnung braucht aber jede Zeile einen Wert") if num else
                  N_("häufigster Wert der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier "
                     "entfernt; für eine Zuordnung braucht aber jede Zeile einen Wert"), fb=fb)
    return tx(N_("Median der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken") if num else
              N_("häufigster Wert der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken"), fb=fb)


def _data_quality(recipe: dict, n: int, rep: dict, anomaly: bool = False) -> tuple[list, str]:
    """Datenqualität der neuen Daten (Schritt 10) – für die Zuordnung (K-Means) und die Bewertung (Anomalie-Modelle).
    Rückgabe: ([{"level", "text", "details"}], Gesamtstufe)."""
    quality = []

    def add(level, text, details=None):
        quality.append({"level": level, "text": text, "details": details or []})

    extra = rep.get("extra_columns") or []
    add(LEVEL_OK, tx("{n} Zeilen; alle {v} benötigten Spalten vorhanden",
                     n=_n(n), v=rep.get('n_required', len(recipe['features'])))
        + (tx(", {n} weitere Spalten werden nicht gebraucht (ignoriert).", n=len(extra)) if extra else "."))
    n_imp = int(np.sum(rep.get("row_imputed", [])))
    if n_imp:
        share = n_imp / n
        add(_level(share, 0.05, 0.2),
            tx("Lücken in {n} Zeilen ({pct}) mit Werten aus dem Training ergänzt. Ergänzte Werte sind Schätzungen "
               "– die Zuordnung dieser Zeilen ist etwas unsicherer.",
               n=_n(n_imp), pct=_pct(share)),
            [f"{md_escape(c)}: {_n(k)} × → {_fill_text(recipe, c)}" for c, k in rep["imputed"].items()])
    else:
        add(LEVEL_OK, tx("Keine Lücken in den benötigten Spalten."))
    n_unk = int(np.sum(rep.get("row_unknown", [])))
    if n_unk:
        share = n_unk / n
        add(_level(share, 0.01, 0.05),
            tx("{n} Zeilen ({pct}) mit Kategorien, die es im Training nicht gab – dort zählen alle Dummies dieser "
               "Spalte als 0 (das Modell kennt die Kategorie nicht).",
               n=_n(n_unk), pct=_pct(share)),
            [f"{md_escape(c)}: " + ", ".join(f"{md_escape(k)} ({v})" for k, v in list(vals.items())[:5])
             for c, vals in rep["unknown"].items()])
    n_out = int(np.sum(rep.get("row_out_of_range", [])))
    if n_out:
        share = n_out / n
        add(_level(share, 0.01, 0.05),
            tx(N_("{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – solche Zeilen sind oft auffällig.")
               if anomaly else
               N_("{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – solche Zeilen liegen oft weit weg "
                  "von allen Clustern."),
               n=_n(n_out), pct=_pct(share)),
            [tx("{c}: {n} Zeile", c=md_escape(c), n=_n(k)) if k == 1 else tx("{c}: {n} Zeilen", c=md_escape(c), n=_n(k))
             for c, k in rep["out_of_range"].items()])
    if rep.get("invalid_numeric"):
        k = sum(rep["invalid_numeric"].values())
        add(_level(k / n, 0.01, 0.05), tx("{n} Werte in Zahlenspalten waren keine Zahl und wurden wie Lücken "
                                          "behandelt.",
                                          n=_n(k)), [f"{md_escape(c)}: {v}" for c, v in rep["invalid_numeric"].items()])
    return quality, max((q["level"] for q in quality), key=_LEVEL_RANK.get)


def assess_assignment(recipe: dict, res: pd.DataFrame, rep: dict, cmp: dict | None = None) -> dict:
    """Fazit zu Schritt 10.

    Rückgabe: {"level": Gesamtstufe der Datenqualität,
               "quality": [{"level", "text", "details": [Text]}], "meaning": [Text], "advice": [Text]}"""
    n = len(res)
    meaning, advice = [], []
    quality, level = _data_quality(recipe, n, rep)

    # --- Was sagt die Zuordnung? ----------------------------------------------------------------------
    lab = np.asarray(rep.get("labels", res[RESULT_CLUSTER].to_numpy()), dtype=int)
    k = len((recipe.get("train_info") or {}).get("sizes") or []) or (int(lab.max()) + 1 if len(lab) else 0)
    share_new = np.bincount(lab, minlength=k) / max(n, 1)
    share_train = np.asarray(recipe.get("cluster_share") or [], dtype=float)
    parts = []
    for j in range(k):
        txt = f"{md_escape(cluster_label(j, recipe.get('cluster_names')))}: {_pct(share_new[j])}"
        if len(share_train) == k:
            txt += tx(" (Training {pct})", pct=_pct(share_train[j]))
        parts.append(txt)
    meaning.append(tx("Verteilung auf die Cluster: ") + " · ".join(parts) + ".")
    if len(share_train) == k and n >= 20:
        diff = float(np.max(np.abs(share_new - share_train)))
        if diff > 0.15:
            meaning.append(tx("Die Verteilung weicht deutlich vom Training ab (bis {pp} Prozentpunkte) – die neuen Daten "
                              "unterscheiden sich vom Training (andere Zeit, andere Gruppe) oder sind eine Auswahl.",
                              pp=f"{diff * 100:.0f}"))
    far = np.asarray(rep.get("row_far", np.zeros(n, dtype=bool)))
    n_far = int(far.sum())
    if n_far:
        share = n_far / n
        meaning.append(tx("**{n} Zeilen ({pct}) liegen weit weg von allen Clustern** – weiter vom nächsten Zentrum "
                          "entfernt als 99 % der Trainingszeilen dieses Clusters (Hinweis 'weit weg von allen "
                          "Clustern'). Das sind Ausreißer oder ein Typ, den es im Training nicht gab.",
                          n=_n(n_far), pct=_pct(share)))
        if share > 0.05:
            level = max(level, LEVEL_WARN, key=_LEVEL_RANK.get)
            advice.append(tx("Viele Zeilen passen zu keinem Cluster: mit aktuellen Daten neu clustern (Schritt 4–6) "
                             "oder prüfen, ob die neuen Daten aus derselben Quelle stammen."))
    else:
        meaning.append(tx("Alle Zeilen liegen so nah an ihrem Cluster-Zentrum wie die Trainingszeilen – die Cluster "
                          "beschreiben auch diese Daten."))
    n_amb = int(np.sum(rep.get("row_ambiguous", [])))
    if n_amb:
        meaning.append(tx("{n} Zeilen ({pct}) liegen fast genau zwischen zwei Clustern (Hinweis 'zwischen zwei "
                          "Clustern') – ihre Zuordnung ist knapp.",
                          n=_n(n_amb), pct=_pct(n_amb / n)))

    # --- Vergleich mit der bekannten Einteilung --------------------------------------------------------
    if cmp is not None:
        gm = recipe.get("group_metrics") or {}
        txt = tx("Vergleich mit '{c}' ({n} Zeilen): Die Zuordnung Cluster → Gruppe aus dem Training trifft {pct}",
                 c=md_escape(recipe.get("group_col")), n=_n(cmp["n"]), pct=_pct(cmp["accuracy"]))
        if gm.get("accuracy") is not None:
            txt += tx(" (Training: {pct})", pct=_pct(gm["accuracy"]))
        txt += tx("; naive Vorhersage (immer '{g}'): {pct}.", g=md_escape(cmp["majority"]), pct=_pct(cmp["baseline"]))
        meaning.append(txt)
        if gm.get("accuracy") is not None and cmp["accuracy"] < gm["accuracy"] - 0.05:
            advice.append(tx("Die Cluster passen hier schlechter zur bekannten Einteilung als im Training – die neuen "
                             "Daten unterscheiden sich vermutlich."))
    if level != LEVEL_OK:
        advice.append(tx("Datenqualität verbessern: Lücken, unbekannte Kategorien und Ausreißer in der Quelle "
                         "prüfen – betroffene Zeilen stehen in der Spalte 'Hinweis'."))
    return {"level": level, "quality": quality, "meaning": meaning, "advice": advice}


def assess_scoring(recipe: dict, res: pd.DataFrame, rep: dict, cmp: dict | None = None, top: int = 5) -> dict:
    """Fazit zu Schritt 10 für die Anomalie-Modelle: Datenqualität, Anteil auffälliger Zeilen im Vergleich zum
    Training, die auffälligsten Zeilen und – wenn die neuen Daten die bekannte Einteilung enthalten – der Vergleich.

    Rückgabe wie assess_assignment: {"level", "quality", "meaning", "advice"}."""
    n = len(res)
    meaning, advice = [], []
    quality, level = _data_quality(recipe, n, rep, anomaly=True)
    key = recipe_model(recipe)
    ti = recipe.get("train_info") or {}
    flag = np.asarray(rep.get("flag", res[RESULT_ANOMALY].to_numpy()), dtype=bool)
    score = np.asarray(rep.get("score", res[RESULT_SCORE].to_numpy()), dtype="float64")
    k, share, train_share = int(flag.sum()), float(flag.mean()) if n else 0.0, ti.get("share")
    txt = tx("**{k} von {n} Zeilen auffällig ({pct})**", k=_n(k), n=_n(n), pct=share_text(share))
    if train_share is not None:
        txt += tx(" – im Training {pct}", pct=share_text(train_share))
    meaning.append(txt + tx("; Schwelle aus dem Training: Score über {thr}.",
                            thr=score_text(ti.get("threshold", 1.0 if key == MODEL_DBSCAN else float("nan")))))
    if k == 0:
        meaning.append(tx("Keine Zeile liegt über der Schwelle – die neuen Daten sehen aus wie die normalen "
                          "Trainingsdaten."))
    if train_share is not None and n >= 20 and share > max(2 * train_share, train_share + 0.02):
        meaning.append(tx("Deutlich mehr auffällige Zeilen als im Training – die neuen Daten unterscheiden sich "
                          "(anderer Zeitraum, andere Marktphase, andere Quelle) oder es gab echte Ereignisse."))
        advice.append(tx("Bei Zeitreihen: Häufen sich die Anomalien in einem Zeitraum, ist das eher ein neues Regime als "
                         "lauter Einzelfälle – dann mit neueren Daten neu trainieren (Schritt 4–6)."))
    if k:
        order = np.argsort(-score, kind="stable")[:top]
        names = rep.get("names")
        label = (lambda i: md_escape(names[i])) if names is not None else (lambda i: tx("Zeile {i}", i=int(i) + 1))
        meaning.append(tx("Am auffälligsten: ") + " · ".join(f"{label(i)} ({score_text(score[i])})" for i in order)
                       + ".")
    labels = rep.get("labels")
    if key == MODEL_DBSCAN and labels is not None:
        labels = np.asarray(labels, dtype=int)
        parts = [tx("Cluster {j}: {pct}", j=j, pct=share_text(float(np.mean(labels == j))))
                 for j in sorted(set(labels.tolist()) - {-1})]
        parts.append(tx("Rauschen: {pct}", pct=share_text(float(np.mean(labels == -1)))))
        meaning.append(tx("Verteilung (DBSCAN): ") + " · ".join(parts) + ".")
    if cmp is not None:
        g = md_escape(recipe.get("group_col"))
        if cmp.get("binary"):
            pos = md_escape(cmp["positive"])
            meaning.append(tx("Vergleich mit '{g}' ({n} Zeilen): Von den auffälligen Zeilen gehören {prec} zu '{pos}' "
                              "(Anteil insgesamt {base}); erkannt werden {rec} aller '{pos}'-Zeilen. ROC-AUC des Scores: "
                              "{auc}.", g=g, n=_n(cmp["n"]), pos=pos,
                              prec="–" if np.isnan(cmp["precision"]) else share_text(cmp["precision"]),
                              base=share_text(cmp["base_rate"]), rec=share_text(cmp["recall"]),
                              auc=f"{cmp['roc_auc']:.3f}"))
        else:
            parts = [f"{md_escape(r['Gruppe'])}: {share_text(r['Anteil auffällig'])}"
                     for r in cmp["table"].to_dict("records")]
            meaning.append(tx("Anteil auffälliger Zeilen je '{g}': ", g=g) + " · ".join(parts) + ".")
    if level != LEVEL_OK:
        advice.append(tx("Datenqualität verbessern: Lücken, unbekannte Kategorien und Ausreißer in der Quelle "
                         "prüfen – betroffene Zeilen stehen in der Spalte 'Hinweis'."))
    return {"level": level, "quality": quality, "meaning": meaning, "advice": advice}
