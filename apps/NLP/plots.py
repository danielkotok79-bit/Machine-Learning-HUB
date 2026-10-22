"""
plots.py – Diagramme (matplotlib/seaborn) für die Schritte 3, 6, 7, 8 und 10 der Textklassifikation.

Jede Funktion gibt eine matplotlib-Figure zurück:
    Streamlit:  st.pyplot(fig)        Jupyter:  fig   (oder fig.savefig("bild.png"))

Farben: feste, farbfehlsichtigkeits-geprüfte Reihenfolge (SERIES). Wörter, die für eine Klasse sprechen, sind blau,
Wörter dagegen orange; Korrelationen nutzen eine divergierende Skala Blau ↔ Grau ↔ Rot mit 0 in der Mitte.
"""
from __future__ import annotations

import math
import re
import warnings
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter, PercentFormatter

from i18n import N_, tx  # Beschriftungen in der gewählten Sprache

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
OTHER_COLOR = "#a8a69f"  # Sammelgruppe "Andere"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blau", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blau_rot", ["#0d366b", "#2a78d6", NEUTRAL, "#e34948", "#8f1f1e"])
FOR, AGAINST = SERIES[0], SERIES[1]  # Wort spricht für die Klasse (blau) bzw. dagegen (orange)
OTHER = N_("Andere")  # Sammelgruppe – angezeigt in der gewählten Sprache (tx(OTHER))
MAX_PANELS = 12  # höchstens so viele Teilgrafiken (Klassen) je Diagramm


def use_style() -> None:
    sns.set_theme(style="whitegrid", rc={
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "lines.linewidth": 2, "legend.frameon": False, "font.size": 10, "figure.dpi": 100,
        "text.parse_math": False,  # '$' aus Texten und Spaltennamen wörtlich ('$5 off' wäre sonst eine Formel)
    })
    plt.rcParams["axes.prop_cycle"] = cycler(color=SERIES)


use_style()


@contextmanager
def _quiet():
    """Veraltungs-Hinweise aus seaborn 0.13 unterdrücken (sns.boxplot ruft noch ax.boxplot(vert=…) auf, sns.heatmap
    cmap.set_bad) – für die Diagramme ohne Bedeutung, sie würden nur das Protokoll füllen."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="vert: bool was deprecated")
        warnings.filterwarnings("ignore", message="The set_bad function will be deprecated")
        yield


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _natural_key(v) -> list:
    """'2 years' < '10+ years', 'A2' < 'A10'."""
    return [(0, int(t), "") if t.isdigit() else (1, 0, t.lower()) for t in re.split(r"(\d+)", str(v))]

def _as_cat(s: pd.Series) -> pd.Series:
    """Werte als Text (für Achsen/Farben); ganze Zahlen ohne '.0'."""
    if pd.api.types.is_float_dtype(s):
        v = s.dropna().to_numpy(dtype="float64")
        if len(v) and np.isfinite(v).all() and (np.abs(v) < 2**53).all() and (np.mod(v, 1) == 0).all():
            return s.astype("Int64").astype(str).where(s.notna(), np.nan)
    return s.astype(str).where(s.notna(), np.nan)

def fold(s: pd.Series, max_groups: int = 8) -> pd.Series:
    """Für Farben höchstens max_groups Gruppen: seltene Werte → 'Andere'."""
    s = _as_cat(s)
    vc = s.value_counts()
    if len(vc) <= max_groups:
        return s
    keep = vc.index[: max_groups - 1]
    return s.where(s.isin(keep) | s.isna(), tx(OTHER))

def _levels(s: pd.Series) -> list:
    levels = sorted(s.dropna().unique(), key=_natural_key)
    other = tx(OTHER)
    if other in levels:
        levels.remove(other)
        levels.append(other)
    return levels

def _palette(levels: list) -> dict:
    pal, i = {}, 0
    for lv in levels:
        if lv == tx(OTHER):
            pal[lv] = OTHER_COLOR
        else:
            pal[lv] = SERIES[i % len(SERIES)]
            i += 1
    return pal

def _rotate(ax, n: int) -> None:
    if n > 8:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def _sample(d: pd.DataFrame, n: int, seed: int = 101) -> pd.DataFrame:
    return d.sample(n, random_state=seed) if len(d) > n else d


def _short(text, n: int = 40) -> str:
    """Lange Klassen- oder Gruppennamen für Titel kürzen."""
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _grid(n: int, ncol_max: int = 3, width: float = 4.4, height: float = 3.0):
    """Teilgrafiken für n Gruppen: höchstens ncol_max nebeneinander. Rückgabe: (fig, Liste der Achsen)."""
    ncol = max(1, min(n, ncol_max))
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(min(width * ncol, 15), height * nrow + 0.4), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.axis("off")
    return fig, flat[:n]


# ---------------------------------------------------------------------------
# 3) Analyse & Visualisierung
# ---------------------------------------------------------------------------
def _empty_fig(title: str, text: str | None = None):
    text = tx("Keine Werte zum Anzeigen (Spalte leer).") if text is None else text
    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=11, transform=ax.transAxes)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig

def _legend_right(ax, title):
    """Legende rechts neben das Diagramm – nur wenn es eine gibt (z. B. nicht bei leeren Daten)."""
    if ax.get_legend() is not None:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=title)

def fig_missing(df: pd.DataFrame):
    pct = (df.isna().mean() * 100).sort_values()
    pct = pct[pct > 0]
    if pct.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(pct) + 1.3))
    ax.barh(pct.index.astype(str), pct.values, color=SERIES[0], height=0.62)
    for y, v in enumerate(pct.values):
        ax.text(v, y, f"  {v:.2f} %", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, pct.max() * 1.25)
    ax.set_xlabel(tx("fehlend in %"))
    ax.set_title(tx("Fehlende Werte je Spalte"))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig

def fig_count(df: pd.DataFrame, x: str, hue: str | None = None, order_by: str = "Wert", top_n: int = 40):
    hue = None if hue == x else hue
    d = pd.DataFrame({x: _as_cat(df[x])})
    vc = d[x].value_counts()
    if vc.empty:
        return _empty_fig(tx("Anzahl je {x}", x=x))
    note = ""
    if len(vc) > top_n:
        d = d[d[x].isin(vc.index[:top_n])]
        note = tx(" – die {top_n} häufigsten von {n}", top_n=top_n, n=len(vc))
    order = (sorted(d[x].dropna().unique(), key=_natural_key) if order_by == "Wert"
             else list(vc.index[:top_n]))  # nach Häufigkeit (vc ist schon absteigend sortiert)
    kw = {}
    if hue:
        d[hue] = fold(df.loc[d.index, hue])
        levels = _levels(d[hue])
        kw = dict(hue=hue, hue_order=levels, palette=_palette(levels))
    else:
        kw = dict(color=SERIES[0])
    fig, ax = plt.subplots(figsize=(min(max(7, 0.38 * len(order) + 2.5), 18), 4.4))
    sns.countplot(data=d, x=x, order=order, ax=ax, saturation=1, edgecolor=SURFACE, linewidth=1, **kw)
    ax.set_title(tx("Anzahl je {x}{note}", x=x, note=note))
    ax.set_ylabel(tx("Anzahl"))
    ax.grid(axis="x", visible=False)
    _rotate(ax, len(order))
    if hue:
        _legend_right(ax, hue)
    fig.tight_layout()
    return fig

def fig_hist(df: pd.DataFrame, x: str, hue: str | None = None, bins: int = 40, kde: bool = False,
             clip: bool = False):
    hue = None if hue == x else hue
    d = df[[x] + ([hue] if hue else [])].dropna(subset=[x]).copy()
    title = tx("Verteilung von {x}", x=x)
    finite = np.isfinite(d[x].to_numpy(dtype="float64"))
    if not finite.all():  # ±inf lässt sich nicht in Klassen einteilen
        title += tx(" – {n} × inf ausgelassen", n=int((~finite).sum()))
        d = d[finite]
    if d.empty:
        return _empty_fig(title)
    if clip and len(d):
        lo, hi = d[x].quantile([0.01, 0.99])
        d = d[d[x].between(lo, hi)]
        title += tx(" (1.–99. Perzentil)")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    if hue:
        d[hue] = fold(d[hue])
        levels = _levels(d[hue])
        sns.histplot(data=d, x=x, hue=hue, hue_order=levels, palette=_palette(levels), bins=bins, kde=kde,
                     element="step", fill=False, linewidth=1.8, common_norm=False, ax=ax)
        _legend_right(ax, hue)
    else:
        sns.histplot(data=d, x=x, bins=bins, kde=kde, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6,
                     alpha=0.9, ax=ax)
    ax.set_title(title)
    ax.set_ylabel(tx("Anzahl"))
    fig.tight_layout()
    return fig

def fig_box(df: pd.DataFrame, x: str, y: str, hue: str | None = None, top_n: int = 30):
    if x == y:
        raise ValueError(tx("Für den Boxplot bitte zwei verschiedene Spalten wählen."))
    d = df[[x, y] + ([hue] if hue and hue not in (x, y) else [])].dropna(subset=[x, y]).copy()
    if d.empty:
        return _empty_fig(tx("{y} je {x}", y=y, x=x), tx("Keine Zeilen mit Werten in beiden Spalten."))
    d[x] = _as_cat(d[x])
    vc = d[x].value_counts()
    if len(vc) > top_n:
        d = d[d[x].isin(vc.index[:top_n])]
    order = sorted(d[x].unique(), key=_natural_key)
    fig, ax = plt.subplots(figsize=(min(max(7, 0.5 * len(order) + 3), 18), 4.6))
    kw = dict(color=SERIES[0])
    if hue and hue not in (x, y):
        d[hue] = fold(d[hue], 4)
        levels = _levels(d[hue])
        kw = dict(hue=hue, hue_order=levels, palette=_palette(levels))
    with _quiet():
        sns.boxplot(data=d, x=x, y=y, order=order, fliersize=1.5, linewidth=1, saturation=1, ax=ax, **kw)
    if "hue" in kw:
        _legend_right(ax, hue)
    ax.set_title(tx("{y} je {x}", y=y, x=x))
    ax.grid(axis="x", visible=False)
    _rotate(ax, len(order))
    fig.tight_layout()
    return fig

def corr_matrix(df: pd.DataFrame, cols: list | None = None, method: str = "pearson") -> pd.DataFrame:
    num = df.select_dtypes(include="number")
    if cols:
        num = num[[c for c in cols if c in num.columns]]
    return num.corr(method=method)

def fig_corr_heatmap(corr: pd.DataFrame, annot: bool = True, triangle: bool = False):
    n = len(corr)
    fig, ax = plt.subplots(figsize=(min(1.5 + 0.62 * n, 20), min(1.0 + 0.5 * n, 16)))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1) if triangle else None
    with _quiet():
        sns.heatmap(corr, mask=mask, cmap=DIV, vmin=-1, vmax=1, center=0, annot=annot and n <= 25, fmt=".2f",
                    annot_kws={"size": 8}, linewidths=0.6, linecolor=SURFACE,
                    cbar_kws={"shrink": 0.7, "label": tx("Korrelation")}, ax=ax)
    ax.set_title(tx("Korrelationsmatrix (−1 blau · 0 grau · +1 rot)"))
    fig.tight_layout()
    return fig


def fig_text_length(lengths, groups=None, unit: str = N_("Zeichen"), bins: int = 50, clip: bool = True,
                    group_name: str = ""):
    """Länge der Texte – wie im Kurs: sns.FacetGrid(yelp, col='stars').map(plt.hist, 'text length').
    Eine Teilgrafik je Gruppe (höchstens 8, seltene → 'Andere') mit gleichen Klassen (bins) für alle Gruppen."""
    d = pd.DataFrame({"n": np.asarray(lengths, dtype="float64")})
    title = tx("Textlänge in {unit}", unit=tx(unit))
    if groups is not None:
        d["g"] = fold(pd.Series(list(groups)), 8).to_numpy()
        d = d.dropna(subset=["g"])
    d = d.dropna(subset=["n"])
    if d.empty:
        return _empty_fig(title)
    hi = float(d["n"].quantile(0.99)) if clip else float(d["n"].max())
    note = tx(" (bis zum 99. Perzentil: {v:.0f})", v=hi) if clip and hi < d["n"].max() else ""
    edges = np.histogram_bin_edges(d["n"].clip(upper=hi), bins=bins)
    if groups is None:
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.hist(d["n"].clip(upper=hi), bins=edges, color=SERIES[0], edgecolor=SURFACE, linewidth=0.5)
        ax.set_title(title + note)
        ax.set_xlabel(tx(unit))
        ax.set_ylabel(tx("Anzahl"))
        fig.tight_layout()
        return fig
    levels = _levels(d["g"])
    pal = _palette(levels)
    fig, axes = _grid(len(levels), 4, 3.6, 2.7)
    for ax, lv in zip(axes, levels):
        v = d.loc[d["g"] == lv, "n"]
        ax.hist(v.clip(upper=hi), bins=edges, color=pal[lv], edgecolor=SURFACE, linewidth=0.4)
        ax.set_title(_short(f"{group_name} = {lv}" if group_name else lv, 30), fontsize=10)
        ax.text(0.98, 0.95, tx("n = {n}\nMedian {m:.0f}", n=len(v), m=float(v.median())), transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlabel(tx(unit), fontsize=9)
    fig.suptitle(title + note, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_top_terms(terms: dict, title: str, top: int = 20, group_name: str = ""):
    """Häufigste Wörter – eine Teilgrafik je Gruppe (waagerechte Balken). terms = pipeline.top_terms()."""
    groups = [g for g, t in terms.items() if len(t)][:MAX_PANELS]
    if not groups:
        return _empty_fig(title, tx("Keine Wörter gefunden (nur Stoppwörter oder einzelne Zeichen?)."))
    n_words = max(len(terms[g].head(top)) for g in groups)
    fig, axes = _grid(len(groups), 3, 4.6, 0.24 * n_words + 1.2)
    pal = _palette(groups)
    for ax, g in zip(axes, groups):
        t = terms[g].head(top).iloc[::-1]
        ax.barh(t["Wort"].astype(str), t["Anzahl"], color=pal[g] if g else SERIES[0], height=0.65)
        ax.set_title(_short(f"{group_name} = {g}" if group_name and g else g or tx("alle Texte"), 34), fontsize=10)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(tx("Anzahl"), fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6) Modellvergleich
# ---------------------------------------------------------------------------
def fig_compare(table: pd.DataFrame, names: list):
    """Die drei Modelle auf den Testdaten: Accuracy, F1 (macro) und ROC-AUC nebeneinander (größer = besser)."""
    metrics = [c for c in ("Accuracy", "F1 (macro)", "ROC-AUC") if c in table and table[c].notna().any()]
    x = np.arange(len(table))
    w = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    values = []
    for i, m in enumerate(metrics):
        v = table[m].astype(float).to_numpy()
        values += [float(a) for a in v if np.isfinite(a)]
        bars = ax.bar(x + (i - (len(metrics) - 1) / 2) * w, v, width=w * 0.92, color=SERIES[i], label=m)
        ax.bar_label(bars, labels=[f"{a:.3f}" if np.isfinite(a) else "" for a in v], fontsize=8, padding=2,
                     color=INK2)
    lo = min(values) if values else 0.0
    ax.set_ylim(max(0.0, lo - 0.08), 1.02)
    ax.set_xticks(x, [_short(n, 32) for n in names])
    ax.set_title(tx("Modellvergleich auf den Testdaten (größer = besser)"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7) Training vs. Validierung
# ---------------------------------------------------------------------------
LABEL_BOX = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85)  # Text über Linien lesbar


def fig_curve(curve: dict, metric_label: str):
    """Fehlerrate auf Training und Validierung je Wert von alpha bzw. C (logarithmische Achse) – senkrechte Linie =
    eingestellter Wert, Punkt = bester Wert, gestrichelt = naive Vorhersage."""
    x = np.asarray(curve["values"], dtype="float64")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, curve["train"], color=SERIES[0], marker="o", markersize=3.5, label=tx("Training"))
    if curve["val"]:
        ax.plot(x[:len(curve["val"])], curve["val"], color=SERIES[1], marker="o", markersize=3.5,
                label=tx("Validierung"))
    if curve.get("baseline") is not None:
        ax.axhline(curve["baseline"], color=MUTED, linewidth=1, linestyle="--", label=tx("naive Vorhersage"))
    ax.set_xscale("log")  # Zahlen statt 10^x (text.parse_math=False zeigte sonst '$\mathdefault…$')
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.xaxis.set_minor_formatter(NullFormatter())
    _rotate(ax, len(x) + 3)
    chosen = curve.get("chosen")
    if chosen is not None:
        ax.axvline(chosen, color=INK2, linewidth=1)
        left = list(x).index(chosen) > 0.75 * (len(x) - 1) if chosen in list(x) else False
        ax.annotate(tx("eingestellt") + f" ({chosen:g})", (chosen, 1), xycoords=("data", "axes fraction"),
                    xytext=(-4 if left else 4, -14), textcoords="offset points", ha="right" if left else "left",
                    fontsize=9, color=INK2, bbox=LABEL_BOX)
    j = curve.get("best_index")
    if j is not None and j < len(curve["val"]):
        ax.scatter([x[j]], [curve["val"][j]], s=52, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        right = j > 0.7 * len(x)
        ax.annotate(tx("bester Wert: {v}", v=f"{x[j]:g}"), (x[j], curve["val"][j]), xytext=(-6 if right else 6, 9),
                    textcoords="offset points", ha="right" if right else "left", color=INK2, fontsize=9,
                    bbox=LABEL_BOX, zorder=4)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0 if max(curve["train"] + curve["val"]) > 0.05
                                                  else 1))
    ax.set_xlabel(curve["param"])
    ax.set_ylabel(metric_label)
    ax.set_title(tx("{metric} je {param}", metric=metric_label, param=curve["param"]))
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8) Evaluation und Modell verstehen
# ---------------------------------------------------------------------------
CM_MAX_INCHES = (13.0, 11.5)  # Obergrenze der Größe (50 Klassen ergaben sonst 68 × 57 Zoll)


def fig_confusion(cm: np.ndarray, labels: list):
    """Confusion Matrix: Anzahl und Anteil je Zeile; ab 13 Klassen nur die Anzahl, ab 26 Klassen nur Farben (die
    Zahlen stehen in der Tabelle), die Größe ist begrenzt."""
    k = len(labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_pct = cm / cm.sum(axis=1, keepdims=True)
    if k <= 12:
        annot = np.array([[f"{cm[i, j]}\n{row_pct[i, j]:.0%}" if np.isfinite(row_pct[i, j]) else f"{cm[i, j]}"
                           for j in range(k)] for i in range(k)])
    elif k <= 25:
        annot = np.array([[f"{cm[i, j]}" for j in range(k)] for i in range(k)])
    else:
        annot = False
    size = (min(1.3 * k + 2.6, CM_MAX_INCHES[0]), min(1.1 * k + 1.8, CM_MAX_INCHES[1]))
    names = [_short(lb, 24) for lb in labels]
    fig, ax = plt.subplots(figsize=size)
    with _quiet():
        sns.heatmap(cm, annot=annot, fmt="", cmap=SEQ, cbar=k > 25, xticklabels=names, yticklabels=names,
                    linewidths=1.5 if k <= 25 else 0.3, linecolor=SURFACE, annot_kws={"size": 10 if k <= 12 else 7},
                    ax=ax)
    ax.set_xlabel(tx("Vorhersage"))
    ax.set_ylabel(tx("Tatsächlich"))
    ax.set_title(tx("Confusion Matrix (Anzahl, Anteil je Zeile)") if k <= 12 else tx("Confusion Matrix (Anzahl)"))
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10 if k <= 12 else 7)
    if k > 8:
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=10 if k <= 12 else 7)
    fig.tight_layout()
    return fig

def fig_proba(scores: np.ndarray, classes: list, y_true: np.ndarray | None = None, threshold: float = 0.5):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    bins = np.linspace(0, 1, 41)
    if y_true is not None:
        for k, cl in enumerate(classes):
            ax.hist(scores[y_true.astype(int) == k], bins=bins, histtype="step", linewidth=2,
                    color=SERIES[k], label=tx("tatsächlich: {cl}", cl=cl))
        ax.legend(loc="upper left")
    else:
        ax.hist(scores, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(threshold, color=INK2, linewidth=1.2)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, tx("  Schwelle {threshold:.2f}", threshold=threshold), color=INK2,
            fontsize=9, va="top")
    ax.set_xlabel(tx("vorhergesagte Wahrscheinlichkeit für '{v}'", v=classes[1]))
    ax.set_ylabel(tx("Anzahl"))
    ax.set_title(tx("Verteilung der Wahrscheinlichkeiten"))
    fig.tight_layout()
    return fig

def fig_top_words(tw: dict, classes: list, top: int = 15):
    """Wörter mit dem größten Gewicht je Klasse (pipeline.top_words) – eine Teilgrafik je Klasse."""
    kind = tw["kind"]
    shown = [c for c in classes if c in tw["classes"]][:MAX_PANELS]
    n_words = max((len(tw["classes"][c].head(top)) for c in shown), default=1)
    fig, axes = _grid(len(shown), 2 if len(shown) == 2 else 3, 4.8, 0.24 * n_words + 1.3)
    for ax, cl in zip(axes, shown):
        t = tw["classes"][cl].head(top).iloc[::-1]
        w = t["Gewicht"].to_numpy(dtype="float64")
        ax.barh(t["Wort"].astype(str), w, color=[FOR if v > 0 else AGAINST for v in w], height=0.65)
        ax.axvline(0, color=AXIS, linewidth=1)
        ax.set_title(tx("spricht für '{cl}'", cl=_short(cl, 28)), fontsize=10)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(tx("Log-Verhältnis (Naive Bayes)") if kind == "nb" else tx("Koeffizient"), fontsize=9)
    title = tx("Wichtigste Wörter je Klasse")
    if len(classes) > len(shown):
        title += tx(" – die ersten {n} von {m} Klassen", n=len(shown), m=len(classes))
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def fig_contrib(table: pd.DataFrame, label: str, top: int = 12):
    """Beitrag der Wörter eines Textes zur vorhergesagten Klasse (pipeline.explain_text) – blau spricht dafür,
    orange dagegen."""
    t = table.head(top).iloc[::-1]
    if t.empty:
        return _empty_fig(tx("Welche Wörter haben entschieden?"),
                          tx("Kein Wort dieses Textes steht im Vokabular – das Modell entscheidet nur nach der "
                             "Häufigkeit der Klassen."))
    v = t["Beitrag"].to_numpy(dtype="float64")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(t) + 1.5))
    ax.barh(t["Wort"].astype(str), v, color=[FOR if a > 0 else AGAINST for a in v], height=0.62)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_title(tx("Welche Wörter haben entschieden? → '{label}'", label=_short(label, 30)))
    ax.set_xlabel(tx("Beitrag (blau: spricht für '{label}', orange: dagegen)", label=_short(label, 24)))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10) Vorhersagen
# ---------------------------------------------------------------------------
def fig_class_counts(pred: pd.Series, train_share: dict | None = None):
    counts = pred.value_counts()
    labels = sorted(set(counts.index) | set(train_share or {}), key=_natural_key)
    share_new = np.array([counts.get(lb, 0) for lb in labels]) / max(len(pred), 1)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(min(max(6, 1.2 * len(labels) + 3), 16), 3.8))
    w = 0.4 if train_share else 0.6
    ax.bar(x - (w / 2 if train_share else 0), share_new * 100, width=w, color=SERIES[0], label=tx("Vorhersage neue Daten"))
    if train_share:
        ax.bar(x + w / 2, [train_share.get(lb, 0) * 100 for lb in labels], width=w, color=SERIES[1],
               label=tx("Anteil im Training"))
        ax.legend(loc="upper right")
    ax.set_xticks(x, [_short(lb, 24) for lb in labels])
    _rotate(ax, len(labels))  # viele Klassen: Beschriftungen schräg statt übereinander
    ax.set_ylabel(tx("Anteil in %"))
    ax.set_title(tx("Vorhergesagte Klassen"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
