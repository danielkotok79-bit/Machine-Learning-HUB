"""
plots.py – Diagramme (matplotlib/seaborn) für die Schritte 3, 7, 8 und 10 (scikit-learn-Modelle).

Jede Funktion gibt eine matplotlib-Figure zurück:
    Streamlit:  st.pyplot(fig)        Jupyter:  fig   (oder fig.savefig("bild.png"))

Farben: feste, farbfehlsichtigkeits-geprüfte Reihenfolge (SERIES). Korrelationen nutzen eine
divergierende Skala Blau ↔ Grau ↔ Rot mit 0 in der Mitte, Mengen eine einfarbige Blau-Skala.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, PercentFormatter

from i18n import N_, get_lang, tx  # Beschriftungen in der gewählten Sprache

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
OTHER_COLOR = "#a8a69f"  # Sammelgruppe "Andere"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blau", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blau_rot", ["#0d366b", "#2a78d6", NEUTRAL, "#e34948", "#8f1f1e"])
POS, NEG = "#e34948", "#2a78d6"  # positive Korrelation rot, negative blau (wie in der Heatmap)
OTHER = N_("Andere")  # Sammelgruppe – angezeigt in der gewählten Sprache (tx(OTHER))


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
        "text.parse_math": False,  # '$' aus Spaltennamen wörtlich ('Preis ($) gegen Kosten ($)' wäre sonst eine Formel)
    })
    plt.rcParams["axes.prop_cycle"] = cycler(color=SERIES)


use_style()


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


def binary_target(s: pd.Series, positive) -> pd.Series:
    """Ziel → 0/1 (1 = positive Klasse), NaN bleibt NaN."""
    return (_as_cat(s) == str(positive)).astype(float).where(s.notna())


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
    sns.boxplot(data=d, x=x, y=y, order=order, fliersize=1.5, linewidth=1, saturation=1, ax=ax, **kw)
    if "hue" in kw:
        _legend_right(ax, hue)
    ax.set_title(tx("{y} je {x}", y=y, x=x))
    ax.grid(axis="x", visible=False)
    _rotate(ax, len(order))
    fig.tight_layout()
    return fig


def fig_scatter(df: pd.DataFrame, x: str, y: str, hue: str | None = None, max_points: int = 5000):
    cols = list(dict.fromkeys([x, y] + ([hue] if hue and hue not in (x, y) else [])))
    d = _sample(df[cols].dropna(subset=[x, y]), max_points).copy()
    if d.empty:
        return _empty_fig(tx("{y} gegen {x}", y=y, x=x), tx("Keine Zeilen mit Werten in beiden Spalten."))
    fig, ax = plt.subplots(figsize=(8, 5))
    kw = dict(color=SERIES[0])
    if hue and hue not in (x, y):
        d[hue] = fold(d[hue], 4)  # Punktwolken: max. 3 Farben + "Andere"
        levels = _levels(d[hue])
        kw = dict(hue=hue, hue_order=levels, palette=_palette(levels))
    sns.scatterplot(data=d, x=x, y=y, s=14, alpha=0.6, edgecolor=SURFACE, linewidth=0.3, ax=ax, **kw)
    if "hue" in kw:
        _legend_right(ax, hue)
    ax.set_title(tx("{y} gegen {x}", y=y, x=x)
                 + (tx(" ({n} Punkte, Stichprobe)", n=len(d)) if len(df) > max_points else ""))
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
    sns.heatmap(corr, mask=mask, cmap=DIV, vmin=-1, vmax=1, center=0, annot=annot and n <= 25, fmt=".2f",
                annot_kws={"size": 8}, linewidths=0.6, linecolor=SURFACE,
                cbar_kws={"shrink": 0.7, "label": tx("Korrelation")}, ax=ax)
    ax.set_title(tx("Korrelationsmatrix (−1 blau · 0 grau · +1 rot)"))
    fig.tight_layout()
    return fig


def corr_with_target(df: pd.DataFrame, target: str, positive=None, method: str = "pearson") -> pd.Series:
    y = df[target]
    if not pd.api.types.is_numeric_dtype(y):
        y = binary_target(y, positive)
    num = df.select_dtypes(include="number").drop(columns=[target], errors="ignore")
    return num.corrwith(y.astype(float), method=method).dropna().sort_values()


def fig_corr_target(corr: pd.Series, target: str, positive=None, top: int | None = 30):
    """Korrelation jeder Spalte mit einer Spalte als Balken. top: nur die stärksten (|r|) zeichnen – bei breiten Daten
    (z. B. 160 Spalten des Sportdaten-Tools) entstünde sonst ein meterhohes Bild, das bei jedem Klick Sekunden kostet;
    die Tabelle daneben in der App zeigt alle."""
    total = len(corr)
    if top and total > top:
        corr = corr.loc[corr.abs().sort_values(ascending=False).index[:top]].sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(corr) + 1.4))
    ax.barh(corr.index.astype(str), corr.values, color=[POS if v > 0 else NEG for v in corr.values], height=0.62)
    ax.axvline(0, color=AXIS, linewidth=1)
    lim = max(abs(corr).max() * 1.15, 0.05) if len(corr) else 1
    ax.set_xlim(-lim, lim)
    label = f"{target} = '{positive}'" if positive is not None else target
    title = tx("Korrelation mit {label}", label=label)
    if len(corr) < total:
        title += tx(" – die {n} stärksten von {m}", n=len(corr), m=total)
    ax.set_title(title)
    ax.set_xlabel(tx("Korrelation (rot positiv, blau negativ)"))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_pairplot(df: pd.DataFrame, cols: list, hue: str | None = None, max_points: int = 1500):
    use = list(dict.fromkeys(list(cols) + ([hue] if hue else [])))
    d = _sample(df[use].dropna(), max_points).copy()
    if d.empty:
        return _empty_fig(tx("Pairplot"), tx("Keine Zeile ohne Lücken in den gewählten Spalten."))
    if hue:
        hue_col = f"{hue} " if hue in cols else hue  # Farbe als eigene (Text-)Spalte
        d[hue_col] = fold(d[hue], 4)
        levels = _levels(d[hue_col])
        g = sns.pairplot(d, vars=cols, hue=hue_col, hue_order=levels, palette=_palette(levels), corner=True,
                         diag_kind="hist", height=2.1,
                         plot_kws={"s": 10, "alpha": 0.6, "edgecolor": SURFACE, "linewidth": 0.2},
                         diag_kws={"element": "step", "fill": False, "linewidth": 1.5})
    else:
        g = sns.pairplot(d, vars=cols, corner=True, diag_kind="hist", height=2.1,
                         plot_kws={"s": 10, "alpha": 0.6, "color": SERIES[0], "edgecolor": SURFACE, "linewidth": 0.2},
                         diag_kws={"color": SERIES[0], "edgecolor": SURFACE})
    g.figure.suptitle(tx("Pairplot ({n} Zeilen)", n=len(d)), x=0.02, ha="left", fontweight="bold", color=INK)
    return g.figure


def target_rate(df: pd.DataFrame, cat: str, target: str, positive, top_n: int = 40) -> pd.DataFrame:
    """Anteil der positiven Klasse je Kategorie (z. B. Ausfallquote je emp_length)."""
    d = pd.DataFrame({cat: _as_cat(df[cat]), "y": binary_target(df[target], positive)}).dropna()
    stats = d.groupby(cat)["y"].agg(["mean", "count"]).rename(columns={"mean": "Anteil", "count": "Anzahl"})
    stats = stats.sort_values("Anzahl", ascending=False).head(top_n)
    return stats.loc[sorted(stats.index, key=_natural_key)]


def fig_target_rate(stats: pd.DataFrame, cat: str, target: str, positive, overall: float):
    fig, ax = plt.subplots(figsize=(min(max(7, 0.4 * len(stats) + 2.5), 18), 4.4))
    ax.bar(stats.index.astype(str), stats["Anteil"] * 100, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.axhline(overall * 100, color=INK2, linewidth=1)
    ax.text(len(stats) - 0.5, overall * 100, tx(" gesamt {overall:.1%}", overall=overall), va="bottom", ha="right",
            color=INK2, fontsize=9)
    ax.set_ylabel(tx("Anteil '{positive}' in %", positive=positive))
    ax.set_title(tx("Anteil {target} = '{positive}' je {cat}", target=target, positive=positive, cat=cat))
    ax.grid(axis="x", visible=False)
    _rotate(ax, len(stats))
    fig.tight_layout()
    return fig


# --- Jointplot, Dichte 2D (KDE), Regression (lmplot), Histogramm je Gruppe ---------------------------------------
JOINT_KINDS = ["scatter", "kde", "hex", "reg"]  # Arten von sns.jointplot (Farbe/hue nur bei scatter und kde)


def _finite_xy(df: pd.DataFrame, x: str, y: str, extra=()) -> pd.DataFrame:
    """Zeilen mit endlichen Zahlen in x und y (NaN und ±inf fallen weg), dazu weitere Spalten (z. B. hue)."""
    if x == y:
        raise ValueError(tx("Bitte zwei verschiedene Spalten wählen."))
    cols = list(dict.fromkeys([x, y, *[c for c in extra if c]]))
    d = df[cols].copy()
    d[x], d[y] = d[x].astype("float64"), d[y].astype("float64")
    return d[np.isfinite(d[[x, y]].to_numpy()).all(axis=1)]


def category_values(s: pd.Series) -> list:
    """Werte einer Spalte als Text, natürlich sortiert – für Auswahllisten (z. B. Filter)."""
    return sorted(_as_cat(s).dropna().unique(), key=_natural_key)


def filter_rows(df: pd.DataFrame, col: str | None, value) -> pd.DataFrame:
    """Nur Zeilen mit col = value (Vergleich als Text, wie in den Auswahllisten) – z. B. species = setosa."""
    return df if not col else df[_as_cat(df[col]) == str(value)]


def _kde_possible(d: pd.DataFrame, x: str, y: str) -> bool:
    """Eine 2D-Dichte braucht Streuung in x und y und keinen exakt linearen Zusammenhang (sonst singulär)."""
    if len(d) < 3 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return False
    r = np.corrcoef(d[x].to_numpy(), d[y].to_numpy())[0, 1]
    return bool(np.isfinite(r) and abs(r) < 0.9999)


def _suptitle(fig, title: str, space: float = 0.5) -> None:
    """Titel über Figure-Diagrammen (JointGrid, FacetGrid), links wie die anderen Titel. Oben wird Platz frei
    gemacht (space in Zoll), damit der Titel auch bei fig.savefig() ohne bbox_inches='tight' in der Figure liegt."""
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - space / h)
    fig.suptitle(title, x=0.02, y=1 - 0.08 / h, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)


def fig_joint(df: pd.DataFrame, x: str, y: str, kind: str = "scatter", hue: str | None = None,
              max_points: int = 5000):
    """sns.jointplot: Punkte (oder Dichte/Waben/Regression) mit den Verteilungen von x und y am Rand."""
    kind = kind if kind in JOINT_KINDS else "scatter"
    hue = hue if hue and hue not in (x, y) and kind in ("scatter", "kde") else None
    d = _finite_xy(df, x, y, [hue])
    if hue:
        d = d.dropna(subset=[hue])
    sampled = len(d) > max_points
    d = _sample(d, max_points).copy()
    title = tx("{y} gegen {x}", y=y, x=x) + (tx(" ({n} Punkte, Stichprobe)", n=len(d)) if sampled else "")
    if len(d) < 2:
        return _empty_fig(title, tx("Keine Zeilen mit Werten in beiden Spalten."))
    if kind == "kde" and not _kde_possible(d, x, y):
        return _empty_fig(title, tx("Zu wenige oder zu gleichförmige Werte für eine Dichte-Schätzung (KDE)."))
    kw = dict(color=SERIES[0])
    if hue:
        d[hue] = fold(d[hue], 8)
        levels = _levels(d[hue])
        kw = dict(hue=hue, hue_order=levels, palette=_palette(levels))
    joint_kws = {"scatter": dict(s=14, alpha=0.6, edgecolor=SURFACE, linewidth=0.3), "kde": {},
                 "hex": dict(gridsize=35), "reg": dict(scatter_kws=dict(s=12, alpha=0.5))}[kind]
    g = sns.jointplot(data=d, x=x, y=y, kind=kind, height=6, joint_kws=joint_kws, **kw)
    g.ax_joint.xaxis.set_major_locator(plt.MaxNLocator(6))  # große Zahlen (z. B. 40000) überlappen sonst
    _suptitle(g.figure, title)
    return g.figure


def fig_kde2d(df: pd.DataFrame, x: str, y: str, filter_col: str | None = None, filter_value=None,
              fill: bool = True, max_points: int = 3000):
    """2D-Dichte (sns.kdeplot mit x und y), optional nur für eine Gruppe – wie iris[iris['species'] == 'setosa']."""
    d = _finite_xy(filter_rows(df, filter_col, filter_value), x, y)
    sampled = len(d) > max_points
    d = _sample(d, max_points)
    title = (tx("Dichte: {y} gegen {x}", y=y, x=x) + (f" – {filter_col} = {filter_value}" if filter_col else "")
             + (tx(" ({n} Punkte, Stichprobe)", n=len(d)) if sampled else ""))
    if d.empty:
        return _empty_fig(title, tx("Keine Zeilen mit Werten in beiden Spalten."))
    if not _kde_possible(d, x, y):
        return _empty_fig(title, tx("Zu wenige oder zu gleichförmige Werte für eine Dichte-Schätzung (KDE)."))
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.kdeplot(data=d, x=x, y=y, fill=fill, cmap="plasma", thresh=0.05, levels=10, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_lm(df: pd.DataFrame, x: str, y: str, hue: str | None = None, col: str | None = None,
           max_points: int = 5000):
    """sns.lmplot: Punkte mit Regressionsgerade und 95-%-Band, getrennt nach Farbe (hue) und Teilgrafiken (col)."""
    hue = hue if hue and hue not in (x, y) else None
    col = col if col and col not in (x, y) else None
    d = _finite_xy(df, x, y, [hue, col]).dropna(subset=[c for c in (hue, col) if c])
    sampled = len(d) > max_points
    d = _sample(d, max_points).copy()
    title = tx("Regression: {y} gegen {x}", y=y, x=x) + (tx(" ({n} Punkte, Stichprobe)", n=len(d)) if sampled else "")
    if len(d) < 2:
        return _empty_fig(title, tx("Keine Zeilen mit Werten in beiden Spalten."))
    kw = {}
    if hue:
        d[hue] = fold(d[hue], 8)
        levels = _levels(d[hue])
        kw.update(hue=hue, hue_order=levels, palette=_palette(levels))
    if col:
        d[col] = fold(d[col], 6) if col != hue else d[col]
        col_levels = _levels(d[col])
        kw.update(col=col, col_order=col_levels, col_wrap=3 if len(col_levels) > 3 else None)
    g = sns.lmplot(data=d, x=x, y=y, height=4.2, aspect=1.15, scatter_kws=dict(s=12, alpha=0.5),
                   line_kws=dict(linewidth=2), **kw)
    _suptitle(g.figure, title, 0.75 if col else 0.5)  # Teilgrafiken haben eigene Titel → mehr Platz
    return g.figure


def fig_hist_groups(df: pd.DataFrame, x: str, group: str, bins: int = 30):
    """Überlagerte Histogramme einer Zahlenspalte je Gruppe – gleiche Klassen, halb durchsichtig (alpha 0.5)."""
    if x == group:
        raise ValueError(tx("Bitte zwei verschiedene Spalten wählen."))
    d = df[[x, group]].copy()
    d[x] = d[x].astype("float64")
    d = d[np.isfinite(d[x].to_numpy())].dropna(subset=[group])
    title = tx("Verteilung von {x} je {group}", x=x, group=group)
    if d.empty:
        return _empty_fig(title, tx("Keine Zeilen mit Werten in beiden Spalten."))
    d[group] = fold(d[group], 8)
    levels = _levels(d[group])
    pal = _palette(levels)
    edges = np.histogram_bin_edges(d[x].to_numpy(), bins=int(bins))  # gleiche Klassen für alle Gruppen
    fig, ax = plt.subplots(figsize=(10, 5))
    for lv in levels:
        ax.hist(d.loc[d[group] == lv, x].to_numpy(), bins=edges, alpha=0.5, color=pal[lv], label=f"{group}={lv}")
    ax.legend(loc="upper right")
    ax.set_xlabel(x)
    ax.set_ylabel(tx("Anzahl"))
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7) Training vs. Validierung
# ---------------------------------------------------------------------------
LABEL_BOX = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85)  # Text über Linien lesbar


def fig_curve(curve: dict, metric_label: str, best_label: str | None = None):
    """Fehler auf Training und Validierung je Parameterwert (Validierungskurve, Elbow-Methode, Lernkurve) –
    senkrechte Linie = eingestellter Wert, Punkt = bester Wert, gestrichelt = naive Vorhersage."""
    x = np.asarray(curve["x"], dtype="float64")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(x, curve["train"], color=SERIES[0], marker="o", markersize=3.5, label=tx("Training"))
    ax.plot(x, curve["val"], color=SERIES[1], marker="o", markersize=3.5, label=tx("Validierung"))
    if curve.get("baseline") is not None:
        ax.axhline(curve["baseline"], color=MUTED, linewidth=1, linestyle="--", label=tx("naive Vorhersage"))
    if curve.get("log_x"):  # C: 0.01 … 100 – Zahlen statt 10^x (text.parse_math=False zeigte sonst '$\mathdefault…$')
        ax.set_xscale("log")
        decades = [v for v in x if v > 0 and float(np.log10(v)).is_integer()]
        ax.set_xticks(decades, [f"{v:g}" for v in decades])
        ax.xaxis.set_minor_formatter(NullFormatter())
    if curve.get("param") == "max_depth":  # letzter Punkt = unbegrenzt (None) bei der tatsächlichen Tiefe
        ax.set_xticks(x, [str(v) if v is not None else "None" for v in curve["values"]])
        _rotate(ax, len(x) + 3)
    if curve.get("chosen_x") is not None:
        ax.axvline(curve["chosen_x"], color=INK2, linewidth=1)
        label = tx("eingestellt") + (f" ({curve['chosen_x']:g})" if curve.get("log_x") else "")
        left = list(x).index(curve["chosen_x"]) > 0.75 * (len(x) - 1) if curve["chosen_x"] in list(x) else False
        ax.annotate(label, (curve["chosen_x"], 1), xycoords=("data", "axes fraction"), xytext=(-4 if left else 4, -14),
                    textcoords="offset points", ha="right" if left else "left", fontsize=9, color=INK2,
                    bbox=LABEL_BOX)
    j = curve.get("best_index")
    if j is not None and best_label:
        ax.scatter([x[j]], [curve["val"][j]], s=52, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        right = j > 0.7 * len(x)  # Beschriftung links vom Punkt, wenn er weit rechts liegt
        ax.annotate(tx("bester Wert: {v}", v=best_label), (x[j], curve["val"][j]), xytext=(-6 if right else 6, 9),
                    textcoords="offset points", ha="right" if right else "left", color=INK2, fontsize=9,
                    bbox=LABEL_BOX, zorder=4)
    if curve.get("metric") == "error":
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    param = tx(curve["label"])  # im Rezept deutsch gespeichert – Anzeige in der aktuellen Sprache
    ax.set_xlabel(param)
    ax.set_ylabel(metric_label)
    if get_lang() != "de" and len(param) > 1 and param[0].isupper() and param[1].islower():
        param = param[0].lower() + param[1:]  # „Error rate by number of trees“ – nur Deutsch schreibt Nomen groß
    ax.set_title(tx("{metric} je {param}", metric=metric_label, param=param))
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def fig_grid(grid: dict):
    """GridSearchCV (SVM): mittlere Validierungsgüte je C und gamma – größer = besser, Rahmen = beste Kombination."""
    rows = pd.DataFrame(grid["rows"])
    linear = rows["gamma"].isna().all()
    if linear:  # linearer Kernel: nur C
        table = rows.set_index("C")[["val"]].T
        table.index = [grid["scoring"]]
    else:
        table = rows.pivot_table(index="gamma", columns="C", values="val").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(min(2.6 + 1.25 * table.shape[1], 10), 1.5 + 0.62 * table.shape[0]))
    sns.heatmap(table, annot=True, fmt=".3f", cmap=SEQ, linewidths=1.5, linecolor=SURFACE, annot_kws={"size": 10},
                cbar_kws={"label": grid["scoring"]}, ax=ax)
    best = grid["best"]
    try:
        r = 0 if linear else list(table.index).index(best["gamma"])
        c = list(table.columns).index(best["C"])
        ax.add_patch(Rectangle((c, r), 1, 1, fill=False, edgecolor=SERIES[1], linewidth=3))
    except (KeyError, ValueError):
        pass
    ax.set_xlabel("C")
    ax.set_ylabel("" if linear else "gamma")
    plt.setp(ax.get_yticklabels(), rotation=0)
    ax.set_title(tx("GridSearchCV: mittlere Validierungsgüte ({s}, cv = {cv})", s=grid["scoring"], cv=grid["cv"]))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8) Modell-Details: Koeffizienten, Wichtigkeit, Entscheidungsbaum
# ---------------------------------------------------------------------------
def fig_bars(names, values, title: str, xlabel: str, signed: bool = False, top: int = 25):
    """Waagerechte Balken, z. B. Koeffizienten (rot positiv, blau negativ) oder feature_importances_."""
    s = pd.Series(np.asarray(values, dtype="float64"), index=[str(n) for n in names]).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(s) + 1.4))
    ax.barh(s.index, s.values, color=[POS if v > 0 else NEG for v in s.values] if signed else SERIES[0], height=0.62)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_tree(estimator, feature_names: list, class_names: list | None = None, max_depth: int = 3):
    """Die obersten Ebenen eines Entscheidungsbaums (sklearn.tree.plot_tree)."""
    from sklearn.tree import plot_tree

    depth = min(max_depth, int(estimator.get_depth()))
    fig, ax = plt.subplots(figsize=(min(4 + 3.2 * 2 ** depth / 2, 18), 2.2 + 1.6 * depth))
    plot_tree(estimator, max_depth=depth, feature_names=[str(f) for f in feature_names],
              class_names=[str(c) for c in class_names] if class_names else None, filled=True, impurity=False,
              proportion=True, rounded=True, fontsize=8, ax=ax)
    ax.set_title(tx("Entscheidungsbaum – die obersten {d} Ebenen", d=depth))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8) Evaluation
# ---------------------------------------------------------------------------
def fig_confusion(cm: np.ndarray, labels: list):
    k = len(labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n{row_pct[i, j]:.0%}" if np.isfinite(row_pct[i, j]) else f"{cm[i, j]}"
                       for j in range(k)] for i in range(k)])
    fig, ax = plt.subplots(figsize=(1.3 * k + 2.6, 1.1 * k + 1.8))
    sns.heatmap(cm, annot=annot, fmt="", cmap=SEQ, cbar=False, xticklabels=labels, yticklabels=labels,
                linewidths=1.5, linecolor=SURFACE, annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel(tx("Vorhersage"))
    ax.set_ylabel(tx("Tatsächlich"))
    ax.set_title(tx("Confusion Matrix (Anzahl, Anteil je Zeile)"))
    plt.setp(ax.get_yticklabels(), rotation=0)
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


def fig_calibration(y_true: np.ndarray, probs: dict, positive: str, bins: int = 10):
    """Kalibrierung (binär): Zeilen nach der vorhergesagten Wahrscheinlichkeit in gleich große Gruppen teilen und je
    Gruppe den tatsächlichen Anteil der positiven Klasse zeigen. Auf der Diagonale = die Wahrscheinlichkeiten stimmen
    (70 % vorhergesagt → in 70 % der Fälle eingetreten). probs: {Name: Wahrscheinlichkeiten}."""
    from sklearn.calibration import calibration_curve

    y = np.asarray(y_true).astype(int)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label=tx("perfekt kalibriert"))
    lo, hi = 1.0, 0.0
    for i, (name, p) in enumerate(probs.items()):
        p = np.asarray(p, dtype="float64")
        ok = np.isfinite(p)
        if ok.sum() < bins or len(np.unique(y[ok])) < 2:
            continue
        true, pred = calibration_curve(y[ok], p[ok], n_bins=bins, strategy="quantile")
        ax.plot(pred, true, color=SERIES[i % len(SERIES)], marker="o", markersize=4, label=name)
        lo, hi = min(lo, pred.min(), true.min()), max(hi, pred.max(), true.max())
    if lo < hi:  # Ausschnitt um die Punkte – sonst kleben sie bei Sportdaten (meist 30–80 %) in der Mitte
        pad = 0.05
        ax.set_xlim(max(lo - pad, 0), min(hi + pad, 1))
        ax.set_ylim(max(lo - pad, 0), min(hi + pad, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(tx("vorhergesagte Wahrscheinlichkeit (Mittel je Gruppe)"))
    ax.set_ylabel(tx("tatsächlicher Anteil '{v}'", v=positive))
    ax.set_title(tx("Kalibrierung: vorhergesagt gegen tatsächlich"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def fig_regression(y_true: np.ndarray, y_pred: np.ndarray, max_points: int = 5000, seed: int = 101):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y_true), max_points, replace=False) if len(y_true) > max_points else slice(None)
    yt, yp = np.asarray(y_true)[idx], np.asarray(y_pred)[idx]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    a1.scatter(yt, yp, s=10, alpha=0.5, color=SERIES[0], edgecolor=SURFACE, linewidth=0.2)
    lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    a1.plot(lims, lims, color=SERIES[7], linewidth=1.5, label=tx("perfekte Vorhersage"))
    a1.set_xlabel(tx("tatsächlich (y_test)"))
    a1.set_ylabel(tx("vorhergesagt"))
    a1.set_title(tx("Vorhersage vs. tatsächlich"))
    a1.legend(loc="upper left")
    resid = np.asarray(y_true) - np.asarray(y_pred)
    a2.hist(resid, bins=50, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    a2.axvline(0, color=INK2, linewidth=1)
    a2.set_xlabel(tx("Residuum = tatsächlich − vorhergesagt"))
    a2.set_ylabel(tx("Anzahl"))
    a2.set_title(tx("Verteilung der Fehler"))
    fig.tight_layout()
    return fig


def fig_importance(imp: pd.DataFrame, top: int = 25):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(d) + 1.4))
    ax.barh(d["Feature"].astype(str), d["Wichtigkeit"], xerr=d["Streuung"], color=SERIES[0], height=0.62,
            error_kw={"ecolor": MUTED, "elinewidth": 1})
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(tx("Verschlechterung ({v}) wenn das Feature gemischt wird", v=imp.attrs.get('metric', 'Score')))
    ax.set_title(tx("Permutation Feature Importance"))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10) Vorhersagen
# ---------------------------------------------------------------------------
def fig_values(values: np.ndarray, xlabel: str, title: str, ref: float | None = None, ref_label: str = ""):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(values, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    if ref is not None:
        ax.axvline(ref, color=INK2, linewidth=1.2)
        ax.text(ref, ax.get_ylim()[1] * 0.95, f"  {ref_label}", color=INK2, fontsize=9, va="top")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(tx("Anzahl"))
    ax.set_title(title)
    fig.tight_layout()
    return fig


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
    ax.set_xticks(x, [str(lb) for lb in labels])
    ax.set_ylabel(tx("Anteil in %"))
    ax.set_title(tx("Vorhergesagte Klassen"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
