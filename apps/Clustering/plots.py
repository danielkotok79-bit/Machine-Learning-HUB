"""
plots.py – Diagramme (matplotlib/seaborn) für die Schritte 3, 7, 8 und 10 (K-Means und die Anomalie-Modelle
Isolation Forest, LOF, DBSCAN – scikit-learn).

Jede Funktion gibt eine matplotlib-Figure zurück:
    Streamlit:  st.pyplot(fig)        Jupyter:  fig   (oder fig.savefig("bild.png"))

Farben: feste, farbfehlsichtigkeits-geprüfte Reihenfolge (SERIES) – Cluster 0, 1, 2 … haben in allen Diagrammen
dieselbe Farbe. Korrelationen und Cluster-Profile nutzen eine divergierende Skala Blau ↔ Grau ↔ Rot mit 0 in der
Mitte, Mengen eine einfarbige Blau-Skala.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.ticker import FuncFormatter, NullFormatter

from i18n import N_, tx  # Beschriftungen in der gewählten Sprache
from pipeline import cluster_label, share_text  # 'Cluster 2' bzw. '2 · Name' – überall gleich

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


def fig_corr_target(corr: pd.Series, target: str, positive=None):
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(corr) + 1.4))
    ax.barh(corr.index.astype(str), corr.values, color=[POS if v > 0 else NEG for v in corr.values], height=0.62)
    ax.axvline(0, color=AXIS, linewidth=1)
    lim = max(abs(corr).max() * 1.15, 0.05) if len(corr) else 1
    ax.set_xlim(-lim, lim)
    label = f"{target} = '{positive}'" if positive is not None else target
    ax.set_title(tx("Korrelation mit {label}", label=label))
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
# Cluster: Farben und Namen
# ---------------------------------------------------------------------------
LABEL_BOX = dict(boxstyle="round,pad=0.2", facecolor=SURFACE, edgecolor="none", alpha=0.85)  # Text über Linien lesbar


def cluster_colors(k: int) -> list:
    """Eine Farbe je Cluster: die feste Reihenfolge SERIES, ab 9 Clustern zusätzlich gleichmäßig verteilte Farbtöne."""
    if k <= len(SERIES):
        return SERIES[:k]
    return SERIES + [to_hex(c) for c in sns.color_palette("husl", k - len(SERIES))]


def _int_ticks(ax, values) -> None:
    values = list(values)
    if len(values) <= 25:
        ax.set_xticks(values, [str(v) for v in values])


# ---------------------------------------------------------------------------
# 7) Wahl von K – Elbow-Methode und Silhouette
# ---------------------------------------------------------------------------
def fig_k_curve(curve: dict, chosen: int | None = None):
    """Links die Elbow-Methode (Inertia je K), rechts die mittlere Silhouette je K. Senkrechte Linie = eingestelltes K,
    Punkt = Knick bzw. höchste Silhouette."""
    x = np.asarray(curve["values"], dtype=int)
    inertia = np.asarray(curve["inertia"], dtype="float64")
    sil = np.asarray(curve["silhouette"], dtype="float64")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    a1.plot(x, inertia, color=SERIES[0], marker="o", markersize=4)
    ok = np.isfinite(sil)
    a2.plot(x[ok], sil[ok], color=SERIES[2], marker="o", markersize=4)
    marks = [(a1, curve.get("elbow"), inertia, SERIES[0], tx("Knick: K = {k}"), 8),
             (a2, curve.get("best_silhouette"), sil, SERIES[2], tx("höchste: K = {k}"), -16)]  # Maximum: Text darunter
    for ax, k, y, color, text, dy in marks:
        if k is not None and k in list(x):
            j = list(x).index(k)
            ax.scatter([k], [y[j]], s=64, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(text.format(k=k), (k, y[j]), xytext=(8, dy), textcoords="offset points", fontsize=9,
                        color=INK2, bbox=LABEL_BOX, zorder=4)
    for ax in (a1, a2):
        if chosen is not None and chosen in list(x):
            ax.axvline(chosen, color=INK2, linewidth=1)
            right = list(x).index(chosen) > 0.75 * (len(x) - 1)
            ax.annotate(tx("eingestellt"), (chosen, 1), xycoords=("data", "axes fraction"), xytext=(-4 if right else 4, -14),
                        textcoords="offset points", ha="right" if right else "left", fontsize=9, color=INK2,
                        bbox=LABEL_BOX)
        _int_ticks(ax, x)
        ax.set_xlabel("K")
    a1.set_ylabel(tx("Inertia (Summe der quadrierten Abstände)"))
    a1.set_title(tx("Elbow-Methode: Inertia je K"))
    a2.set_ylabel(tx("mittlere Silhouette (−1 … 1)"))
    a2.set_title(tx("Silhouette je K – größer = besser getrennt"))
    fig.tight_layout()
    return fig


def fig_silhouette(sv: dict, k: int, names=None):
    """Silhouetten-Diagramm: je Cluster die Silhouette jeder Zeile, sortiert. Breite Bereiche über dem Mittelwert =
    gut getrennter Cluster; Werte unter 0 = Zeile liegt näher an einem anderen Cluster."""
    vals, lab = np.asarray(sv["values"]), np.asarray(sv["labels"])
    colors = cluster_colors(k)
    gap = max(2, int(0.02 * len(vals)))
    fig, ax = plt.subplots(figsize=(8.5, min(2.4 + 0.5 * k, 8)))
    y, ticks = 0, []
    for j in range(k):
        v = np.sort(vals[lab == j])
        if not len(v):
            continue
        ax.fill_betweenx(np.arange(y, y + len(v)), 0, v, color=colors[j], alpha=0.9, linewidth=0)
        ticks.append((y + len(v) / 2, cluster_label(j, names)))
        y += len(v) + gap
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.axvline(sv["mean"], color=INK2, linewidth=1.2, linestyle="--")
    ax.annotate(tx("Mittel {v:.3f}", v=sv["mean"]), (sv["mean"], 0), xycoords=("data", "axes fraction"),
                xytext=(4, 6), textcoords="offset points", fontsize=9, color=INK2, bbox=LABEL_BOX)
    ax.set_yticks([t for t, _ in ticks], [lb for _, lb in ticks])
    ax.set_xlim(min(-0.1, float(vals.min()) - 0.05), 1.0)
    ax.set_ylim(-gap, y)
    ax.set_xlabel(tx("Silhouette je Zeile (−1 … 1)"))
    ax.set_title(tx("Silhouetten-Diagramm – je Cluster sortiert")
                 + (tx(" ({n} Zeilen, Stichprobe)", n=sv["n"]) if sv.get("sampled") else ""))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 8) Cluster-Auswertung
# ---------------------------------------------------------------------------
def fig_sizes(sizes, names=None):
    """Anzahl Zeilen je Cluster (mit Anteil)."""
    sizes = np.asarray(sizes, dtype=int)
    k, total = len(sizes), max(int(sizes.sum()), 1)
    fig, ax = plt.subplots(figsize=(min(max(6.0, 0.95 * k + 3), 14), 3.8))
    ax.bar(range(k), sizes, color=cluster_colors(k), width=0.65)
    for j, v in enumerate(sizes):
        ax.text(j, v, f"{v:,}".replace(",", " ") + f"\n{v / total:.0%}", ha="center", va="bottom", fontsize=9,
                color=INK2)
    ax.set_xticks(range(k), [cluster_label(j, names) for j in range(k)])
    _rotate(ax, k + (4 if any(names or []) else 0))  # Namen sind länger als 'Cluster 2'
    ax.set_ylim(0, max(int(sizes.max()), 1) * 1.3)
    ax.set_ylabel(tx("Anzahl Zeilen"))
    ax.set_title(tx("Größe der Cluster"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_profile(z: pd.DataFrame, names=None, top: int = 25, labels: list | None = None, title: str | None = None):
    """Cluster-Profile: Abweichung jedes Features vom Durchschnitt in Standardabweichungen (z) – rot = über,
    blau = unter dem Durchschnitt. Bei vielen Features die top mit der größten Abweichung. labels/title: eigene
    Spaltennamen und Titel (z. B. normal / auffällig bei den Anomalie-Modellen)."""
    zz = z.copy()
    note = ""
    if len(zz) > top:
        zz = zz.loc[zz.abs().max(axis=1).sort_values(ascending=False).index[:top]]
        note = tx(" – die {n} stärksten von {m}", n=top, m=len(z))
    zz.columns = list(labels) if labels is not None else [cluster_label(int(j), names) for j in zz.columns]
    zz.index = [str(i) for i in zz.index]
    fig, ax = plt.subplots(figsize=(min(3.2 + 1.25 * zz.shape[1], 14), 1.4 + 0.36 * len(zz)))
    sns.heatmap(zz, annot=True, fmt="+.1f", cmap=DIV, center=0, vmin=-2, vmax=2, linewidths=1, linecolor=SURFACE,
                cbar_kws={"label": tx("z = Abweichung in Standardabweichungen")}, annot_kws={"size": 9}, ax=ax)
    ax.set_title((tx("Cluster-Profile: Abweichung vom Durchschnitt (z)") if title is None else title) + note)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_yticklabels(), rotation=0)
    _rotate(ax, zz.shape[1] + (4 if any(names or []) else 0))
    fig.tight_layout()
    return fig


def _scatter_groups(ax, xy: np.ndarray, groups: pd.Series, colors: dict, labels: dict) -> None:
    for g, color in colors.items():
        m = (groups == g).to_numpy()
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=13, alpha=0.65, color=color, edgecolor=SURFACE, linewidth=0.3,
                       label=labels.get(g, str(g)))


def fig_pca(proj: dict, names=None, groups: pd.Series | None = None, group_title: str = ""):
    """Die Cluster in 2D (PCA) – nur zur Darstellung: Punkte = Zeilen (Stichprobe), X = Cluster-Zentren.
    Farbe = Cluster, oder die bekannte Einteilung (groups, gleiche Reihenfolge wie proj['rows'])."""
    xy, centers = np.asarray(proj["coords"]), np.asarray(proj["centers"])
    k = len(centers)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    if groups is None:
        colors = dict(zip(range(k), cluster_colors(k)))
        _scatter_groups(ax, xy, pd.Series(proj["labels"]), colors, {j: cluster_label(j, names) for j in range(k)})
        title = tx("Cluster in 2D (PCA) – Farbe = Cluster")
        legend = tx("Cluster")
    else:  # andere Farben als die Cluster (sonst sähe 'Yes' wie Cluster 1 aus)
        g = fold(pd.Series(np.asarray(groups, dtype=object)), 8).fillna(tx("ohne Wert"))
        levels = _levels(g)
        other = [SERIES[i] for i in (6, 3, 5, 4, 7, 0, 1, 2)]
        colors = {lv: OTHER_COLOR if lv in (tx(OTHER), tx("ohne Wert")) else other[i % len(other)]
                  for i, lv in enumerate(levels)}
        _scatter_groups(ax, xy, g, colors, {})
        title = tx("Cluster in 2D (PCA) – Farbe = {g}", g=group_title)
        legend = group_title
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=190, color=INK, edgecolor=SURFACE, linewidth=1.5, zorder=4,
               label=tx("Zentrum"))
    for j, (cx, cy) in enumerate(centers):
        ax.annotate(str(j), (cx, cy), xytext=(7, 6), textcoords="offset points", fontsize=10, fontweight="bold",
                    color=INK, bbox=LABEL_BOX, zorder=5)
    ev = proj["explained"]
    ax.set_xlabel(tx("PC1 ({pct} der Streuung)", pct=f"{ev[0]:.0%}"))
    ax.set_ylabel(tx("PC2 ({pct} der Streuung)", pct=f"{ev[1]:.0%}"))
    if len(xy) < proj.get("n", len(xy)):
        title += tx(" ({n} Punkte, Stichprobe)", n=len(xy))
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title=legend, markerscale=1.4)
    fig.tight_layout()
    return fig


def fig_cluster_scatter(df: pd.DataFrame, x: str, y: str, labels, names=None, max_points: int = 5000,
                        seed: int = 101):
    """Zwei Features gegeneinander, Farbe = Cluster – wie im Kurs (Grad.Rate gegen Room.Board), nur mit den Clustern
    statt der bekannten Einteilung."""
    if x == y:
        raise ValueError(tx("Bitte zwei verschiedene Spalten wählen."))
    lab = np.asarray(labels)
    d = pd.DataFrame({"x": pd.to_numeric(df[x], errors="coerce").to_numpy(),
                      "y": pd.to_numeric(df[y], errors="coerce").to_numpy(), "c": lab}).dropna()
    title = tx("{y} gegen {x} – Farbe = Cluster", y=y, x=x)
    if d.empty:
        return _empty_fig(title, tx("Keine Zeilen mit Werten in beiden Spalten."))
    n = len(d)
    d = _sample(d, max_points, seed)
    k = int(lab.max()) + 1 if len(lab) else 0
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    _scatter_groups(ax, d[["x", "y"]].to_numpy(), d["c"], dict(zip(range(k), cluster_colors(k))),
                    {j: cluster_label(j, names) for j in range(k)})
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title + (tx(" ({n} Punkte, Stichprobe)", n=len(d)) if n > max_points else ""))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), title=tx("Cluster"), markerscale=1.4)
    fig.tight_layout()
    return fig


def fig_crosstab(ct: pd.DataFrame, names=None, group_title: str = ""):
    """Kreuztabelle Cluster × bekannte Gruppe: Anzahl und Anteil je Cluster (Zeile)."""
    values = ct.to_numpy()
    k, m = values.shape
    with np.errstate(invalid="ignore", divide="ignore"):
        share = values / values.sum(axis=1, keepdims=True)
    annot = np.array([[f"{values[i, j]}\n{share[i, j]:.0%}" if np.isfinite(share[i, j]) else f"{values[i, j]}"
                       for j in range(m)] for i in range(k)])
    fig, ax = plt.subplots(figsize=(min(1.4 * m + 3.6, 14), 0.8 * k + 1.7))
    sns.heatmap(values, annot=annot, fmt="", cmap=SEQ, cbar=False, linewidths=1.5, linecolor=SURFACE,
                xticklabels=[str(c) for c in ct.columns], yticklabels=[cluster_label(int(j), names) for j in ct.index],
                annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel(group_title)
    ax.set_ylabel("")
    ax.set_title(tx("Cluster × {g} (Anzahl, Anteil je Cluster)", g=group_title))
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def fig_bars(names, values, title: str, xlabel: str, signed: bool = False, top: int = 25, colors=None):
    """Waagerechte Balken, z. B. Trennkraft je Feature (R²) oder Abstand zu den Zentren."""
    s = pd.Series(np.asarray(values, dtype="float64"), index=[str(n) for n in names]).head(top)
    colors = list(colors)[: len(s)] if colors is not None else None
    s, colors = s.iloc[::-1], colors[::-1] if colors else None
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(s) + 1.4))
    color = colors or ([POS if v > 0 else NEG for v in s.values] if signed else SERIES[0])
    ax.barh(s.index, s.values, color=color, height=0.62)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_confusion(cm: np.ndarray, labels: list, xlabel: str | None = None, ylabel: str | None = None,
                  title: str | None = None):
    """Confusion Matrix (Anzahl, Anteil je Zeile) – hier: bekannte Gruppe gegen die Gruppe laut Cluster."""
    k = len(labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        row_pct = cm / cm.sum(axis=1, keepdims=True)
    annot = np.array([[f"{cm[i, j]}\n{row_pct[i, j]:.0%}" if np.isfinite(row_pct[i, j]) else f"{cm[i, j]}"
                       for j in range(k)] for i in range(k)])
    fig, ax = plt.subplots(figsize=(1.3 * k + 2.8, 1.1 * k + 1.8))
    sns.heatmap(cm, annot=annot, fmt="", cmap=SEQ, cbar=False, xticklabels=labels, yticklabels=labels,
                linewidths=1.5, linecolor=SURFACE, annot_kws={"size": 10}, ax=ax)
    ax.set_xlabel(xlabel or tx("Gruppe laut Cluster"))
    ax.set_ylabel(ylabel or tx("tatsächliche Gruppe"))
    ax.set_title(title or tx("Confusion Matrix (Anzahl, Anteil je Zeile)"))
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 10) Neue Daten
# ---------------------------------------------------------------------------
def fig_cluster_shares(share_new, share_train=None, names=None):
    """Anteil je Cluster: neue Daten gegen Training."""
    share_new = np.asarray(share_new, dtype="float64")
    k = len(share_new)
    x = np.arange(k)
    train = share_train is not None and len(share_train) == k
    w = 0.4 if train else 0.6
    fig, ax = plt.subplots(figsize=(min(max(6.0, 1.2 * k + 3), 16), 3.8))
    ax.bar(x - (w / 2 if train else 0), share_new * 100, width=w, color=SERIES[0], label=tx("neue Daten"))
    if train:
        ax.bar(x + w / 2, np.asarray(share_train, dtype="float64") * 100, width=w, color=SERIES[1],
               label=tx("Training"))
        ax.legend(loc="upper right")
    ax.set_xticks(x, [cluster_label(j, names) for j in range(k)])
    _rotate(ax, k + (4 if any(names or []) else 0))
    ax.set_ylabel(tx("Anteil in %"))
    ax.set_title(tx("Verteilung auf die Cluster"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Anomalie-Modelle (Isolation Forest, LOF, DBSCAN): Schritte 7, 8 und 10
# ---------------------------------------------------------------------------
ANOMALY_COLOR, NORMAL_COLOR = "#e34948", "#9ec5f4"  # auffällig rot, normal hellblau (Rauschen bei DBSCAN rot)


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype="float64")
    return v[np.isfinite(v)]


def _plain_log_axis(axis):
    """Log-Achse mit 1, 10, 100 … statt 10² – mathtext ist abgeschaltet (text.parse_math), sonst stünde dort
    '$\\mathdefault{10^{2}}$'."""
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ") if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def fig_scores(score, threshold: float, curve: dict | None = None, title: str | None = None):
    """Links die Verteilung des Anomalie-Scores mit der Schwelle (rot), rechts – wenn curve['sorted'] da ist – die
    absteigend sortierte Score-Kurve der auffälligsten Zeilen mit Knick und Schwelle (wie die Elbow-Methode)."""
    vals = _finite(score)
    with_curve = curve is not None and curve.get("sorted") is not None and len(curve["sorted"])
    fig, axes = plt.subplots(1, 2 if with_curve else 1, figsize=(11.5 if with_curve else 8, 4.2))
    a1 = axes[0] if with_curve else axes
    title = tx("Verteilung des Anomalie-Scores") if title is None else title
    if not len(vals):
        a1.text(0.5, 0.5, tx("Keine endlichen Scores."), ha="center", va="center", color=MUTED, transform=a1.transAxes)
        a1.set_title(title)
        a1.axis("off")
    else:
        thr_ok = threshold is not None and np.isfinite(threshold)
        edges = np.histogram_bin_edges(vals, bins=60)
        a1.hist(vals[vals <= threshold] if thr_ok else vals, bins=edges, color=NORMAL_COLOR, edgecolor=SURFACE,
                linewidth=0.4, label=tx("normal"))
        if thr_ok and (vals > threshold).any():
            a1.hist(vals[vals > threshold], bins=edges, color=ANOMALY_COLOR, edgecolor=SURFACE, linewidth=0.4,
                    label=tx("auffällig"))
        if thr_ok:  # Schwelle in der Legende – eine Beschriftung an der Linie läge oft unter der Legende
            a1.axvline(threshold, color=INK2, linewidth=1.2, linestyle="--",
                       label=tx("Schwelle {v}", v=f"{threshold:.4g}"))
        a1.set_yscale("log")
        _plain_log_axis(a1.yaxis)
        a1.set_xlabel(tx("Anomalie-Score (höher = auffälliger)"))
        a1.set_ylabel(tx("Anzahl (log)"))
        a1.set_title(title)
        a1.legend(loc="upper right")
    if with_curve:
        desc = np.asarray(curve["sorted"], dtype="float64")
        x = np.arange(1, len(desc) + 1)
        a2 = axes[1]
        a2.plot(x, desc, color=SERIES[0])
        if threshold is not None and np.isfinite(threshold):
            a2.axhline(threshold, color=INK2, linewidth=1.2, linestyle="--")
        knee = curve.get("knee")
        if knee is not None and knee < len(desc):
            a2.scatter([knee + 1], [desc[knee]], s=64, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3)
            a2.annotate(tx("Knick: {n} Zeilen", n=knee + 1), (knee + 1, desc[knee]), xytext=(8, 8),
                        textcoords="offset points", fontsize=9, color=INK2, bbox=LABEL_BOX, zorder=4)
        a2.set_xlabel(tx("Rang (1 = auffälligste Zeile)"))
        a2.set_ylabel(tx("Anomalie-Score"))
        a2.set_title(tx("Score-Kurve der auffälligsten Zeilen"))
    fig.tight_layout()
    return fig


def fig_kdist(curve: dict, eps: float):
    """DBSCAN, Schritt 7: links die aufsteigend sortierte k-Distanz mit eps (gestrichelt) und Knick, rechts – wenn
    berechnet – der Anteil Rauschen und die Zahl der Cluster je eps."""
    kd = np.asarray(curve["kdist"], dtype="float64")
    grid = curve.get("grid")
    fig, axes = plt.subplots(1, 2 if grid else 1, figsize=(11.5 if grid else 8, 4.2))
    a1 = axes[0] if grid else axes
    a1.plot(np.arange(len(kd)), kd, color=SERIES[0])
    finite = kd[np.isfinite(kd)]
    knee = curve.get("knee")
    if len(finite):  # wenige extreme Ausreißer würden den Knick ganz unten zusammendrücken → oben abschneiden
        top = max(float(np.quantile(finite, 0.995)) * 1.3, float(eps) * 1.4,
                  float(kd[knee]) * 1.6 if knee is not None and knee < len(kd) else 0.0)
        a1.set_ylim(bottom=0)  # Abstände sind nie negativ
        if finite.max() > top:
            a1.set_ylim(0, top)
            a1.annotate(tx("abgeschnitten – höchster Wert {v}", v=f"{finite.max():.4g}"), (1, 1),
                        xycoords="axes fraction", xytext=(-4, -14), textcoords="offset points", ha="right",
                        fontsize=8, color=MUTED)
    a1.axhline(eps, color=INK2, linewidth=1.2, linestyle="--")
    a1.annotate(tx("eingestellt: eps = {v}", v=f"{eps:.4g}"), (0, eps), xycoords=("axes fraction", "data"),
                xytext=(4, 6), textcoords="offset points", fontsize=9, color=INK2, bbox=LABEL_BOX)
    if knee is not None and knee < len(kd):
        a1.scatter([knee], [kd[knee]], s=64, color=SERIES[1], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        a1.annotate(tx("Knick: eps ≈ {v}", v=f"{kd[knee]:.4g}"), (knee, kd[knee]), xytext=(-8, 10),
                    textcoords="offset points", ha="right", fontsize=9, color=INK2, bbox=LABEL_BOX, zorder=4)
    a1.set_xlabel(tx("Zeilen, nach k-Distanz sortiert"))
    a1.set_ylabel(tx("Abstand zum {k}. Nachbarn", k=curve.get("k", "")))
    a1.set_title(tx("k-Distanz – Kernpunkt, wenn höchstens eps"))
    if grid:
        a2 = axes[1]
        e = np.asarray(grid["eps"], dtype="float64")
        noise = np.asarray(grid["noise"], dtype="float64") * 100
        a2.plot(e, noise, color=ANOMALY_COLOR, marker="o", markersize=4)
        for xv, yv, c in zip(e, noise, grid["clusters"]):
            a2.annotate(str(c), (xv, yv), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8,
                        color=INK2)
        a2.axvline(eps, color=INK2, linewidth=1.2, linestyle="--")
        a2.set_xlabel("eps")
        a2.set_ylabel(tx("Rauschen in % (Zahl am Punkt = Cluster)"))
        a2.set_title(tx("Rauschanteil je eps"))
    fig.tight_layout()
    return fig


def fig_anomaly_pca(proj: dict, flag_rows, dbscan: bool = False):
    """Die Zeilen in 2D (PCA) – nur zur Darstellung. Normal hellblau, auffällig rot und obenauf. Bei DBSCAN die
    Cluster in ihren Farben, Rauschen rot. flag_rows = Anomalie ja/nein in der Reihenfolge von proj['rows']."""
    xy = np.asarray(proj["coords"])
    flag = np.asarray(flag_rows, dtype=bool)
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    if dbscan:
        lab = np.asarray(proj["labels"], dtype=int)
        k = int(lab.max()) + 1 if (lab >= 0).any() else 0
        colors = cluster_colors(k) if k else []
        for j in range(k):
            m = lab == j
            ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.6, color=colors[j], edgecolor=SURFACE, linewidth=0.3,
                       label=cluster_label(j))
    else:
        ax.scatter(xy[~flag, 0], xy[~flag, 1], s=10, alpha=0.55, color=NORMAL_COLOR, edgecolor=SURFACE, linewidth=0.2,
                   label=tx("normal"))
    ax.scatter(xy[flag, 0], xy[flag, 1], s=26, alpha=0.9, color=ANOMALY_COLOR, edgecolor=INK, linewidth=0.4,
               marker="X" if dbscan else "o", label=tx("Rauschen (Anomalie)") if dbscan else tx("auffällig"), zorder=4)
    ev = proj["explained"]
    ax.set_xlabel(tx("PC1 ({pct} der Streuung)", pct=f"{ev[0]:.0%}"))
    ax.set_ylabel(tx("PC2 ({pct} der Streuung)", pct=f"{ev[1]:.0%}"))
    title = tx("Anomalien in 2D (PCA)")
    if len(xy) < proj.get("n", len(xy)):
        title += tx(" ({n} Punkte, Stichprobe)", n=len(xy))
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), markerscale=1.3)
    fig.tight_layout()
    return fig


def fig_anomaly_groups(table: pd.DataFrame, overall: float, group_title: str = ""):
    """Anteil auffälliger Zeilen je Gruppe der bekannten Einteilung (z. B. Class) – Linie = Anteil insgesamt."""
    t = table.copy()
    t["Gruppe"] = t["Gruppe"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(t) + 1.5))
    ax.barh(t["Gruppe"][::-1], (t["Anteil auffällig"] * 100)[::-1], color=ANOMALY_COLOR, height=0.6)
    for y, (v, k, n) in enumerate(zip(t["Anteil auffällig"][::-1], t["auffällig"][::-1], t["Zeilen"][::-1])):
        ax.text(v * 100, y, f"  {v:.1%} ({k}/{n})", va="center", fontsize=9, color=INK2)
    ax.axvline(overall * 100, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlim(0, max(float(t["Anteil auffällig"].max()) * 100 * 1.35, overall * 100 * 1.2, 1))
    ax.set_xlabel(tx("auffällig in % (gestrichelt: alle Zeilen)"))
    ax.set_ylabel(group_title)
    ax.set_title(tx("Anteil auffälliger Zeilen je {g}", g=group_title))
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def fig_dbscan_sizes(sizes, n_noise: int):
    """DBSCAN: Zeilen je Cluster und Rauschen (rot)."""
    sizes = [int(v) for v in sizes]
    labels = [cluster_label(j) for j in range(len(sizes))] + [tx("Rauschen")]
    values = sizes + [int(n_noise)]
    colors = list(cluster_colors(len(sizes))) + [ANOMALY_COLOR]
    total = max(sum(values), 1)
    fig, ax = plt.subplots(figsize=(min(max(6.0, 0.95 * len(values) + 3), 14), 3.8))
    ax.bar(range(len(values)), values, color=colors, width=0.65)
    for j, v in enumerate(values):
        ax.text(j, v, f"{v:,}".replace(",", " ") + f"\n{share_text(v / total)}", ha="center", va="bottom",
                fontsize=9, color=INK2)
    ax.set_xticks(range(len(values)), labels)
    _rotate(ax, len(values))
    ax.set_ylim(0, max(max(values), 1) * 1.3)
    ax.set_ylabel(tx("Anzahl Zeilen"))
    ax.set_title(tx("DBSCAN: Cluster und Rauschen"))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig
