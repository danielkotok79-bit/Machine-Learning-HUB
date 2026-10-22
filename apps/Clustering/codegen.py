"""
codegen.py – erzeugt den passenden Python-Code (pandas, scikit-learn) zu den Einstellungen der App.

Zum Nachvollziehen und um eine Pipeline 1:1 in ein Jupyter Notebook oder Skript zu übernehmen. Der erzeugte
Code rechnet dasselbe wie die App (pipeline.py) – das prüft tests/test_codegen_parity.py.
"""
from __future__ import annotations

import re

import pipeline as pl
from i18n import N_, tx  # Kommentare im erzeugten Code in der gewählten Sprache

IMPORTS = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix"""

# Hilfsfunktion für "Zahl extrahieren" – dieselbe Logik wie pipeline.parse_numbers()
TO_NUMBER = r'''def to_number(s):
    """Erste Zahl aus Text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5' → 1234.5"""
    import re

    def norm(t):
        if not isinstance(t, str):
            return float("nan")
        if "," in t and "." in t:
            t = t.replace(".", "").replace(",", ".") if t.rfind(",") > t.rfind(".") else t.replace(",", "")
        elif "," in t:
            t = t.replace(",", "") if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", t) else t.replace(",", ".")
        elif t.count(".") > 1:
            t = t.replace(".", "")
        try:
            return float(t)
        except ValueError:
            return float("nan")

    tok = s.astype(str).where(s.notna()).str.extract(r"(-?\d(?:[\d.,]*\d)?)", expand=False)
    low = s.astype(str).str.strip().str.lower()
    return tok.map(norm).astype(float).where(~low.isin(["true", "false"]), (low == "true").astype(float))'''
_TO_NUMBER_DOC = N_("Erste Zahl aus Text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5' → 1234.5")


def _to_number_code() -> str:
    """TO_NUMBER mit Docstring in der gewählten Sprache."""
    return TO_NUMBER.replace(_TO_NUMBER_DOC, tx(_TO_NUMBER_DOC))


def imports_code() -> str:
    return IMPORTS


def _cm(text) -> str:
    """Kommentartext in EINER Zeile. Spaltennamen, Werte und Definitionen aus den Daten können Zeilenumbrüche
    enthalten – im Kommentar würde der Rest sonst zu einer eigenen Code-Zeile (die beim Ausführen mitläuft)."""
    return " ".join(str(text).splitlines())


def _ident(name: str, suffix: str = "") -> str:
    """Gültiger Python-Variablenname aus einem Spaltennamen ('total acc' → 'total_acc_avg')."""
    v = re.sub(r"\W+", "_", str(name)).strip("_") or "spalte"
    if v[0].isdigit():
        v = "c_" + v
    return v + suffix


def _float(v) -> str:
    """Zahl für den erzeugten Code – exakt der Wert der App (repr ohne Rundung), ganze Zahlen ohne '.0'.
    f"{x:g}" rundete auf 6 Stellen (C = 1.234567 → 1.23457) – der Code rechnete dann ein anderes Modell."""
    f = float(v)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


def _num(v):
    """Ältere Rezepte ohne class_values: '1' → 1, '2.5' → 2.5, 'True' → True, sonst Text."""
    if str(v) in ("True", "False"):
        return str(v) == "True"
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() else f


# ---------------------------------------------------------------------------
# 1) + 5) Laden, Stichprobe
# ---------------------------------------------------------------------------
def load_code(name: str, sep: str = ",", decimal: str = ".", encoding: str = "utf-8",
              sample_frac: float | None = None, seed: int = pl.RANDOM_STATE, strip_columns: bool = False) -> str:
    args = [repr(name)]
    if sep != ",":
        args.append(f"sep={sep!r}")
    if decimal != ".":
        args.append(f"decimal={decimal!r}")
    if encoding.lower().replace("-", "") != "utf8":
        args.append(f"encoding={encoding!r}")
    args.append("low_memory=False")  # ganze Spalte auf einmal → einheitliche Typen (wie die App)
    lines = [f"df = pd.read_csv({', '.join(args)})"]
    if strip_columns:
        lines.append("df.columns = df.columns.str.strip()   # " + tx("Leerzeichen in den Spaltennamen entfernen"))
    if sample_frac:
        lines.append(f"df = df.sample(frac={sample_frac}, random_state={seed}).reset_index(drop=True)   # "
                     + tx("Stichprobe"))
    else:
        lines.append(f"# df = df.sample(frac=0.1, random_state={seed})   # "
                     + tx("optional: Stichprobe für langsame Rechner"))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2) Datenaufbereitung 1
# ---------------------------------------------------------------------------
def _as_text(col: str, kind: str) -> str:
    """Spalte als Text – exakt wie pipeline._to_str (22690.0 → '22690', NaN bleibt NaN)."""
    if kind == "float_int":
        return f"{col}.astype('Int64').astype(str).where({col}.notna())"
    if kind == "float":
        return f"{col}.astype(str).where({col}.notna())"
    if kind in ("int", "bool"):
        return f"{col}.astype(str)"
    return col


def _date_expr(col: str, p: str, fitted: dict, numeric: bool) -> str:
    if numeric and fitted.get("numeric_date"):
        return f"pd.to_datetime({col}.astype('Int64').astype(str), format={(p or '%Y%m%d')!r}, errors='coerce')"
    fmt = f", format={p!r}" if p else ""
    return f"pd.to_datetime({col}{fmt}, errors='coerce')"


def _action_lines(c: str, dst: str, act: str, p: str, fitted: dict, kind: str, state: dict) -> list[str]:
    """Code für eine Aktion: Ergebnis aus df[c] nach df[dst] (dst == c → in der Spalte selbst).
    Bei 'Datum → Jahr + Monat' ist dst der Grundname für <dst>_jahr und <dst>_monat."""
    col, out = f"df[{c!r}]", f"df[{dst!r}]"
    numeric = kind != "text"
    copy = [] if dst == c else [f"{out} = {col}.astype(float)"]
    if act == pl.A_KEEP:
        return [] if dst == c else [f"{out} = {col}.copy()"]
    if act == pl.A_NUM:
        if numeric:
            return ["# " + _cm(tx("{c}: ist bereits eine Zahl – keine Umwandlung nötig", c=c))] + copy
        state["to_number"] = True
        return [f"{out} = to_number({col})"]
    if act in pl.DATE_ACTIONS:
        if numeric and not fitted.get("numeric_date"):
            note = (tx("{c}: ist bereits Jahr als Zahl – keine Umwandlung nötig", c=c) if act == pl.A_YEAR
                    else tx("{c}: ist bereits Monat als Zahl – keine Umwandlung nötig", c=c))
            return ["# " + _cm(note)] + copy
        expr = _date_expr(col, p, fitted, numeric)
        note = "   # " + tx("Datum als Zahl, z. B. 20141013") if numeric else ""
        if act == pl.A_YEAR_MONTH:
            jahr, monat = pl.result_names(dst, act)
            return [f"datum = {expr}{note}", f"df[{jahr!r}] = datum.dt.year", f"df[{monat!r}] = datum.dt.month"]
        return [f"{out} = {expr}.dt.{'year' if act == pl.A_YEAR else 'month'}{note}"]
    if act == pl.A_FIRST:
        return [f"{out} = {_as_text(col, kind)}.str[:{int(float(p)) if p else 1}]"]
    if act == pl.A_LAST:
        return [f"{out} = {_as_text(col, kind)}.str[-{int(float(p)) if p else 5}:]"]
    if act == pl.A_REPLACE:
        mapping = pl.parse_replace(p)
        if numeric:
            mapping = {pl._to_number(k): pl._to_number(v) for k, v in mapping.items()}
        return [f"{out} = {col}.replace({mapping!r})"]
    if act == pl.A_TOPN:
        n = int(float(p)) if p else 10
        lines, src = [], col
        if kind != "text":
            lines.append(f"{out} = {_as_text(col, kind)}")
            src = out
        keep = fitted.get("keep")
        if keep is not None and len(keep) <= 30:
            lines.append(f"top = {keep!r}   # " + tx("die {n} häufigsten Werte der Trainingsdaten", n=n))
        else:
            lines.append(f"top = {src}.value_counts().index[:{n}]")
        lines.append(f"{out} = {src}.where({src}.isin(top) | {src}.isna(), {pl.OTHER!r})")
        return lines
    if act == pl.A_CAT:
        return [f"{out} = {_as_text(col, kind)}   # " + tx("Zahl → Kategorie (Text)")]
    if act == pl.A_CLIP:
        lo, hi = pl.parse_clip(p)
        args = ", ".join(a for a in (f"lower={_float(lo)}" if lo is not None else "",
                                     f"upper={_float(hi)}" if hi is not None else "") if a)
        if numeric:
            return [f"{out} = {col}.astype(float).clip({args})   # " + tx("Werte begrenzen")]
        state["to_number"] = True
        return [f"{out} = to_number({col}).clip({args})   # " + tx("Werte begrenzen")]
    return ["# " + _cm(tx("unbekannte Aktion {act} für {c}", act=repr(act), c=c))]


def _new_column_lines(s: dict, state: dict) -> list[str]:
    """Code für eine neue Spalte (Schritt 2, Abschnitt 'Neue Spalten')."""
    name, kind, fitted = s["name"], s["kind"], s.get("fitted") or {}
    out = f"df[{name!r}]"
    if kind == pl.N_MAP:
        mapping, default = pl.parse_mapping(s["definition"])
        src = f"df[{s['source']!r}]"
        txt = f"{_as_text(src, fitted.get('src_kind', 'text'))}.str.strip()"
        expr = f"txt.map({mapping!r})"
        if default is not None:
            expr += f".where(txt.isna() | txt.isin({list(mapping)!r}), {default!r})"
        return [f"txt = {txt}", f"{out} = {expr}"]
    if kind == pl.N_FORMULA:
        return [f"{out} = ({fitted['code']}).astype(float).replace([np.inf, -np.inf], np.nan)"]
    if kind == pl.N_COND:
        return [f"{out} = ({fitted['code']}).astype(float).where(df[{fitted['used']!r}].notna().all(axis=1))"
                + "   # " + tx("1 = ja, 0 = nein, leer bei Lücken")]
    return _action_lines(s["source"], name, s.get("action") or pl.A_KEEP, s.get("param", ""), fitted,
                         fitted.get("src_kind", "text"), state)


def step2_code(actions: list[dict], numeric_cols: set | None = None, new_columns: list[dict] | None = None) -> str:
    """Code für Schritt 2. Nutzt die beim Übernehmen gemerkten Infos (fitted) über die Rohspalten."""
    numeric_cols = numeric_cols or set()
    lines: list[str] = []
    drops: list[str] = []
    state: dict = {}

    def flush():
        if drops:
            lines.append(f"df = df.drop({drops!r}, axis=1)")
            drops.clear()

    for a in actions:
        c, act = a["column"], a.get("action") or pl.A_KEEP
        p, new = (a.get("param") or "").strip(), (a.get("new_name") or "").strip()
        fitted = a.get("fitted") or {}
        kind = fitted.get("src_kind") or ("float" if c in numeric_cols else "text")
        if act == pl.A_KEEP and (not new or new == c):
            continue
        if act == pl.A_DROP:
            drops.append(c)
            continue
        flush()
        target = new or c
        if act == pl.A_YEAR_MONTH:
            lines += _action_lines(c, target, act, p, fitted, kind, state)
            lines.append(f"df = df.drop({c!r}, axis=1)   # "
                         + _cm(tx("ersetzt durch {target}_jahr und {target}_monat", target=target)))
            continue
        lines += _action_lines(c, c, act, p, fitted, kind, state)
        if new and new != c:
            lines.append(f"df = df.rename(columns={{{c!r}: {new!r}}})")
    flush()
    if new_columns:
        lines += ["", "# " + tx("Neue Spalten")]
        for s in new_columns:
            if s["kind"] == pl.N_COPY:
                act = s.get("action") or pl.A_KEEP
                note = tx("Kopie von {src}", src=s["source"]) + ("" if act == pl.A_KEEP else f" + {tx(act)}")
            else:
                note = f"{tx(s['kind'])}: {s['definition']}"
            lines += ["# " + _cm(f"{s['name']} – {note}")] + _new_column_lines(s, state)
    if state.get("to_number"):
        lines = [_to_number_code(), ""] + lines
    return "\n".join(lines) if lines else "# " + tx("keine Änderungen")


# ---------------------------------------------------------------------------
# 4) Datenaufbereitung 2 – kein Label, kein Split
# ---------------------------------------------------------------------------
def rows_var(r: dict) -> str:
    """Variable mit den Zeilen und Spalten fürs Clustering: 'data' (Arbeitskopie), wenn Schritt 4 Zeilen oder Spalten
    entfernt oder inf ersetzt – df (Daten nach Schritt 2) bleibt dann unverändert, wie in der App (Export in
    Schritt 5, Diagramme in Schritt 3); sonst einfach df."""
    return "data" if r.get("inf_columns") or r.get("dropped_columns") or r.get("dropna_columns") else "df"


def _group_expr(r: dict) -> str:
    """Bekannte Einteilung als Text – exakt wie pipeline.group_values (Zahlen ohne '.0', ohne Rand-Leerzeichen)."""
    g = f"{rows_var(r)}[{r['group_col']!r}]"
    return f"{_as_text(g, r.get('group_kind', 'text'))}.str.strip()"


def step4_code(r: dict) -> str:
    """Code für Schritt 4 in derselben Reihenfolge wie die App: Füllwerte, Kategorien und Scaler aus ALLEN Zeilen
    (Clustering: kein Label, kein Train/Test-Split)."""
    L = []
    v = rows_var(r)
    if v == "data":
        L.append("data = df.copy()   # "
                 + tx("Arbeitskopie fürs Clustering – df (Daten nach Schritt 2) bleibt unverändert"))
    if r.get("inf_columns"):
        L.append(f"data[{r['inf_columns']!r}] = data[{r['inf_columns']!r}].replace([np.inf, -np.inf], np.nan)   # "
                 + tx("inf wie fehlend behandeln"))
    if r.get("dropped_columns"):
        L.append(f"data = data.drop({r['dropped_columns']!r}, axis=1)   # " + tx("Spalten mit zu vielen Lücken"))
    if r.get("dropna_columns"):
        L.append(f"data = data.dropna(subset={r['dropna_columns']!r})")
    cat = r["categorical_features"]
    L.append(f"features = {r['features']!r}")
    if r.get("name_col"):
        L.append(f"names = {v}[{r['name_col']!r}]   # " + tx("Bezeichnung je Zeile – nur zur Anzeige"))
    if r.get("group_col"):
        L.append(f"groups = {_group_expr(r)}   # "
                 + tx("bekannte Einteilung – nur zum Vergleich in Schritt 8, geht NICHT in K-Means"))
    L += ["", f"filled = {v}[features].copy()"]
    if cat:
        L += [f"categorical = {cat!r}   # " + tx("→ Dummies"),
              "filled[categorical] = filled[categorical].apply(lambda s: s.astype(str).str.strip().where(s.notna()))"
              "   # " + tx("als Text, ohne Leerzeichen am Rand")]
    fills = r.get("fills", {})
    if fills:
        simple = {c: f for c, f in fills.items() if f["strategy"] != pl.M_GROUP}
        group, used = [], set()
        for c, f in fills.items():  # z. B. total_acc_avg = Mittelwert von mort_acc je total_acc
            if f["strategy"] != pl.M_GROUP:
                continue
            var = _ident(f["group_col"], "_avg")
            if var in used:
                var = _ident(f"{c}_je_{f['group_col']}", "_avg")
            used.add(var)
            group.append((c, f["group_col"], var, _ident(c, "_mean")))
        L += ["", "# " + tx("Fehlende Werte: Füllwerte aus allen Zeilen (Clustering: kein Train/Test-Split)")]
        if simple:
            expr = {pl.M_MEAN: "filled[{c!r}].mean()", pl.M_MEDIAN: "filled[{c!r}].median()",
                    pl.M_MODE: "filled[{c!r}].mode()[0]"}
            L.append("fill = {")
            for c, f in simple.items():
                value = repr(f["value"]) if f["strategy"] == pl.M_CONST else expr[f["strategy"]].format(c=c)
                L.append(f"    {c!r}: {value},")
            L.append("}")
        for c, g, var, mean_var in group:
            L.append(f"{var} = filled.groupby({g!r})[{c!r}].mean()")
            L.append(f"{mean_var} = filled[{c!r}].mean()")
        if simple:
            L.append("filled = filled.fillna(fill)")
        for c, g, var, mean_var in group:  # zuletzt: die Gruppenspalte ist dann schon gefüllt
            L.append(f"filled[{c!r}] = filled[{c!r}].fillna(filled[{g!r}].map({var})).fillna({mean_var})")
    L.append("")
    if cat:
        L += ["# " + tx("Dummies – Kategorien aus den Daten (neue Daten: unbekannte Kategorie → alle Dummies 0)"),
              "categories = {c: sorted(filled[c].dropna().unique()) for c in categorical}", "",
              "def to_X(part):",
              "    X = part[features].copy()",
              "    for c, cats in categories.items():",
              "        X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)   # " + tx("unbekannt → NaN"),
              f"    X = pd.get_dummies(X, columns=list(categories), drop_first={r['drop_first']}, dtype=float)",
              "    return X.astype(float)   # " + tx("pandas ≥ 2: sonst True/False statt 0/1")]
    else:
        L += ["def to_X(part):", "    return part[features].astype(float)"]
    L += ["", "X_df = to_X(filled)   # " + tx("unskaliert – für die Profile in Originaleinheiten"),
          "feature_columns = list(X_df.columns)", ""]
    kind = r["scaler"]["type"]
    if kind == "keiner":
        L.append("X = X_df.values   # " + tx("keine Skalierung – große Zahlen bestimmen dann die Abstände"))
    else:
        L += [f"scaler = {kind}()",
              "X = scaler.fit_transform(X_df.values)   # "
              + tx("K-Means rechnet mit Abständen → alle Features auf dieselbe Skala")]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
EXCEL_NOTE = {"de": N_("für deutsches Excel"), "pl": N_("für polnisches Excel"),
              "en": N_("für Excel in Großbritannien, USA, Australien …")}
EXPORT_FILES = {"step2": N_("daten_aufbereitet"), "ml": N_("clustering_daten"), "ml_scaled": N_("clustering_daten_skaliert"),
                "clusters": N_("daten_mit_clustern")}


def export_code(what: str, recipe: dict | None = None, excel: str | None = None, names: list | None = None,
                columns: dict | None = None, data_columns: list | None = None) -> str:
    """what = 'step2' (Daten nach Schritt 2), 'ml' / 'ml_scaled' (nach Schritt 4, so sieht K-Means die Daten) oder
    'clusters' (Daten nach Schritt 2 + Cluster). excel = 'de', 'pl' oder 'en': CSV wie Excel in dieser Sprache –
    wie in der App mit entschärften Formeln (=, +, -, @ am Anfang eines Textes bekommen ein ').
    columns = Namen der neuen Spalten (Cluster, Cluster-Name, Abstand) in der gewählten Sprache;
    data_columns = Spalten der Daten nach Schritt 2 (ein schon vorhandener Name bekommt _2 – wie in der App)."""
    kw = note = ""  # Kommentar erst nach der schließenden Klammer (sonst wäre die Klammer auskommentiert)
    if excel:
        sep, dec = pl.EXCEL_CSV[excel]
        kw = (f", sep={sep!r}" if sep != "," else "") + (f", decimal={dec!r}" if dec != "." else "") + \
            ", encoding='utf-8-sig'"
        note = "   # " + tx(EXCEL_NOTE[excel])
    file = repr(tx(EXPORT_FILES[what]) + ".csv")
    r = recipe or {}
    if what == "step2":
        if not excel:
            return f"df.to_csv({file}, index=False)"
        L = ["out = df.copy()"]
    elif what in ("ml", "ml_scaled"):
        L = ["out = X_df.round(6)" if what == "ml" else
             "out = pd.DataFrame(X, columns=feature_columns, index=X_df.index).round(6)   # " + tx("skaliert")]
        taken = list(r.get("feature_columns", []))
        v = rows_var(r)
        if r.get("name_col"):
            col = pl.free_name(taken, r["name_col"])
            taken.insert(0, col)
            L.append(f"out.insert(0, {col!r}, {v}[{r['name_col']!r}].values)   # " + tx("Bezeichnung"))
        if r.get("group_col"):
            col = pl.free_name(taken, r["group_col"])
            L.append(f"out[{col!r}] = {v}[{r['group_col']!r}].values   # " + tx("bekannte Einteilung (nicht im Modell)"))
    else:
        columns = {pl.RESULT_CLUSTER: pl.RESULT_CLUSTER, pl.RESULT_NAME: pl.RESULT_NAME,
                   pl.RESULT_DIST: pl.RESULT_DIST, **(columns or {})}
        taken = [str(c) for c in (data_columns or [])]
        # alle Spalten nach Schritt 2 (auch in Schritt 4 entfernte), nur die Zeilen, die ins Clustering gingen
        L = ["out = df.loc[data.index].copy()   # " + tx("Zeilen, die ins Clustering gingen")
             if rows_var(r) == "data" else "out = df.copy()"]
        col = pl.free_name(taken, columns[pl.RESULT_CLUSTER])
        taken.append(col)
        L.append(f"out[{col!r}] = labels")
        names = pl.clean_names(names, len(names or []))
        if any(names):
            col = pl.free_name(taken, columns[pl.RESULT_NAME])
            taken.append(col)
            L.append(f"out[{col!r}] = pd.Series(labels).map({dict(enumerate(names))!r}).values")
        col = pl.free_name(taken, columns[pl.RESULT_DIST])
        L.append(f"out[{col!r}] = np.round(np.sqrt(((X - kmeans.cluster_centers_[labels]) ** 2).sum(axis=1)), 4)   # "
                 + tx("Abstand zum eigenen Zentrum"))
    if excel:
        L += ["for c in out.columns[~out.dtypes.map(pd.api.types.is_numeric_dtype)]:   # "
              + tx("Excel: Text wie =1+1 nicht als Formel ausführen"),
              "    risky = out[c].notna() & out[c].astype(str).str.match(r'^[=+\\-@\\t\\r]')",
              "    out[c] = out[c].astype(object).where(~risky, \"'\" + out[c].astype(str))"]
    return "\n".join(L + [f"out.to_csv({file}, index=False{kw}){note}"])


# ---------------------------------------------------------------------------
# 6) Modell & Training – K-Means
# ---------------------------------------------------------------------------
def kmeans_code(cfg: pl.ClusterConfig, k: str | int | None = None) -> str:
    """KMeans(...) – nur Parameter, die vom Standard abweichen; n_init immer (Standard 'auto' = 1 Start)."""
    args = [f"n_clusters={k if k is not None else int(cfg.n_clusters)}"]
    if cfg.init != "k-means++":
        args.append(f"init={cfg.init!r}")
    args.append(f"n_init={int(cfg.n_init)}")
    if int(cfg.max_iter) != 300:
        args.append(f"max_iter={int(cfg.max_iter)}")
    if cfg.algorithm != "lloyd":
        args.append(f"algorithm={cfg.algorithm!r}")
    args.append(f"random_state={int(cfg.seed)}")
    return f"KMeans({', '.join(args)})"


def _silhouette_call(x: str, labels: str, n: int, seed: int) -> str:
    """silhouette_score wie in der App: ab 5000 Zeilen auf einer festen Stichprobe."""
    if n > pl.SIL_ROWS:
        return f"silhouette_score({x}, {labels}, sample_size={pl.SIL_ROWS}, random_state={int(seed)})"
    return f"silhouette_score({x}, {labels})"


def model_code(cfg: pl.ClusterConfig, n_rows: int) -> str:
    """Code für Schritt 6: K-Means auf allen Zeilen trainieren (wie im Kurs: kmeans.fit(...))."""
    return "\n".join([
        f"kmeans = {kmeans_code(cfg)}",
        "kmeans.fit(X)   # " + tx("Unsupervised: K-Means bekommt keine Antwort (kein y)"),
        "labels = kmeans.labels_   # " + tx("Cluster-Nummer je Zeile (0 … K−1, die Reihenfolge ist willkürlich)"),
        "print(kmeans.cluster_centers_)   # " + tx("Zentren (skaliert) – wie im Kurs"),
        "print('Inertia:', kmeans.inertia_)",
        f"print('Silhouette:', {_silhouette_call('X', 'labels', n_rows, cfg.seed)})",
    ])


def curve_code(cfg: pl.ClusterConfig, n_rows: int) -> str:
    """Code für Schritt 7: Elbow-Methode (Inertia je K) und Silhouette je K."""
    if not cfg.curve:
        return "# " + tx("keine Elbow-/Silhouette-Kurve (in Schritt 6 ausgeschaltet)")
    L = []
    rows = pl.CURVE_ROWS if n_rows > pl.CURVE_ROWS else n_rows
    if n_rows > pl.CURVE_ROWS:
        L.append(f"rows = np.random.default_rng({int(cfg.seed)}).choice(len(X), {pl.CURVE_ROWS}, replace=False)   # "
                 + tx("Stichprobe – K-Means wird für jedes K neu trainiert"))
        L.append("X_curve = X[rows]")
    else:
        L.append("X_curve = X")
    values = pl.k_values(cfg, rows)
    L += [f"values = {values!r}   # K",
          "inertia, silhouette = [], []",
          "for k in values:",
          f"    km = {kmeans_code(cfg, 'k')}.fit(X_curve)",
          "    inertia.append(km.inertia_)",
          f"    silhouette.append({_silhouette_call('X_curve', 'km.labels_', rows, cfg.seed)} if k > 1 else np.nan)",
          "",
          "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))",
          "ax1.plot(values, inertia, marker='o')",
          f"ax1.set(title={tx('Elbow-Methode')!r}, xlabel='K', ylabel='Inertia')",
          "ax2.plot(values[1:], silhouette[1:], marker='o', color='green')",
          f"ax2.set(title='Silhouette', xlabel='K', ylabel={tx('mittlere Silhouette')!r})",
          "plt.show()"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 8) Cluster-Auswertung
# ---------------------------------------------------------------------------
def eval_code(recipe: dict, cfg: pl.ClusterConfig) -> str:
    """Code für Schritt 8: Größen, Profile, PCA-Darstellung und – falls vorhanden – Vergleich mit der bekannten
    Einteilung wie im Kurs (confusion_matrix, classification_report)."""
    L = ["print(pd.Series(labels).value_counts().sort_index())   # " + tx("Größe der Cluster"), "",
         "# " + tx("Profile: Mittelwert je Cluster in Originaleinheiten (Dummies: Anteil)"),
         "profile = X_df.groupby(labels).mean().T",
         f"profile.insert(0, {tx('Alle')!r}, X_df.mean())",
         "print(profile.round(2))",
         "z = (X_df.groupby(labels).mean() - X_df.mean()) / X_df.std(ddof=0)   # "
         + tx("Abweichung vom Durchschnitt in Standardabweichungen"),
         "sns.heatmap(z.T, annot=True, fmt='+.1f', cmap='RdBu_r', center=0, vmin=-2, vmax=2)",
         "plt.show()", "",
         "# " + tx("PCA auf 2 Dimensionen – nur zur Darstellung (K-Means nutzt alle Features)"),
         f"pca = PCA(n_components={min(2, len(recipe['feature_columns']))}, random_state={int(cfg.seed)})",
         "xy = pca.fit_transform(X)",
         "centers = pca.transform(kmeans.cluster_centers_)"]
    if len(recipe["feature_columns"]) >= 2:
        L += ["plt.figure(figsize=(9, 6))",
              "plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap='tab10', s=12, alpha=0.6)",
              "plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200, c='black')",
              "plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.0%})')",
              "plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.0%})')",
              "plt.show()"]
    g = recipe.get("group_col")
    if g:
        values = recipe.get("group_values") or []
        L += ["", "# " + _cm(tx("Vergleich mit der bekannten Einteilung '{g}' – K-Means hat sie nie gesehen", g=g)),
              "known = groups.notna().values",
              "g, lab = groups.values[known], labels[known]",
              "print(pd.crosstab(lab, g))   # " + tx("Kreuztabelle Cluster × Gruppe")]
        if len(values) == 2 and int(cfg.n_clusters) == 2:
            L += [f"codes = pd.Series(g).map({dict(zip(values, range(2)))!r}).values   # "
                  + _cm(tx("wie im Kurs: {a} → 0, {b} → 1", a=values[0], b=values[1])),
                  "print(confusion_matrix(codes, lab))   # "
                  + tx("Achtung: die Cluster-Nummern sind willkürlich – evtl. vertauscht")]
        L += ["mapping = pd.crosstab(lab, g).idxmax(axis=1)   # " + tx("jeder Cluster → häufigste Gruppe darin"),
              "predicted = pd.Series(lab).map(mapping).values",
              "print(classification_report(g, predicted, zero_division=0))"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 9) Speichern, 10) Neue Daten
# ---------------------------------------------------------------------------
def save_code(folder: str, fmt: str = ".skops") -> str:
    path = f"{folder}/model{fmt}"
    if fmt == ".joblib":
        lines = ["import os", "import joblib", f"os.makedirs({folder!r}, exist_ok=True)",
                 f"joblib.dump(kmeans, {path!r})   # " + tx("Standard von scikit-learn (pickle)"),
                 f"later_model = joblib.load({path!r})   # "
                 + tx("nur vertrauenswürdige Dateien laden – pickle kann beim Laden Code ausführen")]
    else:
        lines = ["import os", "import skops.io as sio", f"os.makedirs({folder!r}, exist_ok=True)",
                 f"sio.dump(kmeans, {path!r})   # " + tx("sicheres Format: beim Laden wird kein Code ausgeführt"),
                 f"unknown = sio.get_untrusted_types(file={path!r})",
                 "trusted = [t for t in unknown if t.startswith('sklearn.')]   # "
                 + tx("nur scikit-learn-Typen vertrauen – alles andere lehnt sio.load ab"),
                 f"later_model = sio.load({path!r}, trusted=trusted)"]
    return ("\n".join(lines) + "\n\n"
            "# " + tx("Wichtig: die Vorverarbeitung gehört zum Modell! Die App speichert sie in") + "\n"
            "# " + _cm(tx("'{folder}/preprocessing.json' (Spalten-Aktionen, Füllwerte, Dummy-Spalten, Scaler)",
                          folder=folder)))


def predict_code(folder: str) -> str:
    return ("import pipeline as pl\n\n"
            f"kmeans, recipe = pl.load_bundle({folder!r})\n"
            f"df_new, _ = pl.read_csv({tx('neue_daten.csv')!r})\n"
            "result, report, comparison = pl.assign_new(kmeans, recipe, df_new)   # "
            + tx("Cluster, Abstand, Hinweis") + "\n"
            f"result.to_csv({tx('cluster_zuordnung.csv')!r}, index=False)\n\n"
            "# " + tx("Ohne pipeline.py: neue Daten genauso aufbereiten wie oben (Schritt 2, fill, to_X) und dann")
            + "\n# " + tx("NUR transformieren – nichts neu fitten:") + " "
            "kmeans.predict(scaler.transform(to_X(df_new_filled).values))")


def full_script(source: dict, actions: list[dict], numeric_cols: set, recipe: dict, cfg: pl.ClusterConfig,
                folder: str = "models/mein_modell", fmt: str = ".skops", new_columns: list[dict] | None = None) -> str:
    """Gesamte Pipeline als lauffähiges Python-Skript (für Jupyter oder die Kommandozeile)."""
    n = int((recipe.get("n_rows") or {}).get("train", 0))
    parts = [
        '"""' + tx("Automatisch erzeugt mit DATA-Analysis-Clustering – entspricht den Einstellungen in der App.")
        + '"""',
        "# " + tx("1) Bibliotheken"), IMPORTS, "",
        "# " + tx("Daten laden (+ Stichprobe)"),
        load_code(source.get("path") or source.get("name", "daten.csv"), source.get("sep", ","),
                  source.get("decimal", "."), source.get("encoding", "utf-8"), source.get("sample_frac"),
                  strip_columns=bool(source.get("strip_columns"))), "",
        "# " + tx("2) Datenaufbereitung 1"), step2_code(actions, numeric_cols, new_columns), "",
        "# " + tx("4) Datenaufbereitung 2"), step4_code(recipe), "",
        "# " + tx("6) Modell & Training") + " – K-Means", model_code(cfg, n), "",
        "# " + tx("7) Wahl von K"), curve_code(cfg, n), "",
        "# " + tx("8) Cluster-Auswertung"), eval_code(recipe, cfg), "",
        "# " + tx("9) Speichern & Laden"), save_code(folder, fmt),
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# 3) Diagramme
# ---------------------------------------------------------------------------
def chart_code(kind: str, **p) -> str:
    def q(v):
        return repr(v)

    hue = f", hue={q(p['hue'])}" if p.get("hue") else ""
    if kind == "overview":
        return "df.info()\ndf.describe().transpose()\n100 * df.isnull().sum() / len(df)"
    if kind == "count":
        order = f", order=sorted(df[{q(p['x'])}].dropna().unique())" if p.get("order_by") == "Wert" else ""
        return f"plt.figure(figsize=(12, 4))\nsns.countplot(x={q(p['x'])}, data=df{hue}{order})"
    if kind == "hist":
        kde = ", kde=True" if p.get("kde") else ""
        el = ", element='step'" if p.get("hue") else ""
        return f"plt.figure(figsize=(12, 4))\nsns.histplot(data=df, x={q(p['x'])}{hue}, bins={p.get('bins', 40)}{kde}{el})"
    if kind == "box":
        return f"sns.boxplot(x={q(p['x'])}, y={q(p['y'])}, data=df{hue})"
    if kind == "scatter":
        return f"sns.scatterplot(x={q(p['x'])}, y={q(p['y'])}, data=df{hue})"
    if kind == "heatmap":
        return ("plt.figure(figsize=(12, 7))\n"
                f"sns.heatmap(df.corr(numeric_only=True, method={q(p.get('method', 'pearson'))}), annot=True, "
                "cmap='RdBu_r', center=0, vmin=-1, vmax=1)   # " + tx("numeric_only=True: nötig ab pandas 2"))
    if kind == "corr_target":
        t = p["target"]
        if p.get("positive") is not None:
            return (f"y = (df[{q(t)}].astype(str) == {q(p['positive'])}).astype(int)\n"
                    "df.select_dtypes('number').corrwith(y).sort_values().plot(kind='barh')")
        return f"df.corr(numeric_only=True)[{q(t)}].sort_values().drop({q(t)}).plot(kind='barh')"
    if kind == "pairplot":
        cols = list(p["cols"]) + ([p["hue"]] if p.get("hue") and p["hue"] not in p["cols"] else [])
        return f"sns.pairplot(df[{cols!r}].sample(min({p.get('n', 1500)}, len(df))){hue}, corner=True)"
    if kind == "rate":
        return (f"rate = df.groupby({q(p['cat'])})[{q(p['target'])}].apply("
                f"lambda s: (s.astype(str) == {q(p['positive'])}).mean())\n"
                "rate.plot(kind='bar')")
    if kind == "groupby":
        return f"df.groupby({q(p['by'])})[{q(p['col'])}].describe()"
    sample = f".sample(min({p.get('n', 5000)}, len(df)), random_state=101)   # " + tx("Stichprobe")
    if kind == "joint":
        jk = p.get("joint_kind", "scatter")
        joint_hue = hue if jk in ("scatter", "kde") else ""
        return (f"d = df{sample}\n"
                f"sns.jointplot(x={q(p['x'])}, y={q(p['y'])}, data=d{joint_hue}"
                + (f", kind={q(jk)})" if jk != "scatter" else ")"))
    if kind == "kde2d":
        lines = []
        if p.get("filter_col"):
            fc, fv = p["filter_col"], p.get("filter_value")
            lines.append(f"df_f = df[df[{q(fc)}] == {_literal(fv, p.get('filter_numeric', False))}]   # "
                         + _cm(tx("nur Zeilen mit {col} = {val}", col=fc, val=fv)))
            lines.append(f"d = df_f{sample.replace('len(df)', 'len(df_f)')}")
        else:
            lines.append(f"d = df{sample}")
        fill = ", fill=True" if p.get("fill", True) else ""
        lines.append(f"sns.kdeplot(x={q(p['x'])}, y={q(p['y'])}, data=d, cmap='plasma'{fill}, thresh=0.05)   # "
                     + tx("seaborn ≥ 0.11: x=/y= als Schlüsselwörter, fill statt shade, thresh statt shade_lowest"))
        return "\n".join(lines)
    if kind == "lm":
        col = f", col={q(p['col'])}" + (", col_wrap=3" if p.get("wrap") else "") if p.get("col") else ""
        return (f"d = df{sample}\n"
                f"sns.lmplot(x={q(p['x'])}, y={q(p['y'])}, data=d{hue}{col}, height=4, aspect=1.15)   # "
                + tx("lmplot erstellt eine eigene Figure: Größe über height/aspect – plt.figure() davor wirkt nicht"))
    if kind == "hist_groups":
        x, g = p["x"], p["group"]
        return ("plt.figure(figsize=(10, 6))\n"
                f"bins = np.histogram_bin_edges(df[{q(x)}].dropna(), bins={p.get('bins', 30)})   # "
                + tx("gleiche Klassen (bins) für alle Gruppen") + "\n"
                f"for value in sorted(df[{q(g)}].dropna().unique()):\n"
                f"    df[df[{q(g)}] == value][{q(x)}].hist(alpha=0.5, bins=bins, label={g + '='!r} + str(value))\n"
                "plt.legend()\n"
                f"plt.xlabel({q(x)})")
    return ""


def _literal(value, numeric: bool) -> str:
    """Wert aus einer Auswahlliste (Text) als Python-Literal für den Vergleich df[spalte] == …"""
    v = str(value)
    if numeric:
        if v in ("True", "False"):
            return v
        try:
            f = float(v)
        except ValueError:
            return repr(v)
        return repr(int(f)) if f.is_integer() else repr(f)
    return repr(v)
