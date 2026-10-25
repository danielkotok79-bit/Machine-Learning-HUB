"""
codegen.py – erzeugt den passenden Python-Code (pandas, scikit-learn) zu den Einstellungen der App.

Zum Nachvollziehen und um eine Pipeline 1:1 in ein Jupyter Notebook oder Skript zu übernehmen. Der erzeugte
Code rechnet dasselbe wie die App (pipeline.py) – das prüft tests/test_codegen_parity.py.
"""
from __future__ import annotations

import re
from dataclasses import replace

import pipeline as pl
from i18n import N_, tx  # Kommentare im erzeugten Code in der gewählten Sprache

IMPORTS = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.metrics import classification_report, confusion_matrix"""

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
# 4) Datenaufbereitung 2
# ---------------------------------------------------------------------------
def time_split_lines(sp: dict) -> list[str]:
    """Zeitlicher Split wie pipeline.time_split: nach der Zeitspalte sortieren (stabil), Test = die jüngsten
    ceil(test_size · n) Zeilen – Zeilen mit demselben Zeitwert wie die erste Testzeile ebenfalls in den Test."""
    col = sp["time_column"]
    inf = ".replace([np.inf, -np.inf], np.nan)" if sp.get("time_removed") else ""
    if sp.get("time_kind") == "date":
        L = [f"t = pd.to_datetime(df[{col!r}], format={sp.get('time_format')!r}, errors='coerce')   # "
             + tx("Zeitspalte → Zeitpunkte")]
    else:
        L = [f"t = df[{col!r}].astype(float){inf}   # " + tx("Zeitspalte als Zahl (z. B. Jahr oder Saison)")]
    if sp.get("time_removed"):
        L.append("df, t = df[t.notna()], t[t.notna()]   # " + tx("Zeilen ohne Zeitwert entfernen"))
    return L + [
        "order = t.sort_values(kind='mergesort').index   # "
        + tx("nach der Zeit sortieren – stabil: gleiche Zeit behält die ursprüngliche Reihenfolge"),
        "df, t = df.loc[order], t.loc[order]",
        f"first_test = t.iloc[-int(np.ceil({sp['test_size']} * len(df)))]   # " + tx("Zeitpunkt der ersten Testzeile"),
        "train, test = df[t < first_test], df[t >= first_test]   # "
        + tx("Test = die jüngsten Zeilen, Training = alles davor (kein Blick in die Zukunft)"),
    ]


def step4_code(r: dict) -> str:
    """Code für Schritt 4 in derselben Reihenfolge wie die App: Split zuerst, dann Füllwerte, Kategorien und
    Scaler nur aus den Trainingsdaten lernen und auf Train und Test anwenden."""
    label, task = r["label"], r["task"]
    L = [f"df = df.dropna(subset=[{label!r}])   # " + tx("Zeilen ohne Label")]
    if r.get("label_inf_removed"):
        L.append(f"df = df[np.isfinite(df[{label!r}])]   # " + tx("Zeilen mit inf im Label"))
    if r.get("inf_columns"):
        L.append(f"df[{r['inf_columns']!r}] = df[{r['inf_columns']!r}].replace([np.inf, -np.inf], np.nan)   # "
                 + tx("inf wie fehlend behandeln"))
    sp = r["split"]
    time_col = sp.get("time_column") if sp.get("mode") == pl.SPLIT_TIME else None
    # Zeitspalte (Split) und Vergleichsspalte (Schritt 8) werden später noch gebraucht – nicht aus df entfernen;
    # als Feature fallen sie trotzdem weg (features enthält sie nicht)
    keep = {time_col, r.get("baseline_column")} - {None}
    dropped = [c for c in r.get("dropped_columns") or [] if c not in keep]
    if dropped:
        L.append(f"df = df.drop({dropped!r}, axis=1)   # " + tx("Spalten mit zu vielen Lücken"))
    if r.get("dropna_columns"):
        L.append(f"df = df.dropna(subset={r['dropna_columns']!r})")
    cat = r["categorical_features"]
    if cat:
        L += [f"categorical = {cat!r}   # " + tx("→ Dummies"),
              "df[categorical] = df[categorical].apply(lambda s: s.astype(str).str.strip().where(s.notna()))   # "
              + tx("als Text, ohne Leerzeichen am Rand")]
    if task != pl.T_REG:
        classes = r["classes"]
        values = r.get("class_values")
        if values is None:  # ältere Rezepte
            values = [_num(c) for c in classes] if r.get("label_is_numeric") else classes
        mapping = dict(zip(values, range(len(values))))
        if r.get("label_kind") == "text":
            L.append(f"df[{label!r}] = df[{label!r}].str.strip().map({mapping!r})")
        elif not all(k == v for k, v in mapping.items()):
            L.append(f"df[{label!r}] = df[{label!r}].map({mapping!r})")
        elif r.get("label_kind") == "bool":
            L.append(f"df[{label!r}] = df[{label!r}].astype(int)   # False/True → 0/1")

    L += ["", f"features = {r['features']!r}", "",
          "# " + tx("Zuerst aufteilen – Füllwerte, Kategorien und Scaler werden danach NUR aus den Trainingsdaten "
                   "gelernt")]
    if time_col:
        L += time_split_lines(sp)
    else:
        strat = (f", stratify=df[{label!r}]"
                 if sp.get("stratify_used", sp.get("stratify")) and task != pl.T_REG else "")
        L.append(f"train, test = train_test_split(df, test_size={sp['test_size']}, random_state={sp['random_state']}"
                 f"{strat})")

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
        L += ["", "# " + tx("Fehlende Werte: Füllwerte aus den Trainingsdaten berechnen …")]
        if simple:
            expr = {pl.M_MEAN: "train[{c!r}].mean()", pl.M_MEDIAN: "train[{c!r}].median()",
                    pl.M_MODE: "train[{c!r}].mode()[0]"}
            L.append("fill = {")
            for c, f in simple.items():
                value = repr(f["value"]) if f["strategy"] == pl.M_CONST else expr[f["strategy"]].format(c=c)
                L.append(f"    {c!r}: {value},")
            L.append("}")
        for c, g, var, mean_var in group:
            L.append(f"{var} = train.groupby({g!r})[{c!r}].mean()")
            L.append(f"{mean_var} = train[{c!r}].mean()")
        L += ["", "def fill_missing(part):",
              '    """' + tx("… und auf Train UND Test anwenden – die Testdaten lernen nichts mit.") + '"""',
              "    part = part.fillna(fill)" if simple else "    part = part.copy()"]
        for c, g, var, mean_var in group:  # zuletzt: die Gruppenspalte ist dann schon gefüllt
            L.append(f"    part[{c!r}] = part[{c!r}].fillna(part[{g!r}].map({var})).fillna({mean_var})")
        L += ["    return part", "", "train, test = fill_missing(train), fill_missing(test)"]

    L.append("")
    if cat:
        L += ["# " + tx("Dummies mit den Kategorien der Trainingsdaten (Kategorie nur im Test → alle Dummies 0)"),
              "categories = {c: sorted(train[c].dropna().unique()) for c in categorical}", "",
              "def to_X(part):",
              "    X = part[features].copy()",
              "    for c, cats in categories.items():",
              "        X[c] = pd.Categorical(X[c].where(X[c].isin(cats)), categories=cats)   # " + tx("unbekannt → NaN"),
              f"    X = pd.get_dummies(X, columns=list(categories), drop_first={r['drop_first']}, dtype=float)",
              "    return X.astype(float)   # " + tx("pandas ≥ 2: sonst True/False statt 0/1")]
    else:
        L += ["def to_X(part):", "    return part[features].astype(float)"]
    L += ["", "X_train, X_test = to_X(train), to_X(test)",
          "feature_columns = list(X_train.columns)",
          "X_train, X_test = X_train.values, X_test.values",
          f"y_train, y_test = train[{label!r}].values, test[{label!r}].values", ""]
    kind = r["scaler"]["type"]
    if kind == "keiner":
        L.append("# " + tx("keine Skalierung"))
    else:
        if kind == "StandardScaler":
            L.append("from sklearn.preprocessing import StandardScaler")
        L += [f"scaler = {kind}()", "X_train = scaler.fit_transform(X_train)   # " + tx("fit nur auf Trainingsdaten!"),
              "X_test = scaler.transform(X_test)"]
    if r.get("label_scaler"):
        L += ["", "# " + tx("Label skalieren (Regression) – Vorhersagen später zurückrechnen"),
              "y_mean, y_std = y_train.mean(), y_train.std()",
              "y_train = (y_train - y_mean) / y_std",
              "y_test = (y_test - y_mean) / y_std"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
EXCEL_NOTE = {"de": N_("für deutsches Excel"), "pl": N_("für polnisches Excel"),
              "en": N_("für Excel in Großbritannien, USA, Australien …")}


def export_code(what: str, german: bool, label: str, scaled: bool = False, split_col: str = "split",
                excel: str | None = None) -> str:
    """what = 'step2' (Daten nach Schritt 2) oder 'ml' (ML-fertige Daten nach Schritt 4).
    excel = 'de', 'pl' oder 'en': CSV wie Excel in dieser Sprache/Region (german=True entspricht 'de')."""
    excel = excel or ("de" if german else None)
    kw = note = ""  # Kommentar erst nach der schließenden Klammer (sonst wäre die Klammer auskommentiert)
    if excel:
        sep, dec = pl.EXCEL_CSV[excel]
        kw = (f", sep={sep!r}" if sep != "," else "") + (f", decimal={dec!r}" if dec != "." else "") + \
            ", encoding='utf-8-sig'"
        note = "   # " + tx(EXCEL_NOTE[excel])
    if what == "step2":
        return f"df.to_csv({tx('daten_aufbereitet') + '.csv'!r}, index=False{kw}){note}"
    s = repr(split_col)
    if scaled:  # Label unskaliert (wie in der App), auch wenn y_train für das Training skaliert wurde
        parts = [f"    pd.DataFrame(X_train, columns=feature_columns, index=train.index)"
                 f".assign(**{{{label!r}: train[{label!r}], {s}: 'train'}}),",
                 f"    pd.DataFrame(X_test, columns=feature_columns, index=test.index)"
                 f".assign(**{{{label!r}: test[{label!r}], {s}: 'test'}}),"]
    else:
        parts = [f"    to_X(train).assign(**{{{label!r}: train[{label!r}], {s}: 'train'}}),",
                 f"    to_X(test).assign(**{{{label!r}: test[{label!r}], {s}: 'test'}}),"]
    return "\n".join(["ml = pd.concat([", *parts, "]).sort_index()   # " + tx("ursprüngliche Reihenfolge der Zeilen"),
                      f"ml.to_csv({tx('ml_daten') + '.csv'!r}, index=False{kw}){note}"])


# ---------------------------------------------------------------------------
# 6) Modell & Training
# ---------------------------------------------------------------------------
def estimator_code(cfg: pl.ModelConfig, task: str, *, plain: bool = False, calib_cv: int = 5, override=None,
                   grid: bool = False) -> str:
    """scikit-learn-Ausdruck für das Modell – nur Parameter, die vom Standard abweichen (wie pipeline.make_model).
    plain=True: SVC ohne CalibratedClassifierCV · override: {Parameter: Code}, z. B. {'n_neighbors': 'k'} für Schleifen
    · grid=True: C/gamma aus GridSearchCV (**grid.best_params_)."""
    clf = task != pl.T_REG
    cw = "'balanced'" if cfg.class_weight and pl.supports_class_weight(cfg, task) else None
    m = cfg.model
    if m == pl.MODEL_GB:
        name = "HistGradientBoosting" + ("Classifier" if clf else "Regressor")
        d = pl.GB_DEFAULTS
        n_iter = min(max(int(cfg.gb_iter), 1), pl.MAX_TREES)
        depth = pl._depth(cfg.gb_depth)
        args = {"max_iter": None if n_iter == d["gb_iter"] else str(n_iter),
                "learning_rate": None if float(cfg.gb_lr) == d["gb_lr"] else _float(cfg.gb_lr),
                "max_leaf_nodes": None if max(int(cfg.gb_leaves), 2) == d["gb_leaves"] else str(max(int(cfg.gb_leaves), 2)),
                "max_depth": str(depth) if depth else None,
                "min_samples_leaf": None if max(int(cfg.gb_leaf), 1) == d["gb_leaf"] else str(max(int(cfg.gb_leaf), 1)),
                "early_stopping": "False", "class_weight": cw if clf else None, "random_state": str(int(cfg.seed))}
    elif m == pl.MODEL_TREES:
        single = cfg.tree_variant == pl.TREE_SINGLE
        name = ("DecisionTree" if single else "RandomForest") + ("Classifier" if clf else "Regressor")
        depth = pl._depth(cfg.max_depth)
        args = {"n_estimators": None if single else str(min(max(int(cfg.n_estimators), 1), pl.MAX_TREES)),
                "max_depth": str(depth) if depth else None,
                "min_samples_leaf": str(int(cfg.min_samples_leaf)) if int(cfg.min_samples_leaf) > 1 else None}
        if not single:
            mf = pl._max_features(cfg.max_features)
            args["max_features"] = None if mf == ("sqrt" if clf else 1.0) else repr(mf)
        args.update(class_weight=cw if clf else None, random_state=str(int(cfg.seed)), n_jobs=None if single else "-1")
    elif m == pl.MODEL_SVM:
        name = "SVC" if clf else "SVR"
        args = {"kernel": repr(cfg.kernel), "C": None if grid else _float(cfg.svm_c),
                "gamma": None if grid else repr(pl.gamma_value(cfg.gamma)),
                "epsilon": None if clf else _float(cfg.epsilon), "class_weight": cw}
    elif m == pl.MODEL_LOGREG:
        name = "LogisticRegression"
        args = {"C": _float(cfg.logreg_c), "max_iter": str(int(cfg.max_iter)), "class_weight": cw,
                "random_state": str(int(cfg.seed))}
    elif m == pl.MODEL_KNN:
        name = "KNeighborsClassifier" if clf else "KNeighborsRegressor"
        args = {"n_neighbors": str(int(cfg.n_neighbors)), "weights": None if cfg.weights == "uniform" else repr(cfg.weights),
                "p": None if int(cfg.p) == 2 else str(int(cfg.p))}
    else:
        name = "LinearRegression"
        args = {"fit_intercept": None if cfg.fit_intercept else "False"}
    args.update(override or {})
    parts = [f"{k}={v}" for k, v in args.items() if v is not None] + (["**grid.best_params_"] if grid else [])
    code = f"{name}({', '.join(parts)})"
    if m == pl.MODEL_SVM and clf and not plain:
        code = f"CalibratedClassifierCV({code}, ensemble=False{'' if calib_cv == 5 else f', cv={calib_cv}'})"
    return code


def _grid_param(cfg: pl.ModelConfig) -> dict:
    return {"C": pl.GRID_C} if cfg.kernel == "linear" else {"C": pl.GRID_C, "gamma": pl.GRID_GAMMA}


def model_code(cfg: pl.ModelConfig, task: str, calib_cv: int = 5, grid_cv: int | None = None,
               time_split: bool = False) -> str:
    """Code für Schritt 6: Modell anlegen und trainieren (optional vorher GridSearchCV wie im Kurs).
    grid_cv: Folds, die die App tatsächlich nutzt (pipeline.grid_cv – bei seltenen Klassen/wenigen Zeilen weniger) ·
    time_split: zeitlicher Split in Schritt 4 → GridSearchCV mit TimeSeriesSplit."""
    L = []
    grid = cfg.model == pl.MODEL_SVM and cfg.grid_search
    if grid:
        base = estimator_code(replace(cfg, grid_search=False), task, plain=True, override={"C": None, "gamma": None})
        folds = grid_cv or min(max(int(cfg.cv), 2), 10)
        cv = f"TimeSeriesSplit(n_splits={folds})" if time_split else str(folds)
        L += [f"param_grid = {_grid_param(cfg)!r}   # " + tx("wie im Kurs"),
              f"grid = GridSearchCV({base}, param_grid, cv={cv}, n_jobs=-1)"
              + ("   # " + tx("zeitlicher Split: jeder Fold validiert auf späteren Zeilen") if time_split else ""),
              "grid.fit(X_train, y_train)",
              "print(grid.best_params_, grid.best_score_)", ""]
    if cfg.model == pl.MODEL_SVM and task != pl.T_REG:
        L.append("# " + tx("Wahrscheinlichkeiten für Schwelle und ROC-AUC: CalibratedClassifierCV statt "
                           "SVC(probability=True) – das ist seit scikit-learn 1.9 veraltet"))
    L += [f"model = {estimator_code(cfg, task, calib_cv=calib_cv, grid=grid)}", "model.fit(X_train, y_train)"]
    if cfg.model == pl.MODEL_LINREG:
        L.append("print('Coefficients:', model.coef_)   # " + tx("Koeffizienten – wie im Kurs: lm.coef_"))
    return "\n".join(L)


def curve_code(cfg: pl.ModelConfig, task: str, label_scaled: bool = False, time_split: bool = False) -> str:
    """Code für Schritt 7: Fehler auf Training und Validierung je Parameterwert – wie die Elbow-Methode im Kurs
    (die App wertet dabei Stichproben von höchstens 5000 Zeilen aus). time_split: Validierungsanteil = der jüngste
    Teil der (nach der Zeit sortierten) Trainingsdaten."""
    clf = task != pl.T_REG
    L = []
    if cfg.model == pl.MODEL_SVM and cfg.grid_search:
        L += ["results = pd.DataFrame(grid.cv_results_)",
              "table = results.pivot_table(index='param_gamma', columns='param_C', values='mean_test_score')"
              if cfg.kernel != "linear" else "table = results.pivot_table(columns='param_C', values='mean_test_score')",
              "sns.heatmap(table, annot=True, fmt='.3f', cmap='Blues')   # " + tx("GridSearchCV: mittlere Validierungsgüte"),
              "plt.show()", ""]
    if not cfg.curve:
        return "\n".join(L) if L else "# " + tx("keine Validierungskurve (in Schritt 6 ausgeschaltet)")
    if cfg.validation == "split" and time_split:
        L.append(f"X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size={_float(cfg.validation_split)}, "
                 "shuffle=False)   # " + tx("Validierung = der jüngste Teil der Trainingsdaten"))
    elif cfg.validation == "split":
        strat = ", stratify=y_train" if clf else ""
        L.append(f"X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size={_float(cfg.validation_split)}, "
                 f"random_state={int(cfg.seed)}{strat})   # " + tx("Validierungsanteil aus den Trainingsdaten"))
    else:
        L.append("X_fit, y_fit, X_val, y_val = X_train, y_train, X_test, y_test   # "
                 + tx("Validierung mit den Testdaten (wie im Kurs)"))
    if clf:
        L.append("error = lambda y, pred: np.mean(pred != y)   # " + tx("Fehlerrate (Error Rate)"))
    else:
        L.append("error = lambda y, pred: np.mean(np.abs(pred - y))" + (" * y_std   # MAE – " + tx("zurück in Originaleinheiten")
                                                                       if label_scaled else "   # MAE"))
    L.append("error_train, error_val = [], []")
    m, x = cfg.model, "values"
    if m == pl.MODEL_GB:
        L += [f"values = {pl.forest_steps(cfg.gb_iter)!r}",
              f"m = {estimator_code(cfg, task)}",
              "m.fit(X_fit, y_fit)   # " + tx("EIN Training – staged_predict liefert die Vorhersage nach jeder Runde"),
              "stages = zip(m.staged_predict(X_fit), m.staged_predict(X_val))",
              "for n, (pred_fit, pred_val) in enumerate(stages, start=1):",
              "    if n in values:",
              "        error_train.append(error(y_fit, pred_fit))",
              "        error_val.append(error(y_val, pred_val))"]
        xlabel = "max_iter"
    elif m == pl.MODEL_TREES and cfg.tree_variant != pl.TREE_SINGLE:
        L += [f"values = {pl.forest_steps(cfg.n_estimators)!r}",
              f"m = {estimator_code(cfg, task)}",
              "m.set_params(warm_start=True)   # " + tx("jeder Schritt fügt nur neue Bäume hinzu"),
              "for n in values:",
              "    m.set_params(n_estimators=n)",
              "    m.fit(X_fit, y_fit)"]
        xlabel = "n_estimators"
    elif m == pl.MODEL_TREES:
        L += ["values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]   # None = " + tx("unbegrenzt"),
              "for d in values:",
              f"    m = {estimator_code(cfg, task, override={'max_depth': 'd'})}",
              "    m.fit(X_fit, y_fit)"]
        xlabel, x = "max_depth", "[str(v) for v in values]"
    elif m in (pl.MODEL_SVM, pl.MODEL_LOGREG):
        values = [0.01, 0.1, 1, 10, 100] if m == pl.MODEL_SVM else [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        override = {"C": "c"}
        if m == pl.MODEL_SVM and cfg.grid_search:  # C aus GridSearchCV liegt immer im Raster
            if cfg.kernel != "linear":
                override["gamma"] = "grid.best_params_['gamma']"  # bestes gamma aus GridSearchCV, wie in der App
        else:  # eingestelltes C gehört zur Kurve (wie in der App)
            values = sorted(set(values) | {float(cfg.svm_c if m == pl.MODEL_SVM else cfg.logreg_c)})
        L += ["values = [" + ", ".join(_float(v) for v in values) + "]",
              "for c in values:",
              f"    m = {estimator_code(replace(cfg, grid_search=False), task, plain=True, override=override)}",
              "    m.fit(X_fit, y_fit)"]
        xlabel = "C"
    elif m == pl.MODEL_KNN:
        k = int(cfg.n_neighbors)
        values = (f"list(range(1, min({pl.KNN_MAX_K}, len(X_fit)) + 1))" if k <= pl.KNN_MAX_K else
                  f"sorted(set(range(1, min({pl.KNN_MAX_K}, len(X_fit)) + 1)) | {{min({k}, len(X_fit))}})")
        L += [f"values = {values}   # " + tx("Elbow-Methode wie im Kurs"),
              "for k in values:",
              f"    m = {estimator_code(cfg, task, override={'n_neighbors': 'k'})}",
              "    m.fit(X_fit, y_fit)"]
        xlabel = "K"
    else:
        L += ["order = np.random.default_rng(" + str(int(cfg.seed)) + ").permutation(len(X_fit))",
              "values = sorted({max(2, int(round(f * len(X_fit)))) for f in np.linspace(0.1, 1.0, 10)})   # "
              + tx("Lernkurve"),
              "for n in values:",
              "    rows = order[:n]",
              f"    m = {estimator_code(cfg, task)}",
              "    m.fit(X_fit[rows], y_fit[rows])",
              "    error_train.append(error(y_fit[rows], m.predict(X_fit[rows])))",
              "    error_val.append(error(y_val, m.predict(X_val)))"]
        xlabel = tx("Trainingszeilen")
    if m not in (pl.MODEL_LINREG, pl.MODEL_GB):
        L += ["    error_train.append(error(y_fit, m.predict(X_fit)))",
              "    error_val.append(error(y_val, m.predict(X_val)))"]
    ylabel = "Error Rate" if clf else "MAE"
    L += ["", "plt.figure(figsize=(10, 6))",
          f"plt.plot({x}, error_val, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=8, "
          f"label={tx('Validierung')!r})",
          f"plt.plot({x}, error_train, marker='o', label={tx('Training')!r})"]
    if m in (pl.MODEL_SVM, pl.MODEL_LOGREG):
        L.append("plt.xscale('log')")
    L += [f"plt.title({ylabel + ' vs. ' + xlabel!r})", f"plt.xlabel({xlabel!r})", f"plt.ylabel({ylabel!r})", "plt.legend()"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 8) Evaluation, 9) Speichern, 10) Vorhersage
# ---------------------------------------------------------------------------
def details_code(cfg: pl.ModelConfig, task: str) -> str:
    """Modell-spezifisch: Koeffizienten (lineare Modelle) bzw. feature_importances_ (Bäume)."""
    if cfg.model == pl.MODEL_LINREG:
        return ("coefficients = pd.DataFrame(model.coef_, feature_columns, columns=['Coefficient'])   # "
                + tx("je 1 Einheit des skalierten Features – die App zeigt zusätzlich Originaleinheiten") + "\n"
                "print(coefficients.sort_values('Coefficient'))")
    if cfg.model == pl.MODEL_LOGREG:
        coef = "model.coef_[0]" if task == pl.T_BIN else "model.coef_.T"
        return (f"coefficients = pd.DataFrame({coef}, feature_columns)   # " + tx("Koeffizienten (log-odds, skalierte Features)")
                + "\nprint(coefficients)")
    if cfg.model == pl.MODEL_TREES:
        return ("importances = pd.Series(model.feature_importances_, feature_columns).sort_values(ascending=False)\n"
                "print(importances.head(25))")
    if cfg.model == pl.MODEL_GB:
        return ("from sklearn.inspection import permutation_importance   # "
                + tx("HistGradientBoosting hat kein feature_importances_") + "\n"
                f"imp = permutation_importance(model, X_test, y_test, n_repeats=3, random_state={int(cfg.seed)})\n"
                "print(pd.Series(imp.importances_mean, feature_columns).sort_values(ascending=False).head(25))")
    if cfg.model == pl.MODEL_SVM:
        svc = "model.calibrated_classifiers_[0].estimator" if task != pl.T_REG else "model"
        return f"print('Support vectors:', {svc}.n_support_)"
    return ""


def eval_code(task: str, threshold: float = 0.5, label_scaled: bool = False, cfg: pl.ModelConfig | None = None,
              baseline_column: str | None = None) -> str:
    """Code für Schritt 8. baseline_column (binär): Vergleich mit einer Wahrscheinlichkeit aus den Daten (z. B. Elo)
    auf denselben Testzeilen."""
    if task == pl.T_BIN:
        code = ("proba = model.predict_proba(X_test)[:, 1]   # " + tx("Wahrscheinlichkeit für Klasse 1") + "\n"
                f"predictions = (proba > {_float(threshold)}).astype(int)\n"
                "print(classification_report(y_test, predictions))\n"
                "print(confusion_matrix(y_test, predictions))\n"
                "from sklearn.metrics import log_loss, brier_score_loss\n"
                "print('Log loss:', log_loss(y_test, proba), '· Brier:', brier_score_loss(y_test, proba))   # "
                + tx("Güte der Wahrscheinlichkeiten – kleiner = besser"))
        if baseline_column:
            code += ("\n\n"
                     f"p_base = pd.to_numeric(df.loc[test.index, {baseline_column!r}], errors='coerce').values   # "
                     + _cm(tx("Vergleich: Wahrscheinlichkeit aus den Daten (z. B. Elo)")) + "\n"
                     "ok = ~np.isnan(p_base)   # " + tx("nur Zeilen mit Wert") + "\n"
                     "print('Log loss – model:', log_loss(y_test[ok], proba[ok]), '· baseline:', "
                     "log_loss(y_test[ok], p_base[ok]))")
    elif task == pl.T_MULTI:
        code = ("predictions = model.predict(X_test)\n"
                "print(classification_report(y_test, predictions))\n"
                "print(confusion_matrix(y_test, predictions))")
    else:
        head = "from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score\n"
        if label_scaled:
            head += ("predictions = model.predict(X_test) * y_std + y_mean   # " + tx("zurück in Originaleinheiten")
                     + "\n"
                     "y_true = y_test * y_std + y_mean\n")
        else:
            head += "predictions = model.predict(X_test)\ny_true = y_test\n"
        code = (head
                + "print('MAE :', mean_absolute_error(y_true, predictions))\n"
                "print('RMSE:', np.sqrt(mean_squared_error(y_true, predictions)))\n"
                "print('Explained variance:', explained_variance_score(y_true, predictions))\n"
                "plt.scatter(y_true, predictions)\n"
                "plt.plot(y_true, y_true, 'r')   # " + tx("perfekte Vorhersage"))
    details = details_code(cfg, task) if cfg is not None else ""
    return code + ("\n\n" + details if details else "")


def save_code(folder: str, fmt: str = ".skops") -> str:
    path = f"{folder}/model{fmt}"
    if fmt == ".joblib":
        lines = ["import os", "import joblib", f"os.makedirs({folder!r}, exist_ok=True)",
                 f"joblib.dump(model, {path!r})   # " + tx("Standard von scikit-learn (pickle)"),
                 f"later_model = joblib.load({path!r})   # "
                 + tx("nur vertrauenswürdige Dateien laden – pickle kann beim Laden Code ausführen")]
    else:
        lines = ["import os", "import skops.io as sio", f"os.makedirs({folder!r}, exist_ok=True)",
                 f"sio.dump(model, {path!r})   # " + tx("sicheres Format: beim Laden wird kein Code ausgeführt"),
                 f"unknown = sio.get_untrusted_types(file={path!r})",
                 "trusted = [t for t in unknown if t.startswith('sklearn.')]   # "
                 + tx("nur scikit-learn-Typen vertrauen – alles andere lehnt sio.load ab"),
                 f"later_model = sio.load({path!r}, trusted=trusted)"]
    return ("\n".join(lines) + "\n\n"
            "# " + tx("Wichtig: die Vorverarbeitung gehört zum Modell! Die App speichert sie in") + "\n"
            "# " + _cm(tx("'{folder}/preprocessing.json' (Füllwerte, Dummy-Spalten, Scaler, Label-Kodierung)",
                          folder=folder)))


def predict_code(folder: str) -> str:
    return ("import pipeline as pl\n\n"
            f"model, recipe = pl.load_bundle({folder!r})\n"
            f"df_new, _ = pl.read_csv({tx('neue_daten.csv')!r})\n"
            "result, report, evaluation = pl.predict_new(model, recipe, df_new)\n"
            f"result.to_csv({tx('vorhersagen.csv')!r}, index=False)")


def full_script(source: dict, actions: list[dict], numeric_cols: set, recipe: dict, cfg: pl.ModelConfig,
                threshold: float = 0.5, folder: str = "models/mein_modell", fmt: str = ".skops",
                new_columns: list[dict] | None = None) -> str:
    """Gesamte Pipeline als lauffähiges Python-Skript (für Jupyter oder die Kommandozeile)."""
    task = recipe["task"]
    scaled = bool(recipe.get("label_scaler"))
    calib = pl.calibration_cv(recipe=recipe) if cfg.model == pl.MODEL_SVM and task != pl.T_REG else 5
    gcv = pl.grid_cv(cfg, task, recipe=recipe) if cfg.model == pl.MODEL_SVM and cfg.grid_search else None
    ts = pl.is_time_split(recipe)
    parts = [
        '"""' + tx("Automatisch erzeugt mit dem Streamlit-ML-Template – entspricht den Einstellungen in der App.")
        + '"""',
        "# " + tx("1) Bibliotheken"), IMPORTS, "",
        "# " + tx("Daten laden (+ Stichprobe)"),
        load_code(source.get("path") or source.get("name", "daten.csv"), source.get("sep", ","),
                  source.get("decimal", "."), source.get("encoding", "utf-8"), source.get("sample_frac"),
                  strip_columns=bool(source.get("strip_columns"))), "",
        "# " + tx("2) Datenaufbereitung 1"), step2_code(actions, numeric_cols, new_columns), "",
        "# " + tx("4) Datenaufbereitung 2"), step4_code(recipe), "",
        "# " + tx("6) Modell & Training") + f" – {pl.model_name(cfg.model)}", model_code(cfg, task, calib, gcv, ts), "",
        "# " + tx("7) Training vs. Validierung"), curve_code(cfg, task, scaled, ts), "",
        "# " + tx("8) Evaluation"), eval_code(task, threshold, scaled, cfg, recipe.get("baseline_column")), "",
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
