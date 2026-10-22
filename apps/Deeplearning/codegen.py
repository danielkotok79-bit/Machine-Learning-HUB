"""
codegen.py – erzeugt den passenden Python-Code (pandas, scikit-learn, Keras) zu den Einstellungen der App.

Zum Nachvollziehen und um eine Pipeline 1:1 in ein Jupyter Notebook oder Skript zu übernehmen. Der erzeugte
Code rechnet dasselbe wie die App (pipeline.py) – das prüft tests/test_codegen_parity.py.
"""
from __future__ import annotations

import re

import pipeline as pl
from i18n import N_, tx  # Kommentare im erzeugten Code in der gewählten Sprache

IMPORTS = """import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import load_model
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
    if r.get("dropped_columns"):
        L.append(f"df = df.drop({r['dropped_columns']!r}, axis=1)   # " + tx("Spalten mit zu vielen Lücken"))
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

    sp = r["split"]
    strat = (f", stratify=df[{label!r}]"
             if sp.get("stratify_used", sp.get("stratify")) and task != pl.T_REG else "")
    L += ["", f"features = {r['features']!r}", "",
          "# " + tx("Zuerst aufteilen – Füllwerte, Kategorien und Scaler werden danach NUR aus den Trainingsdaten "
                   "gelernt"),
          f"train, test = train_test_split(df, test_size={sp['test_size']}, random_state={sp['random_state']}{strat})"]

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
              "    return X.astype(float)   # " + tx("pandas ≥ 2: sonst True/False → Keras-Fehler")]
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
def model_code(cfg: pl.ModelConfig, task: str, n_features: int, n_out: int) -> str:
    L = []
    if cfg.optimizer != "adam" or cfg.learning_rate != 0.001:
        opt_cls = {"adam": "Adam", "rmsprop": "RMSprop", "sgd": "SGD"}[cfg.optimizer]
        L.append(f"from tensorflow.keras.optimizers import {opt_cls}")
        opt = f"{opt_cls}(learning_rate={cfg.learning_rate:g})"
    else:
        opt = "'adam'"
    callbacks = []
    if cfg.early_stopping:
        rb = ", restore_best_weights=True" if cfg.restore_best_weights else ""
        L.append(f"early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience={cfg.patience}{rb})")
        callbacks.append("early_stop")
    if cfg.tensorboard:
        L.append("tensorboard = TensorBoard(log_dir='logs/fit', histogram_freq=1)")
        callbacks.append("tensorboard")
    L += ["", "model = Sequential()",
          f"model.add(Input(shape=({n_features},)))   # " + tx("Keras 3: Eingabeschicht explizit"), ""]
    constraint = f", kernel_constraint=max_norm({cfg.max_norm:g})" if cfg.max_norm else ""
    for layer in cfg.layers:
        L.append(f"model.add(Dense(units={layer['units']}, activation={layer['activation']!r}{constraint}))")
        if layer.get("dropout", 0) > 0:
            L.append(f"model.add(Dropout({layer['dropout']:g}))")
    if cfg.output_activation == "linear":
        L.append(f"model.add(Dense(units={n_out}))   # " + tx("keine Aktivierung = linear"))
    else:
        L.append(f"model.add(Dense(units={n_out}, activation={cfg.output_activation!r}))")
    L.append("")
    L.append(f"model.compile(loss={cfg.loss!r}, optimizer={opt})")
    if cfg.loss in pl.LOSS_SHORT:
        L[-1] += "   # " + tx("= '{short}' (voller Name: .h5 bleibt in Keras 3 ladbar)", short=pl.LOSS_SHORT[cfg.loss])
    L.append("")
    if cfg.class_weight and task != pl.T_REG:
        L += ["from sklearn.utils.class_weight import compute_class_weight",
              "cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)",
              "class_weight = dict(enumerate(cw))", ""]
    val = ("validation_data=(X_test, y_test)" if cfg.validation == "test"
           else f"validation_split={cfg.validation_split:g}")
    fit = ["model.fit(x=X_train,", "          y=y_train,", f"          epochs={cfg.epochs},",
           f"          batch_size={cfg.batch_size},", f"          {val}, verbose=1,"]
    if cfg.class_weight and task != pl.T_REG:
        fit.append("          class_weight=class_weight,")
    if callbacks:
        fit.append(f"          callbacks=[{', '.join(callbacks)}]")
    fit.append("          )")
    return "\n".join(L + fit)


def loss_code(task: str) -> str:
    metric = "mean_absolute_error" if task == pl.T_REG else "accuracy"
    return ("model_loss = pd.DataFrame(model.history.history)\n"
            "model_loss[['loss', 'val_loss']].plot()\n"
            f"# model_loss[['{metric}', 'val_{metric}']].plot()")


# ---------------------------------------------------------------------------
# 8) Evaluation, 9) Speichern, 10) Vorhersage
# ---------------------------------------------------------------------------
def eval_code(task: str, threshold: float = 0.5, label_scaled: bool = False) -> str:
    if task == pl.T_BIN:
        return (f"predictions = (model.predict(X_test) > {threshold:g}).astype('int32')   # "
                + tx("model.predict_classes() gibt es seit TF 2.6 nicht mehr") + "\n"
                "print(classification_report(y_test, predictions))\n"
                "print(confusion_matrix(y_test, predictions))")
    if task == pl.T_MULTI:
        return ("predictions = np.argmax(model.predict(X_test), axis=1)\n"
                "print(classification_report(y_test, predictions))\n"
                "print(confusion_matrix(y_test, predictions))")
    head = "from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score\n"
    if label_scaled:
        head += ("predictions = model.predict(X_test).ravel() * y_std + y_mean   # " + tx("zurück in Originaleinheiten")
                 + "\n"
                 "y_true = y_test * y_std + y_mean\n")
    else:
        head += "predictions = model.predict(X_test).ravel()\ny_true = y_test\n"
    return (head
            + "print('MAE :', mean_absolute_error(y_true, predictions))\n"
            "print('RMSE:', np.sqrt(mean_squared_error(y_true, predictions)))\n"
            "print('Explained variance:', explained_variance_score(y_true, predictions))\n"
            "plt.scatter(y_true, predictions)\n"
            "plt.plot(y_true, y_true, 'r')   # " + tx("perfekte Vorhersage"))


def save_code(folder: str, fmt: str = ".keras") -> str:
    path = f"{folder}/model{fmt}"
    note = "   # " + (tx("HDF5 (älteres Format, in Keras 3 'legacy')") if fmt == ".h5"
                      else tx("empfohlenes Keras-3-Format"))
    return ("import os\n"
            f"os.makedirs({folder!r}, exist_ok=True)\n"
            f"model.save({path!r}){note}\n"
            f"later_model = load_model({path!r}, compile=False)\n\n"
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
                threshold: float = 0.5, folder: str = "models/mein_modell", fmt: str = ".keras",
                new_columns: list[dict] | None = None) -> str:
    """Gesamte Pipeline als lauffähiges Python-Skript (für Jupyter oder die Kommandozeile)."""
    task = recipe["task"]
    parts = [
        '"""' + tx("Automatisch erzeugt mit dem Streamlit-DL-Template – entspricht den Einstellungen in der App.")
        + '"""',
        "# " + tx("1) Bibliotheken"), IMPORTS, "",
        "# " + tx("Daten laden (+ Stichprobe)"),
        load_code(source.get("path") or source.get("name", "daten.csv"), source.get("sep", ","),
                  source.get("decimal", "."), source.get("encoding", "utf-8"), source.get("sample_frac"),
                  strip_columns=bool(source.get("strip_columns"))), "",
        "# " + tx("2) Datenaufbereitung 1"), step2_code(actions, numeric_cols, new_columns), "",
        "# " + tx("4) Datenaufbereitung 2"), step4_code(recipe), "",
        "# " + tx("6) Modell & Training"), model_code(cfg, task, len(recipe["feature_columns"]), pl.n_outputs(recipe)),
        "",
        "# " + tx("7) Loss vs. val_loss"), loss_code(task), "",
        "# " + tx("8) Evaluation"), eval_code(task, threshold, bool(recipe.get("label_scaler"))), "",
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
