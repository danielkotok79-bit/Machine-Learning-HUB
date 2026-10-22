"""
codegen.py – erzeugt den passenden Python-Code (pandas, scikit-learn) zu den Einstellungen der App.

Zum Nachvollziehen und um eine Pipeline 1:1 in ein Jupyter Notebook oder Skript zu übernehmen. Der erzeugte
Code rechnet dasselbe wie die App (pipeline.py) – das prüft tests/test_codegen_parity.py.
"""
from __future__ import annotations

import pipeline as pl
from i18n import N_, tx  # Kommentare im erzeugten Code in der gewählten Sprache

IMPORTS = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score"""

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
EXAMPLE_TEXTS = ("The food was great!", "Never again, the service was terrible.")  # Beispiele im Code (Kurs: Englisch)


def _to_number_code() -> str:
    """TO_NUMBER mit Docstring in der gewählten Sprache."""
    return TO_NUMBER.replace(_TO_NUMBER_DOC, tx(_TO_NUMBER_DOC))


def imports_code() -> str:
    return IMPORTS


def _cm(text) -> str:
    """Kommentartext in EINER Zeile. Spaltennamen, Werte und Definitionen aus den Daten können Zeilenumbrüche
    enthalten – im Kommentar würde der Rest sonst zu einer eigenen Code-Zeile (die beim Ausführen mitläuft)."""
    return " ".join(str(text).splitlines())


def _float(v) -> str:
    """Zahl für den erzeugten Code – exakt der Wert der App (repr ohne Rundung), ganze Zahlen ohne '.0'.
    f"{x:g}" rundete auf 6 Stellen (C = 1.234567 → 1.23457) – der Code rechnete dann ein anderes Modell."""
    f = float(v)
    return str(int(f)) if f.is_integer() and abs(f) < 1e15 else repr(f)


# ---------------------------------------------------------------------------
# 1) Laden, Stichprobe
# ---------------------------------------------------------------------------
def load_code(name: str, sep: str = ",", decimal: str = ".", encoding: str = "utf-8",
              sample_frac: float | None = None, seed: int = pl.RANDOM_STATE, strip_columns: bool = False,
              header: bool = True, names: list | None = None) -> str:
    """pd.read_csv mit den Optionen, mit denen die App gelesen hat. header=False: Datei ohne Kopfzeile (wie
    SMSSpamCollection im Kurs) – die Spaltennamen kommen dann aus names."""
    args = [repr(name)]
    if sep != ",":
        args.append(f"sep={sep!r}")
    if decimal != ".":
        args.append(f"decimal={decimal!r}")
    if encoding.lower().replace("-", "") != "utf8":
        args.append(f"encoding={encoding!r}")
    if not header:
        args += ["header=None", f"names={list(names or [])!r}"]
    args.append("low_memory=False")  # ganze Spalte auf einmal → einheitliche Typen (wie die App)
    note = "   # " + tx("Datei ohne Kopfzeile – Spaltennamen selbst vergeben (wie im Kurs)") if not header else ""
    lines = [f"df = pd.read_csv({', '.join(args)}){note}"]
    if strip_columns and header:
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
    if act == pl.A_LEN:
        return [f"{out} = {_as_text(col, kind)}.str.len().astype(float)   # "
                + tx("Länge in Zeichen (wie im Kurs: .apply(len))")]
    if act == pl.A_WORDS:
        return [f"{out} = {_as_text(col, kind)}.str.split().str.len().astype(float)   # " + tx("Anzahl Wörter")]
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
# 4) Datenaufbereitung 2 – Text, Label, Split, Vektorisierung
# ---------------------------------------------------------------------------
_VEC_DEFAULTS = dict(lowercase=True, stop_words=None, ngram_range=(1, 1), min_df=1, max_df=1.0, max_features=None,
                     binary=False, strip_accents=None, sublinear_tf=False)


def vectorizer_code(vc: pl.VectorizerConfig) -> str:
    """CountVectorizer(…) bzw. TfidfVectorizer(…) – nur Parameter, die vom Standard abweichen (wie
    pipeline.make_vectorizer)."""
    kw = pl.vectorizer_kwargs(vc)
    parts = [f"{k}={_float(v) if k == 'max_df' else repr(v)}" for k, v in kw.items() if v != _VEC_DEFAULTS[k]]
    return f"{'TfidfVectorizer' if vc.kind == pl.VEC_TFIDF else 'CountVectorizer'}({', '.join(parts)})"


def _label_mapping(r: dict) -> dict:
    """{Wert im Label: Zahl} – Text: Klassenname, Zahl/Bool: der Wert selbst (wie pipeline.encode_label)."""
    keys = r["classes"] if r["label_kind"] == "text" else r["class_values"]
    return {k: i for i, k in enumerate(keys)}


def step4_code(r: dict) -> str:
    """Code für Schritt 4 in derselben Reihenfolge wie die App: Zeilen ohne Label (und ggf. andere Klassen bzw. ohne
    Text) entfernen, Label kodieren, ZUERST aufteilen, dann den Vektorisierer nur auf den Trainingstexten fitten."""
    text, label, kind = r["text_col"], r["label"], r["label_kind"]
    vc = pl.vectorizer_config(r.get("vectorizer"))
    L = [f"data = df[[{text!r}, {label!r}]].copy()",
         f"data = data.dropna(subset=[{label!r}])   # " + tx("Zeilen ohne Label")]
    lab = f"data[{label!r}]"
    if kind == "text":
        L += [f"label_text = data[{label!r}].astype(str).str.strip()   # "
              + tx("Klassen als Text, ohne Leerzeichen am Rand"),
              "keep = label_text != ''   # " + tx("ein Label aus nur Leerzeichen zählt als fehlend"),
              "data, label_text = data[keep], label_text[keep]"]
        lab = "label_text"
    if r.get("classes_selected"):
        keys = r["classes"] if kind == "text" else r["class_values"]
        L += [f"keep = {lab}.isin({keys!r})",
              "data = data[keep]   # " + _cm(tx("nur die Klassen {c} (wie im Kurs: nur 1 und 5 Sterne)",
                                                c=", ".join(r["classes"])))]
        if kind == "text":
            L.append("label_text = label_text[keep]")
    L.append(f"texts = data[{text!r}].fillna('').astype(str)   # " + tx("fehlender Text → leerer Text"))
    if r.get("missing_text", pl.TEXT_DROP) == pl.TEXT_DROP:
        L += ["keep = texts.str.strip() != ''   # " + tx("Zeilen ohne Text entfernen"),
              "data, texts = data[keep], texts[keep]"]
        if kind == "text":
            L.append("label_text = label_text[keep]")
    mapping = _label_mapping(r)
    L += [f"y = {lab}.map({mapping!r}).to_numpy()   # " + _cm(tx("Klassen → Zahlen: {m}", m=", ".join(
              f"{c} → {i}" for i, c in enumerate(r["classes"])))),
          f"classes = {r['classes']!r}"]
    sp = r["split"]
    strat = ", stratify=y" if sp.get("stratify_used") else ""
    L += ["", "# " + tx("Zuerst aufteilen – der Vektorisierer lernt sein Vokabular danach NUR aus den Trainingstexten"),
          f"X_train_text, X_test_text, y_train, y_test = train_test_split(texts, y, test_size={_float(sp['test_size'])}, "
          f"random_state={int(sp['random_state'])}{strat})",
          "",
          f"vectorizer = {vectorizer_code(vc)}   # "
          + (tx("TF-IDF: häufige Wörter zählen weniger") if vc.kind == pl.VEC_TFIDF else tx("Bag of Words: Wörter zählen")),
          "X_train = vectorizer.fit_transform(X_train_text)   # " + tx("Vokabular nur aus den Trainingstexten"),
          "X_test = vectorizer.transform(X_test_text)   # " + tx("Testtexte nur umwandeln – unbekannte Wörter fallen weg"),
          "print(X_train.shape, len(vectorizer.vocabulary_))   # " + tx("(Texte, Wörter) und Größe des Vokabulars")]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5) Daten speichern
# ---------------------------------------------------------------------------
EXCEL_NOTE = {"de": N_("für deutsches Excel"), "pl": N_("für polnisches Excel"),
              "en": N_("für Excel in Großbritannien, USA, Australien …")}


def _csv_args(excel: str | None) -> tuple[str, str]:
    """Zusätzliche Argumente für to_csv und der Kommentar dazu (excel = 'de', 'pl', 'en' oder None)."""
    if not excel:
        return "", ""
    sep, dec = pl.EXCEL_CSV[excel]
    kw = (f", sep={sep!r}" if sep != "," else "") + (f", decimal={dec!r}" if dec != "." else "") + ", encoding='utf-8-sig'"
    return kw, "   # " + tx(EXCEL_NOTE[excel])  # Kommentar erst nach der schließenden Klammer


def split_names(r: dict) -> list[str]:
    """Spaltennamen des Exports nach Schritt 4 (wie pipeline.split_frame): Text, Label, Label als Zahl, split."""
    names = [r["text_col"]]
    for n in (r["label"], f"{r['label']}_code", "split"):
        names.append(pl.free_name(names, n))
    return names


def export_code(what: str, r: dict | None = None, excel: str | None = None, columns: dict | None = None) -> str:
    """what = 'step2' (Daten nach Schritt 2), 'split' (Texte + Label + split) oder 'vocab' (Vokabular).
    columns = Spaltennamen des Vokabulars in der gewählten Sprache. excel = 'de', 'pl', 'en' oder None."""
    kw, note = _csv_args(excel)
    if what == "step2":
        return f"df.to_csv({tx('daten_aufbereitet') + '.csv'!r}, index=False{kw}){note}"
    if what == "split":
        t, lab, code, split = split_names(r)
        part = ("    pd.DataFrame({{{t!r}: X_{p}_text, {lab!r}: data.loc[X_{p}_text.index, {label!r}], {code!r}: y_{p}, "
                "{split!r}: '{p}'}}),")
        return "\n".join(["text_data = pd.concat([", part.format(t=t, lab=lab, label=r["label"], code=code, split=split,
                                                                p="train"),
                          part.format(t=t, lab=lab, label=r["label"], code=code, split=split, p="test"),
                          "]).sort_index()   # " + tx("ursprüngliche Reihenfolge der Zeilen"),
                          f"text_data.to_csv({tx('text_daten') + '.csv'!r}, index=False{kw}){note}"])
    names = {k: k for k in ("Wort", "Texte", "Anzahl", "IDF")} | (columns or {})
    vc = pl.vectorizer_config(r.get("vectorizer"))
    lines = ["vocab = pd.DataFrame({",
             f"    {names['Wort']!r}: vectorizer.get_feature_names_out(),",
             f"    {names['Texte']!r}: np.asarray((X_train > 0).sum(axis=0)).ravel(),   # "
             + tx("in wie vielen Trainingstexten")]
    if vc.kind != pl.VEC_TFIDF and not vc.binary:
        lines.append(f"    {names['Anzahl']!r}: np.asarray(X_train.sum(axis=0)).ravel(),   # " + tx("wie oft insgesamt"))
    lines.append("})")
    if vc.kind == pl.VEC_TFIDF:
        lines.append(f"vocab[{names['IDF']!r}] = vectorizer.idf_.round(4)")
    lines += [f"vocab = vocab.sort_values([{names['Texte']!r}, {names['Wort']!r}], ascending=[False, True], "
              "kind='stable')",
              f"vocab.to_csv({tx('vokabular') + '.csv'!r}, index=False{kw}){note}"]
    return "\n".join(lines)


def prediction_names(r: dict, columns: dict | None = None) -> list[str]:
    """Spaltennamen des Exports in Schritt 8 (wie pipeline.predictions_frame)."""
    c = {"tatsächlich": "tatsächlich", pl.RESULT_PRED: pl.RESULT_PRED, "richtig": "richtig", **(columns or {})}
    names = [r["text_col"]]
    probs = [f"P({r['classes'][1]})"] if r["task"] == pl.T_BIN else [f"P({k})" for k in r["classes"]]
    for n in [c["tatsächlich"], c[pl.RESULT_PRED], *probs, c["richtig"]]:
        names.append(pl.free_name(names, n))
    return names


def predictions_export_code(r: dict, threshold: float = 0.5, excel: str | None = None,
                            columns: dict | None = None) -> str:
    """Code für den Download in Schritt 8: Testtexte mit tatsächlicher und vorhergesagter Klasse."""
    kw, note = _csv_args(excel)
    names = prediction_names(r, columns)
    t, act, pred = names[:3]
    lines = [f"result = pd.DataFrame({{{t!r}: X_test_text.to_numpy()}})",
             f"result[{act!r}] = np.array(classes)[y_test]",
             f"result[{pred!r}] = np.array(classes)[predictions]"]
    if r["task"] == pl.T_BIN:
        lines.append(f"result[{names[3]!r}] = proba.round(4)")
    else:
        for k in range(len(r["classes"])):
            lines.append(f"result[{names[3 + k]!r}] = proba[:, {k}].round(4)")
    lines += [f"result[{names[-1]!r}] = predictions == y_test",
              f"result.to_csv({tx('vorhersagen_test') + '.csv'!r}, index=False{kw}){note}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 6) Modell & Training
# ---------------------------------------------------------------------------
def estimator_code(cfg: pl.TextModelConfig, *, plain: bool = False, calib_cv: int = 5, override=None) -> str:
    """scikit-learn-Ausdruck für das Modell – nur Parameter, die vom Standard abweichen (wie pipeline.make_model).
    plain=True: LinearSVC ohne CalibratedClassifierCV · override: {Parameter: Code}, z. B. {'alpha': 'a'} für
    Schleifen."""
    m = cfg.model
    cw = "'balanced'" if cfg.class_weight and m != pl.MODEL_NB else None
    if m == pl.MODEL_NB:
        name = "MultinomialNB"
        args = {"alpha": None if float(cfg.alpha) == 1.0 else _float(cfg.alpha),
                "fit_prior": None if cfg.fit_prior else "False"}
    elif m == pl.MODEL_LOGREG:
        name = "LogisticRegression"
        args = {"C": None if float(cfg.logreg_c) == 1.0 else _float(cfg.logreg_c), "max_iter": str(int(cfg.max_iter)),
                "class_weight": cw, "random_state": str(int(cfg.seed))}
    else:
        name = "LinearSVC"
        args = {"C": None if float(cfg.svc_c) == 1.0 else _float(cfg.svc_c), "class_weight": cw,
                "random_state": str(int(cfg.seed))}
    args.update(override or {})
    code = f"{name}({', '.join(f'{k}={v}' for k, v in args.items() if v is not None)})"
    if m == pl.MODEL_SVC and not plain:
        code = f"CalibratedClassifierCV({code}, ensemble=False{'' if int(calib_cv) == 5 else f', cv={int(calib_cv)}'})"
    return code


def model_code(cfg: pl.TextModelConfig, calib_cv: int = 5) -> str:
    """Code für Schritt 6: Modell anlegen und trainieren."""
    L = []
    if cfg.model == pl.MODEL_SVC:
        L.append("# " + tx("LinearSVC liefert keine Wahrscheinlichkeiten – CalibratedClassifierCV ergänzt sie (für "
                           "Schwelle und ROC-AUC)"))
    L += [f"model = {estimator_code(cfg, calib_cv=calib_cv)}", "model.fit(X_train, y_train)"]
    return "\n".join(L)


def compare_code(cfg: pl.TextModelConfig, task: str, calib_cv: int = 5, models=None) -> str:
    """Code für den Modellvergleich in Schritt 6: alle drei Modelle mit den aktuellen Einstellungen (models = nur
    diese, z. B. ohne LinearSVC, wenn eine Klasse nur einen Trainingstext hat – wie in der App)."""
    from dataclasses import replace

    models = [m for m in pl.MODELS if models is None or m in models]
    L = []
    left_out = [pl.model_name(m) for m in pl.MODELS if m not in models]
    if left_out:
        L.append("# " + tx("Nicht im Vergleich: {m} – eine Klasse hat nur einen Trainingstext (siehe App).",
                           m=", ".join(left_out)))
    L.append("models = {")
    for m in models:
        L.append(f"    {pl.model_name(m)!r}: {estimator_code(replace(cfg, model=m), calib_cv=calib_cv)},")
    L += ["}", "for name, m in models.items():", "    m.fit(X_train, y_train)", "    proba = m.predict_proba(X_test)"]
    if task == pl.T_BIN:
        L += ["    pred = (proba[:, 1] > 0.5).astype(int)", "    auc = roc_auc_score(y_test, proba[:, 1])"]
    else:
        L += ["    pred = proba.argmax(axis=1)", "    auc = roc_auc_score(y_test, proba, multi_class='ovr')"]
    L.append("    print(f\"{name}: Accuracy {accuracy_score(y_test, pred):.4f} · "
             "F1 (macro) {f1_score(y_test, pred, average='macro'):.4f} · ROC-AUC {auc:.4f}\")")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 7) Training vs. Validierung
# ---------------------------------------------------------------------------
def curve_code(cfg: pl.TextModelConfig, vec: pl.VectorizerConfig | None = None, stratify: bool = True) -> str:
    """Code für Schritt 7: Fehlerrate auf Training und Validierung je alpha bzw. C – wie die Elbow-Methode im Kurs.
    stratify: ob der Validierungsanteil stratifiziert gezogen wurde (wie in der App)."""
    if not cfg.curve:
        return "# " + tx("keine Validierungskurve (in Schritt 6 ausgeschaltet)")
    L = []
    if cfg.validation == "split":
        strat = ", stratify=y_train" if stratify else ""
        L += [f"fit_idx, val_idx = train_test_split(np.arange(len(y_train)), test_size={_float(cfg.validation_split)}, "
              f"random_state={int(cfg.seed)}{strat})   # " + tx("Validierungsanteil aus den Trainingsdaten"),
              f"vec_fit = {vectorizer_code(vec or pl.VectorizerConfig())}   # "
              + tx("eigener Vektorisierer – lernt nur aus dem Trainingsteil"),
              "X_fit = vec_fit.fit_transform(X_train_text.iloc[fit_idx])",
              "X_val = vec_fit.transform(X_train_text.iloc[val_idx])",
              "y_fit, y_val = y_train[fit_idx], y_train[val_idx]"]
    else:
        L.append("X_fit, y_fit, X_val, y_val = X_train, y_train, X_test, y_test   # "
                 + tx("Validierung mit den Testdaten (wie im Kurs)"))
    param, chosen = pl.param_of(cfg)
    values = sorted(set(pl.CURVE_VALUES[cfg.model]) | {chosen})
    var = "a" if param == "alpha" else "c"
    L += ["error = lambda y, pred: np.mean(pred != y)   # " + tx("Fehlerrate (Error Rate)"),
          "error_train, error_val = [], []",
          "values = [" + ", ".join(_float(v) for v in values) + "]",
          f"for {var} in values:",
          f"    m = {estimator_code(cfg, plain=True, override={param: var})}"
          + ("   # " + tx("ohne Kalibrierung – schneller") if cfg.model == pl.MODEL_SVC else ""),
          "    m.fit(X_fit, y_fit)",
          "    error_train.append(error(y_fit, m.predict(X_fit)))",
          "    error_val.append(error(y_val, m.predict(X_val)))",
          "",
          "plt.figure(figsize=(10, 6))",
          "plt.plot(values, error_val, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', "
          f"markersize=8, label={tx('Validierung')!r})",
          f"plt.plot(values, error_train, marker='o', label={tx('Training')!r})",
          "plt.xscale('log')",
          f"plt.title({'Error Rate vs. ' + param!r})", f"plt.xlabel({param!r})", "plt.ylabel('Error Rate')",
          "plt.legend()"]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 8) Evaluation, 9) Speichern, 10) Vorhersage
# ---------------------------------------------------------------------------
def weights_code(cfg: pl.TextModelConfig, task: str) -> str:
    """Gewichte der Wörter je Klasse (wie pipeline.word_weights)."""
    est = "model.calibrated_classifiers_[0].estimator" if cfg.model == pl.MODEL_SVC else "model"
    if cfg.model == pl.MODEL_NB:
        if task == pl.T_BIN:
            return ("weights = model.feature_log_prob_[1] - model.feature_log_prob_[0]   # "
                    + tx("log P(Wort | Klasse 1) − log P(Wort | Klasse 0)"))
        return ("flp = model.feature_log_prob_\n"
                "weights = flp - (flp.sum(axis=0) - flp) / (len(classes) - 1)   # "
                + tx("je Klasse: log P(Wort | Klasse) − Mittel der anderen Klassen"))
    if task == pl.T_BIN:
        return f"weights = {est}.coef_[0]   # " + tx("positiv → spricht für Klasse 1")
    return f"weights = {est}.coef_   # " + tx("eine Zeile je Klasse")


def eval_code(task: str, threshold: float = 0.5, cfg: pl.TextModelConfig | None = None, classes: list | None = None,
              top: int = 15, example: str = "") -> str:
    classes = classes or ["0", "1"]
    if task == pl.T_BIN:
        code = ["proba = model.predict_proba(X_test)[:, 1]   # " + _cm(tx("Wahrscheinlichkeit für '{c}'", c=classes[1])),
                f"predictions = (proba > {_float(threshold)}).astype(int)"]
    else:
        code = ["proba = model.predict_proba(X_test)", "predictions = proba.argmax(axis=1)"]
    code += ["print(classification_report(y_test, predictions, target_names=classes))",
             "print(confusion_matrix(y_test, predictions))"]
    if cfg is not None:
        code += ["", "# " + tx("Welche Wörter sprechen für welche Klasse?"),
                 "words = vectorizer.get_feature_names_out()", weights_code(cfg, task)]
        if task == pl.T_BIN:
            code += ["top = pd.Series(weights, index=words)",
                     f"print(classes[1], top.nlargest({top}).index.tolist())",
                     f"print(classes[0], (-top).nlargest({top}).index.tolist())"]
        else:
            code += ["for k, c in enumerate(classes):",
                     f"    print(c, pd.Series(weights[k], index=words).nlargest({top}).index.tolist())"]
    code += ["", "# " + tx("Einzelner Text: Vektorisierer und Modell als Pipeline (wie im Kurs)"),
             "pipe = Pipeline([('vect', vectorizer), ('clf', model)])",
             f"print(pipe.predict_proba([{(example or EXAMPLE_TEXTS[0])!r}]))"]
    return "\n".join(code)


def save_code(folder: str, fmt: str = ".skops") -> str:
    path = f"{folder}/model{fmt}"
    lines = ["import os", "pipe = Pipeline([('vect', vectorizer), ('clf', model)])   # "
             + tx("Vektorisierer und Modell gehören zusammen"), f"os.makedirs({folder!r}, exist_ok=True)"]
    if fmt == ".joblib":
        lines += ["import joblib", f"joblib.dump(pipe, {path!r})   # " + tx("Standard von scikit-learn (pickle)"),
                  f"later_pipe = joblib.load({path!r})   # "
                  + tx("nur vertrauenswürdige Dateien laden – pickle kann beim Laden Code ausführen")]
    else:
        lines += ["import skops.io as sio",
                  f"sio.dump(pipe, {path!r})   # " + tx("sicheres Format: beim Laden wird kein Code ausgeführt"),
                  f"unknown = sio.get_untrusted_types(file={path!r})",
                  "trusted = [t for t in unknown if t.startswith('sklearn.')]   # "
                  + tx("nur scikit-learn-Typen vertrauen – alles andere lehnt sio.load ab"),
                  f"later_pipe = sio.load({path!r}, trusted=trusted)"]
    lines.append(f"print(later_pipe.predict([{EXAMPLE_TEXTS[0]!r}]))")
    return ("\n".join(lines) + "\n\n"
            "# " + tx("Die App speichert zusätzlich das Rezept in") + "\n"
            "# " + _cm(tx("'{folder}/preprocessing.json' (Schritt 2, Textspalte, Klassen, Schwelle)", folder=folder)))


def predict_code(folder: str, recipe: dict | None = None) -> str:
    lines = ["import pipeline as pl", "",
             f"pipe, recipe = pl.load_bundle({folder!r})",
             f"df_new, _ = pl.read_csv({tx('neue_texte.csv')!r})",
             "result, report, evaluation = pl.predict_texts(pipe, recipe, df_new)",
             f"result.to_csv({tx('vorhersagen.csv')!r}, index=False)"]
    if recipe is not None:
        col = recipe["text_col"]
        lines += ["", "# " + tx("Nur einzelne Texte – ohne pipeline.py, direkt mit der scikit-learn-Pipeline:"),
                  f"texts = {list(EXAMPLE_TEXTS)!r}",
                  "proba = pipe.predict_proba(texts)",
                  "print([recipe['classes'][k] for k in proba.argmax(axis=1)])"
                  + ("" if recipe["task"] != pl.T_BIN else "   # " + tx("Schwelle 0.5")),
                  "# " + _cm(tx("Texte aus einer CSV: zuerst Schritt 2 anwenden, dann pipe.predict_proba(df_new[{c}])",
                                c=repr(col)))]
    return "\n".join(lines)


def full_script(source: dict, actions: list[dict], numeric_cols: set, recipe: dict, cfg: pl.TextModelConfig,
                threshold: float = 0.5, folder: str = "models/mein_modell", fmt: str = ".skops",
                new_columns: list[dict] | None = None, curve_stratify: bool = True) -> str:
    """Gesamte Pipeline als lauffähiges Python-Skript (für Jupyter oder die Kommandozeile)."""
    task = recipe["task"]
    calib = pl.calibration_cv(recipe=recipe) if cfg.model == pl.MODEL_SVC else 5
    parts = [
        '"""' + tx("Automatisch erzeugt mit DATA-Analysis-NLP – entspricht den Einstellungen in der App.") + '"""',
        "# " + tx("1) Bibliotheken"), IMPORTS, "",
        "# " + tx("Daten laden (+ Stichprobe)"),
        load_code(source.get("path") or source.get("name", "daten.csv"), source.get("sep", ","),
                  source.get("decimal", "."), source.get("encoding", "utf-8"), source.get("sample_frac"),
                  strip_columns=bool(source.get("strip_columns")), header=source.get("header", True) is not False,
                  names=source.get("names")), "",
        "# " + tx("2) Datenaufbereitung 1"), step2_code(actions, numeric_cols, new_columns), "",
        "# " + tx("4) Datenaufbereitung 2 – Text, Label, Split, Vektorisierung"), step4_code(recipe), "",
        "# " + tx("6) Modell & Training") + f" – {pl.model_name(cfg.model)}", model_code(cfg, calib), "",
        "# " + tx("7) Training vs. Validierung"), curve_code(cfg, pl.vectorizer_config(recipe.get("vectorizer")),
                                                          curve_stratify), "",
        "# " + tx("8) Evaluation"), eval_code(task, threshold, cfg, recipe["classes"]), "",
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
    if kind == "heatmap":
        return ("plt.figure(figsize=(12, 7))\n"
                f"sns.heatmap(df.corr(numeric_only=True, method={q(p.get('method', 'pearson'))}), annot=True, "
                "cmap='RdBu_r', center=0, vmin=-1, vmax=1)   # " + tx("numeric_only=True: nötig ab pandas 2"))
    if kind == "groupby":  # wie im Kurs: stars = yelp.groupby('stars').mean(); stars.corr()
        by = q(p["by"])
        return (f"means = df.groupby({by}).mean(numeric_only=True)   # " + tx("Mittelwert je Gruppe") + "\n"
                "print(means)\n"
                "sns.heatmap(means.corr(), cmap='coolwarm', annot=True)   # " + tx("wie im Kurs"))
    text = q(p.get("text", "text"))
    if kind == "length":
        col = tx("Textlänge") if p.get("unit", "Zeichen") == "Zeichen" else tx("Anzahl Wörter")
        expr = (f"df[{text}].fillna('').astype(str).str.len()" if p.get("unit", "Zeichen") == "Zeichen"
                else f"df[{text}].fillna('').astype(str).str.split().str.len()")
        lines = [f"d = df.assign(**{{{col!r}: {expr}}})   # " + tx("wie im Kurs: yelp['text length'] = yelp['text'].apply(len)")]
        if p.get("group"):
            g = q(p["group"])
            lines += [f"g = sns.FacetGrid(d, col={g}, col_wrap=4)",
                      f"g.map(plt.hist, {col!r}, bins={p.get('bins', 50)})",
                      "plt.show()",
                      f"sns.boxplot(x={g}, y={col!r}, data=d)"]
        else:
            lines.append(f"d[{col!r}].plot(kind='hist', bins={p.get('bins', 50)})")
        return "\n".join(lines)
    if kind == "terms":
        cv = "CountVectorizer(" + ", ".join(
            ([f"stop_words={p['stop_words']!r}"] if p.get("stop_words") else [])
            + ([f"ngram_range=(1, {int(p['ngram_max'])})"] if int(p.get("ngram_max", 1)) > 1 else [])) + ")"
        top = int(p.get("top", 20))
        body = ["counts = cv.fit_transform(part[{t}].fillna('').astype(str))".format(t=text),
                "top = pd.Series(np.asarray(counts.sum(axis=0)).ravel(), index=cv.get_feature_names_out())"
                f".nlargest({top})"]
        lines = [f"cv = {cv}"]
        if p.get("group"):
            lines += [f"for value, part in df.groupby({q(p['group'])}):"] + ["    " + b for b in body] + \
                     ["    print(value, top.index.tolist())"]
        else:
            lines += ["part = df"] + body + ["top.sort_values().plot(kind='barh')"]
        return "\n".join(lines)
    if kind == "examples":  # wie die App: nur vorhandene Texte, höchstens so viele, wie es gibt
        n, seed = int(p.get("n", 3)), p.get("seed", 101)
        if p.get("group"):
            return (f"for value, part in df.groupby({q(p['group'])}):\n"
                    f"    texts = part[{text}].dropna()\n"
                    f"    print(value, texts.sample(min({n}, len(texts)), random_state={seed}).tolist())")
        return (f"texts = df[{text}].dropna()\n"
                f"texts.sample(min({n}, len(texts)), random_state={seed}).tolist()")
    return ""
