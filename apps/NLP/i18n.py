"""
i18n.py – Sprachen der Oberfläche: Deutsch (Original), Englisch, Polnisch
=========================================================================

Alle Texte stehen im Code auf Deutsch. tx("…") liefert den Text in der aktiven Sprache
(Übersetzungen in locales/), N_("…") markiert einen Text nur – übersetzt wird er erst bei der
Anzeige, z. B. Namen von Aktionen, die auch als interne Kennung dienen.

    from i18n import tx
    tx("Daten laden")                      # 'Load data' bei Englisch
    tx("{n} Zeilen", n=5)                  # Platzhalter wie bei str.format

Die Sprache gilt je Thread (in Streamlit: je Sitzung und Durchlauf). Startsprache:
?lang=en in der Adresse, `streamlit run app.py -- --lang pl`, Umgebungsvariable NLP_TEMPLATE_LANG
oder Deutsch. Die Rechenlogik ist in allen Sprachen dieselbe.
"""
from __future__ import annotations

import os
import sys
import threading

LANGUAGES = {"de": "Deutsch", "en": "English", "pl": "Polski"}
_COLUMN = {"en": 0, "pl": 1}  # Spalte im Katalog: {deutsch: (englisch, polnisch)}
_local = threading.local()
_catalog: dict | None = None


def _valid(lang) -> str | None:
    lang = str(lang or "").strip().lower()
    return lang if lang in LANGUAGES else None


def start_lang(query=None) -> str:
    """Startsprache: ?lang=… in der Adresse, --lang … beim Start, NLP_TEMPLATE_LANG, sonst Deutsch."""
    candidates = [query]
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg.startswith("--lang="):
            candidates.append(arg.split("=", 1)[1])
        elif arg == "--lang" and i + 1 < len(args):
            candidates.append(args[i + 1])
    candidates.append(os.environ.get("NLP_TEMPLATE_LANG"))
    return next((v for v in map(_valid, candidates) if v), "de")


def set_lang(lang) -> str:
    _local.lang = _valid(lang) or "de"
    return _local.lang


def get_lang() -> str:
    return getattr(_local, "lang", None) or _valid(os.environ.get("NLP_TEMPLATE_LANG")) or "de"


def catalog() -> dict:
    global _catalog
    if _catalog is None:
        from locales import CATALOG
        _catalog = CATALOG
    return _catalog


def tx(text, /, **kw):
    """Text in der aktiven Sprache; mit kw wie str.format. Unbekannte Texte bleiben deutsch."""
    lang = get_lang()
    if lang != "de" and isinstance(text, str):
        entry = catalog().get(text)
        if entry is not None and entry[_COLUMN[lang]]:
            text = entry[_COLUMN[lang]]
    return text.format(**kw) if kw else text


def N_(text):
    """Text nur markieren (für den Katalog) – übersetzt wird bei der Anzeige mit tx()."""
    return text
