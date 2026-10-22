"""
Übersetzungen der Oberfläche: {deutscher Originaltext: (Englisch, Polnisch)}

Je Modul eine Datei (app, pipeline, plots, codegen). Platzhalter wie {n} müssen in allen
Sprachen gleich bleiben – das prüft tests/test_i18n.py.
"""
from locales import app, codegen, pipeline, plots

CATALOG: dict[str, tuple[str, str]] = {}
for _part in (pipeline, plots, codegen, app):
    CATALOG.update(_part.T)
