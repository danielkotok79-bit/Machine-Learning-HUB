"""
ML Hub – centralna strona do wyszukiwania i uruchamiania 4 aplikacji Streamlit
oraz przeglądania materiałów o Machine Learningu.

Start:   streamlit run app.py

Języki: English (domyślnie) · Polski · Deutsch – przełącznik na górze paska bocznego, wybór zostaje w adresie
(?lang=…). Uruchamiane aplikacje dostają ten sam język (?lang=… w linku, --lang … przy starcie), więc bez zmiany
startują po angielsku.

Aplikacje startują tym samym Pythonem co launcher (sys.executable) przez mlhub_compat.py (ten sam folder):
na Streamlit < 1.56 uzupełnia on st.segmented_control(required=…), którego używają aplikacje – bez tego starsza
wersja (np. 1.51 z Anacondy) kończy się błędem TypeError … 'required'. Aplikacje działają w tle także po
zamknięciu launchera; PID-y i logi są w folderze .mlhub/ obok tego pliku.
"""
from __future__ import annotations

import importlib
import inspect
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from importlib import util as importlib_util
from pathlib import Path
from urllib.parse import urlsplit

import streamlit as st
from packaging.version import Version

st.set_page_config(page_title="ML Hub", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
APPS_DIR = BASE_DIR / "apps"
PDF_DIR = BASE_DIR / "materials"
RUNNER = BASE_DIR / "mlhub_compat.py"
STATE_DIR = BASE_DIR / ".mlhub"
REGISTRY = STATE_DIR / "processes.json"  # PID-y aplikacji uruchomionych przez launcher (dla „Zatrzymaj”)
LOG_DIR = STATE_DIR / "logs"
START_TIMEOUT = 30  # s – tyle launcher czeka, aż nowo uruchomiona aplikacja odpowie
NATIVE_VERSION = "1.56"  # od tej wersji Streamlit aplikacje działają bez poprawki
MIN_VERSION = "1.48"  # starsze nie mają m.in. st.container(horizontal=…) (aplikacja NLP)
TEMPLATE_ENV = ("DL", "ML", "CL", "NLP")  # aplikacje czytają też <X>_TEMPLATE_LANG

LANGS = {"en": "English", "pl": "Polski", "de": "Deutsch"}
DEFAULT_LANG = "en"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
try:
    import mlhub_compat
except ImportError:
    mlhub_compat = None
COMPAT_ACTIVE = bool(mlhub_compat and mlhub_compat.install())  # True → Streamlit < 1.56, poprawka działa
ST_VERSION = Version(st.__version__)

# ---------------------------------------------------------------------------
# Teksty interfejsu (EN / PL / DE)
# ---------------------------------------------------------------------------
TEXTS = {
    "en": {
        "tagline": "Find and **launch** the analysis apps and browse the materials on Machine Learning.",
        "search": "🔍 Search for a model / app",
        "search_ph": "e.g. random forest, k-means, nlp, neural network, supervised, svm …",
        "no_results": "Nothing found – showing all apps. Try another keyword.",
        "apps": "Available apps",
        "state_running": "🟢 **Running** · port {port}",
        "state_external": "🟢 **Running** · port {port} · started outside the launcher",
        "state_busy": "🟠 **Port {port} is used by another program**",
        "state_stopped": "⚪ Not running · port {port}",
        "start": "Start",
        "open": "Open",
        "stop": "Stop",
        "restart": "Restart",
        "ext_old": "This instance was started without the compatibility fix (e.g. by the old launcher) – in "
                   "the browser it shows *TypeError … 'required'*. Click **Restart**.",
        "starting": "Starting {name} on port {port} …",
        "started": "{name} is running – click **Open**.",
        "already": "{name} is already running.",
        "slow_start": "{name} is still starting ({sec} s) – click **Open** in a moment. Last lines of the log:",
        "start_failed": "{name} stopped right after starting (exit code {code}). Last lines of the log:",
        "start_error": "Could not start {name}: {err}",
        "not_found": "App not found: `{path}`",
        "busy_err": "Port {port} is used by another program – close it or change the port in `APPS`.",
        "runner_missing": "`mlhub_compat.py` is missing next to the launcher (`app.py`) – copy it into the same "
                          "folder.",
        "stopped": "{name} stopped.",
        "stop_unknown": "{name} was not started by this launcher – stop it where it was started (Ctrl+C in its "
                        "terminal).",
        "stop_error": "Could not stop {name}: {err}",
        "busy_hint": "Another program is listening on port {port}, so the app cannot start there.",
        "missing": "Missing Python packages: **{pkgs}** – the app will not work without them. Install:",
        "details": "Manual start & log",
        "manual": "Same as the **Start** button (same Python, language: {lang}):",
        "log": "Last lines of the log (`{path}`):",
        "sb_apps": "Apps",
        "sb_materials": "Materials",
        "sb_mat_items": ["Recommendations for small data", "ML in predictions", "How to read the results"],
        "env": "Environment",
        "env_compat_short": "compatibility mode",
        "env_python": "Python: `{exe}`",
        "env_ports": "Apps run in the background on ports 8502–8505 with this Python – also after the launcher "
                     "is closed. PIDs and logs: `{dir}`.",
        "env_ok": "Streamlit {ver} – the apps run natively.",
        "env_compat": "Streamlit {ver}: the apps use `st.segmented_control(required=…)`, which exists only from "
                      "Streamlit {native}. The launcher adds it when starting them (compatibility mode), so they "
                      "work. To run them without the launcher as well, update Streamlit and restart the launcher:",
        "env_no_compat": "Streamlit {ver} is older than {native} and `mlhub_compat.py` is missing next to the "
                         "launcher – the apps will fail with *TypeError … 'required'*. Copy `mlhub_compat.py` next "
                         "to `app.py` or update Streamlit:",
        "env_too_old": "Streamlit {ver} is too old for the apps (minimum {min}, recommended {native} or newer). "
                       "Update and restart the launcher:",
        "materials": "📚 Materials",
        "tabs": ["Small datasets – recommendations", "ML in financial and political predictions",
                 "How to read ML model results", "PDF files"],
        "pdf_title": "PDF files",
        "pdf_intro": "Download the full documents:",
        "pdf_missing": "File missing: {f}",
        "pdf_other": "Other languages",
        "pdf_labels": {"faq": "FAQ – small datasets", "predictions": "ML in predictions",
                       "results": "How to read the results"},
        "footer": "ML Hub · apps run locally on ports 8502–8505 · materials based on the attached PDFs "
                  "(as of 16.10.2026)",
    },
    "pl": {
        "tagline": "Centralna strona do **wyszukiwania i uruchamiania** aplikacji analitycznych oraz przeglądania "
                   "materiałów o Machine Learningu.",
        "search": "🔍 Wyszukaj model / aplikację",
        "search_ph": "np. random forest, k-means, nlp, sieć neuronowa, nadzorowane, svm …",
        "no_results": "Nic nie znaleziono – pokazuję wszystkie aplikacje. Spróbuj innego słowa kluczowego.",
        "apps": "Dostępne aplikacje",
        "state_running": "🟢 **Działa** · port {port}",
        "state_external": "🟢 **Działa** · port {port} · uruchomiona poza launcherem",
        "state_busy": "🟠 **Port {port} zajmuje inny program**",
        "state_stopped": "⚪ Nie działa · port {port}",
        "start": "Uruchom",
        "open": "Otwórz",
        "stop": "Zatrzymaj",
        "restart": "Uruchom ponownie",
        "ext_old": "Ta instancja została uruchomiona bez poprawki zgodności (np. przez stary launcher) – w "
                   "przeglądarce pokaże *TypeError … 'required'*. Kliknij **Uruchom ponownie**.",
        "starting": "Uruchamiam {name} na porcie {port} …",
        "started": "{name} działa – kliknij **Otwórz**.",
        "already": "{name} już działa.",
        "slow_start": "{name} jeszcze się uruchamia ({sec} s) – za chwilę kliknij **Otwórz**. Ostatnie linie logu:",
        "start_failed": "{name} zakończyła się zaraz po starcie (kod {code}). Ostatnie linie logu:",
        "start_error": "Nie udało się uruchomić: {name}: {err}",
        "not_found": "Nie znaleziono aplikacji: `{path}`",
        "busy_err": "Port {port} zajmuje inny program – zamknij go albo zmień port w `APPS`.",
        "runner_missing": "Brak pliku `mlhub_compat.py` obok launchera (`app.py`) – skopiuj go do tego samego "
                          "folderu.",
        "stopped": "Zatrzymano: {name}.",
        "stop_unknown": "{name} nie została uruchomiona przez ten launcher – zatrzymaj ją tam, gdzie ją "
                        "uruchomiono (Ctrl+C w jej terminalu).",
        "stop_error": "Nie udało się zatrzymać: {name}: {err}",
        "busy_hint": "Na porcie {port} nasłuchuje inny program, więc aplikacja nie może tam wystartować.",
        "missing": "Brakuje pakietów Pythona: **{pkgs}** – bez nich aplikacja nie zadziała. Instalacja:",
        "details": "Uruchomienie ręczne i log",
        "manual": "To samo co przycisk **Uruchom** (ten sam Python, język: {lang}):",
        "log": "Ostatnie linie logu (`{path}`):",
        "sb_apps": "Aplikacje",
        "sb_materials": "Materiały",
        "sb_mat_items": ["Zalecenia dla małych danych", "ML w predykcjach", "Jak czytać wyniki"],
        "env": "Środowisko",
        "env_compat_short": "tryb zgodności",
        "env_python": "Python: `{exe}`",
        "env_ports": "Aplikacje działają w tle na portach 8502–8505 z tym Pythonem – także po zamknięciu "
                     "launchera. PID-y i logi: `{dir}`.",
        "env_ok": "Streamlit {ver} – aplikacje działają natywnie.",
        "env_compat": "Streamlit {ver}: aplikacje używają `st.segmented_control(required=…)`, który istnieje "
                      "dopiero od Streamlit {native}. Launcher dodaje go przy starcie (tryb zgodności), więc "
                      "działają. Żeby działały też bez launchera, zaktualizuj Streamlit i uruchom launcher ponownie:",
        "env_no_compat": "Streamlit {ver} jest starszy niż {native}, a obok launchera brakuje `mlhub_compat.py` – "
                         "aplikacje skończą się błędem *TypeError … 'required'*. Skopiuj `mlhub_compat.py` obok "
                         "`app.py` albo zaktualizuj Streamlit:",
        "env_too_old": "Streamlit {ver} jest za stary dla aplikacji (minimum {min}, zalecane {native} lub nowszy). "
                       "Zaktualizuj i uruchom launcher ponownie:",
        "materials": "📚 Materiały informacyjne",
        "tabs": ["Zalecenia dla małych zbiorów danych", "ML w predykcjach finansowych i politycznych",
                 "Jak czytać wyniki modeli ML", "Pliki PDF do pobrania"],
        "pdf_title": "Pliki PDF",
        "pdf_intro": "Pobierz pełne dokumenty:",
        "pdf_missing": "Brak pliku: {f}",
        "pdf_other": "Inne języki",
        "pdf_labels": {"faq": "FAQ – małe zbiory danych", "predictions": "ML w predykcjach",
                       "results": "Interpretacja wyników"},
        "footer": "ML Hub · aplikacje uruchamiane lokalnie na portach 8502–8505 · materiały na podstawie "
                  "załączonych PDF (stan 16.10.2026)",
    },
    "de": {
        "tagline": "Zentrale Seite zum **Suchen und Starten** der Analyse-Apps und für Lernmaterial zu Machine "
                   "Learning.",
        "search": "🔍 Modell / App suchen",
        "search_ph": "z. B. random forest, k-means, nlp, neuronales Netz, überwacht, svm …",
        "no_results": "Nichts gefunden – alle Apps werden angezeigt. Anderes Stichwort versuchen.",
        "apps": "Verfügbare Apps",
        "state_running": "🟢 **Läuft** · Port {port}",
        "state_external": "🟢 **Läuft** · Port {port} · außerhalb des Launchers gestartet",
        "state_busy": "🟠 **Port {port} ist von einem anderen Programm belegt**",
        "state_stopped": "⚪ Läuft nicht · Port {port}",
        "start": "Starten",
        "open": "Öffnen",
        "stop": "Stoppen",
        "restart": "Neu starten",
        "ext_old": "Diese Instanz wurde ohne den Kompatibilitäts-Fix gestartet (z. B. vom alten Launcher) – "
                   "im Browser erscheint *TypeError … 'required'*. Auf **Neu starten** klicken.",
        "starting": "Starte {name} auf Port {port} …",
        "started": "{name} läuft – auf **Öffnen** klicken.",
        "already": "{name} läuft bereits.",
        "slow_start": "{name} startet noch ({sec} s) – gleich auf **Öffnen** klicken. Letzte Zeilen des Logs:",
        "start_failed": "{name} wurde direkt nach dem Start beendet (Code {code}). Letzte Zeilen des Logs:",
        "start_error": "{name} konnte nicht gestartet werden: {err}",
        "not_found": "App nicht gefunden: `{path}`",
        "busy_err": "Port {port} ist von einem anderen Programm belegt – Programm beenden oder den Port in `APPS` "
                    "ändern.",
        "runner_missing": "`mlhub_compat.py` fehlt neben dem Launcher (`app.py`) – in denselben Ordner kopieren.",
        "stopped": "{name} gestoppt.",
        "stop_unknown": "{name} wurde nicht von diesem Launcher gestartet – dort stoppen, wo sie gestartet wurde "
                        "(Strg+C im Terminal).",
        "stop_error": "{name} konnte nicht gestoppt werden: {err}",
        "busy_hint": "Auf Port {port} lauscht ein anderes Programm, daher kann die App dort nicht starten.",
        "missing": "Fehlende Python-Pakete: **{pkgs}** – ohne sie funktioniert die App nicht. Installieren:",
        "details": "Manueller Start & Log",
        "manual": "Dasselbe wie die Schaltfläche **Starten** (gleiches Python, Sprache: {lang}):",
        "log": "Letzte Zeilen des Logs (`{path}`):",
        "sb_apps": "Apps",
        "sb_materials": "Materialien",
        "sb_mat_items": ["Empfehlungen für kleine Daten", "ML in Prognosen", "Ergebnisse richtig lesen"],
        "env": "Umgebung",
        "env_compat_short": "Kompatibilitätsmodus",
        "env_python": "Python: `{exe}`",
        "env_ports": "Die Apps laufen im Hintergrund auf den Ports 8502–8505 mit diesem Python – auch nach dem "
                     "Schließen des Launchers. PIDs und Logs: `{dir}`.",
        "env_ok": "Streamlit {ver} – die Apps laufen nativ.",
        "env_compat": "Streamlit {ver}: Die Apps verwenden `st.segmented_control(required=…)`, das es erst ab "
                      "Streamlit {native} gibt. Der Launcher ergänzt es beim Start (Kompatibilitätsmodus), daher "
                      "funktionieren sie. Damit sie auch ohne Launcher laufen, Streamlit aktualisieren und den "
                      "Launcher neu starten:",
        "env_no_compat": "Streamlit {ver} ist älter als {native}, und neben dem Launcher fehlt `mlhub_compat.py` – "
                         "die Apps brechen mit *TypeError … 'required'* ab. `mlhub_compat.py` neben `app.py` "
                         "kopieren oder Streamlit aktualisieren:",
        "env_too_old": "Streamlit {ver} ist zu alt für die Apps (mindestens {min}, empfohlen {native} oder neuer). "
                       "Aktualisieren und den Launcher neu starten:",
        "materials": "📚 Lernmaterialien",
        "tabs": ["Empfehlungen für kleine Datensätze", "ML in Finanz- und Politikprognosen",
                 "Ergebnisse von ML-Modellen lesen", "PDF-Dateien"],
        "pdf_title": "PDF-Dateien",
        "pdf_intro": "Vollständige Dokumente herunterladen:",
        "pdf_missing": "Datei fehlt: {f}",
        "pdf_other": "Andere Sprachen",
        "pdf_labels": {"faq": "FAQ – kleine Datensätze", "predictions": "ML in Prognosen",
                       "results": "Ergebnisse interpretieren"},
        "footer": "ML Hub · Apps laufen lokal auf den Ports 8502–8505 · Materialien auf Basis der beigefügten PDFs "
                  "(Stand 16.10.2026)",
    },
}

# ---------------------------------------------------------------------------
# Aplikacje
# ---------------------------------------------------------------------------
SKLEARN_STACK = {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "skops": "skops",
                 "matplotlib": "matplotlib", "seaborn": "seaborn"}  # moduł → nazwa pakietu w pip

APPS = {
    "supervised": {
        "dir": "Supervised",
        "port": 8502,
        "icon": "📊",
        "name": {"en": "Supervised Learning", "pl": "Uczenie nadzorowane", "de": "Überwachtes Lernen"},
        "desc": {
            "en": "Tabular CSV data: preparation, visualisation, training and evaluation of supervised-learning "
                  "models (Gradient Boosting, Decision Tree + Random Forest, SVM, Logistic Regression, KNN, Linear "
                  "Regression). Classification and regression, random or time-based split, comparison with e.g. "
                  "Elo. Languages: EN / PL / DE.",
            "pl": "Dane tabelaryczne CSV: przygotowanie, wizualizacja, trening i ewaluacja modeli uczenia "
                  "nadzorowanego (Gradient Boosting, Decision Tree + Random Forest, SVM, Regresja logistyczna, KNN, "
                  "Regresja liniowa). Klasyfikacja i regresja, podział losowy albo czasowy, porównanie np. z Elo. "
                  "Języki: EN / PL / DE.",
            "de": "Tabellarische CSV-Daten: Aufbereitung, Visualisierung, Training und Evaluation von Modellen des "
                  "überwachten Lernens (Gradient Boosting, Decision Tree + Random Forest, SVM, Logistische "
                  "Regression, KNN, Lineare Regression). Klassifikation und Regression, Split zufällig oder zeitlich, "
                  "Vergleich z. B. mit Elo. Sprachen: EN / PL / DE.",
        },
        "models": "Gradient Boosting · Decision Trees + Random Forests · SVM · Logistic Regression · KNN · "
                  "Linear Regression",
        "keywords": ["nadzorowane", "supervised", "überwacht", "klasyfikacja", "classification", "klassifikation",
                     "regresja", "regression", "random forest", "drzewo", "tree", "baum", "svm", "knn",
                     "logistyczna", "logistic", "logistische", "liniowa", "linear", "lineare", "scikit-learn", "rf",
                     "gradient boosting", "boosting", "histgradientboosting", "elo", "nba", "tenis", "tennis",
                     "sport"],
        "packages": SKLEARN_STACK,
    },
    "clustering": {
        "dir": "Clustering",
        "port": 8503,
        "icon": "🔵",
        "name": {"en": "Clustering & anomaly detection", "pl": "Klastrowanie i wykrywanie anomalii",
                 "de": "Clustering & Anomalieerkennung"},
        "desc": {
            "en": "Tabular data: preparation, visualisation, grouping with K-Means and anomaly detection with "
                  "Isolation Forest, LOF, DBSCAN and an autoencoder neural network (unsupervised learning). Cluster profiles, "
                  "the most unusual rows, comparison with a known grouping, assigning or scoring new data. Includes a "
                  "Bybit script that builds crypto-market features. Languages: EN / PL / DE.",
            "pl": "Dane tabelaryczne: przygotowanie, wizualizacja, podział na grupy za pomocą K-Means i wykrywanie "
                  "anomalii (Isolation Forest, LOF, DBSCAN, autoenkoder – sieć neuronowa) – uczenie nienadzorowane. "
                  "Profile klastrów, najbardziej nietypowe wiersze, porównanie ze znanym podziałem, przypisywanie lub "
                  "ocena nowych danych. Ze skryptem Bybit, który buduje cechy rynku krypto. Języki: EN / PL / DE.",
            "de": "Tabellarische Daten: Aufbereitung, Visualisierung, Gruppierung mit K-Means und Anomalieerkennung "
                  "mit Isolation Forest, LOF, DBSCAN und einem Autoencoder (neuronales Netz) – unüberwachtes Lernen. "
                  "Cluster-Profile, die auffälligsten Zeilen, Vergleich mit einer bekannten Einteilung, Zuordnung bzw. "
                  "Bewertung neuer Daten. Mit einem Bybit-Skript, das Kennzahlen des Kryptomarkts erzeugt. "
                  "Sprachen: EN / PL / DE.",
        },
        "models": "K-Means · Isolation Forest · LOF · DBSCAN · Autoencoder",
        "keywords": ["klastrowanie", "clustering", "k-means", "kmeans", "nienadzorowane", "unsupervised",
                     "unüberwacht", "grupy", "groups", "gruppen", "klastry", "cluster", "anomalie", "anomalia",
                     "anomaly", "anomalies", "anomalien", "isolation forest", "lof", "local outlier factor", "dbscan",
                     "outlier", "outliers", "ausreißer", "odstające", "wykrywanie anomalii", "bybit", "krypto",
                     "crypto", "trading", "autoencoder", "autoenkoder", "neural network", "sieć neuronowa",
                     "neuronales netz", "deep learning"],
        "packages": SKLEARN_STACK,
    },
    "nlp": {
        "dir": "NLP",
        "port": 8504,
        "icon": "📝",
        "name": {"en": "NLP – text classification", "pl": "NLP – klasyfikacja tekstu",
                 "de": "NLP – Textklassifikation"},
        "desc": {
            "en": "CSV data with texts: preparation, vectorisation (CountVectorizer / TF-IDF), training and "
                  "evaluation (Naive Bayes, Logistic Regression, Linear SVM). Analysis of important words. "
                  "Languages: EN / PL / DE.",
            "pl": "Dane CSV z tekstami: przygotowanie, wektoryzacja (CountVectorizer / TF-IDF), trening i ewaluacja "
                  "(Naive Bayes, Regresja logistyczna, Linear SVM). Analiza ważnych słów. Języki: EN / PL / DE.",
            "de": "CSV-Daten mit Texten: Aufbereitung, Vektorisierung (CountVectorizer / TF-IDF), Training und "
                  "Evaluation (Naive Bayes, Logistische Regression, Lineare SVM). Analyse wichtiger Wörter. "
                  "Sprachen: EN / PL / DE.",
        },
        "models": "MultinomialNB · Logistic Regression · LinearSVC",
        "keywords": ["nlp", "tekst", "text", "klasyfikacja tekstu", "text classification", "textklassifikation",
                     "naive bayes", "tf-idf", "bag of words", "sentiment", "recenzje", "reviews", "bewertungen",
                     "yelp", "spam"],
        "packages": {**SKLEARN_STACK, "scipy": "scipy"},
    },
    "deeplearning": {
        "dir": "Deeplearning",
        "port": 8505,
        "icon": "🧠",
        "name": {"en": "Deep Learning (neural networks)", "pl": "Deep Learning (sieci neuronowe)",
                 "de": "Deep Learning (neuronale Netze)"},
        "desc": {
            "en": "Tabular CSV data: preparation, visualisation, neural-network training (Keras / TensorFlow), "
                  "evaluation and prediction. Classification and regression. Languages: EN / PL / DE.",
            "pl": "Dane tabelaryczne CSV: przygotowanie, wizualizacja, trening sieci neuronowej (Keras / TensorFlow), "
                  "ewaluacja i predykcja. Klasyfikacja i regresja. Języki: EN / PL / DE.",
            "de": "Tabellarische CSV-Daten: Aufbereitung, Visualisierung, Training eines neuronalen Netzes (Keras / "
                  "TensorFlow), Evaluation und Vorhersage. Klassifikation und Regression. Sprachen: EN / PL / DE.",
        },
        "models": "Keras Sequential (Dense)",
        "keywords": ["deep learning", "sieć neuronowa", "neural network", "neuronales netz", "neural", "keras",
                     "tensorflow", "mlp", "epoki", "epochs", "epochen", "early stopping", "dl"],
        "packages": {"numpy": "numpy", "pandas": "pandas", "sklearn": "scikit-learn", "matplotlib": "matplotlib",
                     "seaborn": "seaborn", "tensorflow": "tensorflow"},
    },
}

# ---------------------------------------------------------------------------
# Materiały informacyjne (EN / PL / DE)
# ---------------------------------------------------------------------------
NOTEBOOKS_TABLE = """
| # | Notebook                                      | Model                          | Data                  |
|---|-----------------------------------------------|--------------------------------|-----------------------|
| 1 | Linear Regression                             | LinearRegression               | Ecommerce Customers (500) |
| 2 | Logistic Regression                           | LogisticRegression             | advertising.csv (1000) |
| 3 | K-Nearest Neighbors                           | KNeighborsClassifier           | KNN_Project_Data (~1000) |
| 4 | Decision Trees & Random Forests               | DecisionTree, RandomForest     | loan_data.csv (9578)  |
| 5 | Support Vector Machines                       | SVC + GridSearchCV             | Iris (150)            |
| 6 | K-Means Clustering                            | KMeans                         | College_Data (777)    |
| 7 | NLP                                           | CountVectorizer/TF-IDF + MultinomialNB | Yelp (10 000)    |
| 8 | Keras                                         | Keras Sequential (Dense)       | LendingClub (396 030) |
"""

MATERIALS = {
    "small": {
        "title": {"en": "Recommendations for small datasets and neural networks",
                  "pl": "Zalecenia dotyczące małych zbiorów danych i sieci neuronowych",
                  "de": "Empfehlungen für kleine Datensätze und neuronale Netze"},
        "body": {
            "en": """
With datasets of **~2000 rows and a few to a dozen features**, neural networks are usually unnecessary and often give worse results than simpler classical models.

### What a "small" dataset means in practice

| Size              | Example                                         | When it makes sense                      |
|-------------------|-------------------------------------------------|------------------------------------------|
| **Very small**    | 2–5k rows, a few to a dozen features            | Almost always classical ML               |
| **Small**         | 10–50k rows                                     | Usually still better: trees / boosting   |
| **Medium**        | 100k+ rows, complex data (images, text)         | Neural networks start to make sense      |
| **Large**         | Millions of rows / large models                 | Neural networks (or larger)              |

A 2000 × 7 dataset is **very small data**.

### Are neural networks worth it?
Rather not. At this size:
- Networks overfit easily.
- They are slower to train and harder to interpret.
- Simpler models usually give better or comparable results.

**Better options to start with:**
- **Random Forest** – robust against overfitting, little tuning.
- **XGBoost / LightGBM** – often the best result on tabular data.
- **Logistic / linear regression** – simple, fast, interpretable.
- **SVM** – good for smaller datasets.

**When to use a neural network anyway?**
- When the data are non-linear and complex and classical models perform poorly.
- When you want to learn neural networks (the result will not necessarily be better).
- When you plan a much larger dataset later.

### Training settings with early stopping
With early stopping you don't have to guess the exact number of epochs. You set an upper limit, and early stopping ends training once the model stops improving.

**Recommendation for ~2000 rows and 7 features:**

| Parameter                 | Suggested value | Comment                               |
|---------------------------|-----------------|---------------------------------------|
| Maximum number of epochs  | 100–300         | Safe upper limit                      |
| Patience (early stopping) | 10–20           | Epochs to wait without improvement    |
| Monitor                   | `val_loss`      | Usually the best choice               |

**Example settings:** `epochs=200`, `patience=15`, optionally `min_delta=0.001`.

**Why?**
- On a small dataset the network usually converges quickly (often in 20–80 epochs).
- Early stopping stops earlier anyway, so a high limit does no harm.
- Too low a limit (20–30) can stop before the optimum.
- Too high a limit (1000+) brings nothing.

**Additional tips:**
- Always use a validation set (e.g. 15–20% of the data).
- If the model reaches a plateau very quickly – lower the learning rate.
- With such a small dataset, watch out for overfitting (regularisation, dropout, a smaller network).

**Summary:** Set 200 epochs with patience=15. Early stopping decides when to finish. On such a dataset you can train the model even on a weak computer (no GPU).
""",
            "pl": """
Przy zbiorach rzędu **~2000 wierszy i kilku–kilkunastu cech** sieci neuronowe są zazwyczaj niepotrzebne i często dają gorsze wyniki niż prostsze modele klasyczne.

### Co oznacza „mały” zbiór danych w praktyce

| Wielkość          | Przykład                                      | Kiedy ma sens                          |
|-------------------|-----------------------------------------------|----------------------------------------|
| **Bardzo mała**   | 2–5 tys. wierszy, kilka–kilkanaście cech     | Prawie zawsze klasyczne ML             |
| **Mała**          | 10–50 tys. wierszy                           | Nadal zwykle lepiej drzewa / boosting  |
| **Średnia**       | 100 tys.+ wierszy, złożone dane (obrazy, tekst) | Sieci neuronowe zaczynają mieć sens |
| **Duża**          | Miliony wierszy / duże modele                | Sieci neuronowe (lub większe)          |

Zbiór 2000 × 7 to **bardzo małe dane**.

### Czy warto używać sieci neuronowych?
Raczej nie. Przy takim rozmiarze:
- Sieci łatwo się przeuczają (overfitting).
- Są wolniejsze w trenowaniu i trudniejsze do interpretacji.
- Prostsze modele zwykle dają lepsze lub porównywalne wyniki.

**Lepsze opcje na start:**
- **Random Forest** – odporny na overfitting, mało strojenia.
- **XGBoost / LightGBM** – często najlepszy wynik na danych tabelarycznych.
- **Regresja logistyczna / liniowa** – prosta, szybka, interpretowalna.
- **SVM** – dobry przy mniejszych zbiorach.

**Kiedy jednak użyć sieci neuronowej?**
- Gdy dane są nieliniowe i złożone, a klasyczne modele słabo działają.
- Gdy chcesz się uczyć sieci neuronowych (wynik niekoniecznie będzie lepszy).
- Gdy później planujesz znacznie większy zbiór danych.

### Ustawienia treningu z early stopping
Przy early stopping nie musisz dokładnie zgadywać liczby epok. Ustawiasz górny limit, a early stopping sam zatrzyma trening, gdy model przestanie się poprawiać.

**Rekomendacja dla ok. 2000 wierszy i 7 cech:**

| Parametr                  | Sugerowana wartość | Komentarz                          |
|---------------------------|--------------------|------------------------------------|
| Maksymalna liczba epok    | 100–300            | Bezpieczny górny limit             |
| Patience (early stopping) | 10–20              | Ile epok czekać bez poprawy        |
| Monitor                   | `val_loss`         | Najczęściej najlepszy wybór        |

**Przykładowe ustawienia:** `epochs=200`, `patience=15`, opcjonalnie `min_delta=0.001`.

**Dlaczego tak?**
- Na małym zbiorze sieć zwykle zbiega się szybko (często w 20–80 epokach).
- Early stopping i tak przerwie wcześniej, więc wysoki limit nie zaszkodzi.
- Zbyt niski limit (20–30) może przerwać przed optimum.
- Zbyt wysoki (1000+) nic nie daje.

**Dodatkowe wskazówki:**
- Zawsze używaj zbioru walidacyjnego (np. 15–20% danych).
- Jeśli model bardzo szybko osiąga plateau – zmniejsz learning rate.
- Przy tak małym zbiorze pilnuj, żeby nie przeuczyć się (regularizacja, dropout, mniejsza sieć).

**Podsumowanie:** Ustaw 200 epok z patience=15. Early stopping sam zdecyduje, kiedy skończyć. Na takim zbiorze model wytrenujesz nawet na słabym komputerze (bez GPU).
""",
            "de": """
Bei Datensätzen mit **~2000 Zeilen und einigen bis etwa einem Dutzend Merkmalen** sind neuronale Netze meist unnötig und liefern oft schlechtere Ergebnisse als einfachere klassische Modelle.

### Was ein „kleiner“ Datensatz in der Praxis bedeutet

| Größe             | Beispiel                                          | Wann sinnvoll                            |
|-------------------|---------------------------------------------------|------------------------------------------|
| **Sehr klein**    | 2–5 Tsd. Zeilen, einige bis ein Dutzend Merkmale  | Fast immer klassisches ML                |
| **Klein**         | 10–50 Tsd. Zeilen                                 | Meist weiterhin besser: Bäume / Boosting |
| **Mittel**        | ab 100 Tsd. Zeilen, komplexe Daten (Bilder, Text) | Neuronale Netze werden sinnvoll          |
| **Groß**          | Millionen Zeilen / große Modelle                  | Neuronale Netze (oder größere)           |

Ein Datensatz mit 2000 × 7 ist **sehr klein**.

### Lohnen sich neuronale Netze?
Eher nicht. Bei dieser Größe:
- passen sich Netze leicht zu stark an (Overfitting),
- sind sie langsamer im Training und schwerer zu interpretieren,
- liefern einfachere Modelle meist bessere oder vergleichbare Ergebnisse.

**Bessere Optionen für den Start:**
- **Random Forest** – robust gegen Overfitting, wenig Tuning.
- **XGBoost / LightGBM** – oft das beste Ergebnis bei Tabellendaten.
- **Logistische / lineare Regression** – einfach, schnell, interpretierbar.
- **SVM** – gut bei kleineren Datensätzen.

**Wann doch ein neuronales Netz?**
- Wenn die Daten nichtlinear und komplex sind und klassische Modelle schwach abschneiden.
- Wenn du neuronale Netze lernen willst (das Ergebnis wird nicht zwingend besser).
- Wenn später ein deutlich größerer Datensatz geplant ist.

### Trainingseinstellungen mit Early Stopping
Mit Early Stopping musst du die Zahl der Epochen nicht genau raten. Du setzt eine Obergrenze, und Early Stopping beendet das Training, sobald sich das Modell nicht mehr verbessert.

**Empfehlung für ca. 2000 Zeilen und 7 Merkmale:**

| Parameter                 | Vorschlag  | Kommentar                                   |
|---------------------------|------------|---------------------------------------------|
| Maximale Epochenzahl      | 100–300    | Sichere Obergrenze                          |
| Patience (Early Stopping) | 10–20      | Wie viele Epochen ohne Verbesserung warten  |
| Monitor                   | `val_loss` | Meist die beste Wahl                        |

**Beispiel:** `epochs=200`, `patience=15`, optional `min_delta=0.001`.

**Warum?**
- Auf einem kleinen Datensatz konvergiert das Netz meist schnell (oft in 20–80 Epochen).
- Early Stopping bricht ohnehin früher ab – eine hohe Grenze schadet nicht.
- Eine zu niedrige Grenze (20–30) kann vor dem Optimum abbrechen.
- Eine zu hohe Grenze (1000+) bringt nichts.

**Weitere Tipps:**
- Immer einen Validierungsdatensatz verwenden (z. B. 15–20 % der Daten).
- Erreicht das Modell sehr schnell ein Plateau – Lernrate senken.
- Bei so wenigen Daten auf Overfitting achten (Regularisierung, Dropout, kleineres Netz).

**Fazit:** 200 Epochen mit patience=15 einstellen. Early Stopping entscheidet selbst, wann Schluss ist. Auf so einem Datensatz trainierst du das Modell auch auf einem schwachen Rechner (ohne GPU).
""",
        },
    },
    "predictions": {
        "title": {"en": "Machine Learning in financial, stock-market and political predictions",
                  "pl": "Machine Learning w predykcjach finansowych, giełdowych i politycznych",
                  "de": "Machine Learning in Finanz-, Börsen- und Politikprognosen"},
        "note": {"en": "*Revised version – 16.10.2026*", "pl": "*Wersja poprawiona – 16.10.2026*",
                 "de": "*Überarbeitete Fassung – 16.10.2026*"},
        "body": {
            "en": """
### 1. The type of learning is not the same as the type of model
- **Supervised learning** – the data have a label (target variable). The model learns to predict it. Number → regression; class → classification.
- **Unsupervised learning** – no labels. The algorithm looks for structure: groups (clustering), directions of variation (PCA), anomalies.
- **Self-supervised learning** – the label is created from the data itself (e.g. the next word). This is how language models learn.
- **Deep learning** is not another type of learning but a family of models (neural networks). A network can learn supervised, unsupervised or self-supervised.

**The key correction:** Deep learning does not stand next to supervised/unsupervised as a third option. An LSTM or Transformer predicting a price learns supervised – just like XGBoost. The proper comparison is simple models (linear, trees) versus neural networks, all within supervised learning.

### 2. Course topics – supervised or unsupervised?
| Topic                          | Type of learning           | Notes |
|--------------------------------|----------------------------|-------|
| Linear Regression              | Supervised (regression)    | Numeric label (e.g. Yearly Amount Spent) |
| Logistic Regression            | Supervised (classification) | Class label (e.g. Clicked on Ad) |
| K-Nearest Neighbors            | Supervised                 | Needs labels; not to be confused with K-Means |
| Decision Trees / Random Forests| Supervised                 | Isolation Forest = anomaly detection (no labels) |
| Support Vector Machines        | Supervised                 | One-Class SVM = anomalies |
| K-Means Clustering             | Unsupervised               | No labels; cluster numbers are arbitrary |
| PCA                            | Unsupervised               | Dimensionality reduction / visualisation, not a predictive model |
| Recommender Systems            | Depends on the approach    | Often closer to supervised (matrix factorisation) |
| NLP                            | Supervised in the course   | In general: supervised, unsupervised and self-supervised |
| Keras (LendingClub)            | Supervised                 | The simplest proof that DL is a type of model, not of learning |
| Autoencoder (Clustering app)   | Unsupervised               | A neural network without labels: learns to rebuild the rows; a large reconstruction error = anomaly |

### 3. What to choose for prediction?
There is no single best approach. Order of decisions:
1. If you want to predict something and you have a label → supervised learning.
2. Only then choose the model, starting with a simple one.

On financial data simple models often win; neural networks are not automatically better.

**Stock market and finance:** On tabular data (technical and fundamental features) tree-based models do well, especially gradient boosting. The first point of reference should be a linear model and a simple baseline. Networks (LSTM, Transformer) on a single instrument with limited history often lose to simpler models, because financial data are noisy and non-stationary.

**Political predictions:** There are few elections, so there are few labels. Professional forecasts rely mainly on statistically combining many polls and fundamental data, not on large ML models. NLP (sentiment) can be an additional feature, but as a stand-alone way of predicting elections it has a poor track record.

**General rules:**
- Do you have a label (election result, price direction, up/down)? → supervised learning. Start with a simple model and a comparison against a baseline.
- Do you want to understand the structure of data without labels? → unsupervised learning.
- Test neural networks only when you have a lot of data and simpler models have reached their limit.

### 4. Before you trust a result
- Validation without data leakage (chronological, walk-forward, purged k-fold + embargo).
- A baseline ("always the majority class" or a simple market rule).
- A clear definition of the label, the horizon and what counts as success.
""",
            "pl": """
### 1. Rodzaj uczenia to nie to samo co rodzaj modelu
- **Uczenie nadzorowane (supervised)** – dane mają etykietę (zmienną docelową). Model uczy się ją przewidywać. Liczba → regresja; klasa → klasyfikacja.
- **Uczenie nienadzorowane (unsupervised)** – brak etykiet. Algorytm szuka struktury: grupy (klastrowanie), kierunki zmienności (PCA), anomalie.
- **Uczenie samonadzorowane (self-supervised)** – etykietę tworzy się z samych danych (np. następne słowo). Tak uczą się modele językowe.
- **Deep learning** to nie kolejny rodzaj uczenia, tylko rodzina modeli (sieci neuronowe). Sieć może uczyć się nadzorowanie, nienadzorowanie lub samonadzorowanie.

**Najważniejsza poprawka:** Deep learning nie stoi obok supervised/unsupervised jako trzecia możliwość. LSTM czy Transformer przewidujący cenę uczy się nadzorowanie – tak samo jak XGBoost. Właściwe porównanie to proste modele (liniowe, drzewa) kontra sieci neuronowe, wszystkie w ramach uczenia nadzorowanego.

### 2. Tematy kursu – nadzorowane czy nienadzorowane?
| Temat                          | Rodzaj uczenia          | Uwagi |
|--------------------------------|-------------------------|-------|
| Linear Regression              | Nadzorowane (regresja)  | Etykieta liczbowa (np. Yearly Amount Spent) |
| Logistic Regression            | Nadzorowane (klasyfikacja) | Etykieta klasowa (np. Clicked on Ad) |
| K-Nearest Neighbors            | Nadzorowane             | Potrzebuje etykiet; nie mylić z K-Means |
| Decision Trees / Random Forests| Nadzorowane             | Isolation Forest = wykrywanie anomalii (bez etykiet) |
| Support Vector Machines        | Nadzorowane             | One-Class SVM = anomalie |
| K-Means Clustering             | Nienadzorowane          | Brak etykiet; numery klastrów losowe |
| PCA                            | Nienadzorowane          | Redukcja wymiarów / wizualizacja, nie model predykcyjny |
| Recommender Systems            | Zależnie od podejścia   | Często bliżej nadzorowanego (faktoryzacja macierzy) |
| NLP                            | W kursie nadzorowane    | Ogólnie: nadzorowane, nienadzorowane i samonadzorowane |
| Keras (LendingClub)            | Nadzorowane             | Najprostszy dowód, że DL to rodzaj modelu, nie uczenia |
| Autoenkoder (aplikacja Clustering) | Nienadzorowane      | Sieć neuronowa bez etykiet: uczy się odtwarzać wiersze; duży błąd rekonstrukcji = anomalia |

### 3. Co wybrać do predykcji?
Nie ma jednego najlepszego podejścia. Kolejność decyzji:
1. Jeśli chcesz coś przewidzieć i masz etykietę → uczenie nadzorowane.
2. Dopiero potem wybierasz model, zaczynając od prostego.

Na danych finansowych proste modele często wygrywają; sieci neuronowe nie są automatycznie lepsze.

**Giełda i finanse:** Na danych tabelarycznych (cechy techniczne i fundamentalne) dobrze wypadają modele drzewiaste, zwłaszcza gradient boosting. Pierwszym punktem odniesienia powinien być model liniowy i prosty baseline. Sieci (LSTM, Transformer) przy pojedynczym instrumencie i ograniczonej historii często przegrywają z prostszymi modelami, bo dane finansowe są zaszumione i niestacjonarne.

**Predykcje polityczne:** Wyborów jest niewiele, więc etykiet jest mało. Profesjonalne prognozy opierają się głównie na statystycznym łączeniu wielu sondaży i danych fundamentalnych, a nie na dużych modelach ML. NLP (sentyment) może być dodatkową cechą, ale jako samodzielny sposób przewidywania wyborów ma słabą historię.

**Ogólne zasady:**
- Masz etykietę (wynik wyborów, kierunek ceny, wzrost/spadek)? → uczenie nadzorowane. Zacznij od prostego modelu i porównania z baseline.
- Chcesz zrozumieć strukturę danych bez etykiet? → uczenie nienadzorowane.
- Sieci neuronowe testuj dopiero wtedy, gdy masz dużo danych, a prostsze modele osiągnęły limit.

### 4. Zanim zaufasz wynikowi
- Walidacja bez przecieku danych (chronologicznie, walk-forward, purged k-fold + embargo).
- Baseline („zawsze klasa większościowa” lub prosta reguła rynkowa).
- Jasna definicja etykiety, horyzontu i tego, co liczy się jako sukces.
""",
            "de": """
### 1. Die Art des Lernens ist nicht dasselbe wie die Art des Modells
- **Überwachtes Lernen (supervised)** – die Daten haben ein Label (Zielvariable). Das Modell lernt, es vorherzusagen. Zahl → Regression; Klasse → Klassifikation.
- **Unüberwachtes Lernen (unsupervised)** – keine Labels. Der Algorithmus sucht Struktur: Gruppen (Clustering), Richtungen der Streuung (PCA), Anomalien.
- **Selbstüberwachtes Lernen (self-supervised)** – das Label entsteht aus den Daten selbst (z. B. das nächste Wort). So lernen Sprachmodelle.
- **Deep Learning** ist keine weitere Art des Lernens, sondern eine Modellfamilie (neuronale Netze). Ein Netz kann überwacht, unüberwacht oder selbstüberwacht lernen.

**Die wichtigste Korrektur:** Deep Learning steht nicht als dritte Möglichkeit neben supervised/unsupervised. Ein LSTM oder Transformer, der einen Preis vorhersagt, lernt überwacht – genau wie XGBoost. Der richtige Vergleich ist: einfache Modelle (linear, Bäume) gegen neuronale Netze, alle im überwachten Lernen.

### 2. Kursthemen – überwacht oder unüberwacht?
| Thema                          | Art des Lernens            | Hinweise |
|--------------------------------|----------------------------|----------|
| Linear Regression              | Überwacht (Regression)     | Zahl als Label (z. B. Yearly Amount Spent) |
| Logistic Regression            | Überwacht (Klassifikation) | Klasse als Label (z. B. Clicked on Ad) |
| K-Nearest Neighbors            | Überwacht                  | Braucht Labels; nicht mit K-Means verwechseln |
| Decision Trees / Random Forests| Überwacht                  | Isolation Forest = Anomalieerkennung (ohne Labels) |
| Support Vector Machines        | Überwacht                  | One-Class SVM = Anomalien |
| K-Means Clustering             | Unüberwacht                | Keine Labels; Cluster-Nummern sind willkürlich |
| PCA                            | Unüberwacht                | Dimensionsreduktion / Visualisierung, kein Vorhersagemodell |
| Recommender Systems            | Je nach Ansatz             | Oft näher am überwachten Lernen (Matrixfaktorisierung) |
| NLP                            | Im Kurs überwacht          | Allgemein: überwacht, unüberwacht und selbstüberwacht |
| Keras (LendingClub)            | Überwacht                  | Der einfachste Beleg, dass DL eine Modellart ist, keine Lernart |
| Autoencoder (Clustering-App)   | Unüberwacht                | Ein neuronales Netz ohne Labels: lernt, die Zeilen nachzubauen; großer Rekonstruktionsfehler = Anomalie |

### 3. Was wählen für Vorhersagen?
Es gibt nicht den einen besten Ansatz. Reihenfolge der Entscheidungen:
1. Willst du etwas vorhersagen und hast ein Label → überwachtes Lernen.
2. Erst dann das Modell wählen – mit einem einfachen beginnen.

Bei Finanzdaten gewinnen oft einfache Modelle; neuronale Netze sind nicht automatisch besser.

**Börse und Finanzen:** Bei Tabellendaten (technische und fundamentale Merkmale) schneiden baumbasierte Modelle gut ab, besonders Gradient Boosting. Erster Vergleichspunkt sollten ein lineares Modell und eine einfache Baseline sein. Netze (LSTM, Transformer) verlieren bei einem einzelnen Instrument und begrenzter Historie oft gegen einfachere Modelle, weil Finanzdaten verrauscht und nicht stationär sind.

**Politische Prognosen:** Es gibt wenige Wahlen, also wenige Labels. Professionelle Prognosen beruhen vor allem auf der statistischen Kombination vieler Umfragen und Fundamentaldaten, nicht auf großen ML-Modellen. NLP (Sentiment) kann ein zusätzliches Merkmal sein, hat als alleinige Methode zur Wahlvorhersage aber eine schwache Bilanz.

**Allgemeine Regeln:**
- Hast du ein Label (Wahlergebnis, Kursrichtung, steigt/fällt)? → überwachtes Lernen. Mit einem einfachen Modell und dem Vergleich mit einer Baseline beginnen.
- Willst du die Struktur von Daten ohne Labels verstehen? → unüberwachtes Lernen.
- Neuronale Netze erst testen, wenn viele Daten vorliegen und einfachere Modelle ihre Grenze erreicht haben.

### 4. Bevor du einem Ergebnis vertraust
- Validierung ohne Datenleck (chronologisch, Walk-Forward, Purged k-Fold + Embargo).
- Eine Baseline („immer die häufigste Klasse“ oder eine einfache Marktregel).
- Eine klare Definition von Label, Horizont und dem, was als Erfolg zählt.
""",
        },
    },
    "results": {
        "title": {"en": "How to read ML model results", "pl": "Jak czytać wyniki modeli ML",
                  "de": "Ergebnisse von ML-Modellen lesen"},
        "body": {
            "en": """
The document interprets the results of 8 course notebooks (versions of 16.10.2026). For each notebook it answers two questions:
1. Were the data suitable for training, and how did training go?
2. How should the results on the test data (which the model did not see during training) be interpreted?

Each chapter has the same structure: key facts → "In short" (Data / Training / Test result) → 1) data and training → 2) results on the test data → what to watch out for → conclusion.

### Notebooks covered
""" + NOTEBOOKS_TABLE + """
### Cheat sheet – are the data suitable for training?
- **How much data, and is it complete?** `df.info()` – Non-Null < number of rows = gaps; `object` = text that needs encoding.
- **Are the values plausible?** `df.describe()` min/max – percentages > 100, negative times/amounts, max many times larger than the 75% value.
- **Feature scale** – differences of ×100–×1000 → scale for KNN, K-Means, SVM, networks.
- **Class balance** – one class ≥ 70–80% → accuracy misleads; compute a baseline.
- **Do the features separate the classes?** pairplot, histograms with `hue=label`, `corr()` with the target – overlapping distributions = weak model.
- **Data leakage?** A feature known only "after the fact"; scaler / vectorizer / imputation fitted before the split.

The full document (in English, and as a bilingual PL + DE version) can be downloaded in the next tab. It contains detailed interpretations of each notebook's results, a comparison of the models and the key lessons.
""",
            "pl": """
Dokument interpretuje wyniki 8 notebooków z kursu (wersje z 16.10.2026). Dla każdego notebooka odpowiada na dwa pytania:
1. Czy dane nadawały się do treningu i jak przebiegł trening?
2. Jak interpretować wyniki na danych testowych (których model nie widział podczas uczenia)?

Każdy rozdział ma budowę: metryczka → „W skrócie” (Dane / Trening / Wynik testu) → 1) dane i trening → 2) wyniki na danych testowych → na co uważać → wniosek.

### Notebooki objęte analizą
""" + NOTEBOOKS_TABLE.replace("| Data ", "| Dane ") + """
### Ściąga – czy dane nadają się do treningu?
- **Ile danych i czy kompletne?** `df.info()` – Non-Null < liczba wierszy = braki; `object` = tekst do zakodowania.
- **Czy wartości możliwe?** `df.describe()` min/max – procent > 100, ujemne czasy/kwoty, max wielokrotnie większe od 75%.
- **Skala cech** – różnice ×100–×1000 → skalowanie przy KNN, K-Means, SVM, sieciach.
- **Balans klas** – jedna klasa ≥ 70–80% → accuracy myli; licz baseline.
- **Czy cechy rozdzielają klasy?** pairplot, histogramy z `hue=etykieta`, `corr()` z targetem – pokrywające się rozkłady = słaby model.
- **Przeciek danych?** Cecha znana dopiero „po fakcie”; scaler / vectorizer / imputacja dopasowane przed podziałem.

Pełny dokument (w wersji dwujęzycznej PL + DE oraz po angielsku) jest dostępny do pobrania w zakładce obok. Zawiera szczegółowe interpretacje wyników każdego notebooka, porównanie modeli i najważniejsze lekcje.
""",
            "de": """
Das Dokument interpretiert die Ergebnisse von 8 Kurs-Notebooks (Stand 16.10.2026). Für jedes Notebook beantwortet es zwei Fragen:
1. Waren die Daten für das Training geeignet, und wie verlief das Training?
2. Wie sind die Ergebnisse auf den Testdaten zu interpretieren (die das Modell beim Lernen nicht gesehen hat)?

Jedes Kapitel ist gleich aufgebaut: Steckbrief → „Kurz gesagt“ (Daten / Training / Testergebnis) → 1) Daten und Training → 2) Ergebnisse auf den Testdaten → worauf achten → Fazit.

### Ausgewertete Notebooks
""" + NOTEBOOKS_TABLE.replace("| Data ", "| Daten") + """
### Spickzettel – sind die Daten für das Training geeignet?
- **Wie viele Daten, und sind sie vollständig?** `df.info()` – Non-Null < Zeilenzahl = Lücken; `object` = Text, der kodiert werden muss.
- **Sind die Werte plausibel?** `df.describe()` min/max – Prozente > 100, negative Zeiten/Beträge, max ein Vielfaches des 75-%-Werts.
- **Skala der Merkmale** – Unterschiede ×100–×1000 → skalieren bei KNN, K-Means, SVM, Netzen.
- **Klassenbalance** – eine Klasse ≥ 70–80 % → Accuracy täuscht; Baseline berechnen.
- **Trennen die Merkmale die Klassen?** Pairplot, Histogramme mit `hue=label`, `corr()` mit dem Ziel – überlappende Verteilungen = schwaches Modell.
- **Datenleck?** Ein Merkmal, das erst „hinterher“ bekannt ist; Scaler / Vectorizer / Imputation vor dem Split angepasst.

Das vollständige Dokument (zweisprachig PL + DE sowie auf Englisch) steht im nächsten Tab zum Download. Es enthält ausführliche Interpretationen der Ergebnisse jedes Notebooks, einen Modellvergleich und die wichtigsten Lehren.
""",
        },
    },
}
# Materiały PDF w folderze materials/: dokument → plik w każdym języku (interpretacja PL i DE jest w jednym pliku)
PDFS = {
    "faq": {"en": "FAQ_EN.pdf", "pl": "FAQ_PL.pdf", "de": "FAQ_DE.pdf"},
    "predictions": {"en": "ML_in_predictions_EN.pdf", "pl": "ML_in_predictions_PL.pdf",
                    "de": "ML_in_predictions_DE.pdf"},
    "results": {"en": "Reading_ML_results_EN.pdf", "pl": "Reading_ML_results_PL-DE.pdf",
                "de": "Reading_ML_results_PL-DE.pdf"},
}

# ---------------------------------------------------------------------------
# Język: domyślnie angielski; ?lang=… w adresie zachowuje wybór przy odświeżeniu strony
# ---------------------------------------------------------------------------
ss = st.session_state
if ss.get("hub_lang") not in LANGS:
    requested = str(st.query_params.get("lang") or "").strip().lower()
    ss["hub_lang"] = requested if requested in LANGS else DEFAULT_LANG
LANG = ss["hub_lang"]
if st.query_params.get("lang") != LANG:
    st.query_params["lang"] = LANG
TX = TEXTS[LANG]


def T(key: str, **kw) -> str:
    text = TX.get(key, TEXTS["en"][key])
    return text.format(**kw) if kw else text


def app_name(key: str) -> str:
    return APPS[key]["name"][LANG]


def _supports(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def shell_join(args) -> str:
    args = [str(a) for a in args]
    return subprocess.list2cmdline(args) if os.name == "nt" else shlex.join(args)


# ---------------------------------------------------------------------------
# Procesy aplikacji
# ---------------------------------------------------------------------------
_NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # localhost nigdy przez proxy
_DETACH = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
           else {"start_new_session": True})  # aplikacja działa dalej po zamknięciu launchera


@st.cache_resource
def _children() -> dict:
    """Procesy uruchomione przez ten launcher – między przebiegami, żeby po zakończeniu je posprzątać (wait)."""
    return {}


def port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def streamlit_ok(port: int) -> bool:
    """Czy na porcie odpowiada serwer Streamlit (a nie inny program)?"""
    try:
        with _NO_PROXY.open(f"http://127.0.0.1:{port}/_stcore/health", timeout=1.5) as resp:
            return resp.status == 200 and resp.read(16).strip() == b"ok"
    except (OSError, ValueError):
        return False


def _read_registry() -> dict:
    try:
        data = json.loads(REGISTRY.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _ensure_state_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ignore = STATE_DIR / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")  # pliki robocze launchera – poza repozytorium git


def _write_registry(reg: dict) -> None:
    try:
        _ensure_state_dir()
        REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    except OSError:
        pass


def _is_ours(pid: int, port: int) -> bool:
    """Czy PID to nadal aplikacja uruchomiona przez launcher (a nie inny proces z tym samym numerem)?"""
    if os.name == "nt":
        return True  # bez /proc – wystarcza, że port odpowiada jak Streamlit
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return True  # system bez /proc (macOS)
    try:
        args = cmdline.read_bytes().split(b"\0")
    except OSError:
        return False
    return any(a.endswith(RUNNER.name.encode()) for a in args) and str(port).encode() in args


def registered_pid(key: str) -> int | None:
    entry = _read_registry().get(key) or {}
    pid, port = entry.get("pid"), APPS[key]["port"]
    if isinstance(pid, int) and entry.get("port") == port and _is_ours(pid, port):
        return pid
    return None


def app_state(key: str) -> str:
    """running = działa (uruchomiona tutaj) · external = działa, uruchomiona poza launcherem ·
    busy = port zajmuje inny program · stopped = nie działa."""
    port = APPS[key]["port"]
    if not port_open(port):
        return "stopped"
    if not streamlit_ok(port):
        return "busy"
    return "running" if registered_pid(key) else "external"


def listening_pid(port: int) -> int | None:
    """PID procesu nasłuchującego na porcie – dla aplikacji uruchomionych poza launcherem (np. przez stary)."""
    try:
        import psutil  # w Anacondzie jest; bez niego Linux: /proc

        for conn in psutil.net_connections(kind="tcp"):
            if conn.laddr and conn.laddr.port == port and conn.status == psutil.CONN_LISTEN and conn.pid:
                return conn.pid
    except Exception:  # noqa: BLE001 – brak psutil / brak uprawnień
        pass
    sockets = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            rows = Path(table).read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            cols = row.split()
            if len(cols) > 9 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:  # 0A = LISTEN
                sockets.add(f"socket:[{cols[9]}]")
    if not sockets:
        return None
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            if any(os.readlink(fd) in sockets for fd in (proc_dir / "fd").iterdir()):
                return int(proc_dir.name)
        except OSError:
            continue
    return None


def started_without_fix(pid: int) -> bool:
    """Instancja na tym samym (starszym niż 1.56) Pythonie, ale bez mlhub_compat.py → w przeglądarce błąd 'required'."""
    if ST_VERSION >= Version(NATIVE_VERSION):
        return False
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        same_python = os.path.realpath(f"/proc/{pid}/exe") == os.path.realpath(sys.executable)
    except OSError:
        return False  # nie wiadomo (np. macOS) – bez ostrzeżenia
    return same_python and not any(a.endswith(RUNNER.name.encode()) for a in args)


def app_command(key: str, lang: str) -> list[str]:
    port = APPS[key]["port"]
    return [sys.executable, str(RUNNER), "run", "app.py", "--server.port", str(port), "--server.headless", "true",
            "--", "--lang", lang]


def app_url(key: str) -> str:
    """Link do aplikacji z językiem launchera – host taki jak w przeglądarce (działa też z innego komputera w LAN)."""
    host = "localhost"
    try:
        header = st.context.headers.get("Host")
    except Exception:  # noqa: BLE001 – starsze wersje / testy bez nagłówków
        header = None
    if header:
        name = urlsplit("//" + header).hostname
        if name:
            host = f"[{name}]" if ":" in name else name
    return f"http://{host}:{APPS[key]['port']}/?lang={LANG}"


def log_path(key: str) -> Path:
    return LOG_DIR / f"{key}.log"


def log_tail(path: Path, lines: int = 25) -> str:
    try:
        text = path.read_bytes()[-20_000:].decode("utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def missing_packages(key: str) -> list[str]:
    return [pip for module, pip in APPS[key]["packages"].items() if importlib_util.find_spec(module) is None]


def launch(key: str) -> tuple:
    """Uruchamia aplikację w tle → (poziom komunikatu, tekst, szczegóły)."""
    app, name = APPS[key], app_name(key)
    port, folder = app["port"], APPS_DIR / app["dir"]
    if not (folder / "app.py").exists():
        return "error", T("not_found", path=folder / "app.py"), None
    state = app_state(key)
    if state in ("running", "external"):
        return "info", T("already", name=name), None
    if state == "busy":
        return "error", T("busy_err", port=port), None
    if not RUNNER.exists():
        return "error", T("runner_missing"), None
    env = {**os.environ, "PYTHONUNBUFFERED": "1", **{f"{p}_TEMPLATE_LANG": LANG for p in TEMPLATE_ENV}}
    try:
        _ensure_state_dir()
        with open(log_path(key), "wb") as log:
            proc = subprocess.Popen(app_command(key, LANG), cwd=folder, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, env=env, **_DETACH)
    except OSError as exc:
        return "error", T("start_error", name=name, err=exc), None
    _children()[key] = proc
    reg = _read_registry()
    reg[key] = {"pid": proc.pid, "port": port, "lang": LANG, "started": datetime.now().isoformat(timespec="seconds")}
    _write_registry(reg)
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline and proc.poll() is None:
        if streamlit_ok(port):
            return "success", T("started", name=name), None
        time.sleep(0.4)
    if proc.poll() is None:
        return "warning", T("slow_start", name=name, sec=START_TIMEOUT), log_tail(log_path(key))
    return "error", T("start_failed", name=name, code=proc.returncode), log_tail(log_path(key))


def _signal(pid: int, sig: int) -> None:
    if os.getpgid(pid) == pid:  # lider własnej grupy (start_new_session=True) → razem z procesami potomnymi
        os.killpg(pid, sig)
    else:
        os.kill(pid, sig)


def _wait_port_closed(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not port_open(port):
            return True
        time.sleep(0.25)
    return not port_open(port)


def stop(key: str, pid: int | None = None) -> tuple:
    """Zatrzymuje aplikację: uruchomioną przez launcher albo – gdy podano pid – działającą poza nim."""
    name, port = app_name(key), APPS[key]["port"]
    pid = pid or registered_pid(key)
    if pid is None:
        return "warning", T("stop_unknown", name=name), None
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=15)
        else:
            _signal(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError as exc:
        return "error", T("stop_error", name=name, err=exc), None
    if not _wait_port_closed(port, 10) and os.name != "nt":
        try:
            _signal(pid, signal.SIGKILL)
        except OSError:
            pass
        _wait_port_closed(port, 3)
    proc = _children().pop(key, None)
    if proc is not None:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    reg = _read_registry()
    reg.pop(key, None)
    _write_registry(reg)
    return "success", T("stopped", name=name), None


def flash(key: str, result: tuple) -> None:
    ss.setdefault("_hub_flash", {})[key] = result


def show_flash(key: str) -> None:
    item = ss.get("_hub_flash", {}).pop(key, None)
    if item:
        level, text, detail = item
        getattr(st, level)(text)
        if detail:
            st.code(detail, language="text")


for _key, _proc in list(_children().items()):  # zakończone procesy: posprzątać (bez procesów-zombie)
    if _proc.poll() is not None:
        _children().pop(_key, None)
importlib.invalidate_caches()  # nowo zainstalowane pakiety są widoczne bez restartu
STATES = {key: app_state(key) for key in APPS}
EXT_PID = {key: listening_pid(APPS[key]["port"]) for key, state in STATES.items() if state == "external"}
STALE = {key for key, pid in EXT_PID.items() if pid and started_without_fix(pid)}  # stara instancja bez poprawki

# ---------------------------------------------------------------------------
# Pasek boczny
# ---------------------------------------------------------------------------
with st.sidebar:
    lang_label = "Language · Język · Sprache"
    if hasattr(st, "segmented_control") and _supports(st.segmented_control, "required"):
        st.segmented_control(lang_label, list(LANGS), key="hub_lang", required=True, format_func=LANGS.get,
                             label_visibility="collapsed")
    else:
        st.radio(lang_label, list(LANGS), key="hub_lang", format_func=LANGS.get, horizontal=True,
                 label_visibility="collapsed")

    st.markdown(f"**{T('sb_apps')}**")
    dots = {"running": "🟢", "external": "🟢", "busy": "🟠", "stopped": "⚪"}
    for key, app in APPS.items():
        label = f"{'🟠' if key in STALE else dots[STATES[key]]} {app['icon']} {app_name(key)}"
        linked = STATES[key] in ("running", "external") and key not in STALE
        st.markdown(f"[{label}]({app_url(key)})" if linked else label)
    st.divider()
    st.markdown(f"**{T('sb_materials')}**")
    st.markdown("\n".join(f"- {item}" for item in TX["sb_mat_items"]))
    st.divider()

    native = ST_VERSION >= Version(NATIVE_VERSION)
    if native:
        env_key = "env_ok"
    elif ST_VERSION < Version(MIN_VERSION):
        env_key = "env_too_old"
    else:
        env_key = "env_compat" if COMPAT_ACTIVE else "env_no_compat"
    title = f"{T('env')} · Streamlit {st.__version__}" + (f" · {T('env_compat_short')}" if env_key == "env_compat"
                                                         else "")
    with st.expander(title, expanded=env_key in ("env_no_compat", "env_too_old")):
        message = T(env_key, ver=st.__version__, native=NATIVE_VERSION, min=MIN_VERSION)
        if env_key == "env_ok":
            st.success(message)
        else:
            (st.info if env_key == "env_compat" else st.error)(message)
            st.code(shell_join([sys.executable, "-m", "pip", "install", "--upgrade",
                                f"streamlit>={NATIVE_VERSION}"]), language="bash")
        st.caption(T("env_python", exe=sys.executable))
        st.caption(T("env_ports", dir=STATE_DIR))

# ---------------------------------------------------------------------------
# Wyszukiwanie i karty aplikacji
# ---------------------------------------------------------------------------
st.title("🤖 ML Hub")
st.markdown(T("tagline"))

search = st.text_input(T("search"), placeholder=T("search_ph"), key="hub_search")
q = (search or "").lower().strip()
filtered = []
for key, app in APPS.items():
    hay = " ".join([*app["name"].values(), *app["desc"].values(), app["models"], *app["keywords"]]).lower()
    if not q or q in hay or any(word in hay for word in q.split()):
        filtered.append(key)
if q and not filtered:
    st.info(T("no_results"))
    filtered = list(APPS)

st.subheader(T("apps"))
cols = st.columns(2)
for i, key in enumerate(filtered):
    app, state, name = APPS[key], STATES[key], app_name(key)
    port = app["port"]
    with cols[i % 2], st.container(border=True):
        st.markdown(f"### {app['icon']} {name}")
        st.caption(app["models"])
        st.write(app["desc"][LANG])
        st.markdown(T(f"state_{state}", port=port))
        show_flash(key)
        missing = missing_packages(key)
        if missing:
            st.warning(T("missing", pkgs=", ".join(missing)))
            st.code(shell_join([sys.executable, "-m", "pip", "install", *missing]), language="bash")

        ext_pid, stale = EXT_PID.get(key), key in STALE
        if stale:
            st.warning(T("ext_old"))
        b1, b2, _ = st.columns([1, 1, 1])
        if state == "running" or (state == "external" and not stale):
            b1.link_button(T("open"), app_url(key), type="primary", icon=":material/open_in_new:")
            if (state == "running" or ext_pid) and b2.button(T("stop"), key=f"stop_{key}",
                                                             icon=":material/stop_circle:"):
                flash(key, stop(key, ext_pid))
                st.rerun()
        elif stale:
            if b1.button(T("restart"), key=f"restart_{key}", type="primary", icon=":material/restart_alt:"):
                with st.spinner(T("starting", name=name, port=port)):
                    result = stop(key, ext_pid)
                    if result[0] == "success":
                        result = launch(key)
                flash(key, result)
                st.rerun()
            b2.link_button(T("open"), app_url(key), icon=":material/open_in_new:")
        elif state == "stopped":
            if b1.button(T("start"), key=f"start_{key}", type="primary", icon=":material/play_arrow:"):
                with st.spinner(T("starting", name=name, port=port)):
                    result = launch(key)
                flash(key, result)
                st.rerun()
        else:
            st.caption(T("busy_hint", port=port))

        with st.expander(T("details")):
            st.markdown(T("manual", lang=LANGS[LANG]))
            st.code(f"cd {shell_join([APPS_DIR / app['dir']])}\n{shell_join(app_command(key, LANG))}",
                    language="bash")
            if log_path(key).exists():
                st.markdown(T("log", path=log_path(key)))
                st.code(log_tail(log_path(key), 15) or "–", language="text")

st.divider()

# ---------------------------------------------------------------------------
# Materiały informacyjne
# ---------------------------------------------------------------------------
st.header(T("materials"))
tab1, tab2, tab3, tab4 = st.tabs(TX["tabs"])

with tab1:
    st.subheader(MATERIALS["small"]["title"][LANG])
    st.markdown(MATERIALS["small"]["body"][LANG])

with tab2:
    st.subheader(MATERIALS["predictions"]["title"][LANG])
    st.markdown(MATERIALS["predictions"]["note"][LANG])
    st.markdown(MATERIALS["predictions"]["body"][LANG])

with tab3:
    st.subheader(MATERIALS["results"]["title"][LANG])
    st.markdown(MATERIALS["results"]["body"][LANG])


def pdf_button(doc: str, fname: str, key: str) -> None:
    """Przycisk pobierania PDF z językiem pliku w etykiecie, np. „ML in predictions (EN)” lub „… (PL + DE)”."""
    path = PDF_DIR / fname
    if not path.exists():
        st.warning(T("pdf_missing", f=fname))
        return
    langs = Path(fname).stem.rsplit("_", 1)[-1].replace("-", " + ")
    st.download_button(f"{TX['pdf_labels'][doc]} ({langs})", path.read_bytes(), file_name=fname,
                       mime="application/pdf", key=key, icon=":material/download:")


with tab4:
    st.subheader(T("pdf_title"))
    st.markdown(T("pdf_intro"))
    for col, (doc, files) in zip(st.columns(len(PDFS)), PDFS.items(), strict=True):
        with col:
            pdf_button(doc, files[LANG], key=f"pdf_{doc}")
    with st.expander(T("pdf_other")):
        for col, (doc, files) in zip(st.columns(len(PDFS)), PDFS.items(), strict=True):
            with col:
                for fname in dict.fromkeys(files.values()):  # kolejność EN, PL, DE; plik PL-DE tylko raz
                    if fname != files[LANG]:
                        pdf_button(doc, fname, key=f"pdf_other_{fname}")

st.divider()
st.caption(T("footer"))
