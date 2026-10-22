# Data-Analysis-Template (Streamlit + Keras)

Streamlit-App für tabellarische CSV-Daten: laden, aufbereiten, visualisieren, ein neuronales Netz (Keras)
trainieren, bewerten und auf neue Daten anwenden – für beliebige Datensätze (binär, Mehrklassen, Regression).
Zu jedem Schritt zeigt die App den passenden Python-Code.

Die App gibt es auf **Deutsch, Englisch und Polnisch** – siehe [Sprachen](#sprachen).
(English: `README_EN.md` · Polski: `README_PL.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # öffnet http://localhost:8501
```

Getestet mit Python 3.13, TensorFlow 2.21 (Keras 3.15), pandas 3.0 und 2.2, scikit-learn 1.9, Streamlit 1.65.
Empfohlen: Python 3.10–3.13.

## Sprachen

Die Sprache lässt sich jederzeit oben in der Seitenleiste umschalten: **Deutsch · English · Polski**. Es ändern
sich nur die Texte – die Berechnungen sind in allen drei Sprachen identisch (Deutsch ist das Original).

| Direkt in einer Sprache starten | |
|---|---|
| Kommandozeile | `streamlit run app.py -- --lang en` (oder `pl`, `de`) – das `--` vor `--lang` ist nötig |
| Adresse im Browser | `http://localhost:8501/?lang=pl` |
| Umgebungsvariable | `DL_TEMPLATE_LANG=en` (Windows: `set DL_TEMPLATE_LANG=en`) |

Ohne Angabe startet die App auf Deutsch. Eine in der Seitenleiste gewählte Sprache steht danach auch in der
Adresse (`?lang=…`), so bleibt sie beim Neuladen der Seite erhalten.

Übersetzt werden alle Texte der Oberfläche, Diagrammtitel und Achsen, Tabellenköpfe, die Kommentare und
Beispiel-Dateinamen im erzeugten Python-Code, das Fazit in Schritt 10 und die Dateinamen der Downloads. In allen
Sprachen gleich bleiben:

* die internen Werte – Aktionen und Strategien stehen in Konfigurationen (`.json`) und in `preprocessing.json`
  mit ihrem deutschen Namen; Konfigurationen und Modelle funktionieren deshalb in allen drei Versionen;
* die Werte, die die App in die Daten schreibt: `Sonstige` (Top-N), `fehlend` (eigene Kategorie für Lücken),
  `_jahr` / `_monat` (Datum → Jahr + Monat); bei „Werte zuordnen“ gehen `sonst=…` und `rest=…` in jeder Sprache;
* die Kommentare im Quellcode (Deutsch).

Wird die Sprache mitten in der Arbeit gewechselt, bleiben schon erzeugte Meldungen (z. B. das Protokoll von
Schritt 2 oder 4) in der alten Sprache, bis der Schritt erneut ausgeführt wird. Texte von Streamlit selbst
(z. B. „Upload“ im Dateifeld oder das Menü ⋮) sind in jeder Version englisch.

Excel-Format in Schritt 5 je Sprache: Deutsch und Polnisch `;` und Dezimalkomma, Englisch `,` und Dezimalpunkt
(Großbritannien, USA, Australien, Kanada, Irland, Neuseeland … verwenden alle dieses Format) – jeweils UTF-8 mit
BOM und mit Schutz vor CSV-Injection.

## Schnellstart

1. **Schritt 1:** CSV hochladen oder Pfad eintragen → *Daten laden*
2. **Schritt 2:** Aktionen je Spalte prüfen → *Übernehmen*; bei Bedarf neue Spalten erstellen
3. **Schritt 3:** Spalten für die Diagramme auswählen und Daten ansehen
4. **Schritt 4:** Label und Features wählen → *Daten für ML vorbereiten*
5. **Schritt 6:** *Training starten* → danach 7, 8, 9, 10

Für den öffentlichen LendingClub-Datensatz (`lending_club_loan_two.csv`) gibt es in Schritt 2 eine
**Beispiel-Konfiguration** (Schritt 2, 4 und 6 fertig eingestellt). Bei langsamem Rechner links die
**Stichprobe** aktivieren (`df.sample(frac=0.1)`).

## Die Schritte

| Seite | Inhalt |
|---|---|
| 1 · Daten & Bibliotheken | CSV laden (Upload oder Pfad; Trennzeichen `;` und Dezimalkomma werden automatisch erkannt), Überblick, Bibliotheks-Versionen, optional Spalten-Beschreibungen |
| 2 · Aufbereitung 1 | je Spalte eine Aktion (behalten, entfernen, Zahl extrahieren, Datum → Jahr / Monat / **Jahr + Monat**, erste/letzte N Zeichen, Werte ersetzen, Top-N …) · **neue Spalten**: Werte zuordnen, Formel, Bedingung → 0/1, Kopie + Aktion |
| 3 · Analyse & Visualisierung | **Auswahl der Spalten** · Überblick, Countplot, Histogramm, Boxplot, Scatter, Heatmap, Korrelation mit einer Spalte, Pairplot, Anteil je Kategorie, Gruppen-Statistik, **Jointplot** (scatter/kde/hex/reg), **Dichte 2D (KDE)** mit Filter auf eine Gruppe, **Regression (lmplot)** mit Farbe (hue) und Teilgrafiken (col), **Histogramm je Gruppe** (überlagert) |
| 4 · Aufbereitung 2 | Label + Problemtyp, Features, fehlende Werte je Spalte (mit Korrelations-Hilfe für den Gruppen-Mittelwert); `train_test_split` **zuerst**, danach Füllwerte, Dummies (Übersicht der neuen 0/1-Spalten) und Skalierung – nur aus den Trainingsdaten gelernt |
| 5 · Daten speichern | Daten nach Schritt 2 oder ML-fertig nach Schritt 4 (mit Spalte `split`) als CSV – international oder für deutsches Excel; eine vorhandene Datei wird nur nach Bestätigung überschrieben |
| 6 · Modell & Training | Hidden Layers (Neuronen, Aktivierung, Dropout), Ausgabe-Aktivierung, loss, optimizer, epochs, batch_size, EarlyStopping/patience – der passende Keras-Code wird live angezeigt |
| 7 · loss vs. val_loss | Kurven + Einordnung (Overfitting, EarlyStopping) |
| 8 · Evaluation | classification_report, confusion_matrix, ROC-AUC, Schwelle, Einzelfall, Feature-Wichtigkeit · Regression: MAE, RMSE, R² |
| 9 · Modell speichern & laden | `models/<name>/model.keras` (oder `.h5`) + `preprocessing.json`, ZIP-Download, ganze Pipeline als Python-Skript; Prüfsummen der Trainingszeilen optional |
| 10 · Vorhersage | neue CSV oder Formular → gleiche Aufbereitung → Vorhersage, Interpretation, CSV-Download und **Fazit** (Datenqualität, Aussage des Ergebnisses, Empfehlung) |

Zu jedem Schritt gibt es ein Feld **„Python-Code zu diesem Schritt“** – so lässt sich alles 1:1 in ein
Jupyter Notebook übernehmen. Einstellungen (inkl. neuer Spalten) lassen sich links unter **Konfiguration**
als JSON speichern und für neue Daten mit gleichen Spalten wieder laden. Beim Laden wird die Datei geprüft:
unbrauchbare Einträge (falscher Typ, außerhalb des erlaubten Bereichs, z. B. 10 Mio. Neuronen) werden ignoriert
und in der Meldung genannt.

### Daten speichern (Schritt 5)

* Für Vorhersagen in Schritt 10 immer die **ursprüngliche** CSV-Datei verwenden, nicht den Export: Das gespeicherte
  Modell wiederholt die Aufbereitung aus Schritt 2 selbst (z. B. würde „Werte ersetzen 1=2; 2=3“ sonst doppelt
  angewendet).
* Format „deutsches Excel“: Texte, die mit `=`, `+`, `-` oder `@` beginnen, bekommen ein `'` davor – sonst würde
  Excel sie als Formel ausführen (CSV-Injection). Im internationalen Format bleiben die Daten unverändert.
* Heißt ein Feature selbst `split`, heißt die train/test-Spalte `split_train_test`.

## Neue Spalten (Schritt 2)

| Art | Beispiel | Ergebnis |
|---|---|---|
| Werte zuordnen | aus `loan_status`: `Fully Paid=1; Charged Off=0` | neue Spalte `loan_repaid` (1/0); nicht genannte Werte bleiben leer oder `sonst=…` |
| Formel | `loan_amnt / annual_inc` | Zahl je Zeile; erlaubt sind `+ - * / ** // %`, Klammern, `abs log log10 sqrt exp round` |
| Bedingung → 0/1 | `dti > 30 and home_ownership == 'RENT'` | 1 = erfüllt, 0 = nicht erfüllt, leer bei Lücken |
| Kopie + Aktion | `earliest_cr_line` + „Datum → Jahr“ | Original bleibt, Ergebnis als zusätzliche Spalte |

Spaltennamen mit Leer- oder Sonderzeichen in Formeln in \`Backticks\` schreiben. Formeln werden geprüft:
nur Rechnen, Vergleiche und die genannten Funktionen – kein beliebiger Python-Code. Damit keine Formel das
Programm blockieren kann, gelten Grenzen: höchstens 1000 Zeichen und 100 Verschachtelungsebenen, Exponent bei
`**` zwischen −100 und 100 (oder eine Spalte), `round(x, n)` mit n von 0 bis 10, Text in Anführungszeichen nur in
Vergleichen. Passt bei „Werte zuordnen“ kein einziger Wert (z. B. andere Groß-/Kleinschreibung), nennt die
Meldung die vorhandenen Werte. Wird eine neue Spalte von einer späteren benutzt, lassen sich beide nur zusammen
entfernen. Neue Spalten gehören zum
gespeicherten Rezept und werden in Schritt 10 für neue Daten automatisch mitberechnet. Eine neue Spalte, die aus
derselben CSV-Spalte wie das Label entsteht (z. B. `loan_repaid` aus `loan_status`), wird nicht als Feature
vorgeschlagen (Datenleck).

## Schritt 10: neue Daten richtig behandeln

* **Nichts neu fitten.** Neue Daten laufen durch das gespeicherte *Rezept* (`preprocessing.json`):
  gleiche Spalten-Aktionen und neue Spalten, Füllwerte, Dummy-Spalten in gleicher Reihenfolge, gleicher
  Scaler, gleiche Label-Kodierung und Schwelle.
* **Gleiche Rohspalten wie im Training** – das Modell bestimmt die Features. Die App zeigt die
  benötigten Spalten und bietet eine CSV-Vorlage an. Fehlende Spalten → Fehlermeldung; zusätzliche
  Spalten (auch das Label) werden ignoriert.
* **Abweichungen** werden behandelt und im **Fazit** erklärt: Lücken → Werte aus dem Training (jede Zeile
  braucht für eine Vorhersage einen Wert – auch dort, wo im Training Zeilen mit Lücken entfernt wurden),
  unbekannte Kategorien → alle Dummies 0, Werte außerhalb des Trainingsbereichs → Hinweis.
* **Daten aus dem Training erkannt:** Das Rezept enthält Prüfsummen (Hashes) einer Stichprobe von bis zu 3000
  Trainingszeilen. Stecken Trainingszeilen in den neuen Daten, nennt das Fazit den Anteil – Kennzahlen auf solchen
  Daten sind zu optimistisch. Aus einer Prüfsumme lässt sich keine Zeile zurückrechnen; wer die Originaldaten hat,
  kann aber prüfen, ob eine bestimmte Zeile im Training war. Bei vertraulichen Daten deshalb in Schritt 9
  *Prüfsummen … mitspeichern* ausschalten, bevor das Modell weitergegeben wird (die Prüfung entfällt dann).
* Zeilen mit unendlichem Label (`inf`) werden in Schritt 4 entfernt und in Schritt 10 nicht mitbewertet.
* Haben die neuen Daten schon Spalten mit dem Namen einer Ergebnisspalte (z. B. `Vorhersage`, `Hinweis` aus einem
  früheren Export), werden sie im Ergebnis ersetzt – die App nennt sie.
* Ist das Label in den neuen Daten vorhanden, wird automatisch evaluiert und mit dem Test aus dem Training
  verglichen.

Ohne App:

```python
import pipeline as pl
model, recipe = pl.load_bundle("models/mein_modell")
df_new, _ = pl.read_csv("neue_daten.csv")
result, report, evaluation = pl.predict_new(model, recipe, df_new)
fazit = pl.assess_prediction(recipe, result, report, evaluation)
```

## Hinweise zu aktuellen Bibliotheksversionen

Ältere Anleitungen und Notebooks enthalten oft Zeilen, die mit aktuellen Versionen nicht mehr laufen:

| älterer Code | Problem heute | im Template |
|---|---|---|
| `model.predict_classes(X)` | seit TensorFlow 2.6 entfernt | `(model.predict(X) > 0.5).astype("int32")` |
| `model.save('x.h5')` mit `loss='mse'` | Keras 3 kann die Datei nicht mehr laden | `.keras` als Standard, volle Loss-Namen, `load_model(..., compile=False)` |
| `df.corr()` | pandas ≥ 2: Fehler bei Textspalten | `df.corr(numeric_only=True)` |
| `df.select_dtypes(['object'])` | pandas 3: Text hat den Typ `str`; `'object'` findet ihn nur noch übergangsweise | Typprüfung über `is_numeric_dtype` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False → Keras-Fehler | `dtype=float` |
| `df.groupby('total_acc').mean()['mort_acc']` | pandas ≥ 2: Fehler bei Textspalten | `df.groupby('total_acc')['mort_acc'].mean()` |
| `sns.distplot` | veraltet | `sns.histplot` |
| `sns.kdeplot(df['a'], df['b'], shade=True, shade_lowest=False)` | seaborn ≥ 0.12: Fehler, nur `data` darf ohne Namen übergeben werden; `shade`/`shade_lowest` veraltet | `sns.kdeplot(x='a', y='b', data=df, fill=True, thresh=0.05)` |
| `plt.figure(figsize=(11, 7))` vor `sns.lmplot(...)` | wirkt nicht – lmplot erzeugt eine eigene Figure (im Notebook: „Figure … with 0 Axes“) | `sns.lmplot(..., height=4, aspect=1.15)` |
| Lücken füllen, **dann** `train_test_split` | Füllwerte enthalten Information aus den Testdaten (Datenleck) | Split zuerst; Füllwerte, Kategorien und Scaler nur aus den Trainingsdaten |

Zusätzliche Optionen (jeweils abschaltbar): `restore_best_weights`, `stratify=y`, einstellbare
Schwelle + ROC-AUC, `class_weight='balanced'`, Validierung aus dem Trainingsanteil statt mit den
Testdaten, Label-Skalierung bei Regression, Mehrklassen (softmax), Datenleck-Warnung,
Permutation Importance.

## Dateien

```
app.py                  Streamlit-Oberfläche (nur UI)
pipeline.py             Rechenlogik – auch im Notebook nutzbar (import pipeline as pl)
plots.py                Diagramme (matplotlib/seaborn), geben Figures zurück
codegen.py              erzeugt den angezeigten bzw. exportierten Python-Code
i18n.py                 Sprachen: tx() übersetzt einen deutschen Text in die aktuelle Sprache
locales/                Übersetzungen {Deutsch: (Englisch, Polnisch)}, eine Datei je Modul
configs/                lending_club_beispiel.json (Beispiel-Konfiguration LendingClub)
.streamlit/config.toml  Upload-Limit 1 GB, helles Theme
tests/                  pytest-Tests + synthetische Testdaten
AUDIT.md                Audit-Bericht (polnisch): gefundene Fehler, Korrekturen, Tests
README.md               Deutsch · README_EN.md Englisch · README_PL.md Polnisch
models/                 entsteht beim Speichern (Schritt 9)
exports/                Standardordner für „In Ordner speichern“ (Schritt 5)
```

Erweitern:
* neue Spalten-Aktion → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* neue Art neuer Spalten → `NEW_KINDS` + `apply_new_columns()` in `pipeline.py` (+ `_new_column_lines()` in `codegen.py`)
* neues Diagramm → Funktion in `plots.py` + Eintrag in `CHARTS` / `page_viz()` in `app.py` (+ `chart_code()` in
  `codegen.py`)
* neue Layer-Vorlage → `PRESETS` in `pipeline.py`
* neuer oder geänderter Text → auf Deutsch in `tx("…")` schreiben (Konstanten: `N_("…")`) und die englische und
  polnische Übersetzung in der passenden Datei in `locales/` ergänzen; `pytest tests/test_i18n.py` meldet fehlende
  Übersetzungen.

## Tests

```bash
pip install pytest
pytest -q                                  # 190 Tests, ca. 4–5 Minuten
python tests/synthetic.py --rows 20000     # synthetische Demo-CSV (keine echten Daten)
```

| Datei | prüft |
|---|---|
| `tests/test_pipeline.py` | Beispiel-Pipeline, Speichern/Laden (.keras/.h5), neue Daten, Regression, Mehrklassen, exportiertes Skript |
| `tests/test_codegen_parity.py` | der angezeigte Python-Code rechnet **exakt** dasselbe wie die App (X_train, X_test, y) – auch in Grenzfällen und mit neuen Spalten |
| `tests/test_new_features.py` | Jahr + Monat, neue Spalten (inkl. Schutz vor beliebigem Code in Formeln), Daten speichern, Fazit |
| `tests/test_audit_regressions.py` | je ein Test pro Fehler aus dem ersten Audit (siehe `AUDIT.md`) |
| `tests/test_audit2_regressions.py` | zweites Audit: Grenzen für Formeln, Prüfung importierter Konfigurationen, `inf` im Label, Export (Excel, Spalte `split`), Diagramme mit leeren Spalten, ZIP-Größe, Fazit |
| `tests/test_app.py` | Klick-Test der Oberfläche (alle 10 Schritte, neue Spalten, Export, Fazit; Bool-Label, Semikolon-CSV, Schutz vor Überschreiben; abhängige neue Spalten, unbrauchbare Werte, Fehler beim Zeichnen, Modell ohne Prüfsummen) |
| `tests/test_audit3_regressions.py` | drittes Audit: kein Code aus Spaltennamen/Werten im angezeigten Python-Code, Schritt 10 mit gleichnamigen Spalten, Diagrammtitel (Notebook, `$`), Sprachwechsel (Tabellen, Export, Adresse) |
| `tests/test_charts.py` | neue Diagramme in Schritt 3 (Jointplot, Dichte 2D, lmplot, Histogramm je Gruppe): ungewöhnliche Daten, angezeigter Code läuft, Klick-Test |
| `tests/test_i18n.py` | Sprachen: jeder Text hat eine englische und polnische Übersetzung mit denselben Platzhaltern, Deutsch unverändert, Excel-CSV je Sprache, erzeugter Code rechnet in jeder Sprache gleich, die ganze App auf Englisch und Polnisch ohne deutsche Reste |

## Hinweise

* Sehr große CSVs besser über „Pfad auf diesem Rechner“ laden als hochladen.
* Während des Trainings nichts anklicken – jede Eingabe startet das Skript neu und bricht das Training ab.
* GPU: TensorFlow nutzt unter Windows nativ nur die CPU (GPU über WSL2). Die App zeigt in Schritt 1, ob eine GPU gefunden wurde.
* **Nur lokal verwenden.** Das Pfad-Feld liest jede Datei, auf die der angemeldete Benutzer Zugriff hat, und
  `models/` ist für alle Nutzer der App gleich. Die App daher nicht ohne Anmeldung ins Netz stellen.
  Modelle (ZIP/.h5/.keras) nur aus vertrauenswürdigen Quellen laden (ein ZIP darf entpackt höchstens 2 GB groß
  sein; Formeln im Rezept werden beim Laden erneut geprüft, gespeicherter Code wird nie ausgeführt).
* **Testkennzahlen sind leicht optimistisch**, wenn die Testdaten auch als Validierung für EarlyStopping dienen
  (Standard „Validierung: Testdaten“) oder die Schwelle in Schritt 8 auf den Testdaten eingestellt wird. Für eine
  ehrliche Schätzung in Schritt 6 „Validierung: Anteil aus Training“ wählen.
* **Kein Datenleck in Schritt 4:** Füllwerte, Kategorien (Dummy-Spalten) und Scaler werden nach dem Split nur
  aus den Trainingsdaten gelernt. Eine Kategorie, die nur in den Testdaten vorkommt, wird wie ein unbekannter
  Wert in neuen Daten behandelt (alle Dummies 0). Aus allen Zeilen (vor dem Split) entstehen nur noch die Liste
  bei „Top-N behalten“ in Schritt 2 und die automatischen Vorschläge in den Tabellen – beides mit geringem Einfluss.
