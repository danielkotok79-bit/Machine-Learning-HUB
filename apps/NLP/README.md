# DATA-Analysis-NLP (Streamlit + scikit-learn, Textklassifikation)

Streamlit-App für CSV-Daten mit Texten: laden, aufbereiten, visualisieren, Texte in Zahlen umwandeln
(**Bag of Words** mit `CountVectorizer` oder **TF-IDF**) und mit drei Modellen klassifizieren –
**Naive Bayes (MultinomialNB)**, **Logistische Regression** und **lineare SVM (LinearSVC)**. Danach zeigt die App,
welche Wörter für welche Klasse sprechen, erklärt einzelne Vorhersagen und ordnet neue Texte ein. Zu jedem Schritt
zeigt sie den passenden Python-Code.

Vorbild sind die NLP-Teile des Kurses: das **Yelp-Projekt** (Bewertungen mit 1 und 5 Sternen, `CountVectorizer` +
`MultinomialNB`, `test_size=0.3`, `random_state=101`) und die **SMS-Spam-Lektion** (`SMSSpamCollection`, Tab-getrennt
ohne Kopfzeile, Pipeline aus `CountVectorizer`, `TfidfTransformer` und `MultinomialNB`). Für beide gibt es eine
fertige Beispiel-Konfiguration.

Die App ist aus der Supervised-Version (DATA-Analysis-SupervisedLearning) entstanden: Schritte 1–3 (Laden,
Spalten-Aktionen, neue Spalten, Diagramme), Rezept (`preprocessing.json`), Speichern/Laden, Konfigurationen und
Sprachen sind gleich geblieben; neu sind die Erkennung von Dateien ohne Kopfzeile, die Text-Aktionen und
Text-Diagramme, Schritt 4 (Text, Label, Klassen, Vektorisierung), die drei Textmodelle mit Vergleich, die
Auswertung der Wörter in Schritt 8 und das Einordnen neuer Texte in Schritt 10.

Die App gibt es auf **Deutsch, Englisch und Polnisch** – siehe [Sprachen](#sprachen).
(English: `README_EN.md` · Polski: `README_PL.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # öffnet http://localhost:8501
```

Getestet mit Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 oder neuer.
Zusätzliche Pakete braucht die NLP-Version nicht (kein nltk – die Stoppwörter stammen aus scikit-learn).

## Sprachen

Die Sprache lässt sich jederzeit oben in der Seitenleiste umschalten: **Deutsch · English · Polski**. Es ändern
sich nur die Texte – die Berechnungen sind in allen drei Sprachen identisch (Deutsch ist das Original).

| Direkt in einer Sprache starten | |
|---|---|
| Kommandozeile | `streamlit run app.py -- --lang en` (oder `pl`, `de`) – das `--` vor `--lang` ist nötig |
| Adresse im Browser | `http://localhost:8501/?lang=pl` |
| Umgebungsvariable | `NLP_TEMPLATE_LANG=de` (Windows: `set NLP_TEMPLATE_LANG=de`) |

Ohne Angabe startet die App auf Englisch. Eine in der Seitenleiste gewählte Sprache steht danach auch in der
Adresse (`?lang=…`), so bleibt sie beim Neuladen der Seite erhalten.

Übersetzt werden alle Texte der Oberfläche, Diagrammtitel und Achsen, Tabellenköpfe (auch im Export des
Vokabulars und der Testvorhersagen), die Kommentare und Beispiel-Dateinamen im erzeugten Python-Code, das Fazit in
Schritt 10 und die Dateinamen der Downloads. In allen Sprachen gleich bleiben:

* die internen Werte – Aktionen stehen in Konfigurationen (`.json`) und in `preprocessing.json` mit ihrem deutschen
  Namen; Konfigurationen und Modelle funktionieren deshalb in allen drei Sprachen;
* die Werte, die die App in die Daten schreibt: `Sonstige` (Top-N), `_jahr` / `_monat` (Datum → Jahr + Monat),
  `Spalte_1`, `Spalte_2` … (Datei ohne Kopfzeile) und die Spalte `Hinweis` im Ergebnis der Rechenlogik
  (in der Oberfläche und im Download übersetzt);
* die Beispieltexte im erzeugten Code („The food was great!“) – die Kursdaten sind englisch;
* die Kommentare im Quellcode (Deutsch).

Wird die Sprache mitten in der Arbeit gewechselt, bleiben schon erzeugte Meldungen (z. B. das Protokoll von
Schritt 2 oder 4) in der alten Sprache, bis der Schritt erneut ausgeführt wird. Texte von Streamlit selbst
(z. B. „Upload“ im Dateifeld oder das Menü ⋮) sind in jeder Version englisch.

Excel-Format in Schritt 5, 8 und 10 je Sprache: Deutsch und Polnisch `;` und Dezimalkomma, Englisch `,` und
Dezimalpunkt – jeweils UTF-8 mit BOM und mit Schutz vor CSV-Injection (in Schritt 5 auch im erzeugten Python-Code).
Texte aus fremden Quellen (Bewertungen, Nachrichten) können mit `=`, `+`, `-` oder `@` beginnen – Excel würde sie
sonst als Formel ausführen.

## Schnellstart – wie im Kurs

**Yelp (Bewertungen, 1 gegen 5 Sterne)**

1. **Schritt 1:** `yelp.csv` laden
2. **Schritt 2:** *Beispiel-Konfiguration laden* → *Übernehmen* (IDs entfernen, neue Spalte `text length`)
3. **Schritt 3:** *Textlänge* je `stars` (wie `sns.FacetGrid(yelp, col='stars')`), *Häufigste Wörter* je Klasse,
   *Mittelwerte je Gruppe* (wie `yelp.groupby('stars').mean()`)
4. **Schritt 4:** *Texte vektorisieren* (nur 1 und 5 Sterne, CountVectorizer, `test_size=0.3`, `random_state=101`)
5. **Schritt 6:** *Training starten* (MultinomialNB) und *Modelle vergleichen* → Schritt 7, 8 (wichtigste Wörter,
   Fehler, eigener Text), 9, 10

**SMS-Spam**

1. **Schritt 1:** `SMSSpamCollection` laden – Tab und fehlende Kopfzeile werden erkannt (Spalten `Spalte_1`,
   `Spalte_2`)
2. **Schritt 2:** *Beispiel-Konfiguration laden* → *Übernehmen* (Umbenennen in `label` und `message`, neue Spalte
   `length` wie `messages['message'].apply(len)`)
3. **Schritt 4:** TF-IDF mit englischen Stoppwörtern, `test_size=0.2` → Schritt 6: Naive Bayes (oder LinearSVC mit
   `class_weight='balanced'`)

Für eigene Daten: in Schritt 4 Textspalte, Label und optional einige Klassen wählen; in Schritt 6 ein Modell – der
Vergleich zeigt, welches am besten passt. Bei langsamem Rechner links die **Stichprobe** aktivieren
(`df.sample(frac=0.1)`).

## Die Schritte

| Seite | Inhalt |
|---|---|
| 1 · Daten & Bibliotheken | CSV laden (Upload oder Pfad); Trennzeichen (auch Tab), Dezimalkomma und **fehlende Kopfzeile** werden automatisch erkannt (die erste Zeile gilt als Daten, wenn ihr Wert in einer Textspalte häufig vorkommt – wie „ham“ in SMSSpamCollection; Kopfzeile: automatisch / ja / nein); Überblick mit erkannten Freitext-Spalten (Ø Wörter), Bibliotheks-Versionen |
| 2 · Aufbereitung 1 | je Spalte eine Aktion (behalten, entfernen, umbenennen, Zahl extrahieren, Datum → Jahr / Monat, erste/letzte N Zeichen, Werte ersetzen, Top-N, **Text → Länge (Zeichen)**, **Text → Anzahl Wörter** …) – Freitext und kurze Texte (Titel, Schlagzeilen) bleiben *behalten*, IDs werden zum Entfernen vorgeschlagen · neue Spalten: Werte zuordnen, Formel, Bedingung → 0/1, Kopie + Aktion · Schnell-Buttons „Länge in Zeichen“ und „Anzahl Wörter“ (wie `text length` im Kurs) |
| 3 · Analyse & Visualisierung | Überblick, Countplot, **Textlänge** je Klasse (Histogramme wie FacetGrid + Boxplot), **häufigste Wörter** insgesamt oder je Klasse (mit/ohne Stoppwörter, Wörter oder Wortpaare), **Beispieltexte** je Klasse, Histogramm, Boxplot, Korrelation, **Mittelwerte je Gruppe** (wie `yelp.groupby('stars').mean()` + Heatmap) |
| 4 · Aufbereitung 2 – Text → Zahlen | **Textspalte** und **Label**, optional nur einige **Klassen** (wie im Kurs 1 und 5 Sterne), positive Klasse, Zeilen ohne Text entfernen oder als leeren Text behandeln, `test_size`, `random_state`, `stratify` · **Vektorisierer**: CountVectorizer oder TfidfVectorizer, Stoppwörter (englisch), `ngram_range`, Kleinschreibung, Akzente, `min_df`, `max_df`, `max_features`, `binary`, `sublinear_tf` · Ergebnis: Größe des Vokabulars, Anteil der Nicht-Nullen, Verteilung der Klassen, **ein Text als Zahlen** (wie `bow_transformer.transform([text])` im Kurs), Vokabular-Tabelle, Hinweise auf doppelte Texte in Training und Test und auf Klassen mit sehr wenigen Trainingstexten (unter 5) |
| 5 · Daten speichern | Daten nach Schritt 2, **Texte nach Schritt 4** (Text, Label, Label als Zahl, `split`) oder das **Vokabular** (Wort, Anzahl Texte, Anzahl bzw. IDF); international oder für Excel |
| 6 · Modell & Training | **MultinomialNB** (alpha, fit_prior), **Logistische Regression** (C, max_iter), **LinearSVC** (C, mit Kalibrierung für Wahrscheinlichkeiten), `class_weight='balanced'`, Seed; Validierungskurve (Testdaten oder Anteil aus dem Training); der scikit-learn-Code wird live angezeigt, die Kurve wächst beim Training mit; Tipps zur Kombination mit dem Vektorisierer (z. B. LinearSVC am besten mit TF-IDF) · **Vergleich aller drei Modelle** (Accuracy, F1, ROC-AUC, Zeit) – hat eine Klasse nur einen Trainingstext, fehlt LinearSVC mit Begründung |
| 7 · Training vs. Validierung | Fehlerrate je alpha bzw. C (logarithmisch) mit Einordnung (Overfitting, Underfitting, „Trainingsfehler 0 ist bei Texten normal“, nicht besser als naiv) und Vorschlag *Besten Wert übernehmen* |
| 8 · Evaluation & Wörter | Accuracy, F1, ROC-AUC, classification_report, Confusion Matrix, Schwelle (binär) · **wichtigste Wörter je Klasse** · **falsch eingeordnete Texte** (sicherste Fehler zuerst) · Einzelfall und **eigener Text** mit eingefärbten Wörtern (blau dafür, orange dagegen) und Beitrag je Wort · Testtexte mit Vorhersage als CSV (international oder für Excel) |
| 9 · Modell speichern & laden | `models/<name>/model.skops` (sicher, Standard) oder `model.joblib` – die **Pipeline aus Vektorisierer und Modell** – + `preprocessing.json` (Rezept, Klassen, Schwelle, Kennzahlen, optional Prüfsummen der Trainingstexte), ZIP-Download, ganze Pipeline als Python-Skript |
| 10 · Neue Texte einordnen | **einzelner Text** (mit Erklärung) oder **CSV** → gleiche Aufbereitung → Vorhersage, Wahrscheinlichkeiten, Hinweise je Zeile, Evaluation (falls das Label enthalten ist), CSV-Download (international oder für Excel) und **Fazit** |

Zu jedem Schritt gibt es ein Feld **„Python-Code zu diesem Schritt“** – so lässt sich alles 1:1 in ein
Jupyter Notebook übernehmen; der Code rechnet exakt dasselbe wie die App (gleiches Vokabular, gleiche Matrizen,
gleiche Wahrscheinlichkeiten, gleiche Kurve, gleiche wichtigste Wörter, gleiche Exporte). Einstellungen (inkl. neuer
Spalten) lassen sich links unter **Konfiguration** als JSON speichern und für neue Daten mit gleichen Spalten wieder
laden. Beim Laden wird die Datei geprüft: unbrauchbare Einträge (falscher Typ, außerhalb des erlaubten Bereichs,
z. B. `ngram_range` (1, 7)) werden ignoriert und in der Meldung genannt. Konfigurationen der anderen Versionen
lassen sich laden: Schritt 2 wird übernommen, deren Label- und Modell-Einstellungen still übergangen.

## Text → Zahlen in dieser App

* **Zuerst aufteilen, dann das Vokabular lernen.** Im Kurs wurde `cv.fit_transform(X)` auf **allen** Texten
  gerechnet und erst danach aufgeteilt – das Vokabular kannte dann schon die Testtexte (ein kleines Datenleck). Die
  App teilt zuerst (`train_test_split`) und fittet den Vektorisierer nur auf den Trainingstexten; die Testtexte
  werden damit nur umgewandelt – genau wie später neue Texte. Wörter, die nur im Test vorkommen, fallen weg (Schritt
  4 zeigt, wie viel Prozent der Testwörter das Vokabular kennt).
* **Sparse Matrix.** Jeder Text wird zu einer Zeile mit einer Spalte je Wort des Vokabulars; fast alle Werte sind 0.
  scikit-learn speichert nur die Nicht-Nullen – deshalb gehen auch 100 000 Texte mit 50 000 Wörtern.
* **CountVectorizer oder TF-IDF.** Count zählt (wie im Kurs); TF-IDF gewichtet Wörter, die in vielen Texten
  vorkommen, herunter – `TfidfVectorizer` entspricht `CountVectorizer` + `TfidfTransformer` aus dem Kurs.
* **Stoppwörter:** die englische Liste von scikit-learn (im Kurs: `stopwords.words('english')` aus nltk, plus
  Satzzeichen entfernen – das macht der Vektorisierer selbst). Für deutsche oder polnische Texte gibt es keine
  eingebaute Liste; dort helfen `max_df` (z. B. 0.9) und `min_df`.
* **Wortpaare** (`ngram_range=(1, 2)`) erkennen z. B. „not good“, machen das Vokabular aber viel größer.
* **Label kodieren:** Klassen werden zu 0/1 (binär, die *positive Klasse* = 1) bzw. 0 … K−1; Zahlen-Labels (Sterne)
  werden über den Wert kodiert, Text-Labels ohne Leerzeichen am Rand („ham “ = „ham“); ein Label aus nur Leerzeichen
  zählt wie ein fehlendes (keine eigene Klasse).
* **Doppelte Texte:** Kommt ein Testtext wortgleich auch im Training vor (bei SMS häufig: Vorlagen), warnt Schritt
  4 – für diese Texte kennt das Modell die Antwort schon.

## Die drei Modelle

| Modell | wichtigster Parameter | Stärken | Wahrscheinlichkeiten |
|---|---|---|---|
| Naive Bayes (MultinomialNB) – wie im Kurs | `alpha` (Glättung, Standard 1.0) | sehr schnell, wenig Daten nötig, gute Baseline | direkt (`predict_proba`) |
| Logistische Regression | `C` (klein = stärkere Regularisierung) | meist genauer bei genug Texten, gut kalibriert, lesbare Gewichte | direkt |
| Lineare SVM (LinearSVC) | `C` | oft am genauesten mit TF-IDF | über `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |

* **Validierungskurve (Schritt 7):** Fehlerrate auf Training und Validierung für alpha = 0.01 … 5 bzw. C = 0.01 … 100
  (LinearSVC: 0.001 … 10, ohne Kalibrierung). Mit „Anteil aus Training“ lernt ein eigener Vektorisierer sein
  Vokabular nur aus dem verbleibenden Trainingsteil – sonst kennte er die Validierungstexte.
* **Trainingsfehler nahe 0 ist bei Texten normal:** Es gibt viel mehr Wörter als Texte, lineare Modelle können die
  Trainingstexte fast immer trennen. Aussagekräftig ist der Validierungsfehler im Vergleich zur naiven Vorhersage.
* **Wichtigste Wörter (Schritt 8):** Naive Bayes – `log P(Wort | Klasse)` minus Mittel der anderen Klassen;
  Logistische Regression und LinearSVC – die Koeffizienten. Für einen einzelnen Text ist der **Beitrag** eines Worts
  = Wert im Text (Anzahl bzw. TF-IDF) × Gewicht für die vorhergesagte Klasse; bei Naive Bayes ergeben die Beiträge
  plus der Prior genau die Log-Odds.
* **TF-IDF und die Parameter:** TF-IDF-Werte sind klein (meist unter 1). Naive Bayes braucht damit oft ein kleineres
  `alpha` (z. B. 0.1 statt 1.0 – sonst glättet es bei großem Vokabular fast alles weg), die Logistische Regression
  ein größeres `C` (z. B. 10); LinearSVC konvergiert mit TF-IDF deutlich schneller als mit rohen Wortzählungen.
  Schritt 6 zeigt dazu einen Tipp, die Validierungskurve in Schritt 7 den besten Wert.
* **LinearSVC und seltene Klassen:** Die Wahrscheinlichkeiten kommen aus einer Kalibrierung mit Kreuzvalidierung –
  dafür braucht jede Klasse mindestens 2 Trainingstexte. Sonst meldet die App das verständlich (Schritt 6), und der
  Vergleich läuft ohne LinearSVC.

## Schritt 10: neue Texte einordnen

* **Nichts neu fitten.** Neue Texte laufen durch das gespeicherte *Rezept* (Schritt 2, z. B. Umbenennen) und die
  gespeicherte *Pipeline* – der Vektorisierer nutzt das Vokabular aus dem Training. Unbekannte Wörter werden
  ignoriert.
* **Nötig ist nur die Textspalte** (Rohspalte der CSV) – die App zeigt sie und bietet eine CSV-Vorlage an; das Label
  ist optional (dann wird ausgewertet). Wurde mit einer Datei ohne Kopfzeile trainiert, neue Daten in derselben Form
  laden (Kopfzeile: nein).
* **Hinweise je Zeile:** leerer Text · kein bekanntes Wort (andere Sprache?) · überwiegend unbekannte Wörter.
* **Fazit:** wie viele Wörter das Modell kennt (im Vergleich zu den Testdaten), Länge der Texte im Vergleich zum
  Training, vorhergesagte Anteile, unsichere Vorhersagen, Vergleich mit dem Test aus dem Training, **wortgleiche
  Texte aus dem Training** (Prüfsummen, optional) und Klassen im Label, die das Modell nicht kennt (z. B. 2–4 Sterne
  bei einem 1-gegen-5-Modell).

Ohne App:

```python
import pipeline as pl
pipe, recipe = pl.load_bundle("models/mein_modell")
df_new, _ = pl.read_csv("neue_texte.csv")
result, report, evaluation = pl.predict_texts(pipe, recipe, df_new)
fazit = pl.assess_prediction(recipe, result, report, evaluation)
pipe.predict_proba(["The food was great!"])   # einzelne Texte direkt mit der scikit-learn-Pipeline
```

## Hinweise zu aktuellen Bibliotheksversionen

| älterer Code | Problem heute | im Template |
|---|---|---|
| `cv.fit_transform(X)` vor `train_test_split` | Vokabular aus allen Texten – die Testtexte fließen ins Modell ein | erst aufteilen, `fit_transform` nur auf `X_train`, `transform` auf `X_test` |
| `CountVectorizer(analyzer=text_process)` mit nltk | langsam; eine eigene Funktion lässt sich mit `skops` nicht sicher laden; nltk-Stoppwörter brauchen einen Download | `CountVectorizer(stop_words='english')` – Satzzeichen entfernt der Vektorisierer selbst |
| `CountVectorizer` + `TfidfTransformer` | geht weiterhin, zwei Schritte | `TfidfVectorizer` (dasselbe in einem Schritt) |
| `cv.get_feature_names()` | in scikit-learn 1.2 entfernt | `get_feature_names_out()` |
| `SVC(probability=True)` | ab scikit-learn 1.9 veraltet | `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |
| `sns.FacetGrid(yelp, col='stars', size=…)` | `size` heißt seit seaborn 0.9 `height` | `col_wrap`/`height` bzw. das Diagramm *Textlänge* |
| `yelp.groupby('stars').mean()` | pandas ≥ 2: Fehler bei Textspalten | `mean(numeric_only=True)` |
| `pd.read_csv(..., sep='\t', names=[...])` | geht weiterhin | die App erkennt Tab und fehlende Kopfzeile selbst; der erzeugte Code nutzt `header=None, names=[...]` |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | beim Laden kann beliebiger Code aus der Datei ausgeführt werden | `.skops` als Standard; `.joblib` nur für vertrauenswürdige Dateien |

## Dateien

```
app.py                  Streamlit-Oberfläche (nur UI)
pipeline.py             Rechenlogik – auch im Notebook nutzbar (import pipeline as pl)
plots.py                Diagramme (matplotlib/seaborn), geben Figures zurück
codegen.py              erzeugt den angezeigten bzw. exportierten Python-Code
i18n.py                 Sprachen: tx() übersetzt einen deutschen Text in die aktuelle Sprache
locales/                Übersetzungen {Deutsch: (Englisch, Polnisch)}, eine Datei je Modul
configs/                yelp_beispiel.json, sms_beispiel.json (Beispiel-Konfigurationen wie im Kurs)
.streamlit/config.toml  Upload-Limit 1 GB, helles Theme
tests/                  pytest-Tests + synthetische Testdaten (yelp_like, sms_like, news_like)
AUDIT.md                Bericht (polnisch): Aufbau, Prüfungen, Grenzen
README.md               Deutsch · README_EN.md Englisch · README_PL.md Polnisch
models/                 entsteht beim Speichern (Schritt 9)
exports/                Standardordner für „In Ordner speichern“ (Schritt 5)
```

Erweitern:
* neue Spalten-Aktion → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* neues Diagramm → Funktion in `plots.py` + Eintrag in `page_viz()` bzw. `page_eval()` in `app.py`
* weiteres Textmodell → `MODELS`, `MODEL_NAMES`, `TextModelConfig`, `make_model()`, `word_weights()` in
  `pipeline.py`, `model_params_ui()` in `app.py`, `estimator_code()` in `codegen.py`
* neuer oder geänderter Text → auf Deutsch in `tx("…")` schreiben (Konstanten: `N_("…")`) und die englische und
  polnische Übersetzung in der passenden Datei in `locales/` ergänzen; `pytest tests/test_i18n.py` meldet fehlende
  Übersetzungen.

## Tests

```bash
pip install pytest
pytest -q                                  # 188 Tests, ca. 4 Minuten
python tests/synthetic.py                  # synthetische Yelp-ähnliche Daten (10 000 Bewertungen, keine echten Daten)
python tests/synthetic.py --sms            # synthetische SMS-Datei wie SMSSpamCollection (Tab, ohne Kopfzeile)
```

| Datei | prüft |
|---|---|
| `tests/test_pipeline.py` | Laden (Tab ohne Kopfzeile, Anführungszeichen, Semikolon, latin-1), Text-Aktionen, Schritt 4 (Klassenauswahl, positive Klasse, Bool-/Zahlen-/Text-Label, leere Texte, verständliche Fehler, doppelte Texte, Datenleck, nur Stoppwörter), alle drei Modelle lernen das Sentiment, Erklärung = Vorhersage (Log-Odds), Mehrklassen und Vergleich, Exporte, Speichern/Laden, neue Texte (Hinweise, Überschneidung mit dem Training, fehlende Spalte), SMS-Ablauf mit Umbenennen |
| `tests/test_codegen_parity.py` | der angezeigte Python-Code rechnet **exakt** dasselbe wie die App: Vokabular, Matrizen, Split, Wahrscheinlichkeiten, Kurve, wichtigste Wörter, Exporte – auch mit Datei ohne Kopfzeile, Klassenauswahl, leeren Texten, Stichprobe, Bool-/Zahlen-Label; Modellvergleich, ganzes Skript und Diagramm-Code laufen |
| `tests/test_audit_regressions.py` | Sicherheit und Robustheit: ZIP-Grenzen, pickle nur nach Bestätigung, eigene Funktionen im Modell werden von skops abgelehnt, beschädigte oder fremde Rezepte, Pipeline passt nicht zum Rezept, kein Code und kein Markdown aus Namen, Formel- und Konfigurationsgrenzen, CSV-Injection, gescheitertes Speichern |
| `tests/test_audit6_regressions.py` | Audit 6: jeder Eintrag von `preprocessing.json`, den Schritt 9 und 10 benutzen, wird beim Laden geprüft (auch Schwelle, Klassenauswahl, Kennzahlen, Spalten-Aktionen, Vektorisierer, Größe), leere Labels, kurze Texte in Schritt 2, Erkennung der Kopfzeile, Confusion Matrix bei vielen Klassen, Code der Beispieltexte, naive Vorhersage aus den Trainingsdaten |
| `tests/test_review_regressions.py` | Funde der Prüfung vor der Auslieferung: Klasse mit nur einem Trainingstext (Hinweis in Schritt 4, verständliche Meldung bei LinearSVC, Vergleich ohne LinearSVC – auch im erzeugten Code), Validierungskurve mit nur einer Klasse im Trainingsteil, Einordnung „nicht besser als naiv“ bei ungünstigem alpha, CSV für Excel in Schritt 8 und 10, Klassennamen wörtlich, polnische Modellnamen |
| `tests/test_app.py` | Klick-Test der Oberfläche: Yelp wie im Kurs (alle 10 Schritte), SMS-Datei ohne Kopfzeile mit LinearSVC, Fehlerfälle und Schutz vor Überschreiben, keine Textspalte, Konfiguration exportieren/importieren |
| `tests/test_charts.py` | Diagramme der NLP-Version (Textlänge, häufigste Wörter, Vergleich, Kurve, wichtigste Wörter, Beitrag) auch mit ungewöhnlichen Daten |
| `tests/test_i18n.py` | Sprachen: jeder Text übersetzt mit denselben Platzhaltern, Excel-CSV je Sprache (App = Code), erzeugter Code rechnet in jeder Sprache gleich, die ganze App auf Englisch und Polnisch ohne deutsche Reste |

## Hinweise

* Sehr große CSVs besser über „Pfad auf diesem Rechner“ laden als hochladen.
* Während des Trainings nichts anklicken – jede Eingabe startet das Skript neu und bricht das Training ab.
* **Laufzeit:** Naive Bayes ist immer schnell. Logistische Regression und LinearSVC brauchen bei großem Vokabular und
  rohen Wortzählungen deutlich länger (gemessen mit 14 000 synthetischen Texten und 28 754 Wörtern auf 2 Kernen:
  Kurve + Training ca. 2 Minuten bzw. 1 Minute; LinearSVC mit TF-IDF und Wortpaaren 15 Sekunden). Schneller: TF-IDF,
  `min_df`/`max_features` in Schritt 4, die Validierungskurve ausschalten oder die Stichprobe nutzen.
* **Bag of Words kennt keine Reihenfolge:** „not good“ und „good, not bad“ sehen fast gleich aus (Wortpaare helfen
  etwas). Ironie, Sarkasmus und gemischte Bewertungen bleiben schwierig – Schritt 8 zeigt diese Fehler. Für
  Bedeutung jenseits einzelner Wörter braucht es Deep Learning (Embeddings, Transformer) – nicht Teil dieser App.
* Ein Modell kennt nur die Sprache und die Themen seiner Trainingstexte; Schritt 10 meldet, wenn neue Texte viele
  unbekannte Wörter enthalten.
* **Nur lokal verwenden.** Das Pfad-Feld liest jede Datei, auf die der angemeldete Benutzer Zugriff hat, und
  `models/` ist für alle Nutzer der App gleich. Die App daher nicht ohne Anmeldung ins Netz stellen.
  Modelle nur aus vertrauenswürdigen Quellen laden (ein ZIP darf entpackt höchstens 2 GB groß sein). `.skops` wird
  ohne Code-Ausführung geladen; `.joblib` ist pickle und wird aus einem hochgeladenen ZIP nur nach ausdrücklicher
  Bestätigung geladen. Geladen werden nur Textmodelle dieser App – Modelle der anderen Versionen lehnt die App mit
  einer klaren Meldung ab, ebenso ein unvollständiges oder beschädigtes `preprocessing.json` und eine Pipeline, die
  nicht zum Rezept passt.
* Das gespeicherte Modell enthält das **Vokabular** (alle Wörter der Trainingstexte) und – wenn nicht abgewählt –
  Prüfsummen von Trainingstexten. Bei vertraulichen Texten das Modell nur an Berechtigte weitergeben.
