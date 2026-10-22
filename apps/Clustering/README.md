# DATA-Analysis-Clustering (Streamlit + scikit-learn, K-Means)

Streamlit-App für tabellarische CSV-Daten: laden, aufbereiten, visualisieren und mit **K-Means** in Gruppen
(Cluster) einteilen – **Unsupervised Learning**: Es gibt kein Label, K-Means sucht selbst nach Zeilen, die sich
ähneln. Danach beschreibt die App die Cluster (Profile, 2D-Karte), vergleicht sie – wenn vorhanden – mit einer
bekannten Einteilung und ordnet neue Daten den Clustern zu. Zu jedem Schritt zeigt die App den passenden
Python-Code.

Vorbild ist das K-Means-Projekt aus dem Kurs: **College_Data** – K-Means mit K = 2 auf allen Zahlenspalten; die
Spalte *Private* (Yes/No) wird dabei ignoriert und erst danach zum Vergleich benutzt. Für diesen Datensatz gibt es
eine fertige Beispiel-Konfiguration.

Die App ist aus der Supervised-Version (DATA-Analysis-SupervisedLearning) entstanden: Schritte 1–3 (Laden,
Spalten-Aktionen, neue Spalten, Diagramme), Rezept (`preprocessing.json`), Speichern/Laden, Konfigurationen und
Sprachen sind gleich geblieben; neu sind Schritt 4 ohne Label und ohne Split, K-Means in Schritt 6, die Wahl von K
in Schritt 7, die Cluster-Auswertung in Schritt 8 und die Zuordnung neuer Daten in Schritt 10.

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

## Sprachen

Die Sprache lässt sich jederzeit oben in der Seitenleiste umschalten: **Deutsch · English · Polski**. Es ändern
sich nur die Texte – die Berechnungen sind in allen drei Sprachen identisch (Deutsch ist das Original).

| Direkt in einer Sprache starten | |
|---|---|
| Kommandozeile | `streamlit run app.py -- --lang en` (oder `pl`, `de`) – das `--` vor `--lang` ist nötig |
| Adresse im Browser | `http://localhost:8501/?lang=pl` |
| Umgebungsvariable | `CL_TEMPLATE_LANG=de` (Windows: `set CL_TEMPLATE_LANG=de`) |

Ohne Angabe startet die App auf Englisch. Eine in der Seitenleiste gewählte Sprache steht danach auch in der
Adresse (`?lang=…`), so bleibt sie beim Neuladen der Seite erhalten.

Übersetzt werden alle Texte der Oberfläche, Diagrammtitel und Achsen, Tabellenköpfe, die Kommentare und
Beispiel-Dateinamen im erzeugten Python-Code, das Fazit in Schritt 10, die Namen der neuen Spalten im Export
(`Cluster`, `Cluster-Name`, `Abstand`) und die Dateinamen der Downloads. In allen Sprachen gleich bleiben:

* die internen Werte – Aktionen und Strategien stehen in Konfigurationen (`.json`) und in `preprocessing.json`
  mit ihrem deutschen Namen; Konfigurationen und Modelle funktionieren deshalb in allen drei Sprachen;
* die Werte, die die App in die Daten schreibt: `Sonstige` (Top-N), `fehlend` (eigene Kategorie für Lücken),
  `_jahr` / `_monat` (Datum → Jahr + Monat); bei „Werte zuordnen“ gehen `sonst=…` und `rest=…` in jeder Sprache;
* die Kommentare im Quellcode (Deutsch).

Wird die Sprache mitten in der Arbeit gewechselt, bleiben schon erzeugte Meldungen (z. B. das Protokoll von
Schritt 2 oder 4) in der alten Sprache, bis der Schritt erneut ausgeführt wird. Texte von Streamlit selbst
(z. B. „Upload“ im Dateifeld oder das Menü ⋮) sind in jeder Version englisch.

Excel-Format in Schritt 5 je Sprache: Deutsch und Polnisch `;` und Dezimalkomma, Englisch `,` und Dezimalpunkt –
jeweils UTF-8 mit BOM und mit Schutz vor CSV-Injection (auch im erzeugten Python-Code).

## Schnellstart – wie im Kurs

1. **Schritt 1:** `College_Data` (CSV, erste Spalte = Name der Hochschule) laden
2. **Schritt 2:** *Beispiel-Konfiguration laden* → *Übernehmen* (Name behalten, Grad.Rate 118 → 100)
3. **Schritt 3:** z. B. Scatterplot Grad.Rate gegen Room.Board mit Farbe = Private, Histogramm je Gruppe
4. **Schritt 4:** *Daten für K-Means vorbereiten* (17 Features, Private nur zum Vergleich, StandardScaler)
5. **Schritt 6:** *Training starten* (K = 2) → Schritt 7 (Wahl von K), 8 (Profile, Vergleich mit Private), 9, 10

Für eigene Daten: in Schritt 4 Features wählen, optional eine bekannte Einteilung zum Vergleich und eine Spalte mit
Namen; in Schritt 6 K wählen – Schritt 7 zeigt, welches K passt. Bei langsamem Rechner links die **Stichprobe**
aktivieren (`df.sample(frac=0.1)`).

## Die Schritte

| Seite | Inhalt |
|---|---|
| 1 · Daten & Bibliotheken | CSV laden (Upload oder Pfad; Trennzeichen `;` und Dezimalkomma werden automatisch erkannt), Überblick, Bibliotheks-Versionen, optional Spalten-Beschreibungen |
| 2 · Aufbereitung 1 | je Spalte eine Aktion (behalten, entfernen, Zahl extrahieren, Datum → Jahr / Monat / Jahr + Monat, erste/letzte N Zeichen, Werte ersetzen, Top-N, **Werte begrenzen (min, max)** …) · neue Spalten: Werte zuordnen, Formel, Bedingung → 0/1, Kopie + Aktion |
| 3 · Analyse & Visualisierung | Auswahl der Spalten · Überblick, Countplot, Histogramm, Boxplot, Scatter, Heatmap, Korrelation mit einer Spalte, Pairplot, Anteil je Kategorie, Gruppen-Statistik, Jointplot, Dichte 2D, lmplot, Histogramm je Gruppe |
| 4 · Aufbereitung 2 – Clustering-Daten | **Features**, optional **bekannte Einteilung** (nur zum Vergleich, z. B. Private) und **Bezeichnung** der Zeilen (z. B. Name der Hochschule); fehlende Werte je Spalte; Dummies und Skalierung (Standard: StandardScaler, drop_first aus) – aus allen Zeilen gelernt, kein Split; Hinweise auf stark korrelierte Spalten und Spalten ohne Streuung. Als Vergleichsgruppe wird nur eine Spalte vorgeschlagen, die so heißt (Private, species, label, segment …); eine ID-Spalte (z. B. CustomerID = 1, 2, 3 …) wird als Bezeichnung vorgeschlagen, nicht als Feature – als Feature gewählt, warnt die App |
| 5 · Daten speichern | Daten nach Schritt 2, Clustering-Daten nach Schritt 4 (unskaliert oder skaliert) oder – nach dem Training – Daten mit **Cluster-Nummer, Cluster-Name und Abstand** zum Zentrum; international oder für Excel |
| 6 · Modell & Training | **K-Means**: K (n_clusters), init (k-means++/random), n_init, max_iter, algorithm, Seed; optional die Elbow- und Silhouette-Kurve für Schritt 7 – der passende scikit-learn-Code wird live angezeigt, die Kurven wachsen beim Training mit |
| 7 · Wahl von K | **Elbow-Methode** (Inertia je K) und **Silhouette** je K mit Einordnung und Vorschlag (*K = … übernehmen*), Davies-Bouldin, Calinski-Harabasz, erklärte Streuung (R²) und das **Silhouetten-Diagramm** des trainierten Modells |
| 8 · Cluster-Auswertung | Größe der Cluster, **Cluster benennen**, **Profile** (Mittelwerte in Originaleinheiten, Heatmap der Abweichung vom Durchschnitt in σ, Text je Cluster), **Trennkraft je Feature (R²)**, **2D-Karte (PCA)** mit Farbe = Cluster oder bekannte Einteilung, zwei Features gegeneinander, **Vergleich mit der bekannten Einteilung** (Kreuztabelle, Zuordnung Cluster → Gruppe, Treffer, naive Vorhersage, ARI, NMI, Confusion Matrix, classification_report – auch der Vergleich wie im Kurs), **typische Vertreter** je Cluster, Einzelfall |
| 9 · Modell speichern & laden | `models/<name>/model.skops` (sicher, Standard) oder `model.joblib` + `preprocessing.json` (Rezept, Einstellungen, Cluster-Namen, Kennzahlen, Zuordnung zur Vergleichsgruppe), ZIP-Download, ganze Pipeline als Python-Skript |
| 10 · Neue Daten zuordnen | neue CSV oder Formular → gleiche Aufbereitung → **nächstes Cluster-Zentrum** (`kmeans.predict`), Abstand, Hinweise, Verteilung im Vergleich zum Training, Vergleich mit der bekannten Einteilung (falls enthalten), CSV-Download und **Fazit** |

Zu jedem Schritt gibt es ein Feld **„Python-Code zu diesem Schritt“** – so lässt sich alles 1:1 in ein
Jupyter Notebook übernehmen; der Code rechnet exakt dasselbe wie die App (gleiche Cluster, gleiche Inertia, gleiche
Kurven). Einstellungen (inkl. neuer Spalten) lassen sich links unter **Konfiguration** als JSON speichern und für
neue Daten mit gleichen Spalten wieder laden. Beim Laden wird die Datei geprüft: unbrauchbare Einträge (falscher
Typ, außerhalb des erlaubten Bereichs, z. B. K = 1000) werden ignoriert und in der Meldung genannt. Konfigurationen
der Supervised- und der Deep-Learning-Version lassen sich laden: Schritt 2, fehlende Werte, Features und Skalierung
werden übernommen, Label und Modell-Einstellungen still übergangen.

## K-Means in dieser App

**Unsupervised:** K-Means bekommt keine Antwort. Es wählt K Startzentren, ordnet jede Zeile dem nächsten Zentrum zu
(euklidischer Abstand in den skalierten Features), setzt jedes Zentrum auf den Mittelwert seiner Zeilen und
wiederholt das, bis sich nichts mehr ändert. Das Ganze `n_init`-mal mit anderen Startzentren – es gewinnt die
Lösung mit der kleinsten **Inertia** (Summe der quadrierten Abstände zum eigenen Zentrum).

* **Kein Train/Test-Split.** Wie im Kurs (`kmeans.fit(df.drop('Private', axis=1))`) lernt K-Means aus allen Zeilen.
  Füllwerte, Kategorien und Scaler stammen deshalb ebenfalls aus allen Zeilen und gelten unverändert für neue
  Daten. Es gibt keine Antwort, die man vor dem Modell verstecken müsste – ein Datenleck wie beim Supervised
  Learning gibt es hier nicht. Ausnahme: Ein Feature aus derselben CSV-Spalte wie die Vergleichsgruppe würde den
  Vergleich in Schritt 8 schönen – davor warnt Schritt 4.
* **Skalierung ist entscheidend.** K-Means rechnet mit Abständen: Ohne Skalierung bestimmen Spalten mit großen
  Zahlen (Apps, F.Undergrad) fast allein die Cluster. Standard ist deshalb `StandardScaler` (Mittelwert 0,
  Standardabweichung 1 je Spalte). Im Kurs wurde nicht skaliert – für den direkten Vergleich in Schritt 4
  „keiner“ wählen.
* **Dummies ohne drop_first.** Für Abstände ist es besser, alle Kategorien zu behalten (symmetrisch); mit
  drop_first läge die weggelassene Kategorie näher an allen anderen.
* **n_init = 10** ausdrücklich: In scikit-learn ≥ 1.4 ist der Standard `n_init='auto'` – das ist bei k-means++ nur
  ein einziger Start.
* **Ausreißer und unmögliche Werte** verschieben die Zentren: in Schritt 2 mit *Werte begrenzen* korrigieren
  (im Kurs: Grad.Rate 118 → 100, hier mit dem Parameter `,100`).

### Wahl von K (Schritt 7)

| Kennzahl | Bedeutung | gut |
|---|---|---|
| Inertia (Elbow-Methode) | Summe der quadrierten Abstände zum eigenen Zentrum; sinkt mit jedem weiteren Cluster | der **Knick**: das K, nach dem die Inertia deutlich langsamer sinkt (die App sucht die Stelle, an der die Abnahme je K mindestens 2,5-mal kleiner wird) |
| Silhouette (−1 … 1) | wie viel näher jede Zeile am eigenen Cluster liegt als am nächsten anderen | größer; > 0.7 stark, > 0.5 vernünftig, > 0.25 schwach, darunter kaum Struktur |
| Davies-Bouldin | Verhältnis von Streuung im Cluster zu Abstand der Cluster | kleiner |
| Calinski-Harabasz | Streuung zwischen den Clustern / Streuung in den Clustern | größer |
| erklärte Streuung (R²) | 1 − Inertia / Gesamtstreuung | größer (steigt aber immer mit K) |

Zeigen Elbow und Silhouette dasselbe K, schlägt die App es vor; sonst beide. Bleibt die Silhouette für alle K
niedrig, macht die App keinen Silhouette-Vorschlag. Am Ende zählt, ob sich die Cluster in Schritt 8 sinnvoll
beschreiben lassen. Die Kurve wird ab 20 000 Zeilen auf einer festen Stichprobe berechnet, die Silhouette ab 5000
Zeilen (wie `silhouette_score(..., sample_size=5000)`); das Modell selbst nutzt immer alle Zeilen.

### Cluster beschreiben und vergleichen (Schritt 8)

* **Cluster-Nummern sind willkürlich.** K-Means weiß nicht, dass Cluster 1 „Yes“ heißen soll. Deshalb zeigt die
  App den Vergleich wie im Kurs (Cluster-Nummer = Gruppen-Nummer, im Kurs 22 %) und den Vergleich nach der
  **Zuordnung**: Jeder Cluster bekommt die Gruppe, die in ihm am häufigsten ist (im Kurs entspricht das 78 %).
  Dazu die naive Vorhersage (immer die häufigste Gruppe), **ARI** und **NMI** – beide unabhängig von den Nummern.
* **Profile:** Mittelwert jedes Features je Cluster in Originaleinheiten und die Abweichung vom Durchschnitt in
  Standardabweichungen (z) – daraus entstehen die Texte „über/unter dem Durchschnitt“ und gute Namen für die
  Cluster. **Trennkraft (R²)**: welcher Anteil der Streuung eines Features zwischen den Clustern liegt.
* **PCA nur für das Bild:** Die 2D-Karte projiziert die Daten auf die zwei Richtungen mit der größten Streuung;
  K-Means rechnet weiter mit allen Spalten.
* **Namen** der Cluster (z. B. „klein & privat“) werden mit dem Modell gespeichert und in Schritt 10 und im Export
  ausgegeben.

## Neue Spalten und Aktionen (Schritt 2)

| Art | Beispiel | Ergebnis |
|---|---|---|
| Werte zuordnen | aus `Private`: `Yes=1; No=0` | neue Spalte (1/0); nicht genannte Werte bleiben leer oder `sonst=…` |
| Formel | `Accept / Apps` | Zahl je Zeile; erlaubt sind `+ - * / ** // %`, Klammern, `abs log log10 sqrt exp round` |
| Bedingung → 0/1 | `Grad.Rate > 80` (Namen mit Punkt in \`Backticks\`) | 1 = erfüllt, 0 = nicht erfüllt, leer bei Lücken |
| Kopie + Aktion | `Grad.Rate` + „Werte begrenzen“ | Original bleibt, Ergebnis als zusätzliche Spalte |
| Aktion *Werte begrenzen* | Parameter `,100` · `0,100` · `10,` · mit Dezimalkomma `0,5;99,5` | Werte unter min / über max werden auf die Grenze gesetzt (`clip`) |

Formeln werden geprüft: nur Rechnen, Vergleiche und die genannten Funktionen – kein beliebiger Python-Code; Grenzen
für Länge, Verschachtelung und Exponenten verhindern, dass eine Formel das Programm blockiert. Neue Spalten gehören
zum gespeicherten Rezept und werden in Schritt 10 für neue Daten automatisch mitberechnet.

## Schritt 10: neue Daten zuordnen

* **Nichts neu fitten.** Neue Daten laufen durch das gespeicherte *Rezept* (`preprocessing.json`): gleiche
  Spalten-Aktionen und neue Spalten, Füllwerte, Dummy-Spalten in gleicher Reihenfolge, gleicher Scaler. Danach kommt
  jede Zeile zum **nächsten Zentrum** – die Zentren bleiben, wie sie sind.
* **Gleiche Rohspalten wie im Training** – die App zeigt die benötigten Spalten und bietet eine CSV-Vorlage an.
  Fehlende Spalten → Fehlermeldung; zusätzliche Spalten werden ignoriert.
* **Hinweise je Zeile:** Lücken ergänzt · unbekannte Kategorie · außerhalb Trainingsbereich · **weit weg von allen
  Clustern** (weiter vom nächsten Zentrum als 99 % der Trainingszeilen dieses Clusters – Ausreißer oder ein neuer
  Typ) · **zwischen zwei Clustern** (Abstand zum nächsten Zentrum ≥ 95 % des Abstands zum zweitnächsten).
* **Fazit:** Datenqualität, Verteilung auf die Cluster im Vergleich zum Training, Anteil der Ausreißer und knappen
  Zuordnungen; enthalten die neuen Daten die bekannte Einteilung, wird mit der Zuordnung Cluster → Gruppe **aus dem
  Training** verglichen (nicht neu bestimmt).
* Haben die neuen Daten schon Spalten mit dem Namen einer Ergebnisspalte (z. B. `Cluster`, `Hinweis` aus einem
  früheren Export), werden sie im Ergebnis ersetzt – die App nennt sie.

Ohne App:

```python
import pipeline as pl
kmeans, recipe = pl.load_bundle("models/mein_modell")
df_new, _ = pl.read_csv("neue_daten.csv")
result, report, comparison = pl.assign_new(kmeans, recipe, df_new)
fazit = pl.assess_assignment(recipe, result, report, comparison)
```

## Hinweise zu aktuellen Bibliotheksversionen

| älterer Code | Problem heute | im Template |
|---|---|---|
| `KMeans(n_clusters=2)` | scikit-learn ≥ 1.4: `n_init='auto'` = 1 Start bei k-means++ – Ergebnis hängt stärker vom Zufall ab | `KMeans(n_clusters=2, n_init=10, random_state=101)` |
| `df['Grad.Rate']['Cazenovia College'] = 100` | pandas: SettingWithCopyWarning (Chained Assignment), in pandas 3 wirkungslos | Aktion *Werte begrenzen* `,100` bzw. `df['Grad.Rate'].clip(upper=100)` |
| `sns.lmplot('Room.Board', 'Grad.Rate', data=df, …)` | seaborn ≥ 0.12: nur `data` darf ohne Namen übergeben werden | `sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, …)` |
| `sns.FacetGrid(df, hue='Private', size=6)` | `size` heißt seit seaborn 0.9 `height` | Histogramm je Gruppe (`hist` mit gemeinsamen bins) |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | beim Laden kann beliebiger Code aus der Datei ausgeführt werden | `.skops` als Standard; `.joblib` nur für vertrauenswürdige Dateien |
| `df.corr()` | pandas ≥ 2: Fehler bei Textspalten | `df.corr(numeric_only=True)` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False statt 0/1 | `dtype=float` |

## Dateien

```
app.py                  Streamlit-Oberfläche (nur UI)
pipeline.py             Rechenlogik – auch im Notebook nutzbar (import pipeline as pl)
plots.py                Diagramme (matplotlib/seaborn), geben Figures zurück
codegen.py              erzeugt den angezeigten bzw. exportierten Python-Code
i18n.py                 Sprachen: tx() übersetzt einen deutschen Text in die aktuelle Sprache
locales/                Übersetzungen {Deutsch: (Englisch, Polnisch)}, eine Datei je Modul
configs/                college_beispiel.json (Beispiel-Konfiguration College_Data wie im Kurs)
.streamlit/config.toml  Upload-Limit 1 GB, helles Theme
tests/                  pytest-Tests + synthetische Testdaten (college_like, mall_like …)
AUDIT.md                Bericht (polnisch): Aufbau, Prüfungen, Grenzen
README.md               Deutsch · README_EN.md Englisch · README_PL.md Polnisch
models/                 entsteht beim Speichern (Schritt 9)
exports/                Standardordner für „In Ordner speichern“ (Schritt 5)
```

Erweitern:
* neue Spalten-Aktion → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* neues Diagramm → Funktion in `plots.py` + Eintrag in `page_viz()` bzw. `page_profiles()` in `app.py`
* weiteres Clustering-Verfahren → `MODEL_NAMES`, `ClusterConfig`, `make_kmeans()`/`train_kmeans()` in
  `pipeline.py`, `page_train()` in `app.py`, `model_code()` in `codegen.py`
* neuer oder geänderter Text → auf Deutsch in `tx("…")` schreiben (Konstanten: `N_("…")`) und die englische und
  polnische Übersetzung in der passenden Datei in `locales/` ergänzen; `pytest tests/test_i18n.py` meldet fehlende
  Übersetzungen.

## Tests

```bash
pip install pytest
pytest -q                                  # 268 Tests, ca. 3 Minuten
python tests/synthetic.py                  # synthetische College-Daten (777 Zeilen, keine echten Daten)
```

| Datei | prüft |
|---|---|
| `tests/test_clustering.py` | K-Means wie im Kurs (= scikit-learn direkt), Reproduzierbarkeit, Elbow und Silhouette (klare Gruppen werden gefunden, kein Vorschlag ohne Struktur, Stichproben bei großen Daten), Einordnung (kleine/dominierende Cluster, nicht konvergiert, doppelte Zeilen), Profile in Originaleinheiten, Trennkraft = erklärte Streuung, PCA, Vergleich mit bekannten Gruppen, Ausreißer und knappe Zuordnungen in neuen Daten |
| `tests/test_codegen_parity.py` | der angezeigte Python-Code rechnet **exakt** dasselbe wie die App: X, Cluster, Inertia, Elbow-/Silhouette-Kurve, Profile, Vergleich, Export – auch in Grenzfällen (Tausendertrennzeichen, Datum als Zahl, inf, Gruppen-Füllwerte, Werte begrenzen, neue Spalten) |
| `tests/test_pipeline.py` | ganzer Ablauf mit LendingClub-ähnlichen Daten, Speichern/Laden (.skops/.joblib, ZIP), neue Daten (unbekannte Kategorien, Ausreißer, fehlende Spalten), exportiertes Skript läuft |
| `tests/test_new_features.py` | Jahr + Monat, neue Spalten, *Werte begrenzen*, neue Spalten für neue Daten, Daten speichern, Fazit |
| `tests/test_audit_regressions.py` … `test_audit4_regressions.py` | Regressionstests aus den vier Audits des Grundgerüsts, übertragen auf Clustering (Zahlen, Datum, CSV-Erkennung, Formel-Grenzen, Konfigurationen, ZIP-Schutz, kein Code aus Namen im angezeigten Code, Sprachwechsel, sicheres Laden, nur K-Means-Modelle) |
| `tests/test_audit5_regressions.py` | Audit 5 (Clustering-App): Export-Code = App auch nach entfernten Zeilen/Spalten, keine ungefragte Vergleichsgruppe, ID-Spalten als Bezeichnung, Text „Wie im Kurs“ mit den echten Gruppen, kein Markdown aus fremden Modellen, beschädigte `preprocessing.json` wird abgelehnt, leere Cluster, Hinweis auf geänderte Einstellungen |
| `tests/test_app.py` | Klick-Test der Oberfläche (alle 10 Schritte wie im Kurs; Semikolon-CSV, geänderte Einstellungen, unbrauchbare Werte, Schutz vor Überschreiben; neue Spalten, Fehler beim Zeichnen, Export) |
| `tests/test_charts.py` | Diagramme von Schritt 3 und die Cluster-Diagramme (Elbow/Silhouette, Silhouetten-Diagramm, Größen, Profile, PCA, Kreuztabelle) |
| `tests/test_i18n.py` | Sprachen: jeder Text übersetzt mit denselben Platzhaltern, Excel-CSV je Sprache (App = Code), erzeugter Code rechnet in jeder Sprache gleich, die ganze App auf Englisch und Polnisch ohne deutsche Reste |

## Hinweise

* Sehr große CSVs besser über „Pfad auf diesem Rechner“ laden als hochladen.
* Während des Trainings nichts anklicken – jede Eingabe startet das Skript neu und bricht das Training ab.
* K-Means ist schnell, aber bei sehr vielen Zeilen und `n_init = 10` dauert es: 400 000 Zeilen × 80 Spalten
  brauchen je K einige Sekunden bis etwa eine Minute (gemessen mit 2 Prozessorkernen – je undeutlicher die Gruppen,
  desto mehr Iterationen). Die Elbow-Kurve nutzt deshalb ab 20 000 Zeilen eine Stichprobe; Tipp: Stichprobe in der
  Seitenleiste oder kleineres `n_init`.
* **Nur lokal verwenden.** Das Pfad-Feld liest jede Datei, auf die der angemeldete Benutzer Zugriff hat, und
  `models/` ist für alle Nutzer der App gleich. Die App daher nicht ohne Anmeldung ins Netz stellen.
  Modelle nur aus vertrauenswürdigen Quellen laden (ein ZIP darf entpackt höchstens 2 GB groß sein). `.skops` wird
  ohne Code-Ausführung geladen; `.joblib` ist pickle und wird aus einem hochgeladenen ZIP nur nach ausdrücklicher
  Bestätigung geladen. Geladen werden nur K-Means-Modelle dieser App – Modelle der Supervised- oder
  Deep-Learning-Version lehnt die App mit einer klaren Meldung ab, ebenso ein unvollständiges oder beschädigtes
  `preprocessing.json` und ein Rezept, das nicht zum Modell passt.
* **K-Means hat Grenzen:** Es findet eher runde, ähnlich große Gruppen und ordnet jede Zeile einem Cluster zu – auch
  Ausreißer. Bei länglichen oder verschachtelten Gruppen oder sehr unterschiedlich großen Clustern (siehe
  Silhouetten-Diagramm) sind andere Verfahren (z. B. DBSCAN, hierarchisches Clustering) besser geeignet.
