# DATA-Analysis-Clustering (Streamlit + scikit-learn: K-Means, Isolation Forest, LOF, DBSCAN)

Streamlit-App für tabellarische CSV-Daten: laden, aufbereiten, visualisieren und mit **K-Means** in Gruppen
(Cluster) einteilen – **Unsupervised Learning**: Es gibt kein Label, K-Means sucht selbst nach Zeilen, die sich
ähneln. Danach beschreibt die App die Cluster (Profile, 2D-Karte), vergleicht sie – wenn vorhanden – mit einer
bekannten Einteilung und ordnet neue Daten den Clustern zu. Zu jedem Schritt zeigt die App den passenden
Python-Code.

**Anomalieerkennung:** Statt Gruppen zu suchen, findet die App mit **Isolation Forest**, **Local Outlier Factor
(LOF)** oder **DBSCAN** (Rauschen = Anomalie) die *ungewöhnlichsten Zeilen* – mit Score, Schwelle, den Features, die
eine Zeile auffällig machen, und dem Vergleich mit einer bekannten Einteilung (z. B. *Class* bei Betrugsdaten). Neue
Daten werden mit derselben Schwelle bewertet. Für Kryptodaten baut `tools/bybit_features.py` aus öffentlichen
Bybit-Marktdaten eine fertige CSV (siehe [Bybit → CSV](#bybit--csv-toolsbybit_featurespy)).

Vorbild ist das K-Means-Projekt aus dem Kurs: **College_Data** – K-Means mit K = 2 auf allen Zahlenspalten; die
Spalte *Private* (Yes/No) wird dabei ignoriert und erst danach zum Vergleich benutzt. Für diesen Datensatz gibt es
eine fertige Beispiel-Konfiguration.

Die App ist aus der Supervised-Version (DATA-Analysis-SupervisedLearning) entstanden: Schritte 1–3 (Laden,
Spalten-Aktionen, neue Spalten, Diagramme), Rezept (`preprocessing.json`), Speichern/Laden, Konfigurationen und
Sprachen sind gleich geblieben; neu sind Schritt 4 ohne Label und ohne Split, K-Means in Schritt 6, die Wahl von K
in Schritt 7, die Cluster-Auswertung in Schritt 8 und die Zuordnung neuer Daten in Schritt 10 – später kamen die
Anomalie-Modelle dazu (Schritte 6–10 passen sich dem gewählten Modell an).

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
4. **Schritt 4:** *Daten fürs Modell vorbereiten* (17 Features, Private nur zum Vergleich, StandardScaler)
5. **Schritt 6:** *Training starten* (K = 2) → Schritt 7 (Wahl von K), 8 (Profile, Vergleich mit Private), 9, 10

Für eigene Daten: in Schritt 4 Features wählen, optional eine bekannte Einteilung zum Vergleich und eine Spalte mit
Namen; in Schritt 6 K wählen – Schritt 7 zeigt, welches K passt. Bei langsamem Rechner links die **Stichprobe**
aktivieren (`df.sample(frac=0.1)`).

## Schnellstart – Anomalien in Kryptodaten (Bybit)

```bash
python tools/bybit_features.py --top 20 --interval 1h --days 120 --split 2026-10-01
```

Das Skript holt öffentliche Marktdaten (ohne API-Schlüssel) und schreibt `bybit_series_linear_60_train.csv` (vor dem
1. Oktober), `bybit_series_linear_60_new.csv` (ab dann) und `bybit_series_linear_60_columns.csv`
(Spaltenbeschreibungen).

1. **Schritt 1:** `…_train.csv` laden (optional `…_columns.csv` als Spalten-Beschreibungen)
2. **Schritt 2:** Die App erkennt die Bybit-Daten → *Beispiel-Konfiguration laden* → *Übernehmen*
3. **Schritt 4:** *Daten fürs Modell vorbereiten* (11 Kennzahlen, `row` = Symbol + Zeit als Bezeichnung)
4. **Schritt 6:** Isolation Forest mit 1 % auffälligen Zeilen ist eingestellt → *Training starten*
5. **Schritt 7:** Passt die Schwelle (Knick der Score-Kurve)? · **Schritt 8:** die auffälligsten Stunden und was sie
   auffällig macht (z. B. `turnover_z +9.1 σ · ret_z −7.4 σ`)
6. **Schritt 9:** speichern · **Schritt 10:** `…_new.csv` mit demselben Modell und derselben Schwelle bewerten

## Die Schritte

| Seite | Inhalt |
|---|---|
| 1 · Daten & Bibliotheken | CSV laden (Upload oder Pfad; Trennzeichen `;` und Dezimalkomma werden automatisch erkannt), Überblick, Bibliotheks-Versionen, optional Spalten-Beschreibungen |
| 2 · Aufbereitung 1 | je Spalte eine Aktion (behalten, entfernen, Zahl extrahieren, Datum → Jahr / Monat / Jahr + Monat, erste/letzte N Zeichen, Werte ersetzen, Top-N, **Werte begrenzen (min, max)** …) · neue Spalten: Werte zuordnen, Formel, Bedingung → 0/1, Kopie + Aktion |
| 3 · Analyse & Visualisierung | Auswahl der Spalten · Überblick, Countplot, Histogramm, Boxplot, Scatter, Heatmap, Korrelation mit einer Spalte, Pairplot, Anteil je Kategorie, Gruppen-Statistik, Jointplot, Dichte 2D, lmplot, Histogramm je Gruppe |
| 4 · Aufbereitung 2 – Clustering-Daten | **Features**, optional **bekannte Einteilung** (nur zum Vergleich, z. B. Private) und **Bezeichnung** der Zeilen (z. B. Name der Hochschule); fehlende Werte je Spalte; Dummies und Skalierung (Standard: StandardScaler, drop_first aus) – aus allen Zeilen gelernt, kein Split; Hinweise auf stark korrelierte Spalten und Spalten ohne Streuung. Als Vergleichsgruppe wird nur eine Spalte vorgeschlagen, die so heißt (Private, species, label, segment …); eine ID-Spalte (z. B. CustomerID = 1, 2, 3 …) wird als Bezeichnung vorgeschlagen, nicht als Feature – als Feature gewählt, warnt die App |
| 5 · Daten speichern | Daten nach Schritt 2, Clustering-Daten nach Schritt 4 (unskaliert oder skaliert) oder – nach dem Training – Daten mit **Cluster-Nummer, Cluster-Name und Abstand** zum Zentrum (K-Means) bzw. mit **Anomalie (1 = auffällig) und Anomalie-Score** (bei DBSCAN auch der Cluster); international oder für Excel |
| 6 · Modell & Training | **Methode wählen** · **K-Means**: K (n_clusters), init (k-means++/random), n_init, max_iter, algorithm, Seed; optional die Elbow- und Silhouette-Kurve für Schritt 7 (die Kurven wachsen beim Training mit) · **Isolation Forest**: n_estimators, max_samples, Seed, contamination (auto oder fester Anteil) · **LOF**: n_neighbors, metric, contamination · **DBSCAN**: eps (Startwert aus der k-Distanz), min_samples, metric, Rauschanteil je eps für Schritt 7 – der passende scikit-learn-Code wird live angezeigt |
| 7 · Wahl von K bzw. Schwelle | K-Means: **Elbow-Methode** (Inertia je K) und **Silhouette** je K mit Einordnung und Vorschlag (*K = … übernehmen*), Davies-Bouldin, Calinski-Harabasz, erklärte Streuung (R²) und das **Silhouetten-Diagramm** · Isolation Forest/LOF: **Verteilung des Scores** mit Schwelle, **Score-Kurve** der auffälligsten Zeilen mit Knick (*contamination = … übernehmen*), Schwelle je Anteil · DBSCAN: **k-Distanz** mit Knick (*eps = … übernehmen*), Rauschanteil und Zahl der Cluster je eps |
| 8 · Auswertung | K-Means: Größe der Cluster, **Cluster benennen**, **Profile** (Mittelwerte in Originaleinheiten, Heatmap der Abweichung vom Durchschnitt in σ, Text je Cluster), **Trennkraft je Feature (R²)**, **2D-Karte (PCA)** mit Farbe = Cluster oder bekannte Einteilung, zwei Features gegeneinander, **Vergleich mit der bekannten Einteilung** (Kreuztabelle, Zuordnung Cluster → Gruppe, Treffer, naive Vorhersage, ARI, NMI, Confusion Matrix, classification_report – auch der Vergleich wie im Kurs), **typische Vertreter** je Cluster, Einzelfall · Anomalie-Modelle: **die auffälligsten Zeilen** mit Score und den Features, die am stärksten vom Durchschnitt abweichen (z), **Profil** auffällig gegen normal, **PCA-Bild** (Anomalien immer im Bild), DBSCAN-Cluster, **Vergleich mit der bekannten Einteilung** (Anteil auffällig je Gruppe; bei zwei Gruppen Precision, Recall, ROC-AUC, Average Precision), Einzelfall |
| 9 · Modell speichern & laden | `models/<name>/model.skops` (sicher, Standard) oder `model.joblib` + `preprocessing.json` (Rezept, Einstellungen, Cluster-Namen bzw. Schwelle und Anteil auffälliger Zeilen, Kennzahlen, Zuordnung zur Vergleichsgruppe), ZIP-Download, ganze Pipeline als Python-Skript |
| 10 · Neue Daten zuordnen / bewerten | neue CSV oder Formular → gleiche Aufbereitung → K-Means: **nächstes Cluster-Zentrum** (`kmeans.predict`), Abstand, Hinweise, Verteilung im Vergleich zum Training · Anomalie-Modelle: **Anomalie und Score mit der Schwelle aus dem Training** (DBSCAN: Cluster des nächsten Kernpunkts oder Rauschen), Anteil auffälliger Zeilen im Vergleich zum Training · Vergleich mit der bekannten Einteilung (falls enthalten), CSV-Download und **Fazit** |

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

## Anomalieerkennung: Isolation Forest, LOF, DBSCAN

Gleiche Schritte 1–5 wie bei K-Means, in Schritt 6 eine andere **Methode**. Alle drei sind unsupervised: Sie lernen,
wie die Zeilen *normalerweise* aussehen, und geben jeder Zeile einen **Anomalie-Score** – höher = auffälliger.
Auffällig (`Anomalie = 1`) ist ein Score über der **Schwelle**.

| Methode | Idee | Score | Schwelle | gut für | Vorsicht |
|---|---|---|---|---|---|
| **Isolation Forest** | Zufallsbäume schneiden die Daten an zufälligen Stellen – ungewöhnliche Zeilen sind nach wenigen Schnitten allein | `−score_samples`, 0 … 1 (um 0.5 normal) | contamination = auto → 0.5, sonst das Quantil für den gewählten Anteil | der Standard für Tabellen; schnell, auch bei vielen Zeilen und Features; braucht keine Skalierung | findet eher globale als lokale Ausreißer |
| **Local Outlier Factor (LOF)** | Dichte um eine Zeile im Vergleich zur Dichte um ihre `n_neighbors` Nachbarn | LOF ≈ 1 normal, deutlich über 1 auffällig | auto → 1.5, sonst das Quantil | lokale Ausreißer neben dichten Gruppen | bilden die Ausreißer selbst eine kleine Gruppe, muss `n_neighbors` größer sein als diese Gruppe; langsamer bei sehr vielen Zeilen; skalieren |
| **DBSCAN** | Kernpunkt = mindestens `min_samples` Zeilen im Radius `eps`; was zu keinem Cluster gehört, ist **Rauschen** | Abstand zum nächsten Kernpunkt in Vielfachen von eps | 1 (= eps): Rauschen ⇔ Score über 1 | Cluster beliebiger Form und Ausreißer in einem Schritt; kein K nötig | `eps` hängt von Skalierung und Zahl der Features ab – Schritt 7 (k-Distanz) zeigt, welches passt |

* **contamination ist eine Entscheidung, keine Wahrheit:** Wie viele Zeilen sollen gemeldet werden? Lieber wenige
  klare Anomalien (0.5–2 %) als eine lange Liste. Schritt 7 zeigt den **Knick der Score-Kurve** – dahinter
  unterscheiden sich die Zeilen kaum noch – und bietet ihn als Vorschlag an.
* **Was macht eine Zeile auffällig?** Schritt 8 zeigt je Zeile die Features mit der größten Abweichung vom
  Durchschnitt (z, in Standardabweichungen) und das Profil *auffällig gegen normal* – ein Blick in die Daten, nicht
  ins Modell.
* **Vergleich mit einer bekannten Einteilung** (z. B. *Class* bei Betrugsdaten): Das Modell hat sie nie gesehen. Bei
  zwei Gruppen gilt die seltenere als „positiv“: **Precision** (Anteil echter Fälle unter den Meldungen), **Recall**
  (Anteil der gefundenen echten Fälle); **ROC-AUC** und **Average Precision** bewerten den Score unabhängig von der
  Schwelle (Zufall: 0.5 bzw. der Anteil der positiven Gruppe).
* **Nicht begrenzen, was überwacht werden soll.** *Werte begrenzen* in Schritt 2 schneidet genau die Extremwerte ab,
  die eine Anomalie ausmachen – für die Anomalieerkennung nur unmögliche Werte korrigieren.
* **Skalierung:** LOF und DBSCAN rechnen mit Abständen → StandardScaler in Schritt 4. Isolation Forest schneidet je
  Feature und braucht keine Skalierung (sie schadet aber nicht).
* **Zeitreihen ohne Blick in die Zukunft:** Kennzahlen je Zeile nur aus der Vergangenheit berechnen (gleitende
  Fenster, z-Werte gegen *frühere* Werte – wie im Bybit-Skript), auf einem früheren Zeitraum trainieren und einen
  späteren in Schritt 10 bewerten. Häufen sich die Anomalien in einem Zeitraum, ist das eher ein neues Regime als
  lauter Einzelfälle → mit neueren Daten neu trainieren.
* **Neue Daten (Schritt 10):** gleiche Aufbereitung und gleiche Schwelle wie im Training. LOF wird mit
  `novelty=True` trainiert (sonst könnte es keine neuen Daten bewerten); bei DBSCAN bekommt eine neue Zeile den
  Cluster des nächsten Kernpunkts, wenn er höchstens eps entfernt ist – sonst −1 (Rauschen = Anomalie).
* Wie bei K-Means rechnet der angezeigte Python-Code exakt dasselbe wie die App (Score, Anomalien, Schwelle, Kurven
  für Schritt 7, Profile, Export).

## Bybit → CSV (tools/bybit_features.py)

Baut aus öffentlichen Marktdaten der Bybit-API v5 eine CSV mit Kennzahlen für die Anomalieerkennung – ohne
API-Schlüssel, nur mit der Standardbibliothek und pandas/numpy (beides braucht die App ohnehin). Die App erkennt die
Datei an ihren Spalten und bietet in Schritt 2 die passende Beispiel-Konfiguration an.

```bash
python tools/bybit_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 120
python tools/bybit_features.py --top 20 --interval 4h --days 365 --split 2026-09-01
python tools/bybit_features.py --mode cross --top 100 --interval 1h --days 30
python tools/bybit_features.py --help
```

**Zeitreihe** (Standard, `--mode series`): eine Zeile je Symbol und abgeschlossener Kerze – „Welche Stunden/Tage
waren ungewöhnlich?“

| Spalte | Bedeutung |
|---|---|
| `row`, `symbol`, `time` | Bezeichnung (Symbol + Kerzenbeginn in UTC), Symbol, Kerzenbeginn – keine Features |
| `ret_pct`, `ret_z` | Rendite der Kerze in %; als z-Wert gegen die vorigen Kerzen (Log-Renditen) |
| `range_pct`, `range_z` | Spanne Hoch–Tief in % des Eröffnungskurses; als z-Wert |
| `turnover_z` | gehandelter Wert in USD als z-Wert – Volumen-Schock |
| `vola_regime` | log2(Volatilität im letzten Fenster / Volatilität davor) – 0 normal, +1 doppelt so hoch |
| `er` | Efficiency Ratio im Fenster: Nettobewegung / Summe aller Bewegungen – 0 seitwärts, 1 gerader Trend |
| `cs_spread_pct` | geschätzter Bid-Ask-Spread in % (Corwin-Schultz aus Hoch/Tief), Mittel im Fenster – Liquiditätsstress |
| `corr_btc`, `excess_z` | Korrelation mit dem Benchmark (Standard BTC) im Fenster; Rendite minus Benchmark-Rendite als z-Wert – eigene Bewegung eines Coins |
| `funding_bp`, `funding_z` | zum Kerzenschluss bekannte Funding-Rate in Basispunkten; als z-Wert (nur Perpetuals) |
| `oi_chg_pct`, `oi_z` | Änderung des Open Interest in %; als z-Wert (nur Perpetuals) |

`ret_pct`, `range_pct` und `oi_chg_pct` sind zum Lesen da; die Beispiel-Konfiguration nimmt die übrigen 11 Spalten
als Features (Isolation Forest, 1 % auffällig). Tipp: In Schritt 4 `symbol` als bekannte Einteilung wählen – dann
zeigt Schritt 8, welche Symbole die meisten auffälligen Kerzen haben.

**Querschnitt** (`--mode cross`): eine Zeile je Symbol über den ganzen Zeitraum – „Welche Tokens verhalten sich anders
als der Rest?“ Spalten: `ret_window_pct`, `vola_ann_pct` (annualisiert), `er`, `cs_spread_pct`, `max_dd_pct`
(größter Rückgang), `corr_btc`, `beta_btc`, `vola_trend` (Volatilität zweite gegen erste Hälfte), `turnover_log10`
(mittlerer Tagesumsatz), `ret_skew`, `ret_kurt` (dicke Ränder), `funding_mean_bp`, `oi_chg_pct` – die
Beispiel-Konfiguration meldet 5 % der Tokens.

* **Kein Blick in die Zukunft:** Jede Kennzahl einer Kerze nutzt nur Daten bis zu ihrem Schluss – z-Werte vergleichen
  mit den *vorigen* `--baseline` Kerzen (Standard 30 Tage), Fenster (`--window`, Standard 1 Tag) enden bei der Kerze,
  Funding und Open Interest sind die zum Kerzenschluss bekannten Werte, die laufende Kerze fehlt. Den Vorlauf für die
  z-Werte holt das Skript zusätzlich vor dem Start. Ein Test prüft das: Daten abschneiden → frühere Zeilen bleiben
  bitgenau gleich.
* `--split DATUM` schreibt `<name>_train.csv` (davor) und `<name>_new.csv` (ab dann): auf dem ersten Teil trainieren,
  den zweiten in Schritt 10 bewerten – wie im Live-Betrieb.
* `<name>_columns.csv` beschreibt die Spalten – in Schritt 1 unter *Optional: Spalten-Beschreibungen laden*.
* Symbole: `--symbols` oder `--top N` (größter 24h-Umsatz, ohne Stablecoins und Futures mit Laufzeit); `--category`
  linear (USDT-Perpetuals, Standard), inverse oder spot (ohne Funding und Open Interest); `--interval` 1 3 5 15 30 60
  120 240 360 720 D W bzw. 15m, 1h, 4h, 1d, 1w; außerdem `--start`/`--end`, `--no-funding`, `--no-oi`,
  `--benchmark`, `--keep-ohlcv`.
* **Grenzen der API:** höchstens 600 Anfragen je 5 Sekunden und IP – das Skript wartet zwischen den Anfragen
  (`--pause`) und wiederholt bei Überlastung. Antwortet Bybit mit **HTTP 403**, ist das IP-Limit überschritten oder
  der Zugang aus diesem Netz bzw. dieser Region gesperrt – mindestens 10 Minuten warten. `--cache ORDNER` (z. B.
  `bybit_cache`, steht in `.gitignore`) speichert die Antworten; mit festem `--start`/`--end` lässt sich ein Lauf dann
  mit `--offline` ohne Netz wiederholen.
* Hat ein Symbol zu wenig Vorlauf (erst kürzlich gelistet), fallen seine ersten Zeilen weg; eine Spalte mit mehr als
  20 % Lücken (z. B. Open Interest ohne Historie) lässt das Skript weg und sagt es.

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

Anomalie-Modelle (Isolation Forest, LOF, DBSCAN) bewerten statt zuzuordnen – mit der Schwelle aus dem Training:

```python
model, recipe = pl.load_bundle("models/mein_anomalie_modell")
result, report, comparison = pl.score_new(model, recipe, df_new)   # Anomalie, Anomalie-Score, Hinweis
fazit = pl.assess_scoring(recipe, result, report, comparison)
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
| `LocalOutlierFactor().fit_predict(X)`, später `lof.predict(neu)` | ohne `novelty=True` gibt es kein `predict` für neue Daten (mit `novelty=True` kein `fit_predict`) | `LocalOutlierFactor(novelty=True)`; Score der Trainingszeilen aus `negative_outlier_factor_` |
| `IsolationForest(contamination=0.1, behaviour='new')` aus Tutorials | `behaviour` gibt es seit scikit-learn 0.24 nicht mehr; 10 % „Anomalien“ sind fast immer zu viele | `contamination='auto'` (Schwelle 0.5) oder ein bewusst kleiner Anteil – Schritt 7 zeigt den Knick |

## Dateien

```
app.py                  Streamlit-Oberfläche (nur UI)
pipeline.py             Rechenlogik – auch im Notebook nutzbar (import pipeline as pl)
plots.py                Diagramme (matplotlib/seaborn), geben Figures zurück
codegen.py              erzeugt den angezeigten bzw. exportierten Python-Code
i18n.py                 Sprachen: tx() übersetzt einen deutschen Text in die aktuelle Sprache
locales/                Übersetzungen {Deutsch: (Englisch, Polnisch)}, eine Datei je Modul
configs/                Beispiel-Konfigurationen: College_Data wie im Kurs, Bybit-Zeitreihe, Bybit-Querschnitt
.streamlit/config.toml  Upload-Limit 1 GB, helles Theme
tools/bybit_features.py Bybit-Marktdaten → CSV mit Kennzahlen für die Anomalieerkennung
tests/                  pytest-Tests + synthetische Testdaten (college_like, mall_like, anomaly_like, FakeBybit …)
AUDIT.md                Bericht (polnisch): Aufbau, Prüfungen, Grenzen
README.md               Deutsch · README_EN.md Englisch · README_PL.md Polnisch
models/                 entsteht beim Speichern (Schritt 9)
exports/                Standardordner für „In Ordner speichern“ (Schritt 5)
```

Erweitern:
* neue Spalten-Aktion → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* neues Diagramm → Funktion in `plots.py` + Eintrag in `page_viz()` bzw. `page_profiles()` in `app.py`
* weiteres Verfahren → in `pipeline.py` `MODEL_NAMES`, `MODEL_CLASSES`, `MODEL_VARS`, `MODEL_WHY` und die Felder in
  `ClusterConfig`. Ein Anomalie-Verfahren zusätzlich in `ANOMALY_MODELS`, `anomaly_params()`, `anomaly_scores()`
  (Score: höher = auffälliger) und `anomaly_threshold()` – Training, Schritte 7–10 und Speichern laufen dann über
  `train_anomaly()`, `anomaly_curve()` und `score_new()`; neue Typen fürs sichere Laden in `SKOPS_TRUSTED`. In
  `app.py` die Parameter in `anomaly_params_ui()` und die Erklärung in `HOW_ANOMALY`, in `codegen.py`
  `anomaly_model_code()` und `predict_code()`; `tests/test_anomaly.py` prüft, dass der Code dasselbe rechnet wie die
  App. Ein Clustering-Verfahren wie K-Means: `make_kmeans()`/`train_kmeans()` und `model_code()` als Vorlage.
* neuer oder geänderter Text → auf Deutsch in `tx("…")` schreiben (Konstanten: `N_("…")`) und die englische und
  polnische Übersetzung in der passenden Datei in `locales/` ergänzen; `pytest tests/test_i18n.py` meldet fehlende
  Übersetzungen.

## Tests

```bash
pip install pytest
pytest -q                                  # 349 Tests, ca. 5 Minuten
python tests/synthetic.py                  # synthetische College-Daten (777 Zeilen, keine echten Daten)
```

| Datei | prüft |
|---|---|
| `tests/test_clustering.py` | K-Means wie im Kurs (= scikit-learn direkt), Reproduzierbarkeit, Elbow und Silhouette (klare Gruppen werden gefunden, kein Vorschlag ohne Struktur, Stichproben bei großen Daten), Einordnung (kleine/dominierende Cluster, nicht konvergiert, doppelte Zeilen), Profile in Originaleinheiten, Trennkraft = erklärte Streuung, PCA, Vergleich mit bekannten Gruppen, Ausreißer und knappe Zuordnungen in neuen Daten |
| `tests/test_codegen_parity.py` | der angezeigte Python-Code rechnet **exakt** dasselbe wie die App: X, Cluster, Inertia, Elbow-/Silhouette-Kurve, Profile, Vergleich, Export – auch in Grenzfällen (Tausendertrennzeichen, Datum als Zahl, inf, Gruppen-Füllwerte, Werte begrenzen, neue Spalten) |
| `tests/test_anomaly.py` | Isolation Forest, LOF, DBSCAN: Score ⇔ Schwelle, contamination, novelty, Rauschen = Anomalie, eps zu klein, Kurven und Vorschläge für Schritt 7, auffälligste Zeilen, Profil, PCA, Speichern/Laden (.skops/.joblib), neue Daten, kein Vertauschen mit K-Means, beschädigte Rezepte – und der angezeigte Code rechnet **exakt** dasselbe (Score, Anomalien, Schwelle, k-Distanz, eps-Raster, Profile, Export, Schritt 10); das exportierte Skript läuft; Diagramme in Grenzfällen |
| `tests/test_bybit_features.py` | Bybit-Skript mit simulierter API (kein Netz): Blättern nach den Grenzen der API, Wiederholen/403/Cache, kein Blick in die Zukunft, laufende Kerze fehlt, Kennzahlen nachgerechnet, Querschnitt, Spot, unbekannte Symbole, Beispiel-Konfiguration passt zur CSV, der eingebaute Kurssturz ist die auffälligste Zeile |
| `tests/test_pipeline.py` | ganzer Ablauf mit LendingClub-ähnlichen Daten, Speichern/Laden (.skops/.joblib, ZIP), neue Daten (unbekannte Kategorien, Ausreißer, fehlende Spalten), exportiertes Skript läuft |
| `tests/test_new_features.py` | Jahr + Monat, neue Spalten, *Werte begrenzen*, neue Spalten für neue Daten, Daten speichern, Fazit |
| `tests/test_audit_regressions.py` … `test_audit4_regressions.py` | Regressionstests aus den vier Audits des Grundgerüsts, übertragen auf Clustering (Zahlen, Datum, CSV-Erkennung, Formel-Grenzen, Konfigurationen, ZIP-Schutz, kein Code aus Namen im angezeigten Code, Sprachwechsel, sicheres Laden, nur K-Means-Modelle) |
| `tests/test_audit5_regressions.py` | Audit 5 (Clustering-App): Export-Code = App auch nach entfernten Zeilen/Spalten, keine ungefragte Vergleichsgruppe, ID-Spalten als Bezeichnung, Text „Wie im Kurs“ mit den echten Gruppen, kein Markdown aus fremden Modellen, beschädigte `preprocessing.json` wird abgelehnt, leere Cluster, Hinweis auf geänderte Einstellungen |
| `tests/test_app.py` | Klick-Test der Oberfläche (alle 10 Schritte wie im Kurs; Semikolon-CSV, geänderte Einstellungen, unbrauchbare Werte, Schutz vor Überschreiben; neue Spalten, Fehler beim Zeichnen, Export) |
| `tests/test_app_anomaly.py` | Klick-Test der drei Anomalie-Modelle durch die Schritte 4–10 und der Bybit-CSV mit ihrer Beispiel-Konfiguration bis zur Bewertung neuer Daten |
| `tests/test_charts.py` | Diagramme von Schritt 3 und die Cluster-Diagramme (Elbow/Silhouette, Silhouetten-Diagramm, Größen, Profile, PCA, Kreuztabelle) |
| `tests/test_i18n.py` | Sprachen: jeder Text übersetzt mit denselben Platzhaltern, Excel-CSV je Sprache (App = Code), erzeugter Code rechnet in jeder Sprache gleich, die ganze App auf Englisch und Polnisch ohne deutsche Reste – auch mit den Anomalie-Modellen |

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
  Bestätigung geladen. Geladen werden nur Modelle dieser App (K-Means, Isolation Forest, LOF, DBSCAN) – Modelle der Supervised- oder
  Deep-Learning-Version lehnt die App mit einer klaren Meldung ab, ebenso ein unvollständiges oder beschädigtes
  `preprocessing.json` und ein Rezept, das nicht zum Modell passt.
* **K-Means hat Grenzen:** Es findet eher runde, ähnlich große Gruppen und ordnet jede Zeile einem Cluster zu – auch
  Ausreißer. Bei länglichen oder verschachtelten Gruppen oder sehr unterschiedlich großen Clustern (siehe
  Silhouetten-Diagramm) passen **DBSCAN** (in Schritt 6 wählbar) oder hierarchisches Clustering besser; Ausreißer
  findet man gezielt mit Isolation Forest oder LOF.
* LOF und DBSCAN vergleichen jede Zeile mit ihren Nachbarn – bei sehr vielen Zeilen (ab etwa 50 000) dauert das und
  braucht Speicher; Isolation Forest bleibt schnell. Die k-Distanz für Schritt 7 nutzt ab 20 000 Zeilen eine
  Stichprobe, das eps-Raster gibt es nur bis 20 000 Zeilen.
