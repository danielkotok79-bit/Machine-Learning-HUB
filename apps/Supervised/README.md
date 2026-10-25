# DATA-Analysis-SupervisedLearning (Streamlit + scikit-learn)

Streamlit-App für tabellarische CSV-Daten: laden, aufbereiten, visualisieren, ein Supervised-Learning-Modell
(scikit-learn) wählen und trainieren, bewerten und auf neue Daten anwenden – für beliebige Datensätze (binär,
Mehrklassen, Regression). Zur Wahl stehen sechs Modelle, sortiert nach ihrer Rangliste:
**Gradient Boosting (HistGradientBoosting), Decision Trees + Random Forests, Support Vector Machines, Logistische
Regression, K-Nearest Neighbors, Lineare Regression** – siehe [Die sechs Modelle](#die-sechs-modelle-schritt-6). Zu
jedem Schritt zeigt die App den passenden Python-Code.

Für Daten mit einer Reihenfolge (Spiele, Kurse, Messreihen) gibt es einen **zeitlichen Split** (Test = die jüngsten
Zeilen), und Schritt 8 vergleicht das Modell auf Wunsch mit einer **Wahrscheinlichkeit aus den Daten** – z. B. der
Siegchance nach Elo aus dem Sportdaten-Tool. Siehe [Sportdaten: zeitlicher Split und Vergleich mit
Elo](#sportdaten-zeitlicher-split-und-vergleich-mit-elo).

Die App ist aus der Deep-Learning-Version (Keras) entstanden: Schritte 1–5 und 10, Aufbereitung, Rezept
(`preprocessing.json`), Konfigurationen und Sprachen sind gleich geblieben; neu sind die Modelle in Schritt 6,
die Validierungskurven in Schritt 7, die Modell-Details in Schritt 8 und die Formate `.skops`/`.joblib` in
Schritt 9.

Die App gibt es auf **Deutsch, Englisch und Polnisch** – siehe [Sprachen](#sprachen).
(English: `README_EN.md` · Polski: `README_PL.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # öffnet http://localhost:8501
```

Getestet mit Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 oder neuer;
TensorFlow wird nicht mehr gebraucht.

## Sprachen

Die Sprache lässt sich jederzeit oben in der Seitenleiste umschalten: **Deutsch · English · Polski**. Es ändern
sich nur die Texte – die Berechnungen sind in allen drei Sprachen identisch (Deutsch ist das Original).

| Direkt in einer Sprache starten | |
|---|---|
| Kommandozeile | `streamlit run app.py -- --lang en` (oder `pl`, `de`) – das `--` vor `--lang` ist nötig |
| Adresse im Browser | `http://localhost:8501/?lang=pl` |
| Umgebungsvariable | `ML_TEMPLATE_LANG=de` (Windows: `set ML_TEMPLATE_LANG=de`) – früher `DL_TEMPLATE_LANG` |

Ohne Angabe startet die App auf Englisch. Eine in der Seitenleiste gewählte Sprache steht danach auch in der
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
4. **Schritt 4:** Label und Features wählen, Split zufällig oder zeitlich → *Daten für ML vorbereiten*
5. **Schritt 6:** Modell wählen (Platz 1: Gradient Boosting) → *Training starten* → danach 7, 8, 9, 10

Für den öffentlichen LendingClub-Datensatz (`lending_club_loan_two.csv`) gibt es in Schritt 2 eine
**Beispiel-Konfiguration** (Schritt 2, 4 und 6 fertig eingestellt: Random Forest mit 100 Bäumen, `max_depth=12`).
Bei langsamem Rechner links die **Stichprobe** aktivieren (`df.sample(frac=0.1)`).

## Die Schritte

| Seite | Inhalt |
|---|---|
| 1 · Daten & Bibliotheken | CSV laden (Upload oder Pfad; Trennzeichen `;` und Dezimalkomma werden automatisch erkannt), Überblick, Bibliotheks-Versionen, optional Spalten-Beschreibungen |
| 2 · Aufbereitung 1 | je Spalte eine Aktion (behalten, entfernen, Zahl extrahieren, Datum → Jahr / Monat / **Jahr + Monat**, erste/letzte N Zeichen, Werte ersetzen, Top-N …) · **neue Spalten**: Werte zuordnen, Formel, Bedingung → 0/1, Kopie + Aktion |
| 3 · Analyse & Visualisierung | **Auswahl der Spalten** · Überblick, Countplot, Histogramm, Boxplot, Scatter, Heatmap, Korrelation mit einer Spalte, Pairplot, Anteil je Kategorie, Gruppen-Statistik, **Jointplot** (scatter/kde/hex/reg), **Dichte 2D (KDE)** mit Filter auf eine Gruppe, **Regression (lmplot)** mit Farbe (hue) und Teilgrafiken (col), **Histogramm je Gruppe** (überlagert) |
| 4 · Aufbereitung 2 | Label + Problemtyp, Features, fehlende Werte je Spalte (mit Korrelations-Hilfe für den Gruppen-Mittelwert); Split **zuerst** – zufällig (`train_test_split`) oder **zeitlich** nach einer Datums-/Saisonspalte (Test = die jüngsten Zeilen) –, danach Füllwerte, Dummies (Übersicht der neuen 0/1-Spalten) und Skalierung – nur aus den Trainingsdaten gelernt · optional **Vergleichsspalte** für Schritt 8 (z. B. `pre_elo_prob_home`) |
| 5 · Daten speichern | Daten nach Schritt 2 oder ML-fertig nach Schritt 4 (mit Spalte `split`) als CSV – international oder für deutsches Excel; eine vorhandene Datei wird nur nach Bestätigung überschrieben |
| 6 · Modell & Training | eines der **sechs Modelle** in der Reihenfolge der Rangliste (mit „Warum?“ und „Am besten für“), dazu seine wichtigsten Parameter wie im Kurs; Validierung, `class_weight='balanced'`, Seed – der passende scikit-learn-Code wird live angezeigt |
| 7 · Training vs. Validierung | Validierungskurve des wichtigsten Parameters (Fehler auf Trainings- und Validierungsdaten, naive Vorhersage als Linie) + Einordnung (Overfitting, Underfitting, kein Signal); *Besten Wert in Schritt 6 übernehmen* (auch die beste Zahl Runden beim Gradient Boosting); SVM mit GridSearchCV: Heatmap C × gamma |
| 8 · Evaluation | classification_report, confusion_matrix, ROC-AUC, **Log Loss**, Schwelle, Einzelfall, Permutation Importance · Regression: MAE, RMSE, R² · **Vergleich mit einer Wahrscheinlichkeit aus den Daten** (z. B. Elo): Accuracy, Log Loss, Brier, ROC-AUC, Unterschied ± Standardfehler, Kalibrierungs-Diagramm · **Modell-Details**: Feature-Wichtigkeit der Bäume (+ Baum-Grafik beim einzelnen Baum), Koeffizienten der Linearen Regression (in Originaleinheiten wie `lm.coef_` im Kurs) und der Logistischen Regression (mit Odds-Faktor), Kennzahlen von Gradient Boosting, SVM (Stützvektoren) und KNN |
| 9 · Modell speichern & laden | `models/<name>/model.skops` (sicher, Standard) oder `model.joblib` + `preprocessing.json`, ZIP-Download, ganze Pipeline als Python-Skript; Prüfsummen der Trainingszeilen optional |
| 10 · Vorhersage | neue CSV oder Formular → gleiche Aufbereitung → Vorhersage, Interpretation, CSV-Download und **Fazit** (Datenqualität, Aussage des Ergebnisses, Empfehlung) |

Zu jedem Schritt gibt es ein Feld **„Python-Code zu diesem Schritt“** – so lässt sich alles 1:1 in ein
Jupyter Notebook übernehmen. Einstellungen (inkl. neuer Spalten) lassen sich links unter **Konfiguration**
als JSON speichern und für neue Daten mit gleichen Spalten wieder laden. Beim Laden wird die Datei geprüft:
unbrauchbare Einträge (falscher Typ, außerhalb des erlaubten Bereichs, z. B. 10 Mio. Bäume) werden ignoriert
und in der Meldung genannt. Konfigurationen der Deep-Learning-Version lassen sich laden: Schritt 2 und 4 werden
übernommen, die Keras-Einstellungen (Layer, Epochen …) still übergangen.

### Daten speichern (Schritt 5)

* Für Vorhersagen in Schritt 10 immer die **ursprüngliche** CSV-Datei verwenden, nicht den Export: Das gespeicherte
  Modell wiederholt die Aufbereitung aus Schritt 2 selbst (z. B. würde „Werte ersetzen 1=2; 2=3“ sonst doppelt
  angewendet).
* Format „deutsches Excel“: Texte, die mit `=`, `+`, `-` oder `@` beginnen, bekommen ein `'` davor – sonst würde
  Excel sie als Formel ausführen (CSV-Injection). Im internationalen Format bleiben die Daten unverändert.
* Heißt ein Feature selbst `split`, heißt die train/test-Spalte `split_train_test`.

## Die sechs Modelle (Schritt 6)

Die Reihenfolge in Schritt 6 ist die Rangliste – Platz 1 ist in der Regel am stärksten:

| Platz | Algorithmus | Warum? | Am besten für |
|---|---|---|---|
| 1 | Gradient Boosting (HistGradientBoosting) | Der beste dieser Liste für Tabellen. Baut Bäume nacheinander – jeder korrigiert die Fehler der vorigen. Meist genauer als Random Forest und auch bei vielen Zeilen schnell; braucht aber eine passende Anzahl Runden (max_iter), sonst lernt er Rauschen mit – Schritt 7 zeigt die beste. | Erste Wahl für Tabellen: Richtung, Sieg/Niederlage (Klassifikation) und Zahlen (Regression) |
| 2 | Decision Trees + Random Forests | Sehr robust und kaum zu verderben. Kommt mit Nichtlinearitäten, Wechselwirkungen zwischen Features und Rauschen zurecht. Random Forest begrenzt das Overfitting eines einzelnen Baums stark. | Vorhersage der Richtung (Klassifikation) und der Renditen |
| 3 | Support Vector Machines | Gut, besonders mit RBF-Kernel. Trennt Klassen gut, ist aber empfindlich gegenüber der Skalierung der Daten und der Wahl der Hyperparameter. | Klassifikation der Richtung (buy/sell) |
| 4 | Logistische Regression | Solide Baseline. Schnell, interpretierbar, stabil. Verliert bei komplexen Zusammenhängen oft gegen Bäume. | Einfache Klassifikation der Richtung |
| 5 | K-Nearest Neighbors | Einfach, aber meist schwächer. Verträgt Rauschen, irrelevante Features und hohe Dimensionalität (typisch für Börsendaten) schlecht. | Eher Experimente / Baseline |
| 6 | Lineare Regression | Am schwächsten für echte Vorhersagen. Setzt einen linearen Zusammenhang voraus, Märkte sind aber stark nichtlinear und verrauscht. | Nur als sehr einfacher Referenzpunkt |

| Modell | scikit-learn | Parameter in Schritt 6 | Kurve in Schritt 7 |
|---|---|---|---|
| Gradient Boosting | `HistGradientBoostingClassifier/Regressor` (`early_stopping=False`) | `max_iter` (Runden), `learning_rate`, `max_leaf_nodes`, `max_depth` (0 = unbegrenzt), `min_samples_leaf` – Standardwerte wie in scikit-learn | Fehler je Anzahl Runden aus **einem** Training (`staged_predict`), mit bestem Wert und *übernehmen* |
| Decision Trees + Random Forests | `RandomForestClassifier/Regressor` oder `DecisionTreeClassifier/Regressor` | Variante (Wald / ein Baum), `n_estimators`, `max_depth` (0 = unbegrenzt), `min_samples_leaf`, `max_features` | Fehler je Anzahl Bäume (der Wald wächst Schritt für Schritt) bzw. je `max_depth` |
| Support Vector Machines | `SVC` / `SVR` | `kernel`, `C`, `gamma`, `epsilon` (SVR); **GridSearchCV** über C = 0.1 … 100 und gamma = 1 … 0.001 wie im Kurs | Fehler je C; mit GridSearchCV zusätzlich die Heatmap aller Kombinationen |
| Logistische Regression | `LogisticRegression` | `C`, `max_iter` | Fehler je C |
| K-Nearest Neighbors | `KNeighborsClassifier/Regressor` | `K (n_neighbors)`, `weights`, Abstand (euklidisch/Manhattan) | Fehlerrate je K = 1 … 40 (Elbow-Methode wie im Kurs) |
| Lineare Regression | `LinearRegression` | `fit_intercept` | Lernkurve: Fehler je Anzahl Trainingszeilen |

* **Passt das Modell zum Problemtyp?** Die Logistische Regression sagt nur Klassen vorher (binär, Mehrklassen), die
  Lineare Regression nur Zahlen (Regression). Bei einem unpassenden Modell erklärt Schritt 6 den Grund, und
  *Training starten* bleibt gesperrt. Die anderen vier Modelle können beides.
* **Gradient Boosting:** Anders als beim Random Forest können mehr Runden schaden – ab einem Punkt lernt das Modell
  Rauschen, der Validierungsfehler steigt. `early_stopping` ist deshalb aus (scikit-learn würde sonst ab 10 000
  Zeilen selbst zufällig 10 % der Trainingsdaten abzweigen); die beste Zahl Runden zeigt die Kurve in Schritt 7.
  `HistGradientBoosting` hat kein `feature_importances_` – die Wichtigkeit zeigt die Permutation Importance in
  Schritt 8. Skalierung braucht es nicht.
* **Fehler in Schritt 7:** Klassifikation = Fehlerrate (1 − Accuracy, wie `np.mean(pred != y_test)` im Kurs),
  Regression = MAE in Originaleinheiten. Die gestrichelte Linie ist die naive Vorhersage (häufigste Klasse bzw.
  Mittelwert) – ein Modell darunter hat etwas gelernt.
* **Validierung:** „Testdaten“ berechnet die Kurve wie im Kurs auf `X_test`; „Anteil aus Training (sauberer)“
  nimmt einen eigenen Teil der Trainingsdaten, damit das Testset bis Schritt 8 unberührt bleibt – beim zeitlichen
  Split den **jüngsten** Teil (`train_test_split(..., shuffle=False)`), GridSearchCV nutzt dann `TimeSeriesSplit`.
  Ohne Validierungskurve vergleicht Schritt 7 nur Training und Test.
* **Besten Wert übernehmen:** Schritt 7 schlägt einen Wert vor, wenn er den Fehler mehr als zufällig senkt, und
  trägt ihn auf Knopfdruck in Schritt 6 ein – für das trainierte Modell; bei der SVM mit GridSearchCV wird die Suche
  dafür ausgeschaltet (sonst würde sie C wieder ersetzen). Danach neu trainieren.
* **Wahrscheinlichkeiten der SVM:** für Schwelle und ROC-AUC (Schritt 8) wird die SVM mit
  `CalibratedClassifierCV(SVC(...), ensemble=False)` kalibriert – `SVC(probability=True)` ist seit scikit-learn 1.9
  veraltet.
* **Große Daten:** SVM und KNN werden bei sehr vielen Zeilen langsam – Schritt 6 weist darauf hin (Tipp:
  Stichprobe in der Seitenleiste), ebenso auf fehlende Skalierung bei SVM und KNN (sie rechnen mit Abständen). Die
  Fehler der Validierungskurven werden auf höchstens 5000 Zeilen je Datensatz gemessen (KNN: 2000), trainiert
  wird mit allen Zeilen.

## Sportdaten: zeitlicher Split und Vergleich mit Elo

Die App erkennt CSV-Dateien im Format des Sportdaten-Tools (Datasety NBA i tenis) am Label `home_win` bzw. `p1_win`
zusammen mit Spalten `pre_*`/`lineup_*` – auch Exporte, in denen die Gruppen `inplay_*` und `post_*` im Tool abgewählt
wurden – und schlägt in Schritt 4 passende Einstellungen vor:

| Vorschlag | Warum |
|---|---|
| Features: nur `pre_*` und `lineup_*` | vor dem Spiel bekannt. `inplay_*` (während) und `post_*` (nach dem Spiel) wären ein Datenleck – wählt man sie trotzdem, warnt Schritt 4 |
| Label `home_win` (NBA) bzw. `p1_win` (Tennis) | Ziel der Vorhersage |
| zeitlicher Split nach `date` (Tennis: `match_date_est`) | Training = ältere Spiele, Test = die jüngsten. Spiele desselben Tages landen nie in beiden Teilen. Die Datumsspalte bleibt dafür in Schritt 2 unverändert (sonst würde „Datum → Jahr“ vorgeschlagen) |
| Vergleichsspalte `pre_elo_prob_home` (Tennis: `pre_elo_prob_p1`) | die Siegchance nach Elo vor dem Spiel – das Tool rechnet Elo Spiel für Spiel nur aus früheren Spielen. Sie gilt für die Heimmannschaft bzw. Spieler 1, also die Klasse `1` – vorgeschlagen wird sie deshalb nur, wenn `1` die positive Klasse ist. Sagt eine gewählte Vergleichsspalte die positive Klasse auf den Trainingszeilen schlechter vorher als Zufall (ROC-AUC unter 0.5), warnen Schritt 4 und Schritt 8 |

**Vergleich in Schritt 8.** Auf denselben Testzeilen stehen Modell, Vergleichsspalte und naive Vorhersage (immer
der Anteil der positiven Klasse im Training) nebeneinander: Accuracy (Schwelle 0.5), Log Loss, Brier und ROC-AUC.
Entscheidend ist der **Log Loss** – er misst, wie gut die Wahrscheinlichkeiten stimmen. Die App rechnet den mittleren
Unterschied je Spiel mit Standardfehler (gepaart, dieselben Spiele) und sagt, ob das Modell die Vergleichsspalte
mehr als zufällig schlägt. Das Kalibrierungs-Diagramm zeigt, ob z. B. 70 % vorhergesagt auch in etwa 70 % der Fälle
eintreten – erst dann taugen die Wahrscheinlichkeiten für einen Vergleich mit Wettquoten.

Beispiel mit den NBA-Daten des Tools (Stand 22.10.2026, 28 382 Spiele seit 2004-05, Test = 20 % jüngste Spiele ab März
2022 – `test_size` 0.2 –, sonst Vorschläge aus Schritt 4; Gradient Boosting auf allen `pre_*`/`lineup_*` mit 300
Runden, `learning_rate=0.05` und der Validierung „Anteil aus Training (sauberer)“): Log Loss 0.621 gegen 0.620 für Elo
allein – kein Vorteil, denn das Modell ist überangepasst. Schritt 7 meldet das und schlägt 50 Runden vor; nach „Besten
Wert in Schritt 6 übernehmen“ und neuem Training: Log Loss **0.613 gegen 0.620**, Unterschied −0.006 ± 0.002 je Spiel
– das Modell schlägt Elo knapp, aber mehr als zufällig. Die Zahl der Runden also nicht raten, sondern in Schritt 7
prüfen. Das passt zum Benchmark des Tools (Training bis Saison 2023, Test ab 2024): nur `pre_*` 0.6083 ≈ Elo 0.6085,
mit `lineup_*` (wer spielt) 0.6022. Im Tennis (ATP, ebenfalls 20 % Test, ab Februar 2021) liegt schon Gradient
Boosting mit den Standardwerten klar vor Elo: 0.608 gegen 0.634.

Der zeitliche Split funktioniert mit jeder Datums- oder Zahlenspalte (Jahr, Saison), die Vergleichsspalte mit jeder
Spalte mit Wahrscheinlichkeiten zwischen 0 und 1 für die positive Klasse (z. B. aus Wettquoten) – auch außerhalb von
Sportdaten. Zeilen ohne Zeitwert werden entfernt; Zeilen mit demselben Zeitwert bleiben zusammen (der Testanteil kann
daher etwas von `test_size` abweichen, z. B. bei ganzen Saisons). Schritt 4 prüft das Ergebnis: Bleiben weniger als 10
Trainingszeilen oder fehlt dem Training eine Klasse (z. B. weil sie erst in den jüngsten Zeilen vorkommt), erklärt
eine Meldung, was zu tun ist; fällt die Grenze auf den ältesten Zeitwert, nennt sie das größte mögliche `test_size`.

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
| `SVC(probability=True)` | scikit-learn 1.9: veraltet (FutureWarning), wird in 1.11 entfernt | `CalibratedClassifierCV(SVC(...), ensemble=False)` |
| `LogisticRegression(penalty='l2')` | scikit-learn 1.8: `penalty` veraltet, wird in 1.10 entfernt | ohne `penalty` (Standard L2), Stärke über `C` |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | beim Laden kann beliebiger Code aus der Datei ausgeführt werden | `.skops` als Standard (`skops.io`, nur bekannte scikit-learn-Typen); `.joblib` nur für vertrauenswürdige Dateien |
| `scaler.fit(df.drop('TARGET CLASS', axis=1))` **vor** `train_test_split` (KNN-Projekt) | Mittelwert und Streuung enthalten Information aus den Testdaten (Datenleck) | Split zuerst, Scaler nur auf `X_train` |
| `df.corr()` | pandas ≥ 2: Fehler bei Textspalten | `df.corr(numeric_only=True)` |
| `df.select_dtypes(['object'])` | pandas 3: Text hat den Typ `str`; `'object'` findet ihn nur noch übergangsweise | Typprüfung über `is_numeric_dtype` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False statt 0/1 | `dtype=float` |
| `df.groupby('total_acc').mean()['mort_acc']` | pandas ≥ 2: Fehler bei Textspalten | `df.groupby('total_acc')['mort_acc'].mean()` |
| `sns.distplot` | veraltet | `sns.histplot` |
| `sns.kdeplot(df['a'], df['b'], shade=True, shade_lowest=False)` | seaborn ≥ 0.12: Fehler, nur `data` darf ohne Namen übergeben werden; `shade`/`shade_lowest` veraltet | `sns.kdeplot(x='a', y='b', data=df, fill=True, thresh=0.05)` |
| `plt.figure(figsize=(11, 7))` vor `sns.lmplot(...)` | wirkt nicht – lmplot erzeugt eine eigene Figure (im Notebook: „Figure … with 0 Axes“) | `sns.lmplot(..., height=4, aspect=1.15)` |
| Lücken füllen, **dann** `train_test_split` | Füllwerte enthalten Information aus den Testdaten (Datenleck) | Split zuerst; Füllwerte, Kategorien und Scaler nur aus den Trainingsdaten |

Zusätzliche Optionen (jeweils abschaltbar): zeitlicher Split, Vergleichsspalte (z. B. Elo), `stratify=y`,
einstellbare Schwelle + ROC-AUC,
`class_weight='balanced'`, Validierung aus dem Trainingsanteil statt mit den Testdaten, GridSearchCV,
Label-Skalierung bei Regression, Datenleck-Warnung, Permutation Importance.

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
* neues Modell → `MODELS`, `MODEL_NAMES`, `MODEL_WHY`, `MODEL_BEST`, `MODEL_TASKS`, `ModelConfig`, `make_model()`
  und `validation_curve()` in `pipeline.py`, `model_params_ui()` in `app.py` (+ `estimator_code()` und
  `curve_code()` in `codegen.py`)
* neuer oder geänderter Text → auf Deutsch in `tx("…")` schreiben (Konstanten: `N_("…")`) und die englische und
  polnische Übersetzung in der passenden Datei in `locales/` ergänzen; `pytest tests/test_i18n.py` meldet fehlende
  Übersetzungen.

## Tests

```bash
pip install pytest
pytest -q                                  # 310 Tests, ca. 5 Minuten
python tests/synthetic.py --rows 20000     # synthetische Demo-CSV (keine echten Daten)
```

| Datei | prüft |
|---|---|
| `tests/test_pipeline.py` | Beispiel-Pipeline, Speichern/Laden (.skops/.joblib), neue Daten, Regression, Mehrklassen, exportiertes Skript |
| `tests/test_models.py` | die sechs Modelle: jedes Modell × Problemtyp (Training, Validierungskurve, Einordnung, Modell-Details, Speichern/Laden), angezeigter scikit-learn-Code (Modell und Kurve) rechnet **exakt** wie die App, Elbow-Methode = `predict()` je K, Koeffizienten in Originaleinheiten, sicheres Laden (.skops/.joblib), Klick-Test Regression (Lineare Regression, KNN, SVM + GridSearchCV) |
| `tests/test_codegen_parity.py` | der angezeigte Python-Code rechnet **exakt** dasselbe wie die App (X_train, X_test, y) – auch in Grenzfällen und mit neuen Spalten |
| `tests/test_new_features.py` | Jahr + Monat, neue Spalten (inkl. Schutz vor beliebigem Code in Formeln), Daten speichern, Fazit |
| `tests/test_audit_regressions.py` | je ein Test pro Fehler aus dem ersten Audit (siehe `AUDIT.md`) |
| `tests/test_audit2_regressions.py` | zweites Audit: Grenzen für Formeln, Prüfung importierter Konfigurationen, `inf` im Label, Export (Excel, Spalte `split`), Diagramme mit leeren Spalten, ZIP-Größe, Fazit |
| `tests/test_app.py` | Klick-Test der Oberfläche (alle 10 Schritte, neue Spalten, Export, Fazit; Bool-Label, Semikolon-CSV, Schutz vor Überschreiben; abhängige neue Spalten, unbrauchbare Werte, Fehler beim Zeichnen, Modell ohne Prüfsummen) |
| `tests/test_audit3_regressions.py` | drittes Audit: kein Code aus Spaltennamen/Werten im angezeigten Python-Code, Schritt 10 mit gleichnamigen Spalten, Diagrammtitel (Notebook, `$`), Sprachwechsel (Tabellen, Export, Adresse) |
| `tests/test_audit4_regressions.py` | viertes Audit (scikit-learn): „Besten Wert übernehmen“ für das trainierte Modell und mit GridSearchCV, exakte Zahlen im angezeigten Code, GridSearchCV mit wenigen Zeilen je Fold, Rückfrage vor pickle, Namen aus Dateien nicht als Markdown, sicherer Lade-Code für `.skops`, Speichern ohne Verlust des alten Modells, Größenlimit für `.skops`, Hinweis zu `drop_first`, Achsentexte in der aktuellen Sprache |
| `tests/test_charts.py` | neue Diagramme in Schritt 3 (Jointplot, Dichte 2D, lmplot, Histogramm je Gruppe): ungewöhnliche Daten, angezeigter Code läuft, Klick-Test |
| `tests/test_audit5_regressions.py` | fünftes Audit (Gradient Boosting, zeitlicher Split, Elo): Vergleichsspalte der anderen Klasse (kein Vorschlag, Warnung in Schritt 4 und 8), zeitlicher Split ohne beide Klassen oder mit zu wenig Training, Erkennung ohne `post_*`/`inplay_*` und ohne Fehlalarm, Code von Schritt 8 bei entfernter Vergleichsspalte, Hinweis „neu vorbereiten“ nach Wechsel der Vergleichsspalte, Meldungen (Grenze beim ältesten Zeitwert, Vergleichsspalte ohne Testwert oder = Label, `test_size` 0/1, GridSearchCV mit `TimeSeriesSplit`), Tempo von Schritt 4 bei breiten Daten |
| `tests/test_sports_timesplit.py` | Sportdaten-Format erkennen (Vorschläge, Datumsspalte bleibt), zeitlicher Split (Training vor dem Test, gleiche Tage zusammen, Zahl/Datum, Fehlermeldungen), angezeigter Code = App auch beim zeitlichen Split, Gradient Boosting (Kurve aus einem Training, bester Wert, Hinweis „mehr Runden“), GridSearchCV mit `TimeSeriesSplit`, Vergleich mit einer Wahrscheinlichkeit (Urteil, Standardfehler, Zeilen ohne Wert), ganzes Skript, Klick-Test mit Sportdaten (Schritt 1–10, „Besten Wert übernehmen“ bei Gradient Boosting) |
| `tests/test_i18n.py` | Sprachen: jeder Text hat eine englische und polnische Übersetzung mit denselben Platzhaltern, Deutsch unverändert, Excel-CSV je Sprache, erzeugter Code rechnet in jeder Sprache gleich, die ganze App auf Englisch und Polnisch ohne deutsche Reste |

## Hinweise

* Sehr große CSVs besser über „Pfad auf diesem Rechner“ laden als hochladen.
* Während des Trainings nichts anklicken – jede Eingabe startet das Skript neu und bricht das Training ab.
* scikit-learn rechnet auf der CPU (keine GPU nötig). Random Forest, KNN und GridSearchCV nutzen alle Kerne
  (`n_jobs=-1`), Gradient Boosting ebenfalls (OpenMP); Schritt 1 zeigt die Anzahl der CPU-Kerne.
* **Nur lokal verwenden.** Das Pfad-Feld liest jede Datei, auf die der angemeldete Benutzer Zugriff hat, und
  `models/` ist für alle Nutzer der App gleich. Die App daher nicht ohne Anmeldung ins Netz stellen.
  Modelle nur aus vertrauenswürdigen Quellen laden (ein ZIP darf entpackt höchstens 2 GB groß sein; Formeln im
  Rezept werden beim Laden erneut geprüft). `.skops` wird ohne Code-Ausführung geladen – nur bekannte
  scikit-learn-Typen sind erlaubt, alles andere lehnt die App ab. `.joblib` ist pickle und kann beim Laden Code
  ausführen: aus einem hochgeladenen ZIP lädt die App eine `.joblib`-Datei nur nach ausdrücklicher Bestätigung.
  Modelle der Deep-Learning-Version (`model.keras`/`.h5`) lassen sich hier nicht laden – die Meldung sagt es.
* **Testkennzahlen sind leicht optimistisch**, wenn die Testdaten auch für die Validierungskurve dienen und danach
  der beste Wert übernommen wird (Standard „Validierung: Testdaten“, wie im Kurs) oder die Schwelle in Schritt 8 auf
  den Testdaten eingestellt wird. Für eine ehrliche Schätzung in Schritt 6 „Anteil aus Training (sauberer)“
  wählen; GridSearchCV sucht ohnehin nur mit Kreuzvalidierung auf den Trainingsdaten.
* **Kein Datenleck in Schritt 4:** Füllwerte, Kategorien (Dummy-Spalten) und Scaler werden nach dem Split nur
  aus den Trainingsdaten gelernt. Eine Kategorie, die nur in den Testdaten vorkommt, wird wie ein unbekannter
  Wert in neuen Daten behandelt (alle Dummies 0). Aus allen Zeilen (vor dem Split) entstehen nur noch die Liste
  bei „Top-N behalten“ in Schritt 2 und die automatischen Vorschläge in den Tabellen – beides mit geringem Einfluss.
