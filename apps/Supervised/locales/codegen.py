"""Übersetzungen für codegen.py (Kommentare im erzeugten Python-Code): {Deutsch: (Englisch, Polnisch)}"""

T = {
    "Erste Zahl aus Text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5' → 1234.5": (
        "First number in the text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5' → 1234.5",
        "Pierwsza liczba z tekstu: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5' → 1234.5"),
    "Leerzeichen in den Spaltennamen entfernen": ("remove spaces in the column names", "usuń spacje w nazwach kolumn"),
    "Stichprobe": ("sample", "próbka"),
    "optional: Stichprobe für langsame Rechner": ("optional: sample for slow computers",
                                                  "opcjonalnie: próbka dla wolnych komputerów"),
    "{c}: ist bereits eine Zahl – keine Umwandlung nötig": ("{c}: already a number – no conversion needed",
                                                           "{c}: już jest liczbą – konwersja niepotrzebna"),
    "{c}: ist bereits Jahr als Zahl – keine Umwandlung nötig": ("{c}: already a year as a number – no conversion needed",
                                                               "{c}: już jest rokiem jako liczba – konwersja niepotrzebna"),
    "{c}: ist bereits Monat als Zahl – keine Umwandlung nötig": (
        "{c}: already a month as a number – no conversion needed",
        "{c}: już jest miesiącem jako liczba – konwersja niepotrzebna"),
    "Datum als Zahl, z. B. 20141013": ("date as a number, e.g. 20141013", "data jako liczba, np. 20141013"),
    "die {n} häufigsten Werte der Trainingsdaten": ("the {n} most frequent values of the training data",
                                                    "najczęstsze wartości danych treningowych (top {n})"),
    "Zahl → Kategorie (Text)": ("number → category (text)", "liczba → kategoria (tekst)"),
    "unbekannte Aktion {act} für {c}": ("unknown action {act} for {c}", "nieznana akcja {act} dla {c}"),
    "1 = ja, 0 = nein, leer bei Lücken": ("1 = yes, 0 = no, empty where values are missing",
                                          "1 = tak, 0 = nie, puste przy brakach"),
    "ersetzt durch {target}_jahr und {target}_monat": ("replaced by {target}_jahr and {target}_monat",
                                                       "zastąpiona przez {target}_jahr i {target}_monat"),
    "Neue Spalten": ("New columns", "Nowe kolumny"),
    "keine Änderungen": ("no changes", "brak zmian"),
    "Zeilen ohne Label": ("rows without a label", "wiersze bez etykiety"),
    "Zeilen mit inf im Label": ("rows with inf in the label", "wiersze z inf w etykiecie"),
    "inf wie fehlend behandeln": ("treat inf as missing", "traktuj inf jak brak"),
    "Spalten mit zu vielen Lücken": ("columns with too many gaps", "kolumny ze zbyt wieloma brakami"),
    "→ Dummies": ("→ dummy columns", "→ kolumny dummy"),
    "als Text, ohne Leerzeichen am Rand": ("as text, without leading/trailing spaces",
                                           "jako tekst, bez spacji na brzegach"),
    "Zuerst aufteilen – Füllwerte, Kategorien und Scaler werden danach NUR aus den Trainingsdaten gelernt": (
        "Split first – fill values, categories and scaler are then learned ONLY from the training data",
        "Najpierw podział – wartości uzupełniające, kategorie i skaler są potem uczone TYLKO na danych "
        "treningowych"),
    "Fehlende Werte: Füllwerte aus den Trainingsdaten berechnen …": (
        "Missing values: compute fill values from the training data …",
        "Brakujące wartości: oblicz wartości uzupełniające z danych treningowych …"),
    "… und auf Train UND Test anwenden – die Testdaten lernen nichts mit.": (
        "… and apply them to train AND test – nothing is learned from the test data.",
        "… i zastosuj je do train ORAZ test – z danych testowych nic się nie uczymy."),
    "Dummies mit den Kategorien der Trainingsdaten (Kategorie nur im Test → alle Dummies 0)": (
        "Dummies with the categories of the training data (category only in test → all dummies 0)",
        "Kolumny dummy z kategoriami danych treningowych (kategoria tylko w teście → wszystkie dummy 0)"),
    "unbekannt → NaN": ("unknown → NaN", "nieznana → NaN"),
    "keine Skalierung": ("no scaling", "bez skalowania"),
    "fit nur auf Trainingsdaten!": ("fit on training data only!", "fit tylko na danych treningowych!"),
    "Label skalieren (Regression) – Vorhersagen später zurückrechnen": (
        "Scale the label (regression) – convert predictions back later",
        "Skalowanie etykiety (regresja) – prognozy później przeliczyć z powrotem"),
    "für deutsches Excel": ("for German Excel", "dla niemieckiego Excela"),
    "für polnisches Excel": ("for Polish Excel", "dla polskiego Excela"),
    "für Excel in Großbritannien, USA, Australien …": ("for Excel in the UK, US, Australia …",
                                                       "dla Excela w Wielkiej Brytanii, USA, Australii …"),
    "ursprüngliche Reihenfolge der Zeilen": ("original order of the rows", "pierwotna kolejność wierszy"),
    "neue_daten.csv": ("new_data.csv", "nowe_dane.csv"),
    "vorhersagen.csv": ("predictions.csv", "prognozy.csv"),
    "zurück in Originaleinheiten": ("back to original units", "z powrotem w oryginalnych jednostkach"),
    "Wichtig: die Vorverarbeitung gehört zum Modell! Die App speichert sie in": (
        "Important: the preprocessing belongs to the model! The app saves it in",
        "Ważne: przetwarzanie wstępne należy do modelu! Aplikacja zapisuje je w"),
    "'{folder}/preprocessing.json' (Füllwerte, Dummy-Spalten, Scaler, Label-Kodierung)": (
        "'{folder}/preprocessing.json' (fill values, dummy columns, scaler, label encoding)",
        "'{folder}/preprocessing.json' (wartości uzupełniające, kolumny dummy, skaler, kodowanie etykiety)"),
    "1) Bibliotheken": ("1) Libraries", "1) Biblioteki"),
    "Daten laden (+ Stichprobe)": ("Load data (+ sample)", "Wczytanie danych (+ próbka)"),
    "2) Datenaufbereitung 1": ("2) Data preparation 1", "2) Przygotowanie danych 1"),
    "4) Datenaufbereitung 2": ("4) Data preparation 2", "4) Przygotowanie danych 2"),
    "6) Modell & Training": ("6) Model & training", "6) Model i trening"),
    "8) Evaluation": ("8) Evaluation", "8) Ewaluacja"),
    "9) Speichern & Laden": ("9) Save & load", "9) Zapis i odczyt"),
    "numeric_only=True: nötig ab pandas 2": ("numeric_only=True: needed from pandas 2", "numeric_only=True: wymagane od pandas 2"),
    "nur Zeilen mit {col} = {val}": ("only rows with {col} = {val}", "tylko wiersze z {col} = {val}"),
    "seaborn ≥ 0.11: x=/y= als Schlüsselwörter, fill statt shade, thresh statt shade_lowest": (
        "seaborn ≥ 0.11: x=/y= as keywords, fill instead of shade, thresh instead of shade_lowest",
        "seaborn ≥ 0.11: x=/y= jako słowa kluczowe, fill zamiast shade, thresh zamiast shade_lowest"),
    "lmplot erstellt eine eigene Figure: Größe über height/aspect – plt.figure() davor wirkt nicht": (
        "lmplot creates its own figure: size via height/aspect – plt.figure() before it has no effect",
        "lmplot tworzy własną figurę: rozmiar przez height/aspect – plt.figure() wcześniej nic nie zmienia"),
    "gleiche Klassen (bins) für alle Gruppen": ("same bins for all groups",
                                                "te same przedziały (bins) dla wszystkich grup"),
    # Schritt 4, 6–9 (ML-Version)
    "pandas ≥ 2: sonst True/False statt 0/1": (
        "pandas ≥ 2: otherwise True/False instead of 0/1",
        "pandas ≥ 2: inaczej True/False zamiast 0/1"),
    "wie im Kurs": (
        "as in the course",
        "jak w kursie"),
    "Wahrscheinlichkeiten für Schwelle und ROC-AUC: CalibratedClassifierCV statt SVC(probability=True) – das "
    "ist seit scikit-learn 1.9 veraltet": (
        "Probabilities for threshold and ROC-AUC: CalibratedClassifierCV instead of SVC(probability=True) – "
        "deprecated since scikit-learn 1.9",
        "Prawdopodobieństwa do progu i ROC-AUC: CalibratedClassifierCV zamiast SVC(probability=True) – "
        "przestarzałe od scikit-learn 1.9"),
    "Koeffizienten – wie im Kurs: lm.coef_": (
        "coefficients – as in the course: lm.coef_",
        "współczynniki – jak w kursie: lm.coef_"),
    "GridSearchCV: mittlere Validierungsgüte": (
        "GridSearchCV: mean validation score",
        "GridSearchCV: średni wynik walidacji"),
    "keine Validierungskurve (in Schritt 6 ausgeschaltet)": (
        "no validation curve (switched off in step 6)",
        "brak krzywej walidacji (wyłączona w kroku 6)"),
    "Validierungsanteil aus den Trainingsdaten": (
        "validation share of the training data",
        "część walidacyjna z danych treningowych"),
    "Validierung mit den Testdaten (wie im Kurs)": (
        "validation with the test data (as in the course)",
        "walidacja na danych testowych (jak w kursie)"),
    "Fehlerrate (Error Rate)": (
        "error rate",
        "odsetek błędów (error rate)"),
    "jeder Schritt fügt nur neue Bäume hinzu": (
        "each step only adds new trees",
        "każdy krok dodaje tylko nowe drzewa"),
    "unbegrenzt": (
        "unlimited",
        "bez limitu"),
    "Elbow-Methode wie im Kurs": (
        "elbow method as in the course",
        "metoda łokcia jak w kursie"),
    "Lernkurve": (
        "learning curve",
        "krzywa uczenia"),
    "je 1 Einheit des skalierten Features – die App zeigt zusätzlich Originaleinheiten": (
        "per 1 unit of the scaled feature – the app also shows original units",
        "na 1 jednostkę przeskalowanej cechy – aplikacja pokazuje też oryginalne jednostki"),
    "Koeffizienten (log-odds, skalierte Features)": (
        "coefficients (log-odds, scaled features)",
        "współczynniki (log-odds, przeskalowane cechy)"),
    "Wahrscheinlichkeit für Klasse 1": (
        "probability of class 1",
        "prawdopodobieństwo klasy 1"),
    "Standard von scikit-learn (pickle)": (
        "scikit-learn standard (pickle)",
        "standard scikit-learn (pickle)"),
    "nur vertrauenswürdige Dateien laden – pickle kann beim Laden Code ausführen": (
        "only load trusted files – pickle can run code when loading",
        "wczytuj tylko zaufane pliki – pickle może wykonać kod przy wczytywaniu"),
    "sicheres Format: beim Laden wird kein Code ausgeführt": (
        "safe format: no code is run when loading",
        "bezpieczny format: przy wczytywaniu nie jest wykonywany żaden kod"),
    "nur scikit-learn-Typen vertrauen – alles andere lehnt sio.load ab": (
        "trust only scikit-learn types – sio.load refuses everything else",
        "ufaj tylko typom scikit-learn – wszystko inne sio.load odrzuci"),
    "Automatisch erzeugt mit dem Streamlit-ML-Template – entspricht den Einstellungen in der App.": (
        "Generated automatically with the Streamlit ML template – matches the settings in the app.",
        "Wygenerowano automatycznie szablonem Streamlit ML – odpowiada ustawieniom w aplikacji."),
    "7) Training vs. Validierung": (
        "7) Training vs. validation",
        "7) Trening a walidacja"),
    "Güte der Wahrscheinlichkeiten – kleiner = besser": (
        "quality of the probabilities – smaller = better",
        "jakość prawdopodobieństw – mniej = lepiej"),
    "Zeitspalte → Zeitpunkte": (
        "time column → points in time",
        "kolumna czasu → punkty w czasie"),
    "Zeitspalte als Zahl (z. B. Jahr oder Saison)": (
        "time column as a number (e.g. year or season)",
        "kolumna czasu jako liczba (np. rok albo sezon)"),
    "Zeilen ohne Zeitwert entfernen": (
        "remove rows without a time value",
        "usuń wiersze bez wartości czasu"),
    "nach der Zeit sortieren – stabil: gleiche Zeit behält die ursprüngliche Reihenfolge": (
        "sort by time – stable: equal times keep the original order",
        "sortuj według czasu – stabilnie: ten sam czas zachowuje pierwotną kolejność"),
    "Zeitpunkt der ersten Testzeile": (
        "time of the first test row",
        "czas pierwszego wiersza testowego"),
    "Test = die jüngsten Zeilen, Training = alles davor (kein Blick in die Zukunft)": (
        "test = the most recent rows, training = everything before (no look into the future)",
        "test = najnowsze wiersze, trening = wszystko wcześniej (bez zaglądania w przyszłość)"),
    "Validierung = der jüngste Teil der Trainingsdaten": (
        "validation = the most recent part of the training data",
        "walidacja = najnowsza część danych treningowych"),
    "EIN Training – staged_predict liefert die Vorhersage nach jeder Runde": (
        "ONE training run – staged_predict gives the prediction after every round",
        "JEDEN trening – staged_predict zwraca prognozę po każdej rundzie"),
    "HistGradientBoosting hat kein feature_importances_": (
        "HistGradientBoosting has no feature_importances_",
        "HistGradientBoosting nie ma feature_importances_"),
    "nur Zeilen mit Wert": (
        "only rows with a value",
        "tylko wiersze z wartością"),
    "zeitlicher Split: jeder Fold validiert auf späteren Zeilen": (
        "time-based split: every fold validates on later rows",
        "podział czasowy: każdy fold waliduje na późniejszych wierszach"),
    "Vergleich: Wahrscheinlichkeit aus den Daten (z. B. Elo)": (
        "comparison: probability from the data (e.g. Elo)",
        "porównanie: prawdopodobieństwo z danych (np. Elo)"),
}
