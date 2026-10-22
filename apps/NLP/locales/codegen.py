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
    "für deutsches Excel": ("for German Excel", "dla niemieckiego Excela"),
    "für polnisches Excel": ("for Polish Excel", "dla polskiego Excela"),
    "für Excel in Großbritannien, USA, Australien …": ("for Excel in the UK, US, Australia …",
                                                       "dla Excela w Wielkiej Brytanii, USA, Australii …"),
    "ursprüngliche Reihenfolge der Zeilen": ("original order of the rows", "pierwotna kolejność wierszy"),
    "vorhersagen.csv": ("predictions.csv", "prognozy.csv"),
    "1) Bibliotheken": ("1) Libraries", "1) Biblioteki"),
    "Daten laden (+ Stichprobe)": ("Load data (+ sample)", "Wczytanie danych (+ próbka)"),
    "2) Datenaufbereitung 1": ("2) Data preparation 1", "2) Przygotowanie danych 1"),
    "6) Modell & Training": ("6) Model & training", "6) Model i trening"),
    "8) Evaluation": ("8) Evaluation", "8) Ewaluacja"),
    "9) Speichern & Laden": ("9) Save & load", "9) Zapis i odczyt"),
    "numeric_only=True: nötig ab pandas 2": ("numeric_only=True: needed from pandas 2", "numeric_only=True: wymagane od pandas 2"),
    # Schritt 4, 6–9 (ML-Version)
    "wie im Kurs": (
        "as in the course",
        "jak w kursie"),
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
    "7) Training vs. Validierung": (
        "7) Training vs. validation",
        "7) Trening a walidacja"),
    # --- NLP-App (Textklassifikation) ---
    "eine Zeile je Klasse": ("one row per class", "jeden wiersz na klasę"),
    "Datei ohne Kopfzeile – Spaltennamen selbst vergeben (wie im Kurs)": (
        "file without a header row – assign the column names yourself (as in the course)",
        "plik bez wiersza nagłówka – nazwy kolumn nadaj samodzielnie (jak na kursie)"),
    "fehlender Text → leerer Text": ("missing text → empty text", "brakujący tekst → pusty tekst"),
    "Zuerst aufteilen – der Vektorisierer lernt sein Vokabular danach NUR aus den Trainingstexten": (
        "Split first – the vectorizer then learns its vocabulary ONLY from the training texts",
        "Najpierw podział – wektoryzator uczy się potem słownika TYLKO z tekstów treningowych"),
    "Vokabular nur aus den Trainingstexten": (
        "vocabulary only from the training texts",
        "słownik tylko z tekstów treningowych"),
    "Testtexte nur umwandeln – unbekannte Wörter fallen weg": (
        "only transform the test texts – unknown words are dropped",
        "teksty testowe tylko przekształć – nieznane słowa odpadają"),
    "(Texte, Wörter) und Größe des Vokabulars": (
        "(texts, words) and size of the vocabulary",
        "(teksty, słowa) i rozmiar słownika"),
    "in wie vielen Trainingstexten": ("in how many training texts", "w ilu tekstach treningowych"),
    "je Klasse: log P(Wort | Klasse) − Mittel der anderen Klassen": (
        "per class: log P(word | class) − mean of the other classes",
        "dla każdej klasy: log P(słowo | klasa) − średnia pozostałych klas"),
    "positiv → spricht für Klasse 1": ("positive → speaks for class 1", "dodatnie → przemawia za klasą 1"),
    "Einzelner Text: Vektorisierer und Modell als Pipeline (wie im Kurs)": (
        "Single text: vectorizer and model as a pipeline (as in the course)",
        "Pojedynczy tekst: wektoryzator i model jako potok (jak na kursie)"),
    "Vektorisierer und Modell gehören zusammen": (
        "vectorizer and model belong together",
        "wektoryzator i model należą do siebie"),
    "'{folder}/preprocessing.json' (Schritt 2, Textspalte, Klassen, Schwelle)": (
        "'{folder}/preprocessing.json' (step 2, text column, classes, threshold)",
        "'{folder}/preprocessing.json' (krok 2, kolumna tekstu, klasy, próg)"),
    "4) Datenaufbereitung 2 – Text, Label, Split, Vektorisierung": (
        "4) Data preparation 2 – text, label, split, vectorization",
        "4) Przygotowanie danych 2 – tekst, etykieta, podział, wektoryzacja"),
    "Textlänge": ("text length", "długość tekstu"),
    "Länge in Zeichen (wie im Kurs: .apply(len))": (
        "length in characters (as in the course: .apply(len))",
        "długość w znakach (jak na kursie: .apply(len))"),
    "Klassen als Text, ohne Leerzeichen am Rand": (
        "classes as text, without surrounding spaces",
        "klasy jako tekst, bez spacji na brzegach"),
    "Klassen → Zahlen: {m}": ("classes → numbers: {m}", "klasy → liczby: {m}"),
    "TF-IDF: häufige Wörter zählen weniger": (
        "TF-IDF: frequent words count less",
        "TF-IDF: częste słowa liczą się mniej"),
    "Bag of Words: Wörter zählen": ("bag of words: count words", "bag of words: zliczanie słów"),
    "wie oft insgesamt": ("how often in total", "ile razy łącznie"),
    "LinearSVC liefert keine Wahrscheinlichkeiten – CalibratedClassifierCV ergänzt sie (für Schwelle und ROC-AUC)": (
        "LinearSVC provides no probabilities – CalibratedClassifierCV adds them (for the threshold and ROC-AUC)",
        "LinearSVC nie daje prawdopodobieństw – CalibratedClassifierCV je dodaje (dla progu i ROC-AUC)"),
    "eigener Vektorisierer – lernt nur aus dem Trainingsteil": (
        "own vectorizer – learns only from the training part",
        "osobny wektoryzator – uczy się tylko z części treningowej"),
    "log P(Wort | Klasse 1) − log P(Wort | Klasse 0)": (
        "log P(word | class 1) − log P(word | class 0)",
        "log P(słowo | klasa 1) − log P(słowo | klasa 0)"),
    "Welche Wörter sprechen für welche Klasse?": (
        "Which words speak for which class?",
        "Które słowa przemawiają za którą klasą?"),
    "Die App speichert zusätzlich das Rezept in": (
        "The app also saves the recipe in",
        "Aplikacja zapisuje też przepis w"),
    "neue_texte.csv": ("new_texts.csv", "nowe_teksty.csv"),
    "Nur einzelne Texte – ohne pipeline.py, direkt mit der scikit-learn-Pipeline:": (
        "Just single texts – without pipeline.py, directly with the scikit-learn pipeline:",
        "Tylko pojedyncze teksty – bez pipeline.py, bezpośrednio potokiem scikit-learn:"),
    "Automatisch erzeugt mit DATA-Analysis-NLP – entspricht den Einstellungen in der App.": (
        "Generated automatically with DATA-Analysis-NLP – matches the settings in the app.",
        "Wygenerowano automatycznie w DATA-Analysis-NLP – odpowiada ustawieniom w aplikacji."),
    "wie im Kurs: yelp['text length'] = yelp['text'].apply(len)": (
        "as in the course: yelp['text length'] = yelp['text'].apply(len)",
        "jak na kursie: yelp['text length'] = yelp['text'].apply(len)"),
    "nur die Klassen {c} (wie im Kurs: nur 1 und 5 Sterne)": (
        "only the classes {c} (as in the course: only 1 and 5 stars)",
        "tylko klasy {c} (jak na kursie: tylko 1 i 5 gwiazdek)"),
    "vokabular": ("vocabulary", "slownik"),
    "vorhersagen_test": ("predictions_test", "prognozy_test"),
    "ohne Kalibrierung – schneller": ("without calibration – faster", "bez kalibracji – szybciej"),
    "Wahrscheinlichkeit für '{c}'": ("probability of '{c}'", "prawdopodobieństwo '{c}'"),
    "Texte aus einer CSV: zuerst Schritt 2 anwenden, dann pipe.predict_proba(df_new[{c}])": (
        "texts from a CSV: apply step 2 first, then pipe.predict_proba(df_new[{c}])",
        "teksty z pliku CSV: najpierw zastosuj krok 2, potem pipe.predict_proba(df_new[{c}])"),
    "Mittelwert je Gruppe": ("mean per group", "średnia w grupie"),
    "Schwelle 0.5": ("threshold 0.5", "próg 0.5"),
    "text_daten": ("text_data", "dane_tekstowe"),
    "Nicht im Vergleich: {m} – eine Klasse hat nur einen Trainingstext (siehe App).": (
        "Not in the comparison: {m} – a class has only one training text (see app).",
        "Poza porównaniem: {m} – jedna klasa ma tylko jeden tekst treningowy (zob. aplikacja)."),
    "ein Label aus nur Leerzeichen zählt als fehlend": (
        "a label of only spaces counts as missing",
        "etykieta złożona tylko ze spacji liczy się jako brakująca"),
}
