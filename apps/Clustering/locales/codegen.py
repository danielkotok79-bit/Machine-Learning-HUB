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
    "inf wie fehlend behandeln": ("treat inf as missing", "traktuj inf jak brak"),
    "Spalten mit zu vielen Lücken": ("columns with too many gaps", "kolumny ze zbyt wieloma brakami"),
    "→ Dummies": ("→ dummy columns", "→ kolumny dummy"),
    "als Text, ohne Leerzeichen am Rand": ("as text, without leading/trailing spaces",
                                           "jako tekst, bez spacji na brzegach"),
    "unbekannt → NaN": ("unknown → NaN", "nieznana → NaN"),
    "für deutsches Excel": ("for German Excel", "dla niemieckiego Excela"),
    "für polnisches Excel": ("for Polish Excel", "dla polskiego Excela"),
    "für Excel in Großbritannien, USA, Australien …": ("for Excel in the UK, US, Australia …",
                                                       "dla Excela w Wielkiej Brytanii, USA, Australii …"),
    "neue_daten.csv": ("new_data.csv", "nowe_dane.csv"),
    "Wichtig: die Vorverarbeitung gehört zum Modell! Die App speichert sie in": (
        "Important: the preprocessing belongs to the model! The app saves it in",
        "Ważne: przetwarzanie wstępne należy do modelu! Aplikacja zapisuje je w"),
    "1) Bibliotheken": ("1) Libraries", "1) Biblioteki"),
    "Daten laden (+ Stichprobe)": ("Load data (+ sample)", "Wczytanie danych (+ próbka)"),
    "2) Datenaufbereitung 1": ("2) Data preparation 1", "2) Przygotowanie danych 1"),
    "4) Datenaufbereitung 2": ("4) Data preparation 2", "4) Przygotowanie danych 2"),
    "6) Modell & Training": ("6) Model & training", "6) Model i trening"),
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
    # --- Clustering-App (K-Means) ---
    "Werte begrenzen": ("clip values", "ogranicz wartości"),
    "Bezeichnung je Zeile – nur zur Anzeige": (
        "row name – for display only",
        "nazwa wiersza – tylko do wyświetlania"),
    "bekannte Einteilung – nur zum Vergleich in Schritt 8, geht NICHT in K-Means": (
        "known grouping – only for the comparison in step 8, NOT used by K-Means",
        "znany podział – tylko do porównania w kroku 8, NIE trafia do K-Means"),
    "Fehlende Werte: Füllwerte aus allen Zeilen (Clustering: kein Train/Test-Split)": (
        "Missing values: fill values from all rows (clustering: no train/test split)",
        "Brakujące wartości: wartości uzupełniające ze wszystkich wierszy (klastrowanie: bez podziału train/test)"),
    "Dummies – Kategorien aus den Daten (neue Daten: unbekannte Kategorie → alle Dummies 0)": (
        "Dummies – categories from the data (new data: unknown category → all dummies 0)",
        "Kolumny dummy – kategorie z danych (nowe dane: nieznana kategoria → wszystkie dummy 0)"),
    "unskaliert – für die Profile in Originaleinheiten": (
        "unscaled – for the profiles in original units",
        "nieskalowane – do profili w jednostkach oryginalnych"),
    "keine Skalierung – große Zahlen bestimmen dann die Abstände": (
        "no scaling – large numbers then dominate the distances",
        "bez skalowania – duże liczby dominują wtedy odległości"),
    "K-Means rechnet mit Abständen → alle Features auf dieselbe Skala": (
        "K-Means works with distances → all features on the same scale",
        "K-Means liczy odległości → wszystkie cechy w tej samej skali"),
    "clustering_daten": ("clustering_data", "dane_klastrowanie"),
    "clustering_daten_skaliert": ("clustering_data_scaled", "dane_klastrowanie_skalowane"),
    "daten_mit_clustern": ("data_with_clusters", "dane_z_klastrami"),
    "skaliert": ("scaled", "skalowane"),
    "Bezeichnung": ("row name", "nazwa wiersza"),
    "bekannte Einteilung (nicht im Modell)": ("known grouping (not in the model)", "znany podział (nie w modelu)"),
    "Abstand zum eigenen Zentrum": ("distance to its own centre", "odległość od własnego centrum"),
    "Unsupervised: K-Means bekommt keine Antwort (kein y)": (
        "Unsupervised: K-Means is never given an answer (no y)",
        "Unsupervised: K-Means nie dostaje odpowiedzi (brak y)"),
    "Cluster-Nummer je Zeile (0 … K−1, die Reihenfolge ist willkürlich)": (
        "cluster number per row (0 … K−1, the order is arbitrary)",
        "numer klastra dla każdego wiersza (0 … K−1, kolejność jest dowolna)"),
    "Zentren (skaliert) – wie im Kurs": ("centres (scaled) – as in the course", "centra (skalowane) – jak w kursie"),
    "keine Elbow-/Silhouette-Kurve (in Schritt 6 ausgeschaltet)": (
        "no elbow/silhouette curve (switched off in step 6)",
        "brak krzywej łokcia/Silhouette (wyłączona w kroku 6)"),
    "Stichprobe – K-Means wird für jedes K neu trainiert": (
        "sample – K-Means is retrained for every K",
        "próbka – K-Means jest trenowany od nowa dla każdego K"),
    "Elbow-Methode": ("Elbow method", "Metoda łokcia"),
    "mittlere Silhouette": ("mean silhouette", "średnia Silhouette"),
    "7) Wahl von K": ("7) Choosing K", "7) Wybór K"),
    "8) Cluster-Auswertung": ("8) Cluster evaluation", "8) Ocena klastrów"),
    "Profile: Mittelwert je Cluster in Originaleinheiten (Dummies: Anteil)": (
        "Profiles: mean per cluster in original units (dummies: share)",
        "Profile: średnia w każdym klastrze w jednostkach oryginalnych (dummy: udział)"),
    "Alle": ("All", "Wszystkie"),
    "Abweichung vom Durchschnitt in Standardabweichungen": (
        "deviation from the average in standard deviations",
        "odchylenie od średniej w odchyleniach standardowych"),
    "PCA auf 2 Dimensionen – nur zur Darstellung (K-Means nutzt alle Features)": (
        "PCA to 2 dimensions – for display only (K-Means uses all features)",
        "PCA do 2 wymiarów – tylko do wizualizacji (K-Means używa wszystkich cech)"),
    "Vergleich mit der bekannten Einteilung '{g}' – K-Means hat sie nie gesehen": (
        "Comparison with the known grouping '{g}' – K-Means never saw it",
        "Porównanie ze znanym podziałem '{g}' – K-Means nigdy go nie widział"),
    "Kreuztabelle Cluster × Gruppe": ("cross table cluster × group", "tabela krzyżowa klaster × grupa"),
    "wie im Kurs: {a} → 0, {b} → 1": ("as in the course: {a} → 0, {b} → 1", "jak w kursie: {a} → 0, {b} → 1"),
    "Achtung: die Cluster-Nummern sind willkürlich – evtl. vertauscht": (
        "careful: the cluster numbers are arbitrary – possibly swapped",
        "uwaga: numery klastrów są dowolne – mogą być zamienione"),
    "jeder Cluster → häufigste Gruppe darin": (
        "each cluster → the most frequent group in it",
        "każdy klaster → najczęstsza grupa w nim"),
    "'{folder}/preprocessing.json' (Spalten-Aktionen, Füllwerte, Dummy-Spalten, Scaler)": (
        "'{folder}/preprocessing.json' (column actions, fill values, dummy columns, scaler)",
        "'{folder}/preprocessing.json' (akcje kolumn, wartości uzupełniające, kolumny dummy, scaler)"),
    "Cluster, Abstand, Hinweis": ("cluster, distance, note", "klaster, odległość, uwaga"),
    "cluster_zuordnung.csv": ("cluster_assignment.csv", "przypisanie_klastrow.csv"),
    "Ohne pipeline.py: neue Daten genauso aufbereiten wie oben (Schritt 2, fill, to_X) und dann": (
        "Without pipeline.py: prepare new data exactly as above (step 2, fill, to_X) and then",
        "Bez pipeline.py: nowe dane przygotować dokładnie jak wyżej (krok 2, fill, to_X), a potem"),
    "NUR transformieren – nichts neu fitten:": (
        "ONLY transform – fit nothing again:",
        "TYLKO transformować – nic nie dopasowywać ponownie:"),
    "Automatisch erzeugt mit DATA-Analysis-Clustering – entspricht den Einstellungen in der App.": (
        "Generated automatically with DATA-Analysis-Clustering – matches the settings in the app.",
        "Wygenerowano automatycznie w DATA-Analysis-Clustering – odpowiada ustawieniom w aplikacji."),
    "Excel: Text wie =1+1 nicht als Formel ausführen": (
        "Excel: do not run text such as =1+1 as a formula",
        "Excel: tekstu jak =1+1 nie wykonywać jako formuły"),
    # --- Audit 5 ---
    "Arbeitskopie fürs Clustering – df (Daten nach Schritt 2) bleibt unverändert": (
        "working copy for the clustering – df (data after step 2) stays unchanged",
        "kopia robocza do klastrowania – df (dane po kroku 2) pozostaje bez zmian"),
    "Zeilen, die ins Clustering gingen": ("rows that went into the clustering", "wiersze, które trafiły do klastrowania"),
}
