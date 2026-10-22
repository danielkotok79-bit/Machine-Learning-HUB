"""Übersetzungen für app.py (Oberfläche): {Deutsch: (Englisch, Polnisch)}"""

T = {
    # Allgemein
    "(keine)": ("(none)", "(brak)"),
    "{label}: keine passenden Spalten": ("{label}: no suitable columns", "{label}: brak pasujących kolumn"),
    "Auswählen …": ("Choose …", "Wybierz …"),
    "Diagramm nicht möglich mit dieser Auswahl: {err}: {exc}": ("Chart not possible with this selection: {err}: {exc}",
                                                               "Wykres niemożliwy przy tym wyborze: {err}: {exc}"),
    "Python-Code zu diesem Schritt": ("Python code for this step", "Kod Pythona do tego kroku"),
    # Spaltennamen der Tabellen
    "Hinweis": ("Note", "Uwaga"),
    "Sicherheit": ("Confidence", "Pewność"),
    "Beispiel": ("Example", "Przykład"),
    "Eindeutig": ("Unique", "Unikalne"),
    "Fehlend": ("Missing", "Brakujące"),
    "Fehlend %": ("Missing %", "Brakujące %"),
    "Neue Spalte": ("New column", "Nowa kolumna"),
    "Spalte": ("Column", "Kolumna"),
    "Aktion": ("Action", "Akcja"),
    "Anteil": ("Share", "Udział"),
    "Art": ("Type", "Rodzaj"),
    "Definition": ("Definition", "Definicja"),
    "Parameter": ("Parameter", "Parametr"),
    "Wirkung": ("Effect", "Działanie"),
    "Klasse": ("Class", "Klasa"),
    "Kodiert": ("Encoded", "Kod"),
    "Modell-Eingänge": ("Model inputs", "Wejścia modelu"),
    "Rolle": ("Role", "Rola"),
    "Strategie": ("Strategy", "Strategia"),
    "Test": ("Test", "Test"),
    "Train": ("Train", "Trening"),
    "Anteil Train": ("Share train", "Udział trening"),
    "Kategorie-Spalte": ("Category column", "Kolumna kategorii"),
    "Kategorien": ("Categories", "Kategorie"),
    "weggelassen (drop_first)": ("left out (drop_first)", "pominięta (drop_first)"),
    "Beispiele": ("Examples", "Przykłady"),
    "Feature": ("Feature", "Cecha"),
    "Streuung": ("Spread", "Rozrzut"),
    "Wichtigkeit": ("Importance", "Ważność"),
    "neue 0/1-Spalten": ("new 0/1 columns", "nowe kolumny 0/1"),
    # Schritt 1
    "Komma  ,": ("Comma  ,", "Przecinek  ,"),
    "Pipe  |": ("Pipe  |", "Kreska  |"),
    "Semikolon  ;": ("Semicolon  ;", "Średnik  ;"),
    "Tabulator": ("Tab", "Tabulator"),
    "Dezimalzeichen": ("Decimal mark", "Separator dziesiętny"),
    "Encoding": ("Encoding", "Kodowanie"),
    "Datei hochladen": ("Upload file", "Prześlij plik"),
    "Pfad auf diesem Rechner": ("Path on this computer", "Ścieżka na tym komputerze"),
    "Quelle": ("Source", "Źródło"),
    "CSV-Datei": ("CSV file", "Plik CSV"),
    "Pfad zur CSV-Datei": ("Path to the CSV file", "Ścieżka do pliku CSV"),
    "z. B. C:\\Daten\\meine_daten.csv oder ../daten/meine_daten.csv": (
        "e.g. C:\\Data\\my_data.csv or ../data/my_data.csv", "np. C:\\Dane\\moje_dane.csv albo ../dane/moje_dane.csv"),
    "Stichprobe geändert – bitte Schritt 2 und 4 erneut übernehmen.": (
        "Sample changed – please apply step 2 and 4 again.", "Zmieniono próbkę – zastosuj ponownie krok 2 i 4."),
    "Lese CSV …": ("Reading CSV …", "Wczytywanie CSV …"),
    "Laden fehlgeschlagen: {exc}": ("Loading failed: {exc}", "Wczytanie nie powiodło się: {exc}"),
    "Bitte zuerst Daten laden (Schritt 1) – dann die Konfiguration importieren.": (
        "Please load data first (step 1) – then import the configuration.",
        "Najpierw wczytaj dane (krok 1) – potem zaimportuj konfigurację."),
    "Ungültige Konfiguration: {exc}": ("Invalid configuration: {exc}", "Nieprawidłowa konfiguracja: {exc}"),
    "Übernommen – jetzt Schritt 2 'Übernehmen' und Schritt 4 'Vorbereiten' klicken.": (
        "Applied – now click 'Apply' in step 2 and 'Prepare' in step 4.",
        "Zastosowano – teraz kliknij 'Zastosuj' w kroku 2 i 'Przygotuj' w kroku 4."),
    " Ungültige Werte ignoriert: {skipped}.": (" Invalid values ignored: {skipped}.",
                                               " Pominięto nieprawidłowe wartości: {skipped}."),
    "1 · Daten laden & Bibliotheken": ("1 · Load data & libraries", "1 · Wczytanie danych i biblioteki"),
    "1) Bibliotheken – werden beim Start automatisch importiert": (
        "1) Libraries – imported automatically at start-up", "1) Biblioteki – importowane automatycznie przy starcie"),
    "CSV-Datei laden": ("Load CSV file", "Wczytaj plik CSV"),
    "Daten laden": ("Load data", "Wczytaj dane"),
    "Optional: Spalten-Beschreibungen laden": ("Optional: load column descriptions", "Opcjonalnie: wczytaj opisy kolumn"),
    "CSV mit 2 Spalten: Spaltenname, Beschreibung. Die Texte erscheinen in Schritt 2 neben den Spalten.": (
        "CSV with 2 columns: column name, description. The texts appear next to the columns in step 2.",
        "CSV z 2 kolumnami: nazwa kolumny, opis. Teksty pojawią się w kroku 2 obok kolumn."),
    "Pfad": ("Path", "Ścieżka"),
    "Beschreibungs-CSV": ("Description CSV", "CSV z opisami"),
    "z. B. ../daten/beschreibungen.csv": ("e.g. ../data/descriptions.csv", "np. ../dane/opisy.csv"),
    "Beschreibungen laden": ("Load descriptions", "Wczytaj opisy"),
    "Fehler: {exc}": ("Error: {exc}", "Błąd: {exc}"),
    "{n} Beschreibungen geladen": ("{n} descriptions loaded", "Wczytano opisy: {n}"),
    "Beschreibung nachschlagen": ("Look up description", "Sprawdź opis"),
    "Noch keine Daten geladen – oben eine CSV-Datei hochladen oder den Pfad eintragen.": (
        "No data loaded yet – upload a CSV file above or enter the path.",
        "Nie wczytano jeszcze danych – prześlij plik CSV powyżej albo wpisz ścieżkę."),
    "Überblick: {name}": ("Overview: {name}", "Przegląd: {name}"),
    "Automatisch erkannt: {v} → gelesen mit Trennzeichen „{v2}“, Dezimalzeichen „{decimal}“, Encoding {encoding}.": (
        "Detected automatically: {v} → read with separator “{v2}”, decimal mark “{decimal}”, encoding {encoding}.",
        "Rozpoznano automatycznie: {v} → wczytano z separatorem „{v2}”, separatorem dziesiętnym „{decimal}”, "
        "kodowaniem {encoding}."),
    "Nur 1 Spalte erkannt – stimmt das Trennzeichen?": ("Only 1 column detected – is the separator right?",
                                                        "Rozpoznano tylko 1 kolumnę – czy separator jest poprawny?"),
    "Zeilen": ("Rows", "Wiersze"),
    "Spalten": ("Columns", "Kolumny"),
    "Zahl / Text": ("Number / text", "Liczba / tekst"),
    "fehlende Zellen": ("missing cells", "brakujące komórki"),
    "Stichprobe aktiv: {n} von {n2} Zeilen": ("Sample active: {n} of {n2} rows", "Aktywna próbka: {n} z {n2} wierszy"),
    "Spalten (df.info)": ("Columns (df.info)", "Kolumny (df.info)"),
    # Schritt 2 – neue Spalten
    "Hinzugefügt: {sources}": ("Added: {sources}", "Dodano: {sources}"),
    " – das trainierte Modell wurde verworfen (Daten geändert).": (
        " – the trained model was discarded (data changed).", " – wytrenowany model odrzucono (dane się zmieniły)."),
    "Nicht entfernt: ": ("Not removed: ", "Nie usunięto: "),
    "'{d}' benutzt {v}": ("'{d}' uses {v}", "'{d}' używa {v}"),
    " – diese Spalte(n) zuerst oder gleichzeitig entfernen.": (" – remove these column(s) first or at the same time.",
                                                               " – usuń te kolumny najpierw albo jednocześnie."),
    "Nicht entfernt: {exc}": ("Not removed: {exc}", "Nie usunięto: {exc}"),
    "Entfernt: {v} – Schritt 2 neu übernommen.": ("Removed: {v} – step 2 applied again.",
                                                  "Usunięto: {v} – krok 2 zastosowano ponownie."),
    "Neue Spalten erstellen": ("Create new columns", "Tworzenie nowych kolumn"),
    "Neue Spalten entstehen aus dem Ergebnis oben (nach den Aktionen) und werden später automatisch auch für neue "
    "Daten berechnet (Schritt 10). Spätere neue Spalten dürfen frühere benutzen.": (
        "New columns are created from the result above (after the actions) and are later calculated automatically "
        "for new data as well (step 10). Later new columns may use earlier ones.",
        "Nowe kolumny powstają z wyniku powyżej (po akcjach) i później są automatycznie liczone także dla nowych "
        "danych (krok 10). Późniejsze nowe kolumny mogą korzystać z wcześniejszych."),
    "Zuerst oben **Übernehmen** klicken – neue Spalten werden aus dem Ergebnis berechnet.": (
        "Click **Apply** above first – new columns are calculated from the result.",
        "Najpierw kliknij powyżej **Zastosuj** – nowe kolumny są liczone z wyniku."),
    "Name der neuen Spalte": ("Name of the new column", "Nazwa nowej kolumny"),
    "z. B. loan_repaid": ("e.g. loan_repaid", "np. loan_repaid"),
    "aus Spalte": ("from column", "z kolumny"),
    "Zuordnung  alt=neu; …  (optional: sonst=…)": ("Mapping  old=new; …  (optional: rest=…)",
                                                   "Przypisanie  stara=nowa; …  (opcjonalnie: rest=…)"),
    "z. B. Fully Paid=1; Charged Off=0": ("e.g. Fully Paid=1; Charged Off=0", "np. Fully Paid=1; Charged Off=0"),
    "Werte in {src}: ": ("Values in {src}: ", "Wartości w {src}: "),
    "z. B. hohe_dti": ("e.g. high_dti", "np. wysokie_dti"),
    "z. B. kredit_zu_einkommen": ("e.g. loan_to_income", "np. kredyt_do_dochodu"),
    "z. B. {example}": ("e.g. {example}", "np. {example}"),
    "Spalten: ": ("Columns: ", "Kolumny: "),
    " · Namen mit Leer-/Sonderzeichen in \\`Backticks\\`": (" · names with spaces/special characters in \\`backticks\\`",
                                                           " · nazwy ze spacjami/znakami specjalnymi w \\`odwrotnych "
                                                           "apostrofach\\`"),
    "z. B. earliest_cr_year": ("e.g. earliest_cr_year", "np. earliest_cr_year"),
    "Ergebnis: zwei Spalten <Name>_jahr und <Name>_monat.": ("Result: two columns <name>_jahr and <name>_monat.",
                                                            "Wynik: dwie kolumny <nazwa>_jahr i <nazwa>_monat."),
    "Spalte hinzufügen": ("Add column", "Dodaj kolumnę"),
    "Kopie von {source}": ("Copy of {source}", "Kopia kolumny {source}"),
    "aus {source}: ": ("from {source}: ", "z {source}: "),
    "Neue Spalten entfernen": ("Remove new columns", "Usuń nowe kolumny"),
    "Entfernen": ("Remove", "Usuń"),
    # Schritt 2
    "2 · Datenaufbereitung 1 – Spalten umwandeln (object → Zahl / Kategorie)": (
        "2 · Data preparation 1 – convert columns (object → number / category)",
        "2 · Przygotowanie danych 1 – konwersja kolumn (object → liczba / kategoria)"),
    "Bitte zuerst in Schritt 1 eine CSV laden.": ("Please load a CSV in step 1 first.", "Najpierw wczytaj plik CSV w kroku 1."),
    "Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann **Übernehmen**. Die "
    "Vorschläge sind automatisch – bitte prüfen: Spalten, die das Ergebnis schon verraten (z. B. erst später "
    "erfasste Daten), sind ein Datenleck und sollten entfernt werden. Text-Spalten mit *behalten* werden in "
    "Schritt 4 zu Dummies.": (
        "One row per CSV column. Choose an **action**, if needed a **parameter** and a **new name**, then click "
        "**Apply**. The suggestions are automatic – please check them: columns that already give away the result "
        "(e.g. data recorded only later) are a data leak and should be dropped. Text columns set to *Keep* become "
        "dummy columns in step 4.",
        "Jeden wiersz na kolumnę CSV. Wybierz **akcję**, w razie potrzeby **parametr** i **nową nazwę**, potem "
        "**Zastosuj**. Propozycje są automatyczne – sprawdź je: kolumny, które zdradzają już wynik (np. dane "
        "zbierane dopiero później), to wyciek danych i należy je usunąć. Kolumny tekstowe z akcją *Zachowaj* "
        "w kroku 4 stają się kolumnami dummy."),
    "Beispiel-Konfiguration laden": ("Load example configuration", "Wczytaj przykładową konfigurację"),
    "Beispiel-Konfiguration geladen (Schritt 2, 4 und 6) – jetzt unten **Übernehmen** klicken.": (
        "Example configuration loaded (step 2, 4 and 6) – now click **Apply** below.",
        "Wczytano przykładową konfigurację (krok 2, 4 i 6) – teraz kliknij poniżej **Zastosuj**."),
    "siehe 'Aktionen & Parameter erklärt'": ("see 'Actions & parameters explained'", "zobacz 'Akcje i parametry – opis'"),
    "Neuer Name": ("New name", "Nowa nazwa"),
    "optional umbenennen, z. B. zip_code": ("optional new name, e.g. zip_code", "opcjonalnie nowa nazwa, np. zip_code"),
    "Beschreibung": ("Description", "Opis"),
    "Aktionen & Parameter erklärt": ("Actions & parameters explained", "Akcje i parametry – opis"),
    "Übernehmen oder neue Spalten verwerfen das trainierte Modell – vorher in Schritt 9 speichern.": (
        "Applying or new columns discard the trained model – save it in step 9 first.",
        "Zastosowanie albo nowe kolumny odrzucają wytrenowany model – najpierw zapisz go w kroku 9."),
    "Übernehmen": ("Apply", "Zastosuj"),
    "Wende Aktionen an …": ("Applying actions …", "Stosowanie akcji …"),
    "Schritt 2 übernommen": ("Step 2 applied", "Krok 2 zastosowany"),
    "Vorschläge wiederherstellen": ("Restore suggestions", "Przywróć propozycje"),
    "Ergebnis: {n} Zeilen × {v} Spalten": ("Result: {n} rows × {v} columns", "Wynik: {n} × {v} (wiersze × kolumny)"),
    "Spalte untersuchen": ("Inspect column", "Zbadaj kolumnę"),
    # Schritt 3
    "Boxplot": ("Box plot", "Wykres pudełkowy"),
    "Countplot": ("Count plot", "Wykres liczności"),
    "Histogramm": ("Histogram", "Histogram"),
    "Korrelation (Heatmap)": ("Correlation (heatmap)", "Korelacja (heatmapa)"),
    "Scatterplot": ("Scatter plot", "Wykres punktowy"),
    "Überblick": ("Overview", "Przegląd"),
    "Anteil je Kategorie": ("Share per category", "Udział według kategorii"),
    "Gruppen-Statistik": ("Group statistics", "Statystyki grup"),
    "Jointplot": ("Joint plot", "Jointplot"),
    "Dichte 2D (KDE)": ("Density 2D (KDE)", "Gęstość 2D (KDE)"),
    "Regression (lmplot)": ("Regression (lmplot)", "Regresja (lmplot)"),
    "Histogramm je Gruppe": ("Histogram per group", "Histogram według grup"),
    "Zwei Zahlenspalten gegeneinander, am Rand die Verteilung jeder Spalte (sns.jointplot).": (
        "Two numeric columns against each other, with the distribution of each column at the edge (sns.jointplot).",
        "Dwie kolumny liczbowe względem siebie, na brzegach rozkład każdej z nich (sns.jointplot)."),
    "Art (kind)": ("Type (kind)", "Rodzaj (kind)"),
    "scatter = Punkte · kde = Dichte (Höhenlinien) · hex = Waben, gut bei sehr vielen Punkten · reg = Punkte mit "
    "Regressionsgerade": (
        "scatter = points · kde = density (contour lines) · hex = hexagons, good for very many points · reg = points "
        "with a regression line",
        "scatter = punkty · kde = gęstość (poziomice) · hex = sześciokąty, dobre przy bardzo wielu punktach · reg = "
        "punkty z prostą regresji"),
    "Farbe nur bei scatter und kde.": ("Colour only with scatter and kde.", "Kolor tylko przy scatter i kde."),
    "Wo liegen die meisten Punkte? Dichte zweier Zahlenspalten (sns.kdeplot) – je heller, desto dichter. Mit Filter "
    "nur eine Gruppe, z. B. iris[iris['species'] == 'setosa'].": (
        "Where are most of the points? Density of two numeric columns (sns.kdeplot) – the lighter, the denser. With "
        "a filter only one group, e.g. iris[iris['species'] == 'setosa'].",
        "Gdzie leży najwięcej punktów? Gęstość dwóch kolumn liczbowych (sns.kdeplot) – im jaśniej, tym gęściej. Z "
        "filtrem tylko jedna grupa, np. iris[iris['species'] == 'setosa']."),
    "Filter: Spalte": ("Filter: column", "Filtr: kolumna"),
    "Filter: Wert": ("Filter: value", "Filtr: wartość"),
    "Flächen füllen (fill)": ("Fill areas (fill)", "Wypełnij obszary (fill)"),
    "Punkte mit Regressionsgerade und 95-%-Band (sns.lmplot) – getrennt nach Farbe (hue) und Teilgrafiken (col), "
    "z. B. int.rate gegen fico je credit.policy und not.fully.paid.": (
        "Points with a regression line and 95 % band (sns.lmplot) – split by colour (hue) and subplots (col), e.g. "
        "int.rate vs. fico by credit.policy and not.fully.paid.",
        "Punkty z prostą regresji i 95-procentowym pasmem (sns.lmplot) – rozdzielone według koloru (hue) i "
        "podwykresów (col), np. int.rate względem fico według credit.policy i not.fully.paid."),
    "Teilgrafiken (col)": ("Subplots (col)", "Podwykresy (col)"),
    "Histogramme einer Zahlenspalte für jede Gruppe übereinander – gleiche Klassen, halb durchsichtig (z. B. fico je "
    "credit.policy).": (
        "Histograms of a numeric column for each group on top of each other – same bins, semi-transparent (e.g. fico "
        "by credit.policy).",
        "Histogramy kolumny liczbowej dla każdej grupy nałożone na siebie – te same przedziały, półprzezroczyste "
        "(np. fico według credit.policy)."),
    "Gruppe": ("Group", "Grupa"),
    "Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis werden sie ersetzt: "
    "{cols}": (
        "The new data already contain columns with the names of the result columns – they are replaced in the "
        "result: {cols}",
        "Nowe dane zawierają już kolumny o nazwach kolumn wyniku – w wyniku zostaną zastąpione: {cols}"),
    "Korrelation mit einer Spalte": ("Correlation with one column", "Korelacja z jedną kolumną"),
    "Pairplot": ("Pair plot", "Pairplot"),
    "3 · Datenanalyse & Visualisierung": ("3 · Data analysis & visualisation", "3 · Analiza danych i wizualizacja"),
    "Bitte zuerst Schritt 2 übernehmen.": ("Please apply step 2 first.", "Najpierw zastosuj krok 2."),
    "Spalten für die Visualisierung auswählen": ("Choose columns for the visualisation", "Wybierz kolumny do wizualizacji"),
    "alle": ("all", "wszystkie"),
    "nur Zahlenspalten": ("number columns only", "tylko kolumny liczbowe"),
    "keine": ("none", "żadne"),
    "{n} von {v} Spalten ausgewählt · {n2} Zeilen": ("{n} of {v} columns selected · {n2} rows",
                                                    "Wybrano {n} z {v} kolumn · wiersze: {n2}"),
    " (keine Auswahl → alle Spalten)": (" (nothing selected → all columns)", " (brak wyboru → wszystkie kolumny)"),
    "Diagramm": ("Chart", "Wykres"),
    "**Zahlen** – `df.describe()`": ("**Numbers** – `df.describe()`", "**Liczby** – `df.describe()`"),
    "**Kategorien / Text**": ("**Categories / text**", "**Kategorie / tekst**"),
    "Keine fehlenden Werte.": ("No missing values.", "Brak brakujących wartości."),
    "x (Kategorie)": ("x (category)", "x (kategoria)"),
    "Farbe (hue)": ("Colour (hue)", "Kolor (hue)"),
    "Häufigkeit": ("Frequency", "Częstość"),
    "Reihenfolge": ("Order", "Kolejność"),
    "Wert": ("Value", "Wartość"),
    "Spalte (Zahl)": ("Column (number)", "Kolumna (liczba)"),
    "KDE-Linie": ("KDE line", "Linia KDE"),
    "Ausreißer abschneiden (1–99 %)": ("Cut off outliers (1–99 %)", "Obetnij wartości odstające (1–99 %)"),
    "y (Zahl)": ("y (number)", "y (liczba)"),
    "max. Punkte": ("max. points", "maks. punktów"),
    "Numerische Spalten": ("Numeric columns", "Kolumny liczbowe"),
    "Methode": ("Method", "Metoda"),
    "Werte anzeigen (annot)": ("Show values (annot)", "Pokaż wartości (annot)"),
    "nur untere Hälfte": ("lower half only", "tylko dolna połowa"),
    "Tabelle (df.corr)": ("Table (df.corr)", "Tabela (df.corr)"),
    "Mindestens 2 numerische Spalten wählen.": ("Choose at least 2 numeric columns.", "Wybierz co najmniej 2 kolumny liczbowe."),
    "Wie stark hängt jede Zahlenspalte mit der gewählten Spalte zusammen? (+1 = gleichläufig, −1 = gegenläufig, "
    "0 = kein linearer Zusammenhang) – z. B. um eine Gruppenspalte für fehlende Werte zu finden oder Features zu "
    "vergleichen.": (
        "How strongly is each number column related to the chosen column? (+1 = same direction, −1 = opposite "
        "direction, 0 = no linear relationship) – e.g. to find a group column for missing values or to compare "
        "features.",
        "Jak silnie każda kolumna liczbowa wiąże się z wybraną kolumną? (+1 = zgodnie, −1 = przeciwnie, "
        "0 = brak zależności liniowej) – np. aby znaleźć kolumnę grupującą dla braków albo porównać cechy."),
    "= 1 (positive Klasse)": ("= 1 (positive class)", "= 1 (klasa pozytywna)"),
    "Keine numerischen Spalten zum Vergleichen.": ("No numeric columns to compare.", "Brak kolumn liczbowych do porównania."),
    "Numerische Spalten (2–6 empfohlen)": ("Numeric columns (2–6 recommended)", "Kolumny liczbowe (zalecane 2–6)"),
    "max. Zeilen (Stichprobe)": ("max. rows (sample)", "maks. wierszy (próbka)"),
    "Mehr als 8 Spalten → sehr langsam und unübersichtlich.": ("More than 8 columns → very slow and hard to read.",
                                                              "Więcej niż 8 kolumn → bardzo wolno i nieczytelnie."),
    "Pairplot erstellen": ("Create pair plot", "Utwórz pairplot"),
    "Zeichne Pairplot …": ("Drawing pair plot …", "Rysowanie pairplotu …"),
    "Anteil einer Klasse je Kategorie – z. B. Ausfallquote je Beschäftigungsdauer.": (
        "Share of a class per category – e.g. default rate per length of employment.",
        "Udział klasy w każdej kategorii – np. odsetek niespłaconych kredytów według stażu pracy."),
    "Zielspalte (2 Werte)": ("Target column (2 values)", "Kolumna docelowa (2 wartości)"),
    "Anteil von": ("Share of", "Udział"),
    "je Kategorie": ("per category", "według kategorii"),
    "Tabelle": ("Table", "Tabela"),
    "gruppieren nach": ("group by", "grupuj według"),
    "Statistik für": ("Statistics for", "Statystyki dla"),
    # Schritt 4
    "Konstante: Wert · Gruppen-Mittelwert: Gruppenspalte": ("Constant: value · group mean: group column",
                                                            "Stała: wartość · średnia grupowa: kolumna grupująca"),
    "4 · Datenaufbereitung 2 – fehlende Werte, Label & Features": (
        "4 · Data preparation 2 – missing values, label & features",
        "4 · Przygotowanie danych 2 – braki, etykieta i cechy"),
    "Label (y) & Problemtyp": ("Label (y) & problem type", "Etykieta (y) i typ problemu"),
    "Label – was soll vorhergesagt werden?": ("Label – what should be predicted?", "Etykieta – co ma być przewidywane?"),
    "Problemtyp": ("Problem type", "Typ problemu"),
    "Positive Klasse (= 1)": ("Positive class (= 1)", "Klasa pozytywna (= 1)"),
    "Diese Klasse wird zu 1 kodiert, die andere zu 0 (z. B. 'Fully Paid' → 1, 'Charged Off' → 0).": (
        "This class is encoded as 1, the other as 0 (e.g. 'Fully Paid' → 1, 'Charged Off' → 0).",
        "Ta klasa jest kodowana jako 1, druga jako 0 (np. 'Fully Paid' → 1, 'Charged Off' → 0)."),
    "Verteilung von {label}": ("Distribution of {label}", "Rozkład {label}"),
    "Features (X)": ("Features (X)", "Cechy (X)"),
    "Features für das Modell": ("Features for the model", "Cechy dla modelu"),
    "{leak} stammt aus derselben CSV-Spalte wie das Label → nicht als Feature verwenden (Datenleck).": (
        "{leak} comes from the same CSV column as the label → do not use it as a feature (data leak).",
        "{leak} pochodzi z tej samej kolumny CSV co etykieta → nie używaj jako cechy (wyciek danych)."),
    "Feature (X)": ("Feature (X)", "Cecha (X)"),
    "Label (y)": ("Label (y)", "Etykieta (y)"),
    "Kategorie → Dummies": ("Category → dummies", "Kategoria → dummy"),
    "Alle Parameter – {n} Features → ca. {n_inputs} Modell-Eingänge": (
        "All parameters – {n} features → about {n_inputs} model inputs",
        "Wszystkie parametry – cechy: {n} → wejścia modelu: ok. {n_inputs}"),
    "Viele Kategorien → viele Dummy-Spalten: {many}. Tipp: Schritt 2 'Top-N' oder entfernen.": (
        "Many categories → many dummy columns: {many}. Tip: step 2 'Keep top N' or drop.",
        "Dużo kategorii → dużo kolumn dummy: {many}. Wskazówka: w kroku 2 'Zachowaj top N' albo usuń."),
    "Fehlende Werte (NULL)": ("Missing values (NULL)", "Brakujące wartości (NULL)"),
    "Die Füllwerte werden erst **nach** dem Train/Test-Split berechnet – nur aus den Trainingsdaten – und dann auf "
    "Train, Test und später neue Daten angewendet. So fließt keine Information aus den Testdaten ins Modell (kein "
    "Datenleck). Dasselbe gilt für die Kategorien (Dummies) und den Scaler.": (
        "The fill values are calculated only **after** the train/test split – from the training data only – and "
        "then applied to train, test and later new data. This way no information from the test data flows into "
        "the model (no data leakage). The same applies to the categories (dummies) and the scaler.",
        "Wartości uzupełniające są liczone dopiero **po** podziale train/test – tylko z danych treningowych – i "
        "potem stosowane do train, test i później nowych danych. Dzięki temu żadna informacja z danych testowych "
        "nie trafia do modelu (brak wycieku danych). To samo dotyczy kategorii (dummy) i skalera."),
    "{n} Zeilen ohne Label werden automatisch entfernt.": ("{n} rows without a label are removed automatically.",
                                                          "Wiersze bez etykiety zostaną automatycznie usunięte: {n}."),
    "Strategien erklärt": ("Strategies explained", "Strategie – opis"),
    "Hilfe: Welche Spalte passt als Gruppe? – Korrelation mit der lückenhaften Spalte": (
        "Help: which column fits as a group? – correlation with the column that has gaps",
        "Pomoc: która kolumna nadaje się na grupę? – korelacja z kolumną z brakami"),
    "Spalte mit Lücken": ("Column with gaps", "Kolumna z brakami"),
    "Keine weiteren Zahlenspalten zum Vergleichen.": ("No other number columns to compare.",
                                                      "Brak innych kolumn liczbowych do porównania."),
    "verschiedene Werte": ("distinct values", "różne wartości"),
    "Am stärksten mit {mc} zusammen hängt **{v}** (r = {v2:+.2f}). Strategie *{m_group}* mit Parameter `{v}` "
    "füllt jede Lücke mit dem Mittelwert von {mc} in derselben {v}-Gruppe.": (
        "**{v}** is most strongly related to {mc} (r = {v2:+.2f}). Strategy *{m_group}* with parameter `{v}` fills "
        "each gap with the mean of {mc} in the same {v} group.",
        "Najsilniej z {mc} wiąże się **{v}** (r = {v2:+.2f}). Strategia *{m_group}* z parametrem `{v}` uzupełnia "
        "każdy brak średnią {mc} w tej samej grupie {v}."),
    "Keine fehlenden Werte in den gewählten Features.": ("No missing values in the selected features.",
                                                         "Brak braków w wybranych cechach."),
    "Train/Test-Split & Skalierung": ("Train/test split & scaling", "Podział train/test i skalowanie"),
    "Klassenverhältnis in Train und Test gleich halten (empfohlen bei unausgewogenen Klassen)": (
        "Keep the class ratio equal in train and test (recommended for imbalanced classes)",
        "Zachowaj te same proporcje klas w train i test (zalecane przy niezrównoważonych klasach)"),
    "Skalierung": ("Scaling", "Skalowanie"),
    "drop_first (Dummies)": ("drop_first (dummies)", "drop_first (dummy)"),
    "Label skalieren": ("Scale label", "Skaluj etykietę"),
    "Neu vorbereiten verwirft das trainierte Modell – vorher in Schritt 9 speichern.": (
        "Preparing again discards the trained model – save it in step 9 first.",
        "Ponowne przygotowanie odrzuca wytrenowany model – najpierw zapisz go w kroku 9."),
    "Daten für ML vorbereiten": ("Prepare data for ML", "Przygotuj dane do ML"),
    "Bereite Daten vor …": ("Preparing data …", "Przygotowywanie danych …"),
    "Daten vorbereitet": ("Data prepared", "Dane przygotowane"),
    "Die Einstellungen oben wurden seit dem Vorbereiten geändert → erneut **Daten für ML vorbereiten**.": (
        "The settings above changed since preparing → click **Prepare data for ML** again.",
        "Ustawienia powyżej zmieniły się od przygotowania → ponownie kliknij **Przygotuj dane do ML**."),
    "Fertig: X_train {shape} · X_test {shape2} · {n} Features → {n2} Modell-Eingänge": (
        "Done: X_train {shape} · X_test {shape2} · {n} features → {n2} model inputs",
        "Gotowe: X_train {shape} · X_test {shape2} · cechy: {n} → wejścia modelu: {n2}"),
    "get_dummies: {n} Kategorie-Spalten → {n_dummy} Dummy-Spalten (0/1)": (
        "get_dummies: {n} category columns → {n_dummy} dummy columns (0/1)",
        "get_dummies: kolumny kategorii: {n} → kolumny dummy (0/1): {n_dummy}"),
    "Jede Kategorie wird zu einer eigenen 0/1-Spalte, z. B. sub_grade → sub_grade_A2, sub_grade_A3 … Mit drop_first "
    "entfällt die erste Kategorie – sie ist daran erkennbar, dass alle anderen 0 sind. Die Kategorien stammen aus "
    "den Trainingsdaten; unbekannte Werte → alle 0.": (
        "Each category becomes its own 0/1 column, e.g. sub_grade → sub_grade_A2, sub_grade_A3 … With drop_first the "
        "first category is left out – it is recognised by all others being 0. The categories come from the "
        "training data; unknown values → all 0.",
        "Każda kategoria staje się osobną kolumną 0/1, np. sub_grade → sub_grade_A2, sub_grade_A3 … Przy drop_first "
        "pierwsza kategoria jest pomijana – rozpoznaje się ją po tym, że wszystkie inne są 0. Kategorie pochodzą "
        "z danych treningowych; nieznane wartości → wszystkie 0."),
    "**X nach Dummy-Kodierung (unskaliert, erste Zeilen)**": ("**X after dummy encoding (unscaled, first rows)**",
                                                             "**X po kodowaniu dummy (bez skalowania, pierwsze wiersze)**"),
    "**X_train nach {type} (erste Zeilen)**": ("**X_train after {type} (first rows)**",
                                               "**X_train po {type} (pierwsze wiersze)**"),
    # Schritt 5
    "Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)": (
        "Data after step 2 – all changes and new columns (readable, e.g. for Excel)",
        "Dane po kroku 2 – wszystkie zmiany i nowe kolumny (czytelne, np. do Excela)"),
    "ML-fertige Daten nach Schritt 4 – Lücken gefüllt, Dummies, Label kodiert, Spalte 'split' (train/test)": (
        "ML-ready data after step 4 – gaps filled, dummies, label encoded, column 'split' (train/test)",
        "Dane gotowe do ML po kroku 4 – braki uzupełnione, dummy, etykieta zakodowana, kolumna 'split' (train/test)"),
    "wie zuvor, zusätzlich skaliert – so sieht das Modell die Daten": (
        "as before, additionally scaled – this is how the model sees the data",
        "jak wyżej, dodatkowo przeskalowane – tak model widzi dane"),
    "CSV international (Komma, Dezimalpunkt)": ("CSV international (comma, decimal point)",
                                                "CSV międzynarodowy (przecinek, kropka dziesiętna)"),
    "CSV für deutsches Excel (Semikolon, Dezimalkomma)": (
        "CSV for Excel – UK, US, Australia … (comma, decimal point)",
        "CSV dla polskiego Excela (średnik, przecinek dziesiętny)"),
    "5 · Daten speichern": ("5 · Save data", "5 · Zapis danych"),
    "Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen Programm.": (
        "Save the processed data as a CSV file – to keep working in Excel or another program.",
        "Zapisz przetworzone dane jako plik CSV – do dalszej pracy w Excelu albo innym programie."),
    "Was speichern?": ("What to save?", "Co zapisać?"),
    "Format": ("Format", "Format"),
    "Für ML-fertige Daten zuerst Schritt 4 ausführen (**Daten für ML vorbereiten**).": (
        "For ML-ready data run step 4 first (**Prepare data for ML**).",
        "Aby zapisać dane gotowe do ML, najpierw wykonaj krok 4 (**Przygotuj dane do ML**)."),
    "Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.": (
        "Sample active: only {n} of {n2} rows will be saved.",
        "Aktywna próbka: zapisana zostanie tylko próbka – {n} z {n2} wierszy."),
    "Label kodiert: ": ("Label encoded: ", "Zakodowana etykieta: "),
    "daten_aufbereitet": ("prepared_data", "dane_przygotowane"),
    "ml_daten": ("ml_data", "dane_ml"),
    "ml_daten_skaliert": ("ml_data_scaled", "dane_ml_skalowane"),
    "mein_modell": ("my_model", "moj_model"),
    "CSV erstellen": ("Create CSV", "Utwórz CSV"),
    "Erstelle CSV …": ("Creating CSV …", "Tworzenie CSV …"),
    "{name}: {n} Zeilen × {v} Spalten · {v2:.1f} MB": ("{name}: {n} rows × {v} columns · {v2:.1f} MB",
                                                       "{name}: {n} × {v} (wiersze × kolumny) · {v2:.1f} MB"),
    "CSV herunterladen": ("Download CSV", "Pobierz CSV"),
    "… oder direkt in diesen Ordner speichern": ("… or save directly to this folder", "… albo zapisz bezpośrednio w tym folderze"),
    "Die Datei **{target}** gibt es schon.": ("The file **{target}** already exists.", "Plik **{target}** już istnieje."),
    "Ja, vorhandene Datei überschreiben": ("Yes, overwrite the existing file", "Tak, nadpisz istniejący plik"),
    "In Ordner speichern": ("Save to folder", "Zapisz w folderze"),
    "Speichern fehlgeschlagen: {exc}": ("Saving failed: {exc}", "Zapis nie powiódł się: {exc}"),
    "Gespeichert: {target}": ("Saved: {target}", "Zapisano: {target}"),
    "Für Vorhersagen (Schritt 10) die **ursprüngliche** CSV-Datei verwenden, nicht diese Datei – das gespeicherte "
    "Modell wiederholt die Aufbereitung selbst.": (
        "For predictions (step 10) use the **original** CSV file, not this file – the saved model repeats the "
        "preparation itself.",
        "Do prognoz (krok 10) używaj **oryginalnego** pliku CSV, nie tego pliku – zapisany model sam powtarza "
        "przygotowanie danych."),
    "Texte, die mit =, +, -, @ beginnen, bekommen ein ' davor, damit Excel sie nicht als Formel ausführt.": (
        "Text starting with =, +, -, @ gets a ' in front so that Excel does not run it as a formula.",
        "Teksty zaczynające się od =, +, -, @ dostają ' na początku, żeby Excel nie wykonał ich jako formuły."),
    # Schritt 6
    "6 · Modell & Training": ("6 · Model & training", "6 · Model i trening"),
    "Bitte zuerst Schritt 4 ausführen (Daten für ML vorbereiten).": ("Please run step 4 first (prepare data for ML).",
                                                                     "Najpierw wykonaj krok 4 (przygotuj dane do ML)."),
    "Validierung": ("Validation", "Walidacja"),
    "Anteil aus Training (sauberer)": ("Share of training data (cleaner)", "Część danych treningowych (czyściej)"),
    "Testdaten": ("Test data", "Dane testowe"),
    "class_weight='balanced' – hilft der seltenen Klasse (z. B. Charged Off)": (
        "class_weight='balanced' – helps the rare class (e.g. Charged Off)",
        "class_weight='balanced' – pomaga rzadkiej klasie (np. Charged Off)"),
    "Seed": ("Seed", "Ziarno (seed)"),
    "Während des Trainings nichts anklicken – jede Eingabe bricht das Training ab.": (
        "Do not click anything during training – any input stops the training.",
        "Nie klikaj niczego podczas treningu – każda zmiana przerywa trening."),
    "Training starten": ("Start training", "Rozpocznij trening"),
    "Starte Training …": ("Starting training …", "Uruchamianie treningu …"),
    "Training fehlgeschlagen: {exc}": ("Training failed: {exc}", "Trening nie powiódł się: {exc}"),
    # Schritt 7
    "Bitte zuerst in Schritt 6 trainieren.": ("Please train in step 6 first.", "Najpierw wytrenuj model w kroku 6."),
    # Schritt 8
    "Explained Variance": ("Explained variance", "Wyjaśniona wariancja"),
    "MAE = {x:.1%} vom Mittelwert des Labels ({x2})": ("MAE = {x:.1%} of the label mean ({x2})",
                                                       "MAE = {x:.1%} średniej etykiety ({x2})"),
    "Accuracy": ("Accuracy", "Dokładność (accuracy)"),
    "Mittel über alle Klassen – fair bei Ungleichgewicht": ("Average over all classes – fair for imbalanced classes",
                                                            "Średnia ze wszystkich klas – uczciwa przy niezrównoważeniu"),
    "0.5 = Zufall, 1.0 = perfekt; unabhängig von der Schwelle": ("0.5 = chance, 1.0 = perfect; independent of the threshold",
                                                                "0.5 = losowo, 1.0 = idealnie; niezależnie od progu"),
    "classification_report (Text)": ("classification_report (text)", "classification_report (tekst)"),
    ":green[:material/check_circle:] richtig": (":green[:material/check_circle:] correct",
                                                ":green[:material/check_circle:] trafnie"),
    ":red[:material/cancel:] falsch": (":red[:material/cancel:] wrong", ":red[:material/cancel:] błędnie"),
    "8 · Modellevaluation": ("8 · Model evaluation", "8 · Ewaluacja modelu"),
    "Entscheidungsschwelle: P({v}) > Schwelle → '{v}'": ("Decision threshold: P({v}) > threshold → '{v}'",
                                                         "Próg decyzyjny: P({v}) > próg → '{v}'"),
    "Standard 0.5. Höher = strenger für Klasse 1. Wird beim Speichern mitgespeichert.": (
        "Default 0.5. Higher = stricter for class 1. Saved together with the model.",
        "Domyślnie 0.5. Wyższy = surowiej dla klasy 1. Zapisywany razem z modelem."),
    "Bewertet auf {n} Testzeilen (X_test, y_test)": ("Evaluated on {n} test rows (X_test, y_test)",
                                                     "Ocena na wierszach testowych: {n} (X_test, y_test)"),
    "**Schwelle verstehen:** Das Modell liefert für jede Zeile eine Wahrscheinlichkeit für '{v}'. Liegt sie über "
    "der Schwelle, lautet die Vorhersage '{v}'. Eine höhere Schwelle erkennt mehr '{v2}' (Recall Klasse 0 steigt), "
    "auf Kosten von Fehlalarmen. Bei unausgewogenen Klassen lohnt es sich, die Schwelle zu verschieben oder in "
    "Schritt 6 Klassengewichte zu nutzen.": (
        "**Understanding the threshold:** the model gives a probability of '{v}' for each row. If it is above the "
        "threshold, the prediction is '{v}'. A higher threshold detects more '{v2}' (recall of class 0 rises), at "
        "the cost of false alarms. With imbalanced classes it is worth moving the threshold or using class "
        "weights in step 6.",
        "**Jak działa próg:** model podaje dla każdego wiersza prawdopodobieństwo '{v}'. Jeśli jest powyżej "
        "progu, prognoza to '{v}'. Wyższy próg wykrywa więcej '{v2}' (rośnie recall klasy 0), kosztem fałszywych "
        "alarmów. Przy niezrównoważonych klasach warto przesunąć próg albo w kroku 6 użyć wag klas."),
    "Einzelfall prüfen": ("Check a single case", "Sprawdź pojedynczy przypadek"),
    "Zufälligen Datensatz aus den Testdaten ziehen und die Vorhersage mit der Wahrheit vergleichen.": (
        "Draw a random record from the test data and compare the prediction with the truth.",
        "Wylosuj rekord z danych testowych i porównaj prognozę z rzeczywistością."),
    "Zufälligen Datensatz ziehen": ("Draw random record", "Wylosuj rekord"),
    "tatsächlich": ("actual", "rzeczywista"),
    "Abweichung: **{v}{x}**": ("Difference: **{v}{x}**", "Różnica: **{v}{x}**"),
    "P({v}) = **{x:.1%}** · Schwelle {thr:.2f} · ": ("P({v}) = **{x:.1%}** · threshold {thr:.2f} · ",
                                                      "P({v}) = **{x:.1%}** · próg {thr:.2f} · "),
    "Sicherheit **{x:.1%}** · ": ("Confidence **{x:.1%}** · ", "Pewność **{x:.1%}** · "),
    "Welche Features sind wichtig?": ("Which features matter?", "Które cechy są ważne?"),
    "Permutation Importance: Werte eines Features werden gemischt – je stärker das Modell dadurch schlechter wird, "
    "desto wichtiger ist das Feature. Dummies einer Kategorie werden gemeinsam gemischt.": (
        "Permutation importance: the values of a feature are shuffled – the more the model gets worse, the more "
        "important the feature. Dummies of one category are shuffled together.",
        "Ważność permutacyjna: wartości cechy są tasowane – im bardziej model się przez to pogarsza, tym ważniejsza "
        "cecha. Kolumny dummy jednej kategorii są tasowane razem."),
    "Permutation Importance berechnen (Stichprobe 2000 Zeilen)": ("Compute permutation importance (sample of 2000 rows)",
                                                                  "Oblicz ważność permutacyjną (próbka 2000 wierszy)"),
    "Berechne …": ("Computing …", "Obliczanie …"),
    # Schritt 9
    "9 · Modell speichern & laden": ("9 · Save & load model", "9 · Zapis i odczyt modelu"),
    "Speichern": ("Save", "Zapisz"),
    "Noch kein trainiertes Modell in dieser Sitzung (Schritt 6).": ("No trained model in this session yet (step 6).",
                                                                    "W tej sesji nie ma jeszcze wytrenowanego modelu "
                                                                    "(krok 6)."),
    "Name (Ordner unter models/)": ("Name (folder in models/)", "Nazwa (folder w models/)"),
    "Prüfsummen von Trainingszeilen mitspeichern (Schritt 10 erkennt damit Trainingsdaten)": (
        "Save checksums of training rows (step 10 uses them to detect training data)",
        "Zapisz sumy kontrolne wierszy treningowych (krok 10 rozpoznaje dzięki nim dane treningowe)"),
    "Bis zu 3000 Prüfsummen (Hashes) zufälliger Trainingszeilen. Aus einer Prüfsumme lässt sich die Zeile nicht "
    "zurückrechnen – wer die Originaldaten hat, kann aber prüfen, ob eine bestimmte Zeile im Training war. Bei "
    "vertraulichen Daten ausschalten, bevor das Modell weitergegeben wird.": (
        "Up to 3000 checksums (hashes) of random training rows. A row cannot be reconstructed from a checksum – but "
        "anyone with the original data can check whether a specific row was in training. Switch this off for "
        "confidential data before passing the model on.",
        "Do 3000 sum kontrolnych (hashy) losowych wierszy treningowych. Z sumy kontrolnej nie da się odtworzyć "
        "wiersza – ale kto ma dane źródłowe, może sprawdzić, czy dany wiersz był w treningu. Przy danych poufnych "
        "wyłącz tę opcję przed przekazaniem modelu."),
    "Unter `models/{safe}` liegt schon ein Modell – Speichern ersetzt es.": (
        "There is already a model in `models/{safe}` – saving replaces it.",
        "W `models/{safe}` jest już model – zapis go zastąpi."),
    "Ja, vorhandenes Modell überschreiben": ("Yes, overwrite the existing model", "Tak, nadpisz istniejący model"),
    "Speichere …": ("Saving …", "Zapisywanie …"),
    "Gespeichert: {folder}": ("Saved: {folder}", "Zapisano: {folder}"),
    "Modell + Rezept als ZIP": ("Model + recipe as ZIP", "Model + recepta jako ZIP"),
    "Komplette Pipeline als Python-Skript (für Jupyter)": ("Complete pipeline as a Python script (for Jupyter)",
                                                           "Cały pipeline jako skrypt Pythona (do Jupytera)"),
    "Laden": ("Load", "Wczytaj"),
    "Aus dem Ordner models/": ("From the folder models/", "Z folderu models/"),
    "ZIP hochladen": ("Upload ZIP", "Prześlij ZIP"),
    "Noch keine gespeicherten Modelle.": ("No saved models yet.", "Brak zapisanych modeli."),
    "Gespeicherte Modelle": ("Saved models", "Zapisane modele"),
    "'{sel}' geladen → Schritt 10": ("'{sel}' loaded → step 10", "Wczytano '{sel}' → krok 10"),
    "ZIP laden": ("Load ZIP", "Wczytaj ZIP"),
    "ZIP geladen → Schritt 10": ("ZIP loaded → step 10", "Wczytano ZIP → krok 10"),
    "**Kennzahlen (Testdaten)**": ("**Metrics (test data)**", "**Metryki (dane testowe)**"),
    "**Training**": ("**Training**", "**Trening**"),
    # Schritt 10
    "\n**So werden neue Daten verarbeitet**\n\n1. **Nichts neu fitten.** Neue Daten werden mit dem *gespeicherten "
    "Rezept* aufbereitet – gleiche\n   Spalten-Aktionen und neue Spalten, gleiche Füllwerte, gleiche Dummy-Spalten in "
    "gleicher Reihenfolge,\n   gleicher Scaler. Würde man `get_dummies` oder `MinMaxScaler` auf den neuen Daten neu "
    "rechnen, passen\n   Spalten und Skalen nicht mehr zum Modell.\n2. **Gleiche Rohspalten wie im Training.** Das "
    "Modell bestimmt die Features – eine Auswahl ist hier nicht\n   mehr möglich. Die Tabelle unten zeigt, welche "
    "CSV-Spalten nötig sind (Vorlage zum Download).\n   Fehlen Spalten → klare Fehlermeldung. Zusätzliche Spalten "
    "(auch das Label) werden ignoriert.\n3. **Abweichungen werden behandelt und gemeldet:** Lücken → Füllwerte aus "
    "dem Training (jede Zeile braucht\n   für eine Vorhersage einen Wert – auch dort, wo im Training Zeilen mit "
    "Lücken entfernt wurden);\n   unbekannte Kategorien → alle Dummies 0; Werte außerhalb des Trainingsbereichs → "
    "Hinweis.\n4. **Interpretation und Fazit:** Bei binär = Wahrscheinlichkeit für die positive Klasse + Schwelle. "
    "Ist das\n   Label in den neuen Daten enthalten, wird automatisch evaluiert. Das Fazit am Ende bewertet die\n   "
    "Datenqualität und erklärt, was das Ergebnis aussagt – auch, ob die Daten schon im Training waren.\n": (
        "\n**How new data are processed**\n\n"
        "1. **Nothing is fitted again.** New data are prepared with the *saved recipe* – the same column actions\n"
        "   and new columns, the same fill values, the same dummy columns in the same order, the same scaler. If\n"
        "   `get_dummies` or `MinMaxScaler` were recomputed on the new data, columns and scales would no longer\n"
        "   match the model.\n"
        "2. **The same raw columns as in training.** The model determines the features – no selection is possible\n"
        "   here. The table below shows which CSV columns are needed (template to download). Missing columns →\n"
        "   clear error message. Additional columns (including the label) are ignored.\n"
        "3. **Deviations are handled and reported:** gaps → fill values from training (every row needs a value\n"
        "   for a prediction – even where rows with gaps were removed in training); unknown categories → all\n"
        "   dummies 0; values outside the training range → note.\n"
        "4. **Interpretation and conclusion:** for binary = probability of the positive class + threshold. If the\n"
        "   label is included in the new data, it is evaluated automatically. The conclusion at the end rates the\n"
        "   data quality and explains what the result means – including whether the data were already in training.\n",
        "\n**Jak przetwarzane są nowe dane**\n\n"
        "1. **Nic nie jest dopasowywane od nowa.** Nowe dane są przygotowywane według *zapisanej recepty* – te same\n"
        "   akcje na kolumnach i nowe kolumny, te same wartości uzupełniające, te same kolumny dummy w tej samej\n"
        "   kolejności, ten sam skaler. Gdyby `get_dummies` albo `MinMaxScaler` liczyć od nowa na nowych danych,\n"
        "   kolumny i skale nie pasowałyby już do modelu.\n"
        "2. **Te same surowe kolumny co w treningu.** Model określa cechy – tutaj nie można ich już wybierać.\n"
        "   Tabela poniżej pokazuje, które kolumny CSV są potrzebne (szablon do pobrania). Brak kolumn → jasny\n"
        "   komunikat o błędzie. Dodatkowe kolumny (także etykieta) są pomijane.\n"
        "3. **Odchylenia są obsługiwane i zgłaszane:** braki → wartości uzupełniające z treningu (każdy wiersz\n"
        "   potrzebuje wartości do prognozy – także tam, gdzie w treningu usunięto wiersze z brakami); nieznane\n"
        "   kategorie → wszystkie dummy 0; wartości poza zakresem treningowym → uwaga.\n"
        "4. **Interpretacja i podsumowanie:** przy binarnym = prawdopodobieństwo klasy pozytywnej + próg. Jeśli\n"
        "   etykieta jest w nowych danych, ewaluacja odbywa się automatycznie. Podsumowanie na końcu ocenia jakość\n"
        "   danych i wyjaśnia, co oznacza wynik – także to, czy dane były już w treningu.\n"),
    "**Lesart:** `P({pos})` ist die vom Modell geschätzte Wahrscheinlichkeit für '{pos}'. Über der Schwelle "
    "{thr:.2f} → Vorhersage '{pos}', sonst '{neg}'. Werte nahe 0 oder 1 = sicher, um {thr:.2f} = unsicher.": (
        "**How to read it:** `P({pos})` is the probability of '{pos}' estimated by the model. Above the threshold "
        "{thr:.2f} → prediction '{pos}', otherwise '{neg}'. Values close to 0 or 1 = certain, around {thr:.2f} = "
        "uncertain.",
        "**Jak czytać:** `P({pos})` to oszacowane przez model prawdopodobieństwo '{pos}'. Powyżej progu {thr:.2f} → "
        "prognoza '{pos}', inaczej '{neg}'. Wartości bliskie 0 lub 1 = pewne, w okolicy {thr:.2f} = niepewne."),
    "\n\n**Ergebnis:** {n} von {n2} Zeilen ({v:.1%}) werden als '{pos}' vorhergesagt, {n3} als '{neg}'.": (
        "\n\n**Result:** {n} of {n2} rows ({v:.1%}) are predicted as '{pos}', {n3} as '{neg}'.",
        "\n\n**Wynik:** {n} z {n2} wierszy ({v:.1%}) ma prognozę '{pos}', {n3} – '{neg}'."),
    " Zum Vergleich: In den Trainingsdaten waren tatsächlich {train_share:.1%} '{pos}'. Ein Unterschied zeigt, wie "
    "oft das Modell danebenliegt oder dass die Daten anders sind – Einordnung im Fazit unten.": (
        " For comparison: in the training data {train_share:.1%} were actually '{pos}'. A difference shows how often "
        "the model is wrong or that the data are different – see the conclusion below.",
        " Dla porównania: w danych treningowych faktycznie {train_share:.1%} to '{pos}'. Różnica pokazuje, jak często "
        "model się myli albo że dane są inne – ocena w podsumowaniu poniżej."),
    "**Lesart:** `Vorhersage` = Klasse mit der höchsten Wahrscheinlichkeit, `Sicherheit` = diese "
    "Wahrscheinlichkeit; `P(…)` je Klasse summieren sich zu 1.": (
        "**How to read it:** `Prediction` = class with the highest probability, `Confidence` = this probability; "
        "`P(…)` per class add up to 1.",
        "**Jak czytać:** `Prognoza` = klasa z najwyższym prawdopodobieństwem, `Pewność` = to prawdopodobieństwo; "
        "`P(…)` dla klas sumują się do 1."),
    "**Lesart:** `Vorhersage` ist der geschätzte Wert von `{label}` (gleiche Einheit).": (
        "**How to read it:** `Prediction` is the estimated value of `{label}` (same unit).",
        "**Jak czytać:** `Prognoza` to oszacowana wartość `{label}` (ta sama jednostka)."),
    " Mittelwert der Vorhersagen: **{x}** (Training: {x2}).": (" Mean of the predictions: **{x}** (training: {x2}).",
                                                              " Średnia prognoz: **{x}** (trening: {x2})."),
    " Typische Abweichung laut Test: ± {x} (MAE).": (" Typical error according to the test: ± {x} (MAE).",
                                                    " Typowy błąd według testu: ± {x} (MAE)."),
    "Verteilung der Vorhersagen": ("Distribution of the predictions", "Rozkład prognoz"),
    "Vorhersage {label}": ("Prediction {label}", "Prognoza {label}"),
    "Mittelwert Training": ("Training mean", "Średnia w treningu"),
    "Fazit": ("Conclusion", "Podsumowanie"),
    "**Datenqualität: {v} {level}**": ("**Data quality: {v} {level}**", "**Jakość danych: {v} {level}**"),
    "**Was das Ergebnis aussagt**": ("**What the result means**", "**Co oznacza wynik**"),
    "**Empfehlung**": ("**Recommendation**", "**Zalecenie**"),
    "10 · Vorhersagen auf neuen Daten": ("10 · Predictions on new data", "10 · Prognozy na nowych danych"),
    "Aktuelles Modell (diese Sitzung)": ("Current model (this session)", "Bieżący model (ta sesja)"),
    "Geladenes Modell: {name}": ("Loaded model: {name}", "Wczytany model: {name}"),
    "Kein Modell vorhanden – in Schritt 6 trainieren oder in Schritt 9 laden.": (
        "No model available – train one in step 6 or load one in step 9.",
        "Brak modelu – wytrenuj go w kroku 6 albo wczytaj w kroku 9."),
    "Modell": ("Model", "Model"),
    "So funktioniert Schritt 10": ("How step 10 works", "Jak działa krok 10"),
    "Benötigte Spalten": ("Required columns", "Wymagane kolumny"),
    "CSV-Vorlage (nur Kopfzeile)": ("CSV template (header only)", "Szablon CSV (tylko nagłówek)"),
    "vorlage_neue_daten.csv": ("template_new_data.csv", "szablon_nowe_dane.csv"),
    "Schwelle": ("Threshold", "Próg"),
    "Eingabe": ("Input", "Dane wejściowe"),
    "Einzelner Datensatz (Formular)": ("Single record (form)", "Pojedynczy rekord (formularz)"),
    "Vorhersage berechnen": ("Compute predictions", "Oblicz prognozy"),
    "Bereite neue Daten auf und sage vorher …": ("Preparing new data and predicting …",
                                                 "Przygotowywanie nowych danych i prognozowanie …"),
    "{exc}. Die CSV braucht dieselben Rohspalten wie die Trainingsdaten (siehe Tabelle oben / CSV-Vorlage).": (
        "{exc}. The CSV needs the same raw columns as the training data (see table above / CSV template).",
        "{exc}. Plik CSV musi mieć te same surowe kolumny co dane treningowe (zob. tabela powyżej / szablon CSV)."),
    "{n} Vorhersagen für {name}": ("{n} predictions for {name}", "Prognozy dla {name}: {n}"),
    "{v} Hinweise zur Datenqualität und Einordnung → **Fazit** unten": (
        "{v} Notes on data quality and interpretation → **Conclusion** below",
        "{v} Uwagi o jakości danych i interpretacji → **Podsumowanie** poniżej"),
    "Label '{label}' ist enthalten → Evaluation auf {n} Zeilen": ("Label '{label}' is included → evaluation on {n} rows",
                                                                 "Etykieta '{label}' jest w danych → ewaluacja na "
                                                                 "wierszach: {n}"),
    "Ergebnis": ("Result", "Wynik"),
    "Ergebnis als CSV": ("Result as CSV", "Wynik jako CSV"),
    "vorhersagen_{stem}.csv": ("predictions_{stem}.csv", "prognozy_{stem}.csv"),
    "Werte eingeben (vorbelegt mit Median bzw. häufigstem Wert aus dem Training). Eingegeben wird die Form nach "
    "Schritt 2, z. B. term als Zahl.": (
        "Enter values (pre-filled with the median or most frequent value from training). Enter the form after "
        "step 2, e.g. term as a number.",
        "Wpisz wartości (wstępnie mediana albo najczęstsza wartość z treningu). Wpisuje się postać po kroku 2, "
        "np. term jako liczbę."),
    "Python-Code (Vorhersage ohne App)": ("Python code (prediction without the app)", "Kod Pythona (prognoza bez aplikacji)"),
    # Navigation und Seitenleiste
    "1 · Daten & Bibliotheken": ("1 · Data & libraries", "1 · Dane i biblioteki"),
    "2 · Aufbereitung 1 – Spalten": ("2 · Preparation 1 – columns", "2 · Przygotowanie 1 – kolumny"),
    "3 · Analyse & Visualisierung": ("3 · Analysis & visualisation", "3 · Analiza i wizualizacja"),
    "4 · Aufbereitung 2 – ML-Daten": ("4 · Preparation 2 – ML data", "4 · Przygotowanie 2 – dane ML"),
    "8 · Evaluation": ("8 · Evaluation", "8 · Ewaluacja"),
    "10 · Vorhersage neue Daten": ("10 · Predict new data", "10 · Prognoza dla nowych danych"),
    "Schritt": ("Step", "Krok"),
    "**Stichprobe** – für langsame Rechner": ("**Sample** – for slow computers", "**Próbka** – dla wolnych komputerów"),
    "Nur Stichprobe verwenden (df.sample)": ("Use a sample only (df.sample)", "Używaj tylko próbki (df.sample)"),
    "Standard: alle Daten (100 %). Gilt ab dem Laden für alle Schritte.": (
        "Default: all data (100 %). Applies to all steps from loading on.",
        "Domyślnie: wszystkie dane (100 %). Obowiązuje we wszystkich krokach od wczytania."),
    "Daten: {name} ({n} Zeilen)": ("Data: {name} ({n} rows)", "Dane: {name} (wiersze: {n})"),
    "Aufbereitung 1 übernommen": ("Preparation 1 applied", "Przygotowanie 1 zastosowane"),
    "ML-Daten vorbereitet": ("ML data prepared", "Dane ML przygotowane"),
    "ML-Daten: X_train {shape}": ("ML data: X_train {shape}", "Dane ML: X_train {shape}"),
    "Modell trainiert": ("Model trained", "Model wytrenowany"),
    "Geladen: {name}": ("Loaded: {name}", "Wczytano: {name}"),
    "Modell geladen (Schritt 9)": ("Model loaded (step 9)", "Model wczytany (krok 9)"),
    "Konfiguration speichern / laden": ("Save / load configuration", "Zapis / odczyt konfiguracji"),
    "Alle Einstellungen (Schritt 2 inkl. neuer Spalten, 4, 6) als JSON – für geänderte Daten mit gleichen Spalten.": (
        "All settings (step 2 incl. new columns, 4, 6) as JSON – for changed data with the same columns.",
        "Wszystkie ustawienia (krok 2 z nowymi kolumnami, 4, 6) jako JSON – dla zmienionych danych z tymi samymi "
        "kolumnami."),
    "Exportieren (JSON)": ("Export (JSON)", "Eksportuj (JSON)"),
    "Importieren": ("Import", "Importuj"),
    "Konfiguration übernehmen": ("Apply configuration", "Zastosuj konfigurację"),
    # Allgemein, Schritt 1, 2 und 4 (ML-Version)
    "Lade Bibliotheken (pandas, scikit-learn) …": (
        "Loading libraries (pandas, scikit-learn) …",
        "Wczytywanie bibliotek (pandas, scikit-learn) …"),
    "Koeffizient": (
        "Coefficient",
        "Współczynnik"),
    "Koeffizient (skaliert)": (
        "Coefficient (scaled)",
        "Współczynnik (skalowany)"),
    "Odds-Faktor": (
        "Odds ratio",
        "Iloraz szans"),
    "Platz": (
        "Rank",
        "Miejsce"),
    "Algorithmus": (
        "Algorithm",
        "Algorytm"),
    "Warum?": (
        "Why?",
        "Dlaczego?"),
    "Am besten für": (
        "Best for",
        "Do czego najlepiej"),
    "scikit-learn rechnet auf der CPU – {n} Kerne; Random Forest und KNN nutzen alle (n_jobs=-1).": (
        "scikit-learn computes on the CPU – {n} cores; Random Forest and KNN use all of them (n_jobs=-1).",
        "scikit-learn liczy na CPU – rdzenie: {n}; Random Forest i KNN używają wszystkich (n_jobs=-1)."),
    "LendingClub-Datensatz erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und 6 passend ein (78 "
    "Features, Random Forest mit 100 Bäumen, max_depth 12).": (
        "LendingClub dataset detected: the example configuration sets up step 2, 4 and 6 to match (78 features, "
        "Random Forest with 100 trees, max_depth 12).",
        "Rozpoznano zbiór LendingClub: przykładowa konfiguracja ustawia odpowiednio krok 2, 4 i 6 (78 cech, "
        "Random Forest ze 100 drzewami, max_depth 12)."),
    "SVM und KNN rechnen mit Abständen und brauchen skalierte Features; die Logistische Regression "
    "konvergiert damit schneller. Bäume (Random Forest) brauchen keine Skalierung.": (
        "SVM and KNN work with distances and need scaled features; Logistic Regression converges faster with "
        "them. Trees (Random Forest) need no scaling.",
        "SVM i KNN liczą na odległościach i potrzebują przeskalowanych cech; Logistic Regression dzięki temu "
        "szybciej się zbiega. Drzewa (Random Forest) nie potrzebują skalowania."),
    "y wird standardisiert, Vorhersagen automatisch zurückgerechnet. Wichtig für die SVM (epsilon und C "
    "beziehen sich auf die Skala von y); Bäume, KNN und die Lineare Regression liefern mit und ohne "
    "dieselben Vorhersagen.": (
        "y is standardised, predictions are converted back automatically. Important for the SVM (epsilon and C "
        "refer to the scale of y); trees, KNN and Linear Regression give the same predictions with or without it.",
        "y jest standaryzowane, prognozy są automatycznie przeliczane z powrotem. Ważne dla SVM (epsilon i C "
        "odnoszą się do skali y); drzewa, KNN i Linear Regression dają z nim i bez niego te same prognozy."),
    # Schritt 6 (ML-Version)
    "Fehler je Anzahl Bäume – der Wald wächst Schritt für Schritt (wie Epochen)": (
        "Error by number of trees – the forest grows step by step (like epochs)",
        "Błąd według liczby drzew – las rośnie krok po kroku (jak epoki)"),
    "Fehler je max_depth – von 1 bis unbegrenzt": (
        "Error by max_depth – from 1 to unlimited",
        "Błąd według max_depth – od 1 aż po brak limitu"),
    "Fehler je C (0.01 … 100)": (
        "Error by C (0.01 … 100)",
        "Błąd według C (0.01 … 100)"),
    "Fehler je C (0.001 … 1000)": (
        "Error by C (0.001 … 1000)",
        "Błąd według C (0.001 … 1000)"),
    "Fehlerrate je K = 1 … 40 (Elbow-Methode wie im Kurs)": (
        "Error rate by K = 1 … 40 (elbow method as in the course)",
        "Odsetek błędów według K = 1 … 40 (metoda łokcia jak w kursie)"),
    "Lernkurve: Fehler je Anzahl Trainingszeilen": (
        "Learning curve: error by number of training rows",
        "Krzywa uczenia: błąd według liczby wierszy treningowych"),
    "nur Regression": (
        "regression only",
        "tylko regresja"),
    "nur Klassifikation": (
        "classification only",
        "tylko klasyfikacja"),
    "Fehlerrate": (
        "Error rate",
        "Odsetek błędów"),
    "**{v}** · {n_feat} Features (Modell-Eingänge) · {n} Trainingszeilen · {n2} Testzeilen": (
        "**{v}** · {n_feat} features (model inputs) · {n} training rows · {n2} test rows",
        "**{v}** · cechy (wejścia modelu): {n_feat} · wiersze treningowe: {n} · wiersze testowe: {n2}"),
    "Modell wählen": (
        "Choose a model",
        "Wybór modelu"),
    "Reihenfolge = Rangliste (Platz 1 ist in der Regel am stärksten)": (
        "Order = ranking (rank 1 is usually the strongest)",
        "Kolejność = ranking (miejsce 1 jest zwykle najmocniejsze)"),
    "Rangliste als Tabelle": (
        "Ranking as a table",
        "Ranking jako tabela"),
    "Parameter: {name}": (
        "Parameters: {name}",
        "Parametry: {name}"),
    "Variante": (
        "Variant",
        "Wariant"),
    "Anzahl Bäume (im Kurs 600). Mehr Bäume = stabiler, aber langsamer und eine größere Modelldatei.": (
        "Number of trees (600 in the course). More trees = more stable, but slower and a larger model file.",
        "Liczba drzew (w kursie 600). Więcej drzew = stabilniej, ale wolniej i większy plik modelu."),
    "Maximale Tiefe – 0 = unbegrenzt (None). Kleiner = einfacher, weniger Overfitting.": (
        "Maximum depth – 0 = unlimited (None). Smaller = simpler, less overfitting.",
        "Maksymalna głębokość – 0 = bez limitu (None). Mniej = prostszy model, mniej przeuczenia."),
    "Mindestanzahl Zeilen je Blatt – größer = glattere Vorhersagen und kleinere Bäume.": (
        "Minimum number of rows per leaf – larger = smoother predictions and smaller trees.",
        "Minimalna liczba wierszy w liściu – więcej = gładsze prognozy i mniejsze drzewa."),
    "Features, die je Split zufällig zur Auswahl stehen: sqrt = √(Anzahl Features), 1.0 = alle.": (
        "Features available at random for each split: sqrt = √(number of features), 1.0 = all.",
        "Cechy losowo dostępne przy każdym podziale: sqrt = √(liczba cech), 1.0 = wszystkie."),
    "alle (1.0)": (
        "all (1.0)",
        "wszystkie (1.0)"),
    "rbf = Standard (nichtlinear) · linear = gerade Trennlinie · poly und sigmoid selten nötig": (
        "rbf = default (non-linear) · linear = straight separating line · poly and sigmoid rarely needed",
        "rbf = domyślny (nieliniowy) · linear = prosta linia podziału · poly i sigmoid rzadko potrzebne"),
    "Strafe für Fehler im Training: groß = genaue Trennung (Gefahr von Overfitting), klein = glattere Grenze.": (
        "Penalty for errors in training: large = precise separation (risk of overfitting), small = smoother "
        "boundary.",
        "Kara za błędy w treningu: duże = dokładny podział (ryzyko przeuczenia), małe = gładsza granica."),
    "Reichweite eines Trainingspunkts beim RBF-Kernel: groß = sehr lokal (Gefahr von Overfitting). 'scale' = "
    "1 / (Anzahl Features · Varianz).": (
        "Reach of a training point with the RBF kernel: large = very local (risk of overfitting). 'scale' = 1 / "
        "(number of features · variance).",
        "Zasięg punktu treningowego przy kernelu RBF: duże = bardzo lokalnie (ryzyko przeuczenia). 'scale' = 1 / "
        "(liczba cech · wariancja)."),
    "SVR: Fehler kleiner als epsilon werden nicht bestraft (Einheit von y – bei skaliertem Label in "
    "Standardabweichungen).": (
        "SVR: errors smaller than epsilon are not penalised (unit of y – in standard deviations if the label is "
        "scaled).",
        "SVR: błędy mniejsze niż epsilon nie są karane (jednostka y – przy skalowanej etykiecie w odchyleniach "
        "standardowych)."),
    "GridSearchCV: C und gamma automatisch suchen (wie im Kurs)": (
        "GridSearchCV: search C and gamma automatically (as in the course)",
        "GridSearchCV: automatycznie szukaj C i gamma (jak w kursie)"),
    "Probiert C = 0.1 … 100 und gamma = 1 … 0.001 mit Kreuzvalidierung auf den Trainingsdaten und trainiert "
    "danach mit der besten Kombination – C und gamma oben werden dabei ersetzt.": (
        "Tries C = 0.1 … 100 and gamma = 1 … 0.001 with cross-validation on the training data and then trains "
        "with the best combination – C and gamma above are replaced.",
        "Sprawdza C = 0.1 … 100 i gamma = 1 … 0.001 z walidacją krzyżową na danych treningowych, a potem trenuje "
        "z najlepszą kombinacją – C i gamma powyżej zostają zastąpione."),
    "Anzahl Folds der Kreuzvalidierung": (
        "Number of cross-validation folds",
        "Liczba foldów walidacji krzyżowej"),
    "Kehrwert der Regularisierung: klein = stärkere Regularisierung (einfacheres Modell), groß = schwächere.": (
        "Inverse of the regularisation: small = stronger regularisation (simpler model), large = weaker.",
        "Odwrotność regularyzacji: małe = silniejsza regularyzacja (prostszy model), duże = słabsza."),
    "Maximale Iterationen des Lösers – erhöhen, wenn eine ConvergenceWarning erscheint (Standard in "
    "scikit-learn: 100).": (
        "Maximum iterations of the solver – increase if a ConvergenceWarning appears (default in scikit-learn: "
        "100).",
        "Maksymalna liczba iteracji solvera – zwiększ, gdy pojawi się ConvergenceWarning (domyślnie w "
        "scikit-learn: 100)."),
    "Anzahl Nachbarn: klein = sehr lokal (Gefahr von Overfitting), groß = glatter. Den besten Wert zeigt "
    "Schritt 7 (Elbow-Methode).": (
        "Number of neighbours: small = very local (risk of overfitting), large = smoother. Step 7 shows the best "
        "value (elbow method).",
        "Liczba sąsiadów: mało = bardzo lokalnie (ryzyko przeuczenia), dużo = gładziej. Najlepszą wartość "
        "pokazuje krok 7 (metoda łokcia)."),
    "uniform = jeder Nachbar zählt gleich · distance = nähere Nachbarn zählen mehr": (
        "uniform = every neighbour counts the same · distance = closer neighbours count more",
        "uniform = każdy sąsiad liczy się tak samo · distance = bliżsi sąsiedzi liczą się bardziej"),
    "Achsenabschnitt schätzen (fit_intercept)": (
        "Estimate the intercept (fit_intercept)",
        "Szacuj wyraz wolny (fit_intercept)"),
    "Die Lineare Regression hat keine Hyperparameter. Schritt 7 zeigt die Lernkurve (helfen mehr Daten?), "
    "Schritt 8 die Koeffizienten (wie im Kurs: lm.coef_).": (
        "Linear Regression has no hyperparameters. Step 7 shows the learning curve (does more data help?), step "
        "8 the coefficients (as in the course: lm.coef_).",
        "Linear Regression nie ma hiperparametrów. Krok 7 pokazuje krzywą uczenia (czy więcej danych pomaga?), "
        "krok 8 współczynniki (jak w kursie: lm.coef_)."),
    "Training & Validierung": (
        "Training & validation",
        "Trening i walidacja"),
    "Validierungskurve für Schritt 7": (
        "Validation curve for step 7",
        "Krzywa walidacji do kroku 7"),
    "Trainiert das Modell zusätzlich für mehrere Werte des wichtigsten Parameters und misst den Fehler auf "
    "Trainings- und Validierungsdaten – dauert etwas länger.": (
        "Additionally trains the model for several values of the most important parameter and measures the error "
        "on training and validation data – takes a little longer.",
        "Dodatkowo trenuje model dla kilku wartości najważniejszego parametru i mierzy błąd na danych "
        "treningowych i walidacyjnych – trwa nieco dłużej."),
    "Testdaten: Die Kurve wird auf X_test berechnet (wie die Elbow-Methode im Kurs) – einfach, aber wer "
    "danach den besten Wert wählt, hat die Testdaten mitbenutzt. Sauberer: eigener Validierungsanteil aus "
    "den Trainingsdaten, damit das Testset bis Schritt 8 unberührt bleibt.": (
        "Test data: the curve is computed on X_test (like the elbow method in the course) – simple, but whoever "
        "then picks the best value has used the test data as well. Cleaner: a separate validation share of the "
        "training data, so the test set stays untouched until step 8.",
        "Dane testowe: krzywa jest liczona na X_test (jak metoda łokcia w kursie) – prosto, ale kto potem "
        "wybiera najlepszą wartość, korzystał już z danych testowych. Czyściej: osobna część walidacyjna z "
        "danych treningowych, aby zbiór testowy pozostał nietknięty do kroku 8."),
    "Schritt 7 zeigt: {what}.": (
        "Step 7 shows: {what}.",
        "Krok 7 pokazuje: {what}."),
    "Ohne Validierungskurve vergleicht Schritt 7 nur Training und Test.": (
        "Without a validation curve, step 7 only compares training and test.",
        "Bez krzywej walidacji krok 7 porównuje tylko trening i test."),
    "SVM: Die Rechenzeit wächst etwa quadratisch mit der Zahl der Zeilen – bei {n} Trainingszeilen kann das "
    "Training sehr lange dauern (mit GridSearchCV ein Vielfaches). Tipp: Stichprobe in der Seitenleiste oder "
    "ein anderes Modell.": (
        "SVM: the computing time grows roughly quadratically with the number of rows – with {n} training rows "
        "training can take very long (with GridSearchCV many times longer). Tip: a sample in the sidebar or "
        "another model.",
        "SVM: czas obliczeń rośnie mniej więcej kwadratowo z liczbą wierszy – przy liczbie wierszy treningowych "
        "{n} trening może trwać bardzo długo (z GridSearchCV wielokrotnie dłużej). Wskazówka: próbka w pasku "
        "bocznym albo inny model."),
    "KNN vergleicht jede Vorhersage mit allen {n} Trainingszeilen – Schritt 8 und 10 werden langsamer. Tipp: "
    "Stichprobe in der Seitenleiste.": (
        "KNN compares every prediction with all {n} training rows – steps 8 and 10 become slower. Tip: a sample "
        "in the sidebar.",
        "KNN porównuje każdą prognozę ze wszystkimi wierszami treningowymi ({n}) – kroki 8 i 10 działają "
        "wolniej. Wskazówka: próbka w pasku bocznym."),
    "Viele Zeilen und unbegrenzte Tiefe: Jeder Baum wird sehr groß (Arbeitsspeicher, Modelldatei in Schritt "
    "9). Tipp: max_depth z. B. 12–20 oder min_samples_leaf erhöhen.": (
        "Many rows and unlimited depth: every tree becomes very large (memory, model file in step 9). Tip: "
        "max_depth e.g. 12–20 or increase min_samples_leaf.",
        "Dużo wierszy i nieograniczona głębokość: każde drzewo staje się bardzo duże (pamięć, plik modelu w "
        "kroku 9). Wskazówka: max_depth np. 12–20 albo zwiększ min_samples_leaf."),
    "{name} rechnet mit Abständen – ohne Skalierung (Schritt 4) bestimmen Features mit großen Zahlen fast "
    "allein das Ergebnis. In Schritt 4 MinMaxScaler oder StandardScaler wählen.": (
        "{name} works with distances – without scaling (step 4) features with large numbers almost alone "
        "determine the result. Choose MinMaxScaler or StandardScaler in step 4.",
        "{name} liczy na odległościach – bez skalowania (krok 4) cechy o dużych liczbach niemal same decydują o "
        "wyniku. W kroku 4 wybierz MinMaxScaler albo StandardScaler."),
    "scikit-learn-Code mit diesen Einstellungen": (
        "scikit-learn code with these settings",
        "Kod scikit-learn z tymi ustawieniami"),
    "Aktuelles Modell: {name} · trainiert {v} · {seconds} s. Neues Training ersetzt es.": (
        "Current model: {name} · trained {v} · {seconds} s. A new training run replaces it.",
        "Bieżący model: {name} · wytrenowany {v} · {seconds} s. Nowy trening go zastąpi."),
    "Fertig · {seconds} s": (
        "Done · {seconds} s",
        "Gotowe · {seconds} s"),
    "Training abgeschlossen ({metric}): Training {t} · Test {v} → weiter mit Schritt 7.": (
        "Training finished ({metric}): training {t} · test {v} → continue with step 7.",
        "Trening zakończony ({metric}): trening {t} · test {v} → dalej w kroku 7."),
    "GridSearchCV: beste Kombination {best} (mittlere Validierungsgüte {score:.3f}) – damit wurde das Modell "
    "trainiert.": (
        "GridSearchCV: best combination {best} (mean validation score {score:.3f}) – the model was trained with "
        "it.",
        "GridSearchCV: najlepsza kombinacja {best} (średni wynik walidacji {score:.3f}) – z nią wytrenowano "
        "model."),
    "Das Modell ist nicht vollständig konvergiert (ConvergenceWarning) – max_iter erhöhen oder in Schritt 4 "
    "skalieren.": (
        "The model did not fully converge (ConvergenceWarning) – increase max_iter or scale in step 4.",
        "Model nie zbiegł się w pełni (ConvergenceWarning) – zwiększ max_iter albo włącz skalowanie w kroku 4."),
    "Modell (scikit-learn)": (
        "Model (scikit-learn)",
        "Model (scikit-learn)"),
    # Schritt 7 (ML-Version)
    "7 · Auswertung Training vs. Validierung": (
        "7 · Evaluating training vs. validation",
        "7 · Analiza: trening a walidacja"),
    "{name} · {metric} – kleiner = besser. Naive Vorhersage: man sagt immer {how} vorher.": (
        "{name} · {metric} – smaller = better. Naive prediction: always predicting {how}.",
        "{name} · {metric} – mniej = lepiej. Naiwna prognoza: zawsze przewiduje się {how}."),
    "Anteil aus Training": (
        "share of the training data",
        "część danych treningowych"),
    "Validierung ({where})": (
        "Validation ({where})",
        "Walidacja ({where})"),
    "Besten Wert in Schritt 6 übernehmen": (
        "Apply the best value in step 6",
        "Przejmij najlepszą wartość do kroku 6"),
    " GridSearchCV ist dafür ausgeschaltet (sonst würde die Suche C wieder ersetzen); gamma = {g} aus der Suche.": (
        " GridSearchCV has been switched off for this (otherwise the search would replace C again); gamma = {g} "
        "from the search.",
        " GridSearchCV został w tym celu wyłączony (inaczej wyszukiwanie znów zastąpiłoby C); gamma = {g} z "
        "wyszukiwania."),
    " GridSearchCV ist dafür ausgeschaltet (sonst würde die Suche C wieder ersetzen).": (
        " GridSearchCV has been switched off for this (otherwise the search would replace C again).",
        " GridSearchCV został w tym celu wyłączony (inaczej wyszukiwanie znów zastąpiłoby C)."),
    "0 (unbegrenzt)": (
        "0 (unlimited)",
        "0 (bez limitu)"),
    "Übernommen: {param} = {v} – jetzt in Schritt 6 neu trainieren.": (
        "Applied: {param} = {v} – now train again in step 6.",
        "Przejęto: {param} = {v} – teraz wytrenuj model ponownie w kroku 6."),
    "Validierung mit den Testdaten (wie im Kurs): Wer danach den besten Wert wählt, schätzt die Kennzahlen "
    "in Schritt 8 etwas zu optimistisch. Sauberer: in Schritt 6 „Anteil aus Training“.": (
        "Validation with the test data (as in the course): whoever then picks the best value estimates the "
        "metrics in step 8 somewhat too optimistically. Cleaner: “share of training data” in step 6.",
        "Walidacja na danych testowych (jak w kursie): kto potem wybiera najlepszą wartość, szacuje metryki w "
        "kroku 8 nieco zbyt optymistycznie. Czyściej: w kroku 6 „część danych treningowych”."),
    "Fehler gemessen auf Stichproben von höchstens {n} Zeilen – das Modell selbst nutzt alle Trainingszeilen.": (
        "Errors measured on samples of at most {n} rows – the model itself uses all training rows.",
        "Błędy mierzone na próbkach (maks. liczba wierszy: {n}) – sam model korzysta ze wszystkich wierszy "
        "treningowych."),
    "Die Validierungskurve ist in Schritt 6 ausgeschaltet – mit Kurve zeigt diese Seite den Verlauf über den "
    "wichtigsten Parameter.": (
        "The validation curve is switched off in step 6 – with the curve this page shows the course over the "
        "most important parameter.",
        "Krzywa walidacji jest wyłączona w kroku 6 – z krzywą ta strona pokazuje przebieg błędu dla "
        "najważniejszego parametru."),
    "Beste Kombination: **{best}** · mittlere Validierungsgüte ({s}) **{score:.3f}** mit {cv}-facher "
    "Kreuzvalidierung auf den Trainingsdaten – damit wurde das Modell trainiert.": (
        "Best combination: **{best}** · mean validation score ({s}) **{score:.3f}** with {cv}-fold "
        "cross-validation on the training data – the model was trained with it.",
        "Najlepsza kombinacja: **{best}** · średni wynik walidacji ({s}) **{score:.3f}** przy {cv}-krotnej "
        "walidacji krzyżowej na danych treningowych – z nią wytrenowano model."),
    "Alle Kombinationen": (
        "All combinations",
        "Wszystkie kombinacje"),
    "7 · Training vs. Validierung": (
        "7 · Training vs. validation",
        "7 · Trening a walidacja"),
    # Schritt 8 – Modell-Details (ML-Version)
    "Modell-Details: {name}": (
        "Model details: {name}",
        "Szczegóły modelu: {name}"),
    "feature_importances_: wie viel jedes Feature zu den Splits der Bäume beiträgt (Dummies einer Kategorie "
    "zusammengezählt, Summe = 1). Schnell, aber bei Features mit vielen verschiedenen Werten zu hoch – die "
    "Permutation Importance oben ist robuster.": (
        "feature_importances_: how much each feature contributes to the splits of the trees (dummies of one "
        "category added up, sum = 1). Fast, but too high for features with many different values – the "
        "permutation importance above is more robust.",
        "feature_importances_: ile każda cecha wnosi do podziałów w drzewach (kolumny dummy jednej kategorii "
        "zsumowane, suma = 1). Szybko, ale zawyżone dla cech o wielu różnych wartościach – ważność permutacyjna "
        "powyżej jest bardziej odporna."),
    "Feature-Wichtigkeit der Bäume": (
        "Feature importance of the trees",
        "Ważność cech w drzewach"),
    "Anteil an der Verbesserung der Splits": (
        "Share of the improvement of the splits",
        "Udział w poprawie podziałów"),
    "Entscheidungsbaum ansehen (oberste Ebenen)": (
        "View the decision tree (top levels)",
        "Zobacz drzewo decyzyjne (górne poziomy)"),
    "Koeffizient = Änderung der Vorhersage, wenn das Feature um 1 Einheit steigt und alle anderen gleich "
    "bleiben (Originaleinheiten, wie im Kurs). „skaliert“ = je 1 Einheit nach der Skalierung aus Schritt 4 – "
    "nur diese Werte sind zwischen den Features vergleichbar.": (
        "Coefficient = change in the prediction when the feature rises by 1 unit and all others stay the same "
        "(original units, as in the course). “scaled” = per 1 unit after the scaling from step 4 – only these "
        "values are comparable between features.",
        "Współczynnik = zmiana prognozy, gdy cecha rośnie o 1 jednostkę, a wszystkie inne pozostają bez zmian "
        "(oryginalne jednostki, jak w kursie). „skalowany” = na 1 jednostkę po skalowaniu z kroku 4 – tylko te "
        "wartości można porównywać między cechami."),
    "Steigt **{f}** um 1 (alle anderen gleich), ändert sich `{label}` um **{c}**.": (
        "If **{f}** rises by 1 (all others unchanged), `{label}` changes by **{c}**.",
        "Gdy **{f}** rośnie o 1 (pozostałe bez zmian), `{label}` zmienia się o **{c}**."),
    "Koeffizienten auf den skalierten Features (log-odds): positiv → höhere Wahrscheinlichkeit für '{pos}'. "
    "Odds-Faktor = exp(Koeffizient): Faktor, um den sich die Chancen je +1 (skaliert) ändern.": (
        "Coefficients on the scaled features (log-odds): positive → higher probability of '{pos}'. Odds ratio = "
        "exp(coefficient): factor by which the odds change per +1 (scaled).",
        "Współczynniki na przeskalowanych cechach (log-odds): dodatni → wyższe prawdopodobieństwo '{pos}'. "
        "Iloraz szans = exp(współczynnik): czynnik, o jaki zmieniają się szanse przy +1 (po skalowaniu)."),
    "Ein Koeffizient je Feature und Klasse (log-odds, skalierte Features): positiv → die Klasse wird "
    "wahrscheinlicher.": (
        "One coefficient per feature and class (log-odds, scaled features): positive → the class becomes more "
        "likely.",
        "Jeden współczynnik na cechę i klasę (log-odds, przeskalowane cechy): dodatni → klasa staje się bardziej "
        "prawdopodobna."),
    "Ohne drop_first (Schritt 4) summieren sich die Dummies einer Kategorie zu 1 – dann sind ihre "
    "Koeffizienten nicht eindeutig (nur die Vorhersagen). Zum Interpretieren in Schritt 4 drop_first "
    "einschalten.": (
        "Without drop_first (step 4) the dummies of a category add up to 1 – then their coefficients are not "
        "unique (only the predictions are). To interpret them, switch on drop_first in step 4.",
        "Bez drop_first (krok 4) kolumny dummy jednej kategorii sumują się do 1 – wtedy ich współczynniki nie są "
        "jednoznaczne (jednoznaczne są tylko prognozy). Aby je interpretować, włącz drop_first w kroku 4."),
    "Koeffizienten (skalierte Features)": (
        "Coefficients (scaled features)",
        "Współczynniki (przeskalowane cechy)"),
    "Koeffizient (rot positiv, blau negativ)": (
        "Coefficient (red positive, blue negative)",
        "Współczynnik (czerwony dodatni, niebieski ujemny)"),
    # Schritt 9 (ML-Version)
    "Gespeichert werden **zwei Dateien** in `models/<name>/`: das scikit-learn-Modell (`model.skops` oder "
    "`model.joblib`) und `preprocessing.json` – das Rezept aus Schritt 2 und 4 (Spalten-Aktionen, Füllwerte, "
    "Kategorien/Dummy-Spalten, Scaler, Label-Kodierung, Schwelle) mit den Einstellungen aus Schritt 6. Ohne "
    "das Rezept lassen sich neue Daten nicht korrekt aufbereiten.": (
        "**Two files** are saved in `models/<name>/`: the scikit-learn model (`model.skops` or `model.joblib`) "
        "and `preprocessing.json` – the recipe from step 2 and 4 (column actions, fill values, categories/dummy "
        "columns, scaler, label encoding, threshold) with the settings from step 6. Without the recipe new data "
        "cannot be prepared correctly.",
        "Zapisywane są **dwa pliki** w `models/<nazwa>/`: model scikit-learn (`model.skops` albo `model.joblib`) "
        "i `preprocessing.json` – recepta z kroku 2 i 4 (akcje na kolumnach, wartości uzupełniające, "
        "kategorie/kolumny dummy, skaler, kodowanie etykiety, próg) z ustawieniami z kroku 6. Bez recepty nowych "
        "danych nie da się poprawnie przygotować."),
    ".skops (empfohlen, sicher)": (
        ".skops (recommended, safe)",
        ".skops (zalecany, bezpieczny)"),
    ".joblib (scikit-learn-Standard, pickle)": (
        ".joblib (scikit-learn standard, pickle)",
        ".joblib (standard scikit-learn, pickle)"),
    ".skops lädt nur bekannte scikit-learn-Bausteine – kein fremder Code. .joblib (pickle) kann beim Laden "
    "beliebigen Code ausführen: nur eigene oder vertrauenswürdige Dateien laden.": (
        ".skops only loads known scikit-learn building blocks – no foreign code. .joblib (pickle) can run any "
        "code when loading: only load your own or trusted files.",
        ".skops wczytuje tylko znane elementy scikit-learn – żadnego obcego kodu. .joblib (pickle) może przy "
        "wczytywaniu wykonać dowolny kod: wczytuj tylko własne albo zaufane pliki."),
    "{name} · {model} · Label {label} · {v}": (
        "{name} · {model} · label {label} · {v}",
        "{name} · {model} · etykieta {label} · {v}"),
    "ZIP (model.skops/.joblib + preprocessing.json)": (
        "ZIP (model.skops/.joblib + preprocessing.json)",
        "ZIP (model.skops/.joblib + preprocessing.json)"),
    "Das ZIP enthält ein .joblib-Modell (pickle). Pickle kann beim Laden beliebigen Code ausführen – nur "
    "Dateien aus vertrauenswürdiger Quelle laden (z. B. eigene).": (
        "The ZIP contains a .joblib model (pickle). Pickle can run any code when loading – only load files from "
        "a trusted source (e.g. your own).",
        "ZIP zawiera model .joblib (pickle). Pickle może przy wczytywaniu wykonać dowolny kod – wczytuj tylko "
        "pliki z zaufanego źródła (np. własne)."),
    "Ich vertraue dieser Datei – .joblib laden": (
        "I trust this file – load .joblib",
        "Ufam temu plikowi – wczytaj .joblib"),
    "**Geladen: {name}** · {model} · Label `{label}` · {v} · {n} Features → {n2} Eingänge · gespeichert {v2}": (
        "**Loaded: {name}** · {model} · label `{label}` · {v} · {n} features → {n2} inputs · saved {v2}",
        "**Wczytano: {name}** · {model} · etykieta `{label}` · {v} · cechy: {n} → wejścia: {n2} · zapisano {v2}"),
}
