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
    "Modell-Eingänge": ("Model inputs", "Wejścia modelu"),
    "Rolle": ("Role", "Rola"),
    "Strategie": ("Strategy", "Strategia"),
    "Kategorie-Spalte": ("Category column", "Kolumna kategorii"),
    "Kategorien": ("Categories", "Kategorie"),
    "weggelassen (drop_first)": ("left out (drop_first)", "pominięta (drop_first)"),
    "Beispiele": ("Examples", "Przykłady"),
    "Feature": ("Feature", "Cecha"),
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
    "Feature (X)": ("Feature (X)", "Cecha (X)"),
    "Kategorie → Dummies": ("Category → dummies", "Kategoria → dummy"),
    "Viele Kategorien → viele Dummy-Spalten: {many}. Tipp: Schritt 2 'Top-N' oder entfernen.": (
        "Many categories → many dummy columns: {many}. Tip: step 2 'Keep top N' or drop.",
        "Dużo kategorii → dużo kolumn dummy: {many}. Wskazówka: w kroku 2 'Zachowaj top N' albo usuń."),
    "Fehlende Werte (NULL)": ("Missing values (NULL)", "Brakujące wartości (NULL)"),
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
    "Skalierung": ("Scaling", "Skalowanie"),
    "drop_first (Dummies)": ("drop_first (dummies)", "drop_first (dummy)"),
    "Neu vorbereiten verwirft das trainierte Modell – vorher in Schritt 9 speichern.": (
        "Preparing again discards the trained model – save it in step 9 first.",
        "Ponowne przygotowanie odrzuca wytrenowany model – najpierw zapisz go w kroku 9."),
    "Bereite Daten vor …": ("Preparing data …", "Przygotowywanie danych …"),
    "Daten vorbereitet": ("Data prepared", "Dane przygotowane"),
    "get_dummies: {n} Kategorie-Spalten → {n_dummy} Dummy-Spalten (0/1)": (
        "get_dummies: {n} category columns → {n_dummy} dummy columns (0/1)",
        "get_dummies: kolumny kategorii: {n} → kolumny dummy (0/1): {n_dummy}"),
    "**X nach Dummy-Kodierung (unskaliert, erste Zeilen)**": ("**X after dummy encoding (unscaled, first rows)**",
                                                             "**X po kodowaniu dummy (bez skalowania, pierwsze wiersze)**"),
    # Schritt 5
    "Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)": (
        "Data after step 2 – all changes and new columns (readable, e.g. for Excel)",
        "Dane po kroku 2 – wszystkie zmiany i nowe kolumny (czytelne, np. do Excela)"),
    "CSV international (Komma, Dezimalpunkt)": ("CSV international (comma, decimal point)",
                                                "CSV międzynarodowy (przecinek, kropka dziesiętna)"),
    "CSV für deutsches Excel (Semikolon, Dezimalkomma)": (
        "CSV for Excel – UK, US, Australia … (comma, decimal point)",
        "CSV dla polskiego Excela (średnik, przecinek dziesiętny)"),
    "5 · Daten speichern": ("5 · Save data", "5 · Zapis danych"),
    "Was speichern?": ("What to save?", "Co zapisać?"),
    "Format": ("Format", "Format"),
    "Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.": (
        "Sample active: only {n} of {n2} rows will be saved.",
        "Aktywna próbka: zapisana zostanie tylko próbka – {n} z {n2} wierszy."),
    "daten_aufbereitet": ("prepared_data", "dane_przygotowane"),
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
    "Texte, die mit =, +, -, @ beginnen, bekommen ein ' davor, damit Excel sie nicht als Formel ausführt.": (
        "Text starting with =, +, -, @ gets a ' in front so that Excel does not run it as a formula.",
        "Teksty zaczynające się od =, +, -, @ dostają ' na początku, żeby Excel nie wykonał ich jako formuły."),
    # Schritt 6
    "Während des Trainings nichts anklicken – jede Eingabe bricht das Training ab.": (
        "Do not click anything during training – any input stops the training.",
        "Nie klikaj niczego podczas treningu – każda zmiana przerywa trening."),
    "Training starten": ("Start training", "Rozpocznij trening"),
    "Starte Training …": ("Starting training …", "Uruchamianie treningu …"),
    "Training fehlgeschlagen: {exc}": ("Training failed: {exc}", "Trening nie powiódł się: {exc}"),
    # Schritt 7
    "Bitte zuerst in Schritt 6 trainieren.": ("Please train in step 6 first.", "Najpierw wytrenuj model w kroku 6."),
    # Schritt 8
    "Einzelfall prüfen": ("Check a single case", "Sprawdź pojedynczy przypadek"),
    # Schritt 9
    "9 · Modell speichern & laden": ("9 · Save & load model", "9 · Zapis i odczyt modelu"),
    "Speichern": ("Save", "Zapisz"),
    "Noch kein trainiertes Modell in dieser Sitzung (Schritt 6).": ("No trained model in this session yet (step 6).",
                                                                    "W tej sesji nie ma jeszcze wytrenowanego modelu "
                                                                    "(krok 6)."),
    "Name (Ordner unter models/)": ("Name (folder in models/)", "Nazwa (folder w models/)"),
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
    # Schritt 10
    "Fazit": ("Conclusion", "Podsumowanie"),
    "**Datenqualität: {v} {level}**": ("**Data quality: {v} {level}**", "**Jakość danych: {v} {level}**"),
    "**Empfehlung**": ("**Recommendation**", "**Zalecenie**"),
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
    "Eingabe": ("Input", "Dane wejściowe"),
    "Einzelner Datensatz (Formular)": ("Single record (form)", "Pojedynczy rekord (formularz)"),
    "{exc}. Die CSV braucht dieselben Rohspalten wie die Trainingsdaten (siehe Tabelle oben / CSV-Vorlage).": (
        "{exc}. The CSV needs the same raw columns as the training data (see table above / CSV template).",
        "{exc}. Plik CSV musi mieć te same surowe kolumny co dane treningowe (zob. tabela powyżej / szablon CSV)."),
    "{v} Hinweise zur Datenqualität und Einordnung → **Fazit** unten": (
        "{v} Notes on data quality and interpretation → **Conclusion** below",
        "{v} Uwagi o jakości danych i interpretacji → **Podsumowanie** poniżej"),
    "Ergebnis": ("Result", "Wynik"),
    "Ergebnis als CSV": ("Result as CSV", "Wynik jako CSV"),
    # Navigation und Seitenleiste
    "1 · Daten & Bibliotheken": ("1 · Data & libraries", "1 · Dane i biblioteki"),
    "2 · Aufbereitung 1 – Spalten": ("2 · Preparation 1 – columns", "2 · Przygotowanie 1 – kolumny"),
    "3 · Analyse & Visualisierung": ("3 · Analysis & visualisation", "3 · Analiza i wizualizacja"),
    "Schritt": ("Step", "Krok"),
    "**Stichprobe** – für langsame Rechner": ("**Sample** – for slow computers", "**Próbka** – dla wolnych komputerów"),
    "Nur Stichprobe verwenden (df.sample)": ("Use a sample only (df.sample)", "Używaj tylko próbki (df.sample)"),
    "Standard: alle Daten (100 %). Gilt ab dem Laden für alle Schritte.": (
        "Default: all data (100 %). Applies to all steps from loading on.",
        "Domyślnie: wszystkie dane (100 %). Obowiązuje we wszystkich krokach od wczytania."),
    "Daten: {name} ({n} Zeilen)": ("Data: {name} ({n} rows)", "Dane: {name} (wiersze: {n})"),
    "Aufbereitung 1 übernommen": ("Preparation 1 applied", "Przygotowanie 1 zastosowane"),
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
    # Schritt 6 (ML-Version)
    "scikit-learn-Code mit diesen Einstellungen": (
        "scikit-learn code with these settings",
        "Kod scikit-learn z tymi ustawieniami"),
    "Fertig · {seconds} s": (
        "Done · {seconds} s",
        "Gotowe · {seconds} s"),
    "Modell (scikit-learn)": (
        "Model (scikit-learn)",
        "Model (scikit-learn)"),
    # Schritt 9 (ML-Version)
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
    # --- Clustering-App (K-Means) ---
    "Cluster-Name": ("Cluster name", "Nazwa klastra"),
    "Name": ("Name", "Nazwa"),
    "typische Vertreter": ("Typical members", "Typowi przedstawiciele"),
    "Randfälle (weit vom Zentrum)": ("Edge cases (far from the centre)", "Przypadki brzegowe (daleko od centrum)"),
    "Ladung": ("Loading", "Ładunek"),
    "Vergleichsgruppe": ("Comparison group", "Grupa porównawcza"),
    "4 · Aufbereitung 2 – Clustering-Daten": (
        "4 · Preparation 2 – clustering data",
        "4 · Przygotowanie 2 – dane do klastrowania"),
    "6 · Modell & Training (K-Means)": ("6 · Model & training (K-Means)", "6 · Model i trening (K-Means)"),
    "7 · Wahl von K (Elbow & Silhouette)": (
        "7 · Choosing K (elbow & silhouette)",
        "7 · Wybór K (łokieć i Silhouette)"),
    "8 · Cluster-Auswertung": ("8 · Cluster evaluation", "8 · Ocena klastrów"),
    "10 · Neue Daten zuordnen": ("10 · Assign new data", "10 · Przypisz nowe dane"),
    "scikit-learn rechnet auf der CPU – {n} Kerne; K-Means nutzt sie automatisch alle.": (
        "scikit-learn computes on the CPU – {n} cores; K-Means uses all of them automatically.",
        "scikit-learn liczy na CPU – {n} rdzeni; K-Means automatycznie używa wszystkich."),
    "Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann **Übernehmen**. "
    "Die Vorschläge sind automatisch – bitte prüfen. Für K-Means zählt nur, was Zeilen ähnlich oder verschieden "
    "macht: IDs und Freitext entfernen; eine Spalte mit Namen (z. B. der Hochschule) behalten und in Schritt 4 "
    "als Bezeichnung wählen. Unmögliche Werte und Ausreißer verschieben die Cluster-Zentren – z. B. Grad.Rate "
    "118 % mit *Werte begrenzen* auf 100 setzen. Text-Spalten mit *behalten* werden in Schritt 4 zu Dummies.": (
        "One row per CSV column. Choose an **action**, if needed a **parameter** and a **new name**, then "
        "**Apply**. The suggestions are automatic – please check them. For K-Means only what makes rows similar "
        "or different counts: drop IDs and free text; keep a column with names (e.g. of the college) and choose "
        "it as the row name in step 4. Impossible values and outliers shift the cluster centres – e.g. set "
        "Grad.Rate 118 % to 100 with *clip values*. Text columns with *keep* become dummies in step 4.",
        "Jeden wiersz na kolumnę CSV. Wybierz **akcję**, w razie potrzeby **parametr** i **nową nazwę**, potem "
        "**Zastosuj**. Propozycje są automatyczne – sprawdź je. Dla K-Means liczy się tylko to, co czyni wiersze "
        "podobnymi lub różnymi: usuń identyfikatory i wolny tekst; zachowaj kolumnę z nazwami (np. uczelni) i "
        "wybierz ją w kroku 4 jako nazwę wiersza. Niemożliwe wartości i wartości odstające przesuwają centra "
        "klastrów – np. ustaw Grad.Rate 118 % na 100 przez *ogranicz wartości*. Kolumny tekstowe z akcją "
        "*zachowaj* stają się w kroku 4 kolumnami dummy."),
    "College-Datensatz aus dem K-Means-Kurs erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und 6 wie im "
    "Kurs ein – Name der Hochschule als Bezeichnung, Grad.Rate auf 100 begrenzt, 17 Features, Private nur zum "
    "Vergleich, K = 2. Anders als im Kurs mit StandardScaler; für den direkten Vergleich in Schritt 4 'keiner' "
    "wählen.": (
        "College data set from the K-Means course detected: the example configuration sets steps 2, 4 and 6 as "
        "in the course – college name as row name, Grad.Rate clipped to 100, 17 features, Private only for "
        "comparison, K = 2. Unlike the course it uses StandardScaler; for a direct comparison choose 'none' in "
        "step 4.",
        "Wykryto zbiór College z kursu K-Means: przykładowa konfiguracja ustawia kroki 2, 4 i 6 jak w kursie – "
        "nazwa uczelni jako nazwa wiersza, Grad.Rate ograniczone do 100, 17 cech, Private tylko do porównania, K "
        "= 2. Inaczej niż w kursie ze StandardScaler; do bezpośredniego porównania wybierz w kroku 4 'brak'."),
    "4 · Datenaufbereitung 2 – Clustering-Daten (Features, Lücken, Skalierung)": (
        "4 · Data preparation 2 – clustering data (features, gaps, scaling)",
        "4 · Przygotowanie danych 2 – dane do klastrowania (cechy, braki, skalowanie)"),
    "**Unsupervised Learning:** K-Means bekommt keine Antwort – kein Label, kein y. Es sucht Gruppen von Zeilen, "
    "die sich in den Features ähneln. Eine bekannte Einteilung (im Kurs die Spalte *Private*) geht nicht ins "
    "Modell; sie dient nur dem Vergleich in Schritt 8.": (
        "**Unsupervised learning:** K-Means is never given an answer – no label, no y. It looks for groups of "
        "rows that are similar in the features. A known grouping (in the course the column *Private*) does not "
        "go into the model; it is only used for the comparison in step 8.",
        "**Uczenie nienadzorowane:** K-Means nie dostaje odpowiedzi – brak etykiety, brak y. Szuka grup wierszy "
        "podobnych pod względem cech. Znany podział (w kursie kolumna *Private*) nie trafia do modelu; służy "
        "tylko do porównania w kroku 8."),
    "Vergleichsgruppe & Bezeichnung (optional)": (
        "Comparison group & row name (optional)",
        "Grupa porównawcza i nazwa wiersza (opcjonalnie)"),
    "Bekannte Einteilung – nur zum Vergleich": (
        "Known grouping – for comparison only",
        "Znany podział – tylko do porównania"),
    "Geht NICHT ins Clustering – erst danach wird verglichen: Finden die Cluster diese Gruppen wieder? (im Kurs: "
    "Private = Yes/No). Spalten mit 2–30 verschiedenen Werten.": (
        "NOT used for the clustering – compared only afterwards: do the clusters find these groups again? (in "
        "the course: Private = Yes/No). Columns with 2–30 different values.",
        "NIE trafia do klastrowania – porównanie dopiero potem: czy klastry odnajdują te grupy? (w kursie: "
        "Private = Yes/No). Kolumny z 2–30 różnymi wartościami."),
    "Bezeichnung der Zeilen – nur zur Anzeige": (
        "Row name – for display only",
        "Nazwa wierszy – tylko do wyświetlania"),
    "Text, der eine Zeile benennt (z. B. Name der Hochschule): zeigt typische Vertreter je Cluster und steht im "
    "Export – nie ein Feature.": (
        "Text that names a row (e.g. the college name): shows typical members of each cluster and is included in "
        "the export – never a feature.",
        "Tekst nazywający wiersz (np. nazwa uczelni): pokazuje typowych przedstawicieli klastrów i trafia do "
        "eksportu – nigdy nie jest cechą."),
    "Tipp: Eine Spalte mit Namen (z. B. 'Unnamed: 0' – die erste Spalte von College_Data, im Kurs index_col=0) "
    "in Schritt 2 behalten statt entfernen; dann lässt sie sich hier als Bezeichnung wählen.": (
        "Tip: keep a column with names (e.g. 'Unnamed: 0' – the first column of College_Data, index_col=0 in the "
        "course) in step 2 instead of dropping it; then it can be chosen here as the row name.",
        "Wskazówka: kolumnę z nazwami (np. 'Unnamed: 0' – pierwsza kolumna College_Data, w kursie index_col=0) w "
        "kroku 2 zachowaj zamiast usuwać; wtedy można ją tu wybrać jako nazwę wiersza."),
    "Features (X) – daraus berechnet K-Means die Ähnlichkeit": (
        "Features (X) – K-Means computes the similarity from these",
        "Cechy (X) – z nich K-Means liczy podobieństwo"),
    "Features für K-Means": ("Features for K-Means", "Cechy dla K-Means"),
    "{leak} stammt aus derselben CSV-Spalte wie die Vergleichsgruppe – K-Means kennt die Antwort dann schon, der "
    "Vergleich in Schritt 8 wäre geschönt.": (
        "{leak} comes from the same CSV column as the comparison group – K-Means would then already know the "
        "answer and the comparison in step 8 would look better than it is.",
        "{leak} pochodzi z tej samej kolumny CSV co grupa porównawcza – K-Means znałby wtedy już odpowiedź, a "
        "porównanie w kroku 8 wypadłoby lepiej niż w rzeczywistości."),
    "Alle Spalten – {n} Features → ca. {n_inputs} Modell-Eingänge": (
        "All columns – {n} features → approx. {n_inputs} model inputs",
        "Wszystkie kolumny – {n} cech → ok. {n_inputs} wejść modelu"),
    "Beim Clustering gibt es keinen Train/Test-Split – K-Means bekommt keine Antwort, die man vor ihm verstecken "
    "müsste. Füllwerte, Kategorien (Dummies) und Scaler werden deshalb aus allen Zeilen gelernt und später "
    "unverändert auf neue Daten angewendet (Schritt 10).": (
        "Clustering has no train/test split – K-Means is never given an answer that would have to be hidden from "
        "it. Fill values, categories (dummies) and the scaler are therefore learned from all rows and later "
        "applied unchanged to new data (step 10).",
        "Przy klastrowaniu nie ma podziału train/test – K-Means nie dostaje odpowiedzi, którą trzeba by przed "
        "nim ukryć. Wartości uzupełniające, kategorie (dummy) i scaler są więc wyuczone ze wszystkich wierszy i "
        "później stosowane bez zmian do nowych danych (krok 10)."),
    "Skalierung & Dummies": ("Scaling & dummies", "Skalowanie i dummy"),
    "K-Means rechnet mit Abständen: Ohne Skalierung bestimmt die Spalte mit den größten Zahlen (z. B. Apps oder "
    "F.Undergrad) fast allein die Cluster. StandardScaler = je Spalte Mittelwert 0 und Standardabweichung 1 "
    "(üblich für K-Means), MinMaxScaler = 0 … 1. Im Kurs wurde nicht skaliert.": (
        "K-Means works with distances: without scaling, the column with the largest numbers (e.g. Apps or "
        "F.Undergrad) almost alone decides the clusters. StandardScaler = mean 0 and standard deviation 1 per "
        "column (usual for K-Means), MinMaxScaler = 0 … 1. The course did not scale.",
        "K-Means liczy odległości: bez skalowania kolumna z największymi liczbami (np. Apps lub F.Undergrad) "
        "prawie sama decyduje o klastrach. StandardScaler = w każdej kolumnie średnia 0 i odchylenie standardowe "
        "1 (typowe dla K-Means), MinMaxScaler = 0 … 1. W kursie nie skalowano."),
    "Für Abstände besser aus: Mit drop_first liegt die weggelassene Kategorie (alle Dummies 0) näher an allen "
    "anderen als diese untereinander.": (
        "Better off for distances: with drop_first the dropped category (all dummies 0) is closer to all others "
        "than they are to each other.",
        "Dla odległości lepiej wyłączyć: z drop_first pominięta kategoria (wszystkie dummy 0) leży bliżej "
        "wszystkich innych niż one między sobą."),
    "Ohne Skalierung zählen große Zahlen viel mehr als kleine – das Ergebnis hängt dann von den Einheiten ab "
    "(Euro oder Tausend Euro). Für K-Means meist StandardScaler wählen.": (
        "Without scaling, large numbers count much more than small ones – the result then depends on the units "
        "(euros or thousands of euros). For K-Means usually choose StandardScaler.",
        "Bez skalowania duże liczby liczą się znacznie bardziej niż małe – wynik zależy wtedy od jednostek (euro "
        "czy tysiące euro). Dla K-Means zwykle wybierz StandardScaler."),
    "Hinweis: Dummies (0/1) werden mitskaliert – seltene Kategorien bekommen dadurch großes Gewicht. Bei vielen "
    "seltenen Kategorien MinMaxScaler erwägen.": (
        "Note: dummies (0/1) are scaled too – rare categories get a large weight as a result. With many rare "
        "categories consider MinMaxScaler.",
        "Uwaga: kolumny dummy (0/1) też są skalowane – rzadkie kategorie dostają przez to dużą wagę. Przy wielu "
        "rzadkich kategoriach rozważ MinMaxScaler."),
    "Daten für K-Means vorbereiten": ("Prepare data for K-Means", "Przygotuj dane dla K-Means"),
    "Die Einstellungen oben wurden seit dem Vorbereiten geändert → erneut **Daten für K-Means vorbereiten**.": (
        "The settings above have changed since preparing → **Prepare data for K-Means** again.",
        "Ustawienia powyżej zmieniły się od przygotowania → ponownie **Przygotuj dane dla K-Means**."),
    "Fertig: X {shape} – {n} Zeilen, {n2} Features → {n3} Spalten (Modell-Eingänge)": (
        "Done: X {shape} – {n} rows, {n2} features → {n3} columns (model inputs)",
        "Gotowe: X {shape} – {n} wierszy, {n2} cech → {n3} kolumn (wejścia modelu)"),
    "**Vergleichsgruppe {g}** – geht nicht ins Modell, wird erst in Schritt 8 verglichen:": (
        "**Comparison group {g}** – not part of the model, compared only in step 8:",
        "**Grupa porównawcza {g}** – nie trafia do modelu, porównanie dopiero w kroku 8:"),
    "Jede Kategorie wird zu einer eigenen 0/1-Spalte, z. B. Region → Region_North, Region_South … Mit drop_first "
    "entfällt die erste Kategorie – sie ist daran erkennbar, dass alle anderen 0 sind. Unbekannte Werte in neuen "
    "Daten → alle 0.": (
        "Every category becomes its own 0/1 column, e.g. Region → Region_North, Region_South … With drop_first "
        "the first category is left out – it is recognisable by all others being 0. Unknown values in new data → "
        "all 0.",
        "Każda kategoria staje się osobną kolumną 0/1, np. Region → Region_North, Region_South … Z drop_first "
        "pierwsza kategoria odpada – rozpoznaje się ją po tym, że wszystkie inne są 0. Nieznane wartości w "
        "nowych danych → wszystkie 0."),
    "**X nach {type} (erste Zeilen) – so sieht K-Means die Daten**": (
        "**X after {type} (first rows) – this is how K-Means sees the data**",
        "**X po {type} (pierwsze wiersze) – tak K-Means widzi dane**"),
    "Clustering-Daten nach Schritt 4 – Lücken gefüllt, Dummies; Bezeichnung und Vergleichsgruppe als "
    "Zusatzspalten": (
        "Clustering data after step 4 – gaps filled, dummies; row name and comparison group as extra columns",
        "Dane do klastrowania po kroku 4 – braki uzupełnione, dummy; nazwa wiersza i grupa porównawcza jako "
        "dodatkowe kolumny"),
    "wie zuvor, zusätzlich skaliert – genau diese Zahlen sieht K-Means": (
        "as above, additionally scaled – exactly these numbers are what K-Means sees",
        "jak wyżej, dodatkowo skalowane – dokładnie te liczby widzi K-Means"),
    "Daten nach Schritt 2 mit Cluster-Nummer, Cluster-Name und Abstand zum Zentrum (nach Schritt 6)": (
        "Data after step 2 with cluster number, cluster name and distance to the centre (after step 6)",
        "Dane po kroku 2 z numerem klastra, nazwą klastra i odległością od centrum (po kroku 6)"),
    "Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen Programm, "
    "nach dem Training auch mit der Cluster-Nummer je Zeile.": (
        "Save the prepared data as a CSV file – to keep working in Excel or another program, after training also "
        "with the cluster number of each row.",
        "Zapisz przygotowane dane jako plik CSV – do dalszej pracy w Excelu lub innym programie, po treningu "
        "także z numerem klastra każdego wiersza."),
    "Für die Clustering-Daten zuerst Schritt 4 ausführen (**Daten für K-Means vorbereiten**).": (
        "For the clustering data run step 4 first (**Prepare data for K-Means**).",
        "Aby uzyskać dane do klastrowania, najpierw wykonaj krok 4 (**Przygotuj dane dla K-Means**)."),
    "Für die Cluster-Nummer je Zeile zuerst in Schritt 6 trainieren.": (
        "For the cluster number of each row, train in step 6 first.",
        "Aby uzyskać numer klastra dla każdego wiersza, najpierw wytrenuj model w kroku 6."),
    "Die Cluster-Nummern sind willkürlich (0 … K−1). Abstand = euklidischer Abstand zum eigenen Zentrum in den "
    "skalierten Features – groß = untypisch für den Cluster.": (
        "The cluster numbers are arbitrary (0 … K−1). Distance = Euclidean distance to its own centre in the "
        "scaled features – large = untypical for the cluster.",
        "Numery klastrów są dowolne (0 … K−1). Odległość = odległość euklidesowa od własnego centrum w "
        "skalowanych cechach – duża = nietypowy dla klastra."),
    "Für die Zuordnung neuer Daten (Schritt 10) die **ursprüngliche** CSV-Datei verwenden, nicht diese Datei – "
    "das gespeicherte Modell wiederholt die Aufbereitung selbst.": (
        "To assign new data (step 10) use the **original** CSV file, not this file – the saved model repeats the "
        "preparation itself.",
        "Do przypisania nowych danych (krok 10) użyj **oryginalnego** pliku CSV, nie tego pliku – zapisany model "
        "sam powtarza przygotowanie."),
    "6 · Modell & Training – K-Means": ("6 · Model & training – K-Means", "6 · Model i trening – K-Means"),
    "Bitte zuerst Schritt 4 ausführen (Daten für K-Means vorbereiten).": (
        "Please run step 4 first (prepare data for K-Means).",
        "Najpierw wykonaj krok 4 (przygotuj dane dla K-Means)."),
    "**K-Means** · {n} Zeilen · {p} Features (Modell-Eingänge) · Skalierung: {scaler}": (
        "**K-Means** · {n} rows · {p} features (model inputs) · scaling: {scaler}",
        "**K-Means** · {n} wierszy · {p} cech (wejścia modelu) · skalowanie: {scaler}"),
    "So arbeitet K-Means": ("How K-Means works", "Jak działa K-Means"),
    "K (n_clusters)": ("K (n_clusters)", "K (n_clusters)"),
    "Anzahl der Cluster – wie im Kurs KMeans(n_clusters=2). Welches K passt, zeigt Schritt 7 (Elbow-Methode und "
    "Silhouette).": (
        "Number of clusters – as in the course KMeans(n_clusters=2). Step 7 shows which K fits (elbow method and "
        "silhouette).",
        "Liczba klastrów – jak w kursie KMeans(n_clusters=2). Które K pasuje, pokazuje krok 7 (metoda łokcia i "
        "Silhouette)."),
    "k-means++ = Startzentren gut verteilt (Standard, meist schneller und besser) · random = zufällige Zeilen "
    "als Startzentren": (
        "k-means++ = well spread starting centres (default, usually faster and better) · random = random rows as "
        "starting centres",
        "k-means++ = dobrze rozłożone centra startowe (domyślnie, zwykle szybciej i lepiej) · random = losowe "
        "wiersze jako centra startowe"),
    "Wie oft K-Means mit anderen Startzentren läuft – das beste Ergebnis (kleinste Inertia) gewinnt. "
    "scikit-learn nimmt sonst 'auto' = 1 Lauf bei k-means++.": (
        "How often K-Means runs with other starting centres – the best result (smallest inertia) wins. Otherwise "
        "scikit-learn uses 'auto' = 1 run with k-means++.",
        "Ile razy K-Means startuje z innymi centrami – wygrywa najlepszy wynik (najmniejsza inercja). Inaczej "
        "scikit-learn używa 'auto' = 1 przebieg przy k-means++."),
    "Höchstzahl der Schritte Zuordnen → Verschieben je Lauf (Standard 300).": (
        "Maximum number of assign → move steps per run (default 300).",
        "Maksymalna liczba kroków przypisz → przesuń na przebieg (domyślnie 300)."),
    "lloyd = klassisches K-Means (Standard) · elkan = gleiches Ergebnis, bei gut getrennten Clustern manchmal "
    "schneller": (
        "lloyd = classic K-Means (default) · elkan = same result, sometimes faster with well separated clusters",
        "lloyd = klasyczny K-Means (domyślnie) · elkan = ten sam wynik, przy dobrze rozdzielonych klastrach "
        "czasem szybciej"),
    "Seed (random_state)": ("Seed (random_state)", "Seed (random_state)"),
    "Gleicher Seed = gleiche Startzentren = gleiches Ergebnis.": (
        "Same seed = same starting centres = same result.",
        "Ten sam seed = te same centra startowe = ten sam wynik."),
    "Wahl von K für Schritt 7": ("Choosing K for step 7", "Wybór K dla kroku 7"),
    "Elbow- und Silhouette-Kurve berechnen (K = 1 … k_max)": (
        "Compute elbow and silhouette curve (K = 1 … k_max)",
        "Oblicz krzywą łokcia i Silhouette (K = 1 … k_max)"),
    "Trainiert K-Means zusätzlich für jedes K von 1 bis k_max und misst Inertia und Silhouette – Schritt 7 zeigt "
    "damit, welches K passt. Dauert etwas länger.": (
        "Additionally trains K-Means for every K from 1 to k_max and measures inertia and silhouette – step 7 "
        "uses this to show which K fits. Takes a little longer.",
        "Dodatkowo trenuje K-Means dla każdego K od 1 do k_max i mierzy inercję oraz Silhouette – krok 7 "
        "pokazuje na tej podstawie, które K pasuje. Trwa nieco dłużej."),
    "Ohne Skalierung (Schritt 4) bestimmen Features mit großen Zahlen fast allein die Abstände – und damit die "
    "Cluster. In Schritt 4 StandardScaler wählen.": (
        "Without scaling (step 4), features with large numbers almost alone decide the distances – and therefore "
        "the clusters. Choose StandardScaler in step 4.",
        "Bez skalowania (krok 4) cechy z dużymi liczbami prawie same decydują o odległościach – a więc o "
        "klastrach. Wybierz w kroku 4 StandardScaler."),
    "{n} Zeilen: K-Means mit n_init = {i} kann einige Minuten dauern (die Kurve für Schritt 7 nutzt eine "
    "Stichprobe von {s} Zeilen). Tipp: n_init senken oder Stichprobe in der Seitenleiste.": (
        "{n} rows: K-Means with n_init = {i} can take a few minutes (the curve for step 7 uses a sample of {s} "
        "rows). Tip: lower n_init or use a sample in the sidebar.",
        "{n} wierszy: K-Means z n_init = {i} może trwać kilka minut (krzywa dla kroku 7 używa próbki {s} "
        "wierszy). Wskazówka: zmniejsz n_init lub użyj próbki w panelu bocznym."),
    "Aktuelles Modell: K-Means mit K = {k} · trainiert {v} · {seconds} s. Neues Training ersetzt es.": (
        "Current model: K-Means with K = {k} · trained {v} · {seconds} s. A new training replaces it.",
        "Aktualny model: K-Means z K = {k} · wytrenowany {v} · {seconds} s. Nowy trening go zastąpi."),
    "K-Means fertig: K = {k} · Silhouette {s} · Inertia {i} · {it} Iterationen → weiter mit Schritt 7.": (
        "K-Means done: K = {k} · silhouette {s} · inertia {i} · {it} iterations → continue with step 7.",
        "K-Means gotowy: K = {k} · Silhouette {s} · inercja {i} · {it} iteracji → dalej krok 7."),
    "K-Means hat max_iter erreicht, ohne stabil zu werden – max_iter erhöhen.": (
        "K-Means reached max_iter without becoming stable – increase max_iter.",
        "K-Means osiągnął max_iter, nie stabilizując się – zwiększ max_iter."),
    "Hinweis von scikit-learn: {w}": ("Note from scikit-learn: {w}", "Uwaga od scikit-learn: {w}"),
    "Übernommen: K = {k} – jetzt in Schritt 6 neu trainieren.": (
        "Applied: K = {k} – now retrain in step 6.",
        "Przejęto: K = {k} – teraz wytrenuj ponownie w kroku 6."),
    "7 · Wahl von K – Elbow-Methode & Silhouette": (
        "7 · Choosing K – elbow method & silhouette",
        "7 · Wybór K – metoda łokcia i Silhouette"),
    "−1 … 1: Wie viel näher liegt jede Zeile an ihrem eigenen Cluster als am nächsten anderen? Größer = besser "
    "getrennt.": (
        "−1 … 1: how much closer is each row to its own cluster than to the next other one? Higher = better "
        "separated.",
        "−1 … 1: o ile bliżej leży każdy wiersz swojego klastra niż najbliższego innego? Większa = lepiej "
        "rozdzielone."),
    "Summe der quadrierten Abstände zum eigenen Zentrum – kleiner = kompakter, sinkt aber mit jedem zusätzlichen "
    "Cluster.": (
        "Sum of squared distances to the own centre – smaller = more compact, but it falls with every additional "
        "cluster.",
        "Suma kwadratów odległości od własnego centrum – mniejsza = bardziej zwarte, ale maleje z każdym "
        "dodatkowym klastrem."),
    "Kleiner = besser: Cluster kompakt und weit voneinander entfernt.": (
        "Smaller = better: compact clusters far apart from each other.",
        "Mniejszy = lepiej: klastry zwarte i daleko od siebie."),
    "erklärte Streuung (R²)": ("explained variance (R²)", "wyjaśniona zmienność (R²)"),
    "Anteil der Gesamtstreuung, den die Cluster erklären = 1 − Inertia / Gesamtstreuung.": (
        "Share of the total variance that the clusters explain = 1 − inertia / total variance.",
        "Udział całkowitej zmienności wyjaśniony przez klastry = 1 − inercja / całkowita zmienność."),
    "**So liest man die Kurven:** Die Inertia sinkt mit jedem weiteren Cluster – gesucht ist der *Knick* "
    "(Ellbogen), nach dem sie nur noch langsam sinkt. Die Silhouette misst, wie gut die Cluster getrennt sind: "
    "Ihr Maximum ist ein guter Kandidat. Beide sind Hinweise – am Ende zählt, ob sich die Cluster in Schritt 8 "
    "sinnvoll beschreiben lassen.": (
        "**How to read the curves:** the inertia falls with every additional cluster – look for the *bend* "
        "(elbow) after which it only falls slowly. The silhouette measures how well the clusters are separated: "
        "its maximum is a good candidate. Both are hints – in the end what counts is whether the clusters can be "
        "described sensibly in step 8.",
        "**Jak czytać krzywe:** inercja maleje z każdym kolejnym klastrem – szukamy *załamania* (łokcia), po "
        "którym maleje już tylko powoli. Silhouette mierzy, jak dobrze klastry są rozdzielone: jej maksimum to "
        "dobry kandydat. Oba to wskazówki – na końcu liczy się, czy klastry da się sensownie opisać w kroku 8."),
    "Davies-Bouldin: kleiner = besser · Calinski-Harabasz: größer = besser.": (
        "Davies-Bouldin: smaller = better · Calinski-Harabasz: larger = better.",
        "Davies-Bouldin: mniejszy = lepiej · Calinski-Harabasz: większy = lepiej."),
    "Die Elbow-/Silhouette-Kurve ist in Schritt 6 ausgeschaltet – mit Kurve zeigt diese Seite, welches K zu den "
    "Daten passt.": (
        "The elbow/silhouette curve is switched off in step 6 – with the curve this page shows which K suits the "
        "data.",
        "Krzywa łokcia/Silhouette jest wyłączona w kroku 6 – z krzywą ta strona pokazuje, które K pasuje do "
        "danych."),
    "Silhouetten-Diagramm": ("Silhouette plot", "Wykres Silhouette"),
    "Silhouette nicht berechenbar (weniger als 2 Cluster in den Daten).": (
        "Silhouette cannot be computed (fewer than 2 clusters in the data).",
        "Nie można obliczyć Silhouette (mniej niż 2 klastry w danych)."),
    "**Lesart:** Jede Zeile ist ein waagerechter Strich, je Cluster nach der Silhouette sortiert. Nahe 1 = liegt "
    "mitten im eigenen Cluster, um 0 = auf der Grenze zum nächsten Cluster, unter 0 = näher an einem anderen "
    "Cluster (vermutlich falsch zugeordnet). Ein guter Cluster ragt weit über die gestrichelte Linie "
    "(Mittelwert) hinaus und hat kaum Werte unter 0.": (
        "**How to read it:** every row is a horizontal line, sorted by silhouette within each cluster. Close to "
        "1 = in the middle of its own cluster, around 0 = on the border to the next cluster, below 0 = closer to "
        "another cluster (probably assigned wrongly). A good cluster reaches far beyond the dashed line (mean) "
        "and has hardly any values below 0.",
        "**Jak czytać:** każdy wiersz to pozioma kreska, w każdym klastrze posortowana według Silhouette. Blisko "
        "1 = w środku własnego klastra, około 0 = na granicy z najbliższym klastrem, poniżej 0 = bliżej innego "
        "klastra (prawdopodobnie źle przypisany). Dobry klaster sięga daleko poza przerywaną linię (średnią) i "
        "prawie nie ma wartości poniżej 0."),
    "8 · Cluster-Auswertung – Profile, Darstellung, Vergleich": (
        "8 · Cluster evaluation – profiles, display, comparison",
        "8 · Ocena klastrów – profile, wizualizacja, porównanie"),
    "K-Means mit K = {k} auf {n} Zeilen · Silhouette {s} · erklärte Streuung {r2}": (
        "K-Means with K = {k} on {n} rows · silhouette {s} · explained variance {r2}",
        "K-Means z K = {k} na {n} wierszach · Silhouette {s} · wyjaśniona zmienność {r2}"),
    "Größe und Namen der Cluster": ("Size and names of the clusters", "Wielkość i nazwy klastrów"),
    "**Cluster benennen:** K-Means vergibt nur Nummern (0 … K−1, die Reihenfolge ist willkürlich). Ein Name aus "
    "dem Profil unten macht Ergebnisse lesbar – er wird mit dem Modell gespeichert und in Schritt 10 ausgegeben.": (
        "**Name the clusters:** K-Means only gives numbers (0 … K−1, the order is arbitrary). A name based on "
        "the profile below makes results readable – it is saved with the model and used in step 10.",
        "**Nazwij klastry:** K-Means nadaje tylko numery (0 … K−1, kolejność jest dowolna). Nazwa na podstawie "
        "profilu poniżej czyni wyniki czytelnymi – zostaje zapisana z modelem i podana w kroku 10."),
    "Cluster-Profile – was macht jeden Cluster aus?": (
        "Cluster profiles – what characterises each cluster?",
        "Profile klastrów – co wyróżnia każdy klaster?"),
    "**{label}** ({n} Zeilen · {pct})": ("**{label}** ({n} rows · {pct})", "**{label}** ({n} wierszy · {pct})"),
    "über dem Durchschnitt: ": ("above average: ", "powyżej średniej: "),
    "unter dem Durchschnitt: ": ("below average: ", "poniżej średniej: "),
    "liegt nahe am Durchschnitt aller Zeilen": (
        "close to the average of all rows",
        "blisko średniej wszystkich wierszy"),
    "z = Abweichung des Cluster-Mittelwerts vom Durchschnitt aller Zeilen in Standardabweichungen (σ): +1 = eine "
    "Standardabweichung darüber. Rot = über, Blau = unter dem Durchschnitt. Dummies: Anteil der Kategorie im "
    "Cluster.": (
        "z = deviation of the cluster mean from the average of all rows in standard deviations (σ): +1 = one "
        "standard deviation above. Red = above, blue = below average. Dummies: share of the category in the "
        "cluster.",
        "z = odchylenie średniej klastra od średniej wszystkich wierszy w odchyleniach standardowych (σ): +1 = "
        "jedno odchylenie standardowe powyżej. Czerwony = powyżej, niebieski = poniżej średniej. Dummy: udział "
        "kategorii w klastrze."),
    "**Mittelwerte je Cluster in Originaleinheiten** (Dummies: Anteil 0 … 1)": (
        "**Means per cluster in original units** (dummies: share 0 … 1)",
        "**Średnie w klastrach w jednostkach oryginalnych** (dummy: udział 0 … 1)"),
    "Welche Features trennen die Cluster?": (
        "Which features separate the clusters?",
        "Które cechy rozdzielają klastry?"),
    "Trennkraft je Feature (R²)": ("Separating power per feature (R²)", "Siła rozdzielania cechy (R²)"),
    "Anteil der Streuung des Features, den die Cluster erklären (0 … 1)": (
        "share of the feature's variance explained by the clusters (0 … 1)",
        "udział zmienności cechy wyjaśniony przez klastry (0 … 1)"),
    "R² nahe 1: Das Feature unterscheidet sich fast nur zwischen den Clustern – es prägt die Einteilung. Nahe 0: "
    "innerhalb der Cluster genauso verschieden wie insgesamt. Dummies einer Kategorie zählen zusammen.": (
        "R² close to 1: the feature differs almost only between the clusters – it shapes the grouping. Close to "
        "0: as varied within the clusters as overall. Dummies of one category count together.",
        "R² blisko 1: cecha różni się prawie wyłącznie między klastrami – kształtuje podział. Blisko 0: wewnątrz "
        "klastrów tak samo zróżnicowana jak ogółem. Dummy jednej kategorii liczą się razem."),
    "Darstellung": ("Display", "Wizualizacja"),
    "2D-Karte (PCA)": ("2D map (PCA)", "Mapa 2D (PCA)"),
    "Zwei Features": ("Two features", "Dwie cechy"),
    "Farbe": ("Colour", "Kolor"),
    "bekannte Einteilung: {g}": ("known grouping: {g}", "znany podział: {g}"),
    "**PCA nur für das Bild:** Die Daten haben {p} Dimensionen (Spalten) – für die Grafik werden sie auf die 2 "
    "Richtungen mit der größten Streuung projiziert (Hauptkomponenten PC1, PC2). K-Means selbst rechnet mit "
    "allen Spalten; Cluster, die sich hier überlappen, können im vollen Raum getrennt sein. Zusammen zeigen PC1 "
    "und PC2 {pct} der Streuung.": (
        "**PCA only for the picture:** the data have {p} dimensions (columns) – for the chart they are projected "
        "onto the 2 directions with the largest variance (principal components PC1, PC2). K-Means itself works "
        "with all columns; clusters that overlap here can be separated in the full space. Together PC1 and PC2 "
        "show {pct} of the variance.",
        "**PCA tylko do obrazka:** dane mają {p} wymiarów (kolumn) – na wykresie rzutuje się je na 2 kierunki o "
        "największej zmienności (składowe główne PC1, PC2). Sam K-Means liczy na wszystkich kolumnach; klastry, "
        "które tu się nakładają, mogą być rozdzielone w pełnej przestrzeni. Razem PC1 i PC2 pokazują {pct} "
        "zmienności."),
    "Ladung = Gewicht eines Features in der Hauptkomponente (die 5 größten je PC).": (
        "Loading = weight of a feature in the principal component (the 5 largest per PC).",
        "Ładunek = waga cechy w składowej głównej (5 największych na PC)."),
    "Mindestens 2 Zahlen-Features nötig.": (
        "At least 2 numeric features needed.",
        "Potrzebne co najmniej 2 cechy liczbowe."),
    "Wie im Kurs (Grad.Rate gegen Room.Board, Farbe = Private) – nur mit den gefundenen Clustern als Farbe. Zwei "
    "Features zeigen nur einen Ausschnitt: Die Cluster entstehen aus allen Features.": (
        "As in the course (Grad.Rate vs. Room.Board, colour = Private) – only with the clusters found as colour. "
        "Two features show only part of the picture: the clusters come from all features.",
        "Jak w kursie (Grad.Rate względem Room.Board, kolor = Private) – tylko z kolorem według znalezionych "
        "klastrów. Dwie cechy pokazują tylko wycinek: klastry powstają ze wszystkich cech."),
    "Typische Vertreter je Cluster": ("Typical members of each cluster", "Typowi przedstawiciele klastrów"),
    "Die Zeilen am nächsten am Zentrum ihres Clusters – daran lässt sich ein Cluster oft am schnellsten "
    "benennen. Randfälle liegen am weitesten vom Zentrum entfernt.": (
        "The rows closest to the centre of their cluster – often the quickest way to name a cluster. Edge cases "
        "are furthest from the centre.",
        "Wiersze najbliżej centrum swojego klastra – często najszybszy sposób, by nazwać klaster. Przypadki "
        "brzegowe leżą najdalej od centrum."),
    "Zufällige Zeile ziehen: Zu welchem Cluster gehört sie, und wie weit ist sie von allen Zentren entfernt?": (
        "Draw a random row: which cluster does it belong to, and how far is it from all centres?",
        "Wylosuj wiersz: do którego klastra należy i jak daleko jest od wszystkich centrów?"),
    "Zufällige Zeile ziehen": ("Draw a random row", "Wylosuj wiersz"),
    "Abstand zum Zentrum": ("Distance to the centre", "Odległość od centrum"),
    "Euklidischer Abstand in den skalierten Features. Zum Vergleich: Median im Cluster {m}.": (
        "Euclidean distance in the scaled features. For comparison: median in the cluster {m}.",
        "Odległość euklidesowa w skalowanych cechach. Dla porównania: mediana w klastrze {m}."),
    "Bekannte Einteilung {g}: **{v}**": ("Known grouping {g}: **{v}**", "Znany podział {g}: **{v}**"),
    "Abstand zu allen Zentren": ("Distance to all centres", "Odległość od wszystkich centrów"),
    "Abstand (skalierte Features) – der kleinste bestimmt den Cluster": (
        "distance (scaled features) – the smallest decides the cluster",
        "odległość (skalowane cechy) – najmniejsza decyduje o klastrze"),
    "Vergleich mit der bekannten Einteilung: {g}": (
        "Comparison with the known grouping: {g}",
        "Porównanie ze znanym podziałem: {g}"),
    "Keine Zeile hat einen Wert in '{g}'.": (
        "No row has a value in '{g}'.",
        "Żaden wiersz nie ma wartości w '{g}'."),
    "K-Means hat '{g}' nie gesehen. Weil die Cluster-Nummern willkürlich sind, bekommt jeder Cluster die Gruppe, "
    "die in ihm am häufigsten ist – dann wird gezählt, wie oft das stimmt.": (
        "K-Means never saw '{g}'. Because the cluster numbers are arbitrary, every cluster gets the group that "
        "is most frequent in it – then we count how often that is right.",
        "K-Means nigdy nie widział '{g}'. Ponieważ numery klastrów są dowolne, każdy klaster dostaje grupę, "
        "która jest w nim najczęstsza – potem liczy się, jak często to się zgadza."),
    "Treffer nach Zuordnung": ("Accuracy after mapping", "Trafność po przypisaniu"),
    "Anteil der Zeilen, deren Gruppe mit der Gruppe ihres Clusters übereinstimmt (Purity).": (
        "Share of rows whose group matches the group of their cluster (purity).",
        "Udział wierszy, których grupa zgadza się z grupą ich klastra (purity)."),
    "Immer die häufigste Gruppe ('{v}') – so oft läge man auch ohne Clustering richtig.": (
        "Always the most frequent group ('{v}') – this is how often you would be right even without clustering.",
        "Zawsze najczęstsza grupa ('{v}') – tak często miałoby się rację nawet bez klastrowania."),
    "Adjusted Rand Index: 0 = nicht besser als Zufall, 1 = identische Einteilung – unabhängig von den "
    "Cluster-Nummern.": (
        "Adjusted Rand Index: 0 = no better than chance, 1 = identical grouping – independent of the cluster "
        "numbers.",
        "Adjusted Rand Index: 0 = nie lepiej niż losowo, 1 = identyczny podział – niezależnie od numerów "
        "klastrów."),
    "Normalized Mutual Information: 0 = Cluster verraten nichts über die Gruppe, 1 = alles.": (
        "Normalized Mutual Information: 0 = clusters reveal nothing about the group, 1 = everything.",
        "Normalized Mutual Information: 0 = klastry nic nie mówią o grupie, 1 = wszystko."),
    "**Zuordnung Cluster → Gruppe**": ("**Mapping cluster → group**", "**Przypisanie klaster → grupa**"),
    "- {label} → **{v}** ({hit} von {total})": (
        "- {label} → **{v}** ({hit} of {total})",
        "- {label} → **{v}** ({hit} z {total})"),
    "Keine Mehrheit in einem Cluster: {v} – K-Means hat diese Gruppe(n) nicht als eigenen Cluster gefunden.": (
        "No majority in any cluster: {v} – K-Means did not find this group (these groups) as a cluster of its "
        "own.",
        "Brak większości w żadnym klastrze: {v} – K-Means nie znalazł tej grupy (tych grup) jako osobnego "
        "klastra."),
    "Die Cluster treffen die Gruppen nicht besser als die naive Vorhersage – sie folgen einer anderen Struktur "
    "in den Daten als '{g}'.": (
        "The clusters do not match the groups better than the naive prediction – they follow a different "
        "structure in the data than '{g}'.",
        "Klastry nie trafiają grup lepiej niż prognoza naiwna – odzwierciedlają inną strukturę w danych niż "
        "'{g}'."),
    "**Wie im Kurs** (Cluster-Nummer direkt mit der Gruppen-Nummer {codes} verglichen): {raw:.1%} – nach der "
    "Zuordnung {acc:.1%}.": (
        "**As in the course** (cluster number compared directly with the group number {codes}): {raw:.1%} – "
        "after the mapping {acc:.1%}.",
        "**Jak w kursie** (numer klastra porównany wprost z numerem grupy {codes}): {raw:.1%} – po przypisaniu "
        "{acc:.1%}."),
    "Der Unterschied liegt nur an den willkürlichen Nummern: K-Means weiß nicht, dass Cluster 1 '{v}' heißen "
    "soll.": (
        "The difference is only due to the arbitrary numbers: K-Means does not know that cluster 1 is supposed "
        "to mean '{v}'.",
        "Różnica wynika tylko z dowolnych numerów: K-Means nie wie, że klaster 1 ma oznaczać '{v}'."),
    "Hier gehören mehrere Cluster zur selben Gruppe – die Zuordnung ist deshalb mehr als eine Umnummerierung.": (
        "Here several clusters belong to the same group – so the mapping is more than a renumbering.",
        "Tutaj kilka klastrów należy do tej samej grupy – przypisanie to więc coś więcej niż zmiana numeracji."),
    "Die Nummern passen hier schon zu den Gruppen.": (
        "Here the numbers already match the groups.",
        "Tutaj numery już pasują do grup."),
    "leer – keine Zeile liegt diesem Zentrum am nächsten (doppelte Zeilen, K kleiner wählen)": (
        "empty – no row is closest to this centre (duplicate rows, choose a smaller K)",
        "pusty – żaden wiersz nie leży najbliżej tego centrum (powtórzone wiersze, wybierz mniejsze K)"),
    "{cols}: sieht aus wie eine ID (ganze Zahlen, jede Zeile ein anderer Wert) – als Feature würde K-Means nach "
    "Zeilennummern gruppieren. Besser als Bezeichnung der Zeilen wählen und hier abwählen.": (
        "{cols}: looks like an ID (whole numbers, a different value in every row) – as a feature, K-Means would "
        "group by row numbers. Better choose it as the row name and deselect it here.",
        "{cols}: wygląda jak identyfikator (liczby całkowite, w każdym wierszu inna wartość) – jako cecha "
        "sprawiłby, że K-Means grupowałby według numerów wierszy. Lepiej wybrać go jako nazwę wiersza i odznaczyć "
        "tutaj."),
    "Confusion Matrix und classification_report nach der Zuordnung": (
        "Confusion matrix and classification_report after the mapping",
        "Macierz pomyłek i classification_report po przypisaniu"),
    "Gespeichert werden **zwei Dateien** in `models/<name>/`: das K-Means-Modell (`model.skops` oder "
    "`model.joblib`, mit den Cluster-Zentren) und `preprocessing.json` – das Rezept aus Schritt 2 und 4 "
    "(Spalten-Aktionen, Füllwerte, Kategorien/Dummy-Spalten, Scaler) mit den Einstellungen aus Schritt 6, den "
    "Cluster-Namen aus Schritt 8 und den Kennzahlen. Ohne das Rezept lassen sich neue Daten nicht korrekt "
    "zuordnen.": (
        "**Two files** are saved in `models/<name>/`: the K-Means model (`model.skops` or `model.joblib`, with "
        "the cluster centres) and `preprocessing.json` – the recipe from steps 2 and 4 (column actions, fill "
        "values, categories/dummy columns, scaler) with the settings from step 6, the cluster names from step 8 "
        "and the metrics. Without the recipe new data cannot be assigned correctly.",
        "Zapisywane są **dwa pliki** w `models/<name>/`: model K-Means (`model.skops` lub `model.joblib`, z "
        "centrami klastrów) oraz `preprocessing.json` – receptura z kroków 2 i 4 (akcje kolumn, wartości "
        "uzupełniające, kategorie/kolumny dummy, scaler) z ustawieniami z kroku 6, nazwami klastrów z kroku 8 i "
        "miarami. Bez receptury nowych danych nie da się poprawnie przypisać."),
    "Cluster-Namen: {v}": ("Cluster names: {v}", "Nazwy klastrów: {v}"),
    " – noch keine Namen (Schritt 8: Cluster benennen)": (
        " – no names yet (step 8: name the clusters)",
        " – jeszcze bez nazw (krok 8: nazwij klastry)"),
    "{name} · K-Means · K = {k} · {n} Features · {v}": (
        "{name} · K-Means · K = {k} · {n} features · {v}",
        "{name} · K-Means · K = {k} · {n} cech · {v}"),
    "**Geladen: {name}** · K-Means · K = {k} · {n} Features → {n2} Eingänge · gespeichert {v}": (
        "**Loaded: {name}** · K-Means · K = {k} · {n} features → {n2} inputs · saved {v}",
        "**Wczytano: {name}** · K-Means · K = {k} · {n} cech → {n2} wejść · zapisano {v}"),
    "Cluster: ": ("Clusters: ", "Klastry: "),
    "**Kennzahlen (Training)**": ("**Metrics (training)**", "**Miary (trening)**"),
    "**Einstellungen**": ("**Settings**", "**Ustawienia**"),
    "10 · Neue Daten den Clustern zuordnen": (
        "10 · Assign new data to the clusters",
        "10 · Przypisz nowe dane do klastrów"),
    "Cluster zuordnen": ("Assign clusters", "Przypisz klastry"),
    "Bereite neue Daten auf und ordne zu …": (
        "Preparing new data and assigning …",
        "Przygotowuję nowe dane i przypisuję …"),
    "{n} Zeilen zugeordnet ({name})": ("{n} rows assigned ({name})", "Przypisano {n} wierszy ({name})"),
    "**Lesart:** `Cluster` = Nummer des nächsten Zentrums, `Abstand` = euklidischer Abstand dazu (skalierte "
    "Features – klein = typisch für den Cluster). `Hinweis` nennt ergänzte Lücken, unbekannte Kategorien, Werte "
    "außerhalb des Trainingsbereichs, Zeilen weit weg von allen Clustern und knappe Zuordnungen zwischen zwei "
    "Clustern.": (
        "**How to read it:** `Cluster` = number of the nearest centre, `Distance` = Euclidean distance to it "
        "(scaled features – small = typical for the cluster). `Note` names filled gaps, unknown categories, "
        "values outside the training range, rows far from all clusters and close calls between two clusters.",
        "**Jak czytać:** `Klaster` = numer najbliższego centrum, `Odległość` = odległość euklidesowa od niego "
        "(skalowane cechy – mała = typowy dla klastra). `Uwaga` wymienia uzupełnione braki, nieznane kategorie, "
        "wartości poza zakresem treningowym, wiersze daleko od wszystkich klastrów i przypisania na granicy "
        "dwóch klastrów."),
    "'{g}' ist enthalten → Vergleich mit der Zuordnung aus dem Training ({n} Zeilen)": (
        "'{g}' is included → comparison with the mapping from training ({n} rows)",
        "'{g}' jest w danych → porównanie z przypisaniem z treningu ({n} wierszy)"),
    "cluster_{stem}.csv": ("clusters_{stem}.csv", "klastry_{stem}.csv"),
    "**Was die Zuordnung aussagt**": ("**What the assignment means**", "**Co mówi przypisanie**"),
    "Werte eingeben (vorbelegt mit Median bzw. häufigstem Wert aus dem Training). Eingegeben wird die Form nach "
    "Schritt 2.": (
        "Enter values (prefilled with the median or most frequent value from training). Enter the form after "
        "step 2.",
        "Wpisz wartości (wstępnie wypełnione medianą lub najczęstszą wartością z treningu). Wpisuje się postać "
        "po kroku 2."),
    "Hinweis: {v}": ("Note: {v}", "Uwaga: {v}"),
    "Python-Code (Zuordnung ohne App)": (
        "Python code (assignment without the app)",
        "Kod Python (przypisanie bez aplikacji)"),
    "Clustering-Daten: X {shape}": ("Clustering data: X {shape}", "Dane do klastrowania: X {shape}"),
    "Clustering-Daten vorbereitet": ("Clustering data prepared", "Dane do klastrowania przygotowane"),
    "K-Means trainiert (K = {k})": ("K-Means trained (K = {k})", "K-Means wytrenowany (K = {k})"),
    "K-Means trainiert": ("K-Means trained", "K-Means wytrenowany"),
    "**So arbeitet K-Means**\n\n1. **K Startzentren** wählen – mit *k-means++* gut verteilt, mit *random* "
    "zufällige Zeilen.\n2. **Zuordnen:** Jede Zeile kommt zum nächsten Zentrum (euklidischer Abstand in den "
    "skalierten Features).\n3. **Verschieben:** Jedes Zentrum wandert in den Mittelwert seiner Zeilen. Schritt 2 "
    "und 3 wiederholen, bis sich\n   nichts mehr ändert (höchstens *max_iter*-mal).\n4. Alles **n_init-mal** mit "
    "anderen Startzentren – es gewinnt die Lösung mit der kleinsten **Inertia** (Summe der\n   quadrierten "
    "Abstände jeder Zeile zu ihrem Zentrum).\n\nK-Means findet eher runde, ähnlich große Gruppen. K gibt man vor "
    "– welches K zu den Daten passt, zeigt Schritt 7.": (
        "**How K-Means works**\n\n1. Choose **K starting centres** – well spread with *k-means++*, random rows "
        "with *random*.\n2. **Assign:** every row goes to the nearest centre (Euclidean distance in the scaled "
        "features).\n3. **Move:** every centre moves to the mean of its rows. Repeat steps 2 and 3 until nothing "
        "changes any more\n   (at most *max_iter* times).\n4. Do all of this **n_init times** with other "
        "starting centres – the solution with the smallest **inertia** wins (sum\n   of the squared distances of "
        "every row to its centre).\n\nK-Means tends to find round groups of similar size. You choose K – step 7 "
        "shows which K suits the data.",
        "**Jak działa K-Means**\n\n1. Wybierz **K centrów startowych** – przy *k-means++* dobrze rozłożone, przy "
        "*random* losowe wiersze.\n2. **Przypisz:** każdy wiersz trafia do najbliższego centrum (odległość "
        "euklidesowa w skalowanych cechach).\n3. **Przesuń:** każde centrum przesuwa się do średniej swoich "
        "wierszy. Powtarzaj kroki 2 i 3, aż nic się już nie\n   zmienia (najwyżej *max_iter* razy).\n4. Całość "
        "**n_init razy** z innymi centrami startowymi – wygrywa rozwiązanie z najmniejszą **inercją** (suma\n   "
        "kwadratów odległości każdego wiersza od jego centrum).\n\nK-Means znajduje raczej okrągłe grupy "
        "podobnej wielkości. K podaje się z góry – które K pasuje do danych, pokazuje\nkrok 7."),
    "\n**So werden neue Daten zugeordnet**\n\n1. **Nichts neu fitten.** Neue Daten werden mit dem *gespeicherten "
    "Rezept* aufbereitet – gleiche Spalten-Aktionen und\n   neue Spalten, gleiche Füllwerte, gleiche "
    "Dummy-Spalten in gleicher Reihenfolge, gleicher Scaler. Danach kommt jede\n   Zeile zum **nächsten "
    "Cluster-Zentrum** (`kmeans.predict`) – die Zentren bleiben, wie sie sind.\n2. **Gleiche Rohspalten wie im "
    "Training.** Das Modell bestimmt die Features – die Tabelle unten zeigt, welche\n   CSV-Spalten nötig sind "
    "(Vorlage zum Download). Fehlen Spalten → klare Fehlermeldung. Zusätzliche Spalten werden\n   ignoriert.\n3. "
    "**Abweichungen werden behandelt und gemeldet:** Lücken → Füllwerte aus dem Training; unbekannte Kategorien "
    "→ alle\n   Dummies 0; Werte außerhalb des Trainingsbereichs → Hinweis. Zeilen, die weiter vom nächsten "
    "Zentrum entfernt sind\n   als 99 % der Trainingszeilen dieses Clusters, bekommen den Hinweis *weit weg von "
    "allen Clustern*.\n4. **Interpretation und Fazit:** Verteilung auf die Cluster im Vergleich zum Training, "
    "Ausreißer und knappe\n   Zuordnungen – und wenn die neuen Daten die bekannte Einteilung enthalten (z. B. "
    "Private), der Vergleich mit der\n   Zuordnung Cluster → Gruppe aus dem Training.\n": (
        "\n**How new data are assigned**\n\n1. **Nothing is fitted again.** New data are prepared with the "
        "*saved recipe* – same column actions and new columns,\n   same fill values, same dummy columns in the "
        "same order, same scaler. Then every row goes to the **nearest cluster\n   centre** (`kmeans.predict`) – "
        "the centres stay as they are.\n2. **Same raw columns as in training.** The model determines the "
        "features – the table below shows which CSV columns\n   are needed (template to download). Missing "
        "columns → clear error message. Additional columns are ignored.\n3. **Deviations are handled and "
        "reported:** gaps → fill values from training; unknown categories → all dummies 0;\n   values outside "
        "the training range → note. Rows further from the nearest centre than 99 % of the training rows of\n   "
        "that cluster get the note *far from all clusters*.\n4. **Interpretation and conclusion:** distribution "
        "across the clusters compared with training, outliers and close\n   calls – and, if the new data contain "
        "the known grouping (e.g. Private), the comparison with the cluster → group\n   assignment from "
        "training.\n",
        "\n**Jak przypisywane są nowe dane**\n\n1. **Nic nie jest dopasowywane od nowa.** Nowe dane są "
        "przygotowywane *zapisaną recepturą* – te same akcje kolumn i\n   nowe kolumny, te same wartości "
        "uzupełniające, te same kolumny dummy w tej samej kolejności, ten sam scaler. Potem\n   każdy wiersz "
        "trafia do **najbliższego centrum klastra** (`kmeans.predict`) – centra pozostają bez zmian.\n2. **Te "
        "same surowe kolumny co w treningu.** Model wyznacza cechy – tabela poniżej pokazuje, które kolumny CSV "
        "są\n   potrzebne (szablon do pobrania). Brakujące kolumny → jasny komunikat błędu. Dodatkowe kolumny są "
        "pomijane.\n3. **Odchylenia są obsługiwane i zgłaszane:** braki → wartości uzupełniające z treningu; "
        "nieznane kategorie → wszystkie\n   dummy 0; wartości poza zakresem treningowym → uwaga. Wiersze dalej "
        "od najbliższego centrum niż 99 % wierszy\n   treningowych tego klastra dostają uwagę *daleko od "
        "wszystkich klastrów*.\n4. **Interpretacja i podsumowanie:** rozkład na klastry w porównaniu z "
        "treningiem, wartości odstające i przypisania na\n   granicy – a jeśli nowe dane zawierają znany podział "
        "(np. Private), porównanie z przypisaniem klaster → grupa z\n   treningu.\n"),
}
