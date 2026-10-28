"""Übersetzungen für pipeline.py: {Deutsch: (Englisch, Polnisch)}"""

T = {
    # Problemtypen
    "keiner": ("none", "brak"),
    # Spalten-Aktionen (Schritt 2)
    "behalten": ("Keep", "Zachowaj"),
    "entfernen": ("Drop", "Usuń"),
    "Zahl extrahieren": ("Extract number", "Wyodrębnij liczbę"),
    "Datum → Jahr": ("Date → year", "Data → rok"),
    "Datum → Monat": ("Date → month", "Data → miesiąc"),
    "Datum → Jahr + Monat": ("Date → year + month", "Data → rok + miesiąc"),
    "Text: erste N Zeichen": ("Text: first N characters", "Tekst: pierwsze N znaków"),
    "Text: letzte N Zeichen": ("Text: last N characters", "Tekst: ostatnie N znaków"),
    "Werte ersetzen": ("Replace values", "Zamień wartości"),
    "Top-N behalten, Rest → Sonstige": ("Keep top N, rest → Sonstige (other)",
                                        "Zachowaj top N, resztę → Sonstige (inne)"),
    "als Kategorie (Text)": ("As category (text)", "Jako kategoria (tekst)"),
    "unverändert; Text bleibt Kategorie und wird in Schritt 4 zu Dummies": (
        "unchanged; text stays a category and becomes dummy columns in step 4",
        "bez zmian; tekst pozostaje kategorią i w kroku 4 staje się kolumnami dummy"),
    "Spalte löschen (z. B. emp_title, title oder Datenleck wie issue_d)": (
        "delete the column (e.g. emp_title, title or a data leak such as issue_d)",
        "usuń kolumnę (np. emp_title, title albo wyciek danych jak issue_d)"),
    "erste Zahl aus dem Text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5 €' → 1234.5": (
        "first number in the text: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5 €' → 1234.5",
        "pierwsza liczba z tekstu: ' 36 months' → 36, '13,5 %' → 13.5, '$1,200' → 1200, '1.234,5 €' → 1234.5"),
    "Datum → Jahr: 'Jun-1990' → 1990, auch Zahl 20141013 → 2014": (
        "date → year: 'Jun-1990' → 1990, also the number 20141013 → 2014",
        "data → rok: 'Jun-1990' → 1990, także liczba 20141013 → 2014"),
    "optional Format, z. B. %b-%Y": ("optional format, e.g. %b-%Y", "opcjonalnie format, np. %b-%Y"),
    "Datum → Monat (1–12)": ("date → month (1–12)", "data → miesiąc (1–12)"),
    "optional Format, z. B. %d.%m.%Y": ("optional format, e.g. %d.%m.%Y", "opcjonalnie format, np. %d.%m.%Y"),
    "zwei neue Zahlenspalten: <Name>_jahr und <Name>_monat ('Oct-2014' → 2014 und 10)": (
        "two new number columns: <name>_jahr and <name>_monat ('Oct-2014' → 2014 and 10)",
        "dwie nowe kolumny liczbowe: <nazwa>_jahr i <nazwa>_monat ('Oct-2014' → 2014 i 10)"),
    "N (Standard 1)": ("N (default 1)", "N (domyślnie 1)"),
    "erste N Zeichen: 'B4' → 'B'": ("first N characters: 'B4' → 'B'", "pierwsze N znaków: 'B4' → 'B'"),
    "N (Standard 5)": ("N (default 5)", "N (domyślnie 5)"),
    "letzte N Zeichen: Adresse → PLZ '22690'": ("last N characters: address → ZIP code '22690'",
                                                "ostatnie N znaków: adres → kod pocztowy '22690'"),
    "Werte ersetzen: NONE,ANY=OTHER": ("replace values: NONE,ANY=OTHER", "zamiana wartości: NONE,ANY=OTHER"),
    "alt1,alt2=neu; alt3=neu2": ("old1,old2=new; old3=new2", "stara1,stara2=nowa; stara3=nowa2"),
    "N (Standard 10)": ("N (default 10)", "N (domyślnie 10)"),
    "nur die N häufigsten Werte behalten, Rest → 'Sonstige'": (
        "keep only the N most frequent values, rest → 'Sonstige'",
        "zachowaj tylko N najczęstszych wartości, resztę → 'Sonstige'"),
    "Zahlen als Kategorie behandeln (Codes, PLZ als Zahl …)": (
        "treat numbers as categories (codes, ZIP codes stored as numbers …)",
        "traktuj liczby jako kategorie (kody, kody pocztowe zapisane jako liczby …)"),
    # Fehlende Werte (Schritt 4)
    "Zeilen entfernen": ("Drop rows", "Usuń wiersze"),
    "Spalte entfernen": ("Drop column", "Usuń kolumnę"),
    "Mittelwert": ("Mean", "Średnia"),
    "Median": ("Median", "Mediana"),
    "häufigster Wert": ("Most frequent value", "Najczęstsza wartość"),
    "Konstante": ("Constant", "Stała"),
    "Gruppen-Mittelwert": ("Group mean", "Średnia grupowa"),
    "df.dropna(subset=[…]) – sinnvoll bei < 1 % fehlend": ("df.dropna(subset=[…]) – useful when < 1 % is missing",
                                                          "df.dropna(subset=[…]) – sensowne przy < 1 % braków"),
    "Spalte komplett weglassen – bei sehr vielen Lücken": ("leave out the whole column – when there are very many gaps",
                                                           "pomiń całą kolumnę – przy bardzo wielu brakach"),
    "Median der Spalte in den Trainingsdaten (robust gegen Ausreißer)": (
        "median of the column in the training data (robust to outliers)",
        "mediana kolumny w danych treningowych (odporna na wartości odstające)"),
    "Mittelwert der Spalte in den Trainingsdaten": ("mean of the column in the training data",
                                                    "średnia kolumny w danych treningowych"),
    "häufigster Wert in den Trainingsdaten – auch für Kategorien": (
        "most frequent value in the training data – also for categories",
        "najczęstsza wartość w danych treningowych – także dla kategorii"),
    "Wert, z. B. 0 oder fehlend": ("value, e.g. 0 or fehlend", "wartość, np. 0 albo fehlend"),
    "fester Wert; bei Kategorien wird 'fehlend' eine eigene Kategorie": (
        "fixed value; for categories 'fehlend' (missing) becomes a category of its own",
        "stała wartość; przy kategoriach 'fehlend' (brak) staje się osobną kategorią"),
    "Gruppenspalte, z. B. total_acc": ("group column, e.g. total_acc", "kolumna grupująca, np. total_acc"),
    "Mittelwert je Gruppe in den Trainingsdaten, z. B. mort_acc je total_acc – passende Gruppe: Korrelation "
    "(Hilfe unten)": (
        "mean per group in the training data, e.g. mort_acc per total_acc – finding a suitable group: "
        "correlation (help below)",
        "średnia w grupie w danych treningowych, np. mort_acc według total_acc – dobór grupy: korelacja "
        "(pomoc poniżej)"),
    # Fehlermeldungen und Protokoll
    "Es fehlen Spalten: ": ("Missing columns: ", "Brakuje kolumn: "),
    "Parameter '{param}' ist keine ganze Zahl.": ("Parameter '{param}' is not a whole number.",
                                                  "Parametr '{param}' nie jest liczbą całkowitą."),
    "Parameter N muss ≥ 1 sein.": ("Parameter N must be ≥ 1.", "Parametr N musi być ≥ 1."),
    "Werte ersetzen: '{part}' hat kein '=' (Format: alt1,alt2=neu)": (
        "Replace values: '{part}' has no '=' (format: old1,old2=new)",
        "Zamień wartości: '{part}' nie zawiera '=' (format: stara1,stara2=nowa)"),
    "Werte ersetzen: Parameter fehlt (Format: alt1,alt2=neu)": (
        "Replace values: parameter missing (format: old1,old2=new)",
        "Zamień wartości: brak parametru (format: stara1,stara2=nowa)"),
    "'{v}' ist keine Zahl (Spalte ist numerisch).": ("'{v}' is not a number (the column is numeric).",
                                                     "'{v}' nie jest liczbą (kolumna jest liczbowa)."),
    "Die Datei enthält keine Konfiguration (JSON-Objekt erwartet).": (
        "The file does not contain a configuration (JSON object expected).",
        "Plik nie zawiera konfiguracji (oczekiwano obiektu JSON)."),
    "{key} ({v} Einträge)": ("{key} ({v} entries)", "{key} (wpisy: {v})"),
    "Bibliothek": ("Library", "Biblioteka"),
    "Version": ("Version", "Wersja"),
    "Datei nicht gefunden: {path}": ("File not found: {path}", "Nie znaleziono pliku: {path}"),
    "Trennzeichen": ("Separator", "Separator"),
    "Dezimalkomma": ("Decimal comma", "Przecinek dziesiętny"),
    "Die Beschreibungs-CSV braucht mindestens 2 Spalten (Name, Beschreibung).": (
        "The description CSV needs at least 2 columns (name, description).",
        "Plik CSV z opisami musi mieć co najmniej 2 kolumny (nazwa, opis)."),
    # Datentypen
    "Bool": ("Bool", "Logiczna (bool)"),
    "Kategorie": ("Category", "Kategoria"),
    "Text": ("Text", "Tekst"),
    "Zahl (float)": ("Number (float)", "Liczba (float)"),
    "Zahl (int)": ("Number (int)", "Liczba (int)"),
    "unverändert": ("unchanged", "bez zmian"),
    "ist bereits numerisch": ("is already numeric", "jest już liczbowa"),
    "Zahl extrahiert ({v:.0%} gültig)": ("number extracted ({v:.0%} valid)", "wyodrębniono liczbę ({v:.0%} poprawnych)"),
    "Zahl, aber kein Datum – erwartet Text wie 'Oct-2014' oder eine Zahl wie 20141013.": (
        "A number but not a date – expected text such as 'Oct-2014' or a number such as 20141013.",
        "Liczba, ale nie data – oczekiwano tekstu jak 'Oct-2014' albo liczby jak 20141013."),
    "ist bereits numerisch (unverändert)": ("is already numeric (unchanged)", "jest już liczbowa (bez zmian)"),
    "({v:.0%} erkannt)": ("({v:.0%} recognised)", "(rozpoznano {v:.0%})"),
    "Jahr + Monat {found}": ("year + month {found}", "rok + miesiąc {found}"),
    "Jahr aus Datum {found}": ("year from date {found}", "rok z daty {found}"),
    "Monat aus Datum {found}": ("month from date {found}", "miesiąc z daty {found}"),
    "erste {n} Zeichen": ("first {n} character(s)", "pierwsze znaki (N = {n})"),
    "letzte {n} Zeichen": ("last {n} character(s)", "ostatnie znaki (N = {n})"),
    "Werte ersetzt {mapping}": ("values replaced {mapping}", "zamieniono wartości {mapping}"),
    "Top-{n} behalten, Rest → '{other}'": ("top {n} kept, rest → '{other}'", "zachowano top {n}, reszta → '{other}'"),
    "Unbekannte Aktion: {act}": ("Unknown action: {act}", "Nieznana akcja: {act}"),
    "{col}: entfernt": ("{col}: removed", "{col}: usunięta"),
    "Spalte '{col}' ist nicht in den Daten.": ("Column '{col}' is not in the data.", "Kolumny '{col}' nie ma w danych."),
    "Spalte '{col}' ({act}): {exc}": ("Column '{col}' ({act}): {exc}", "Kolumna '{col}' ({act}): {exc}"),
    "Neuer Name {clash} (für '{col}') existiert bereits als Spalte.": (
        "New name {clash} (for '{col}') already exists as a column.",
        "Nowa nazwa {clash} (dla '{col}') już istnieje jako kolumna."),
    ", umbenannt in '{target}'": (", renamed to '{target}'", ", zmieniono nazwę na '{target}'"),
    # Neue Spalten
    "Werte zuordnen": ("Map values", "Przypisz wartości"),
    "Formel": ("Formula", "Formuła"),
    "Bedingung → 0/1": ("Condition → 0/1", "Warunek → 0/1"),
    "Kopie + Aktion": ("Copy + action", "Kopia + akcja"),
    "Quellspalte · alt=neu; …; sonst=…": ("Source column · old=new; …; rest=…",
                                          "Kolumna źródłowa · stara=nowa; …; rest=…"),
    "z. B. loan_repaid aus loan_status: Fully Paid=1; Charged Off=0 – nicht genannte Werte bleiben leer (oder "
    "'sonst=…'); sind alle neuen Werte Zahlen, wird die Spalte eine Zahlenspalte": (
        "e.g. loan_repaid from loan_status: Fully Paid=1; Charged Off=0 – values not listed stay empty (or use "
        "'rest=…'); if all new values are numbers, the column becomes a number column",
        "np. loan_repaid z loan_status: Fully Paid=1; Charged Off=0 – wartości niewymienione zostają puste (albo "
        "'rest=…'); jeśli wszystkie nowe wartości są liczbami, kolumna staje się liczbowa"),
    "Rechnen mit Zahlenspalten: + − * / ** // % und Klammern; Funktionen abs, log, log10, sqrt, exp, round – "
    "z. B. loan_amnt / annual_inc": (
        "Calculate with number columns: + − * / ** // % and brackets; functions abs, log, log10, sqrt, exp, "
        "round – e.g. loan_amnt / annual_inc",
        "Obliczenia na kolumnach liczbowych: + − * / ** // % i nawiasy; funkcje abs, log, log10, sqrt, exp, "
        "round – np. loan_amnt / annual_inc"),
    "1 = erfüllt, 0 = nicht erfüllt, leer bei Lücken: dti > 30 · home_ownership == 'RENT' · (dti > 30) and "
    "(term == 60)": (
        "1 = true, 0 = false, empty where values are missing: dti > 30 · home_ownership == 'RENT' · (dti > 30) "
        "and (term == 60)",
        "1 = spełniony, 0 = niespełniony, puste przy brakach: dti > 30 · home_ownership == 'RENT' · (dti > 30) "
        "and (term == 60)"),
    "Bedingung": ("Condition", "Warunek"),
    "Quellspalte · Aktion · Parameter": ("Source column · action · parameter", "Kolumna źródłowa · akcja · parametr"),
    "wie eine Aktion aus der Tabelle oben – das Original bleibt erhalten": (
        "like an action from the table above – the original is kept",
        "jak akcja z tabeli powyżej – oryginał zostaje zachowany"),
    "Zuordnung": ("Mapping", "Przypisanie"),
    "Zuordnung: mindestens ein Wert nötig, z. B. Fully Paid=1": (
        "Mapping: at least one value needed, e.g. Fully Paid=1",
        "Przypisanie: potrzebna co najmniej jedna wartość, np. Fully Paid=1"),
    "Formel fehlt.": ("Formula missing.", "Brak formuły."),
    "Formel zu lang (höchstens {max_formula_len} Zeichen).": ("Formula too long (at most {max_formula_len} characters).",
                                                             "Formuła za długa (maksymalnie {max_formula_len} znaków)."),
    "Formel ungültig ({msg}).": ("Invalid formula ({msg}).", "Nieprawidłowa formuła ({msg})."),
    "Formel zu stark verschachtelt.": ("Formula nested too deeply.", "Formuła jest zbyt zagnieżdżona."),
    "Formel zu stark verschachtelt – bitte in mehrere neue Spalten aufteilen (z. B. erst teil1 = …, dann "
    "gesamt = teil1 + …).": (
        "Formula nested too deeply – please split it into several new columns (e.g. first part1 = …, then "
        "total = part1 + …).",
        "Formuła jest zbyt zagnieżdżona – podziel ją na kilka nowych kolumn (np. najpierw czesc1 = …, potem "
        "suma = czesc1 + …)."),
    "Zahl in der Formel ist zu groß.": ("A number in the formula is too large.", "Liczba w formule jest za duża."),
    "Unbekannte Spalte '{name}' – Namen mit Leer-/Sonderzeichen in `Backticks`.": (
        "Unknown column '{name}' – put names with spaces or special characters in `backticks`.",
        "Nieznana kolumna '{name}' – nazwy ze spacjami lub znakami specjalnymi zapisz w `odwrotnych apostrofach`."),
    "Text in Anführungszeichen ist nur in Vergleichen erlaubt, z. B. home_ownership == 'RENT'.": (
        "Text in quotes is only allowed in comparisons, e.g. home_ownership == 'RENT'.",
        "Tekst w cudzysłowie jest dozwolony tylko w porównaniach, np. home_ownership == 'RENT'."),
    "Potenz: Der Exponent muss eine Zahl zwischen -100 und 100 sein (oder eine Spalte enthalten).": (
        "Power: the exponent must be a number between -100 and 100 (or contain a column).",
        "Potęga: wykładnik musi być liczbą od -100 do 100 (albo zawierać kolumnę)."),
    "Vergleich nicht erlaubt – nur == != < <= > >=.": ("Comparison not allowed – only == != < <= > >=.",
                                                       "Niedozwolone porównanie – tylko == != < <= > >=."),
    "round(x, n): n muss eine ganze Zahl von 0 bis 10 sein.": ("round(x, n): n must be a whole number from 0 to 10.",
                                                              "round(x, n): n musi być liczbą całkowitą od 0 do 10."),
    "In Formeln nicht erlaubt: {what}. Erlaubt: + - * / ** // %, Vergleiche, and/or/not, {formula_funcs}.": (
        "Not allowed in formulas: {what}. Allowed: + - * / ** // %, comparisons, and/or/not, {formula_funcs}.",
        "Niedozwolone w formułach: {what}. Dozwolone: + - * / ** // %, porównania, and/or/not, {formula_funcs}."),
    "Die Formel muss mindestens eine Spalte enthalten.": ("The formula must contain at least one column.",
                                                          "Formuła musi zawierać co najmniej jedną kolumnę."),
    "Formel lässt sich nicht berechnen: {exc}": ("The formula cannot be calculated: {exc}",
                                                 "Nie można obliczyć formuły: {exc}"),
    "Die Formel ergibt keinen Wert je Zeile.": ("The formula does not give a value per row.",
                                                "Formuła nie daje wartości dla każdego wiersza."),
    "Neue Spalte '{name}'": ("New column '{name}'", "Nowa kolumna '{name}'"),
    "Name fehlt.": ("Name missing.", "Brak nazwy."),
    "unbekannte Art '{kind}'.": ("unknown type '{kind}'.", "nieznany rodzaj '{kind}'."),
    "Quellspalte fehlt.": ("Source column missing.", "Brak kolumny źródłowej."),
    "Spalte '{src}' gibt es nicht.": ("Column '{src}' does not exist.", "Kolumna '{src}' nie istnieje."),
    "{label}: {clash} gibt es schon – bitte anderen Namen wählen.": (
        "{label}: {clash} already exists – please choose another name.",
        "{label}: {clash} już istnieje – wybierz inną nazwę."),
    "Kein Wert von '{src}' passt zur Zuordnung – Schreibweise prüfen (Groß-/Kleinschreibung). Vorhandene Werte "
    "z. B.: {seen}": (
        "No value of '{src}' matches the mapping – check the spelling (upper/lower case). Existing values "
        "e.g.: {seen}",
        "Żadna wartość z '{src}' nie pasuje do przypisania – sprawdź pisownię (wielkość liter). Istniejące "
        "wartości np.: {seen}"),
    " ({unmapped} Werte ohne Zuordnung → leer)": (" ({unmapped} values without a mapping → empty)",
                                                   " (wartości bez przypisania: {unmapped} → puste)"),
    "aus {src} zugeordnet": ("mapped from {src}", "przypisano z {src}"),
    "Eine Bedingung muss wahr/falsch ergeben, z. B. dti > 30.": ("A condition must give true/false, e.g. dti > 30.",
                                                                "Warunek musi dawać prawdę/fałsz, np. dti > 30."),
    "Bedingung {definition} → 1 bei {n} Zeilen": ("condition {definition} → 1 in {n} rows",
                                                  "warunek {definition} → 1 w wierszach: {n}"),
    "Die Formel ergibt keine Zahl – nur mit Zahlenspalten rechnen.": (
        "The formula does not give a number – only calculate with number columns.",
        "Formuła nie daje liczby – licz tylko na kolumnach liczbowych."),
    " ({n_inf} × Division durch 0 → leer)": (" ({n_inf} × division by 0 → empty)", " ({n_inf} × dzielenie przez 0 → puste)"),
    "'entfernen' ist hier nicht sinnvoll – Aktion wählen oder 'behalten' für eine Kopie.": (
        "'Drop' makes no sense here – choose an action, or 'Keep' for a copy.",
        "'Usuń' nie ma tu sensu – wybierz akcję albo 'Zachowaj' dla kopii."),
    "Kopie von {src}": ("Copy of {src}", "Kopia kolumny {src}"),
    "{names} (neu): {msg}": ("{names} (new): {msg}", "{names} (nowa): {msg}"),
    # Schritt 4
    "Unbekannte Features: {unknown}": ("Unknown features: {unknown}", "Nieznane cechy: {unknown}"),
    "Bitte mindestens ein Feature auswählen.": ("Please select at least one feature.", "Wybierz co najmniej jedną cechę."),
    "{c}: {n_inf} unendliche Werte (inf) als fehlend behandelt": (
        "{c}: {n_inf} infinite values (inf) treated as missing",
        "{c}: wartości nieskończone (inf) potraktowane jako braki: {n_inf}"),
    "Spalten ohne Werte entfernt: {empty}": ("Columns without values removed: {empty}",
                                             "Usunięto kolumny bez wartości: {empty}"),
    "Keine Features mit Werten übrig.": ("No features with values left.", "Nie zostały żadne cechy z wartościami."),
    "Spalten wegen fehlender Werte entfernt: {drop_cols}": ("Columns removed because of missing values: {drop_cols}",
                                                           "Usunięto kolumny z powodu braków: {drop_cols}"),
    "Nach dem Entfernen der Spalten bleiben keine Features übrig.": (
        "No features are left after removing the columns.", "Po usunięciu kolumn nie zostały żadne cechy."),
    "{v} Zeilen mit fehlenden Werten in {dropna_cols} entfernt": (
        "{v} rows with missing values in {dropna_cols} removed",
        "Usunięto wiersze z brakami w {dropna_cols}: {v}"),
    "'{c}' ist eine Kategorie – '{strat}' geht nur bei Zahlen. Bitte '{m_mode}' oder '{m_const}' wählen.": (
        "'{c}' is a category – '{strat}' only works for numbers. Please choose '{m_mode}' or '{m_const}'.",
        "'{c}' to kategoria – '{strat}' działa tylko dla liczb. Wybierz '{m_mode}' albo '{m_const}'."),
    "Gruppen-Mittelwert für '{c}': Parameter muss eine andere Feature-Spalte sein (z. B. total_acc) – angegeben: "
    "'{g}'.": (
        "Group mean for '{c}': the parameter must be another feature column (e.g. total_acc) – given: '{g}'.",
        "Średnia grupowa dla '{c}': parametr musi być inną kolumną cech (np. total_acc) – podano: '{g}'."),
    "Unbekannte Strategie '{strat}' für '{c}'.": ("Unknown strategy '{strat}' for '{c}'.",
                                                  "Nieznana strategia '{strat}' dla '{c}'."),
    "Mittelwert je {v}": ("mean per {v}", "średnia według {v}"),
    "{c}: {n_na} fehlende Werte → {strat} ({shown})": ("{c}: {n_na} missing values → {strat} ({shown})",
                                                       "{c}: braki: {n_na} → {strat} ({shown})"),
    "'{c}' hat {n} Kategorien → sehr viele Dummy-Spalten. Tipp: Schritt 2 'Top-N behalten' oder entfernen.": (
        "'{c}' has {n} categories → very many dummy columns. Tip: step 2 'Keep top N' or drop it.",
        "'{c}' ma bardzo wiele kategorii ({n}) → bardzo dużo kolumn dummy. Wskazówka: w kroku 2 'Zachowaj top N' albo usuń."),
    "Doppelte Spaltennamen nach Dummy-Kodierung: {dupes} – bitte in Schritt 2 umbenennen.": (
        "Duplicate column names after dummy encoding: {dupes} – please rename them in step 2.",
        "Zduplikowane nazwy kolumn po kodowaniu dummy: {dupes} – zmień nazwy w kroku 2."),
    # Schritt 9 + 10
    "ZIP enthält keine preprocessing.json.": ("The ZIP contains no preprocessing.json.",
                                              "ZIP nie zawiera preprocessing.json."),
    "ZIP zu groß: entpackt {v:.1f} GB (höchstens {v2:.0f} GB).": ("ZIP too large: {v:.1f} GB unpacked (at most {v2:.0f} GB).",
                                                                 "ZIP za duży: po rozpakowaniu {v:.1f} GB (maksymalnie "
                                                                 "{v2:.0f} GB)."),
    "Zahl": ("Number", "Liczba"),
    "Mittelwert je {group_col}": ("mean per {group_col}", "średnia według {group_col}"),
    "CSV-Spalte": ("CSV column", "Kolumna CSV"),
    "Feature im Modell": ("Feature in the model", "Cecha w modelu"),
    "Typ": ("Type", "Typ"),
    "Füllwert bei Lücke": ("Fill value for gaps", "Wartość przy braku"),
    "Werte im Training": ("Values in training", "Wartości w treningu"),
    "Die Datei enthält keine Datenzeilen (nur Kopfzeile?).": ("The file contains no data rows (header only?).",
                                                              "Plik nie zawiera wierszy danych (tylko nagłówek?)."),
    "Lücken ergänzt": ("gaps filled", "uzupełniono braki"),
    "unbekannte Kategorie": ("unknown category", "nieznana kategoria"),
    "außerhalb Trainingsbereich": ("outside training range", "poza zakresem treningowym"),
    # Fazit (Schritt 10)
    "gut": ("good", "dobra"),
    "kritisch": ("critical", "krytyczna"),
    "mit Einschränkungen": ("with limitations", "z zastrzeżeniami"),
    "Mittelwert je {group_col} aus den Trainingsdaten": ("mean per {group_col} from the training data",
                                                         "średnia według {group_col} z danych treningowych"),
    "fester Wert {v}": ("fixed value {v}", "stała wartość {v}"),
    "{what} der Trainingsdaten ({v})": ("{what} of the training data ({v})", "{what} z danych treningowych ({v})"),
    "Mittelwert der Trainingsdaten ({v})": ("mean of the training data ({v})", "średnia z danych treningowych ({v})"),
    "Median der Trainingsdaten ({v})": ("median of the training data ({v})", "mediana z danych treningowych ({v})"),
    "häufigster Wert der Trainingsdaten ({v})": ("most frequent value of the training data ({v})",
                                                 "najczęstsza wartość z danych treningowych ({v})"),
    "Median der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken": (
        "median of the training data ({fb}) – there were no gaps here in training",
        "mediana z danych treningowych ({fb}) – w treningu nie było tu braków"),
    "häufigster Wert der Trainingsdaten ({fb}) – im Training gab es hier keine Lücken": (
        "most frequent value of the training data ({fb}) – there were no gaps here in training",
        "najczęstsza wartość z danych treningowych ({fb}) – w treningu nie było tu braków"),
    "{n} Zeilen; alle {v} benötigten Spalten vorhanden": ("{n} rows; all {v} required columns present",
                                                         "Wiersze: {n}; wszystkie wymagane kolumny ({v}) są obecne"),
    ", {n} weitere Spalten werden nicht gebraucht (ignoriert).": (", {n} more columns are not needed (ignored).",
                                                                 ", dodatkowe kolumny ({n}) nie są potrzebne "
                                                                 "(pominięte)."),
    "Keine Lücken in den benötigten Spalten.": ("No gaps in the required columns.",
                                                "Wymagane kolumny są kompletne (bez braków)."),
    "{n} Zeilen ({pct}) mit Kategorien, die es im Training nicht gab – dort zählen alle Dummies dieser Spalte als "
    "0 (das Modell kennt die Kategorie nicht).": (
        "{n} rows ({pct}) with categories that did not exist in training – all dummies of this column count as 0 "
        "there (the model does not know the category).",
        "Wiersze z kategoriami, których nie było w treningu: {n} ({pct}) – wszystkie dummy tej kolumny liczą się "
        "tam jako 0 (model nie zna tej kategorii)."),
    "{c}: {n} Zeile": ("{c}: {n} row", "{c}: {n} wiersz"),
    "{c}: {n} Zeilen": ("{c}: {n} rows", "{c}: wiersze: {n}"),
    "{n} Werte in Zahlenspalten waren keine Zahl und wurden wie Lücken behandelt.": (
        "{n} values in number columns were not numbers and were treated like gaps.",
        "Wartości w kolumnach liczbowych, które nie były liczbami i potraktowano je jak braki: {n}."),
    "Datenqualität verbessern: Lücken, unbekannte Kategorien und Ausreißer in der Quelle prüfen – betroffene "
    "Zeilen stehen in der Spalte 'Hinweis'.": (
        "Improve data quality: check gaps, unknown categories and outliers in the source – affected rows are "
        "marked in the column 'Note'.",
        "Popraw jakość danych: sprawdź braki, nieznane kategorie i wartości odstające w źródle – wiersze, których "
        "to dotyczy, są oznaczone w kolumnie 'Uwaga'."),
    # Schritt 8 – Modell-Details (ML-Version)
    "Abstand": (
        "Distance",
        "Odległość"),
    # Schritt 9 – Speichern & Laden (ML-Version)
    "Das Modell ist eine .joblib-Datei (pickle) – sie kann beim Laden Programmcode ausführen. Nur laden, "
    "wenn die Datei aus einer vertrauenswürdigen Quelle stammt (Häkchen setzen), oder das Modell als .skops "
    "speichern.": (
        "The model is a .joblib file (pickle) – it can run program code when loaded. Only load it if the file "
        "comes from a trusted source (tick the box), or save the model as .skops.",
        "Model jest plikiem .joblib (pickle) – przy wczytywaniu może wykonać kod programu. Wczytuj go tylko, "
        "jeśli plik pochodzi z zaufanego źródła (zaznacz pole), albo zapisz model jako .skops."),
    "Das Modell enthält Typen, die nicht als sicher gelten – Laden abgelehnt: {types}": (
        "The model contains types that are not considered safe – loading refused: {types}",
        "Model zawiera typy, które nie są uznane za bezpieczne – odmowa wczytania: {types}"),
    # --- Clustering-App (K-Means) ---
    "Werte begrenzen (min, max)": ("Clip values (min, max)", "Ogranicz wartości (min, max)"),
    "min,max – z. B. 0,100 oder ,100": ("min,max – e.g. 0,100 or ,100", "min,max – np. 0,100 lub ,100"),
    "Zahlen auf einen Bereich begrenzen (clip): ',100' macht aus Grad.Rate 118 den Wert 100 – unmögliche Werte "
    "und Ausreißer verschieben die Cluster-Zentren": (
        "limit numbers to a range (clip): ',100' turns Grad.Rate 118 into 100 – impossible values and outliers "
        "shift the cluster centres",
        "ogranicz liczby do zakresu (clip): ',100' zmienia Grad.Rate 118 na 100 – niemożliwe wartości i wartości "
        "odstające przesuwają centra klastrów"),
    "Werte begrenzen: Format min,max – z. B. 0,100 oder ,100 (nur Obergrenze).": (
        "Clip values: format min,max – e.g. 0,100 or ,100 (upper limit only).",
        "Ogranicz wartości: format min,max – np. 0,100 lub ,100 (tylko górna granica)."),
    "Werte begrenzen: mindestens eine Grenze angeben – z. B. ,100.": (
        "Clip values: give at least one limit – e.g. ,100.",
        "Ogranicz wartości: podaj co najmniej jedną granicę – np. ,100."),
    "Werte begrenzen: Grenzen müssen endliche Zahlen sein.": (
        "Clip values: limits must be finite numbers.",
        "Ogranicz wartości: granice muszą być skończonymi liczbami."),
    "Werte begrenzen: min ({lo}) ist größer als max ({hi}).": (
        "Clip values: min ({lo}) is greater than max ({hi}).",
        "Ogranicz wartości: min ({lo}) jest większe niż max ({hi})."),
    "auf {span} begrenzt ({n} Werte geändert)": (
        "clipped to {span} ({n} values changed)",
        "ograniczono do {span} (zmieniono {n} wartości)"),
    "{cols}: als Vergleichsgruppe bzw. Bezeichnung gewählt – nicht als Feature verwendet.": (
        "{cols}: chosen as comparison group or row name – not used as a feature.",
        "{cols}: wybrano jako grupę porównawczą lub nazwę wiersza – nie jest używana jako cecha."),
    "Spalte '{c}' ist nicht in den Daten.": ("Column '{c}' is not in the data.", "Kolumny '{c}' nie ma w danych."),
    "Nur noch {n} Zeilen übrig – zu wenig für ein Clustering (mindestens {m}).": (
        "Only {n} rows left – too few for clustering (at least {m}).",
        "Zostało tylko {n} wierszy – za mało do klastrowania (co najmniej {m})."),
    "Ohne Streuung (überall derselbe Wert): {const} – tragen nichts zum Clustering bei.": (
        "No variation (the same value everywhere): {const} – they contribute nothing to the clustering.",
        "Bez zmienności (wszędzie ta sama wartość): {const} – nic nie wnoszą do klastrowania."),
    "Stark korreliert (|r| ≥ 0.9): {pairs} – solche Paare zählen im Abstand fast doppelt; wer das nicht möchte, "
    "lässt eine Spalte des Paares weg.": (
        "Strongly correlated (|r| ≥ 0.9): {pairs} – such pairs count almost twice in the distance; if you do not "
        "want that, leave out one column of each pair.",
        "Silnie skorelowane (|r| ≥ 0.9): {pairs} – takie pary liczą się w odległości prawie podwójnie; kto tego "
        "nie chce, pomija jedną kolumnę z pary."),
    "Clustering-Daten: {n} Zeilen × {p} Spalten – Füllwerte, Kategorien und {scaler} aus allen Zeilen gelernt "
    "(kein Train/Test-Split: K-Means bekommt keine Antwort)": (
        "Clustering data: {n} rows × {p} columns – fill values, categories and {scaler} learned from all rows "
        "(no train/test split: K-Means is never given an answer)",
        "Dane do klastrowania: {n} wierszy × {p} kolumn – wartości uzupełniające, kategorie i {scaler} wyuczone "
        "ze wszystkich wierszy (bez podziału train/test: K-Means nie dostaje odpowiedzi)"),
    "Vergleichsgruppe '{c}' hat {k} verschiedene Werte – für den Vergleich sind 2–{m} nötig; sie wird nicht "
    "verwendet.": (
        "Comparison group '{c}' has {k} different values – the comparison needs 2–{m}; it is not used.",
        "Grupa porównawcza '{c}' ma {k} różnych wartości – do porównania potrzeba 2–{m}; nie zostanie użyta."),
    "{n} Zeilen ohne Wert in '{c}' – sie werden geclustert, fehlen aber im Vergleich.": (
        "{n} rows without a value in '{c}' – they are clustered but left out of the comparison.",
        "{n} wierszy bez wartości w '{c}' – są klastrowane, ale nie biorą udziału w porównaniu."),
    "Feature(s) {same} stammen aus derselben CSV-Spalte wie die Vergleichsgruppe '{c}' – K-Means kennt die "
    "Antwort dann schon, der Vergleich in Schritt 8 wäre geschönt.": (
        "Feature(s) {same} come from the same CSV column as the comparison group '{c}' – K-Means would then "
        "already know the answer and the comparison in step 8 would look better than it is.",
        "Cecha(-y) {same} pochodzi z tej samej kolumny CSV co grupa porównawcza '{c}' – K-Means znałby wtedy już "
        "odpowiedź, a porównanie w kroku 8 wypadłoby lepiej niż w rzeczywistości."),
    "K muss zwischen 2 und {m} liegen.": ("K must be between 2 and {m}.", "K musi mieścić się między 2 a {m}."),
    "K = {k} ist zu groß für {n} Zeilen.": (
        "K = {k} is too large for {n} rows.",
        "K = {k} jest za duże dla {n} wierszy."),
    "Trainiere K-Means (K = {k}, n_init = {n}) …": (
        "Training K-Means (K = {k}, n_init = {n}) …",
        "Trenuję K-Means (K = {k}, n_init = {n}) …"),
    "K-Means fertig – {n} Iterationen": ("K-Means done – {n} iterations", "K-Means gotowy – {n} iteracji"),
    "Elbow- und Silhouette-Kurve: K = {k} ({i} von {n})": (
        "Elbow and silhouette curve: K = {k} ({i} of {n})",
        "Krzywa łokcia i Silhouette: K = {k} ({i} z {n})"),
    "nicht berechenbar": ("not computable", "nie do obliczenia"),
    "starke Struktur – die Cluster sind klar getrennt": (
        "strong structure – the clusters are clearly separated",
        "silna struktura – klastry są wyraźnie rozdzielone"),
    "vernünftige Struktur": ("reasonable structure", "rozsądna struktura"),
    "schwache Struktur – die Cluster überlappen; Ergebnis mit Vorsicht deuten": (
        "weak structure – the clusters overlap; interpret the result with care",
        "słaba struktura – klastry nachodzą na siebie; wynik interpretować ostrożnie"),
    "kaum Struktur – die Cluster überlappen stark (bei vielen Features ist die Silhouette oft niedrig; ob die "
    "Cluster trotzdem sinnvoll sind, zeigen die Profile in Schritt 8)": (
        "hardly any structure – the clusters overlap strongly (with many features the silhouette is often low; "
        "the profiles in step 8 show whether the clusters still make sense)",
        "prawie brak struktury – klastry mocno na siebie nachodzą (przy wielu cechach Silhouette jest często "
        "niska; czy klastry mimo to mają sens, pokazują profile w kroku 8)"),
    "**Silhouette für K = {k}: {s}** – {verdict}.": (
        "**Silhouette for K = {k}: {s}** – {verdict}.",
        "**Silhouette dla K = {k}: {s}** – {verdict}."),
    "**Elbow-Methode:** kein deutlicher Knick – die Inertia sinkt gleichmäßig.": (
        "**Elbow method:** no clear bend – the inertia falls evenly.",
        "**Metoda łokcia:** brak wyraźnego załamania – inercja maleje równomiernie."),
    "**Silhouette** am höchsten bei **K = {b}** ({s:.3f}).": (
        "**Silhouette** highest at **K = {b}** ({s:.3f}).",
        "**Silhouette** najwyższa przy **K = {b}** ({s:.3f})."),
    "Beide Methoden zeigen **K = {k2}**.": (
        "Both methods point to **K = {k2}**.",
        "Obie metody wskazują **K = {k2}**."),
    " Eingestellt ist K = {k}.": (" Currently set: K = {k}.", " Ustawione jest K = {k}."),
    "K = {v} übernehmen": ("Use K = {v}", "Przejmij K = {v}"),
    "K = {v} übernehmen ({what})": ("Use K = {v} ({what})", "Przejmij K = {v} ({what})"),
    "Elbow": ("elbow", "łokieć"),
    "Silhouette": ("Silhouette", "Silhouette"),
    "Elbow und Silhouette zeigen verschiedene K – beide ausprobieren: Am Ende zählt, ob sich die Cluster in "
    "Schritt 8 sinnvoll beschreiben lassen.": (
        "Elbow and silhouette point to different K – try both: what counts in the end is whether the clusters "
        "can be described sensibly in step 8.",
        "Łokieć i Silhouette wskazują różne K – wypróbuj oba: na końcu liczy się, czy klastry da się sensownie "
        "opisać w kroku 8."),
    "Kurve auf einer Stichprobe von {n} Zeilen – das Modell selbst nutzt alle Zeilen.": (
        "Curve computed on a sample of {n} rows – the model itself uses all rows.",
        "Krzywa policzona na próbce {n} wierszy – sam model używa wszystkich wierszy."),
    ":orange[**Sehr kleiner Cluster:**] Cluster {j} hat nur {v} Zeilen ({pct}) – oft Ausreißer. Tipp: Ausreißer "
    "in Schritt 2 begrenzen (Werte begrenzen) oder K verkleinern.": (
        ":orange[**Very small cluster:**] cluster {j} has only {v} rows ({pct}) – often outliers. Tip: clip "
        "outliers in step 2 (clip values) or reduce K.",
        ":orange[**Bardzo mały klaster:**] klaster {j} ma tylko {v} wierszy ({pct}) – często wartości odstające. "
        "Wskazówka: ogranicz wartości odstające w kroku 2 (ogranicz wartości) albo zmniejsz K."),
    ":orange[**Ein Cluster dominiert:**] Cluster {j} enthält {pct} der Zeilen – die Daten trennen sich kaum, "
    "oder wenige Ausreißer bilden die übrigen Cluster.": (
        ":orange[**One cluster dominates:**] cluster {j} contains {pct} of the rows – the data hardly separate, "
        "or a few outliers form the other clusters.",
        ":orange[**Jeden klaster dominuje:**] klaster {j} zawiera {pct} wierszy – dane prawie się nie "
        "rozdzielają albo kilka wartości odstających tworzy pozostałe klastry."),
    ":orange[**Nicht konvergiert:**] K-Means hat max_iter erreicht – max_iter in Schritt 6 erhöhen.": (
        ":orange[**Not converged:**] K-Means reached max_iter – increase max_iter in step 6.",
        ":orange[**Brak zbieżności:**] K-Means osiągnął max_iter – zwiększ max_iter w kroku 6."),
    ":orange[**Hinweis von scikit-learn:**] {w}": (
        ":orange[**Note from scikit-learn:**] {w}",
        ":orange[**Uwaga od scikit-learn:**] {w}"),
    "Cluster {j}": ("Cluster {j}", "Klaster {j}"),
    "Kein scikit-learn-Modell gefunden (model.skops oder model.joblib).": (
        "No scikit-learn model found (model.skops or model.joblib).",
        "Nie znaleziono modelu scikit-learn (model.skops lub model.joblib)."),
    "weit weg von allen Clustern": ("far from all clusters", "daleko od wszystkich klastrów"),
    "zwischen zwei Clustern": ("between two clusters", "między dwoma klastrami"),
    "Lücken in {n} Zeilen ({pct}) mit Werten aus dem Training ergänzt. Ergänzte Werte sind Schätzungen – die "
    "Zuordnung dieser Zeilen ist etwas unsicherer.": (
        "Gaps in {n} rows ({pct}) filled with values from training. Filled values are estimates – the assignment "
        "of these rows is somewhat less certain.",
        "Braki w {n} wierszach ({pct}) uzupełniono wartościami z treningu. Uzupełnione wartości to szacunki – "
        "przypisanie tych wierszy jest nieco mniej pewne."),
    "{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – solche Zeilen liegen oft weit weg von allen "
    "Clustern.": (
        "{n} rows ({pct}) with values outside the training range – such rows are often far from all clusters.",
        "{n} wierszy ({pct}) z wartościami poza zakresem treningowym – takie wiersze często leżą daleko od "
        "wszystkich klastrów."),
    "Median der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für eine Zuordnung "
    "braucht aber jede Zeile einen Wert": (
        "median of the training data ({fb}) – in training, rows with a gap here were removed; but an assignment "
        "needs a value in every row",
        "mediana z danych treningowych ({fb}) – w treningu wiersze z brakiem tutaj usunięto; do przypisania "
        "każdy wiersz potrzebuje jednak wartości"),
    "häufigster Wert der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für eine "
    "Zuordnung braucht aber jede Zeile einen Wert": (
        "most frequent value of the training data ({fb}) – in training, rows with a gap here were removed; but "
        "an assignment needs a value in every row",
        "najczęstsza wartość z danych treningowych ({fb}) – w treningu wiersze z brakiem tutaj usunięto; do "
        "przypisania każdy wiersz potrzebuje jednak wartości"),
    "Verteilung auf die Cluster: ": ("Distribution across the clusters: ", "Rozkład na klastry: "),
    " (Training {pct})": (" (training {pct})", " (trening {pct})"),
    "Die Verteilung weicht deutlich vom Training ab (bis {pp} Prozentpunkte) – die neuen Daten unterscheiden "
    "sich vom Training (andere Zeit, andere Gruppe) oder sind eine Auswahl.": (
        "The distribution differs clearly from training (up to {pp} percentage points) – the new data differ "
        "from the training data (another period, another group) or are a selection.",
        "Rozkład wyraźnie różni się od treningu (do {pp} punktów procentowych) – nowe dane różnią się od "
        "treningowych (inny okres, inna grupa) albo są wyborem."),
    "**{n} Zeilen ({pct}) liegen weit weg von allen Clustern** – weiter vom nächsten Zentrum entfernt als 99 % "
    "der Trainingszeilen dieses Clusters (Hinweis 'weit weg von allen Clustern'). Das sind Ausreißer oder ein "
    "Typ, den es im Training nicht gab.": (
        "**{n} rows ({pct}) lie far from all clusters** – further from the nearest centre than 99 % of the "
        "training rows of that cluster (note 'far from all clusters'). These are outliers or a type that did not "
        "exist in training.",
        "**{n} wierszy ({pct}) leży daleko od wszystkich klastrów** – dalej od najbliższego centrum niż 99 % "
        "wierszy treningowych tego klastra (uwaga 'daleko od wszystkich klastrów'). To wartości odstające albo "
        "typ, którego nie było w treningu."),
    "Viele Zeilen passen zu keinem Cluster: mit aktuellen Daten neu clustern (Schritt 4–6) oder prüfen, ob die "
    "neuen Daten aus derselben Quelle stammen.": (
        "Many rows fit no cluster: cluster again with current data (steps 4–6) or check whether the new data "
        "come from the same source.",
        "Wiele wierszy nie pasuje do żadnego klastra: sklastruj ponownie na aktualnych danych (kroki 4–6) albo "
        "sprawdź, czy nowe dane pochodzą z tego samego źródła."),
    "Alle Zeilen liegen so nah an ihrem Cluster-Zentrum wie die Trainingszeilen – die Cluster beschreiben auch "
    "diese Daten.": (
        "All rows are as close to their cluster centre as the training rows – the clusters also describe these "
        "data.",
        "Wszystkie wiersze leżą tak blisko centrum swojego klastra jak wiersze treningowe – klastry opisują "
        "także te dane."),
    "{n} Zeilen ({pct}) liegen fast genau zwischen zwei Clustern (Hinweis 'zwischen zwei Clustern') – ihre "
    "Zuordnung ist knapp.": (
        "{n} rows ({pct}) lie almost exactly between two clusters (note 'between two clusters') – their "
        "assignment is a close call.",
        "{n} wierszy ({pct}) leży prawie dokładnie między dwoma klastrami (uwaga 'między dwoma klastrami') – ich "
        "przypisanie jest na granicy."),
    "Vergleich mit '{c}' ({n} Zeilen): Die Zuordnung Cluster → Gruppe aus dem Training trifft {pct}": (
        "Comparison with '{c}' ({n} rows): the cluster → group assignment from training is right for {pct}",
        "Porównanie z '{c}' ({n} wierszy): przypisanie klaster → grupa z treningu trafia w {pct}"),
    " (Training: {pct})": (" (training: {pct})", " (trening: {pct})"),
    "; naive Vorhersage (immer '{g}'): {pct}.": (
        "; naive prediction (always '{g}'): {pct}.",
        "; prognoza naiwna (zawsze '{g}'): {pct}."),
    "Die Cluster passen hier schlechter zur bekannten Einteilung als im Training – die neuen Daten unterscheiden "
    "sich vermutlich.": (
        "Here the clusters match the known grouping worse than in training – the new data probably differ.",
        "Tutaj klastry gorzej pasują do znanego podziału niż w treningu – nowe dane prawdopodobnie się różnią."),
    "**Elbow-Methode:** Knick bei **K = {e}** – danach sinkt die Inertia deutlich langsamer.": (
        "**Elbow method:** bend at **K = {e}** – after that the inertia falls much more slowly.",
        "**Metoda łokcia:** załamanie przy **K = {e}** – potem inercja maleje wyraźnie wolniej."),
    " Das ist der Rand des geprüften Bereichs – k_max in Schritt 6 erhöhen, um größere K zu sehen.": (
        " That is the edge of the tested range – increase k_max in step 6 to see larger K.",
        " To skraj sprawdzanego zakresu – zwiększ k_max w kroku 6, aby zobaczyć większe K."),
    "Die Silhouette bleibt für alle K niedrig – die Daten zerfallen kaum in getrennte Gruppen; die Silhouette "
    "liefert hier keinen Vorschlag.": (
        "The silhouette stays low for every K – the data hardly fall apart into separate groups; the silhouette "
        "gives no suggestion here.",
        "Silhouette pozostaje niska dla każdego K – dane prawie nie rozpadają się na odrębne grupy; Silhouette "
        "nie daje tu propozycji."),
    # --- Audit 5 ---
    ":orange[**Leerer Cluster:**] {clusters} – keine Zeile liegt diesem Zentrum am nächsten. Die Daten haben "
    "weniger verschiedene Zeilen als K (doppelte Zeilen) – K verkleinern.": (
        ":orange[**Empty cluster:**] {clusters} – no row is closest to this centre. The data have fewer distinct "
        "rows than K (duplicate rows) – reduce K.",
        ":orange[**Pusty klaster:**] {clusters} – żaden wiersz nie leży najbliżej tego centrum. Dane mają mniej "
        "różnych wierszy niż K (powtórzone wiersze) – zmniejsz K."),
    "preprocessing.json ist unvollständig oder beschädigt (Eintrag: {part}) – das Modell bitte neu speichern "
    "(Schritt 9).": (
        "preprocessing.json is incomplete or damaged (entry: {part}) – please save the model again (step 9).",
        "preprocessing.json jest niekompletny lub uszkodzony (wpis: {part}) – zapisz model ponownie (krok 9)."),
    "Modell und preprocessing.json passen nicht zusammen: das Modell hat {k} Cluster und erwartet {n} Eingänge, "
    "das Rezept liefert {m}.": (
        "Model and preprocessing.json do not match: the model has {k} clusters and expects {n} inputs, the recipe "
        "provides {m}.",
        "Model i preprocessing.json do siebie nie pasują: model ma {k} klastrów i oczekuje {n} wejść, receptura "
        "dostarcza {m}."),
    # --- Anomalieerkennung (Isolation Forest, LOF, DBSCAN) ---
    "Gruppen (Cluster) ähnlicher Zeilen finden – K gibt man vor, Schritt 7 hilft bei der Wahl. Ausreißer zeigt "
    "Schritt 10: Zeilen weit weg von allen Zentren.": (
        "Find groups (clusters) of similar rows – you set K, step 7 helps you choose it. Outliers show up in step "
        "10: rows far from all centres.",
        "Znajdowanie grup (klastrów) podobnych wierszy – K podajesz sam, krok 7 pomaga je dobrać. Wartości "
        "odstające pokazuje krok 10: wiersze daleko od wszystkich centrów."),
    "Anomalien finden: Zufällige Schnitte isolieren jede Zeile – ungewöhnliche Zeilen sind nach wenigen "
    "Schnitten allein. Schnell, auch bei vielen Zeilen und Features; der Standard für Tabellen.": (
        "Find anomalies: random splits isolate every row – unusual rows are alone after only a few splits. Fast, "
        "also with many rows and features; the standard choice for tables.",
        "Wykrywanie anomalii: losowe cięcia izolują każdy wiersz – nietypowe wiersze zostają same już po kilku "
        "cięciach. Szybki, także przy wielu wierszach i cechach; standard dla danych tabelarycznych."),
    "Anomalien finden: vergleicht die Dichte um eine Zeile mit der Dichte um ihre Nachbarn – findet auch lokale "
    "Ausreißer neben dichten Gruppen. Bei sehr vielen Zeilen langsamer.": (
        "Find anomalies: compares the density around a row with the density around its neighbours – also finds "
        "local outliers next to dense groups. Slower with very many rows.",
        "Wykrywanie anomalii: porównuje gęstość wokół wiersza z gęstością wokół jego sąsiadów – znajduje też "
        "lokalne wartości odstające obok gęstych grup. Wolniejszy przy bardzo wielu wierszach."),
    "Cluster beliebiger Form plus Rauschen: Zeilen in dünn besetzten Bereichen sind Rauschen = Anomalien. Kein K "
    "nötig, dafür eps (Radius) und min_samples – eps zeigt Schritt 7 (k-Distanz).": (
        "Clusters of any shape plus noise: rows in sparse regions are noise = anomalies. No K needed, but eps "
        "(radius) and min_samples – step 7 shows a good eps (k-distance).",
        "Klastry dowolnego kształtu plus szum: wiersze w rzadko obsadzonych obszarach to szum = anomalie. Nie "
        "trzeba K, za to eps (promień) i min_samples – dobre eps pokazuje krok 7 (k-odległość)."),
    "**{k} von {n} Zeilen auffällig ({pct})**": (
        "**{k} of {n} rows anomalous ({pct})**",
        "**Nietypowe: {k} z {n} wierszy ({pct})**"),
    "Unbekanntes Modell: {model}": ("Unknown model: {model}", "Nieznany model: {model}"),
    " – im Training {pct}": (" – in training {pct}", " – w treningu {pct}"),
    "contamination muss zwischen 0 und 0.5 liegen (Anteil auffälliger Zeilen, z. B. 0.01 = 1 %).": (
        "contamination must be between 0 and 0.5 (share of anomalous rows, e.g. 0.01 = 1 %).",
        "contamination musi być między 0 a 0.5 (udział nietypowych wierszy, np. 0.01 = 1 %)."),
    "eps muss größer als 0 sein.": ("eps must be greater than 0.", "eps musi być większe od 0."),
    "Trainiere {name} …": ("Training {name} …", "Trenowanie: {name} …"),
    "{name} fertig – {n} auffällige Zeilen": (
        "{name} done – {n} anomalous rows",
        "{name} gotowy – nietypowe wiersze: {n}"),
    "**{k} von {n} Zeilen auffällig ({pct})** – Schwelle: Score über {thr}.": (
        "**{k} of {n} rows anomalous ({pct})** – threshold: score above {thr}.",
        "**Nietypowe: {k} z {n} wierszy ({pct})** – próg: wskaźnik powyżej {thr}."),
    "**{c} Cluster, {k} Zeilen Rauschen ({pct})** bei eps = {eps} und min_samples = {m}.": (
        "**{c} clusters, {k} rows of noise ({pct})** at eps = {eps} and min_samples = {m}.",
        "**Klastry: {c}, szum: {k} wierszy ({pct})** przy eps = {eps} i min_samples = {m}."),
    "Kernpunkte: {c} von {n} Zeilen ({pct}).": (
        "Core points: {c} of {n} rows ({pct}).",
        "Punkty rdzeniowe: {c} z {n} wierszy ({pct})."),
    "Dieses Modell stammt nicht aus der Clustering-Version (Aufgabe: {task}) – hier lassen sich nur Modelle "
    "laden, die mit dieser App gespeichert wurden (K-Means, Isolation Forest, LOF, DBSCAN).": (
        "This model does not come from the clustering version (task: {task}) – only models saved with this app can "
        "be loaded here (K-Means, Isolation Forest, LOF, DBSCAN).",
        "Ten model nie pochodzi z wersji do klastrowania (zadanie: {task}) – tutaj można wczytać tylko modele "
        "zapisane w tej aplikacji (K-Means, Isolation Forest, LOF, DBSCAN)."),
    "Die Modelldatei enthält kein {name}-Modell ({type}).": (
        "The model file does not contain a {name} model ({type}).",
        "Plik modelu nie zawiera modelu {name} ({type})."),
    "Modell und preprocessing.json passen nicht zusammen: das {name}-Modell erwartet {n} Eingänge, das Rezept "
    "liefert {m}.": (
        "Model and preprocessing.json do not match: the {name} model expects {n} inputs, the recipe provides {m}.",
        "Model i preprocessing.json do siebie nie pasują: model {name} oczekuje {n} wejść, receptura dostarcza {m}."),
    "Das LOF-Modell wurde ohne novelty=True trainiert – damit lassen sich keine neuen Daten bewerten.": (
        "The LOF model was trained without novelty=True – it cannot score new data.",
        "Model LOF wytrenowano bez novelty=True – nie da się nim ocenić nowych danych."),
    "Anomalie-Modell: neue Daten mit score_new() bewerten (assign_new ist für K-Means).": (
        "Anomaly model: score new data with score_new() (assign_new is for K-Means).",
        "Model anomalii: nowe dane oceniaj przez score_new() (assign_new jest dla K-Means)."),
    "K-Means-Modell: neue Daten mit assign_new() zuordnen (score_new ist für die Anomalie-Modelle).": (
        "K-Means model: assign new data with assign_new() (score_new is for the anomaly models).",
        "Model K-Means: nowe dane przypisuj przez assign_new() (score_new jest dla modeli anomalii)."),
    "; Schwelle aus dem Training: Score über {thr}.": (
        "; threshold from training: score above {thr}.",
        "; próg z treningu: wskaźnik powyżej {thr}."),
    "Keine Zeile liegt über der Schwelle – die neuen Daten sehen aus wie die normalen Trainingsdaten.": (
        "No row is above the threshold – the new data look like the normal training data.",
        "Żaden wiersz nie przekracza progu – nowe dane wyglądają jak normalne dane treningowe."),
    "Deutlich mehr auffällige Zeilen als im Training – die neuen Daten unterscheiden sich (anderer Zeitraum, "
    "andere Marktphase, andere Quelle) oder es gab echte Ereignisse.": (
        "Clearly more anomalous rows than in training – the new data differ (another period, another market phase, "
        "another source) or there were real events.",
        "Wyraźnie więcej nietypowych wierszy niż w treningu – nowe dane się różnią (inny okres, inna faza rynku, "
        "inne źródło) albo wydarzyło się coś rzeczywistego."),
    "Bei Zeitreihen: Häufen sich die Anomalien in einem Zeitraum, ist das eher ein neues Regime als lauter "
    "Einzelfälle – dann mit neueren Daten neu trainieren (Schritt 4–6).": (
        "For time series: if the anomalies pile up in one period, that is more likely a new regime than many single "
        "cases – then retrain with more recent data (steps 4–6).",
        "Dla szeregów czasowych: jeśli anomalie skupiają się w jednym okresie, to raczej nowy reżim niż wiele "
        "pojedynczych przypadków – wtedy wytrenuj ponownie na nowszych danych (kroki 4–6)."),
    "Cluster {j}: {pct}": ("Cluster {j}: {pct}", "Klaster {j}: {pct}"),
    "Rauschen: {pct}": ("noise: {pct}", "szum: {pct}"),
    "contamination = {pct}: Die Schwelle ist das passende Quantil der Trainings-Scores – etwa dieser Anteil der "
    "Zeilen gilt als auffällig.": (
        "contamination = {pct}: the threshold is the matching quantile of the training scores – about this share of "
        "rows counts as anomalous.",
        "contamination = {pct}: próg to odpowiedni kwantyl wskaźników treningowych – mniej więcej taki udział "
        "wierszy uznaje się za nietypowe."),
    ":orange[**Keine auffällige Zeile:**] Kein Score liegt über der Schwelle – mit contamination einen festen "
    "Anteil wählen.": (
        ":orange[**No anomalous row:**] no score is above the threshold – choose a fixed share with contamination.",
        ":orange[**Brak nietypowych wierszy:**] żaden wskaźnik nie przekracza progu – wybierz stały udział przez "
        "contamination."),
    "**Knick der Score-Kurve:** Nach den auffälligsten {n} Zeilen ({pct}) flacht die Kurve ab – diese Zeilen "
    "heben sich deutlich vom Rest ab. Ein Kandidat für contamination.": (
        "**Knee of the score curve:** after the {n} most anomalous rows ({pct}) the curve flattens – these rows "
        "clearly stand out from the rest. A candidate for contamination.",
        "**Załamanie krzywej wskaźnika:** po {n} najbardziej nietypowych wierszach ({pct}) krzywa się wypłaszcza – "
        "te wiersze wyraźnie odstają od reszty. Kandydat na contamination."),
    ":red[**Kein Kernpunkt:**] eps ist zu klein – alle Zeilen sind Rauschen. eps vergrößern (Knick der "
    "k-Distanz).": (
        ":red[**No core point:**] eps is too small – all rows are noise. Increase eps (knee of the k-distance).",
        ":red[**Brak punktów rdzeniowych:**] eps jest za małe – wszystkie wiersze to szum. Zwiększ eps (załamanie "
        "k-odległości)."),
    "**Knick der k-Distanz bei eps ≈ {eps}:** Rechts davon liegen die Zeilen deutlich weiter von ihren Nachbarn "
    "entfernt – Kandidaten für Rauschen.": (
        "**Knee of the k-distance at eps ≈ {eps}:** to the right of it, rows lie clearly further from their "
        "neighbours – candidates for noise.",
        "**Załamanie k-odległości przy eps ≈ {eps}:** na prawo od niego wiersze leżą wyraźnie dalej od sąsiadów – "
        "kandydaci na szum."),
    "k-Distanz für {n} zufällige Zeilen gemessen (Nachbarn aus allen Zeilen).": (
        "k-distance measured for {n} random rows (neighbours from all rows).",
        "k-odległość zmierzona dla {n} losowych wierszy (sąsiedzi ze wszystkich wierszy)."),
    "Rauschanteil je eps nur bis {m} Zeilen – DBSCAN müsste für jedes eps neu laufen. Tipp: Stichprobe in der "
    "Seitenleiste.": (
        "Noise share per eps only up to {m} rows – DBSCAN would have to run again for every eps. Tip: sample in the "
        "sidebar.",
        "Udział szumu dla eps tylko do {m} wierszy – DBSCAN musiałby działać od nowa dla każdego eps. Wskazówka: "
        "próbka w pasku bocznym."),
    "Zeile {i}": ("row {i}", "wiersz {i}"),
    "Vergleich mit '{g}' ({n} Zeilen): Von den auffälligen Zeilen gehören {prec} zu '{pos}' (Anteil insgesamt "
    "{base}); erkannt werden {rec} aller '{pos}'-Zeilen. ROC-AUC des Scores: {auc}.": (
        "Comparison with '{g}' ({n} rows): of the anomalous rows, {prec} belong to '{pos}' (overall share {base}); "
        "{rec} of all '{pos}' rows are detected. ROC-AUC of the score: {auc}.",
        "Porównanie z '{g}' ({n} wierszy): spośród nietypowych wierszy {prec} należy do '{pos}' (udział ogółem "
        "{base}); wykryto {rec} wszystkich wierszy '{pos}'. ROC-AUC wskaźnika: {auc}."),
    "Rauschanteil je eps: eps = {e} ({i} von {n})": (
        "Noise share per eps: eps = {e} ({i} of {n})",
        "Udział szumu dla eps: eps = {e} ({i} z {n})"),
    ":orange[**Viele auffällige Zeilen ({pct}):**] Für eine Liste der stärksten Ausreißer mit contamination "
    "einen festen, kleineren Anteil wählen (z. B. 1 %).": (
        ":orange[**Many anomalous rows ({pct}):**] for a list of the strongest outliers choose a fixed, smaller "
        "share with contamination (e.g. 1 %).",
        ":orange[**Dużo nietypowych wierszy ({pct}):**] aby dostać listę najsilniejszych odstających, wybierz "
        "przez contamination stały, mniejszy udział (np. 1 %)."),
    "**Score-Kurve ohne deutlichen Knick:** Die Scores sinken gleichmäßig – die Schwelle ist dann eine Frage des "
    "gewünschten Anteils (Tabelle unten).": (
        "**Score curve without a clear knee:** the scores fall evenly – the threshold is then a question of the "
        "share you want (table below).",
        "**Krzywa wskaźnika bez wyraźnego załamania:** wskaźniki spadają równomiernie – próg jest wtedy kwestią "
        "pożądanego udziału (tabela poniżej)."),
    ":orange[**Mehr als die Hälfte ist Rauschen:**] eps ist zu klein oder min_samples zu groß.": (
        ":orange[**More than half is noise:**] eps is too small or min_samples too large.",
        ":orange[**Ponad połowa to szum:**] eps jest za małe albo min_samples za duże."),
    "**k-Distanz ohne deutlichen Knick:** eps nach dem gewünschten Rauschanteil wählen (Tabelle unten).": (
        "**k-distance without a clear knee:** choose eps by the noise share you want (table below).",
        "**k-odległość bez wyraźnego załamania:** wybierz eps według pożądanego udziału szumu (tabela poniżej)."),
    "{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – solche Zeilen sind oft auffällig.": (
        "{n} rows ({pct}) with values outside the training range – such rows are often anomalous.",
        "{n} wierszy ({pct}) z wartościami poza zakresem treningowym – takie wiersze są często nietypowe."),
    "Am auffälligsten: ": ("Most anomalous: ", "Najbardziej nietypowe: "),
    "Verteilung (DBSCAN): ": ("Distribution (DBSCAN): ", "Rozkład (DBSCAN): "),
    "contamination = auto: Die Schwelle kommt aus dem Verfahren selbst – Score 0.5 wie im Original-Paper zu "
    "Isolation Forest.": (
        "contamination = auto: the threshold comes from the method itself – score 0.5 as in the original Isolation "
        "Forest paper.",
        "contamination = auto: próg wynika z samej metody – wskaźnik 0.5 jak w oryginalnej publikacji o Isolation "
        "Forest."),
    "contamination = auto: Die Schwelle kommt aus dem Verfahren selbst – LOF-Wert 1.5.": (
        "contamination = auto: the threshold comes from the method itself – LOF value 1.5.",
        "contamination = auto: próg wynika z samej metody – wartość LOF 1.5."),
    "contamination = {pct} übernehmen": ("Use contamination = {pct}", "Przejmij contamination = {pct}"),
    ":orange[**Ein Cluster, fast kein Rauschen:**] eps ist eher zu groß – kaum eine Zeile gilt als Anomalie.": (
        ":orange[**One cluster, almost no noise:**] eps is probably too large – hardly any row counts as an anomaly.",
        ":orange[**Jeden klaster, prawie brak szumu:**] eps jest raczej za duże – prawie żaden wiersz nie jest "
        "anomalią."),
    "eps = {eps} übernehmen": ("Use eps = {eps}", "Przejmij eps = {eps}"),
    "Anteil auffälliger Zeilen je '{g}': ": (
        "Share of anomalous rows per '{g}': ",
        "Udział nietypowych wierszy wg '{g}': "),
    # Audit 6
    "DBSCAN: Bei eps = {eps} liegen im Mittel {m} Zeilen im Umkreis jeder Zeile – zusammen etwa {p} Nachbarpaare, "
    "die DBSCAN gleichzeitig im Speicher hält (Grenze: {lim}). eps verkleinern (Knick der k-Distanz in Schritt 7) "
    "oder eine Stichprobe nehmen (Seitenleiste). Viele genau gleiche Zeilen (z. B. nur kategoriale Features) wirken "
    "wie ein großes eps.": (
        "DBSCAN: with eps = {eps}, on average {m} rows lie within the radius of every row – together about {p} "
        "neighbour pairs that DBSCAN keeps in memory at the same time (limit: {lim}). Decrease eps (knee of the "
        "k-distance in step 7) or take a sample (sidebar). Many identical rows (e.g. only categorical features) act "
        "like a large eps.",
        "DBSCAN: przy eps = {eps} w promieniu każdego wiersza leży średnio {m} wierszy – razem ok. {p} par sąsiadów, "
        "które DBSCAN trzyma jednocześnie w pamięci (limit: {lim}). Zmniejsz eps (załamanie k-odległości w kroku 7) "
        "albo weź próbkę (pasek boczny). Wiele identycznych wierszy (np. same cechy kategoryczne) działa jak duże "
        "eps."),
    "LOF: {n} Zeilen × n_neighbors {k} = {p} Abstände im Speicher (Grenze: {lim}). n_neighbors senken oder eine "
    "Stichprobe nehmen (Seitenleiste).": (
        "LOF: {n} rows × n_neighbors {k} = {p} distances in memory (limit: {lim}). Lower n_neighbors or take a "
        "sample (sidebar).",
        "LOF: {n} wierszy × n_neighbors {k} = {p} odległości w pamięci (limit: {lim}). Zmniejsz n_neighbors albo "
        "weź próbkę (pasek boczny)."),
    "Alle Zeilen sind nach der Vorbereitung gleich – es gibt nichts Auffälliges zu finden. In Schritt 4 Features "
    "wählen, die sich zwischen den Zeilen unterscheiden.": (
        "After preparation all rows are identical – there is nothing anomalous to find. In step 4 choose features "
        "that differ between the rows.",
        "Po przygotowaniu wszystkie wiersze są jednakowe – nie ma czego wykrywać. W kroku 4 wybierz cechy, które "
        "różnią się między wierszami."),
    ":orange[**Viele genau gleiche Zeilen:**] Bei {pct} der Zeilen ist die k-Distanz 0 – sie haben mindestens "
    "min_samples − 1 genau gleiche Zeilen (typisch, wenn alle Features kategorial sind). Mit dem kleinsten eps ({e}) "
    "bilden genau gleiche Zeilen die Cluster, seltene Kombinationen sind Rauschen.": (
        ":orange[**Many identical rows:**] for {pct} of the rows the k-distance is 0 – they have at least "
        "min_samples − 1 identical rows (typical when all features are categorical). With the smallest eps ({e}) "
        "identical rows form the clusters and rare combinations are noise.",
        ":orange[**Wiele identycznych wierszy:**] dla {pct} wierszy k-odległość wynosi 0 – mają co najmniej "
        "min_samples − 1 identycznych wierszy (typowe, gdy wszystkie cechy są kategoryczne). Przy najmniejszym eps "
        "({e}) klastry tworzą identyczne wiersze, a rzadkie kombinacje są szumem."),
    "Rauschanteil je eps: eps ab {e} übersprungen – DBSCAN hielte dafür zu viele Nachbarpaare gleichzeitig im "
    "Speicher.": (
        "Noise share per eps: eps from {e} on skipped – DBSCAN would keep too many neighbour pairs in memory at "
        "the same time.",
        "Udział szumu dla eps: eps od {e} pominięte – DBSCAN trzymałby jednocześnie w pamięci zbyt wiele par "
        "sąsiadów."),
}
