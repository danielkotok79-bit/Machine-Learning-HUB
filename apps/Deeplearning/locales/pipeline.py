"""Übersetzungen für pipeline.py: {Deutsch: (Englisch, Polnisch)}"""

T = {
    # Problemtypen
    "Binär (2 Klassen → sigmoid)": ("Binary (2 classes → sigmoid)", "Binarny (2 klasy → sigmoid)"),
    "Mehrklassen (softmax)": ("Multiclass (softmax)", "Wieloklasowy (softmax)"),
    "Nicht binär / Regression (linear, mse)": ("Not binary / regression (linear, mse)",
                                               "Nie binarny / regresja (linear, mse)"),
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
    "layers (ungültige Schichten)": ("layers (invalid layers)", "layers (nieprawidłowe warstwy)"),
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
    # Label & Problemtyp
    "Binär braucht genau 2 verschiedene Label-Werte – gefunden: {k}.": (
        "Binary needs exactly 2 different label values – found: {k}.",
        "Binarny wymaga dokładnie 2 różnych wartości etykiety – znaleziono: {k}."),
    "Mehrklassen braucht 2–200 Klassen – gefunden: {k}.": ("Multiclass needs 2–200 classes – found: {k}.",
                                                          "Wieloklasowy wymaga 2–200 klas – znaleziono: {k}."),
    "Regression braucht ein numerisches Label (Text-Label → Binär oder Mehrklassen).": (
        "Regression needs a numeric label (text label → binary or multiclass).",
        "Regresja wymaga etykiety liczbowej (etykieta tekstowa → binarny albo wieloklasowy)."),
    # Schritt 4
    "Label '{label}' ist nicht in den Daten.": ("Label '{label}' is not in the data.", "Etykiety '{label}' nie ma w danych."),
    "Unbekannte Features: {unknown}": ("Unknown features: {unknown}", "Nieznane cechy: {unknown}"),
    "Bitte mindestens ein Feature auswählen.": ("Please select at least one feature.", "Wybierz co najmniej jedną cechę."),
    "{v} Zeilen ohne Label entfernt": ("{v} rows without a label removed", "Usunięto wiersze bez etykiety: {v}"),
    "{label_inf} Zeilen mit unendlichem Label (inf) entfernt": (
        "{label_inf} rows with an infinite label (inf) removed",
        "Usunięto wiersze z nieskończoną etykietą (inf): {label_inf}"),
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
    "Nur noch {n} Zeilen übrig – zu wenig für Training und Test.": (
        "Only {n} rows left – too few for training and test.",
        "Zostało wierszy: {n} – za mało na trening i test."),
    "Label konnte nicht vollständig kodiert werden.": ("The label could not be fully encoded.",
                                                       "Nie udało się w pełni zakodować etykiety."),
    "Klassen: ": ("Classes: ", "Klasy: "),
    "Stratify nicht möglich (eine Klasse hat < 2 Zeilen) – Split ohne stratify.": (
        "Stratify not possible (a class has < 2 rows) – split without stratify.",
        "Stratify niemożliwe (jedna klasa ma < 2 wiersze) – podział bez stratify."),
    "Stratify nicht möglich: Testmenge ({n_test} Zeilen) kleiner als Anzahl Klassen ({k}) – Split ohne stratify. "
    "Tipp: test_size erhöhen.": (
        "Stratify not possible: test set ({n_test} rows) smaller than the number of classes ({k}) – split without "
        "stratify. Tip: increase test_size.",
        "Stratify niemożliwe: zbiór testowy (wiersze: {n_test}) jest mniejszy niż liczba klas ({k}) – podział bez "
        "stratify. Wskazówka: zwiększ test_size."),
    "Split zuerst: Füllwerte, Kategorien und Scaler nur aus den {n} Trainingszeilen gelernt – die {n2} Testzeilen "
    "werden damit nur umgewandelt (kein Datenleck)": (
        "Split first: fill values, categories and scaler learned only from the {n} training rows – the {n2} test "
        "rows are only transformed with them (no data leakage)",
        "Najpierw podział: wartości uzupełniające, kategorie i skaler wyuczone tylko na wierszach treningowych "
        "({n}) – wiersze testowe ({n2}) są nimi tylko przekształcane (brak wycieku danych)"),
    "Spalten ohne Werte in den Trainingsdaten entfernt: {empty_tr}": (
        "Columns without values in the training data removed: {empty_tr}",
        "Usunięto kolumny bez wartości w danych treningowych: {empty_tr}"),
    "Keine Features mit Werten in den Trainingsdaten übrig.": (
        "No features with values in the training data left.",
        "Nie zostały żadne cechy z wartościami w danych treningowych."),
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
    "Testdaten: {unknown_test} Zeilen mit Kategorien, die in den Trainingsdaten nicht vorkommen → dort alle "
    "Dummies 0 (wie bei neuen Daten)": (
        "Test data: {unknown_test} rows with categories that do not occur in the training data → all dummies 0 "
        "there (as with new data)",
        "Dane testowe: wiersze z kategoriami, których nie ma w danych treningowych: {unknown_test} → tam wszystkie "
        "dummy = 0 (jak przy nowych danych)"),
    "Verdacht auf Datenleck – fast perfekte Korrelation mit dem Label: ": (
        "Possible data leakage – almost perfect correlation with the label: ",
        "Podejrzenie wycieku danych – prawie idealna korelacja z etykietą: "),
    ". Gehört diese Spalte wirklich zu den Features?": (". Does this column really belong to the features?",
                                                        ". Czy ta kolumna naprawdę powinna być cechą?"),
    "Feature(s) {same_source} stammen aus derselben CSV-Spalte wie das Label → Datenleck!": (
        "Feature(s) {same_source} come from the same CSV column as the label → data leakage!",
        "Cechy {same_source} pochodzą z tej samej kolumny CSV co etykieta → wyciek danych!"),
    "Label skaliert: (y − {x}) / {x2} – Vorhersagen werden automatisch zurückgerechnet": (
        "Label scaled: (y − {x}) / {x2} – predictions are converted back automatically",
        "Etykieta przeskalowana: (y − {x}) / {x2} – prognozy są automatycznie przeliczane z powrotem"),
    # Schritt 6
    "Beispiel binär (30 → 15, Dropout 0.5)": ("Example binary (30 → 15, dropout 0.5)",
                                              "Przykład binarny (30 → 15, dropout 0.5)"),
    "Beispiel Regression (4 × 19)": ("Example regression (4 × 19)", "Przykład regresji (4 × 19)"),
    "Beispiel LendingClub (78 → 39 → 19, Dropout 0.2)": ("Example LendingClub (78 → 39 → 19, dropout 0.2)",
                                                          "Przykład LendingClub (78 → 39 → 19, dropout 0.2)"),
    "Faustregel (n → n/2 → n/4)": ("Rule of thumb (n → n/2 → n/4)", "Reguła kciuka (n → n/2 → n/4)"),
    "Layer {i}: Neuronen und Dropout müssen Zahlen sein.": ("Layer {i}: neurons and dropout must be numbers.",
                                                           "Warstwa {i}: neurony i dropout muszą być liczbami."),
    "Layer {i}: Neuronen muss zwischen 1 und {max_units} liegen.": (
        "Layer {i}: neurons must be between 1 and {max_units}.",
        "Warstwa {i}: liczba neuronów musi być od 1 do {max_units}."),
    "Layer {i}: Dropout muss ≥ 0 und < 1 sein (üblich 0 – 0.5).": (
        "Layer {i}: dropout must be ≥ 0 and < 1 (usually 0 – 0.5).",
        "Warstwa {i}: dropout musi być ≥ 0 i < 1 (zwykle 0 – 0.5)."),
    "Layer {i}: unbekannte Aktivierung '{act}'.": ("Layer {i}: unknown activation '{act}'.",
                                                   "Warstwa {i}: nieznana aktywacja '{act}'."),
    "Bitte mindestens einen Hidden Layer anlegen.": ("Please create at least one hidden layer.",
                                                     "Dodaj co najmniej jedną warstwę ukrytą."),
    "Höchstens {max_layers} Hidden Layers.": ("At most {max_layers} hidden layers.",
                                              "Maksymalnie {max_layers} warstw ukrytych."),
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
    "Median der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für eine Vorhersage "
    "braucht aber jede Zeile einen Wert": (
        "median of the training data ({fb}) – in training, rows with a gap here were removed; but for a "
        "prediction every row needs a value",
        "mediana z danych treningowych ({fb}) – w treningu wiersze z brakiem w tej kolumnie usunięto; do "
        "prognozy każdy wiersz potrzebuje jednak wartości"),
    "häufigster Wert der Trainingsdaten ({fb}) – im Training wurden Zeilen mit Lücke hier entfernt; für eine "
    "Vorhersage braucht aber jede Zeile einen Wert": (
        "most frequent value of the training data ({fb}) – in training, rows with a gap here were removed; but "
        "for a prediction every row needs a value",
        "najczęstsza wartość z danych treningowych ({fb}) – w treningu wiersze z brakiem w tej kolumnie usunięto; "
        "do prognozy każdy wiersz potrzebuje jednak wartości"),
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
    "Lücken in {n} Zeilen ({pct}) mit Werten aus dem Training ergänzt. Ergänzte Werte sind Schätzungen – die "
    "Vorhersage dieser Zeilen ist etwas unsicherer.": (
        "Gaps in {n} rows ({pct}) filled with values from training. Filled values are estimates – the prediction "
        "for these rows is somewhat less certain.",
        "Braki w wierszach: {n} ({pct}) uzupełniono wartościami z treningu. Uzupełnione wartości są szacunkami – "
        "prognoza dla tych wierszy jest nieco mniej pewna."),
    "Keine Lücken in den benötigten Spalten.": ("No gaps in the required columns.",
                                                "Wymagane kolumny są kompletne (bez braków)."),
    "{n} Zeilen ({pct}) mit Kategorien, die es im Training nicht gab – dort zählen alle Dummies dieser Spalte als "
    "0 (das Modell kennt die Kategorie nicht).": (
        "{n} rows ({pct}) with categories that did not exist in training – all dummies of this column count as 0 "
        "there (the model does not know the category).",
        "Wiersze z kategoriami, których nie było w treningu: {n} ({pct}) – wszystkie dummy tej kolumny liczą się "
        "tam jako 0 (model nie zna tej kategorii)."),
    "{n} Zeilen ({pct}) mit Werten außerhalb des Trainingsbereichs – dort muss das Modell extrapolieren, die "
    "Vorhersage ist unsicherer.": (
        "{n} rows ({pct}) with values outside the training range – the model has to extrapolate there, the "
        "prediction is less certain.",
        "Wiersze z wartościami poza zakresem treningowym: {n} ({pct}) – model musi tam ekstrapolować, prognoza "
        "jest mniej pewna."),
    "{c}: {n} Zeile": ("{c}: {n} row", "{c}: {n} wiersz"),
    "{c}: {n} Zeilen": ("{c}: {n} rows", "{c}: wiersze: {n}"),
    "{n} Werte in Zahlenspalten waren keine Zahl und wurden wie Lücken behandelt.": (
        "{n} values in number columns were not numbers and were treated like gaps.",
        "Wartości w kolumnach liczbowych, które nie były liczbami i potraktowano je jak braki: {n}."),
    "{n} Zeilen haben im Label einen Wert, den es im Training nicht gab – sie fehlen in der Evaluation.": (
        "{n} rows have a label value that did not exist in training – they are missing from the evaluation.",
        "Wiersze z wartością etykiety, której nie było w treningu: {n} – nie są uwzględnione w ewaluacji."),
    "**Ca. {pct} der Zeilen ({n} von {n2}) kennt das Modell schon aus dem Training** (gleiche Werte wie "
    "Trainingszeilen). Für diese Zeilen hat es die richtige Antwort beim Lernen gesehen – Kennzahlen auf solchen "
    "Daten sind zu optimistisch.": (
        "**The model already knows about {pct} of the rows ({n} of {n2}) from training** (same values as training "
        "rows). For these rows it saw the right answer while learning – metrics on such data are too optimistic.",
        "**Model zna już z treningu ok. {pct} wierszy ({n} z {n2})** (te same wartości co wiersze treningowe). Dla "
        "tych wierszy widział poprawną odpowiedź podczas uczenia – metryki na takich danych są zbyt optymistyczne."),
    "Für eine ehrliche Bewertung nur Daten verwenden, die nicht im Training waren – oder die Kennzahlen aus "
    "Schritt 8 (Testdaten) heranziehen.": (
        "For an honest assessment use only data that were not in training – or refer to the metrics from step 8 "
        "(test data).",
        "Do uczciwej oceny używaj tylko danych, których nie było w treningu – albo korzystaj z metryk z kroku 8 "
        "(dane testowe)."),
    "Vorhergesagt '{pos}': {pct} der Zeilen": ("Predicted '{pos}': {pct} of the rows", "Prognoza '{pos}': {pct} wierszy"),
    " · tatsächlich in diesen Daten: {pct}": (" · actually in these data: {pct}", " · faktycznie w tych danych: {pct}"),
    " · in den Trainingsdaten: {pct}": (" · in the training data: {pct}", " · w danych treningowych: {pct}"),
    "häufiger": ("more often", "częściej"),
    "seltener": ("less often", "rzadziej"),
    "Das Modell sagt '{pos}' {direction} voraus, als es tatsächlich vorkommt: Es erkennt {pct} der echten '{pos}' "
    "(Recall), und {pct2} seiner '{pos}'-Vorhersagen stimmen (Precision). Der Unterschied ist kein Fehler der App, "
    "sondern die Trefferquote des Modells bei der Schwelle {threshold:.2f}.": (
        "The model predicts '{pos}' {direction} than it actually occurs: it detects {pct} of the real '{pos}' "
        "(recall), and {pct2} of its '{pos}' predictions are correct (precision). The difference is not an error "
        "in the app but the model's hit rate at the threshold {threshold:.2f}.",
        "Model przewiduje '{pos}' {direction}, niż występuje w rzeczywistości: rozpoznaje {pct} prawdziwych "
        "'{pos}' (recall), a {pct2} jego prognoz '{pos}' jest trafnych (precision). Różnica nie jest błędem "
        "aplikacji, tylko skutecznością modelu przy progu {threshold:.2f}."),
    "Mehr '{pos}' erkennen: Schwelle senken (z. B. {v:.2f}) oder in Schritt 6 class_weight='balanced' – kostet "
    "mehr Fehlalarme.": (
        "Detect more '{pos}': lower the threshold (e.g. {v:.2f}) or use class_weight='balanced' in step 6 – costs "
        "more false alarms.",
        "Aby wykrywać więcej '{pos}': obniż próg (np. {v:.2f}) albo w kroku 6 użyj class_weight='balanced' – "
        "kosztem większej liczby fałszywych alarmów."),
    "Weniger Fehlalarme für '{pos}': Schwelle erhöhen (z. B. {v:.2f}).": (
        "Fewer false alarms for '{pos}': raise the threshold (e.g. {v:.2f}).",
        "Mniej fałszywych alarmów dla '{pos}': podnieś próg (np. {v:.2f})."),
    "Der vorhergesagte Anteil weicht deutlich vom Training ab – die neuen Daten unterscheiden sich womöglich vom "
    "Training (andere Zeit, andere Gruppe).": (
        "The predicted share differs clearly from training – the new data may differ from the training data "
        "(different period, different group).",
        "Przewidywany udział wyraźnie odbiega od treningu – nowe dane mogą różnić się od treningowych (inny "
        "okres, inna grupa)."),
    "{pct} der Vorhersagen liegen nahe an der Schwelle (±0.1) und sind unsicher; Werte nahe 0 oder 1 sind sicher.": (
        "{pct} of the predictions are close to the threshold (±0.1) and uncertain; values close to 0 or 1 are "
        "certain.",
        "{pct} prognoz leży blisko progu (±0.1) i jest niepewnych; wartości bliskie 0 lub 1 są pewne."),
    "Durchschnittliche Sicherheit der Vorhersagen: {pct}; {pct2} der Zeilen unter 50 % (unsicher).": (
        "Average confidence of the predictions: {pct}; {pct2} of the rows below 50 % (uncertain).",
        "Średnia pewność prognoz: {pct}; {pct2} wierszy poniżej 50 % (niepewne)."),
    "Die Verteilung der vorhergesagten Klassen weicht deutlich vom Training ab.": (
        "The distribution of the predicted classes differs clearly from training.",
        "Rozkład przewidywanych klas wyraźnie odbiega od treningu."),
    "Mittelwert der Vorhersagen: {x} (Training: {x2}).": ("Mean of the predictions: {x} (training: {x2}).",
                                                         "Średnia prognoz: {x} (trening: {x2})."),
    "besser als": ("better than", "lepiej"),
    "schlechter als": ("worse than", "gorzej"),
    "vergleichbar mit": ("comparable to", "porównywalnie"),
    "Abweichung (MAE) hier {x} – {verdict} dem Test aus dem Training ({x2}).": (
        "Error (MAE) here {x} – {verdict} the test from training ({x2}).",
        "Błąd (MAE) tutaj {x}, w teście z treningu {x2} – {verdict}."),
    " – vermutlich, weil das Modell einen Teil der Zeilen aus dem Training kennt": (
        " – probably because the model knows part of the rows from training",
        " – prawdopodobnie dlatego, że model zna część wierszy z treningu"),
    "{name} hier {new:.3f} – besser als im Test ({old:.3f}){why}.": (
        "{name} here {new:.3f} – better than in the test ({old:.3f}){why}.",
        "{name} tutaj {new:.3f} – lepiej niż w teście ({old:.3f}){why}."),
    "{name} hier {new:.3f} – schlechter als im Test ({old:.3f}): Die neuen Daten unterscheiden sich vom Training "
    "(Datenqualität, andere Zeit/Gruppe).": (
        "{name} here {new:.3f} – worse than in the test ({old:.3f}): the new data differ from the training data "
        "(data quality, different period/group).",
        "{name} tutaj {new:.3f} – gorzej niż w teście ({old:.3f}): nowe dane różnią się od treningowych (jakość "
        "danych, inny okres/grupa)."),
    "Prüfen, ob sich die neuen Daten vom Training unterscheiden; ggf. mit aktuelleren Daten neu trainieren.": (
        "Check whether the new data differ from the training data; retrain with more recent data if necessary.",
        "Sprawdź, czy nowe dane różnią się od treningowych; w razie potrzeby wytrenuj model na nowszych danych."),
    "{name} hier {new:.3f}, im Test {old:.3f}.": ("{name} here {new:.3f}, in the test {old:.3f}.",
                                                  "{name} tutaj {new:.3f}, w teście {old:.3f}."),
    "{name} hier {new:.3f} – vergleichbar mit dem Test ({old:.3f}): Das Modell arbeitet auf diesen Daten so gut "
    "wie erwartet.": (
        "{name} here {new:.3f} – comparable to the test ({old:.3f}): the model performs as well as expected on "
        "these data.",
        "{name} tutaj {new:.3f} – porównywalnie z testem ({old:.3f}): model działa na tych danych tak dobrze, jak "
        "oczekiwano."),
    "Datenqualität verbessern: Lücken, unbekannte Kategorien und Ausreißer in der Quelle prüfen – betroffene "
    "Zeilen stehen in der Spalte 'Hinweis'.": (
        "Improve data quality: check gaps, unknown categories and outliers in the source – affected rows are "
        "marked in the column 'Note'.",
        "Popraw jakość danych: sprawdź braki, nieznane kategorie i wartości odstające w źródle – wiersze, których "
        "to dotyczy, są oznaczone w kolumnie 'Uwaga'."),
}
