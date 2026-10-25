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
    # Problemtypen (ML-Version)
    "Binär (2 Klassen)": (
        "Binary (2 classes)",
        "Binarny (2 klasy)"),
    "Mehrklassen": (
        "Multiclass",
        "Wieloklasowy"),
    "Regression (Zahl vorhersagen)": (
        "Regression (predict a number)",
        "Regresja (prognoza liczby)"),
    # Schritt 6 – Modelle (Rangliste) (ML-Version)
    "Decision Trees + Random Forests": (
        "Decision Trees + Random Forests",
        "Decision Trees + Random Forests"),
    "Support Vector Machines": (
        "Support Vector Machines",
        "Support Vector Machines"),
    "Logistische Regression": (
        "Logistic Regression",
        "Logistic Regression"),
    "K-Nearest Neighbors": (
        "K-Nearest Neighbors",
        "K-Nearest Neighbors"),
    "Lineare Regression": (
        "Linear Regression",
        "Linear Regression"),
    "Sehr robust und kaum zu verderben. Kommt mit Nichtlinearitäten, Wechselwirkungen zwischen Features und "
    "Rauschen zurecht. Random Forest begrenzt das Overfitting eines einzelnen Baums stark.": (
        "Very robust and hard to get wrong. Handles non-linearities, feature interactions and noise. Random Forest "
        "strongly limits the overfitting of a single tree.",
        "Bardzo solidny i trudny do zepsucia. Radzi sobie z nieliniowościami, interakcjami cech, szumem. Random "
        "Forest mocno ogranicza overfitting pojedynczego drzewa."),
    "Gut, besonders mit RBF-Kernel. Trennt Klassen gut, ist aber empfindlich gegenüber der Skalierung der "
    "Daten und der Wahl der Hyperparameter.": (
        "Good, especially with the RBF kernel. Separates classes well, but is sensitive to data scaling and to "
        "the choice of hyperparameters.",
        "Dobry, szczególnie z kernelem RBF. Potrafi dobrze oddzielać klasy, ale jest wrażliwy na skalowanie "
        "danych i dobór hiperparametrów."),
    "Solide Baseline. Schnell, interpretierbar, stabil. Verliert bei komplexen Zusammenhängen oft gegen "
    "Bäume.": (
        "Solid baseline. Fast, interpretable, stable. Often loses to trees on complex relationships.",
        "Solidny baseline. Szybki, interpretowalny, stabilny. Często przegrywa z drzewami przy złożonych "
        "zależnościach."),
    "Einfach, aber meist schwächer. Verträgt Rauschen, irrelevante Features und hohe Dimensionalität "
    "(typisch für Börsendaten) schlecht.": (
        "Simple, but usually weaker. Copes poorly with noise, irrelevant features and high dimensionality "
        "(typical of market data).",
        "Prosty, ale zwykle słabszy. Źle znosi szum, nieistotne cechy i wysoką wymiarowość (typową dla danych "
        "giełdowych)."),
    "Am schwächsten für echte Vorhersagen. Setzt einen linearen Zusammenhang voraus, Märkte sind aber stark "
    "nichtlinear und verrauscht.": (
        "The weakest for real predictions. Assumes a linear relationship, but markets are highly non-linear and "
        "noisy.",
        "Najsłabszy do realnych predykcji. Zakłada liniową zależność, a rynki są mocno nieliniowe i zaszumione."),
    "Vorhersage der Richtung (Klassifikation) und der Renditen": (
        "Predicting direction (classification) and returns",
        "Predykcja kierunku (klasyfikacja) i zwrotów"),
    "Klassifikation der Richtung (buy/sell)": (
        "Direction classification (buy/sell)",
        "Klasyfikacja kierunku (buy/sell)"),
    "Einfache Klassifikation der Richtung": (
        "Simple direction classification",
        "Prosta klasyfikacja kierunku"),
    "Eher Experimente / Baseline": (
        "Rather experiments / baseline",
        "Raczej eksperymenty / baseline"),
    "Nur als sehr einfacher Referenzpunkt": (
        "Only as a very simple reference point",
        "Tylko jako bardzo prosty punkt odniesienia"),
    "Random Forest (viele Bäume)": (
        "Random Forest (many trees)",
        "Random Forest (wiele drzew)"),
    "Decision Tree (ein Baum)": (
        "Decision Tree (one tree)",
        "Decision Tree (jedno drzewo)"),
    "Unbekanntes Modell: {m}": (
        "Unknown model: {m}",
        "Nieznany model: {m}"),
    "Logistische Regression sagt Klassen vorher. Für eine Zahl als Label (Regression) z. B. Gradient Boosting "
    "(Platz 1) oder Lineare Regression (Platz 6) wählen – oder in Schritt 4 den Problemtyp ändern.": (
        "Logistic Regression predicts classes. For a number as the label (regression) choose e.g. Gradient "
        "Boosting (rank 1) or Linear Regression (rank 6) – or change the problem type in step 4.",
        "Logistic Regression przewiduje klasy. Dla liczby jako etykiety (regresja) wybierz np. Gradient Boosting "
        "(miejsce 1) albo Linear Regression (miejsce 6) – albo zmień typ problemu w kroku 4."),
    "Lineare Regression sagt eine Zahl vorher. Für Klassen z. B. Gradient Boosting (Platz 1) oder Logistische "
    "Regression (Platz 4) wählen – oder in Schritt 4 den Problemtyp Regression wählen.": (
        "Linear Regression predicts a number. For classes choose e.g. Gradient Boosting (rank 1) or Logistic "
        "Regression (rank 4) – or choose the problem type regression in step 4.",
        "Linear Regression przewiduje liczbę. Dla klas wybierz np. Gradient Boosting (miejsce 1) albo Logistic "
        "Regression (miejsce 4) – albo w kroku 4 wybierz typ problemu regresja."),
    "K = {k} ist größer als die Anzahl der Trainingszeilen ({n}).": (
        "K = {k} is larger than the number of training rows ({n}).",
        "K = {k} jest większe niż liczba wierszy treningowych ({n})."),
    "SVM: Jede Klasse braucht mindestens 2 Trainingszeilen (für die Wahrscheinlichkeiten).": (
        "SVM: every class needs at least 2 training rows (for the probabilities).",
        "SVM: każda klasa potrzebuje co najmniej 2 wierszy treningowych (do prawdopodobieństw)."),
    "GridSearchCV: {n} Kombinationen × {cv} Folds …": (
        "GridSearchCV: {n} combinations × {cv} folds …",
        "GridSearchCV: kombinacje parametrów: {n}, foldy: {cv} …"),
    "GridSearchCV: keine gültige Bewertung (zu wenige Zeilen je Fold) – cv verkleinern oder ohne GridSearchCV "
    "trainieren.": (
        "GridSearchCV: no valid score (too few rows per fold) – reduce cv or train without GridSearchCV.",
        "GridSearchCV: brak poprawnej oceny (za mało wierszy w foldzie) – zmniejsz cv albo trenuj bez "
        "GridSearchCV."),
    "Trainiere das Modell auf allen Trainingsdaten ({n} Zeilen) …": (
        "Training the model on all training data ({n} rows) …",
        "Trenowanie modelu na wszystkich danych treningowych (wiersze: {n}) …"),
    "Validierungskurve: {param} = {v} ({i}/{n})": (
        "Validation curve: {param} = {v} ({i}/{n})",
        "Krzywa walidacji: {param} = {v} ({i}/{n})"),
    "Elbow-Methode: nächste Nachbarn suchen …": (
        "Elbow method: searching nearest neighbours …",
        "Metoda łokcia: szukanie najbliższych sąsiadów …"),
    "None (unbegrenzt, Tiefe {d})": (
        "None (unlimited, depth {d})",
        "None (bez limitu, głębokość {d})"),
    "Anzahl Bäume (n_estimators)": (
        "Number of trees (n_estimators)",
        "Liczba drzew (n_estimators)"),
    "Trainingszeilen": (
        "Training rows",
        "Wiersze treningowe"),
    # Schritt 7 – Training vs. Validierung (ML-Version)
    "max_depth begrenzen oder min_samples_leaf erhöhen": (
        "limit max_depth or increase min_samples_leaf",
        "ogranicz max_depth albo zwiększ min_samples_leaf"),
    "max_depth begrenzen, min_samples_leaf erhöhen oder Random Forest wählen": (
        "limit max_depth, increase min_samples_leaf or choose Random Forest",
        "ogranicz max_depth, zwiększ min_samples_leaf albo wybierz Random Forest"),
    "C oder gamma verkleinern": (
        "reduce C or gamma",
        "zmniejsz C albo gamma"),
    "C verkleinern (stärkere Regularisierung)": (
        "reduce C (stronger regularisation)",
        "zmniejsz C (silniejsza regularyzacja)"),
    "K erhöhen": (
        "increase K",
        "zwiększ K"),
    "weniger Features oder mehr Trainingsdaten": (
        "fewer features or more training data",
        "mniej cech albo więcej danych treningowych"),
    "max_depth erhöhen (0 = unbegrenzt) oder mehr Features je Split (max_features)": (
        "increase max_depth (0 = unlimited) or more features per split (max_features)",
        "zwiększ max_depth (0 = bez limitu) albo więcej cech na podział (max_features)"),
    "max_depth erhöhen oder Random Forest wählen": (
        "increase max_depth or choose Random Forest",
        "zwiększ max_depth albo wybierz Random Forest"),
    "C oder gamma vergrößern, RBF-Kernel versuchen": (
        "increase C or gamma, try the RBF kernel",
        "zwiększ C albo gamma, spróbuj kernela RBF"),
    "C vergrößern oder ein nichtlineares Modell wählen (Random Forest)": (
        "increase C or choose a non-linear model (Random Forest)",
        "zwiększ C albo wybierz model nieliniowy (Random Forest)"),
    "K verkleinern": (
        "reduce K",
        "zmniejsz K"),
    "ein nichtlineares Modell wählen (Random Forest) oder bessere Features bilden": (
        "choose a non-linear model (Random Forest) or build better features",
        "wybierz model nieliniowy (Random Forest) albo zbuduj lepsze cechy"),
    "Beim Random Forest lernt jeder Baum seine Stichprobe fast auswendig": (
        "In a Random Forest each tree learns its sample almost by heart",
        "W Random Forest każde drzewo uczy się swojej próbki niemal na pamięć"),
    "Mit weights='distance' ist jeder Trainingspunkt sein eigener nächster Nachbar": (
        "With weights='distance' every training point is its own nearest neighbour",
        "Przy weights='distance' każdy punkt treningowy jest swoim własnym najbliższym sąsiadem"),
    "Mit K = 1 ist jeder Trainingspunkt sein eigener nächster Nachbar": (
        "With K = 1 every training point is its own nearest neighbour",
        "Przy K = 1 każdy punkt treningowy jest swoim własnym najbliższym sąsiadem"),
    "den Mittelwert": (
        "the mean",
        "średnią"),
    "die häufigste Klasse": (
        "the most frequent class",
        "najczęstszą klasę"),
    "**Nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber {b}, wenn man immer {how} "
    "vorhersagt. Das Modell findet in diesen Features kaum ein Signal – Features prüfen (Schritt 2–4) oder "
    "ein anderes Modell testen.": (
        "**Not better than a naive prediction:** validation {v} versus {b} when always predicting {how}. The "
        "model finds hardly any signal in these features – check the features (step 2–4) or test another model.",
        "**Nie lepiej niż naiwna prognoza:** walidacja {v} wobec {b}, gdy zawsze przewiduje się {how}. Model "
        "prawie nie znajduje sygnału w tych cechach – sprawdź cechy (kroki 2–4) albo przetestuj inny model."),
    "**Trainingsfehler {t} ist hier normal:** {why}. Aussagekräftig ist nur der Validierungsfehler {v} – zum "
    "Vergleich: naive Vorhersage {b}.": (
        "**Training error {t} is normal here:** {why}. Only the validation error {v} is meaningful – for "
        "comparison: naive prediction {b}.",
        "**Błąd treningowy {t} jest tu normalny:** {why}. Miarodajny jest tylko błąd walidacji {v} – dla "
        "porównania: naiwna prognoza {b}."),
    "**Overfitting:** Training {t}, Validierung {v} – das Modell lernt die Trainingsdaten deutlich besser, "
    "als es auf neuen Daten vorhersagt. Gegenmittel: {fix}.": (
        "**Overfitting:** training {t}, validation {v} – the model learns the training data much better than it "
        "predicts new data. Remedies: {fix}.",
        "**Przeuczenie (overfitting):** trening {t}, walidacja {v} – model uczy się danych treningowych wyraźnie "
        "lepiej, niż prognozuje nowe dane. Środki zaradcze: {fix}."),
    "**Underfitting:** Training {t} und Validierung {v} sind ähnlich hoch und kaum besser als die naive "
    "Vorhersage ({b}) – das Modell ist zu einfach für die Daten. Gegenmittel: {fix}.": (
        "**Underfitting:** training {t} and validation {v} are similarly high and hardly better than the naive "
        "prediction ({b}) – the model is too simple for the data. Remedies: {fix}.",
        "**Niedouczenie (underfitting):** trening {t} i walidacja {v} są podobnie wysokie i niewiele lepsze od "
        "naiwnej prognozy ({b}) – model jest zbyt prosty dla tych danych. Środki zaradcze: {fix}."),
    "Training {t} und Validierung {v} liegen nah beieinander und deutlich unter der naiven Vorhersage ({b}) "
    "– keine Anzeichen für Overfitting.": (
        "Training {t} and validation {v} are close together and clearly below the naive prediction ({b}) – no "
        "signs of overfitting.",
        "Trening {t} i walidacja {v} są blisko siebie i wyraźnie poniżej naiwnej prognozy ({b}) – brak oznak "
        "przeuczenia."),
    "Der Validierungsfehler sinkt bis zur letzten Datenmenge – mehr Trainingsdaten würden vermutlich helfen.": (
        "The validation error keeps falling up to the largest amount of data – more training data would probably "
        "help.",
        "Błąd walidacji spada aż do największej ilości danych – więcej danych treningowych prawdopodobnie by "
        "pomogło."),
    "Die Lernkurve ist am Ende flach – mehr Daten derselben Art bringen kaum noch etwas; bessere Features "
    "oder ein anderes Modell schon eher.": (
        "The learning curve is flat at the end – more data of the same kind hardly help; better features or "
        "another model are more promising.",
        "Krzywa uczenia jest na końcu płaska – więcej danych tego samego rodzaju niewiele da; lepsze cechy albo "
        "inny model raczej tak."),
    "Ab etwa {n} Bäumen ändert sich der Validierungsfehler kaum noch ({e} bei {n2} Bäumen). Mehr Bäume "
    "schaden nicht, kosten aber Rechenzeit und Speicher.": (
        "From about {n} trees on, the validation error hardly changes ({e} with {n2} trees). More trees do no "
        "harm, but cost computing time and memory.",
        "Od liczby drzew ok. {n} błąd walidacji prawie się już nie zmienia ({e} przy liczbie drzew {n2}). Więcej "
        "drzew nie szkodzi, ale kosztuje czas obliczeń i pamięć."),
    "**Bester Wert laut Validierungskurve:** {param} = {best} (Fehler {e}). Eingestellt ist {chosen} (Fehler "
    "{e2}).": (
        "**Best value according to the validation curve:** {param} = {best} (error {e}). Currently set: {chosen} "
        "(error {e2}).",
        "**Najlepsza wartość według krzywej walidacji:** {param} = {best} (błąd {e}). Ustawiono: {chosen} (błąd "
        "{e2})."),
    "Der eingestellte Wert {param} = {chosen} liegt am Minimum der Validierungskurve (oder nur knapp "
    "daneben).": (
        "The value set, {param} = {chosen}, lies at the minimum of the validation curve (or only just next to "
        "it).",
        "Ustawiona wartość {param} = {chosen} leży w minimum krzywej walidacji (albo tuż obok)."),
    # Schritt 8 – Modell-Details (ML-Version)
    "Bäume": (
        "Trees",
        "Drzewa"),
    "Tiefe (Mittel)": (
        "Depth (mean)",
        "Głębokość (średnio)"),
    "Blätter (Mittel)": (
        "Leaves (mean)",
        "Liście (średnio)"),
    "Tiefe": (
        "Depth",
        "Głębokość"),
    "Blätter": (
        "Leaves",
        "Liście"),
    "Achsenabschnitt (intercept)": (
        "Intercept",
        "Wyraz wolny (intercept)"),
    "Iterationen": (
        "Iterations",
        "Iteracje"),
    "Stützvektoren": (
        "Support vectors",
        "Wektory nośne"),
    "Stützvektoren je Klasse": (
        "Support vectors per class",
        "Wektory nośne w klasach"),
    "Abstand": (
        "Distance",
        "Odległość"),
    "euklidisch (p=2)": (
        "Euclidean (p=2)",
        "euklidesowa (p=2)"),
    "Manhattan (p=1)": (
        "Manhattan (p=1)",
        "Manhattan (p=1)"),
    "gespeicherte Trainingszeilen": (
        "stored training rows",
        "zapisane wiersze treningowe"),
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
    "Kein scikit-learn-Modell gefunden (model.skops oder model.joblib) – Modelle der Deep-Learning-Version "
    "(.keras/.h5) lassen sich hier nicht laden.": (
        "No scikit-learn model found (model.skops or model.joblib) – models of the deep learning version "
        "(.keras/.h5) cannot be loaded here.",
        "Nie znaleziono modelu scikit-learn (model.skops albo model.joblib) – modeli z wersji deep learning "
        "(.keras/.h5) nie da się tu wczytać."),
    # Zeitlicher Split, Vergleichsspalte (Schritt 4)
    "zufällig (train_test_split)": (
        "random (train_test_split)",
        "losowy (train_test_split)"),
    "zeitlich – Test = die jüngsten Zeilen": (
        "by time – test = the most recent rows",
        "czasowy – test = najnowsze wiersze"),
    "Zeitspalte '{col}': weder eine Zahl (z. B. Jahr, Saison) noch ein erkennbares Datum (z. B. 2024-01-31).": (
        "Time column '{col}': neither a number (e.g. year, season) nor a recognisable date (e.g. 2024-01-31).",
        "Kolumna czasu '{col}': ani liczba (np. rok, sezon), ani rozpoznawalna data (np. 2024-01-31)."),
    "Unbekannte Art des Splits: {v}": (
        "Unknown kind of split: {v}",
        "Nieznany rodzaj podziału: {v}"),
    "Zeitlicher Split: bitte eine Zeitspalte wählen (z. B. date oder season).": (
        "Time-based split: please choose a time column (e.g. date or season).",
        "Podział czasowy: wybierz kolumnę czasu (np. date albo season)."),
    "Die Zeitspalte darf nicht das Label sein.": (
        "The time column must not be the label.",
        "Kolumna czasu nie może być etykietą."),
    "Vergleichsspalte '{col}' ist nicht in den Daten.": (
        "Comparison column '{col}' is not in the data.",
        "Kolumny porównawczej '{col}' nie ma w danych."),
    "{n} Zeilen ohne Wert in der Zeitspalte '{col}' entfernt": (
        "{n} rows without a value in the time column '{col}' removed",
        "Usunięto wiersze bez wartości w kolumnie czasu '{col}': {n}"),
    "Zeitlicher Split nach '{col}': Training {n} Zeilen ({a} … {b}), Test {n2} Zeilen ({c} … {d}) – das Modell "
    "lernt nur aus der Vergangenheit": (
        "Time-based split by '{col}': training {n} rows ({a} … {b}), test {n2} rows ({c} … {d}) – the model only "
        "learns from the past",
        "Podział czasowy według '{col}': trening – wiersze: {n} ({a} … {b}), test – wiersze: {n2} ({c} … {d}) – "
        "model uczy się tylko z przeszłości"),
    "Testanteil {pct} statt {pct2}: Zeilen mit demselben Zeitwert ('{col}') bleiben zusammen – eine Saison oder "
    "ein Tag wird nicht zwischen Training und Test geteilt.": (
        "Test share {pct} instead of {pct2}: rows with the same time value ('{col}') stay together – a season or "
        "a day is not split between training and test.",
        "Udział testu {pct} zamiast {pct2}: wiersze o tej samej wartości czasu ('{col}') zostają razem – sezon "
        "ani dzień nie są dzielone między trening i test."),
    "Vergleichsspalte '{col}' wird ignoriert – den Vergleich gibt es nur bei binären Labels.": (
        "Comparison column '{col}' is ignored – the comparison is only available for binary labels.",
        "Kolumna porównawcza '{col}' jest pomijana – porównanie jest dostępne tylko dla etykiet binarnych."),
    "Vergleichsspalte '{col}' muss Wahrscheinlichkeiten zwischen 0 und 1 enthalten.": (
        "Comparison column '{col}' must contain probabilities between 0 and 1.",
        "Kolumna porównawcza '{col}' musi zawierać prawdopodobieństwa od 0 do 1."),
    "Vergleichsspalte '{col}': {n} Testzeilen ohne Wert – der Vergleich nutzt nur Zeilen mit Wert": (
        "Comparison column '{col}': {n} test rows without a value – the comparison only uses rows with a value",
        "Kolumna porównawcza '{col}': wiersze testowe bez wartości: {n} – porównanie używa tylko wierszy z wartością"),
    "Vergleich": (
        "Comparison",
        "Porównanie"),
    # Gradient Boosting (Schritt 6–8)
    "Gradient Boosting (HistGradientBoosting)": (
        "Gradient Boosting (HistGradientBoosting)",
        "Gradient Boosting (HistGradientBoosting)"),
    "Der beste dieser Liste für Tabellen. Baut Bäume nacheinander – jeder korrigiert die Fehler der vorigen. "
    "Meist genauer als Random Forest und auch bei vielen Zeilen schnell; braucht aber eine passende Anzahl Runden "
    "(max_iter), sonst lernt er Rauschen mit – Schritt 7 zeigt die beste.": (
        "The best of this list for tables. Builds trees one after another – each corrects the errors of the "
        "previous ones. Usually more accurate than Random Forest and fast even with many rows; but it needs a "
        "suitable number of rounds (max_iter), otherwise it also learns noise – step 7 shows the best one.",
        "Najlepszy z tej listy dla danych tabelarycznych. Buduje drzewa po kolei – każde poprawia błędy "
        "poprzednich. Zwykle dokładniejszy niż Random Forest i szybki także przy wielu wierszach; potrzebuje jednak "
        "odpowiedniej liczby rund (max_iter), inaczej uczy się też szumu – najlepszą pokazuje krok 7."),
    "Erste Wahl für Tabellen: Richtung, Sieg/Niederlage (Klassifikation) und Zahlen (Regression)": (
        "First choice for tables: direction, win/loss (classification) and numbers (regression)",
        "Pierwszy wybór dla tabel: kierunek, wygrana/przegrana (klasyfikacja) i liczby (regresja)"),
    "max_iter senken (bester Wert: Validierungskurve), learning_rate verkleinern oder min_samples_leaf erhöhen": (
        "lower max_iter (best value: validation curve), reduce learning_rate or increase min_samples_leaf",
        "zmniejsz max_iter (najlepsza wartość: krzywa walidacji), zmniejsz learning_rate albo zwiększ "
        "min_samples_leaf"),
    "max_iter oder learning_rate erhöhen, größere Bäume (max_leaf_nodes)": (
        "increase max_iter or learning_rate, larger trees (max_leaf_nodes)",
        "zwiększ max_iter albo learning_rate, większe drzewa (max_leaf_nodes)"),
    "Gradient Boosting: {n} Runden trainieren …": (
        "Gradient Boosting: training {n} rounds …",
        "Gradient Boosting: trening, rundy: {n} …"),
    "Anzahl Runden (max_iter)": (
        "Number of rounds (max_iter)",
        "Liczba rund (max_iter)"),
    "Der Validierungsfehler sinkt bis zur letzten Runde – mehr Runden (max_iter) oder eine größere learning_rate "
    "könnten noch helfen.": (
        "The validation error keeps falling until the last round – more rounds (max_iter) or a larger "
        "learning_rate might still help.",
        "Błąd walidacji spada aż do ostatniej rundy – więcej rund (max_iter) albo większy learning_rate mogą "
        "jeszcze pomóc."),
    "Runden (n_iter_)": (
        "Rounds (n_iter_)",
        "Rundy (n_iter_)"),
    # Audyt 5
    " (zeitlicher Split: Training {a} … {b})": (
        " (time-based split: training {a} … {b})",
        " (podział czasowy: trening {a} … {b})"),
    "Zeitlicher Split nicht möglich: Bei test_size = {ts} fällt die Grenze auf den ältesten Zeitwert ({v}) – weil "
    "Zeilen mit demselben Zeitwert zusammenbleiben, kämen alle Zeilen in den Test. test_size höchstens {mx} wählen.": (
        "Time-based split not possible: with test_size = {ts} the boundary falls on the oldest time value ({v}) – since "
        "rows with the same time value stay together, all rows would end up in the test set. Choose test_size of at "
        "most {mx}.",
        "Podział czasowy niemożliwy: przy test_size = {ts} granica wypada na najstarszej wartości czasu ({v}) – "
        "ponieważ wiersze o tej samej wartości czasu zostają razem, wszystkie wiersze trafiłyby do testu. Wybierz "
        "test_size najwyżej {mx}."),
    "Zeitlicher Split nicht möglich: Nur {pct} der Zeilen liegen nach dem ältesten Zeitwert ({v}) – die Spalte trennt "
    "kaum Zeiträume. Eine feinere Zeitspalte wählen (z. B. Datum statt Saison).": (
        "Time-based split not possible: only {pct} of the rows come after the oldest time value ({v}) – the column "
        "hardly separates periods. Choose a finer time column (e.g. date instead of season).",
        "Podział czasowy niemożliwy: tylko {pct} wierszy leży po najstarszej wartości czasu ({v}) – kolumna prawie nie "
        "rozdziela okresów. Wybierz dokładniejszą kolumnę czasu (np. datę zamiast sezonu)."),
    "Zeitlicher Split nicht möglich: Alle Zeilen haben denselben Zeitwert ({v}) – die Spalte trennt keine Zeiträume. "
    "Andere Zeitspalte wählen.": (
        "Time-based split not possible: all rows have the same time value ({v}) – the column does not separate "
        "periods. Choose another time column.",
        "Podział czasowy niemożliwy: wszystkie wiersze mają tę samą wartość czasu ({v}) – kolumna nie rozdziela "
        "okresów. Wybierz inną kolumnę czasu."),
    "test_size muss zwischen 0 und 1 liegen (z. B. 0.2) – angegeben: {v}.": (
        "test_size must be between 0 and 1 (e.g. 0.2) – given: {v}.",
        "test_size musi być między 0 a 1 (np. 0.2) – podano: {v}."),
    "Die Vergleichsspalte darf nicht das Label sein.": (
        "The comparison column must not be the label.",
        "Kolumna porównawcza nie może być etykietą."),
    "Nur {n} Trainingszeilen{period} – zu wenig zum Lernen (mindestens 10). test_size verkleinern oder eine feinere "
    "Zeitspalte wählen.": (
        "Only {n} training rows{period} – too few to learn from (at least 10). Reduce test_size or choose a finer time "
        "column.",
        "Tylko wiersze treningowe: {n}{period} – za mało do nauki (co najmniej 10). Zmniejsz test_size albo wybierz "
        "dokładniejszą kolumnę czasu."),
    "In den Trainingsdaten{period} kommt nur die Klasse '{cls}' vor – das Modell könnte nichts anderes vorhersagen. "
    "test_size ändern, eine andere Zeitspalte oder den zufälligen Split (mit stratify) wählen.": (
        "The training data{period} contain only the class '{cls}' – the model could predict nothing else. Change "
        "test_size, choose another time column or the random split (with stratify).",
        "W danych treningowych{period} jest tylko klasa '{cls}' – model nie mógłby przewidzieć niczego innego. Zmień "
        "test_size, wybierz inną kolumnę czasu albo podział losowy (ze stratify)."),
    "Klassen ohne eine einzige Trainingszeile{period}: {cls} – das Modell kann sie nie vorhersagen.": (
        "Classes without a single training row{period}: {cls} – the model can never predict them.",
        "Klasy bez ani jednego wiersza treningowego{period}: {cls} – model nigdy ich nie przewidzi."),
    "GridSearchCV: keine gültige Bewertung – beim zeitlichen Split lernen die ersten Folds (TimeSeriesSplit) nur aus "
    "den ältesten Zeilen; dort fehlt vermutlich eine Klasse oder es sind zu wenige Zeilen. cv verkleinern oder ohne "
    "GridSearchCV trainieren.": (
        "GridSearchCV: no valid score – with the time-based split the first folds (TimeSeriesSplit) learn only from the "
        "oldest rows; a class is probably missing there or there are too few rows. Reduce cv or train without "
        "GridSearchCV.",
        "GridSearchCV: brak poprawnej oceny – przy podziale czasowym pierwsze foldy (TimeSeriesSplit) uczą się tylko "
        "na najstarszych wierszach; prawdopodobnie brakuje tam jakiejś klasy albo jest za mało wierszy. Zmniejsz cv "
        "albo trenuj bez GridSearchCV."),
    "Vergleichsspalte '{col}' hat in den Testzeilen keinen einzigen Wert.": (
        "Comparison column '{col}' has not a single value in the test rows.",
        "Kolumna porównawcza '{col}' nie ma w wierszach testowych ani jednej wartości."),
    "Vergleichsspalte '{col}' sagt '{pos}' auf den Trainingszeilen schlechter vorher als Zufall (ROC-AUC {auc:.2f}) – "
    "vermutlich ist sie die Wahrscheinlichkeit für '{neg}'. Dann in Schritt 4 '{neg}' als positive Klasse wählen, "
    "sonst ist der Vergleich in Schritt 8 wertlos.": (
        "Comparison column '{col}' predicts '{pos}' worse than chance on the training rows (ROC-AUC {auc:.2f}) – it "
        "is probably the probability of '{neg}'. Then choose '{neg}' as the positive class in step 4, otherwise the "
        "comparison in step 8 is worthless.",
        "Kolumna porównawcza '{col}' przewiduje '{pos}' na wierszach treningowych gorzej niż przypadek (ROC-AUC "
        "{auc:.2f}) – zapewne to prawdopodobieństwo dla '{neg}'. Wtedy w kroku 4 wybierz '{neg}' jako klasę "
        "pozytywną, inaczej porównanie w kroku 8 nic nie znaczy."),
}
