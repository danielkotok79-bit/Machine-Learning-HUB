"""Übersetzungen für pipeline.py: {Deutsch: (Englisch, Polnisch)}"""

T = {
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
    "{v} Zeilen ohne Label entfernt": ("{v} rows without a label removed", "Usunięto wiersze bez etykiety: {v}"),
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
    # Schritt 9 + 10
    "ZIP enthält keine preprocessing.json.": ("The ZIP contains no preprocessing.json.",
                                              "ZIP nie zawiera preprocessing.json."),
    "ZIP zu groß: entpackt {v:.1f} GB (höchstens {v2:.0f} GB).": ("ZIP too large: {v:.1f} GB unpacked (at most {v2:.0f} GB).",
                                                                 "ZIP za duży: po rozpakowaniu {v:.1f} GB (maksymalnie "
                                                                 "{v2:.0f} GB)."),
    "CSV-Spalte": ("CSV column", "Kolumna CSV"),
    "Typ": ("Type", "Typ"),
    "Die Datei enthält keine Datenzeilen (nur Kopfzeile?).": ("The file contains no data rows (header only?).",
                                                              "Plik nie zawiera wierszy danych (tylko nagłówek?)."),
    # Fazit (Schritt 10)
    "gut": ("good", "dobra"),
    "kritisch": ("critical", "krytyczna"),
    "mit Einschränkungen": ("with limitations", "z zastrzeżeniami"),
    ", {n} weitere Spalten werden nicht gebraucht (ignoriert).": (", {n} more columns are not needed (ignored).",
                                                                 ", dodatkowe kolumny ({n}) nie są potrzebne "
                                                                 "(pominięte)."),
    " · tatsächlich in diesen Daten: {pct}": (" · actually in these data: {pct}", " · faktycznie w tych danych: {pct}"),
    " · in den Trainingsdaten: {pct}": (" · in the training data: {pct}", " · w danych treningowych: {pct}"),
    "häufiger": ("more often", "częściej"),
    "seltener": ("less often", "rzadziej"),
    "Mehr '{pos}' erkennen: Schwelle senken (z. B. {v:.2f}) oder in Schritt 6 class_weight='balanced' – kostet "
    "mehr Fehlalarme.": (
        "Detect more '{pos}': lower the threshold (e.g. {v:.2f}) or use class_weight='balanced' in step 6 – costs "
        "more false alarms.",
        "Aby wykrywać więcej '{pos}': obniż próg (np. {v:.2f}) albo w kroku 6 użyj class_weight='balanced' – "
        "kosztem większej liczby fałszywych alarmów."),
    "Weniger Fehlalarme für '{pos}': Schwelle erhöhen (z. B. {v:.2f}).": (
        "Fewer false alarms for '{pos}': raise the threshold (e.g. {v:.2f}).",
        "Mniej fałszywych alarmów dla '{pos}': podnieś próg (np. {v:.2f})."),
    "{pct} der Vorhersagen liegen nahe an der Schwelle (±0.1) und sind unsicher; Werte nahe 0 oder 1 sind sicher.": (
        "{pct} of the predictions are close to the threshold (±0.1) and uncertain; values close to 0 or 1 are "
        "certain.",
        "{pct} prognoz leży blisko progu (±0.1) i jest niepewnych; wartości bliskie 0 lub 1 są pewne."),
    "Die Verteilung der vorhergesagten Klassen weicht deutlich vom Training ab.": (
        "The distribution of the predicted classes differs clearly from training.",
        "Rozkład przewidywanych klas wyraźnie odbiega od treningu."),
    "{name} hier {new:.3f} – besser als im Test ({old:.3f}){why}.": (
        "{name} here {new:.3f} – better than in the test ({old:.3f}){why}.",
        "{name} tutaj {new:.3f} – lepiej niż w teście ({old:.3f}){why}."),
    "{name} hier {new:.3f}, im Test {old:.3f}.": ("{name} here {new:.3f}, in the test {old:.3f}.",
                                                  "{name} tutaj {new:.3f}, w teście {old:.3f}."),
    # Problemtypen (ML-Version)
    "Binär (2 Klassen)": (
        "Binary (2 classes)",
        "Binarny (2 klasy)"),
    "Mehrklassen": (
        "Multiclass",
        "Wieloklasowy"),
    # Schritt 6 – Modelle (Rangliste) (ML-Version)
    "Logistische Regression": (
        "Logistic Regression",
        "Regresja logistyczna"),
    "Unbekanntes Modell: {m}": (
        "Unknown model: {m}",
        "Nieznany model: {m}"),
    "Validierungskurve: {param} = {v} ({i}/{n})": (
        "Validation curve: {param} = {v} ({i}/{n})",
        "Krzywa walidacji: {param} = {v} ({i}/{n})"),
    # Schritt 7 – Training vs. Validierung (ML-Version)
    "Training {t} und Validierung {v} liegen nah beieinander und deutlich unter der naiven Vorhersage ({b}) "
    "– keine Anzeichen für Overfitting.": (
        "Training {t} and validation {v} are close together and clearly below the naive prediction ({b}) – no "
        "signs of overfitting.",
        "Trening {t} i walidacja {v} są blisko siebie i wyraźnie poniżej naiwnej prognozy ({b}) – brak oznak "
        "przeuczenia."),
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
    # --- NLP-App (Textklassifikation) ---
    "Text → Länge (Zeichen)": ("Text → length (characters)", "Tekst → długość (znaki)"),
    "Text → Anzahl Wörter": ("Text → number of words", "Tekst → liczba słów"),
    "Zeilen ohne Text entfernen": ("Drop rows without text", "Usuń wiersze bez tekstu"),
    "als leeren Text behandeln": ("Treat as empty text", "Traktuj jako pusty tekst"),
    "CountVectorizer – Wörter zählen (Bag of Words, wie im Kurs)": (
        "CountVectorizer – count words (bag of words, as in the course)",
        "CountVectorizer – zliczanie słów (bag of words, jak na kursie)"),
    "TfidfVectorizer – TF-IDF (Wörter, die in vielen Texten vorkommen, zählen weniger)": (
        "TfidfVectorizer – TF-IDF (words that appear in many texts count less)",
        "TfidfVectorizer – TF-IDF (słowa występujące w wielu tekstach liczą się mniej)"),
    "Englisch (Liste von scikit-learn)": ("English (scikit-learn list)", "Angielskie (lista scikit-learn)"),
    "Naive Bayes (MultinomialNB)": ("Naive Bayes (MultinomialNB)", "Naiwny Bayes (MultinomialNB)"),
    "Lineare SVM (LinearSVC)": ("Linear SVM (LinearSVC)", "Liniowa SVM (LinearSVC)"),
    "Der Klassiker für Texte und das Modell aus dem Kurs: zählt, wie typisch jedes Wort für eine Klasse ist. "
    "Sehr schnell, braucht wenig Daten, gute Baseline – nimmt aber an, dass Wörter unabhängig voneinander sind.": (
        "The classic for text and the model from the course: counts how typical each word is for a class. Very "
        "fast, needs little data, a good baseline – but it assumes that words are independent of each other.",
        "Klasyk dla tekstów i model z kursu: liczy, jak typowe jest każde słowo dla danej klasy. Bardzo szybki, "
        "potrzebuje mało danych, dobry punkt odniesienia – zakłada jednak, że słowa są od siebie niezależne."),
    "Gewichtet jedes Wort und liefert gut kalibrierte Wahrscheinlichkeiten. Meist genauer als Naive Bayes, wenn "
    "genug Texte da sind; die Gewichte sind leicht zu lesen.": (
        "Weights every word and delivers well-calibrated probabilities. Usually more accurate than Naive Bayes "
        "when there are enough texts; the weights are easy to read.",
        "Waży każde słowo i daje dobrze skalibrowane prawdopodobieństwa. Zwykle dokładniejsza niż naiwny Bayes, "
        "gdy tekstów jest wystarczająco dużo; wagi łatwo odczytać."),
    "Sucht die Trennlinie mit dem größten Abstand zwischen den Klassen – bei Texten mit TF-IDF oft das genaueste "
    "der drei. Wahrscheinlichkeiten kommen über eine Kalibrierung (CalibratedClassifierCV).": (
        "Looks for the boundary with the largest margin between the classes – with TF-IDF on text often the most "
        "accurate of the three. Probabilities come from a calibration (CalibratedClassifierCV).",
        "Szuka granicy o największym marginesie między klasami – dla tekstów z TF-IDF często najdokładniejsza z "
        "trzech. Prawdopodobieństwa pochodzą z kalibracji (CalibratedClassifierCV)."),
    "schneller Start, wenige Texte, Baseline": (
        "quick start, few texts, baseline",
        "szybki start, mało tekstów, punkt odniesienia"),
    "Wahrscheinlichkeiten und Schwelle, lesbare Gewichte": (
        "probabilities and threshold, readable weights",
        "prawdopodobieństwa i próg, czytelne wagi"),
    "viele Texte, TF-IDF, höchste Genauigkeit": (
        "many texts, TF-IDF, highest accuracy",
        "dużo tekstów, TF-IDF, najwyższa dokładność"),
    "alpha vergrößern (stärkere Glättung) oder in Schritt 4 min_df erhöhen": (
        "increase alpha (stronger smoothing) or raise min_df in step 4",
        "zwiększ alpha (silniejsze wygładzanie) albo podnieś min_df w kroku 4"),
    "C verkleinern (stärkere Regularisierung) oder in Schritt 4 min_df erhöhen": (
        "decrease C (stronger regularization) or raise min_df in step 4",
        "zmniejsz C (silniejsza regularyzacja) albo podnieś min_df w kroku 4"),
    "alpha verkleinern, in Schritt 4 Wortpaare (ngram 2) oder TF-IDF probieren": (
        "decrease alpha, try word pairs (ngram 2) or TF-IDF in step 4",
        "zmniejsz alpha, w kroku 4 spróbuj par słów (ngram 2) albo TF-IDF"),
    "C vergrößern, in Schritt 4 Wortpaare (ngram 2) probieren": (
        "increase C, try word pairs (ngram 2) in step 4",
        "zwiększ C, w kroku 4 spróbuj par słów (ngram 2)"),
    "leerer Text": ("empty text", "pusty tekst"),
    "kein bekanntes Wort": ("no known word", "brak znanego słowa"),
    "überwiegend unbekannte Wörter": ("mostly unknown words", "głównie nieznane słowa"),
    "unverändert – eine Textspalte (z. B. die Bewertung) wird in Schritt 4 die Textspalte": (
        "unchanged – a text column (e.g. the review) becomes the text column in step 4",
        "bez zmian – kolumna tekstowa (np. recenzja) staje się w kroku 4 kolumną tekstu"),
    "Spalte löschen (z. B. IDs wie review_id oder user_id)": (
        "delete the column (e.g. IDs such as review_id or user_id)",
        "usuń kolumnę (np. identyfikatory jak review_id albo user_id)"),
    "Anzahl Zeichen je Text – wie im Kurs: text.apply(len); meist als neue Spalte (Kopie + Aktion)": (
        "number of characters per text – as in the course: text.apply(len); usually as a new column (copy + "
        "action)",
        "liczba znaków w tekście – jak na kursie: text.apply(len); zwykle jako nowa kolumna (kopia + akcja)"),
    "Anzahl Wörter je Text (durch Leerzeichen getrennt); meist als neue Spalte (Kopie + Aktion)": (
        "number of words per text (separated by spaces); usually as a new column (copy + action)",
        "liczba słów w tekście (rozdzielonych spacjami); zwykle jako nowa kolumna (kopia + akcja)"),
    "Split zuerst: Vokabular nur aus den {n} Trainingstexten gelernt ({v} Wörter{grams}) – die {n2} Testtexte "
    "werden damit nur umgewandelt (kein Datenleck)": (
        "Split first: vocabulary learned only from the {n} training texts ({v} words{grams}) – the {n2} test "
        "texts are only transformed with it (no data leakage)",
        "Najpierw podział: słownik nauczony tylko z {n} tekstów treningowych ({v} słów{grams}) – {n2} tekstów "
        "testowych jest nim tylko przekształcanych (brak wycieku danych)"),
    "Trainiere das Modell auf allen Trainingsdaten ({n} Texte) …": (
        "Training the model on all training data ({n} texts) …",
        "Trenuję model na wszystkich danych treningowych ({n} tekstów) …"),
    "Fertig.": ("Done.", "Gotowe."),
    "Vorhergesagt '{pos}': {pct} der Texte": (
        "Predicted '{pos}': {pct} of the texts",
        "Przewidziano '{pos}': {pct} tekstów"),
    "keine Kopfzeile": ("no header row", "brak wiersza nagłówka"),
    "Länge in Zeichen": ("length in characters", "długość w znakach"),
    "Anzahl Wörter": ("number of words", "liczba słów"),
    "Unbekannter Vektorisierer: {v}": ("Unknown vectorizer: {v}", "Nieznany wektoryzator: {v}"),
    "Unbekannte Stoppwort-Liste: {v}": ("Unknown stop word list: {v}", "Nieznana lista stop words: {v}"),
    "Textspalte": ("Text column", "Kolumna tekstu"),
    "Label": ("Label", "Etykieta"),
    "Textspalte und Label müssen verschiedene Spalten sein.": (
        "Text column and label must be different columns.",
        "Kolumna tekstu i etykieta muszą być różnymi kolumnami."),
    "Die Spalte '{c}' enthält Zahlen, keinen Text – bitte die Spalte mit den Texten wählen.": (
        "The column '{c}' contains numbers, not text – please choose the column with the texts.",
        "Kolumna '{c}' zawiera liczby, a nie tekst – wybierz kolumnę z tekstami."),
    "test_size muss zwischen 0 und 1 liegen.": (
        "test_size must be between 0 and 1.",
        "test_size musi leżeć między 0 a 1."),
    "{v} Zeilen ohne Text entfernt": ("{v} rows without text removed", "Usunięto {v} wierszy bez tekstu"),
    "Das Label hat nur {k} Klasse – für eine Klassifikation sind mindestens 2 nötig.": (
        "The label has only {k} class – a classification needs at least 2.",
        "Etykieta ma tylko {k} klasę – klasyfikacja wymaga co najmniej 2."),
    "Das Label hat {k} Klassen – höchstens {m} sind möglich. Ist das wirklich das Label (oder eine ID / ein "
    "Text)? Sonst in Schritt 4 nur einige Klassen auswählen.": (
        "The label has {k} classes – at most {m} are possible. Is this really the label (or an ID / a text)? "
        "Otherwise select only some classes in step 4.",
        "Etykieta ma {k} klas – możliwe jest najwyżej {m}. Czy to na pewno etykieta (a nie identyfikator / "
        "tekst)? W przeciwnym razie wybierz w kroku 4 tylko niektóre klasy."),
    "Nur noch {n} Zeilen übrig – zu wenig für Training und Test (mindestens {m}).": (
        "Only {n} rows left – too few for training and test (at least {m}).",
        "Zostało tylko {n} wierszy – za mało na trening i test (co najmniej {m})."),
    "Die Klasse(n) {c} kommen in den Trainingsdaten nicht vor – stratify einschalten, test_size verkleinern oder "
    "die Klasse in Schritt 4 abwählen.": (
        "The class(es) {c} do not occur in the training data – turn on stratify, reduce test_size or deselect "
        "the class in step 4.",
        "Klasy {c} nie występują w danych treningowych – włącz stratify, zmniejsz test_size albo odznacz klasę w "
        "kroku 4."),
    "Texte ohne ein einziges Wort aus dem Vokabular: {a} im Training, {b} im Test (nur Stoppwörter, Zahlen, "
    "einzelne Zeichen oder unbekannte Wörter)": (
        "Texts without a single word from the vocabulary: {a} in training, {b} in test (only stop words, "
        "numbers, single characters or unknown words)",
        "Teksty bez ani jednego słowa ze słownika: {a} w treningu, {b} w teście (same stop words, liczby, "
        "pojedyncze znaki albo nieznane słowa)"),
    "Sehr großes Vokabular ({v} Wörter) – Training und Speichern dauern länger. Tipp: min_df erhöhen (z. B. 2) "
    "oder max_features begrenzen.": (
        "Very large vocabulary ({v} words) – training and saving take longer. Tip: raise min_df (e.g. 2) or "
        "limit max_features.",
        "Bardzo duży słownik ({v} słów) – trening i zapis trwają dłużej. Wskazówka: podnieś min_df (np. 2) albo "
        "ogranicz max_features."),
    "{pct} der Testtexte ({n}) kommen wortgleich auch im Training vor – für sie kennt das Modell die Antwort "
    "schon; die Testgenauigkeit ist dadurch etwas zu optimistisch. Tipp: doppelte Texte in den Daten prüfen.": (
        "{pct} of the test texts ({n}) also appear word for word in training – for them the model already knows "
        "the answer, so the test accuracy is somewhat too optimistic. Tip: check the data for duplicate texts.",
        "{pct} tekstów testowych ({n}) występuje słowo w słowo także w treningu – dla nich model zna już "
        "odpowiedź, więc dokładność na teście jest nieco zbyt optymistyczna. Wskazówka: sprawdź dane pod kątem "
        "duplikatów."),
    "Textspalte und Label stammen aus derselben CSV-Spalte ({c}) → Datenleck!": (
        "Text column and label come from the same CSV column ({c}) → data leakage!",
        "Kolumna tekstu i etykieta pochodzą z tej samej kolumny CSV ({c}) → wyciek danych!"),
    "Vergleich: {m} ({i}/{n})": ("Comparison: {m} ({i}/{n})", "Porównanie: {m} ({i}/{n})"),
    "**Nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber {b}, wenn man immer die häufigste "
    "Klasse vorhersagt. In diesen Texten steckt für das Modell kaum ein Signal – Textspalte, Label und "
    "Vektorisierung prüfen (Schritt 4).": (
        "**Not better than a naive prediction:** validation {v} compared with {b} when always predicting the "
        "most frequent class. The model finds hardly any signal in these texts – check the text column, label "
        "and vectorization (step 4).",
        "**Nie lepiej niż naiwna prognoza:** walidacja {v} wobec {b}, gdy zawsze przewiduje się najczęstszą "
        "klasę. Model nie znajduje w tych tekstach prawie żadnego sygnału – sprawdź kolumnę tekstu, etykietę i "
        "wektoryzację (krok 4)."),
    "Dieses Modell stammt nicht aus der NLP-Version ({app}) – hier lassen sich nur Textklassifikations-Modelle "
    "laden, die mit dieser App gespeichert wurden.": (
        "This model does not come from the NLP version ({app}) – only text classification models saved with this "
        "app can be loaded here.",
        "Ten model nie pochodzi z wersji NLP ({app}) – tutaj można wczytać tylko modele klasyfikacji tekstu "
        "zapisane w tej aplikacji."),
    "Die Modelldatei enthält keine Pipeline aus Vektorisierer und Textmodell ({type}).": (
        "The model file does not contain a pipeline of vectorizer and text model ({type}).",
        "Plik modelu nie zawiera potoku z wektoryzatora i modelu tekstowego ({type})."),
    "Die Pipeline in der Modelldatei ist nicht trainiert.": (
        "The pipeline in the model file is not trained.",
        "Potok w pliku modelu nie jest wytrenowany."),
    "Modell und preprocessing.json passen nicht zusammen: das Modell kennt {k} Klassen, das Rezept {m}.": (
        "Model and preprocessing.json do not match: the model knows {k} classes, the recipe {m}.",
        "Model i preprocessing.json nie pasują do siebie: model zna {k} klas, przepis {m}."),
    "Vektorisierer und Modell passen nicht zusammen ({a} Wörter im Vokabular, das Modell erwartet {b}).": (
        "Vectorizer and model do not match ({a} words in the vocabulary, the model expects {b}).",
        "Wektoryzator i model nie pasują do siebie ({a} słów w słowniku, model oczekuje {b})."),
    "Modell und preprocessing.json passen nicht zusammen: laut Rezept {m}, in der Datei {type}.": (
        "Model and preprocessing.json do not match: according to the recipe {m}, in the file {type}.",
        "Model i preprocessing.json nie pasują do siebie: według przepisu {m}, w pliku {type}."),
    "{n} Texte aus der Spalte '{c}'": ("{n} texts from the column '{c}'", "{n} tekstów z kolumny '{c}'"),
    "{n} leere Texte ({pct}) – dort sagt das Modell nur nach der Häufigkeit der Klassen im Training voraus.": (
        "{n} empty texts ({pct}) – for them the model predicts only from the class frequencies in training.",
        "{n} pustych tekstów ({pct}) – dla nich model przewiduje tylko na podstawie częstości klas w treningu."),
    "{n} Texte ({pct}) enthalten kein einziges Wort aus dem Vokabular des Trainings (andere Sprache, nur Zahlen, "
    "Tippfehler?) – ihre Vorhersage ist geraten.": (
        "{n} texts ({pct}) contain not a single word from the training vocabulary (other language, only numbers, "
        "typos?) – their prediction is a guess.",
        "{n} tekstów ({pct}) nie zawiera ani jednego słowa ze słownika treningowego (inny język, same liczby, "
        "literówki?) – ich prognoza to zgadywanie."),
    "{n} Texte ({pct}) bestehen überwiegend aus Wörtern, die das Modell nicht kennt – die Vorhersage stützt sich "
    "auf wenige Wörter.": (
        "{n} texts ({pct}) consist mostly of words the model does not know – the prediction rests on few words.",
        "{n} tekstów ({pct}) składa się głównie ze słów, których model nie zna – prognoza opiera się na niewielu "
        "słowach."),
    "Die neuen Texte sind im Mittel {a} Wörter lang, im Training {b} – sehr andere Texte als im Training.": (
        "The new texts are {a} words long on average, in training {b} – very different texts from training.",
        "Nowe teksty mają średnio {a} słów, w treningu {b} – bardzo inne teksty niż w treningu."),
    "{n} Zeilen haben im Label eine Klasse, die das Modell nicht kennt (trainiert mit: {c}) – sie fehlen in der "
    "Evaluation; vorhergesagt wird für sie trotzdem eine der bekannten Klassen.": (
        "{n} rows have a class in the label that the model does not know (trained with: {c}) – they are missing "
        "from the evaluation; one of the known classes is still predicted for them.",
        "{n} wierszy ma w etykiecie klasę, której model nie zna (trenowany na: {c}) – brakuje ich w ewaluacji; "
        "mimo to przewidywana jest dla nich jedna ze znanych klas."),
    "**Ca. {pct} der Texte ({n} von {n2}) kennt das Modell wortgleich aus dem Training.** Für diese Texte hat es "
    "die richtige Antwort beim Lernen gesehen – Kennzahlen auf solchen Daten sind zu optimistisch.": (
        "**The model knows about {pct} of the texts ({n} of {n2}) word for word from training.** For these texts "
        "it saw the correct answer while learning – metrics on such data are too optimistic.",
        "**Model zna ok. {pct} tekstów ({n} z {n2}) słowo w słowo z treningu.** Dla tych tekstów widział "
        "poprawną odpowiedź podczas nauki – wskaźniki na takich danych są zbyt optymistyczne."),
    "Für eine ehrliche Bewertung nur Texte verwenden, die nicht im Training waren – oder die Kennzahlen aus "
    "Schritt 8 (Testdaten) heranziehen.": (
        "For an honest assessment use only texts that were not in training – or rely on the metrics from step 8 "
        "(test data).",
        "Do uczciwej oceny używaj tylko tekstów, których nie było w treningu – albo opieraj się na wskaźnikach z "
        "kroku 8 (dane testowe)."),
    "Durchschnittliche Sicherheit der Vorhersagen: {pct}; {pct2} der Texte unter 50 % (unsicher).": (
        "Average confidence of the predictions: {pct}; {pct2} of the texts below 50 % (uncertain).",
        "Średnia pewność prognoz: {pct}; {pct2} tekstów poniżej 50 % (niepewne)."),
    "Texte prüfen: leere Texte, andere Sprache oder ganz andere Themen als im Training – betroffene Zeilen "
    "stehen in der Spalte 'Hinweis'.": (
        "Check the texts: empty texts, another language or completely different topics than in training – "
        "affected rows are marked in the column 'Hinweis' (note).",
        "Sprawdź teksty: puste teksty, inny język albo zupełnie inne tematy niż w treningu – dotknięte wiersze "
        "są oznaczone w kolumnie 'Hinweis' (uwaga)."),
    "{what} '{c}' ist nicht in den Daten.": (
        "{what} '{c}' is not in the data.",
        "{what} '{c}' nie występuje w danych."),
    "Bitte mindestens 2 Klassen auswählen (vorhanden: {v}).": (
        "Please select at least 2 classes (available: {v}).",
        "Wybierz co najmniej 2 klasy (dostępne: {v})."),
    "Nur die Klassen {keep}: {v} Zeilen mit anderen Klassen entfernt (wie im Kurs: nur 1 und 5 Sterne)": (
        "Only the classes {keep}: {v} rows with other classes removed (as in the course: only 1 and 5 stars)",
        "Tylko klasy {keep}: usunięto {v} wierszy z innymi klasami (jak na kursie: tylko 1 i 5 gwiazdek)"),
    "{v} Zeilen ohne Text – als leerer Text behandelt (das Modell sagt dort nur nach der Häufigkeit der Klassen "
    "voraus)": (
        "{v} rows without text – treated as empty text (the model predicts there only from the class frequencies)",
        "{v} wierszy bez tekstu – traktowane jako pusty tekst (model przewiduje tam tylko na podstawie częstości "
        "klas)"),
    "**Trainingsfehler {t} ist bei Texten normal:** Es gibt viel mehr Wörter als Texte, das Modell kann die "
    "Trainingstexte fast immer trennen. Aussagekräftig ist der Validierungsfehler {v} – zum Vergleich: naive "
    "Vorhersage {b}. Liegt er deutlich über dem Minimum der Kurve, hilft: {fix}.": (
        "**A training error of {t} is normal for text:** there are many more words than texts, so the model can "
        "almost always separate the training texts. What matters is the validation error {v} – for comparison: "
        "naive prediction {b}. If it is clearly above the minimum of the curve, this helps: {fix}.",
        "**Błąd treningowy {t} jest normalny dla tekstów:** słów jest dużo więcej niż tekstów, więc model prawie "
        "zawsze potrafi rozdzielić teksty treningowe. Liczy się błąd walidacji {v} – dla porównania: naiwna "
        "prognoza {b}. Jeśli jest wyraźnie powyżej minimum krzywej, pomaga: {fix}."),
    "Die Kurve nutzt die Testdaten (wie im Kurs). Wer danach einen Wert auswählt, hat die Testdaten schon zur "
    "Wahl benutzt – die Testgenauigkeit ist dann leicht zu optimistisch. Sauberer: in Schritt 6 „Anteil aus "
    "Training“ wählen.": (
        "The curve uses the test data (as in the course). Choosing a value afterwards means the test data has "
        "already been used for the choice – the test accuracy is then slightly too optimistic. Cleaner: choose "
        "“Share of training” in step 6.",
        "Krzywa używa danych testowych (jak na kursie). Kto potem wybiera wartość, użył już danych testowych do "
        "wyboru – dokładność na teście jest wtedy nieco zbyt optymistyczna. Czyściej: w kroku 6 wybierz „Część "
        "danych treningowych”."),
    "preprocessing.json ist keine gültige JSON-Datei ({exc}).": (
        "preprocessing.json is not a valid JSON file ({exc}).",
        "preprocessing.json nie jest poprawnym plikiem JSON ({exc})."),
    "Spalte im Modell": ("Column in the model", "Kolumna w modelu"),
    "Text (nötig)": ("Text (required)", "Tekst (wymagany)"),
    "Label (optional – dann wird die Vorhersage geprüft)": (
        "Label (optional – then the prediction is checked)",
        "Etykieta (opcjonalnie – wtedy prognoza zostanie sprawdzona)"),
    "Das Modell kennt nur {pct} der Wörter in diesen Texten – in den Testdaten waren es {pct2}. Die neuen Texte "
    "unterscheiden sich deutlich vom Training (andere Sprache, anderes Thema, andere Schreibweise?).": (
        "The model knows only {pct} of the words in these texts – in the test data it was {pct2}. The new texts "
        "differ clearly from training (other language, other topic, other spelling?).",
        "Model zna tylko {pct} słów w tych tekstach – w danych testowych było to {pct2}. Nowe teksty wyraźnie "
        "różnią się od treningu (inny język, inny temat, inna pisownia?)."),
    "Das Modell wurde nur mit den Klassen {c} trainiert (Schritt 4). Für alle Klassen in Schritt 4 keine Auswahl "
    "treffen und neu trainieren.": (
        "The model was trained only with the classes {c} (step 4). For all classes, make no selection in step 4 "
        "and train again.",
        "Model był trenowany tylko na klasach {c} (krok 4). Aby użyć wszystkich klas, nie zaznaczaj niczego w "
        "kroku 4 i wytrenuj ponownie."),
    " · in den {n} Zeilen mit bekanntem Label: vorhergesagt {pct}, tatsächlich {pct2}": (
        " · in the {n} rows with a known label: predicted {pct}, actual {pct2}",
        " · w {n} wierszach ze znaną etykietą: przewidziano {pct}, faktycznie {pct2}"),
    " – vermutlich, weil das Modell einen Teil der Texte aus dem Training kennt": (
        " – probably because the model knows some of the texts from training",
        " – prawdopodobnie dlatego, że model zna część tekstów z treningu"),
    "Kein Wort bleibt im Vokabular – die Texte bestehen nur aus Stoppwörtern oder einzelnen Zeichen, oder min_df "
    "/ max_df sind zu streng ({exc}).": (
        "No word remains in the vocabulary – the texts consist only of stop words or single characters, or "
        "min_df / max_df are too strict ({exc}).",
        "W słowniku nie zostało żadne słowo – teksty składają się tylko ze stop words albo pojedynczych znaków, "
        "albo min_df / max_df są zbyt surowe ({exc})."),
    " und Wortpaare": (" and word pairs", " i par słów"),
    "**Overfitting:** Training {t}, Validierung {v} – das Modell lernt die Trainingstexte deutlich besser, als "
    "es neue Texte einordnet. Gegenmittel: {fix}.": (
        "**Overfitting:** training {t}, validation {v} – the model learns the training texts much better than it "
        "classifies new texts. Remedy: {fix}.",
        "**Przeuczenie:** trening {t}, walidacja {v} – model uczy się tekstów treningowych dużo lepiej, niż "
        "klasyfikuje nowe teksty. Środek zaradczy: {fix}."),
    "Das Modell kennt {pct} der Wörter in diesen Texten": (
        "The model knows {pct} of the words in these texts",
        "Model zna {pct} słów w tych tekstach"),
    "Das Modell sagt '{pos}' {direction} voraus, als es tatsächlich vorkommt: Es erkennt {pct} der echten "
    "'{pos}' (Recall), und {pct2} seiner '{pos}'-Vorhersagen stimmen (Precision) – bei der Schwelle "
    "{threshold:.2f}.": (
        "The model predicts '{pos}' {direction} than it actually occurs: it detects {pct} of the real '{pos}' "
        "(recall), and {pct2} of its '{pos}' predictions are correct (precision) – at the threshold "
        "{threshold:.2f}.",
        "Model przewiduje '{pos}' {direction}, niż faktycznie występuje: wykrywa {pct} prawdziwych '{pos}' "
        "(recall), a {pct2} jego prognoz '{pos}' jest trafnych (precision) – przy progu {threshold:.2f}."),
    "Der vorhergesagte Anteil weicht deutlich vom Training ab – die neuen Texte unterscheiden sich womöglich vom "
    "Training (anderes Thema, andere Quelle, andere Zeit).": (
        "The predicted share differs clearly from training – the new texts may differ from training (other "
        "topic, other source, other time).",
        "Przewidywany udział wyraźnie odbiega od treningu – nowe teksty mogą różnić się od treningu (inny temat, "
        "inne źródło, inny czas)."),
    "{name} hier {new:.3f} – schlechter als im Test ({old:.3f}): Die neuen Texte unterscheiden sich vom Training "
    "(Thema, Sprache, Quelle, Zeit).": (
        "{name} here {new:.3f} – worse than in the test ({old:.3f}): the new texts differ from training (topic, "
        "language, source, time).",
        "{name} tutaj {new:.3f} – gorzej niż w teście ({old:.3f}): nowe teksty różnią się od treningu (temat, "
        "język, źródło, czas)."),
    "Prüfen, ob sich die neuen Texte vom Training unterscheiden; ggf. mit aktuelleren Texten neu trainieren.": (
        "Check whether the new texts differ from training; if so, retrain with more recent texts.",
        "Sprawdź, czy nowe teksty różnią się od treningu; w razie potrzeby wytrenuj ponownie na nowszych "
        "tekstach."),
    "**Underfitting:** Training {t} und Validierung {v} sind ähnlich hoch und kaum besser als die naive "
    "Vorhersage ({b}) – das Modell ist zu einfach. Gegenmittel: {fix}.": (
        "**Underfitting:** training {t} and validation {v} are similarly high and hardly better than the naive "
        "prediction ({b}) – the model is too simple. Remedy: {fix}.",
        "**Niedouczenie:** trening {t} i walidacja {v} są podobnie wysokie i niewiele lepsze niż naiwna prognoza "
        "({b}) – model jest zbyt prosty. Środek zaradczy: {fix}."),
    " (Testdaten: {pct}).": (" (test data: {pct}).", " (dane testowe: {pct})."),
    "{name} hier {new:.3f} – vergleichbar mit dem Test ({old:.3f}): Das Modell arbeitet auf diesen Texten so gut "
    "wie erwartet.": (
        "{name} here {new:.3f} – comparable to the test ({old:.3f}): the model works on these texts as well as "
        "expected.",
        "{name} tutaj {new:.3f} – porównywalnie z testem ({old:.3f}): model działa na tych tekstach tak dobrze, "
        "jak oczekiwano."),
    "preprocessing.json ist unvollständig oder beschädigt (Eintrag: {part}) – das Modell bitte neu speichern "
    "(Schritt 9).": (
        "preprocessing.json is incomplete or damaged (entry: {part}) – please save the model again (step 9).",
        "preprocessing.json jest niekompletny lub uszkodzony (wpis: {part}) – zapisz model ponownie (krok 9)."),
    "Kein scikit-learn-Modell gefunden (model.skops oder model.joblib).": (
        "No scikit-learn model found (model.skops or model.joblib).",
        "Nie znaleziono modelu scikit-learn (model.skops lub model.joblib)."),
    "Sehr wenige Trainingstexte für {c} (jeweils weniger als {m}) – solche Klassen lernt das Modell kaum, ihre "
    "Kennzahlen sind Zufall. Tipp: diese Klassen in Schritt 4 abwählen oder mehr Texte sammeln.": (
        "Very few training texts for {c} (fewer than {m} each) – the model can hardly learn such classes, and "
        "their metrics are down to chance. Tip: deselect these classes in step 4 or collect more texts.",
        "Bardzo mało tekstów treningowych dla {c} (każda poniżej {m}) – takich klas model prawie się nie nauczy, "
        "a ich miary są dziełem przypadku. Wskazówka: odznacz te klasy w kroku 4 albo zbierz więcej tekstów."),
    " Hat eine Klasse nur einen Trainingstext, ist LinearSVC nicht möglich.": (
        " If a class has only one training text, LinearSVC is not possible.",
        " Jeśli klasa ma tylko jeden tekst treningowy, LinearSVC nie jest możliwy."),
    "LinearSVC braucht für die Wahrscheinlichkeiten (Kalibrierung mit Kreuzvalidierung) mindestens 2 "
    "Trainingstexte je Klasse – {c}: nur einer. Die Klasse in Schritt 4 abwählen oder Naive Bayes bzw. "
    "Logistische Regression wählen.": (
        "For its probabilities (calibration with cross-validation) LinearSVC needs at least 2 training texts per "
        "class – {c}: only one. Deselect the class in step 4 or choose Naive Bayes or logistic regression.",
        "Do prawdopodobieństw (kalibracja z walidacją krzyżową) LinearSVC potrzebuje co najmniej 2 tekstów "
        "treningowych na klasę – {c}: tylko jeden. Odznacz klasę w kroku 4 albo wybierz Naive Bayes lub regresję "
        "logistyczną."),
    "Validierungskurve nicht möglich: Im Trainingsteil bleibt nur eine Klasse übrig (zu wenige Texte für „Anteil "
    "aus Training“). In Schritt 6 die Testdaten zur Validierung wählen oder die Kurve ausschalten.": (
        "Validation curve not possible: only one class is left in the training part (too few texts for “share of "
        "training data”). In step 6 choose the test data for validation or switch the curve off.",
        "Krzywa walidacji niemożliwa: w części treningowej zostaje tylko jedna klasa (za mało tekstów dla "
        "„części danych treningowych”). W kroku 6 wybierz dane testowe do walidacji albo wyłącz krzywą."),
    "**Mit {param} = {chosen} nicht besser als eine naive Vorhersage:** Validierung {v} gegenüber {b}, wenn man "
    "immer die häufigste Klasse vorhersagt. Andere Werte der Kurve sind deutlich besser – der eingestellte Wert "
    "passt nicht zu diesen Daten (z. B. alpha zu groß bei TF-IDF).": (
        "**With {param} = {chosen} no better than a naive prediction:** validation {v} versus {b} when always "
        "predicting the most frequent class. Other values on the curve are clearly better – the chosen value "
        "does not suit this data (e.g. alpha too large with TF-IDF).",
        "**Przy {param} = {chosen} nie lepiej niż naiwna prognoza:** walidacja {v} wobec {b}, gdy zawsze "
        "przewiduje się najczęstszą klasę. Inne wartości krzywej są wyraźnie lepsze – ustawiona wartość nie "
        "pasuje do tych danych (np. alpha za duże przy TF-IDF)."),
    "preprocessing.json ist zu groß ({v:.0f} MB) – das ist kein Rezept dieser App.": (
        "preprocessing.json is too large ({v:.0f} MB) – this is not a recipe of this app.",
        "preprocessing.json jest za duży ({v:.0f} MB) – to nie jest przepis tej aplikacji."),
    "Vektorisierer und preprocessing.json passen nicht zusammen: laut Rezept {kind}, in der Datei {type}.": (
        "Vectorizer and preprocessing.json do not match: according to the recipe {kind}, in the file {type}.",
        "Wektoryzator i preprocessing.json nie pasują do siebie: według przepisu {kind}, w pliku {type}."),
}
