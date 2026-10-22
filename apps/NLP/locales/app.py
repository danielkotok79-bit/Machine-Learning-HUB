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
    "Rolle": ("Role", "Rola"),
    "Test": ("Test", "Test"),
    "Train": ("Train", "Trening"),
    "Anteil Train": ("Share train", "Udział trening"),
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
    "Stichprobe geändert – bitte Schritt 2 und 4 erneut übernehmen.": (
        "Sample changed – please apply step 2 and 4 again.", "Zmieniono próbkę – zastosuj ponownie krok 2 i 4."),
    "Lese CSV …": ("Reading CSV …", "Wczytywanie CSV …"),
    "Laden fehlgeschlagen: {exc}": ("Loading failed: {exc}", "Wczytanie nie powiodło się: {exc}"),
    "Bitte zuerst Daten laden (Schritt 1) – dann die Konfiguration importieren.": (
        "Please load data first (step 1) – then import the configuration.",
        "Najpierw wczytaj dane (krok 1) – potem zaimportuj konfigurację."),
    "Ungültige Konfiguration: {exc}": ("Invalid configuration: {exc}", "Nieprawidłowa konfiguracja: {exc}"),
    " Ungültige Werte ignoriert: {skipped}.": (" Invalid values ignored: {skipped}.",
                                               " Pominięto nieprawidłowe wartości: {skipped}."),
    "1 · Daten laden & Bibliotheken": ("1 · Load data & libraries", "1 · Wczytanie danych i biblioteki"),
    "1) Bibliotheken – werden beim Start automatisch importiert": (
        "1) Libraries – imported automatically at start-up", "1) Biblioteki – importowane automatycznie przy starcie"),
    "CSV-Datei laden": ("Load CSV file", "Wczytaj plik CSV"),
    "Daten laden": ("Load data", "Wczytaj dane"),
    "Fehler: {exc}": ("Error: {exc}", "Błąd: {exc}"),
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
    "aus Spalte": ("from column", "z kolumny"),
    "Zuordnung  alt=neu; …  (optional: sonst=…)": ("Mapping  old=new; …  (optional: rest=…)",
                                                   "Przypisanie  stara=nowa; …  (opcjonalnie: rest=…)"),
    "Werte in {src}: ": ("Values in {src}: ", "Wartości w {src}: "),
    "z. B. {example}": ("e.g. {example}", "np. {example}"),
    "Spalten: ": ("Columns: ", "Kolumny: "),
    " · Namen mit Leer-/Sonderzeichen in \\`Backticks\\`": (" · names with spaces/special characters in \\`backticks\\`",
                                                           " · nazwy ze spacjami/znakami specjalnymi w \\`odwrotnych "
                                                           "apostrofach\\`"),
    "Ergebnis: zwei Spalten <Name>_jahr und <Name>_monat.": ("Result: two columns <name>_jahr and <name>_monat.",
                                                            "Wynik: dwie kolumny <nazwa>_jahr i <nazwa>_monat."),
    "Spalte hinzufügen": ("Add column", "Dodaj kolumnę"),
    "Kopie von {source}": ("Copy of {source}", "Kopia kolumny {source}"),
    "aus {source}: ": ("from {source}: ", "z {source}: "),
    "Neue Spalten entfernen": ("Remove new columns", "Usuń nowe kolumny"),
    "Entfernen": ("Remove", "Usuń"),
    # Schritt 2
    "Bitte zuerst in Schritt 1 eine CSV laden.": ("Please load a CSV in step 1 first.", "Najpierw wczytaj plik CSV w kroku 1."),
    "Beispiel-Konfiguration laden": ("Load example configuration", "Wczytaj przykładową konfigurację"),
    "Beispiel-Konfiguration geladen (Schritt 2, 4 und 6) – jetzt unten **Übernehmen** klicken.": (
        "Example configuration loaded (step 2, 4 and 6) – now click **Apply** below.",
        "Wczytano przykładową konfigurację (krok 2, 4 i 6) – teraz kliknij poniżej **Zastosuj**."),
    "siehe 'Aktionen & Parameter erklärt'": ("see 'Actions & parameters explained'", "zobacz 'Akcje i parametry – opis'"),
    "Neuer Name": ("New name", "Nowa nazwa"),
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
    "Überblick": ("Overview", "Przegląd"),
    "Die neuen Daten enthalten schon Spalten mit den Namen der Ergebnisspalten – im Ergebnis werden sie ersetzt: "
    "{cols}": (
        "The new data already contain columns with the names of the result columns – they are replaced in the "
        "result: {cols}",
        "Nowe dane zawierają już kolumny o nazwach kolumn wyniku – w wyniku zostaną zastąpione: {cols}"),
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
    "Numerische Spalten": ("Numeric columns", "Kolumny liczbowe"),
    "Methode": ("Method", "Metoda"),
    "Werte anzeigen (annot)": ("Show values (annot)", "Pokaż wartości (annot)"),
    "nur untere Hälfte": ("lower half only", "tylko dolna połowa"),
    "Tabelle (df.corr)": ("Table (df.corr)", "Tabela (df.corr)"),
    "Mindestens 2 numerische Spalten wählen.": ("Choose at least 2 numeric columns.", "Wybierz co najmniej 2 kolumny liczbowe."),
    "Tabelle": ("Table", "Tabela"),
    "gruppieren nach": ("group by", "grupuj według"),
    # Schritt 4
    "Positive Klasse (= 1)": ("Positive class (= 1)", "Klasa pozytywna (= 1)"),
    "Verteilung von {label}": ("Distribution of {label}", "Rozkład {label}"),
    # Schritt 5
    "Daten nach Schritt 2 – alle Änderungen und neuen Spalten (lesbar, z. B. für Excel)": (
        "Data after step 2 – all changes and new columns (readable, e.g. for Excel)",
        "Dane po kroku 2 – wszystkie zmiany i nowe kolumny (czytelne, np. do Excela)"),
    "CSV international (Komma, Dezimalpunkt)": ("CSV international (comma, decimal point)",
                                                "CSV międzynarodowy (przecinek, kropka dziesiętna)"),
    "5 · Daten speichern": ("5 · Save data", "5 · Zapis danych"),
    "Die bearbeiteten Daten als CSV-Datei speichern – zum Weiterarbeiten in Excel oder einem anderen Programm.": (
        "Save the processed data as a CSV file – to keep working in Excel or another program.",
        "Zapisz przetworzone dane jako plik CSV – do dalszej pracy w Excelu albo innym programie."),
    "Was speichern?": ("What to save?", "Co zapisać?"),
    "Format": ("Format", "Format"),
    "Stichprobe aktiv: Es werden nur {n} von {n2} Zeilen gespeichert.": (
        "Sample active: only {n} of {n2} rows will be saved.",
        "Aktywna próbka: zapisana zostanie tylko próbka – {n} z {n2} wierszy."),
    "Label kodiert: ": ("Label encoded: ", "Zakodowana etykieta: "),
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
    "Validierung": ("Validation", "Walidacja"),
    "Anteil aus Training (sauberer)": ("Share of training data (cleaner)", "Część danych treningowych (czyściej)"),
    "Testdaten": ("Test data", "Dane testowe"),
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
    "Accuracy": ("Accuracy", "Dokładność (accuracy)"),
    "Mittel über alle Klassen – fair bei Ungleichgewicht": ("Average over all classes – fair for imbalanced classes",
                                                            "Średnia ze wszystkich klas – uczciwa przy niezrównoważeniu"),
    "0.5 = Zufall, 1.0 = perfekt; unabhängig von der Schwelle": ("0.5 = chance, 1.0 = perfect; independent of the threshold",
                                                                "0.5 = losowo, 1.0 = idealnie; niezależnie od progu"),
    "classification_report (Text)": ("classification_report (text)", "classification_report (tekst)"),
    ":green[:material/check_circle:] richtig": (":green[:material/check_circle:] correct",
                                                ":green[:material/check_circle:] trafnie"),
    ":red[:material/cancel:] falsch": (":red[:material/cancel:] wrong", ":red[:material/cancel:] błędnie"),
    "Entscheidungsschwelle: P({v}) > Schwelle → '{v}'": ("Decision threshold: P({v}) > threshold → '{v}'",
                                                         "Próg decyzyjny: P({v}) > próg → '{v}'"),
    "Standard 0.5. Höher = strenger für Klasse 1. Wird beim Speichern mitgespeichert.": (
        "Default 0.5. Higher = stricter for class 1. Saved together with the model.",
        "Domyślnie 0.5. Wyższy = surowiej dla klasy 1. Zapisywany razem z modelem."),
    "tatsächlich": ("actual", "rzeczywista"),
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
    "Modell + Rezept als ZIP": ("Model + recipe as ZIP", "Model + przepis jako ZIP"),
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
    "**Lesart:** `P({pos})` ist die vom Modell geschätzte Wahrscheinlichkeit für '{pos}'. Über der Schwelle "
    "{thr:.2f} → Vorhersage '{pos}', sonst '{neg}'. Werte nahe 0 oder 1 = sicher, um {thr:.2f} = unsicher.": (
        "**How to read it:** `P({pos})` is the probability of '{pos}' estimated by the model. Above the threshold "
        "{thr:.2f} → prediction '{pos}', otherwise '{neg}'. Values close to 0 or 1 = certain, around {thr:.2f} = "
        "uncertain.",
        "**Jak czytać:** `P({pos})` to oszacowane przez model prawdopodobieństwo '{pos}'. Powyżej progu {thr:.2f} → "
        "prognoza '{pos}', inaczej '{neg}'. Wartości bliskie 0 lub 1 = pewne, w okolicy {thr:.2f} = niepewne."),
    "**Lesart:** `Vorhersage` = Klasse mit der höchsten Wahrscheinlichkeit, `Sicherheit` = diese "
    "Wahrscheinlichkeit; `P(…)` je Klasse summieren sich zu 1.": (
        "**How to read it:** `Prediction` = class with the highest probability, `Confidence` = this probability; "
        "`P(…)` per class add up to 1.",
        "**Jak czytać:** `Prognoza` = klasa z najwyższym prawdopodobieństwem, `Pewność` = to prawdopodobieństwo; "
        "`P(…)` dla klas sumują się do 1."),
    "Fazit": ("Conclusion", "Podsumowanie"),
    "**Was das Ergebnis aussagt**": ("**What the result means**", "**Co oznacza wynik**"),
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
    "Eingabe": ("Input", "Dane wejściowe"),
    "Vorhersage berechnen": ("Compute predictions", "Oblicz prognozy"),
    "Ergebnis": ("Result", "Wynik"),
    "Ergebnis als CSV": ("Result as CSV", "Wynik jako CSV"),
    "vorhersagen_{stem}.csv": ("predictions_{stem}.csv", "prognozy_{stem}.csv"),
    "Python-Code (Vorhersage ohne App)": ("Python code (prediction without the app)", "Kod Pythona (prognoza bez aplikacji)"),
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
    "Warum?": (
        "Why?",
        "Dlaczego?"),
    "Am besten für": (
        "Best for",
        "Do czego najlepiej"),
    # Schritt 6 (ML-Version)
    "Fehlerrate": (
        "Error rate",
        "Odsetek błędów"),
    "Modell wählen": (
        "Choose a model",
        "Wybór modelu"),
    "Parameter: {name}": (
        "Parameters: {name}",
        "Parametry: {name}"),
    "Maximale Iterationen des Lösers – erhöhen, wenn eine ConvergenceWarning erscheint (Standard in "
    "scikit-learn: 100).": (
        "Maximum iterations of the solver – increase if a ConvergenceWarning appears (default in scikit-learn: "
        "100).",
        "Maksymalna liczba iteracji solvera – zwiększ, gdy pojawi się ConvergenceWarning (domyślnie w "
        "scikit-learn: 100)."),
    "Training & Validierung": (
        "Training & validation",
        "Trening i walidacja"),
    "Validierungskurve für Schritt 7": (
        "Validation curve for step 7",
        "Krzywa walidacji do kroku 7"),
    "Schritt 7 zeigt: {what}.": (
        "Step 7 shows: {what}.",
        "Krok 7 pokazuje: {what}."),
    "Ohne Validierungskurve vergleicht Schritt 7 nur Training und Test.": (
        "Without a validation curve, step 7 only compares training and test.",
        "Bez krzywej walidacji krok 7 porównuje tylko trening i test."),
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
    "Modell (scikit-learn)": (
        "Model (scikit-learn)",
        "Model (scikit-learn)"),
    # Schritt 7 (ML-Version)
    "7 · Auswertung Training vs. Validierung": (
        "7 · Evaluating training vs. validation",
        "7 · Analiza: trening a walidacja"),
    "Anteil aus Training": (
        "share of the training data",
        "część danych treningowych"),
    "Validierung ({where})": (
        "Validation ({where})",
        "Walidacja ({where})"),
    "Besten Wert in Schritt 6 übernehmen": (
        "Apply the best value in step 6",
        "Przejmij najlepszą wartość do kroku 6"),
    "Übernommen: {param} = {v} – jetzt in Schritt 6 neu trainieren.": (
        "Applied: {param} = {v} – now train again in step 6.",
        "Przejęto: {param} = {v} – teraz wytrenuj model ponownie w kroku 6."),
    "7 · Training vs. Validierung": (
        "7 · Training vs. validation",
        "7 · Trening a walidacja"),
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
    # --- NLP-App (Textklassifikation) ---
    "Wort": ("Word", "Słowo"),
    "Texte": ("Texts", "Teksty"),
    "IDF": ("IDF", "IDF"),
    "Gewicht": ("Weight", "Waga"),
    "Beitrag": ("Contribution", "Wkład"),
    "Zeile": ("Row", "Wiersz"),
    "Sekunden": ("Seconds", "Sekundy"),
    "Wörter": ("words", "słowa"),
    "richtig": ("correct", "poprawnie"),
    "automatisch erkennen": ("detect automatically", "wykryj automatycznie"),
    "ja": ("yes", "tak"),
    "nein – Spalten heißen Spalte_1, Spalte_2 …": (
        "no – columns are named Spalte_1, Spalte_2 …",
        "nie – kolumny nazywają się Spalte_1, Spalte_2 …"),
    "Häufigste Wörter": ("Most frequent words", "Najczęstsze słowa"),
    "Beispieltexte": ("Example texts", "Przykładowe teksty"),
    "Mittelwerte je Gruppe": ("Means per group", "Średnie w grupach"),
    "Texte nach Schritt 4 – Text, Label, Label als Zahl und Spalte 'split' (train/test)": (
        "Texts after step 4 – text, label, label as a number and column 'split' (train/test)",
        "Teksty po kroku 4 – tekst, etykieta, etykieta jako liczba i kolumna 'split' (train/test)"),
    "Vokabular nach Schritt 4 – jedes Wort mit Anzahl Texte (und IDF bei TF-IDF)": (
        "Vocabulary after step 4 – every word with its number of texts (and IDF with TF-IDF)",
        "Słownik po kroku 4 – każde słowo z liczbą tekstów (i IDF przy TF-IDF)"),
    "CSV für Excel (Format der gewählten Sprache)": (
        "CSV for Excel (format of the selected language)",
        "CSV dla Excela (format wybranego języka)"),
    "Fehlerrate je alpha (0.01 … 5) – wie stark geglättet wird": (
        "error rate per alpha (0.01 … 5) – how strongly the counts are smoothed",
        "odsetek błędów dla alpha (0.01 … 5) – jak silne jest wygładzanie"),
    "Fehlerrate je C (0.01 … 100) – wie stark regularisiert wird": (
        "error rate per C (0.01 … 100) – how strongly the model is regularized",
        "odsetek błędów dla C (0.01 … 100) – jak silna jest regularyzacja"),
    "Fehlerrate je C (0.001 … 10) – wie stark regularisiert wird": (
        "error rate per C (0.001 … 10) – how strongly the model is regularized",
        "odsetek błędów dla C (0.001 … 10) – jak silna jest regularyzacja"),
    "4 · Aufbereitung 2 – Text → Zahlen": (
        "4 · Preparation 2 – text → numbers",
        "4 · Przygotowanie 2 – tekst → liczby"),
    "8 · Evaluation & Wörter": ("8 · Evaluation & words", "8 · Ewaluacja i słowa"),
    "10 · Neue Texte einordnen": ("10 · Classify new texts", "10 · Klasyfikacja nowych tekstów"),
    "Übernommen – jetzt Schritt 2 'Übernehmen' und Schritt 4 'Texte vektorisieren' klicken.": (
        "Applied – now click 'Apply' in step 2 and 'Vectorize texts' in step 4.",
        "Przejęto – teraz kliknij 'Zastosuj' w kroku 2 i 'Wektoryzuj teksty' w kroku 4."),
    "Eine Tabelle mit einer Textspalte (Bewertung, Nachricht, Beschreibung …) und einer Spalte mit der Klasse "
    "(z. B. Sterne, spam/ham, Thema). Wie im Kurs: yelp.csv oder SMSSpamCollection (Tab, ohne Kopfzeile).": (
        "A table with a text column (review, message, description …) and a column with the class (e.g. stars, "
        "spam/ham, topic). As in the course: yelp.csv or SMSSpamCollection (tab, no header row).",
        "Tabela z kolumną tekstu (recenzja, wiadomość, opis …) i kolumną z klasą (np. gwiazdki, spam/ham, "
        "temat). Jak na kursie: yelp.csv albo SMSSpamCollection (tabulator, bez wiersza nagłówka)."),
    "Textspalten (Freitext)": ("Text columns (free text)", "Kolumny tekstowe (wolny tekst)"),
    "Yelp-Datensatz aus dem NLP-Projekt des Kurses erkannt: Die Beispiel-Konfiguration stellt Schritt 2, 4 und 6 "
    "wie im Kurs ein – IDs entfernen, neue Spalte text length, nur 1 und 5 Sterne, CountVectorizer und Naive "
    "Bayes (test_size 0.3, random_state 101).": (
        "Yelp dataset from the course's NLP project detected: the example configuration sets steps 2, 4 and 6 as "
        "in the course – drop IDs, new column text length, only 1 and 5 stars, CountVectorizer and Naive Bayes "
        "(test_size 0.3, random_state 101).",
        "Wykryto zbiór Yelp z projektu NLP kursu: przykładowa konfiguracja ustawia kroki 2, 4 i 6 jak na kursie "
        "– usuwa identyfikatory, dodaje kolumnę text length, tylko 1 i 5 gwiazdek, CountVectorizer i naiwny "
        "Bayes (test_size 0.3, random_state 101)."),
    "Zwei Spalten ohne Kopfzeile – wie SMSSpamCollection aus dem Kurs? Die Beispiel-Konfiguration benennt sie in "
    "label und message um, ergänzt die Länge und stellt TF-IDF + Naive Bayes ein (wie die Pipeline im Kurs).": (
        "Two columns without a header row – like SMSSpamCollection from the course? The example configuration "
        "renames them to label and message, adds the length and sets TF-IDF + Naive Bayes (like the pipeline in "
        "the course).",
        "Dwie kolumny bez wiersza nagłówka – jak SMSSpamCollection z kursu? Przykładowa konfiguracja zmienia ich "
        "nazwy na label i message, dodaje długość i ustawia TF-IDF + naiwny Bayes (jak potok na kursie)."),
    "2 · Datenaufbereitung 1 – Spalten umwandeln": (
        "2 · Data preparation 1 – transform columns",
        "2 · Przygotowanie danych 1 – przekształcanie kolumn"),
    "Eine Zeile je CSV-Spalte. **Aktion**, ggf. **Parameter** und **Neuer Name** wählen, dann **Übernehmen**. "
    "Die Vorschläge sind automatisch – bitte prüfen: Die **Textspalte** bleibt unverändert (*behalten*) – sie "
    "wird erst in Schritt 4 in Zahlen umgewandelt. IDs (z. B. review_id, user_id) entfernen; Spalten, die die "
    "Antwort schon verraten, sind ein Datenleck.": (
        "One row per CSV column. Choose **Action**, if needed **Parameter** and **New name**, then click "
        "**Apply**. The suggestions are automatic – please check them: the **text column** stays unchanged "
        "(*keep*) – it is only turned into numbers in step 4. Drop IDs (e.g. review_id, user_id); columns that "
        "already give away the answer are a data leak.",
        "Jeden wiersz na kolumnę CSV. Wybierz **Akcję**, w razie potrzeby **Parametr** i **Nową nazwę**, potem "
        "**Zastosuj**. Propozycje są automatyczne – sprawdź je: **kolumna tekstu** pozostaje bez zmian "
        "(*zachowaj*) – na liczby zamienia się ją dopiero w kroku 4. Usuń identyfikatory (np. review_id, "
        "user_id); kolumny, które zdradzają odpowiedź, to wyciek danych."),
    "4 · Datenaufbereitung 2 – Text, Label & Vektorisierung": (
        "4 · Data preparation 2 – text, label & vectorization",
        "4 · Przygotowanie danych 2 – tekst, etykieta i wektoryzacja"),
    "Text (X) & Label (y)": ("Text (X) & label (y)", "Tekst (X) i etykieta (y)"),
    "Train/Test-Split": ("Train/test split", "Podział train/test"),
    "Vektorisierung – Text → Zahlen": ("Vectorization – text → numbers", "Wektoryzacja – tekst → liczby"),
    "Texte vektorisieren": ("Vectorize texts", "Wektoryzuj teksty"),
    "Fertig: X_train {shape} · X_test {shape2} · Vokabular {v} Wörter · {pct} der Werte sind nicht 0 (sparse "
    "Matrix)": (
        "Done: X_train {shape} · X_test {shape2} · vocabulary {v} words · {pct} of the values are not 0 (sparse "
        "matrix)",
        "Gotowe: X_train {shape} · X_test {shape2} · słownik {v} słów · {pct} wartości jest różnych od 0 "
        "(macierz rzadka)"),
    "**Trainingstexte:** Ø {c:.0f} Zeichen · Ø {w:.0f} Wörter": (
        "**Training texts:** Ø {c:.0f} characters · Ø {w:.0f} words",
        "**Teksty treningowe:** Ø {c:.0f} znaków · Ø {w:.0f} słów"),
    "**Beispiel: ein Trainingstext als Zahlen**": (
        "**Example: a training text as numbers**",
        "**Przykład: tekst treningowy jako liczby**"),
    "**{v}** · {k} Klassen · {n_feat} Wörter im Vokabular · {n} Trainingstexte · {n2} Testtexte": (
        "**{v}** · {k} classes · {n_feat} words in the vocabulary · {n} training texts · {n2} test texts",
        "**{v}** · {k} klas · {n_feat} słów w słowniku · {n} tekstów treningowych · {n2} tekstów testowych"),
    "Vergleich: alle drei Modelle": ("Comparison: all three models", "Porównanie: wszystkie trzy modele"),
    "Trainiert Naive Bayes, Logistische Regression und LinearSVC mit den aktuellen Einstellungen (alpha, C, "
    "class_weight) auf denselben Trainingstexten und misst sie auf den Testdaten – Schwelle 0.5.": (
        "Trains Naive Bayes, logistic regression and LinearSVC with the current settings (alpha, C, "
        "class_weight) on the same training texts and measures them on the test data – threshold 0.5.",
        "Trenuje naiwny Bayes, regresję logistyczną i LinearSVC z bieżącymi ustawieniami (alpha, C, "
        "class_weight) na tych samych tekstach treningowych i mierzy je na danych testowych – próg 0.5."),
    "Modelle vergleichen": ("Compare models", "Porównaj modele"),
    "{name} · {metric} – kleiner = besser. Naive Vorhersage: man sagt immer die häufigste Klasse vorher.": (
        "{name} · {metric} – lower = better. Naive prediction: always predict the most frequent class.",
        "{name} · {metric} – mniej = lepiej. Naiwna prognoza: zawsze przewiduje się najczęstszą klasę."),
    "bekannte Wörter": ("known words", "znane słowa"),
    ":blue[**blau**] = spricht für '{c}', :orange[**orange**] = spricht dagegen (die wichtigsten Wörter dieses "
    "Textes).": (
        ":blue[**blue**] = speaks for '{c}', :orange[**orange**] = speaks against it (the most important words "
        "of this text).",
        ":blue[**niebieski**] = przemawia za '{c}', :orange[**pomarańczowy**] = przemawia przeciw (najważniejsze "
        "słowa tego tekstu)."),
    "8 · Modellevaluation & wichtige Wörter": (
        "8 · Model evaluation & important words",
        "8 · Ewaluacja modelu i ważne słowa"),
    "Bewertet auf {n} Testtexten (X_test, y_test) · {name}": (
        "Evaluated on {n} test texts (X_test, y_test) · {name}",
        "Oceniono na {n} tekstach testowych (X_test, y_test) · {name}"),
    "Testtexte mit Vorhersage als CSV": ("Test texts with prediction as CSV", "Teksty testowe z prognozą jako CSV"),
    "Falsch eingeordnete Texte": ("Misclassified texts", "Błędnie sklasyfikowane teksty"),
    "Die sichersten Fehler zuerst – oft Ironie, gemischte Bewertungen oder ein falsches Label in den Daten. Ein "
    "Blick darauf zeigt, wo das Modell an seine Grenzen kommt.": (
        "The most confident mistakes first – often irony, mixed reviews or a wrong label in the data. A look at "
        "them shows where the model reaches its limits.",
        "Najpewniejsze błędy najpierw – często ironia, mieszane recenzje albo błędna etykieta w danych. Rzut oka "
        "na nie pokazuje, gdzie model dochodzi do swoich granic."),
    "Einzelfall: zufälliger Testtext": (
        "Single case: random test text",
        "Pojedynczy przypadek: losowy tekst testowy"),
    "Zufälligen Testtext ziehen": ("Draw a random test text", "Wylosuj tekst testowy"),
    "Eigenen Text testen": ("Test your own text", "Przetestuj własny tekst"),
    "Beispiel aus den Testdaten einsetzen": (
        "Insert an example from the test data",
        "Wstaw przykład z danych testowych"),
    "Gespeichert werden **zwei Dateien** in `models/<name>/`: die scikit-learn-Pipeline aus Vektorisierer und "
    "Modell (`model.skops` oder `model.joblib`, wie im Kurs: Pipeline([('bow', …), ('classifier', …)])) und "
    "`preprocessing.json` – das Rezept aus Schritt 2 und 4 (Spalten-Aktionen, Textspalte, Klassen, Schwelle) mit "
    "den Einstellungen aus Schritt 6. Neue Texte werden damit genau so umgewandelt wie die Trainingstexte.": (
        "**Two files** are saved in `models/<name>/`: the scikit-learn pipeline of vectorizer and model "
        "(`model.skops` or `model.joblib`, as in the course: Pipeline([('bow', …), ('classifier', …)])) and "
        "`preprocessing.json` – the recipe from steps 2 and 4 (column actions, text column, classes, threshold) "
        "with the settings from step 6. New texts are then transformed exactly like the training texts.",
        "Zapisywane są **dwa pliki** w `models/<name>/`: potok scikit-learn z wektoryzatora i modelu "
        "(`model.skops` albo `model.joblib`, jak na kursie: Pipeline([('bow', …), ('classifier', …)])) oraz "
        "`preprocessing.json` – przepis z kroków 2 i 4 (akcje kolumn, kolumna tekstu, klasy, próg) z "
        "ustawieniami z kroku 6. Nowe teksty są nim przekształcane dokładnie tak jak teksty treningowe."),
    "\n\n**Ergebnis:** {n} von {n2} Texten ({v:.1%}) werden als '{pos}' eingeordnet, {n3} als '{neg}'.": (
        "\n\n**Result:** {n} of {n2} texts ({v:.1%}) are classified as '{pos}', {n3} as '{neg}'.",
        "\n\n**Wynik:** {n} z {n2} tekstów ({v:.1%}) sklasyfikowano jako '{pos}', {n3} jako '{neg}'."),
    "scikit-learn rechnet auf der CPU – {n} Kerne. Naive Bayes, Logistische Regression und LinearSVC sind auch "
    "bei vielen Texten schnell.": (
        "scikit-learn computes on the CPU – {n} cores. Naive Bayes, logistic regression and LinearSVC are fast "
        "even with many texts.",
        "scikit-learn liczy na CPU – {n} rdzeni. Naiwny Bayes, regresja logistyczna i LinearSVC są szybkie nawet "
        "przy wielu tekstach."),
    "Die Datei hat keine Kopfzeile – die Spalten heißen {cols}. In Schritt 2 unter **Neuer Name** umbenennen "
    "(wie im Kurs: label, message).": (
        "The file has no header row – the columns are named {cols}. Rename them in step 2 under **New name** (as "
        "in the course: label, message).",
        "Plik nie ma wiersza nagłówka – kolumny nazywają się {cols}. Zmień ich nazwy w kroku 2 w polu **Nowa "
        "nazwa** (jak na kursie: label, message)."),
    "Keine Spalte mit Freitext gefunden (im Mittel mindestens 3 Wörter je Wert) – für die Textklassifikation "
    "braucht es eine Spalte mit Texten.": (
        "No column with free text found (at least 3 words per value on average) – text classification needs a "
        "column with texts.",
        "Nie znaleziono kolumny z wolnym tekstem (średnio co najmniej 3 słowa na wartość) – klasyfikacja tekstu "
        "wymaga kolumny z tekstami."),
    "Keine Textspalte in den Daten – für die Textklassifikation braucht es eine Spalte mit Texten (in Schritt 2 "
    "behalten bzw. 'als Kategorie (Text)').": (
        "No text column in the data – text classification needs a column with texts (in step 2 keep it or use "
        "'As category (text)').",
        "Brak kolumny tekstowej w danych – klasyfikacja tekstu wymaga kolumny z tekstami (w kroku 2 zachowaj ją "
        "albo użyj 'Jako kategoria (tekst)')."),
    "Textspalte – was gelesen wird": ("Text column – what is read", "Kolumna tekstu – co jest czytane"),
    "Label – was vorhergesagt werden soll": (
        "Label – what should be predicted",
        "Etykieta – co ma być przewidywane"),
    "Es gibt keine weitere Spalte für das Label.": (
        "There is no other column for the label.",
        "Nie ma innej kolumny na etykietę."),
    "Zeilen ohne Text": ("Rows without text", "Wiersze bez tekstu"),
    "Bitte mindestens 2 Klassen auswählen – oder keine (dann alle).": (
        "Please select at least 2 classes – or none (then all).",
        "Wybierz co najmniej 2 klasy – albo żadnej (wtedy wszystkie)."),
    "{n} Klassen – höchstens {m} sind möglich. Bitte einige Klassen auswählen.": (
        "{n} classes – at most {m} are possible. Please select some classes.",
        "{n} klas – możliwe jest najwyżej {m}. Wybierz niektóre klasy."),
    "Textspalte und Label stammen aus derselben CSV-Spalte → Datenleck!": (
        "Text column and label come from the same CSV column → data leakage!",
        "Kolumna tekstu i etykieta pochodzą z tej samej kolumny CSV → wyciek danych!"),
    "So wird Text zu Zahlen": ("How text becomes numbers", "Jak tekst zamienia się w liczby"),
    "Vektorisierer": ("Vectorizer", "Wektoryzator"),
    "Stoppwörter": ("Stop words", "Stop words"),
    "Kleinschreibung (lowercase)": ("Lower case (lowercase)", "Małe litery (lowercase)"),
    "Akzente entfernen": ("Remove accents", "Usuń akcenty"),
    "Neu vektorisieren verwirft das trainierte Modell – vorher in Schritt 9 speichern.": (
        "Vectorizing again discards the trained model – save it in step 9 first.",
        "Ponowna wektoryzacja odrzuca wytrenowany model – najpierw zapisz go w kroku 9."),
    "Die Einstellungen oben wurden seit dem Vektorisieren geändert → erneut **Texte vektorisieren**.": (
        "The settings above have changed since vectorizing → click **Vectorize texts** again.",
        "Ustawienia powyżej zmieniły się od wektoryzacji → kliknij ponownie **Wektoryzuj teksty**."),
    "**Testtexte:** {pct} ihrer Wörter stehen im Vokabular – der Rest war im Training nie zu sehen und wird "
    "ignoriert.": (
        "**Test texts:** {pct} of their words are in the vocabulary – the rest never appeared in training and is "
        "ignored.",
        "**Teksty testowe:** {pct} ich słów jest w słowniku – reszty nigdy nie było w treningu i jest pomijana."),
    "Text {i} von {n} · Klasse '{c}'": ("Text {i} of {n} · class '{c}'", "Tekst {i} z {n} · klasa '{c}'"),
    "Anderer Text": ("Another text", "Inny tekst"),
    "{n} von {v} Spalten sind hier nicht 0 – wie im Kurs: bow_transformer.transform([text])": (
        "{n} of {v} columns are not 0 here – as in the course: bow_transformer.transform([text])",
        "{n} z {v} kolumn jest tu różnych od 0 – jak na kursie: bow_transformer.transform([text])"),
    "Vokabular (häufigste Wörter in den Trainingstexten)": (
        "Vocabulary (most frequent words in the training texts)",
        "Słownik (najczęstsze słowa w tekstach treningowych)"),
    "{n} Wörter – die ersten 500 nach Anzahl Texte. Komplett als CSV in Schritt 5.": (
        "{n} words – the first 500 by number of texts. Complete as CSV in step 5.",
        "{n} słów – pierwsze 500 według liczby tekstów. W całości jako CSV w kroku 5."),
    "Dafür zuerst Schritt 4 ausführen (**Texte vektorisieren**).": (
        "For this, run step 4 first (**Vectorize texts**).",
        "Najpierw wykonaj krok 4 (**Wektoryzuj teksty**)."),
    "Bitte zuerst Schritt 4 ausführen (Texte vektorisieren).": (
        "Please run step 4 first (vectorize texts).",
        "Najpierw wykonaj krok 4 (wektoryzacja tekstów)."),
    "Die drei Modelle als Tabelle": ("The three models as a table", "Trzy modele w tabeli"),
    "Sehr großes Vokabular ({v} Wörter): Logistische Regression und LinearSVC brauchen länger. Tipp: in Schritt "
    "4 min_df erhöhen oder max_features begrenzen.": (
        "Very large vocabulary ({v} words): logistic regression and LinearSVC take longer. Tip: raise min_df or "
        "limit max_features in step 4.",
        "Bardzo duży słownik ({v} słów): regresja logistyczna i LinearSVC potrzebują więcej czasu. Wskazówka: w "
        "kroku 4 podnieś min_df albo ogranicz max_features."),
    "Höchste Accuracy: **{m}**. Mit den Einstellungen alpha = {a:g}, C = {c:g} (Logistische Regression), C = "
    "{s:g} (LinearSVC){cw}.": (
        "Highest accuracy: **{m}**. With the settings alpha = {a:g}, C = {c:g} (logistic regression), C = {s:g} "
        "(LinearSVC){cw}.",
        "Najwyższa accuracy: **{m}**. Przy ustawieniach alpha = {a:g}, C = {c:g} (regresja logistyczna), C = "
        "{s:g} (LinearSVC){cw}."),
    "Python-Code zum Vergleich": ("Python code for the comparison", "Kod Python do porównania"),
    "Die Validierungskurve ist in Schritt 6 ausgeschaltet – mit Kurve zeigt diese Seite den Verlauf über alpha "
    "bzw. C.": (
        "The validation curve is turned off in step 6 – with the curve this page shows the course over alpha or "
        "C.",
        "Krzywa walidacji jest wyłączona w kroku 6 – z krzywą ta strona pokazuje przebieg dla alpha lub C."),
    "Anteil richtig eingeordneter Texte – naive Vorhersage (immer die häufigste Klasse der Trainingsdaten): "
    "{b:.3f}": (
        "Share of correctly classified texts – naive prediction (always the most frequent class of the training "
        "data): {b:.3f}",
        "Odsetek poprawnie sklasyfikowanych tekstów – naiwna prognoza (zawsze najczęstsza klasa danych "
        "treningowych): {b:.3f}"),
    "Wörter (nach Kleinschreibung und Stoppwörtern), die im Vokabular stehen – nur sie zählen.": (
        "Words (after lower-casing and stop words) that are in the vocabulary – only they count.",
        "Słowa (po zamianie na małe litery i usunięciu stop words), które są w słowniku – tylko one się liczą."),
    "**Schwelle verstehen:** Das Modell liefert für jeden Text eine Wahrscheinlichkeit für '{v}'. Liegt sie über "
    "der Schwelle, lautet die Vorhersage '{v}'. Eine höhere Schwelle macht '{v}' seltener (weniger Fehlalarme, "
    "aber mehr übersehene '{v}'); eine niedrigere erkennt mehr '{v}'.": (
        "**Understanding the threshold:** the model delivers a probability of '{v}' for every text. If it is "
        "above the threshold, the prediction is '{v}'. A higher threshold makes '{v}' rarer (fewer false alarms, "
        "but more missed '{v}'); a lower one detects more '{v}'.",
        "**Jak działa próg:** model podaje dla każdego tekstu prawdopodobieństwo '{v}'. Jeśli jest powyżej "
        "progu, prognoza to '{v}'. Wyższy próg sprawia, że '{v}' pojawia się rzadziej (mniej fałszywych alarmów, "
        "ale więcej przeoczonych '{v}'); niższy wykrywa więcej '{v}'."),
    "Naive Bayes: wie viel wahrscheinlicher ein Wort in dieser Klasse ist als in den anderen (log P(Wort | "
    "Klasse) minus Mittel der anderen). Je größer, desto typischer.": (
        "Naive Bayes: how much more likely a word is in this class than in the others (log P(word | class) minus "
        "the mean of the others). The larger, the more typical.",
        "Naiwny Bayes: o ile bardziej prawdopodobne jest słowo w tej klasie niż w pozostałych (log P(słowo | "
        "klasa) minus średnia pozostałych). Im większe, tym bardziej typowe."),
    "Koeffizient des Worts für die Klasse – positiv: das Wort macht die Klasse wahrscheinlicher. Ein Wort zählt "
    "so oft, wie es im Text vorkommt (bzw. mit seinem TF-IDF-Wert).": (
        "Coefficient of the word for the class – positive: the word makes the class more likely. A word counts "
        "as often as it occurs in the text (or with its TF-IDF value).",
        "Współczynnik słowa dla klasy – dodatni: słowo czyni klasę bardziej prawdopodobną. Słowo liczy się tyle "
        "razy, ile występuje w tekście (albo z jego wartością TF-IDF)."),
    "Kein Testtext falsch eingeordnet.": (
        "No test text misclassified.",
        "Żaden tekst testowy nie został źle sklasyfikowany."),
    "Einordnen": ("Classify", "Klasyfikuj"),
    "Prüfsummen von Trainingstexten mitspeichern (Schritt 10 erkennt damit Trainingsdaten)": (
        "Also save checksums of training texts (step 10 uses them to recognize training data)",
        "Zapisz też sumy kontrolne tekstów treningowych (krok 10 rozpoznaje dzięki nim dane treningowe)"),
    "Hinweis: Das Vokabular (alle Wörter der Trainingstexte) steckt im Modell – bei vertraulichen Texten das "
    "Modell nur an Berechtigte weitergeben.": (
        "Note: the vocabulary (all words of the training texts) is part of the model – with confidential texts, "
        "share the model only with authorized people.",
        "Uwaga: słownik (wszystkie słowa tekstów treningowych) jest częścią modelu – przy poufnych tekstach "
        "udostępniaj model tylko uprawnionym osobom."),
    "**Geladen: {name}** · {model} · Text `{text}` → Label `{label}` · {v} · Klassen: {classes} · gespeichert "
    "{v2}": (
        "**Loaded: {name}** · {model} · text `{text}` → label `{label}` · {v} · classes: {classes} · saved {v2}",
        "**Wczytano: {name}** · {model} · tekst `{text}` → etykieta `{label}` · {v} · klasy: {classes} · "
        "zapisano {v2}"),
    "**Vektorisierer**": ("**Vectorizer**", "**Wektoryzator**"),
    " Zum Vergleich: In den Trainingsdaten waren {train_share:.1%} '{pos}'. Einordnung im Fazit unten.": (
        "For comparison: {train_share:.1%} of the training data were '{pos}'. Assessment in the conclusion "
        "below.",
        "Dla porównania: w danych treningowych {train_share:.1%} stanowiło '{pos}'. Ocena w podsumowaniu "
        "poniżej."),
    "**Qualität der Texte: {v} {level}**": (
        "**Quality of the texts: {v} {level}**",
        "**Jakość tekstów: {v} {level}**"),
    "Schwelle: P({v}) > Schwelle → '{v}'": ("Threshold: P({v}) > threshold → '{v}'", "Próg: P({v}) > próg → '{v}'"),
    "Einzelner Text": ("Single text", "Pojedynczy tekst"),
    "vorlage_neue_texte.csv": ("template_new_texts.csv", "szablon_nowe_teksty.csv"),
    "Kopfzeile": ("Header row", "Wiersz nagłówka"),
    "z. B. C:\\Daten\\yelp.csv oder ../daten/SMSSpamCollection": (
        "e.g. C:\\Data\\yelp.csv or ../data/SMSSpamCollection",
        "np. C:\\Dane\\yelp.csv albo ../dane/SMSSpamCollection"),
    "Freitext erkannt: ": ("Free text detected: ", "Wykryto wolny tekst: "),
    "**Schnell: Textlänge als neue Spalte** – wie im Kurs: `yelp['text length'] = yelp['text'].apply(len)`. Gut "
    "für Schritt 3 (Länge je Klasse).": (
        "**Quick: text length as a new column** – as in the course: `yelp['text length'] = "
        "yelp['text'].apply(len)`. Useful for step 3 (length per class).",
        "**Szybko: długość tekstu jako nowa kolumna** – jak na kursie: `yelp['text length'] = "
        "yelp['text'].apply(len)`. Przydatne w kroku 3 (długość według klasy)."),
    "optional umbenennen, z. B. message": (
        "optionally rename, e.g. message",
        "opcjonalnie zmień nazwę, np. message"),
    "Die Spalte mit den Texten, z. B. text (Yelp) oder message (SMS).": (
        "The column with the texts, e.g. text (Yelp) or message (SMS).",
        "Kolumna z tekstami, np. text (Yelp) albo message (SMS)."),
    "Klassen (leer = alle)": ("Classes (empty = all)", "Klasy (puste = wszystkie)"),
    "Das Label hat {n} verschiedene Werte – das ist kein Label (eher eine ID oder ein Text).": (
        "The label has {n} different values – that is not a label (rather an ID or a text).",
        "Etykieta ma {n} różnych wartości – to nie jest etykieta (raczej identyfikator albo tekst)."),
    "Leere oder fehlende Texte enthalten kein Wort – entfernen ist meist richtig.": (
        "Empty or missing texts contain no words – removing them is usually right.",
        "Puste albo brakujące teksty nie zawierają słów – usunięcie ich jest zwykle właściwe."),
    "Anteil der Texte für den Test – im Kurs 0.3 (Yelp) bzw. 0.2 (SMS).": (
        "Share of the texts for the test – in the course 0.3 (Yelp) or 0.2 (SMS).",
        "Udział tekstów do testu – na kursie 0.3 (Yelp) albo 0.2 (SMS)."),
    "Klassenverhältnis in Train und Test gleich halten (empfohlen, z. B. nur 13 % spam)": (
        "Keep the class ratio equal in train and test (recommended, e.g. only 13 % spam)",
        "Zachowaj ten sam stosunek klas w train i test (zalecane, np. tylko 13 % spamu)"),
    "Sehr häufige Wörter ohne Inhalt (the, and, is …) entfernen – die Liste von scikit-learn ist englisch. Im "
    "Kurs: stopwords.words('english') aus nltk.": (
        "Remove very frequent words without content (the, and, is …) – the scikit-learn list is English. In the "
        "course: stopwords.words('english') from nltk.",
        "Usuń bardzo częste słowa bez treści (the, and, is …) – lista scikit-learn jest angielska. Na kursie: "
        "stopwords.words('english') z nltk."),
    "(1, 2) = auch Wortpaare wie 'not good' – erkennt Verneinungen, aber das Vokabular wird viel größer.": (
        "(1, 2) = also word pairs such as 'not good' – detects negations, but the vocabulary becomes much larger.",
        "(1, 2) = także pary słów jak 'not good' – wykrywa zaprzeczenia, ale słownik staje się dużo większy."),
    "café = cafe, Müller = Muller (strip_accents)": (
        "café = cafe, Müller = Muller (strip_accents)",
        "café = cafe, Müller = Muller (strip_accents)"),
    "Wort muss in mindestens so vielen Trainingstexten vorkommen – 2 oder 5 entfernt Tippfehler und Einzelfälle, "
    "das Vokabular wird kleiner.": (
        "A word must occur in at least this many training texts – 2 or 5 removes typos and one-offs, the "
        "vocabulary gets smaller.",
        "Słowo musi występować w co najmniej tylu tekstach treningowych – 2 albo 5 usuwa literówki i pojedyncze "
        "przypadki, słownik się zmniejsza."),
    "Wörter, die in mehr als diesem Anteil der Texte vorkommen, weglassen (z. B. 0.9) – eine Art automatischer "
    "Stoppwörter.": (
        "Leave out words that occur in more than this share of the texts (e.g. 0.9) – a kind of automatic stop "
        "words.",
        "Pomiń słowa występujące w więcej niż tym odsetku tekstów (np. 0.9) – rodzaj automatycznych stop words."),
    "Nur die häufigsten N Wörter behalten – 0 = alle.": (
        "Keep only the N most frequent words – 0 = all.",
        "Zachowaj tylko N najczęstszych słów – 0 = wszystkie."),
    "Nur vorhanden (1) oder nicht (0) statt Anzahl": (
        "Only present (1) or not (0) instead of the count",
        "Tylko występuje (1) albo nie (0) zamiast liczby"),
    "Nur TF-IDF: 1 + log(Anzahl) statt Anzahl – ein Wort, das 10-mal vorkommt, zählt nicht 10-mal so viel.": (
        "TF-IDF only: 1 + log(count) instead of the count – a word that occurs 10 times does not count 10 times "
        "as much.",
        "Tylko TF-IDF: 1 + log(liczba) zamiast liczby – słowo występujące 10 razy nie liczy się 10 razy więcej."),
    "Texte vektorisiert": ("Texts vectorized", "Teksty zwektoryzowane"),
    "Klassenanteile als Vorwissen (fit_prior)": (
        "Class shares as prior knowledge (fit_prior)",
        "Udziały klas jako wiedza a priori (fit_prior)"),
    "LinearSVC liefert keine Wahrscheinlichkeiten – CalibratedClassifierCV ergänzt sie (für Schwelle und "
    "ROC-AUC), das Training dauert dadurch etwas länger.": (
        "LinearSVC provides no probabilities – CalibratedClassifierCV adds them (for the threshold and ROC-AUC), "
        "which makes training take a little longer.",
        "LinearSVC nie daje prawdopodobieństw – CalibratedClassifierCV je dodaje (dla progu i ROC-AUC), przez co "
        "trening trwa nieco dłużej."),
    "Trainiert das Modell zusätzlich für mehrere Werte von alpha bzw. C und misst die Fehlerrate auf Trainings- "
    "und Validierungsdaten – dauert etwas länger.": (
        "Additionally trains the model for several values of alpha or C and measures the error rate on training "
        "and validation data – takes a little longer.",
        "Dodatkowo trenuje model dla kilku wartości alpha lub C i mierzy odsetek błędów na danych treningowych i "
        "walidacyjnych – trwa nieco dłużej."),
    "Testdaten: Die Kurve wird auf X_test berechnet (wie die Elbow-Methode im Kurs) – einfach, aber wer danach "
    "den besten Wert wählt, hat die Testdaten mitbenutzt. Sauberer: eigener Validierungsanteil aus den "
    "Trainingsdaten (mit eigenem Vokabular), damit das Testset bis Schritt 8 unberührt bleibt.": (
        "Test data: the curve is computed on X_test (like the elbow method in the course) – simple, but choosing "
        "the best value afterwards means the test data has been used too. Cleaner: a separate validation share "
        "of the training data (with its own vocabulary), so the test set stays untouched until step 8.",
        "Dane testowe: krzywa jest liczona na X_test (jak metoda łokcia na kursie) – prosto, ale kto potem "
        "wybiera najlepszą wartość, użył też danych testowych. Czyściej: osobna część walidacyjna z danych "
        "treningowych (z własnym słownikiem), aby zbiór testowy pozostał nietknięty aż do kroku 8."),
    "class_weight='balanced' – hilft der seltenen Klasse (z. B. spam). Naive Bayes kennt das nicht – dort "
    "fit_prior ausschalten.": (
        "class_weight='balanced' – helps the rare class (e.g. spam). Naive Bayes does not have it – turn off "
        "fit_prior there.",
        "class_weight='balanced' – pomaga rzadkiej klasie (np. spam). Naiwny Bayes tego nie ma – tam wyłącz "
        "fit_prior."),
    "Das Modell ist nicht vollständig konvergiert (ConvergenceWarning) – max_iter erhöhen oder C verkleinern.": (
        "The model did not fully converge (ConvergenceWarning) – increase max_iter or decrease C.",
        "Model nie osiągnął pełnej zbieżności (ConvergenceWarning) – zwiększ max_iter albo zmniejsz C."),
    "Vergleiche …": ("Comparing …", "Porównuję …"),
    "z. B. The food was great, but the service was terribly slow.": (
        "e.g. The food was great, but the service was terribly slow.",
        "np. The food was great, but the service was terribly slow."),
    "Bis zu 5000 Prüfsummen (Hashes) von Trainingstexten. Aus einer Prüfsumme lässt sich der Text nicht "
    "zurückrechnen – wer einen Text hat, kann aber prüfen, ob er im Training war. Bei vertraulichen Texten "
    "ausschalten, bevor das Modell weitergegeben wird.": (
        "Up to 5000 checksums (hashes) of training texts. A text cannot be recovered from a checksum – but "
        "anyone who has a text can check whether it was in training. Turn this off for confidential texts before "
        "sharing the model.",
        "Do 5000 sum kontrolnych (hashy) tekstów treningowych. Z sumy kontrolnej nie da się odtworzyć tekstu – "
        "ale kto ma tekst, może sprawdzić, czy był w treningu. Przy poufnych tekstach wyłącz to przed "
        "udostępnieniem modelu."),
    "Bitte einen Text eingeben.": ("Please enter a text.", "Wpisz tekst."),
    "Das Modell wurde mit einer Datei ohne Kopfzeile trainiert – für neue Daten dieselbe Form verwenden "
    "(Kopfzeile: nein) oder eine CSV mit den Spaltennamen aus der Tabelle oben.": (
        "The model was trained with a file without a header row – use the same form for new data (header row: "
        "no) or a CSV with the column names from the table above.",
        "Model był trenowany na pliku bez wiersza nagłówka – dla nowych danych użyj tej samej formy (wiersz "
        "nagłówka: nie) albo pliku CSV z nazwami kolumn z tabeli powyżej."),
    "{n} Texte eingeordnet ({name})": ("{n} texts classified ({name})", "Sklasyfikowano {n} tekstów ({name})"),
    "Texte vektorisiert: X_train {shape}": (
        "Texts vectorized: X_train {shape}",
        "Teksty zwektoryzowane: X_train {shape}"),
    "Modell trainiert: {name}": ("Model trained: {name}", "Model wytrenowany: {name}"),
    "Hat die erste Zeile die Spaltennamen? SMSSpamCollection aus dem Kurs hat keine – dort heißen die Spalten "
    "dann Spalte_1, Spalte_2 (in Schritt 2 umbenennen, z. B. label und message).": (
        "Does the first row contain the column names? SMSSpamCollection from the course has none – there the "
        "columns are then called Spalte_1, Spalte_2 (rename them in step 2, e.g. label and message).",
        "Czy pierwszy wiersz zawiera nazwy kolumn? SMSSpamCollection z kursu ich nie ma – kolumny nazywają się "
        "wtedy Spalte_1, Spalte_2 (zmień nazwy w kroku 2, np. na label i message)."),
    "z. B. sentiment": ("e.g. sentiment", "np. sentiment"),
    "z. B. 1=negativ; 2=negativ; 4=positiv; 5=positiv": (
        "e.g. 1=negative; 2=negative; 4=positive; 5=positive",
        "np. 1=negatywna; 2=negatywna; 4=pozytywna; 5=pozytywna"),
    "Nur einige Klassen verwenden – wie im Kurs nur 1 und 5 Sterne (deutlich negativ / positiv). Zeilen mit "
    "anderen Klassen werden entfernt.": (
        "Use only some classes – as in the course only 1 and 5 stars (clearly negative / positive). Rows with "
        "other classes are removed.",
        "Użyj tylko niektórych klas – jak na kursie tylko 1 i 5 gwiazdek (wyraźnie negatywne / pozytywne). "
        "Wiersze z innymi klasami są usuwane."),
    "Diese Klasse wird zu 1 kodiert, die andere zu 0 (z. B. spam → 1, ham → 0). Die Schwelle in Schritt 8 "
    "bezieht sich auf sie.": (
        "This class is encoded as 1, the other as 0 (e.g. spam → 1, ham → 0). The threshold in step 8 refers to "
        "it.",
        "Ta klasa jest kodowana jako 1, druga jako 0 (np. spam → 1, ham → 0). Próg w kroku 8 odnosi się do niej."),
    "verwendet": ("used", "używana"),
    "{n} Zeilen ohne Label werden entfernt.": (
        "{n} rows without a label are removed.",
        "{n} wierszy bez etykiety zostanie usuniętych."),
    "Teile auf und lerne das Vokabular …": (
        "Splitting and learning the vocabulary …",
        "Dzielę i uczę się słownika …"),
    "Glättung (Laplace): auch Wörter, die in einer Klasse nie vorkamen, bekommen eine kleine Wahrscheinlichkeit. "
    "Klein = das Modell vertraut einzelnen Wörtern mehr (Gefahr von Overfitting), groß = glatter. Standard 1.0.": (
        "Smoothing (Laplace): even words that never occurred in a class get a small probability. Small = the "
        "model trusts single words more (risk of overfitting), large = smoother. Default 1.0.",
        "Wygładzanie (Laplace'a): nawet słowa, które nigdy nie wystąpiły w danej klasie, dostają małe "
        "prawdopodobieństwo. Małe = model bardziej ufa pojedynczym słowom (ryzyko przeuczenia), duże = gładziej. "
        "Domyślnie 1.0."),
    "Ein: die Häufigkeit der Klassen im Training zählt mit (Standard). Aus: alle Klassen gleich wahrscheinlich – "
    "hilft manchmal der seltenen Klasse.": (
        "On: the frequency of the classes in training counts too (default). Off: all classes equally likely – "
        "sometimes helps the rare class.",
        "Włączone: liczy się też częstość klas w treningu (domyślnie). Wyłączone: wszystkie klasy jednakowo "
        "prawdopodobne – czasem pomaga rzadkiej klasie."),
    "Vergleich fehlgeschlagen: {exc}": ("Comparison failed: {exc}", "Porównanie nie powiodło się: {exc}"),
    "Die Kurve nutzt die reine LinearSVC (ohne Kalibrierung, schneller) – die Kennzahlen in Schritt 8 können "
    "leicht abweichen.": (
        "The curve uses the plain LinearSVC (without calibration, faster) – the metrics in step 8 may differ "
        "slightly.",
        "Krzywa używa samej LinearSVC (bez kalibracji, szybciej) – wskaźniki w kroku 8 mogą się nieco różnić."),
    "Validierung: {n} der {n2} Trainingstexte; das Vokabular dafür wurde nur aus den übrigen {n3} Texten gelernt.": (
        "Validation: {n} of the {n2} training texts; the vocabulary for it was learned only from the other {n3} "
        "texts.",
        "Walidacja: {n} z {n2} tekstów treningowych; słownik do niej nauczono tylko z pozostałych {n3} tekstów."),
    "Neuen Text eingeben, z. B. eine Bewertung oder eine Nachricht …": (
        "Enter a new text, e.g. a review or a message …",
        "Wpisz nowy tekst, np. recenzję albo wiadomość …"),
    "{v} Hinweise zur Qualität der Texte und Einordnung → **Fazit** unten": (
        "{v} Notes on the quality of the texts and assessment → **Conclusion** below",
        "{v} Uwagi o jakości tekstów i ocena → **Podsumowanie** poniżej"),
    "Label '{label}' ist enthalten → Evaluation auf {n} Texten": (
        "Label '{label}' is included → evaluation on {n} texts",
        "Etykieta '{label}' jest obecna → ewaluacja na {n} tekstach"),
    "{c} (Ø {n:.0f} Wörter)": ("{c} (Ø {n:.0f} words)", "{c} (Ø {n:.0f} słów)"),
    "z. B. text length": ("e.g. text length", "np. text length"),
    "Wie lang sind die Texte je Klasse? Wie im Kurs: Länge berechnen und je Klasse ein Histogramm "
    "(sns.FacetGrid) – unterscheiden sich die Klassen schon in der Länge?": (
        "How long are the texts per class? As in the course: compute the length and draw one histogram per class "
        "(sns.FacetGrid) – do the classes already differ in length?",
        "Jak długie są teksty w każdej klasie? Jak na kursie: oblicz długość i narysuj histogram dla każdej "
        "klasy (sns.FacetGrid) – czy klasy różnią się już długością?"),
    "Kehrwert der Regularisierung: klein = stärkere Regularisierung (einfacheres Modell), groß = schwächere. Bei "
    "Texten oft 1–10.": (
        "Inverse of the regularization: small = stronger regularization (simpler model), large = weaker. For "
        "text often 1–10.",
        "Odwrotność regularyzacji: małe = silniejsza regularyzacja (prostszy model), duże = słabsza. Dla tekstów "
        "często 1–10."),
    "Strafe für Fehler im Training: groß = genaue Trennung (Gefahr von Overfitting), klein = breiterer Rand, "
    "einfacheres Modell. Bei Texten oft 0.1–1.": (
        "Penalty for errors in training: large = exact separation (risk of overfitting), small = wider margin, "
        "simpler model. For text often 0.1–1.",
        "Kara za błędy w treningu: duże = dokładne rozdzielenie (ryzyko przeuczenia), małe = szerszy margines, "
        "prostszy model. Dla tekstów często 0.1–1."),
    "Kein Wort dieses Textes steht im Vokabular des Modells – die Vorhersage folgt nur der Häufigkeit der "
    "Klassen im Training (andere Sprache?).": (
        "No word of this text is in the model's vocabulary – the prediction follows only the class frequencies "
        "in training (another language?).",
        "Żadne słowo tego tekstu nie występuje w słowniku modelu – prognoza wynika tylko z częstości klas w "
        "treningu (inny język?)."),
    "Wandle die neuen Texte um und ordne sie ein …": (
        "Transforming and classifying the new texts …",
        "Przekształcam i klasyfikuję nowe teksty …"),
    "{exc}. Die CSV braucht dieselbe Textspalte wie die Trainingsdaten (siehe Tabelle oben / CSV-Vorlage).": (
        "{exc}. The CSV needs the same text column as the training data (see table above / CSV template).",
        "{exc}. Plik CSV potrzebuje tej samej kolumny tekstu co dane treningowe (zob. tabela powyżej / szablon "
        "CSV)."),
    "z. B. stimmen": ("e.g. votes", "np. glosy"),
    "z. B. positiv": ("e.g. positive", "np. pozytywna"),
    "Einheit": ("Unit", "Jednostka"),
    "Welche Wörter kommen am häufigsten vor – insgesamt oder je Klasse? Ohne Stoppwörter (the, and, is …) sind "
    "die Unterschiede zwischen den Klassen besser zu sehen.": (
        "Which words occur most often – overall or per class? Without stop words (the, and, is …) the "
        "differences between the classes are easier to see.",
        "Które słowa występują najczęściej – łącznie albo w każdej klasie? Bez stop words (the, and, is …) "
        "różnice między klasami są lepiej widoczne."),
    "je Gruppe": ("per group", "w grupie"),
    "Stoppwörter entfernen": ("Remove stop words", "Usuń stop words"),
    "Häufigste Wörter je {g}": ("Most frequent words per {g}", "Najczęstsze słowa według {g}"),
    "Häufigste Wörter in {c}": ("Most frequent words in {c}", "Najczęstsze słowa w {c}"),
    "Andere Texte": ("Other texts", "Inne teksty"),
    "Gezeigt: die ersten {n} von {m} Gruppen.": (
        "Shown: the first {n} of {m} groups.",
        "Pokazano: pierwsze {n} z {m} grup."),
    "Gezählt auf einer Stichprobe von 20 000 Texten.": (
        "Counted on a sample of 20 000 texts.",
        "Policzono na próbce 20 000 tekstów."),
    "Texte je Gruppe": ("Texts per group", "Tekstów na grupę"),
    "Wortpaare": ("word pairs", "pary słów"),
    "**{g} = {v}** · {n} Texte": ("**{g} = {v}** · {n} texts", "**{g} = {v}** · {n} tekstów"),
    "Keine Zahlenspalten – in Schritt 2 z. B. die Textlänge als neue Spalte anlegen.": (
        "No numeric columns – e.g. create the text length as a new column in step 2.",
        "Brak kolumn liczbowych – np. utwórz w kroku 2 długość tekstu jako nową kolumnę."),
    "Wie im Kurs: stars = yelp.groupby('stars').mean() – Mittelwert jeder Zahlenspalte je Gruppe und die "
    "Korrelation dieser Mittelwerte (z. B. cool, useful, funny, text length).": (
        "As in the course: stars = yelp.groupby('stars').mean() – mean of every numeric column per group and the "
        "correlation of these means (e.g. cool, useful, funny, text length).",
        "Jak na kursie: stars = yelp.groupby('stars').mean() – średnia każdej kolumny liczbowej w grupie i "
        "korelacja tych średnich (np. cool, useful, funny, text length)."),
    "**So wird Text zu Zahlen (Bag of Words)**\n\n1. **Wörter zerlegen** – Kleinschreibung, Satzzeichen weg, "
    "optional Stoppwörter (the, and, is …) entfernen.\n2. **Vokabular** – jedes Wort aus den **Trainingstexten** "
    "bekommt eine Spalte (oft Tausende).\n3. **Zählen** – jeder Text wird zu einer Zeile: wie oft kommt jedes "
    "Wort vor (CountVectorizer). Die meisten Werte\n   sind 0 – daher eine *sparse* Matrix, die nur die "
    "Nicht-Nullen speichert.\n4. **TF-IDF** (optional) – Wörter, die in fast allen Texten vorkommen, zählen "
    "weniger; seltene, typische Wörter mehr.\n\nWie im Kurs mit `CountVectorizer` (und `TfidfTransformer`) – nur "
    "in der richtigen Reihenfolge: **erst aufteilen**,\ndann das Vokabular nur aus den Trainingstexten lernen. "
    "Die Testtexte werden damit umgewandelt wie später neue Texte.": (
        "**How text becomes numbers (bag of words)**\n\n1. **Split into words** – lower case, punctuation "
        "removed, optionally stop words (the, and, is …) removed.\n2. **Vocabulary** – every word from the "
        "**training texts** gets its own column (often thousands).\n3. **Count** – every text becomes one row: "
        "how often does each word occur (CountVectorizer). Most values\n   are 0 – hence a *sparse* matrix that "
        "stores only the non-zeros.\n4. **TF-IDF** (optional) – words that occur in almost all texts count less; "
        "rare, typical words count more.\n\nAs in the course with `CountVectorizer` (and `TfidfTransformer`) – "
        "just in the right order: **split first**,\nthen learn the vocabulary from the training texts only. The "
        "test texts are transformed with it like new texts later.",
        "**Jak tekst zamienia się w liczby (bag of words)**\n\n1. **Podział na słowa** – małe litery, bez "
        "interpunkcji, opcjonalnie bez stop words (the, and, is …).\n2. **Słownik** – każde słowo z **tekstów "
        "treningowych** dostaje własną kolumnę (często tysiące).\n3. **Zliczanie** – każdy tekst staje się "
        "jednym wierszem: ile razy występuje każde słowo (CountVectorizer).\n   Większość wartości to 0 – "
        "dlatego macierz *rzadka* (sparse), która zapisuje tylko niezerowe wartości.\n4. **TF-IDF** "
        "(opcjonalnie) – słowa występujące w prawie wszystkich tekstach liczą się mniej; rzadkie, typowe\n   "
        "słowa bardziej.\n\nJak na kursie z `CountVectorizer` (i `TfidfTransformer`) – tylko we właściwej "
        "kolejności: **najpierw podział**,\npotem słownik tylko z tekstów treningowych. Teksty testowe są nim "
        "przekształcane tak jak później nowe teksty."),
    "\n**So werden neue Texte eingeordnet**\n\n1. **Nichts neu fitten.** Die gespeicherte Pipeline wandelt neue "
    "Texte mit dem **Vokabular aus dem Training** in\n   Zahlen um – würde man den Vektorisierer neu fitten, "
    "passten die Spalten nicht mehr zum Modell. Wörter, die das\n   Modell nie gesehen hat, werden "
    "ignoriert.\n2. **Gleiche Rohspalte wie im Training.** Nötig ist nur die Textspalte (Tabelle unten, Vorlage "
    "zum Download);\n   Schritt 2 (z. B. Umbenennen) wird automatisch wiederholt. Fehlt sie → klare "
    "Fehlermeldung. Weitere Spalten\n   werden ignoriert.\n3. **Abweichungen werden gemeldet:** leere Texte, "
    "Texte ohne ein einziges bekanntes Wort (andere Sprache?) und\n   Texte aus überwiegend unbekannten Wörtern "
    "– Spalte *Hinweis*.\n4. **Interpretation und Fazit:** Bei binär = Wahrscheinlichkeit für die positive "
    "Klasse + Schwelle. Ist das Label\n   in den neuen Daten enthalten, wird automatisch evaluiert. Das Fazit "
    "bewertet die Texte und erklärt, was das\n   Ergebnis aussagt – auch, ob Texte schon im Training waren.\n": (
        "\n**How new texts are classified**\n\n1. **Nothing is refitted.** The saved pipeline turns new texts "
        "into numbers with the **vocabulary from\n   training** – refitting the vectorizer would no longer match "
        "the model's columns. Words the model has never\n   seen are ignored.\n2. **Same raw column as in "
        "training.** Only the text column is needed (table below, template to download);\n   step 2 (e.g. "
        "renaming) is repeated automatically. If it is missing → clear error message. Other columns\n   are "
        "ignored.\n3. **Deviations are reported:** empty texts, texts without a single known word (another "
        "language?) and\n   texts made mostly of unknown words – column *Hinweis* (note).\n4. **Interpretation "
        "and conclusion:** for binary = probability of the positive class + threshold. If the\n   label is "
        "included in the new data, it is evaluated automatically. The conclusion rates the texts and\n   "
        "explains what the result means – including whether texts were already in training.\n",
        "\n**Jak klasyfikowane są nowe teksty**\n\n1. **Nic nie jest dopasowywane od nowa.** Zapisany potok "
        "zamienia nowe teksty na liczby **słownikiem z\n   treningu** – ponowne dopasowanie wektoryzatora nie "
        "pasowałoby już do kolumn modelu. Słowa, których model\n   nigdy nie widział, są pomijane.\n2. **Ta sama "
        "surowa kolumna co w treningu.** Potrzebna jest tylko kolumna tekstu (tabela poniżej, szablon do\n   "
        "pobrania); krok 2 (np. zmiana nazwy) jest powtarzany automatycznie. Jeśli jej brakuje → czytelny "
        "komunikat\n   błędu. Pozostałe kolumny są pomijane.\n3. **Odchylenia są zgłaszane:** puste teksty, "
        "teksty bez ani jednego znanego słowa (inny język?) i teksty\n   złożone głównie z nieznanych słów – "
        "kolumna *Hinweis* (uwaga).\n4. **Interpretacja i podsumowanie:** przy binarnej = prawdopodobieństwo "
        "klasy pozytywnej + próg. Jeśli nowe\n   dane zawierają etykietę, ewaluacja odbywa się automatycznie. "
        "Podsumowanie ocenia teksty i wyjaśnia, co\n   oznacza wynik – także to, czy teksty były już w "
        "treningu.\n"),
    "{m} fehlt im Vergleich: {why}": (
        "{m} is missing from the comparison: {why}",
        "{m} nie ma w porównaniu: {why}"),
    "CSV für Excel": (
        "CSV for Excel",
        "CSV dla Excela"),
    "Trennzeichen und Dezimalzeichen wie Excel in der gewählten Sprache (Deutsch und Polnisch: ; und Komma, "
    "Englisch: , und Punkt), UTF-8 mit BOM. Texte, die mit = + - @ beginnen, bekommen ein ' davor – Excel führt "
    "sie sonst als Formel aus.": (
        "Separator and decimal mark as in Excel for the chosen language (German and Polish: ; and comma, "
        "English: , and point), UTF-8 with BOM. Texts starting with = + - @ get a ' in front – otherwise Excel "
        "runs them as a formula.",
        "Separator i znak dziesiętny jak w Excelu dla wybranego języka (niemiecki i polski: ; i przecinek, "
        "angielski: , i kropka), UTF-8 z BOM. Teksty zaczynające się od = + - @ dostają ' na początku – inaczej "
        "Excel wykona je jako formułę."),
    "Tipp: LinearSVC arbeitet am besten mit TF-IDF (Schritt 4) – mit rohen Wortzählungen langer Texte dauert das "
    "Training länger und konvergiert oft nicht vollständig.": (
        "Tip: LinearSVC works best with TF-IDF (step 4) – with raw word counts of long texts training takes "
        "longer and often does not fully converge.",
        "Wskazówka: LinearSVC działa najlepiej z TF-IDF (krok 4) – przy surowych liczbach słów długich tekstów "
        "trening trwa dłużej i często nie osiąga pełnej zbieżności."),
    "Tipp: Mit TF-IDF sind die Werte klein (meist unter 1) – alpha = 1 glättet dann stark, besonders bei großem "
    "Vokabular (Wortpaare). Oft passt ein kleineres alpha, z. B. 0.1; die Validierungskurve in Schritt 7 zeigt "
    "den besten Wert.": (
        "Tip: with TF-IDF the values are small (mostly below 1) – alpha = 1 then smooths heavily, especially "
        "with a large vocabulary (word pairs). A smaller alpha, e.g. 0.1, often fits better; the validation "
        "curve in step 7 shows the best value.",
        "Wskazówka: przy TF-IDF wartości są małe (zwykle poniżej 1) – alpha = 1 mocno wtedy wygładza, zwłaszcza "
        "przy dużym słowniku (pary słów). Często lepiej pasuje mniejsze alpha, np. 0.1; krzywa walidacji w kroku "
        "7 pokazuje najlepszą wartość."),
    "Tipp: Mit TF-IDF braucht die Logistische Regression meist ein größeres C, z. B. 10 (die Werte sind normiert "
    "und klein); die Validierungskurve in Schritt 7 zeigt den besten Wert.": (
        "Tip: with TF-IDF logistic regression usually needs a larger C, e.g. 10 (the values are normalised and "
        "small); the validation curve in step 7 shows the best value.",
        "Wskazówka: przy TF-IDF regresja logistyczna potrzebuje zwykle większego C, np. 10 (wartości są "
        "znormalizowane i małe); krzywa walidacji w kroku 7 pokazuje najlepszą wartość."),
    "LinearSVC ist nicht vollständig konvergiert (ConvergenceWarning) – C verkleinern oder in Schritt 4 TF-IDF "
    "wählen (normierte Werte statt roher Wortzählungen).": (
        "LinearSVC did not fully converge (ConvergenceWarning) – decrease C or choose TF-IDF in step 4 "
        "(normalised values instead of raw word counts).",
        "LinearSVC nie osiągnął pełnej zbieżności (ConvergenceWarning) – zmniejsz C albo wybierz TF-IDF w kroku "
        "4 (znormalizowane wartości zamiast surowych liczb słów)."),
    "Nur kurze Texte gefunden (im Mittel unter 3 Wörtern, z. B. Titel): {cols} – das geht auch; die Textspalte "
    "wird in Schritt 4 gewählt.": (
        "Only short texts found (fewer than 3 words on average, e.g. titles): {cols} – that works too; the text "
        "column is chosen in step 4.",
        "Znaleziono tylko krótkie teksty (średnio poniżej 3 słów, np. tytuły): {cols} – to też działa; kolumnę "
        "tekstu wybiera się w kroku 4."),
}
