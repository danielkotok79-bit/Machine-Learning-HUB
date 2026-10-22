"""Übersetzungen für plots.py: {Deutsch: (Englisch, Polnisch)}"""

T = {
    "Andere": ("Other", "Inne"),
    "Keine Werte zum Anzeigen (Spalte leer).": ("No values to show (column empty).", "Brak wartości do pokazania (kolumna pusta)."),
    "fehlend in %": ("missing in %", "brakujące w %"),
    "Fehlende Werte je Spalte": ("Missing values per column", "Brakujące wartości w kolumnach"),
    "Anzahl je {x}": ("Count per {x}", "Liczność według {x}"),
    " – die {top_n} häufigsten von {n}": (" – the {top_n} most frequent of {n}", " – top {top_n} z {n}"),
    "Anzahl je {x}{note}": ("Count per {x}{note}", "Liczność według {x}{note}"),
    "Anzahl": ("Count", "Liczność"),
    "Verteilung von {x}": ("Distribution of {x}", "Rozkład {x}"),
    " – {n} × inf ausgelassen": (" – {n} × inf left out", " – pominięto {n} × inf"),
    " (1.–99. Perzentil)": (" (1st–99th percentile)", " (1.–99. percentyl)"),
    "Für den Boxplot bitte zwei verschiedene Spalten wählen.": ("Please choose two different columns for the box plot.",
                                                                "Do wykresu pudełkowego wybierz dwie różne kolumny."),
    "Keine Zeilen mit Werten in beiden Spalten.": ("No rows with values in both columns.",
                                                   "Brak wierszy z wartościami w obu kolumnach."),
    "{y} je {x}": ("{y} by {x}", "{y} według {x}"),
    "Korrelation": ("Correlation", "Korelacja"),
    "Korrelationsmatrix (−1 blau · 0 grau · +1 rot)": ("Correlation matrix (−1 blue · 0 grey · +1 red)",
                                                       "Macierz korelacji (−1 niebieski · 0 szary · +1 czerwony)"),
    "Vorhersage": ("Prediction", "Prognoza"),
    "Tatsächlich": ("Actual", "Rzeczywista"),
    "Confusion Matrix (Anzahl, Anteil je Zeile)": ("Confusion matrix (count, share per row)",
                                                   "Macierz pomyłek (liczność, udział w wierszu)"),
    "tatsächlich: {cl}": ("actual: {cl}", "rzeczywista: {cl}"),
    "  Schwelle {threshold:.2f}": ("  threshold {threshold:.2f}", "  próg {threshold:.2f}"),
    "vorhergesagte Wahrscheinlichkeit für '{v}'": ("predicted probability of '{v}'",
                                                   "przewidywane prawdopodobieństwo '{v}'"),
    "Verteilung der Wahrscheinlichkeiten": ("Distribution of the probabilities", "Rozkład prawdopodobieństw"),
    "vorhergesagt": ("predicted", "przewidywana"),
    "Vorhersage neue Daten": ("Prediction for new data", "Prognoza dla nowych danych"),
    "Anteil im Training": ("Share in training", "Udział w treningu"),
    "Anteil in %": ("Share in %", "Udział w %"),
    "Vorhergesagte Klassen": ("Predicted classes", "Przewidywane klasy"),
    # Schritt 7 und 8 (ML-Version)
    "Training": (
        "Training",
        "Trening"),
    "naive Vorhersage": (
        "naive prediction",
        "naiwna prognoza"),
    "eingestellt": (
        "set",
        "ustawiono"),
    "bester Wert: {v}": (
        "best value: {v}",
        "najlepsza wartość: {v}"),
    "{metric} je {param}": (
        "{metric} by {param}",
        "{metric} a {param}"),  # „według“ verlangt den Genitiv („liczby drzew“) – „a“ passt zu jedem Parameter
    # --- NLP-App (Textklassifikation) ---
    "Zeichen": ("characters", "znaki"),
    "Textlänge in {unit}": ("Text length in {unit}", "Długość tekstu ({unit})"),
    "Wichtigste Wörter je Klasse": ("Most important words per class", "Najważniejsze słowa w każdej klasie"),
    " (bis zum 99. Perzentil: {v:.0f})": (" (up to the 99th percentile: {v:.0f})", " (do 99. percentyla: {v:.0f})"),
    "Modellvergleich auf den Testdaten (größer = besser)": (
        "Model comparison on the test data (higher = better)",
        "Porównanie modeli na danych testowych (więcej = lepiej)"),
    " – die ersten {n} von {m} Klassen": (" – the first {n} of {m} classes", " – pierwsze {n} z {m} klas"),
    "Welche Wörter haben entschieden? → '{label}'": (
        "Which words decided? → '{label}'",
        "Które słowa zdecydowały? → '{label}'"),
    "Beitrag (blau: spricht für '{label}', orange: dagegen)": (
        "Contribution (blue: speaks for '{label}', orange: against)",
        "Wkład (niebieski: przemawia za '{label}', pomarańczowy: przeciw)"),
    "n = {n}\nMedian {m:.0f}": ("n = {n}\nmedian {m:.0f}", "n = {n}\nmediana {m:.0f}"),
    "Keine Wörter gefunden (nur Stoppwörter oder einzelne Zeichen?).": (
        "No words found (only stop words or single characters?).",
        "Nie znaleziono słów (same stop words albo pojedyncze znaki?)."),
    "spricht für '{cl}'": ("speaks for '{cl}'", "przemawia za '{cl}'"),
    "Welche Wörter haben entschieden?": ("Which words decided?", "Które słowa zdecydowały?"),
    "Kein Wort dieses Textes steht im Vokabular – das Modell entscheidet nur nach der Häufigkeit der Klassen.": (
        "No word of this text is in the vocabulary – the model decides only by the class frequencies.",
        "Żadne słowo tego tekstu nie występuje w słowniku – model decyduje tylko na podstawie częstości klas."),
    "Log-Verhältnis (Naive Bayes)": ("log ratio (Naive Bayes)", "logarytm ilorazu (naiwny Bayes)"),
    "alle Texte": ("all texts", "wszystkie teksty"),
    "Confusion Matrix (Anzahl)": (
        "Confusion matrix (count)",
        "Macierz pomyłek (liczność)"),
}
