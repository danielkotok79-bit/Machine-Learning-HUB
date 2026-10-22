# Audyt kodu – Data-Analysis-Template (Streamlit + Keras)

> Trzy audyty: **audyt 1** (poniżej), **audyt 2** – całe oprogramowanie po rozszerzeniach, **audyt 3** – kod
> wersji 6 i 7 (na końcu pliku). Pomiędzy nimi: **wersja 6** – wersje językowe (angielska i polska) i sposób
> sprawdzenia, że wersja niemiecka się nie zmieniła, oraz **wersja 7** – cztery nowe wykresy w kroku 3.

**Data:** 14.10.2026 · **Zakres:** `app.py`, `pipeline.py`, `plots.py`, `codegen.py`, testy
**Metoda:** przegląd kodu + ok. 30 sond z przypadkami brzegowymi + testy w przeglądarce (Playwright) na
działającym serwerze. Każdy potwierdzony błąd został naprawiony i zabezpieczony testem regresyjnym.

**Wynik:** 24 naprawione problemy (6 poważnych) + zmiana metodyczna: uzupełnianie braków dopiero po podziale
na zbiór treningowy i testowy (sekcja „Zmiana metodyczna”). Testy: **67 zamiast 12**, wszystkie przechodzą na
pandas 3.0 i 2.2; składnia sprawdzona dla Pythona 3.10–3.13. Modele zapisane poprzednimi wersjami nadal się
wczytują (test `test_recipes_from_version_1_still_work`).

---

## Wysoka waga – awaria albo po cichu błędne wyniki

| # | Problem (wersja 1) | Skutek | Poprawka |
|---|---|---|---|
| 1 | Etykieta typu bool (`True`/`False`) | strona 4 kończyła się wyjątkiem – krok 4 nie do użycia | kodowanie etykiety po wartości (`label_kind`, `class_values`); w kodzie: `.astype(int)` |
| 2 | „Zahl extrahieren” przy separatorach tysięcy | **po cichu złe dane:** `$1,200` → 1.2, `1,234,567` → 1.234, `1.234,5` → 1.234 | nowa funkcja `parse_numbers` (format US i niemiecki, waluty, jednostki) |
| 3 | Wartości ±inf w kolumnie liczbowej | błąd skalera (`Input X contains infinity`) | inf traktowane jak brak → wypełniane; to samo dla nowych danych |
| 4 | Wieloklasowa etykieta z liczbami niecałkowitymi (0.5 / 1.0 / 1.5) | błąd „Label konnte nicht vollständig kodiert werden” | kodowanie po wartości, kolejność klas numeryczna |
| 5 | Wygenerowany kod (podgląd i skrypt z kroku 9) liczył **inaczej** niż aplikacja | m.in. data jako liczba → rok 1970, `22690.0` zamiast `22690`, `SyntaxError` przy nazwach ze spacją, `stratify=y` mimo że aplikacja go wyłączyła, średnie grupowe liczone po wypełnieniu | `codegen.py` przepisany; 14 testów porównuje liczbowo `X_train`, `X_test`, `y` z aplikacji i z wygenerowanego kodu |
| 6 | ZIP z modelem: ścieżka `model_file` w `preprocessing.json` nie była sprawdzana | spreparowany ZIP mógł wczytać plik modelu spoza archiwum (ścieżka absolutna lub `../`) | używana jest tylko nazwa pliku |

## Średnia waga

| # | Problem | Poprawka |
|---|---|---|
| 7 | „Datum → Jahr/Monat” na liczbach typu `20141013` – kolumna zostawała bez zmian | rozpoznawanie formatu RRRRMMDD |
| 8 | Kolumna bez żadnej wartości + strategia „häufigster Wert” → `IndexError` | puste kolumny usuwane z ostrzeżeniem |
| 9 | `stratify` przy wielu klasach / małym zbiorze testowym → błąd scikit-learn | automatyczny podział bez `stratify` + ostrzeżenie, zapis `stratify_used` w recepcie |
| 10 | Spacje na brzegach kategorii (`RENT` vs `RENT `) → osobne dummies, w nowych danych „nieznana kategoria”; etykieta `yes`/`yes ` → „3 klasy” | ujednolicone `strip()` w treningu, nowych danych i walidacji etykiety |
| 11 | Cecha bool z luką w nowych danych (`True`/puste/`False`) → fałszywe ostrzeżenie i złe wartości | tekst `True`/`False` → 1/0 |
| 12 | CSV ze średnikiem i przecinkiem dziesiętnym przy ustawieniach domyślnych → 1 kolumna | automatyczne rozpoznanie separatora i przecinka dziesiętnego; pola wyboru i wygenerowany kod przejmują wykryte ustawienia |
| 13 | Ścieżka w cudzysłowie (Windows „Kopiuj jako ścieżkę”) → plik nie znaleziony | cudzysłowy i spacje są usuwane |
| 14 | Import konfiguracji z ręcznie zmienionymi wartościami (`"600"` jako tekst, test_size 0.9) → wyjątek Streamlit | `coerce_setting`: typ + zakres, nieprawidłowe wartości pomijane z komunikatem |
| 15 | Zapis w kroku 9 po cichu nadpisywał istniejący model o tej samej nazwie | ostrzeżenie + pole potwierdzenia |
| 16 | Pusty plik z nowymi danymi (tylko nagłówek) → niezrozumiały błąd Keras | jasny komunikat |

## Niska waga

| # | Problem | Poprawka |
|---|---|---|
| 17 | `id(model)` jako klucz pamięci podręcznej – po ponownym treningu Python może użyć tego samego id → stare wyniki / próg | unikalny token (`uuid`) na każdy trening i każdy wczytany model |
| 18 | Edytor braków danych pokazywał stare liczby po ponownym kroku 2 | sygnatura zawiera token danych |
| 19 | ZIP do pobrania był pakowany przy każdym kliknięciu na stronie 9 | pamięć podręczna (zależna od daty zmiany plików) |
| 20 | Countplot sortowany wg częstości: zbiór budowany dla każdej kategorii (O(n·k)) – bardzo wolno przy wielu kategoriach | posortowane `value_counts()` |
| 21 | Histogram z wartościami inf | inf pomijane, liczba podana w tytule |
| 22 | Spacje w nazwach kolumn CSV: aplikacja je usuwała, wygenerowany skrypt nie → `KeyError` | `df.columns.str.strip()` w kodzie (gdy potrzebne) + `low_memory=False` jak w aplikacji |
| 23 | Etykiety w nowych danych, których nie było w treningu, znikały po cichu z ewaluacji | ostrzeżenie z liczbą wierszy |
| 24 | `RuntimeWarning` przy sprawdzaniu liczb całkowitych z inf / bardzo dużymi wartościami | funkcja `_whole` |

---

## Zmiana metodyczna (14.10.2026): najpierw podział, potem uzupełnianie braków

Wcześniej wartości do uzupełniania liczone były na wszystkich wierszach **przed**
`train_test_split`, czyli z udziałem zbioru testowego (mały wyciek informacji). Teraz kolejność w kroku 4 jest taka:

1. kodowanie etykiety → `train_test_split` (te same wiersze w train/test co wcześniej – podział zależy tylko od
   liczby wierszy, `random_state` i `y`),
2. **wyłącznie z wierszy treningowych:** wartości do uzupełniania (mediana, średnia, najczęstsza wartość, średnia
   grupowa), wartości zapasowe dla nowych danych, kategorie (kolumny dummies) i skaler,
3. wiersze testowe przechodzą przez tę samą receptę co później nowe dane – niczego się z nich nie uczymy.

Skutki: kategoria występująca tylko w zbiorze testowym jest traktowana jak nieznana (wszystkie dummies = 0,
informacja w logu), a kolumna bez żadnej wartości w wierszach treningowych jest usuwana z ostrzeżeniem.
Wygenerowany kod Pythona ma tę samą kolejność (`train, test = train_test_split(df, …)`, potem `fill_missing`
i `to_X`); 14 testów porównuje go liczbowo z aplikacją. Nowe recepty mają `"fit_on": "train"`.

Z wszystkich wierszy (przed podziałem) powstają już tylko: lista „Top-N behalten” z kroku 2 oraz automatyczne
podpowiedzi w tabelach – oba z niewielkim wpływem.

## Uwagi metodyczne (nie błędy – opisane w README)

* **Optymistyczne metryki testowe:** domyślnie zbiór testowy służy też jako walidacja dla
  EarlyStopping, a próg w kroku 8 ustawia się na danych testowych. Uczciwsza ocena: w kroku 6 wybrać
  „Validierung: Anteil aus Training”.
* **Bezpieczeństwo:** aplikacja jest do użytku lokalnego. Pole ścieżki czyta każdy plik dostępny dla
  zalogowanego użytkownika, a folder `models/` jest wspólny – nie wystawiać aplikacji do sieci bez logowania.
  Modele (ZIP / `.h5` / `.keras`) wczytywać tylko z zaufanych źródeł. Projekt nie używa `pickle`
  (recepta to czysty JSON).

## Co sprawdzono i działa poprawnie

* Recepta (`preprocessing.json`): surowe wiersze testowe dają w kroku 10 **identyczne** prognozy jak w treningu.
* Skaler tylko na `X_train`; kolejność kolumn dummies stała także dla nowych danych.
* Obejście problemu Keras 3 z `.h5` + `loss='mse'` (pełne nazwy funkcji straty, `compile=False`).
* Zgodność: pandas 2.2 i 3.0, Python 3.10–3.13, Streamlit 1.65, TensorFlow 2.21 / Keras 3.15.

## Testy

| Plik | Liczba | Co sprawdza |
|---|---|---|
| `tests/test_pipeline.py` | 11 | pipeline przykładowy (LendingClub), zapis/odczyt `.keras`/`.h5`, nowe dane, regresja, wiele klas, eksport skryptu |
| `tests/test_codegen_parity.py` | 14 | wygenerowany kod = aplikacja (liczbowo), także w przypadkach brzegowych |
| `tests/test_audit_regressions.py` | 40 | jeden test na każdy błąd z tego audytu, zgodność ze starymi modelami, uzupełnianie tylko z danych treningowych |
| `tests/test_app.py` | 2 | klikanie przez wszystkie 10 kroków; etykieta bool, CSV ze średnikiem, ochrona przed nadpisaniem |

Uruchomienie: `pip install pytest` → `pytest -q` (ok. 1 minuty).

## Aktualizacja

Wystarczy podmienić pliki projektu. Zapisane wcześniej modele (`models/…`) działają dalej; nowe recepty mają
`template_version: 3`. Po aktualizacji warto raz ponownie kliknąć „Daten für ML vorbereiten” i wytrenować model,
bo wartości uzupełniające liczone są teraz inaczej.

---

## Rozszerzenia (14.10.2026, po audycie)

Na życzenie dodane funkcje – szczegóły w README:

* usunięte wszystkie odniesienia do kursu – aplikacja jest uniwersalna; tytuł **Data-Analysis-Template**;
  emoji (u części użytkowników widoczne jako puste kwadraty) zastąpione ikonami Material,
* wybór kolumn do wizualizacji bezpośrednio w kroku 3,
* akcja „Datum → Jahr + Monat” (dwie kolumny `<nazwa>_jahr`, `<nazwa>_monat`),
* nowe kolumny w kroku 2: przypisanie wartości, formuła, warunek → 0/1, kopia + akcja – zapisywane w recepcie
  i liczone automatycznie dla nowych danych; formuły sprawdzane (tylko działania, porównania i kilka funkcji),
* krok 4: pomoc z korelacją przy wyborze kolumny grupującej, przegląd kolumn dummies,
* nowy krok 5 „Daten speichern” (CSV po kroku 2 lub gotowe dane ML po kroku 4, także w formacie dla
  niemieckiego Excela),
* krok 10: podsumowanie („Fazit”) – jakość danych, znaczenie wyniku, zalecenia, rozpoznawanie wierszy
  z danych treningowych (sumy kontrolne próbki wierszy w recepcie).

Testy: **87** (dodatkowo `tests/test_new_features.py` i rozszerzone testy zgodności kodu), wszystkie przechodzą
na pandas 3.0 i 2.2.


---

# Audyt 2 – całe oprogramowanie po rozszerzeniach (14.10.2026)

**Zakres:** cały kod po zmianach z wersji 4 – nowe kolumny (przypisanie wartości, formuła, warunek, kopia + akcja),
krok 5 „Daten speichern”, Fazit w kroku 10, sumy kontrolne wierszy treningowych, wybór kolumn w kroku 3,
import/eksport konfiguracji. Pliki: `app.py`, `pipeline.py`, `codegen.py`, `plots.py`, testy.
**Metoda:** przegląd kodu, ok. 40 sond z przypadkami brzegowymi i celowo „złośliwymi” danymi (formuły, pliki
konfiguracji, ZIP), porównanie z wersją 4, test klikany (AppTest) i test w prawdziwej przeglądarce na działającym
serwerze (Playwright, 18 kontroli). Każdy potwierdzony błąd został naprawiony i zabezpieczony testem regresyjnym.

**Wynik:** 14 naprawionych problemów (3 poważne). Testy: **125 zamiast 87** – wszystkie przechodzą na pandas 3.0
i 2.2; składnia sprawdzona dla Pythona 3.10. Modele i konfiguracje zapisane wcześniejszymi wersjami nadal się
wczytują.

## Wysoka waga – zawieszenie, awaria albo bezużyteczny model

| # | Problem (wersja 4) | Skutek | Poprawka |
|---|---|---|---|
| B1 | Formuła w „Neue Spalten” bez limitów | `9 ** 9 ** 9 + a` liczy liczbę o setkach milionów cyfr – aplikacja nie reaguje przez wiele minut; `'x' * 10**8` zajmuje setki MB; bardzo głębokie zagnieżdżenie (`------…a`) → `RecursionError` i awaria strony | wykładnik `**` tylko od −100 do 100 (albo kolumna), liczby przy potędze liczone jako zmiennoprzecinkowe (przepełnienie → od razu komunikat), tekst w cudzysłowie tylko w porównaniach, maks. 1000 znaków i 100 poziomów zagnieżdżenia, `round(x, n)` tylko z n = 0–10 |
| B2 | Import ręcznie zmienionej lub uszkodzonej konfiguracji (JSON) | lista zamiast obiektu → nieobsłużony wyjątek; `"w_features": 5` → `TypeError` w krokach 2–4 przy każdym kliknięciu; warstwa z 10⁹ neuronów → brak pamięci przy budowie modelu; `Infinity` → nieobsłużony `OverflowError` | nowa funkcja `sanitize_config`: typy, zakresy, długości list i tekstów (maks. 50 warstw, 4096 neuronów, 200 nowych kolumn); błędne części są pomijane i wymienione w komunikacie; aplikacja jest też odporna na błędne wartości w stanie sesji; plik z BOM (Notatnik w Windows) jest akceptowany |
| B3 | Etykieta z wartościami ±inf (regresja) | model uczony na `inf` (loss = inf/nan – bezużyteczny); jeśli `inf` trafi do zbioru testowego – błąd scikit-learn w kroku 8 i 10 | wiersze z `inf` w etykiecie usuwane w kroku 4 (log, pole `label_inf_removed` w recepcie, ta sama linia w wygenerowanym kodzie); w kroku 10 ocena tylko na skończonych etykietach |

## Średnia waga

| # | Problem | Poprawka |
|---|---|---|
| B4 | Eksport ML (krok 5): cecha o nazwie `split` była **po cichu nadpisywana** kolumną train/test | kolumna train/test nazywa się wtedy `split_train_test` (aplikacja i wygenerowany kod) |
| B5 | Eksport „dla niemieckiego Excela”: tekst zaczynający się od `=`, `+`, `-`, `@` Excel wykonuje jako formułę (CSV-Injection, np. `=HYPERLINK(…)`) | w tym formacie takie komórki dostają `'` na początku; format międzynarodowy bez zmian; informacja w aplikacji |
| B6 | „Werte zuordnen” z inną pisownią (`fully paid=1` zamiast `Fully Paid`) → kolumna **po cichu pusta**, potem niezrozumiałe błędy w kroku 4 | komunikat z przykładami istniejących wartości |
| B7 | Krok 3: nietypowe dane (np. histogram z kolorem dla pustej kolumny) → wyjątek seaborn i awaria całej strony | wszystkie wykresy obsługują puste dane („Keine Werte zum Anzeigen”); dodatkowo osłona: błąd wykresu = ostrzeżenie zamiast awarii strony |
| B8 | Usunięcie nowej kolumny, z której korzysta inna nowa kolumna → niejasny błąd z kroku 2 | sprawdzenie zależności: „'B' benutzt 'A' – zuerst oder gleichzeitig entfernen”, nic nie jest zmieniane |
| B9 | Krok 5 „In Ordner speichern” nadpisywał istniejący plik bez pytania | ostrzeżenie + pole potwierdzenia (plik zapisany w tej samej sesji można zastąpić bez pytania) |
| B10 | Model ZIP (krok 9): brak limitu rozmiaru po rozpakowaniu („ZIP-bomba” może zapełnić dysk/pamięć) | maks. 2 GB po rozpakowaniu, sprawdzane przed rozpakowaniem |
| B11 | Prywatność: recepta zawiera do 3000 sum kontrolnych (hashy) wierszy treningowych – kto ma dane źródłowe, może sprawdzić, czy dany wiersz był w treningu | w kroku 9 opcja „Prüfsummen … mitspeichern” (domyślnie włączona, z wyjaśnieniem); bez nich Fazit pomija tylko to jedno sprawdzenie; opis w README |

## Niska waga

| # | Problem | Poprawka |
|---|---|---|
| B12 | Wygenerowany kod eksportu ML (krok 5) różnił się od aplikacji: przy skalowanej etykiecie (regresja) zapisywał wartości skalowane, wiersze w innej kolejności (najpierw train, potem test) | ten sam wynik co w aplikacji – test porównuje oba pliki CSV |
| B13 | Eksport ML z aplikacji: klasy jako `0.0`/`1.0` | klasy jako liczby całkowite `0`/`1`, w regresji oryginalne wartości |
| B14 | Fazit: udział przewidywanej klasy liczony z zaokrąglonych prawdopodobieństw (`0.50004` → `0.5`), sama prognoza z niezaokrąglonych – drobna rozbieżność; nieaktualne komentarze i docstring | udział liczony z kolumny „Vorhersage”; komentarze poprawione |

## Co sprawdzono i działa poprawnie

* **Formuły są bezpieczne:** tylko działania, porównania i kilka funkcji; nazwy zawsze tłumaczone na `df['…']`;
  próby typu `__import__('os')`, `a.__class__`, `lambda`, indeksowanie są odrzucane. Formuły z recepty (także
  z obcego ZIP) są przy wczytaniu ponownie sprawdzane – zapisany kod nigdy nie jest wykonywany.
* ZIP z modelem: rozpakowywane są tylko oczekiwane nazwy plików, ścieżka `model_file` z JSON jest ignorowana.
* Sumy kontrolne: 400 000 wierszy w ok. 0,8 s; rozpoznawanie danych treningowych w kroku 10 działa.
* „Najpierw podział, potem uzupełnianie” – aplikacja i wygenerowany kod liczą identycznie (19 testów porównawczych,
  w tym nowe kolumny, `inf` w etykiecie i eksport z kroku 5).
* Nieudane usunięcie nowej kolumny przywraca poprzedni stan; brak ostrzeżeń Deprecation/FutureWarning z kodu
  projektu (testy porównawcze traktują je jako błędy).
* Test w przeglądarce (prawdziwy serwer): import uszkodzonych konfiguracji, nowe kolumny i ich zależności, wykresy,
  eksport z potwierdzeniem nadpisania, trening, zapis bez sum kontrolnych – 18/18.

## Testy (stan po audycie 2)

| Plik | Liczba | Co sprawdza |
|---|---|---|
| `tests/test_pipeline.py` | 11 | pipeline przykładowy, zapis/odczyt `.keras`/`.h5`, nowe dane, regresja, wiele klas, eksport skryptu |
| `tests/test_codegen_parity.py` | 19 | wygenerowany kod = aplikacja (liczbowo), także eksport z kroku 5 |
| `tests/test_new_features.py` | 19 | rok + miesiąc, nowe kolumny, zapis danych, Fazit |
| `tests/test_audit_regressions.py` | 40 | błędy z audytu 1 |
| `tests/test_audit2_regressions.py` | 33 | błędy z audytu 2 (B1–B14) |
| `tests/test_app.py` | 3 | klikanie przez wszystkie 10 kroków; etykieta bool i CSV ze średnikiem; zabezpieczenia z audytu 2 |

Uruchomienie: `pip install pytest` → `pytest -q` (ok. 2 minut).

## Aktualizacja do wersji 5

Wystarczy podmienić pliki projektu. Format recepty się nie zmienił (nowe pole `label_inf_removed` jest
opcjonalne) – zapisane modele działają dalej, konfiguracje JSON z poprzednich wersji wczytują się bez zmian.

---

# Wersja 6 – wersja angielska i polska (15.10.2026)

**Zakres:** przetłumaczenie wszystkich tekstów z niemieckiego na angielski i polski. Silnik obliczeń
(przygotowanie danych, trening, ewaluacja, prognozy) pozostał bez zmian.

## Co dodano

* Przełącznik języka na górze paska bocznego: **Deutsch · English · Polski**. Start od razu w danym języku:
  `streamlit run app.py -- --lang pl`, adres `?lang=en` albo zmienna `DL_TEMPLATE_LANG`; domyślnie niemiecki.
* `i18n.py` + katalog `locales/` (748 tekstów `{niemiecki: (angielski, polski)}`, osobny plik dla każdego modułu).
  Teksty w kodzie pozostają niemieckie i są tłumaczone dopiero przy wyświetlaniu (`tx("…")`, stałe oznaczone
  `N_("…")`). Wartości wewnętrzne – akcje, strategie, nazwy kolumn wyników, wartości wpisywane do danych
  (`Sonstige`, `fehlend`, `_jahr`/`_monat`) – są takie same we wszystkich językach, dlatego konfiguracje i
  zapisane modele są wymienne między wersjami językowymi.
* Przetłumaczone są: interfejs, komunikaty i ostrzeżenia, wykresy (tytuły, osie, legendy), nagłówki tabel,
  komentarze i przykładowe nazwy plików w generowanym kodzie, podsumowanie (Fazit) w kroku 10, nazwy pobieranych
  plików oraz kolumny pliku z prognozami.
* Krok 5 – CSV dla Excela zależnie od języka: niemiecki i polski `;` + przecinek dziesiętny, angielski `,` +
  kropka dziesiętna (Wielka Brytania, USA, Australia, Kanada, Irlandia i Nowa Zelandia używają tego samego
  formatu, więc wystarcza jedna opcja). Zawsze UTF-8 z BOM i ochrona przed CSV injection; pokazywany kod eksportu
  odpowiada wybranemu formatowi.

## Znaleziony błąd (istniał od wersji 4)

| # | Problem | Skutek | Poprawka |
|---|---|---|---|
| C1 | Kod eksportu „dla Excela” w kroku 5: komentarz `# für deutsches Excel` był wstawiony przed nawiasem zamykającym `to_csv(…)` | skopiowany fragment kodu nie działał (SyntaxError); sam plik CSV z aplikacji był poprawny | komentarz za nawiasem; test wykonuje wygenerowany kod i wczytuje zapisany plik (de/pl/en) |

## Jak sprawdzono, że wersja niemiecka się nie zmieniła

* Porównanie wersji 5 i 6 na tych samych danych: 38 wyników (kod generowany dla kroków 2–9, kod wykresów,
  dzienniki kroków 2 i 4, komunikaty błędów, tabela kolumn, wyniki i podsumowanie kroku 10) – identyczne, z jedynym
  wyjątkiem poprawki C1.
* Cała niemiecka aplikacja przeklikana w wersji 5 i 6 (68 widoków: wszystkie strony, wszystkie wykresy, trening,
  zapis, prognoza, formularz) – teksty identyczne; różnice tylko: C1, godzina zapisu modelu i losowy rekord w
  kroku 8.
* Wszystkie dotychczasowe testy (125) przechodzą bez zmian.

## Jak sprawdzono tłumaczenia

* `tests/test_i18n.py` (16 testów): każdy tekst ma tłumaczenie angielskie i polskie (bez pustych i bez
  nieużywanych wpisów), te same symbole zastępcze `{…}`; wybór języka przy starcie; CSV dla Excela w każdym
  języku; wygenerowany kod po angielsku i po polsku liczy dokładnie to samo co aplikacja; cała aplikacja
  (10 kroków) po angielsku i po polsku bez niemieckich pozostałości.
* Skan wszystkich widocznych tekstów po angielsku i po polsku (447 tekstów interfejsu, ok. 300 tekstów na
  wykresach, wszystkie bloki kodu): brak niemieckich słów – poza nazwą „Data-Analysis-Template” i wartościami
  danych (`Sonstige`, `fehlend`).
* Test w przeglądarce (prawdziwy serwer): start z `--lang pl` i `?lang=en`, przełączanie języka w trakcie pracy,
  eksport dla Excela po angielsku i po polsku (plik i pokazany kod), trening i ewaluacja po polsku – 19/19.
* Polskie teksty z liczbami są sformułowane tak, żeby odmiana była poprawna dla każdej liczby (np. „wiersze:
  12”, „12 z 40 wierszy”).

## Ograniczenia (opisane w README)

* Po zmianie języka w trakcie pracy już utworzone komunikaty (np. dziennik kroku 2 lub 4) pozostają w poprzednim
  języku, dopóki krok nie zostanie wykonany ponownie.
* Teksty samego Streamlita (np. „Upload” w polu pliku, menu ⋮) są zawsze po angielsku; komentarze w kodzie
  źródłowym są po niemiecku.
* Poza aplikacją (`import pipeline`) komunikaty są domyślnie niemieckie; `i18n.set_lang("pl")` przełącza
  komunikaty i podsumowanie. Nazwy kolumn wyników (`Vorhersage`, `Sicherheit`, `Hinweis`) pozostają niemieckie.

## Testy (stan wersji 6)

141 testów (125 dotychczasowych + 16 w `tests/test_i18n.py`) – wszystkie przechodzą na pandas 3.0 i 2.2;
składnia sprawdzona dla Pythona 3.10. Uruchomienie: `pytest -q` (ok. 3–4 minuty).

## Aktualizacja do wersji 6

Wystarczy podmienić pliki projektu (nowe: `i18n.py`, `locales/`, `README_EN.md`, `README_PL.md`,
`tests/test_i18n.py`). Format recepty i konfiguracji się nie zmienił – zapisane modele i konfiguracje działają w
każdej wersji językowej.

---

# Wersja 7 – nowe wykresy w kroku 3 (15.10.2026)

Dodane cztery wykresy z kursu – na końcu listy w kroku 3, we wszystkich trzech językach. Dotychczasowe wykresy,
ich kolejność i reszta aplikacji są bez zmian.

| Wykres (DE · EN · PL) | Odpowiada kodowi z kursu | Ustawienia |
|---|---|---|
| Jointplot · Joint plot · Jointplot | `sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')`, `kind='kde'` | x, y, rodzaj: scatter / kde / hex / reg, kolor (hue – tylko scatter i kde), maks. punktów |
| Dichte 2D (KDE) · Density 2D (KDE) · Gęstość 2D (KDE) | `sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap='plasma', shade=True, shade_lowest=False)` | x, y, filtr na jedną grupę (kolumna = wartość, np. species = setosa), wypełnienie |
| Regression (lmplot) · Regression (lmplot) · Regresja (lmplot) | `sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid')` | x, y, kolor (hue), podwykresy (col, do 6 wartości) |
| Histogramm je Gruppe · Histogram per group · Histogram według grup | `loans[loans['credit.policy'] == 1]['fico'].hist(alpha=0.5, bins=30, …)` dla każdej grupy | kolumna liczbowa, grupa (2–8 wartości), bins |

* **Pokazywany kod** używa aktualnej składni: `sns.kdeplot(x=…, y=…, data=…, fill=True, thresh=0.05)` – stara forma
  z kursu (dwie kolumny bez nazw, `shade`, `shade_lowest`) daje w seaborn 0.13 błąd lub ostrzeżenia; przy lmplot
  komentarz, że `plt.figure(figsize=…)` przed `sns.lmplot` nic nie zmienia (rozmiar przez `height`/`aspect`).
  Histogramy grup mają wspólne przedziały (`np.histogram_bin_edges`), więc słupki grup leżą dokładnie nad sobą.
* **Zabezpieczenia:** puste kolumny, `inf`, liczby całkowite z lukami (`Int64`), wartości stałe albo dokładnie
  liniowe (gęstości KDE nie da się wtedy policzyć – zamiast błędu pokazywany jest komunikat), ta sama kolumna jako
  x i y. Przy dużych danych losowa próbka (domyślnie 3000 punktów dla jointplot/KDE, 5000 dla lmplot – suwak).
* **Poprawka z wersji 6 (C2):** po zmianie języka w pasku bocznym następne kliknięcie cofało aplikację do kroku 1 –
  przeglądarka odsyłała nazwę strony jeszcze w poprzednim języku, której Streamlit już nie rozpoznawał. Wybór strony
  jest teraz odświeżany przy każdym przebiegu, tak jak pozostałe ustawienia (`w_…`), więc po zmianie języka
  przeglądarka od razu dostaje nową nazwę. Sprawdzone w przeglądarce na prawdziwym serwerze: DE → EN → PL, po każdej
  zmianie i po kolejnym kliknięciu zostają strona i wybrany wykres (12/12). Test klikania w Pythonie (AppTest) tego
  błędu nie odtwarza, bo nie przechowuje wartości tak jak przeglądarka.
* **Testy:** `tests/test_charts.py` (39 testów) – każda odmiana wykresu, nietypowe dane, wykonanie pokazywanego
  kodu (filtr w kodzie wybiera te same wiersze co aplikacja, histogram ma te same etykiety), test klikania w
  aplikacji. Razem **180 testów** – wszystkie przechodzą na pandas 3.0 i 2.2. Teksty nowych wykresów (interfejs i
  wykresy) sprawdzone skanem po angielsku i po polsku – bez niemieckich pozostałości.

---

# Audyt 3 – kod wersji 6 i 7 (15.10.2026)

**Zakres:** wszystko, co powstało po audycie 2: warstwa językowa (`i18n.py`, `locales/`, tłumaczenia we wszystkich
modułach, przełącznik w pasku bocznym), CSV dla Excela zależnie od języka, cztery nowe wykresy w kroku 3 (funkcje
w `plots.py`, interfejs, generowany kod) oraz poprawka nawigacji z wersji 7.
**Metoda:** przegląd kodu, sondy z nietypowymi danymi we wszystkich trzech językach (CSV ze średnikiem i przecinkiem
dziesiętnym, nazwy kolumn z `$`, umlautami i nazwami jak nagłówki tabel lub kolumny wyniku, tekst z podziałem
wierszy, typy nullable `boolean`/`Float64`/`Int64`), próby wstrzyknięcia kodu, testy w przeglądarce na prawdziwym
serwerze. Każdy potwierdzony błąd jest naprawiony i ma test regresyjny (`tests/test_audit3_regressions.py`).

**Wynik:** 7 naprawionych problemów – 1 związany z bezpieczeństwem, 1 średni, 5 drobnych.

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| A1 | wysoka (bezpieczeństwo) | Nazwy kolumn, wartości (filtr w „Gęstość 2D”), nazwy i definicje nowych kolumn oraz folder modelu trafiały bez zmian do komentarzy w pokazywanym i eksportowanym kodzie Pythona | podział wiersza w nazwie lub wartości z CSV zaczynał w kodzie nową linię: spreparowany plik mógł przemycić dowolne polecenie do skryptu, który uruchamia się w Jupyterze; w zwykłym przypadku skrypt kończył się błędem składni | komentarze z danymi zawsze w jednej linii (`codegen._cm`); test sprawdza wszystkie miejsca we wszystkich językach |
| B1 | średnia | Krok 10 po angielsku/polsku: nowe dane z kolumną o nazwie kolumny wyniku („Prediction”, „Note”, „Confidence” / „Prognoza”, „Uwaga”, „Pewność”), np. ponowna prognoza na pobranym wcześniej pliku wyników | strona kroku 10 kończyła się błędem „Duplicate column names” | jak po niemiecku przy „Vorhersage”: kolumny wyniku zastępują kolumny o tej samej nazwie; nowy komunikat (we wszystkich językach) wymienia zastąpione kolumny |
| C1 | niska | Jointplot i lmplot: tytuł nad rysunkiem | przy `fig.savefig()` w notatniku (bez `bbox_inches='tight'`) tytuł był obcięty | miejsce na tytuł wewnątrz rysunku |
| C2 | niska (od wersji 1, wszystkie wykresy) | Dwa znaki `$` w jednym tekście, np. „Kosten ($) gegen Preis ($)” | matplotlib traktował tekst między `$` jako wzór: znaki `$` znikały, reszta kursywą bez spacji | `text.parse_math = False` – `$` pokazywany dosłownie |
| C3 | niska | Tabele po angielsku/polsku tłumaczyły też nazwę indeksu – a tam stoi nazwa kolumny z danych (np. „Typ”, „Klasse” → „Type”, „Class”) | mylący nagłówek w tabelach liczności | nazwa indeksu bez tłumaczenia |
| C4 | niska | Krok 5: po zmianie języka już utworzony CSV (format międzynarodowy) zachowywał nazwę pliku w poprzednim języku | niespójna nazwa pobieranego pliku | język należy do klucza – CSV jest tworzony na nowo |
| C5 | niska | Odświeżenie strony przywracało język z adresu (`?lang=…`), a nie ostatni wybór w pasku bocznym | po F5 nagle inny język | wybrany język jest zapisywany w adresie; niemiecki bez `?lang` – bez zmian |

## Co sprawdzono i działa poprawnie

* **Katalog tłumaczeń** (774 teksty): każdy tekst z kodu ma tłumaczenie EN i PL, brak nieużywanych i zdublowanych
  kluczy, te same symbole zastępcze, spacje na początku/końcu, pogrubienia, backticki i podziały wierszy.
  Tłumaczenia nie są wykonywane przy imporcie modułów (stałe tylko oznaczone `N_`), więc zawsze obowiązuje bieżący
  język; wszystkie callbacki ustawiają język sesji; brak wątków roboczych, które zgubiłyby język.
* **Zmiana języka w trakcie pracy:** strona, wybrany wykres i ustawienia zostają (DE → EN → PL, także po kolejnym
  kliknięciu – poprawka z wersji 7); niezapisana zmiana w tabeli kroku 2 przetrwa zmianę języka i zostaje
  zastosowana (sprawdzone w przeglądarce).
* **Pełny przebieg kroków 1–10 z nietypowymi danymi** w DE/EN/PL: wszystkie wykresy (14), eksport dla Excela,
  trening, zapis, prognoza – bez błędów (po poprawkach).
* **Nowe wykresy:** typy nullable, `inf`, puste i stałe kolumny, dokładnie liniowe dane (KDE), ta sama kolumna
  jako x i y – zawsze komunikat zamiast błędu.
* **Wersja niemiecka bez zmian:** 38 wyników obliczeń i kodu identycznych jak w wersji 7; 76 widoków niemieckiej
  aplikacji identycznych (różni się tylko losowy rekord w kroku 8 i godzina zapisu modelu).

## Uwagi (bez zmian w silniku)

* Po niemiecku kolumny nowych danych o nazwie „Vorhersage”, „Hinweis” lub „P(…)” były i są zastępowane wynikiem –
  teraz aplikacja o tym informuje.
* Liczby w interfejsie mają we wszystkich językach kropkę dziesiętną (jak w pandas); przecinek dziesiętny dostaje
  CSV dla niemieckiego i polskiego Excela.

## Testy (stan po audycie 3)

190 testów (180 + 10 w `tests/test_audit3_regressions.py`) – wszystkie przechodzą na pandas 3.0 i 2.2; składnia
sprawdzona dla Pythona 3.10. Testy w przeglądarce: języki i eksport 19/19, zmiana języka 12/12, nowe wykresy 9/9,
język w adresie 7/7.
