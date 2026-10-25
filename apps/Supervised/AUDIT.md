# Audyt kodu – Data-Analysis-Template (Streamlit + Keras, od wersji 9: scikit-learn)

> Pięć audytów: **audyt 1** (poniżej), **audyt 2** – całe oprogramowanie po rozszerzeniach, **audyt 3** – kod wersji 6
> i 7, **audyt 4** – silnik scikit-learn z wersji 9, **audyt 5** – Gradient Boosting, podział czasowy i porównanie z
> Elo (na końcu pliku). Pomiędzy nimi: **wersja 6** – wersje językowe (angielska i polska) i sposób sprawdzenia, że
> wersja niemiecka się nie zmieniła, **wersja 7** – cztery nowe wykresy w kroku 3, oraz **wersja 9** – silnik
> supervised learning (scikit-learn, pięć modeli) zamiast Keras.

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

---

# Wersja 9 – silnik supervised learning (scikit-learn) zamiast deep learning (16.10.2026)

**Zakres:** wymiana silnika – zamiast sieci neuronowej (Keras/TensorFlow) pięć modeli supervised learning ze
scikit-learn. Model wybiera się w kroku 6; kolejność i opisy pochodzą z rankingu (Miejsce · Algorytm · Dlaczego? ·
Do czego najlepiej) – po polsku dosłownie, po niemiecku i angielsku przetłumaczone. Kroki 1–5 i 10, przygotowanie
danych, recepta (`preprocessing.json`), konfiguracje, języki i podział na pliki pozostały bez zmian. Kod modeli
oparty na notatnikach kursu (Decision Trees und Random Forests, Support Vector Machines, Logistische Regression,
K Nearest Neighbors, Lineare Regression) – tylko części z modelem (trening, GridSearchCV, metoda łokcia,
współczynniki), nie analiza i przygotowanie danych.

## Co się zmieniło

| Krok | Wersja 8 (Keras) | Wersja 9 (scikit-learn) |
|---|---|---|
| 6 · Model i trening | warstwy ukryte, aktywacje, loss, optimizer, epochs, batch_size, EarlyStopping | wybór modelu (ranking 1–5 z „Dlaczego?” i „Do czego najlepiej”, ranking także jako tabela), parametry modelu, walidacja, `class_weight='balanced'`, seed; kod scikit-learn na żywo; pasek postępu z krzywą na żywo |
| 7 · Trening a walidacja | loss a val_loss | krzywa walidacji najważniejszego parametru (błąd na treningu i walidacji, naiwna prognoza jako linia), ocena: przeuczenie / niedouczenie / brak sygnału / „uczenie na pamięć”; przycisk *Przejmij najlepszą wartość do kroku 6*; przy GridSearchCV mapa ciepła C × gamma i tabela wszystkich kombinacji |
| 8 · Ewaluacja | – | dotychczasowa ewaluacja bez zmian + **szczegóły modelu**: ważność cech w drzewach (kolumny dummy jednej kategorii zsumowane), rysunek drzewa (pojedyncze drzewo), współczynniki Linear Regression w oryginalnych jednostkach (jak `lm.coef_`) i po skalowaniu, Logistic Regression z ilorazem szans, wektory nośne SVM, dane KNN |
| 9 · Zapis i odczyt | `model.keras` / `.h5` | `model.skops` (domyślnie, bezpieczny) albo `model.joblib`; ZIP i skrypt całego pipeline’u jak dotąd |
| 10 · Prognoza | – | bez zmian – te same funkcje działają z modelem scikit-learn |

## Pięć modeli – co wzięto z kursu

| Model | Kod z kursu | W aplikacji | Krzywa w kroku 7 |
|---|---|---|---|
| 1 · Decision Trees + Random Forests | `DecisionTreeClassifier()`, `RandomForestClassifier(n_estimators=600)` | las albo jedno drzewo, `n_estimators` (domyślnie 100), `max_depth`, `min_samples_leaf`, `max_features`; także regresja | błąd według liczby drzew – las rośnie przez `warm_start` (wynik identyczny z jednym `fit`, test) – albo według `max_depth` |
| 2 · Support Vector Machines | `SVC()`, `GridSearchCV(SVC(), param_grid, refit=True)` z C = 0.1 … 100, gamma = 1 … 0.001 | `kernel`, `C`, `gamma`, `epsilon` (SVR), GridSearchCV z tą samą siatką i wyborem `cv`; także regresja (SVR) | błąd według C; mapa ciepła GridSearchCV |
| 3 · Logistic Regression | `LogisticRegression()` | `C`, `max_iter` (domyślnie 1000 – przy 100 ze scikit-learn często ConvergenceWarning); tylko klasyfikacja | błąd według C |
| 4 · K-Nearest Neighbors | `KNeighborsClassifier(n_neighbors=1)`, metoda łokcia `for i in range(1,40)` z `error_rate` | K, `weights`, odległość euklidesowa/Manhattan; także regresja | odsetek błędów dla K = 1 … 40 |
| 5 · Linear Regression | `LinearRegression()`, `lm.coef_`, tabela współczynników | `fit_intercept`; tylko regresja | krzywa uczenia (błąd według liczby wierszy treningowych) |

## Decyzje

1. **Model niepasujący do typu problemu** (Logistic Regression przy regresji, Linear Regression przy
   klasyfikacji): w kroku 6 przy opisie dopisek „tylko klasyfikacja / tylko regresja”, komunikat z propozycją i
   zablokowany przycisk treningu – zamiast błędu scikit-learn.
2. **Prawdopodobieństwa SVM** (próg, ROC-AUC, kolumna `P(…)` w kroku 10): `CalibratedClassifierCV(SVC(…),
   ensemble=False)` – `SVC(probability=True)` jest od scikit-learn 1.9 przestarzałe (FutureWarning, usunięcie w
   1.11). Kalibracja z 5 foldami, z mniejszą liczbą tylko przy bardzo rzadkich klasach. Krzywa w kroku 7 i
   GridSearchCV używają zwykłego `SVC` – jak w kursie.
3. **LogisticRegression bez `penalty`** – parametr przestarzały od scikit-learn 1.8 (domyślnie L2, siła przez `C`).
4. **Walidacja w kroku 7:** domyślnie „Dane testowe” – jak metoda łokcia w kursie; opcja „Część danych
   treningowych (czyściej)” zostawia zbiór testowy nietknięty aż do kroku 8. Błąd: klasyfikacja = odsetek błędów,
   regresja = MAE w oryginalnych jednostkach; linia przerywana = naiwna prognoza (najczęstsza klasa albo średnia).
5. **Szybkość:** krzywe mierzą błąd na najwyżej 5000 wierszy z każdego zbioru (KNN: 2000), trenowane jest zawsze na
   wszystkich wierszach. Metoda łokcia liczy wszystkie K z jednego wyszukiwania sąsiadów – te same prognozy co
   `predict()` dla każdego K osobno, z wyjątkiem remisu odległości na K-tym miejscu (wtedy kolejności nie ustala też
   scikit-learn); test sprawdza obie rzeczy. Las przy walidacji na danych testowych jest od razu gotowym modelem.
6. **„Najlepsza wartość”** jest proponowana tylko wtedy, gdy poprawa jest większa niż szum (błąd standardowy) – nie
   dla liczby drzew i krzywej uczenia (więcej drzew lub danych nie szkodzi).
7. **Zapis:** `.skops` wczytuje tylko znane typy scikit-learn (lista dozwolonych typów); `.joblib` (pickle) z
   przesłanego ZIP-a tylko po zaznaczeniu potwierdzenia; modele z wersji 8 (`.keras`/`.h5`) → jasny komunikat.
8. **float64 zamiast float32 w kroku 4.** Keras potrzebował float32, scikit-learn liczy w float64. Teraz macierz X
   w aplikacji jest bit w bit taka sama jak w pokazywanym kodzie, a modele dają te same wyniki co kod skopiowany
   do Jupytera (test z tolerancją 1e-9). Z float32 różnice rzędu 1e-7 zmieniały pojedyncze prognozy SVM, a etykieta
   regresji była zaokrąglana (ceny rzędu 500 000 do ok. 0,03). Koszt: macierz ML zajmuje dwa razy więcej pamięci
   (np. 396 000 × 78 → ok. 250 MB).
9. **Zmienne środowiskowe** `ML_TEMPLATE_LANG` i `ML_TEMPLATE_MODELS` (zamiast `DL_…`) – obie aplikacje mogą
   działać obok siebie z osobnymi folderami modeli. Bez ustawień aplikacja startuje po angielsku, jak w wersji 8
   (README podawało niemiecki – poprawione).
10. **Konfiguracje z wersji 8** wczytują się: kroki 2 i 4 są przejmowane, ustawienia Keras (warstwy, epoki …)
    pomijane bez komunikatu; nowe ustawienia modeli mają własne limity (np. najwyżej 5000 drzew, K ≤ 1000).
11. **Brakujące pliki w ZIP-ie** (`locales/`, `configs/`, `.streamlit/`): `locales/` przejęte z projektu ML
    (wersja 8) i uzupełnione o nowe teksty; `configs/lending_club_beispiel.json` (teraz z Random Forest: 100 drzew,
    `max_depth` 12, `min_samples_leaf` 50, lista cech jak w konfiguracji zapisanej w aplikacji) i
    `.streamlit/config.toml` odtworzone według README.

## Znalezione w teście w przeglądarce (poprawione)

| # | Problem | Skutek | Poprawka |
|---|---|---|---|
| E1 | Oś X krzywych w skali logarytmicznej (C przy SVM i Logistic Regression) | zamiast 0.01 … 100 pod osią stał tekst `$\mathdefault{10^{-2}}$` – skutek uboczny poprawki C2 z audytu 3 (`text.parse_math = False`) | etykiety jako liczby (0.01, 0.1, 1 …), ustawiona wartość przy linii, np. „ustawiono (0.37)”; test rysuje wszystkie wykresy kroków 7 i 8 i sprawdza etykiety osi |
| E2 | Ocena „błąd treningowy jest tu normalny” dla każdego lasu | przy `min_samples_leaf=50` (przykładowa konfiguracja) błąd treningowy 13,8 % był tłumaczony tym, że każde drzewo uczy się swojej próbki na pamięć | ta ocena tylko wtedy, gdy błąd treningowy wynosi najwyżej połowę błędu walidacji; inaczej zwykła ocena (bez przeuczenia / przeuczenie / niedouczenie) |
| E3 | Opis osi i tytuł krzywej (liczba drzew, wiersze treningowe) zapisywane w języku treningu | po zmianie języka po treningu wykres i tabela w kroku 7 częściowo po niemiecku | tłumaczenie dopiero przy wyświetlaniu; po polsku tytuł „Odsetek błędów a liczba drzew” zamiast niegramatycznego „według Liczba drzew” |
| E4 | Układ: etykieta „Krzywa walidacji do kroku 7” obcięta (krok 6), trzecia kolumna tabeli współczynników obcięta (krok 8) | gorsza czytelność | szersze kolumny |
| E5 | Opisy „najlepsza wartość” i „ustawiono” nachodziły na linie wykresu | gorsza czytelność | opis nad punktem, na jasnym tle; przy prawej krawędzi po lewej stronie linii |

## Testy (stan wersji 9)

* `tests/test_models.py` (56 testów): każdy model × każdy pasujący typ problemu (trening, krzywa, ocena, szczegóły
  modelu, zapis/odczyt `.skops` i `.joblib` → te same prognozy w kroku 10), modele niepasujące, las z
  `warm_start` = zwykły `fit`, GridSearchCV jak w kursie, oceny w kroku 7 (przeuczenie, uczenie na pamięć, brak
  sygnału), szybka metoda łokcia = `predict()` dla każdego K (także z remisami), współczynniki w oryginalnych
  jednostkach (3 skalery × skalowanie etykiety), odrzucanie nieznanych typów w `.skops`, `.joblib` z ZIP-a tylko po
  zgodzie, modele z wersji 8, **pokazywany kod modelu i krzywej liczy to samo co aplikacja** (8 kombinacji,
  tolerancja 1e-9), kod ewaluacji i zapisu działa, czytelne osie wszystkich wykresów kroków 7 i 8, test klikania
  regresji (Linear Regression, KNN z „najlepszą wartością”, SVM + GridSearchCV, zapis `.skops`, krok 10).
* Dotychczasowe testy dostosowane do nowego silnika (trening przez `ModelConfig`, formaty `.skops`/`.joblib`,
  klucze konfiguracji); testy przygotowania danych, wykresów, języków i audytów bez zmian merytorycznych.
* Razem **248 testów** – wszystkie przechodzą (Python 3.13, pandas 3.0, scikit-learn 1.9, skops 0.16,
  Streamlit 1.66); składnia sprawdzona dla Pythona 3.10. Uruchomienie: `pytest -q` (ok. 6 minut).
* Test w przeglądarce (prawdziwy serwer, Playwright, syntetyczne dane LendingClub i domów): kroki 1–10 po
  niemiecku ze wszystkimi pięcioma modelami (las z przykładowej konfiguracji, zapis `.skops`, prognoza z
  podsumowaniem; KNN z przejęciem najlepszej wartości; SVM + GridSearchCV z mapą ciepła; Linear Regression
  zablokowana przy klasyfikacji; Logistic Regression ze współczynnikami), kroki 6–8 po angielsku i po polsku bez
  niemieckich pozostałości, regresja po polsku (Linear Regression: krzywa uczenia, współczynniki) – **42/42** po
  poprawkach E1–E5.

## Aktualizacja do wersji 9

* `pip install -r requirements.txt` (nowe: `skops`; TensorFlow nie jest już potrzebny).
* Modele z wersji 8 (`model.keras`/`.h5`) trzeba wytrenować ponownie; konfiguracje (`.json`) działają dalej.
* Zmienne środowiskowe `DL_TEMPLATE_LANG` / `DL_TEMPLATE_MODELS` → `ML_TEMPLATE_LANG` / `ML_TEMPLATE_MODELS`.

---

# Audyt 4 – silnik supervised learning z wersji 9 (16.10.2026)

**Zakres:** cały kod powstały lub zmieniony w wersji 9: silnik w `pipeline.py` (konfiguracja modeli, trening,
krzywe walidacji, GridSearchCV, ocena w kroku 7, szczegóły modelu, zapis i wczytywanie `.skops`/`.joblib`, ZIP,
konfiguracje), kroki 6–9 w `app.py`, generowany i eksportowany kod w `codegen.py`, nowe wykresy w `plots.py`, nowe
tłumaczenia, przykładowa konfiguracja i testy.
**Metoda:** przegląd każdej nowej funkcji; sondy z macierzą 5 modeli × 3 typy problemu × opcje (warianty drzew,
kernele, GridSearchCV, `class_weight`, `weights`/`p`, `fit_intercept`) × walidacja „dane testowe”/„część treningowa”;
przypadki brzegowe (rzadka klasa, minimum 20 wierszy, stała etykieta, etykieta bool i liczbowa, jedna cecha, stałe
cechy, `n_estimators=1`, drzewo o głębokości 1); eksportowany skrypt uruchomiony dla każdego modelu; testy klikania
(AppTest) po niemiecku i po polsku; próby z przygotowanymi plikami ZIP, receptami i nazwami. Każdy potwierdzony błąd
jest naprawiony i ma test regresyjny (`tests/test_audit4_regressions.py`).

**Wynik:** 11 naprawionych problemów – 4 średnie, 7 drobnych (w tym 4 związane z bezpieczeństwem). Żaden nie
zmieniał wyników trenowanego modelu przy domyślnych ustawieniach.

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| B1 | średnia | „Przejmij najlepszą wartość do kroku 6” po zmianie modelu w kroku 6 (bez nowego treningu) | wartość trafiała do aktualnie wybranego modelu – np. C = 0.001 z Logistic Regression do SVM – a wytrenowany model nie dostawał nic | wartość zawsze dla wytrenowanego modelu (krzywa należy do niego); wybór modelu i wariantu drzewa wraca do niego |
| B2 | średnia | „Przejmij najlepszą wartość” przy SVM z GridSearchCV | przycisk nic nie zmieniał – przy następnym treningu GridSearchCV znowu zastępował C swoim wynikiem | GridSearchCV zostaje w tym celu wyłączony, gamma z wyszukiwania przejęte (krzywa była liczona z nim); komunikat to mówi |
| B3 | średnia | Pokazywany i eksportowany kod zaokrąglał liczby do 6 cyfr (`f"{C:g}"`) | przy C = 1.234567 kod trenował inny model niż aplikacja (różnice prognoz do 0,0015 przy SVR) | dokładna wartość (C, epsilon, próg, `test_size`); test zgodności z tolerancją 1e-9 |
| B4 | średnia | SVR z GridSearchCV przy małej liczbie wierszy (np. 13 wierszy treningowych, cv = 10) | foldy z 1 wierszem → R² = NaN dla wszystkich kombinacji → „najlepsza” kombinacja przypadkowa, a mapa ciepła w kroku 7 kończyła się błędem | przy regresji najwyżej n/2 foldów (co najmniej 2 wiersze w foldzie), przy klasach nie więcej niż wierszy najrzadszej klasy – ta sama liczba w komunikacie, w pokazywanym kodzie i w skrypcie; gdy mimo to brak poprawnej oceny → zrozumiały komunikat |
| C1 | niska | Kod krzywej (krok 7) bez ustawionej wartości spoza siatki (np. C = 1.234567, K = 50); przy bardzo małych danych K > liczba wierszy | inny przebieg krzywej niż w aplikacji; przy 20 wierszach kod KNN kończył się błędem | te same wartości co aplikacja: K do min(40, wiersze), ustawiona wartość dodana, krzywa uczenia co najmniej 2 wiersze |
| C2 | niska (bezpieczeństwo) | ZIP z `model.skops` i `model.joblib`, recepta wskazuje `.joblib` | aplikacja nie pytała o zgodę na pickle, a wczytanie kończyło się komunikatem „zaznacz pole”, którego nie było (kod nie był wykonywany) | pytanie o zgodę dotyczy dokładnie pliku, który zostanie wczytany – ta sama reguła co przy wczytywaniu (recepta może wskazać tylko `model.skops` albo `model.joblib`) |
| C3 | niska (bezpieczeństwo) | Nazwa modelu z obcej recepty i nazwa przesłanego ZIP-a trafiały do Markdown (krok 9, krok 10, pasek boczny) | np. `![](http://…)` mogło wczytać obcy obrazek (śledzenie otwarcia) | nieznany model → „?”, nazwy plików escapowane, etykieta w backtickach bez backticków i podziałów wiersza |
| C4 | niska (bezpieczeństwo) | Pokazywany kod wczytywania `.skops` ufał wszystkim typom z pliku | skopiowany do wczytywania cudzych plików omijał ochronę skops | ufa tylko typom `sklearn.*`, tak jak aplikacja – wszystko inne `sio.load` odrzuca |
| C5 | niska | Zapis modelu najpierw kasował poprzedni plik | gdy zapis się nie udał (brak miejsca, błąd skops), poprzedni model o tej nazwie był stracony | zapis do plików tymczasowych, potem podmiana; plik w drugim formacie usuwany dopiero po udanym zapisie |
| C6 | niska (bezpieczeństwo) | `.skops` to także ZIP – bez limitu rozmiaru po rozpakowaniu | spreparowany plik mógł zająć całą pamięć przy wczytywaniu | ten sam limit 2 GB co dla ZIP-a modelu |
| C7 | niska | Linear Regression bez `drop_first` (krok 4) | kolumny dummy jednej kategorii sumują się do 1 – ich współczynniki nie są jednoznaczne, a krok 8 opisywał je jak jednoznaczne | ostrzeżenie w kroku 8 z radą, by włączyć `drop_first` |

Drobne poprawki bez osobnego numeru: po polsku „Głębokość (średnio)” jak „Liście (średnio)”.

## Co sprawdzono i działa poprawnie

* **Macierz modeli** (13 konfiguracji, każda dla pasujących typów problemu, × 2 rodzaje walidacji): trening, krzywa,
  ocena w kroku 7, szczegóły modelu, zapis i wczytanie `.skops` oraz `.joblib` (także z `class_weight`), prognoza
  nowych danych = prognoza aplikacji; pokazywany kod modelu i krzywej liczy to samo co aplikacja (tolerancja 1e-9, po
  poprawkach B3 i C1).
* **Eksportowany skrypt** (krok 9) uruchomiony dla każdego modelu × typu problemu × formatu: model w skrypcie =
  model w aplikacji (1e-9), zapis i ponowne wczytanie działają.
* **Przypadki brzegowe:** rzadka klasa (SVM → zrozumiały komunikat, pozostałe modele działają), minimum 20 wierszy
  (każdy model, obie walidacje), stała etykieta regresji ze skalowaniem, etykieta bool i liczbowa, jedna cecha, stałe
  cechy, `n_estimators = 1`, drzewo o głębokości 1 – bez awarii, wszystkie błędy i wykresy skończone.
* **Wiele klas (Iris) w aplikacji:** każdy model przez kroki 6–10 po niemiecku i po polsku – bez błędów.
* **Bezpieczeństwo:** `.skops` z obcym typem → odmowa; `.joblib` z ZIP-a tylko po zgodzie; recepta nie może wskazać
  innego pliku niż `model.skops`/`model.joblib`; modele z wersji 8 → komunikat; konfiguracje z nieznanymi
  wartościami (model, kernel, gamma …) → wartości domyślne, liczby w dozwolonych granicach.
* **Teksty:** katalog tłumaczeń kompletny (904 teksty, te same symbole zastępcze), nowe teksty po angielsku i po
  polsku sprawdzone.

## Uwagi (bez zmian)

* `.joblib` z folderu `models/` wczytuje się bez pytania – to własny folder użytkownika; aplikacja jest przeznaczona
  do użytku lokalnego (README, „Nur lokal verwenden”).
* Metoda łokcia liczy wszystkie K z jednego wyszukiwania sąsiadów; przy remisie odległości na K-tym miejscu wybór
  sąsiada może się różnić od osobnego `predict()` – kolejności remisów nie ustala też scikit-learn (wersja 9, punkt 5).
* Kod krzywej dla pojedynczego drzewa używa stałej listy głębokości (1 … 20, None); aplikacja dobiera głębokości do
  głębokości pełnego drzewa – wynik dla tych samych głębokości jest ten sam.

## Testy (stan po audycie 4)

266 testów (248 + 18 w `tests/test_audit4_regressions.py`) – wszystkie przechodzą (Python 3.13, pandas 3.0,
scikit-learn 1.9, skops 0.16, Streamlit 1.66); składnia sprawdzona dla Pythona 3.10. 15 nowych testów odtwarza błędy
B1–B4 i C1–C7 – uruchomione na kodzie sprzed audytu wszystkie kończą się niepowodzeniem (sprawdzone); 3 testy
zabezpieczają poprawkę E3 z wersji 9 (opisy osi w bieżącym języku). Uruchomienie: `pytest -q` (ok. 5 minut).

---

# Audyt 5 – Gradient Boosting, podział czasowy i porównanie z Elo (25.10.2026)

**Zakres:** kod z aktualizacji z 24.10.2026 (commit `3f349fd`): Gradient Boosting
(`HistGradientBoostingClassifier`/`Regressor`, krzywa z `staged_predict`, szczegóły modelu, zapis `.skops`), podział
czasowy (`time_split`, walidacja na najnowszej części treningu, GridSearchCV z `TimeSeriesSplit`), rozpoznawanie
formatu narzędzia z danymi sportowymi (`match_data_layout`), kolumna porównawcza i porównanie w kroku 8
(`compare_probabilities`, wykres kalibracji), zmiany w krokach 2, 4 i 6–8 w `app.py`, generowany i eksportowany kod w
`codegen.py`, nowe teksty i tłumaczenia, przykład NBA w README i testy w `tests/test_sports_timesplit.py`.
**Metoda:** przegląd każdej nowej funkcji; sondy z prawdziwymi danymi z narzędzia „Datasety NBA i tenis” (NBA: 28 382
mecze, 144 kolumny; ATP: 78 874 mecze, 178 kolumn) przez logikę aplikacji i AppTest, interfejs po angielsku i po
polsku z syntetycznymi danymi w tym formacie; eksporty z narzędzia z odznaczonymi grupami (`post_*`, `inplay_*`);
przypadki brzegowe podziału czasowego (całe sezony, jedna wartość czasu, klasy tylko w najnowszych wierszach, mało
wierszy, `test_size` 0 i 1, liczby i daty, typy nullable); kolumna porównawcza z lukami, dla drugiej klasy, usunięta w
kroku 4, równa etykiecie; benchmark narzędzia odtworzony niezależnie od aplikacji; pomiary czasu kroku 4 na szerokich
danych; pełne testy także w środowisku jak w CI (pandas 3.1, numpy 2.5.4). Każdy potwierdzony błąd jest naprawiony i –
poza README (A3) i samym testem (C7) – ma test regresyjny (`tests/test_audit5_regressions.py`).

**Wynik:** 13 naprawionych problemów – 4 średnie (w tym błędny wniosek w README), 7 drobnych, 2 dotyczące szybkości.
Przy ustawieniach proponowanych dla danych z narzędzia (etykieta `home_win`/`p1_win`, klasa pozytywna `1`, kolumna
`pre_elo_prob_*`) model i porównanie liczyły poprawnie; błędy dotyczyły innych ustawień, eksportów bez niektórych
grup, komunikatów i szybkości – oraz wniosku w przykładzie NBA.

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| A1 | średnia | Kolumna porównawcza dla drugiej klasy: klasa pozytywna `0` (wygrana gości), a `pre_elo_prob_home` to szansa gospodarzy (klasa `1`) | aplikacja i tak proponowała Elo jako porównanie, a krok 8 ogłaszał „model lepszy od Elo” (dane NBA: log loss 0.62 wobec 0.94) – porównanie bez wartości i bez żadnej uwagi | Elo proponowane tylko przy klasie pozytywnej `1`; krok 4 liczy ROC-AUC kolumny porównawczej na wierszach treningowych (test zostaje nietknięty) – poniżej 0.5 ostrzeżenie w kroku 4 i w kroku 8 z radą, którą klasę wybrać |
| A2 | średnia | Podział czasowy bez obu klas albo z bardzo małym treningiem | np. 2 wiersze z 2020 i 118 z 2021, `test_size` 0.3 → cały 2021 w teście, trening na 2 wierszach, tylko uwaga „udział testu 98 %”; gdy we wczesnym okresie jest jedna klasa, Gradient Boosting „uczył się” modelu, który zawsze mówi `0` | co najmniej 10 wierszy treningowych i obie klasy w treningu – inaczej komunikat z radą (mniejszy `test_size`, dokładniejsza kolumna czasu, losowy podział ze stratify); przy wielu klasach ostrzeżenie o klasach bez wiersza treningowego |
| A3 | średnia | README (DE/EN/PL), przykład NBA: „Gradient Boosting, 300 rund – log loss 0.621 wobec 0.620 dla Elo, brak wyraźnej różnicy” | wniosek z przeuczonego modelu. Przy tych samych ustawieniach (walidacja „część danych treningowych”) krok 7 pokazuje przeuczenie i proponuje 50 rund – po przejęciu model jest lepszy od Elo: 0.613 wobec 0.620, różnica −0.006 ± 0.002 na mecz, prawie 3 błędy standardowe | przykład policzony od nowa i opisany krok po kroku (300 rund → propozycja z kroku 7 → wynik); dodane `test_size` 0.2 oraz wyniki benchmarku narzędzia i tenisa |
| A4 | średnia | Rozpoznawanie formatu wymagało kolumn `post_*`/`inplay_*` | eksport z narzędzia bez tych grup (można je odznaczyć) nie był rozpoznany: brak propozycji, a w kroku 2 proponowane „Data → rok” dla `date` – po jego zastosowaniu podział czasowy był możliwy już tylko według roku; z kolei obce dane z `pre_tax_income` i `post_code` uchodziły za dane sportowe | rozpoznawanie po etykiecie `home_win`/`p1_win` razem z kolumnami `pre_*`/`lineup_*`; `post_*`/`inplay_*` są opcjonalne |
| C1 | niska | Kolumna porównawcza z lukami i strategią „usuń kolumnę” w kroku 4 | pokazywany i eksportowany kod kroku 8 kończył się `KeyError` – kod kroku 4 usuwał z `df` kolumnę, z której krok 8 czyta porównanie | kod kroku 4 zostawia w `df` kolumnę czasu i kolumnę porównawczą (jako cechy i tak odpadają) |
| C2 | niska | Zmiana kolumny porównawczej po przygotowaniu danych | brak uwagi „zmienione od przygotowania” – krok 8 porównywał dalej ze starą kolumną | kolumna porównawcza należy do porównania ustawień w kroku 4 |
| C3 | niska | Komunikat, gdy granica podziału wypada na najstarszej wartości czasu (np. 100 wierszy z 2020, 20 z 2021, `test_size` 0.5) | „najnowsze wiersze mają tę samą wartość czasu co najstarsze” – nieprawda, a rada nie mówiła, jaki `test_size` zadziała | osobne komunikaty: jedna wartość czasu w całej kolumnie; granica na najstarszej wartości – z największym możliwym `test_size` (tu 0.16, sprawdzone, że działa); mniej niż 5 % wierszy po najstarszej wartości – rada, by wybrać dokładniejszą kolumnę czasu (suwak zaczyna się od 0.05) |
| C4 | niska | Kolumna porównawcza bez żadnej wartości w wierszach testowych; etykieta jako kolumna porównawcza | mylący komunikat „musi zawierać prawdopodobieństwa od 0 do 1”; etykieta jako „porównanie” z samą sobą była przyjmowana | jasne komunikaty dla obu przypadków |
| C5 | niska | `test_size` 0 albo 1 (tylko przez `pipeline.py` – suwak w aplikacji na to nie pozwala) | podział czasowy: `IndexError`; losowy: komunikat scikit-learn po angielsku | sprawdzenie na początku z komunikatem w języku aplikacji |
| C6 | niska | SVM z GridSearchCV przy podziale czasowym, gdy pierwsze foldy `TimeSeriesSplit` mają tylko jedną klasę | komunikat „za mało wierszy w foldzie”, choć przyczyną jest kolejność czasowa | komunikat wyjaśnia foldy `TimeSeriesSplit` (pierwsze uczą się tylko na najstarszych wierszach) |
| C7 | niska (testy) | Test klikania „Przejmij najlepszą wartość” dla Gradient Boosting był warunkowy | gdy krok 7 nic nie proponował, test przechodził bez naciśnięcia przycisku | ustawienia, przy których propozycja jest pewna (400 rund, `learning_rate` 1, `min_samples_leaf` 2); test sprawdza pokazywany kod, wartość w kroku 6 i wybór modelu |
| P1 | niska (szybkość) | Krok 4, pomoc „korelacja z kolumną z lukami” przy szerokich danych | wykres ze wszystkimi kolumnami liczbowymi (NBA 134, ATP 163 słupki – obraz wysoki na 47–57 cali) rysowany przy każdym kliknięciu w kroku 4: każde kliknięcie 4,6–5,0 s (NBA) i 6,3–7,8 s (ATP) | wykres pokazuje 30 najsilniejszych korelacji (tytuł to mówi), tabela obok – wszystkie; kliknięcie 1,7–1,8 s (NBA) i 1,8–2,3 s (ATP) |
| P2 | niska (szybkość) | Krok 4, propozycje strategii dla brakujących wartości | korelacje liczone kolumna po kolumnie (ATP: 67 kolumn z lukami × 49 kandydatów, 20 s) – pierwsze wejście do kroku 4 trwało 26 s | `suggest_missing_all`: kandydaci sprawdzani raz, wszystkie korelacje w jednym kroku – te same propozycje (test), 3 s; pierwsze wejście do kroku 4: 5 s |

Drobne poprawki bez osobnego numeru: liczba testów w głównym README (EN/DE/PL) nie była zaktualizowana 24.10 –
teraz Supervised 310, razem 956.

## Co sprawdzono i działa poprawnie

* **Gradient Boosting:** `n_iter_` zawsze = `max_iter` (bez early stopping), krzywa z `staged_predict` z jednego
  treningu = osobne treningi z mniejszą liczbą rund; szczegóły modelu, zapis i wczytanie `.skops` (zaufany tylko typ
  `TreePredictor` z scikit-learn) dla wszystkich typów problemu.
* **Podział czasowy:** trening zawsze starszy niż test, wiersze o tej samej wartości czasu razem, daty i liczby, typy
  nullable (`Int64`, `Float64`, `boolean`), wiersze bez wartości czasu usuwane; pokazywany i eksportowany kod daje te
  same `X_train`/`X_test`/`y` co aplikacja.
* **Prawdziwe dane z narzędzia:** NBA i ATP przez logikę aplikacji (kroki 2–8) i AppTest (kroki 1–4); migawka z
  narzędzia (mecze bez wyniku: id + `pre_*` + `lineup_*`) → prognoza w kroku 10 działa; interfejs po angielsku i po
  polsku z syntetycznymi danymi w formacie narzędzia (kroki 1–10) bez niemieckich pozostałości. Benchmark narzędzia
  odtworzony co do czwartego miejsca (trening do sezonu 2023, test od 2024, log loss: `pre_*` 0.6083, `pre_*` +
  `lineup_*` 0.6022, Elo 0.6085). Tenis (ATP, test 20 %): Gradient Boosting z wartościami domyślnymi 0.608 wobec Elo
  0.634.
* **Porównanie w kroku 8:** log loss, Brier i ROC-AUC = scikit-learn; błąd standardowy różnicy liczony parami; wiersze
  bez wartości pominięte; wykres kalibracji dla skrajnych danych (stałe prawdopodobieństwa, mało wierszy, jedna klasa w
  koszyku) – bez błędów i ostrzeżeń.
* **Teksty:** katalog tłumaczeń kompletny (989 tekstów, te same symbole zastępcze), bez nieaktualnych wpisów.

## Uwagi (bez zmian)

* Krzywa w kroku 7 przy walidacji „część danych treningowych” bierze przy podziale czasowym najnowsze wiersze treningu
  (`shuffle=False`); mecze z jednego dnia mogą trafić częściowo do części walidacyjnej – wpływ pomijalny (między
  treningiem a testem takie mecze zawsze zostają razem).
* Domyślna walidacja „dane testowe” (jak w kursie): kto wybiera najlepszą wartość według niej, ocenia krok 8 zbyt
  optymistycznie – krok 7 to mówi i poleca „część danych treningowych”; przykład NBA w README korzysta z tej czystej
  walidacji.
* Krzywa w kroku 7 mierzy błąd klasyfikacji (1 − accuracy), a porównanie w kroku 8 – log loss. Błąd klasyfikacji
  jest miarą zgrubną: na danych NBA log loss wyraźnie rośnie między 50 a 600 rundami (0.613 → 0.629), a błąd
  klasyfikacji zmienia się tylko w granicach szumu (0.353 → 0.358, tolerancja 0.007) – przy 600 rundach krok 7
  pokazuje przeuczenie, ale nie proponuje mniejszej wartości (przy 300 rundach proponuje 50). Krzywa log loss dla
  klasyfikacji binarnej byłaby czulsza – to zmiana projektu (kod w aplikacji, testy zgodności), do rozważenia.
* Kod wykresu „korelacja z kolumną” w kroku 3 rysuje wszystkie kolumny – w notatniku nie ma tego problemu z szybkością
  co w aplikacji (P1).
* Nazwy kolumn trafiają do tekstów Markdown bez escapowania – jak w całej aplikacji; to własne dane użytkownika, a
  aplikacja jest przeznaczona do użytku lokalnego.
* `HistGradientBoosting` liczy na wszystkich rdzeniach (OpenMP); na mocno obciążonym komputerze trening bywa
  wielokrotnie wolniejszy (zmierzone dla NBA: ok. 4 s bez obciążenia, ok. 5 min przy równolegle działających testach).

## Testy (stan po audycie 5)

310 testów (293 + 17 w `tests/test_audit5_regressions.py`) – wszystkie przechodzą (Python 3.13, pandas 3.0.5 oraz
3.1.0 jak w CI, numpy 2.5, scikit-learn 1.9.1, skops 0.16, Streamlit 1.66); składnia sprawdzona dla Pythona 3.10. Nowe
testy odtwarzają błędy A1, A2, A4, C1–C6, P1 i P2 – uruchomione na kodzie sprzed audytu wszystkie kończą się
niepowodzeniem (sprawdzone); C7 to poprawiony test klikania w `tests/test_sports_timesplit.py`. Uruchomienie:
`pytest -q` (ok. 5–8 minut).
