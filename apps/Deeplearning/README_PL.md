# Data-Analysis-Template (Streamlit + Keras)

Aplikacja Streamlit do danych tabelarycznych CSV: wczytanie, przygotowanie, wizualizacja, trening sieci
neuronowej (Keras), ewaluacja i zastosowanie do nowych danych – dla dowolnych zbiorów (klasyfikacja binarna,
wieloklasowa, regresja). Do każdego kroku aplikacja pokazuje odpowiedni kod w Pythonie.

Aplikacja jest dostępna **po niemiecku, angielsku i polsku** – zob. [Języki](#języki).
(Deutsch: `README.md` · English: `README_EN.md`)

## Uruchomienie

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py -- --lang pl  # otwiera http://localhost:8501 po polsku
```

Przetestowano z Python 3.13, TensorFlow 2.21 (Keras 3.15), pandas 3.0 i 2.2, scikit-learn 1.9, Streamlit 1.65.
Zalecane: Python 3.10–3.13.

## Języki

Język można w każdej chwili zmienić na górze paska bocznego: **Deutsch · English · Polski**. Zmieniają się
tylko teksty – obliczenia są identyczne we wszystkich trzech językach (oryginałem jest wersja niemiecka).

| Start od razu w wybranym języku | |
|---|---|
| wiersz poleceń | `streamlit run app.py -- --lang pl` (albo `en`, `de`) – uwaga na `--` przed `--lang` |
| adres w przeglądarce | `http://localhost:8501/?lang=pl` |
| zmienna środowiskowa | `DL_TEMPLATE_LANG=pl` (Windows: `set DL_TEMPLATE_LANG=pl`) |

Bez tych ustawień aplikacja startuje po angielsku. Język wybrany w pasku bocznym trafia też do adresu
(`?lang=…`), więc zostaje po odświeżeniu strony.

**Tłumaczone są:** wszystkie teksty interfejsu (strony, przyciski, podpowiedzi, komunikaty, ostrzeżenia), tytuły
i opisy osi wykresów, nagłówki tabel, komentarze i przykładowe nazwy plików w generowanym kodzie Pythona,
podsumowanie w kroku 10, nazwy pobieranych plików oraz nazwy kolumn w pliku CSV z prognozami (Prognoza, Pewność,
Uwaga).

**We wszystkich językach takie same** – dzięki temu konfiguracje, modele i wygenerowany kod działają we
wszystkich trzech wersjach:

* Wartości wewnętrzne: akcje i strategie są zapisywane pod nazwami niemieckimi w konfiguracjach (`.json`) i w
  `preprocessing.json`. Konfigurację albo model zapisany w wersji polskiej można wczytać w wersji angielskiej lub
  niemieckiej i odwrotnie.
* Wartości, które aplikacja wpisuje do danych: `Sonstige` (grupa „reszta” przy „Zachowaj top N”), `fehlend`
  (osobna kategoria dla braków) oraz końcówki kolumn `_jahr` / `_monat` („Data → rok + miesiąc”).
* W „Przypisz wartości” wartość domyślną dla pozostałych wartości można zapisać jako `rest=…` albo `sonst=…` –
  w każdym języku.
* Komentarze w kodzie źródłowym (`app.py`, `pipeline.py`, …) są po niemiecku.

Po zmianie języka w trakcie pracy komunikaty, które już powstały (np. dziennik kroku 2 albo 4), pozostają w
poprzednim języku, dopóki dany krok nie zostanie wykonany ponownie. Teksty pochodzące z samego Streamlita (np.
„Upload” w polu pliku albo menu ⋮) są w każdej wersji po angielsku.

### Format Excela w kroku 5 (Zapis danych)

| Język | Opcja dla Excela | Separator | Znak dziesiętny |
|---|---|---|---|
| Deutsch | CSV für deutsches Excel | `;` | `,` |
| Polski | CSV dla polskiego Excela | `;` | `,` |
| English | CSV for Excel – UK, US, Australia … | `,` | `.` |

Polski Excel (polskie ustawienia regionalne Windows) używa – tak jak niemiecki – średnika jako separatora i
przecinka jako znaku dziesiętnego. Wszystkie warianty dla Excela są zapisywane jako UTF-8 z BOM (Excel poprawnie
pokazuje polskie znaki: ą, ć, ę, ł, ń, ó, ś, ź, ż) i są chronione przed CSV injection (zob. niżej). Opcja „CSV
międzynarodowy (przecinek, kropka dziesiętna)” jest taka sama we wszystkich językach.
Excel bierze separator z ustawień regionalnych Windows, a nie z języka Excela: jeśli plik otwiera się w jednej
kolumnie, wybierz drugi format.

## Szybki start

1. **Krok 1:** prześlij plik CSV albo wpisz ścieżkę → *Wczytaj dane*
2. **Krok 2:** sprawdź akcje dla kolumn → *Zastosuj*; w razie potrzeby utwórz nowe kolumny
3. **Krok 3:** wybierz kolumny do wykresów i obejrzyj dane
4. **Krok 4:** wybierz etykietę i cechy → *Przygotuj dane do ML*
5. **Krok 6:** *Rozpocznij trening* → potem 7, 8, 9, 10

Dla publicznego zbioru LendingClub (`lending_club_loan_two.csv`) w kroku 2 jest **przykładowa konfiguracja**
(kroki 2, 4 i 6 gotowo ustawione). Na wolnym komputerze włącz w pasku bocznym **próbkę** (`df.sample(frac=0.1)`).

## Kroki

| Strona | Zawartość |
|---|---|
| 1 · Dane i biblioteki | wczytanie CSV (przesłanie albo ścieżka; separator `;` i przecinek dziesiętny są rozpoznawane automatycznie), przegląd, wersje bibliotek, opcjonalnie opisy kolumn |
| 2 · Przygotowanie 1 – kolumny | jedna akcja na kolumnę (Zachowaj, Usuń, Wyodrębnij liczbę, Data → rok / miesiąc / **rok + miesiąc**, pierwsze/ostatnie N znaków, Zamień wartości, Zachowaj top N …) · **nowe kolumny**: Przypisz wartości, Formuła, Warunek → 0/1, Kopia + akcja |
| 3 · Analiza i wizualizacja | **wybór kolumn** · Przegląd, Wykres liczności, Histogram, Wykres pudełkowy, Wykres punktowy, Korelacja (heatmapa), Korelacja z jedną kolumną, Pairplot, Udział według kategorii, Statystyki grup, **Jointplot** (scatter/kde/hex/reg), **Gęstość 2D (KDE)** z filtrem na jedną grupę, **Regresja (lmplot)** z kolorem (hue) i podwykresami (col), **Histogram według grup** (nałożone) |
| 4 · Przygotowanie 2 – dane ML | etykieta + typ problemu, cechy, brakujące wartości dla każdej kolumny (z pomocą korelacji przy średniej grupowej); **najpierw** `train_test_split`, potem wartości uzupełniające, kolumny dummy (przegląd nowych kolumn 0/1) i skalowanie – uczone tylko na danych treningowych |
| 5 · Zapis danych | dane po kroku 2 albo gotowe do ML po kroku 4 (z kolumną `split`) jako CSV – międzynarodowy albo dla Excela w języku aplikacji; istniejący plik jest nadpisywany dopiero po potwierdzeniu |
| 6 · Model i trening | warstwy ukryte (neurony, aktywacja, dropout), aktywacja wyjścia, loss, optimizer, epochs, batch_size, EarlyStopping/patience – odpowiedni kod Keras jest pokazywany na bieżąco |
| 7 · loss a val_loss | krzywe + interpretacja (przeuczenie, EarlyStopping) |
| 8 · Ewaluacja | classification_report, confusion_matrix, ROC-AUC, próg, pojedynczy przypadek, ważność cech · regresja: MAE, RMSE, R² |
| 9 · Zapis i odczyt modelu | `models/<nazwa>/model.keras` (albo `.h5`) + `preprocessing.json`, pobieranie ZIP, cały pipeline jako skrypt Pythona; opcjonalnie sumy kontrolne wierszy treningowych |
| 10 · Prognoza dla nowych danych | nowy CSV albo formularz → to samo przygotowanie → prognoza, interpretacja, pobranie CSV i **podsumowanie** (jakość danych, co mówi wynik, zalecenie) |

Każdy krok ma pole **„Kod Pythona do tego kroku”** – wszystko można przenieść 1:1 do notatnika Jupyter.
Ustawienia (łącznie z nowymi kolumnami) można zapisać w pasku bocznym w **Zapis / odczyt konfiguracji** jako JSON
i wczytać ponownie dla nowych danych z tymi samymi kolumnami. Przy wczytywaniu plik jest sprawdzany: wpisy
bezużyteczne (zły typ, poza dozwolonym zakresem, np. 10 mln neuronów) są pomijane i wymieniane w komunikacie.

### Zapis danych (krok 5)

* Do prognoz w kroku 10 zawsze używaj **oryginalnego** pliku CSV, a nie eksportu: zapisany model sam powtarza
  przygotowanie z kroku 2 (np. „Zamień wartości 1=2; 2=3” zostałoby inaczej zastosowane dwa razy).
* Formaty dla Excela: teksty zaczynające się od `=`, `+`, `-` albo `@` dostają z przodu `'` – inaczej Excel
  wykonałby je jako formułę (CSV injection). W formacie międzynarodowym dane pozostają bez zmian.
* Jeśli jakaś cecha sama nazywa się `split`, kolumna train/test nazywa się `split_train_test`.

## Nowe kolumny (krok 2)

| Rodzaj | Przykład | Wynik |
|---|---|---|
| Przypisz wartości | z `loan_status`: `Fully Paid=1; Charged Off=0` | nowa kolumna `loan_repaid` (1/0); niewymienione wartości pozostają puste albo `rest=…` |
| Formuła | `loan_amnt / annual_inc` | liczba w każdym wierszu; dozwolone są `+ - * / ** // %`, nawiasy, `abs log log10 sqrt exp round` |
| Warunek → 0/1 | `dti > 30 and home_ownership == 'RENT'` | 1 = spełniony, 0 = niespełniony, puste przy brakach |
| Kopia + akcja | `earliest_cr_line` + „Data → rok” | oryginał zostaje, wynik jako dodatkowa kolumna |

Nazwy kolumn ze spacjami lub znakami specjalnymi zapisuj w formułach w \`backtickach\`. Formuły są sprawdzane:
tylko obliczenia, porównania i wymienione funkcje – żadnego dowolnego kodu Pythona. Aby żadna formuła nie mogła
zablokować programu, obowiązują limity: maksymalnie 1000 znaków i 100 poziomów zagnieżdżenia, wykładnik przy `**`
od −100 do 100 (albo kolumna), `round(x, n)` z n od 0 do 10, tekst w cudzysłowie tylko w porównaniach. Jeśli w
„Przypisz wartości” nie pasuje ani jedna wartość (np. inna wielkość liter), komunikat wymienia istniejące wartości.
Jeśli nowa kolumna jest używana przez późniejszą, obie można usunąć tylko razem. Nowe kolumny należą do zapisanej
recepty i w kroku 10 są automatycznie liczone dla nowych danych. Nowa kolumna utworzona z tej samej kolumny CSV co
etykieta (np. `loan_repaid` z `loan_status`) nie jest proponowana jako cecha (wyciek danych).

## Krok 10: prawidłowe traktowanie nowych danych

* **Nic nie jest dopasowywane od nowa.** Nowe dane przechodzą przez zapisaną *receptę* (`preprocessing.json`):
  te same akcje kolumn i nowe kolumny, wartości uzupełniające, kolumny dummy w tej samej kolejności, ten sam
  skaler, to samo kodowanie etykiety i próg.
* **Te same surowe kolumny co w treningu** – to model określa cechy. Aplikacja pokazuje wymagane kolumny i
  oferuje szablon CSV. Brakujące kolumny → komunikat o błędzie; dodatkowe kolumny (także etykieta) są pomijane.
* **Odchylenia** są obsługiwane i wyjaśniane w **podsumowaniu**: braki → wartości z treningu (każdy wiersz
  potrzebuje wartości do prognozy – także tam, gdzie w treningu usunięto wiersze z brakami), nieznane kategorie →
  wszystkie dummy 0, wartości spoza zakresu treningowego → uwaga.
* **Rozpoznawanie danych treningowych:** recepta zawiera sumy kontrolne (hashe) próbki do 3000 wierszy
  treningowych. Jeśli wiersze treningowe znajdują się w nowych danych, podsumowanie podaje ich udział – wskaźniki
  na takich danych są zbyt optymistyczne. Z sumy kontrolnej nie da się odtworzyć wiersza; kto ma dane oryginalne,
  może jednak sprawdzić, czy dany wiersz był w treningu. Dlatego przy danych poufnych przed przekazaniem modelu
  wyłącz w kroku 9 *Zapisz sumy kontrolne wierszy treningowych* (sprawdzenie jest wtedy pomijane).
* Wiersze z nieskończoną etykietą (`inf`) są usuwane w kroku 4 i nie są oceniane w kroku 10.
* Jeśli nowe dane mają już kolumny o nazwie kolumny wyniku (np. `Prognoza` lub `Uwaga` z wcześniejszego eksportu),
  w wyniku zostają one zastąpione – aplikacja je wymienia.
* Jeśli nowe dane zawierają etykietę, są automatycznie ewaluowane i porównywane z testem z treningu.

Bez aplikacji:

```python
import i18n
import pipeline as pl

i18n.set_lang("pl")  # komunikaty i podsumowanie po polsku (domyślnie: niemiecki)
model, recipe = pl.load_bundle("models/moj_model")
df_new, _ = pl.read_csv("nowe_dane.csv")
result, report, evaluation = pl.predict_new(model, recipe, df_new)
podsumowanie = pl.assess_prediction(recipe, result, report, evaluation)
```

Poza aplikacją nazwy kolumn w `result` pozostają niemieckie (`Vorhersage`, `Sicherheit`, `Hinweis`) – aplikacja
tłumaczy je tylko do wyświetlenia i pobrania.

## Uwagi do aktualnych wersji bibliotek

Starsze poradniki i notatniki często zawierają linie, które z aktualnymi wersjami już nie działają:

| starszy kod | problem dziś | w szablonie |
|---|---|---|
| `model.predict_classes(X)` | usunięte od TensorFlow 2.6 | `(model.predict(X) > 0.5).astype("int32")` |
| `model.save('x.h5')` z `loss='mse'` | Keras 3 nie potrafi już wczytać pliku | `.keras` jako standard, pełne nazwy loss, `load_model(..., compile=False)` |
| `df.corr()` | pandas ≥ 2: błąd przy kolumnach tekstowych | `df.corr(numeric_only=True)` |
| `df.select_dtypes(['object'])` | pandas 3: tekst ma typ `str`; `'object'` znajduje go już tylko przejściowo | sprawdzanie typu przez `is_numeric_dtype` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False → błąd Keras | `dtype=float` |
| `df.groupby('total_acc').mean()['mort_acc']` | pandas ≥ 2: błąd przy kolumnach tekstowych | `df.groupby('total_acc')['mort_acc'].mean()` |
| `sns.distplot` | przestarzałe | `sns.histplot` |
| `sns.kdeplot(df['a'], df['b'], shade=True, shade_lowest=False)` | seaborn ≥ 0.12: błąd, bez nazwy można przekazać tylko `data`; `shade`/`shade_lowest` przestarzałe | `sns.kdeplot(x='a', y='b', data=df, fill=True, thresh=0.05)` |
| `plt.figure(figsize=(11, 7))` przed `sns.lmplot(...)` | nic nie zmienia – lmplot tworzy własną figurę (w notatniku: „Figure … with 0 Axes”) | `sns.lmplot(..., height=4, aspect=1.15)` |
| uzupełnianie braków, **potem** `train_test_split` | wartości uzupełniające zawierają informację z danych testowych (wyciek danych) | najpierw podział; wartości uzupełniające, kategorie i skaler tylko z danych treningowych |

Dodatkowe opcje (każdą można wyłączyć): `restore_best_weights`, `stratify=y`, regulowany próg + ROC-AUC,
`class_weight='balanced'`, walidacja na części danych treningowych zamiast na danych testowych, skalowanie
etykiety przy regresji, wiele klas (softmax), ostrzeżenie o wycieku danych, permutation importance.

## Pliki

```
app.py                  interfejs Streamlit (tylko UI)
pipeline.py             logika obliczeń – do użycia także w notatniku (import pipeline as pl)
plots.py                wykresy (matplotlib/seaborn), zwracają obiekty Figure
codegen.py              tworzy pokazywany lub eksportowany kod Pythona
i18n.py                 języki: tx() tłumaczy niemiecki tekst na bieżący język
locales/                tłumaczenia {niemiecki: (angielski, polski)}, jeden plik na moduł
configs/                lending_club_beispiel.json (przykładowa konfiguracja LendingClub)
.streamlit/config.toml  limit przesyłania 1 GB, jasny motyw
tests/                  testy pytest + syntetyczne dane testowe
AUDIT.md                raport z audytu (po polsku): znalezione błędy, poprawki, testy
README.md               niemiecki · README_EN.md angielski · README_PL.md polski
models/                 powstaje przy zapisie (krok 9)
exports/                domyślny folder dla „Zapisz w folderze” (krok 5)
```

Rozbudowa:
* nowa akcja kolumny → `ACTIONS` + `_transform_column()` w `pipeline.py` (+ `_action_lines()` w `codegen.py`)
* nowy rodzaj nowych kolumn → `NEW_KINDS` + `apply_new_columns()` w `pipeline.py` (+ `_new_column_lines()` w
  `codegen.py`)
* nowy wykres → funkcja w `plots.py` + wpis w `CHARTS` / `page_viz()` w `app.py` (+ `chart_code()` w
  `codegen.py`)
* nowy szablon warstw → `PRESETS` w `pipeline.py`
* nowy lub zmieniony tekst → napisz go po niemiecku w `tx("…")` (stałe: `N_("…")`) i dodaj tłumaczenie
  angielskie i polskie w odpowiednim pliku w `locales/`; `pytest tests/test_i18n.py` zgłasza brakujące
  tłumaczenia.

## Testy

```bash
pip install pytest
pytest -q                                  # 190 testów, ok. 4–5 minut
python tests/synthetic.py --rows 20000     # syntetyczny przykładowy CSV (bez prawdziwych danych)
```

| Plik | sprawdza |
|---|---|
| `tests/test_pipeline.py` | przykładowy pipeline, zapis/odczyt (.keras/.h5), nowe dane, regresja, wiele klas, eksportowany skrypt |
| `tests/test_codegen_parity.py` | pokazywany kod Pythona liczy **dokładnie** to samo co aplikacja (X_train, X_test, y) – także w przypadkach brzegowych i z nowymi kolumnami |
| `tests/test_new_features.py` | rok + miesiąc, nowe kolumny (łącznie z ochroną przed dowolnym kodem w formułach), zapis danych, podsumowanie |
| `tests/test_audit_regressions.py` | po jednym teście na każdy błąd z pierwszego audytu (zob. `AUDIT.md`) |
| `tests/test_audit2_regressions.py` | drugi audyt: limity formuł, sprawdzanie importowanych konfiguracji, `inf` w etykiecie, eksport (Excel, kolumna `split`), wykresy z pustymi kolumnami, rozmiar ZIP, podsumowanie |
| `tests/test_app.py` | test klikania interfejsu (wszystkie 10 kroków, nowe kolumny, eksport, podsumowanie; etykieta bool, CSV ze średnikiem, ochrona przed nadpisaniem; zależne nowe kolumny, bezużyteczne wartości, błędy przy rysowaniu, model bez sum kontrolnych) |
| `tests/test_audit3_regressions.py` | trzeci audyt: żaden kod z nazw kolumn/wartości w pokazywanym kodzie Pythona, krok 10 z kolumnami o tej samej nazwie, tytuły wykresów (notatnik, `$`), zmiana języka (tabele, eksport, adres) |
| `tests/test_charts.py` | nowe wykresy w kroku 3 (jointplot, gęstość 2D, lmplot, histogram według grup): nietypowe dane, pokazywany kod działa, test klikania |
| `tests/test_i18n.py` | języki: każdy tekst ma tłumaczenie angielskie i polskie z tymi samymi symbolami zastępczymi, niemiecki bez zmian, CSV dla Excela zależnie od języka, wygenerowany kod liczy to samo w każdym języku, cała aplikacja po angielsku i po polsku bez niemieckich pozostałości |

## Uwagi

* Bardzo duże pliki CSV lepiej wczytywać przez „Ścieżka na tym komputerze” niż przesyłać.
* W trakcie treningu niczego nie klikaj – każda zmiana uruchamia skrypt od nowa i przerywa trening.
* GPU: pod Windows TensorFlow natywnie używa tylko CPU (GPU przez WSL2). Krok 1 pokazuje, czy znaleziono GPU.
* **Używaj tylko lokalnie.** Pole ścieżki odczytuje każdy plik, do którego zalogowany użytkownik ma dostęp, a
  `models/` jest wspólny dla wszystkich użytkowników aplikacji. Dlatego nie udostępniaj aplikacji w sieci bez
  logowania. Modele (ZIP/.h5/.keras) wczytuj tylko z zaufanych źródeł (ZIP po rozpakowaniu może mieć maksymalnie
  2 GB; formuły w recepcie są przy wczytywaniu sprawdzane ponownie, zapisany kod nigdy nie jest wykonywany).
* **Wskaźniki testowe są lekko optymistyczne**, jeśli dane testowe służą też jako walidacja dla EarlyStopping
  (domyślnie „Walidacja: Dane testowe”) albo jeśli próg w kroku 8 jest ustawiany na danych testowych. Dla
  uczciwej oceny wybierz w kroku 6 „Walidacja: Część danych treningowych (czyściej)”.
* **Brak wycieku danych w kroku 4:** wartości uzupełniające, kategorie (kolumny dummy) i skaler są uczone po
  podziale tylko na danych treningowych. Kategoria występująca tylko w danych testowych jest traktowana jak
  nieznana wartość w nowych danych (wszystkie dummy 0). Ze wszystkich wierszy (przed podziałem) powstają już tylko
  lista dla „Zachowaj top N” w kroku 2 i automatyczne propozycje w tabelach – oba z niewielkim wpływem.
