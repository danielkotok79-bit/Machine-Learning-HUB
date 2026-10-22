# DATA-Analysis-NLP (Streamlit + scikit-learn, klasyfikacja tekstów)

Aplikacja Streamlit dla danych CSV z tekstami: wczytywanie, przygotowanie, wizualizacja, zamiana tekstów na liczby
(**bag of words** z `CountVectorizer` albo **TF-IDF**) i klasyfikacja trzema modelami – **naiwny Bayes
(MultinomialNB)**, **regresja logistyczna** i **liniowa SVM (LinearSVC)**. Następnie aplikacja pokazuje, które słowa
przemawiają za którą klasą, wyjaśnia pojedyncze prognozy i klasyfikuje nowe teksty. Do każdego kroku pokazuje
odpowiedni kod Python.

Wzorem są części kursu poświęcone NLP: **projekt Yelp** (recenzje z 1 i 5 gwiazdkami, `CountVectorizer` +
`MultinomialNB`, `test_size=0.3`, `random_state=101`) oraz **lekcja o spamie SMS** (`SMSSpamCollection`, rozdzielany
tabulatorami, bez wiersza nagłówka, potok z `CountVectorizer`, `TfidfTransformer` i `MultinomialNB`). Dla obu
przypadków jest gotowa przykładowa konfiguracja.

Aplikacja powstała z wersji supervised (DATA-Analysis-SupervisedLearning): kroki 1–3 (wczytywanie, akcje kolumn,
nowe kolumny, wykresy), przepis (`preprocessing.json`), zapis/odczyt, konfiguracje i języki pozostały takie same;
nowe są: rozpoznawanie plików bez wiersza nagłówka, akcje i wykresy tekstowe, krok 4 (tekst, etykieta, klasy,
wektoryzacja), trzy modele tekstowe z porównaniem, analiza słów w kroku 8 i klasyfikacja nowych tekstów w kroku 10.

Aplikacja jest dostępna **po niemiecku, angielsku i polsku** – zob. [Języki](#języki).
(Deutsch: `README.md` · English: `README_EN.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # otwiera http://localhost:8501
```

Przetestowano z Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 lub nowszy.
Wersja NLP nie potrzebuje dodatkowych pakietów (bez nltk – stop words pochodzą ze scikit-learn).

## Języki

Język można w każdej chwili przełączyć na górze paska bocznego: **Deutsch · English · Polski**. Zmieniają się tylko
teksty – obliczenia są identyczne we wszystkich trzech językach (oryginałem jest niemiecki).

| Start od razu w danym języku | |
|---|---|
| Wiersz poleceń | `streamlit run app.py -- --lang pl` (albo `en`, `de`) – `--` przed `--lang` jest konieczne |
| Adres w przeglądarce | `http://localhost:8501/?lang=pl` |
| Zmienna środowiskowa | `NLP_TEMPLATE_LANG=pl` (Windows: `set NLP_TEMPLATE_LANG=pl`) |

Bez wskazania aplikacja startuje po angielsku. Język wybrany w pasku bocznym pojawia się też w adresie (`?lang=…`),
dzięki czemu zostaje po odświeżeniu strony.

Tłumaczone są wszystkie teksty interfejsu, tytuły i osie wykresów, nagłówki tabel (także w eksporcie słownika i
prognoz dla danych testowych), komentarze i przykładowe nazwy plików w generowanym kodzie Python, podsumowanie w
kroku 10 oraz nazwy pobieranych plików. We wszystkich językach takie same pozostają:

* wartości wewnętrzne – akcje są zapisane w konfiguracjach (`.json`) i w `preprocessing.json` pod niemiecką nazwą;
  dzięki temu konfiguracje i modele działają we wszystkich trzech językach;
* wartości, które aplikacja wpisuje do danych: `Sonstige` (top N), `_jahr` / `_monat` (data → rok + miesiąc),
  `Spalte_1`, `Spalte_2` … (plik bez wiersza nagłówka) oraz kolumna `Hinweis` w wyniku logiki obliczeń
  (w interfejsie i w pobranym pliku przetłumaczona);
* przykładowe teksty w generowanym kodzie („The food was great!”) – dane z kursu są po angielsku;
* komentarze w kodzie źródłowym (po niemiecku).

Jeśli język zmieni się w trakcie pracy, już wygenerowane komunikaty (np. protokół kroku 2 lub 4) zostają w starym
języku, dopóki krok nie zostanie wykonany ponownie. Teksty samego Streamlit (np. „Upload” w polu pliku albo menu ⋮)
są w każdej wersji po angielsku.

Format Excela w krokach 5, 8 i 10 zależnie od języka: niemiecki i polski `;` i przecinek dziesiętny, angielski `,` i
kropka dziesiętna – zawsze UTF-8 z BOM i z ochroną przed CSV injection (w kroku 5 także w generowanym kodzie
Python). Teksty z obcych źródeł (recenzje, wiadomości) mogą zaczynać się od `=`, `+`, `-` albo `@` – Excel
wykonałby je inaczej jako formułę.

## Szybki start – jak na kursie

**Yelp (recenzje, 1 kontra 5 gwiazdek)**

1. **Krok 1:** wczytaj `yelp.csv`
2. **Krok 2:** *Wczytaj przykładową konfigurację* → *Zastosuj* (usuwa identyfikatory, nowa kolumna `text length`)
3. **Krok 3:** *Długość tekstu* według `stars` (jak `sns.FacetGrid(yelp, col='stars')`), *Najczęstsze słowa* w
   każdej klasie, *Średnie w grupach* (jak `yelp.groupby('stars').mean()`)
4. **Krok 4:** *Wektoryzuj teksty* (tylko 1 i 5 gwiazdek, CountVectorizer, `test_size=0.3`, `random_state=101`)
5. **Krok 6:** *Rozpocznij trening* (MultinomialNB) i *Porównaj modele* → kroki 7, 8 (najważniejsze słowa, błędy,
   własny tekst), 9, 10

**Spam SMS**

1. **Krok 1:** wczytaj `SMSSpamCollection` – tabulator i brak wiersza nagłówka zostaną rozpoznane (kolumny
   `Spalte_1`, `Spalte_2`)
2. **Krok 2:** *Wczytaj przykładową konfigurację* → *Zastosuj* (zmiana nazw na `label` i `message`, nowa kolumna
   `length` jak `messages['message'].apply(len)`)
3. **Krok 4:** TF-IDF z angielskimi stop words, `test_size=0.2` → krok 6: naiwny Bayes (albo LinearSVC z
   `class_weight='balanced'`)

Dla własnych danych: w kroku 4 wybierz kolumnę tekstu, etykietę i opcjonalnie niektóre klasy; w kroku 6 model –
porównanie pokaże, który pasuje najlepiej. Na wolnym komputerze włącz po lewej **próbkę** (`df.sample(frac=0.1)`).

## Kroki

| Strona | Zawartość |
|---|---|
| 1 · Dane i biblioteki | wczytanie CSV (upload albo ścieżka); separator (także tabulator), przecinek dziesiętny i **brak wiersza nagłówka** są rozpoznawane automatycznie (pierwszy wiersz uznawany jest za dane, gdy jego wartość często występuje w kolumnie tekstowej – jak „ham” w SMSSpamCollection; wiersz nagłówka: automatycznie / tak / nie); przegląd z rozpoznanymi kolumnami z wolnym tekstem (Ø słów), wersje bibliotek |
| 2 · Przygotowanie 1 | akcja dla każdej kolumny (zachowaj, usuń, zmień nazwę, wyodrębnij liczbę, data → rok / miesiąc, pierwsze/ostatnie N znaków, zamień wartości, top N, **tekst → długość (znaki)**, **tekst → liczba słów** …) – wolny tekst i krótkie teksty (tytuły, nagłówki) zostają *zachowane*, identyfikatory są proponowane do usunięcia · nowe kolumny: przypisanie wartości, formuła, warunek → 0/1, kopia + akcja · szybkie przyciski „długość w znakach” i „liczba słów” (jak `text length` na kursie) |
| 3 · Analiza i wizualizacja | przegląd, countplot, **długość tekstu** w każdej klasie (histogramy jak FacetGrid + boxplot), **najczęstsze słowa** łącznie albo w każdej klasie (z/bez stop words, słowa albo pary słów), **przykładowe teksty** z każdej klasy, histogram, boxplot, korelacja, **średnie w grupach** (jak `yelp.groupby('stars').mean()` + heatmapa) |
| 4 · Przygotowanie 2 – tekst → liczby | **kolumna tekstu** i **etykieta**, opcjonalnie tylko niektóre **klasy** (jak na kursie 1 i 5 gwiazdek), klasa pozytywna, wiersze bez tekstu usunąć albo traktować jako pusty tekst, `test_size`, `random_state`, `stratify` · **wektoryzator**: CountVectorizer albo TfidfVectorizer, stop words (angielskie), `ngram_range`, małe litery, akcenty, `min_df`, `max_df`, `max_features`, `binary`, `sublinear_tf` · wynik: rozmiar słownika, udział wartości niezerowych, rozkład klas, **tekst jako liczby** (jak `bow_transformer.transform([text])` na kursie), tabela słownika, ostrzeżenia o duplikatach tekstów w treningu i teście oraz o klasach z bardzo małą liczbą tekstów treningowych (poniżej 5) |
| 5 · Zapis danych | dane po kroku 2, **teksty po kroku 4** (tekst, etykieta, etykieta jako liczba, `split`) albo **słownik** (słowo, liczba tekstów, liczba wystąpień albo IDF); format międzynarodowy albo dla Excela |
| 6 · Model i trening | **MultinomialNB** (alpha, fit_prior), **regresja logistyczna** (C, max_iter), **LinearSVC** (C, z kalibracją dla prawdopodobieństw), `class_weight='balanced'`, seed; krzywa walidacji (dane testowe albo część danych treningowych); kod scikit-learn jest pokazywany na żywo, krzywa rośnie w trakcie treningu; wskazówki do połączenia z wektoryzatorem (np. LinearSVC najlepiej z TF-IDF) · **porównanie wszystkich trzech modeli** (accuracy, F1, ROC-AUC, czas) – gdy klasa ma tylko jeden tekst treningowy, LinearSVC jest pomijany z podaniem powodu |
| 7 · Trening a walidacja | odsetek błędów dla alpha lub C (skala logarytmiczna) z oceną (przeuczenie, niedouczenie, „błąd treningowy 0 jest normalny dla tekstów”, nie lepiej niż naiwnie) i propozycją *Przejmij najlepszą wartość* |
| 8 · Ewaluacja i słowa | accuracy, F1, ROC-AUC, classification_report, macierz pomyłek, próg (binarnie) · **najważniejsze słowa w każdej klasie** · **błędnie sklasyfikowane teksty** (najpewniejsze błędy najpierw) · pojedynczy przypadek i **własny tekst** z pokolorowanymi słowami (niebieskie za, pomarańczowe przeciw) i wkładem każdego słowa · teksty testowe z prognozą jako CSV (międzynarodowy albo dla Excela) |
| 9 · Zapis i odczyt modelu | `models/<nazwa>/model.skops` (bezpieczny, domyślnie) albo `model.joblib` – **potok z wektoryzatora i modelu** – + `preprocessing.json` (przepis, klasy, próg, wskaźniki, opcjonalnie sumy kontrolne tekstów treningowych), pobranie ZIP, cały potok jako skrypt Python |
| 10 · Klasyfikacja nowych tekstów | **pojedynczy tekst** (z wyjaśnieniem) albo **CSV** → to samo przygotowanie → prognoza, prawdopodobieństwa, uwagi dla wierszy, ewaluacja (jeśli etykieta jest obecna), pobranie CSV (międzynarodowy albo dla Excela) i **podsumowanie** |

Każdy krok ma pole **„Kod Python do tego kroku”** – wszystko można przenieść 1:1 do notatnika Jupyter; kod liczy
dokładnie to samo co aplikacja (ten sam słownik, te same macierze, te same prawdopodobieństwa, ta sama krzywa, te
same najważniejsze słowa, te same eksporty). Ustawienia (łącznie z nowymi kolumnami) można zapisać po lewej w sekcji
**Konfiguracja** jako JSON i wczytać ponownie dla nowych danych z tymi samymi kolumnami. Przy wczytywaniu plik jest
sprawdzany: nieprzydatne wpisy (zły typ, poza dozwolonym zakresem, np. `ngram_range` (1, 7)) są pomijane i
wymieniane w komunikacie. Można wczytać konfiguracje innych wersji: krok 2 zostaje przejęty, ich ustawienia
etykiety i modelu są po cichu pomijane.

## Tekst → liczby w tej aplikacji

* **Najpierw podział, potem słownik.** Na kursie `cv.fit_transform(X)` liczono na **wszystkich** tekstach i dopiero
  potem dzielono – słownik znał już teksty testowe (mały wyciek danych). Aplikacja najpierw dzieli
  (`train_test_split`) i dopasowuje wektoryzator tylko do tekstów treningowych; teksty testowe są nim tylko
  przekształcane – dokładnie tak jak później nowe teksty. Słowa występujące tylko w teście odpadają (krok 4 pokazuje,
  jaki odsetek słów testowych zna słownik).
* **Macierz rzadka.** Każdy tekst staje się wierszem z kolumną dla każdego słowa słownika; prawie wszystkie wartości
  to 0. scikit-learn zapisuje tylko wartości niezerowe – dlatego działa to nawet dla 100 000 tekstów i 50 000 słów.
* **CountVectorizer albo TF-IDF.** Count liczy (jak na kursie); TF-IDF obniża wagę słów występujących w wielu
  tekstach – `TfidfVectorizer` odpowiada `CountVectorizer` + `TfidfTransformer` z kursu.
* **Stop words:** angielska lista scikit-learn (na kursie: `stopwords.words('english')` z nltk plus usuwanie
  interpunkcji – wektoryzator robi to sam). Dla tekstów niemieckich czy polskich nie ma wbudowanej listy; pomagają
  tam `max_df` (np. 0.9) i `min_df`.
* **Pary słów** (`ngram_range=(1, 2)`) rozpoznają np. „not good”, ale bardzo powiększają słownik.
* **Kodowanie etykiety:** klasy stają się 0/1 (binarnie, *klasa pozytywna* = 1) albo 0 … K−1; etykiety liczbowe
  (gwiazdki) są kodowane według wartości, tekstowe bez spacji na brzegach („ham ” = „ham”); etykieta złożona tylko
  ze spacji liczy się jako brakująca (nie jest osobną klasą).
* **Duplikaty tekstów:** jeśli tekst testowy występuje słowo w słowo także w treningu (częste przy szablonach SMS),
  krok 4 ostrzega – dla takich tekstów model zna już odpowiedź.

## Trzy modele

| Model | najważniejszy parametr | zalety | prawdopodobieństwa |
|---|---|---|---|
| Naiwny Bayes (MultinomialNB) – jak na kursie | `alpha` (wygładzanie, domyślnie 1.0) | bardzo szybki, potrzebuje mało danych, dobry punkt odniesienia | bezpośrednio (`predict_proba`) |
| Regresja logistyczna | `C` (małe = silniejsza regularyzacja) | zwykle dokładniejsza przy wystarczającej liczbie tekstów, dobrze skalibrowana, czytelne wagi | bezpośrednio |
| Liniowa SVM (LinearSVC) | `C` | często najdokładniejsza z TF-IDF | przez `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |

* **Krzywa walidacji (krok 7):** odsetek błędów w treningu i walidacji dla alpha = 0.01 … 5 albo C = 0.01 … 100
  (LinearSVC: 0.001 … 10, bez kalibracji). Przy „części danych treningowych” osobny wektoryzator uczy się słownika
  tylko z pozostałej części treningowej – inaczej znałby teksty walidacyjne.
* **Błąd treningowy bliski 0 jest normalny dla tekstów:** słów jest dużo więcej niż tekstów, więc modele liniowe
  prawie zawsze potrafią rozdzielić teksty treningowe. Liczy się błąd walidacji w porównaniu z naiwną prognozą.
* **Najważniejsze słowa (krok 8):** naiwny Bayes – `log P(słowo | klasa)` minus średnia pozostałych klas; regresja
  logistyczna i LinearSVC – współczynniki. Dla pojedynczego tekstu **wkład** słowa = wartość w tekście (liczba albo
  TF-IDF) × waga dla przewidzianej klasy; dla naiwnego Bayesa wkłady plus prior dają dokładnie logarytm szans.
* **TF-IDF a parametry:** wartości TF-IDF są małe (zwykle poniżej 1). Naiwny Bayes potrzebuje wtedy często
  mniejszego `alpha` (np. 0.1 zamiast 1.0 – inaczej przy dużym słowniku wygładza prawie wszystko), regresja
  logistyczna większego `C` (np. 10); LinearSVC osiąga zbieżność z TF-IDF dużo szybciej niż z surowymi liczbami
  słów. Krok 6 pokazuje wskazówkę, krzywa walidacji w kroku 7 najlepszą wartość.
* **LinearSVC i rzadkie klasy:** prawdopodobieństwa pochodzą z kalibracji z walidacją krzyżową – do niej każda klasa
  potrzebuje co najmniej 2 tekstów treningowych. Inaczej aplikacja mówi to zrozumiale (krok 6), a porównanie
  odbywa się bez LinearSVC.

## Krok 10: klasyfikacja nowych tekstów

* **Nic nie jest dopasowywane od nowa.** Nowe teksty przechodzą przez zapisany *przepis* (krok 2, np. zmiana nazw) i
  zapisany *potok* – wektoryzator używa słownika z treningu. Nieznane słowa są pomijane.
* **Potrzebna jest tylko kolumna tekstu** (surowa kolumna pliku CSV) – aplikacja ją pokazuje i oferuje szablon CSV;
  etykieta jest opcjonalna (wtedy prognozy są oceniane). Jeśli model trenowano na pliku bez wiersza nagłówka, nowe
  dane wczytaj w tej samej formie (wiersz nagłówka: nie).
* **Uwagi dla wierszy:** pusty tekst · brak znanego słowa (inny język?) · głównie nieznane słowa.
* **Podsumowanie:** ile słów zna model (w porównaniu z danymi testowymi), długość tekstów w porównaniu z treningiem,
  przewidywane udziały, niepewne prognozy, porównanie z testem z treningu, **teksty znane słowo w słowo z treningu**
  (sumy kontrolne, opcjonalnie) i klasy w etykiecie, których model nie zna (np. 2–4 gwiazdki przy modelu 1 kontra 5).

Bez aplikacji:

```python
import pipeline as pl
pipe, recipe = pl.load_bundle("models/moj_model")
df_new, _ = pl.read_csv("nowe_teksty.csv")
result, report, evaluation = pl.predict_texts(pipe, recipe, df_new)
podsumowanie = pl.assess_prediction(recipe, result, report, evaluation)
pipe.predict_proba(["The food was great!"])   # pojedyncze teksty bezpośrednio potokiem scikit-learn
```

## Uwagi do aktualnych wersji bibliotek

| starszy kod | problem dziś | w szablonie |
|---|---|---|
| `cv.fit_transform(X)` przed `train_test_split` | słownik ze wszystkich tekstów – teksty testowe trafiają do modelu | najpierw podział, `fit_transform` tylko na `X_train`, `transform` na `X_test` |
| `CountVectorizer(analyzer=text_process)` z nltk | wolne; własnej funkcji nie da się bezpiecznie wczytać przez `skops`; stop words z nltk wymagają pobrania | `CountVectorizer(stop_words='english')` – interpunkcję usuwa sam wektoryzator |
| `CountVectorizer` + `TfidfTransformer` | nadal działa, dwa kroki | `TfidfVectorizer` (to samo w jednym kroku) |
| `cv.get_feature_names()` | usunięte w scikit-learn 1.2 | `get_feature_names_out()` |
| `SVC(probability=True)` | przestarzałe od scikit-learn 1.9 | `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |
| `sns.FacetGrid(yelp, col='stars', size=…)` | `size` nazywa się od seaborn 0.9 `height` | `col_wrap`/`height` albo wykres *Długość tekstu* |
| `yelp.groupby('stars').mean()` | pandas ≥ 2: błąd przy kolumnach tekstowych | `mean(numeric_only=True)` |
| `pd.read_csv(..., sep='\t', names=[...])` | nadal działa | aplikacja sama rozpoznaje tabulator i brak wiersza nagłówka; generowany kod używa `header=None, names=[...]` |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | przy wczytywaniu może zostać wykonany dowolny kod z pliku | `.skops` domyślnie; `.joblib` tylko dla zaufanych plików |

## Pliki

```
app.py                  interfejs Streamlit (tylko UI)
pipeline.py             logika obliczeń – także do użycia w notatniku (import pipeline as pl)
plots.py                wykresy (matplotlib/seaborn), zwracają obiekty Figure
codegen.py              generuje wyświetlany lub eksportowany kod Python
i18n.py                 języki: tx() tłumaczy niemiecki tekst na bieżący język
locales/                tłumaczenia {niemiecki: (angielski, polski)}, jeden plik na moduł
configs/                yelp_beispiel.json, sms_beispiel.json (przykładowe konfiguracje jak na kursie)
.streamlit/config.toml  limit uploadu 1 GB, jasny motyw
tests/                  testy pytest + syntetyczne dane testowe (yelp_like, sms_like, news_like)
AUDIT.md                raport (po polsku): budowa, kontrole, ograniczenia
README.md               niemiecki · README_EN.md angielski · README_PL.md polski
models/                 powstaje przy zapisie (krok 9)
exports/                domyślny folder dla „Zapisz w folderze” (krok 5)
```

Rozszerzanie:
* nowa akcja kolumny → `ACTIONS` + `_transform_column()` w `pipeline.py` (+ `_action_lines()` w `codegen.py`)
* nowy wykres → funkcja w `plots.py` + wpis w `page_viz()` lub `page_eval()` w `app.py`
* kolejny model tekstowy → `MODELS`, `MODEL_NAMES`, `TextModelConfig`, `make_model()`, `word_weights()` w
  `pipeline.py`, `model_params_ui()` w `app.py`, `estimator_code()` w `codegen.py`
* nowy lub zmieniony tekst → napisz go po niemiecku w `tx("…")` (stałe: `N_("…")`) i dodaj tłumaczenie angielskie i
  polskie w odpowiednim pliku w `locales/`; `pytest tests/test_i18n.py` zgłasza brakujące tłumaczenia.

## Testy

```bash
pip install pytest
pytest -q                                  # 188 testów, ok. 4 minuty
python tests/synthetic.py                  # syntetyczne dane podobne do Yelp (10 000 recenzji, bez prawdziwych danych)
python tests/synthetic.py --sms            # syntetyczny plik SMS jak SMSSpamCollection (tabulator, bez nagłówka)
```

| Plik | sprawdza |
|---|---|
| `tests/test_pipeline.py` | wczytywanie (tabulator bez nagłówka, cudzysłowy, średnik, latin-1), akcje tekstowe, krok 4 (wybór klas, klasa pozytywna, etykieta bool/liczbowa/tekstowa, puste teksty, czytelne błędy, duplikaty, wyciek danych, same stop words), wszystkie trzy modele uczą się sentymentu, wyjaśnienie = prognoza (logarytm szans), wiele klas i porównanie, eksporty, zapis/odczyt, nowe teksty (uwagi, pokrycie z treningiem, brakująca kolumna), przebieg SMS ze zmianą nazw |
| `tests/test_codegen_parity.py` | wyświetlany kod Python liczy **dokładnie** to samo co aplikacja: słownik, macierze, podział, prawdopodobieństwa, krzywa, najważniejsze słowa, eksporty – także z plikiem bez nagłówka, wyborem klas, pustymi tekstami, próbką, etykietą bool/liczbową; porównanie modeli, cały skrypt i kod wykresów działają |
| `tests/test_audit_regressions.py` | bezpieczeństwo i odporność: limity ZIP, pickle tylko po potwierdzeniu, własne funkcje w modelu są odrzucane przez skops, uszkodzone lub obce przepisy, potok niepasujący do przepisu, brak kodu i Markdown z nazw, limity formuł i konfiguracji, CSV injection, nieudany zapis |
| `tests/test_audit6_regressions.py` | audyt 6: każdy wpis `preprocessing.json` używany w krokach 9 i 10 jest sprawdzany przy wczytaniu (także próg, wybór klas, miary, akcje kolumn, wektoryzator, rozmiar), puste etykiety, krótkie teksty w kroku 2, rozpoznawanie wiersza nagłówka, macierz pomyłek przy wielu klasach, kod przykładowych tekstów, naiwna prognoza z danych treningowych |
| `tests/test_review_regressions.py` | ustalenia przeglądu przed wydaniem: klasa z tylko jednym tekstem treningowym (uwaga w kroku 4, zrozumiały komunikat dla LinearSVC, porównanie bez LinearSVC – także w generowanym kodzie), krzywa walidacji z tylko jedną klasą w części treningowej, ocena „nie lepiej niż naiwnie” przy niekorzystnym alpha, CSV dla Excela w krokach 8 i 10, nazwy klas dosłownie, polskie nazwy modeli |
| `tests/test_app.py` | test klikania interfejsu: Yelp jak na kursie (wszystkie 10 kroków), plik SMS bez nagłówka z LinearSVC, przypadki błędów i ochrona przed nadpisaniem, brak kolumny tekstowej, eksport/import konfiguracji |
| `tests/test_charts.py` | wykresy wersji NLP (długość tekstu, najczęstsze słowa, porównanie, krzywa, najważniejsze słowa, wkład) także z nietypowymi danymi |
| `tests/test_i18n.py` | języki: każdy tekst przetłumaczony z tymi samymi symbolami zastępczymi, CSV dla Excela zależnie od języka (aplikacja = kod), generowany kod liczy tak samo w każdym języku, cała aplikacja po angielsku i polsku bez niemieckich pozostałości |

## Uwagi

* Bardzo duże pliki CSV lepiej wczytywać przez „Ścieżka na tym komputerze” niż przez upload.
* Podczas treningu nic nie klikaj – każda zmiana uruchamia skrypt od nowa i przerywa trening.
* **Czas obliczeń:** naiwny Bayes jest zawsze szybki. Regresja logistyczna i LinearSVC potrzebują przy dużym słowniku
  i surowych liczbach słów dużo więcej czasu (pomiar: 14 000 syntetycznych tekstów i 28 754 słów na 2 rdzeniach:
  krzywa + trening ok. 2 minuty i 1 minuta; LinearSVC z TF-IDF i parami słów 15 sekund). Szybciej: TF-IDF,
  `min_df`/`max_features` w kroku 4, wyłączenie krzywej walidacji albo próbka.
* **Bag of words nie zna kolejności słów:** „not good” i „good, not bad” wyglądają prawie tak samo (pary słów trochę
  pomagają). Ironia, sarkazm i mieszane recenzje pozostają trudne – krok 8 pokazuje takie błędy. Znaczenie wykraczające
  poza pojedyncze słowa wymaga deep learningu (embeddingi, transformery) – to nie jest częścią tej aplikacji.
* Model zna tylko język i tematy swoich tekstów treningowych; krok 10 zgłasza, gdy nowe teksty zawierają wiele
  nieznanych słów.
* **Używać tylko lokalnie.** Pole ścieżki czyta każdy plik, do którego ma dostęp zalogowany użytkownik, a `models/`
  jest wspólny dla wszystkich użytkowników aplikacji. Dlatego nie wystawiaj aplikacji do internetu bez logowania.
  Wczytuj modele tylko z zaufanych źródeł (ZIP po rozpakowaniu może mieć najwyżej 2 GB). `.skops` jest wczytywany bez
  wykonywania kodu; `.joblib` to pickle i z przesłanego ZIP-a jest wczytywany tylko po wyraźnym potwierdzeniu.
  Wczytywane są tylko modele tekstowe tej aplikacji – modele innych wersji aplikacja odrzuca z czytelnym
  komunikatem, podobnie jak niekompletny lub uszkodzony `preprocessing.json` i potok niepasujący do przepisu.
* Zapisany model zawiera **słownik** (wszystkie słowa tekstów treningowych) i – jeśli nie wyłączono – sumy kontrolne
  tekstów treningowych. Przy poufnych tekstach udostępniaj model tylko uprawnionym osobom.
