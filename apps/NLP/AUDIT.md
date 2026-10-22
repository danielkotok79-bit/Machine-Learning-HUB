# Raport – DATA-Analysis-NLP (klasyfikacja tekstów), wersja 1

> Nowa aplikacja na szkielecie **DATA-Analysis-SupervisedLearning** (wersja 9, po czterech audytach – szczegóły
> w `AUDIT.md` tamtej aplikacji) i z doświadczeniami z **DATA-Analysis-Clustering** (audyt 5). Ten raport opisuje,
> co przejęto, co jest nowe, jakie decyzje metodyczne podjęto, jak aplikacja odwzorowuje projekty NLP z kursu, co
> znaleziono i poprawiono w trakcie budowy i w przeglądzie przed wydaniem oraz jak całość sprawdzono. **Audyt 6** –
> pierwszy audyt kodu tej aplikacji (numeracja kontynuuje audyty szkieletu i wersji Clustering) – jest na końcu pliku.

**Data:** 17.10.2026 · **Zakres:** `app.py`, `pipeline.py`, `plots.py`, `codegen.py`, `i18n.py`, `locales/`,
`configs/yelp_beispiel.json`, `configs/sms_beispiel.json`, testy
**Metoda:** każda nowa funkcja z testem; liczbowe porównanie aplikacji z pokazywanym i eksportowanym kodem Python
(dokładnie, bez tolerancji); testy klikania (Streamlit AppTest) we wszystkich trzech językach; przegląd kodu z ok. 40
sondami (nietypowe dane i ustawienia) i pomiarem czasu obliczeń; test w przeglądarce (Playwright) na działającym
serwerze z syntetycznymi danymi Yelp (3000 recenzji) i SMS (3000 wiadomości, plik bez nagłówka) – nie są to prawdziwe
dane z kursu.
**Wynik:** **159 testów** – wszystkie przechodzą (Python 3.13, pandas 3.0, scikit-learn 1.9, skops 0.16,
Streamlit 1.66); składnia sprawdzona dla Pythona 3.10 i 3.11; test w przeglądarce **56/56**. Dwa problemy
znalezione w trakcie budowy i siedem z przeglądu przed wydaniem są poprawione; każda poprawka z przeglądu ma test
regresyjny (sekcja „Znalezione i poprawione”).

---

## Co przejęto bez zmian

* **Kroki 1–3:** wczytywanie CSV (automatyczne rozpoznanie separatora, także tabulatora, i przecinka dziesiętnego,
  ścieżki w cudzysłowie, spacje w nagłówkach), wszystkie akcje kolumn, nowe kolumny (przypisanie wartości, formuła,
  warunek → 0/1, kopia + akcja – formuły bez dowolnego kodu, z limitami), wykresy ze szkieletu.
* **Przepis** (`preprocessing.json`): wszystko, czego aplikacja uczy się z danych, jest zapisane i na nowych
  danych stosowane bez zmian – nic nie jest dopasowywane od nowa.
* **Zapis i odczyt:** `.skops` domyślnie (bez wykonywania kodu), `.joblib` tylko dla zaufanych plików (z ZIP-a tylko
  po potwierdzeniu), limit 2 GB po rozpakowaniu, zapis przez pliki tymczasowe, ochrona przed nadpisaniem.
* **Konfiguracje** z kontrolą typów i zakresów, **języki** DE/EN/PL, CSV dla Excela wg języka z ochroną przed CSV
  injection, ochrona przed Markdown z obcych nazw (`md_escape`).
* **Testy regresyjne bezpieczeństwa i odporności z audytów szkieletu** – przeniesione na teksty
  (`tests/test_audit_regressions.py`, 58 testów: ZIP, pickle, skops, uszkodzone i obce przepisy, potok niepasujący
  do przepisu, brak kodu i Markdown z nazw, limity formuł i konfiguracji, CSV injection, nieudany zapis).

## Co nowe

| Krok | Supervised (wersja 9) | NLP (ta aplikacja) |
|---|---|---|
| 1 | wczytywanie CSV | dodatkowo **plik bez nagłówka** (kolumny `Spalte_1`, `Spalte_2` …; nagłówek: automatycznie / tak / nie) jak `SMSSpamCollection`, podgląd kolumn z wolnym tekstem (Ø słów) |
| 2 | akcje kolumn | dodatkowo **Tekst → długość (znaki)** i **Tekst → liczba słów** (szybkie przyciski jak `text length` na kursie); wolny tekst „zachowaj”, identyfikatory do usunięcia; rozpoznanie zbiorów z kursu (Yelp, SMSSpamCollection) z przykładową konfiguracją |
| 3 | wykresy | dodatkowo **długość tekstów** wg klasy (histogramy jak `FacetGrid` + boxplot), **najczęstsze słowa** ogółem lub wg klasy (stop words, pary słów), **przykładowe teksty** wg klasy, **średnie wg grupy** |
| 4 | etykieta, podział, uzupełnianie | **kolumna tekstu, etykieta**, opcjonalnie tylko wybrane **klasy** (jak 1 i 5 gwiazdek), klasa pozytywna, wiersze bez tekstu; **najpierw podział, potem słownik tylko z tekstów treningowych**; `CountVectorizer` lub `TfidfVectorizer` (stop words, n-gramy, małe litery, akcenty, `min_df`, `max_df`, `max_features`, `binary`, `sublinear_tf`); jeden tekst jako liczby, słownik; uwagi: duplikaty train/test, tekst i etykieta z tej samej kolumny CSV, bardzo duży słownik, klasy z mniej niż 5 tekstami treningowymi |
| 5 | dane z prognozą | dane po kroku 2, **teksty po kroku 4** (z kolumną `split`), **słownik** |
| 6 | pięć modeli | **MultinomialNB**, **regresja logistyczna**, **LinearSVC** (z kalibracją dla prawdopodobieństw), porównanie trzech modeli, wskazówki do połączenia modelu z wektoryzatorem |
| 7 | krzywa walidacji / uczenia | błąd w treningu i walidacji wg `alpha` / `C` (dane testowe albo część treningu z **własnym słownikiem**), ocena (overfitting, underfitting, „błąd treningowy bliski 0 jest normalny”, „nie lepiej niż naiwnie”), propozycja najlepszej wartości z przyciskiem „przejmij” |
| 8 | macierz pomyłek, ważność cech | dodatkowo **najważniejsze słowa wg klasy**, **błędnie sklasyfikowane teksty** (najpewniejsze najpierw), pojedynczy przypadek i **własny tekst** z pokolorowanymi słowami i wkładem każdego słowa, CSV międzynarodowy lub dla Excela |
| 9 | model + przepis | **potok** (wektoryzator + model) + przepis (klasy, próg, miary, opcjonalnie sumy kontrolne tekstów treningowych) |
| 10 | prognoza | **pojedynczy tekst** z wyjaśnieniem albo **CSV**; uwagi w wierszach (pusty tekst, brak znanego słowa, przeważnie nieznane słowa), pokrycie słownika w porównaniu z testem, długość tekstów, **teksty znane z treningu** (sumy kontrolne), klasy w etykiecie nieznane modelowi, porównanie z testem, podsumowanie |

## Decyzje metodyczne

| Decyzja | Dlaczego |
|---|---|
| Najpierw `train_test_split`, potem `fit_transform` wektoryzatora tylko na tekstach treningowych | na kursie `cv.fit_transform(X)` liczono na wszystkich tekstach przed podziałem – słownik znał już teksty testowe (mały wyciek danych); teraz test jest traktowany jak przyszłe nowe teksty, krok 4 pokazuje, jaki procent słów testowych zna słownik |
| Podział przez pozycje: `train_test_split(np.arange(n), …)` | daje ten sam podział co `train_test_split(texts, y, …)` (zależy tylko od liczby wierszy, seeda i `y`) – pokazywany kod liczy dokładnie to samo co aplikacja |
| `stratify` domyślnie włączone (konfiguracja Yelp: wyłączone, jak na kursie); automatycznie wyłączane z komunikatem, gdy klasa ma mniej niż 2 wiersze albo test jest mniejszy niż liczba klas | przy 13 % spamu proporcje w treningu i teście powinny być równe; brak cichych błędów scikit-learn |
| Etykiety liczbowe kodowane przez wartość (`5.0` = `5` = `'5'`), tekstowe bez spacji na brzegach (`'ham '` = `'ham'`), bool także z tekstu `true`/`1` | nowe dane z innym typem kolumny trafiają do tych samych klas |
| Wybór klas (1 i 5 gwiazdek) zapisany w przepisie | w kroku 10 wiersze z innymi klasami (2–4 gwiazdki) są wymieniane, nie wchodzą do ewaluacji, a podsumowanie radzi trening na wszystkich klasach |
| LinearSVC przez `CalibratedClassifierCV(LinearSVC(), ensemble=False, cv=k)`, `k` = min(5, najmniejsza liczba tekstów treningowych w klasie), co najmniej 2 | prawdopodobieństwa dla progu i ROC-AUC; `SVC(probability=True)` jest w scikit-learn 1.9 przestarzałe; `ensemble=False` → jeden model na wszystkich danych, jego wagi słów są czytelne |
| Krzywa dla LinearSVC bez kalibracji | szybciej; krok 7 mówi, że miary w kroku 8 mogą się nieco różnić |
| Krzywa na „części danych treningowych”: osobny wektoryzator dopasowany tylko do części uczącej | inaczej słownik znałby teksty walidacyjne; domyślnie dane testowe jak metoda łokcia na kursie – z uwagą, że wybór wartości na teście czyni test nieco zbyt optymistycznym |
| Najlepsza wartość krzywej: przy remisie ustawiona; propozycja tylko, gdy poprawa jest większa niż błąd standardowy (max(0,003; √(e(1−e)/n))) | brak propozycji z powodu szumu |
| Ocena „błąd treningowy bliski 0 jest normalny dla tekstów” zamiast „overfitting”, gdy błąd treningowy ≤ 2 % | słów jest dużo więcej niż tekstów, modele liniowe prawie zawsze rozdzielają teksty treningowe; liczy się błąd walidacji wobec prognozy naiwnej |
| Wagi słów: naiwny Bayes – `log P(słowo \| klasa)` minus średnia pozostałych klas (binarnie: różnica); modele liniowe – współczynniki; **wkład** = wartość w tekście × waga dla przewidzianej klasy | ten sam rachunek w aplikacji i w kodzie; dla naiwnego Bayesa wkłady plus prior dają dokładnie logarytm szans (test) |
| Kolorowanie tylko 12 najważniejszych słów tekstu, tylko pojedyncze słowa (pary słów w tabeli), najwyżej 4000 znaków | czytelność i czas; tokenizacja jak w wektoryzatorze (ten sam `token_pattern`, małe litery, akcenty) |
| Sumy kontrolne tekstów treningowych: hash znormalizowanego tekstu (małe litery, ujednolicone spacje), najwyżej 5000 (próbka), z możliwością wyłączenia | krok 10 rozpoznaje teksty znane z treningu (szacunek przeskalowany z próbki); z sumy nie da się odtworzyć tekstu, ale da się sprawdzić, czy dany tekst był w treningu – przy poufnych tekstach wyłączyć |
| Pokrycie słownika w teście jako punkt odniesienia; ostrzeżenie, gdy nowe teksty mają o ponad 15 punktów procentowych mniej znanych słów; uwaga „przeważnie nieznane słowa” poniżej 50 % | inny język, temat albo pisownia są widoczne, zanim ktoś zaufa prognozie |
| Stop words tylko z angielskiej listy scikit-learn (bez nltk) | brak dodatkowego pobierania; dla tekstów niemieckich i polskich pomagają `max_df` i `min_df` (README) |

## Zgodność z projektami NLP z kursu

**Yelp** (`configs/yelp_beispiel.json`)

| Notebook z kursu | W aplikacji |
|---|---|
| `yelp = pd.read_csv('yelp.csv')` | krok 1; identyfikatory (`business_id`, `review_id`, `user_id`), `date` i `type` usuwane w kroku 2 |
| `yelp['text length'] = yelp['text'].apply(len)` | krok 2: nowa kolumna „kopia + akcja: Tekst → długość (znaki)” albo szybki przycisk |
| `sns.FacetGrid(yelp, col='stars')` z histogramem długości, boxplot, countplot | krok 3: „Długość tekstów” wg `stars` (histogramy + boxplot), countplot |
| `yelp.groupby('stars').mean()`, `.corr()`, heatmapa | krok 3: „Średnie wg grupy” (`mean(numeric_only=True)` – pandas ≥ 2) + heatmapa |
| `yelp_class = yelp[(yelp.stars == 1) \| (yelp.stars == 5)]` | krok 4: klasy 1 i 5 |
| `cv = CountVectorizer()`, `X = cv.fit_transform(X)` | krok 4: `CountVectorizer()` – ale dopiero po podziale (zob. decyzje) |
| `train_test_split(X, y, test_size=0.3, random_state=101)` | krok 4: `test_size=0.3`, `random_state=101`, bez `stratify` |
| `nb = MultinomialNB()`, `nb.fit(X_train, y_train)` | krok 6 |
| `confusion_matrix`, `classification_report` | krok 8 |
| `Pipeline([('bow', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', MultinomialNB())])` | krok 4: `TfidfVectorizer` (= `CountVectorizer` + `TfidfTransformer`); zapis w kroku 9 jako `Pipeline([('vect', …), ('clf', …)])` |

**SMS Spam** (`configs/sms_beispiel.json`)

| Notebook z kursu | W aplikacji |
|---|---|
| `pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])` | krok 1: tabulator i brak nagłówka rozpoznawane same (`Spalte_1`, `Spalte_2`); krok 2: zmiana nazw na `label` i `message`; generowany kod: `header=None, names=[…]` |
| `messages['length'] = messages['message'].apply(len)` | krok 2: nowa kolumna `length` |
| `text_process` (interpunkcja + stop words z nltk) i `CountVectorizer(analyzer=text_process)` | `CountVectorizer(stop_words='english')` – interpunkcję usuwa sam wektoryzator; własna funkcja w modelu nie dałaby się bezpiecznie wczytać przez skops (test: model z `analyzer=text_process` jest odrzucany) |
| `TfidfTransformer` + `MultinomialNB` w `Pipeline` | `TfidfVectorizer` + `MultinomialNB`, potok zapisany jako całość |
| `train_test_split(…, test_size=0.2)` | `test_size=0.2` ze `stratify` (13 % spamu) |

Test w przeglądarce sprawdza oba przebiegi (rozpoznanie zbioru, przykładowa konfiguracja, zmiana nazw, ostrzeżenie
o duplikatach SMS). Wyniki różnią się od kursu, bo dane są syntetyczne, a słownik powstaje dopiero po podziale.

## Znalezione i poprawione

### W trakcie budowy

| # | Problem | Skutek | Poprawka |
|---|---|---|---|
| N1 | `recipe_problem` przepuszczał `lineage = []` (lista zamiast słownika) w ręcznie zmienionym `preprocessing.json` | model się wczytywał, a krok 10 kończył się wyjątkiem | wpisy słownikowe przepisu: brak albo słownik – inaczej czytelny komunikat przy wczytaniu; przypadek w teście `test_broken_recipe_fails_at_loading` |
| N2 | seaborn 0.13 z Matplotlib 3.11 zgłaszał w aplikacji ostrzeżenia o przestarzałych parametrach (`vert` w boxplocie, `set_bad` w heatmapie) | ostrzeżenia w konsoli i w testach | wywołania seaborn w `plots.py` w `_quiet()`; testy aplikacji przechodzą bez ostrzeżeń |

### W przeglądzie przed wydaniem

Przegląd całego nowego kodu (silnik kroków 4–10, strony 4–10, generowany kod, wykresy, tłumaczenia) z ok. 40
sondami na poziomie silnika, przejściami kliknięciami (AppTest) dla czterech zbiorów i pomiarem czasu obliczeń.
Każdy potwierdzony problem jest poprawiony i ma test w `tests/test_review_regressions.py`; wszystkie 6 testów
uruchomione na kodzie sprzed poprawek kończą się niepowodzeniem (sprawdzone na kopii z cofniętymi poprawkami).

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| R1 | średnia | LinearSVC przy klasie z **jednym** tekstem treningowym (rzadka klasa, podział bez `stratify`) | trening kończył się angielskim komunikatem scikit-learn „Requesting 2-fold cross-validation but provided less than 2 examples for at least one class”; strona kroku 6 urywała się (bez sekcji porównania), a porównanie modeli zawodziło w całości | `svc_problem`: zrozumiały komunikat z nazwą klasy i radą; krok 4 ostrzega przed klasami z mniej niż 5 tekstami treningowymi (z dopiskiem o LinearSVC przy jednym tekście); porównanie liczy naiwnego Bayesa i regresję logistyczną, a LinearSVC pomija z podaniem powodu – także w pokazywanym kodzie porównania; strona po błędzie treningu wyświetla się dalej |
| R2 | niska | Krzywa walidacji z „częścią danych treningowych”, gdy w części uczącej zostaje tylko jedna klasa (bardzo mało tekstów) | regresja logistyczna i LinearSVC: błąd scikit-learn „This solver needs samples of at least 2 classes”; naiwny Bayes liczył bezsensowną krzywą | czytelny komunikat z radą (walidacja na danych testowych albo krzywa wyłączona) |
| R3 | średnia (bezpieczeństwo) | Pobieranie w krokach 8 i 10 tylko jako CSV międzynarodowy bez ochrony – a README obiecywało format Excela w kroku 8 | teksty z obcych źródeł (recenzje, SMS) zaczynające się od `=`, `+`, `-`, `@` mogły zostać wykonane przez Excel jako formuła (np. `=HYPERLINK(…)`); dokumentacja niezgodna z aplikacją | drugi przycisk **„CSV dla Excela”** (separator i znak dziesiętny wg języka, UTF-8 z BOM, ochrona formuł) w krokach 8 i 10; obie wersje tworzone tylko przy zmianie modelu, progu, wyniku lub języka |
| R4 | niska | Komunikaty kroku 4 (błędy, ostrzeżenia) i błędy treningu w kroku 6 wyświetlane jako Markdown | nazwy klas i kolumn z danych (np. `*Kultur*`) zmieniały formatowanie komunikatu – ta sama klasa problemu co C2 w audycie 5 | `md_text` dla tych komunikatów |
| R5 | niska (metodyka) | Ocena w kroku 7 „nie lepiej niż prognoza naiwna – w tekstach prawie nie ma sygnału”, choć winna była tylko ustawiona wartość parametru | przykład: naiwny Bayes z TF-IDF i parami słów przy `alpha = 1` – błąd równy prognozie naiwnej, przy `alpha = 0.01` wyraźnie niższy; komunikat kierował do sprawdzania danych zamiast parametru | gdy inne wartości krzywej są wyraźnie lepsze (< 90 % błędu naiwnego), ocena mówi, że ustawiona wartość nie pasuje do danych; propozycja najlepszej wartości jak dotąd |
| R6 | niska (użyteczność) | Ostrzeżenie o braku zbieżności dla LinearSVC radziło „zwiększ `max_iter`” – LinearSVC nie ma tego pola w aplikacji; brak wskazówek do połączeń model–wektoryzator | pomiar (14 000 syntetycznych tekstów, 28 754 słów, 2 rdzenie): LinearSVC z surowymi liczbami słów 56 s i brak zbieżności, z TF-IDF i parami słów 15 s, zbieżny i dokładniejszy; naiwny Bayes z TF-IDF przy `alpha = 1` (szczególnie z parami słów) i regresja logistyczna z TF-IDF przy `C = 1` gorsze niż przy `alpha = 0.1` / `C = 10` (sprawdzone na trzech zbiorach) | ostrzeżenie zależne od modelu (LinearSVC: zmniejsz `C` albo wybierz TF-IDF); wskazówki w kroku 6: LinearSVC + `CountVectorizer`, naiwny Bayes + TF-IDF (`alpha`), regresja logistyczna + TF-IDF (`C`) |
| R7 | kosmetyka | Polska nazwa modelu „Logistic Regression” obok „Naiwny Bayes” i „Liniowa SVM” | niespójna lista modeli po polsku | „Regresja logistyczna” |

## Co sprawdzono i działa poprawnie

* **Pokazywany i eksportowany kod = aplikacja** (`tests/test_codegen_parity.py`, porównanie dokładne, bez
  tolerancji): słownik, macierze rzadkie, podział, `y`, prawdopodobieństwa, krzywa, najważniejsze słowa, eksporty
  i miary porównania – także dla pliku bez nagłówka, wybranych klas, pustych tekstów zachowanych jako puste, próbki,
  etykiet bool/liczbowych/tekstowych ze spacjami, TF-IDF z parami słów i regresji logistycznej z walidacją na części
  treningu, LinearSVC z kalibracją, wielu klas bez krzywej. Cały skrypt z kroku 9 uruchamia się i zapisuje ten sam
  potok; fragment kodu z kroku 10 i kod wykresów działają.
* **Sondy na poziomie silnika** (wszystkie z oczekiwanym wynikiem albo czytelnym komunikatem): `min_df` = 1000,
  `max_df` mniejsze niż `min_df`, `max_df` = 0, same stop words, same pojedyncze znaki; etykieta bool z tekstem
  `"True"`/`"false"` w nowych danych; etykieta 1.0/5.0 z nowymi wartościami `"5"`, `"1.0"` i nieznaną `"3"`;
  `'ham '` = `'ham'`; klasa pozytywna pierwsza alfabetycznie; brakujące teksty jako puste (uwagi i poziom
  „krytyczny”); `lowercase=False` z usuwaniem akcentów (kolorowanie zgodne ze słownikiem); pary słów w wyjaśnieniu;
  zapis i odczyt wszystkich trzech modeli w `.skops` i `.joblib` oraz z ZIP-a (identyczne prawdopodobieństwa);
  przepisy wersji Supervised/Clustering i zła struktura odrzucane; 50 klas (wszystkie modele) i 51 klas
  (komunikat); 19 wierszy (komunikat); tekst bez znanego słowa; brak błędnych klasyfikacji; kolumna tekstu z liczbami
  i tekstem, typ `category`; tekst 1 MB (0,2 s, kolorowanie ograniczone).
* **Przejścia kliknięciami** (AppTest): niemieckie wiadomości z czterema klasami (TF-IDF, pary słów, akcenty,
  wszystkie wykresy, eksporty dla Excela, LinearSVC z walidacją na części treningu, porównanie, „przejmij najlepszą
  wartość”, własny tekst z umlautami, zapis i odczyt, krok 10 – pojedynczy tekst i CSV z ewaluacją, zmiana języka);
  Markdown w tekstach i nazwach klas (`![x](http://…)`, `**gut**`, `$schlecht$`, `<b>`) – wszędzie wyświetlany
  dosłownie (krok 3, krok 4, etykiety metryk i suwaka w kroku 8, krok 10); Yelp z pięcioma klasami (podsumowanie dla
  wielu klas); rzadka klasa z LinearSVC (R1).
* **Bezpieczeństwo:** wszystkie zabezpieczenia szkieletu działają dalej (testy regresyjne); skops odrzuca potok
  z własną funkcją (`analyzer=text_process` jak na kursie); sumy kontrolne tekstów treningowych można wyłączyć przed
  przekazaniem modelu; README ostrzega, że słownik w modelu zawiera wszystkie słowa tekstów treningowych.
* **Języki:** katalog tłumaczeń kompletny (809 tekstów, te same symbole zastępcze w trzech językach); cała aplikacja
  po angielsku i po polsku bez niemieckich pozostałości (AppTest i przeglądarka); generowany kod liczy w każdym języku
  to samo.
* **Python 3.10:** składnia wszystkich plików sprawdzona parserem 3.10 (`ast`) i kompilatorem 3.11 (f-stringi sprzed
  PEP 701).

## Test w przeglądarce (Playwright, prawdziwy serwer)

Kroki 1–10 po niemiecku jak w projekcie Yelp: wczytanie `yelp.csv` (ścieżka), rozpoznanie zbioru i przykładowa
konfiguracja (nowa kolumna `text length`), wykresy długości tekstów, najczęstszych słów, przykładowych tekstów
i średnich wg grupy, krok 4 z `text` i `stars` już wybranymi (klasy 1 i 5), eksport słownika, wskazówka TF-IDF przy
LinearSVC, trening naiwnego Bayesa, porównanie trzech modeli, krzywa, ocena (najważniejsze słowa, własny tekst
z pokolorowanymi słowami, pojedynczy przypadek, CSV międzynarodowy i dla Excela), zapis `.skops` i odczyt, krok 10 –
pojedynczy tekst i 500 nowych recenzji z podsumowaniem (oba formaty CSV). Następnie kroki 4, 6, 7, 8 i 10 po
angielsku i po polsku – bez błędów i bez niemieckich pozostałości (także protokół kroku 4 po ponownej wektoryzacji).
Na końcu plik SMS bez nagłówka: rozpoznanie tabulatora i braku nagłówka, przykładowa konfiguracja (zmiana nazw),
ostrzeżenie o duplikatach, wskazówka do `alpha` przy naiwnym Bayesie z TF-IDF, trening LinearSVC i rozpoznanie spamu
w pojedynczym tekście. Wynik: **56/56**. Błędów w aplikacji test nie znalazł; poprawiono dwie rzeczy w samym
skrypcie testowym (ścieżka do danych liczona od katalogu serwera; w kroku 10 wybór „pojedynczy tekst”, bo tryb CSV
pozostał z wcześniejszego przebiegu – zamierzone: ustawienia przetrwają zmianę strony).

## Ograniczenia i uwagi (opisane w README)

* **Bag of words nie zna kolejności słów:** „not good” i „good, not bad” wyglądają prawie tak samo (pary słów trochę
  pomagają); ironia i mieszane recenzje pozostają trudne – krok 8 pokazuje takie błędy. Znaczenie poza pojedynczymi
  słowami wymaga deep learningu (embeddingi, transformery) – to nie jest częścią tej aplikacji.
* **Stop words tylko po angielsku**; model zna tylko język i tematy swoich tekstów treningowych – krok 10 zgłasza
  teksty z wieloma nieznanymi słowami.
* **Czas obliczeń:** regresja logistyczna i LinearSVC na surowych liczbach słów przy dużym słowniku potrzebują minut
  (pomiar w R6; regresja logistyczna: krzywa + trening ok. 2 min). Szybciej: TF-IDF, `min_df` / `max_features`,
  krzywa wyłączona albo próbka. Naiwny Bayes zawsze poniżej sekundy.
* **Krzywa na danych testowych** (domyślnie, jak na kursie) czyni test nieco zbyt optymistycznym, jeśli według niej
  wybiera się wartość – aplikacja to mówi i proponuje „część danych treningowych”.
* **Teksty znane z treningu** w kroku 10 to szacunek z próbki do 5000 sum kontrolnych; normalizacja obejmuje tylko
  wielkość liter i spacje – teksty różniące się interpunkcją liczą się jako różne.
* **CSV międzynarodowy** w krokach 8 i 10 jest bez ochrony dla Excela (do pandas i innych programów); do otwierania
  w Excelu służy przycisk „CSV dla Excela”.
* **Zmiana języka w trakcie pracy:** protokół kroków 2 i 4 zostaje w starym języku, dopóki krok nie zostanie
  wykonany ponownie (zachowanie szkieletu; test w przeglądarce sprawdza, że ponowna wektoryzacja daje protokół
  w nowym języku).
* **`.joblib` z folderu `models/`** wczytuje się bez pytania – to własny folder użytkownika; aplikacja jest
  przeznaczona do użytku lokalnego.
* **Modele nie są wymienne między aplikacjami:** model tekstowy działa tylko tutaj, modele Supervised / Clustering
  tylko tam; konfiguracje (`.json`) można wczytywać w każdej aplikacji (nieznane ustawienia są pomijane, krok 2
  przejmowany).
* Polskie formy liczby mnogiej są uproszczone („tekstów”, „klas”) – jak w aplikacji bazowej.

## Testy (stan wersji 1)

| Plik | Testy | sprawdza |
|---|---|---|
| `tests/test_pipeline.py` | 38 | wczytywanie (tabulator bez nagłówka, cudzysłowy, średnik, latin-1), akcje tekstowe, krok 4 (wybór klas, klasa pozytywna, rodzaje etykiet, puste teksty, komunikaty, duplikaty, wyciek, same stop words), trzy modele uczą się sentymentu, wyjaśnienie = prognoza (logarytm szans), wiele klas i porównanie, eksporty, zapis/odczyt, nowe teksty, przebieg SMS |
| `tests/test_codegen_parity.py` | 27 | pokazywany i eksportowany kod = aplikacja (dokładnie) |
| `tests/test_audit_regressions.py` | 58 | bezpieczeństwo i odporność z audytów szkieletu, przeniesione na teksty |
| `tests/test_review_regressions.py` | 6 | poprawki z przeglądu przed wydaniem (R1–R7) |
| `tests/test_charts.py` | 8 | wykresy wersji NLP, także z nietypowymi danymi |
| `tests/test_i18n.py` | 17 | tłumaczenia, CSV dla Excela wg języka, aplikacja po angielsku i polsku |
| `tests/test_app.py` | 5 | testy klikania: Yelp jak na kursie (10 kroków), SMS bez nagłówka z LinearSVC, przypadki błędne, brak kolumny tekstu, konfiguracja |

Razem **159 testów** – wszystkie przechodzą; uruchomienie `pytest -q` (ok. 4 minuty).

## Uruchomienie

* `pip install -r requirements.txt`, `streamlit run app.py` (te same biblioteki co wersja Supervised; bez nltk).
* Zmienne środowiskowe: `NLP_TEMPLATE_LANG` (język startowy; bez ustawienia aplikacja startuje po angielsku jak
  pozostałe wersje) i `NLP_TEMPLATE_MODELS` (folder modeli) – odpowiedniki `ML_TEMPLATE_…` / `CL_TEMPLATE_…`.

---

# Audyt 6 – kod aplikacji NLP (18.10.2026)

**Zakres:** cały kod wersji 1: silnik w `pipeline.py` (wczytywanie z rozpoznawaniem nagłówka, akcje tekstowe,
krok 4, modele, krzywa, porównanie, ocena i wyjaśnienia, zapis i odczyt z kontrolą przepisu, nowe teksty
i podsumowanie), strony 1–10 w `app.py`, generowany kod w `codegen.py`, wykresy w `plots.py`, tłumaczenia.
**Metoda:** przegląd każdej funkcji; sondy: 17 ręcznie zmienionych plików `preprocessing.json` (z ZIP-a i z folderu
`models/`, także kliknięciami w krokach 9 i 10), etykiety złożone tylko ze spacji (trening i nowe dane), krótkie
teksty w kroku 2, nagłówek z wartością równą nazwie kolumny, dwa sklejone eksporty CSV, plik bez nagłówka z rzadką
klasą w pierwszym wierszu, 40 i 50 klas (wykresy, czas), zgodność aplikacji z kodem w 6 przypadkach brzegowych,
50 000 nowych tekstów (czas), przejścia kliknięciami (AppTest) dla 40 klas i krótkich tekstów, próbka 45 polskich
tłumaczeń. Każdy potwierdzony błąd jest poprawiony i ma test regresyjny (`tests/test_audit6_regressions.py`).

**Wynik:** 8 naprawionych problemów – 3 średnie, 4 drobne, 1 kosmetyczny. Żaden nie zmieniał wyników przy
przykładowych konfiguracjach Yelp i SMS.

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| B1 | średnia (odporność) | `preprocessing.json` był sprawdzany niepełnie: bez `classes_selected`, wartości w `metrics`, zawartości `column_actions` i `new_columns` (tylko „czy to słownik”), `lineage`, `split`, `missing_text`; próg dopuszczalny w całym przedziale 0–1; brak porównania rodzaju wektoryzatora z potokiem i limitu rozmiaru pliku | ręcznie zmieniony lub uszkodzony plik wczytywał się, a potem: krok 10 przerywał się wyjątkiem (`classes_selected` = 5 albo tekst w `metrics` – w podsumowaniu; próg 0,999 – już przy otwarciu strony, bo suwak ma zakres 0,05–0,95), prognoza kończyła się komunikatem „Fehler: 'column'” albo podsumowanie wymieniało klasy, których model nie zna; krok 9 pokazywał ustawienia wektoryzatora niezgodne z modelem; `preprocessing.json` o rozmiarze 2 GB z ZIP-a trafiłby w całości do pamięci | `recipe_problem` sprawdza każdy wpis używany w krokach 9 i 10 (typy, próg w granicach suwaków, wybór klas jako podzbiór klas, liczby w miarach, struktura akcji kolumn i nowych kolumn), `check_recipe` porównuje rodzaj wektoryzatora z potokiem, limit 50 MB dla `preprocessing.json` (ten sam co przy pytaniu o pickle); granice progu zdefiniowane w jednym miejscu (`THRESHOLD_MIN/MAX`) dla suwaków, konfiguracji i przepisu |
| B2 | średnia | Etykieta złożona tylko ze spacji (np. komórka „ ” w CSV) była osobną klasą „” | z „ham”/„spam” i kilkoma pustymi komórkami powstawało po cichu zadanie z trzema klasami (bez progu, inne miary); w kroku 10 puste etykiety liczone jako „klasa, której model nie zna” | pusta etykieta = brak etykiety: wiersze usuwane w kroku 4 (z wpisem w protokole), nie pojawiają się w wyborze klas, rozkładzie ani na wykresach, w nowych danych nie są „nieznane”; ten sam filtr w generowanym kodzie (test zgodności) |
| B3 | średnia | Propozycje w kroku 2: kolumna tekstowa z ponad 50 różnymi wartościami i średnio mniej niż 3 słowami była proponowana do usunięcia; krok 1 ostrzegał „brak wolnego tekstu” | krótkie teksty (tytuły recenzji, nagłówki, tweety; sprawdzone: średnio 2 słowa, 393 różne wartości) znikały po „Zastosuj” – kolumna tekstu nie trafiała do kroku 4 | do usunięcia proponowane są tylko identyfikatory (wartości bez spacji, średnio ≤ 1,1 słowa); krótkie teksty zostają, krok 1 pokazuje je jako możliwą kolumnę tekstu zamiast ostrzeżenia |
| C1 | niska | Rozpoznawanie braku nagłówka: wystarczało jedno wystąpienie nazwy kolumny jako wartości w kolumnie tekstowej | plik z nagłówkiem, w którym kategoria raz nazywa się jak kolumna (np. „typ” w kolumnie „typ”), albo dwa sklejone eksporty CSV (nagłówek drugi raz w środku) były czytane jak plik bez nagłówka: kolumny `Spalte_1`, `Spalte_2`, nagłówek jako wiersz danych | wartość musi wystąpić co najmniej 2 razy i w 1 % wierszy (SMSSpamCollection: „ham” w 86 % wierszy); plik bez nagłówka z rzadszą klasą w pierwszym wierszu (spam, 13 %) nadal jest rozpoznawany |
| C2 | niska | Macierz pomyłek rosła bez ograniczeń razem z liczbą klas | 40 klas: obraz ok. 100 mln pikseli (ostrzeżenie Pillow „DecompressionBombWarning”), krok 8 ok. 10 s; 50 klas: 68 × 57 cali; podpisy wykresu klas w kroku 10 nachodziły na siebie | rozmiar ograniczony (13 × 11,5 cala), od 13 klas tylko liczności, od 26 tylko kolory ze skalą; podpisy pochylone; krok 8 dla 40 klas: 3,5 s |
| C3 | niska | Kod „Przykładowe teksty” (krok 3) bez `dropna` | przy brakujących tekstach kod pokazywał NaN i inne teksty niż aplikacja; bez grupy kończył się błędem, gdy n było większe niż liczba wierszy | kod jak aplikacja: tylko istniejące teksty, `min(n, len(texts))`; test porównuje wynik z aplikacją |
| C4 | niska (metodyka) | „Prognoza naiwna” w podpowiedzi przy accuracy (kroki 8 i 10) liczona z najczęstszej klasy ocenianych danych | w kroku 10 przy innym rozkładzie klas niż w treningu wartość myliła (np. same recenzje 1-gwiazdkowe: „naiwnie” 1,000, choć model naiwny z treningu przewiduje zawsze „5” i ma 0,000) | najczęstsza klasa danych treningowych (z `y_train` albo z przepisu), jak w kroku 7 |
| C5 | kosmetyka | „Model + recepta jako ZIP” – w pozostałych polskich tekstach „przepis” | niespójna terminologia | „Model + przepis jako ZIP” (także w tym raporcie: „przepis”) |

## Co sprawdzono i działa poprawnie

* **Zgodność aplikacji z kodem** w przypadkach brzegowych (dokładnie, jak w `tests/test_codegen_parity.py`): krzywa
  na części treningu bez `stratify` (klasa z jednym tekstem treningowym), LinearSVC z kalibracją `cv` = 3 (także
  w skrypcie z kroku 9, gdzie `cv` liczone jest z przepisu), etykieta bool z klasą pozytywną `False`, etykieta
  liczbowa typu float z wyborem klas 1/5, klasą pozytywną 1 i pustymi tekstami, etykiety złożone ze spacji (po
  poprawce B2), trigramy z `binary`, `max_features` i LinearSVC z walidacją na 35 % treningu.
* **Ręcznie zmienione przepisy:** z 17 wariantów 14 jest teraz odrzucanych przy wczytaniu z nazwą wpisu (przed
  audytem: 2); formuła z wywołaniem funkcji w `new_columns` kończy się przy prognozie czytelnym komunikatem
  („Niedozwolone w formułach: Call”) – kod nigdy nie jest wykonywany; `text_stats.mean_words` = 0 i tekst w
  `meta.header` działają bez błędów.
* **Czas:** 50 000 nowych tekstów w kroku 10: 2,3 s (podsumowanie < 0,1 s); najczęstsze słowa w kroku 3 na
  próbce 20 000 tekstów z parami słów: 0,6 s; sortowanie najważniejszych słów dla 500 000 słów × 10 klas: 0,8 s.
* **40 klas** przez kroki 4–10 (AppTest): bez błędów; krok 8: 3,5 s (przed poprawką C2: 10 s).
* **Tłumaczenia:** próbka 45 polskich tekstów – jedna niespójność (C5); katalog kompletny (814 tekstów).
* **Bezpieczeństwo:** wszystkie zabezpieczenia z wersji 1 działają dalej (testy regresyjne); żaden wpis przepisu
  nie prowadzi do wykonania kodu.
* **Test w przeglądarce** (prawdziwy serwer, Playwright): cały przebieg z wersji 1 oraz nowy krok – ręcznie
  zmieniony model w folderze `models/` (próg 0,999) wybrany i wczytywany w kroku 9: komunikat o uszkodzonym
  `preprocessing.json` z nazwą wpisu (`threshold`), bez wyjątku, wcześniej wczytany model pozostaje aktywny –
  **57/57**.

## Uwagi (bez zmian)

* Powtórzony nagłówek w środku pliku (dwa sklejone eksporty) zostaje po poprawce C1 zwykłym wierszem danych – np.
  klasa „label” z jednym tekstem; krok 4 ostrzega o klasie z bardzo małą liczbą tekstów, w wyborze klas można ją
  pominąć.
* Sumy kontrolne tekstów treningowych korzystają z funkcji skrótu pandas (`hash_pandas_object`); gdyby przyszła
  wersja pandas ją zmieniła, krok 10 nie rozpoznawałby tekstów z treningu modeli zapisanych wcześniej (szacunek
  byłby za niski, prognozy bez zmian).
* Etykieta z wartościami różniącymi się tylko wielkością liter („Spam” i „spam”) to nadal dwie klasy – tak jak
  w danych.
* Nazwy kolumn z własnego pliku CSV w komunikatach kroku 2 nie są escapowane (własne dane – jak w aplikacji bazowej).

## Testy (stan po audycie 6)

188 testów (159 + 29 w `tests/test_audit6_regressions.py`) – wszystkie przechodzą (Python 3.13, pandas 3.0,
scikit-learn 1.9, skops 0.16, Streamlit 1.66); składnia sprawdzona dla Pythona 3.10 i 3.11. Z 29 nowych testów 27
uruchomionych na kodzie sprzed audytu kończy się niepowodzeniem (sprawdzone) – odtwarzają problemy B1–C5; 2 sprawdzają,
że poprawki nie psują prawidłowych przypadków (poprawny przepis wczytuje się i prognozuje, plik bez nagłówka jest
nadal rozpoznawany). Uruchomienie: `pytest -q` (ok. 4 minuty).
