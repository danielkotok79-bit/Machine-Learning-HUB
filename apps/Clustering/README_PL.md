# DATA-Analysis-Clustering (Streamlit + scikit-learn: K-Means, Isolation Forest, LOF, DBSCAN, autoenkoder)

Aplikacja Streamlit dla tabelarycznych danych CSV: wczytaj, przygotuj, zwizualizuj i podziel na grupy (klastry)
za pomocą **K-Means** – **uczenie nienadzorowane**: nie ma etykiety, K-Means sam szuka wierszy, które są do siebie
podobne. Potem aplikacja opisuje klastry (profile, mapa 2D), porównuje je – jeśli jest dostępny – ze znanym podziałem
i przypisuje nowe dane do klastrów. Do każdego kroku aplikacja pokazuje pasujący kod Python.

**Wykrywanie anomalii:** zamiast szukać grup, aplikacja znajduje za pomocą **Isolation Forest**, **Local Outlier
Factor (LOF)**, **DBSCAN** (szum = anomalia) albo **autoenkodera** (sieć neuronowa – deep learning bez etykiety)
*najbardziej nietypowe wiersze* – ze wskaźnikiem, progiem, cechami,
które czynią wiersz nietypowym, i porównaniem ze znanym podziałem (np. *Class* w danych o oszustwach). Nowe dane są
oceniane z tym samym progiem. Dla danych krypto skrypt `tools/bybit_features.py` buduje z publicznych danych rynkowych
Bybit gotowy plik CSV (zob. [Bybit → CSV](#bybit--csv-toolsbybit_featurespy)).

Wzorem jest projekt K-Means z kursu: **College_Data** – K-Means z K = 2 na wszystkich kolumnach liczbowych; kolumna
*Private* (Yes/No) jest przy tym pomijana i używana dopiero potem do porównania. Dla tego zbioru jest gotowa
przykładowa konfiguracja.

Aplikacja powstała z wersji Supervised (DATA-Analysis-SupervisedLearning): kroki 1–3 (wczytywanie, akcje kolumn,
nowe kolumny, wykresy), receptura (`preprocessing.json`), zapis/wczytywanie, konfiguracje i języki pozostały bez
zmian; nowe są krok 4 bez etykiety i bez podziału, K-Means w kroku 6, wybór K w kroku 7, ocena klastrów w kroku 8
i przypisywanie nowych danych w kroku 10 – później doszły modele anomalii (kroki 6–10 dopasowują się do wybranego
modelu).

Aplikacja jest dostępna po **niemiecku, angielsku i polsku** – zob. [Języki](#języki).
(Deutsch: `README.md` · English: `README_EN.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # otwiera http://localhost:8501
```

Przetestowano z Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 lub nowszy.

## Języki

Język można w każdej chwili przełączyć na górze panelu bocznego: **Deutsch · English · Polski**. Zmieniają się
tylko teksty – obliczenia są identyczne we wszystkich trzech językach (oryginałem jest niemiecki).

| Start od razu w danym języku | |
|---|---|
| Wiersz poleceń | `streamlit run app.py -- --lang pl` (albo `en`, `de`) – `--` przed `--lang` jest konieczne |
| Adres w przeglądarce | `http://localhost:8501/?lang=pl` |
| Zmienna środowiskowa | `CL_TEMPLATE_LANG=pl` (Windows: `set CL_TEMPLATE_LANG=pl`) |

Bez tych ustawień aplikacja startuje po angielsku. Język wybrany w panelu bocznym trafia też do adresu (`?lang=…`),
dzięki czemu zostaje zachowany po odświeżeniu strony.

Tłumaczone są wszystkie teksty interfejsu, tytuły i osie wykresów, nagłówki tabel, komentarze i przykładowe nazwy
plików w generowanym kodzie Python, podsumowanie w kroku 10, nazwy nowych kolumn w eksporcie (`Klaster`,
`Nazwa klastra`, `Odległość`) i nazwy pobieranych plików. We wszystkich językach takie same pozostają:

* wartości wewnętrzne – akcje i strategie są zapisane w konfiguracjach (`.json`) i w `preprocessing.json` pod
  niemiecką nazwą; dzięki temu konfiguracje i modele działają we wszystkich trzech językach;
* wartości, które aplikacja wpisuje do danych: `Sonstige` (top N), `fehlend` (osobna kategoria dla braków),
  `_jahr` / `_monat` (data → rok + miesiąc); przy „Przypisz wartości” `sonst=…` i `rest=…` działają w każdym języku;
* komentarze w kodzie źródłowym (po niemiecku).

Jeśli język zmieni się w trakcie pracy, już wyświetlone komunikaty (np. protokół kroku 2 lub 4) pozostają w starym
języku, dopóki krok nie zostanie wykonany ponownie. Teksty samego Streamlit (np. „Upload” w polu pliku czy menu ⋮)
są w każdej wersji po angielsku.

Format Excel w kroku 5 zależnie od języka: niemiecki i polski `;` i przecinek dziesiętny, angielski `,` i kropka
dziesiętna – zawsze UTF-8 z BOM i z ochroną przed CSV injection (także w generowanym kodzie Python).

## Szybki start – jak w kursie

1. **Krok 1:** wczytaj `College_Data` (CSV, pierwsza kolumna = nazwa uczelni)
2. **Krok 2:** *Wczytaj przykładową konfigurację* → *Zastosuj* (zachowaj nazwę, Grad.Rate 118 → 100)
3. **Krok 3:** np. wykres punktowy Grad.Rate względem Room.Board z kolorem = Private, histogram wg grup
4. **Krok 4:** *Przygotuj dane dla modelu* (17 cech, Private tylko do porównania, StandardScaler)
5. **Krok 6:** *Rozpocznij trening* (K = 2) → krok 7 (wybór K), 8 (profile, porównanie z Private), 9, 10

Dla własnych danych: w kroku 4 wybierz cechy, opcjonalnie znany podział do porównania i kolumnę z nazwami; w kroku 6
wybierz K – krok 7 pokazuje, które K pasuje. Na wolnym komputerze włącz po lewej **próbkę** (`df.sample(frac=0.1)`).

## Szybki start – anomalie w danych krypto (Bybit)

```bash
python tools/bybit_features.py --top 20 --interval 1h --days 120 --split 2026-10-01
```

Skrypt pobiera publiczne dane rynkowe (bez klucza API) i zapisuje `bybit_series_linear_60_train.csv` (przed
1 października), `bybit_series_linear_60_new.csv` (od tego dnia) i `bybit_series_linear_60_columns.csv` (opisy
kolumn).

1. **Krok 1:** wczytaj `…_train.csv` (opcjonalnie `…_columns.csv` jako opisy kolumn)
2. **Krok 2:** aplikacja rozpoznaje dane Bybit → *Wczytaj przykładową konfigurację* → *Zastosuj*
3. **Krok 4:** *Przygotuj dane dla modelu* (11 wskaźników, `row` = symbol + czas jako nazwa wiersza)
4. **Krok 6:** ustawiony jest Isolation Forest z 1 % nietypowych wierszy → *Rozpocznij trening*
5. **Krok 7:** czy próg pasuje (załamanie krzywej wskaźnika)? · **Krok 8:** najbardziej nietypowe godziny i to, co
   je wyróżnia (np. `turnover_z +9.1 σ · ret_z −7.4 σ`)
6. **Krok 9:** zapisz · **Krok 10:** oceń `…_new.csv` tym samym modelem i z tym samym progiem

## Kroki

| Strona | Zawartość |
|---|---|
| 1 · Dane i biblioteki | wczytanie CSV (upload lub ścieżka; separator `;` i przecinek dziesiętny są wykrywane automatycznie), przegląd, wersje bibliotek, opcjonalnie opisy kolumn |
| 2 · Przygotowanie 1 | jedna akcja na kolumnę (zachowaj, usuń, wyodrębnij liczbę, data → rok / miesiąc / rok + miesiąc, pierwsze/ostatnie N znaków, zamień wartości, top N, **ogranicz wartości (min, max)** …) · nowe kolumny: przypisz wartości, formuła, warunek → 0/1, kopia + akcja |
| 3 · Analiza i wizualizacja | wybór kolumn · przegląd, countplot, histogram, boxplot, scatter, heatmapa, korelacja z jedną kolumną, pairplot, udział wg kategorii, statystyki grup, jointplot, gęstość 2D, lmplot, histogram wg grup |
| 4 · Przygotowanie 2 – dane do klastrowania | **cechy**, opcjonalnie **znany podział** (tylko do porównania, np. Private) i **nazwa wiersza** (np. nazwa uczelni); brakujące wartości dla każdej kolumny; kolumny dummy i skalowanie (domyślnie StandardScaler, drop_first wyłączone) – wyuczone ze wszystkich wierszy, bez podziału; uwagi o silnie skorelowanych kolumnach i kolumnach bez zmienności. Znany podział jest proponowany tylko dla kolumny o takiej nazwie (Private, species, label, segment …); kolumna z identyfikatorem (np. CustomerID = 1, 2, 3 …) jest proponowana jako nazwa wiersza, nie jako cecha – wybrana jako cecha, wywołuje ostrzeżenie |
| 5 · Zapis danych | dane po kroku 2, dane do klastrowania po kroku 4 (nieskalowane lub skalowane) albo – po treningu – dane z **numerem klastra, nazwą klastra i odległością** od centrum (K-Means) lub z **anomalią (1 = nietypowy) i wskaźnikiem anomalii** (przy DBSCAN także klaster); międzynarodowo lub dla Excela |
| 6 · Model i trening | **wybór metody** · **K-Means**: K (n_clusters), init (k-means++/random), n_init, max_iter, algorithm, seed; opcjonalnie krzywa łokcia i Silhouette dla kroku 7 (krzywe rosną w trakcie treningu) · **Isolation Forest**: n_estimators, max_samples, seed, contamination (auto albo stały udział) · **LOF**: n_neighbors, metric, contamination · **DBSCAN**: eps (wartość startowa z k-odległości), min_samples, metric, udział szumu dla eps dla kroku 7 · **Autoenkoder**: warstwy (propozycja na podstawie liczby cech, np. 16,3,16), aktywacja, epoki, alpha, learning rate, early stopping, contamination; po treningu **krzywa uczenia** – pasujący kod scikit-learn jest pokazywany na bieżąco |
| 7 · Wybór K albo progu | K-Means: **metoda łokcia** (inercja dla K) i **Silhouette** dla K z oceną i propozycją (*Przejmij K = …*), Davies-Bouldin, Calinski-Harabasz, wyjaśniona zmienność (R²) i **wykres Silhouette** · Isolation Forest/LOF/autoenkoder: **rozkład wskaźnika** z progiem, **krzywa wskaźnika** najbardziej nietypowych wierszy z załamaniem (*Przejmij contamination = …*), próg dla każdego udziału · DBSCAN: **k-odległość** z załamaniem (*Przejmij eps = …*), udział szumu i liczba klastrów dla eps |
| 8 · Ocena | K-Means: wielkość klastrów, **nazywanie klastrów**, **profile** (średnie w jednostkach oryginalnych, heatmapa odchylenia od średniej w σ, tekst dla każdego klastra), **siła rozdzielania cechy (R²)**, **mapa 2D (PCA)** z kolorem = klaster lub znany podział, dwie cechy względem siebie, **porównanie ze znanym podziałem** (tabela krzyżowa, przypisanie klaster → grupa, trafność, prognoza naiwna, ARI, NMI, macierz pomyłek, classification_report – także porównanie jak w kursie), **typowi przedstawiciele** klastrów, pojedynczy przypadek · modele anomalii: **najbardziej nietypowe wiersze** ze wskaźnikiem i cechami najbardziej odbiegającymi od średniej (z), **profil** nietypowe a normalne, **obraz PCA** (anomalie zawsze na obrazie), klastry DBSCAN, przy autoenkoderze **najgorzej odtworzone cechy** i **mapa 2D wąskiego gardła**, **porównanie ze znanym podziałem** (udział nietypowych w grupie; przy dwóch grupach precision, recall, ROC-AUC, average precision), pojedynczy przypadek |
| 9 · Zapis i wczytanie modelu | `models/<nazwa>/model.skops` (bezpieczny, domyślny) lub `model.joblib` + `preprocessing.json` (receptura, ustawienia, nazwy klastrów albo próg i udział nietypowych wierszy, miary, przypisanie do grupy porównawczej), pobranie ZIP, cały pipeline jako skrypt Python |
| 10 · Przypisz / oceń nowe dane | nowy CSV lub formularz → to samo przygotowanie → K-Means: **najbliższe centrum klastra** (`kmeans.predict`), odległość, uwagi, rozkład w porównaniu z treningiem · modele anomalii: **anomalia i wskaźnik z progiem z treningu** (DBSCAN: klaster najbliższego punktu rdzeniowego albo szum), udział nietypowych wierszy w porównaniu z treningiem · porównanie ze znanym podziałem (jeśli jest w danych), pobranie CSV i **podsumowanie** |

Każdy krok ma pole **„Kod Python do tego kroku”** – wszystko można przenieść 1:1 do Jupyter Notebook; kod liczy
dokładnie to samo co aplikacja (te same klastry, ta sama inercja, te same krzywe). Ustawienia (łącznie z nowymi
kolumnami) można zapisać po lewej w sekcji **Konfiguracja** jako JSON i wczytać ponownie dla nowych danych o tych
samych kolumnach. Przy wczytywaniu plik jest sprawdzany: nieużyteczne wpisy (zły typ, poza dozwolonym zakresem, np.
K = 1000) są pomijane i wymieniane w komunikacie. Można wczytać konfiguracje wersji Supervised i Deep Learning: krok
2, brakujące wartości, cechy i skalowanie są przejmowane, etykieta i ustawienia modelu są po cichu pomijane.

## K-Means w tej aplikacji

**Uczenie nienadzorowane:** K-Means nie dostaje odpowiedzi. Wybiera K centrów startowych, przypisuje każdy wiersz do
najbliższego centrum (odległość euklidesowa w skalowanych cechach), przesuwa każde centrum do średniej jego wierszy
i powtarza to, aż nic się już nie zmienia. Całość `n_init` razy z innymi centrami startowymi – wygrywa rozwiązanie
z najmniejszą **inercją** (sumą kwadratów odległości od własnego centrum).

* **Bez podziału train/test.** Jak w kursie (`kmeans.fit(df.drop('Private', axis=1))`) K-Means uczy się na
  wszystkich wierszach. Wartości uzupełniające, kategorie i scaler pochodzą więc także ze wszystkich wierszy
  i obowiązują bez zmian dla nowych danych. Nie ma odpowiedzi, którą trzeba by ukryć przed modelem – wycieku danych
  jak w uczeniu nadzorowanym tu nie ma. Wyjątek: cecha z tej samej kolumny CSV co grupa porównawcza upiększyłaby
  porównanie w kroku 8 – krok 4 przed tym ostrzega.
* **Skalowanie jest kluczowe.** K-Means liczy odległości: bez skalowania kolumny z dużymi liczbami (Apps,
  F.Undergrad) prawie same decydują o klastrach. Domyślny jest więc `StandardScaler` (średnia 0, odchylenie
  standardowe 1 w każdej kolumnie). W kursie nie skalowano – do bezpośredniego porównania wybierz w kroku 4 „brak”.
* **Kolumny dummy bez drop_first.** Dla odległości lepiej zachować wszystkie kategorie (symetrycznie); z drop_first
  pominięta kategoria leżałaby bliżej wszystkich innych.
* **n_init = 10** jawnie: w scikit-learn ≥ 1.4 domyślne jest `n_init='auto'` – przy k-means++ to tylko jeden start.
* **Wartości odstające i niemożliwe** przesuwają centra: popraw je w kroku 2 akcją *ogranicz wartości* (w kursie:
  Grad.Rate 118 → 100, tutaj parametrem `,100`).

### Wybór K (krok 7)

| Miara | Znaczenie | dobrze |
|---|---|---|
| Inercja (metoda łokcia) | suma kwadratów odległości od własnego centrum; maleje z każdym kolejnym klastrem | **załamanie**: K, po którym inercja maleje wyraźnie wolniej (aplikacja szuka miejsca, w którym spadek na jedno K staje się co najmniej 2,5 raza mniejszy) |
| Silhouette (−1 … 1) | o ile bliżej każdy wiersz leży swojego klastra niż najbliższego innego | większa; > 0.7 silna, > 0.5 rozsądna, > 0.25 słaba, poniżej prawie brak struktury |
| Davies-Bouldin | stosunek rozrzutu w klastrach do odległości między klastrami | mniejszy |
| Calinski-Harabasz | zmienność między klastrami / zmienność w klastrach | większy |
| Wyjaśniona zmienność (R²) | 1 − inercja / całkowita zmienność | większa (ale zawsze rośnie z K) |

Jeśli łokieć i Silhouette wskazują to samo K, aplikacja je proponuje; w przeciwnym razie oba. Jeśli Silhouette
pozostaje niska dla każdego K, aplikacja nie daje propozycji z Silhouette. Na końcu liczy się, czy klastry da się
sensownie opisać w kroku 8. Od 20 000 wierszy krzywa jest liczona na stałej próbce, Silhouette od 5000 wierszy (jak
`silhouette_score(..., sample_size=5000)`); sam model zawsze używa wszystkich wierszy.

### Opisywanie i porównywanie klastrów (krok 8)

* **Numery klastrów są dowolne.** K-Means nie wie, że klaster 1 ma oznaczać „Yes”. Dlatego aplikacja pokazuje
  porównanie jak w kursie (numer klastra = numer grupy, w kursie 22 %) i porównanie po **przypisaniu**: każdy klaster
  dostaje grupę, która jest w nim najczęstsza (w kursie odpowiada to 78 %). Do tego prognoza naiwna (zawsze
  najczęstsza grupa), **ARI** i **NMI** – obie niezależne od numerów.
* **Profile:** średnia każdej cechy w klastrze w jednostkach oryginalnych i odchylenie od średniej w odchyleniach
  standardowych (z) – z tego powstają teksty „powyżej/poniżej średniej” i dobre nazwy klastrów. **Siła rozdzielania
  (R²)**: jaka część zmienności cechy przypada na różnice między klastrami.
* **PCA tylko do obrazka:** mapa 2D rzutuje dane na dwa kierunki o największej zmienności; K-Means nadal liczy na
  wszystkich kolumnach.
* **Nazwy** klastrów (np. „mała i prywatna”) są zapisywane z modelem i podawane w kroku 10 oraz w eksporcie.

## Wykrywanie anomalii: Isolation Forest, LOF, DBSCAN, autoenkoder

Te same kroki 1–5 co przy K-Means, w kroku 6 inna **metoda**. Wszystkie trzy są nienadzorowane: uczą się, jak
wiersze *zwykle* wyglądają, i dają każdemu wierszowi **wskaźnik anomalii** – wyższy = bardziej nietypowy. Wiersz jest
nietypowy (`Anomalia = 1`), gdy jego wskaźnik przekracza **próg**.

| Metoda | Idea | Wskaźnik | Próg | dobre do | uwaga |
|---|---|---|---|---|---|
| **Isolation Forest** | losowe drzewa tną dane w losowych miejscach – nietypowe wiersze zostają same już po kilku cięciach | `−score_samples`, 0 … 1 (ok. 0.5 normalny) | contamination = auto → 0.5, inaczej kwantyl dla wybranego udziału | standard dla tabel; szybki, także przy wielu wierszach i cechach; nie wymaga skalowania | znajduje raczej globalne niż lokalne wartości odstające |
| **Local Outlier Factor (LOF)** | gęstość wokół wiersza w porównaniu z gęstością wokół jego `n_neighbors` sąsiadów | LOF ≈ 1 normalny, wyraźnie powyżej 1 nietypowy | auto → 1.5, inaczej kwantyl | lokalne wartości odstające obok gęstych grup | jeśli wartości odstające same tworzą małą grupę, `n_neighbors` musi być większe niż ta grupa; wolniejszy przy bardzo wielu wierszach; skaluj dane |
| **DBSCAN** | punkt rdzeniowy = co najmniej `min_samples` wierszy w promieniu `eps`; co nie należy do żadnego klastra, jest **szumem** | odległość do najbliższego punktu rdzeniowego w wielokrotnościach eps | 1 (= eps): szum ⇔ wskaźnik powyżej 1 | klastry dowolnego kształtu i wartości odstające w jednym kroku; nie trzeba K | `eps` zależy od skalowania i liczby cech – które pasuje, pokazuje krok 7 (k-odległość) |
| **Autoenkoder** | sieć neuronowa z wejściem = celem: odtwarza każdy wiersz przez **wąskie gardło** (np. 16 → 3 → 16 neuronów) | **błąd rekonstrukcji** (RMSE dla wiersza, w skalowanych cechach) | stały udział (contamination, domyślnie 1 %) – kwantyl błędów w treningu | nietypowe **kombinacje** cech, które pojedynczo wyglądają normalnie; zależności nieliniowe, wiele cech | wymaga skalowania i prawdziwego wąskiego gardła (mniej neuronów w środku niż cech); trening trwa dłużej, wynik zależy od seeda |

* **contamination to decyzja, nie prawda:** ile wierszy ma zostać zgłoszonych? Lepiej kilka wyraźnych anomalii
  (0.5–2 %) niż długa lista. Krok 7 pokazuje **załamanie krzywej wskaźnika** – za nim wiersze prawie się już nie
  różnią – i proponuje je.
* **Co czyni wiersz nietypowym?** Krok 8 pokazuje dla każdego wiersza cechy o największym odchyleniu od średniej
  (z, w odchyleniach standardowych) i profil *nietypowe a normalne* – wgląd w dane, nie w model.
* **Porównanie ze znanym podziałem** (np. *Class* w danych o oszustwach): model nigdy go nie widział. Przy dwóch
  grupach rzadsza liczy się jako „pozytywna”: **precision** (udział prawdziwych przypadków wśród zgłoszeń),
  **recall** (udział znalezionych prawdziwych przypadków); **ROC-AUC** i **average precision** oceniają wskaźnik
  niezależnie od progu (przypadek: 0.5 albo udział grupy pozytywnej). W kroku 10 grupa pozytywna pozostaje ta
  z treningu – także wtedy, gdy w nowych danych (np. próbce z wieloma oszustwami) rzadsza jest druga.
* **eps (DBSCAN):** wartość startowa w kroku 6 pochodzi z k-odległości próbki i jest liczona od nowa po nowym
  przygotowaniu (np. innym scalerze), innym `min_samples` albo `metric` – tak samo propozycja przejęta w kroku 7; eps
  wpisane ręcznie zostaje. Przy wielu identycznych wierszach (np. same cechy kategoryczne) k-odległość często wynosi
  0; krok 7 proponuje wtedy najmniejsze eps (1e-6): identyczne wiersze tworzą klastry, rzadkie kombinacje są szumem.
* **Autoenkoder = deep learning bez etykiety:** sieć (scikit-learn `MLPRegressor`, optymalizator Adam, bez
  TensorFlow) uczy się tylko typowych wzorców; **błąd rekonstrukcji** pokazuje, co do nich nie pasuje. Po treningu krok
  6 pokazuje **krzywą uczenia** (błąd w każdej epoce; z early stopping R² na 10 % odłożonych wierszy – model zachowuje
  wagi najlepszej epoki). Krok 8 pokazuje dla wiersza **najgorzej odtworzone cechy** (wgląd w model: może tam być
  także normalna wartość, jeśli nie pasuje do pozostałych cech) i **mapę 2D wąskiego gardła** – własną, nieliniową
  kompresję sieci (przy więcej niż 2 neuronach przez PCA). Próg jest zapisywany razem z modelem (`threshold_`);
  pokazany kod liczy dokładnie to samo.
* **Warstwy i aktywacja:** propozycja na podstawie liczby cech (8 → 8,2,8; 11 → 16,3,16; 30 → 32,8,32), głębiej np.
  32,16,4,16,32. `tanh` w małych sieciach zwykle działa lepiej niż `relu`. Jeśli najwęższa warstwa ma co najmniej tyle
  neuronów, ile jest cech, sieć może po prostu przepuścić wiersze – aplikacja ostrzega.
* **Trening także na małych danych:** batch_size = liczba wierszy/20 (8 … 200), czyli ok. 20 kroków uczenia na epokę –
  przy domyślnym ustawieniu scikit-learn (200 wierszy na krok) sieć przy kilkuset wierszach robiła tylko jeden–dwa kroki
  na epokę i prawie nie znajdowała nietypowych kombinacji. Early stopping dopiero od 200 wierszy (inaczej do walidacji
  zostałoby za mało wierszy); poniżej sieć uczy się na wszystkich wierszach, a krok 6 o tym informuje.
* **Nie ograniczaj tego, co chcesz monitorować.** *Ogranicz wartości* w kroku 2 obcina właśnie te skrajne wartości,
  które tworzą anomalię – przy wykrywaniu anomalii poprawiaj tylko wartości niemożliwe.
* **Skalowanie:** LOF i DBSCAN liczą odległości → StandardScaler w kroku 4. Isolation Forest tnie po jednej cesze
  i nie potrzebuje skalowania (choć ono nie szkodzi).
* **Szeregi czasowe bez zaglądania w przyszłość:** wskaźniki każdego wiersza licz tylko z przeszłości (okna
  kroczące, wartości z względem *wcześniejszych* wartości – jak w skrypcie Bybit), trenuj na wcześniejszym okresie,
  a późniejszy oceniaj w kroku 10. Jeśli anomalie skupiają się w jednym okresie, to raczej nowy reżim niż wiele
  pojedynczych przypadków → wytrenuj ponownie na nowszych danych.
* **Nowe dane (krok 10):** to samo przygotowanie i ten sam próg co w treningu. LOF jest trenowany z `novelty=True`
  (inaczej nie mógłby oceniać nowych danych); przy DBSCAN nowy wiersz dostaje klaster najbliższego punktu
  rdzeniowego, jeśli jest on najwyżej o eps dalej – inaczej −1 (szum = anomalia).
* Tak jak przy K-Means pokazywany kod Python liczy dokładnie to samo co aplikacja (wskaźnik, anomalie, próg, krzywe
  dla kroku 7, profile, eksport).

## Bybit → CSV (tools/bybit_features.py)

Buduje z publicznych danych rynkowych API Bybit v5 plik CSV ze wskaźnikami do wykrywania anomalii – bez klucza API,
tylko z biblioteką standardową i pandas/numpy (obie aplikacja i tak potrzebuje). Aplikacja rozpoznaje plik po
kolumnach i w kroku 2 proponuje pasującą przykładową konfigurację.

```bash
python tools/bybit_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 120
python tools/bybit_features.py --top 20 --interval 4h --days 365 --split 2026-09-01
python tools/bybit_features.py --mode cross --top 100 --interval 1h --days 30
python tools/bybit_features.py --help
```

**Szereg czasowy** (domyślnie, `--mode series`): jeden wiersz na symbol i zamkniętą świecę – „Które godziny/dni były
nietypowe?”

| Kolumna | Znaczenie |
|---|---|
| `row`, `symbol`, `time` | nazwa wiersza (symbol + początek świecy w UTC), symbol, początek świecy – nie są cechami |
| `ret_pct`, `ret_z` | zwrot świecy w %; jako wartość z względem poprzednich świec (zwroty logarytmiczne) |
| `range_pct`, `range_z` | rozpiętość max–min w % ceny otwarcia; jako wartość z |
| `turnover_z` | obrót w USD jako wartość z – szok wolumenu |
| `vola_regime` | log2(zmienność w ostatnim oknie / zmienność wcześniej) – 0 normalnie, +1 dwa razy większa |
| `er` | efficiency ratio w oknie: ruch netto / suma wszystkich ruchów – 0 bokiem, 1 prosty trend |
| `cs_spread_pct` | szacowany spread bid-ask w % (Corwin-Schultz z max/min), średnia w oknie – stres płynności |
| `corr_btc`, `excess_z` | korelacja z benchmarkiem (domyślnie BTC) w oknie; zwrot minus zwrot benchmarku jako wartość z – własny ruch coina |
| `funding_bp`, `funding_z` | funding znany na zamknięciu świecy w punktach bazowych; jako wartość z (tylko kontrakty perpetual) |
| `oi_chg_pct`, `oi_z` | zmiana open interest w % w czasie jednej świecy, kończącym się jeden interwał OI przed zamknięciem świecy (świece 1h: 15 min); jako wartość z (tylko kontrakty perpetual, świece od 5 minut) |

`ret_pct`, `range_pct` i `oi_chg_pct` służą do czytania; przykładowa konfiguracja bierze pozostałe 11 kolumn jako
cechy (Isolation Forest, 1 % nietypowych). Wskazówka: wybierz w kroku 4 `symbol` jako znany podział – wtedy krok 8
pokaże, które symbole mają najwięcej nietypowych świec.

**Przekrój** (`--mode cross`): jeden wiersz na symbol za cały okres – „Które tokeny zachowują się inaczej niż reszta?”
Kolumny: `ret_window_pct`, `vola_ann_pct` (w skali roku), `er`, `cs_spread_pct`, `max_dd_pct` (największy spadek),
`corr_btc`, `beta_btc`, `vola_trend` (zmienność drugiej względem pierwszej połowy), `turnover_log10` (średni obrót
dzienny), `ret_skew`, `ret_kurt` (grube ogony), `funding_mean_bp`, `oi_chg_pct` – przykładowa konfiguracja zgłasza
5 % tokenów.

* **Bez zaglądania w przyszłość:** każdy wskaźnik świecy używa tylko danych do jej zamknięcia – wartości z porównują
  z *poprzednimi* `--baseline` świecami (domyślnie 30 dni), okna (`--window`, domyślnie 1 dzień) kończą się na
  świecy, funding i open interest to wartości znane na zamknięciu świecy, trwająca świeca jest pomijana. Rozbieg dla
  wartości z skrypt pobiera dodatkowo przed startem. Sprawdza to test: obcięcie danych → wcześniejsze wiersze
  pozostają co do bitu takie same.
* **Open interest ostrożnie:** Bybit nie dokumentuje, czy znacznik czasu wartości open interest oznacza początek jej
  interwału, czy moment pomiaru. Skrypt zakłada więc, że wartość jest znana dopiero jeden interwał po swoim znaczniku
  czasu, i pobiera open interest w drobnej siatce (najwyżej ¼ świecy, co najmniej 5 min) – dzięki temu zmiana nadal
  obejmuje prawie całą świecę. Przy świecach 1- i 3-minutowych open interest jest pomijany (istnieje tylko co
  5 minut).
* `--split DATA` zapisuje `<nazwa>_train.csv` (wcześniej) i `<nazwa>_new.csv` (od tej daty): trenuj na pierwszej
  części, drugą oceniaj w kroku 10 – jak na żywo.
* `<nazwa>_columns.csv` opisuje kolumny – w kroku 1 w sekcji *Opcjonalnie: wczytaj opisy kolumn*.
* Symbole: `--symbols` albo `--top N` (największy obrót 24h, bez stablecoinów i kontraktów z terminem
  wygaśnięcia); `--category` linear (kontrakty perpetual USDT, domyślnie), inverse albo spot (bez fundingu i open
  interest); `--interval` 1 3 5 15 30 60 120 240 360 720 D W albo 15m, 1h, 4h, 1d, 1w; poza tym `--start`/`--end`,
  `--no-funding`, `--no-oi`, `--benchmark`, `--keep-ohlcv`.
* **Limity API:** najwyżej 600 zapytań na 5 sekund z jednego IP – skrypt czeka między zapytaniami (`--pause`) i
  ponawia przy przeciążeniu. Jeśli Bybit odpowie **HTTP 403**, przekroczono limit IP albo dostęp z tej sieci/regionu
  jest zablokowany – odczekaj co najmniej 10 minut. `--cache FOLDER` (np. `bybit_cache`, wpisany w `.gitignore`)
  zachowuje odpowiedzi; przy stałych `--start`/`--end` przebieg można potem powtórzyć bez sieci z `--offline`.
* Jeśli symbol ma za krótką historię (niedawno notowany), jego pierwsze wiersze odpadają; kolumnę z ponad 20 %
  braków (np. open interest bez historii) skrypt pomija i o tym informuje.

## Nowe kolumny i akcje (krok 2)

| Rodzaj | Przykład | Wynik |
|---|---|---|
| Przypisz wartości | z `Private`: `Yes=1; No=0` | nowa kolumna (1/0); niewymienione wartości pozostają puste lub `sonst=…` |
| Formuła | `Accept / Apps` | liczba dla każdego wiersza; dozwolone `+ - * / ** // %`, nawiasy, `abs log log10 sqrt exp round` |
| Warunek → 0/1 | `Grad.Rate > 80` (nazwy z kropką w \`backtickach\`) | 1 = spełniony, 0 = niespełniony, puste przy brakach |
| Kopia + akcja | `Grad.Rate` + „ogranicz wartości” | oryginał zostaje, wynik jako dodatkowa kolumna |
| Akcja *ogranicz wartości* | parametr `,100` · `0,100` · `10,` · z przecinkiem dziesiętnym `0,5;99,5` | wartości poniżej min / powyżej max są ustawiane na granicę (`clip`) |

Formuły są sprawdzane: tylko obliczenia, porównania i wymienione funkcje – żadnego dowolnego kodu Python; limity
długości, zagnieżdżenia i wykładników zapobiegają zablokowaniu programu przez formułę. Nowe kolumny należą do
zapisanej receptury i w kroku 10 są automatycznie liczone dla nowych danych.

## Krok 10: przypisywanie nowych danych

* **Nic nie jest dopasowywane od nowa.** Nowe dane przechodzą przez zapisaną *recepturę* (`preprocessing.json`): te
  same akcje kolumn i nowe kolumny, wartości uzupełniające, kolumny dummy w tej samej kolejności, ten sam scaler.
  Potem każdy wiersz trafia do **najbliższego centrum** – centra pozostają bez zmian.
* **Te same surowe kolumny co w treningu** – aplikacja pokazuje potrzebne kolumny i oferuje szablon CSV. Brakujące
  kolumny → komunikat błędu; dodatkowe kolumny są pomijane.
* **Uwagi dla każdego wiersza:** uzupełniono braki · nieznana kategoria · poza zakresem treningowym · **daleko od
  wszystkich klastrów** (dalej od najbliższego centrum niż 99 % wierszy treningowych tego klastra – wartość odstająca
  albo nowy typ) · **między dwoma klastrami** (odległość od najbliższego centrum ≥ 95 % odległości od drugiego).
* **Podsumowanie:** jakość danych, rozkład na klastry w porównaniu z treningiem, udział wartości odstających
  i przypisań na granicy; jeśli nowe dane zawierają znany podział, do porównania używane jest przypisanie klaster →
  grupa **z treningu** (nie wyznaczane na nowo).
* Jeśli nowe dane mają już kolumny o nazwie kolumny wynikowej (np. `Klaster`, `Uwaga` z wcześniejszego eksportu),
  są one w wyniku zastępowane – aplikacja je wymienia.

Bez aplikacji:

```python
import pipeline as pl
kmeans, recipe = pl.load_bundle("models/moj_model")
df_new, _ = pl.read_csv("nowe_dane.csv")
result, report, comparison = pl.assign_new(kmeans, recipe, df_new)
podsumowanie = pl.assess_assignment(recipe, result, report, comparison)
```

Modele anomalii (Isolation Forest, LOF, DBSCAN, autoenkoder) oceniają zamiast przypisywać – z progiem z treningu:

```python
model, recipe = pl.load_bundle("models/moj_model_anomalii")
result, report, comparison = pl.score_new(model, recipe, df_new)   # Anomalie, Anomalie-Score, Hinweis
podsumowanie = pl.assess_scoring(recipe, result, report, comparison)
```

## Uwagi do aktualnych wersji bibliotek

| starszy kod | problem dziś | w szablonie |
|---|---|---|
| `KMeans(n_clusters=2)` | scikit-learn ≥ 1.4: `n_init='auto'` = 1 start przy k-means++ – wynik bardziej zależy od przypadku | `KMeans(n_clusters=2, n_init=10, random_state=101)` |
| `df['Grad.Rate']['Cazenovia College'] = 100` | pandas: SettingWithCopyWarning (chained assignment), w pandas 3 bez efektu | akcja *ogranicz wartości* `,100` lub `df['Grad.Rate'].clip(upper=100)` |
| `sns.lmplot('Room.Board', 'Grad.Rate', data=df, …)` | seaborn ≥ 0.12: tylko `data` można podać bez nazwy | `sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, …)` |
| `sns.FacetGrid(df, hue='Private', size=6)` | od seaborn 0.9 `size` nazywa się `height` | histogram wg grup (`hist` ze wspólnymi przedziałami) |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | przy wczytywaniu może zostać wykonany dowolny kod z pliku | `.skops` domyślnie; `.joblib` tylko dla zaufanych plików |
| `df.corr()` | pandas ≥ 2: błąd przy kolumnach tekstowych | `df.corr(numeric_only=True)` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False zamiast 0/1 | `dtype=float` |
| `LocalOutlierFactor().fit_predict(X)`, później `lof.predict(nowe)` | bez `novelty=True` nie ma `predict` dla nowych danych (a z `novelty=True` nie ma `fit_predict`) | `LocalOutlierFactor(novelty=True)`; wskaźnik wierszy treningowych z `negative_outlier_factor_` |
| `IsolationForest(contamination=0.1, behaviour='new')` z poradników | `behaviour` nie istnieje od scikit-learn 0.24; 10 % „anomalii” to prawie zawsze za dużo | `contamination='auto'` (próg 0.5) albo świadomie mały udział – krok 7 pokazuje załamanie |

## Pliki

```
app.py                  interfejs Streamlit (tylko UI)
pipeline.py             logika obliczeń – działa też w notebooku (import pipeline as pl)
plots.py                wykresy (matplotlib/seaborn), zwracają Figure
codegen.py              generuje pokazywany lub eksportowany kod Python
i18n.py                 języki: tx() tłumaczy niemiecki tekst na bieżący język
locales/                tłumaczenia {niemiecki: (angielski, polski)}, jeden plik na moduł
configs/                przykładowe konfiguracje: College_Data jak w kursie, szereg czasowy Bybit, przekrój Bybit
.streamlit/config.toml  limit uploadu 1 GB, jasny motyw
tools/bybit_features.py dane rynkowe Bybit → CSV ze wskaźnikami do wykrywania anomalii
tests/                  testy pytest + syntetyczne dane testowe (college_like, mall_like, anomaly_like, FakeBybit …)
AUDIT.md                raport (po polsku): budowa, kontrole, ograniczenia
README.md               niemiecki · README_EN.md angielski · README_PL.md polski
models/                 powstaje przy zapisie (krok 9)
exports/                domyślny folder dla „Zapisz w folderze” (krok 5)
```

Rozszerzanie:
* nowa akcja kolumny → `ACTIONS` + `_transform_column()` w `pipeline.py` (+ `_action_lines()` w `codegen.py`)
* nowy wykres → funkcja w `plots.py` + wpis w `page_viz()` lub `page_profiles()` w `app.py`
* inna metoda → w `pipeline.py` `MODEL_NAMES`, `MODEL_CLASSES`, `MODEL_VARS`, `MODEL_WHY` i pola w `ClusterConfig`.
  Metoda wykrywania anomalii dodatkowo w `ANOMALY_MODELS`, `anomaly_params()`, `anomaly_scores()` (wskaźnik: wyższy =
  bardziej nietypowy) i `anomaly_threshold()` – trening, kroki 7–10 i zapis działają wtedy przez `train_anomaly()`,
  `anomaly_curve()` i `score_new()`; jeśli metoda potrzebuje dużo pamięci, limit w `size_problem()`; nowe typy do
  bezpiecznego wczytywania w `SKOPS_TRUSTED`. W `app.py` parametry w `anomaly_params_ui()` i wyjaśnienie w
  `HOW_ANOMALY`, w `codegen.py` `anomaly_model_code()` i `predict_code()`; `tests/test_anomaly.py` sprawdza, że kod
  liczy to samo co aplikacja. Metoda klastrowania jak K-Means: wzorem są `make_kmeans()`/`train_kmeans()` i
  `model_code()`.
* nowy lub zmieniony tekst → napisz po niemiecku w `tx("…")` (stałe: `N_("…")`) i dodaj tłumaczenie angielskie
  i polskie w odpowiednim pliku w `locales/`; `pytest tests/test_i18n.py` zgłasza brakujące tłumaczenia.

## Testy

```bash
pip install pytest
pytest -q                                  # 385 testów, ok. 6 minut
python tests/synthetic.py                  # syntetyczne dane College (777 wierszy, nie są to prawdziwe dane)
```

| Plik | sprawdza |
|---|---|
| `tests/test_clustering.py` | K-Means jak w kursie (= bezpośrednio scikit-learn), powtarzalność, łokieć i Silhouette (wyraźne grupy są znajdowane, brak propozycji bez struktury, próbki przy dużych danych), ocena (małe/dominujące klastry, brak zbieżności, powtórzone wiersze), profile w jednostkach oryginalnych, siła rozdzielania = wyjaśniona zmienność, PCA, porównanie ze znanymi grupami, wartości odstające i przypisania na granicy w nowych danych |
| `tests/test_codegen_parity.py` | pokazywany kod Python liczy **dokładnie** to samo co aplikacja: X, klastry, inercja, krzywa łokcia/Silhouette, profile, porównanie, eksport – także w przypadkach granicznych (separatory tysięcy, data jako liczba, inf, wartości grupowe, ogranicz wartości, nowe kolumny) |
| `tests/test_anomaly.py` | Isolation Forest, LOF, DBSCAN, autoenkoder: wskaźnik ⇔ próg, contamination, novelty, szum = anomalia, za małe eps, krzywe i propozycje dla kroku 7, najbardziej nietypowe wiersze, profil, PCA, zapis/wczytanie (.skops/.joblib), nowe dane, brak pomyłki z K-Means, uszkodzone receptury – a pokazywany kod liczy **dokładnie** to samo (wskaźnik, anomalie, próg, k-odległość, siatka eps, profile, eksport, krok 10); wyeksportowany skrypt działa; wykresy w przypadkach granicznych |
| `tests/test_autoencoder.py` | autoenkoder: warstwy (propozycja, sprawdzanie), próg = kwantyl błędów rekonstrukcji, contamination 1 % bez podania, znajduje wbudowane oszustwa, znajduje nietypowe kombinacje (gdzie Isolation Forest zawodzi), batch_size i early stopping na małych danych (nawet 10 wierszy), wąskie gardło = wyjście enkodera (tanh, relu), mapa 2D, najgorzej odtworzone cechy, zapis/odczyt bez pickle, niekompletne modele, krzywa uczenia, konfiguracja, krok 6 sprawdza warstwy i wyszarza early stopping przy małych danych |
| `tests/test_bybit_features.py` | skrypt Bybit z symulowanym API (bez sieci): stronicowanie w limitach API, ponawianie/403/cache, bez zaglądania w przyszłość, trwająca świeca pominięta, wskaźniki przeliczone ponownie, przekrój, spot, nieznane symbole, przykładowa konfiguracja pasuje do CSV, wbudowany krach jest najbardziej nietypowym wierszem |
| `tests/test_pipeline.py` | cały przebieg z danymi podobnymi do LendingClub, zapis/wczytanie (.skops/.joblib, ZIP), nowe dane (nieznane kategorie, wartości odstające, brakujące kolumny), wyeksportowany skrypt działa |
| `tests/test_new_features.py` | rok + miesiąc, nowe kolumny, *ogranicz wartości*, nowe kolumny dla nowych danych, zapis danych, podsumowanie |
| `tests/test_audit_regressions.py` … `test_audit4_regressions.py` | testy regresji z czterech audytów aplikacji bazowej, przeniesione na klastrowanie (liczby, daty, rozpoznawanie CSV, limity formuł, konfiguracje, ochrona ZIP, brak kodu z nazw w pokazywanym kodzie, zmiana języka, bezpieczne wczytywanie, tylko modele K-Means) |
| `tests/test_audit5_regressions.py` | audyt 5 (aplikacja Clustering): kod eksportu = aplikacja także po usunięciu wierszy/kolumn, brak nieproszonej grupy porównawczej, kolumny ID jako nazwa wiersza, tekst „jak w kursie” z prawdziwymi grupami, brak Markdown z obcych modeli, odrzucanie uszkodzonego `preprocessing.json`, puste klastry, uwaga o zmienionych ustawieniach |
| `tests/test_audit6_regressions.py` | audyt 6 (wykrywanie anomalii, skrypt Bybit): grupa pozytywna w kroku 10 jak w treningu, wartość startowa eps po nowym przygotowaniu, propozycja eps nigdy 0, ochrona pamięci dla DBSCAN i LOF (trening i tabela eps, kod = aplikacja), identyczne wiersze, małe dane, linia eps na wykresie k-odległości, open interest bez zaglądania w przyszłość, drobna siatka OI, brak OI przy świecach 1-minutowych |
| `tests/test_app.py` | test klikania interfejsu (wszystkie 10 kroków jak w kursie; CSV ze średnikiem, zmienione ustawienia, nieużyteczne wartości, ochrona przed nadpisaniem; nowe kolumny, błędy rysowania, eksport) |
| `tests/test_app_anomaly.py` | test klikania trzech modeli anomalii przez kroki 4–10 oraz CSV z Bybit z przykładową konfiguracją aż do oceny nowych danych |
| `tests/test_charts.py` | wykresy kroku 3 i wykresy klastrów (łokieć/Silhouette, wykres Silhouette, wielkości, profile, PCA, tabela krzyżowa) |
| `tests/test_i18n.py` | języki: każdy tekst przetłumaczony z tymi samymi placeholderami, CSV dla Excela wg języka (aplikacja = kod), generowany kod liczy tak samo w każdym języku, cała aplikacja po angielsku i polsku bez niemieckich pozostałości – także z modelami anomalii |

## Uwagi

* Bardzo duże pliki CSV lepiej wczytywać przez „Ścieżka na tym komputerze” niż przez upload.
* W trakcie treningu niczego nie klikaj – każde kliknięcie uruchamia skrypt od nowa i przerywa trening.
* K-Means jest szybki, ale przy bardzo wielu wierszach i `n_init = 10` trwa to chwilę: 400 000 wierszy × 80 kolumn
  to od kilku sekund do ok. minuty na jedno K (zmierzone na 2 rdzeniach procesora – im mniej wyraźne grupy, tym
  więcej iteracji). Krzywa łokcia od 20 000 wierszy używa więc próbki; wskazówka: próbka w panelu bocznym lub
  mniejsze `n_init`.
* **Używaj tylko lokalnie.** Pole ścieżki czyta każdy plik, do którego zalogowany użytkownik ma dostęp, a `models/`
  jest wspólne dla wszystkich użytkowników aplikacji. Nie wystawiaj więc aplikacji do internetu bez logowania.
  Wczytuj modele tylko z zaufanych źródeł (ZIP po rozpakowaniu może mieć najwyżej 2 GB). `.skops` jest wczytywany
  bez wykonywania kodu; `.joblib` to pickle i z wgranego ZIP jest wczytywany tylko po wyraźnym potwierdzeniu.
  Wczytywane są tylko modele tej aplikacji (K-Means, Isolation Forest, LOF, DBSCAN, autoenkoder) – modele wersji Supervised lub Deep Learning aplikacja odrzuca
  z jasnym komunikatem, podobnie jak niekompletny lub uszkodzony `preprocessing.json` i recepturę, która nie pasuje
  do modelu.
* **K-Means ma ograniczenia:** znajduje raczej okrągłe grupy podobnej wielkości i każdy wiersz przypisuje do
  klastra – także wartości odstające. Przy wydłużonych lub zagnieżdżonych grupach albo klastrach o bardzo różnej
  wielkości (zob. wykres Silhouette) lepiej sprawdzają się **DBSCAN** (do wyboru w kroku 6) albo klastrowanie
  hierarchiczne; wartości odstające znajduje się bezpośrednio za pomocą Isolation Forest albo LOF.
* LOF i DBSCAN porównują każdy wiersz z sąsiadami – przy bardzo wielu wierszach (od ok. 50 000) trwa to i zajmuje
  pamięć; Isolation Forest pozostaje szybki. K-odległość dla kroku 7 od 20 000 wierszy używa próbki, siatka eps jest
  liczona tylko do 20 000 wierszy.
* Autoenkoder uczy się epoka po epoce na wszystkich wierszach – przy bardzo wielu wierszach (od ok. 100 000) trwa to
  kilka minut. Wskazówka: próbka w pasku bocznym albo mniej epok.
* **Ochrona pamięci:** DBSCAN (scikit-learn) trzyma wszystkie sąsiedztwa jednocześnie w pamięci – przy zbyt dużym eps
  albo wielu identycznych wierszach rośnie to z kwadratem liczby wierszy. Aplikacja szacuje liczbę par sąsiadów na
  400 losowych wierszach i powyżej 40 mln par (ok. 1 GB) odmawia treningu z wyjaśnieniem; w tabeli eps w kroku 7
  pomija takie wartości eps. Tak samo LOF, gdy wiersze × `n_neighbors` przekracza 40 mln.
