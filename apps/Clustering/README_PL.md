# DATA-Analysis-Clustering (Streamlit + scikit-learn, K-Means)

Aplikacja Streamlit dla tabelarycznych danych CSV: wczytaj, przygotuj, zwizualizuj i podziel na grupy (klastry)
za pomocą **K-Means** – **uczenie nienadzorowane**: nie ma etykiety, K-Means sam szuka wierszy, które są do siebie
podobne. Potem aplikacja opisuje klastry (profile, mapa 2D), porównuje je – jeśli jest dostępny – ze znanym podziałem
i przypisuje nowe dane do klastrów. Do każdego kroku aplikacja pokazuje pasujący kod Python.

Wzorem jest projekt K-Means z kursu: **College_Data** – K-Means z K = 2 na wszystkich kolumnach liczbowych; kolumna
*Private* (Yes/No) jest przy tym pomijana i używana dopiero potem do porównania. Dla tego zbioru jest gotowa
przykładowa konfiguracja.

Aplikacja powstała z wersji Supervised (DATA-Analysis-SupervisedLearning): kroki 1–3 (wczytywanie, akcje kolumn,
nowe kolumny, wykresy), receptura (`preprocessing.json`), zapis/wczytywanie, konfiguracje i języki pozostały bez
zmian; nowe są krok 4 bez etykiety i bez podziału, K-Means w kroku 6, wybór K w kroku 7, ocena klastrów w kroku 8
i przypisywanie nowych danych w kroku 10.

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
4. **Krok 4:** *Przygotuj dane dla K-Means* (17 cech, Private tylko do porównania, StandardScaler)
5. **Krok 6:** *Rozpocznij trening* (K = 2) → krok 7 (wybór K), 8 (profile, porównanie z Private), 9, 10

Dla własnych danych: w kroku 4 wybierz cechy, opcjonalnie znany podział do porównania i kolumnę z nazwami; w kroku 6
wybierz K – krok 7 pokazuje, które K pasuje. Na wolnym komputerze włącz po lewej **próbkę** (`df.sample(frac=0.1)`).

## Kroki

| Strona | Zawartość |
|---|---|
| 1 · Dane i biblioteki | wczytanie CSV (upload lub ścieżka; separator `;` i przecinek dziesiętny są wykrywane automatycznie), przegląd, wersje bibliotek, opcjonalnie opisy kolumn |
| 2 · Przygotowanie 1 | jedna akcja na kolumnę (zachowaj, usuń, wyodrębnij liczbę, data → rok / miesiąc / rok + miesiąc, pierwsze/ostatnie N znaków, zamień wartości, top N, **ogranicz wartości (min, max)** …) · nowe kolumny: przypisz wartości, formuła, warunek → 0/1, kopia + akcja |
| 3 · Analiza i wizualizacja | wybór kolumn · przegląd, countplot, histogram, boxplot, scatter, heatmapa, korelacja z jedną kolumną, pairplot, udział wg kategorii, statystyki grup, jointplot, gęstość 2D, lmplot, histogram wg grup |
| 4 · Przygotowanie 2 – dane do klastrowania | **cechy**, opcjonalnie **znany podział** (tylko do porównania, np. Private) i **nazwa wiersza** (np. nazwa uczelni); brakujące wartości dla każdej kolumny; kolumny dummy i skalowanie (domyślnie StandardScaler, drop_first wyłączone) – wyuczone ze wszystkich wierszy, bez podziału; uwagi o silnie skorelowanych kolumnach i kolumnach bez zmienności. Znany podział jest proponowany tylko dla kolumny o takiej nazwie (Private, species, label, segment …); kolumna z identyfikatorem (np. CustomerID = 1, 2, 3 …) jest proponowana jako nazwa wiersza, nie jako cecha – wybrana jako cecha, wywołuje ostrzeżenie |
| 5 · Zapis danych | dane po kroku 2, dane do klastrowania po kroku 4 (nieskalowane lub skalowane) albo – po treningu – dane z **numerem klastra, nazwą klastra i odległością** od centrum; międzynarodowo lub dla Excela |
| 6 · Model i trening | **K-Means**: K (n_clusters), init (k-means++/random), n_init, max_iter, algorithm, seed; opcjonalnie krzywa łokcia i Silhouette dla kroku 7 – pasujący kod scikit-learn jest pokazywany na bieżąco, krzywe rosną w trakcie treningu |
| 7 · Wybór K | **metoda łokcia** (inercja dla K) i **Silhouette** dla K z oceną i propozycją (*Przejmij K = …*), Davies-Bouldin, Calinski-Harabasz, wyjaśniona zmienność (R²) i **wykres Silhouette** wytrenowanego modelu |
| 8 · Ocena klastrów | wielkość klastrów, **nazywanie klastrów**, **profile** (średnie w jednostkach oryginalnych, heatmapa odchylenia od średniej w σ, tekst dla każdego klastra), **siła rozdzielania cechy (R²)**, **mapa 2D (PCA)** z kolorem = klaster lub znany podział, dwie cechy względem siebie, **porównanie ze znanym podziałem** (tabela krzyżowa, przypisanie klaster → grupa, trafność, prognoza naiwna, ARI, NMI, macierz pomyłek, classification_report – także porównanie jak w kursie), **typowi przedstawiciele** klastrów, pojedynczy przypadek |
| 9 · Zapis i wczytanie modelu | `models/<nazwa>/model.skops` (bezpieczny, domyślny) lub `model.joblib` + `preprocessing.json` (receptura, ustawienia, nazwy klastrów, miary, przypisanie do grupy porównawczej), pobranie ZIP, cały pipeline jako skrypt Python |
| 10 · Przypisz nowe dane | nowy CSV lub formularz → to samo przygotowanie → **najbliższe centrum klastra** (`kmeans.predict`), odległość, uwagi, rozkład w porównaniu z treningiem, porównanie ze znanym podziałem (jeśli jest w danych), pobranie CSV i **podsumowanie** |

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

## Pliki

```
app.py                  interfejs Streamlit (tylko UI)
pipeline.py             logika obliczeń – działa też w notebooku (import pipeline as pl)
plots.py                wykresy (matplotlib/seaborn), zwracają Figure
codegen.py              generuje pokazywany lub eksportowany kod Python
i18n.py                 języki: tx() tłumaczy niemiecki tekst na bieżący język
locales/                tłumaczenia {niemiecki: (angielski, polski)}, jeden plik na moduł
configs/                college_beispiel.json (przykładowa konfiguracja College_Data jak w kursie)
.streamlit/config.toml  limit uploadu 1 GB, jasny motyw
tests/                  testy pytest + syntetyczne dane testowe (college_like, mall_like …)
AUDIT.md                raport (po polsku): budowa, kontrole, ograniczenia
README.md               niemiecki · README_EN.md angielski · README_PL.md polski
models/                 powstaje przy zapisie (krok 9)
exports/                domyślny folder dla „Zapisz w folderze” (krok 5)
```

Rozszerzanie:
* nowa akcja kolumny → `ACTIONS` + `_transform_column()` w `pipeline.py` (+ `_action_lines()` w `codegen.py`)
* nowy wykres → funkcja w `plots.py` + wpis w `page_viz()` lub `page_profiles()` w `app.py`
* inna metoda klastrowania → `MODEL_NAMES`, `ClusterConfig`, `make_kmeans()`/`train_kmeans()` w `pipeline.py`,
  `page_train()` w `app.py`, `model_code()` w `codegen.py`
* nowy lub zmieniony tekst → napisz po niemiecku w `tx("…")` (stałe: `N_("…")`) i dodaj tłumaczenie angielskie
  i polskie w odpowiednim pliku w `locales/`; `pytest tests/test_i18n.py` zgłasza brakujące tłumaczenia.

## Testy

```bash
pip install pytest
pytest -q                                  # 268 testów, ok. 3 minuty
python tests/synthetic.py                  # syntetyczne dane College (777 wierszy, nie są to prawdziwe dane)
```

| Plik | sprawdza |
|---|---|
| `tests/test_clustering.py` | K-Means jak w kursie (= bezpośrednio scikit-learn), powtarzalność, łokieć i Silhouette (wyraźne grupy są znajdowane, brak propozycji bez struktury, próbki przy dużych danych), ocena (małe/dominujące klastry, brak zbieżności, powtórzone wiersze), profile w jednostkach oryginalnych, siła rozdzielania = wyjaśniona zmienność, PCA, porównanie ze znanymi grupami, wartości odstające i przypisania na granicy w nowych danych |
| `tests/test_codegen_parity.py` | pokazywany kod Python liczy **dokładnie** to samo co aplikacja: X, klastry, inercja, krzywa łokcia/Silhouette, profile, porównanie, eksport – także w przypadkach granicznych (separatory tysięcy, data jako liczba, inf, wartości grupowe, ogranicz wartości, nowe kolumny) |
| `tests/test_pipeline.py` | cały przebieg z danymi podobnymi do LendingClub, zapis/wczytanie (.skops/.joblib, ZIP), nowe dane (nieznane kategorie, wartości odstające, brakujące kolumny), wyeksportowany skrypt działa |
| `tests/test_new_features.py` | rok + miesiąc, nowe kolumny, *ogranicz wartości*, nowe kolumny dla nowych danych, zapis danych, podsumowanie |
| `tests/test_audit_regressions.py` … `test_audit4_regressions.py` | testy regresji z czterech audytów aplikacji bazowej, przeniesione na klastrowanie (liczby, daty, rozpoznawanie CSV, limity formuł, konfiguracje, ochrona ZIP, brak kodu z nazw w pokazywanym kodzie, zmiana języka, bezpieczne wczytywanie, tylko modele K-Means) |
| `tests/test_audit5_regressions.py` | audyt 5 (aplikacja Clustering): kod eksportu = aplikacja także po usunięciu wierszy/kolumn, brak nieproszonej grupy porównawczej, kolumny ID jako nazwa wiersza, tekst „jak w kursie” z prawdziwymi grupami, brak Markdown z obcych modeli, odrzucanie uszkodzonego `preprocessing.json`, puste klastry, uwaga o zmienionych ustawieniach |
| `tests/test_app.py` | test klikania interfejsu (wszystkie 10 kroków jak w kursie; CSV ze średnikiem, zmienione ustawienia, nieużyteczne wartości, ochrona przed nadpisaniem; nowe kolumny, błędy rysowania, eksport) |
| `tests/test_charts.py` | wykresy kroku 3 i wykresy klastrów (łokieć/Silhouette, wykres Silhouette, wielkości, profile, PCA, tabela krzyżowa) |
| `tests/test_i18n.py` | języki: każdy tekst przetłumaczony z tymi samymi placeholderami, CSV dla Excela wg języka (aplikacja = kod), generowany kod liczy tak samo w każdym języku, cała aplikacja po angielsku i polsku bez niemieckich pozostałości |

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
  Wczytywane są tylko modele K-Means tej aplikacji – modele wersji Supervised lub Deep Learning aplikacja odrzuca
  z jasnym komunikatem, podobnie jak niekompletny lub uszkodzony `preprocessing.json` i recepturę, która nie pasuje
  do modelu.
* **K-Means ma ograniczenia:** znajduje raczej okrągłe grupy podobnej wielkości i każdy wiersz przypisuje do
  klastra – także wartości odstające. Przy wydłużonych lub zagnieżdżonych grupach albo klastrach o bardzo różnej
  wielkości (zob. wykres Silhouette) lepiej sprawdzają się inne metody (np. DBSCAN, klastrowanie hierarchiczne).
