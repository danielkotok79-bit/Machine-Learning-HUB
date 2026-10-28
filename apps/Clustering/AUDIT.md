# Raport – DATA-Analysis-Clustering (K-Means), wersja 1

> Nowa aplikacja na szkielecie **DATA-Analysis-SupervisedLearning** (wersja 9, po czterech audytach – szczegóły
> w `AUDIT.md` tamtej aplikacji). Ten raport opisuje, co przejęto, co jest nowe, jakie decyzje metodyczne podjęto,
> co znaleziono i poprawiono w trakcie budowy oraz jak całość sprawdzono. **Audyt 5** – pierwszy audyt kodu tej
> aplikacji (numeracja kontynuuje cztery audyty szkieletu) – jest na końcu pliku.

**Data:** 17.10.2026 · **Zakres:** `app.py`, `pipeline.py`, `plots.py`, `codegen.py`, `i18n.py`, `locales/`,
`configs/college_beispiel.json`, testy
**Metoda:** każda nowa funkcja z testem; liczbowe porównanie aplikacji z pokazywanym i eksportowanym kodem Python;
testy klikania (Streamlit AppTest) we wszystkich trzech językach; test w przeglądarce (Playwright) na działającym
serwerze z syntetycznymi danymi College (777 wierszy, nie są to prawdziwe dane z kursu).
**Wynik:** **247 testów** – wszystkie przechodzą (Python 3.13, pandas 3.0, scikit-learn 1.9, skops 0.16,
Streamlit 1.66); składnia sprawdzona dla Pythona 3.10 i 3.11; test w przeglądarce **42/42**. Sześć problemów
znalezionych w trakcie budowy jest poprawionych i zabezpieczonych testami (sekcja „Znalezione w trakcie budowy”).

---

## Co przejęto bez zmian

* **Kroki 1–3:** wczytywanie CSV (automatyczne rozpoznanie `;` i przecinka dziesiętnego, ścieżki w cudzysłowie,
  spacje w nagłówkach), wszystkie akcje kolumn, nowe kolumny (przypisanie wartości, formuła, warunek → 0/1,
  kopia + akcja – formuły bez dowolnego kodu, z limitami), wszystkie wykresy kroku 3.
* **Receptura** (`preprocessing.json`): wszystko, czego aplikacja uczy się z danych, jest zapisane i na nowych
  danych stosowane bez zmian – nic nie jest dopasowywane od nowa.
* **Zapis i odczyt:** `.skops` domyślnie (bez wykonywania kodu), `.joblib` tylko dla zaufanych plików (z ZIP-a tylko
  po potwierdzeniu), limit 2 GB po rozpakowaniu, zapis przez pliki tymczasowe, ochrona przed nadpisaniem.
* **Konfiguracje** z kontrolą typów i zakresów, **języki** DE/EN/PL, CSV dla Excela wg języka z ochroną przed CSV
  injection, ochrona przed Markdown z obcych nazw.
* **Wszystkie testy regresyjne z czterech audytów** (93 testy w `tests/test_audit*_regressions.py`) – przeniesione
  na klastrowanie tam, gdzie dotyczyły etykiety albo modeli supervised (np. „wczytywane są tylko modele K-Means”
  zamiast „pięć modeli”); wszystkie przechodzą.

## Co nowe (kroki 4–10)

| Krok | Supervised (wersja 9) | Clustering (ta aplikacja) |
|---|---|---|
| 4 | etykieta, podział train/test, uzupełnianie z wierszy treningowych | **bez etykiety i bez podziału**: cechy, opcjonalnie znany podział (tylko do porównania) i nazwa wiersza (tylko do wyświetlania); wartości uzupełniające, kategorie i scaler ze wszystkich wierszy; StandardScaler domyślnie, drop_first wyłączone; uwagi o silnie skorelowanych kolumnach (\|r\| ≥ 0,9), kolumnach bez zmienności i cechach z tej samej kolumny CSV co grupa porównawcza |
| 5 | dane z prognozą | dane z numerem klastra, nazwą klastra i odległością od centrum |
| 6 | pięć modeli | **K-Means** (K, init, n_init, max_iter, algorithm, seed) + krzywa łokcia i Silhouette rosnąca na żywo |
| 7 | krzywa walidacji / uczenia | **metoda łokcia**, **Silhouette**, Davies-Bouldin, Calinski-Harabasz, R², wykres Silhouette, propozycja K z przyciskiem „przejmij” |
| 8 | macierz pomyłek, ważność cech | wielkości i **nazwy** klastrów, **profile** (jednostki oryginalne + odchylenie w σ, tekst dla każdego klastra), **siła rozdzielania** cech (R²), **mapa PCA 2D**, dwie cechy, **porównanie ze znanym podziałem**, typowi przedstawiciele, pojedynczy przypadek |
| 9 | model + receptura | K-Means + receptura z nazwami klastrów, miarami i przypisaniem klaster → grupa |
| 10 | prognoza | **najbliższe centrum**, odległość, uwagi (daleko od wszystkich klastrów, między dwoma klastrami), rozkład w porównaniu z treningiem, porównanie z przypisaniem z treningu, podsumowanie |

## Decyzje metodyczne

| Decyzja | Dlaczego |
|---|---|
| Bez podziału train/test; wartości uzupełniające, kategorie i scaler ze wszystkich wierszy (`"fit_on": "all"`) | jak w kursie (`kmeans.fit(df.drop('Private', axis=1))`); nie ma odpowiedzi, którą trzeba by ukryć przed modelem, więc nie ma też wycieku danych jak w uczeniu nadzorowanym |
| Znany podział (Private) nigdy nie jest cechą; cecha z tej samej kolumny CSV co grupa → ostrzeżenie | inaczej porównanie w kroku 8 byłoby upiększone |
| StandardScaler domyślnie, „brak” do wyboru | K-Means liczy odległości – bez skalowania Apps i F.Undergrad decydowałyby prawie same; do porównania 1:1 z kursem wybrać „brak” |
| drop_first wyłączone | z drop_first pominięta kategoria (same zera) leży bliżej wszystkich pozostałych – odległości przestają być symetryczne |
| `n_init=10` jawnie | scikit-learn ≥ 1.4 ma `n_init='auto'` = tylko jeden start przy k-means++; 10 startów odpowiada też dawnemu zachowaniu z czasów kursu |
| Łokieć metodą ilorazu: spadek inercji na jedno K przed punktem ÷ spadek po nim ≥ 2,5; liczą się tylko miejsca ze spadkiem przed ≥ 5 % całego spadku | niezależna od skali; gdy krzywa maleje równomiernie, aplikacja mówi „brak wyraźnego łokcia” zamiast zgadywać |
| Silhouette nie daje propozycji, gdy jej maksimum ≤ 0,25 | przy prawie braku struktury maksimum jest przypadkowe |
| Krzywa od 20 000 wierszy na stałej próbce, Silhouette od 5000 wierszy (`sample_size=5000`) | czas obliczeń; ta sama próbka w aplikacji i w pokazywanym kodzie (ten sam seed); sam model zawsze na wszystkich wierszach |
| Profile przez odwrócenie skalowania; z = (średnia klastra − średnia ogółem) / odchylenie standardowe | czytelne w jednostkach oryginalnych i porównywalne między cechami |
| Opis klastra: cechy z \|z\| ≥ 0,3 (najwyżej po 3 powyżej i poniżej średniej) | zob. K3 poniżej |
| PCA tylko do obrazka (najwyżej 5000 punktów) | K-Means liczy na wszystkich cechach; klastry nakładające się w 2D mogą być rozdzielone w pełnej przestrzeni – aplikacja to mówi |
| Porównanie z grupą: przypisanie większościowe klaster → grupa, porównanie „jak w kursie” (numer klastra = numer grupy; tylko gdy K = liczba grup), prognoza naiwna, ARI, NMI | numery klastrów są dowolne; ARI i NMI od numerów nie zależą |
| Nowe dane porównywane z przypisaniem **z treningu** | wyznaczanie przypisania od nowa na nowych danych upiększałoby wynik |
| „Daleko od wszystkich klastrów”: odległość większa niż 99. percentyl odległości wierszy treningowych tego klastra; „między dwoma klastrami”: odległość od najbliższego ≥ 95 % odległości od drugiego centrum | ostrzeżenia dla wartości odstających, nowych typów i przypisań na granicy |

## Zgodność z projektem K-Means z kursu (College_Data)

| Notebook z kursu | W aplikacji |
|---|---|
| `pd.read_csv('College_Data', index_col=0)` | krok 1; pierwsza kolumna (`Unnamed: 0`) zostaje w kroku 2 jako „College” – nazwa wiersza, nie cecha (przykładowa konfiguracja) |
| `sns.lmplot('Room.Board', 'Grad.Rate', data=df, hue='Private', fit_reg=False)`, scatter F.Undergrad / Outstate, `FacetGrid` z histogramami Outstate i Grad.Rate | krok 3: scatter / lmplot z kolorem = Private, histogram wg grup |
| `df['Grad.Rate']['Cazenovia College'] = 100` | akcja „ogranicz wartości” z parametrem `,100` (w przykładowej konfiguracji) |
| `KMeans(n_clusters=2)`, `kmeans.fit(df.drop('Private', axis=1))` | krok 4: Private jako znany podział, 17 cech; krok 6: K = 2, `n_init=10`, `random_state=101` |
| `kmeans.cluster_centers_` | krok 6 (kod), krok 8 (profile w jednostkach oryginalnych) |
| `converter` (Yes → 1, No → 0), `confusion_matrix`, `classification_report` | krok 8: porównanie „jak w kursie” (No = 0, Yes = 1) i po przypisaniu, macierz pomyłek i classification_report |

`test_kmeans_like_the_course_without_scaling`: bez skalowania aplikacja daje **te same klastry i tę samą inercję**
co `KMeans(n_clusters=2, n_init=10, random_state=101)` wywołany bezpośrednio na liczbach. Przykładowa konfiguracja
używa StandardScaler, więc wyniki różnią się od kursu – do porównania 1:1 wystarczy w kroku 4 wybrać „brak”.

## Znalezione w trakcie budowy (poprawione)

| # | Problem | Skutek | Poprawka |
|---|---|---|---|
| K1 | Łokieć wyznaczany metodą „Kneedle” (największa odległość od prostej łączącej końce krzywej) | dla czterech wyraźnie rozdzielonych grup wskazywał K = 3 | metoda ilorazu (zob. decyzje); testy: wyraźne grupy, prosta, szum bez grup, brak spadku |
| K2 | Ocena Silhouette ≤ 0,25 brzmiała „brak struktury” | przy wielu cechach (College: 17) Silhouette jest często niska, choć klastry dają się sensownie opisać – ocena zniechęcała bez powodu | „prawie brak struktury – klastry mocno na siebie nachodzą (przy wielu cechach Silhouette jest często niska; czy klastry mimo to mają sens, pokazują profile w kroku 8)”; przy niskiej Silhouette dla każdego K brak propozycji z Silhouette |
| K3 | Opis klastra tylko dla cech z \|z\| ≥ 0,5 | przy K = 2 duży klaster (77 % wierszy) często nie miał żadnego opisu – jego średnie leżą blisko średniej ogółem | próg 0,3 |
| K4 | Uwaga „między dwoma klastrami” od stosunku odległości 0,9 | zbyt wiele wierszy oznaczonych jako niepewne | próg 0,95 |
| K5 | Eksport w pokazywanym kodzie (krok 5, format Excel) nie zabezpieczał tekstów zaczynających się od `=`, `+`, `-`, `@` (albo tabulatora) | plik z kodu różnił się od pliku z aplikacji, a Excel mógł wykonać tekst jako formułę (CSV injection) – znalezione przez test zgodności | ta sama ochrona co w aplikacji; test porównuje oba pliki bajt po bajcie |
| K6 | Domyślna nazwa wiersza (krok 4) dla danych bez kolumny tekstowej | proponowana była ciągła kolumna liczbowa (np. kwota) | proponowane są tylko kolumny tekstowe; jednoznaczne liczby całkowite (ID) można wybrać ręcznie |

## Co sprawdzono i działa poprawnie

* **Pokazywany i eksportowany kod = aplikacja** (`tests/test_codegen_parity.py`, porównanie dokładne, bez
  tolerancji): X, klastry, inercja, krzywa łokcia, Silhouette, Davies-Bouldin i Calinski-Harabasz (także na
  próbkach), profile, porównanie z grupą i wszystkie eksporty – również przy separatorach tysięcy, dacie jako
  liczbie, inf, wartościach grupowych, akcji „ogranicz wartości”, nowych kolumnach, MinMaxScaler z drop_first,
  `init='random'` i `algorithm='elkan'`. Cały skrypt z kroku 9 uruchamia się i daje te same klastry.
* **Wybór K:** wyraźne grupy są znajdowane przez obie metody (K = 4 → propozycja „przejmij K = 4”, Silhouette > 0,7,
  zgodność z prawdziwymi grupami 100 %, ARI = 1); dane bez struktury → brak propozycji; ostrzeżenia o bardzo małym
  klastrze, dominującym klastrze, braku zbieżności i powtórzonych wierszach (uwaga scikit-learn zamiast błędu).
* **Ocena:** profile = średnie w jednostkach oryginalnych; suma siły rozdzielania = wyjaśniona zmienność; kolumny
  dummy jednej kategorii liczone razem; przypisanie większościowe także przy K > liczba grup (grupy bez klastra są
  wymieniane).
* **Nowe dane:** te same klastry co w treningu dla tych samych wierszy (także po zapisie i odczycie `.skops` /
  `.joblib` i z ZIP-a); nieznane kategorie, brakujące kolumny, wartości odstające i przypisania na granicy dają
  zrozumiałe uwagi.
* **Bezpieczeństwo:** odczyt odrzuca modele inne niż K-Means i receptury wersji Supervised / Deep Learning z jasnym
  komunikatem; `.skops` z obcym typem → odmowa; wszystkie zabezpieczenia z audytów 1–4 (ZIP, ścieżki, formuły,
  Markdown) działają dalej – testy regresyjne przechodzą.
* **Języki:** katalog tłumaczeń kompletny (828 tekstów, te same symbole zastępcze w trzech językach); cała aplikacja
  po angielsku i po polsku bez niemieckich pozostałości (AppTest i przeglądarka); generowany kod liczy w każdym
  języku to samo.
* **Python 3.10:** składnia wszystkich plików sprawdzona parserem 3.10 (`ast`) i 3.11 (f-stringi sprzed PEP 701);
  brak funkcji bibliotek dostępnych dopiero od 3.11.

## Test w przeglądarce (Playwright, prawdziwy serwer)

Kroki 1–10 po niemiecku jak w kursie: wczytanie College_Data (ścieżka), rozpoznanie zbioru i przykładowa
konfiguracja (Grad.Rate ograniczone do 100 – „1 wartość zmieniona”), scatterplot, krok 4 z Private i College już
wybranymi (17 cech, uwaga o korelacjach), eksport CSV, trening K = 2 z kodem `n_init=10`, krzywe
i wykres Silhouette, ocena (porównanie z Private, typowi przedstawiciele, 7 wykresów, edytor nazw, pojedynczy
przypadek), zapis `.skops` i odczyt, przypisanie 90 nowych wierszy z podsumowaniem. Następnie kroki 4, 6, 7, 8 i 10
po angielsku i po polsku – bez błędów i bez niemieckich pozostałości. Wynik: **42/42**. Błędów w aplikacji test
nie znalazł; poprawiono dwie kontrole w samym skrypcie testowym (Streamlit 1.66 pokazuje wybraną wartość listy
w polu wejściowym, nie w tekście strony; polskie „nie wie” to nie niemiecki).

## Ograniczenia i uwagi (opisane w README)

* **Założenia K-Means:** raczej okrągłe grupy podobnej wielkości; każdy wiersz trafia do jakiegoś klastra – także
  wartości odstające. Wydłużone lub zagnieżdżone grupy lepiej rozpoznają inne metody (DBSCAN, klastrowanie
  hierarchiczne).
* **Łokieć jest ostrożny:** przy łagodnie zakrzywionej krzywej (np. spadek 2224, potem 1026 – iloraz 2,2)
  aplikacja nie wskazuje łokcia. To celowe – lepiej „brak wyraźnego łokcia” niż przypadkowe K; Silhouette i profile
  w kroku 8 pomagają wtedy zdecydować.
* **Próbki:** od 20 000 wierszy krzywa, od 5000 wierszy Silhouette są szacunkiem na stałej próbce; model zawsze
  używa wszystkich wierszy.
* **Zmiana języka w trakcie pracy:** protokół kroków 2 i 4 zostaje w starym języku, dopóki krok nie zostanie
  wykonany ponownie (zachowanie przejęte ze szkieletu; test w przeglądarce sprawdza, że ponowne „Przygotuj dane”
  daje protokół w nowym języku).
* **`.joblib` z folderu `models/`** wczytuje się bez pytania – to własny folder użytkownika; aplikacja jest
  przeznaczona do użytku lokalnego.
* **Modele nie są wymienne między aplikacjami:** model K-Means działa tylko tutaj, modele Supervised / Deep Learning
  tylko tam; konfiguracje (`.json`) można wczytywać w każdej z trzech aplikacji (nieznane ustawienia są pomijane).

## Testy (stan wersji 1)

| Plik | Testy | sprawdza |
|---|---|---|
| `tests/test_clustering.py` | 26 | silnik K-Means: kurs = scikit-learn, powtarzalność, łokieć i Silhouette, próbki, oceny, profile, siła rozdzielania, PCA, porównanie z grupami, przypisanie nowych danych |
| `tests/test_codegen_parity.py` | 26 | pokazywany i eksportowany kod = aplikacja (dokładnie) |
| `tests/test_pipeline.py` | 8 | cały przebieg, zapis/odczyt, nowe dane, eksportowany skrypt |
| `tests/test_new_features.py` | 29 | rok + miesiąc, nowe kolumny, ogranicz wartości, eksport, podsumowanie |
| `tests/test_audit*_regressions.py` | 93 | regresje z czterech audytów szkieletu, przeniesione na klastrowanie |
| `tests/test_charts.py` | 46 | wykresy kroku 3 i wykresy klastrów |
| `tests/test_i18n.py` | 16 | tłumaczenia, CSV dla Excela wg języka, aplikacja po angielsku i polsku |
| `tests/test_app.py` | 3 | testy klikania: wszystkie 10 kroków jak w kursie i przypadki szczególne |

Razem **247 testów** – wszystkie przechodzą; uruchomienie `pytest -q` (ok. 3 minuty).

## Uruchomienie

* `pip install -r requirements.txt`, `streamlit run app.py` (te same biblioteki co wersja Supervised).
* Zmienne środowiskowe: `CL_TEMPLATE_LANG` (język startowy) i `CL_TEMPLATE_MODELS` (folder modeli) – odpowiedniki
  `ML_TEMPLATE_LANG` / `ML_TEMPLATE_MODELS` wersji Supervised.

---

# Audyt 5 – kod aplikacji Clustering (17.10.2026)

**Zakres:** cały kod wersji 1: silnik w `pipeline.py` (krok 4 bez etykiety, trening, krzywa łokcia i Silhouette,
ocena w kroku 7, profile, siła rozdzielania, PCA, porównanie z grupą, zapis i odczyt, przypisanie nowych danych,
podsumowanie), strony 4–10 w `app.py`, generowany kod w `codegen.py`, wykresy klastrów w `plots.py`, nowe
tłumaczenia i testy.
**Metoda:** przegląd każdej funkcji; ok. 30 sond z nietypowymi danymi (10 wierszy i K = 9, same powtórzone wiersze,
kolumny bez zmienności, tylko kategorie, grupa jako liczba 0/1 i z nieznanymi wartościami w nowych danych, 25 000
wierszy z próbkami, cechy bool, nazwy z Markdown); pełne przejścia kliknięciami (AppTest) przez kroki 4–10 dla pięciu
zbiorów (Mall, Iris, duplikaty, minimalny, stały); przygotowane pliki ZIP z uszkodzonym `preprocessing.json`;
eksportowany skrypt z usuwaniem wierszy i kolumn; test w przeglądarce. Każdy potwierdzony błąd jest poprawiony i ma
test regresyjny (`tests/test_audit5_regressions.py`).

**Wynik:** 9 naprawionych problemów – 3 średnie, 6 drobnych (1 związany z bezpieczeństwem). Żaden nie zmieniał
klastrów przy ustawieniach z przykładowej konfiguracji College.

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| B1 | średnia | Pokazywany kod kroku 4 zmieniał `df` (usuwał wiersze z lukami i kolumny z wieloma lukami, zamieniał inf na NaN) | kod eksportu z kroku 5, uruchomiony po kodzie kroku 4, zapisywał inne dane niż aplikacja: „Dane po kroku 2” bez usuniętych wierszy i kolumn i z NaN zamiast inf (sprawdzone: 280 × 18 zamiast 300 × 19), „Dane z klastrami” bez usuniętych kolumn | kod kroku 4 pracuje na kopii `data`, `df` (dane po kroku 2) zostaje bez zmian; eksport z klastrami bierze wszystkie kolumny `df` dla wierszy z `data` – jak aplikacja; test porównuje wszystkie 4 eksporty (międzynarodowo i dla Excela) bajt po bajcie |
| B2 | średnia | Domyślna grupa porównawcza w kroku 4: gdy żadna kolumna nie nazywa się jak etykieta, aplikacja brała ostatnią kolumnę (jeśli miała 2–30 wartości) | np. w danych kart kredytowych (CC GENERAL) TENURE (7 wartości) stawała się bez pytania grupą porównawczą i znikała z cech – klastry liczone bez niej; po odznaczeniu grupy kolumna nie wracała do cech | propozycja tylko dla kolumn nazwanych jak podział (Private, species, label, class, segment, group, type, y …); kolumna, która przestaje być grupą lub nazwą wiersza, wraca do cech, jeśli należy do propozycji |
| B3 | średnia | Kolumna z identyfikatorem (np. CustomerID = 1, 2, 3 … w Mall Customers) była domyślnie cechą | K-Means grupował częściowo według numeru wiersza – po StandardScaler identyfikator waży tyle co każda inna cecha | nowa funkcja `looks_like_id` (liczby całkowite, każda wartość inna, nazwa kończy się na ID/Nr/Number/Index albo wartości 1, 2, 3 … / 0, 1, 2 …): taka kolumna nie jest proponowana jako cecha, tylko jako nazwa wiersza (gdy nie ma kolumny tekstowej z nazwami); wybrana jako cecha → ostrzeżenie |
| C1 | niska | Tekst „Jak w kursie” w kroku 8 zawsze kończył się „…że klaster 1 ma oznaczać 'Yes'” | dla innych danych (np. Iris: setosa/versicolor/virginica) zdanie było błędne; gdy kilka klastrów trafia do tej samej grupy albo numery już pasują, wyjaśnienie „różnica wynika tylko z numerów” było nieprawdziwe | nazwa prawdziwej grupy o numerze 1; osobne zdania dla przypisania wiele-do-jednego i dla numerów, które już pasują |
| C2 | niska (bezpieczeństwo) | Nazwy klastrów, nazwy kolumn i wartości grup z wczytanego modelu (ZIP) trafiały do Markdown: podsumowanie kroku 10, nagłówek porównania, etykiety pól formularza (Streamlit pokazuje w etykietach także obrazki) i komunikaty błędów | np. `![](http://…)` jako nazwa klastra mogło wczytać obcy obrazek (śledzenie otwarcia) – ta sama klasa problemu co C3 w audycie 4 szkieletu, ale dla nowych pól | `pl.md_escape` (wspólne z `md_text`) dla wszystkich tekstów z pliku w krokach 8–10 i dla komunikatów błędów przy wczytywaniu i przypisywaniu; escapowany jest też `$` (formuły) |
| C3 | niska | `preprocessing.json` był sprawdzany tylko pod kątem `task` | uszkodzony lub ręcznie zmieniony plik wczytywał się, a potem: krok 9 kończył się wyjątkiem przy każdym wyświetleniu (brak `features`), krok 10 błędem (tekst w `train_info`, zły `group_mapping`) albo **liczył po cichu źle** (nieznany scaler działał jak „brak”) | `recipe_problem` / `check_recipe`: typy i długości wszystkich używanych wpisów (cechy, kategorie, kolumny, parametry scalera, zakresy, wartości uzupełniające, mapowanie, nazwy, ustawienia) oraz zgodność z modelem (liczba wejść i klastrów) – przy wczytaniu jasny komunikat |
| C4 | niska | Uwaga „Bez zmienności” sprawdzała, czy odchylenie standardowe = 0 | kolumna z samymi 0.1 ma przez zaokrąglenia odchylenie 2.8e-17 – uwaga się nie pojawiała | rozstęp (max − min) = 0 – sprawdzenie dokładne |
| C5 | niska | Pusty klaster (powtórzone wiersze, mniej różnych wierszy niż K) opisany jako „Bardzo mały klaster … tylko 0 wierszy – często wartości odstające”, a w profilach „blisko średniej” | myląca diagnoza | osobna uwaga „Pusty klaster … – zmniejsz K” w kroku 7 i „pusty” w opisie profili w kroku 8 |
| C6 | niska | Uwaga „ustawienia zmieniły się od przygotowania” w kroku 4 brała pod uwagę tylko cechy, grupę i nazwę | po zmianie skalowania, drop_first albo strategii braków strona dalej pokazywała stare dane jako aktualne – trening w kroku 6 używał starego przygotowania | porównanie także skalowania, drop_first i strategii braków |

## Co sprawdzono i działa poprawnie

* **Dane brzegowe:** 10 wierszy i K = 9 (krzywa K = 1 … 9, uwagi, profile, PCA, przypisanie); same powtórzone
  wiersze (pusty klaster, uwaga scikit-learn, bez awarii); wszystkie wiersze jednakowe (Silhouette „–”, R² „–”, bez
  awarii); tylko kategorie z MinMaxScaler (siła rozdzielania liczona razem dla kolumn dummy jednej kategorii); grupa
  0/1 jako liczba, w nowych danych jako liczba zmiennoprzecinkowa z lukami i nieznaną wartością (przypisanie
  z treningu zostaje, nieznana grupa liczy się jako błąd); cecha bool, w nowych danych tekst „True”/„False”; nazwa
  wiersza z lukami i liczbowe ID jako nazwa; znaki specjalne i podział wiersza w nazwach klastrów po zapisie
  i odczycie (`.skops` i ZIP).
* **Duże dane:** 25 000 wierszy – krzywa na stałej próbce 20 000 wierszy, Silhouette na próbce 5000 wierszy (średnia
  na wykresie Silhouette = wartość w kroku 7), razem 6,6 s.
* **Eksportowany skrypt** (krok 9) z usuwaniem wierszy i kolumn w kroku 4: te same klastry co aplikacja.
* **Pełne przejścia kliknięciami** (AppTest) przez kroki 4–10 dla zbiorów Mall, Iris, duplikaty, minimalny i stały:
  przygotowanie, trening, wybór K, ocena z pojedynczym przypadkiem, 4 eksporty, zapis, przypisanie pliku i formularz
  – bez błędów.
* **Bezpieczeństwo:** 7 przygotowanych plików ZIP z uszkodzonym `preprocessing.json` – wszystkie odrzucone przy
  wczytaniu z jasnym komunikatem; żaden tekst z wczytanego modelu nie jest już interpretowany jako Markdown.
* **Teksty:** katalog tłumaczeń kompletny (838 tekstów), nowe teksty po angielsku i po polsku sprawdzone.
* **Test w przeglądarce** (prawdziwy serwer, Playwright): cały przebieg z wersji 1 (kroki 1–10 po niemiecku, kroki
  4–10 po angielsku i po polsku) oraz nowe kontrole – tekst „Jak w kursie” z grupą 'Yes' dla College, dane jak Mall
  Customers z Kaggle (bez grupy porównawczej, CustomerID jako nazwa wiersza, nie jako cecha, przygotowanie 200 × 5) –
  **46/46**.

## Uwagi (bez zmian)

* Prognoza naiwna w kroku 10 liczy najczęstszą grupę w nowych danych (zwykła miara odniesienia); gdy nowe dane
  zawierają tylko jedną grupę, wynosi 100 % i porównanie niewiele mówi.
* Przy remisie w przypisaniu większościowym aplikacja bierze pierwszą grupę w porządku naturalnym („2” przed „10”),
  pokazywany kod (`idxmax` na tabeli krzyżowej) pierwszą alfabetycznie – różnica tylko przy dokładnie równych liczbach.
* Nazwy kolumn z własnego pliku CSV w krokach 2–4 nie są escapowane (własne dane użytkownika – jak w aplikacji
  bazowej); escapowane są teksty z wczytanych modeli i komunikaty błędów.
* Wynik kroku 10 do pobrania to CSV międzynarodowy (przecinek), bez ochrony dla Excela – jak w aplikacji bazowej.
* Polskie formy liczby mnogiej są uproszczone („wierszy”, „klastrów”) – jak w aplikacji bazowej.

## Testy (stan po audycie 5)

268 testów (247 + 21 w `tests/test_audit5_regressions.py`) – wszystkie przechodzą (Python 3.13, pandas 3.0,
scikit-learn 1.9, skops 0.16, Streamlit 1.66); składnia sprawdzona dla Pythona 3.10 i 3.11. Wszystkie 21 nowych
testów uruchomione na kodzie sprzed audytu kończą się niepowodzeniem (sprawdzone) – 19 odtwarza błędy B1–B3
i C1–C6, 2 sprawdzają nowe funkcje pomocnicze (`rows_var`, `looks_like_id`). W `tests/test_new_features.py` jeden test
uwzględnia teraz escapowane nazwy kolumn w podsumowaniu. Uruchomienie: `pytest -q` (ok. 3 minuty).

---

# Wersja 2 – wykrywanie anomalii (Isolation Forest, LOF, DBSCAN) i skrypt Bybit (28.10.2026)

**Zakres:** w kroku 6 obok K-Means trzy nowe metody – **Isolation Forest**, **Local Outlier Factor** (zawsze
`novelty=True`) i **DBSCAN** (szum = anomalia); kroki 5, 7, 8, 9 i 10 dopasowują się do wybranego modelu; generowany
kod (`codegen.py`), wykresy (`plots.py`), ok. 230 nowych tekstów EN/PL; skrypt `tools/bybit_features.py` (publiczne
dane rynkowe Bybit → CSV ze wskaźnikami) z dwiema przykładowymi konfiguracjami (szereg czasowy, przekrój); karta
aplikacji w launcherze; README w trzech językach.

## Decyzje metodyczne

| Temat | Decyzja |
|---|---|
| Wskaźnik | jedna skala dla wszystkich metod – wyższy = bardziej nietypowy: IF `−score_samples` (0 … 1), LOF wartość LOF (≈ 1 normalny), DBSCAN odległość do najbliższego punktu rdzeniowego / eps (powyżej 1 = szum) |
| Próg | `contamination = auto` (IF 0.5, LOF 1.5) albo stały udział; DBSCAN 1 (= eps). Próg i udział nietypowych wierszy są zapisane w `preprocessing.json` i w kroku 10 używane bez zmian |
| LOF | `novelty=True`: wskaźnik wierszy treningowych z `negative_outlier_factor_` (każdy wiersz bez siebie jako sąsiada), nowe dane przez `score_samples` / `predict` |
| DBSCAN, nowe dane | klaster najbliższego punktu rdzeniowego, jeśli leży najwyżej o eps – inaczej −1 (jak definicja punktu brzegowego) |
| Krok 7 | IF/LOF: próg dla udziałów 0.1–10 % i załamanie (uproszczony Kneedle) krzywej 20 % najbardziej nietypowych wierszy → propozycja contamination. DBSCAN: k-odległość (k = min_samples, wiersz liczy się sam) z załamaniem → propozycja eps; udział szumu i liczba klastrów dla kwantyli k-odległości (do 20 000 wierszy) |
| Krok 8 | najbardziej nietypowe wiersze z odchyleniami z (wgląd w dane, nie w model), profil nietypowe/normalne, PCA z anomaliami zawsze na wykresie, porównanie ze znanym podziałem (precision, recall, ROC-AUC, average precision; rzadsza grupa = pozytywna) |
| Skrypt Bybit | bez zaglądania w przyszłość: wartości z względem *poprzednich* świec, okna kończą się na świecy, funding i open interest znane na zamknięciu świecy, trwająca świeca pominięta, rozbieg pobierany przed startem. Stronicowanie wstecz tylko z `end` / `endTime` (niezależne od kolejności sortowania API), open interest przez kursor; limit 600 zapytań / 5 s, ponawianie przy 10006 i 5xx, przy 403 koniec z jasnym komunikatem |

## Znalezione w trakcie budowy (poprawione)

| # | Problem | Poprawka |
|---|---|---|
| N1 | LOF nie wykrywał wartości odstających, które same tworzą małą grupę (24 podobne „oszustwa” przy `n_neighbors = 20`) – ich sąsiadami były one same, LOF ≈ 1 | opis w pomocy `n_neighbors`, w tabeli metod i w README; test z `n_neighbors = 60` |
| N2 | Oś logarytmiczna w kroku 7 pokazywała `$\mathdefault{10^{2}}$` – aplikacja wyłącza mathtext (`text.parse_math`, ochrona przed `$` w nazwach kolumn) | własny formatter osi (1, 10, 100) + test |
| N3 | Legenda zasłaniała opis progu na histogramie wskaźnika | próg jako wpis w legendzie |
| N4 | k-odległość: kilka skrajnych wartości (np. krach) spłaszczało krzywą, załamanie było nieczytelne | oś y obcięta (max z 1.3 × kwantyl 99.5 %, 1.4 × eps, 1.6 × załamanie) z dopiskiem najwyższej wartości; oś od 0 |
| N5 | Udziały na wykresie DBSCAN zaokrąglane do 0 % / 100 % (przy 0.46 % szumu) | `share_text` (0.46 %, 99.5 %) |
| N6 | `pca_projection` bez `labels` dla IF/LOF (modele bez `labels_`) kończyła się wyjątkiem | domyślnie same 0 |
| N7 | `recipe_problem` przepuszczał próg NaN/∞ i udział spoza 0 … 1 w ręcznie zmienionym `preprocessing.json` | kontrola skończoności i zakresu |
| N8 | Wartości uzupełniające w kroku 10 z szumem zaokrągleń (0.17886999999999997) | `format_value` z 12 cyframi znaczącymi |

## Co sprawdzono

* **Kod = aplikacja** dla IF, LOF i DBSCAN (wskaźnik, anomalie, próg, progi dla udziałów, załamanie, k-odległość,
  siatka eps, profile, odchylenia z, porównanie, eksport, krok 10 bez aplikacji, cały skrypt z kroku 9) – także
  z kategoriami, lukami, MinMaxScaler, próbką i metryką manhattan.
* **Skrypt Bybit** na symulowanym API (prawdziwe `api.bybit.com` nie było osiągalne ze środowiska testowego):
  limity stron, ponawianie, 403, cache/offline, brak zaglądania w przyszłość (obcięte dane → wcześniejsze wiersze co do
  bitu takie same), wskaźniki przeliczone niezależnie, wbudowany krach jako najbardziej nietypowy wiersz (IF, 1 %) –
  także po `--split` w kroku 10.
* **Składnia** zmienionych plików i generowanego kodu sprawdzona dla Pythona 3.10 i 3.11.
* **Test w przeglądarce** (Playwright, prawdziwy serwer, po polsku i angielsku): dane Bybit → przykładowa
  konfiguracja → Isolation Forest → kroki 7, 8, 9 i 10 – bez błędów, wykresy czytelne; krach na pierwszym miejscu
  w kroku 10.

## Testy (stan wersji 2)

349 testów (268 + 81 nowych: 33 w `tests/test_anomaly.py`, 5 w `tests/test_app_anomaly.py`, 39 w
`tests/test_bybit_features.py`, 4 w `tests/test_i18n.py`) – wszystkie przechodzą (Python 3.13, pandas 3.1,
scikit-learn 1.9, skops 0.16, Streamlit 1.66). Uruchomienie: `pytest -q` (ok. 5 minut).

---

# Audyt 6 – wykrywanie anomalii i skrypt Bybit (29.10.2026)

**Zakres:** cały kod wersji 2: część anomalii w `pipeline.py` (parametry, wskaźnik, próg, krzywe i ocena w kroku 7,
profile, porównanie ze znanym podziałem, zapis i odczyt, ocena nowych danych, podsumowanie), strony 6–10 w `app.py`
dla Isolation Forest, LOF i DBSCAN, generowany kod (`codegen.py`), nowe wykresy (`plots.py`), tłumaczenia, skrypt
`tools/bybit_features.py`, przykładowe konfiguracje i testy.
**Metoda:** przegląd każdej nowej funkcji; sondy z nietypowymi danymi (12 wierszy, wszystkie wiersze jednakowe,
jedna cecha, połowa duplikatów, same kategorie, 3000–20 000 wierszy z dużym eps, nowe dane z odwróconymi proporcjami
grup, zmiana scalera po treningu); pomiar pamięci DBSCAN, LOF i Isolation Forest w osobnych procesach; przejścia
kliknięciami (AppTest); skrypt Bybit na symulowanym API z zaostrzonym testem „bez zaglądania w przyszłość”;
dokumentacja API Bybit (znaczenie znacznika czasu open interest). Każdy potwierdzony błąd jest poprawiony i ma test
regresyjny (`tests/test_audit6_regressions.py`).

**Wynik:** 9 naprawionych problemów – 4 średnie, 5 drobnych. Poprawki B1–B3 i C1–C5 nie zmieniają wyników przy
ustawieniach z przykładowych konfiguracji Bybit (Isolation Forest, 1 %, świece 1h); B4 zmienia w CSV ze skryptu
kolumny `oi_chg_pct` i `oi_z` (okno open interest kończy się 15 min przed zamknięciem świecy 1h).

| # | Waga | Problem | Skutek | Poprawka |
|---|---|---|---|---|
| B1 | średnia | Krok 10: przy porównaniu ze znanym podziałem „pozytywna” była grupa rzadsza **w nowych danych** | próbka z wieloma oszustwami (15 oszustw, 5 normalnych) była oceniana względem grupy „normalna”: precision 0, recall 0, ROC-AUC 0.0 – choć model trafił wszystkie oszustwa | grupa pozytywna z treningu (`anomaly_metrics.positive` w `preprocessing.json`); rzadsza grupa tylko, gdy model zapisano bez porównania; w kroku 10 podpis „Pozytywna = '1' – jak w treningu” |
| B2 | średnia | DBSCAN (scikit-learn) trzyma wszystkie sąsiedztwa jednocześnie w pamięci; LOF – wiersze × `n_neighbors` odległości | przy dużym eps albo wielu identycznych wierszach pamięć rośnie z kwadratem liczby wierszy (pomiar: 15–35 B na parę; 8000 wierszy i eps obejmujące wszystko → 0,8–1,3 GB, 20 000 wierszy → ok. 6–14 GB) – awaria aplikacji bez komunikatu; także w tabeli eps kroku 7 (kwantyl 99 % k-odległości) | `neighbor_pairs` szacuje liczbę par na 400 losowych wierszach (dokładność ok. ±25 %, poniżej √40 mln wierszy bez liczenia); powyżej 40 mln par trening odmawia z wyjaśnieniem i podpowiedzią (mniejsze eps, próbka), tabela eps pomija za duże eps – a pokazywany kod i skrypt z kroku 9 pomijają te same; LOF powyżej 40 mln odległości tak samo |
| B3 | średnia | Wartość startowa eps (krok 6) liczona tylko raz | po nowym przygotowaniu z innym scalerem eps zostawało: StandardScaler → eps 1.049 (3 klastry, 17 wierszy szumu), po zmianie na MinMaxScaler nadal 1.049 zamiast 0.1649 → **0 wierszy szumu**, bez ostrzeżenia | wartość startowa liczona od nowa po nowym przygotowaniu, zmianie `min_samples` lub `metric`, o ile eps pochodzi z danych (wartość startowa albo propozycja przejęta w kroku 7); wartość wpisana ręcznie zostaje |
| B4 | średnia | Skrypt Bybit: open interest ze znacznikiem czasu równym zamknięciu świecy był używany dla tej świecy | Bybit nie dokumentuje, czy znacznik oznacza początek interwału, czy moment pomiaru; jeśli początek, wartość jest znana dopiero po interwale → zaglądanie w przyszłość w `oi_chg_pct` / `oi_z` (ważne przy `--split`, jak na żywo) | ostrożnie: wartość znana dopiero jeden interwał po znaczniku (`oi_step_ms`); w trybie series drobniejsza siatka OI (najwyżej ¼ świecy, np. 15 min dla świec 1h), więc zmiana nadal obejmuje prawie całą świecę; zaostrzony test (obcięcie OI o trwający interwał → wcześniejsze wiersze bez zmian) |
| C1 | niska | Same kategorie → wiele identycznych wierszy, k-odległość 0 | krok 7 proponował „eps = 0 przejmij” (eps musi być > 0; pole w kroku 6 po cichu ustawiało 1e-6) | propozycja co najmniej 1e-6 (także w pokazywanym kodzie) i uwaga „Wiele identycznych wierszy”: przy najmniejszym eps klastry tworzą identyczne wiersze, rzadkie kombinacje są szumem |
| C2 | niska | Wszystkie wiersze jednakowe po przygotowaniu | Isolation Forest zgłaszał **wszystkie** wiersze jako nietypowe (wskaźnik = próg 0.5, zaokrąglenie), LOF i DBSCAN żadnego | trening odmawia z jasnym komunikatem (wybierz w kroku 4 cechy, które się różnią) |
| C3 | niska | Poniżej 100 wierszy załamanie krzywej wskaźnika szukane w więcej niż 20 % wierszy (co najmniej 20 punktów) | przy 12 wierszach krok 7 proponował contamination = 50 % (IF i LOF) | załamanie liczy się tylko wśród 20 % najbardziej nietypowych wierszy – w aplikacji i w pokazywanym kodzie |
| C4 | niska | Wykres k-odległości: gdy eps leżało ponad wszystkimi k-odległościami, oś y kończyła się poniżej eps | brak linii i podpisu „ustawione: eps” – akurat w sytuacji z B3 (eps za duże), przy samych zerach oś do 0.055; ostrzeżenie `tight_layout` | oś y zawsze obejmuje eps |
| C5 | niska | Skrypt Bybit, świece 1- i 3-minutowe: open interest istnieje tylko co 5 minut | `oi_chg_pct` = 0 w 4 z 5 świec i skok w co piątej – sztuczny wzorzec, który model może uznać za nietypowy | open interest pomijany przy świecach krótszych niż 5 minut, z informacją w konsoli |

## Co sprawdzono i działa poprawnie

* **Dane brzegowe:** 12 wierszy, jedna cecha, połowa duplikatów – trening, kroki 7 i 8 bez błędów; LOF bez żadnej
  anomalii; DBSCAN z samym szumem (bez punktów rdzeniowych) przez kroki 7, 8, 5, 9 i 10; Isolation Forest nie zgłasza
  powtarzających się wierszy, gdy stanowią 30–70 % danych.
* **Zapis i odczyt:** ZIP bez pickle (`.skops`) dla wszystkich trzech modeli, zgodność modelu z rezeptem.
* **Pamięć Isolation Forest:** ok. 0,1 węzła na wiersz próbki (drzewa mają ograniczoną głębokość) – 100 drzew
  × 50 000 wierszy → 42 MB; ochrona nie jest potrzebna.
* **Kod = aplikacja** po poprawkach (wskaźnik, próg, załamanie, propozycja eps, tabela eps z pominiętymi wartościami).
* **Skrypt Bybit:** stronicowanie, ponawianie, cache, przekrój, spot, nieznane symbole i przykładowe konfiguracje –
  bez zmian; krach w symulacji nadal na pierwszym miejscu (także po `--split`), spadek open interest nadal w świecy
  krachu (−15 %).

## Uwagi (bez zmian)

* LOF z `novelty=True`: wskaźnik wierszy treningowych pochodzi z `negative_outlier_factor_` (wiersz bez siebie jako
  sąsiada), nowych danych z `score_samples` – ten sam wiersz oceniony ponownie w kroku 10 może mieć nieco inny
  wskaźnik (tak działa scikit-learn).
* Pole „krzywa dla kroku 7” w kroku 6 jest wspólne dla K-Means (łokieć i Silhouette) i DBSCAN (tabela eps).
* Skrypt Bybit nie uzupełnia brakujących świec (godziny bez transakcji): wartości z porównują z poprzednimi N
  wierszami, nie N jednostkami czasu – przy płynnych kontraktach perpetual rzadkie.
* Szacunek par sąsiadów pochodzi z próbki 400 wierszy (±25 %); limit 40 mln par (ok. 1 GB) ma zapas.
* Prawdziwe `api.bybit.com` nie było osiągalne ze środowiska testowego – znaczenia znacznika czasu open interest nie
  dało się sprawdzić na żywych danych, stąd ostrożne założenie w B4.

## Testy (stan po audycie 6)

362 testy (349 + 13 w `tests/test_audit6_regressions.py`) – wszystkie przechodzą (Python 3.13, pandas 3.1,
scikit-learn 1.9, skops 0.16, Streamlit 1.66); składnia sprawdzona dla Pythona 3.10 i 3.11. Wszystkie 13 nowych
testów uruchomione na kodzie sprzed audytu kończą się niepowodzeniem (sprawdzone) – 10 odtwarza błędy B1, B3, B4
i C1–C5, 3 sprawdzają nową ochronę pamięci z B2 (samej awarii z braku pamięci nie da się bezpiecznie odtworzyć
w teście – zmierzono ją osobno). Symulowane API Bybit (`tests/synthetic.py`) zwraca teraz open interest w żądanym
interwale (dane co 5 minut, spadek w trakcie świecy krachu); `tests/test_bybit_features.py` sprawdza dodatkowo
drobną siatkę OI. Uruchomienie: `pytest -q` (ok. 5 minut).
