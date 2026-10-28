# 🤖 ML Hub – uczenie maszynowe w Pythonie w czterech aplikacjach Streamlit

[![Tests](https://github.com/danielkotok79-bit/Machine-Learning-HUB/actions/workflows/tests.yml/badge.svg)](https://github.com/danielkotok79-bit/Machine-Learning-HUB/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)

**ML Hub** to lokalny launcher dla czterech aplikacji Streamlit, które prowadzą plik CSV przez cały proces uczenia
maszynowego: wczytanie i czyszczenie danych, eksplorację, trening modelu, ewaluację, zapis modelu i jego użycie na
nowych danych. Na każdym kroku aplikacje pokazują też odpowiedni kod w Pythonie, więc służą również do nauki.

🌍 Launcher i wszystkie aplikacje są dostępne po **angielsku** (domyślnie), **polsku** i **niemiecku**.  
📖 [README in English](README.md) · [README auf Deutsch](README_DE.md)

![ML Hub – launcher](docs/screenshots/ml-hub.png)

## Aplikacje

| Aplikacja | Rodzaj uczenia | Modele | Przykładowe dane |
|---|---|---|---|
| 📊 [**Uczenie nadzorowane**](apps/Supervised) | nadzorowane – klasyfikacja i regresja | Gradient Boosting, drzewo decyzyjne + Random Forest, SVM, regresja logistyczna, KNN, regresja liniowa; podział losowy albo czasowy, porównanie z Elo | LendingClub, NBA/tenis |
| 🔵 [**Klastrowanie**](apps/Clustering) | nienadzorowane | K-Means (metoda łokcia, silhouette, profile klastrów, mapa PCA) · wykrywanie anomalii: Isolation Forest, LOF, DBSCAN | dane uczelni (College) · dane rynku krypto z Bybit (skrypt w zestawie) |
| 📝 [**NLP – klasyfikacja tekstu**](apps/NLP) | nadzorowane, tekst | Bag of Words / TF-IDF + Naive Bayes, regresja logistyczna, liniowy SVM | recenzje Yelp, SMS spam |
| 🧠 [**Deep Learning**](apps/Deeplearning) | nadzorowane – klasyfikacja i regresja | Keras Sequential (Dense) na TensorFlow, EarlyStopping | LendingClub |

Wszystkie cztery aplikacje mają te same dziesięć kroków:

1. **Wczytanie danych** – upload CSV albo ścieżka do pliku; separator i przecinek dziesiętny są rozpoznawane automatycznie
2. **Przygotowanie kolumn** – jedna akcja na kolumnę (zostaw, usuń, wyciągnij liczbę, data → rok/miesiąc, przypisz wartości …) i nowe kolumny z formuł
3. **Eksploracja** – wykres liczności, histogram, wykres pudełkowy, wykres punktowy, mapa korelacji, pair plot, statystyki grup …
4. **Dane do ML** – etykieta i cechy; **najpierw** `train_test_split`, potem braki danych, kolumny dummy i skalowanie uczone tylko na danych treningowych (bez przecieku danych)
5. **Zapis danych** – przygotowane dane jako CSV (format międzynarodowy albo dla Excela)
6. **Model i trening** – wybór modelu z „Dlaczego?” i „Najlepszy do”, najważniejsze parametry
7. **Trening vs. walidacja** – krzywa walidacji albo krzywe straty, diagnoza przeuczenia i niedouczenia
8. **Ewaluacja** – classification report, macierz pomyłek, ROC-AUC, próg decyzyjny, ważność cech · regresja: MAE, RMSE, R²
9. **Zapis i wczytanie modelu** – `.skops` (bezpieczny) albo `.joblib` + `preprocessing.json`, pobranie ZIP, cały pipeline jako skrypt Python
10. **Predykcja dla nowych danych** – to samo przygotowanie stosowane automatycznie, potem predykcja, wyjaśnienie i wniosek

Klastrowanie i NLP dopasowują kroki 4–10 do swojego zadania (np. wybór K i profile klastrów, próg i najbardziej
nietypowe wiersze przy anomaliach albo najważniejsze słowa dla każdej klasy). Szczegóły są w README każdej aplikacji (DE · EN · PL).

![Uczenie nadzorowane – krok 8: ewaluacja modelu](docs/screenshots/supervised-evaluation.png)
*Uczenie nadzorowane, krok 8: ewaluacja modelu Random Forest na syntetycznych danych podobnych do LendingClub (zob. [Wypróbuj bez własnych danych](#wypróbuj-bez-własnych-danych)).*

## Szybki start

Wymagania: **Python 3.10 lub nowszy** (aplikacja Deep Learning potrzebuje Pythona 3.10–3.13 ze względu na TensorFlow).

```bash
git clone https://github.com/danielkotok79-bit/Machine-Learning-HUB.git
cd Machine-Learning-HUB
python -m venv .venv
source .venv/bin/activate                         # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # launcher + Supervised, Clustering, NLP
pip install -r apps/Deeplearning/requirements.txt # tylko dla aplikacji Deep Learning (TensorFlow)
streamlit run app.py                              # otwiera http://localhost:8501
```

W launcherze kliknij **Uruchom** przy wybranej aplikacji, a potem **Otwórz**. Aplikacje działają w tle na portach
**8502–8505** (także po zamknięciu launchera) i startują w języku launchera; **Zatrzymaj** je kończy. PID-y i logi są
w folderze `.mlhub/`. Jeśli brakuje jakiegoś pakietu, launcher pokazuje dokładne polecenie `pip install`.

Każdą aplikację można też uruchomić osobno:

```bash
cd apps/Supervised
streamlit run app.py -- --lang pl                 # albo en, de · otwiera http://localhost:8501
```

> **Starszy Streamlit (np. 1.51 z Anacondy)?** Aplikacje używają `st.segmented_control(required=…)`, który istnieje
> dopiero od Streamlit 1.56. Launcher uruchamia aplikacje przez `mlhub_compat.py`, który na starszych wersjach
> dodaje ten parametr, więc z launchera wszystko działa. Bez launchera uruchom w folderze aplikacji
> `python ../../mlhub_compat.py run app.py` zamiast `streamlit run app.py` albo zaktualizuj Streamlit:
> `pip install --upgrade "streamlit>=1.56"`.

## Wypróbuj bez własnych danych

Zbiory danych z kursu nie są częścią repozytorium. Każda aplikacja potrafi wygenerować **syntetyczny** plik z tymi
samymi kolumnami i formatami co oryginalny zbiór, więc wszystko można wypróbować od razu:

| Aplikacja | Polecenie (w folderze aplikacji) | Tworzy |
|---|---|---|
| Supervised | `python tests/synthetic.py` | `demo_lending_club_synthetisch.csv` (20 000 wierszy) |
| Clustering | `python tests/synthetic.py` · `python tools/bybit_features.py --top 10 --days 60` | `demo_college_synthetisch.csv` (777 wierszy) · prawdziwe wskaźniki krypto z publicznych danych Bybit (potrzebny internet) |
| NLP | `python tests/synthetic.py` · `python tests/synthetic.py --sms` | `demo_yelp_synthetisch.csv` (10 000 recenzji) · `demo_sms_synthetisch.txt` |
| Deep Learning | `python tests/synthetic.py` | `demo_lending_club_synthetisch.csv` (20 000 wierszy) |

Wczytaj plik w kroku 1. Dla tych zbiorów aplikacje oferują gotową przykładową konfigurację (`configs/`), która
ustawia przygotowanie danych i model jak w projektach z kursu.

## Materiały

Oprócz aplikacji launcher zawiera krótkie materiały do czytania i trzy dokumenty do pobrania – każdy po angielsku,
po polsku i po niemiecku. Launcher proponuje wersję w wybranym języku, pozostałe są w sekcji *Inne języki*. Wszystkie
pliki są w folderze [`materials/`](materials):

| Dokument | Treść | EN | PL | DE |
|---|---|---|---|---|
| FAQ | małe zbiory danych i sieci neuronowe – kiedy sieć ma sens, ustawienia early stopping | [PDF](materials/FAQ_EN.pdf) | [PDF](materials/FAQ_PL.pdf) | [PDF](materials/FAQ_DE.pdf) |
| ML w predykcjach | uczenie maszynowe w predykcjach finansowych, giełdowych i politycznych | [PDF](materials/ML_in_predictions_EN.pdf) | [PDF](materials/ML_in_predictions_PL.pdf) | [PDF](materials/ML_in_predictions_DE.pdf) |
| Interpretacja wyników | jak czytać wyniki 8 notebooków ML z kursu (dane, trening, wyniki na danych testowych) | [PDF](materials/Reading_ML_results_EN.pdf) | [PDF (PL + DE)](materials/Reading_ML_results_PL-DE.pdf) | [PDF (PL + DE)](materials/Reading_ML_results_PL-DE.pdf) |

Wersja polska i niemiecka trzeciego dokumentu to jeden dwujęzyczny plik (najpierw część polska, potem niemiecka).

## Testy

**1037 automatycznych testów** (pytest, w tym testy całego interfejsu przez Streamlit `AppTest`) – uruchamiane
osobno dla każdej aplikacji:

```bash
pip install pytest
cd apps/Supervised
pytest -q
```

| Aplikacja | Testy |
|---|---|
| Supervised | 310 |
| Clustering | 349 |
| NLP | 188 |
| Deep Learning | 190 |

W folderze każdej aplikacji jest też `AUDIT.md` (po polsku) z wynikami przeglądów kodu: znalezione błędy, sposób
ich poprawienia i testy regresyjne, które je pokrywają.

## Struktura projektu

```
.
├── app.py               launcher (ML Hub)
├── mlhub_compat.py      poprawka zgodności dla Streamlit < 1.56
├── requirements.txt
├── materials/           dokumenty PDF (EN · PL · DE) dostępne w launcherze
├── docs/screenshots/
└── apps/
    ├── Supervised/      📊 scikit-learn – sześć modeli uczenia nadzorowanego
    ├── Clustering/      🔵 K-Means · Isolation Forest · LOF · DBSCAN
    ├── NLP/             📝 klasyfikacja tekstu
    └── Deeplearning/    🧠 Keras / TensorFlow
```

Każda aplikacja ma ten sam układ:

```
app.py              interfejs Streamlit
pipeline.py         logika obliczeń – do użycia także w notebooku (import pipeline as pl)
plots.py            wykresy (matplotlib / seaborn)
codegen.py          generuje kod Python pokazywany w aplikacji
i18n.py, locales/   języki – teksty w kodzie są po niemiecku, locales/ zawiera tłumaczenia angielskie i polskie
configs/            przykładowe konfiguracje
tests/              testy pytest + syntetyczne dane testowe
README.md           niemiecki · README_EN.md angielski · README_PL.md polski
AUDIT.md            raport z przeglądu kodu (po polsku)
```

Komentarze w kodzie źródłowym są po niemiecku – w oryginalnym języku kodu.

## Technologie

Python · Streamlit · pandas · NumPy · scikit-learn · SciPy · skops · matplotlib · seaborn · TensorFlow / Keras · pytest
