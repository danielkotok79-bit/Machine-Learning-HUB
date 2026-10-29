# 🤖 ML Hub – machine learning with Python in four Streamlit apps

[![Tests](https://github.com/danielkotok79-bit/Machine-Learning-HUB/actions/workflows/tests.yml/badge.svg)](https://github.com/danielkotok79-bit/Machine-Learning-HUB/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikitlearn&logoColor=white)

**ML Hub** is a local launcher for four Streamlit apps that take a CSV file through a complete machine-learning
workflow: loading and cleaning the data, exploring it, training a model, evaluating it, saving it and using it on
new data. Every step also shows the matching Python code, so the apps work as a learning tool as well.

🌍 The launcher and all apps are available in **English** (default), **Polish** and **German**.  
📖 [README po polsku](README_PL.md) · [README auf Deutsch](README_DE.md)

![ML Hub – launcher](docs/screenshots/ml-hub.png)

## The apps

| App | Type of learning | Models | Example data |
|---|---|---|---|
| 📊 [**Supervised Learning**](apps/Supervised) | supervised – classification and regression | Gradient Boosting, Decision Tree + Random Forest, SVM, Logistic Regression, KNN, Linear Regression; random or time-based split, comparison with Elo | LendingClub, NBA/tennis |
| 🔵 [**Clustering**](apps/Clustering) | unsupervised | K-Means (elbow, silhouette, cluster profiles, PCA map) · anomaly detection: Isolation Forest, LOF, DBSCAN, autoencoder (neural network) | College data · crypto market data from Bybit (script included) |
| 📝 [**NLP – text classification**](apps/NLP) | supervised, text | Bag of Words / TF-IDF + Naive Bayes, Logistic Regression, Linear SVM | Yelp reviews, SMS spam |
| 🧠 [**Deep Learning**](apps/Deeplearning) | supervised – classification and regression | Keras Sequential (Dense) on TensorFlow, EarlyStopping | LendingClub |

All four apps follow the same ten steps:

1. **Load data** – CSV upload or file path; separator and decimal comma are detected automatically
2. **Prepare columns** – one action per column (keep, drop, extract number, date → year/month, map values …) and new columns from formulas
3. **Explore** – count plot, histogram, box plot, scatter plot, correlation heatmap, pair plot, group statistics …
4. **ML data** – label and features; `train_test_split` **first**, then missing values, dummies and scaling learned from the training data only (no data leakage)
5. **Save data** – the prepared data as CSV (international or Excel format)
6. **Model & training** – choice of model with "Why?" and "Best for", the most important parameters
7. **Training vs. validation** – validation curve or loss curves, diagnosis of over- and underfitting
8. **Evaluation** – classification report, confusion matrix, ROC-AUC, decision threshold, feature importance · regression: MAE, RMSE, R²
9. **Save & load the model** – `.skops` (safe) or `.joblib` + `preprocessing.json`, ZIP download, the whole pipeline as a Python script
10. **Predict new data** – the same preparation is applied automatically, then prediction, explanation and a conclusion

Clustering and NLP adapt steps 4–10 to their task (e.g. choosing K and cluster profiles, the anomaly threshold and the
most unusual rows, or the most important words per class). Details are in the README of each app (DE · EN · PL).

![Supervised Learning – step 8: model evaluation](docs/screenshots/supervised-evaluation.png)
*Supervised Learning, step 8: evaluation of a Random Forest on synthetic LendingClub-like data (see [Try it without your own data](#try-it-without-your-own-data)).*

## Quick start

Requirements: **Python 3.10 or newer** (the Deep Learning app needs Python 3.10–3.13 because of TensorFlow).

```bash
git clone https://github.com/danielkotok79-bit/Machine-Learning-HUB.git
cd Machine-Learning-HUB
python -m venv .venv
source .venv/bin/activate                         # Windows: .venv\Scripts\activate
pip install -r requirements.txt                   # launcher + Supervised, Clustering, NLP
pip install -r apps/Deeplearning/requirements.txt # only for the Deep Learning app (TensorFlow)
streamlit run app.py                              # opens http://localhost:8501
```

In the launcher, click **Start** next to an app and then **Open**. The apps run in the background on ports
**8502–8505** (also after the launcher is closed) and get the language of the launcher; **Stop** ends them.
Process IDs and logs are kept in the `.mlhub/` folder. If a package is missing, the launcher shows the exact
`pip install` command.

Every app can also be started on its own:

```bash
cd apps/Supervised
streamlit run app.py -- --lang en                 # or pl, de · opens http://localhost:8501
```

> **Older Streamlit (e.g. 1.51 from Anaconda)?** The apps use `st.segmented_control(required=…)`, which exists from
> Streamlit 1.56. The launcher starts the apps through `mlhub_compat.py`, which adds this parameter on older
> versions, so everything works from the launcher. Without the launcher, run
> `python ../../mlhub_compat.py run app.py` in the app folder instead of `streamlit run app.py`, or update Streamlit:
> `pip install --upgrade "streamlit>=1.56"`.

## Try it without your own data

The course datasets are not part of this repository. Each app can generate a **synthetic** file with the same
columns and formats as the original dataset, so everything can be tried out right away:

| App | Command (in the app folder) | Creates |
|---|---|---|
| Supervised | `python tests/synthetic.py` | `demo_lending_club_synthetisch.csv` (20 000 rows) |
| Clustering | `python tests/synthetic.py` · `python tools/bybit_features.py --top 10 --days 60` | `demo_college_synthetisch.csv` (777 rows) · real crypto features from public Bybit data (internet needed) |
| NLP | `python tests/synthetic.py` · `python tests/synthetic.py --sms` | `demo_yelp_synthetisch.csv` (10 000 reviews) · `demo_sms_synthetisch.txt` |
| Deep Learning | `python tests/synthetic.py` | `demo_lending_club_synthetisch.csv` (20 000 rows) |

Load the file in step 1. For these datasets the apps offer a ready-made example configuration (`configs/`), which
sets up the preparation and the model as in the course projects.

## Materials

Besides the apps, the launcher contains short reading material and three documents to download – each in English,
Polish and German. The launcher offers the version in the selected language; the others are under *Other
languages*. All files are in [`materials/`](materials):

| Document | Content | EN | PL | DE |
|---|---|---|---|---|
| FAQ | small datasets and neural networks – when a network makes sense, settings for early stopping | [PDF](materials/FAQ_EN.pdf) | [PDF](materials/FAQ_PL.pdf) | [PDF](materials/FAQ_DE.pdf) |
| ML in predictions | machine learning in financial, stock-market and political predictions | [PDF](materials/ML_in_predictions_EN.pdf) | [PDF](materials/ML_in_predictions_PL.pdf) | [PDF](materials/ML_in_predictions_DE.pdf) |
| How to read the results | how to read the results of 8 ML course notebooks (data, training, test results) | [PDF](materials/Reading_ML_results_EN.pdf) | [PDF (PL + DE)](materials/Reading_ML_results_PL-DE.pdf) | [PDF (PL + DE)](materials/Reading_ML_results_PL-DE.pdf) |

The Polish and German versions of the third document are one bilingual file (Polish part first, then German).

## Tests

**1070 automated tests** (pytest, including Streamlit `AppTest` tests of the whole interface) – run per app:

```bash
pip install pytest
cd apps/Supervised
pytest -q
```

| App | Tests |
|---|---|
| Supervised | 310 |
| Clustering | 382 |
| NLP | 188 |
| Deep Learning | 190 |

Each app folder also contains an `AUDIT.md` (in Polish) with the results of the code reviews: the bugs found, how
they were fixed and the regression tests that cover them.

## Project structure

```
.
├── app.py               launcher (ML Hub)
├── mlhub_compat.py      compatibility fix for Streamlit < 1.56
├── requirements.txt
├── materials/           PDF documents (EN · PL · DE) offered in the launcher
├── docs/screenshots/
└── apps/
    ├── Supervised/      📊 scikit-learn – six supervised models
    ├── Clustering/      🔵 K-Means · Isolation Forest · LOF · DBSCAN · Autoencoder
    ├── NLP/             📝 text classification
    └── Deeplearning/    🧠 Keras / TensorFlow
```

Every app has the same layout:

```
app.py              Streamlit interface
pipeline.py         computation logic – also usable in a notebook (import pipeline as pl)
plots.py            charts (matplotlib / seaborn)
codegen.py          generates the Python code shown in the app
i18n.py, locales/   languages – the texts are written in German in the code, locales/ holds English and Polish
configs/            example configurations
tests/              pytest tests + synthetic test data
README.md           German · README_EN.md English · README_PL.md Polish
AUDIT.md            code review report (Polish)
```

The comments in the source code are in German, the original language of the code.

## Tech stack

Python · Streamlit · pandas · NumPy · scikit-learn · SciPy · skops · matplotlib · seaborn · TensorFlow / Keras · pytest
