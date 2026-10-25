# DATA-Analysis-SupervisedLearning (Streamlit + scikit-learn)

Streamlit app for tabular CSV data: load, prepare, visualise, choose and train a supervised learning model
(scikit-learn), evaluate it and apply it to new data – for any dataset (binary, multi-class, regression). Six
models are available, sorted by their ranking: **Gradient Boosting (HistGradientBoosting), Decision Trees + Random
Forests, Support Vector Machines, Logistic Regression, K-Nearest Neighbors, Linear Regression** – see [The six
models](#the-six-models-step-6). For every step the app shows the matching Python code.

For data with an order (games, prices, measurement series) there is a **time-based split** (test = the most recent
rows), and step 8 can compare the model with a **probability from the data** – e.g. the win probability by Elo from
the sports data tool. See [Sports data: time-based split and comparison with
Elo](#sports-data-time-based-split-and-comparison-with-elo).

The app grew out of the deep learning version (Keras): steps 1–5 and 10, data preparation, the recipe
(`preprocessing.json`), configurations and languages are unchanged; new are the models in step 6, the validation
curves in step 7, the model details in step 8 and the formats `.skops`/`.joblib` in step 9.

The app is available in **German, English and Polish** – see [Languages](#languages).
(Deutsch: `README.md` · Polski: `README_PL.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py -- --lang en  # opens http://localhost:8501 in English
```

Tested with Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 or newer;
TensorFlow is no longer needed.

## Languages

The language can be switched at any time at the top of the sidebar: **Deutsch · English · Polski**. Only the
texts change – the calculations are identical in all three languages (German is the original).

| Start directly in a language | |
|---|---|
| command line | `streamlit run app.py -- --lang en` (or `pl`, `de`) – note the `--` before `--lang` |
| address in the browser | `http://localhost:8501/?lang=en` |
| environment variable | `ML_TEMPLATE_LANG=pl` (Windows: `set ML_TEMPLATE_LANG=pl`) – formerly `DL_TEMPLATE_LANG` |

Without any of these the app starts in English. A language chosen in the sidebar is then also written to the
address (`?lang=…`), so it stays when the page is reloaded.

**Translated:** all texts of the interface (pages, buttons, help texts, messages, warnings), chart titles and
axis labels, table headers, the comments and example file names in the generated Python code, the conclusion in
step 10, the file names of the downloads and the column names of the prediction CSV (Prediction, Confidence,
Note).

**The same in every language** – so that configurations, models and generated code work in all three versions:

* Internal values: actions and strategies are stored under their German names in configurations (`.json`) and in
  `preprocessing.json`. A configuration or a model saved in the English version can be loaded in the Polish or
  German version and vice versa.
* Values that the app writes into the data: `Sonstige` (rest group of "Keep top N"), `fehlend` (own category for
  missing values) and the column suffixes `_jahr` / `_monat` ("Date → year + month").
* In "Map values" the default for all other values can be written as `rest=…` or `sonst=…` in every language.
* The comments in the source code (`app.py`, `pipeline.py`, …) are German.

If the language is switched while working, messages that have already been produced (e.g. the log of step 2
or 4) stay in the previous language until the step is run again. Texts that come from Streamlit itself (e.g.
"Upload" in the file field or the ⋮ menu) are English in every version.

### Excel format in step 5 (Save data)

| Language | Excel option | Separator | Decimal sign |
|---|---|---|---|
| Deutsch | CSV für deutsches Excel | `;` | `,` |
| Polski | CSV dla polskiego Excela | `;` | `,` |
| English | CSV for Excel – UK, US, Australia … | `,` | `.` |

English-speaking countries (UK, US, Australia, Canada, Ireland, New Zealand …) all use the decimal point and the
comma as list separator, so one option covers all of them. All Excel variants are written as UTF-8 with BOM (so
Excel shows special characters such as ä, ł or é correctly) and are protected against CSV injection (see below).
The option "CSV international (comma, decimal point)" is the same in all languages.
Excel uses the list separator of the Windows region settings, not the language of Excel: if a file opens in a
single column, use the other format.

## Quick start

1. **Step 1:** upload a CSV or enter a path → *Load data*
2. **Step 2:** check the action for each column → *Apply*; create new columns if needed
3. **Step 3:** choose the columns for the charts and look at the data
4. **Step 4:** choose label and features, split at random or by time → *Prepare data for ML*
5. **Step 6:** choose a model (rank 1: Gradient Boosting) → *Start training* → then 7, 8, 9, 10

For the public LendingClub dataset (`lending_club_loan_two.csv`) step 2 offers an **example configuration**
(steps 2, 4 and 6 preset: Random Forest with 100 trees, `max_depth=12`). On a slow computer, switch on the
**sample** in the sidebar (`df.sample(frac=0.1)`).

## The steps

| Page | Content |
|---|---|
| 1 · Data & libraries | load CSV (upload or path; separator `;` and decimal comma are detected automatically), overview, library versions, optional column descriptions |
| 2 · Preparation 1 – columns | one action per column (Keep, Drop, Extract number, Date → year / month / **year + month**, first/last N characters, Replace values, Keep top N …) · **new columns**: Map values, Formula, Condition → 0/1, Copy + action |
| 3 · Analysis & visualisation | **choice of columns** · Overview, Count plot, Histogram, Box plot, Scatter plot, Correlation (heatmap), Correlation with one column, Pair plot, Share per category, Group statistics, **Joint plot** (scatter/kde/hex/reg), **Density 2D (KDE)** with a filter for one group, **Regression (lmplot)** with colour (hue) and subplots (col), **Histogram per group** (overlaid) |
| 4 · Preparation 2 – ML data | label + problem type, features, missing values per column (with correlation help for the group mean); split **first** – at random (`train_test_split`) or **by time** using a date/season column (test = the most recent rows) –, then fill values, dummies (overview of the new 0/1 columns) and scaling – learned from the training data only · optional **comparison column** for step 8 (e.g. `pre_elo_prob_home`) |
| 5 · Save data | data after step 2 or ML-ready after step 4 (with column `split`) as CSV – international or for Excel in the language of the app; an existing file is only overwritten after confirmation |
| 6 · Model & training | one of the **six models** in the order of the ranking (with "Why?" and "Best for"), plus its most important parameters as in the course; validation, `class_weight='balanced'`, seed – the matching scikit-learn code is shown live |
| 7 · Training vs. validation | validation curve of the most important parameter (error on training and validation data, naive prediction as a line) + interpretation (overfitting, underfitting, no signal); *Apply the best value in step 6* (also the best number of rounds for Gradient Boosting); SVM with GridSearchCV: heat map C × gamma |
| 8 · Evaluation | classification_report, confusion_matrix, ROC-AUC, **log loss**, threshold, single case, permutation importance · regression: MAE, RMSE, R² · **comparison with a probability from the data** (e.g. Elo): accuracy, log loss, Brier, ROC-AUC, difference ± standard error, calibration chart · **model details**: feature importance of the trees (+ tree chart for a single tree), coefficients of the Linear Regression (in original units like `lm.coef_` in the course) and of the Logistic Regression (with odds ratio), key figures of Gradient Boosting, SVM (support vectors) and KNN |
| 9 · Save & load model | `models/<name>/model.skops` (safe, default) or `model.joblib` + `preprocessing.json`, ZIP download, the whole pipeline as a Python script; checksums of training rows optional |
| 10 · Predict new data | new CSV or form → same preparation → prediction, interpretation, CSV download and **conclusion** (data quality, what the result says, recommendation) |

Every step has a box **"Python code for this step"** – so everything can be copied 1:1 into a Jupyter notebook.
Settings (incl. new columns) can be saved as JSON in the sidebar under **Save / load configuration** and loaded
again for new data with the same columns. When loading, the file is checked: unusable entries (wrong type,
outside the allowed range, e.g. 10 million trees) are ignored and named in the message. Configurations of the
deep learning version can be loaded: steps 2 and 4 are applied, the Keras settings (layers, epochs …) are skipped
silently.

### Save data (step 5)

* For predictions in step 10 always use the **original** CSV file, not the export: the saved model repeats the
  preparation from step 2 itself (e.g. "Replace values 1=2; 2=3" would otherwise be applied twice).
* Excel formats: texts that start with `=`, `+`, `-` or `@` get a `'` in front – otherwise Excel would execute
  them as a formula (CSV injection). In the international format the data stay unchanged.
* If a feature itself is called `split`, the train/test column is called `split_train_test`.

## The six models (step 6)

The order in step 6 is the ranking – rank 1 is usually the strongest:

| Rank | Algorithm | Why? | Best for |
|---|---|---|---|
| 1 | Gradient Boosting (HistGradientBoosting) | The best of this list for tables. Builds trees one after another – each corrects the errors of the previous ones. Usually more accurate than Random Forest and fast even with many rows; but it needs a suitable number of rounds (max_iter), otherwise it also learns noise – step 7 shows the best one. | First choice for tables: direction, win/loss (classification) and numbers (regression) |
| 2 | Decision Trees + Random Forests | Very robust and hard to get wrong. Handles non-linearities, feature interactions and noise. Random Forest strongly limits the overfitting of a single tree. | Predicting direction (classification) and returns |
| 3 | Support Vector Machines | Good, especially with the RBF kernel. Separates classes well, but is sensitive to data scaling and to the choice of hyperparameters. | Direction classification (buy/sell) |
| 4 | Logistic Regression | Solid baseline. Fast, interpretable, stable. Often loses to trees on complex relationships. | Simple direction classification |
| 5 | K-Nearest Neighbors | Simple, but usually weaker. Copes poorly with noise, irrelevant features and high dimensionality (typical of market data). | Rather experiments / baseline |
| 6 | Linear Regression | The weakest for real predictions. Assumes a linear relationship, but markets are highly non-linear and noisy. | Only as a very simple reference point |

| Model | scikit-learn | Parameters in step 6 | Curve in step 7 |
|---|---|---|---|
| Gradient Boosting | `HistGradientBoostingClassifier/Regressor` (`early_stopping=False`) | `max_iter` (rounds), `learning_rate`, `max_leaf_nodes`, `max_depth` (0 = unlimited), `min_samples_leaf` – defaults as in scikit-learn | error by number of rounds from **one** training run (`staged_predict`), with best value and *apply* |
| Decision Trees + Random Forests | `RandomForestClassifier/Regressor` or `DecisionTreeClassifier/Regressor` | variant (forest / one tree), `n_estimators`, `max_depth` (0 = unlimited), `min_samples_leaf`, `max_features` | error by number of trees (the forest grows step by step) or by `max_depth` |
| Support Vector Machines | `SVC` / `SVR` | `kernel`, `C`, `gamma`, `epsilon` (SVR); **GridSearchCV** over C = 0.1 … 100 and gamma = 1 … 0.001 as in the course | error by C; with GridSearchCV also the heat map of all combinations |
| Logistic Regression | `LogisticRegression` | `C`, `max_iter` | error by C |
| K-Nearest Neighbors | `KNeighborsClassifier/Regressor` | `K (n_neighbors)`, `weights`, distance (Euclidean/Manhattan) | error rate by K = 1 … 40 (elbow method as in the course) |
| Linear Regression | `LinearRegression` | `fit_intercept` | learning curve: error by number of training rows |

* **Does the model fit the problem type?** Logistic Regression only predicts classes (binary, multi-class),
  Linear Regression only numbers (regression). For an unsuitable model step 6 explains why, and *Start training*
  stays disabled. The other four models can do both.
* **Gradient Boosting:** unlike with Random Forest, more rounds can hurt – from some point on the model learns
  noise and the validation error rises. `early_stopping` is therefore off (otherwise scikit-learn would set aside a
  random 10 % of the training data itself from 10 000 rows on); the curve in step 7 shows the best number of rounds.
  `HistGradientBoosting` has no `feature_importances_` – the permutation importance in step 8 shows the importance.
  It does not need scaling.
* **Error in step 7:** classification = error rate (1 − accuracy, like `np.mean(pred != y_test)` in the course),
  regression = MAE in original units. The dashed line is the naive prediction (most frequent class or mean) – a
  model below it has learned something.
* **Validation:** "Test data" computes the curve on `X_test` as in the course; "Share of training data (cleaner)"
  takes a separate part of the training data so that the test set stays untouched until step 8 – with the
  time-based split the **most recent** part (`train_test_split(..., shuffle=False)`); GridSearchCV then uses
  `TimeSeriesSplit`. Without a validation curve, step 7 only compares training and test.
* **Apply the best value:** step 7 suggests a value when it lowers the error by more than chance, and enters it
  in step 6 at the push of a button – for the trained model; for the SVM with GridSearchCV the search is switched
  off for this (otherwise it would replace C again). Then train again.
* **Probabilities of the SVM:** for the threshold and ROC-AUC (step 8) the SVM is calibrated with
  `CalibratedClassifierCV(SVC(...), ensemble=False)` – `SVC(probability=True)` is deprecated since
  scikit-learn 1.9.
* **Large data:** SVM and KNN get slow with very many rows – step 6 points this out (tip: sample in the sidebar),
  and also missing scaling for SVM and KNN (they compute distances). The errors of the validation curves are
  measured on at most 5000 rows per data set (KNN: 2000); training uses all rows.

## Sports data: time-based split and comparison with Elo

The app recognises CSV files in the format of the sports data tool (Datasety NBA i tenis) by the label `home_win` or
`p1_win` together with `pre_*`/`lineup_*` columns – also exports in which the groups `inplay_*` and `post_*` were
unticked in the tool – and proposes suitable settings in step 4:

| Proposal | Why |
|---|---|
| features: only `pre_*` and `lineup_*` | known before the game. `inplay_*` (during) and `post_*` (after the game) would be a data leak – if you select them anyway, step 4 warns |
| label `home_win` (NBA) or `p1_win` (tennis) | target of the prediction |
| time-based split by `date` (tennis: `match_date_est`) | training = older games, test = the most recent ones. Games of the same day never end up in both parts. For this the date column stays unchanged in step 2 (otherwise "Date → year" would be proposed) |
| comparison column `pre_elo_prob_home` (tennis: `pre_elo_prob_p1`) | the win probability by Elo before the game – the tool computes Elo game by game from earlier games only. It refers to the home team or player 1, i.e. class `1` – so it is only proposed when `1` is the positive class. If a chosen comparison column predicts the positive class worse than chance on the training rows (ROC-AUC below 0.5), steps 4 and 8 warn |

**Comparison in step 8.** On the same test rows, the model, the comparison column and the naive prediction (always
the share of the positive class in training) are shown side by side: accuracy (threshold 0.5), log loss, Brier and
ROC-AUC. What matters is the **log loss** – it measures how well the probabilities fit. The app computes the mean
difference per game with its standard error (paired, the same games) and says whether the model beats the
comparison column by more than chance. The calibration chart shows whether e.g. 70 % predicted also happens in about
70 % of cases – only then are the probabilities fit for a comparison with betting odds.

Example with the tool's NBA data (as of 22.10.2026, 28,382 games since 2004-05, test = the most recent 20 % from March
2022 – `test_size` 0.2 –, otherwise proposals from step 4; Gradient Boosting on all `pre_*`/`lineup_*` with 300
rounds, `learning_rate=0.05` and the validation "Share of training data (cleaner)"): log loss 0.621 against 0.620 for
Elo alone – no advantage, because the model overfits. Step 7 reports this and proposes 50 rounds; after "Apply the
best value in step 6" and training again: log loss **0.613 against 0.620**, a difference of −0.006 ± 0.002 per game –
the model beats Elo narrowly, but by more than chance. So don't guess the number of rounds, check it in step 7. This
matches the tool's benchmark (training up to season 2023, test from 2024): `pre_*` only 0.6083 ≈ Elo 0.6085, with
`lineup_*` (who plays) 0.6022. In tennis (ATP, also 20 % test, from February 2021) even Gradient Boosting with the
default values is clearly ahead of Elo: 0.608 against 0.634.

The time-based split works with any date or number column (year, season), the comparison column with any column of
probabilities between 0 and 1 for the positive class (e.g. from betting odds) – also outside sports data. Rows
without a time value are removed; rows with the same time value stay together (so the test share can differ a little
from `test_size`, e.g. with whole seasons). Step 4 checks the result: if fewer than 10 training rows remain or the
training data lacks a class (e.g. because it only occurs in the most recent rows), a message explains what to do; if
the boundary falls on the oldest time value, it names the largest possible `test_size`.

## New columns (step 2)

| Kind | Example | Result |
|---|---|---|
| Map values | from `loan_status`: `Fully Paid=1; Charged Off=0` | new column `loan_repaid` (1/0); values that are not listed stay empty, or `rest=…` |
| Formula | `loan_amnt / annual_inc` | number per row; allowed are `+ - * / ** // %`, brackets, `abs log log10 sqrt exp round` |
| Condition → 0/1 | `dti > 30 and home_ownership == 'RENT'` | 1 = true, 0 = false, empty where values are missing |
| Copy + action | `earliest_cr_line` + "Date → year" | the original stays, the result is an additional column |

In formulas, write column names with spaces or special characters in \`backticks\`. Formulas are checked: only
arithmetic, comparisons and the functions listed – no arbitrary Python code. So that no formula can block the
program, there are limits: at most 1000 characters and 100 nesting levels, exponent of `**` between −100 and 100
(or a column), `round(x, n)` with n from 0 to 10, text in quotes only in comparisons. If not a single value
matches in "Map values" (e.g. different upper/lower case), the message lists the existing values. If a new
column is used by a later one, both can only be removed together. New columns belong to the saved recipe and are
calculated automatically for new data in step 10. A new column that is created from the same CSV column as the
label (e.g. `loan_repaid` from `loan_status`) is not suggested as a feature (data leakage).

## Step 10: handling new data correctly

* **Nothing is fitted again.** New data run through the saved *recipe* (`preprocessing.json`): the same column
  actions and new columns, fill values, dummy columns in the same order, the same scaler, the same label encoding
  and threshold.
* **The same raw columns as in training** – the model defines the features. The app shows the required columns
  and offers a CSV template. Missing columns → error message; additional columns (also the label) are ignored.
* **Deviations** are handled and explained in the **conclusion**: gaps → values from the training (every row
  needs a value for a prediction – even where rows with gaps were removed in training), unknown categories → all
  dummies 0, values outside the training range → note.
* **Training data detected:** the recipe contains checksums (hashes) of a sample of up to 3000 training rows. If
  training rows are among the new data, the conclusion states their share – metrics on such data are too
  optimistic. A checksum cannot be turned back into a row; but whoever has the original data can check whether a
  particular row was used in training. For confidential data, therefore switch off *Save checksums of training
  rows* in step 9 before passing the model on (the check is then skipped).
* Rows with an infinite label (`inf`) are removed in step 4 and not evaluated in step 10.
* If the new data already contain columns with the name of a result column (e.g. `Prediction` or `Note` from an
  earlier export), they are replaced in the result – the app names them.
* If the label is present in the new data, they are evaluated automatically and compared with the test from the
  training.

Without the app:

```python
import i18n
import pipeline as pl

i18n.set_lang("en")  # messages and conclusion in English (default: German)
model, recipe = pl.load_bundle("models/my_model")
df_new, _ = pl.read_csv("new_data.csv")
result, report, evaluation = pl.predict_new(model, recipe, df_new)
conclusion = pl.assess_prediction(recipe, result, report, evaluation)
```

Outside the app the column names of `result` stay German (`Vorhersage`, `Sicherheit`, `Hinweis`) – the app
translates them only for display and download.

## Notes on current library versions

Older tutorials and notebooks often contain lines that no longer run with current versions:

| older code | problem today | in the template |
|---|---|---|
| `SVC(probability=True)` | scikit-learn 1.9: deprecated (FutureWarning), removed in 1.11 | `CalibratedClassifierCV(SVC(...), ensemble=False)` |
| `LogisticRegression(penalty='l2')` | scikit-learn 1.8: `penalty` deprecated, removed in 1.10 | without `penalty` (default L2), strength via `C` |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | loading can run arbitrary code from the file | `.skops` as default (`skops.io`, only known scikit-learn types); `.joblib` only for trusted files |
| `scaler.fit(df.drop('TARGET CLASS', axis=1))` **before** `train_test_split` (KNN project) | mean and spread contain information from the test data (data leakage) | split first, scaler only on `X_train` |
| `df.corr()` | pandas ≥ 2: error with text columns | `df.corr(numeric_only=True)` |
| `df.select_dtypes(['object'])` | pandas 3: text has the type `str`; `'object'` only finds it for a transition period | type check with `is_numeric_dtype` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False instead of 0/1 | `dtype=float` |
| `df.groupby('total_acc').mean()['mort_acc']` | pandas ≥ 2: error with text columns | `df.groupby('total_acc')['mort_acc'].mean()` |
| `sns.distplot` | deprecated | `sns.histplot` |
| `sns.kdeplot(df['a'], df['b'], shade=True, shade_lowest=False)` | seaborn ≥ 0.12: error, only `data` may be passed without a name; `shade`/`shade_lowest` deprecated | `sns.kdeplot(x='a', y='b', data=df, fill=True, thresh=0.05)` |
| `plt.figure(figsize=(11, 7))` before `sns.lmplot(...)` | has no effect – lmplot creates its own figure (in the notebook: "Figure … with 0 Axes") | `sns.lmplot(..., height=4, aspect=1.15)` |
| fill gaps, **then** `train_test_split` | the fill values contain information from the test data (data leakage) | split first; fill values, categories and scaler only from the training data |

Additional options (each can be switched off): time-based split, comparison column (e.g. Elo), `stratify=y`,
adjustable threshold + ROC-AUC,
`class_weight='balanced'`, validation from the training share instead of the test data, GridSearchCV, label
scaling for regression, data-leakage warning, permutation importance.

## Files

```
app.py                  Streamlit interface (UI only)
pipeline.py             computation logic – also usable in a notebook (import pipeline as pl)
plots.py                charts (matplotlib/seaborn), return figures
codegen.py              creates the Python code that is shown or exported
i18n.py                 languages: tx() translates a German text into the current language
locales/                translations {German: (English, Polish)}, one file per module
configs/                lending_club_beispiel.json (example configuration LendingClub)
.streamlit/config.toml  upload limit 1 GB, light theme
tests/                  pytest tests + synthetic test data
AUDIT.md                audit report (Polish): bugs found, fixes, tests
README.md               German · README_EN.md English · README_PL.md Polish
models/                 created when saving (step 9)
exports/                default folder for "Save to folder" (step 5)
```

Extending:
* new column action → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* new kind of new column → `NEW_KINDS` + `apply_new_columns()` in `pipeline.py` (+ `_new_column_lines()` in
  `codegen.py`)
* new chart → function in `plots.py` + entry in `CHARTS` / `page_viz()` in `app.py` (+ `chart_code()` in
  `codegen.py`)
* new model → `MODELS`, `MODEL_NAMES`, `MODEL_WHY`, `MODEL_BEST`, `MODEL_TASKS`, `ModelConfig`, `make_model()` and
  `validation_curve()` in `pipeline.py`, `model_params_ui()` in `app.py` (+ `estimator_code()` and `curve_code()`
  in `codegen.py`)
* new or changed text → write it in German inside `tx("…")` (constants: `N_("…")`) and add the English and Polish
  translation to the matching file in `locales/`; `pytest tests/test_i18n.py` reports missing translations.

## Tests

```bash
pip install pytest
pytest -q                                  # 310 tests, about 5 minutes
python tests/synthetic.py --rows 20000     # synthetic demo CSV (no real data)
```

| File | checks |
|---|---|
| `tests/test_pipeline.py` | example pipeline, save/load (.skops/.joblib), new data, regression, multi-class, exported script |
| `tests/test_models.py` | the six models: every model × problem type (training, validation curve, interpretation, model details, save/load), the scikit-learn code shown (model and curve) computes **exactly** like the app, elbow method = `predict()` per K, coefficients in original units, safe loading (.skops/.joblib), click test regression (Linear Regression, KNN, SVM + GridSearchCV) |
| `tests/test_codegen_parity.py` | the Python code shown computes **exactly** the same as the app (X_train, X_test, y) – also in edge cases and with new columns |
| `tests/test_new_features.py` | year + month, new columns (incl. protection against arbitrary code in formulas), save data, conclusion |
| `tests/test_audit_regressions.py` | one test for each bug from the first audit (see `AUDIT.md`) |
| `tests/test_audit2_regressions.py` | second audit: limits for formulas, check of imported configurations, `inf` in the label, export (Excel, column `split`), charts with empty columns, ZIP size, conclusion |
| `tests/test_app.py` | click test of the interface (all 10 steps, new columns, export, conclusion; bool label, semicolon CSV, overwrite protection; dependent new columns, unusable values, errors while drawing, model without checksums) |
| `tests/test_audit3_regressions.py` | third audit: no code from column names/values in the Python code shown, step 10 with columns of the same name, chart titles (notebook, `$`), language switch (tables, export, address) |
| `tests/test_audit4_regressions.py` | fourth audit (scikit-learn): “apply the best value” for the trained model and with GridSearchCV, exact numbers in the code shown, GridSearchCV with few rows per fold, question before pickle, names from files not as Markdown, safe loading code for `.skops`, saving without losing the old model, size limit for `.skops`, note on `drop_first`, axis texts in the current language |
| `tests/test_charts.py` | new charts in step 3 (joint plot, density 2D, lmplot, histogram per group): unusual data, the code shown runs, click test |
| `tests/test_audit5_regressions.py` | fifth audit (Gradient Boosting, time-based split, Elo): comparison column of the other class (no proposal, warning in steps 4 and 8), time-based split without both classes or with too little training data, recognition without `post_*`/`inplay_*` and without false alarms, step 8 code with a removed comparison column, “prepare again” note after changing the comparison column, messages (boundary on the oldest time value, comparison column without test values or = label, `test_size` 0/1, GridSearchCV with `TimeSeriesSplit`), speed of step 4 with wide data |
| `tests/test_sports_timesplit.py` | recognising the sports data format (proposals, date column kept), time-based split (training before test, same days together, number/date, error messages), the code shown = app also with the time-based split, Gradient Boosting (curve from one training run, best value, note "more rounds"), GridSearchCV with `TimeSeriesSplit`, comparison with a probability (verdict, standard error, rows without a value), whole script, click test with sports data (steps 1–10, “apply the best value” with Gradient Boosting) |
| `tests/test_i18n.py` | languages: every text has an English and a Polish translation with the same placeholders, German unchanged, Excel CSV per language, the generated code computes the same in every language, the whole app in English and Polish without German leftovers |

## Notes

* Very large CSVs are better loaded via "Path on this computer" than uploaded.
* Don't click anything during the training – every input restarts the script and stops the training.
* scikit-learn computes on the CPU (no GPU needed). Random Forest, KNN and GridSearchCV use all cores
  (`n_jobs=-1`), Gradient Boosting too (OpenMP); step 1 shows the number of CPU cores.
* **Use locally only.** The path field reads every file the logged-in user has access to, and `models/` is the
  same for all users of the app. Therefore do not put the app on a network without a login. Load models only
  from trustworthy sources (a ZIP may be at most 2 GB unpacked; formulas in the recipe are checked again when
  loading). `.skops` is loaded without running code – only known scikit-learn types are allowed, the app refuses
  everything else. `.joblib` is pickle and can run code when loading: from an uploaded ZIP the app loads a
  `.joblib` file only after explicit confirmation. Models of the deep learning version (`model.keras`/`.h5`)
  cannot be loaded here – the message says so.
* **Test metrics are slightly optimistic** if the test data also serve for the validation curve and the best value
  is applied afterwards (default "Validation: Test data", as in the course) or if the threshold in step 8 is set
  on the test data. For an honest estimate choose "Share of training data (cleaner)" in step 6; GridSearchCV
  searches with cross-validation on the training data anyway.
* **No data leakage in step 4:** fill values, categories (dummy columns) and scaler are learned after the split
  from the training data only. A category that occurs only in the test data is treated like an unknown value in
  new data (all dummies 0). Only two things are still derived from all rows (before the split): the list for
  "Keep top N" in step 2 and the automatic suggestions in the tables – both with little influence.
