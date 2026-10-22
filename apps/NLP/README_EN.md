# DATA-Analysis-NLP (Streamlit + scikit-learn, text classification)

Streamlit app for CSV data with texts: load, prepare, visualize, turn texts into numbers (**bag of words** with
`CountVectorizer` or **TF-IDF**) and classify them with three models – **Naive Bayes (MultinomialNB)**,
**logistic regression** and **linear SVM (LinearSVC)**. The app then shows which words speak for which class,
explains individual predictions and classifies new texts. For every step it shows the matching Python code.

The models are the NLP parts of the course: the **Yelp project** (reviews with 1 and 5 stars, `CountVectorizer` +
`MultinomialNB`, `test_size=0.3`, `random_state=101`) and the **SMS spam lecture** (`SMSSpamCollection`,
tab-separated without a header row, pipeline of `CountVectorizer`, `TfidfTransformer` and `MultinomialNB`). There is a
ready-made example configuration for both.

The app grew out of the supervised version (DATA-Analysis-SupervisedLearning): steps 1–3 (loading, column actions,
new columns, charts), the recipe (`preprocessing.json`), saving/loading, configurations and languages stayed the
same; new are the detection of files without a header row, the text actions and text charts, step 4 (text, label,
classes, vectorization), the three text models with a comparison, the word analysis in step 8 and classifying new
texts in step 10.

The app is available in **German, English and Polish** – see [Languages](#languages).
(Deutsch: `README.md` · Polski: `README_PL.md`)

## Start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # opens http://localhost:8501
```

Tested with Python 3.13, scikit-learn 1.9, skops 0.16, pandas 3.0, Streamlit 1.66. Python 3.10 or newer.
The NLP version needs no additional packages (no nltk – the stop words come from scikit-learn).

## Languages

Switch the language at any time at the top of the sidebar: **Deutsch · English · Polski**. Only the texts change –
the calculations are identical in all three languages (German is the original).

| Start directly in a language | |
|---|---|
| Command line | `streamlit run app.py -- --lang en` (or `pl`, `de`) – the `--` before `--lang` is required |
| Browser address | `http://localhost:8501/?lang=pl` |
| Environment variable | `NLP_TEMPLATE_LANG=de` (Windows: `set NLP_TEMPLATE_LANG=de`) |

Without a choice the app starts in English. A language chosen in the sidebar also appears in the address
(`?lang=…`), so it survives reloading the page.

Translated are all texts of the interface, chart titles and axes, table headers (also in the export of the
vocabulary and of the test predictions), the comments and example file names in the generated Python code, the
conclusion in step 10 and the file names of downloads. The same in all languages:

* the internal values – actions are stored in configurations (`.json`) and in `preprocessing.json` with their German
  names; configurations and models therefore work in all three languages;
* the values the app writes into the data: `Sonstige` (top N), `_jahr` / `_monat` (date → year + month), `Spalte_1`,
  `Spalte_2` … (file without a header row) and the column `Hinweis` in the result of the engine (translated in the
  interface and in the download);
* the example texts in the generated code ("The food was great!") – the course data is English;
* the comments in the source code (German).

If you switch the language in the middle of your work, messages that already exist (e.g. the log of step 2 or 4)
stay in the old language until the step is run again. Texts of Streamlit itself (e.g. "Upload" in the file field or
the ⋮ menu) are English in every version.

Excel format in steps 5, 8 and 10 per language: German and Polish `;` and decimal comma, English `,` and decimal
point – always UTF-8 with BOM and with protection against CSV injection (in step 5 also in the generated Python
code). Texts from other sources (reviews, messages) may start with `=`, `+`, `-` or `@` – Excel would otherwise run
them as a formula.

## Quick start – as in the course

**Yelp (reviews, 1 vs. 5 stars)**

1. **Step 1:** load `yelp.csv`
2. **Step 2:** *Load example configuration* → *Apply* (drop IDs, new column `text length`)
3. **Step 3:** *Text length* per `stars` (like `sns.FacetGrid(yelp, col='stars')`), *Most frequent words* per class,
   *Means per group* (like `yelp.groupby('stars').mean()`)
4. **Step 4:** *Vectorize texts* (only 1 and 5 stars, CountVectorizer, `test_size=0.3`, `random_state=101`)
5. **Step 6:** *Start training* (MultinomialNB) and *Compare models* → steps 7, 8 (most important words, mistakes,
   your own text), 9, 10

**SMS spam**

1. **Step 1:** load `SMSSpamCollection` – tab and missing header row are detected (columns `Spalte_1`, `Spalte_2`)
2. **Step 2:** *Load example configuration* → *Apply* (rename to `label` and `message`, new column `length` like
   `messages['message'].apply(len)`)
3. **Step 4:** TF-IDF with English stop words, `test_size=0.2` → step 6: Naive Bayes (or LinearSVC with
   `class_weight='balanced'`)

For your own data: in step 4 choose the text column, the label and optionally some classes; in step 6 a model – the
comparison shows which fits best. On a slow computer activate the **sample** on the left (`df.sample(frac=0.1)`).

## The steps

| Page | Content |
|---|---|
| 1 · Data & libraries | load a CSV (upload or path); separator (also tab), decimal comma and a **missing header row** are detected automatically (the first row counts as data when its value is frequent in a text column – like "ham" in SMSSpamCollection; header row: automatic / yes / no); overview with the detected free-text columns (Ø words), library versions |
| 2 · Preparation 1 | one action per column (keep, drop, rename, extract number, date → year / month, first/last N characters, replace values, top N, **text → length (characters)**, **text → number of words** …) – free text and short texts (titles, headlines) stay *keep*, IDs are suggested for dropping · new columns: map values, formula, condition → 0/1, copy + action · quick buttons "length in characters" and "number of words" (like `text length` in the course) |
| 3 · Analysis & visualization | overview, countplot, **text length** per class (histograms like FacetGrid + box plot), **most frequent words** overall or per class (with/without stop words, words or word pairs), **example texts** per class, histogram, box plot, correlation, **means per group** (like `yelp.groupby('stars').mean()` + heatmap) |
| 4 · Preparation 2 – text → numbers | **text column** and **label**, optionally only some **classes** (as in the course 1 and 5 stars), positive class, drop rows without text or treat them as empty text, `test_size`, `random_state`, `stratify` · **vectorizer**: CountVectorizer or TfidfVectorizer, stop words (English), `ngram_range`, lower case, accents, `min_df`, `max_df`, `max_features`, `binary`, `sublinear_tf` · result: size of the vocabulary, share of non-zeros, class distribution, **one text as numbers** (like `bow_transformer.transform([text])` in the course), vocabulary table, warnings about duplicate texts in training and test and about classes with very few training texts (fewer than 5) |
| 5 · Save data | data after step 2, **texts after step 4** (text, label, label as a number, `split`) or the **vocabulary** (word, number of texts, count or IDF); international or for Excel |
| 6 · Model & training | **MultinomialNB** (alpha, fit_prior), **logistic regression** (C, max_iter), **LinearSVC** (C, with calibration for probabilities), `class_weight='balanced'`, seed; validation curve (test data or a share of training); the scikit-learn code is shown live, the curve grows during training; tips on the combination with the vectorizer (e.g. LinearSVC works best with TF-IDF) · **comparison of all three models** (accuracy, F1, ROC-AUC, time) – if a class has only one training text, LinearSVC is left out with the reason |
| 7 · Training vs. validation | error rate per alpha or C (logarithmic) with an assessment (overfitting, underfitting, "a training error of 0 is normal for text", not better than naive) and the suggestion *Apply best value* |
| 8 · Evaluation & words | accuracy, F1, ROC-AUC, classification_report, confusion matrix, threshold (binary) · **most important words per class** · **misclassified texts** (most confident mistakes first) · single case and **your own text** with coloured words (blue for, orange against) and the contribution of each word · test texts with prediction as CSV (international or for Excel) |
| 9 · Save & load model | `models/<name>/model.skops` (safe, default) or `model.joblib` – the **pipeline of vectorizer and model** – + `preprocessing.json` (recipe, classes, threshold, metrics, optionally checksums of training texts), ZIP download, the whole pipeline as a Python script |
| 10 · Classify new texts | **single text** (with explanation) or **CSV** → same preparation → prediction, probabilities, notes per row, evaluation (if the label is included), CSV download (international or for Excel) and **conclusion** |

Every step has a field **"Python code for this step"** – so everything can be copied 1:1 into a Jupyter notebook;
the code computes exactly the same as the app (same vocabulary, same matrices, same probabilities, same curve, same
most important words, same exports). Settings (including new columns) can be saved as JSON on the left under
**Configuration** and loaded again for new data with the same columns. When loading, the file is checked: unusable
entries (wrong type, outside the allowed range, e.g. `ngram_range` (1, 7)) are ignored and named in the message.
Configurations of the other versions can be loaded: step 2 is applied, their label and model settings are skipped
silently.

## Text → numbers in this app

* **Split first, then learn the vocabulary.** In the course `cv.fit_transform(X)` was run on **all** texts and only
  then split – the vocabulary already knew the test texts (a small data leak). The app splits first
  (`train_test_split`) and fits the vectorizer on the training texts only; the test texts are only transformed with
  it – exactly like new texts later. Words that occur only in the test set are dropped (step 4 shows what share of
  the test words the vocabulary knows).
* **Sparse matrix.** Every text becomes a row with one column per word of the vocabulary; almost all values are 0.
  scikit-learn stores only the non-zeros – that is why 100 000 texts with 50 000 words still work.
* **CountVectorizer or TF-IDF.** Count counts (as in the course); TF-IDF down-weights words that occur in many texts
  – `TfidfVectorizer` equals `CountVectorizer` + `TfidfTransformer` from the course.
* **Stop words:** the English list of scikit-learn (in the course: `stopwords.words('english')` from nltk plus
  removing punctuation – the vectorizer does that itself). There is no built-in list for German or Polish texts;
  `max_df` (e.g. 0.9) and `min_df` help there.
* **Word pairs** (`ngram_range=(1, 2)`) detect e.g. "not good" but make the vocabulary much larger.
* **Label encoding:** classes become 0/1 (binary, the *positive class* = 1) or 0 … K−1; numeric labels (stars) are
  encoded by value, text labels without surrounding spaces ("ham " = "ham"); a label of only spaces counts as
  missing (not a class of its own).
* **Duplicate texts:** if a test text also appears word for word in training (frequent with SMS templates), step 4
  warns – the model already knows the answer for these texts.

## The three models

| Model | main parameter | strengths | probabilities |
|---|---|---|---|
| Naive Bayes (MultinomialNB) – as in the course | `alpha` (smoothing, default 1.0) | very fast, needs little data, good baseline | directly (`predict_proba`) |
| Logistic regression | `C` (small = stronger regularization) | usually more accurate with enough texts, well calibrated, readable weights | directly |
| Linear SVM (LinearSVC) | `C` | often the most accurate with TF-IDF | via `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |

* **Validation curve (step 7):** error rate on training and validation for alpha = 0.01 … 5 or C = 0.01 … 100
  (LinearSVC: 0.001 … 10, without calibration). With "share of training" a separate vectorizer learns its vocabulary
  only from the remaining training part – otherwise it would know the validation texts.
* **A training error close to 0 is normal for text:** there are many more words than texts, so linear models can
  almost always separate the training texts. What matters is the validation error compared with the naive
  prediction.
* **Most important words (step 8):** Naive Bayes – `log P(word | class)` minus the mean of the other classes;
  logistic regression and LinearSVC – the coefficients. For a single text the **contribution** of a word = value in
  the text (count or TF-IDF) × weight for the predicted class; for Naive Bayes the contributions plus the prior give
  exactly the log odds.
* **TF-IDF and the parameters:** TF-IDF values are small (mostly below 1). Naive Bayes then often needs a smaller
  `alpha` (e.g. 0.1 instead of 1.0 – otherwise it smooths almost everything away with a large vocabulary), logistic
  regression a larger `C` (e.g. 10); LinearSVC converges much faster with TF-IDF than with raw word counts. Step 6
  shows a tip, the validation curve in step 7 the best value.
* **LinearSVC and rare classes:** the probabilities come from a calibration with cross-validation – it needs at
  least 2 training texts per class. Otherwise the app says so clearly (step 6) and the comparison runs without
  LinearSVC.

## Step 10: classify new texts

* **Nothing is refitted.** New texts go through the saved *recipe* (step 2, e.g. renaming) and the saved *pipeline* –
  the vectorizer uses the vocabulary from training. Unknown words are ignored.
* **Only the text column is needed** (the raw column of the CSV) – the app shows it and offers a CSV template; the
  label is optional (then the predictions are evaluated). If the model was trained with a file without a header
  row, load new data in the same form (header row: no).
* **Notes per row:** empty text · no known word (another language?) · mostly unknown words.
* **Conclusion:** how many words the model knows (compared with the test data), length of the texts compared with
  training, predicted shares, uncertain predictions, comparison with the test from training, **texts known word for
  word from training** (checksums, optional) and classes in the label that the model does not know (e.g. 2–4 stars
  for a 1-vs-5 model).

Without the app:

```python
import pipeline as pl
pipe, recipe = pl.load_bundle("models/my_model")
df_new, _ = pl.read_csv("new_texts.csv")
result, report, evaluation = pl.predict_texts(pipe, recipe, df_new)
conclusion = pl.assess_prediction(recipe, result, report, evaluation)
pipe.predict_proba(["The food was great!"])   # single texts directly with the scikit-learn pipeline
```

## Notes on current library versions

| older code | problem today | in the template |
|---|---|---|
| `cv.fit_transform(X)` before `train_test_split` | vocabulary from all texts – the test texts flow into the model | split first, `fit_transform` only on `X_train`, `transform` on `X_test` |
| `CountVectorizer(analyzer=text_process)` with nltk | slow; a custom function cannot be loaded safely with `skops`; nltk stop words need a download | `CountVectorizer(stop_words='english')` – the vectorizer removes punctuation itself |
| `CountVectorizer` + `TfidfTransformer` | still works, two steps | `TfidfVectorizer` (the same in one step) |
| `cv.get_feature_names()` | removed in scikit-learn 1.2 | `get_feature_names_out()` |
| `SVC(probability=True)` | deprecated since scikit-learn 1.9 | `CalibratedClassifierCV(LinearSVC(), ensemble=False)` |
| `sns.FacetGrid(yelp, col='stars', size=…)` | `size` has been called `height` since seaborn 0.9 | `col_wrap`/`height` or the *Text length* chart |
| `yelp.groupby('stars').mean()` | pandas ≥ 2: error with text columns | `mean(numeric_only=True)` |
| `pd.read_csv(..., sep='\t', names=[...])` | still works | the app detects tab and the missing header row itself; the generated code uses `header=None, names=[...]` |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | loading can execute arbitrary code from the file | `.skops` as default; `.joblib` only for trusted files |

## Files

```
app.py                  Streamlit interface (UI only)
pipeline.py             engine – also usable in a notebook (import pipeline as pl)
plots.py                charts (matplotlib/seaborn), return figures
codegen.py              generates the displayed or exported Python code
i18n.py                 languages: tx() translates a German text into the current language
locales/                translations {German: (English, Polish)}, one file per module
configs/                yelp_beispiel.json, sms_beispiel.json (example configurations as in the course)
.streamlit/config.toml  upload limit 1 GB, light theme
tests/                  pytest tests + synthetic test data (yelp_like, sms_like, news_like)
AUDIT.md                report (Polish): structure, checks, limits
README.md               German · README_EN.md English · README_PL.md Polish
models/                 created when saving (step 9)
exports/                default folder for "Save to folder" (step 5)
```

Extending:
* new column action → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* new chart → function in `plots.py` + entry in `page_viz()` or `page_eval()` in `app.py`
* another text model → `MODELS`, `MODEL_NAMES`, `TextModelConfig`, `make_model()`, `word_weights()` in `pipeline.py`,
  `model_params_ui()` in `app.py`, `estimator_code()` in `codegen.py`
* new or changed text → write it in German in `tx("…")` (constants: `N_("…")`) and add the English and Polish
  translation in the matching file in `locales/`; `pytest tests/test_i18n.py` reports missing translations.

## Tests

```bash
pip install pytest
pytest -q                                  # 188 tests, about 4 minutes
python tests/synthetic.py                  # synthetic Yelp-like data (10 000 reviews, no real data)
python tests/synthetic.py --sms            # synthetic SMS file like SMSSpamCollection (tab, no header row)
```

| File | checks |
|---|---|
| `tests/test_pipeline.py` | loading (tab without header row, quotes, semicolon, latin-1), text actions, step 4 (class selection, positive class, bool/numeric/text label, empty texts, clear errors, duplicate texts, data leak, only stop words), all three models learn the sentiment, explanation = prediction (log odds), multiclass and comparison, exports, saving/loading, new texts (notes, overlap with training, missing column), SMS workflow with renaming |
| `tests/test_codegen_parity.py` | the displayed Python code computes **exactly** the same as the app: vocabulary, matrices, split, probabilities, curve, most important words, exports – also with a file without a header row, class selection, empty texts, sample, bool/numeric label; model comparison, the whole script and the chart code run |
| `tests/test_audit_regressions.py` | security and robustness: ZIP limits, pickle only after confirmation, custom functions in the model are refused by skops, broken or foreign recipes, pipeline does not match the recipe, no code and no Markdown from names, formula and configuration limits, CSV injection, failed save |
| `tests/test_audit6_regressions.py` | audit 6: every entry of `preprocessing.json` that steps 9 and 10 use is checked when loading (also threshold, class selection, metrics, column actions, vectorizer, size), empty labels, short texts in step 2, header row detection, confusion matrix with many classes, code of the example texts, naive prediction from the training data |
| `tests/test_review_regressions.py` | findings of the review before delivery: class with only one training text (note in step 4, clear message for LinearSVC, comparison without LinearSVC – also in the generated code), validation curve with only one class in the training part, verdict "no better than naive" with an unsuitable alpha, CSV for Excel in steps 8 and 10, class names shown literally, Polish model names |
| `tests/test_app.py` | click test of the interface: Yelp as in the course (all 10 steps), SMS file without a header row with LinearSVC, error cases and protection against overwriting, no text column, export/import of the configuration |
| `tests/test_charts.py` | charts of the NLP version (text length, most frequent words, comparison, curve, most important words, contribution) also with unusual data |
| `tests/test_i18n.py` | languages: every text translated with the same placeholders, Excel CSV per language (app = code), generated code computes the same in every language, the whole app in English and Polish without German leftovers |

## Notes

* Load very large CSVs via "Path on this computer" rather than uploading them.
* Do not click anything during training – every input restarts the script and stops the training.
* **Run time:** Naive Bayes is always fast. Logistic regression and LinearSVC take much longer with a large vocabulary
  and raw word counts (measured with 14,000 synthetic texts and 28,754 words on 2 cores: curve + training about 2
  minutes and 1 minute; LinearSVC with TF-IDF and word pairs 15 seconds). Faster: TF-IDF, `min_df`/`max_features` in
  step 4, switching off the validation curve or using the sample.
* **Bag of words knows no word order:** "not good" and "good, not bad" look almost the same (word pairs help a
  little). Irony, sarcasm and mixed reviews remain hard – step 8 shows these mistakes. Meaning beyond single words
  needs deep learning (embeddings, transformers) – not part of this app.
* A model only knows the language and topics of its training texts; step 10 reports when new texts contain many
  unknown words.
* **Use locally only.** The path field reads every file the logged-in user can access, and `models/` is shared by
  all users of the app. So do not put the app on the internet without a login. Load models only from trusted
  sources (a ZIP may be at most 2 GB unpacked). `.skops` is loaded without executing code; `.joblib` is pickle and is
  loaded from an uploaded ZIP only after explicit confirmation. Only text models of this app are loaded – the app
  refuses models of the other versions with a clear message, as well as an incomplete or damaged
  `preprocessing.json` and a pipeline that does not match the recipe.
* The saved model contains the **vocabulary** (all words of the training texts) and – unless turned off – checksums
  of training texts. With confidential texts, share the model only with authorized people.
