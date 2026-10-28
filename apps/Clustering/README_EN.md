# DATA-Analysis-Clustering (Streamlit + scikit-learn: K-Means, Isolation Forest, LOF, DBSCAN)

Streamlit app for tabular CSV data: load, prepare, visualise and split into groups (clusters) with **K-Means** –
**unsupervised learning**: there is no label, K-Means itself looks for rows that are similar. Afterwards the app
describes the clusters (profiles, 2D map), compares them – if available – with a known grouping and assigns new data
to the clusters. For every step the app shows the matching Python code.

**Anomaly detection:** instead of looking for groups, the app finds the *most unusual rows* with **Isolation
Forest**, **Local Outlier Factor (LOF)** or **DBSCAN** (noise = anomaly) – with a score, a threshold, the features
that make a row stand out and the comparison with a known grouping (e.g. *Class* in fraud data). New data are scored
with the same threshold. For crypto data, `tools/bybit_features.py` builds a ready-to-use CSV from public Bybit market
data (see [Bybit → CSV](#bybit--csv-toolsbybit_featurespy)).

The model is the K-Means project from the course: **College_Data** – K-Means with K = 2 on all numeric columns; the
column *Private* (Yes/No) is ignored for this and only used afterwards for comparison. There is a ready-made example
configuration for this data set.

The app grew out of the supervised version (DATA-Analysis-SupervisedLearning): steps 1–3 (loading, column actions,
new columns, charts), the recipe (`preprocessing.json`), saving/loading, configurations and languages are unchanged;
new are step 4 without a label and without a split, K-Means in step 6, choosing K in step 7, the cluster evaluation in
step 8 and assigning new data in step 10 – later the anomaly models were added (steps 6–10 adapt to the chosen model).

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

## Languages

The language can be switched at any time at the top of the sidebar: **Deutsch · English · Polski**. Only the texts
change – the calculations are identical in all three languages (German is the original).

| Start directly in a language | |
|---|---|
| Command line | `streamlit run app.py -- --lang en` (or `pl`, `de`) – the `--` before `--lang` is required |
| Browser address | `http://localhost:8501/?lang=pl` |
| Environment variable | `CL_TEMPLATE_LANG=de` (Windows: `set CL_TEMPLATE_LANG=de`) |

Without one of these the app starts in English. A language chosen in the sidebar is then also put into the address
(`?lang=…`), so it is kept when the page is reloaded.

Translated are all texts of the interface, chart titles and axes, table headers, the comments and example file names
in the generated Python code, the conclusion in step 10, the names of the new columns in the export (`Cluster`,
`Cluster name`, `Distance`) and the file names of the downloads. The same in all languages:

* the internal values – actions and strategies are stored in configurations (`.json`) and in `preprocessing.json`
  with their German names; configurations and models therefore work in all three languages;
* the values the app writes into the data: `Sonstige` (top N), `fehlend` (own category for gaps), `_jahr` /
  `_monat` (date → year + month); for "Map values" `sonst=…` and `rest=…` work in every language;
* the comments in the source code (German).

If the language is switched in the middle of the work, messages that already exist (e.g. the log of step 2 or 4)
stay in the old language until the step is run again. Texts of Streamlit itself (e.g. "Upload" in the file field or
the ⋮ menu) are English in every version.

Excel format in step 5 per language: German and Polish `;` and decimal comma, English `,` and decimal point – always
UTF-8 with BOM and with protection against CSV injection (also in the generated Python code).

## Quick start – as in the course

1. **Step 1:** load `College_Data` (CSV, first column = name of the college)
2. **Step 2:** *Load example configuration* → *Apply* (keep the name, Grad.Rate 118 → 100)
3. **Step 3:** e.g. scatter plot Grad.Rate vs. Room.Board with colour = Private, histogram per group
4. **Step 4:** *Prepare data for the model* (17 features, Private only for comparison, StandardScaler)
5. **Step 6:** *Start training* (K = 2) → step 7 (choosing K), 8 (profiles, comparison with Private), 9, 10

For your own data: choose the features in step 4, optionally a known grouping for comparison and a column with
names; choose K in step 6 – step 7 shows which K fits. On a slow computer activate the **sample** on the left
(`df.sample(frac=0.1)`).

## Quick start – anomalies in crypto data (Bybit)

```bash
python tools/bybit_features.py --top 20 --interval 1h --days 120 --split 2026-10-01
```

The script fetches public market data (no API key) and writes `bybit_series_linear_60_train.csv` (before
1 October), `bybit_series_linear_60_new.csv` (from then on) and `bybit_series_linear_60_columns.csv` (column
descriptions).

1. **Step 1:** load `…_train.csv` (optionally `…_columns.csv` as column descriptions)
2. **Step 2:** the app recognises the Bybit data → *Load example configuration* → *Apply*
3. **Step 4:** *Prepare data for the model* (11 metrics, `row` = symbol + time as the row name)
4. **Step 6:** Isolation Forest with 1 % anomalous rows is preset → *Start training*
5. **Step 7:** does the threshold fit (knee of the score curve)? · **Step 8:** the most anomalous hours and what
   makes them stand out (e.g. `turnover_z +9.1 σ · ret_z −7.4 σ`)
6. **Step 9:** save · **Step 10:** score `…_new.csv` with the same model and the same threshold

## The steps

| Page | Content |
|---|---|
| 1 · Data & libraries | load a CSV (upload or path; separator `;` and decimal comma are detected automatically), overview, library versions, optional column descriptions |
| 2 · Preparation 1 | one action per column (keep, drop, extract number, date → year / month / year + month, first/last N characters, replace values, top N, **clip values (min, max)** …) · new columns: map values, formula, condition → 0/1, copy + action |
| 3 · Analysis & visualisation | choice of columns · overview, count plot, histogram, box plot, scatter, heat map, correlation with one column, pair plot, share per category, group statistics, joint plot, 2D density, lmplot, histogram per group |
| 4 · Preparation 2 – clustering data | **features**, optionally a **known grouping** (only for comparison, e.g. Private) and a **row name** (e.g. the college name); missing values per column; dummies and scaling (default: StandardScaler, drop_first off) – learned from all rows, no split; notes on strongly correlated columns and columns without variation. A comparison group is only suggested for a column named like one (Private, species, label, segment …); an ID column (e.g. CustomerID = 1, 2, 3 …) is suggested as the row name, not as a feature – if chosen as a feature, the app warns |
| 5 · Save data | data after step 2, clustering data after step 4 (unscaled or scaled) or – after training – data with **cluster number, cluster name and distance** to the centre (K-Means) or with **anomaly (1 = anomalous) and anomaly score** (DBSCAN: also the cluster); international or for Excel |
| 6 · Model & training | **choose the method** · **K-Means**: K (n_clusters), init (k-means++/random), n_init, max_iter, algorithm, seed; optionally the elbow and silhouette curve for step 7 (the curves grow during training) · **Isolation Forest**: n_estimators, max_samples, seed, contamination (auto or a fixed share) · **LOF**: n_neighbors, metric, contamination · **DBSCAN**: eps (starting value from the k-distance), min_samples, metric, noise share per eps for step 7 – the matching scikit-learn code is shown live |
| 7 · Choosing K or the threshold | K-Means: **elbow method** (inertia by K) and **silhouette** by K with assessment and suggestion (*Use K = …*), Davies-Bouldin, Calinski-Harabasz, explained variance (R²) and the **silhouette plot** · Isolation Forest/LOF: **distribution of the score** with the threshold, **score curve** of the most anomalous rows with its knee (*Use contamination = …*), threshold per share · DBSCAN: **k-distance** with its knee (*Use eps = …*), noise share and number of clusters per eps |
| 8 · Evaluation | K-Means: cluster sizes, **naming the clusters**, **profiles** (means in original units, heat map of the deviation from the average in σ, a text per cluster), **separating power per feature (R²)**, **2D map (PCA)** with colour = cluster or known grouping, two features against each other, **comparison with the known grouping** (cross table, mapping cluster → group, accuracy, naive prediction, ARI, NMI, confusion matrix, classification_report – also the comparison as in the course), **typical members** of each cluster, single case · anomaly models: **the most anomalous rows** with the score and the features that deviate most from the average (z), **profile** anomalous vs. normal, **PCA picture** (anomalies always included), DBSCAN clusters, **comparison with the known grouping** (share anomalous per group; with two groups precision, recall, ROC-AUC, average precision), single case |
| 9 · Save & load model | `models/<name>/model.skops` (safe, default) or `model.joblib` + `preprocessing.json` (recipe, settings, cluster names or threshold and share of anomalous rows, metrics, mapping to the comparison group), ZIP download, the whole pipeline as a Python script |
| 10 · Assign / score new data | new CSV or form → same preparation → K-Means: **nearest cluster centre** (`kmeans.predict`), distance, notes, distribution compared with training · anomaly models: **anomaly and score with the threshold from training** (DBSCAN: cluster of the nearest core point or noise), share of anomalous rows compared with training · comparison with the known grouping (if included), CSV download and **conclusion** |

Every step has a field **"Python code for this step"** – so everything can be copied 1:1 into a Jupyter notebook; the
code computes exactly what the app computes (same clusters, same inertia, same curves). Settings (incl. new columns)
can be saved as JSON on the left under **Configuration** and loaded again for new data with the same columns. On
loading, the file is checked: unusable entries (wrong type, outside the allowed range, e.g. K = 1000) are ignored and
named in the message. Configurations of the supervised and the deep-learning version can be loaded: step 2, missing
values, features and scaling are applied, label and model settings are silently skipped.

## K-Means in this app

**Unsupervised:** K-Means is never given an answer. It chooses K starting centres, assigns every row to the nearest
centre (Euclidean distance in the scaled features), moves every centre to the mean of its rows and repeats this until
nothing changes any more. All of this `n_init` times with other starting centres – the solution with the smallest
**inertia** (sum of the squared distances to the own centre) wins.

* **No train/test split.** As in the course (`kmeans.fit(df.drop('Private', axis=1))`), K-Means learns from all rows.
  Fill values, categories and the scaler therefore also come from all rows and apply unchanged to new data. There is
  no answer that would have to be hidden from the model – a data leak as in supervised learning does not exist here.
  Exception: a feature from the same CSV column as the comparison group would make the comparison in step 8 look
  better than it is – step 4 warns about it.
* **Scaling is essential.** K-Means works with distances: without scaling, columns with large numbers (Apps,
  F.Undergrad) almost alone decide the clusters. The default is therefore `StandardScaler` (mean 0, standard
  deviation 1 per column). The course did not scale – for a direct comparison choose "none" in step 4.
* **Dummies without drop_first.** For distances it is better to keep all categories (symmetric); with drop_first the
  dropped category would be closer to all the others.
* **n_init = 10** explicitly: in scikit-learn ≥ 1.4 the default is `n_init='auto'` – with k-means++ that is a
  single start only.
* **Outliers and impossible values** shift the centres: correct them in step 2 with *clip values* (in the course:
  Grad.Rate 118 → 100, here with the parameter `,100`).

### Choosing K (step 7)

| Metric | Meaning | good |
|---|---|---|
| Inertia (elbow method) | sum of the squared distances to the own centre; falls with every additional cluster | the **bend**: the K after which the inertia falls much more slowly (the app looks for the point where the decrease per K becomes at least 2.5 times smaller) |
| Silhouette (−1 … 1) | how much closer every row is to its own cluster than to the next other one | higher; > 0.7 strong, > 0.5 reasonable, > 0.25 weak, below that hardly any structure |
| Davies-Bouldin | ratio of the spread within clusters to the distance between clusters | lower |
| Calinski-Harabasz | variance between the clusters / variance within the clusters | higher |
| Explained variance (R²) | 1 − inertia / total variance | higher (but it always rises with K) |

If elbow and silhouette point to the same K, the app suggests it; otherwise both. If the silhouette stays low for
every K, the app makes no silhouette suggestion. What counts in the end is whether the clusters can be described
sensibly in step 8. From 20,000 rows the curve is computed on a fixed sample, the silhouette from 5,000 rows (like
`silhouette_score(..., sample_size=5000)`); the model itself always uses all rows.

### Describing and comparing clusters (step 8)

* **Cluster numbers are arbitrary.** K-Means does not know that cluster 1 is supposed to mean "Yes". The app therefore
  shows the comparison as in the course (cluster number = group number, 22 % in the course) and the comparison after
  the **mapping**: every cluster gets the group that is most frequent in it (in the course this corresponds to 78 %).
  In addition the naive prediction (always the most frequent group), **ARI** and **NMI** – both independent of the
  numbers.
* **Profiles:** the mean of every feature per cluster in original units and the deviation from the average in
  standard deviations (z) – this gives the texts "above/below average" and good names for the clusters.
  **Separating power (R²)**: which share of a feature's variance lies between the clusters.
* **PCA only for the picture:** the 2D map projects the data onto the two directions with the largest variance;
  K-Means keeps working with all columns.
* **Names** of the clusters (e.g. "small & private") are saved with the model and used in step 10 and in the export.

## Anomaly detection: Isolation Forest, LOF, DBSCAN

Same steps 1–5 as for K-Means, another **method** in step 6. All three are unsupervised: they learn what the rows
*normally* look like and give every row an **anomaly score** – higher = more anomalous. A row is anomalous
(`Anomaly = 1`) if its score is above the **threshold**.

| Method | Idea | Score | Threshold | good for | careful |
|---|---|---|---|---|---|
| **Isolation Forest** | random trees split the data at random points – unusual rows are alone after a few splits | `−score_samples`, 0 … 1 (around 0.5 normal) | contamination = auto → 0.5, otherwise the quantile for the chosen share | the standard for tables; fast, also with many rows and features; needs no scaling | finds global rather than local outliers |
| **Local Outlier Factor (LOF)** | density around a row compared with the density around its `n_neighbors` neighbours | LOF ≈ 1 normal, clearly above 1 anomalous | auto → 1.5, otherwise the quantile | local outliers next to dense groups | if the outliers form a small group of their own, `n_neighbors` must be larger than that group; slower with very many rows; scale the data |
| **DBSCAN** | core point = at least `min_samples` rows within the radius `eps`; whatever belongs to no cluster is **noise** | distance to the nearest core point in multiples of eps | 1 (= eps): noise ⇔ score above 1 | clusters of any shape and outliers in one step; no K needed | `eps` depends on the scaling and the number of features – step 7 (k-distance) shows which fits |

* **contamination is a decision, not a truth:** how many rows should be reported? Better a few clear anomalies
  (0.5–2 %) than a long list. Step 7 shows the **knee of the score curve** – after it the rows hardly differ any more –
  and offers it as a suggestion.
* **What makes a row anomalous?** Step 8 shows for every row the features with the largest deviation from the
  average (z, in standard deviations) and the profile *anomalous vs. normal* – a look into the data, not into the
  model.
* **Comparison with a known grouping** (e.g. *Class* in fraud data): the model never saw it. With two groups the
  rarer one counts as "positive": **precision** (share of real cases among the flagged rows), **recall** (share of the
  real cases that are found); **ROC-AUC** and **average precision** rate the score independently of the threshold
  (chance: 0.5 or the share of the positive group). In step 10 the positive group stays the one from training – even
  if the other one is rarer in the new data (e.g. a sample with many fraud cases).
* **eps (DBSCAN):** the start value in step 6 comes from the k-distance of a sample and is recomputed after a new
  preparation (e.g. another scaler), another `min_samples` or `metric` – so is a suggestion applied in step 7; an eps
  typed in by hand stays. With many identical rows (e.g. only categorical features) the k-distance is often 0; step 7
  then suggests the smallest eps (1e-6): identical rows form the clusters, rare combinations are noise.
* **Do not clip what you want to monitor.** *Clip values* in step 2 cuts off exactly the extreme values that make an
  anomaly – for anomaly detection only correct impossible values.
* **Scaling:** LOF and DBSCAN work with distances → StandardScaler in step 4. Isolation Forest splits one feature at
  a time and needs no scaling (it does no harm, though).
* **Time series without looking ahead:** compute the metrics of every row from the past only (rolling windows,
  z-scores against *earlier* values – as in the Bybit script), train on an earlier period and score a later one in
  step 10. If the anomalies pile up in one period, that is more likely a new regime than many single cases → retrain
  with more recent data.
* **New data (step 10):** same preparation and same threshold as in training. LOF is trained with `novelty=True`
  (otherwise it could not score new data); with DBSCAN a new row gets the cluster of the nearest core point if it is
  at most eps away – otherwise −1 (noise = anomaly).
* As with K-Means, the Python code shown computes exactly what the app computes (score, anomalies, threshold, the
  curves for step 7, profiles, export).

## Bybit → CSV (tools/bybit_features.py)

Builds a CSV with metrics for anomaly detection from public market data of the Bybit API v5 – no API key, only the
standard library and pandas/numpy (the app needs both anyway). The app recognises the file by its columns and offers
the matching example configuration in step 2.

```bash
python tools/bybit_features.py --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1h --days 120
python tools/bybit_features.py --top 20 --interval 4h --days 365 --split 2026-09-01
python tools/bybit_features.py --mode cross --top 100 --interval 1h --days 30
python tools/bybit_features.py --help
```

**Time series** (default, `--mode series`): one row per symbol and closed candle – "Which hours/days were unusual?"

| Column | Meaning |
|---|---|
| `row`, `symbol`, `time` | row name (symbol + candle start in UTC), symbol, candle start – not features |
| `ret_pct`, `ret_z` | return of the candle in %; as a z-score against the previous candles (log returns) |
| `range_pct`, `range_z` | high–low range in % of the open; as a z-score |
| `turnover_z` | traded value in USD as a z-score – volume shock |
| `vola_regime` | log2(volatility in the last window / volatility before) – 0 normal, +1 twice as high |
| `er` | efficiency ratio in the window: net move / sum of all moves – 0 sideways, 1 straight trend |
| `cs_spread_pct` | estimated bid-ask spread in % (Corwin-Schultz from high/low), mean over the window – liquidity stress |
| `corr_btc`, `excess_z` | correlation with the benchmark (default BTC) in the window; return minus benchmark return as a z-score – a coin's own move |
| `funding_bp`, `funding_z` | funding rate known at candle close in basis points; as a z-score (perpetuals only) |
| `oi_chg_pct`, `oi_z` | change of the open interest in % over one candle length that ends one open interest interval before the candle close (1h candles: 15 min); as a z-score (perpetuals only, candles of 5 minutes or longer) |

`ret_pct`, `range_pct` and `oi_chg_pct` are for reading; the example configuration uses the other 11 columns as
features (Isolation Forest, 1 % anomalous). Tip: choose `symbol` as the known grouping in step 4 – step 8 then shows
which symbols have the most anomalous candles.

**Cross-section** (`--mode cross`): one row per symbol over the whole period – "Which tokens behave differently from
the rest?" Columns: `ret_window_pct`, `vola_ann_pct` (annualised), `er`, `cs_spread_pct`, `max_dd_pct` (largest
drop), `corr_btc`, `beta_btc`, `vola_trend` (volatility of the second vs. the first half), `turnover_log10` (mean
daily turnover), `ret_skew`, `ret_kurt` (fat tails), `funding_mean_bp`, `oi_chg_pct` – the example configuration
flags 5 % of the tokens.

* **No look-ahead:** every metric of a candle uses only data up to its close – z-scores compare with the *previous*
  `--baseline` candles (default 30 days), windows (`--window`, default 1 day) end at the candle, funding and open
  interest are the values known at candle close, the candle that is still running is left out. The script fetches
  the warm-up for the z-scores in addition, before the start. A test checks it: cut off the data → earlier rows stay
  bit for bit the same.
* **Open interest, carefully:** Bybit does not document whether the timestamp of an open interest value means the
  start of its interval or the time of measurement. The script therefore assumes that a value is known only one
  interval after its timestamp and fetches open interest on a fine grid (at most ¼ of the candle, at least 5 min) –
  so the change still covers almost the whole candle. With 1- and 3-minute candles open interest is left out (it
  exists only every 5 minutes).
* `--split DATE` writes `<name>_train.csv` (before) and `<name>_new.csv` (from then on): train on the first part,
  score the second in step 10 – as in live operation.
* `<name>_columns.csv` describes the columns – in step 1 under *Optional: load column descriptions*.
* Symbols: `--symbols` or `--top N` (largest 24h turnover, without stablecoins and dated futures); `--category`
  linear (USDT perpetuals, default), inverse or spot (without funding and open interest); `--interval` 1 3 5 15 30 60
  120 240 360 720 D W or 15m, 1h, 4h, 1d, 1w; also `--start`/`--end`, `--no-funding`, `--no-oi`, `--benchmark`,
  `--keep-ohlcv`.
* **API limits:** at most 600 requests per 5 seconds and IP – the script waits between requests (`--pause`) and
  retries when Bybit is overloaded. If Bybit answers **HTTP 403**, the IP limit was exceeded or access from this
  network/region is blocked – wait at least 10 minutes. `--cache FOLDER` (e.g. `bybit_cache`, listed in `.gitignore`)
  keeps the answers; with a fixed `--start`/`--end` a run can then be repeated without network using `--offline`.
* If a symbol has too little history (listed recently), its first rows are dropped; a column with more than 20 % gaps
  (e.g. open interest without history) is left out – the script says so.

## New columns and actions (step 2)

| Kind | Example | Result |
|---|---|---|
| Map values | from `Private`: `Yes=1; No=0` | new column (1/0); values not named stay empty or `sonst=…` |
| Formula | `Accept / Apps` | a number per row; allowed are `+ - * / ** // %`, brackets, `abs log log10 sqrt exp round` |
| Condition → 0/1 | `Grad.Rate > 80` (names with a dot in \`backticks\`) | 1 = true, 0 = false, empty for gaps |
| Copy + action | `Grad.Rate` + "clip values" | the original stays, the result as an additional column |
| Action *clip values* | parameter `,100` · `0,100` · `10,` · with decimal comma `0,5;99,5` | values below min / above max are set to the limit (`clip`) |

Formulas are checked: only arithmetic, comparisons and the functions named – no arbitrary Python code; limits for
length, nesting and exponents prevent a formula from blocking the program. New columns belong to the saved recipe
and are computed automatically for new data in step 10.

## Step 10: assigning new data

* **Nothing is fitted again.** New data run through the saved *recipe* (`preprocessing.json`): same column actions
  and new columns, fill values, dummy columns in the same order, same scaler. Then every row goes to the **nearest
  centre** – the centres stay as they are.
* **Same raw columns as in training** – the app shows the required columns and offers a CSV template. Missing columns
  → error message; additional columns are ignored.
* **Notes per row:** gaps filled · unknown category · outside training range · **far from all clusters** (further
  from the nearest centre than 99 % of the training rows of that cluster – an outlier or a new type) · **between two
  clusters** (distance to the nearest centre ≥ 95 % of the distance to the second nearest).
* **Conclusion:** data quality, distribution across the clusters compared with training, share of outliers and close
  calls; if the new data contain the known grouping, the cluster → group mapping **from training** is used for the
  comparison (not determined again).
* If the new data already have columns with the name of a result column (e.g. `Cluster`, `Note` from an earlier
  export), they are replaced in the result – the app names them.

Without the app:

```python
import pipeline as pl
kmeans, recipe = pl.load_bundle("models/my_model")
df_new, _ = pl.read_csv("new_data.csv")
result, report, comparison = pl.assign_new(kmeans, recipe, df_new)
conclusion = pl.assess_assignment(recipe, result, report, comparison)
```

Anomaly models (Isolation Forest, LOF, DBSCAN) score instead of assigning – with the threshold from training:

```python
model, recipe = pl.load_bundle("models/my_anomaly_model")
result, report, comparison = pl.score_new(model, recipe, df_new)   # Anomalie, Anomalie-Score, Hinweis
conclusion = pl.assess_scoring(recipe, result, report, comparison)
```

## Notes on current library versions

| older code | problem today | in the template |
|---|---|---|
| `KMeans(n_clusters=2)` | scikit-learn ≥ 1.4: `n_init='auto'` = 1 start with k-means++ – the result depends more on chance | `KMeans(n_clusters=2, n_init=10, random_state=101)` |
| `df['Grad.Rate']['Cazenovia College'] = 100` | pandas: SettingWithCopyWarning (chained assignment), has no effect in pandas 3 | action *clip values* `,100` or `df['Grad.Rate'].clip(upper=100)` |
| `sns.lmplot('Room.Board', 'Grad.Rate', data=df, …)` | seaborn ≥ 0.12: only `data` may be passed without a name | `sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, …)` |
| `sns.FacetGrid(df, hue='Private', size=6)` | `size` is called `height` since seaborn 0.9 | histogram per group (`hist` with shared bins) |
| `pickle.dump(model, ...)` / `joblib.dump(...)` | loading can run arbitrary code from the file | `.skops` as default; `.joblib` only for trusted files |
| `df.corr()` | pandas ≥ 2: error with text columns | `df.corr(numeric_only=True)` |
| `pd.get_dummies(...)` | pandas ≥ 2: True/False instead of 0/1 | `dtype=float` |
| `LocalOutlierFactor().fit_predict(X)`, later `lof.predict(new)` | without `novelty=True` there is no `predict` for new data (and with `novelty=True` no `fit_predict`) | `LocalOutlierFactor(novelty=True)`; the score of the training rows from `negative_outlier_factor_` |
| `IsolationForest(contamination=0.1, behaviour='new')` from tutorials | `behaviour` no longer exists since scikit-learn 0.24; 10 % "anomalies" are almost always too many | `contamination='auto'` (threshold 0.5) or a deliberately small share – step 7 shows the knee |

## Files

```
app.py                  Streamlit interface (UI only)
pipeline.py             computation – also usable in a notebook (import pipeline as pl)
plots.py                charts (matplotlib/seaborn), return figures
codegen.py              generates the Python code shown or exported
i18n.py                 languages: tx() translates a German text into the current language
locales/                translations {German: (English, Polish)}, one file per module
configs/                example configurations: College_Data as in the course, Bybit time series, Bybit cross-section
.streamlit/config.toml  upload limit 1 GB, light theme
tools/bybit_features.py Bybit market data → CSV with metrics for anomaly detection
tests/                  pytest tests + synthetic test data (college_like, mall_like, anomaly_like, FakeBybit …)
AUDIT.md                report (Polish): structure, checks, limits
README.md               German · README_EN.md English · README_PL.md Polish
models/                 created when saving (step 9)
exports/                default folder for "Save to folder" (step 5)
```

Extending:
* new column action → `ACTIONS` + `_transform_column()` in `pipeline.py` (+ `_action_lines()` in `codegen.py`)
* new chart → function in `plots.py` + entry in `page_viz()` or `page_profiles()` in `app.py`
* another method → in `pipeline.py` `MODEL_NAMES`, `MODEL_CLASSES`, `MODEL_VARS`, `MODEL_WHY` and the fields in
  `ClusterConfig`. An anomaly method also in `ANOMALY_MODELS`, `anomaly_params()`, `anomaly_scores()` (score: higher =
  more anomalous) and `anomaly_threshold()` – training, steps 7–10 and saving then run through `train_anomaly()`,
  `anomaly_curve()` and `score_new()`; if the method needs a lot of memory, a limit in `size_problem()`; new types for
  safe loading in `SKOPS_TRUSTED`. In `app.py` the parameters in `anomaly_params_ui()` and the explanation in
  `HOW_ANOMALY`, in `codegen.py` `anomaly_model_code()` and `predict_code()`; `tests/test_anomaly.py` checks that the
  code computes the same as the app. A clustering method like K-Means: use `make_kmeans()`/`train_kmeans()` and
  `model_code()` as a template.
* new or changed text → write it in German in `tx("…")` (constants: `N_("…")`) and add the English and Polish
  translation to the matching file in `locales/`; `pytest tests/test_i18n.py` reports missing translations.

## Tests

```bash
pip install pytest
pytest -q                                  # 362 tests, approx. 5 minutes
python tests/synthetic.py                  # synthetic college data (777 rows, not real data)
```

| File | checks |
|---|---|
| `tests/test_clustering.py` | K-Means as in the course (= scikit-learn directly), reproducibility, elbow and silhouette (clear groups are found, no suggestion without structure, samples for large data), assessment (small/dominating clusters, not converged, duplicate rows), profiles in original units, separating power = explained variance, PCA, comparison with known groups, outliers and close calls in new data |
| `tests/test_codegen_parity.py` | the Python code shown computes **exactly** the same as the app: X, clusters, inertia, elbow/silhouette curve, profiles, comparison, export – also in edge cases (thousands separators, dates as numbers, inf, group fill values, clip values, new columns) |
| `tests/test_anomaly.py` | Isolation Forest, LOF, DBSCAN: score ⇔ threshold, contamination, novelty, noise = anomaly, eps too small, curves and suggestions for step 7, most anomalous rows, profile, PCA, saving/loading (.skops/.joblib), new data, no mix-up with K-Means, damaged recipes – and the code shown computes **exactly** the same (score, anomalies, threshold, k-distance, eps grid, profiles, export, step 10); the exported script runs; charts in edge cases |
| `tests/test_bybit_features.py` | the Bybit script with a simulated API (no network): paging within the API limits, retries/403/cache, no look-ahead, the running candle is left out, metrics recomputed, cross-section, spot, unknown symbols, the example configuration fits the CSV, the built-in crash is the most anomalous row |
| `tests/test_pipeline.py` | the whole pipeline with LendingClub-like data, saving/loading (.skops/.joblib, ZIP), new data (unknown categories, outliers, missing columns), the exported script runs |
| `tests/test_new_features.py` | year + month, new columns, *clip values*, new columns for new data, saving data, conclusion |
| `tests/test_audit_regressions.py` … `test_audit4_regressions.py` | regression tests from the four audits of the base app, applied to clustering (numbers, dates, CSV detection, formula limits, configurations, ZIP protection, no code from names in the code shown, language switch, safe loading, only K-Means models) |
| `tests/test_audit5_regressions.py` | audit 5 (clustering app): export code = app even after removed rows/columns, no unasked comparison group, ID columns as row names, "as in the course" text with the real groups, no Markdown from foreign models, a damaged `preprocessing.json` is refused, empty clusters, note on changed settings |
| `tests/test_audit6_regressions.py` | audit 6 (anomaly detection, Bybit script): positive group in step 10 as in training, eps start value after a new preparation, eps suggestion never 0, memory protection for DBSCAN and LOF (training and eps table, code = app), identical rows, small data, eps line in the k-distance chart, open interest without look-ahead, fine open interest grid, no open interest with 1-minute candles |
| `tests/test_app.py` | click test of the interface (all 10 steps as in the course; semicolon CSV, changed settings, unusable values, overwrite protection; new columns, chart errors, export) |
| `tests/test_app_anomaly.py` | click test of the three anomaly models through steps 4–10 and of the Bybit CSV with its example configuration up to scoring new data |
| `tests/test_charts.py` | the charts of step 3 and the cluster charts (elbow/silhouette, silhouette plot, sizes, profiles, PCA, cross table) |
| `tests/test_i18n.py` | languages: every text translated with the same placeholders, Excel CSV per language (app = code), generated code computes the same in every language, the whole app in English and Polish without German leftovers – also with the anomaly models |

## Notes

* Very large CSVs are better loaded via "Path on this computer" than uploaded.
* Do not click anything during training – every input restarts the script and stops the training.
* K-Means is fast, but with very many rows and `n_init = 10` it takes time: 400,000 rows × 80 columns need a few
  seconds to about one minute per K (measured with 2 CPU cores – the less distinct the groups, the more iterations).
  The elbow curve therefore uses a sample from 20,000 rows; tip: a sample in the sidebar or a smaller `n_init`.
* **Use locally only.** The path field reads every file the logged-in user has access to, and `models/` is the same
  for all users of the app. So do not put the app on the internet without a login. Load models only from trusted
  sources (a ZIP may be at most 2 GB unpacked). `.skops` is loaded without running code; `.joblib` is pickle and is
  loaded from an uploaded ZIP only after explicit confirmation. Only models of this app (K-Means, Isolation Forest, LOF, DBSCAN) are loaded – the app
  refuses models of the supervised or deep-learning version with a clear message, as well as an incomplete or damaged
  `preprocessing.json` and a recipe that does not match the model.
* **K-Means has limits:** it tends to find round groups of similar size and assigns every row to a cluster – outliers
  too. For elongated or nested groups or clusters of very different sizes (see the silhouette plot), other methods
  (e.g. **DBSCAN** – available in step 6 – or hierarchical clustering) are better suited; outliers are found directly
  with Isolation Forest or LOF.
* LOF and DBSCAN compare every row with its neighbours – with very many rows (from about 50,000) this takes time and
  memory; Isolation Forest stays fast. The k-distance for step 7 uses a sample from 20,000 rows, the eps grid exists
  only up to 20,000 rows.
* **Memory protection:** DBSCAN (scikit-learn) keeps all neighbourhoods in memory at the same time – with too large
  an eps or many identical rows this grows with the square of the number of rows. The app estimates the number of
  neighbour pairs from 400 random rows and refuses training above 40 million pairs (about 1 GB) with a note; in the
  eps table of step 7 it skips such eps values. The same for LOF when rows × `n_neighbors` exceeds 40 million.
