# UK Road Collision Analytics (STATS19, 2005-2019)

**Official DfT road-collision data for Great Britain, cleaned and reconciled with published totals, then used for effect-size statistics, a backtested monthly forecast and a leakage-free severity model.**

Author: Maryam · Code: MIT · Data: Open Government Licence v3.0

---

## Key results

> **Status: TBD.** This version of the repository was built and tested without access to the
> DfT data files (see [PROGRESS.md](PROGRESS.md), "Mode"). Every result below is filled in
> only from an executed notebook cell. No number here is estimated.

| Track | Result | Source |
|---|---|---|
| Data quality | Pipeline vs RAS0101 collision counts, largest yearly difference: TBD | `notebooks/01_reconciliation.ipynb` |
| Statistics | Strongest KSI association (Cramér's V): TBD | `notebooks/02_eda_and_tests.ipynb` |
| Count model | Poisson dispersion and model used (Poisson / Negative Binomial): TBD | `notebooks/02_eda_and_tests.ipynb` |
| Forecasting | Test MAE 2017-2019, seasonal naive vs ETS vs SARIMA: TBD | `notebooks/03_forecasting.ipynb` |
| Forecasting | 2015 total forecast from data to 2014 vs official: TBD | `notebooks/03_forecasting.ipynb` |
| Severity model | Test PR-AUC 2017-2019 (LightGBM, logistic regression, base rate): TBD | `notebooks/04_severity_model.ipynb` |

## Dashboard

*Dashboard screenshot: TBD. Save it as `dashboard/screenshots/overview.png` and embed it here, after the Power BI report is built from `dashboard/data/` using
[`dashboard/measures.md`](dashboard/measures.md).*

## Data and licence

- **Source:** Department for Transport, [Road Safety Data (STATS19)](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data):
  the collision and casualty tables (1979 to latest year), filtered to **2005-2019**.
  2020 onwards is excluded from modelling, because COVID-19 travel changes and further
  reporting changes break comparability.
- **Official totals:** table [RAS0101](https://www.gov.uk/government/statistical-data-sets/reported-road-accidents-vehicles-and-casualties-tables-for-great-britain),
  used for reconciliation.
- **Code lists:** the DfT Road Safety Open Dataset data guide (`.xlsx`), used to decode every coded field.
- No data is committed. `data/raw/` and `data/processed/` are gitignored.
- Attribution: *Contains public sector information licensed under the Open Government Licence v3.0. Source: Department for Transport, Road Safety Data (STATS19).*
  See [DATA_LICENSE.md](DATA_LICENSE.md).

## How to run

### Local (Python 3.11)

```bash
git clone https://github.com/maryam843/uk-road-accident-analytics.git
cd uk-road-accident-analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
nbstripout --install --attributes .gitattributes   # keeps notebook diffs clean

python -m src.download    # several GB into data/raw/ (resumable), logs to DOWNLOAD_LOG.json
python -m src.clean       # chunked cleaning -> data/processed/*.parquet + dashboard/data/*.csv
jupyter lab notebooks/    # run 01 -> 04 in order

pytest -q && ruff check .
```

Set `DATA_DIR=/some/path` to keep the data outside the repo. Notebooks 01-04 have
`keep_output` set in their metadata, so their executed outputs survive the nbstripout filter
when committed.

### Google Colab (free tier)

Click a badge and run all cells. The badges point at the `main` branch. The first cell clones the repo and installs
`requirements.txt`; restart the runtime once if Colab asks. The second cell downloads and
cleans the data if `data/processed/` is empty. The CSVs are read in 500k-row chunks and
stored as Parquet, so the large collision file is never loaded into memory at once. While the repo is
private, Colab needs GitHub access (File → Open notebook → GitHub → include private repos).
To keep the data between sessions, mount Drive and set `os.environ["DATA_DIR"]` to a Drive
folder before the setup cell.

| Notebook | Open |
|---|---|
| 00 Kaggle data audit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/uk-road-accident-analytics/blob/main/notebooks/00_kaggle_data_audit.ipynb) |
| 01 Reconciliation with RAS0101 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/uk-road-accident-analytics/blob/main/notebooks/01_reconciliation.ipynb) |
| 02 EDA and statistical tests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/uk-road-accident-analytics/blob/main/notebooks/02_eda_and_tests.ipynb) |
| 03 Forecasting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/uk-road-accident-analytics/blob/main/notebooks/03_forecasting.ipynb) |
| 04 Severity model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maryam843/uk-road-accident-analytics/blob/main/notebooks/04_severity_model.ipynb) |

Example Colab cell to run the whole pipeline without a notebook:

```python
!git clone -q https://github.com/maryam843/uk-road-accident-analytics.git
%cd uk-road-accident-analytics
%pip install -q -r requirements.txt
!python -m src.download && python -m src.clean
```

## Project structure

```
├── src/
│   ├── config.py        paths (DATA_DIR), URLs, study window, split years, random_state=42
│   ├── download.py      streaming download with resume, size check, SHA-256, download log
│   ├── clean.py         chunked cleaning, code decoding, aggregates, dashboard CSVs, RAS0101 reader
│   ├── features.py      pre-event features, year splits, explicit ColumnTransformer
│   └── evaluate.py      forecast/classification metrics, effect sizes, backtest, plot style
├── notebooks/
│   ├── 00_kaggle_data_audit.ipynb   why the old Kaggle copy was rejected
│   ├── 01_reconciliation.ipynb      pipeline vs official RAS0101 totals
│   ├── 02_eda_and_tests.ipynb       EDA, effect sizes, Poisson / Negative Binomial count model
│   ├── 03_forecasting.ipynb         seasonal naive vs ETS vs SARIMA, rolling-origin backtest
│   ├── 04_severity_model.ipynb      KSI classifier, per-year test metrics, SHAP
│   └── 05-07_*.ipynb                planned work (markdown plans only)
├── tests/               pytest suite + synthetic fixture generator (tests/fixtures/)
├── dashboard/           measures.md (DAX + build guide), data/ (aggregated CSVs), screenshots/
├── reports/figures/     figures saved by the notebooks
├── docs/INTERVIEW_NOTES.md   every key decision, explained
└── data/                gitignored: raw/, processed/, kaggle/
```

## Methods

**Data pipeline.** The CSV header is read first and column names are normalised to
`collision_*`. The file is then read in 500k-row chunks and filtered to 2005-2019. Duplicate
collision indexes are dropped and counted. `-1`, "data missing or out of range" and
"unknown (self reported)" become NaN, never a category. Speed limits other than
20/30/40/50/60/70 are set to NaN. Bad coordinates are flagged, never imputed. Only casualties
whose collision survived cleaning are kept. Every step's count goes to `CLEAN_LOG.json`.

**Reconciliation.** Yearly collisions, fatal collisions, fatal-or-serious collisions
(unadjusted and adjusted), people killed and all casualties are compared with RAS0101. The
notebook asserts that collision counts match within 0.5% every year.

**Statistics.** Chi-square with Cramér's V, Welch t with Cohen's d, and ANOVA with
eta-squared. Effect sizes are interpreted rather than p-values, because n is in the millions.
Daily counts are modelled with a Poisson GLM (year + month + weekday as categorical). It
switches to Negative Binomial when Pearson χ²/df > 1.5. Results are incidence rate ratios
with 95% CIs.

**Forecasting.** Monthly collisions, trained on 2005-2016 and tested on 2017-2019.
Seasonal naive is the baseline, compared with ETS and with SARIMA (order chosen once by
`auto_arima` on the training data, then fixed). Rolling-origin backtest from 2012-12 every 12
months, horizon 12. A replica check forecasts the 2015 total from data to 2014. Ljung-Box and
ACF residual checks.

**Severity model.** Binary KSI vs slight at collision level, using pre-event features only.
Police attendance, casualty counts and every casualty field are excluded as leakage. Year
splits: train 2005-2015, validate 2016, test 2017-2019, asserted disjoint by collision index.
Models: dummy baselines, class-balanced logistic regression, and LightGBM with class
weighting, early stopping and a 25-configuration random search on 2016. Reported: PR-AUC,
ROC-AUC, balanced accuracy, macro-F1 and KSI recall/precision at a 2016-chosen threshold,
overall and per test year. SHAP explains the model; it does not establish causes.

## Limitations

- **Injury collisions only.** STATS19 covers collisions reported to the police in which
  someone was injured. Damage-only collisions are not included.
- **Under-reporting.** Many slight injuries never reach the police, so slight counts (and the
  denominator of every KSI share) are undercounts. Serious injuries are under-reported too,
  to a lesser degree.
- **Severity reporting change.** From 2016, injury-based reporting (CRASH/COPA) records more
  injuries as serious. KSI trend charts use DfT's adjusted figures and are labelled
  "adjusted for changes in police severity reporting (DfT)". The classifier's test years
  carry this known label shift, so metrics are shown per year.
- **Exposure is not modelled.** Counts are not divided by traffic volume, so "more collisions"
  does not mean "more dangerous per journey".
- **2020 onwards is excluded.**
- **Approximate regions.** Region comes from police force area, which does not match ONS
  regions exactly.

## Origins

Rebuilt and extended solo from two university projects: a solo data analysis and forecasting project and a two-person data mining project (teammate acknowledged with consent). All code in this repository is my own re-implementation.
