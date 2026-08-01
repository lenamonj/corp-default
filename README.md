<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/uv-managed-DE5FE9?style=for-the-badge&logo=uv&logoColor=white" alt="uv">
  <img src="https://img.shields.io/badge/XGBoost-Classifier-EC6C37?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/Data-Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/SHAP-Explainability-4B0082?style=for-the-badge" alt="SHAP">
  <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="MIT License">
</p>

<h1 align="center">Corporate Default Prediction</h1>

<p align="center">
  <strong>A bankruptcy screen for US public companies, built on 78,682 firm-year observations from NYSE/NASDAQ (1999-2018) and benchmarked against Altman Z (1968) and penalised logistic regression. Two-year forward hazard label, temporal split with a two-year embargo, cluster-bootstrap confidence intervals.</strong>
</p>

<p align="center">
  <em>No Bloomberg terminal. No paid data feeds. Just a free Kaggle dataset and scikit-learn.</em>
</p>

---

## Why This Exists

Predicting corporate default is one of the core problems in credit analysis. Most models are locked behind expensive platforms or proprietary datasets. This project demonstrates the full workflow using entirely free, public data - from raw accounting items to a benchmarked, interval-tested screening model with interpretable feature importance.

The model uses a **temporal split with a two-year embargo** - train on 1999-2006, validate on 2009-2011, test on 2014-2016, with the boundary years dropped because a two-year label window would otherwise straddle them.

This defends against look-ahead leakage. It does not make the company sets disjoint: 2,264 of the 3,854 test companies, 58.7%, also appear in training. The notebook quantifies what that is worth by rerunning the evaluation on firms absent from training entirely.

---

## What It Produces

A single Jupyter notebook that runs the complete pipeline:

| Stage | What Happens |
|-------|-------------|
| **Data Loading** | 78,682 firm-year observations, 8,971 companies, 18 accounting variables |
| **Label Construction** | Two-year discrete-time hazard label - does the firm make its final filing in `t+1` or `t+2`? Terminal-year rows of failed companies and rows whose outcome window exceeds 2018 are dropped, leaving 72,393 rows and 1,123 positives (1.551% base rate) |
| **EDA** | Class balance, feature distributions by outcome, correlation matrix |
| **Feature Engineering** | 14 credit ratios derived from raw accounting and market data, including the five Altman Z components (see table below) |
| **Temporal Split** | Train 1999-2006, embargo 2007-2008, Validate 2009-2011, embargo 2012-2013, Test 2014-2016 |
| **Baselines** | Altman Z-score (1968) and penalised logistic regression, scored through the same evaluation code as XGBoost |
| **XGBoost** | Gradient-boosted classifier tuned by randomized search on PR-AUC, with `scale_pos_weight` for class imbalance and a minimum-boosting-rounds floor on which configurations may be selected |
| **Evaluation** | ROC-AUC, PR-AUC, capture rate at a 1/5/10% review budget, cluster-bootstrap confidence intervals, paired bootstrap of pairwise model differences |
| **Sensitivity** | Unseen-company holdout check and a 1/2/3-year horizon comparison |
| **Interpretability** | SHAP summary and bar plots showing top 15 feature contributions |

---

## Results

Test set is 2014-2016 held out entirely: 9,954 firm-years containing 160 labelled failures, a 1.61% base rate.

| Model | ROC-AUC | ROC 95% CI | PR-AUC | PR 95% CI | Capture@5% | Lift@5% |
| --- | --- | --- | --- | --- | --- | --- |
| Altman Z (1968) | 0.7166 | [0.677, 0.753] | 0.0282 | [0.022, 0.036] | 5.6% | 1.12x |
| Logistic regression | 0.7316 | [0.687, 0.771] | 0.0441 | [0.028, 0.067] | 15.0% | 3.00x |
| XGBoost (tuned) | 0.7760 | [0.736, 0.817] | 0.0533 | [0.038, 0.077] | 18.8% | 3.75x |

PR-AUC here is the trapezoidal `auc(recall, precision)` rather than `average_precision_score`; the former runs slightly optimistic in level, and since it is applied identically to all three models, no comparison between them turns on the choice.

Confidence intervals are a cluster bootstrap over companies (1,000 resamples, 3,854 test companies), not rows - firm-years from the same company are correlated, so resampling rows would understate the uncertainty.

All three ROC-AUC intervals above overlap, and that fact says nothing. Two marginal intervals overlapping is not a test of equality when both estimates are computed on the same rows, because their errors move together. The notebook therefore bootstraps the *difference* between each pair of models over identical company resamples:

| Comparison | Metric | Difference | 95% CI | Excludes 0 |
| --- | --- | --- | --- | --- |
| Altman Z - Logistic regression | ROC-AUC | -0.0151 | [-0.0605, +0.0318] | no |
| Altman Z - Logistic regression | PR-AUC | -0.0159 | [-0.0370, -0.0018] | yes |
| Altman Z - XGBoost | ROC-AUC | -0.0594 | [-0.0984, -0.0219] | yes |
| Altman Z - XGBoost | PR-AUC | -0.0251 | [-0.0477, -0.0118] | yes |
| Logistic regression - XGBoost | ROC-AUC | -0.0444 | [-0.0792, -0.0113] | yes |
| Logistic regression - XGBoost | PR-AUC | -0.0092 | [-0.0320, +0.0145] | no |

XGBoost beats Altman Z on both metrics and beats logistic regression on ROC-AUC, with the interval clear of zero in all three cases. It does not separate from logistic regression on PR-AUC - the gap is 0.0092 and the interval spans zero, which at 160 test positives is unsurprising. Altman Z and logistic regression are likewise not separable on ROC-AUC. The claim the data supports is that gradient boosting buys a genuine improvement in global ranking over both comparators, and an improvement in precision-recall over the 1968 formula only.

The same result in the unit a credit analyst works in: capture rate at a fixed review budget.

| Budget | Names reviewed | Altman Z caught (lift) | Logistic caught (lift) | XGBoost caught (lift) |
| --- | --- | --- | --- | --- |
| 1% | 100 | 1 of 160 (0.62x) | 7 of 160 (4.35x) | 10 of 160 (6.22x) |
| 5% | 498 | 9 of 160 (1.12x) | 24 of 160 (3.00x) | 30 of 160 (3.75x) |
| 10% | 995 | 21 of 160 (1.31x) | 47 of 160 (2.94x) | 66 of 160 (4.13x) |

XGBoost leads at all three budgets. Altman inverts at 1% - a lift below 1.0x is worse than a random draw of the same size, meaning the 1968 formula's most distressed-looking names default less often than average. The widest absolute gap is at 10%: 66 defaults against logistic regression's 47 from the same 995 names, which puts 66 of the test period's 160 failures inside a tenth of the universe.

On companies absent from the training set (3,702 rows, 37.2% of test, 66 defaults, 1,590 companies) XGBoost keeps its lead rather than surrendering it: 0.739 against 0.672 for Altman Z and 0.672 for logistic regression. All three degrade off the full test set and XGBoost degrades least (-0.037, against -0.045 for Altman and -0.060 for logistic). This is consistent with the ranking advantage not being an artifact of having seen a firm's accounting fingerprint before, though 66 positives is thin enough that the ordering is indicative rather than measured.

Horizon sensitivity (train and test windows held fixed at 1999-2005 and 2014-2015, only the label window varies) shows ROC-AUC falling monotonically as the forecast reaches further out: 0.814 at one year, 0.784 at two, 0.740 at three. PR-AUC runs the other way, 0.0447 to 0.0559 to 0.0681, but that is mechanical rather than contradictory - the base rate rises from 0.90% to 1.56% to 2.30% across the three labels, and PR-AUC is not comparable across different base rates. Two years is a compromise between ranking accuracy and a lead time long enough to act on, not the winner on either metric.

These figures come from executing the notebook against the exact package set in `uv.lock`. Two consecutive executions produce identical metrics.

---

## Quick Start

This project is managed with [uv](https://docs.astral.sh/uv/). The only prerequisite is uv itself - you do **not** need to install Python first, because uv reads `.python-version` and fetches CPython 3.13 for you.

If you don't have uv:

```bash
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 1. Clone and set up

```bash
git clone https://github.com/lenamonj/corp-default.git
cd corp-default
uv run python -m ipykernel install --user --name corp-default --display-name "corp-default (uv)"
```

That single command does everything. `uv run` creates `.venv`, installs every package pinned in `uv.lock` (including JupyterLab, so you don't need Jupyter preinstalled), then registers a Jupyter kernel named `corp-default`. There is no separate `uv sync` step and nothing to activate.

The notebook is saved requesting a kernel named `corp-default`, which is why it selects itself once registered. Registration is a command rather than a committed file because a kernel stores an absolute path to *your* `.venv`, which is different on every machine.

### 2. Get the data

The `data/` folder ships empty. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset), unzip it, and place the CSV so the path is exactly:

```text
data/american_bankruptcy.csv
```

### 3. Run

Every `uv` command must be run **from the repo root** (the folder containing `pyproject.toml`):

```bash
cd corp-default
uv run jupyter lab corp_default.ipynb
```

JupyterLab picks the `corp-default` kernel automatically, because the notebook file requests it by name. Note the naming: the folder is hyphenated, the notebook is underscored.

> **Run it from the wrong directory and uv fails quietly.** With no `pyproject.toml` in the current folder or any parent, `uv run` has no project to attach to, so it falls back to whatever is on your `PATH` - a different Jupyter, a different Python, none of the locked packages. There's no error, just wrong versions. If something behaves oddly, check `pwd` first.

In **VS Code**, open the notebook and choose **corp-default (uv)** in the kernel picker at the top right. VS Code treats the notebook's kernel request as a hint rather than a command, so you select it once per workspace and it sticks after that.

> **If `corp-default` isn't in the VS Code dropdown:** the extension caches its kernel list at startup, so a kernel registered while VS Code was already open won't appear yet. Run **Developer: Reload Window** from the Command Palette (`Ctrl+Shift+P`). It will be there. Registering the kernel *before* opening VS Code avoids this entirely.

Run the first code cell before anything else. It prints the active interpreter, which should end in `corp-default\.venv\Scripts\python.exe`. If it doesn't, the wrong kernel is selected and every import error below it is a symptom of that, not a real dependency problem.

---

## Working With uv

| Command | What it does |
| ------- | ------------ |
| `uv sync` | Make `.venv` match `uv.lock` exactly - adds what's missing, removes what drifted in |
| `uv add <pkg>` | Add a dependency, update `pyproject.toml` and `uv.lock`, install it |
| `uv remove <pkg>` | The reverse |
| `uv run <cmd>` | Run a command inside the project env without activating anything |
| `uv lock --upgrade` | Re-resolve to newer versions, then commit the changed lock |

You never need to activate the venv. `uv run` handles it, and `uv sync` runs implicitly before each `uv run`.

`pyproject.toml` and `uv.lock` are both committed and both matter: the first records intent (loose floors), the second records the exact resolved graph. Reproducibility comes from the lock.

---

## Engineered Features

14 credit ratios computed from raw accounting and market data, five of which are the Altman Z-score's own components:

| Ratio | Formula | Measures |
| --- | --- | --- |
| `current_ratio` | Current Assets / Total Assets | Liquidity |
| `debt_to_assets` | Long-Term Debt / Total Assets | Leverage |
| `total_liabilities_to_assets` | Total Liabilities / Total Assets | Gearing |
| `net_margin` | Net Income / Net Sales | Profitability |
| `ebitda_margin` | EBITDA / Net Sales | Operating efficiency |
| `roa` | Net Income / Total Assets | Return on assets |
| `gross_margin` | Gross Profit / Net Sales | Pricing power |
| `asset_turnover` | Net Sales / Total Assets | Asset utilization |
| `debt_to_ebitda` | Long-Term Debt / EBITDA, blank if EBITDA <= 0 | Debt serviceability |
| `negative_ebitda` | 1 if EBITDA <= 0 | Distress flag |
| `working_capital_to_assets` | (Current Assets - Current Liabilities) / Total Assets | Altman: liquidity |
| `retained_earnings_to_assets` | Retained Earnings / Total Assets | Altman: cumulative profitability |
| `ebit_to_assets` | EBIT / Total Assets | Altman: operating return |
| `mve_to_liabilities` | Market Cap / Total Liabilities | Altman: market cushion |

`asset_turnover` is the Altman Z-score's fifth term; the last four rows above are its remaining components. `market_cap` and `mve_to_liabilities` are market observations rather than financial-statement items - see Limitations. These 14 ratios plus the 17 raw accounting and market items behind them make up the model's 31 features; identifiers and label-construction columns are excluded. The dataset ships 18 raw variables; `total_revenue` is dropped as an exact duplicate of `net_sales`, leaving 17.

---

## Design Decisions

- **Two-year discrete-time hazard label over the raw status flag** - `status_label` is constant per company, so used directly it encodes "this company fails at some point" rather than a firm-year event. The label used here asks, from fiscal year `t`, whether the firm makes its final filing in `t+1` or `t+2`.
- **Two-year embargo between splits** - the label reaches two years past the observation year, so a one-year gap between train and validation, or validation and test, would let outcome information leak across the boundary. Embargo years are dropped entirely rather than assigned to either side.
- **Preprocessing fit on train only** - winsorisation bounds and imputation medians are computed on the training split alone and applied unchanged to validation and test, rather than fit on the full panel.
- **Baselines over a bare AUC** - Altman Z (1968) and penalised logistic regression are scored through the same evaluation code as XGBoost, so any XGBoost number is read against a fifty-year-old formula and a linear model, not reported in isolation.
- **Capture rate over a probability threshold** - a fixed cutoff is the wrong unit for a credit screen. Analyst time is the binding constraint, so the question posed is how many defaults are caught at a given review budget (top 1%, 5%, 10% of names by score).
- **Cluster bootstrap over row bootstrap** - firm-years from the same company are correlated, so resampling individual rows understates uncertainty. Confidence intervals resample whole companies instead.
- **Paired differences over overlapping intervals** - two marginal intervals overlapping does not establish that two models are equivalent when both are estimated on the same rows. Every pairwise comparison bootstraps the difference over identical company resamples, so the shared sampling noise cancels.
- **A minimum boosting-round floor on model selection** - scoring each configuration at its own early-stopping optimum and then taking the best of those is a double maximum over a noisy statistic. On 157 validation positives it will find spikes, and it found one at round 1. `MIN_BOOST_ROUNDS = 10` excludes configurations that never boosted, as a constraint on model form decided ahead of any test-set result.

---

## Limitations

**The label is derived, not given.** `status_label` in the source data is constant per company - every row of a firm that eventually failed carries `failed`, including rows from fifteen years prior. Used directly it encodes lifetime outcome rather than a firm-year event, and the resulting firm-year default rate falls from 7.2% to 1.3% across the panel purely because failed firms exit. This notebook derives a two-year horizon label instead.

**Censoring is treated as survival.** Only 32% of companies labelled `alive` are still reporting in 2018. The rest are delisted, acquired, or stop filing, and the dataset does not say which. All are labelled 0. Since those exits are not independent of credit quality, this biases the model mildly toward optimism. The dataset does not permit a fix.

**Company sets overlap across splits.** 2,264 of the 3,854 test companies, 58.7%, appear in training. The hazard label is time-varying, so firm identity no longer gives away the answer as it did under the company-constant label, but the unseen-company check shows performance does fall on firms the model has not seen.

**One test positive in five cannot be confirmed.** `status_label` carries no date. A firm-year is positive when the firm is marked `failed` and its last row in the panel falls one or two years later, and "last row in the panel" is inferred from the absence of a subsequent one. For an interior year that inference is sound, because a later non-filing year is visible in the data. For a firm whose last row is 2018 it is not, because the panel simply ends there. 35 of the 160 test positives (21.9%) rest on a 2018 final filing, against none in train and none in validation - the asymmetry sits only in test. Its direction is conservative: any of those 35 that in fact kept filing after 2018 is a mislabelled positive, and label noise of that kind depresses measured performance rather than flattering it.

**Winsorisation compresses the later period.** Clipping bounds are fitted on 1999-2006, so exactly 1% of training rows sit at the upper bound by construction, but 3.4% of test rows do for `total_assets`, and similarly for other level variables. The panel's firms are larger in 2014-2016 than in 1999-2006 and everything above the training 99th percentile is flattened onto one value. Refitting the bounds on test would be leakage, so the compression is accepted; what it discards is dispersion among the largest test firms.

**Market data is in the feature set.** `market_cap` and `mve_to_liabilities` are market observations, not financial statement items. Equity value collapsing ahead of bankruptcy is a large part of the signal, which makes this a hybrid market/fundamental model rather than a pure accounting one.

**Bankruptcy is not default.** The label is a Chapter 11 or Chapter 7 filing by a listed US company. It is not a missed coupon, a distressed exchange, or a rating migration.

---

## Dataset

[American Companies Bankruptcy Prediction](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)

- 78,682 firm-year observations from 8,971 NYSE/NASDAQ companies
- 18 accounting variables per observation
- Binary target: `alive` / `failed`, assigned per company rather than per firm-year. 609 of the 8,971 companies, 6.8%, are eventually labelled `failed` - a company-level lifetime rate, not a firm-year default rate. This notebook derives a two-year forward hazard label instead: 1,123 positive firm-years of 72,393 (1.551% base rate)
- Period: 1999-2018
- License: CC0 Public Domain

---

## Dependencies

Declared in `pyproject.toml`, pinned to exact versions in `uv.lock`.

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation |
| `numpy` | Numerics |
| `xgboost` | Gradient-boosted classifier |
| `scikit-learn` | Metrics, train/test utilities |
| `shap` | Model interpretability |
| `matplotlib` | Plotting |
| `seaborn` | Heatmaps and styled plots |
| `ipykernel` | Jupyter kernel for the notebook |
| `ipywidgets` | Progress-bar rendering for `tqdm`, which `shap` pulls in |

`jupyterlab` sits in the dev group (`[dependency-groups]`), installed by `uv sync` but excluded from the runtime dependency set.

There is deliberately no `requirements.txt`. `uv.lock` supersedes it, pinning every transitive package to an exact version and hash. If you need one for a tool that can't read a lockfile, generate it on demand rather than committing a second copy that drifts:

```bash
uv export --no-hashes --no-dev --format requirements.txt -o requirements.txt
```

---

## Project Structure

```text
.
├── corp_default.ipynb    # Full pipeline, one notebook
├── pyproject.toml        # Dependency declarations (source of truth)
├── uv.lock               # Exact resolved versions, commit this
├── .python-version       # 3.13, fetched by uv if you lack it
├── data/
│   └── README.md         # Dataset instructions; the CSV itself is gitignored
├── LICENSE               # MIT License
└── README.md             # You are here
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with Python, XGBoost, and public data. No proprietary tools required.</sub>
</p>
