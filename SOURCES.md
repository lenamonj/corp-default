# Sources

Every fact in this casebook that comes from outside the repository has a row here.
A fact that is not in this file does not appear in a notebook or in the README.
All sources were accessed on 2026-11-11 unless a row says otherwise.

## The dataset and the paper

Paper: Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., Poggi, A. (2022).
Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks.
*Future Internet* 14(8), 244. https://doi.org/10.3390/fi14080244. Licence CC BY 4.0.
The publisher page refuses automated access, so the text was read from the University of Parma
repository copy of the published version: https://air.unipr.it/retrieve/c744856e-2113-4092-a35f-1e9a84e66a7f/futureinternet-14-00244-v2.pdf
(23 pages, SHA-256 `8f01652042d9f4d597b67455d91f472fc8f8e209fbc8fb0eed86aaf157d118b1`).

| Claim | Value | Source |
|---|---|---|
| Companies in the dataset, as the paper states | 8,262 | Paper, pp. 1, 2, 4, and 21 (abstract, contributions, Section 3, conclusions) |
| Firm-year observations | 78,682 | Paper, Section 3, p. 4 |
| Period of the financial variables | 1999 to 2018 | Paper, abstract and Section 3 |
| Exchanges | New York Stock Exchange and NASDAQ | Paper, Section 3, p. 4 |
| What counts as bankruptcy | A filing under Chapter 11 or Chapter 7 of the Bankruptcy Code | Paper, Section 3, p. 4 |
| How the label is assigned | "we label the fiscal year before the chapter filing as 'bankruptcy' (1) for the next year" | Paper, Section 3, p. 4 |
| How the label sits in the file (computed here, not from the paper) | `status_label` does not hold the per-year label the paper describes. It is constant within each company: 609 companies are `failed` in every one of their 5,220 rows, and no company mixes the two values | `data/american_bankruptcy.csv` |
| The paper's split | Train 1999 to 2011, validation 2012 to 2014, test 2015 to 2018 | Paper, Section 3, p. 4, and Section 6, p. 11 |
| Missing values | "The dataset has no missing values or synthetic and imputed added values" | Paper, Section 3, p. 4 |
| Variable definitions X1 to X18 | Current assets, cost of goods sold, depreciation and amortization, EBITDA, inventory, net income, total receivables, market value, net sales, total assets, total long-term debt, EBIT, gross profit, total current liabilities, retained earnings, total revenue, total liabilities, total operating expenses | Paper, Table 2, pp. 5 to 6 |
| Firms and bankruptcies by year | Table 1 lists years 2000 to 2019; firms 5,308 in 2000 falling to 2,723 in 2019; bankrupt firms 3, 7, 10, 17 in 2000 to 2003 and 609 in all | Paper, Table 1, p. 5 |
| Training scheme | Balanced by random undersampling of healthy firms, 100 runs per model | Paper, Section 6.1, p. 12 |
| Best average validation AUC, one year of data (WL = 1) | Random forest 0.748; best single run, neural network 0.856 | Paper, Section 6.1, p. 12 |
| Test AUC 2015 to 2018, task T1, WL = 1 | SVM 0.585, ANN 0.861, logistic regression 0.755, AdaBoost 0.658, random forest 0.717, gradient boosting 0.759, XGBoost 0.786 | Paper, Table 4, p. 17 |
| Altman thresholds as the paper states them | "Altman suggested two thresholds, 1.81 and 2.99." The sentence carries no citation; Altman (1968) is cited a paragraph earlier | Paper, Section 2, p. 3 (secondary; the primary is under Classic scores) |

Cross-check of the paper against the file, computed here on 2026-11-11: the paper's Table 1 matches
`data/american_bankruptcy.csv` exactly in all 20 years once the paper's year is read as the file's
year plus one. Firms observed in file year `t` equal the paper's total firms in year `t + 1`, and
failed companies whose last file year is `t` equal the paper's bankrupt firms in year `t + 1`.
The paper's yearly totals sum to 78,682 firm-years and 609 failures, both equal to the file.
The file holds 8,971 distinct `company_name` values, not 8,262. No subset tried reproduces 8,262
(companies with two or more years: 8,216; companies never labelled failed: 8,362). The difference is unreconciled.

Follow-up paper by the same authors, cited by the dataset's README: Pellegrino, M., Lombardo, G.,
Adosoglou, G., Cagnoni, S., Pardalos, P. M., Poggi, A. (2024). A Multi-Head LSTM Architecture for
Bankruptcy Prediction with Time Series Accounting Data. *Future Internet* 16(3), 79.
https://doi.org/10.3390/fi16030079. Read from
https://mdpi-res.com/d_attachment/futureinternet/futureinternet-16-00079/article_deploy/futureinternet-16-00079.pdf
(SHA-256 `4221ea0548877201c652b77f7499e3f5097d317a6d7469a3b6412bfd29690ed0`).

| Claim | Value | Source |
|---|---|---|
| Company count in the follow-up | 8,262, again | Pellegrino et al. (2024), abstract, p. 1 |
| Industry codes | "each company in the dataset was categorized using the Standard Industrial Classification (SIC) system ... specifying major groups within each firm" | Pellegrino et al. (2024), p. 7 |
| Units and data vendor | Not stated | Pellegrino et al. (2024) |

## Where the file can be obtained

| Copy | Location | Licence as labelled | SHA-256 of the CSV | Result of comparison with the Kaggle file |
|---|---|---|---|---|
| Kaggle, "US Company Bankruptcy Prediction Dataset", uploaded by Utkarsh Singh, version 3, last updated 2023-05-27 | https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset | CC0: Public Domain | `cff2c899a97ecd629415cb22f59186000e74e1c0a78cfae036c0a53025419b5e` | This is the file the repo uses (11,254,588 bytes, UTF-8 with BOM) |
| Kaggle public download endpoint, no login | https://www.kaggle.com/api/v1/datasets/download/utkarshx27/american-companies-bankruptcy-prediction-dataset | as above | `cff2c899...419b5e` for `american_bankruptcy.csv` inside the zip | Byte-identical. Zip SHA-256 `bb947405f112fa0f7eaab8c71368319744e29d096050779192996258af2b1309` |
| Authors' repository, original upload, commit `7f56a80b73c2f32582fcc01d28acb6adc665ae40` (2022-07-26) | https://raw.githubusercontent.com/sowide/bankruptcy_dataset/7f56a80b73c2f32582fcc01d28acb6adc665ae40/american_bankruptcy_dataset.csv | None at that commit; the repository's CC BY 4.0 `LICENSE.md` was added on 2022-08-09 (commit `a938168`) | `1aff805af8d2f4368ffcd876814a4589e09905a9cbac289a7b780ed0f7a2478a` | Same content, different bytes. Same 78,682 (company, year) keys in the same order, same `status_label`, every one of the 18 items equal within 0.0005. Differences are formatting only: `year` written as a float, numbers at full float precision, no BOM. 13,196,887 bytes |
| Authors' repository, current file, commit `279e4275101e5df7a39e8e6b3179b66c442557f3` ("Dataset Update", 2024-04-10) | https://raw.githubusercontent.com/sowide/bankruptcy_dataset/279e4275101e5df7a39e8e6b3179b66c442557f3/american_bankruptcy_dataset.csv (12,395,824 bytes) | CC BY 4.0 | `d64ed85ae786d75113455ae238feb24015d19716c10ec1de3a90a52eddc7b04a` | Not the same data. Same keys and labels, but `year` is renamed `fyear`, the X columns are in a different order (its X2 is Kaggle X10, its X3 is Kaggle X2, and so on), and of 1,416,276 item cells 64.5% are 1,000 times the Kaggle value (for example Kaggle 511.267 appears as 511267), 32.4% are equal, and 3.1% are zero in both. Because Kaggle X9 equals X16 in every row, values alone cannot tell which of its X14 and X17 is which. Two extra columns, `Division` (letters A to J) and `MajorGroup` (73 values), constant within each company and never missing. The README does not define them; Pellegrino et al. (2024) describe SIC divisions and major groups |
| Authors' repository HEAD, commit `8bcd8db` (2024-08-06) | https://github.com/sowide/bankruptcy_dataset | CC BY 4.0 | not compared | Adds `dataset_paper.zip` (8,603,541 bytes): train, validation and test financial CSVs and text from filing items 1, 5, and 7. Not used here |

The authors' README says the non-anonymised version cannot be released "because of the license used
to retrieve the dataset". Neither the paper, the README, nor the Kaggle page names the data vendor or
states the units of the items.

Licence note: the authors publish the data under CC BY 4.0. The Kaggle copy is a re-upload labelled
CC0 by someone other than the authors. Citing the paper meets the attribution the original licence asks for.

## Classic scores

Altman (2000) was read from a course-hosted copy because the Stern URL returned HTTP 500 and the
Wayback Machine rate-limited the request. Canonical URL: https://pages.stern.nyu.edu/~ealtman/Zscores.pdf.
Copy read: https://moodle2.units.it/pluginfile.php/322514/mod_resource/content/0/Zscores.pdf
(SHA-256 `cc9acbf25f3618afcd39820689d5950eeebaec478fea71e4d8d513fd7f763074`). Title page: Edward I.
Altman, "Predicting Financial Distress of Companies: Revisiting the Z-Score and ZETA Models", July 2000,
adapted from Altman (1968) and Altman, Haldeman and Narayanan (1977).
The document has no printed page numbers; pages below count PDF pages from the title page.

| Claim | Value | Source |
|---|---|---|
| Z (1968), as published | Z = 0.012 X1 + 0.014 X2 + 0.033 X3 + 0.006 X4 + 0.999 X5, X1 to X4 in percent, X5 in times | Altman (2000), p. 9 and p. 12 |
| Z, the form used here | Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5 with X1 to X4 as ratios | Altman (2000), p. 13 |
| Z variables | X1 working capital / total assets; X2 retained earnings / total assets; X3 EBIT / total assets; X4 market value of equity / book value of total liabilities; X5 sales / total assets | Altman (2000), p. 9 |
| Z zones | "all firms having a Z score of greater than 2.99 clearly fall into the 'non-bankrupt' sector, while those firms having a Z below 1.81 are all bankrupt. The area between 1.81 and 2.99 will be defined as the 'zone of ignorance' or 'gray area'" | Altman (1968), p. 606 (see the note below the table) |
| Z cut-off with the lowest error in the original sample | 2.675, the midpoint of 2.67 to 2.68 | Altman (1968), p. 607; Altman (2000), p. 18 |
| Z lower bound restated | 1.81 | Altman (2000), pp. 18, 20, 22 |
| Z zones as Altman presents them in his own slides | "Z > 2.99 Safe Zone; 1.8 < Z < 2.99 Grey Zone; Z < 1.80 Distress Zone" | Altman, lecture slides, SGH Warsaw, 2016-04-14, slide 12; Altman, slides, LSE Systemic Risk Centre, 2019-10-16, slide 10 |
| Z' (private firms, book equity in X4) | Z' = 0.717 X1 + 0.847 X2 + 3.107 X3 + 0.420 X4 + 0.998 X5 | Altman (2000), p. 25 |
| Z' zones | Table 6 prints both "Z'<1.21 = Zone I" and "gray area = 1.23 to 2.90". Altman's 2016 slides give "Z' > 2.90 Safe; 1.23 < Z' < 2.90 Grey; Z' < 1.23 Distress" | Altman (2000), p. 26, Table 6; SGH slides, slide 30 |
| Z'' (non-manufacturers, no X5, book equity in X4) | Z'' = 6.56 X1 + 3.26 X2 + 6.72 X3 + 1.05 X4 | Altman (2000), p. 27 |
| Z'' constant in the emerging-market version | +3.25 | Altman (2000), p. 27 |
| Z'' zones, for the version with the +3.25 constant only | "Z'' > 5.85 Safe Zone; 4.35 < Z'' < 5.85 Grey Zone; Z'' < 4.35 Distress Zone" | SGH slides, slide 31. The form without the constant (1.10 and 2.60) is not stated there and is not used |
| Original sample | 66 manufacturers, 33 bankrupt (Chapter X petitions 1946 to 1965) and 33 paired non-bankrupt | Altman (2000), p. 7 |
| Altman (1968) bibliographic record | Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *Journal of Finance* 23(4), 589 to 609. https://doi.org/10.1111/j.1540-6261.1968.tb00843.x | Crossref record |
| Altman (2018) bibliographic record | Altman, E. I. (2018). Applications of Distress Prediction Models: What Have We Learned After 50 Years from the Z-Score Models? *International Journal of Financial Studies* 6(3), 70. https://doi.org/10.3390/ijfs6030070 | Crossref record; PDF from https://mdpi-res.com/d_attachment/ijfs/ijfs-06-00070/article_deploy/ijfs-06-00070.pdf |
| Zmijewski (1984) bibliographic record | Zmijewski, M. E. (1984). Methodological Issues Related to the Estimation of Financial Distress Prediction Models. *Journal of Accounting Research* 22, 59 onward. https://doi.org/10.2307/2490859 | Crossref record, which gives the first page only; text not read |

Altman (1968) was read from a JSTOR-stamped scan of the article hosted by a third party, because the
publisher and JSTOR copies are paywalled: https://www.calctopia.com/papers/Altman1968.pdf
(SHA-256 `34ba13a102ee4f1767762786e2720e9c6211e4d3d9252fb45856ca45cb21dd99`; journal p. 606 is PDF page 19).
Altman's slides: SGH Warsaw, https://ssl-www.sgh.waw.pl/sites/sgh.waw.pl/files/2022-01/eward-altman-lecture-series-2016-edward-altman-evolution-of-bankruptcy-prediction-models-prezentacja.pdf
(SHA-256 `0aa5ad0300ad0261efbc36cfe7ce8fea815e123cb012769542bea486b85773f0`); LSE Systemic Risk Centre,
https://www.systemicrisk.ac.uk/sites/default/files/images/Evolution%20of%20Altman%20Z-Score.lse.pdf
(SHA-256 `965e842fbc2ccbff1155ce38fedc14a4f4cba78af6d773b804f5051f902b9c5b`).

## Recession dates

| Claim | Value | Source |
|---|---|---|
| 2001 recession | Peak March 2001 (2001Q1), trough November 2001 (2001Q4) | NBER, US Business Cycle Expansions and Contractions, https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions |
| 2007 to 2009 recession | Peak December 2007 (2007Q4), trough June 2009 (2009Q2) | same |

## Literature

Bibliographic records from Crossref, arXiv, and the NeurIPS proceedings. A sentence that states what a
paper finds needs its own row, from the paper's text, before it appears in a chapter.

| Topic | Citation | Record |
|---|---|---|
| Hazard models | Shumway, T. (2001). Forecasting Bankruptcy More Accurately: A Simple Hazard Model. *Journal of Business* 74(1), 101 to 124. https://doi.org/10.1086/209665 | Crossref |
| Hazard models | Campbell, J. Y., Hilscher, J., Szilagyi, J. (2008). In Search of Distress Risk. *Journal of Finance* 63(6), 2899 to 2939. https://doi.org/10.1111/j.1540-6261.2008.01416.x | Crossref |
| Precision-recall interpolation | Davis, J., Goadrich, M. (2006). The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning*, 233 to 240. https://doi.org/10.1145/1143844.1143874 | Crossref |
| Average precision in scikit-learn | AP = sum over n of (R_n - R_(n-1)) P_n. "This implementation is not interpolated and is different from computing the area under the precision-recall curve with the trapezoidal rule, which uses linear interpolation and can be too optimistic." | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html (scikit-learn 1.9.1 documentation) |
| SHAP | Lundberg, S. M., Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems 30*. https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html; arXiv:1705.07874 | NeurIPS proceedings, arXiv |
| SHAP for trees | Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., Lee, S.-I. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence* 2(1), 56 to 67. https://doi.org/10.1038/s42256-019-0138-9 | Crossref |
| Tabular deep learning | Grinsztajn, L., Oyallon, E., Varoquaux, G. (2022). Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data? *Advances in Neural Information Processing Systems 35*, Datasets and Benchmarks Track. https://doi.org/10.52202/068431-0037; arXiv:2207.08815 | NeurIPS proceedings, arXiv |
| Tabular deep learning | Shwartz-Ziv, R., Armon, A. (2022). Tabular Data: Deep Learning Is Not All You Need. *Information Fusion* 81, 84 to 90. https://doi.org/10.1016/j.inffus.2021.11.011 | Crossref |

## Fonts

All three families are under the SIL Open Font License 1.1. None of the three copyright lines names a
Reserved Font Name. Each project ships static TTF files at the needed weights, so no instancing is needed.

| Family | Files | Location, pinned to the last commit that touched them | Licence |
|---|---|---|---|
| Archivo | `Archivo-SemiBold.ttf` (186,700 bytes), `Archivo-ExtraBold.ttf` (191,700 bytes) | https://github.com/Omnibus-Type/Archivo/tree/b5d63988ce19d044d3e10362de730af00526b672/fonts/ttf | `OFL.txt` in the repository root: "Copyright 2020 The Archivo Project Authors" |
| Public Sans | `PublicSans-Regular.ttf` (84,836 bytes), `PublicSans-SemiBold.ttf` (84,412 bytes) | https://github.com/uswds/public-sans/tree/052e128daf9dc880db9789e045867ffc0e1e87dd/fonts/ttf | `OFL.txt` in the repository root: "Copyright 2015 The Public Sans Project Authors" |
| JetBrains Mono | `JetBrainsMono-Medium.ttf` (270,316 bytes), `JetBrainsMono-Bold.ttf` (274,096 bytes) | https://github.com/JetBrains/JetBrainsMono/tree/02bb50b082dad9ef8a0f33ac393839202b760223/fonts/ttf | `OFL.txt` in the repository root: "Copyright 2020 The JetBrains Mono Project Authors" |

## Tooling

| Claim | Value | Source |
|---|---|---|
| uv build backend | `[build-system] requires = ["uv_build>=X,<Y"]`, `build-backend = "uv_build"`; default module discovery is `src/<normalised project name>/__init__.py`. The page's current example, and this repo's pin, is `uv_build>=0.13.7,<0.14` (uv and uv-build 0.13.7 on PyPI, released 2026-11-03). uv 0.12.0 changed nothing in the build backend's configuration | https://docs.astral.sh/uv/concepts/build-backend/; https://github.com/astral-sh/uv/blob/0.13.5/changelogs/0.12.x.md |
| PyTorch CPU wheels with uv | An index `pytorch-cpu` at `https://download.pytorch.org/whl/cpu` with `explicit = true`, bound to `torch` under `[tool.uv.sources]` | https://docs.astral.sh/uv/guides/integration/pytorch/ |

## Not verified

These are not used until a primary source is read.

| Item | Why |
|---|---|
| Altman Z'' zone cut-offs without the +3.25 constant (1.10 and 2.60) | No Altman-authored text read states them |
| Zmijewski (1984) coefficients and last page | The article is behind the JSTOR paywall and no open copy was found. Secondary sources disagree on the sign and direction of the liquidity term |
| Units of the 18 items | Not stated in either paper, the authors' README, or the Kaggle page |
| Data vendor | Not named in either paper, the authors' README, or the Kaggle page |
| The 8,262 company count | Stated four times by the 2022 paper, once by the 2024 paper, and once by the README; the file holds 8,971 and no subset tried gives 8,262 |
