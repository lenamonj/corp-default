# Data

This folder ships empty. The data is not redistributed here, and one command fetches it:

```bash
uv run python -m tools.get_data
```

Run it from the repo root. It writes two files and checks each download against the SHA-256 recorded in [SOURCES.md](../SOURCES.md).

| File | What it is | Where it comes from | License as labeled |
| --- | --- | --- | --- |
| `data/american_bankruptcy.csv` | The panel every chapter reads: one row for each company and fiscal year | [Kaggle](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset), a copy of the dataset of Lombardo et al. (2022) | CC0 Public Domain |
| `data/industry.csv` | The SIC division and major group of each company | The authors' own [repository](https://github.com/sowide/bankruptcy_dataset), from which only the company name and the two industry codes are kept | CC BY 4.0 |

A file that is already present with the right hash is not downloaded again.
A download whose hash does not match is refused.

If a notebook stops with a message that names this page, the files are missing: run the command above.

Cite the paper when you use the data:

Lombardo, G., Pellegrino, M., Adosoglou, G., Cagnoni, S., Pardalos, P. M., Poggi, A. (2022). Machine Learning for Bankruptcy Prediction in the American Stock Market: Dataset and Benchmarks. *Future Internet* 14(8):244. [doi:10.3390/fi14080244](https://doi.org/10.3390/fi14080244)
