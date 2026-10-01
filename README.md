# Calendar time, foreign trading and the weekend: evidence from three changes of the trading week

Samir Orujov, School of Business, ADA University, Baku

Code, public data, pre-analysis plans and results for the paper, which is under review at the *Journal of International Financial Markets, Institutions & Money*.

## What the paper does

Three exchanges moved their trading week: Tadawul (Saudi Arabia) in 2013, Dubai and Abu Dhabi in 2022, and Tel Aviv in 2026. Each kept the weekend closure at 64 to 70 hours but took one US trading session out of it. The paper uses these changes, together with weeknights on which the US market is closed, to ask whether the variance of returns over a closure builds up with calendar time or with trading abroad.

- Calendar time is rejected everywhere: a weekend carries the variance of 1.2 to 2.5 weeknights, not 3.4 to 4.1.
- A weeknight without a US session carries about 30 per cent less variance for Gulf shares, and half as much for Israeli shares listed in New York.
- At the weekend, the pre-specified test cannot separate the trading-time prediction from no change. Exploratory measures suggest that information moved into the first session after the weekend.

JEL: G14, G15, C22. Keywords: weekend effect, non-trading periods, information arrival, dual listing, trading calendar, Gulf stock markets.

## Where to start

| If you want to | Look in |
|---|---|
| read the paper | `manuscript/manuscript.pdf` and `manuscript/supplementary_material.pdf` |
| see the pre-analysis plans | `pap/`: the Tel Aviv plan (01), the Gulf plan (02, fixed before any Gulf statistic was computed), two exploratory addenda, and their SHA-256 digests in `pap/HASHES.txt` |
| reproduce the Tel Aviv results | `python code/run_all.py` (scripts `code/00`–`08`) |
| reproduce the Gulf results | `code/gcc/09`–`17` (needs licensed data, see below) |
| check a number in the paper | `output/`: every number in the paper is read from these result files and tables |

## Layout

```
code/         Tel Aviv pipeline (00–08), Gulf pipeline (gcc/09–17), figures and tables (pooled/20)
data/raw/     official Tel Aviv Stock Exchange end-of-day records (indices and stocks)
lseg_jobs/    scripts that retrieve the Gulf share prices from LSEG Workspace
pap/          pre-analysis plans, addenda and their digests
output/       result files (JSON) and LaTeX tables
manuscript/   LaTeX source and PDFs of the paper and the supplementary material
```

## Running the code

Run everything from the repository root with Python 3.11 and pandas, NumPy, SciPy and Matplotlib (the LSEG scripts also need `lseg-data` and `python-dotenv`):

```
python code/run_all.py
```

## Data

- **Tel Aviv:** the official end-of-day records of the Tel Aviv Stock Exchange are included in `data/raw/tase/`.
- **Gulf markets (Tadawul, DFM, ADX):** share prices are licensed from LSEG Workspace and cannot be redistributed. With a Workspace licence, `lseg_jobs/` retrieves them into `data/raw/lseg/`. The cell-level statistics derived from them are in `output/`, so every Gulf number in the paper can be checked without the raw data.
- **US session dates and the Yahoo TA-125 series** used in the appendix on missing sessions come from Yahoo Finance and are not redistributed; download the daily history of `^GSPC` and `^TA125.TA`.

`REPLICATION.md` is the short note that accompanied the journal submission.

## Citation

Orujov, S. (2026). Calendar time, foreign trading and the weekend: evidence from three changes of the trading week. Working paper, ADA University.

## Licence

Code: MIT (see `LICENSE`). Tel Aviv Stock Exchange data remain subject to the exchange's terms of use. The text of the paper remains the author's; please cite it rather than reuse it.
