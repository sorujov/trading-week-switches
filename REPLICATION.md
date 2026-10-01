# Replication package

Run everything from this folder's root.

- `pap/` pre-analysis plans (01 Tel Aviv; 02 Gulf, hashed before any Gulf gap statistic) and addenda B and C (exploratory), with SHA-256 hashes in `HASHES.txt`.
- `code/00–08*.py` Tel Aviv pipeline (`python code/run_all.py`), from the official TASE records in `data/raw/tase/`.
- `code/gcc/09–17*.py` Gulf pipeline and exploratory checks; `code/pooled/20_figures_tables.py` figures and tables.
- `output/` results files and LaTeX tables from which every number in the paper is taken.

Data not included:
- Gulf share prices (Tadawul, DFM, ADX) are licensed from LSEG Workspace and may not be redistributed. `lseg_jobs/` would retrieve them with a Workspace licence; `code/gcc/10_build_gcc.py` expects them in `data/raw/lseg/`. The cell-level statistics derived from them are in `output/`.
- NYSE session dates (S&P 500 index, `data/raw/tase/GSPC.csv`) and the Yahoo Finance TA-125 series used in the appendix on missing sessions are from Yahoo Finance and are not redistributed; download ^GSPC and ^TA125.TA daily history.
