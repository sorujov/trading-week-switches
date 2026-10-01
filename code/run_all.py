"""Rebuild every table and figure from data/raw. Data retrieval itself (TASE website service, Yahoo Finance)
is documented in README.md; raw files are included in the replication package."""
import subprocess, sys
for s in ["01_build.py", "02_estimate.py", "03_vendor_and_exploratory.py", "04_stocks.py",
          "06_stock_decomposition.py", "08_revision.py", "05_figures.py", "07_tables.py"]:
    print("==", s, flush=True); subprocess.run([sys.executable, f"code/{s}"], check=True)
