#!/usr/bin/env python3
from pathlib import Path
import pandas as pd

# Hard-coded parquet paths (edit here if your filenames differ)
ABS_CANDIDATES = [
    Path("processed/flu/data_abs.parquet"),
    Path("processed/flu/dataset_abs.parquet"),
]
EM_CANDIDATES = [
    Path("processed/flu/data_em.parquet"),
    Path("processed/flu/dataset_em.parquet"),
]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def first_existing(paths):
    for p in paths:
        q = p if p.is_absolute() else (Path.cwd() / p)
        if q.exists():
            return q
    return None

def show_parquet(title, path: Path | None):
    print(f"\n=== {title} ===")
    if path is None:
        print("Not found. Update the hard-coded paths in this script.")
        return
    print("Path:", path)
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"Failed to read parquet: {e}\nInstall pyarrow or fastparquet if missing.")
        return
    print("Shape:", df.shape)
    print("-- head(5) --")
    print(df.head(5).to_string(index=False))
    print("-- tail(5) --")
    print(df.tail(5).to_string(index=False))

def main():
    abs_p = first_existing(ABS_CANDIDATES)
    em_p = first_existing(EM_CANDIDATES)
    show_parquet("ABS", abs_p)
    show_parquet("EM", em_p)

if __name__ == "__main__":
    main()
