"""Download the Hillstrom email-marketing experiment and save it to data/hillstrom.csv.

64,000 customers who bought in the last 12 months were randomly split three ways:
Mens E-Mail, Womens E-Mail, or No E-Mail. Outcomes were tracked for two weeks.

Run:  python src/get_data.py
"""
from pathlib import Path

import pandas as pd

URL = "https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz"
OUT = Path(__file__).resolve().parents[1] / "data" / "hillstrom.csv"


def main() -> None:
    df = pd.read_csv(URL)               # pandas downloads and unzips the .gz file
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows and {df.shape[1]} columns to {OUT}")


if __name__ == "__main__":
    main()