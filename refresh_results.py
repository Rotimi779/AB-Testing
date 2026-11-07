"""
Recompute experiment results and refresh what the dashboard reads.

    python refresh_results.py
    streamlit run charts.py

Steps:
1. Recompute the per-experiment summary (statistical_tests.build_experiment_summary).
2. Write data/summary_and_analysis.csv.
3. Replace the summary_and_analysis table in ab_testing.db, and refresh the
   experiment_policy table so the database matches the policy file.

Only these two tables are rewritten. Run `python load_csv_files.py` first if the raw
CSV files (sessions, assignments, experiments, ...) have changed.
"""
import sqlite3
from pathlib import Path

import pandas as pd

from statistical_tests import (
    DATA_DIR,
    POLICY_CSV_NAME,
    build_experiment_summary,
    write_summary_csv,
)

DB_PATH = Path("ab_testing.db")
SUMMARY_TABLE = "summary_and_analysis"
POLICY_TABLE = "experiment_policy"


def refresh_results(data_dir=DATA_DIR, db_path=DB_PATH) -> pd.DataFrame:
    summary_df = build_experiment_summary(data_dir)
    csv_path = write_summary_csv(summary_df, data_dir)
    policy_df = pd.read_csv(Path(data_dir) / POLICY_CSV_NAME)
    with sqlite3.connect(db_path) as conn:
        summary_df.to_sql(SUMMARY_TABLE, conn, if_exists="replace", index=False)
        policy_df.to_sql(POLICY_TABLE, conn, if_exists="replace", index=False)
    print(f"Wrote {csv_path} and refreshed tables '{SUMMARY_TABLE}' and '{POLICY_TABLE}' in {db_path}.")
    return summary_df


if __name__ == "__main__":
    df = refresh_results()
    print(df[["experiment_name", "difference_pp", "p_value", "decision", "validity_status"]].to_string(index=False))
    print("\nRestart `streamlit run charts.py` (or clear its cache) to see the refreshed results.")
