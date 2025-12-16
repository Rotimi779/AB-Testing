import sqlite3
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")       
DB_PATH = Path("ab_testing.db")

def csv_to_table_name(csv_path: Path) -> str:
    """
    Convert file name to a safe SQLite table name.
    E.g. user_sessions.csv -> user_sessions
    """
    return csv_path.stem.lower()

def load_all_csvs():
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        for csv_file in csv_files:
            table_name = csv_to_table_name(csv_file)
            df = pd.read_csv(csv_file)

            df.to_sql(
                table_name,
                conn,
                if_exists="replace",
                index=False
            )



load_all_csvs()
