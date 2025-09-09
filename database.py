"""Simple SQLite helpers for the CTI pipeline."""

from pathlib import Path
import sqlite3
import pandas as pd

DB_PATH = Path("data/processed/cti.db")


def load_table(table: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load a table from the SQLite database into a DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM {table}", conn)


def save_table(df: pd.DataFrame, table: str, db_path: Path = DB_PATH) -> None:
    """Write a DataFrame to a table in the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the CTI SQLite database.")
    parser.add_argument("--table", help="Table name to display", default=None)
    args = parser.parse_args()

    if args.table:
        df = load_table(args.table)
        print(df.head())
    else:
        with sqlite3.connect(DB_PATH) as conn:
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )
        print("Available tables:")
        print(tables["name"].tolist())
