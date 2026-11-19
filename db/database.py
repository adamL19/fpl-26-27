"""SQLite helpers: create schema, replace tables from DataFrames, load them back."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_FILE.read_text())
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def replace_table(conn: sqlite3.Connection, table: str, df: pd.DataFrame,
                  where: str | None = None, params: tuple = ()) -> None:
    """Delete rows (all, or matching `where`) and append `df`, keeping only
    columns that exist in the schema.  Missing columns become NULL."""
    cols = _columns(conn, table)
    out = df.reindex(columns=cols)
    with conn:
        conn.execute(f"DELETE FROM {table}" + (f" WHERE {where}" if where else ""), params)
        out.to_sql(table, conn, if_exists="append", index=False)


def read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f"SELECT * FROM {table}", conn)
