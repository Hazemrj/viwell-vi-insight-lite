"""Tiny helper that gives every script the same database connection."""
import re

from sqlalchemy import create_engine, text

from .config import DB_URL, SQL_DIR


def get_engine():
    return create_engine(DB_URL, pool_pre_ping=True)


def run_sql_file(engine, path):
    """Run a .sql file made of ';'-separated statements (no stored procedures)."""
    sql = (SQL_DIR / path).read_text()
    sql = re.sub(r"--[^\n]*", "", sql)          # remove SQL comments (none of our SQL has '--' inside strings)
    with engine.begin() as conn:
        for stmt in sql.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))
