# data_loader.py

import sqlite3
import pandas as pd
from pathlib import Path
import os

# Paths
DB_PATH = Path("db/immune_cells.db")
CSV_PATH = Path("data/cell-count.csv")

# Immune cell columns
CELL_COLS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

def initialize_database(db_path=DB_PATH):
    """Create SQLite schema for immune cell data."""
    # Ensure the parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = ON;")

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS projects (
        project_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT UNIQUE
    );

    CREATE TABLE IF NOT EXISTS subjects (
        subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject TEXT UNIQUE,
        project_id INTEGER NOT NULL,
        treatment TEXT NOT NULL,
        response TEXT NOT NULL,
        sample_type TEXT NOT NULL,
        condition TEXT NOT NULL,
        sex TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(project_id)
    );

    CREATE TABLE IF NOT EXISTS samples (
        sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample TEXT UNIQUE,
        subject_id INTEGER NOT NULL,
        time_from_treatment_start INTEGER NOT NULL,
        total_count INTEGER,
        FOREIGN KEY(subject_id) REFERENCES subjects(subject_id)
    );

    CREATE TABLE IF NOT EXISTS cell_counts (
        cell_count_id INTEGER PRIMARY KEY AUTOINCREMENT,
        sample_id INTEGER NOT NULL,
        cell_type TEXT NOT NULL,
        count INTEGER NOT NULL,
        percentage REAL,
        FOREIGN KEY(sample_id) REFERENCES samples(sample_id)
    );
    """)
    conn.commit()
    return conn, cur

def load_csv_to_db(csv_path=CSV_PATH, db_path=DB_PATH):
    """Load CSV data into SQLite database."""
    df = pd.read_csv(csv_path)
    df['response'] = df['response'].fillna('N/A').astype(str)

    # Validate required columns
    required_cols = ["project", "subject", "treatment", "response", "sample_type", "condition", "sample", "time_from_treatment_start"] + CELL_COLS
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    conn, cur = initialize_database(db_path)

    # Insert projects
    for project in df["project"].unique():
        cur.execute("INSERT OR IGNORE INTO projects (project_name) VALUES (?);", (project,))
    conn.commit()
    project_lookup = {name: pid for pid, name in cur.execute("SELECT project_id, project_name FROM projects;").fetchall()}

    # Insert subjects
    for subject in df["subject"].unique():
        subj_rows = df[df["subject"] == subject]
        treatment_vals = subj_rows["treatment"].unique()
        response_vals = subj_rows["response"].unique()
        sample_type_vals = subj_rows["sample_type"].unique()
        condition_vals = subj_rows["condition"].unique()
        sex = subj_rows["sex"].unique()
        if len(treatment_vals) > 1 or len(response_vals) > 1 or len(sample_type_vals) > 1 or len(condition_vals) > 1 or len(sex) > 1:
            raise ValueError(f"Inconsistent metadata for subject {subject}")
        cur.execute("""
            INSERT OR IGNORE INTO subjects (subject, project_id, treatment, response, sample_type, condition, sex)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (
            subject,
            project_lookup[subj_rows["project"].iloc[0]],
            treatment_vals[0],
            response_vals[0],
            sample_type_vals[0],
            condition_vals[0],
            sex[0]
        ))
    conn.commit()
    subject_lookup = {name: sid for sid, name in cur.execute("SELECT subject_id, subject FROM subjects;").fetchall()}

    # Insert samples
    for _, row in df.iterrows():
        total_count = row[CELL_COLS].sum()
        cur.execute("""
            INSERT OR IGNORE INTO samples (sample, subject_id, time_from_treatment_start, total_count)
            VALUES (?, ?, ?, ?);
        """, (row["sample"], subject_lookup[row["subject"]], row["time_from_treatment_start"], total_count))
    conn.commit()
    sample_lookup = {name: sid for sid, name in cur.execute("SELECT sample_id, sample FROM samples;").fetchall()}

    # Insert cell counts
    for _, row in df.iterrows():
        total_count = row[CELL_COLS].sum()
        for cell in CELL_COLS:
            count = row[cell]
            percentage = count / total_count * 100 if total_count > 0 else 0.0
            cur.execute("""
                INSERT INTO cell_counts (sample_id, cell_type, count, percentage)
                VALUES (?, ?, ?, ?);
            """, (sample_lookup[row["sample"]], cell, count, percentage))

    conn.commit()
    conn.close()
    print(f"Database loaded successfully at {db_path}")
