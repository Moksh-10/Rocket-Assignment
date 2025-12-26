import sqlite3
from pathlib import Path
import os

print("Running from:", os.getcwd())

DB_PATH = Path("research_agent.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    # ====================== research_runs ======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS research_runs (
        id TEXT PRIMARY KEY,
        original_query TEXT NOT NULL,
        classification TEXT,
        primary_intent TEXT,
        relationship TEXT,
        difficulty_level TEXT,

        final_answer TEXT,                 
        markdown_path TEXT,                
        html_path TEXT,                    

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # ====================== sub_questions ======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sub_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        research_run_id TEXT NOT NULL,
        sub_question TEXT NOT NULL,
        position INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (research_run_id) REFERENCES research_runs(id)
    )
    """)

    # ====================== search_results ======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS search_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sub_question_id INTEGER NOT NULL,
        search_query TEXT NOT NULL,
        search_type TEXT NOT NULL,          -- depth / breadth
        url TEXT,
        title TEXT,
        snippet TEXT,
        score REAL,
        source_engine TEXT DEFAULT 'tavily',
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (sub_question_id) REFERENCES sub_questions(id)
    )
    """)

    # ====================== sub_question_summaries ======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sub_question_summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sub_question_id INTEGER NOT NULL UNIQUE,
        summary_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (sub_question_id) REFERENCES sub_questions(id)
    )
    """)

    # ====================== judge_evaluations ======================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS judge_evaluations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        research_run_id TEXT NOT NULL,
        overall_status TEXT NOT NULL,
        confidence_score REAL,
        coverage_assessment TEXT,          -- JSON
        detected_gaps TEXT,                -- JSON
        recommended_follow_up TEXT,        -- JSON
        termination_reasoning TEXT,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (research_run_id) REFERENCES research_runs(id)
    )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized / verified")

