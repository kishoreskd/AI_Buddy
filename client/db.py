import sqlite3
import json
from typing import List, Tuple, Optional

DB_NAME = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            title TEXT,  -- NEW COLUMN
            question TEXT,
            answer TEXT,
            document_info TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.commit()
    conn.close()


def save_message(
    session_id: str,
    question: str,
    answer: str,
    document_info: Optional[List[dict]] = None,
    title: Optional[str] = None,
):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    doc_json = json.dumps(document_info or [])

    if title:  # Save title only once per session
        c.execute("SELECT COUNT(*) FROM chats WHERE session_id = ?", (session_id,))
        if c.fetchone()[0] == 0:
            c.execute(
                "INSERT INTO chats (session_id, title, question, answer, document_info) VALUES (?, ?, ?, ?, ?)",
                (session_id, title, question, answer, doc_json),
            )
        else:
            c.execute(
                "INSERT INTO chats (session_id, question, answer, document_info) VALUES (?, ?, ?, ?)",
                (session_id, question, answer, doc_json),
            )
    else:
        c.execute(
            "INSERT INTO chats (session_id, question, answer, document_info) VALUES (?, ?, ?, ?)",
            (session_id, question, answer, doc_json),
        )

    conn.commit()
    conn.close()


def get_session_ids() -> List[str]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT session_id FROM chats ORDER BY timestamp DESC")
    sessions = [row[0] for row in c.fetchall()]
    conn.close()
    return sessions


def get_session_messages(session_id: str) -> List[Tuple[str, str, List[dict]]]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT question, answer, document_info FROM chats WHERE session_id=? ORDER BY id",
        (session_id,),
    )
    rows = c.fetchall()
    conn.close()
    return [(q, a, json.loads(d)) for q, a, d in rows]


def get_sessions_with_titles() -> List[Tuple[str, str]]:
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute(
        "SELECT DISTINCT session_id, COALESCE(title, session_id) FROM chats ORDER BY timestamp DESC"
    )
    sessions = c.fetchall()
    conn.close()
    return sessions


def clear_history():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM chats")
    conn.commit()
    conn.close()
