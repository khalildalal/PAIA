from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


class ProgressStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    profile_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS course_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    course_code TEXT NOT NULL,
                    title TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS course_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES course_documents(id) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    quiz_title TEXT,
                    quiz_topic TEXT,
                    score INTEGER NOT NULL,
                    total INTEGER NOT NULL,
                    percent REAL NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS quiz_attempt_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    is_correct INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE,
                    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
                )
                """
            )

            conn.execute(
                """
                INSERT OR IGNORE INTO users (username, password, is_admin, profile_json)
                VALUES (?, ?, ?, ?)
                """,
                ("admin", hash_password("admin123"), 1, json.dumps({})),
            )
            conn.commit()

    # -------------------------
    # Auth
    # -------------------------
    def create_user(self, username: str, password: str) -> bool:
        clean_username = (username or "").strip()
        if not clean_username or not password:
            return False

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO users (username, password, is_admin, profile_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (clean_username, hash_password(password), 0, json.dumps({})),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def authenticate(self, username: str, password: str) -> bool:
        clean_username = (username or "").strip()
        if not clean_username or not password:
            return False

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT password
                FROM users
                WHERE username = ?
                """,
                (clean_username,),
            ).fetchone()

        if not row:
            return False

        return row["password"] == hash_password(password)

    def is_admin(self, username: str) -> bool:
        clean_username = (username or "").strip()
        if not clean_username:
            return False

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT is_admin
                FROM users
                WHERE username = ?
                """,
                (clean_username,),
            ).fetchone()

        return bool(row and row["is_admin"] == 1)

    # -------------------------
    # Profiles
    # -------------------------
    def save_profile(self, username: str, profile_data: dict) -> None:
        clean_username = (username or "").strip()
        if not clean_username:
            return

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE users
                SET profile_json = ?
                WHERE username = ?
                """,
                (json.dumps(profile_data, indent=2), clean_username),
            )
            conn.commit()

    def load_profile(self, username: str) -> Optional[dict]:
        clean_username = (username or "").strip()
        if not clean_username:
            return None

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT profile_json
                FROM users
                WHERE username = ?
                """,
                (clean_username,),
            ).fetchone()

        if not row or not row["profile_json"]:
            return None

        try:
            return json.loads(row["profile_json"])
        except json.JSONDecodeError:
            return None

    # -------------------------
    # Admin
    # -------------------------
    def list_users(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT username, is_admin, profile_json
                FROM users
                ORDER BY LOWER(username)
                """
            ).fetchall()

        users: list[dict] = []
        for row in rows:
            try:
                profile = json.loads(row["profile_json"]) if row["profile_json"] else {}
            except json.JSONDecodeError:
                profile = {}

            users.append(
                {
                    "username": row["username"],
                    "is_admin": bool(row["is_admin"]),
                    "profile": profile,
                }
            )
        return users

    def delete_user(self, username: str) -> None:
        clean_username = (username or "").strip()
        if not clean_username or clean_username == "admin":
            return

        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM users
                WHERE username = ?
                """,
                (clean_username,),
            )
            conn.commit()

    # -------------------------
    # Course RAG
    # -------------------------
    def add_course_document(self, course_code: str, title: str, filename: str) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO course_documents (course_code, title, filename)
                VALUES (?, ?, ?)
                """,
                (course_code.strip(), title.strip(), filename.strip()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def add_course_chunk(
        self,
        *,
        document_id: int,
        chunk_index: int,
        chunk_text: str,
        embedding: list[float],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO course_chunks (document_id, chunk_index, chunk_text, embedding_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    json.dumps(embedding),
                ),
            )
            conn.commit()

    def list_course_documents(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    d.id,
                    d.course_code,
                    d.title,
                    d.filename,
                    d.uploaded_at,
                    COUNT(c.id) AS chunk_count
                FROM course_documents d
                LEFT JOIN course_chunks c ON c.document_id = d.id
                GROUP BY d.id, d.course_code, d.title, d.filename, d.uploaded_at
                ORDER BY d.uploaded_at DESC, d.id DESC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_course_chunks(self, course_code: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if course_code:
                rows = conn.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.chunk_text,
                        c.embedding_json,
                        d.course_code,
                        d.title,
                        d.filename
                    FROM course_chunks c
                    JOIN course_documents d ON d.id = c.document_id
                    WHERE d.course_code = ?
                    ORDER BY d.id, c.chunk_index
                    """,
                    (course_code.strip(),),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        c.id,
                        c.document_id,
                        c.chunk_index,
                        c.chunk_text,
                        c.embedding_json,
                        d.course_code,
                        d.title,
                        d.filename
                    FROM course_chunks c
                    JOIN course_documents d ON d.id = c.document_id
                    ORDER BY d.id, c.chunk_index
                    """
                ).fetchall()

        result: list[dict] = []
        for row in rows:
            try:
                emb = json.loads(row["embedding_json"]) if row["embedding_json"] else []
            except json.JSONDecodeError:
                emb = []

            result.append(
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "chunk_text": row["chunk_text"],
                    "embedding": emb,
                    "course_code": row["course_code"],
                    "title": row["title"],
                    "filename": row["filename"],
                }
            )
        return result

    def delete_course_document(self, document_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                DELETE FROM course_documents
                WHERE id = ?
                """,
                (document_id,),
            )
            conn.commit()

    # -------------------------
    # Quiz analytics
    # -------------------------
    def record_quiz_attempt(
        self,
        *,
        username: str,
        quiz_title: str,
        quiz_topic: str,
        score: int,
        total: int,
        percent: float,
        results: list[dict],
    ) -> None:
        clean_username = (username or "").strip()
        if not clean_username:
            return

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO quiz_attempts (username, quiz_title, quiz_topic, score, total, percent)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_username,
                    quiz_title,
                    quiz_topic,
                    int(score),
                    int(total),
                    float(percent),
                ),
            )
            attempt_id = int(cursor.lastrowid)

            for item in results:
                conn.execute(
                    """
                    INSERT INTO quiz_attempt_topics (attempt_id, username, topic, is_correct)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        clean_username,
                        str(item.get("topic", "Unknown Topic")),
                        1 if item.get("is_correct") else 0,
                    ),
                )
            conn.commit()

    def get_students_per_topic(self, selected_topic: str | None = None) -> list[dict]:
        with self._connect() as conn:
            if selected_topic and selected_topic != "All Topics":
                rows = conn.execute(
                    """
                    SELECT
                        topic,
                        COUNT(DISTINCT username) AS student_count
                    FROM quiz_attempt_topics
                    WHERE topic = ?
                    GROUP BY topic
                    ORDER BY topic
                    """,
                    (selected_topic,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        topic,
                        COUNT(DISTINCT username) AS student_count
                    FROM quiz_attempt_topics
                    GROUP BY topic
                    ORDER BY topic
                    """
                ).fetchall()

        return [dict(row) for row in rows]

    def get_quiz_progress_for_student(self, username: str) -> list[dict]:
        clean_username = (username or "").strip()
        if not clean_username:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    id,
                    quiz_title,
                    quiz_topic,
                    score,
                    total,
                    percent,
                    created_at
                FROM quiz_attempts
                WHERE username = ?
                ORDER BY id
                """,
                (clean_username,),
            ).fetchall()

        result = []
        for idx, row in enumerate(rows, start=1):
            result.append(
                {
                    "attempt_no": idx,
                    "quiz_title": row["quiz_title"],
                    "quiz_topic": row["quiz_topic"],
                    "score": row["score"],
                    "total": row["total"],
                    "percent": row["percent"],
                    "created_at": row["created_at"],
                }
            )
        return result

    def list_student_usernames(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT username
                FROM users
                WHERE is_admin = 0
                ORDER BY LOWER(username)
                """
            ).fetchall()
        return [row["username"] for row in rows]

    def list_quiz_topics(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT topic
                FROM quiz_attempt_topics
                WHERE topic IS NOT NULL AND TRIM(topic) != ''
                ORDER BY LOWER(topic)
                """
            ).fetchall()
        return [row["topic"] for row in rows]