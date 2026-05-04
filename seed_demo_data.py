from __future__ import annotations

import json
import os
import random
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from database import hash_password
from student_model import StudentProfile

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.getenv("DATA_DIR")
    or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    or str(BASE_DIR / "data")
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "student_profiles.db"

PASSWORD = "student123"

TOPICS = [
    "Chapter 1: Descriptive Statistics",
    "Chapter 2: Probability",
    "Chapter 3: Discrete Random Variables and Probability Distributions",
    "Chapter 4: Continuous Random Variables and Probability Distributions",
    "Chapter 5: Joint Probability Distributions and Random Samples",
]

STUDENTS = [
    "Nour Ahmed", "Sara Khoury", "Maya Haddad", "Lea Nassar", "Rita Mansour",
    "Yara Saad", "Lara Daher", "Tala Farah", "Celine Karam", "Mira Chidiac",
    "Rana Habib", "Jana Aoun", "Dina Saliba", "Hiba Fares", "Lina Abi Rached",
    "Karim Haddad", "Omar Khoury", "Ali Mansour", "Fadi Nassar", "Elie Saad",
    "Georges Karam", "Rami Daher", "Tony Farah", "Nadim Aoun", "Hadi Habib",
    "Ziad Saliba", "Charbel Fares", "Joe Abi Rached", "Marc Chidiac", "Sami Nehme",
    "Layal Haddad", "Christina Khoury", "Marwa Mansour", "Dalia Nassar", "Aya Saad",
    "Zeina Daher", "Reem Farah", "Nadine Karam", "Carla Aoun", "Joelle Habib",
    "Hussein Saliba", "Mahmoud Fares", "Bilal Abi Rached", "Tarek Chidiac", "Bassel Nehme",
    "Anthony Haddad", "Patrick Khoury", "Wael Mansour", "Rabih Nassar", "Samir Saad",
]


def username_from_name(name: str) -> str:
    return name.lower().replace(" ", ".")


def ensure_tables(conn: sqlite3.Connection) -> None:
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

    conn.commit()


def seed_student(conn: sqlite3.Connection, name: str, index: int) -> None:
    username = username_from_name(name)

    random.seed(index)

    profile = StudentProfile(name=username)

    quiz_scores = []

    for quiz_no in range(1, 6):
        topic = TOPICS[(index + quiz_no) % len(TOPICS)]

        if index % 5 == 0:
            score = random.randint(4, 7)
        elif index % 5 == 1:
            score = random.randint(5, 8)
        elif index % 5 == 2:
            score = random.randint(6, 9)
        elif index % 5 == 3:
            score = random.randint(7, 10)
        else:
            score = random.randint(3, 8)

        quiz_scores.append((quiz_no, topic, score))

        correct_count = score
        wrong_count = 10 - score

        for _ in range(correct_count):
            profile.update_performance(topic, True)

        for _ in range(wrong_count):
            profile.update_performance(topic, False)

    conn.execute(
        """
        INSERT OR REPLACE INTO users (username, password, is_admin, profile_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            username,
            hash_password(PASSWORD),
            0,
            json.dumps(profile.to_dict(), indent=2),
        ),
    )

    conn.execute(
        """
        DELETE FROM quiz_attempt_topics
        WHERE username = ?
        """,
        (username,),
    )

    conn.execute(
        """
        DELETE FROM quiz_attempts
        WHERE username = ?
        """,
        (username,),
    )

    for quiz_no, topic, score in quiz_scores:
        percent = round((score / 10) * 100, 2)

        cursor = conn.execute(
            """
            INSERT INTO quiz_attempts (username, quiz_title, quiz_topic, score, total, percent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                username,
                f"Demo Quiz {quiz_no}",
                topic,
                score,
                10,
                percent,
            ),
        )

        attempt_id = int(cursor.lastrowid)

        for q_index in range(1, 11):
            is_correct = 1 if q_index <= score else 0

            conn.execute(
                """
                INSERT INTO quiz_attempt_topics (attempt_id, username, topic, is_correct)
                VALUES (?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    username,
                    topic,
                    is_correct,
                ),
            )


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        ensure_tables(conn)

        for index, name in enumerate(STUDENTS, start=1):
            seed_student(conn, name, index)

        conn.commit()

    print("Demo dataset added successfully.")
    print(f"Students added: {len(STUDENTS)}")
    print("Quizzes per student: 5")
    print("Questions per quiz: 10")
    print(f"Default password for demo students: {PASSWORD}")
    print(f"Database path: {DB_PATH}")


if __name__ == "__main__":
    main()