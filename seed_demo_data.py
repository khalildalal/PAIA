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


def score_for_group(index: int) -> int:
    """
    Create realistic different levels:
    - first 15 students: advanced
    - next 25 students: intermediate
    - last 10 students: beginner
    """
    if index <= 15:
        return random.randint(8, 10)
    if index <= 40:
        return random.randint(5, 8)
    return random.randint(2, 5)


def force_level_from_accuracy(profile: StudentProfile) -> None:
    """
    Demo-data rule:
    Force the displayed level to match quiz performance clearly.
    This avoids high-accuracy students staying intermediate because of average mastery.
    """
    acc = profile.accuracy()

    if acc >= 0.82:
        profile.level = "advanced"
    elif acc >= 0.55:
        profile.level = "intermediate"
    else:
        profile.level = "beginner"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        for index, name in enumerate(STUDENTS, start=1):
            username = username_from_name(name)
            random.seed(index)

            profile = StudentProfile(name=username)

            # Force student profile topics to match RAG topics only.
            profile.topic_mastery = {topic: 0.5 for topic in TOPICS}
            profile.weak_topics = {}

            conn.execute(
                """
                INSERT OR REPLACE INTO users (username, password, is_admin, profile_json)
                VALUES (?, ?, ?, ?)
                """,
                (username, hash_password(PASSWORD), 0, json.dumps({})),
            )

            conn.execute("DELETE FROM quiz_attempt_topics WHERE username = ?", (username,))
            conn.execute("DELETE FROM quiz_attempts WHERE username = ?", (username,))

            for quiz_no in range(1, 6):
                topic = TOPICS[(index + quiz_no) % len(TOPICS)]
                score = score_for_group(index)
                percent = round((score / 10) * 100, 2)

                results = []

                for q_no in range(1, 11):
                    correct = q_no <= score
                    profile.update_performance(topic, correct)

                    results.append({
                        "topic": topic,
                        "is_correct": correct,
                    })

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

                attempt_id = cursor.lastrowid

                for item in results:
                    conn.execute(
                        """
                        INSERT INTO quiz_attempt_topics (attempt_id, username, topic, is_correct)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            attempt_id,
                            username,
                            item["topic"],
                            1 if item["is_correct"] else 0,
                        ),
                    )

            force_level_from_accuracy(profile)

            conn.execute(
                """
                UPDATE users
                SET profile_json = ?
                WHERE username = ?
                """,
                (json.dumps(profile.to_dict(), indent=2), username),
            )

        conn.commit()

    print("DONE: 50 Lebanese student users added.")
    print("Levels include advanced, intermediate, and beginner.")
    print("Each user has 5 quizzes.")
    print("Each quiz has 10 question results.")
    print("Topics are based on RAG chapters.")
    print("Password for all demo students: student123")
    print(f"Database used: {DB_PATH}")


if __name__ == "__main__":
    main()