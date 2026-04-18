"""
Quiz grading engine

This file contains the logic for checking user quiz answers.

What this file does:
1. Normalizes answer text
2. Tries to interpret numeric answers
3. Compares expected answers with user answers
4. Grades an entire quiz
5. Produces detailed per-question feedback

Why this matters:
The main app creates quizzes, but this file decides whether answers are correct.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any, Dict, List


class QuizEngine:
    """Grade quizzes and compare student answers with expected answers."""
    def normalize(self, value: str) -> str:
        """Lowercase and simplify text so answer comparisons become more reliable."""
        return re.sub(r"\s+", " ", str(value).strip().lower())

    def try_parse_number(self, text: str):
        """Try to convert a text answer into a numeric value."""
        if text is None:
            return None

        raw = self.normalize(text)
        if not raw:
            return None

        raw = raw.replace("%", " /100")
        raw = raw.replace("^", "**")
        raw = raw.replace("=", " ")
        raw = raw.strip()

        try:
            if "/" in raw and all(part.strip().replace("-", "").isdigit() for part in raw.split("/", 1)):
                return float(Fraction(raw))
            return float(eval(raw, {"__builtins__": {}}, {}))
        except Exception:
            return None

    def answers_match(self, expected: str, user: str) -> bool:
        """Check whether the expected answer and user answer should count as the same."""
        a = self.normalize(expected)
        b = self.normalize(user)

        if not b:
            return False

        if a == b:
            return True

        na = self.try_parse_number(a)
        nb = self.try_parse_number(b)
        if na is not None and nb is not None and math.isclose(na, nb, rel_tol=1e-6, abs_tol=1e-9):
            return True

        compact_a = a.replace(" ", "")
        compact_b = b.replace(" ", "")
        return compact_a == compact_b

    def grade_quiz(self, quiz: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """Grade all questions in a quiz and return score details and explanations."""
        results: List[Dict[str, Any]] = []
        score = 0

        for idx, question in enumerate(quiz["questions"], start=1):
            user_answer = answers.get(str(idx), "") or ""
            expected_answer = question["answer"]
            is_correct = self.answers_match(expected_answer, user_answer)

            if is_correct:
                score += 1

            results.append(
                {
                    "index": idx,
                    "topic": question.get("topic", quiz.get("topic", "Basic Probability")),
                    "user_answer": user_answer,
                    "expected_answer": expected_answer,
                    "is_correct": is_correct,
                    "explanation": question.get("explanation", "No explanation available."),
                }
            )

        total = len(quiz["questions"])
        percent = (score / total) * 100 if total else 0.0

        if percent >= 85:
            feedback = "Excellent work. You are ready for harder probability practice."
        elif percent >= 60:
            feedback = "Good work. You understand a lot already, but a little more practice will help."
        else:
            feedback = "Keep going. Focus on the explanations and repeat your weakest topics."

        return {
            "score": score,
            "total": total,
            "percent": percent,
            "results": results,
            "feedback": feedback,
        }
