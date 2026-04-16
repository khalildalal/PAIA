from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_TOPICS = [
    "Basic Probability",
    "Conditional Probability",
    "Bayes' Theorem",
    "Permutations",
    "Combinations",
    "Random Variables",
    "Expected Value",
]


@dataclass
class StudentProfile:
    name: str
    level: str = "beginner"
    correct_answers: int = 0
    wrong_answers: int = 0
    weak_topics: Dict[str, int] = field(default_factory=dict)
    topic_mastery: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        for topic in DEFAULT_TOPICS:
            self.topic_mastery.setdefault(topic, 0.5)

    def update_performance(self, topic: str, correct: bool) -> None:
        if topic not in self.topic_mastery:
            self.topic_mastery[topic] = 0.5

        if correct:
            self.correct_answers += 1
            self.topic_mastery[topic] = min(1.0, self.topic_mastery[topic] + 0.08)
            if topic in self.weak_topics:
                self.weak_topics[topic] -= 1
                if self.weak_topics[topic] <= 0:
                    del self.weak_topics[topic]
        else:
            self.wrong_answers += 1
            self.topic_mastery[topic] = max(0.0, self.topic_mastery[topic] - 0.12)
            self.weak_topics[topic] = self.weak_topics.get(topic, 0) + 1

        self.adjust_level()

    def adjust_level(self) -> None:
        total = self.correct_answers + self.wrong_answers
        if total < 6:
            self.level = "beginner"
            return

        acc = self.accuracy()
        avg_mastery = sum(self.topic_mastery.values()) / len(self.topic_mastery)

        if acc >= 0.82 and avg_mastery >= 0.70:
            self.level = "advanced"
        elif acc >= 0.55 and avg_mastery >= 0.45:
            self.level = "intermediate"
        else:
            self.level = "beginner"

    def accuracy(self) -> float:
        total = self.correct_answers + self.wrong_answers
        return 0.0 if total == 0 else self.correct_answers / total

    def get_weak_topics(self) -> Dict[str, int]:
        return dict(sorted(self.weak_topics.items(), key=lambda item: item[1], reverse=True))

    def weakest_topic(self) -> str | None:
        return next(iter(self.get_weak_topics().keys()), None)

    def recommend_next_topic(self) -> str:
        weakest = self.weakest_topic()
        if weakest:
            return weakest
        return min(self.topic_mastery, key=self.topic_mastery.get)

    def topic_list(self) -> List[str]:
        items = list(self.topic_mastery.keys())
        for topic in DEFAULT_TOPICS:
            if topic not in items:
                items.append(topic)
        return items

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "level": self.level,
            "correct_answers": self.correct_answers,
            "wrong_answers": self.wrong_answers,
            "weak_topics": self.weak_topics,
            "topic_mastery": self.topic_mastery,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StudentProfile":
        if not data:
            return cls(name="Student")
        return cls(
            name=data.get("name", "Student"),
            level=data.get("level", "beginner"),
            correct_answers=int(data.get("correct_answers", 0)),
            wrong_answers=int(data.get("wrong_answers", 0)),
            weak_topics=data.get("weak_topics", {}) or {},
            topic_mastery=data.get("topic_mastery", {}) or {},
        )
