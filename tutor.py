from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Sequence

from prompts import build_quiz_json_schema, build_system_prompt


class ProbabilityTutor:
    def __init__(self, client: Any, model: str = "gpt-5.2") -> None:
        self.client = client
        self.model = model

    def _text(self, instructions: str, user_input: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            store=False,
        )
        return self._postprocess_math(response.output_text)

    def _input_items(self, instructions: str, input_items: list[dict[str, Any]]) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=input_items,
            store=False,
        )
        return self._postprocess_math(response.output_text)

    def _json(self, instructions: str, user_input: str, schema: dict[str, Any]) -> dict[str, Any]:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "schema": schema["schema"],
                    "strict": True,
                }
            },
            store=False,
        )
        return json.loads(response.output_text)

    def _build_instructions(
        self,
        student_level: str,
        weak_topics: Optional[Dict[str, int]] = None,
    ) -> str:
        return build_system_prompt(student_level, weak_topics or {})

    def _solve_format_block(self) -> str:
        return """Required format:
1. Topic
2. Given
3. Required
4. Formula or main idea
5. Step-by-step solution
6. Final answer
7. Quick check"""

    def _math_style_block(self) -> str:
        return """Math and formatting rules:
- Use inline LaTeX with single dollar signs like $X$, $P(X=3)$, and $P(X \\ge 1)$ when math appears inside a sentence.
- Keep words and inline math on the same line whenever possible.
- Use block LaTeX with $$ ... $$ only for longer standalone derivations or multi-line equations when truly necessary.
- Do not put single symbols like $X$ or short expressions like $P(X=3)$ on their own line.
- Keep ordinary text as plain text.
- Do not put ordinary numbers alone on separate lines.
- Keep the response neat, readable, and student-friendly."""

    def _similar_problem_line(self, include_similar: bool) -> str:
        if include_similar:
            return "After the final answer, include one similar practice problem without solving it."
        return "Do not add a similar practice problem."

    def _history_to_text(
        self,
        chat_history: Optional[Sequence[tuple[str, str]]],
        max_messages: int = 10,
    ) -> str:
        if not chat_history:
            return "No previous conversation."

        recent_messages = chat_history[-max_messages:]
        lines: list[str] = []

        for role, msg in recent_messages:
            speaker = "Student" if role == "user" else "Teacher"
            lines.append(f"{speaker}: {msg}")

        return "\n".join(lines)

    def _teacher_style_block(self, student_level: str) -> str:
        return f"""Teacher behavior:
- Act like a real teacher, not just a chatbot.
- Adapt the explanation to a {student_level} student.
- Teach for understanding, not only for the final answer.
- If the student seems confused, explain again in a simpler way.
- If needed, explain the same idea using a different method.
- Use examples only when they are helpful or requested.
- Be patient, clear, and organized.
- Prefer short, focused explanations over long messy answers."""

    def _chat_decision_block(self) -> str:
        return """How to respond:
- If the student asks for a concept, explain the concept clearly.
- If the student asks for a solution, solve it step by step.
- If the student asks for an example, give an example and explain it.
- If the student asks for practice, generate practice.
- If the student asks for a quiz, create a short quiz.
- If the student asks to continue the previous question, continue naturally from the conversation history.
- If the student makes a mistake, correct it gently and explain why.
- When useful, end with one short check-for-understanding question."""

    def _is_first_teacher_reply(
        self,
        chat_history: Optional[Sequence[tuple[str, str]]],
    ) -> bool:
        if not chat_history:
            return True
        return not any(role == "assistant" for role, _ in chat_history)

    def _course_context_block(self, course_context: str | None) -> str:
        if not course_context:
            return """Course grounding:
- No course material was retrieved for this request.
- Answer normally, but stay within the tutor scope."""
        return f"""Course grounding:
- Use the following course material whenever it is relevant.
- Prefer the course material when it directly supports the answer.
- Do not quote large chunks verbatim.
- If the course material is incomplete for the exact question, say so briefly and then help using sound probability reasoning.

Relevant course material:
{course_context}
"""

    def _postprocess_math(self, text: str) -> str:
        if not text:
            return text

        cleaned = text

        # Convert isolated block math like $$X$$ or $$P(X=3)$$ into inline math.
        cleaned = re.sub(
            r"\$\$\s*([A-Za-z0-9\\{}()[\]=+\-*/.,<>| ]{1,80})\s*\$\$",
            lambda m: f"${m.group(1).strip()}$",
            cleaned,
        )

        # Remove line breaks that split a sentence around very short inline math.
        cleaned = re.sub(r"([A-Za-z0-9,;:])\n+\s*(\$[^$\n]{1,80}\$)", r"\1 \2", cleaned)
        cleaned = re.sub(r"(\$[^$\n]{1,80}\$)\n+\s*([A-Za-z])", r"\1 \2", cleaned)

        # Collapse repeated blank lines a bit.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned

    def solve_problem(
        self,
        problem_text: str,
        student_level: str = "beginner",
        weak_topics: Optional[Dict[str, int]] = None,
        include_similar: bool = True,
        include_hint: bool = False,
        course_context: str | None = None,
    ) -> str:
        instructions = self._build_instructions(student_level, weak_topics)

        hint_line = (
            "Add a short hint before the full solution."
            if include_hint
            else "Do not add a hint before the solution."
        )
        similar_line = self._similar_problem_line(include_similar)

        user_input = f"""
Solve this probability problem carefully and teach it like a real teacher.

{self._course_context_block(course_context)}

Problem:
{problem_text}

{self._solve_format_block()}

{self._teacher_style_block(student_level)}

Extra requirements:
- First explain the idea simply, then solve.
- If useful, include one short intuitive explanation before the formula.
- Be mathematically careful.
- {hint_line}
- {similar_line}

{self._math_style_block()}
"""
        return self._text(instructions, user_input)

    def solve_from_document_text(
        self,
        document_text: str,
        student_level: str = "beginner",
        weak_topics: Optional[Dict[str, int]] = None,
        include_similar: bool = True,
        course_context: str | None = None,
    ) -> str:
        instructions = self._build_instructions(student_level, weak_topics)
        similar_line = self._similar_problem_line(include_similar)

        user_input = f"""
A student uploaded a worksheet or document. Read it and identify the probability question or questions.

{self._course_context_block(course_context)}

Document content:
{document_text[:16000]}

Instructions:
- If there is one main probability question, solve it step by step.
- If there are multiple questions, list them first and solve the first one unless the document clearly asks for all.
- If the text is unclear, clearly say what is unclear.

{self._solve_format_block()}

{self._teacher_style_block(student_level)}

Extra requirements:
- Teach like a helpful teacher, not just a solver.
- {similar_line}

{self._math_style_block()}
"""
        return self._text(instructions, user_input)

    def solve_from_image(
        self,
        image_data_url: str,
        student_level: str = "beginner",
        weak_topics: Optional[Dict[str, int]] = None,
        include_similar: bool = True,
        course_context: str | None = None,
    ) -> str:
        instructions = self._build_instructions(student_level, weak_topics)

        input_items = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Read this uploaded worksheet image carefully.

{self._course_context_block(course_context)}

Tasks:
- Extract the probability question accurately.
- If multiple questions appear, list them first and solve the first one unless the worksheet clearly asks for all.

{self._solve_format_block()}

{self._teacher_style_block(student_level)}

Extra requirements:
- Teach like a patient teacher, not just a solver.
- {self._similar_problem_line(include_similar)}

{self._math_style_block()}
""".strip(),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_data_url,
                    },
                ],
            }
        ]
        return self._input_items(instructions, input_items)

    def generate_exercise(
        self,
        topic: str,
        difficulty: str = "beginner",
        weak_topics: Optional[Dict[str, int]] = None,
        with_solution: bool = False,
        course_context: str | None = None,
    ) -> str:
        instructions = self._build_instructions(difficulty, weak_topics)

        solution_part = (
            "Include a full step-by-step solution after the exercise."
            if with_solution
            else "Include only the exercise and one short hint."
        )

        user_input = f"""
Generate one original probability exercise.

{self._course_context_block(course_context)}

Topic: {topic}
Difficulty: {difficulty}

Requirements:
- The exercise should align with the retrieved course material when possible.
- The exercise must be clear and unambiguous.
- Avoid repeating a very common textbook example unless you make it slightly original.
- Keep the wording student-friendly.
- Match the exercise to the student's level.
- {solution_part}

{self._math_style_block()}
"""
        return self._text(instructions, user_input)

    def generate_structured_quiz(
        self,
        topic: str,
        difficulty: str,
        num_questions: int,
        style: str,
        weak_topics: Optional[Dict[str, int]] = None,
        course_context: str | None = None,
    ) -> dict[str, Any]:
        instructions = self._build_instructions(difficulty, weak_topics)
        schema = build_quiz_json_schema()

        if topic == "All Topics":
            topic_instruction = """
Quiz coverage:
- Generate a GENERAL quiz that covers MULTIPLE topics across the full probability and introductory statistics syllabus.
- Distribute questions across DIFFERENT topics when possible.
- Try to cover multiple chapters such as descriptive statistics, probability rules, discrete random variables, continuous random variables, and joint distributions/random samples.
- Do NOT focus the whole quiz on a single topic.
- Set each question's "topic" field to the specific topic used for that question.
"""
        else:
            topic_instruction = f"""
Quiz coverage:
- Main topic: {topic}
- Keep the quiz focused mainly on this topic.
- Set each question's "topic" field appropriately.
"""

        user_input = f"""
Create a probability quiz in JSON.

{self._course_context_block(course_context)}

{topic_instruction}

Difficulty: {difficulty}
Number of questions: {num_questions}
Question style: {style}

Requirements:
- Ground the quiz in the retrieved course material when possible.
- Each question must include:
  - topic
  - question
  - answer
  - explanation
  - hint
- For multiple choice questions, include exactly 4 options and the answer must exactly match one option.
- For short-answer questions, keep the final answer concise.
- Make the quiz suitable for a student at the given difficulty level.
- Keep explanations clear and teacher-like.
- If the quiz topic is "All Topics", diversify the questions across different topics instead of repeating the same one.
"""
        return self._json(instructions, user_input, schema)

    def chat(
        self,
        user_message: str,
        student_level: str = "beginner",
        weak_topics: Optional[Dict[str, int]] = None,
        chat_history: Optional[Sequence[tuple[str, str]]] = None,
        course_context: str | None = None,
    ) -> str:
        instructions = self._build_instructions(student_level, weak_topics)
        history_text = self._history_to_text(chat_history)
        is_first_message = self._is_first_teacher_reply(chat_history)

        identity_line = (
            'Start the reply with exactly: "Hi! I’m Maram Dalal." Then continue directly without adding another greeting.'
            if is_first_message
            else 'Do not mention your name unless the user explicitly asks for it.'
        )

        user_input = f"""
Reply as a real adaptive probability teacher.

{self._course_context_block(course_context)}

Previous conversation:
{history_text}

New student message:
{user_message}

Identity behavior:
- {identity_line}

{self._teacher_style_block(student_level)}

{self._chat_decision_block()}

Conversation behavior:
- Continue naturally from the previous conversation when relevant.
- If the student refers to an earlier question, use the conversation history.
- Do not restart the topic unless needed.
- If the student asks "why", explain the reason clearly.
- If the student asks "another way", explain using a different method.
- If the student asks "I don't understand", simplify and slow down.
- If the student asks for examples, give examples only if they are requested or clearly useful.
- If the student asks for practice, generate practice.
- If the student asks for a quiz, create a quiz.
- If the student asks about a previous mistake, point out the exact misunderstanding gently.
- If the student asks to continue, continue from the last question or explanation.

Very important behavior:
- If the student only greets you or says something casual like "hi", "hello", "thanks", "ok", or similar, reply naturally and briefly.
- In that case, do not start teaching, do not give examples, do not give practice, and do not give a warm-up question unless the student asks.
- If the student asks a real math question, then switch into teacher mode and teach clearly.
- Do not repeat greetings. Only greet once at the start of the first reply.
- Do not give two greetings in the same reply.

Response style:
- Sound like a teacher talking to a student.
- Be clear, natural, and direct.
- Do not be robotic.
- Do not dump formulas unless needed.
- Start with intuition when useful, then show the math.
- Prefer understanding over memorization.

{self._math_style_block()}
"""
        return self._text(instructions, user_input)