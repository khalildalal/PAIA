from __future__ import annotations

import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from openai import OpenAI
from werkzeug.utils import secure_filename

from database import ProgressStore
from document_utils import (
    chunk_text,
    detect_kind,
    encode_image_to_data_url,
    extract_document_text,
)
from quiz_engine import QuizEngine
from student_model import StudentProfile
from tutor import ProbabilityTutor

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.getenv("DATA_DIR")
    or os.getenv("RAILWAY_VOLUME_MOUNT_PATH")
    or str(BASE_DIR / "data")
)
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

store = ProgressStore(str(DATA_DIR / "student_profiles.db"))
quiz_engine = QuizEngine()

ALLOWED_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
ALLOWED_DOCUMENT_EXTENSIONS = {"pdf", "docx", "txt", "md"}
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# -----------------------------
# Syllabus mapping
# -----------------------------
SYLLABUS_TOPIC_MAP = {
    "Chapter 1: Descriptive Statistics": [
        "descriptive statistics",
        "statistics",
        "population",
        "sample",
        "categorical variable",
        "discrete variable",
        "continuous variable",
        "stem-and-leaf",
        "stem and leaf",
        "dotplot",
        "histogram",
        "frequency",
        "relative frequency",
        "mean",
        "median",
        "mode",
        "range",
        "variance",
        "standard deviation",
        "quartile",
        "percentile",
        "boxplot",
    ],
    "Chapter 2: Probability": [
        "probability",
        "sample space",
        "event",
        "simple event",
        "compound event",
        "axiom",
        "axioms",
        "counting",
        "product rule",
        "conditional probability",
        "bayes",
        "bayes theorem",
        "independence",
        "independent events",
        "union",
        "intersection",
        "complement",
        "mutually exclusive",
        "disjoint",
        "counting techniques",
    ],
    "Chapter 3: Discrete Random Variables and Probability Distributions": [
        "random variable",
        "discrete random variable",
        "bernoulli",
        "pmf",
        "probability mass function",
        "cdf",
        "expected value",
        "binomial",
        "hypergeometric",
        "negative binomial",
        "poisson",
        "poisson process",
        "geometric distribution",
        "probability distribution",
        "discrete distribution",
    ],
    "Chapter 4: Continuous Random Variables and Probability Distributions": [
        "continuous random variable",
        "continuous distribution",
        "pdf",
        "probability density function",
        "density curve",
        "uniform distribution",
        "normal distribution",
        "exponential distribution",
        "gamma distribution",
        "chi-squared",
        "chi squared",
        "continuous cdf",
        "percentiles of a continuous distribution",
        "continuous probability",
    ],
    "Chapter 5: Joint Probability Distributions and Random Samples": [
        "joint distribution",
        "joint pmf",
        "joint pdf",
        "jointly distributed",
        "marginal",
        "conditional distribution",
        "covariance",
        "correlation",
        "random samples",
        "random sample",
        "central limit theorem",
        "clt",
        "linear combination",
        "independent random variables",
        "statistics and random samples",
    ],
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i",
    "if", "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "what", "when", "where", "which", "with", "you", "your", "find", "explain",
    "solve", "calculate", "compute", "give", "show", "derive", "about", "let",
    "then", "than", "using", "use", "student", "topic", "difficulty"
}


# -----------------------------
# OpenAI / Tutor
# -----------------------------
def get_client() -> Optional[OpenAI]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def get_tutor() -> Optional[ProbabilityTutor]:
    client = get_client()
    if client is None:
        return None
    return ProbabilityTutor(client=client)


def embed_text(text: str) -> list[float]:
    client = get_client()
    if client is None:
        raise RuntimeError("OPENAI_API_KEY is missing.")

    clean = " ".join((text or "").split()).strip()
    if not clean:
        return []

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=clean,
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return -1.0

    return dot / (norm_a * norm_b)


# -----------------------------
# Retrieval helpers
# -----------------------------
def normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split()).strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9\-\+']+", normalize_text(text))
    return [t for t in tokens if t and t not in STOPWORDS and len(t) > 1]


def build_topic_query(topic: str, difficulty: str, weak_topics: dict[str, int]) -> str:
    weak_text = ", ".join(weak_topics.keys()) if weak_topics else "None"
    return (
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty}\n"
        f"Student weak topics: {weak_text}\n"
        "Retrieve course material relevant to teaching or assessing this topic."
    )


def detect_course_code_from_syllabus(query: str) -> str | None:
    q = normalize_text(query)
    if not q:
        return None

    scores = {code: 0 for code in SYLLABUS_TOPIC_MAP}

    for code, terms in SYLLABUS_TOPIC_MAP.items():
        for term in terms:
            term_norm = normalize_text(term)
            if term_norm in q:
                scores[code] += 3 if " " in term_norm else 1

    best_code = max(scores, key=scores.get)
    return best_code if scores[best_code] > 0 else None


def bm25_scores(query: str, chunks: list[dict], k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
    query_terms = tokenize(query)
    if not query_terms or not chunks:
        return {}

    doc_tf: dict[int, Counter] = {}
    doc_len: dict[int, int] = {}
    df: defaultdict[str, int] = defaultdict(int)

    for chunk in chunks:
        chunk_id = int(chunk["id"])
        tokens = tokenize(chunk.get("chunk_text", ""))
        tf = Counter(tokens)
        doc_tf[chunk_id] = tf
        doc_len[chunk_id] = len(tokens)

        for term in set(tf.keys()):
            df[term] += 1

    n_docs = len(chunks)
    avgdl = sum(doc_len.values()) / n_docs if n_docs else 0.0
    if avgdl == 0:
        return {}

    scores: dict[int, float] = {}

    for chunk in chunks:
        chunk_id = int(chunk["id"])
        tf = doc_tf[chunk_id]
        dl = doc_len[chunk_id]
        score = 0.0

        for term in query_terms:
            if term not in tf:
                continue

            term_df = df.get(term, 0)
            idf = math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            freq = tf[term]
            denom = freq + k1 * (1 - b + b * (dl / avgdl))
            score += idf * ((freq * (k1 + 1)) / denom)

        scores[chunk_id] = score

    return scores


def retrieve_course_context(
    query: str,
    *,
    top_k: int = 4,
    course_code: str | None = None,
) -> str:
    clean_query = " ".join((query or "").split()).strip()
    if not clean_query:
        return ""

    chunks = store.get_course_chunks(course_code=course_code)
    if not chunks:
        return ""

    try:
        query_embedding = embed_text(clean_query)
    except Exception:
        query_embedding = []

    semantic_scores: dict[int, float] = {}
    if query_embedding:
        for chunk in chunks:
            chunk_id = int(chunk["id"])
            emb = chunk.get("embedding") or []
            semantic_scores[chunk_id] = max(0.0, cosine_similarity(query_embedding, emb))

    lexical_scores = bm25_scores(clean_query, chunks)

    max_sem = max(semantic_scores.values()) if semantic_scores else 0.0
    max_lex = max(lexical_scores.values()) if lexical_scores else 0.0

    scored: list[tuple[float, dict, float, float]] = []
    for chunk in chunks:
        chunk_id = int(chunk["id"])

        sem = semantic_scores.get(chunk_id, 0.0)
        lex = lexical_scores.get(chunk_id, 0.0)

        sem_norm = sem / max_sem if max_sem > 0 else 0.0
        lex_norm = lex / max_lex if max_lex > 0 else 0.0

        hybrid = 0.65 * sem_norm + 0.35 * lex_norm

        if hybrid > 0:
            scored.append((hybrid, chunk, sem_norm, lex_norm))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[:top_k]

    if not selected:
        return ""

    context_parts: list[str] = []
    for rank, (score, chunk, sem_norm, lex_norm) in enumerate(selected, start=1):
        title = chunk.get("title", "Untitled")
        code = chunk.get("course_code", "Course")
        idx = chunk.get("chunk_index", 0)
        text = chunk.get("chunk_text", "")
        context_parts.append(
            f"[Source {rank} | {code} | {title} | chunk {idx} | hybrid {score:.3f} | semantic {sem_norm:.3f} | bm25 {lex_norm:.3f}]\n{text}"
        )

    return "\n\n".join(context_parts)


def retrieve_course_context_auto(query: str, top_k: int = 4) -> tuple[str, str | None]:
    detected = detect_course_code_from_syllabus(query)

    if detected:
        focused_context = retrieve_course_context(query, top_k=top_k, course_code=detected)
        if focused_context.strip():
            return focused_context, detected

    return retrieve_course_context(query, top_k=top_k), detected


# -----------------------------
# Session / User helpers
# -----------------------------
def current_username() -> Optional[str]:
    return session.get("user")


def is_authenticated() -> bool:
    return current_username() is not None


def load_student() -> StudentProfile:
    username = current_username()
    if not username:
        return StudentProfile(name="Student")

    saved = store.load_profile(username)
    student = StudentProfile.from_dict(saved) if saved else StudentProfile(name=username)
    student.name = username
    return student


def save_student(student: StudentProfile) -> None:
    username = current_username()
    if not username:
        return
    student.name = username
    store.save_profile(username, student.to_dict())


def get_chat_history() -> list[tuple[str, str]]:
    raw = session.get("chat_history", [])
    history: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            history.append((item.get("role", "user"), item.get("message", "")))
    return history


def set_chat_history(history: list[tuple[str, str]]) -> None:
    session["chat_history"] = [{"role": role, "message": msg} for role, msg in history]


def clear_learning_state() -> None:
    session.pop("chat_history", None)
    session.pop("current_quiz", None)
    session.pop("quiz_results", None)


def get_theme() -> str:
    theme = request.cookies.get("theme_preference", "system")
    if theme not in {"light", "dark", "system"}:
        return "system"
    return theme


# -----------------------------
# Admin analytics
# -----------------------------
def build_admin_dataframe(users: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for user in users:
        profile = user.get("profile") or {}
        correct = int(profile.get("correct_answers", 0) or 0)
        wrong = int(profile.get("wrong_answers", 0) or 0)
        total = correct + wrong
        accuracy = round((correct / total) * 100, 2) if total else 0.0
        weak_topics = profile.get("weak_topics", {}) or {}
        weakest_topic = next(iter(weak_topics.keys()), "None")

        rows.append(
            {
                "username": str(user["username"]),
                "role": "Admin" if user["is_admin"] else "Student",
                "level": str(profile.get("level", "beginner")),
                "correct_answers": correct,
                "wrong_answers": wrong,
                "total_answers": total,
                "accuracy_percent": accuracy,
                "weak_topics_count": len(weak_topics),
                "weakest_topic": str(weakest_topic),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    numeric_cols = [
        "correct_answers",
        "wrong_answers",
        "total_answers",
        "accuracy_percent",
        "weak_topics_count",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    return df


def plotly_template_for_theme(theme: str) -> str:
    return "plotly_dark" if theme != "light" else "plotly_white"


def _nice_axis_top(max_value: float, *, minimum: float = 1.0, headroom_ratio: float = 0.18, extra: float = 0.0) -> float:
    if max_value <= 0:
        return minimum
    padded = max_value * (1.0 + headroom_ratio) + extra
    return max(minimum, padded)


def build_admin_figures(
    df: pd.DataFrame,
    theme: str,
    selected_topic: str,
    selected_student: str,
) -> tuple[dict[str, str], list[str], list[str]]:
    figures: dict[str, str] = {}
    template = plotly_template_for_theme(theme)

    topic_names = store.list_quiz_topics()
    student_names = store.list_student_usernames()

    if not df.empty:
        student_df = df[df["role"] == "Student"].copy()

        if not student_df.empty:
            accuracy_df = df.copy().sort_values("accuracy_percent", ascending=False)
            max_accuracy = float(accuracy_df["accuracy_percent"].max()) if not accuracy_df.empty else 0.0
            accuracy_top = min(110.0, _nice_axis_top(max_accuracy, minimum=10.0, headroom_ratio=0.12, extra=4.0))

            fig_accuracy = px.bar(
                accuracy_df,
                x="username",
                y="accuracy_percent",
                color="role",
                title="Accuracy by User",
                text="accuracy_percent",
                barmode="group",
                color_discrete_map={"Admin": "#60A5FA", "Student": "#A78BFA"},
                hover_data={
                    "level": True,
                    "correct_answers": True,
                    "wrong_answers": True,
                    "accuracy_percent": ":.2f",
                },
                template=template,
            )
            fig_accuracy.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
            fig_accuracy.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=60, b=40),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                yaxis=dict(range=[0, accuracy_top], automargin=True),
                xaxis=dict(automargin=True),
            )
            figures["accuracy"] = fig_accuracy.to_json()

            level_counts = (
                student_df.groupby("level", as_index=False)
                .size()
                .rename(columns={"size": "count"})
            )
            fig_levels = px.pie(
                level_counts,
                names="level",
                values="count",
                title="Student Level Distribution",
                hole=0.6,
                color="level",
                color_discrete_map={
                    "beginner": "#F59E0B",
                    "intermediate": "#38BDF8",
                    "advanced": "#34D399",
                },
                template=template,
            )
            fig_levels.update_layout(height=380, margin=dict(l=20, r=20, t=60, b=20))
            figures["levels"] = fig_levels.to_json()

            engagement_df = student_df.copy().sort_values("total_answers", ascending=False)
            max_engagement = float(engagement_df["total_answers"].max()) if not engagement_df.empty else 0.0
            engagement_top = _nice_axis_top(max_engagement, minimum=5.0, headroom_ratio=0.22, extra=1.0)

            fig_engagement = px.bar(
                engagement_df,
                x="username",
                y="total_answers",
                title="Engagement by Student",
                text="total_answers",
                color_discrete_sequence=["#A78BFA"],
                hover_data={
                    "correct_answers": True,
                    "wrong_answers": True,
                    "total_answers": True,
                },
                template=template,
            )
            fig_engagement.update_traces(textposition="outside", cliponaxis=False)
            fig_engagement.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=60, b=40),
                showlegend=False,
                yaxis=dict(range=[0, engagement_top], automargin=True),
                xaxis=dict(automargin=True),
            )
            figures["engagement"] = fig_engagement.to_json()

            weak_df = student_df.copy().sort_values("weak_topics_count", ascending=False)
            max_weak = float(weak_df["weak_topics_count"].max()) if not weak_df.empty else 0.0
            weak_top = _nice_axis_top(max_weak, minimum=5.0, headroom_ratio=0.22, extra=1.0)

            fig_weak = px.bar(
                weak_df,
                x="username",
                y="weak_topics_count",
                title="Weak Topics by Student",
                text="weak_topics_count",
                color_discrete_sequence=["#F472B6"],
                hover_data={
                    "weakest_topic": True,
                    "weak_topics_count": True,
                },
                template=template,
            )
            fig_weak.update_traces(textposition="outside", cliponaxis=False)
            fig_weak.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=60, b=40),
                showlegend=False,
                yaxis=dict(range=[0, weak_top], automargin=True),
                xaxis=dict(automargin=True),
            )
            figures["weak"] = fig_weak.to_json()

    selected_topic_value = selected_topic if selected_topic else "All Topics"
    topic_rows = store.get_students_per_topic(selected_topic_value)
    if topic_rows:
        topic_df = pd.DataFrame(topic_rows)
        max_students = float(topic_df["student_count"].max()) if not topic_df.empty else 0.0
        students_top = _nice_axis_top(max_students, minimum=2.0, headroom_ratio=0.22, extra=1.0)

        fig_topic_students = px.bar(
            topic_df,
            x="topic",
            y="student_count",
            text="student_count",
            title="Students per Topic",
            color_discrete_sequence=["#22C55E"],
            template=template,
        )
        fig_topic_students.update_traces(textposition="outside", cliponaxis=False)
        fig_topic_students.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=60, b=60),
            showlegend=False,
            yaxis=dict(range=[0, students_top], automargin=True),
            xaxis=dict(automargin=True),
        )
        figures["topic_students"] = fig_topic_students.to_json()

    selected_student_value = selected_student.strip() if selected_student else ""
    if selected_student_value:
        progress_rows = store.get_quiz_progress_for_student(selected_student_value)
        if progress_rows:
            progress_df = pd.DataFrame(progress_rows)
            progress_top = min(
                110.0,
                _nice_axis_top(float(progress_df["percent"].max()), minimum=10.0, headroom_ratio=0.15, extra=3.0),
            )

            fig_progress = px.line(
                progress_df,
                x="attempt_no",
                y="percent",
                markers=True,
                hover_data={
                    "quiz_title": True,
                    "quiz_topic": True,
                    "score": True,
                    "total": True,
                    "created_at": True,
                },
                title=f"Quiz-to-Quiz Progress for {selected_student_value}",
                template=template,
            )
            fig_progress.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=60, b=40),
                yaxis=dict(range=[0, progress_top], automargin=True, title="Score (%)"),
                xaxis=dict(automargin=True, title="Quiz Attempt"),
            )
            figures["student_progress"] = fig_progress.to_json()

    return figures, topic_names, student_names


# -----------------------------
# Rendering / guards
# -----------------------------
def render_page(
    template_name: str,
    *,
    page_title: str,
    active_tab: str,
    **context,
):
    student = load_student() if is_authenticated() else None
    return render_template(
        template_name,
        page_title=page_title,
        active_tab=active_tab,
        student=student,
        current_user=current_username(),
        is_admin=store.is_admin(current_username() or ""),
        theme_mode=get_theme(),
        **context,
    )


def ensure_logged_in():
    if not is_authenticated():
        return redirect(url_for("login"))
    return None


def persist_quiz(quiz: dict | None) -> None:
    if quiz is None:
        session.pop("current_quiz", None)
    else:
        session["current_quiz"] = quiz


def get_persisted_quiz() -> Optional[dict]:
    quiz = session.get("current_quiz")
    return quiz if isinstance(quiz, dict) else None


def persist_quiz_results(results: dict | None) -> None:
    if results is None:
        session.pop("quiz_results", None)
    else:
        session["quiz_results"] = results


def get_persisted_quiz_results() -> Optional[dict]:
    results = session.get("quiz_results")
    return results if isinstance(results, dict) else None


# -----------------------------
# Upload helpers
# -----------------------------
def save_temp_upload(uploaded_file) -> str:
    suffix = ""
    filename = secure_filename(uploaded_file.filename or "")
    if "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded_file.save(tmp.name)
        return tmp.name


def index_course_file(temp_path: str, filename: str, course_code: str, title: str) -> tuple[bool, str]:
    kind = detect_kind(filename)
    if kind != "document":
        return False, "Unsupported document type."

    extracted = extract_document_text(temp_path, filename)
    if not extracted:
        return False, "Could not extract text from the uploaded file."

    chunks = chunk_text(extracted, chunk_size=900, overlap=150)
    if not chunks:
        return False, "No usable text was found after chunking."

    try:
        embeddings = []
        for chunk in chunks:
            embeddings.append(embed_text(chunk))
    except Exception as exc:
        return False, f"Embedding failed: {exc}"

    document_id = store.add_course_document(
        course_code=course_code,
        title=title,
        filename=filename,
    )

    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings), start=1):
        store.add_course_chunk(
            document_id=document_id,
            chunk_index=idx,
            chunk_text=chunk,
            embedding=emb,
        )

    return True, f"Indexed {len(chunks)} chunks successfully."


def bootstrap_default_courses() -> None:
    default_courses = [
        ("Chapter 1: Descriptive Statistics", "Chapter 1: Descriptive Statistics", "C1.pdf"),
        ("Chapter 2: Probability", "Chapter 2: Probability", "C2.pdf"),
        (
            "Chapter 3: Discrete Random Variables and Probability Distributions",
            "Chapter 3: Discrete Random Variables and Probability Distributions",
            "C3.pdf",
        ),
        (
            "Chapter 4: Continuous Random Variables and Probability Distributions",
            "Chapter 4: Continuous Random Variables and Probability Distributions",
            "C4.pdf",
        ),
        (
            "Chapter 5: Joint Probability Distributions and Random Samples",
            "Chapter 5: Joint Probability Distributions and Random Samples",
            "C5.pdf",
        ),
    ]

    existing = store.list_course_documents()
    existing_codes = {doc["course_code"] for doc in existing}

    for course_code, title, filename in default_courses:
        if course_code in existing_codes:
            continue

        pdf_path = BASE_DIR / filename
        if not pdf_path.exists():
            continue

        ok, message = index_course_file(
            temp_path=str(pdf_path),
            filename=filename,
            course_code=course_code,
            title=title,
        )
        print(f"[RAG bootstrap] {course_code}: {message}")


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    if is_authenticated():
        return redirect(url_for("solve"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if is_authenticated():
        return redirect(url_for("solve"))

    if request.method == "POST":
        action = request.form.get("action", "login")

        if action == "login":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if store.authenticate(username, password):
                session["user"] = username
                clear_learning_state()
                flash("Logged in successfully.", "success")
                return redirect(url_for("solve"))
            flash("Invalid username or password.", "error")

        elif action == "register":
            username = request.form.get("new_username", "").strip()
            password = request.form.get("new_password", "")
            if store.create_user(username, password):
                flash("User created successfully. You can now log in.", "success")
            else:
                flash("User already exists or the input is invalid.", "error")

    return render_page("login.html", page_title="Login", active_tab="login")


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("Logged out.", "success")
    return redirect(url_for("login"))


@app.route("/solve", methods=["GET", "POST"])
def solve():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    tutor = get_tutor()
    answer = None
    extracted_text = None
    input_mode = request.form.get("input_mode", "Text")
    include_similar = request.form.get("include_similar") == "on" or request.method == "GET"
    show_hint = request.form.get("show_hint") == "on"

    if request.method == "POST":
        if tutor is None:
            flash("OPENAI_API_KEY is missing. Set it before running the app.", "error")
        else:
            student = load_student()

            if input_mode == "Text":
                problem = request.form.get("problem_text", "").strip()
                if not problem:
                    flash("Please enter an exercise.", "error")
                else:
                    course_context, detected_course = retrieve_course_context_auto(problem, top_k=4)
                    if detected_course:
                        flash(f"Using course: {detected_course}", "success")

                    answer = tutor.solve_problem(
                        problem_text=problem,
                        student_level=student.level,
                        weak_topics=student.get_weak_topics(),
                        include_similar=include_similar,
                        include_hint=show_hint,
                        course_context=course_context,
                    )

            elif input_mode == "Image":
                uploaded = request.files.get("image_file")
                if uploaded is None or uploaded.filename == "":
                    flash("Please upload an image.", "error")
                else:
                    ext = uploaded.filename.rsplit(".", 1)[-1].lower()
                    if ext not in ALLOWED_IMAGE_EXTENSIONS:
                        flash("Unsupported image type.", "error")
                    else:
                        temp_path = save_temp_upload(uploaded)
                        mime = uploaded.mimetype or "image/png"
                        image_url = encode_image_to_data_url(temp_path, mime)
                        query_hint = f"Probability problem from uploaded image: {uploaded.filename}"
                        course_context, detected_course = retrieve_course_context_auto(query_hint, top_k=4)
                        if detected_course:
                            flash(f"Using course: {detected_course}", "success")

                        answer = tutor.solve_from_image(
                            image_data_url=image_url,
                            student_level=student.level,
                            weak_topics=student.get_weak_topics(),
                            include_similar=include_similar,
                            course_context=course_context,
                        )

            else:
                uploaded = request.files.get("document_file")
                if uploaded is None or uploaded.filename == "":
                    flash("Please upload a document.", "error")
                else:
                    filename = secure_filename(uploaded.filename)
                    ext = filename.rsplit(".", 1)[-1].lower()
                    if ext not in ALLOWED_DOCUMENT_EXTENSIONS:
                        flash("Unsupported document type.", "error")
                    else:
                        temp_path = save_temp_upload(uploaded)
                        kind = detect_kind(filename)
                        if kind != "document":
                            flash("Unsupported document type.", "error")
                        else:
                            extracted_text = extract_document_text(temp_path, filename)
                            if not extracted_text:
                                flash("I could not extract text from this file.", "error")
                            else:
                                course_context, detected_course = retrieve_course_context_auto(
                                    extracted_text[:1500],
                                    top_k=4,
                                )
                                if detected_course:
                                    flash(f"Using course: {detected_course}", "success")

                                answer = tutor.solve_from_document_text(
                                    document_text=extracted_text,
                                    student_level=student.level,
                                    weak_topics=student.get_weak_topics(),
                                    include_similar=include_similar,
                                    course_context=course_context,
                                )

    return render_page(
        "solve.html",
        page_title="Solve",
        active_tab="solve",
        answer=answer,
        extracted_text=extracted_text,
        input_mode=input_mode,
        include_similar=include_similar,
        show_hint=show_hint,
    )


@app.route("/practice", methods=["GET", "POST"])
def practice():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    tutor = get_tutor()
    student = load_student()
    exercise = None
    topics = student.topic_list()

    topic_mode = request.form.get("topic_mode", "Choose manually")
    chosen_topic = request.form.get("topic", topics[0] if topics else "Basic Probability")
    difficulty = request.form.get("difficulty", "adaptive")
    with_solution = request.form.get("with_solution") == "on"

    if request.method == "POST":
        if tutor is None:
            flash("OPENAI_API_KEY is missing. Set it before running the app.", "error")
        else:
            topic = chosen_topic
            if topic_mode == "Use weakest topic":
                topic = student.weakest_topic() or chosen_topic
            elif topic_mode == "Adaptive next topic":
                topic = student.recommend_next_topic()

            effective_difficulty = student.level if difficulty == "adaptive" else difficulty
            rag_query = build_topic_query(topic, effective_difficulty, student.get_weak_topics())
            course_context, detected_course = retrieve_course_context_auto(rag_query, top_k=6)
            if detected_course:
                flash(f"Using course: {detected_course}", "success")

            exercise = tutor.generate_exercise(
                topic=topic,
                difficulty=effective_difficulty,
                weak_topics=student.get_weak_topics(),
                with_solution=with_solution,
                course_context=course_context,
            )

    return render_page(
        "practice.html",
        page_title="Practice",
        active_tab="practice",
        topics=topics,
        exercise=exercise,
        topic_mode=topic_mode,
        chosen_topic=chosen_topic,
        difficulty=difficulty,
        with_solution=with_solution,
    )


@app.route("/quiz", methods=["GET", "POST"])
def quiz():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    tutor = get_tutor()
    student = load_student()
    topics = ["All Topics"] + student.topic_list()

    quiz = get_persisted_quiz()
    quiz_results = get_persisted_quiz_results()

    if request.method == "POST":
        action = request.form.get("action", "")

        if action == "create":
            if tutor is None:
                flash("OPENAI_API_KEY is missing. Set it before running the app.", "error")
            else:
                topic_mode = request.form.get("topic_mode", "Choose manually")
                chosen_topic = request.form.get("topic", topics[0] if topics else "All Topics")
                difficulty = request.form.get("difficulty", "adaptive")
                num_questions = min(10, max(3, int(request.form.get("num_questions", 10))))
                question_style = request.form.get("question_style", "Mixed")

                topic = chosen_topic
                if topic_mode == "Use weakest topic":
                    topic = student.weakest_topic() or chosen_topic
                elif topic_mode == "Adaptive mix":
                    topic = "Adaptive mixed probability practice with emphasis on the student's weak topics"

                if chosen_topic == "All Topics" and topic_mode == "Choose manually":
                    topic = "All Topics"

                effective_difficulty = student.level if difficulty == "adaptive" else difficulty
                rag_query = build_topic_query(topic, effective_difficulty, student.get_weak_topics())

                if topic == "All Topics":
                    course_context = retrieve_course_context(rag_query, top_k=10, course_code=None)
                    detected_course = None
                else:
                    course_context, detected_course = retrieve_course_context_auto(rag_query, top_k=6)

                if detected_course:
                    flash(f"Using course: {detected_course}", "success")

                quiz = tutor.generate_structured_quiz(
                    topic=topic,
                    difficulty=effective_difficulty,
                    num_questions=num_questions,
                    style=question_style,
                    weak_topics=student.get_weak_topics(),
                    course_context=course_context,
                )
                persist_quiz(quiz)
                persist_quiz_results(None)
                return redirect(url_for("quiz"))

        elif action == "submit":
            if quiz:
                answers = {
                    str(i): request.form.get(f"answer_{i}", "")
                    for i in range(1, len(quiz.get("questions", [])) + 1)
                }
                quiz_results = quiz_engine.grade_quiz(quiz=quiz, answers=answers)
                persist_quiz_results(quiz_results)

                for item in quiz_results["results"]:
                    student.update_performance(item["topic"], item["is_correct"])
                save_student(student)

                current_user = current_username()
                if current_user:
                    store.record_quiz_attempt(
                        username=current_user,
                        quiz_title=str(quiz.get("title", "Quiz")),
                        quiz_topic=str(quiz.get("topic", "Unknown Topic")),
                        score=int(quiz_results.get("score", 0)),
                        total=int(quiz_results.get("total", 0)),
                        percent=float(quiz_results.get("percent", 0.0)),
                        results=quiz_results.get("results", []),
                    )

                return redirect(url_for("quiz"))

        elif action == "discard":
            persist_quiz(None)
            persist_quiz_results(None)
            return redirect(url_for("quiz"))

        elif action == "restart":
            persist_quiz(None)
            persist_quiz_results(None)
            return redirect(url_for("quiz"))

    return render_page(
        "quiz.html",
        page_title="Quiz",
        active_tab="quiz",
        topics=topics,
        quiz=quiz,
        quiz_results=quiz_results,
    )


@app.route("/tutor-chat", methods=["GET", "POST"])
def tutor_chat():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    tutor = get_tutor()
    student = load_student()
    history = get_chat_history()

    if request.method == "POST":
        action = request.form.get("action", "send")
        if action == "clear":
            set_chat_history([])
            return redirect(url_for("tutor_chat"))

        prompt = request.form.get("prompt", "").strip()
        if prompt:
            history.append(("user", prompt))
            if tutor is None:
                flash("OPENAI_API_KEY is missing. Set it before running the app.", "error")
            else:
                course_context, detected_course = retrieve_course_context_auto(prompt, top_k=4)
                if detected_course:
                    flash(f"Using course: {detected_course}", "success")

                answer = tutor.chat(
                    user_message=prompt,
                    student_level=student.level,
                    weak_topics=student.get_weak_topics(),
                    chat_history=history,
                    course_context=course_context,
                )
                history.append(("assistant", answer))
            set_chat_history(history)
            return redirect(url_for("tutor_chat"))

    return render_page(
        "tutor_chat.html",
        page_title="Tutor Chat",
        active_tab="tutor_chat",
        chat_history=history,
    )


@app.route("/course-materials", methods=["GET", "POST"])
def course_materials():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    if not store.is_admin(current_username() or ""):
        flash("Admin only.", "error")
        return redirect(url_for("solve"))

    if request.method == "POST":
        action = request.form.get("action", "").strip()

        if action == "upload_course":
            uploaded = request.files.get("course_file")
            course_code = request.form.get("course_code", "").strip()
            title = request.form.get("title", "").strip()

            if not uploaded or not uploaded.filename:
                flash("Please choose a course file.", "error")
            elif not course_code:
                flash("Please provide a chapter name.", "error")
            elif not title:
                flash("Please provide a title.", "error")
            else:
                filename = secure_filename(uploaded.filename)
                temp_path = save_temp_upload(uploaded)
                ok, message = index_course_file(temp_path, filename, course_code, title)
                flash(message, "success" if ok else "error")
            return redirect(url_for("course_materials"))

        if action == "delete_course":
            document_id = request.form.get("document_id", "").strip()
            if document_id.isdigit():
                store.delete_course_document(int(document_id))
                flash("Course document deleted.", "success")
            return redirect(url_for("course_materials"))

    documents = store.list_course_documents()
    return render_page(
        "course_materials.html",
        page_title="Course Materials",
        active_tab="course_materials",
        documents=documents,
    )


@app.route("/admin", methods=["GET", "POST"])
def admin():
    redirect_response = ensure_logged_in()
    if redirect_response:
        return redirect_response

    if not store.is_admin(current_username() or ""):
        flash("Admin only.", "error")
        return redirect(url_for("solve"))

    if request.method == "POST":
        action = request.form.get("action", "")
        if action == "delete_user":
            username = request.form.get("username", "").strip()
            if username and username != "admin":
                store.delete_user(username)
                flash(f"Deleted {username}.", "success")
            return redirect(url_for("admin"))

    users = store.list_users()
    df = build_admin_dataframe(users)

    selected_topic = request.args.get("topic_filter", "All Topics")
    selected_student = request.args.get("student_filter", "").strip()

    figures, topic_options, student_options = build_admin_figures(
        df,
        get_theme(),
        selected_topic,
        selected_student,
    )

    filtered_role = request.args.get("role", "All")
    search_text = request.args.get("search", "").strip().lower()

    filtered_df = df.copy()
    if filtered_role != "All" and not filtered_df.empty:
        filtered_df = filtered_df[filtered_df["role"] == filtered_role]
    if search_text and not filtered_df.empty:
        filtered_df = filtered_df[
            filtered_df["username"].str.lower().str.contains(search_text, na=False)
        ]

    table_rows = filtered_df.to_dict(orient="records") if not filtered_df.empty else []

    overview = {
        "total_users": len(df),
        "total_students": int((df["role"] == "Student").sum()) if not df.empty else 0,
        "total_admins": int((df["role"] == "Admin").sum()) if not df.empty else 0,
        "avg_accuracy": float(df["accuracy_percent"].mean()) if not df.empty else 0.0,
    }

    return render_page(
        "admin.html",
        page_title="Admin Dashboard",
        active_tab="admin",
        overview=overview,
        figures=figures,
        users=users,
        table_rows=table_rows,
        filtered_role=filtered_role,
        search_text=search_text,
        selected_topic=selected_topic,
        selected_student=selected_student,
        topic_options=topic_options,
        student_options=student_options,
    )


@app.context_processor
def inject_utilities():
    return {
        "nav_items": [
            ("solve", "Solve", "solve"),
            ("practice", "Practice", "practice"),
            ("quiz", "Quiz", "quiz"),
            ("tutor_chat", "Tutor Chat", "tutor_chat"),
            ("course_materials", "Course Materials", "course_materials"),
            ("admin", "Admin", "admin"),
        ],
    }


bootstrap_default_courses()

if __name__ == "__main__":
    import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)