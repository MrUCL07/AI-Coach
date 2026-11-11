"""AI Coach: explains a topic in one paragraph, then quizzes you with 5 questions.

Stack: Ollama (local LLM) for generation + Streamlit for the UI.
Run with:  streamlit run app.py
"""

import html
import io
import json
import uuid
from datetime import datetime
from pathlib import Path

import ollama
import streamlit as st

st.set_page_config(page_title="AI Coach", page_icon="🎓", layout="centered")

DEFAULT_MODEL = "llama3.2"
DEFAULT_VISION_MODEL = "llava"  # run: ollama pull llava
WHISPER_SIZE = "base"  # "tiny" = fastest, "small" = more accurate
VISION_HINTS = ("llava", "vision", "moondream", "gemma3", "qwen2.5vl", "minicpm", "bakllava")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stTextInput, .stTextArea, button {
    font-family: 'Poppins', sans-serif;
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 780px;}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #6C5CE7 0%, #8E7CF8 45%, #00B8D9 100%);
    border-radius: 24px;
    padding: 2.2rem 2rem;
    color: #fff;
    text-align: center;
    box-shadow: 0 12px 32px rgba(108, 92, 231, 0.35);
    margin-bottom: 1.4rem;
}
.hero h1 {margin: 0; font-size: 2.4rem; font-weight: 700; color: #fff; padding: 0;}
.hero p {margin: .5rem 0 0; font-size: 1.02rem; opacity: .92; color: #fff;}

/* ---------- Step tracker ---------- */
.steps {display: flex; gap: .6rem; margin-bottom: 1.4rem;}
.step {
    flex: 1; text-align: center; padding: .65rem .4rem; border-radius: 14px;
    font-size: .85rem; font-weight: 500;
    background: rgba(128,128,128,.12); color: rgba(128,128,128,.95);
    border: 1px solid rgba(128,128,128,.18);
}
.step.active {
    background: linear-gradient(135deg, #6C5CE7, #8E7CF8); color: #fff;
    border-color: transparent; box-shadow: 0 6px 16px rgba(108,92,231,.35);
}
.step.done {
    background: rgba(0, 184, 148, .15); color: #00B894; border-color: rgba(0,184,148,.35);
}

/* ---------- Cards ---------- */
.card {
    background: rgba(128,128,128,.08);
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 18px; padding: 1.4rem 1.5rem; margin: .4rem 0 1.2rem;
    box-shadow: 0 4px 18px rgba(0,0,0,.06);
}
.card.explain {border-left: 6px solid #6C5CE7;}
.card .tag {
    display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; color: #6C5CE7; background: rgba(108,92,231,.14);
    padding: .2rem .7rem; border-radius: 999px; margin-bottom: .6rem;
}
.card h3 {margin: .1rem 0 .7rem; font-size: 1.35rem; font-weight: 600;}
.card p {margin: 0; line-height: 1.75; font-size: 1.02rem; text-align: justify;}

/* ---------- Question cards ---------- */
.q {
    display: flex; gap: .9rem; align-items: flex-start;
    background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.18);
    border-radius: 16px; padding: .9rem 1.1rem; margin: 1rem 0 .4rem;
}
.q .num {
    flex: 0 0 auto; width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, #6C5CE7, #00B8D9); color: #fff;
    display: flex; align-items: center; justify-content: center; font-weight: 600;
}
.q .txt {font-weight: 500; line-height: 1.5; padding-top: .2rem;}

/* ---------- Score ---------- */
.score-wrap {display: flex; flex-direction: column; align-items: center; margin: 1rem 0 1.4rem;}
.ring {
    width: 150px; height: 150px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 8px 24px rgba(0,0,0,.12);
}
.ring .inner {
    width: 114px; height: 114px; border-radius: 50%;
    background: var(--background-color, #fff);
    display: flex; flex-direction: column; align-items: center; justify-content: center;
}
.ring .pct {font-size: 1.9rem; font-weight: 700; line-height: 1;}
.ring .sub {font-size: .78rem; opacity: .7; margin-top: .2rem;}
.verdict {font-size: 1.2rem; font-weight: 600; margin-top: .8rem;}

.fb {
    border-radius: 14px; padding: .8rem 1.1rem; margin: .6rem 0;
    border-left: 6px solid; background: rgba(128,128,128,.08); line-height: 1.55;
}
.fb.s2 {border-color: #00B894; background: rgba(0,184,148,.10);}
.fb.s1 {border-color: #FDCB6E; background: rgba(253,203,110,.14);}
.fb.s0 {border-color: #FF7675; background: rgba(255,118,117,.12);}
.fb b {font-weight: 600;}

.overall {
    background: linear-gradient(135deg, rgba(108,92,231,.14), rgba(0,184,217,.14));
    border: 1px solid rgba(108,92,231,.3); border-radius: 16px;
    padding: 1rem 1.2rem; margin-top: 1rem; line-height: 1.6;
}

/* ---------- Inputs & buttons ---------- */
.stTextInput input, .stTextArea textarea {
    border-radius: 14px !important; border: 1.5px solid rgba(128,128,128,.3) !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #6C5CE7 !important; box-shadow: 0 0 0 3px rgba(108,92,231,.2) !important;
}
.stButton > button, .stFormSubmitButton > button {
    width: 100%; border: none; border-radius: 14px; padding: .7rem 1rem;
    font-weight: 600; color: #fff;
    background: linear-gradient(135deg, #6C5CE7, #8E7CF8);
    box-shadow: 0 6px 16px rgba(108,92,231,.35); transition: all .2s ease;
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px); box-shadow: 0 10px 22px rgba(108,92,231,.45); color: #fff;
}
[data-testid="stForm"] {border: none; padding: 0;}
.stTabs [data-baseweb="tab"] {font-weight: 600; font-size: .95rem;}
[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.2);}

/* ---------- History buttons (sidebar) ---------- */
[class*="st-key-hist_open_"] button {
    background: rgba(128,128,128,.12); color: inherit; box-shadow: none;
    text-align: left; justify-content: flex-start; font-weight: 500;
    padding: .5rem .8rem; font-size: .85rem;
}
[class*="st-key-hist_open_"] button:hover {
    background: rgba(108,92,231,.22); color: inherit; box-shadow: none; transform: none;
}
[class*="st-key-hist_del_"] button, [class*="st-key-hist_clear"] button {
    background: rgba(255,118,117,.15); color: inherit; box-shadow: none;
    padding: .5rem .4rem; font-size: .85rem;
}
[class*="st-key-hist_del_"] button:hover, [class*="st-key-hist_clear"] button:hover {
    background: rgba(255,118,117,.35); color: inherit; box-shadow: none; transform: none;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
def ask_llm(model: str, system: str, user: str, json_mode: bool = False) -> str:
    """Send one request to the local Ollama model and return the text reply."""
    kwargs = {"format": "json"} if json_mode else {}
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": 0.4},
        **kwargs,
    )
    return response["message"]["content"].strip()


def explain_topic(model: str, topic: str, level: str) -> str:
    system = (
        "You are a friendly, clear AI coach. You explain topics to learners "
        "in exactly ONE paragraph of about 120-170 words. No bullet points, "
        "no headings, no lists. Use simple language and one short example."
    )
    user = f"Explain this topic for a {level} learner: {topic}"
    return ask_llm(model, system, user)


def make_questions(model: str, topic: str, explanation: str) -> list[str]:
    """Ask the model for 5 summary-style questions, returned as JSON."""
    system = (
        "You are a coach who checks understanding. Reply ONLY with JSON in "
        'this exact shape: {"questions": ["q1", "q2", "q3", "q4", "q5"]}. '
        "Write exactly 5 short-answer questions that ask the learner to "
        "summarize or restate ideas from the explanation in their own words. "
        "Do not include answers."
    )
    user = f"Topic: {topic}\n\nExplanation:\n{explanation}"

    for _ in range(3):  # retry a few times in case the JSON is malformed
        try:
            data = json.loads(ask_llm(model, system, user, json_mode=True))
            questions = [str(q).strip() for q in data["questions"] if str(q).strip()]
            if len(questions) >= 5:
                return questions[:5]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    raise ValueError("The model did not return 5 valid questions. Try again.")


def evaluate_answers(
    model: str, topic: str, explanation: str, questions: list[str], answers: list[str]
) -> dict:
    """Grade each answer 0-2 and give short feedback."""
    system = (
        "You are a kind but honest coach grading a learner's answers against "
        "the explanation. Reply ONLY with JSON in this shape: "
        '{"results": [{"score": 0, "feedback": "..."}, ...], "overall": "..."}. '
        "Give exactly one result per question, in order. Score: 0 = wrong or "
        "empty, 1 = partly correct, 2 = correct. Keep feedback to one or two "
        "sentences. 'overall' is a short encouraging summary."
    )
    qa = "\n".join(
        f"Q{i + 1}: {q}\nAnswer: {a or '(no answer)'}"
        for i, (q, a) in enumerate(zip(questions, answers))
    )
    user = f"Topic: {topic}\n\nExplanation:\n{explanation}\n\n{qa}"

    for _ in range(3):
        try:
            data = json.loads(ask_llm(model, system, user, json_mode=True))
            results = data["results"]
            if len(results) == len(questions):
                for r in results:
                    r["score"] = max(0, min(2, int(r.get("score", 0))))
                return data
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    raise ValueError("The model could not grade the answers. Try again.")


def installed_models() -> list[str]:
    """List models pulled in Ollama; empty list if Ollama isn't reachable."""
    try:
        listing = ollama.list()
        return [m["model"] for m in listing["models"]]
    except Exception:
        return []


# ----------------------------------------------------------------------------
# Speech & image input helpers
# ----------------------------------------------------------------------------
def clean_topic(text: str) -> str:
    """Keep only the first line of the model's reply, without quotes/extra marks."""
    text = text.strip()
    line = text.splitlines()[0] if text else ""
    return line.strip(" \"'`*.").strip()[:100]


@st.cache_resource(show_spinner=False)
def load_whisper(size: str):
    """Load the local speech-to-text model once and reuse it."""
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


def transcribe_audio(audio_file) -> str:
    """Convert a recorded clip to text using Whisper (runs on your computer)."""
    try:
        whisper = load_whisper(WHISPER_SIZE)
    except ImportError:
        raise RuntimeError("Speech input needs faster-whisper. Run: pip install faster-whisper")
    segments, _ = whisper.transcribe(io.BytesIO(audio_file.getvalue()), beam_size=1)
    return " ".join(seg.text.strip() for seg in segments).strip()


def extract_topic(model: str, request: str) -> str:
    """Turn a spoken sentence like 'can you explain black holes?' into 'Black holes'."""
    system = (
        "You turn a learner's request into a short topic name of at most 8 words. "
        "Reply with ONLY the topic name. No quotes, no explanation."
    )
    topic = clean_topic(ask_llm(model, system, f"Request: {request}"))
    return topic or request.strip()[:100]


def prepare_image(image_file) -> bytes:
    """Shrink the image so the vision model answers faster."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_file.getvalue())).convert("RGB")
    img.thumbnail((1024, 1024))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def topic_from_image(vision_model: str, image_file) -> str:
    """Ask a vision model what subject the picture is about."""
    response = ollama.chat(
        model=vision_model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Look at this image. What subject or concept would a learner "
                    "want to study from it? Reply with ONLY a short topic name "
                    "of at most 8 words."
                ),
                "images": [prepare_image(image_file)],
            }
        ],
        options={"temperature": 0.2},
    )
    topic = clean_topic(response["message"]["content"])
    if not topic:
        raise ValueError("The vision model returned no topic. Try a clearer image.")
    return topic


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def esc(text) -> str:
    """Escape model/user text before putting it inside HTML."""
    return html.escape(str(text)).replace("\n", "<br>")


def step_tracker(stage: int) -> None:
    """stage: 0 = choose topic, 1 = reading, 2 = results."""
    labels = ["1 · Pick a topic", "2 · Learn & answer", "3 · See your score"]
    items = []
    for i, label in enumerate(labels):
        cls = "active" if i == stage else ("done" if i < stage else "")
        items.append(f'<div class="step {cls}">{label}</div>')
    st.markdown(f'<div class="steps">{"".join(items)}</div>', unsafe_allow_html=True)


def score_ring(total: int, maximum: int) -> None:
    pct = round(100 * total / maximum)
    if pct >= 80:
        color, verdict = "#00B894", "🌟 Excellent understanding!"
    elif pct >= 50:
        color, verdict = "#FDCB6E", "👍 Good effort, almost there!"
    else:
        color, verdict = "#FF7675", "📚 Give the explanation another read."
    st.markdown(
        f"""
        <div class="score-wrap">
          <div class="ring" style="background: conic-gradient({color} {pct}%, rgba(128,128,128,.25) 0);">
            <div class="inner"><div class="pct">{total}/{maximum}</div><div class="sub">{pct}%</div></div>
          </div>
          <div class="verdict">{verdict}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
for key, default in {
    "topic": "",
    "explanation": "",
    "questions": [],
    "evaluation": None,
    "level_used": "beginner",
    "current_id": None,
}.items():
    st.session_state.setdefault(key, default)


# ----------------------------------------------------------------------------
# Chat history (saved to history.json next to app.py)
# ----------------------------------------------------------------------------
HISTORY_FILE = Path(__file__).parent / "history.json"


def load_history() -> list[dict]:
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save_history() -> None:
    try:
        HISTORY_FILE.write_text(
            json.dumps(st.session_state.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        st.warning(f"Couldn't save history: {e}")


if "history" not in st.session_state:
    st.session_state.history = load_history()


def entry_score(entry: dict):
    """Return (total, maximum) if the entry was graded, else None."""
    ev = entry.get("evaluation")
    if not ev:
        return None
    return sum(r["score"] for r in ev["results"]), 2 * len(ev["results"])


def reset():
    st.session_state.update(
        topic="", explanation="", questions=[], evaluation=None, current_id=None
    )
    for i in range(5):
        st.session_state.pop(f"answer_{i}", None)


def add_history_entry(topic: str, level: str, explanation: str, questions: list[str]):
    entry = {
        "id": uuid.uuid4().hex[:8],
        "created": datetime.now().strftime("%d %b %Y, %H:%M"),
        "topic": topic,
        "level": level,
        "explanation": explanation,
        "questions": questions,
        "answers": [],
        "evaluation": None,
    }
    st.session_state.history.insert(0, entry)
    save_history()
    st.session_state.update(
        topic=topic,
        explanation=explanation,
        questions=questions,
        evaluation=None,
        level_used=level,
        current_id=entry["id"],
    )


def save_results(answers: list[str], evaluation: dict) -> None:
    for entry in st.session_state.history:
        if entry["id"] == st.session_state.current_id:
            entry["answers"] = answers
            entry["evaluation"] = evaluation
            break
    save_history()


def open_entry(entry_id: str) -> None:
    """Callback: load a saved session back into the main view."""
    for entry in st.session_state.history:
        if entry["id"] == entry_id:
            reset()
            st.session_state.update(
                topic=entry["topic"],
                explanation=entry["explanation"],
                questions=entry["questions"],
                evaluation=entry.get("evaluation"),
                level_used=entry.get("level", "beginner"),
                current_id=entry["id"],
            )
            saved = entry.get("answers", [])
            for i in range(len(entry["questions"])):
                st.session_state[f"answer_{i}"] = saved[i] if i < len(saved) else ""
            break


def delete_entry(entry_id: str) -> None:
    st.session_state.history = [
        e for e in st.session_state.history if e["id"] != entry_id
    ]
    save_history()
    if st.session_state.current_id == entry_id:
        reset()


def clear_history() -> None:
    st.session_state.history = []
    save_history()
    reset()


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    models = installed_models()
    if models:
        model = st.selectbox("Ollama model", models)
    else:
        st.warning("Couldn't reach Ollama. Is it running (`ollama serve`)?")
        model = st.text_input("Ollama model", DEFAULT_MODEL)
    if models:
        guess = next((m for m in models if any(h in m.lower() for h in VISION_HINTS)), None)
        vision_model = st.selectbox(
            "Vision model (for images)",
            models,
            index=models.index(guess) if guess else 0,
            help="Must be a model that can read images, e.g. llava or llama3.2-vision.",
        )
        if guess is None:
            st.caption("No vision model found. Run `ollama pull llava` to use image input.")
    else:
        vision_model = st.text_input("Vision model (for images)", DEFAULT_VISION_MODEL)
    level = st.selectbox("Your level", ["beginner", "intermediate", "advanced"])
    st.markdown("---")
    if st.button("➕ New topic", key="new_topic"):
        reset()
        st.rerun()

    st.markdown("## 📜 History")
    if not st.session_state.history:
        st.caption("Your past topics will show up here.")
    for entry in st.session_state.history:
        score = entry_score(entry)
        icon = "✅" if score else "📖"
        suffix = f" · {score[0]}/{score[1]}" if score else ""
        active = "▶ " if entry["id"] == st.session_state.current_id else ""
        title = entry["topic"] if len(entry["topic"]) <= 20 else entry["topic"][:19] + "…"
        c1, c2 = st.columns([5, 1])
        c1.button(
            f"{active}{icon} {title}{suffix}",
            key=f"hist_open_{entry['id']}",
            help=f"{entry['topic']} ({entry['level']}) · {entry['created']}",
            on_click=open_entry,
            args=(entry["id"],),
        )
        c2.button(
            "🗑",
            key=f"hist_del_{entry['id']}",
            help="Delete this entry",
            on_click=delete_entry,
            args=(entry["id"],),
        )
    if st.session_state.history:
        st.button(
            "Clear all history",
            key="hist_clear",
            on_click=clear_history,
        )
    st.markdown("---")
    st.caption("Runs fully on your computer with Ollama. No data leaves your machine.")

# ----------------------------------------------------------------------------
# Main UI
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 AI Coach</h1>
      <p>Type any topic, get a clear explanation, then test yourself with 5 questions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stage = 2 if st.session_state.evaluation else (1 if st.session_state.explanation else 0)
step_tracker(stage)

st.markdown("#### 🎯 How do you want to choose a topic?")
tab_type, tab_speak, tab_image = st.tabs(["⌨️ Type", "🎤 Speak", "🖼️ Image"])

with tab_type:
    st.caption("Type a topic or a full question in the box below these tabs.")

with tab_speak:
    st.caption("Record yourself, e.g. “Explain how black holes form.” Transcribed locally with Whisper.")
    audio = st.audio_input("Record your question", key="audio_in")
    if st.button("🎤 Use this recording", key="use_audio"):
        if audio is None:
            st.warning("Record something first.")
        else:
            try:
                with st.spinner("👂 Listening..."):
                    heard = transcribe_audio(audio)
                    if not heard:
                        raise ValueError("No speech detected. Try again closer to the microphone.")
                    found = extract_topic(model, heard)
                st.session_state.topic_input = found
                st.success(f"I heard: “{heard}” → topic: **{found}**")
            except Exception as e:
                st.error(f"Couldn't use the recording: {e}")

with tab_image:
    st.caption(f"Upload or snap a picture (diagram, textbook page, object...). Uses `{vision_model}`.")
    uploaded = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "webp"], key="img_upload"
    )
    with st.expander("📷 Or use your camera"):
        snapshot = st.camera_input("Take a photo", key="img_camera")
    image_file = uploaded if uploaded is not None else snapshot
    if image_file is not None:
        st.image(image_file, width=260)
    if st.button("🔎 Detect topic from image", key="use_image"):
        if image_file is None:
            st.warning("Upload or take a picture first.")
        else:
            try:
                with st.spinner("👀 Looking at the image..."):
                    found = topic_from_image(vision_model, image_file)
                st.session_state.topic_input = found
                st.success(f"Topic detected: **{found}**")
            except Exception as e:
                st.error(
                    f"Couldn't read the image: {e}. "
                    f"Is `{vision_model}` pulled? Try: ollama pull {vision_model}"
                )

with st.form("topic_form"):
    topic = st.text_input(
        "What do you want to learn? (you can edit the detected topic here)",
        key="topic_input",
        placeholder="e.g. Transformers in LLMs, Photosynthesis, Blockchain...",
    )
    submitted = st.form_submit_button("✨ Explain it")

if submitted:
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        reset()
        try:
            with st.spinner("🧠 Thinking..."):
                explanation = explain_topic(model, topic.strip(), level)
            with st.spinner("📝 Preparing your questions..."):
                questions = make_questions(model, topic.strip(), explanation)
            add_history_entry(topic.strip(), level, explanation, questions)
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Explanation + quiz
if st.session_state.explanation:
    st.markdown(
        f"""
        <div class="card explain">
          <span class="tag">📖 Explanation · {esc(st.session_state.level_used)}</span>
          <h3>{esc(st.session_state.topic)}</h3>
          <p>{esc(st.session_state.explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📝 Check your understanding")
    st.caption("Answer each question in your own words, like a short summary.")

    with st.form("quiz_form"):
        answers = []
        for i, q in enumerate(st.session_state.questions):
            st.markdown(
                f'<div class="q"><div class="num">{i + 1}</div>'
                f'<div class="txt">{esc(q)}</div></div>',
                unsafe_allow_html=True,
            )
            answers.append(
                st.text_area(
                    f"Answer {i + 1}",
                    key=f"answer_{i}",
                    height=90,
                    placeholder="Type your answer here...",
                    label_visibility="collapsed",
                )
            )
        graded = st.form_submit_button("🚀 Submit answers")

    if graded:
        try:
            with st.spinner("🔍 Grading your answers..."):
                evaluation = evaluate_answers(
                    model,
                    st.session_state.topic,
                    st.session_state.explanation,
                    st.session_state.questions,
                    answers,
                )
            st.session_state.evaluation = evaluation
            save_results(answers, evaluation)
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Results
evaluation = st.session_state.evaluation
if evaluation:
    results = evaluation["results"]
    total = sum(r["score"] for r in results)
    maximum = 2 * len(results)

    st.markdown("---")
    st.markdown("### 🏆 Your results")
    score_ring(total, maximum)

    icons = {2: "🟢 Correct", 1: "🟡 Partly correct", 0: "🔴 Needs work"}
    for i, r in enumerate(results):
        st.markdown(
            f'<div class="fb s{r["score"]}"><b>Q{i + 1} · {icons[r["score"]]}</b><br>'
            f'{esc(r.get("feedback", ""))}</div>',
            unsafe_allow_html=True,
        )

    overall = evaluation.get("overall", "")
    if overall:
        st.markdown(
            f'<div class="overall">💬 <b>Coach says:</b> {esc(overall)}</div>',
            unsafe_allow_html=True,
        )
