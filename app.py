"""
GAIA Benchmark Solver — Hugging Face AI Agents Course (Unit 4)
==============================================================
Solves GAIA Level 1 questions using smolagents

Setup (HF Space secrets):
    GROQ_API_KEY  —(required)
    HF_TOKEN           — your HF token (required for login/submission)
    SPACE_ID           — automatically set by HF Spaces
"""

import os
import re
import base64
import tempfile
import traceback

import requests
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from smolagents import CodeAgent, LiteLLMModel, Tool
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"


# ---------------------------------------------------------------------------
# Model — Claude via LiteLLM (reliable, fast, cheap)
# ---------------------------------------------------------------------------

def build_model() -> LiteLLMModel:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Add it to your HF Space secrets or .env file. "
        )

    return LiteLLMModel(
        model_id="groq/llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class DownloadTaskFileTool(Tool):
    """
    Downloads the file attached to a GAIA task from the scoring API.

    Text-based files (CSV, JSON, TXT) are returned as decoded strings.
    Binary files (audio, images) are saved to a temp path and the path
    is returned so other tools can process them.

    Endpoint: GET {GAIA_API_BASE}/files/{task_id}
    """

    name = "download_task_file"
    description = (
        "Downloads the file attached to the current GAIA task and returns its contents. "
        "For CSV/JSON/TXT returns the text directly. "
        "For audio/image files, saves to disk and returns the local path. "
        "Call this FIRST whenever the question mentions 'the file', 'the image', "
        "'the audio', 'attached', etc."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The GAIA task UUID whose file should be downloaded.",
        }
    }
    output_type = "string"

    _TEXT_EXTS = {".csv", ".txt", ".json", ".tsv", ".md", ".xml", ".html"}
    _AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
    _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}

    def forward(self, task_id: str) -> str:
        url = f"{GAIA_API_BASE}/files/{task_id}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as exc:
            return f"[DownloadTaskFileTool ERROR] {exc}"

        # Detect filename / extension
        content_disp = r.headers.get("Content-Disposition", "")
        content_type = r.headers.get("Content-Type", "")

        filename = ""
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip().strip('"')

        ext = os.path.splitext(filename)[-1].lower()

        # Infer from MIME if extension is missing
        if not ext:
            mime_map = {
                "csv": ".csv", "json": ".json", "text": ".txt",
                "mpeg": ".mp3", "audio": ".mp3",
                "png": ".png", "jpeg": ".jpg", "image": ".png",
            }
            for key, val in mime_map.items():
                if key in content_type:
                    ext = val
                    break

        # Text files — return content directly
        if ext in self._TEXT_EXTS:
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return r.content.decode("latin-1")

        # Binary files — save and return path
        suffix = ext or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(r.content)
            local_path = tmp.name

        if ext in self._AUDIO_EXTS:
            return (
                f"Audio file saved to: {local_path}\n"
                f"Pass this path to transcribe_audio to get the text."
            )
        if ext in self._IMAGE_EXTS:
            return (
                f"Image file saved to: {local_path}\n"
                f"Describe the image content to answer the question."
            )
        return f"File saved to: {local_path} (type: {ext or 'unknown'})"


class AudioTranscriptionTool(Tool):
    """
    Transcribes audio files using OpenAI Whisper (base model).
    Accepts a local path or an HTTPS URL.
    Downloads remote URLs before transcription.
    """

    name = "transcribe_audio"
    description = (
        "Transcribes an audio file (mp3, wav, m4a) and returns the full text. "
        "Accepts a local filesystem path or a direct download URL. "
        "Use this whenever a question involves an audio file."
    )
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Local file path or HTTP/HTTPS URL to the audio file.",
        }
    }
    output_type = "string"

    def forward(self, audio_path: str) -> str:
        try:
            import whisper  # noqa: PLC0415
        except ImportError:
            return "Whisper not installed. Run: pip install openai-whisper"

        local_path = audio_path

        # Download if URL
        if audio_path.startswith("http://") or audio_path.startswith("https://"):
            try:
                r = requests.get(audio_path, timeout=60)
                r.raise_for_status()
                suffix = os.path.splitext(audio_path.split("?")[0])[-1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(r.content)
                    local_path = tmp.name
            except Exception as exc:
                return f"[AudioTranscriptionTool] Download failed: {exc}"

        try:
            model_w = whisper.load_model("base")
            result = model_w.transcribe(local_path)
            return result["text"].strip()
        except Exception as exc:
            return f"[AudioTranscriptionTool] Transcription failed: {exc}"


class YouTubeTranscriptTool(Tool):
    """
    Fetches the full text transcript of a YouTube video.
    Prefers English captions; falls back to any available language.
    """

    name = "get_youtube_transcript"
    description = (
        "Fetches the complete transcript of a YouTube video given its URL or video ID. "
        "Use this immediately whenever a question contains a YouTube link."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "Full YouTube URL or bare video ID.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        # Extract video ID from various URL formats
        video_id = url
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]

        try:
            try:
                entries = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except Exception:
                entries = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join(e["text"] for e in entries)
        except Exception as exc:
            return (
                f"[YouTubeTranscriptTool] Could not get transcript for '{video_id}': {exc}. "
                "Try searching the web for a summary of this video instead."
            )


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

# This prompt is injected as agent instructions — NOT prepended to each question.
# smolagents CodeAgent accepts it via the `system_prompt` kwarg on CodeAgent
# or via the managed_agents approach. Here we inject it into the task context.

ANSWER_RULES = """
ANSWER FORMAT RULES (follow strictly — answers are scored by exact match):
- Return ONLY the raw answer. No explanation, no preamble, no punctuation added.
- WRONG: "The answer is 4"   RIGHT: "4"
- WRONG: "Based on research: Paris"   RIGHT: "Paris"
- Lists → alphabetical order, comma-separated, e.g. "apple, banana, cherry"
- Numbers → no trailing zeros unless significant; no thousands separators unless asked
- Dates → use the format the question implies; default ISO: YYYY-MM-DD
- Botany rule: tomatoes, peppers, cucumbers, avocados, corn kernels = FRUITS
  Only say "vegetable" if the question explicitly asks about culinary context
- Reversed text questions: decode the reversal, answer the actual question
- For YES/NO questions: answer exactly "Yes" or "No"
"""


# ---------------------------------------------------------------------------
# GAIA Solver
# ---------------------------------------------------------------------------

class GAIASolver:
    """
    Wraps a smolagents CodeAgent to solve GAIA benchmark tasks.

    Each task dict contains:
        task_id   : str  — UUID used to fetch attached files
        question  : str  — The question to answer
        file_name : str  — Optional filename hint (may be empty)
    """

    def __init__(self, model: LiteLLMModel):
        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                YouTubeTranscriptTool(),
                AudioTranscriptionTool(),
                DownloadTaskFileTool(),
            ],
            model=model,
            max_steps=15,
            verbosity_level=1,
            additional_authorized_imports=[
                "pandas", "numpy", "re", "math", "datetime",
                "collections", "json", "csv", "itertools", "string",
            ],
        )

    def solve(self, task: dict) -> str:
        """Solve one GAIA task and return a clean answer string."""
        task_id = task.get("task_id", "")
        question = task.get("question", "")
        file_name = task.get("file_name", "")

        # Build the full prompt sent to the agent
        lines = [ANSWER_RULES, "---"]
        lines.append(f"Task ID: {task_id}")
        if file_name:
            lines.append(
                f"This task has an attached file: '{file_name}'. "
                f"Call download_task_file(task_id='{task_id}') to access it BEFORE answering."
            )
        lines.append(f"\nQuestion: {question}")
        prompt = "\n".join(lines)

        try:
            # reset=True is critical — prevents state leaking between questions
            raw = self.agent.run(prompt, reset=True)
            return self._clean(str(raw))
        except Exception as exc:
            print(f"[ERROR] Task {task_id}: {exc}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Minimal cleanup of LLM output.
        Only strips patterns that are unambiguously wrong for exact-match scoring.
        """
        s = raw.strip()

        # Strip markdown code fences wrapping the whole answer
        if s.startswith("```") and s.endswith("```"):
            inner = s[3:-3].strip()
            # Remove language hint on first line (e.g. ```python)
            first_line, _, rest = inner.partition("\n")
            s = rest.strip() if first_line.isalpha() else inner

        # Strip explicit "The answer is / Final answer:" prefixes (case-insensitive)
        s = re.sub(
            r"^\s*(the\s+)?(final\s+)?(answer\s+(is|:)|result\s*:|value\s*:)\s*",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

        # Strip symmetric surrounding quotes
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and len(s) > 1:
                s = s[1:-1].strip()
                break

        return s


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_questions() -> list[dict]:
    r = requests.get(f"{GAIA_API_BASE}/questions", timeout=30)
    r.raise_for_status()
    return r.json()


def submit_answers(username: str, space_id: str, answers: list[dict]) -> dict:
    payload = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers,
    }
    r = requests.post(f"{GAIA_API_BASE}/submit", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Gradio runner
# ---------------------------------------------------------------------------

def run_benchmark(profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame | None]:
    """
    Orchestrates the full benchmark run:
      1. Validate login
      2. Build model + solver
      3. Fetch questions
      4. Solve each question
      5. Submit and display score
    """
    if profile is None:
        return "⚠️  Please log in with your Hugging Face account first.", None

    username = profile.username
    space_id = os.getenv("SPACE_ID", "")

    print(f"Starting benchmark for user: {username}")

    # Build model & solver
    try:
        model = build_model()
        solver = GAIASolver(model)
    except EnvironmentError as exc:
        return f"❌ Configuration error: {exc}", None

    # Fetch questions
    try:
        questions = fetch_questions()
        print(f"Fetched {len(questions)} questions.")
    except Exception as exc:
        return f"❌ Failed to fetch questions: {exc}", None

    # Solve
    submissions: list[dict] = []
    rows: list[dict] = []

    for i, task in enumerate(questions, start=1):
        tid = task.get("task_id", f"task_{i}")
        q_preview = task.get("question", "")[:80]
        print(f"\n[{i:02d}/{len(questions)}] {tid}: {q_preview}...")

        answer = solver.solve(task)
        print(f"  → Answer: {answer!r}")

        submissions.append({"task_id": tid, "submitted_answer": answer})
        rows.append({
            "Task ID": tid,
            "Question": task.get("question", "")[:120],
            "File": task.get("file_name", "") or "—",
            "Answer": answer or "(empty)",
        })

    # Submit
    try:
        result = submit_answers(username, space_id, submissions)
        score = result.get("score", "N/A")
        message = result.get("message", "")
        status = f"✅ Score: {score}%\n{message}"
    except Exception as exc:
        status = f"⚠️  Submission failed: {exc}"

    return status, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="GAIA Solver") as demo:
    gr.Markdown(
        """
        # 🏅 GAIA Benchmark Solver
        **Hugging Face AI Agents Course — Unit 4**

        **Before running:** make sure `GROQ_API_KEY` is set in your Space secrets.
        
        1. Log in below with your Hugging Face account
        2. Click **Run Benchmark**
        3. Wait ~5–10 min for all 20 questions to be solved and submitted
        """
    )

    gr.LoginButton()
    btn = gr.Button("🚀 Run Benchmark", variant="primary", size="lg")
    status_box = gr.Textbox(label="Result", lines=4)
    table = gr.DataFrame(label="Answers", wrap=True)

    btn.click(fn=run_benchmark, outputs=[status_box, table])


if __name__ == "__main__":
    demo.launch()