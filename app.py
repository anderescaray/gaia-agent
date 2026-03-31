"""
GAIA Benchmark Solver — Hugging Face AI Agents Course (Unit 4)
==============================================================
A multimodal agent capable of answering GAIA benchmark questions using
web search, webpage visits, YouTube transcription, audio transcription,
and attached file handling (CSV, images, PDFs, audio).

Architecture:
    - Model  : Qwen/Qwen2.5-72B-Instruct via HF Inference (LiteLLM)
    - Agent  : smolagents CodeAgent (code-first reasoning loop)
    - Tools  : Search, Web, YouTube, Audio, File handling

Author: <your-name>
"""

import os
import re
import io
import base64
import requests
import tempfile
import traceback

import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, Tool
from youtube_transcript_api import YouTubeTranscriptApi

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Base URL for the GAIA scoring API
GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"

# ---------------------------------------------------------------------------
# Model — Use a reliable HF Inference endpoint
# ---------------------------------------------------------------------------

def build_model() -> LiteLLMModel:
    """
    Instantiate the LLM backend.

    Qwen2.5-72B-Instruct is strong for multi-step reasoning.
    Falls back gracefully if the token is missing.
    """
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN is not set. Add it to your .env or HF Space secrets.")

    return LiteLLMModel(
        model_id="huggingface/Qwen/Qwen2.5-72B-Instruct",
        api_key=HF_TOKEN,
        # Increase timeout — GAIA tasks can require long web browsing chains
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Custom Tools
# ---------------------------------------------------------------------------

class AudioTranscriptionTool(Tool):
    """
    Transcribes audio files (mp3, wav, m4a) using OpenAI Whisper (base model).

    Accepts either a local filesystem path or a remote URL.
    Downloads remote files to a temp directory before transcription.
    """

    name = "transcribe_audio"
    description = (
        "Transcribes an audio file and returns its full text. "
        "Accepts a local path or a direct download URL. "
        "Use this whenever a question involves an audio or mp3 file."
    )
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Local file path or HTTP/HTTPS URL pointing to the audio file.",
        }
    }
    output_type = "string"

    def forward(self, audio_path: str) -> str:
        # Lazy-import Whisper to avoid slow startup when the tool is not used
        try:
            import whisper  # noqa: PLC0415
        except ImportError:
            return "Whisper is not installed. Run: pip install openai-whisper"

        try:
            local_path = audio_path

            # Download if URL
            if audio_path.startswith("http://") or audio_path.startswith("https://"):
                response = requests.get(audio_path, timeout=60)
                response.raise_for_status()

                suffix = os.path.splitext(audio_path)[-1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(response.content)
                    local_path = tmp.name

            model_whisper = whisper.load_model("base")
            result = model_whisper.transcribe(local_path)
            return result["text"].strip()

        except Exception as exc:
            return f"[AudioTranscriptionTool] Failed: {exc}"


class YouTubeTranscriptTool(Tool):
    """
    Retrieves the full transcript of a YouTube video.

    Uses youtube-transcript-api; no browser or cookies required.
    Prefers English captions but falls back to any available language.
    """

    name = "get_youtube_transcript"
    description = (
        "Fetches the full text transcript of a YouTube video given its URL or video ID. "
        "Use this for questions that reference a YouTube link."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "Full YouTube URL (e.g. https://www.youtube.com/watch?v=...) or bare video ID.",
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
            # Try English first, then any available language
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            except Exception:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            return " ".join(entry["text"] for entry in transcript_list)

        except Exception as exc:
            return (
                f"[YouTubeTranscriptTool] Could not retrieve transcript for '{video_id}': {exc}. "
                "Try searching the web for a summary of this video."
            )


class DownloadFileTool(Tool):
    """
    Downloads a file attached to a GAIA question from the scoring API
    and returns its content as text (for CSV/JSON/TXT) or a local path (for binary files).

    The GAIA API exposes attached files at:
        GET {GAIA_API_BASE}/files/{task_id}
    """

    name = "download_task_file"
    description = (
        "Downloads the file attached to the current GAIA task and returns its content. "
        "For spreadsheets and text files, returns the raw text. "
        "For images or audio, saves to disk and returns the local path so other tools can process it. "
        "Use this whenever the question mentions 'the attached file', 'the file', 'the image', 'the audio', etc."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The GAIA task ID (UUID) whose attached file should be downloaded.",
        }
    }
    output_type = "string"

    TEXT_EXTENSIONS = {".csv", ".txt", ".json", ".md", ".tsv", ".xml", ".html"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}

    def forward(self, task_id: str) -> str:
        url = f"{GAIA_API_BASE}/files/{task_id}"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Determine file type from Content-Disposition or Content-Type
            content_disp = response.headers.get("Content-Disposition", "")
            content_type = response.headers.get("Content-Type", "")

            # Try to parse filename from header
            filename = ""
            if "filename=" in content_disp:
                filename = content_disp.split("filename=")[-1].strip().strip('"')

            ext = os.path.splitext(filename)[-1].lower() if filename else ""

            # Infer extension from MIME type if not available
            if not ext:
                if "csv" in content_type:
                    ext = ".csv"
                elif "image" in content_type:
                    ext = ".png"
                elif "audio" in content_type or "mpeg" in content_type:
                    ext = ".mp3"
                elif "json" in content_type:
                    ext = ".json"
                elif "text" in content_type:
                    ext = ".txt"

            # --- Text/data files: return content directly ---
            if ext in self.TEXT_EXTENSIONS:
                try:
                    return response.content.decode("utf-8")
                except UnicodeDecodeError:
                    return response.content.decode("latin-1")

            # --- Binary files: save to disk and return path ---
            suffix = ext or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(response.content)
                local_path = tmp.name

            if ext in self.IMAGE_EXTENSIONS:
                return f"[Image saved to {local_path}] Use vision capabilities or describe the image."
            if ext in self.AUDIO_EXTENSIONS:
                return f"[Audio saved to {local_path}] Pass this path to transcribe_audio."

            return f"[File saved to {local_path}] Extension: {ext}"

        except Exception as exc:
            return f"[DownloadFileTool] Failed to download file for task '{task_id}': {exc}"


# ---------------------------------------------------------------------------
# Agent System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert research agent solving GAIA benchmark questions.
Your goal is to return the EXACT correct answer — no explanations, no preamble.

RULES:
1. OUTPUT FORMAT: Return only the raw answer. If the answer is "42", output "42".
   Do NOT say "The answer is 42" or "Based on my research, 42".
2. LISTS: Sort alphabetically unless the question specifies another order. Use comma+space separation.
3. NUMBERS: Use the exact precision the question implies. Do not round unless asked.
4. SCIENCE: In botanical classification, tomatoes, peppers, cucumbers, and corn kernels are FRUITS.
   Only use "vegetable" if the question explicitly asks about culinary usage.
5. FILES: If the question references an attached file, ALWAYS call download_task_file first.
6. AUDIO: If the question involves an mp3 or audio, call transcribe_audio on the file path.
7. YOUTUBE: If the question contains a YouTube URL, call get_youtube_transcript immediately.
8. WEB SEARCH: If you need external data, search before answering. Verify facts from at least 2 sources.
9. MULTI-STEP: Break the problem into steps. Execute code to verify calculations. Never guess.
10. DATES: Use ISO 8601 format (YYYY-MM-DD) unless the question specifies another format.
"""


# ---------------------------------------------------------------------------
# GAIA Solver
# ---------------------------------------------------------------------------

class GAIASolver:
    """
    Orchestrates the smolagents CodeAgent to solve GAIA benchmark tasks.

    Each task may include:
    - A natural language question
    - An optional attached file (image, audio, CSV, etc.)

    The solver injects the task_id into the question context so the agent
    can call download_task_file when needed.
    """

    def __init__(self, model: LiteLLMModel):
        self.tools = [
            DuckDuckGoSearchTool(),
            VisitWebpageTool(),
            YouTubeTranscriptTool(),
            AudioTranscriptionTool(),
            DownloadFileTool(),
        ]
        self.agent = CodeAgent(
            tools=self.tools,
            model=model,
            # GAIA tasks can require deep research chains — allow enough steps
            max_steps=20,
            verbosity_level=1,
            additional_authorized_imports=[
                "pandas", "numpy", "re", "math", "datetime",
                "collections", "json", "csv", "itertools",
            ],
        )

    def solve(self, task: dict) -> str:
        """
        Solve a single GAIA task dict containing 'task_id', 'question',
        and optionally 'file_name'.

        Returns the cleaned answer string.
        """
        task_id = task.get("task_id", "")
        question = task.get("question", "")
        file_name = task.get("file_name", "")

        # Build a context-rich prompt for the agent
        context_lines = [f"Task ID: {task_id}"]
        if file_name:
            context_lines.append(
                f"Attached file: '{file_name}' — call download_task_file(task_id='{task_id}') to access it."
            )
        context_lines.append(f"\nQuestion: {question}")
        full_prompt = "\n".join(context_lines)

        try:
            raw_answer = self.agent.run(full_prompt, reset=True)
            return self._clean_answer(str(raw_answer))
        except Exception as exc:
            print(f"[ERROR] Task {task_id} failed: {exc}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _clean_answer(raw: str) -> str:
        """
        Strip common LLM verbosity patterns that violate GAIA's exact-match scoring.
        Conservative cleaning — only removes clear preamble patterns.
        """
        cleaned = raw.strip()

        # Remove markdown code fences if the entire answer is wrapped
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned[3:-3].strip()
            if cleaned.startswith(("python", "text", "json")):
                cleaned = cleaned.split("\n", 1)[-1].strip()

        # Remove explicit "The answer is:" type prefixes
        preamble_pattern = re.compile(
            r"^\s*(the\s+)?(final\s+)?(answer\s+is[:\s]*|result[:\s]*|value[:\s]*)",
            re.IGNORECASE,
        )
        cleaned = preamble_pattern.sub("", cleaned).strip()

        # Remove surrounding quotes only if symmetrically present
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        return cleaned


# ---------------------------------------------------------------------------
# Gradio UI & Benchmark Runner
# ---------------------------------------------------------------------------

def fetch_questions() -> list[dict]:
    """Fetch all questions from the GAIA scoring API."""
    response = requests.get(f"{GAIA_API_BASE}/questions", timeout=30)
    response.raise_for_status()
    return response.json()


def submit_answers(username: str, space_id: str, answers: list[dict]) -> dict:
    """Submit answers to the GAIA scoring API and return the result."""
    payload = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers,
    }
    response = requests.post(f"{GAIA_API_BASE}/submit", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def run_benchmark(profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame | None]:
    """
    Main benchmark runner invoked by the Gradio button.

    1. Validates the user is logged in via HF OAuth
    2. Fetches questions from the GAIA API
    3. Solves each question with the agent
    4. Submits answers and displays the score
    """
    if profile is None:
        return "⚠️ Please log in with your Hugging Face account first.", None

    username = profile.username
    space_id = os.getenv("SPACE_ID", "")

    status_log: list[str] = [f"🔐 Logged in as: {username}"]

    # --- Build model & solver ---
    try:
        model = build_model()
        solver = GAIASolver(model)
    except EnvironmentError as exc:
        return str(exc), None

    # --- Fetch questions ---
    try:
        questions = fetch_questions()
        status_log.append(f"📋 Fetched {len(questions)} questions.")
    except Exception as exc:
        return f"❌ Failed to fetch questions: {exc}", None

    # --- Solve ---
    submissions: list[dict] = []
    results_rows: list[dict] = []

    for i, task in enumerate(questions, start=1):
        task_id = task.get("task_id", f"task_{i}")
        question_preview = task.get("question", "")[:80]
        print(f"\n[{i}/{len(questions)}] Solving task {task_id}: {question_preview}...")

        answer = solver.solve(task)
        submissions.append({"task_id": task_id, "submitted_answer": answer})
        results_rows.append({
            "Task ID": task_id,
            "Question (preview)": task.get("question", "")[:120],
            "File": task.get("file_name", ""),
            "Submitted Answer": answer,
        })

    # --- Submit ---
    try:
        result = submit_answers(username, space_id, submissions)
        score = result.get("score", "N/A")
        message = result.get("message", "")
        status = f"✅ Submission complete — Score: {score}%\n{message}"
    except Exception as exc:
        status = f"⚠️ Submission failed: {exc}\nAnswers were generated but not submitted."

    return status, pd.DataFrame(results_rows)


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Soft(), title="GAIA Benchmark Solver") as demo:
    gr.Markdown(
        """
        # 🏅 GAIA Benchmark Solver
        **Hugging Face AI Agents Course — Unit 4 Final Assignment**

        This agent uses web search, file handling, YouTube transcription, and audio transcription
        to solve [GAIA benchmark](https://huggingface.co/datasets/gaia-benchmark/GAIA) questions.

        **Instructions:**
        1. Log in with your Hugging Face account below.
        2. Click **Run Benchmark** to start solving all questions.
        3. Results and your score will appear automatically.
        """
    )

    gr.LoginButton()

    run_btn = gr.Button("🚀 Run Full Benchmark", variant="primary", size="lg")
    status_box = gr.Textbox(
        label="Submission Status",
        lines=4,
        placeholder="Results will appear here after the benchmark completes...",
    )
    results_table = gr.DataFrame(label="Task Results", wrap=True)

    run_btn.click(
        fn=run_benchmark,
        outputs=[status_box, results_table],
    )

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch()