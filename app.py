"""
GAIA Benchmark Solver — Hugging Face AI Agents Course (Unit 4)
==============================================================
Entry point: Gradio UI + benchmark runner.

Architecture:
    agent/model.py   — LLM backend (Anthropic Claude Haiku via LiteLLM)
    agent/solver.py  — GAIASolver: smolagents CodeAgent orchestration
    agent/prompts.py — Exact-match answer formatting rules
    tools/download.py — GAIA attached file downloader
    tools/audio.py   — Whisper-based audio transcription
    tools/youtube.py — YouTube transcript extraction

Setup (HF Space secrets):
    ANTHROPIC_API_KEY  — from console.anthropic.com (required)
    HF_TOKEN           — your HF token (required for login + submission)
    SPACE_ID           — set automatically by HF Spaces
"""

import os

import requests
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from agent import build_model, GAIASolver

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def fetch_questions() -> list[dict]:
    """Fetch all 20 GAIA Level 1 questions from the scoring API."""
    r = requests.get(f"{GAIA_API_BASE}/questions", timeout=30)
    r.raise_for_status()
    return r.json()


def submit_answers(username: str, space_id: str, answers: list[dict]) -> dict:
    """Submit solved answers to the scoring API and return the result."""
    payload = {
        "username": username,
        "agent_code": f"https://huggingface.co/spaces/{space_id}/tree/main",
        "answers": answers,
    }
    r = requests.post(f"{GAIA_API_BASE}/submit", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame | None]:
    """
    Full benchmark pipeline invoked by the Gradio button:
      1. Validate HF login
      2. Build model + solver
      3. Fetch questions from GAIA API
      4. Solve each question with the agent
      5. Submit answers and return score + results table
    """
    if profile is None:
        return "⚠️  Please log in with your Hugging Face account first.", None

    username = profile.username
    space_id = os.getenv("SPACE_ID", "")
    print(f"Starting benchmark for user: {username}")

    # Build model and solver
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

    # Solve all questions
    submissions: list[dict] = []
    rows: list[dict] = []

    for i, task in enumerate(questions, start=1):
        tid = task.get("task_id", f"task_{i}")
        preview = task.get("question", "")[:80]
        print(f"\n[{i:02d}/{len(questions)}] {tid}: {preview}...")

        answer = solver.solve(task)
        print(f"  → Answer: {answer!r}")

        submissions.append({"task_id": tid, "submitted_answer": answer})
        rows.append({
            "Task ID": tid,
            "Question": task.get("question", "")[:120],
            "File": task.get("file_name", "") or "—",
            "Answer": answer or "(empty)",
        })

    # Submit to scoring API
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

with gr.Blocks(theme=gr.themes.Soft(), title="GAIA Benchmark Solver") as demo:
    gr.Markdown(
        """
        # 🏅 GAIA Benchmark Solver
        **Hugging Face AI Agents Course — Unit 4**

        **Before running:** make sure `ANTHROPIC_API_KEY` is set in your Space secrets.

        1. Log in with your Hugging Face account below
        2. Click **Run Benchmark**
        3. Wait ~20–30 min for all 20 questions to be solved and submitted
        """
    )

    gr.LoginButton()
    btn = gr.Button("🚀 Run Benchmark", variant="primary", size="lg")
    status_box = gr.Textbox(label="Result", lines=4)
    table = gr.DataFrame(label="Answers", wrap=True)

    btn.click(fn=run_benchmark, outputs=[status_box, table])


if __name__ == "__main__":
    demo.launch()