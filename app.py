import os
import pandas as pd
import gradio as gr
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool, Tool
from youtube_transcript_api import YouTubeTranscriptApi

# --- Environment Setup ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model Configuration ---
# Using Qwen 2.5 72B for deep reasoning. If you get timeout errors, switch to 32B-Coder.
model = LiteLLMModel(
    model_id="huggingface/Qwen/Qwen2.5-72B-Instruct",
    api_key=HF_TOKEN
)

# --- Custom YouTube Tool ---
class YouTubeTranscriptTool(Tool):
    name = "get_youtube_transcript"
    description = "Retrieves the transcript of a YouTube video. Essential for questions about video content."
    inputs = {"url": {"type": "string", "description": "The YouTube URL or video ID."}}
    output_type = "string"

    def forward(self, url: str):
        video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else url
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([t['text'] for t in transcript])
        except Exception as e:
            return f"Error retrieving transcript: {str(e)}. Please search for video details on the web instead."

# --- Agent Initialization ---
# Note: system_prompt is handled via prompt_templates or direct injection in 2026 versions
search_tool = DuckDuckGoSearchTool()
visit_page = VisitWebpageTool()
youtube_tool = YouTubeTranscriptTool()

agent = CodeAgent(
    tools=[search_tool, visit_page, youtube_tool],
    model=model,
    max_steps=15,
    verbosity_level=1,
    additional_authorized_imports=["pandas", "numpy", "re", "math", "datetime", "collections"]
)

# --- Business Logic Wrapper ---
class GAIAResearcher:
    """Professional Agent Wrapper for GAIA Benchmark Compliance."""
    
    def __init__(self):
        self.instructions = (
            "SYSTEM INSTRUCTIONS:\n"
            "1. BOTANY: Strictly categorize seeds (tomatoes, peppers, etc.) as FRUITS, not vegetables.\n"
            "2. YOUTUBE: Always check transcripts for video links.\n"
            "3. FORMAT: Output ONLY the final answer value. No explanations.\n"
            "4. SEARCH: If the first source is unclear, cross-reference with at least one more site.\n"
        )

    def __call__(self, question: str) -> str:
        print(f"\n[RESEARCHING]: {question[:120]}...")
        
        # Injecting instructions directly into the prompt to avoid constructor errors
        full_prompt = f"{self.instructions}\n\nTask: {question}"
        
        try:
            result = agent.run(full_prompt)
            # Ensure the output is a string and stripped of conversational noise
            answer = str(result).strip()
            # Emergency cleanup for 'The answer is' patterns
            if "answer is" in answer.lower():
                answer = answer.split("is")[-1].strip().rstrip('.')
            return answer
        except Exception as e:
            print(f"[RUNTIME ERROR]: {e}")
            return "Execution Error"

# --- Evaluation & UI Framework ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_evaluation_suite(profile: gr.OAuthProfile | None):
    if not profile:
        return "Authentication error: Please login with Hugging Face.", None

    researcher = GAIAResearcher()
    space_id = os.getenv("SPACE_ID")
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        # Fetch questions from the course API
        questions_resp = requests.get(f"{DEFAULT_API_URL}/questions")
        questions_resp.raise_for_status()
        questions = questions_resp.json()
        
        answers_payload = []
        detailed_log = []

        for item in questions:
            task_id = item['task_id']
            question_text = item['question']
            
            final_answer = researcher(question_text)
            
            answers_payload.append({"task_id": task_id, "submitted_answer": final_answer})
            detailed_log.append({"Task ID": task_id, "Result": final_answer})

        # Submit results
        submission = {
            "username": profile.username,
            "agent_code": agent_code,
            "answers": answers_payload
        }
        
        submit_resp = requests.post(f"{DEFAULT_API_URL}/submit", json=submission)
        submit_resp.raise_for_status()
        data = submit_resp.json()
        
        summary = (
            f"✅ Evaluation Complete\n"
            f"Score: {data.get('score')}% ({data.get('correct_count')}/20)\n"
            f"Message: {data.get('message')}"
        )
        return summary, pd.DataFrame(detailed_log)

    except Exception as e:
        return f"System Error: {str(e)}", None

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 High-Precision GAIA Agent")
    gr.Markdown("Advanced autonomous researcher with YouTube and Botanical specialized reasoning.")
    
    gr.LoginButton()
    run_btn = gr.Button("Start Benchmark Evaluation", variant="primary")
    
    status_msg = gr.Textbox(label="Status", lines=3)
    results_df = gr.DataFrame(label="Task Results")

    run_btn.click(fn=run_evaluation_suite, outputs=[status_msg, results_df])

if __name__ == "__main__":
    demo.launch()