import os
import pandas as pd
import gradio as gr
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, VisitWebpageTool

# --- Configuration & Environment ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# --- Model Initialization ---
# Using Qwen2.5-Coder-32B for its specialized ability to write and execute Python code
model = LiteLLMModel(
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
    api_key=HF_TOKEN
)

# --- Toolset Definition ---
search_tool = DuckDuckGoSearchTool()
visit_page_tool = VisitWebpageTool()

# --- Agent Core Logic ---
# Note: system_prompt is removed from __init__ to ensure compatibility with smolagents v1.1.0+
smol_agent = CodeAgent(
    tools=[search_tool, visit_page_tool],
    model=model,
    max_steps=15,
    verbosity_level=1,
    additional_authorized_imports=["pandas", "numpy", "re", "math", "datetime", "statistics"]
)

class GAIAAssistant:
    """Professional wrapper for the GAIA benchmark evaluation."""
    
    def __init__(self):
        # We define the personality here to keep the code clean and modular
        self.system_instructions = (
            "You are an expert AI researcher. Solve the task step-by-step.\n"
            "1. Search for info. 2. Use 'visit_webpage' for deep reading if needed.\n"
            "3. Use Python to process data. 4. Output ONLY the final concise answer.\n"
        )

    def __call__(self, question: str) -> str:
        print(f"\n[TASK RECEIVED]: {question[:120]}...")
        
        # We prepend the instructions to the user question
        full_task = f"{self.system_instructions}\nTask: {question}"
        
        try:
            result = smol_agent.run(full_task)
            final_output = str(result).strip()
            print(f"[TASK COMPLETED]: {final_output}")
            return final_output
        except Exception as e:
            print(f"[EXECUTION ERROR]: {e}")
            return "Execution Error"

# --- Course Evaluation Framework ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

def run_evaluation_suite(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if not profile:
        return "Authentication required. Please log in via Hugging Face.", None

    username = profile.username
    questions_url = f"{DEFAULT_API_URL}/questions"
    submit_url = f"{DEFAULT_API_URL}/submit"

    assistant = GAIAAssistant()
    agent_code_link = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions = response.json()
    except Exception as e:
        return f"Failed to retrieve questions: {e}", None

    results = []
    payload = []
    
    for item in questions:
        task_id = item.get("task_id")
        text = item.get("question")
        
        answer = assistant(text)
        payload.append({"task_id": task_id, "submitted_answer": answer})
        results.append({"Task ID": task_id, "Question": text, "Agent Answer": answer})

    submission = {
        "username": username,
        "agent_code": agent_code_link,
        "answers": payload
    }

    try:
        res = requests.post(submit_url, json=submission, timeout=60)
        res.raise_for_status()
        data = res.json()
        
        summary = (
            f"✅ Submission Successful\n"
            f"User: {data.get('username')}\n"
            f"Global Score: {data.get('score')}% "
            f"({data.get('correct_count')}/{data.get('total_attempted')} correct)\n"
            f"Status: {data.get('message')}"
        )
        return summary, pd.DataFrame(results)
    except Exception as e:
        return f"Submission Failed: {e}", pd.DataFrame(results)

# --- Gradio UI Deployment ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🚀 Advanced GAIA Solver")
    gr.Markdown("Autonomous agent specialized in high-precision information retrieval and data processing.")
    
    gr.LoginButton()
    evaluate_btn = gr.Button("Execute Benchmark & Submit Score", variant="primary")
    
    status_box = gr.Textbox(label="Evaluation Status", lines=4)
    results_df = gr.DataFrame(label="Detailed Execution Log")

    evaluate_btn.click(
        fn=run_evaluation_suite,
        outputs=[status_box, results_df]
    )

if __name__ == "__main__":
    interface.launch()