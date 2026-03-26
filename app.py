import os
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv

from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

# 1. CONFIGURATION
load_dotenv()
hf_token = os.getenv("HF_TOKEN")

# 2. CONFIGURATE THE AGENT (smolagents)
# Use Qwen 2.5 Coder 32B for its great reasoning ability
model = LiteLLMModel(
    model_id="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct", 
    api_key=hf_token 
)

search_tool = DuckDuckGoSearchTool()

# Initialize the agent outside the class to optimize performance
smol_agent = CodeAgent(
    tools=[search_tool],
    model=model,
    max_steps=10,
    verbosity_level=1,
    additional_authorized_imports=["pandas", "numpy", "re", "math"]
)

# 3. CONSTANTS AND CLASS
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

class BasicAgent:
    def __init__(self):
        print("--- smolagents agent initialized successfully ---")

    def __call__(self, question: str) -> str:
        """
        This function is called by the course evaluator.
        """
        print(f"\n[PROCESSING QUESTION]: {question[:100]}...")
        
        # Prompt específico para que el agente sea muy directo (clave para GAIA)
        prompt = f"""Solve the following question precisely. 
        IMPORTANT: Your final answer must be extremely concise. 
        If the answer is a number, date, or name, return ONLY that value. 
        Do not include any conversational text or explanations.

        Question: {question}"""
        
        try:
            # Ejecutamos la lógica de smolagents
            result = smol_agent.run(prompt)
            final_answer = str(result).strip()
            print(f"[ANSWER GENERATED]: {final_answer}")
            return final_answer
        except Exception as e:
            print(f"Error in the agent: {e}")
            return "Error processing the answer."

# 4. EVALUATION AND SUBMISSION LOGIC (Keeping the course's logic)
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")

    if not profile:
        return "Please log in with Hugging Face using the button above.", None

    username = profile.username
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # Instantiate our agent
    try:
        agent = BasicAgent()
    except Exception as e:
        return f"Error initializing the agent: {e}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    # Get questions
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Error getting questions: {e}", None

    # Execute agent on all questions
    results_log = []
    answers_payload = []
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Answer": answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Answer": f"ERROR: {e}"})

    # Submit to scoring server
    submission_data = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        
        status = (
            f"✅ Submission successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score')}% "
            f"({result_data.get('correct_count')}/{result_data.get('total_attempted')} correct)\n"
            f"Message: {result_data.get('message')}"
        )
        return status, pd.DataFrame(results_log)
    except Exception as e:
        return f"Error submitting: {e}", pd.DataFrame(results_log)

# 5. GRADIO INTERFACE
with gr.Blocks() as demo:
    gr.Markdown("# 🤖 Final Agent - AI Agents Course")
    gr.Markdown("Click the button to make the agent solve the GAIA benchmark and submit your score.")
    
    gr.LoginButton()
    run_button = gr.Button("🚀 Run Evaluation and Submit All", variant="primary")
    
    status_output = gr.Textbox(label="Submission Status", lines=5)
    results_table = gr.DataFrame(label="Answer Details")

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    demo.launch()