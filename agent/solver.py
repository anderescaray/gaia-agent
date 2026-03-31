import re
import traceback

from smolagents import CodeAgent, LiteLLMModel
from smolagents import DuckDuckGoSearchTool, VisitWebpageTool

from agent.prompts import ANSWER_RULES
from tools.download import DownloadTaskFileTool
from tools.audio import AudioTranscriptionTool
from tools.youtube import YouTubeTranscriptTool


class GAIASolver:
    """
    Wraps a smolagents CodeAgent to solve GAIA benchmark tasks.

    Each task dict is expected to contain:
        task_id   : str  — UUID used to fetch attached files from the API
        question  : str  — The natural language question to answer
        file_name : str  — Optional filename hint (may be empty string)

    The solver injects task context (ID + file hint) into the prompt so the
    agent can call download_task_file when an attached file is present.
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
            # Allow enough steps for deep research chains (web + file + verify)
            max_steps=15,
            verbosity_level=1,
            additional_authorized_imports=[
                "pandas", "numpy", "re", "math", "datetime",
                "collections", "json", "csv", "itertools", "string",
            ],
        )

    def solve(self, task: dict) -> str:
        """Solve one GAIA task and return a cleaned answer string."""
        task_id = task.get("task_id", "")
        question = task.get("question", "")
        file_name = task.get("file_name", "")

        # Build structured prompt — rules first, then task context
        lines = [ANSWER_RULES, "---", f"Task ID: {task_id}"]
        if file_name:
            lines.append(
                f"This task has an attached file: '{file_name}'. "
                f"Call download_task_file(task_id='{task_id}') to access it BEFORE answering."
            )
        lines.append(f"\nQuestion: {question}")
        prompt = "\n".join(lines)

        try:
            # reset=True prevents context leaking between consecutive tasks
            raw = self.agent.run(prompt, reset=True)
            return self._clean(str(raw))
        except Exception as exc:
            print(f"[ERROR] Task {task_id}: {exc}")
            traceback.print_exc()
            return ""

    @staticmethod
    def _clean(raw: str) -> str:
        """
        Minimal post-processing of LLM output for exact-match scoring.
        Only removes patterns that are unambiguously wrong (preamble, fences).
        Conservative by design — avoids corrupting valid answers.
        """
        s = raw.strip()

        # Remove markdown code fences wrapping the entire answer
        if s.startswith("```") and s.endswith("```"):
            inner = s[3:-3].strip()
            first_line, _, rest = inner.partition("\n")
            s = rest.strip() if first_line.isalpha() else inner

        # Remove "The answer is:" / "Final answer:" type prefixes
        s = re.sub(
            r"^\s*(the\s+)?(final\s+)?(answer\s+(is|:)|result\s*:|value\s*:)\s*",
            "",
            s,
            flags=re.IGNORECASE,
        ).strip()

        # Remove symmetric surrounding quotes
        for q in ('"', "'"):
            if s.startswith(q) and s.endswith(q) and len(s) > 1:
                s = s[1:-1].strip()
                break

        return s