import os
import tempfile

import requests
from smolagents import Tool

GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"


class DownloadTaskFileTool(Tool):
    """
    Downloads the file attached to a GAIA task from the scoring API.

    Text-based files (CSV, JSON, TXT, etc.) are decoded and returned as strings
    so the agent can process them directly. Binary files (audio, images) are
    saved to a temporary path and the path is returned so other tools
    (e.g. AudioTranscriptionTool) can process them in a follow-up step.

    API endpoint: GET {GAIA_API_BASE}/files/{task_id}
    """

    name = "download_task_file"
    description = (
        "Downloads the file attached to the current GAIA task and returns its contents. "
        "For CSV/JSON/TXT returns the decoded text directly. "
        "For audio/image files, saves to disk and returns the local path. "
        "Call this FIRST whenever the question mentions 'the file', 'the image', "
        "'the audio', or 'attached'."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The GAIA task UUID whose attached file should be downloaded.",
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

        content_disp = r.headers.get("Content-Disposition", "")
        content_type = r.headers.get("Content-Type", "")

        # Resolve filename and extension from response headers
        filename = ""
        if "filename=" in content_disp:
            filename = content_disp.split("filename=")[-1].strip().strip('"')
        ext = os.path.splitext(filename)[-1].lower()

        # Fall back to MIME type if extension is not available
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

        # Return text files as decoded strings
        if ext in self._TEXT_EXTS:
            try:
                return r.content.decode("utf-8")
            except UnicodeDecodeError:
                return r.content.decode("latin-1")

        # Save binary files to disk and return the path
        suffix = ext or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(r.content)
            local_path = tmp.name

        if ext in self._AUDIO_EXTS:
            return f"Audio file saved to: {local_path}\nPass this path to transcribe_audio."
        if ext in self._IMAGE_EXTS:
            return f"Image file saved to: {local_path}\nDescribe the image to answer the question."

        return f"File saved to: {local_path} (type: {ext or 'unknown'})"