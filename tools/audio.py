import os
import tempfile

import requests
from smolagents import Tool


class AudioTranscriptionTool(Tool):
    """
    Transcribes audio files (mp3, wav, m4a) using OpenAI Whisper (base model).

    Accepts either a local filesystem path or a remote HTTP/HTTPS URL.
    Remote files are downloaded to a temporary directory before transcription.
    Whisper is lazy-imported to avoid slow startup when the tool is not used.
    """

    name = "transcribe_audio"
    description = (
        "Transcribes an audio file and returns the full text. "
        "Accepts a local file path or a direct download URL (http/https). "
        "Use this whenever a question involves an mp3, wav, or other audio file."
    )
    inputs = {
        "audio_path": {
            "type": "string",
            "description": "Local filesystem path or HTTP/HTTPS URL to the audio file.",
        }
    }
    output_type = "string"

    def forward(self, audio_path: str) -> str:
        try:
            import whisper  # noqa: PLC0415
        except ImportError:
            return "Whisper is not installed. Run: pip install openai-whisper"

        local_path = audio_path

        # Download remote file to a temp location before transcribing
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