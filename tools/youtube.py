from smolagents import Tool
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeTranscriptTool(Tool):
    """
    Fetches the full text transcript of a YouTube video.

    Uses youtube-transcript-api — no browser, cookies, or authentication needed.
    Prefers English captions and falls back to any available language.
    Supports full URLs (youtube.com/watch?v=, youtu.be/) and bare video IDs.
    """

    name = "get_youtube_transcript"
    description = (
        "Fetches the complete text transcript of a YouTube video. "
        "Accepts a full YouTube URL or a bare video ID. "
        "Use this immediately whenever a question contains a YouTube link."
    )
    inputs = {
        "url": {
            "type": "string",
            "description": "Full YouTube URL (e.g. https://www.youtube.com/watch?v=...) or bare video ID.",
        }
    }
    output_type = "string"

    def forward(self, url: str) -> str:
        # Extract video ID from common YouTube URL formats
        video_id = url
        if "v=" in url:
            video_id = url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[-1].split("?")[0]

        try:
            # Prefer English; fall back to any available language
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