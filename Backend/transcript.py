from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


class NoCaptionsError(Exception):
    pass


def get_transcript(video_id: str):
    """Return caption snippets (each has .text and .start). Prefers manual captions."""
    api = YouTubeTranscriptApi()
    try:
        available = list(api.list(video_id))
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        raise NoCaptionsError(str(e))

    if not available:
        raise NoCaptionsError("No transcripts available")

    manual = [t for t in available if not t.is_generated]
    chosen = (manual or available)[0]
    return chosen.fetch()