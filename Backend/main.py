from dotenv import load_dotenv

load_dotenv()  # must run before importing rag (reads env at import time)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import rag
from transcript import get_transcript, NoCaptionsError

app = FastAPI(title="YT Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestReq(BaseModel):
    video_id: str


class ChatReq(BaseModel):
    video_id: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestReq):
    if rag.has_video(req.video_id):
        return {"status": "already_ingested"}

    try:
        snippets = get_transcript(req.video_id)
    except NoCaptionsError:
        raise HTTPException(status_code=404, detail="No captions found for this video.")
    except Exception as e:
        print(f"Transcript fetch failed for {req.video_id}: {e!r}")
        raise HTTPException(
            status_code=502,
            detail="Could not fetch captions from YouTube (video unavailable or request blocked).",
        )

    n_chunks = rag.ingest(req.video_id, snippets)
    return {"status": "ok", "chunks": n_chunks}


@app.post("/chat")
def chat(req: ChatReq):
    if not rag.has_video(req.video_id):
        raise HTTPException(status_code=400, detail="Video not ingested yet.")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")

    answer, _ = rag.ask(req.video_id, req.question)
    return {"answer": answer}