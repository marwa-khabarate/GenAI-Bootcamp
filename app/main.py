from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent import PharmaAgent
from app.retrieval import CorpusIndex
from app.schemas import AskRequest, AskResponse, ExamplesResponse


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    index = CorpusIndex(ROOT / "corpus")
    app.state.agent = PharmaAgent(index)
    app.state.corpus_chunks = len(index.chunks)
    yield


app = FastAPI(title="Pharma Research Assistant", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    index = app.state.agent.index
    return {"status": "ok", "corpus_chunks": app.state.corpus_chunks, "retrieval_mode": index.embedding_mode}


@app.get("/examples", response_model=ExamplesResponse)
def examples() -> ExamplesResponse:
    return app.state.agent.examples()


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        return app.state.agent.answer(payload.question.strip())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="The research workflow could not complete.") from exc

