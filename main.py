from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent import root_agent
import dotenv
import os
import time
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# ── Setup ──────────────────────────────────────────────────────────────────────
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Job Prep AI Agent",
    description="Search jobs, select one, get interview prep and research guide.",
    version="3.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ──────────────────────────────────────────────────────────────────
APP_NAME  = "job-search-agent"
USER_ID   = "bachi"
SESSION_ID = "bachi-session"

# ── Shared session service ─────────────────────────────────────────────────────
session_service = InMemorySessionService()


# ── Create persistent session at startup ──────────────────────────────────────
@app.on_event("startup")
async def startup():
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={},
    )
    logger.info(f"Persistent session created: {SESSION_ID}")


# ── Request / Response Models ──────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str

    @validator("query")
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()


class ChatResponse(BaseModel):
    query: str
    response: str
    time_taken_seconds: float
    session_state: dict


# ── Core Agent Runner ──────────────────────────────────────────────────────────
async def run_agent(query: str) -> dict:
    start_time = time.time()

    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name=APP_NAME,
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    # Fetch updated session state
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    state = dict(session.state) if session and session.state else {}

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Query: '{query}' | Time: {elapsed}s")

    return {
        "query": query,
        "response": final_response,
        "time_taken_seconds": elapsed,
        "session_state": state,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {
        "status": "healthy",
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
        "model": "gemini-2.5-flash",
        "agent": "Job Prep Pipeline",
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_post(request: ChatRequest):
    """
    Send a message to the job prep agent.

    Turn 1 — search for jobs:
      { "query": "data science jobs in Bangalore" }
      → returns 5 jobs and asks you to pick one

    Turn 2 — select a job:
      { "query": "2" }
      → returns interview questions + research guide
    """
    try:
        return await run_agent(request.query)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_get(query: str = Query(..., min_length=1, max_length=1000)):
    """
    Same as POST /chat but via GET.
    Example: /chat?query=data science jobs in Bangalore
    """
    try:
        return await run_agent(query)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", tags=["Session"])
async def reset_session():
    """
    Reset the conversation — clears selected_job, interview_prep, research_guide.
    Call this when you want to start a fresh job search.
    """
    await session_service.delete_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={},
    )
    logger.info("Session reset.")
    return {"message": "Session reset. Ready for a new job search."}


@app.get("/state", tags=["Session"])
async def get_state():
    """
    See the current session state.
    Shows selected_job, interview_prep, research_guide as they get populated.
    """
    session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Restart the server.")
    return {"state": dict(session.state)}


# Add this endpoint
@app.get("/")
def serve_ui():
    return FileResponse("index.html")
