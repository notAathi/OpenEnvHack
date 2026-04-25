from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
import os

from environment.env import ConflictResolutionEnv
from environment.models import Action, Observation

app = FastAPI(title="Executive Conflict Resolution OpenEnv")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_envs: Dict[str, ConflictResolutionEnv] = {}
DEFAULT_SESSION = "default"


class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: str = "easy"


class StepRequest(BaseModel):
    session_id: Optional[str] = None
    action: Action


@app.get("/")
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"status": "ok", "env": "executive-conflict-resolution-openenv"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None):
    if req is None:
        req = ResetRequest()
    session_id = req.session_id or DEFAULT_SESSION
    env = ConflictResolutionEnv()
    _envs[session_id] = env
    return env.reset(task_level=req.task_id)


@app.post("/step")
def step(req: StepRequest):
    session_id = req.session_id or DEFAULT_SESSION
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found. Call /reset first.")
    obs, reward, done, info = env.step(req.action)
    return {"observation": obs.model_dump(), "reward": {"value": reward}, "done": done, "info": info}


@app.get("/state/{session_id}")
def state(session_id: str):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state().model_dump()


@app.get("/state")
def state_default():
    env = _envs.get(DEFAULT_SESSION)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return env.state().model_dump()


@app.get("/score/{session_id}")
def score(session_id: str):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"final_score": env.final_score()}


@app.get("/score")
def score_default():
    env = _envs.get(DEFAULT_SESSION)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"final_score": env.final_score()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
