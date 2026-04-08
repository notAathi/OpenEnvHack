from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn

from environment.env import EmailTriageEnv
from environment.models import Action, Observation

app = FastAPI(title="Email Triage OpenEnv")

_envs: Dict[str, EmailTriageEnv] = {}


class ResetRequest(BaseModel):
    session_id: str
    task_id: str = "easy"


class StepRequest(BaseModel):
    session_id: str
    action: Action


@app.get("/")
def root():
    return {"status": "ok", "env": "email-triage-openenv"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    env = EmailTriageEnv()
    _envs[req.session_id] = env
    return env.reset(task_level=req.task_id)


@app.post("/step")
def step(req: StepRequest):
    env = _envs.get(req.session_id)
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


@app.get("/score/{session_id}")
def score(session_id: str):
    env = _envs.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"final_score": env.final_score()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
