"""SDLC Pipeline — OpenAI Agents SDK.

Architecture:
  - OpenAI Agents SDK with function_tool and Agent.as_tool()
  - Orchestrator delegates to specialists via as_tool()
  - RunHooks capture events from all agents in real-time
  - SQLite persistence for tasks and events
  - Custom polling dashboard at /oai-pipeline/

Run: python sdlc_pipeline.py
Dashboard: http://localhost:9099/oai-pipeline/
"""

import asyncio
import json
import os
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel as PydanticBaseModel

from agents import (
    Agent,
    Runner,
    RunConfig,
    RunHooks,
    RunContextWrapper,
    RunResult,
    function_tool,
    set_tracing_disabled,
    ModelSettings,
)

# Disable tracing to avoid OpenAI telemetry noise
set_tracing_disabled(True)

WORKSPACE = os.path.expanduser("~/workspace")

# Global ref to current running task for event capture
_current_task = None


# --- Tools (same across all frameworks) ---

@function_tool
def read_file(path: str) -> str:
    """Read a file from the filesystem. Use absolute paths."""
    try:
        with open(path) as f:
            content = f.read()
        if len(content) > 10000:
            content = content[:10000] + "\n... (truncated)"
        return json.dumps({"content": content})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def write_file(path: str, content: str) -> str:
    """Write content to a file. Creates parent directories. Use absolute paths."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def list_files(dir: str) -> str:
    """List files and directories at a path. Use absolute paths."""
    try:
        entries = os.listdir(dir)
        return json.dumps({"files": sorted(e + ("/" if os.path.isdir(os.path.join(dir, e)) else "") for e in entries)})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def run_command(command: str, cwd: str = "") -> str:
    """Execute a shell command. Defaults to workspace/repos/. Use for git, build, test."""
    if not cwd:
        cwd = os.path.join(WORKSPACE, "repos")
    try:
        result = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60)
        stdout = result.stdout[:10000] if len(result.stdout) > 10000 else result.stdout
        stderr = result.stderr[:5000] if len(result.stderr) > 5000 else result.stderr
        return json.dumps({"stdout": stdout, "stderr": stderr, "exit_code": result.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "timed out after 60s"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Replace a specific text snippet in a file. Use for modifying parts of large files."""
    try:
        with open(path) as f:
            content = f.read()
        if old_text not in content:
            return json.dumps({"error": f"old_text not found in {path}"})
        with open(path, "w") as f:
            f.write(content.replace(old_text, new_text, 1))
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


@function_tool
def append_to_file(path: str, content: str) -> str:
    """Append content to end of an existing file."""
    try:
        with open(path, "a") as f:
            f.write(content)
        return json.dumps({"success": True})
    except Exception as e:
        return json.dumps({"error": str(e)})


DEV_TOOLS = [read_file, write_file, list_files, run_command, replace_in_file, append_to_file]


# --- Global instruction (same across all frameworks) ---

def load_global_instruction():
    parts = []
    agents_path = os.path.join(WORKSPACE, "AGENTS.md")
    if os.path.exists(agents_path):
        with open(agents_path) as f:
            parts.append(f.read())
    parts.append(f"""# Workspace
- Root: {WORKSPACE}
- Context: {WORKSPACE}/markdowns/<repo>/CLAUDE.md (READ-ONLY)
- Repos: {WORKSPACE}/repos/<repo>/ (clone here, work here)
- Read CLAUDE.md before working on any repo
- Absolute paths only. Never push to main/master/develop. Never commit secrets.""")
    return "\n\n".join(parts)


GLOBAL = load_global_instruction()


# --- Specialist agents ---

architect = Agent(
    name="architect",
    instructions=GLOBAL + "\n\n# Architect\nDesign systems, break down requirements. Read CLAUDE.md and code first. Max 10-15 files. Output: numbered task breakdown with files, acceptance criteria.",
    tools=DEV_TOOLS,
    model="gpt-5.5",
)

developer = Agent(
    name="developer",
    instructions=GLOBAL + "\n\n# Developer\nWrite clean code following repo patterns. Clone repos, create branches. For large files: use replace_in_file not write_file. Run builds to verify.",
    tools=DEV_TOOLS,
    model="gpt-5.5",
)

tester = Agent(
    name="tester",
    instructions=GLOBAL + "\n\n# Tester\nRead implementation first. Write tests. Run them via run_command. Report results.",
    tools=DEV_TOOLS,
    model="gpt-5.5",
)

reviewer = Agent(
    name="reviewer",
    instructions=GLOBAL + "\n\n# Reviewer\nReview: bugs, error handling, security, readability, test coverage. Score 0-100. Severity: blocker/major/minor/nit.",
    tools=DEV_TOOLS,
    model="gpt-5.5",
)

SPECIALISTS = {"architect": architect, "developer": developer, "tester": tester, "reviewer": reviewer}


# --- Event capture helper ---

def _add_event(author, text="", tool_calls=None, tool_results=None, is_error=False):
    """Add an event to the current task if one is running."""
    global _current_task
    if not _current_task:
        return
    _current_task.add_event({
        "author": author,
        "timestamp": time.time(),
        "text": text[:500] if text else "",
        "tool_calls": tool_calls or [],
        "tool_results": tool_results or [],
        "is_error": is_error,
    })


# --- Manual specialist tools with event capture ---

async def _run_specialist(agent_name: str, task_prompt: str) -> str:
    """Run a specialist and capture events."""
    global _current_task
    agent = SPECIALISTS[agent_name]

    if _current_task:
        _current_task.stage = agent_name
        _current_task.save()
        _add_event("orchestrator", tool_calls=[f"call_{agent_name}"])

    try:
        result = await Runner.run(
            agent,
            task_prompt,
            max_turns=25,
            run_config=RunConfig(tracing_disabled=True),
        )

        # Extract events from result items
        from agents import ToolCallItem, ToolCallOutputItem, MessageOutputItem
        for item in result.new_items:
            if isinstance(item, ToolCallItem):
                name = item.raw_item.name if hasattr(item.raw_item, 'name') else str(type(item.raw_item).__name__)
                args = ""
                if hasattr(item.raw_item, 'arguments'):
                    try:
                        args_dict = json.loads(item.raw_item.arguments) if isinstance(item.raw_item.arguments, str) else item.raw_item.arguments
                        if isinstance(args_dict, dict):
                            parts = []
                            for k, v in args_dict.items():
                                vs = str(v)[:60]
                                parts.append(f"{k}={vs}")
                            args = ", ".join(parts)
                    except:
                        pass
                call_str = f"{name}({args})" if args else name
                _add_event(agent_name, tool_calls=[call_str])

            elif isinstance(item, ToolCallOutputItem):
                output = item.output if isinstance(item.output, str) else str(item.output)
                _add_event(agent_name, tool_results=[output[:150]])

            elif isinstance(item, MessageOutputItem):
                text = ""
                if hasattr(item, 'raw_item') and hasattr(item.raw_item, 'content'):
                    for part in item.raw_item.content:
                        if hasattr(part, 'text'):
                            text += part.text
                if text:
                    _add_event(agent_name, text=text[:500])

        final_text = result.final_output or "No output"
        _add_event("orchestrator", tool_results=[f"call_{agent_name}: done"])
        return final_text[:5000]

    except Exception as e:
        error_msg = f"Error from {agent_name}: {e}"
        _add_event(agent_name, text=error_msg[:300], is_error=True)
        return error_msg[:500]


@function_tool
async def call_architect(task: str) -> str:
    """Delegate to the architect. They design systems, analyze code, produce implementation plans."""
    return await _run_specialist("architect", task)


@function_tool
async def call_developer(task: str) -> str:
    """Delegate to the developer. They implement code, create branches, run builds."""
    return await _run_specialist("developer", task)


@function_tool
async def call_tester(task: str) -> str:
    """Delegate to the tester. They write and run tests, report coverage."""
    return await _run_specialist("tester", task)


@function_tool
async def call_reviewer(task: str) -> str:
    """Delegate to the reviewer. They review code for bugs, security, quality. Score 0-100."""
    return await _run_specialist("reviewer", task)


# --- Orchestrator ---

orchestrator = Agent(
    name="orchestrator",
    instructions=GLOBAL + """

# Orchestrator
You are a dispatcher. You ONLY dispatch work to specialists. You NEVER do work yourself.

## How you work
1. Receive a goal
2. Discover available specialists from your tools — read their descriptions to understand what each one does
3. Pick the right specialist for the current step and call them with a clear task
4. Read the result
5. Decide: done -> summarize. Next step -> call another specialist. Need input -> ask user.

## Rules
- NEVER do work yourself. ALWAYS delegate to specialists.
- ONE specialist per step. Read result. Then decide.
- Pass context from previous results when delegating.
- Never re-assign the exact same task.""",
    tools=[call_architect, call_developer, call_tester, call_reviewer],
    model="gpt-5.5",
)


# --- SQLite persistence ---

DB_PATH = os.path.join(os.path.dirname(__file__), "oai_pipeline.db")


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
        task_id TEXT PRIMARY KEY, goal TEXT, status TEXT DEFAULT 'pending',
        stage TEXT DEFAULT '', created_at REAL, completed_at REAL, error TEXT, summary TEXT
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, author TEXT,
        timestamp REAL, text TEXT, tool_calls TEXT, tool_results TEXT, is_error INTEGER DEFAULT 0
    )""")
    conn.commit()
    conn.close()


_init_db()


@dataclass
class TaskInfo:
    task_id: str
    goal: str
    status: str = "pending"
    stage: str = ""
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    events: list = field(default_factory=list)
    error: Optional[str] = None
    summary: Optional[str] = None

    def save(self):
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?,?,?)",
            (self.task_id, self.goal, self.status, self.stage, self.created_at, self.completed_at, self.error, self.summary))
        conn.commit()
        conn.close()

    def add_event(self, ev):
        self.events.append(ev)
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO events (task_id, author, timestamp, text, tool_calls, tool_results, is_error) VALUES (?,?,?,?,?,?,?)",
            (self.task_id, ev.get("author", ""), ev.get("timestamp", 0), ev.get("text", ""),
             json.dumps(ev.get("tool_calls", [])), json.dumps(ev.get("tool_results", [])),
             1 if ev.get("is_error") else 0))
        conn.commit()
        conn.close()


def _load_tasks():
    result = {}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT * FROM tasks ORDER BY created_at DESC"):
        task = TaskInfo(task_id=row["task_id"], goal=row["goal"], status=row["status"],
                        stage=row["stage"], created_at=row["created_at"],
                        completed_at=row["completed_at"], error=row["error"], summary=row["summary"])
        for ev_row in conn.execute("SELECT * FROM events WHERE task_id=? ORDER BY id", (task.task_id,)):
            task.events.append({"author": ev_row["author"], "timestamp": ev_row["timestamp"],
                                "text": ev_row["text"], "tool_calls": json.loads(ev_row["tool_calls"]),
                                "tool_results": json.loads(ev_row["tool_results"]), "is_error": bool(ev_row["is_error"])})
        result[task.task_id] = task
    conn.close()
    return result


tasks = _load_tasks()


# --- Background runner ---

async def run_pipeline(task_id: str):
    global _current_task
    task = tasks[task_id]
    task.status = "running"
    task.save()
    _current_task = task

    try:
        result = await Runner.run(
            orchestrator,
            task.goal,
            max_turns=25,
            run_config=RunConfig(tracing_disabled=True),
        )

        # Capture orchestrator's final output
        if result.final_output:
            _add_event("orchestrator", text=result.final_output[:500])

        task.summary = result.final_output[:2000] if result.final_output else ""
        task.status = "done"
        task.completed_at = time.time()
        task.save()

    except Exception as e:
        task.status = "failed"
        task.error = str(e)[:500]
        task.completed_at = time.time()
        task.save()
    finally:
        _current_task = None


# --- FastAPI ---

app = FastAPI(title="SDLC Pipeline — OpenAI Agents SDK")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class SubmitRequest(PydanticBaseModel):
    goal: str


@app.post("/oai-pipeline/api/submit")
async def submit_task(req: SubmitRequest):
    task_id = str(uuid.uuid4())[:8]
    task = TaskInfo(task_id=task_id, goal=req.goal)
    task.save()
    tasks[task_id] = task
    asyncio.create_task(run_pipeline(task_id))
    return {"task_id": task_id, "status": "pending"}


@app.get("/oai-pipeline/api/tasks")
async def list_tasks_api():
    return [{"task_id": t.task_id, "goal": t.goal, "status": t.status, "stage": t.stage,
             "created_at": t.created_at, "completed_at": t.completed_at,
             "event_count": len(t.events), "error": t.error}
            for t in sorted(tasks.values(), key=lambda x: x.created_at, reverse=True)]


@app.get("/oai-pipeline/api/task/{task_id}")
async def get_task_api(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"task_id": task.task_id, "goal": task.goal, "status": task.status, "stage": task.stage,
            "created_at": task.created_at, "completed_at": task.completed_at, "error": task.error,
            "summary": task.summary, "event_count": len(task.events)}


@app.get("/oai-pipeline/api/task/{task_id}/events")
async def get_events_api(task_id: str, since: int = 0):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {"task_id": task_id, "status": task.status, "stage": task.stage,
            "total": len(task.events), "events": task.events[since:]}


@app.get("/oai-pipeline/api/task/{task_id}/stats")
async def get_stats_api(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    stats = {}
    for ev in task.events:
        a = ev.get("author", "?")
        if a not in stats:
            stats[a] = {"events": 0, "tool_calls": 0, "tool_errors": 0, "text_chars": 0}
        stats[a]["events"] += 1
        stats[a]["tool_calls"] += len(ev.get("tool_calls", []))
        stats[a]["tool_errors"] += 1 if ev.get("is_error") else 0
        stats[a]["text_chars"] += len(ev.get("text", ""))
    return {"task_id": task_id, "stats": stats}


@app.delete("/oai-pipeline/api/task/{task_id}")
async def cancel_task_api(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    task.status = "cancelled"
    task.completed_at = time.time()
    task.save()
    return {"task_id": task_id, "status": "cancelled"}


@app.get("/oai-pipeline/", response_class=HTMLResponse)
@app.get("/oai-pipeline", response_class=HTMLResponse)
async def dashboard():
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text()


if __name__ == "__main__":
    import uvicorn
    print("SDLC Pipeline (OpenAI Agents SDK): http://localhost:9099/oai-pipeline/")
    uvicorn.run(app, host="127.0.0.1", port=9099)
