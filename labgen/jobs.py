"""Фоновые задачи генерации: граф LangGraph выполняется в потоке, а страница
показывает прогресс по шагам графа."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .agent.graph import NODE_LABELS

MAX_JOBS = 200


@dataclass
class JobStep:
    node: str
    label: str
    log: list[str]
    at: float


@dataclass
class Job:
    id: str
    owner_id: int | None
    title: str
    status: str = "running"  # running | done | error
    steps: list[JobStep] = field(default_factory=list)
    lab_id: str | None = None
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "lab_id": self.lab_id,
            "error": self.error,
            "elapsed": round((self.finished or time.time()) - self.started, 1),
            "steps": [{"node": s.node, "label": s.label, "log": s.log} for s in self.steps],
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, owner_id: int | None, title: str, work: Callable[[Callable[[str, list[str]], None]], str]) -> Job:
        """``work(on_step)`` выполняет генерацию и возвращает id созданной работы."""
        job = Job(id=uuid.uuid4().hex[:12], owner_id=owner_id, title=title)
        with self._lock:
            self._jobs[job.id] = job
            if len(self._jobs) > MAX_JOBS:
                for old in sorted(self._jobs.values(), key=lambda j: j.started)[: len(self._jobs) - MAX_JOBS]:
                    self._jobs.pop(old.id, None)

        def on_step(node: str, log: list[str]) -> None:
            job.steps.append(JobStep(node=node, label=NODE_LABELS.get(node, node), log=log, at=time.time()))

        def run() -> None:
            try:
                job.lab_id = work(on_step)
                job.status = "done"
            except Exception as exc:  # показываем пользователю причину сбоя
                job.error = str(exc) or type(exc).__name__
                job.status = "error"
            finally:
                job.finished = time.time()

        threading.Thread(target=run, daemon=True, name=f"labgen-job-{job.id}").start()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def wait(self, job_id: str, timeout: float = 120.0) -> Job | None:
        """Для тестов и CLI: дождаться завершения задачи."""
        deadline = time.time() + timeout
        job = self.get(job_id)
        while job is not None and job.status == "running" and time.time() < deadline:
            time.sleep(0.05)
        return job
