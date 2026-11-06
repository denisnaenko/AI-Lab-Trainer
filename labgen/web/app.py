"""Веб-интерфейс преподавателя и JSON API (FastAPI)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import markdown
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError

from ..catalog import list_topics
from ..levels import LEVELS
from ..models import Mutant, Requirement
from ..pipeline.generator import TopicNotFound
from ..llm.base import LLMError
from ..service import EXPORT_KINDS, ApprovalError, LabService
from ..storage import LabNotFound

HERE = Path(__file__).parent


def render_md(text: str) -> str:
    return markdown.markdown(text or "", extensions=["tables", "fenced_code"])


class CreateLabRequest(BaseModel):
    topic: str
    level: int = 1
    mode: str = "auto"
    topic_id: str | None = None
    materials: str | None = None


class GradeRequest(BaseModel):
    code: str
    student: str | None = None


def create_app(service: LabService | None = None) -> FastAPI:
    service = service or LabService()
    app = FastAPI(title="AI Lab Trainer", description="ИИ-агент для создания лабораторных работ по агентной разработке")
    app.state.service = service
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["md"] = render_md
    templates.env.globals.update(levels=LEVELS, export_kinds=EXPORT_KINDS)

    def page(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
        context.setdefault("llm_available", service.llm_available)
        context.setdefault("llm_name", getattr(service.provider, "name", ""))
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def get_record(lab_id: str):
        try:
            return service.store.get(lab_id)
        except LabNotFound:
            raise HTTPException(status_code=404, detail="Лабораторная работа не найдена")

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return page(request, "error.html", status_code=exc.status_code, message=exc.detail)

    # --- страницы -------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return page(request, "index.html", labs=service.store.list(), topics=list_topics(), form={})

    @app.post("/labs", response_class=HTMLResponse)
    def create_lab(
        request: Request,
        topic: str = Form(...),
        level: int = Form(1),
        mode: str = Form("auto"),
        topic_id: str = Form(""),
        materials: str = Form(""),
        simulate: int = Form(0),
    ):
        form = {"topic": topic, "level": level, "mode": mode, "topic_id": topic_id, "materials": materials}
        try:
            record = service.create(
                topic.strip(),
                level,
                mode=mode,
                topic_id=topic_id or None,
                materials=materials or None,
                simulate=simulate,
            )
        except (TopicNotFound, ValueError, LLMError) as exc:
            return page(
                request, "index.html", status_code=400,
                labs=service.store.list(), topics=list_topics(), form=form, error=str(exc),
            )
        return RedirectResponse(f"/labs/{record.id}", status_code=303)

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog(request: Request):
        return page(request, "catalog.html", topics=list_topics())

    @app.get("/labs/{lab_id}", response_class=HTMLResponse)
    def lab_page(request: Request, lab_id: str, message: str = "", error: str = ""):
        record = get_record(lab_id)
        return page(
            request,
            "lab.html",
            record=record,
            spec=record.spec,
            tests_map=record.spec.tests_by_requirement(),
            grades=service.store.list_grades(lab_id)[:20],
            versions=service.store.versions(lab_id),
            message=message,
            error=error,
        )

    @app.get("/labs/{lab_id}/edit", response_class=HTMLResponse)
    def edit_page(request: Request, lab_id: str):
        record = get_record(lab_id)
        mutants_json = json.dumps([m.model_dump() for m in record.spec.mutants], ensure_ascii=False, indent=2)
        return page(request, "edit.html", record=record, spec=record.spec, mutants_json=mutants_json, error="")

    @app.post("/labs/{lab_id}/edit", response_class=HTMLResponse)
    async def save_edit(request: Request, lab_id: str):
        record = get_record(lab_id)
        form = await request.form()
        changes: dict = {}
        for field in ("title", "summary", "statement_md", "starter_code", "reference_solution", "tests_public", "tests_hidden"):
            if field in form:
                changes[field] = str(form[field]).replace("\r\n", "\n")
        minutes = str(form.get("estimated_minutes", "")).strip()
        changes["estimated_minutes"] = int(minutes) if minutes.isdigit() else None
        requirements = []
        error = ""
        for i in range(int(form.get("req_count", 0)) + 1):
            req_id = str(form.get(f"req_id_{i}", "")).strip()
            text = str(form.get(f"req_text_{i}", "")).strip()
            if not req_id and not text:
                continue
            try:
                requirements.append(
                    Requirement(
                        id=req_id,
                        text=text,
                        points=int(str(form.get(f"req_points_{i}", "0")) or 0),
                        auto=form.get(f"req_auto_{i}") == "on",
                    ).model_dump()
                )
            except (ValidationError, ValueError) as exc:
                error = f"Требование «{req_id}»: {exc}"
        if requirements:
            changes["requirements"] = requirements
        try:
            mutants = json.loads(str(form.get("mutants_json", "[]")) or "[]")
            changes["mutants"] = [Mutant.model_validate(m).model_dump() for m in mutants]
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            error = error or f"Мутанты: некорректный JSON ({exc})"
        if error:
            mutants_json = str(form.get("mutants_json", "[]"))
            return page(request, "edit.html", status_code=400, record=record, spec=record.spec, mutants_json=mutants_json, error=error)
        record = service.update(lab_id, changes, note=str(form.get("note", "")).strip())
        status = "проверка пройдена" if record.validation and record.validation.passed else "проверка нашла ошибки"
        return RedirectResponse(f"/labs/{lab_id}?message={quote('Сохранено, ' + status)}#validation", status_code=303)

    @app.post("/labs/{lab_id}/validate")
    def revalidate(lab_id: str):
        get_record(lab_id)
        record = service.validate(lab_id)
        status = "пройдена" if record.validation and record.validation.passed else "нашла ошибки"
        return RedirectResponse(f"/labs/{lab_id}?message={quote('Проверка ' + status)}#validation", status_code=303)

    @app.post("/labs/{lab_id}/approve")
    def approve(lab_id: str):
        get_record(lab_id)
        try:
            service.approve(lab_id)
        except ApprovalError as exc:
            return RedirectResponse(f"/labs/{lab_id}?error={quote(str(exc))}", status_code=303)
        return RedirectResponse(f"/labs/{lab_id}?message={quote('Работа утверждена к публикации')}", status_code=303)

    @app.post("/labs/{lab_id}/unapprove")
    def unapprove(lab_id: str):
        get_record(lab_id)
        service.unapprove(lab_id)
        return RedirectResponse(f"/labs/{lab_id}?message={quote('Работа возвращена в черновики')}", status_code=303)

    @app.post("/labs/{lab_id}/restore/{version}")
    def restore(lab_id: str, version: int):
        get_record(lab_id)
        try:
            service.restore_version(lab_id, version)
        except LabNotFound:
            raise HTTPException(404, "Версия не найдена")
        return RedirectResponse(f"/labs/{lab_id}?message={quote(f'Восстановлена версия {version}')}", status_code=303)

    @app.post("/labs/{lab_id}/delete")
    def delete(lab_id: str):
        get_record(lab_id)
        service.store.delete(lab_id)
        return RedirectResponse("/", status_code=303)

    @app.get("/labs/{lab_id}/export/{kind}")
    def export(lab_id: str, kind: str):
        get_record(lab_id)
        if kind not in EXPORT_KINDS:
            raise HTTPException(404, "Неизвестный формат экспорта")
        filename, data = service.export(lab_id, kind)
        return Response(
            data,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )

    @app.post("/labs/{lab_id}/grade")
    async def grade(
        lab_id: str,
        student: str = Form(""),
        code: str = Form(""),
        file: UploadFile | None = File(None),
        llm_feedback: str = Form(""),
    ):
        get_record(lab_id)
        if file is not None and file.filename:
            code = (await file.read()).decode("utf-8", errors="replace")
            student = student or Path(file.filename).stem
        if not code.strip():
            return RedirectResponse(f"/labs/{lab_id}?error={quote('Загрузите файл или вставьте код решения')}#grading", status_code=303)
        grade_id, _ = service.grade(lab_id, code, student=student or None, llm_feedback=llm_feedback == "on")
        return RedirectResponse(f"/labs/{lab_id}/grades/{grade_id}", status_code=303)

    @app.get("/labs/{lab_id}/grades/{grade_id}", response_class=HTMLResponse)
    def grade_page(request: Request, lab_id: str, grade_id: str):
        record = get_record(lab_id)
        try:
            report = service.store.get_grade(lab_id, grade_id)
        except LabNotFound:
            raise HTTPException(404, "Результат проверки не найден")
        return page(request, "grade.html", record=record, report=report)

    # --- JSON API (для интеграций: CI репозитория, плагины LMS) --------------------------

    @app.get("/api/labs")
    def api_list():
        return [
            {
                "id": r.id, "title": r.spec.title, "level": r.level, "status": r.status, "version": r.version,
                "validation_passed": bool(r.validation and r.validation.passed), "updated_at": r.updated_at,
            }
            for r in service.store.list()
        ]

    @app.get("/api/labs/{lab_id}")
    def api_get(lab_id: str):
        return json.loads(get_record(lab_id).model_dump_json())

    @app.post("/api/labs", status_code=201)
    def api_create(body: CreateLabRequest):
        try:
            record = service.create(body.topic, body.level, mode=body.mode, topic_id=body.topic_id, materials=body.materials)
        except (TopicNotFound, ValueError, LLMError) as exc:
            raise HTTPException(400, str(exc))
        return {"id": record.id, "validation_passed": bool(record.validation and record.validation.passed)}

    @app.post("/api/labs/{lab_id}/grade")
    def api_grade(lab_id: str, body: GradeRequest):
        get_record(lab_id)
        grade_id, report = service.grade(lab_id, body.code, student=body.student)
        return {"grade_id": grade_id, **json.loads(report.model_dump_json())}

    @app.get("/api/health")
    def health():
        return {"status": "ok", "llm": service.llm_available}

    return app
