"""Веб-интерфейс преподавателя и JSON API (FastAPI)."""

from __future__ import annotations

import hashlib
import json
import secrets
from pathlib import Path
from urllib.parse import quote

import markdown
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError
from starlette.middleware.sessions import SessionMiddleware

from ..accounts import AccountError, User
from ..agent.graph import NODE_LABELS
from ..catalog import TOPICS, list_topics
from ..levels import LEVELS
from ..llm import PROVIDERS, LLMError, check_connection
from ..models import LabRecord, Mutant, Requirement, test_names
from ..service import EXPORT_KINDS, ApprovalError, LabService
from ..storage import LabNotFound

HERE = Path(__file__).parent
DIFFICULTY = {1: ("Базовый", "easy"), 2: ("Средний", "medium"), 3: ("Продвинутый", "hard")}


def render_md(text: str) -> str:
    return markdown.markdown(text or "", extensions=["tables", "fenced_code"])


class LoginRequired(Exception):
    pass


class CodeRequest(BaseModel):
    code: str
    student: str | None = None


class CreateLabRequest(BaseModel):
    topic: str
    level: int = 1
    source: str = "auto"
    topic_id: str | None = None
    materials: str | None = None


def create_app(service: LabService | None = None) -> FastAPI:
    service = service or LabService()
    app = FastAPI(title="AI Lab Trainer", description="ИИ-агент для создания лабораторных работ по агентной разработке")
    app.state.service = service
    session_secret = hashlib.sha256(("labgen-session:" + service.settings.resolve_secret()).encode()).hexdigest()
    app.add_middleware(
        SessionMiddleware,
        secret_key=session_secret,
        session_cookie="labgen_session",
        same_site="lax",
        max_age=14 * 24 * 3600,
    )
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.filters["md"] = render_md
    templates.env.globals.update(
        levels=LEVELS, difficulty=DIFFICULTY, export_kinds=EXPORT_KINDS, providers=PROVIDERS, topics_by_id=TOPICS,
        node_labels=NODE_LABELS, test_names=test_names,
    )

    # --- сессия, пользователь, CSRF ------------------------------------------------------

    def current_user(request: Request) -> User | None:
        uid = request.session.get("uid")
        return service.accounts.get_user(uid) if uid else None

    def require_user(request: Request) -> User:
        user = current_user(request)
        if user is None:
            raise LoginRequired()
        return user

    def csrf_token(request: Request) -> str:
        token = request.session.get("csrf")
        if not token:
            token = secrets.token_urlsafe(32)
            request.session["csrf"] = token
        return token

    async def csrf_protect(request: Request) -> None:
        token = request.headers.get("x-csrf-token")
        if not token:
            content_type = request.headers.get("content-type", "")
            if content_type.startswith(("application/x-www-form-urlencoded", "multipart/form-data")):
                token = (await request.form()).get("csrf")
        expected = request.session.get("csrf", "")
        if not token or not expected or not secrets.compare_digest(str(token), expected):
            raise HTTPException(403, "Сессия устарела — обновите страницу и повторите действие")

    def flash(request: Request, text: str, kind: str = "success") -> None:
        request.session.setdefault("flash", []).append({"text": text, "kind": kind})

    def page(request: Request, name: str, status_code: int = 200, **context) -> HTMLResponse:
        user = context.get("user", current_user(request))
        default = None
        if user is not None:
            creds = service.accounts.credentials(user.id)
            default = next((c for c in creds.values() if c.is_default), None)
        context.update(
            user=user,
            csrf=csrf_token(request),
            flashes=request.session.pop("flash", []),
            default_llm=default,
            path=request.url.path,
        )
        return templates.TemplateResponse(request, name, context, status_code=status_code)

    def get_lab(lab_id: str, user: User) -> LabRecord:
        try:
            record = service.store.get(lab_id)
        except LabNotFound:
            raise HTTPException(404, "Лабораторная работа не найдена")
        if record.owner_id not in (None, user.id):
            raise HTTPException(404, "Лабораторная работа не найдена")
        return record

    def safe_next(value: str | None) -> str:
        return value if value and value.startswith("/") and not value.startswith("//") else "/"

    @app.exception_handler(LoginRequired)
    async def login_required(request: Request, exc: LoginRequired):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Требуется вход в аккаунт"}, status_code=401)
        return RedirectResponse(f"/login?next={quote(request.url.path)}", status_code=303)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
        return page(request, "error.html", status_code=exc.status_code, message=exc.detail, code=exc.status_code)

    # --- вход и регистрация ----------------------------------------------------------------

    @app.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/"):
        if current_user(request):
            return RedirectResponse(safe_next(next), status_code=303)
        return page(request, "login.html", next=safe_next(next), form={})

    @app.post("/login", dependencies=[Depends(csrf_protect)])
    def login(request: Request, email: str = Form(...), password: str = Form(...), next: str = Form("/")):
        user = service.accounts.authenticate(email, password)
        if user is None:
            return page(request, "login.html", status_code=400, next=safe_next(next), form={"email": email},
                        error="Неверная почта или пароль")
        request.session.clear()
        request.session["uid"] = user.id
        flash(request, f"С возвращением, {user.name}!")
        return RedirectResponse(safe_next(next), status_code=303)

    @app.get("/register", response_class=HTMLResponse)
    def register_page(request: Request):
        if current_user(request):
            return RedirectResponse("/", status_code=303)
        return page(request, "register.html", form={}, allowed=service.settings.allow_registration)

    @app.post("/register", dependencies=[Depends(csrf_protect)])
    def register(
        request: Request,
        name: str = Form(""),
        email: str = Form(...),
        password: str = Form(...),
        password2: str = Form(...),
    ):
        form = {"name": name, "email": email}
        if not service.settings.allow_registration:
            raise HTTPException(403, "Регистрация отключена администратором")
        error = None
        if password != password2:
            error = "Пароли не совпадают"
        else:
            try:
                user = service.accounts.create_user(email, name, password)
            except AccountError as exc:
                error = str(exc)
        if error:
            return page(request, "register.html", status_code=400, form=form, error=error, allowed=True)
        request.session.clear()
        request.session["uid"] = user.id
        flash(request, "Аккаунт создан. Подключите LLM, чтобы генерировать работы по любой теме.")
        return RedirectResponse("/account#llm", status_code=303)

    @app.post("/logout", dependencies=[Depends(csrf_protect)])
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/", status_code=303)

    # --- аккаунт и интеграции LLM ----------------------------------------------------------

    @app.get("/account", response_class=HTMLResponse)
    def account(request: Request, user: User = Depends(require_user)):
        creds = service.accounts.credentials(user.id)
        return page(request, "account.html", user=user, creds=creds)

    @app.post("/account/profile", dependencies=[Depends(csrf_protect)])
    def account_profile(request: Request, name: str = Form(...), user: User = Depends(require_user)):
        service.accounts.update_profile(user.id, name)
        flash(request, "Профиль сохранён")
        return RedirectResponse("/account#profile", status_code=303)

    @app.post("/account/password", dependencies=[Depends(csrf_protect)])
    def account_password(
        request: Request,
        old_password: str = Form(...),
        new_password: str = Form(...),
        user: User = Depends(require_user),
    ):
        try:
            service.accounts.change_password(user.id, old_password, new_password)
            flash(request, "Пароль изменён")
        except AccountError as exc:
            flash(request, str(exc), "error")
        return RedirectResponse("/account#security", status_code=303)

    @app.post("/account/llm/{provider}", dependencies=[Depends(csrf_protect)])
    def save_llm(
        request: Request,
        provider: str,
        api_key: str = Form(""),
        model: str = Form(""),
        base_url: str = Form(""),
        json_mode: str = Form(""),
        user: User = Depends(require_user),
    ):
        if provider not in PROVIDERS:
            raise HTTPException(404, "Неизвестный провайдер")
        try:
            service.accounts.save_credential(
                user.id, provider, api_key=api_key or None, model=model, base_url=base_url, json_mode=json_mode == "on"
            )
            flash(request, f"{PROVIDERS[provider].title}: настройки сохранены. Проверьте подключение.")
        except AccountError as exc:
            flash(request, f"{PROVIDERS[provider].title}: {exc}", "error")
        return RedirectResponse(f"/account#llm-{provider}", status_code=303)

    @app.post("/account/llm/{provider}/delete", dependencies=[Depends(csrf_protect)])
    def delete_llm(request: Request, provider: str, user: User = Depends(require_user)):
        service.accounts.delete_credential(user.id, provider)
        flash(request, f"{PROVIDERS.get(provider, PROVIDERS['local']).title}: интеграция удалена")
        return RedirectResponse("/account#llm", status_code=303)

    @app.post("/account/llm/{provider}/default", dependencies=[Depends(csrf_protect)])
    def default_llm(request: Request, provider: str, user: User = Depends(require_user)):
        try:
            service.accounts.set_default(user.id, provider)
            flash(request, f"{PROVIDERS[provider].title} — модель по умолчанию")
        except (AccountError, KeyError) as exc:
            flash(request, str(exc), "error")
        return RedirectResponse("/account#llm", status_code=303)

    @app.post("/api/account/llm/{provider}/test", dependencies=[Depends(csrf_protect)])
    def test_llm(provider: str, user: User = Depends(require_user)):
        config = service.accounts.llm_config(user.id, provider)
        if config is None:
            raise HTTPException(404, "Интеграция не настроена")
        ok, message = check_connection(config)
        service.accounts.record_check(user.id, provider, ok, message)
        return {"ok": ok, "message": message}

    # --- задачи (лабораторные) -----------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        user = current_user(request)
        if user is None:
            return page(request, "landing.html", topics=list_topics())
        labs = service.store.list(user.id)
        stats = {
            "total": len(labs),
            "approved": sum(r.status == "approved" for r in labs),
            "by_level": {
                n: {"total": sum(r.level == n for r in labs), "approved": sum(r.level == n and r.status == "approved" for r in labs)}
                for n in LEVELS
            },
        }
        creds = service.accounts.credentials(user.id)
        return page(request, "labs.html", user=user, labs=labs, stats=stats, creds=creds, topics=list_topics())

    @app.get("/catalog", response_class=HTMLResponse)
    def catalog(request: Request):
        return page(request, "catalog.html", topics=list_topics())

    @app.get("/labs/new", response_class=HTMLResponse)
    def new_lab(request: Request, topic_id: str = "", level: int = 1, user: User = Depends(require_user)):
        creds = service.accounts.credentials(user.id)
        topic = TOPICS.get(topic_id)
        form = {"topic": topic.title if topic else "", "topic_id": topic_id, "level": level,
                "source": "catalog" if topic else "auto"}
        return page(request, "new.html", user=user, creds=creds, topics=list_topics(), form=form)

    def _resolve_source(user: User, source: str):
        """Источник генерации → (режим, модель)."""
        if source == "catalog":
            return "catalog", None
        if source == "auto":
            llm = service.user_llm(user.id)
            return ("llm", llm) if llm else ("catalog", None)
        if source in PROVIDERS:
            return "llm", service.user_llm(user.id, source)
        raise ValueError("Неизвестный источник генерации")

    @app.post("/labs", dependencies=[Depends(csrf_protect)])
    def create_lab(
        request: Request,
        topic: str = Form(...),
        level: int = Form(1),
        source: str = Form("auto"),
        topic_id: str = Form(""),
        materials: str = Form(""),
        simulate: int = Form(0),
        user: User = Depends(require_user),
    ):
        try:
            mode, llm = _resolve_source(user, source)
        except (LLMError, ValueError) as exc:
            flash(request, str(exc), "error")
            return RedirectResponse("/labs/new", status_code=303)
        job = service.start_create(
            user.id, topic.strip(), level, mode=mode, llm=llm, topic_id=topic_id or None,
            materials=materials or None, simulate=max(0, min(simulate, 10)),
        )
        return RedirectResponse(f"/jobs/{job.id}", status_code=303)

    @app.get("/jobs/{job_id}", response_class=HTMLResponse)
    def job_page(request: Request, job_id: str, user: User = Depends(require_user)):
        job = service.jobs.get(job_id)
        if job is None or job.owner_id != user.id:
            raise HTTPException(404, "Задача генерации не найдена")
        return page(request, "job.html", user=user, job=job)

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str, user: User = Depends(require_user)):
        job = service.jobs.get(job_id)
        if job is None or job.owner_id != user.id:
            raise HTTPException(404, "Задача генерации не найдена")
        return job.to_dict()

    @app.get("/labs/{lab_id}", response_class=HTMLResponse)
    def lab_page(request: Request, lab_id: str, user: User = Depends(require_user)):
        record = get_lab(lab_id, user)
        return page(
            request,
            "workspace.html",
            user=user,
            record=record,
            spec=record.spec,
            tests_map=record.spec.tests_by_requirement(),
            grades=service.store.list_grades(lab_id)[:30],
            versions=service.store.versions(lab_id),
        )

    @app.get("/labs/{lab_id}/edit", response_class=HTMLResponse)
    def edit_page(request: Request, lab_id: str, user: User = Depends(require_user)):
        record = get_lab(lab_id, user)
        mutants_json = json.dumps([m.model_dump() for m in record.spec.mutants], ensure_ascii=False, indent=2)
        return page(request, "edit.html", user=user, record=record, spec=record.spec, mutants_json=mutants_json)

    @app.post("/labs/{lab_id}/edit", dependencies=[Depends(csrf_protect)])
    async def save_edit(request: Request, lab_id: str, user: User = Depends(require_user)):
        record = get_lab(lab_id, user)
        form = await request.form()
        changes: dict = {}
        for field in ("title", "summary", "statement_md", "starter_code", "reference_solution", "tests_public", "tests_hidden"):
            if field in form:
                changes[field] = str(form[field]).replace("\r\n", "\n")
        minutes = str(form.get("estimated_minutes", "")).strip()
        changes["estimated_minutes"] = int(minutes) if minutes.isdigit() else None
        requirements, error = [], ""
        for i in range(int(form.get("req_count", 0)) + 1):
            req_id = str(form.get(f"req_id_{i}", "")).strip()
            text = str(form.get(f"req_text_{i}", "")).strip()
            if not req_id and not text:
                continue
            try:
                requirements.append(
                    Requirement(
                        id=req_id, text=text, points=int(str(form.get(f"req_points_{i}", "0")) or 0),
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
            return page(request, "edit.html", status_code=400, user=user, record=record, spec=record.spec,
                        mutants_json=str(form.get("mutants_json", "[]")), error=error)
        record = service.update(lab_id, changes, note=str(form.get("note", "")).strip())
        passed = record.validation and record.validation.passed
        flash(request, "Сохранено, проверка пройдена" if passed else "Сохранено, но проверка нашла ошибки",
              "success" if passed else "error")
        return RedirectResponse(f"/labs/{lab_id}#check", status_code=303)

    @app.post("/labs/{lab_id}/validate", dependencies=[Depends(csrf_protect)])
    def revalidate(request: Request, lab_id: str, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        record = service.validate(lab_id)
        passed = record.validation and record.validation.passed
        flash(request, "Проверка пройдена" if passed else "Проверка нашла ошибки", "success" if passed else "error")
        return RedirectResponse(f"/labs/{lab_id}#check", status_code=303)

    @app.post("/labs/{lab_id}/approve", dependencies=[Depends(csrf_protect)])
    def approve(request: Request, lab_id: str, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        try:
            service.approve(lab_id)
            flash(request, "Работа утверждена к публикации")
        except ApprovalError as exc:
            flash(request, str(exc), "error")
        return RedirectResponse(f"/labs/{lab_id}", status_code=303)

    @app.post("/labs/{lab_id}/unapprove", dependencies=[Depends(csrf_protect)])
    def unapprove(request: Request, lab_id: str, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        service.unapprove(lab_id)
        flash(request, "Работа возвращена в черновики")
        return RedirectResponse(f"/labs/{lab_id}", status_code=303)

    @app.post("/labs/{lab_id}/restore/{version}", dependencies=[Depends(csrf_protect)])
    def restore(request: Request, lab_id: str, version: int, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        try:
            service.restore_version(lab_id, version)
        except LabNotFound:
            raise HTTPException(404, "Версия не найдена")
        flash(request, f"Восстановлена версия {version}")
        return RedirectResponse(f"/labs/{lab_id}#history", status_code=303)

    @app.post("/labs/{lab_id}/delete", dependencies=[Depends(csrf_protect)])
    def delete(request: Request, lab_id: str, user: User = Depends(require_user)):
        record = get_lab(lab_id, user)
        if record.owner_id != user.id:
            flash(request, "Общую работу каталога может удалить только администратор", "error")
            return RedirectResponse(f"/labs/{lab_id}#history", status_code=303)
        service.store.delete(lab_id)
        flash(request, "Работа удалена")
        return RedirectResponse("/", status_code=303)

    @app.get("/labs/{lab_id}/export/{kind}")
    def export(lab_id: str, kind: str, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        if kind not in EXPORT_KINDS:
            raise HTTPException(404, "Неизвестный формат экспорта")
        filename, data = service.export(lab_id, kind)
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.post("/labs/{lab_id}/grade", dependencies=[Depends(csrf_protect)])
    async def grade_upload(
        request: Request,
        lab_id: str,
        student: str = Form(""),
        file: UploadFile | None = File(None),
        llm_feedback: str = Form(""),
        user: User = Depends(require_user),
    ):
        get_lab(lab_id, user)
        if file is None or not file.filename:
            flash(request, "Выберите файл решения solution.py", "error")
            return RedirectResponse(f"/labs/{lab_id}#submissions", status_code=303)
        code = (await file.read()).decode("utf-8", errors="replace")
        llm = service.user_llm(user.id) if llm_feedback == "on" else None
        grade_id, _ = service.grade(lab_id, code, student=student or Path(file.filename).stem, llm=llm)
        return RedirectResponse(f"/labs/{lab_id}/grades/{grade_id}", status_code=303)

    @app.get("/labs/{lab_id}/grades/{grade_id}", response_class=HTMLResponse)
    def grade_page(request: Request, lab_id: str, grade_id: str, user: User = Depends(require_user)):
        record = get_lab(lab_id, user)
        try:
            report = service.store.get_grade(lab_id, grade_id)
        except LabNotFound:
            raise HTTPException(404, "Результат проверки не найден")
        return page(request, "grade.html", user=user, record=record, report=report)

    # --- JSON API -----------------------------------------------------------------------------

    @app.post("/api/labs/{lab_id}/run", dependencies=[Depends(csrf_protect)])
    def api_run(lab_id: str, body: CodeRequest, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        cases = service.run_public(lab_id, body.code)
        return {"cases": cases, "passed": sum(c["passed"] for c in cases), "total": len(cases)}

    @app.post("/api/labs/{lab_id}/submit", dependencies=[Depends(csrf_protect)])
    def api_submit(lab_id: str, body: CodeRequest, user: User = Depends(require_user)):
        get_lab(lab_id, user)
        grade_id, report = service.grade(lab_id, body.code, student=body.student or f"{user.name} (редактор)")
        return {"grade_id": grade_id, "url": f"/labs/{lab_id}/grades/{grade_id}", **json.loads(report.model_dump_json())}

    @app.get("/api/labs")
    def api_list(user: User = Depends(require_user)):
        return [
            {
                "id": r.id, "title": r.spec.title, "level": r.level, "status": r.status, "version": r.version,
                "validation_passed": bool(r.validation and r.validation.passed), "updated_at": r.updated_at,
                "owner_id": r.owner_id,
            }
            for r in service.store.list(user.id)
        ]

    @app.get("/api/labs/{lab_id}")
    def api_get(lab_id: str, user: User = Depends(require_user)):
        return json.loads(get_lab(lab_id, user).model_dump_json())

    @app.post("/api/labs", status_code=202, dependencies=[Depends(csrf_protect)])
    def api_create(body: CreateLabRequest, user: User = Depends(require_user)):
        try:
            mode, llm = _resolve_source(user, body.source)
        except (LLMError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        job = service.start_create(user.id, body.topic, body.level, mode=mode, llm=llm,
                                   topic_id=body.topic_id, materials=body.materials)
        return {"job_id": job.id, "status_url": f"/api/jobs/{job.id}"}

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app
