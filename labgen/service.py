"""Сервисный слой: общий для веб-интерфейса и командной строки."""

from __future__ import annotations

from .accounts import AccountStore
from .agent.graph import ProgressCallback, generate_lab
from .config import Settings, load_settings
from .exporters.gitverse import export_student_repo
from .exporters.moodle import export_moodle
from .exporters.teacher import export_teacher_kit
from .grader import grade_solution
from .jobs import JobManager
from .llm import LabLLM, LLMError, config_from_settings
from .models import GradeReport, LabRecord, LabSpec, Requirement
from .pipeline.reviewer import review_spec
from .pipeline.validator import validate_spec
from .sandbox import run_tests
from .storage import LabStore

EXPORT_KINDS = {
    "student": "Репозиторий студента (Gitverse)",
    "teacher": "Комплект преподавателя",
    "moodle": "Moodle: CodeRunner XML",
}


class ApprovalError(RuntimeError):
    pass


class LabService:
    def __init__(
        self,
        settings: Settings | None = None,
        store: LabStore | None = None,
        accounts: AccountStore | None = None,
        default_llm: LabLLM | None = None,
        use_settings_llm: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or LabStore(self.settings.data_dir)
        self.accounts = accounts or AccountStore(self.settings.data_dir / "app.db", self.settings.resolve_secret())
        self.jobs = JobManager()
        if default_llm is None and use_settings_llm:
            config = config_from_settings(self.settings)
            default_llm = LabLLM.from_config(config) if config else None
        self.default_llm = default_llm  # LLM из окружения: для CLI

    # --- модели пользователя ---------------------------------------------------------

    def user_llm(self, user_id: int | None, provider: str | None = None) -> LabLLM | None:
        """LLM пользователя (по умолчанию или указанный провайдер); без пользователя — LLM из окружения."""
        if user_id is None:
            return self.default_llm
        config = self.accounts.llm_config(user_id, provider)
        if config is None:
            if provider:
                raise LLMError("Эта интеграция не настроена в аккаунте")
            return None
        return LabLLM.from_config(config)

    # --- создание -----------------------------------------------------------------

    def create(
        self,
        request: str,
        level: int,
        *,
        mode: str = "auto",
        llm: LabLLM | None = None,
        owner_id: int | None = None,
        materials: str | None = None,
        topic_id: str | None = None,
        simulate: int | None = None,
        on_step: ProgressCallback | None = None,
    ) -> LabRecord:
        result = generate_lab(
            request,
            level,
            mode=mode,  # type: ignore[arg-type]
            llm=llm,
            materials=materials,
            topic_id=topic_id,
            max_attempts=self.settings.max_generation_attempts,
            simulate=self.settings.simulate_students if simulate is None else simulate,
            on_step=on_step,
        )
        record = LabRecord(
            id=self.store.new_id(result.spec.title, level),
            request_topic=request,
            level=level,
            mode=result.mode,
            owner_id=owner_id,
            llm_name=result.llm_name,
            spec=result.spec,
            validation=result.report,
            review=result.review,
            generation_log=result.log,
            simulation=result.simulation,
        )
        return self.store.save(record, new_version=False)

    def start_create(self, owner_id: int | None, request: str, level: int, **kwargs):
        """Генерация в фоне; прогресс — по шагам графа LangGraph."""
        title = f"{request} · уровень {level}"
        return self.jobs.start(
            owner_id,
            title,
            lambda on_step: self.create(request, level, owner_id=owner_id, on_step=on_step, **kwargs).id,
        )

    # --- редактирование -------------------------------------------------------------

    def update(self, lab_id: str, changes: dict, *, revalidate: bool = True, note: str = "") -> LabRecord:
        record = self.store.get(lab_id)
        data = record.spec.model_dump()
        for key, value in changes.items():
            if key == "requirements" and isinstance(value, list):
                data[key] = [Requirement.model_validate(item).model_dump() for item in value]
            elif key in data:
                data[key] = value
        record.spec = LabSpec.model_validate(data)
        record.status = "draft"
        record.generation_log.append(f"Изменено преподавателем{': ' + note if note else ''} (версия {record.version + 1}).")
        if revalidate:
            record.validation = validate_spec(record.spec)
            record.review = review_spec(record.spec)
        return self.store.save(record)

    def validate(self, lab_id: str) -> LabRecord:
        record = self.store.get(lab_id)
        record.validation = validate_spec(record.spec)
        return self.store.save(record, new_version=False)

    def approve(self, lab_id: str) -> LabRecord:
        record = self.store.get(lab_id)
        if record.validation is None or not record.validation.passed:
            raise ApprovalError("Нельзя утвердить работу, не прошедшую автоматическую проверку. Исправьте ошибки и запустите проверку.")
        record.status = "approved"
        record.generation_log.append("Утверждено преподавателем к публикации.")
        return self.store.save(record, new_version=False)

    def unapprove(self, lab_id: str) -> LabRecord:
        record = self.store.get(lab_id)
        record.status = "draft"
        return self.store.save(record, new_version=False)

    def restore_version(self, lab_id: str, version: int) -> LabRecord:
        old = self.store.get_version(lab_id, version)
        record = self.store.get(lab_id)
        record.spec = old.spec
        record.validation = old.validation
        record.review = old.review
        record.status = "draft"
        record.generation_log.append(f"Восстановлена версия {version}.")
        return self.store.save(record)

    # --- экспорт, запуск и проверка ------------------------------------------------------

    def export(self, lab_id: str, kind: str) -> tuple[str, bytes]:
        record = self.store.get(lab_id)
        if kind == "student":
            return export_student_repo(record.spec)
        if kind == "teacher":
            return export_teacher_kit(record.spec, record)
        if kind == "moodle":
            return export_moodle(record.spec)
        raise ValueError(f"Неизвестный формат экспорта: {kind}")

    def run_public(self, lab_id: str, code: str) -> list[dict]:
        """«Запустить»: только публичные тесты, без сохранения результата."""
        spec = self.store.get(lab_id).spec
        run = run_tests(code, {"test_public.py": spec.tests_public}, per_test_timeout=5)
        cases = []
        for name in run.expected:
            outcome = run.outcome(name)
            first = next((line for line in outcome.message.splitlines() if line.strip()), "")
            cases.append({"name": name, "passed": outcome.passed, "message": first[:400]})
        if run.collection_error and not run.passed_names:
            last = [line for line in run.collection_error.splitlines() if line.strip()]
            return [{"name": "Запуск решения", "passed": False, "message": (last[-1] if last else "ошибка запуска")[:400]}]
        return cases

    def grade(
        self, lab_id: str, code: str, student: str | None = None, llm: LabLLM | None = None
    ) -> tuple[str, GradeReport]:
        record = self.store.get(lab_id)
        report = grade_solution(record.spec, code, student=student, llm=llm)
        grade_id = self.store.save_grade(lab_id, report)
        return grade_id, report
