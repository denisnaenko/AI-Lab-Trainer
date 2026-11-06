"""Сервисный слой: общий для веб-интерфейса и командной строки."""

from __future__ import annotations

from .config import Settings, load_settings
from .exporters.gitverse import export_student_repo
from .exporters.moodle import export_moodle
from .exporters.teacher import export_teacher_kit
from .grader import grade_solution
from .llm import provider_from_settings
from .llm.base import LLMProvider
from .models import GradeReport, LabRecord, LabSpec, Requirement
from .pipeline.orchestrator import generate_lab
from .pipeline.reviewer import review_spec
from .pipeline.validator import validate_spec
from .storage import LabStore

EXPORT_KINDS = {
    "student": "Репозиторий студента (Gitverse)",
    "teacher": "Комплект преподавателя",
    "moodle": "Moodle: CodeRunner XML и описание задания",
}


class ApprovalError(RuntimeError):
    pass


class LabService:
    def __init__(
        self,
        settings: Settings | None = None,
        store: LabStore | None = None,
        provider: LLMProvider | None = None,
        use_settings_provider: bool = True,
    ) -> None:
        self.settings = settings or load_settings()
        self.store = store or LabStore(self.settings.data_dir)
        if provider is None and use_settings_provider:
            provider = provider_from_settings(self.settings)
        self.provider = provider

    @property
    def llm_available(self) -> bool:
        return self.provider is not None

    # --- создание -----------------------------------------------------------------

    def create(
        self,
        request: str,
        level: int,
        *,
        mode: str = "auto",
        materials: str | None = None,
        topic_id: str | None = None,
        simulate: int | None = None,
    ) -> LabRecord:
        result = generate_lab(
            request,
            level,
            mode=mode,  # type: ignore[arg-type]
            provider=self.provider,
            materials=materials,
            topic_id=topic_id,
            max_attempts=self.settings.max_generation_attempts,
            simulate=self.settings.simulate_students if simulate is None else simulate,
        )
        record = LabRecord(
            id=self.store.new_id(result.spec.title, level),
            request_topic=request,
            level=level,
            mode=result.mode,
            spec=result.spec,
            validation=result.report,
            review=result.review,
            generation_log=result.log,
            simulation=result.simulation,
        )
        return self.store.save(record, new_version=False)

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

    # --- экспорт и проверка ------------------------------------------------------------

    def export(self, lab_id: str, kind: str) -> tuple[str, bytes]:
        record = self.store.get(lab_id)
        if kind == "student":
            return export_student_repo(record.spec)
        if kind == "teacher":
            return export_teacher_kit(record.spec, record)
        if kind == "moodle":
            return export_moodle(record.spec)
        raise ValueError(f"Неизвестный формат экспорта: {kind}")

    def grade(self, lab_id: str, code: str, student: str | None = None, llm_feedback: bool = False) -> tuple[str, GradeReport]:
        record = self.store.get(lab_id)
        report = grade_solution(record.spec, code, student=student, provider=self.provider if llm_feedback else None)
        grade_id = self.store.save_grade(lab_id, report)
        return grade_id, report
