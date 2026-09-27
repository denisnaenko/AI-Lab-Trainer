"""Moodle export.

Moodle's core web services cannot create course activities, so publishing is split:

* Import (primary, works on any Moodle with the CodeRunner plugin): generate a Moodle
  XML question-bank file with a CodeRunner "python3" question per lab. CodeRunner runs
  the tests on Moodle's Jobe sandbox, so auto-grading happens inside Moodle.
* Web services (optional): with a token, look up courses and push grades from our own
  grader for assignment-based labs via `mod_assign_save_grade`.
"""

from __future__ import annotations

from pathlib import Path

from lab_trainer.export.base import Exporter, ExportResult
from lab_trainer.models import LabAssignment


class MoodleXmlExporter(Exporter):
    target = "moodle-xml"

    def _export(self, lab: LabAssignment, out_dir: Path) -> ExportResult:
        # TODO: render <quiz><question type="coderunner"> with questiontext = brief +
        # instructions (HTML), answerpreload = starter, answer = reference solution,
        # testcases generated from visible/hidden tests, and grading via a
        # template that runs pytest. Needs a check against a real Moodle import.
        raise NotImplementedError


class MoodleWebService:
    """REST client for /webservice/rest/server.php (token auth, moodlewsrestformat=json)."""

    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def call(self, function: str, **params) -> dict:
        # TODO: httpx.post(f"{self.base_url}/webservice/rest/server.php",
        #   data={"wstoken": self.token, "wsfunction": function,
        #         "moodlewsrestformat": "json", **params})
        raise NotImplementedError

    def save_grade(self, assignment_id: int, user_id: int, grade: float, feedback: str) -> None:
        # TODO: self.call("mod_assign_save_grade", ...)
        raise NotImplementedError
