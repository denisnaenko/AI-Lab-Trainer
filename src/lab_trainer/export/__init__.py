from lab_trainer.export.base import Exporter, ExportResult, NotApprovedError
from lab_trainer.export.gitverse import GitverseRepoExporter
from lab_trainer.export.moodle import MoodleXmlExporter

EXPORTERS: dict[str, type[Exporter]] = {
    "gitverse": GitverseRepoExporter,
    "moodle-xml": MoodleXmlExporter,
}

__all__ = [
    "EXPORTERS",
    "ExportResult",
    "Exporter",
    "GitverseRepoExporter",
    "MoodleXmlExporter",
    "NotApprovedError",
]
