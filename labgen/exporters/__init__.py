"""Экспорт лабораторной работы: репозиторий студента (Gitverse), комплект преподавателя, Moodle."""

from __future__ import annotations

import io
import re
import zipfile

from ..models import LabSpec

TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e", "ж": "zh", "з": "z", "и": "i",
        "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
        "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "", "ы": "y", "ь": "",
        "э": "e", "ю": "yu", "я": "ya",
    }
)


def slugify(text: str, limit: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower().translate(TRANSLIT)).strip("-")
    return slug[:limit].rstrip("-") or "lab"


def lab_slug(spec: LabSpec) -> str:
    return f"{slugify(spec.title)}-l{spec.level}"


def make_zip(files: dict[str, str | bytes], root: str = "") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            data = content.encode("utf-8") if isinstance(content, str) else content
            archive.writestr(f"{root}/{path}" if root else path, data)
    return buffer.getvalue()
