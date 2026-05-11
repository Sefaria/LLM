import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SegmentRecord:
    tref: str
    text: str
    segment_index: int


def bootstrap_sefaria(sefaria_project_path: Path) -> None:
    if str(sefaria_project_path) not in sys.path:
        sys.path.insert(0, str(sefaria_project_path))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "sefaria.settings")

    import django

    django.setup()


def load_segment_records(
    sefaria_project_path: Path,
    tref: str,
    lang: str = "he",
    version_title: Optional[str] = None,
) -> list[SegmentRecord]:
    bootstrap_sefaria(sefaria_project_path)

    from sefaria.model import Ref

    oref = Ref(tref)
    segment_refs = oref.all_segment_refs() if not oref.is_segment_level() else [oref]
    rows = []
    for i, segment_ref in enumerate(segment_refs, start=1):
        if version_title:
            text = segment_ref.text(lang, vtitle=version_title).text
        else:
            text = segment_ref.text(lang).text
        if not isinstance(text, str):
            continue
        cleaned = text.strip()
        if not cleaned:
            continue
        rows.append(SegmentRecord(tref=segment_ref.normal(), text=cleaned, segment_index=i))
    return rows

