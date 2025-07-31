from dataclasses import dataclass
from typing import Any
@dataclass
class SheetScoringInput:
    sheet_content: dict[str, Any]

