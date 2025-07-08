from dataclasses import dataclass
from typing import List
import json
import os
from typing import List, Optional
@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

class DataHandler:
    def __init__(
        self,
        golden_standard_filename: Optional[str] = None,
        predicted_filename: Optional[str] = None,
        considered_slugs_filename: Optional[str] = None,
    ):
        self.golden_standard_filename   = golden_standard_filename
        self.predicted_filename         = predicted_filename
        self.considered_slugs_filename  = considered_slugs_filename
    @staticmethod
    def _ensure_file(path: Optional[str]) -> bool:
        """Return True only when a usable file path was supplied *and* exists."""
        return bool(path) and os.path.isfile(path)


    def _jsonl_to_labelled_refs(self, jsonl_filename: str) -> List["LabelledRef"]:
        lrs: List["LabelledRef"] = []
        with open(jsonl_filename, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                ref = data["ref"]
                slugs = data["slugs"]
                lrs.append(LabelledRef(ref, slugs))
        return lrs


    def _read_first_column_or_array(self, file_path: str) -> List[str]:
        ext = file_path.rsplit(".", 1)[-1].lower()

        if ext == "csv":
            with open(file_path, newline="", encoding="utf-8") as f:
                return [row[0] for row in csv.reader(f)]

        elif ext == "json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if not (isinstance(data, dict) and isinstance(next(iter(data.values())), list)):
                raise ValueError("JSON must be an object with an array as its first value")
            return next(iter(data.values()))

        elif ext == "jsonl":
            with open(file_path, encoding="utf-8") as f:
                first = json.loads(f.readline())
            if not (isinstance(first, dict) and isinstance(next(iter(first.values())), list)):
                raise ValueError("JSONL’s first object must contain an array as its first value")
            return next(iter(first.values()))

        raise ValueError("Unsupported file type (supported: .csv, .json, .jsonl)")


    # ---------- public API -------------------------------------------------- #

    def get_golden_standard(self) -> List["LabelledRef"]:
        if not self._ensure_file(self.golden_standard_filename):
            return []  # or raise FileNotFoundError(...)
        return self._jsonl_to_labelled_refs(self.golden_standard_filename)


    def get_predicted(self) -> List["LabelledRef"]:
        if not self._ensure_file(self.predicted_filename):
            return []
        return self._jsonl_to_labelled_refs(self.predicted_filename)


    def get_considered_slugs(self) -> List[str]:
        if not self._ensure_file(self.considered_slugs_filename):
            return []
        return self._read_first_column_or_array(self.considered_slugs_filename)