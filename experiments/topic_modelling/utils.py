from dataclasses import dataclass
from typing import List
import json

@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

class DataHandler:
    def __init__(self, golden_standard_filename, predicted_filename, considered_slugs_filename):
        self.golden_standard_filename = golden_standard_filename
        self.predicted_filename = predicted_filename
        self.considered_slugs_filename = considered_slugs_filename

    def _jsonl_to_labelled_refs(self, jsonl_filename) -> List[LabelledRef]:
        lrs = []
        with open(jsonl_filename, 'r') as file:
            for line in file:
                data = json.loads(line)
                ref = data['ref']
                slugs = data['slugs']
                lrs.append(LabelledRef(ref, slugs))
        return lrs

    def _read_first_column_or_array(self, file_path):
        file_extension = file_path.split('.')[-1]
        if file_extension == 'csv':
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                first_column = [row[0] for row in reader]
                return first_column
        elif file_extension == 'json':
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict):
                    first_array = next(iter(data.values()))
                    if isinstance(first_array, list):
                        return first_array
                    else:
                        raise ValueError("Invalid JSON format. Expecting an array as a value for the first key.")
                else:
                    raise ValueError("Invalid JSON format. Expecting a dictionary.")
        elif file_extension == 'jsonl':
            with open(file_path, 'r') as file:
                first_line = file.readline()
                data = json.loads(first_line)
                if isinstance(data, dict):
                    first_array = next(iter(data.values()))
                    if isinstance(first_array, list):
                        return first_array
                    else:
                        raise ValueError("Invalid JSONL format. Expecting an array as a value for the first key.")
                else:
                    raise ValueError("Invalid JSONL format. Expecting a dictionary.")
        else:
            raise ValueError("Unsupported file format. Only CSV, JSON, and JSONL are supported.")

    def get_golden_standard(self) -> List[LabelledRef]:
        return self._jsonl_to_labelled_refs(self.golden_standard_filename)

    def get_predicted(self) -> List[LabelledRef]:
        return self._jsonl_to_labelled_refs(self.predicted_filename)

    def get_considered_slugs(self) -> List[LabelledRef]:
        return self._read_first_column_or_array(self.considered_slugs_filename)