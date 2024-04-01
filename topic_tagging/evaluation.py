import csv
import json
from dataclasses import dataclass, field
from typing import List

@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

class Evaluator:
    def __init__(self, golden_standard: List[LabelledRef], evaluated: List[LabelledRef], considered_labels: List[str]):
        self.golden_standard = golden_standard
        self.evaluated = evaluated
        self.considered_labels = considered_labels

        self.golden_standard_projection = self.get_projection_of_labelled_refs(self.golden_standard)
        self.evaluated_projection = self.get_projection_of_labelled_refs(self.evaluated)

    def get_projection_of_labelled_refs(self, lrs :List[LabelledRef]) -> List[LabelledRef]:
        # Remove irrelevant slugs from the slugs list
        projected = []
        for ref in lrs:
            projected.append(LabelledRef(ref.ref, [slug for slug in ref.slugs if slug in self.considered_labels]))
        return projected

class DataHandler:
    def __init__(self, golden_standard_filename, evaluated_filename, considered_slugs_filename):
        self.golden_standard_filename = golden_standard_filename
        self.evaluated_filename = evaluated_filename
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

    def get_evaluated(self) -> List[LabelledRef]:
        return self._jsonl_to_labelled_refs(self.evaluated_filename)

    def get_evaluated(self) -> List[LabelledRef]:
        return self._read_first_column_or_array(self.considered_slugs_filename)





if __name__ == "__main__":
    # Create some sample data
    golden_standard = [
        LabelledRef("ref1", ["a", "b", "c"]),
        LabelledRef("ref2", ["b", "c", "d"]),
        LabelledRef("ref3", ["c", "d", "e"])
    ]

    evaluated = [
        LabelledRef("ref1", ["a", "c", "e"]),
        LabelledRef("ref2", ["b", "d"]),
        LabelledRef("ref3", ["c", "d", "e"])
    ]

    considered_labels = ["a", "b", "c"]

    # Initialize Evaluator
    evaluator = Evaluator(golden_standard, evaluated, considered_labels)

    # Test accessing the projections
    print("Golden Standard Projection:")
    for ref in evaluator.golden_standard_projection:
        print(ref)

    print("\nEvaluated Projection:")
    for ref in evaluator.evaluated_projection:
        print(ref)

    handler = DataHandler("golden_standard_labels_march_2024.jsonl", '', '')
    print(handler.get_golden_standard())