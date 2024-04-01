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
        self.golden_standard_projection = self.filter_out_refs_not_in_evaluated(self.golden_standard_projection)

        self.evaluated_projection = self.get_projection_of_labelled_refs(self.evaluated)

    def get_projection_of_labelled_refs(self, lrs :List[LabelledRef]) -> List[LabelledRef]:
        # Remove irrelevant slugs from the slugs list
        projected = []
        for ref in lrs:
            projected.append(LabelledRef(ref.ref, [slug for slug in ref.slugs if slug in self.considered_labels]))
        return projected

    def filter_out_refs_not_in_evaluated(self, lrs :List[LabelledRef]) -> List[LabelledRef]:
        evaluated_refs = [laballed_ref.ref for laballed_ref in self.evaluated]
        projected = [labelled_ref for labelled_ref in lrs if labelled_ref.ref in evaluated_refs]
        return projected

    def compute_accuracy(self) -> float:
        correct_predictions = 0
        total_predictions = 0

        for golden_ref, evaluated_ref in zip(self.golden_standard_projection, self.evaluated_projection):
            total_predictions += 1
            if set(golden_ref.slugs) == set(evaluated_ref.slugs):
                correct_predictions += 1

        if total_predictions == 0:
            return 0.0

        accuracy = correct_predictions / total_predictions
        return accuracy

    def compute_metrics_for_single_ref(self, golden_standard_ref: LabelledRef, evaluated_ref: LabelledRef):
        golden_standard = golden_standard_ref.slugs
        evaluated = evaluated_ref.slugs

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for item in evaluated:
            if item in golden_standard:
                true_positives += 1
            else:
                false_positives += 1

        for item in golden_standard:
            if item not in evaluated:
                false_negatives += 1

        return true_positives, false_positives, false_negatives

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

    def get_considered_slugs(self) -> List[LabelledRef]:
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

    handler = DataHandler("golden_standard_labels_march_2024.jsonl", 'golden_standard_labels_march_2024.jsonl', 'all_slugs_and_titles_for_prodigy.csv')
    evaluator = Evaluator(handler.get_golden_standard(), handler.get_evaluated(), handler.get_considered_slugs())
    for g, e in zip(evaluator.golden_standard_projection, evaluator.evaluated_projection):
        print(evaluator.compute_metrics_for_single_ref(g, e))