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

        self.golden_standard_projection = self.get_projection_of_evaluated(self.golden_standard)
        self.evaluated_projection = self.get_projection_of_evaluated(self.evaluated)

    def get_projection_of_evaluated(self, lrs :List[LabelledRef]) -> List[LabelledRef]:
        # Remove irrelevant slugs from the slugs list
        projected = []
        for ref in lrs:
            projected.append(LabelledRef(ref.ref, [slug for slug in ref.slugs if slug in self.considered_labels]))
        return projected



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