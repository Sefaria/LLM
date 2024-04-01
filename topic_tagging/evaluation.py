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
        self.evaluated_projection = self.get_projection_of_evaluated()

    def get_projection_of_evaluated(self) -> List[LabelledRef]:
        # Remove irrelevant slugs from the slugs list
        projected = []
        for ref in self.evaluated:
            projected.append(LabelledRef(ref.ref, [slug for slug in ref.slugs if slug in self.considered_labels]))
        return projected



if __name__ == "__main__":
    # Create instances of LabelledRef
    labelled_ref1 = LabelledRef(ref='example_ref1', slugs=['slug1', 'slug2'])
    labelled_ref2 = LabelledRef(ref='example_ref2', slugs=['slug3', 'slug4', 'slug5'])

    # Print out the instances
    print(labelled_ref1)
    print(labelled_ref2)