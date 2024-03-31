from dataclasses import dataclass, field
from typing import List

@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

if __name__ == "__main__":
    # Create instances of LabelledRef
    labelled_ref1 = LabelledRef(ref='example_ref1', slugs=['slug1', 'slug2'])
    labelled_ref2 = LabelledRef(ref='example_ref2', slugs=['slug3', 'slug4', 'slug5'])

    # Print out the instances
    print(labelled_ref1)
    print(labelled_ref2)