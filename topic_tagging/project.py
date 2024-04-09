from tagging import *
from evaluation import *

def tagging_experiment(dest_jsonl_filename):
    # tag_sample_refs(dest_jsonl=dest_jsonl_filename)
    evaluate_results(dest_jsonl_filename)


if __name__ == "__main__":
    tagging_experiment('exp_results.jsonl')