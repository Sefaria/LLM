from langsmith import evaluate, Client
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run
from experiments.topic_modelling.llm_filtering import SequentialRefTopicFilter
from experiments.topic_modelling.utils import DataHandler



llm_filter = SequentialRefTopicFilter(llm = ChatOpenAI(model="gpt-4o-mini", temperature=0), max_topics=10)
dh = DataHandler( predicted_filename= "../../evaluation_data/predictions.jsonl")
predicted = dh.get_predicted()  #


def get_example(x: dict) -> dict:
    ref = x.get("ref")
    if ref is None:
        return
    # answer = llm.invoke(x['q']).content
    predicted_labelled_ref = [lr for lr in predicted if lr.ref == ref ][0]
    filtered_slugs = llm_filter.filter_ref(predicted_labelled_ref)
    return {'ref': ref, 'slugs': filtered_slugs}
def check_answer(root_run: Run, example: Example) -> dict:
    try:
        ref = root_run.outputs.get("ref")
        predicted_labelled_ref = [lr for lr in dh.get_predicted() if lr.ref == ref][0]
        projected_gold   = set(example.outputs.get("slugs")) & set(predicted_labelled_ref.slugs)
        pred   = set(root_run.outputs.get("slugs"))

        # basic overlap stats
        overlap = projected_gold & pred
        tp = len(overlap)
        fp = len(pred - projected_gold)
        fn = len(projected_gold - pred)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        # LangSmith looks for a top-level numeric `score` (0-1 scale)
        return {
            "score": f1,
            "precision": precision,
            "recall": recall,
            "n_gold": len(projected_gold),
            "n_pred": len(pred),
            "true_positives": sorted(overlap),
            "false_positives": sorted(pred - projected_gold),
            "false_negatives": sorted(projected_gold - pred),
        }

    except Exception as e:
        # Even on error, return *something* so LangSmith doesn't raise
        return {
            "score": 0.0,
            "error": str(e),
        }

if __name__ == '__main__':
    client = Client()

    evaluate(
        get_example,
        data="topic-tagging-gold",
        evaluators=[check_answer],
        experiment_prefix="dummy_experiment",
    )