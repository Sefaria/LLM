from functools import partial
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
import json, csv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from util.sefaria_specific import get_ref_text_with_fallback




@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

def jsonl_to_labelled_refs(jsonl_filename) -> List[LabelledRef]:
    lrs = []
    with open(jsonl_filename, 'r') as file:
        for line in file:
            data = json.loads(line)
            ref = data['ref']
            slugs = data['slugs']
            lrs.append(LabelledRef(ref, slugs))
    return lrs

def file_to_slugs(csv_or_json_path) -> List[str]:
    file_path =csv_or_json_path
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

class Predictor:

    def __init__(self, docs_num, above_mean_threshold_factor, power_relevance_fun):
        self.docs_num = docs_num
        self.above_mean_threshold_factor = above_mean_threshold_factor
        self.power_relevance_fun = power_relevance_fun

    def _slugs_string_to_list(self, slugs_string):
        return [s for s in slugs_string.split('$') if s]

    def _get_keys_above_mean(self, d, threshold_factor=1.0):
        if not d:
            return []
        values = list(d.values())
        mean_value = sum(values) / len(values)

        # Define the threshold as mean + threshold_factor * standard deviation
        threshold = mean_value + threshold_factor * (sum((x - mean_value) ** 2 for x in values) / len(values)) ** 0.5

        return [key for key, value in d.items() if value > threshold]

    def _get_recommended_slugs_weighted_frequency_map(self, docs, compute_weight_fn, ref_to_ignore="$$$"):
        recommended_slugs = defaultdict(lambda: 0)
        for doc, score in docs:
            if doc.metadata['Ref'] == ref_to_ignore:
                continue
            slugs = self._slugs_string_to_list(doc.metadata['Slugs'])
            for slug in slugs:
                recommended_slugs[slug] += compute_weight_fn(score)*1
        return recommended_slugs

    def _euclidean_relevance_to_l2(self, euclidean_relevance_score):
        return math.sqrt(2) * (1 - euclidean_relevance_score)

    def _l2_to_cosine_similarity(self, l2_distance):
        return 1 - (l2_distance ** 2) / 2

    def _cosine_to_one_minus_sine(self, cos_theta):
        sin_theta = math.sqrt(1 - cos_theta ** 2)
        return 1 - sin_theta

    def _euclidean_relevance_to_one_minus_sine(self, euclidean_relevance_score, power=1):
        one_minus_sine = self._cosine_to_one_minus_sine(
            self._l2_to_cosine_similarity(
                self._euclidean_relevance_to_l2(
                    euclidean_relevance_score)))
        return one_minus_sine ** power

    def predict(self, refs):
        pass


class InRAMPredictor(Predictor):

    def __init__(self, close_topics, *hyperparameters):
        """
        no calls to vector db
        @param close_topics:
        @param hyperparameters:
        """
        super().__init__(*hyperparameters)


class VectorDBPredictor(Predictor):

    def __init__(self, vector_db_dir, **hyperparameters):
        super().__init__(**hyperparameters)
        self.vector_db_dir = vector_db_dir
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_db = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)

    def _get_closest_docs_by_text_similarity(self, query, k=100, score_threshold=0.0):
        docs = self.vector_db.similarity_search_with_relevance_scores(
            query.lower(), k=k, score_threshold=score_threshold
        )
        return docs

    def predict(self, refs):
        results = []
        for ref in refs:
            text = get_ref_text_with_fallback(ref, 'en', auto_translate=True)
            docs = self._get_closest_docs_by_text_similarity(text, self.docs_num)
            recommended_slugs_sine = self._get_recommended_slugs_weighted_frequency_map(docs, lambda score: self._euclidean_relevance_to_one_minus_sine(score, self.power_relevance_fun), ref)
            best_slugs_sine = self._get_keys_above_mean(recommended_slugs_sine, self.above_mean_threshold_factor)
            results.append(LabelledRef(ref=ref, slugs=best_slugs_sine))
        return results


class PredictorFactory:

    @staticmethod
    def create(name, *args, **kwargs):
        if name == 'inrampredictor':
            return InRAMPredictor(*args, **kwargs)
        elif name == 'vectordb':
            return VectorDBPredictor(*args, **kwargs)


class Evaluator:

    def __init__(self, gold_standard: List[LabelledRef], considered_labels: List[str]):
        self.gold_standard = gold_standard
        self.considered_labels = considered_labels

    def evaluate(self, predictions: list):
        pass


class EvaluationController:

    def __init__(self, evaluator: Evaluator, predictor: Predictor):
        self.evaluator = evaluator
        self.predictor = predictor

    def evaluate(self, refs: list):
        predictions = self.predictor.predict(refs)
        return self.evaluator.evaluate(predictions)


def optimization_run(evaluator, refs, close_topics, hyperparameters):
    controller = EvaluationController(evaluator, InRAMPredictor(close_topics, hyperparameters))
    return controller.evaluate(refs)


if __name__ == '__main__':
    # close_topics = "hello"
    # refs = gold_standard
    # evaluator = Evaluator(gold_standard, considered_labels)
    # optimization_run_bound = partial(optimization_run, evaluator, close_topics, refs)
    hyperparameters = {"docs_num" : 500,
                       "above_mean_threshold_factor" : 2.5,
                       "power_relevance_fun" : 10}
    predictor = VectorDBPredictor(".chromadb_openai", **hyperparameters)
    print(predictor.predict(["Genesis 1:1", "Isaiah 2:3"]))
