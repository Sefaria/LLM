from functools import partial
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
import json, csv
import pickle
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from util.sefaria_specific import get_ref_text_with_fallback, get_passage_refs
from sefaria.model import library
from sefaria.helper.llm.topic_prompt import make_llm_topic
from sefaria_llm_interface.common.topic import Topic
from sefaria.model.topic import Topic as SefariaTopic
from experiments.topic_source_curation.common import get_topic_str_for_prompts
from experiments.topic_source_curation.gather.source_gatherer import _is_text_about_topic
from util.general import run_parallel
import optuna
from tqdm import tqdm
from collections import Counter

toc = library.get_topic_toc_json_recursive()





@dataclass
class LabelledRef:
    ref: str
    slugs: List[str]

    def __repr__(self):
        return f"LabelledRef(ref='{self.ref}', slugs={self.slugs})"

class FileMemoizer:

    def __init__(self, cache_filename):
        self.cache_filename = cache_filename
        self._cache = {}

    def load(self):
        """Load cache from a file if it exists."""
        if os.path.exists(self.cache_filename):
            with open(self.cache_filename, 'rb') as file:
                self._cache = pickle.load(file)
        else:
            self._cache = {}

    def save(self):
        """Save cache to a file."""
        with open(self.cache_filename, 'wb') as file:
            pickle.dump(self._cache, file)

    def memoize(self, func):
        """Decorator to memoize the function."""

        def wrapper(*args):
            cache_key = args[1:]  # Skip 'self' when caching instance method

            if cache_key in self._cache:
                # print("Fetching from cache")
                return self._cache[cache_key]

            # print("Computing and caching")
            result = func(*args)  # Pass all args including 'self' to the function
            self._cache[cache_key] = result
            return result

        return wrapper

vector_queries_memoizer = FileMemoizer("cached_vector_queries.pkl")
vector_queries_memoizer.load()

ref_to_text_memoizer = FileMemoizer("cached_ref_to_text.pkl")
ref_to_text_memoizer.load()



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


def filter_topics_relevant_to_text(text: str, topics: list[Topic], verbose=True):
    topic_descriptions = []
    for topic in topics:
        topic_descriptions += [get_topic_str_for_prompts(topic, verbose=False)]
    unit_func = partial(_is_text_about_topic, text=text, connections_between_text_and_topic=["is about", "mentions implicitly or explicitly"])
    is_about_text_list = run_parallel(topic_descriptions, unit_func, 50,
                                       desc="filter irrelevant topics", disable=not verbose)
    filtered_items = []
    if verbose:
        print("---FILTERING---")
    for is_about_topic, topic in zip(is_about_text_list, topics):
        if is_about_topic:
            filtered_items += [topic]
        else:
            pass
            # if verbose:
            #     print(item.ref)
            #     print(key(item))
    if verbose:
        print('after filtering: ', len(filtered_items))
    return filtered_items

def add_implied_toc_slugs(labelled_refs: List[LabelledRef], considered_labels=None):

    def _find_parent_slug(data_structure, target_slug):
        # Access the 'children' list
        children = data_structure['children']

        # Iterate through each category dictionary
        for category in children:
            # If the target slug matches the current category's slug
            if category['slug'] == target_slug:
                # Return the parent slug
                return data_structure['slug']
            # If the current category has children, recursively search within them
            if 'children' in category:
                result = _find_parent_slug(category, target_slug)
                # If the parent slug is found in any of the children, return it
                if result:
                    return result
        # If the target slug is not found in any children, return None
        return None

    for lr in labelled_refs:
        for slug in lr.slugs:
            parent = (_find_parent_slug({"children": toc, "slug": None}, slug))
            if parent and parent not in lr.slugs:
                # print(f"{parent} missing parent of {slug}")
                lr.slugs.append(parent)

        if considered_labels:
            projected = []
            for slug in lr.slugs:
                if slug in considered_labels:
                    projected.append(slug)
            lr.slugs = projected
    return labelled_refs


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

    def _euclidean_relevance_to_cosine_similarity(self, euclidean_relevance_score, power=1):
        cosine_similarity = self._l2_to_cosine_similarity(
            self._euclidean_relevance_to_l2(
            euclidean_relevance_score
        ))
        return cosine_similarity ** power

    def predict(self, refs):
        pass

# class PredictorWithCacheWrapper:
#     def __init__(self, predictor, cache_file="predictions_cache.json"):
#         self.predictor = predictor
#         self.cache_file = cache_file
#         self._load_cache()
#
#     def _load_cache(self):
#         if os.path.exists(self.cache_file):
#             with open(self.cache_file, "r", encoding="utf-8") as f:
#                 try:
#                     self.cache = json.load(f)
#                 except json.JSONDecodeError:
#                     self.cache = {}
#         else:
#             self.cache = {}
#
#     def _save_cache(self):
#         with open(self.cache_file, "w", encoding="utf-8") as f:
#             json.dump(self.cache, f, ensure_ascii=False, indent=2)
#
#     def __getattr__(self, name):
#         return getattr(self.predictor, name)
#
#     def predict(self, refs):
#         results = []
#         refs_to_predict = []
#
#         for ref in refs:
#             if ref in self.cache:
#                 tqdm.write(f"Loading cached prediction for: {ref}")
#                 results.append(self.cache[ref])
#             else:
#                 refs_to_predict.append(ref)
#
#         if refs_to_predict:
#             for ref in refs_to_predict:
#                 prediction = self.predictor.predict([ref])
#                 results.append(prediction)
#                 self.cache[ref] = prediction
#                 self._save_cache()
#         return results

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


    def _count_slug_appearances(self):
        all_docs = self.vector_db.get()
        slug_counts = Counter()
        for metadata in all_docs['metadatas']:  # Assuming docs is a list of (Document, score) tuples
            slugs = metadata['Slugs']
            if slugs:
                for slug in slugs.split('$'):
                    if slug == "":
                        continue
                    slug_counts[slug] += 1

        return dict(slug_counts)
    @vector_queries_memoizer.memoize
    def _get_all_similar_docs(self, query):
        docs = self.vector_db.similarity_search_with_relevance_scores(
            query.lower(), k=10000, score_threshold=0.0
        )
        return docs

    def _get_closest_docs_by_text_similarity(self, query, k=100, score_threshold=0.0):
        docs = self._get_all_similar_docs(query)
        return docs[:k]

    # @ref_to_text_memoizer.memoize
    def _get_en_text_for_ref(self, ref):
        text = get_ref_text_with_fallback(ref, 'en', auto_translate=True)
        return text

    def _filter_irrelevent_slugs_by_llm(self, text, slugs):
        sefaria_topics = [SefariaTopic.init(slugs) for slugs in slugs]
        topic_objs = [make_llm_topic(sefaria_topic) for sefaria_topic in sefaria_topics]
        filtered_topics = filter_topics_relevant_to_text(text, topic_objs, verbose=False)
        filtered_slugs = [topic.slug for topic in filtered_topics]
        return filtered_slugs

    def predict(self, refs):
        results = []
        for ref in refs:
            text = self._get_en_text_for_ref(ref)
            docs = self._get_closest_docs_by_text_similarity(text, self.docs_num)
            recommended_slugs = self._get_recommended_slugs_weighted_frequency_map(docs, lambda score: self._euclidean_relevance_to_one_minus_sine(score, self.power_relevance_fun), ref)
            # recommended_slugs = self._get_recommended_slugs_weighted_frequency_map(docs, lambda score: self._euclidean_relevance_to_cosine_similarity(score), ref)
            best_slugs_sine = self._get_keys_above_mean(recommended_slugs, self.above_mean_threshold_factor)
            llm_filtered_slugs = self._filter_irrelevent_slugs_by_llm(text, best_slugs_sine)
            results.append(LabelledRef(ref=ref, slugs=llm_filtered_slugs))
        results = add_implied_toc_slugs(results)
        return results

class ContextVectorDBPredictor(VectorDBPredictor):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_frequency_map = self._count_slug_appearances()
        self.slugs_frequency_partitions = self._partition_slugs_by_frequency()

    def _partition_slugs_by_frequency(self):
        slug_counts = self.total_frequency_map
        # Step 1: Sort slugs by frequency (descending order)
        sorted_slugs = sorted(slug_counts.items(), key=lambda x: x[1], reverse=True)

        # Step 2: Determine the number of sets
        num_sets = max(1, int(math.sqrt(len(slug_counts))))  # Ensure at least 1 set

        # Step 3: Initialize sets
        partitions = [set() for _ in range(num_sets)]

        # Step 4: Distribute slugs into sets (round-robin fashion)
        for i, (slug, count) in enumerate(sorted_slugs):
            partitions[i % num_sets].add(slug)

        return partitions

    def _get_wider_context_ref_list(self, tref):
        try:
            ref_list, full_ref = get_passage_refs(tref)
            # return ref_list + [full_ref]
            return [full_ref, tref]
        except:
            return [tref]

    def _predict_single_ref(self, ref):
        text = self._get_en_text_for_ref(ref)
        docs = self._get_closest_docs_by_text_similarity(text, self.docs_num)
        result = []
        for slugs_partition in self.slugs_frequency_partitions:
            all_recommended_slugs = self._get_recommended_slugs_weighted_frequency_map(docs, lambda
                score: self._euclidean_relevance_to_one_minus_sine(score, self.power_relevance_fun), ref)
            recommended_slugs = {slug: all_recommended_slugs[slug] for slug in all_recommended_slugs if slug in slugs_partition}
            best_slugs_sine = self._get_keys_above_mean(recommended_slugs, self.above_mean_threshold_factor)
            result += best_slugs_sine
        ###without partitioning:
        # all_recommended_slugs = self._get_recommended_slugs_weighted_frequency_map(docs, lambda
        #     score: self._euclidean_relevance_to_one_minus_sine(score, self.power_relevance_fun), ref)
        # best_slugs_sine = self._get_keys_above_mean(all_recommended_slugs, self.above_mean_threshold_factor)
        # result = best_slugs_sine
        return result

    def predict(self, refs):
        results = []
        for ref in tqdm(refs, desc="Tagging refs"):
            tqdm.write(f"Tagging main ref: {ref}")
            predicted_slugs = set()
            ref_list = self._get_wider_context_ref_list(ref)
            for expanded_ref in tqdm(ref_list, desc="Tagging expanded refs;"):
                tqdm.write(f"tagging expanded ref: {expanded_ref}")
                predicted_slugs.update(self._predict_single_ref(expanded_ref))
            results.append(LabelledRef(ref=ref, slugs=list(predicted_slugs)))
        results = add_implied_toc_slugs(results)
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
        self.considered_labels = considered_labels
        gold_standard = self._get_projection_of_labelled_refs(gold_standard)
        self.gold_standard = gold_standard

    def evaluate(self, predictions: List[LabelledRef]):
        predictions = self._get_projection_of_labelled_refs(predictions)
        refs_in_pred = [lr.ref for lr in predictions]
        refs_in_gold = [lr.ref for lr in self.gold_standard]
        predictions_filtered = [lr for lr in predictions if lr.ref in refs_in_gold]
        gold_filtered = [lr for lr in self.gold_standard if lr.ref in refs_in_pred]
        result = self._compute_total_recall(gold_filtered, predictions_filtered)
        return result
    def get_total_recall(self, predictions: List[LabelledRef]):
        predictions = self._get_projection_of_labelled_refs(predictions)
        refs_in_pred = [lr.ref for lr in predictions]
        refs_in_gold = [lr.ref for lr in self.gold_standard]
        predictions_filtered = [lr for lr in predictions if lr.ref in refs_in_gold]
        gold_filtered = [lr for lr in self.gold_standard if lr.ref in refs_in_pred]
        result = self._compute_total_recall(gold_filtered, predictions_filtered)
        return result

    def get_total_precision(self, predictions: List[LabelledRef]):
        predictions = self._get_projection_of_labelled_refs(predictions)
        refs_in_pred = [lr.ref for lr in predictions]
        refs_in_gold = [lr.ref for lr in self.gold_standard]
        predictions_filtered = [lr for lr in predictions if lr.ref in refs_in_gold]
        gold_filtered = [lr for lr in self.gold_standard if lr.ref in refs_in_pred]
        result = self._compute_total_precision(gold_filtered, predictions_filtered)
        return result

    def _get_projection_of_labelled_refs(self, lrs :List[LabelledRef]) -> List[LabelledRef]:
        # Remove irrelevant slugs from the slugs list
        projected = []
        for ref in lrs:
            projected.append(LabelledRef(ref.ref, [slug for slug in ref.slugs if slug in self.considered_labels]))
        return projected

    def _compute_metrics_for_refs_pair(self, golden_standard_ref: LabelledRef, predicted_ref: LabelledRef):
        golden_standard = golden_standard_ref.slugs
        predicted = predicted_ref.slugs

        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for item in predicted:
            if item in golden_standard:
                true_positives += 1
            else:
                false_positives += 1

        for item in golden_standard:
            if item not in predicted:
                false_negatives += 1

        return true_positives, false_positives, false_negatives

    def _compute_f1_score(self, gold_standard: List[LabelledRef], predictions: List[LabelledRef]):
        precision = self._compute_total_precision(gold_standard, predictions)
        recall = self._compute_total_recall(gold_standard, predictions)

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score

    def _compute_weighted_f1_score(self, gold_standard: List[LabelledRef], predictions: List[LabelledRef], beta=1.0):
        """
        - beta (float): The weight of recall relative to precision (default is 1.0, i.e., F1 score).
        """
        precision = self._compute_total_precision(gold_standard, predictions)
        recall = self._compute_total_recall(gold_standard, predictions)
        if precision + recall == 0:
            return 0.0
        beta_squared = beta ** 2
        weighted_f1 = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        return weighted_f1

    def _compute_total_recall(self, gold_standard: List[LabelledRef], predictions: List[LabelledRef]):
        total_true_positives = 0
        total_false_negatives = 0

        for golden_standard_ref, predicted_ref in zip(gold_standard, predictions):
            true_positives, _, false_negatives = self._compute_metrics_for_refs_pair(golden_standard_ref, predicted_ref)
            # if false_negatives != 0:
            #     print("oh!")
            total_true_positives += true_positives
            total_false_negatives += false_negatives

        total_recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0
        return total_recall

    def _compute_total_precision(self, gold_standard: List[LabelledRef], predictions: List[LabelledRef]):
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0

        for golden_standard_ref, predicted_ref in zip(gold_standard, predictions):
            true_positives, false_positives, false_negatives = self._compute_metrics_for_refs_pair(golden_standard_ref, predicted_ref)
            total_true_positives += true_positives
            total_false_positives += false_positives
            total_false_negatives += false_negatives

        total_precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0
        return total_precision


class EvaluationController:

    def __init__(self, evaluator: Evaluator, predictor: Predictor):
        self.evaluator = evaluator
        self.predictor = predictor

    def evaluate(self, refs: List[str]):
        predictions = self.predictor.predict(refs)
        return self.evaluator.evaluate(predictions)


def optimization_run(evaluator, refs, db_dir, hyperparameters):
    controller = EvaluationController(evaluator, VectorDBPredictor(db_dir, **hyperparameters))
    return controller.evaluate(refs)


if __name__ == '__main__':
    # close_topics = "hello"
    # refs = gold_standard
    # evaluator = Evaluator(gold_standard, considered_labels)
    # hyperparameters = {"docs_num" : 500,
    #                    "above_mean_threshold_factor" : 2.5,
    #                    "power_relevance_fun" : 10}
    considered_labels = file_to_slugs("evaluation_data/all_slugs_in_training_set.csv")
    gold_standard = jsonl_to_labelled_refs("evaluation_data/gold.jsonl")
    gold_standard = add_implied_toc_slugs(gold_standard)
    refs_to_evaluate = [labelled_ref.ref for labelled_ref in gold_standard]
    evaluator = Evaluator(gold_standard, considered_labels)

    optimization_run_bound = partial(optimization_run, evaluator, refs_to_evaluate, ".chromadb_openai")
    def objective(trial):
        docs_num_magnitude = trial.suggest_int("docs_num_magnitude", 10, 100)
        above_mean_threshold_factor = trial.suggest_float("above_mean_threshold_factor", 0.25, 5.0)
        power_relevance_fun = trial.suggest_int("power_relevance_fun", 1, 20)
        hyperparameters = {
            "docs_num": docs_num_magnitude*10,
            "above_mean_threshold_factor": above_mean_threshold_factor,
            "power_relevance_fun": power_relevance_fun
        }
        score = optimization_run_bound(hyperparameters)
        return score


    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1000)
    print(study.best_params)
    print(study.best_params)
    print(study.best_trial)

    vector_queries_memoizer.save()
    ref_to_text_memoizer.save()
