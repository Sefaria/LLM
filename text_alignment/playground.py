import re
import django
from functools import lru_cache
# import anthropic
django.setup()
from sefaria.model import *

# from langchain.cache import SQLiteCache
# import langchain
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

model = SentenceTransformer('sentence-transformers/LaBSE')
def naive_tokenizer(text: str):
    return text.split()


class SegmentSplitter:
    end_of_prefix_token = "$"
    last_prefix = False

    def __init__(self, entire_segment):
        self.entire_segment = entire_segment
        self.tokens_list = [self.end_of_prefix_token] + naive_tokenizer(entire_segment)

    def get_current_prefix(self):
        result_list = self._get_prefix_until_token(self.tokens_list, self.end_of_prefix_token)
        result = ' '.join(result_list)
        return result
    def make_next_prefix(self):
        self.tokens_list = (
            self._swap_token_with_next(self.tokens_list, self.end_of_prefix_token))
        if self.tokens_list.index(self.end_of_prefix_token) + 1 == len(self.tokens_list):
            self.last_prefix = True

    def _swap_token_with_next(self, lst, token):
        try:
            # Find the index of the token
            token_index = lst.index(token)

            # Swap the token with the next element
            lst[token_index], lst[token_index + 1] = lst[token_index + 1], lst[token_index]
            return lst
        except ValueError:
            print(f"Token '{token}' not found in the list.")

    def _get_prefix_until_token(self, lst, token):
        prefix = []
        for item in lst:
            if item == token:
                break
            prefix.append(item)
        return prefix



def compute_similarity(vector_1, vector_2):
    return np.matmul(vector_1, np.transpose(vector_2))
    # return np.linalg.norm(vector_1 - vector_2)

def load_model():
    model = SentenceTransformer('sentence-transformers/LaBSE')
    return model

def embed_and_compute_similarity(text1, text2):
    # model = load_model()
    text1_embedding = model.encode(text1)
    text2_embedding = model.encode(text2)
    similarity = compute_similarity(text1_embedding, text2_embedding)
    return similarity

def find_best_semantic_prefix(source, to_align):
    model = load_model()
    source_embedding = model.encode(source)
    splitter = SegmentSplitter(to_align)

    max_similarity = 0
    best_aligned_text = None

    while not splitter.last_prefix:
        aligned_text = splitter.get_current_prefix()
        aligned_embedding = model.encode(aligned_text)
        similarity = compute_similarity(source_embedding, aligned_embedding)

        if similarity > max_similarity:
            max_similarity = similarity
            best_aligned_text = aligned_text

        splitter.make_next_prefix()
    return (best_aligned_text, max_similarity)

def align_target(source_list, to_align: str):
    to_align = to_align.split()
    to_align = ' '.join(to_align)
    aligned_pairs = []
    for segment in source_list:
        best_aligned_text, _ = find_best_semantic_prefix(segment, to_align)
        aligned_pairs.append((segment, best_aligned_text))
        to_align = to_align.replace(best_aligned_text, "")
    return aligned_pairs

def generate_partitions(lis, n):
    if n == 1:
        yield [lis]
        return
    for i in range(1, len(lis)):
        prefix = lis[:i]
        suffix = lis[i:]
        for rest in generate_partitions(suffix, n - 1):
            yield [prefix] + rest

def score_partition(source_partition, target_partition):
    aggregated = 0
    for s, t in zip(source_partition, target_partition):
        similarity = embed_and_compute_similarity(s, t)
        aggregated += similarity
    return aggregated

def align_target_naive(source_list, to_align: str):
    num_of_segments = len(source_list)
    all_possible_partitions = list(generate_partitions(to_align.split(), num_of_segments))
    all_possible_partitions_temp = []
    for partition in all_possible_partitions:
        partition_temp = []
        for chunk in partition:
            chunk_str = ' '.join(chunk)
            partition_temp.append(chunk_str)
        all_possible_partitions_temp.append(partition_temp)
    all_possible_partitions = all_possible_partitions_temp

    best_score = 0
    best_partition = None
    for partition in tqdm(all_possible_partitions, desc="Processing partitions", unit="partition"):
        score = score_partition(source_list, partition)
        if score > best_score:
            best_score = score
            best_partition = partition

    return best_partition


if __name__ == '__main__':
    print("hi")
#     source = """
# “Reuben, you are my firstborn, my might, and the firstfruits of my strength, preeminent in dignity and preeminent in power.
# """
#     to_align = """
# Reuben, thou art my firstborn, my might, and the beginning of my strength, the excellency of dignity, and the excellency of power:    And he took some of his brethren, even five men, and presented them unto Pharaoh.
# Unstable as water, thou shalt not excel; because thou wentest up to thy father's bed; then defiledst thou it: he went up to my couch.
#     """
#
#     prefix, similarity_rate = find_best_semantic_prefix(source, to_align)
#     print(prefix)
#     print(similarity_rate)
    source = [
        "To Sherlock Holmes she is always the woman.",
        "I have seldom heard him mention her under any other name.",
        "In his eyes she eclipses and predominates the whole of her sex.",
        "It was not that he felt any emotion akin to love for Irene Adler."
    ]
    to_align = """
    Pour Sherlock Holmes c’est toujours "la femme". Il ne parle jamais d’elle que sous cette dénomination ; à ses yeux elle éclipse le sexe faible tout entier. Ne croyez pourtant pas qu’il ait eu de l’amour, voire même de l’affection pour Irène Adler. 
    """
    # aligned_pairs = align_target(source, to_align)
    # for pair in aligned_pairs:
    #     print("Source:")
    #     print(pair[0])
    #     print("Aligned:")
    #     print(pair[1])



    # Example usage:
    aligned = align_target_naive(source, to_align)
    print(aligned)
    # score = score_partition(source, ['Pour', 'Sherlock', 'Holmes c’est toujours "la femme". Il ne parle jamais d’elle que sous cette dénomination ; à ses yeux elle éclipse le sexe faible tout entier. Ne croyez pourtant', 'pas qu’il ait eu de l’amour, voire même de l’affection pour Irène Adler.'])
    # print(score)
    # score = score_partition(source, ['Pour Sherlock Holmes c’est toujours "la femme"', "Il ne parle jamais d’elle que sous cette dénomination ;", "à ses yeux elle éclipse le sexe faible tout entier.", "Ne croyez pourtant pas qu’il ait eu de l’amour, voire même de l’affection pour Irène Adler."])
    # print(score)
