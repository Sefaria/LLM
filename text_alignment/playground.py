import re
import django
# import anthropic
django.setup()
from sefaria.model import *
from sefaria.client.wrapper import get_links
from sefaria.datatype.jagged_array import JaggedTextArray
from util.openai import get_completion_openai, count_tokens_openai
from langchain.chat_models import ChatAnthropic
from langchain.schema import HumanMessage
from langchain.cache import SQLiteCache
import langchain
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")
from sentence_transformers import SentenceTransformer
import numpy as np


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



def find_best_semantic_prefix(source, to_align):
    model = SentenceTransformer('sentence-transformers/LaBSE')
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

if __name__ == '__main__':
    print("hi")
    source = """
    So Joseph went in and told Pharaoh, “My father and my brothers, with their flocks and herds and all that they possess, have come from the land of Canaan. They are now in the land of Goshen.”
    """
    to_align = """
    Then Joseph came and told Pharaoh, and said, My father and my brethren, and their flocks, and their herds, and all that they have, are come out of the land of Canaan; and, behold, they are in the land of Goshen. 
    And he took some of his brethren, even five men, and presented them unto Pharaoh.
    """

    prefix, similarity_rate = find_best_semantic_prefix(source, to_align)
    print(prefix)
    print(similarity_rate)






