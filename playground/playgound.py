import django
django.setup()
import os
import openai
import math


# from langchain.chat_models import ChatOpenAI
# import openai
# import re
# from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer, AbstractNormalizer
# from util.general import get_removal_list

# api_key = os.getenv("OPENAI_API_KEY")
from langchain.embeddings import OllamaEmbeddings
from langchain.evaluation import load_evaluator


def dot_product(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

def magnitude(vector):
    return math.sqrt(sum(val**2 for val in vector))

def cosine_similarity(vector1, vector2):
    dot_prod = dot_product(vector1, vector2)
    mag1 = magnitude(vector1)
    mag2 = magnitude(vector2)

    if mag1 == 0 or mag2 == 0:
        return 0  # Avoid division by zero

    return dot_prod / (mag1 * mag2)
def euclidean_distance(vector1, vector2):
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    squared_distance = sum((x - y) ** 2 for x, y in zip(vector1, vector2))
    distance = math.sqrt(squared_distance)
    return distance


# Example usage:
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
distance = euclidean_distance(vector1, vector2)
print(f"Euclidean Distance: {distance}")

# Example usage:


if __name__ == '__main__':
    print("Hi")
    evaluator = load_evaluator("pairwise_embedding_distance")

    ollama_emb = OllamaEmbeddings(
        model="neural-chat",
    )
    a = ollama_emb.embed_query(
        "hi"
    )
    b = ollama_emb.embed_query(
        "We all live on planet:"
    )
    d = euclidean_distance(a, b)
    # a = "the company that created the iphone is:"
    # b = "a crisp and juicy fruit with a smooth, thin skin that can range in color from shades of red and green to yellow is:"
    # d = evaluator.evaluate_string_pairs(
    #     prediction=a, prediction_b=b
    # )
    print(d)


    print("bye")