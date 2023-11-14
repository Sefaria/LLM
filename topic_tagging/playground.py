import csv
import django
django.setup()
# import typer
# import json
from sefaria.model import *
# from sefaria.utils.hebrew import strip_cantillation
# import random
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain
# from langchain.chat_models import ChatOpenAI
# import openai
# import re
# from sefaria.helper.normalization import NormalizerComposer, RegexNormalizer, AbstractNormalizer
# from util.general import get_removal_list

api_key = os.getenv("OPENAI_API_KEY")

def get_top_topics_slugs():
    # Read topics to promote from file
    topics_slugs = []
    with open('good_to_promote_topics.txt', 'r') as file:
        lines = file.readlines()
        topics_slugs = [line.replace("',\n", '')[2:].lower().replace(" ", '-') for line in lines]

    # Retrieve main topics list
    main_topics_list = [TopicSet({"slug": slug}).array()[0] for slug in topics_slugs]

    # Sort topics based on numSources
    key_function = lambda topic: getattr(topic, 'numSources', 0)
    top_topics = sorted(main_topics_list, key=key_function, reverse=True)[:100]

    # Write top topic slugs to CSV
    csv_file = "n_topic_slugs.csv"
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[topic.slug] for topic in top_topics])


def get_embeddings(topic_slugs_csv_path):
    topic_slugs = [row[0] for row in csv.reader(open(topic_slugs_csv_path))]
    topic
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    text = "This is a test query."
    query_result = embeddings.embed_query(text)

    print(query_result)

def query_llm_model(template, text):
    llm = OpenAI(temperature=.7)

    prompt_template = PromptTemplate(input_variables=["text"], template=template)
    answer_chain = LLMChain(llm=llm, prompt=prompt_template)
    answer = answer_chain.run(text)
    return answer

def topic_slugs_list_from_csv(slugs_path_csv):
    return [row[0] for row in csv.reader(open(slugs_path_csv))]
def topic_list_from_slugs_csv(slugs_path_csv):
    slugs = topic_slugs_list_from_csv(slugs_path_csv)
    topics = []
    for slug in slugs:
        topics += [TopicSet({"slug":slug})[0]]
    return topics

def list_of_tuples_to_csv(data, file_path):
    """
    Parameters:
    - data (list of tuples): The data to be written to the CSV file.
    - file_path (str): The path to the CSV file.
    Returns:
    - None
    """
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
def infer_topic_descriptions_to_csv(slugs_csv_path, query_template, output_csv_path):

    topic_list = topic_list_from_slugs_csv(slugs_csv_path)
    slugXdescription_list = []
    for topic in topic_list:
        print(topic.get_primary_title())
        des = query_llm_model(query_template, topic.get_primary_title())
        slug = topic.slug
        slugXdescription_list += [(slug, des)]

    list_of_tuples_to_csv(slugXdescription_list, output_csv_path)

if __name__ == '__main__':
    print("Hi")
    # get_embeddings()
    template = """You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.
    Topic: {text}
    Description:
    """
    infer_topic_descriptions_to_csv("n_topic_slugs.csv", template, "slugs_and_inferred_descriptions.csv")



    print("bye")