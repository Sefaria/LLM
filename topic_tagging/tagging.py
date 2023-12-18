import csv
import django
django.setup()
# We can do the same thing with a SQLite cache
# from langchain.cache import SQLiteCache
# set_llm_cache(SQLiteCache(database_path=".langchain.db"))
# import typer
import re
import json
from sefaria.model import *
# from sefaria.utils.hebrew import strip_cantillation
# import random
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
import  numpy as np

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


def embed_text(query):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, request_timeout=120)
    text = query
    query_result = embeddings.embed_query(text)
    return query_result

def format_string(fstring, arguments):
    try:
        formatted_string = fstring.format(*arguments)
        return formatted_string
    except IndexError:
        print("Error: Number of arguments does not match placeholders in the string.")
    except KeyError:
        print("Error: Invalid placeholder in the string.")
    except Exception as e:
        print(f"Error: {e}")

def query_llm_model(template_fstirng, arguments_list):
    formatted_string = template_fstirng.format(*arguments_list)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.5, request_timeout=120)
    user_prompt = PromptTemplate.from_template("# Input\n{text}")
    human_message = HumanMessage(content=user_prompt.format(text=formatted_string))
    answer = llm([human_message])

    return answer.content

def topic_slugs_list_from_csv(slugs_path_csv):
    return [row[0] for row in csv.reader(open(slugs_path_csv))]
def topic_list_from_slugs_csv(slugs_path_csv):
    slugs = topic_slugs_list_from_csv(slugs_path_csv)
    topics = []
    for slug in slugs:
        topics += [TopicSet({"slug":slug})[0]]
    return topics

def slugs_and_description_dict_from_csv():
    return dict((row[0], row[1]) for row in csv.reader(open('slugs_and_inferred_descriptions.csv')))

def list_of_tuples_to_csv(data, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(data)
def _get_top_n_orefs_for_topic(slug, top_n=10):
    from sefaria.helper.topic import get_topic

    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    result = []
    for d in out['refs']['about']['refs'][:top_n]:
        try:
            result.append(Ref(d['ref']))
        except Exception as e:
            print (e)
    return result
def _get_first_k_categorically_distinct_refs(refs, k):
    distinct_categories = set()
    result = []

    for ref in refs:
        category = ref.primary_category
        if category is not None and category not in distinct_categories:
            distinct_categories.add(category)
            result.append(ref)

            if len(result) == k:
                break

    return result
def _concatenate_passages(passages, separetaor_token):
    result = ""

    for i, passage in enumerate(passages, 1):
        result += f"{separetaor_token} {i}:\n{passage}\n"

    return result
def _get_relevant_example_passages_from_slug(slug, n=3, discard_longest = 0):
    refs = _get_top_n_orefs_for_topic(slug, 200)
    refs = _get_first_k_categorically_distinct_refs(refs, n)
    def _get_length_of_ref(ref):
        return len(ref.text().text)
    for i in range(0, discard_longest):
        longest_element = max(refs, key=_get_length_of_ref)
        refs.remove(longest_element)
    text = _concatenate_passages([ref.tref + ': ' + str(ref.text().text) for ref in refs], "Passage")

    return text

def infer_topic_descriptions_to_csv(slugs_csv_path, output_csv_path):
    template = """You are a humanities scholar specializing in Judaism. Given a Topic or a term and a list of passages that relate to that topic, write a description for that topic from a Jewish perspective.
    Don't quote the passages in your answer, don't summarize the passages, but rather explain the general concept to the Topic.
    Topic: {0}
    Related Passages: {1}
    Description:
    """


    topic_list = topic_list_from_slugs_csv(slugs_csv_path)
    slugXdescription_list = []
    # for topic in topic_list:
    #     print(topic.get_primary_title())
    #     des = query_llm_model(template, topic.get_primary_title())
    #     slug = topic.slug
    #     slugXdescription_list += [(slug, des)]
    for topic in topic_list:
        print(topic.get_primary_title())
        slug = topic.slug
        # if slug != "peace":
        #     continue
        passages = _get_relevant_example_passages_from_slug(slug)
        try:
            des = query_llm_model(template, [topic.get_primary_title(), passages])
        except:
            try:
                passages = _get_relevant_example_passages_from_slug(slug, discard_longest=1)
                des = query_llm_model(template, [topic.get_primary_title(), passages])
            except:
                try:
                    passages = _get_relevant_example_passages_from_slug(slug, discard_longest=2)
                    des = query_llm_model(template, [topic.get_primary_title(), passages])
                except:
                    des = query_llm_model(template, [topic.get_primary_title(), ''])



        slugXdescription_list += [(slug, des)]

    list_of_tuples_to_csv(slugXdescription_list, output_csv_path)

def embed_topic_descriptions_to_jsonl(slugs_and_descriptions_csv, output_jsonl_path):
    list_of_embeddings = []
    slugXdes_list = [(row[0], row[1]) for row in csv.reader(open(slugs_and_descriptions_csv))]
    for slugXdes in slugXdes_list:
        slug = slugXdes[0]
        des = slugXdes[1]
        list_of_embeddings += [{"slug": slug, "embedding":embed_text(des)}]
    with open(output_jsonl_path, 'w') as jsonl_file:
        for item in list_of_embeddings:
            jsonl_file.write(json.dumps(item) + '\n')




class TopicsData:
    def __init__(self, data_jsonl_filename):
        self.data_jsonl_filename = data_jsonl_filename

    def _read_jsonl_into_list_of_dicts(self):
        with open(self.data_jsonl_filename, 'r') as file:
            data_list = [json.loads(line) for line in file]
        return data_list

    def _get_dict_by_key_value(self, dict_list, key, value):
        for d in dict_list:
            if d.get(key) == value:
                return d
        return None

    def _write_list_of_dicts_to_jsonl(self, list_of_dicts):
        with open(self.data_jsonl_filename, 'w') as jsonl_file:
            for item in list_of_dicts:
                jsonl_file.write(json.dumps(item) + '\n')

    def get_description(self, slug):
        topics_data_list = self._read_jsonl_into_list_of_dicts()
        topic_dict = self._get_dict_by_key_value(topics_data_list, "slug", slug)
        description = None
        if topic_dict:
            description = topic_dict.get("description")
        return description

    def get_embedding(self, slug):
        topics_data_list = self._read_jsonl_into_list_of_dicts()
        topic_dict = self._get_dict_by_key_value(topics_data_list, "slug", slug)
        embedding = None
        if topic_dict:
            embedding = topic_dict.get("embedding")
        return embedding

    def set_description(self, slug, description):
        topics_data_list = self._read_jsonl_into_list_of_dicts()
        topic_dict = self._get_dict_by_key_value(topics_data_list, "slug", slug)
        if topic_dict:
            topic_dict["description"] = description
        else:
            topics_data_list += [{"slug": slug, "description": description}]
        self._write_list_of_dicts_to_jsonl(topics_data_list)

    def set_embedding(self, slug, embedding):
        topics_data_list = self._read_jsonl_into_list_of_dicts()
        topic_dict = self._get_dict_by_key_value(topics_data_list, "slug", slug)
        if topic_dict:
            topic_dict["embedding"] = embedding
        else:
            topics_data_list += {"slug": slug, "embedding": embedding}
        self._write_list_of_dicts_to_jsonl(topics_data_list)

    def get_slugs_and_descriptions_dict(self):
        data_list_of_dicts = self._read_jsonl_into_list_of_dicts()
        result_dict = {}
        for topic_dict in data_list_of_dicts:
            slug = topic_dict.get("slug")
            description = topic_dict.get("description")
            if description:
                result_dict[slug] = description
        return result_dict

    def get_slugs_and_embeddings_dict(self):
        data_list_of_dicts = self._read_jsonl_into_list_of_dicts()
        result_dict = {}
        for topic_dict in data_list_of_dicts:
            slug = topic_dict.get("slug")
            embedding = topic_dict.get("embedding")
            if embedding:
                result_dict[slug] = embedding
        return result_dict


class TopicsEmbedder:
    description_generation_prompt_template = """You are a humanities scholar specializing in Judaism. Given a Topic or a term and a list of passages that relate to that topic, write a description for that topic from a Jewish perspective.
    Don't quote the passages in your answer, don't summarize the passages, but rather explain the general concept to the Topic.
    Topic: {0}
    Related Passages: {1}
    Description:
    """

    def __init__(self, data_handler: TopicsData):
        self.data_handler = data_handler

    def _get_top_n_orefs_for_topic(self, slug, top_n=10):
        from sefaria.helper.topic import get_topic

        out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
        result = []
        for d in out['refs']['about']['refs'][:top_n]:
            try:
                result.append(Ref(d['ref']))
            except Exception as e:
                print(e)
        return result

    def _get_first_k_categorically_distinct_refs(self, refs, k):
        distinct_categories = set()
        result = []

        for ref in refs:
            category = ref.primary_category
            if category is not None and category not in distinct_categories:
                distinct_categories.add(category)
                result.append(ref)

                if len(result) == k:
                    break

        return result
    def _get_relevant_example_passages_from_slug(self, slug, n=3, discard_longest=0):
        refs = self._get_top_n_orefs_for_topic(slug, 200)
        refs = self._get_first_k_categorically_distinct_refs(refs, n)

        def _get_length_of_ref(ref):
            return len(ref.text().text)

        for i in range(0, discard_longest):
            longest_element = max(refs, key=_get_length_of_ref)
            refs.remove(longest_element)
        text = _concatenate_passages([ref.tref + ': ' + str(ref.text().text) for ref in refs], "Passage")
        return text

    def _get_topic_object(self, slug):
        return TopicSet({"slug": slug})[0]

    def _embed_text(self, text):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, request_timeout=120)
        query_result = embeddings.embed_query(text)
        return query_result

    def _try_to_get_description_based_on_sources(self, slug, query_template):
        source_passages = self._get_relevant_example_passages_from_slug(slug)
        topic_object = self._get_topic_object(slug)
        topic_title = topic_object.get_primary_title()
        try:
            des = query_llm_model(query_template, [topic_title, source_passages])
        except:
            try:
                source_passages = self.__get_relevant_example_passages_from_slug(slug, discard_longest=1)
                des = query_llm_model(query_template, [topic_title, source_passages])
            except:
                try:
                    source_passages = self.__get_relevant_example_passages_from_slug(slug, discard_longest=2)
                    des = query_llm_model(query_template, [topic_title, source_passages])
                except:
                    des = query_llm_model(query_template, [topic_title, ''])
        return des

    def generate_description(self, slug):
        description = self._try_to_get_description_based_on_sources(slug, self.description_generation_prompt_template)
        self.data_handler.set_description(slug, description)

    def generate_embedding(self, slug):
        description = self.data_handler.get_description(slug)
        if not description:
            raise ValueError(f"No description found for slug: {slug}")
        embedding = self._embed_text(description)
        self.data_handler.set_embedding(slug, embedding)


class TopicsVectorSpace:
    embed_topic_template = (
        "You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.\n"
        "Topic: {0}\nDescription:"
    )

    def __init__(self, data_handler: TopicsData):
        slug_embeddings_dict = data_handler.get_slugs_and_embeddings_dict()
        for slug in slug_embeddings_dict.keys():
            slug_embeddings_dict[slug] = np.array(slug_embeddings_dict[slug])
        self.slug_embeddings_dict = slug_embeddings_dict

    def _embed_and_get_embedding(self, template, topic):
        return np.array(embed_text(query_llm_model(template, [topic])))
    def _embedding_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    def get_nearest_topic(self, slug):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, slug)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slug = sorted_data[0][0]
        return sefaria_slug

class TopicVerifier:
    verifier_template = (
        "You are a humanities scholar specializing in Judaism. Given a passage, a topic, and a description of that topic, return YES if the passage can be tagged with this topic. "
        "Note: Even if the topic is not mentioned explicitly in the passage, but the passage refers to the general concept of the topic, or the general concept coud be found within the passage, return YES."
        "If it's not a good topic tagging for the passage, return NO. If you are unsure, return NO.\n"
        "Passage: {0}\nPossible Topic: {1}\nTopic Description: {2}"
    )

    def __init__(self, data_handler: TopicsData):
        self.slug_descriptions_dict = data_handler.get_slugs_and_descriptions_dict()

    def verify_topic(self, sefaria_slug, segment_text):
        description = self.slug_descriptions_dict[sefaria_slug]
        ver_response = query_llm_model(self.verifier_template, [segment_text, sefaria_slug, description]).replace('# Output',
                                                                                                    '').strip()
        ver_approved = False
        if "YES" in ver_response:
            ver_approved = True
        if "NO" in ver_response:
            ver_approved = False
        return ver_approved
class TopicTagger:
    infer_topics_template = (
        "You are a humanities scholar specializing in Judaism. "
        "Given the following text segment, generate an unnumbered list of relevant short topics or tags. "
        "Topics can also be general concepts and creative, and don't have to appear in the text itself. "
        "Try to infer the 'theme' of the segment, what it tries to teach us, and generate topics accordingly"
        "If the text talks about a general concept that does not appear explicitly in the text, I still consider that as a valid topic for the text.\n"
        "The Text: {0}"
    )


    def __init__(self, topics_space: TopicsVectorSpace, verifier: TopicVerifier):
        self.verifier = verifier
        self.topics_space = topics_space


    def _get_inferred_topics(self, template, segment_text):
        answer = query_llm_model(template, [segment_text])
        return [topic.strip() for topic in answer.split("\n-")]

    def _load_existing_topics(self, jsonl_path):
        existing_topics_dicts = [json.loads(line) for line in
                                 open(jsonl_path, 'r')]
        for topic_dict in existing_topics_dicts:
            topic_dict["embedding"] = np.array(topic_dict["embedding"])
        return existing_topics_dicts

    def _embed_and_get_embedding(self, template, topic):
        return np.array(embed_text(query_llm_model(template, [topic])))

    def _get_Sefaria_nearest_topic(self, inferred_topic):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, inferred_topic)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slug = sorted_data[0][0]
        # print(f"GPT tagged passage with the topic: {inferred_topic}, which is similar to Sefaria's topic: {sefaria_slug}")
        return sefaria_slug


    def tag_segment(self, segment_text):
        model_topics = self._get_inferred_topics(self.infer_topics_template, segment_text)
        verified_slugs = set()

        for inferred_topic in model_topics:
            sefaria_slug = self.topics_space.get_nearest_topic(inferred_topic)
            verified = self.verifier.verify_topic(sefaria_slug, segment_text)
            print(f"LLM Tag: {inferred_topic}")
            print(f"Sefaria Nearest Slug: {sefaria_slug}")
            if verified:
                # print("https://www.sefaria.org/topics/" + sefaria_slug)
                print("LLM Verification: Accept")
                verified_slugs.add(sefaria_slug)
            else:
                print("LLM Verification: Reject")

        return model_topics, set(verified_slugs)




if __name__ == '__main__':
    print("Hi")

    data_handler = TopicsData("experiment.jsonl")
    embedder = TopicsEmbedder(data_handler)
    # embedder.generate_description("shabbat")
    # embedder.generate_embedding("shabbat")
    # embedder.generate_description("money")
    # embedder.generate_embedding("money")

    # embedder.generate_description("happiness")
    # embedder.generate_embedding("happiness")

    verifier = TopicVerifier(data_handler)
    topics_space = TopicsVectorSpace(data_handler)
    tagger = TopicTagger(topics_space=topics_space, verifier=verifier)
    tagger.tag_segment(Ref("Kohelet_Rabbah.1.7.1").text().text)



    print("bye")