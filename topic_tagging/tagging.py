import django
django.setup()
import json
from sefaria.model import *
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
import  numpy as np


api_key = os.getenv("OPENAI_API_KEY")

class LLMOracle:
    def __init__(self, chat_model_name="gpt-3.5-turbo", temperature=.5):
        self.chat_model_name = chat_model_name
        self.temperature = temperature

    def query_llm(self, template_fstring, arguments_list):
        formatted_string = template_fstring.format(*arguments_list)
        llm = ChatOpenAI(model=self.chat_model_name, temperature=self.temperature, request_timeout=120)
        user_prompt = PromptTemplate.from_template("# Input\n{text}")
        human_message = HumanMessage(content=user_prompt.format(text=formatted_string))
        answer = llm([human_message])

        return answer.content

    def embed_text(self, text):
        embeddings = OpenAIEmbeddings(openai_api_key=api_key, request_timeout=120)
        query_result = embeddings.embed_query(text)
        return query_result
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
            topics_data_list += [{"slug": slug, "embedding": embedding}]
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

    def __init__(self, data_handler: TopicsData, oracle: LLMOracle):
        self.data_handler = data_handler
        self.oracle = oracle

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

    def _concatenate_passages(self, passages, separetaor_token):
        result = ""
        for i, passage in enumerate(passages, 1):
            result += f"{separetaor_token} {i}:\n{passage}\n"

        return result
    def _get_relevant_example_passages_from_slug(self, slug, n=3, discard_longest=0):
        refs = self._get_top_n_orefs_for_topic(slug, 200)
        refs = self._get_first_k_categorically_distinct_refs(refs, n)

        def _get_length_of_ref(ref):
            return len(ref.text().text)

        for i in range(0, discard_longest):
            longest_element = max(refs, key=_get_length_of_ref)
            refs.remove(longest_element)
        text = self._concatenate_passages([ref.tref + ': ' + str(ref.text().text) for ref in refs], "Passage")
        return text

    def _get_topic_object(self, slug):
        return TopicSet({"slug": slug})[0]
    def _try_to_get_description_based_on_sources(self, slug, query_template):
        # source_passages = self._get_relevant_example_passages_from_slug(slug)
        #
        # topic_object = self._get_topic_object(slug)
        # topic_title = topic_object.get_primary_title()
        # try:
        #     des = self.oracle.query_llm(query_template, [topic_title, source_passages])
        # except:
        #     try:
        #         source_passages = self._get_relevant_example_passages_from_slug(slug, discard_longest=1)
        #         des = self.oracle.query_llm(query_template, [topic_title, source_passages])
        #     except:
        #         try:
        #             source_passages = self._get_relevant_example_passages_from_slug(slug, discard_longest=2)
        #             des = self.oracle.query_llm(query_template, [topic_title, source_passages])
        #         except:
        #             des = self.oracle.query_llm(query_template, [topic_title, ''])
        # return des
        # source_passages = self._get_relevant_example_passages_from_slug(slug)
        topic_object = self._get_topic_object(slug)
        topic_title = topic_object.get_primary_title()

        discard_attempts = [0, 1, 2]
        for discard_count in discard_attempts:
            try:
                source_passages = self._get_relevant_example_passages_from_slug(slug, discard_longest=discard_count)
                des = self.oracle.query_llm(query_template, [topic_title, source_passages])
                break  # Break the loop if successful
            except Exception as e:
                if discard_count == discard_attempts[-1]:
                    des = self.oracle.query_llm(query_template, [topic_title, ''])
        return des

    def generate_description(self, slug):
        description = self._try_to_get_description_based_on_sources(slug, self.description_generation_prompt_template)
        self.data_handler.set_description(slug, description)

    def generate_embedding(self, slug):
        description = self.data_handler.get_description(slug)
        if not description:
            raise ValueError(f"No description found for slug: {slug}")
        embedding = self.oracle.embed_text(description)
        self.data_handler.set_embedding(slug, embedding)

    def generate_description_and_embedding(self, slug):
        self.generate_description(slug)
        self.generate_embedding(slug)


class TopicsVectorSpace:
    embed_topic_template = (
        "You are a humanities scholar specializing in Judaism. Given a topic or a term, write a description for that topic from a Jewish perspective.\n"
        "Topic: {0}\nDescription:"
    )

    def __init__(self, data_handler: TopicsData, oracle: LLMOracle):
        slug_embeddings_dict = data_handler.get_slugs_and_embeddings_dict()
        for slug in slug_embeddings_dict.keys():
            slug_embeddings_dict[slug] = np.array(slug_embeddings_dict[slug])
        self.slug_embeddings_dict = slug_embeddings_dict
        self.oracle = oracle

    def _embed_and_get_embedding(self, template, topic):
        text = self.oracle.query_llm(template, [topic])
        embedding = np.array(self.oracle.embed_text(text))
        return np.array(embedding)
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

    def __init__(self, data_handler: TopicsData, oracle: LLMOracle):
        self.slug_descriptions_dict = data_handler.get_slugs_and_descriptions_dict()
        self.oracle = oracle

    def verify_topic(self, sefaria_slug, segment_text):
        description = self.slug_descriptions_dict[sefaria_slug]
        ver_response = self.oracle.query_llm(self.verifier_template, [segment_text, sefaria_slug, description]).replace('# Output',
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


    def __init__(self, topics_space: TopicsVectorSpace, verifier: TopicVerifier, oracle: LLMOracle):
        self.verifier = verifier
        self.topics_space = topics_space
        self.oracle = oracle


    def _get_inferred_topics(self, template, segment_text):
        answer = self.oracle.query_llm(template, [segment_text])
        return [topic.strip() for topic in answer.split("\n-")]

    def _load_existing_topics(self, jsonl_path):
        existing_topics_dicts = [json.loads(line) for line in
                                 open(jsonl_path, 'r')]
        for topic_dict in existing_topics_dicts:
            topic_dict["embedding"] = np.array(topic_dict["embedding"])
        return existing_topics_dicts

    def _embed_and_get_embedding(self, template, topic):
        text = self.oracle.query_llm((template, [topic]))
        embedding = self.oracle.embed_text(text)
        return np.array(embedding)

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
    oracle = LLMOracle()
    embedder = TopicsEmbedder(data_handler, oracle)
    embedder.generate_description_and_embedding("maharsha")

    toc = library.get_topic_toc_json_recursive()

    # verifier = TopicVerifier(data_handler, oracle)
    # topics_space = TopicsVectorSpace(data_handler, oracle)
    # tagger = TopicTagger(topics_space=topics_space, verifier=verifier, oracle=oracle)
    # tagger.tag_segment(Ref("Kohelet_Rabbah.1.7.1").text().text)



    print("bye")