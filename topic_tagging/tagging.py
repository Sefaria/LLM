import django
django.setup()
import csv
import json
import re
from sefaria.model import *
from util.general import get_raw_ref_text, get_by_xml_tag, normalizer
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.chains import SimpleSequentialChain
import  numpy as np
import random
random.seed(613)


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
        self._check_file_existence()

    def _check_file_existence(self):
        if not os.path.exists(self.data_jsonl_filename):
            raise FileNotFoundError(f"The file {self.data_jsonl_filename} does not exist.")
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

    def generate_description_and_embedding_idempotent(self, slug):
        if self.data_handler.get_description(slug) and self.data_handler.get_embedding(slug):
            return
        if self.data_handler.get_description(slug):
            self.generate_embedding(slug)
        else:
            self.generate_description_and_embedding(slug)


class TopicsVectorSpace:
    embed_topic_template = (
        "You are a humanities scholar specializing in Judaism."
        "Given a topic or a term, write a description for that topic from a Jewish perspective."
        "Context may be provided in a Context text which belongs to Topic. Use Context to understand the general Topic better.\n"
        "Context: {0}"
        "Topic: {1}\nDescription:"
    )

    def __init__(self, data_handler: TopicsData, oracle: LLMOracle):
        slug_embeddings_dict = data_handler.get_slugs_and_embeddings_dict()
        for slug in slug_embeddings_dict.keys():
            slug_embeddings_dict[slug] = np.array(slug_embeddings_dict[slug])
        self.slug_embeddings_dict = slug_embeddings_dict
        self.oracle = oracle

    def _embed_and_get_embedding(self, template, topic, context=''):
        text = self.oracle.query_llm(template, [context, topic])
        embedding = np.array(self.oracle.embed_text(text))
        return np.array(embedding)
    def _embedding_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)
    def get_nearest_nearest_slug_from_arbitrary(self, topic_name):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, topic_name)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slug = sorted_data[0][0]
        return sefaria_slug
    def get_all_slugs_sorted_by_distance_from_arbitrary(self, topic_name, context_segment):
        inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, topic_name, context_segment)
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], inferred_topic_embedding))
        sefaria_slugs = [t[0] for t in sorted_data]
        return sefaria_slugs
    def get_nearest_k_slugs(self, slug, k: int):
        # inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, slug)
        slugs_embedding = self.slug_embeddings_dict[slug]
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], slugs_embedding))
        k_neighbours = [t[0] for t in sorted_data[:k]]
        return k_neighbours

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

    def _extract_text_to_list(self, text):
        # Split the text into lines
        lines = text.split('\n')

        # Remove any empty lines
        lines = [line.strip() for line in lines if line.strip()]

        # Check if lines are numbered or not
        if re.match(r'^\d+\.', lines[0]):
            # Extract numbered items
            items = [re.sub(r'^\d+\.\s*', '', line) for line in lines]
        else:
            # Extract non-numbered items
            items = [line.lstrip('- ').lstrip() for line in lines]

        return items

    def _get_inferred_topics(self, template, segment_text):
        answer = self.oracle.query_llm(template, [segment_text])
        # answer = answer.lstrip('- ').lstrip()
        # return [topic.strip() for topic in answer.split("\n-")]
        answer = self._extract_text_to_list(answer)
        return answer


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

    def _translate_ref(self, tref: str, context: str = None):
        oref = Ref(tref)
        text = get_raw_ref_text(oref, 'he')
        identity_message = HumanMessage(
            content="You are a Jewish scholar knowledgeable in all Torah and Jewish texts. Your "
                    "task is to translate the Hebrew text wrapped in <input> tags. Context may be "
                    "provided in <context> tags. Use context to provide context to <input> "
                    "text. Don't translate <context>. Only translate <input> text. Output "
                    "translation wrapped in <translation> tags.")
        task_prompt = f"<input>{text}</input>"
        if context:
            task_prompt = f"<context>{context}</context>{task_prompt}"
        task_message = HumanMessage(content=task_prompt)
        llm = ChatAnthropic(model="claude-2", temperature=0, max_tokens_to_sample=1000000)
        response_message = llm([identity_message, task_message])
        translation = get_by_xml_tag(response_message.content, 'translation')
        if translation is None:
            print("TRANSLATION FAILED")
            print(tref)
            print(response_message.content)
            return response_message.content
        return translation

    def _concatenate_strings(self, input_data):
        if isinstance(input_data, list):
            concatenated_string = ' '.join(input_data)
        elif isinstance(input_data, str):
            concatenated_string = input_data
        else:
            # Handle other cases or raise an exception as needed
            raise ValueError("Input must be a list of strings or a single string")

        return concatenated_string

    def tag_ref(self, tref):
        english_text = Ref(tref).text(lang="en").as_string()
        english_text = normalizer.normalize(english_text)
        if english_text == '':
            english_text = self._translate_ref(tref)
        return self.tag_segment(english_text)
    def tag_segment(self, segment_text, number_of_rejects_till_break=1):
        model_topics = self._get_inferred_topics(self.infer_topics_template, segment_text)
        verified_slugs = set()
        rejected_slugs = set()

        for inferred_topic in model_topics:
            sefaria_slugs = self.topics_space.get_all_slugs_sorted_by_distance_from_arbitrary(inferred_topic, segment_text)
            num_of_rejects = 0
            for sefaria_slug in sefaria_slugs:
                if num_of_rejects >= number_of_rejects_till_break:
                    break
                if sefaria_slug in rejected_slugs:
                    num_of_rejects += 1
                    continue
                if sefaria_slug in verified_slugs:
                    continue
                verified = self.verifier.verify_topic(sefaria_slug, segment_text)
                # print(f"LLM Tag: {inferred_topic}")
                # print(f"Sefaria Nearest Slug: {sefaria_slug}")
                if verified:
                    # print("https://www.sefaria.org/topics/" + sefaria_slug)
                    # print("LLM Verification: Accept")
                    verified_slugs.add(sefaria_slug)
                else:
                    num_of_rejects += 1
                    rejected_slugs.add(sefaria_slug)
                    # print("LLM Verification: Reject")

        return model_topics, set(verified_slugs)

def get_slug_values(data):
    slug_values = []

    if isinstance(data, list):
        for item in data:
            slug_values.extend(get_slug_values(item))
    elif isinstance(data, dict):
        if "slug" in data:
            slug_values.append(data["slug"])
        for key, value in data.items():
            slug_values.extend(get_slug_values(value))

    return slug_values
def get_values_by_key(data, key):
    if isinstance(data, list):
        # If it's a list, iterate through its elements
        return [value for sub_data in data for value in get_values_by_key(sub_data, key)]
    elif isinstance(data, dict):
        # If it's a dictionary, check if the key exists and collect its value
        values = [data[key]] if key in data else []
        # Recursively call the function on dictionary values
        for sub_data in data.values():
            values.extend(get_values_by_key(sub_data, key))
        return values
    else:
        # If it's neither a list nor a dictionary, return an empty list
        return []
def embed_toc():
    from tqdm import tqdm
    data_handler = TopicsData("embedding_all_toc.jsonl")
    oracle = LLMOracle()
    embedder = TopicsEmbedder(data_handler, oracle)
    toc = library.get_topic_toc_json_recursive()
    all_slugs = get_values_by_key(toc, "slug")

    for slug in tqdm(all_slugs, desc="Embedding Slugs", unit="slug"):
        try:
            embedder.generate_description_and_embedding_idempotent(slug)
        except Exception as e:
            print(f"Failed embedding slug {slug}, exception: {e}")


def load_refs_from_csv(file_name):
    slugs = []
    with open(file_name, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            row[0] = row[0].rstrip("<d>")
            slugs += row
    return slugs

def tag_sample_refs(source_csv="refs_sample.csv", dest_jsonl="sample_refs_tagged.jsonl"):
    from tqdm import tqdm


    def ref_exists(file_path, ref):
        # Check if the Ref already exists in the file
        with open(file_path, 'r') as jsonl_file:
            for line in jsonl_file:
                record = json.loads(line)
                if record['ref'] == ref:
                    return True
        return False



    trefs = load_refs_from_csv(source_csv)
    data_handler = TopicsData("embedding_all_toc.jsonl")
    oracle = LLMOracle()
    vector_space = TopicsVectorSpace(data_handler, oracle)
    verifier = TopicVerifier(data_handler, oracle)
    tagger = TopicTagger(vector_space, verifier, oracle)

    fieldnames = ['ref', 'original_unverified_slugs', 'slugs']

    # Check if destination file exists
    if os.path.exists(dest_jsonl):
        # If it exists, open in append mode
        mode = 'a'
    else:
        # If it doesn't exist, create a new file
        mode = 'w'


    with open(dest_jsonl, mode, newline='') as jsonl_file:
        for tref in tqdm(trefs, desc="Tagging Refs", unit="Ref"):
            # Check if the Ref already exists in the file
            if not ref_exists(dest_jsonl, tref):
                try:
                    model_topics, verified_slugs = tagger.tag_ref(tref)
                    record = {
                        fieldnames[0]: tref,
                        fieldnames[1]: list(model_topics),
                        fieldnames[2]: list(verified_slugs)
                    }
                    jsonl_file.write(json.dumps(record) + '\n')
                except Exception as e:
                    print(f"Problem with ref: {tref}, Exception: {e}")



def make_stream_file_for_prodigy(source_csv_file="sample_refs_tagged.csv", target_jsonl_file="tagging_data_for_prodigy.jsonl"):
    file_to_list_of_tuples = lambda path: [(row[0], row[2].
                                            replace("slichot", "selichot").
                                            replace("tablets","the-tablets").
                                            replace("bezalel-ben-abraham-ashkenazi", "betzalel-ben-abraham-ashkenazi").
                                            replace("judah-hehasid", "yehuda-hechasid").
                                            replace("alshikh", "moses-alshikh").
                                            replace("moses-and-the-the-tablets", "moses-and-the-tablets").
                                            replace("chanukah","chanukkah").
                                            replace("moshe-cordovero","moses-cordovero").
                                            rstrip().split(", ")) for row in csv.reader(open(path, 'r')) if len(row) >= 3][1:]
    def create_text_dict(ref, slugs_list):
        titles_list = [TopicSet({"slug": slug}).array()[0].get_primary_title() for slug in slugs_list]
        topic_dicts = []
        for slug, title in zip(slugs_list, titles_list):
            topic_dicts.append({"slug":slug, "title":title})
        return {"ref": ref, "hebrew_text": Ref(ref).text(lang="he").as_string(), "english_text":Ref(ref).text(lang="en").as_string(), "topics": topic_dicts}

    result = []
    ref_X_slugs = file_to_list_of_tuples(source_csv_file)
    for ref, slugs in ref_X_slugs:
        result.append(create_text_dict(ref, slugs))
    with open(target_jsonl_file, 'w') as jsonl_file:
        for item in result:
            json_line = json.dumps(item, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


if __name__ == '__main__':
    print("Hi")


    # tree = generate_html_tree(toc)
    # create_html_page(tree)
    # make_stream_file_for_prodigy()
    tag_sample_refs()

    print("bye")