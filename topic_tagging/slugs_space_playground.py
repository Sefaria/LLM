import numpy as np
import os
import json
# from sklearn.feature_extraction.text import TfidfVectorizer
import math
from tqdm import tqdm


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

class TopicsVectorSpace:

    def __init__(self, data_handler: TopicsData):
        slug_embeddings_dict = data_handler.get_slugs_and_embeddings_dict()
        for slug in slug_embeddings_dict.keys():
            slug_embeddings_dict[slug] = np.array(slug_embeddings_dict[slug])
        self.slug_embeddings_dict = slug_embeddings_dict


    def _embedding_distance(self, embedding1, embedding2):
        return np.linalg.norm(embedding1 - embedding2)


    def get_nearest_k_slugs(self, slug, k: int, such_that_predicate=lambda x: True):
        # inferred_topic_embedding = self._embed_and_get_embedding(self.embed_topic_template, slug)
        slugs_embedding = self.slug_embeddings_dict[slug]
        sefaria_topic_embeddings_list = [(key, value) for key, value in self.slug_embeddings_dict.items()]
        sorted_data = sorted(sefaria_topic_embeddings_list,
                             key=lambda x: self._embedding_distance(x[1], slugs_embedding))

        sorted_data = [t for t in sorted_data if such_that_predicate(t[0])]
        k_neighbours = [t[0] for t in sorted_data[:k]]

        return k_neighbours


    def _get_all_parshiot_slugs(self):
        prefix = "parashat-"
        filtered_keys = [key for key in self.slug_embeddings_dict.keys() if key.startswith(prefix)]
        return filtered_keys
    def create_simulated_documents_tf_idf(self, num_of_nearest_slug_to_consider=10):
        parshiot_slugs = self._get_all_parshiot_slugs()
        parasha_documents_dict = {}
        # num_of_slugs = len(self.slug_embeddings_dict.keys())
        num_of_slugs = num_of_nearest_slug_to_consider
        for parasha in parshiot_slugs:
            sorted_slugs_for_parasha = self.get_nearest_k_slugs(parasha, num_of_slugs, such_that_predicate=lambda x: not x.startswith("parashat-"))
            document_simulation = []
            for index, slug in enumerate(sorted_slugs_for_parasha):
                document_simulation.append((slug,num_of_slugs-index))
            parasha_documents_dict[parasha] = document_simulation
        return parasha_documents_dict




    def calculate_tf_idf_for_parshiot(self):
        documents = self.create_simulated_documents_tf_idf()
        # Step 1: Calculate Term Frequencies (TF)
        term_frequencies = {}
        document_frequencies = {}

        for doc_name, word_count_list in documents.items():
            total_words_in_doc = sum(count for _, count in word_count_list)

            tf_values = {}
            for word, count in word_count_list:
                tf_values[word] = count / total_words_in_doc

                # Document Frequencies (DF)
                if word in document_frequencies:
                    document_frequencies[word] += 1
                else:
                    document_frequencies[word] = 1

            term_frequencies[doc_name] = tf_values

        # Step 2: Calculate Inverse Document Frequencies (IDF)
        num_documents = len(documents)
        # idf_values = {word: math.log(num_documents / (df + 1)) + 1 for word, df in document_frequencies.items()}
        idf_values = {word: math.log(num_documents / df) for word, df in document_frequencies.items()}

        # Step 3: Calculate TF-IDF
        tf_idf_scores = {}

        for doc_name, tf_values in term_frequencies.items():
            tf_idf_values = {word: tf * idf_values[word] for word, tf in tf_values.items()}
            tf_idf_scores[doc_name] = tf_idf_values

        return tf_idf_scores

    def sort_parshiot_by_tf_idf(self):
        tf_idf_scores = self.calculate_tf_idf_for_parshiot()
        sorted_words_dict = {}
        for doc_name, tf_idf_values in tf_idf_scores.items():
            sorted_words = sorted(tf_idf_values, key=tf_idf_values.get, reverse=False)
            sorted_words_dict[doc_name] = sorted_words

        return sorted_words_dict

    def get_k_nearest_tf_idf_slugs_for_parasha(self, parasha, k):
        sorted_parashiot_dict = self.sort_parshiot_by_tf_idf()
        return sorted_parashiot_dict[parasha][:k]


slugs = [
'elul',
'teshuvah',
'selichot',
'shofar',
'high-holidays',
'rosh-hashanah',
'yom-kippur',
'sukkot',
'simchat-torah',
'the-four-species',
'rain',
'pomegranates',
'bar-mitzvah',
'bat-mitzvah',
'circumcision',
'marriage',
'sheva-brachot',
'mikvah',
'honoring-parents',
'health',
'mother',
'friendship',
'nature',
'prayer',
'tzitzit',
'tefillin',
'challah',
'rosh-chodesh',
'ketubah',
'kashrut',
'torah',
'shema',
'strength',
'13-principles-of-faith',
'creation',
'the-tree-of-knowledge',
'gan-eden',
'honi-hamagel',
'sigd',
'noah',
'tower-of-babel',
'rainbows',
'abraham',
'covenants',
'sarah',
'binding-of-isaac',
'dreams',
'joseph',
'judah',
'chanukkah',
'light',
'menorah',
'hasmoneans',
'Maccabees',
'moses',
'the-song-of-the-sea',
'the-ten-commandments',
'the-tablets',
'the-ten-plagues',
'yitro',
'justice',
'mishkan',
'tenth-of-tevet',
'tu-bshvat',
'amalek',
'leap-year',
'joy',
'purim',
'the-scroll-of-esther',
'mishloach-manot',
'esther',
'elijah',
'pirkei-avot',
'seven-species',
'converts',
'rabbi-meir',
'bar-kochba',
'shimon-bar-yochai',
'the-17th-of-tammuz',
'ushpizin',
'hoshana raba',
'shemini-atzeret',
'song-of-songs',
'ecclesiastes',
'tzom-gedaliah',
'taanit esther',
'mordechai',
'lashon-hara',
'isaac',
'jacob',
'rachel',
'leah',
'noah and the ark',
'hallel',
'maarat-hamachpelah',
'tishrei',
'chesvan',
'kislev',
'tevet',
'shevat',
'adar',
'nissan',
'iyar',
'sivan',
'tamuz',
'av',
'birkat-hamazon',
'tashlich',
'hatarat-nedarim',
'shabbat-candles',
'king-david',
'king-solomon',
'conversion',
'yahrzeit',
'mourning',
'sacrifices',
'kiddush',
'havdalah',
'blessings',
'birkat-kohanim',
'high-priests',
'acacia-trees',
'lecha-dodi',
'haftarah',
'motzi המוציא',
'mishnah',
'talmud',
'midrash',
'tosafot',
'violence',
'peace',
'priestly-garments',
'prophecy',
'shemoneh-esrei',
'leadership',
'visiting-the-sick',
'adam-and-eve',
'dinah',
'shacharit',
'plague-of-blood',
'plague-of-the-firstborn',
'splitting-of-the-red-sea',
'tallit',
'beit-hillel',
'beit shamai',
'al-chet',
'aaron',
'mezuzah',
'forgiveness-(מחילה)',
'kol-nidre',
'the-four-parshiot',
'bezalel',
'divine-names',
'ishmael',
'oils1',
'pidyon-haben',
]

if __name__ == '__main__':
    from pprint import pprint
    import csv
    print("Hi")
    data_handler = TopicsData("embedding_all_toc.jsonl")
    vector_space = TopicsVectorSpace(data_handler)
    vector_space.get_nearest_k_slugs("rosh-hashanah", 10)
    slugXneighbours = []
    for slug in slugs:
        try:
            neighbours = vector_space.get_nearest_k_slugs(slug, 10)
            neighbours = ', '.join(neighbours)
            slugXneighbours += [(slug, neighbours)]
            # pprint(vector_space.get_nearest_k_slugs(slugs, 10))
        except Exception as e:
            print(slug)
    with open("slugs_and_neighbours.csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header if needed
        # csv_writer.writerow(['ID', 'Name', 'Age'])
        csv_writer.writerows(slugXneighbours)





