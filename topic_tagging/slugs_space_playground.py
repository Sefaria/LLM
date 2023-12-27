import numpy as np
import os
import json
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


if __name__ == '__main__':
    print("Hi")
    data_handler = TopicsData("embedding_all_toc.jsonl")
    vector_space = TopicsVectorSpace(data_handler)
    neighbours = vector_space.get_nearest_k_slugs("parashat-vayechi", 10, such_that_predicate=lambda x: not x.startswith("parashat-"))
    print(neighbours)