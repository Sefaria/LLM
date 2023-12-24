import django
django.setup()
from sefaria.model import *
import os
import random
from tqdm import tqdm
random.seed(613)

def get_all_refs_from_category(topic_category):
    source_refs = []
    sub_topics = topic_category.topics_by_link_type_recursively(linkType='displays-under')
    for sub_topic in sub_topics:
        sources = [*sub_topic.link_set(_class="refTopic").array()]
        sources = [source for source in sources if not source.ref.startswith("Sheet")]
        source_refs += [source.ref for source in sources]
    return source_refs

def create_quintessence_of_texts_by_topic_noahs_method(sample_size = 5000):
    top_level_categories = TopicSet({"isTopLevelDisplay": True})
    cat_sample_size = int(sample_size/len(top_level_categories))
    sample = {}
    for topic_category in top_level_categories:
        cat_refs = get_all_refs_from_category(topic_category)
        sample[topic_category.slug] = random.sample(cat_refs, cat_sample_size)
    return sample

def get_all_seg_refs_from_library_category(category: str):
    all_refs = []
    indexes = library.get_indexes_in_category(category)
    for index in tqdm(indexes, desc=f"Processing Indexes for Category: {category}", unit="index"):
        try:
            all_refs += library.get_index(index).all_segment_refs()
        except Exception as e:
            print(f"Problem with index {index}, Exception: {e}")
    return all_refs
def create_quintessence_of_texts_by_library_noahs_method(sample_size = 5000) -> Dictionary:
    sample = {}
    top_level_categories = library.get_top_categories()
    cat_sample_size = int(sample_size / len(top_level_categories))
    for category in top_level_categories:
        cat_refs = get_all_seg_refs_from_library_category(category)
        sample[category] = random.sample(cat_refs, cat_sample_size)
    return sample


def dict_of_lists_to_csv(file_path, data):
    import csv
    # Extract the keys (header) and values (columns) from the dictionary
    headers = list(data.keys())
    columns = list(data.values())

    # Transpose the columns to create rows
    rows = zip(*columns)

    # Write to CSV file
    with open(file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write the header
        csv_writer.writerow(headers)

        # Write the rows
        csv_writer.writerows(rows)
def create_quintessence_of_texts_by_library_noahs_method_to_csv():
    samples = create_quintessence_of_texts_by_library_noahs_method()
    dict_of_lists_to_csv("quintessence_of_library_noah_method.csv", samples)

def create_quintessence_of_texts_by_topics_noahs_method_to_csv():
    samples = create_quintessence_of_texts_by_topic_noahs_method()
    dict_of_lists_to_csv("quintessence_of_topics_noah_method.csv", samples)

if __name__ == "__main__":
    create_quintessence_of_texts_by_topics_noahs_method_to_csv()

