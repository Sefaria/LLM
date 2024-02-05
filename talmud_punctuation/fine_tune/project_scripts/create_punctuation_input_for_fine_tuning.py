import django
django.setup()
import typer
import json, csv
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation, sanitize
import random
import re

seed_value = 613
random.seed(seed_value)
sample_size = 1000

train_proportion = 0.8

masechtot_ordered = ["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma", "Sukkah", "Beitzah",
                          "Taanit", "Megillah", "Moed Katan", "Chagigah",
                          "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
                          "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot",
                          "Avodah Zarah", "Horayot",
                          "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah",
                          "Tamid", "Niddah"]

task_desciption = 'Punctuate this Talmudic passage based on the commentary I provide. Extract the relevant punctuation marks (, : . ? ! "" ; —) from the commentary and put them in the original. Output only the original Aramaic passage with punctuation without "cantilation" or "nikud".\n'
last_masechet = "Chagigah"

def read_csv(file_path):
    csv_list = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_list = list(csv_reader)


    data = []
    keys = csv_list[0]
    for row in csv_list[1:]:
        new_dict = {}
        for key, row_value in zip(keys, row):
            new_dict[key] = row_value
        data.append(new_dict)
    return data

def create_new_context(task_desciption, non_punctuated, steinsaltz, punctuated):
    return (
    {"messages": [{"role": "system", "content": task_desciption},
                  {"role": "user", "content": f'Original Talmudic Passage:\n{non_punctuated}\n Commentary:\n{steinsaltz}'},
                  {"role": "assistant", "content": punctuated}]}
    )

def parse_manual_data(csv_filename="horayot_inferences_vocalized_for_train.csv"):
    rows = read_csv(csv_filename)
    samples = []
    for row in rows:
        if row["Inference Vocalized Corrected"] != row["Inference Vocalized"]:
            segment_ref = Ref(row['Ref'])
            non_punctuated = segment_ref.text('he', "William Davidson Edition - Aramaic").text
            punctuated = strip_cantillation(row["Inference Vocalized Corrected"], strip_vowels=True)
            steinsaltz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
            samples.append(create_new_context(task_desciption, non_punctuated, steinsaltz, punctuated))
    return samples


def merge_and_delete(list1, list2):
    # Calculate the number of elements to delete from list1
    delete_count = len(list2)

    # Randomly choose elements to delete from list1
    elements_to_delete = random.sample(range(len(list1)), delete_count)
    elements_to_delete.sort(reverse=True)

    # Merge list2 into list1
    for index in elements_to_delete:
        del list1[index]
    list1.extend(list2)

def create_data(output_training_filename: str, output_validation_filename: str, manual_data_csv_filename: str):
    all_samples = []
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
    for masechet in masechtot_ordered:
        print("creating data from Masechet " + masechet)
        all_segment_refs = Ref(masechet).all_segment_refs()
        for segment_ref in all_segment_refs:
            non_punctuated = segment_ref.text('he', "William Davidson Edition - Aramaic").text
            punctuated = strip_cantillation(segment_ref.text('he').text, strip_vowels=True)
            steinsaltz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
            all_samples.append(create_new_context(task_desciption, non_punctuated, steinsaltz, punctuated))
        if masechet == last_masechet:
            break

    #get only limited num of samples
    samples_trimmed = []
    samples_trimmed = random.sample(all_samples, sample_size)

    if (manual_data_csv_filename):
        manual_samples = parse_manual_data(manual_data_csv_filename)
        merge_and_delete(samples_trimmed, manual_samples)

    # Calculate the number of items for training
    num_train = int(len(samples_trimmed) * train_proportion)

    # Use random.sample to partition the list according to the seed
    train_samples = random.sample(samples_trimmed, num_train)
    validation_samples = [item for item in samples_trimmed if item not in train_samples]

    with open(output_training_filename, 'w', encoding='utf-8') as jsonl_file:
        for json_obj in train_samples:
            # Use ensure_ascii=False to encode Unicode characters
            json_line = json.dumps(json_obj, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')
    with open(output_validation_filename, 'w', encoding='utf-8') as jsonl_file:
        for json_obj in validation_samples:
            # Use ensure_ascii=False to encode Unicode characters
            json_line = json.dumps(json_obj, ensure_ascii=False)
            jsonl_file.write(json_line + '\n')


    print("TRAINING SAMPLES: "  + str(len(train_samples)))
    print("VALIDATION SAMPLES: " + str(len(validation_samples)))

if __name__ == '__main__':
    # typer.run(create_data)
    # create_data("output/gpt_punctuation_training.jsonl", "output/gpt_punctuation_validation.jsonl")
    parse_manual_data()