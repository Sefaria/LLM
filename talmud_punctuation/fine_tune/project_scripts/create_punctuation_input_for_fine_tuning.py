import django
django.setup()
import typer
import json
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation, sanitize
import random
import re

seed_value = 613
random.seed(seed_value)
sample_size = 500

train_proportion = 0.8

masechtot_ordered = ["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma", "Sukkah", "Beitzah",
                          "Taanit", "Megillah", "Moed Katan", "Chagigah",
                          "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
                          "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot",
                          "Avodah Zarah", "Horayot",
                          "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah",
                          "Tamid", "Niddah"]

task_desciption = "Given an unpunctuated passage of the Talmud, output a punctuated version of the passage:"
last_masechet = "Gittin"


def create_new_context(task_desciption, non_punctuated, punctuated):
    return (
    {"messages": [{"role": "system", "content": task_desciption},
                  {"role": "user", "content": f'{non_punctuated}'},
                  {"role": "assistant", "content": punctuated}]}
    )

def create_data(output_training_filename: str, output_validation_filename: str):
    all_samples = []
    punctuationre = re.compile(
        r'[\.\!\?\:\,\u05F4]+(?![\u0591-\u05bd\u05bf-\u05c5\u05c7\u200d\u05d0-\u05eA](?:[\.\!\?\:\,\u05F4\s]|$))|—\s')
    for masechet in masechtot_ordered:
        print("creating data from Masechet " + masechet)
        all_segment_refs = Ref(masechet).all_segment_refs()
        for segment_ref in all_segment_refs:
            non_punctuated = segment_ref.text('he', "William Davidson Edition - Vocalized Aramaic").text
            non_punctuated = punctuationre.sub('', non_punctuated)
            punctuated = segment_ref.text('he').text
            # steinsalz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
            all_samples.append(create_new_context(task_desciption, non_punctuated, punctuated))
        if masechet == last_masechet:
            break

    #get only limited num of samples
    samples_trimmed = []
    samples_trimmed = random.sample(all_samples, sample_size)

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
    typer.run(create_data)