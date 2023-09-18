import django
django.setup()
import typer
import json
import srsly
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation
import random


masechtot_ordered = ["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma", "Sukkah", "Beitzah",
                          "Taanit", "Megillah", "Moed Katan", "Chagigah",
                          "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
                          "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot",
                          "Avodah Zarah", "Horayot",
                          "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah",
                          "Tamid", "Niddah"]

training_examples = []
validation_examples = []
task_desciption = "Given a pair ('no_punctuation_talmud': <text>, 'steinsaltz': <text>), where 'no_punctuation_talmud' is a passage from the Talmud without punctuation marks and 'steinsaltz' is a passage that contains the Talmudic text, expands on it, and interprets it, output a punctuated version of the Talmudic passage using the punctuation that appears in the 'steinsaltz' interpretation."
last_masechet = "Gittin"

def true_with_probability(p):
    return random.random() < p
def create_new_context(task_desciption, non_punctuated, steinsalz, punctuated):
    return (
    {"messages": [{"role": "system", "content": task_desciption},
                  {"role": "user", "content": f'("no_punctuation_talmud": {non_punctuated}, "steinsaltz": {steinsalz})'},
                  {"role": "assistant", "content": punctuated}]}
    )

def create_data(output_training_filename: str, output_validation_filename: str):
    for masechet in masechtot_ordered:
        print("creating data from Masechet " + masechet)
        all_segment_refs = Ref(masechet).all_segment_refs()
        for segment_ref in all_segment_refs:
            non_punctuated = segment_ref.text('he', "William Davidson Edition - Aramaic").text
            punctuated = strip_cantillation(segment_ref.text('he').text, strip_vowels=True)
            steinsalz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
            if true_with_probability(0.8):
                training_examples.append(create_new_context(task_desciption, non_punctuated, steinsalz, punctuated))
            else:
                validation_examples.append(create_new_context(task_desciption, non_punctuated, steinsalz, punctuated))
        if masechet == last_masechet:
            break

    srsly.write_jsonl(output_training_filename, training_examples)
    srsly.write_jsonl(output_validation_filename, validation_examples)
    print("TRAINING SAMPLES: "  + str(len(training_examples)))
    print("VALIDATION SAMPLES: " + str(len(validation_examples)))

if __name__ == '__main__':
    typer.run(create_data)