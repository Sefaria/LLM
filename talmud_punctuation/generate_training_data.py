import django


django.setup()
superuser_id = 171118
# import statistics
import json
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation


masechtot_ordered = ["Berakhot", "Shabbat", "Eruvin", "Pesachim", "Rosh Hashanah", "Yoma", "Sukkah", "Beitzah",
                          "Taanit", "Megillah", "Moed Katan", "Chagigah",
                          "Yevamot", "Ketubot", "Nedarim", "Nazir", "Sotah", "Gittin", "Kiddushin",
                          "Bava Kamma", "Bava Metzia", "Bava Batra", "Sanhedrin", "Makkot", "Shevuot",
                          "Avodah Zarah", "Horayot",
                          "Zevachim", "Menachot", "Chullin", "Bekhorot", "Arakhin", "Temurah", "Keritot", "Meilah",
                          "Tamid", "Niddah"]

{"messages": [{"role": "system", "content": "<explain what the task is>"}, {"role": "user", "content": "steinsaltz and no-punctuation talmud"}, {"role": "assistant", "content": "punctuation talmud"}]}
examples = []
task_desciption = "Given this non-puncutated talmudic passage and a punctuated interpretation of this passage, output a punctuated version of the passage based on the interpretation I provided"
last_masechet = "Gittin"
def create_new_context(task_desciption, non_punctuated, steinsalz, punctuated):
    return (
    {"messages": [{"role": "system", "content": task_desciption},
                  {"role": "user", "content": {"no_punctuation_talmud": non_punctuated, "steinsaltz": steinsalz}},
                  {"role": "assistant", "content": punctuated}]}
    )


if __name__ == '__main__':

    for masechet in masechtot_ordered:
        print(masechet)
        all_segment_refs = Ref(masechet).all_segment_refs()
        for segment_ref in all_segment_refs:
            non_punctuated = segment_ref.text('he', "William Davidson Edition - Aramaic").text
            punctuated = strip_cantillation(segment_ref.text('he').text, strip_vowels=True)
            steinsalz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
            examples.append(create_new_context(task_desciption, non_punctuated, steinsalz, punctuated))
        if masechet == last_masechet:
            break
    with open("punctuation_examples.json", 'w') as f:
        json.dump(examples, f)
    print("end of script")