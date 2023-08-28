import django
django.setup()
from sefaria.model import *
import csv
from get_normalizer import get_normalizer


if __name__ == '__main__':
    lang = 'en'
    normalizer = get_normalizer()
    with open("/Users/nss/Downloads/all_topic_prompts.csv", "w") as fout:
        cout = csv.DictWriter(fout, ["Topic Name", "Reference", "Title", "Prompt", "Source Hebrew", "Source English"])
        cout.writeheader()
        for rtl in RefTopicLinkSet({f"descriptions.{lang}": {"$exists": True}}):
            oref = Ref(rtl.ref)
            he = oref.text("he").ja().flatten_to_string()
            en = oref.text("en").ja().flatten_to_string()
            topic = Topic.init(rtl.toTopic)
            cout.writerow({
                "Topic Name": topic.get_primary_title("en"),
                "Reference": rtl.ref,
                "Title": rtl.descriptions[lang]['title'],
                "Prompt": rtl.descriptions[lang]['prompt'],
                "Source Hebrew": normalizer.normalize(he),
                "Source English": normalizer.normalize(en),
            })
