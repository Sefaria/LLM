import json
import glob
import os
import re

def load_chosen_sources(slug):
    with open(f"output/curation_{slug}.json") as fin:
        return json.load(fin)


def load_all_chosen_sources():
    file_pattern = os.path.join("output", 'curation_*')
    slugs = [re.search(r"curation_(.*)\.json", filename).group(1) for filename in glob.glob(file_pattern)]
    for slug in slugs:
        yield load_chosen_sources(slug), slug


if __name__ == '__main__':
    for sources, slug in load_all_chosen_sources():
        for source in sources:
            if "Tze'enah URe'enah" in source['ref']:
                print(slug)

"""
Daf Shevui
rabbi-yehoshua-b-levi
sin-of-peor
kal-vechomer
david-and-the-temple
the-four-parshiot
the-curtain
malachi
etrog
ecclesiastes
mezuzah
mezuzah
cup-for-the-blessing
abaye
zimri
judahs-blessing
aarons-sin
rabbi-eliezer-b-hyrcanus
safek
prophetess
ulla
machloket
mice
cedars
rav-zera
welcoming
moses-birth
willows
forgiveness-(מחילה)
thoughts
air
slaves
asa
sukkah
zachor
creation-of-light-and-luminaires
oxen
rome
society
hezekiah
music
jacobs-oath
winds
partitions
marta-the-daughter-of-boethus
horses
zechariah
social-justice
belshatzar
tamar
minyan
minyan
diaspora
abigail
listening
identity
fresh-grain
alcohol
ahasuerus
boaz
consent
priestly-blessing
"""