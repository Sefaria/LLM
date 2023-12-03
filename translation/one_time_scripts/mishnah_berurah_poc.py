from typing import List
import django
django.setup()
from sefaria.model import *
from translation.poc import translate_segment
from util.general import get_raw_ref_text, normalizer
import random
from tqdm import tqdm
import csv
from functools import partial


def choose_sections(n) -> List[Ref]:
    sa = library.get_index("Shulchan Arukh, Orach Chayim")
    assert isinstance(sa, Index)
    alt = sa.get_alt_structure("Topic")
    sections = []
    for child in alt.children:
        refs = Ref(child.wholeRef).split_spanning_ref()
        sections += [random.choice(refs)]
    chosen_sections = random.sample(sections, n)
    return [Ref(f"Mishnah Berurah {sa_ref.sections[0]}") for sa_ref in chosen_sections]


def translate_section(oref, se_he_list):
    for segment_oref, sa_text in zip(oref.all_segment_refs(), se_he_list):
        yield translate_segment(segment_oref.normal(), context=sa_text)


def get_section_text(oref, lang):
    for segment_oref in oref.all_segment_refs():
        yield normalizer.normalize(get_raw_ref_text(segment_oref, lang))


def get_sa_section_text(mb_oref, lang):
    section_num = mb_oref.sections[0]
    sa_oref = Ref(f"Shulchan Arukh, Orach Chayim {section_num}")
    for mb_seg in mb_oref.all_segment_refs():
        sa_segs = []
        for l in mb_seg.linkset():
            opposite = l.ref_opposite(mb_oref)
            if opposite and sa_oref.contains(opposite):
                sa_segs += [opposite]
        assert len(sa_segs) == 1
        sa_seg = sa_segs[0]
        yield normalizer.normalize(get_raw_ref_text(sa_seg, lang))


def get_csv_rows(get_segs, get_context_seg=None, context_lang='he'):
    segs = get_segs()
    context_segs = [get_context_seg(seg) for seg in segs] if get_context_seg else [None]*len(segs)
    rows = []
    for seg, context_seg in tqdm(zip(segs, context_segs), total=len(segs)):
        seg_he = get_raw_ref_text(seg, 'he')
        context_text = get_raw_ref_text(context_seg, context_lang) if context_seg else None
        seg_en = translate_segment(seg.normal(), context_text)
        rows += [{
            "He": seg_he,
            "En": seg_en,
            "Context": context_text,
            "Ref": seg.normal(),
        }]
    return rows


def get_random_segs_from_book(title, n) -> List[Ref]:
    index = library.get_index(title)
    return random.sample(index.all_segment_refs(), n)


def get_segs_from_ref(ref: Ref):
    return ref.all_segment_refs()


if __name__ == '__main__':
    tref = "Iggerot HaRambam, Maamar Tekhiyat HaMetim"
    rows = get_csv_rows(partial(get_segs_from_ref, Ref(tref)))
    with open(f"../output/{tref}_poc.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['Ref', 'He', 'En', 'Context'])
        cout.writeheader()
        cout.writerows(rows)


