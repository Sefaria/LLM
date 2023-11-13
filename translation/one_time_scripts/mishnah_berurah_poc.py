from typing import List
import django
django.setup()
from sefaria.model import *
from translation.poc import translate_segment
from util.general import get_raw_ref_text, normalizer
import random
from tqdm import tqdm
import csv


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


def get_csv_rows():
    mb_sections = choose_sections(10)
    rows = []
    for mb_section in tqdm(mb_sections):
        mb_he_list = list(get_section_text(mb_section, 'he'))
        sa_he_list = list(get_sa_section_text(mb_section, 'he'))
        mb_en_list = list(translate_section(mb_section, sa_he_list))
        mb_segments = mb_section.all_segment_refs()
        for mb_he, mb_en, sa_he, mb_seg in zip(mb_he_list, mb_en_list, sa_he_list, mb_segments):
            rows += [{
                "MB He": mb_he,
                "MB En": mb_en,
                "SA He": sa_he,
                "MB Ref": mb_seg.normal(),
            }]
    return rows


if __name__ == '__main__':
    rows = get_csv_rows()
    with open("../output/mb_poc.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['MB Ref', 'MB He', 'MB En', 'SA He'])
        cout.writeheader()
        cout.writerows(rows)


