import django
django.setup()
from sefaria.model.text import Ref


def filter_invalid_refs(trefs, key=None):
    key = key or (lambda x: x)
    out = []
    for tref in trefs:
        try:
            Ref(key(tref))
        except:
            continue
        out += [tref]
    return out
