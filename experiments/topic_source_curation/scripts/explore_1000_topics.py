import django
django.setup()
from sefaria.model import *
from experiments.topic_source_curation.curator import get_topics_to_curate
from collections import defaultdict


if __name__ == '__main__':
    counts = []
    yo = defaultdict(int)
    topics = get_topics_to_curate()
    for topic in topics:
        yo[topic.slug] += 1
        t = Topic.init(topic.slug)
        assert isinstance(t, Topic)
        counts += [(t.slug, t.numSources)]
    counts.sort(key=lambda x: x[1], reverse=True)
    for t, c in counts:
        print(t, c)
    for slug, num in yo.items():
        if num > 1:
            print(slug, num)


