from srsly import read_jsonl, write_jsonl
from collections import defaultdict
import random

random.seed(61345)

export_filename = "/Users/nss/Downloads/en_library.jsonl"

def get_export():
    return read_jsonl(export_filename)

def calculate_size_of_sample(docs):
    size = 0
    for doc in docs:
        size += len(doc['text'])
    return size

def sample_to_size(export, target_size):
    sample_num = len(export) // 2
    jump = sample_num
    sample = export
    while jump > 1:
        sample = random.sample(export, sample_num)
        sample_size = calculate_size_of_sample(sample)
        if sample_size > target_size:
            sample_num -= (jump // 2)
        elif sample_size < target_size:
            sample_num += (jump // 2)
        else:
            # exactly somehow
            return sample
        jump //= 2
    return sample


def calc_stats(export):
    cat_counts = defaultdict(int)
    for doc in export:
        cat_counts[doc['metadata']['docCategory']] += 1
    return sorted(cat_counts.items(), key=lambda x: x[0], reverse=True)


if __name__ == '__main__':
    target_size = 1e6
    yo_export = list(get_export())
    export_stats = calc_stats(yo_export)
    yo = sample_to_size(yo_export, target_size)
    print('-----')
    sample_stats = calc_stats(yo)
    print(len(yo))
    print(calculate_size_of_sample(yo))
    for (ecat, ecount), (scat, scount) in zip(export_stats, sample_stats):
        eperc = round(ecount / len(yo_export) * 100, 2)
        sperc = round(scount / len(yo) * 100, 2)
        print(ecat, ecount, scount)
        print(ecat, eperc, sperc)
    write_jsonl(f'output/sample_for_embedding_{target_size}.jsonl', [d['text'] for d in yo])
