import django
django.setup()
from sefaria.model.text import Ref
from typing import Any, Union, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

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



def run_parallel(items: list[Any], unit_func: Callable, max_workers: int, **tqdm_kwargs) -> list:
    def _pbar_wrapper(pbar, item):
        unit = unit_func(item)
        with pbar.get_lock():
            pbar.update(1)
        return unit


    with tqdm(total=len(items), **tqdm_kwargs) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in items:
                futures.append(executor.submit(_pbar_wrapper, pbar, item))

    output = [future.result() for future in futures if future.result() is not None]
    return output
