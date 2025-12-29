import django
from tqdm import tqdm

django.setup()

from sefaria.model.text import Ref

from utils import links_collection

# Toggle to use remote linker DB via SSH tunnel
USE_REMOTE = False


def run(use_remote: bool = False):
    num_book_links = 0
    num_non_segment_links = 0
    query = {"generated_by": "add_links_from_text"}
    with links_collection(use_remote=use_remote) as links:
        count = links.count_documents(query)
        for link in tqdm(links.find(query), total=count):
            for tref in link.get('refs', []):
                try:
                    oref = Ref(tref)
                except Exception:
                    continue
                if oref.is_book_level():
                    num_book_links += 1
                elif not oref.is_segment_level():
                    num_non_segment_links += 1
    print("Num book level links:", num_book_links)
    print("Num non-segment level links:", num_non_segment_links)

if __name__ == '__main__':
    run(use_remote=USE_REMOTE)
