import pytest
from linker.fine_tune.scripts import run_on_validation_set as r


@pytest.mark.parametrize(('original_text', 'doc', 'expected_doc'), (
    ['a test', r.EntityDoc('a, test', [r.Entity('test', 3, 7)]), r.EntityDoc('a test', [r.Entity('test', 2, 6)])],
    ['a, test', r.EntityDoc('a test', [r.Entity('test', 2, 6)]), r.EntityDoc('a, test', [r.Entity('test', 3, 7)])],
    ['a test.', r.EntityDoc('a test', [r.Entity('test', 2, 6)]), r.EntityDoc('a test.', [r.Entity('test', 2, 6)])],
))
def test_realign_entities(original_text, doc, expected_doc):
    new_doc = r.realign_entities(original_text, doc)
    new_doc.validate(original_text)
    assert new_doc == expected_doc
