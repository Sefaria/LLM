import pytest
from app.util.sentencizer import _combine_small_sentences


@pytest.mark.parametrize(('sentences', 'expected_output'), [
    [['a b', 'c d'], ['a b', 'c d']],
    [['a', 'c d'], ['a c d']],
    [['a', 'c d', 'e'], ['a c d e']],
    [['a b', 'c', 'e'], ['a b c e']],
    [['a b', 'c', 'e f', 'g h'], ['a b c e f', 'g h']],
])
def test_combine_small_sentences(sentences, expected_output):
    actual_output = _combine_small_sentences(sentences, min_words=2)
    assert actual_output == expected_output
