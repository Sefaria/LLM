import pytest
import util.general as g


@pytest.mark.parametrize(('orig', 'new', 'expected_removal_list'), [
    ('a test', ' a test', [((0, 0), ' ')]),  # insertion at beginning
    ('a test', 'a test ', [((6, 6), ' ')]),  # insertion at end
    (' a test', 'a test', [((0, 1), '')]),  # deletion at beginning
    ('a test ', 'a test', [((6, 7), '')]),  # deletion at end
    #('a test', 'a t3st', []),  # replacement
])
def test_get_removal_list(orig, new, expected_removal_list):
    actual_removal_list = g.get_removal_list(orig, new)
    assert actual_removal_list == expected_removal_list
