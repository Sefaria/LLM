"""
Pytest test suite for LLM Segment Resolver using CSV test cases.

This test reads test cases from a CSV file and validates that the resolver
produces the expected results for each case.
"""
import os
import csv
import pytest
from typing import List, Dict, Any
from llm_segment_resolver import LLMSegmentResolver


def load_test_cases_from_csv(csv_filename: str) -> List[Dict[str, Any]]:
    """Load test cases from CSV file."""
    if not os.path.exists(csv_filename):
        return []

    test_cases = []
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_cases.append(row)

    return test_cases


def create_mock_chunk(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Create a chunk dict from test case data."""
    # Determine language priority
    lang = test_case.get('citing_language', 'en')

    return {
        'ref': test_case.get('citing_ref', ''),
        'he': test_case.get('citing_text', '') if lang == 'he' else '',
        'en': test_case.get('citing_text', '') if lang == 'en' else '',
        'language': lang,
        'versionTitle': test_case.get('citing_version_title', ''),
        'spans': []  # Will be populated below
    }


def create_mock_link(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Create a link dict from test case data."""
    original_refs = test_case.get('original_link_refs', '').split('|')
    original_refs = [ref.strip() for ref in original_refs if ref.strip()]

    return {
        'refs': original_refs
    }


def create_mock_span(test_case: Dict[str, Any]) -> Dict[str, Any]:
    """Create a citation span from test case data."""
    non_segment_ref = test_case.get('non_segment_ref', '')

    # Parse char range
    char_start = test_case.get('citation_char_start', '')
    char_end = test_case.get('citation_char_end', '')

    char_range = []
    if char_start and char_end:
        try:
            char_range = [int(char_start), int(char_end)]
        except (ValueError, TypeError):
            char_range = [0, 0]
    else:
        char_range = [0, 0]

    return {
        'ref': non_segment_ref,
        'type': 'citation',
        'charRange': char_range
    }


# Fixture to load CSV test cases
@pytest.fixture(scope='session')
def csv_test_cases():
    """Load test cases from CSV file."""
    csv_filename = os.environ.get('TEST_CSV_FILE', 'test_cases.csv')
    return load_test_cases_from_csv(csv_filename)


# Fixture to create resolver instance
@pytest.fixture(scope='session')
def resolver():
    """Create a single resolver instance for all tests."""
    return LLMSegmentResolver()


def pytest_generate_tests(metafunc):
    """Dynamically generate test parameters based on CSV row count."""
    if 'test_case_index' in metafunc.fixturenames:
        # Get the CSV file path
        csv_filename = os.environ.get('TEST_CSV_FILE', 'test_cases.csv')

        # Load test cases to get count
        test_cases = load_test_cases_from_csv(csv_filename)

        # Generate indices for actual test cases
        indices = list(range(len(test_cases)))

        # Parametrize with actual indices
        metafunc.parametrize('test_case_index', indices)


class TestResolverFromCSV:
    """Test suite for validating resolver against CSV test cases."""

    def test_csv_file_exists(self, csv_test_cases):
        """Test that the CSV file exists and contains test cases."""
        assert len(csv_test_cases) > 0, "No test cases found in CSV file"

    def test_resolver_matches_csv(self, csv_test_cases, resolver, test_case_index):
        """Test that resolver produces expected results for each CSV test case."""
        test_case = csv_test_cases[test_case_index]

        # Skip if no expected result
        expected_resolved = test_case.get('expected_resolved_ref', '')
        if not expected_resolved:
            pytest.skip("Test case has no expected resolution")

        # Create mock data structures
        chunk = create_mock_chunk(test_case)
        link = create_mock_link(test_case)
        span = create_mock_span(test_case)

        # Add span to chunk
        chunk['spans'] = [span]

        # Run resolver
        result = resolver.resolve(link, chunk)

        # Validate result
        assert result is not None, f"Resolver failed to resolve test case {test_case_index + 1}"

        # Check resolved ref matches
        actual_resolved = result.get('resolved_ref')

        # For backward compatibility, also check resolved_refs if single resolution
        if not actual_resolved and result.get('resolved_refs'):
            resolved_refs = result['resolved_refs']
            if len(resolved_refs) == 1:
                actual_resolved = resolved_refs[0]

        # For multiple resolutions, check if expected is in any of them
        if not actual_resolved and result.get('resolutions'):
            for res in result['resolutions']:
                if res.get('resolved_ref') == expected_resolved:
                    actual_resolved = expected_resolved
                    break

        assert actual_resolved == expected_resolved, (
            f"Test case {test_case_index + 1} failed:\n"
            f"  Expected: {expected_resolved}\n"
            f"  Got: {actual_resolved}\n"
            f"  Citing ref: {test_case.get('citing_ref', 'N/A')}\n"
            f"  Non-segment ref: {test_case.get('non_segment_ref', 'N/A')}"
        )

        # Optionally check selected segments
        expected_segments = test_case.get('expected_segments', '').split('|')
        expected_segments = [seg.strip() for seg in expected_segments if seg.strip()]

        if expected_segments:
            actual_segments = result.get('selected_segments', [])
            assert set(actual_segments) == set(expected_segments), (
                f"Test case {test_case_index + 1} - segments mismatch:\n"
                f"  Expected: {expected_segments}\n"
                f"  Got: {actual_segments}"
            )


class TestResolverDetailed:
    """More detailed tests with better diagnostics."""

    def test_all_cases_summary(self, csv_test_cases, resolver):
        """Run all test cases and provide a summary of results."""
        if not csv_test_cases:
            pytest.skip("No test cases in CSV")

        results = {
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'failures': []
        }

        for idx, test_case in enumerate(csv_test_cases):
            expected_resolved = test_case.get('expected_resolved_ref', '')
            if not expected_resolved:
                results['skipped'] += 1
                continue

            # Create mock structures
            chunk = create_mock_chunk(test_case)
            link = create_mock_link(test_case)
            span = create_mock_span(test_case)
            chunk['spans'] = [span]

            # Resolve
            result = resolver.resolve(link, chunk)

            if not result:
                results['failed'] += 1
                results['failures'].append({
                    'index': idx + 1,
                    'citing_ref': test_case.get('citing_ref', ''),
                    'non_segment_ref': test_case.get('non_segment_ref', ''),
                    'expected': expected_resolved,
                    'got': None,
                    'reason': 'No result returned'
                })
                continue

            # Get actual result
            actual_resolved = result.get('resolved_ref')
            if not actual_resolved and result.get('resolved_refs'):
                resolved_refs = result['resolved_refs']
                if len(resolved_refs) == 1:
                    actual_resolved = resolved_refs[0]

            # Check match
            if actual_resolved == expected_resolved:
                results['passed'] += 1
            else:
                results['failed'] += 1
                results['failures'].append({
                    'index': idx + 1,
                    'citing_ref': test_case.get('citing_ref', ''),
                    'non_segment_ref': test_case.get('non_segment_ref', ''),
                    'expected': expected_resolved,
                    'got': actual_resolved,
                    'reason': result.get('reason', 'No reason provided')
                })

        # Print summary
        total = results['passed'] + results['failed']
        accuracy = (results['passed'] / total * 100) if total > 0 else 0

        print(f"\n{'=' * 80}")
        print("TEST SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total test cases: {len(csv_test_cases)}")
        print(f"Passed: {results['passed']}")
        print(f"Failed: {results['failed']}")
        print(f"Skipped: {results['skipped']}")
        print(f"Accuracy: {accuracy:.1f}%")

        if results['failures']:
            print(f"\n{'=' * 80}")
            print("FAILURES")
            print(f"{'=' * 80}")
            for failure in results['failures'][:10]:  # Show first 10
                print(f"\nTest #{failure['index']}")
                print(f"  Citing: {failure['citing_ref']}")
                print(f"  Non-segment: {failure['non_segment_ref']}")
                print(f"  Expected: {failure['expected']}")
                print(f"  Got: {failure['got']}")
                print(f"  Reason: {failure['reason']}")

            if len(results['failures']) > 10:
                print(f"\n... and {len(results['failures']) - 10} more failures")

        print(f"\n{'=' * 80}")

        # Assert overall success
        assert results['failed'] == 0, (
            f"{results['failed']} test case(s) failed. "
            f"Accuracy: {accuracy:.1f}%. See output above for details."
        )


# Standalone function to run tests on a specific CSV file
def run_tests_on_csv(csv_filename: str, verbose: bool = True):
    """
    Run tests on a specific CSV file programmatically.

    Args:
        csv_filename: Path to CSV file
        verbose: Whether to print verbose output

    Returns:
        dict with test results
    """
    import sys

    # Set environment variable for pytest
    os.environ['TEST_CSV_FILE'] = csv_filename

    # Run pytest
    args = [__file__, '-v'] if verbose else [__file__]
    exit_code = pytest.main(args)

    return {
        'success': exit_code == 0,
        'exit_code': exit_code
    }


if __name__ == "__main__":
    # Allow running as script
    import sys

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        print(f"Running tests on: {csv_file}")
        result = run_tests_on_csv(csv_file)
        sys.exit(result['exit_code'])
    else:
        # Run pytest normally
        pytest.main([__file__, '-v'])

