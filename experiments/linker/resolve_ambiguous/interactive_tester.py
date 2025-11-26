"""
Interactive utility for testing and validating LLM segment resolver results.
Allows manual validation and saves results to CSV for future testing.
"""
import os
import csv
from datetime import datetime
from llm_segment_resolver import LLMSegmentResolver
from utils import get_random_non_segment_links_with_chunks


def print_separator(char="-", length=80):
    print(char * length)


def print_result(index: int, item: dict, result: dict):
    """Print a single resolution result in a readable format."""
    link = item["link"]
    chunk = item["chunk"]

    print(f"\n{'=' * 80}")
    print(f"RESULT #{index + 1}")
    print(f"{'=' * 80}")

    print(f"\nCiting ref: {chunk.get('ref')}")
    print(f"Citing text: {chunk.get('he', chunk.get('en', ''))[:200]}...")

    print(f"\nOriginal link refs: {link.get('refs')}")

    if result:
        print(f"\n✓ RESOLVED")
        # Handle both single and multiple resolutions
        if result.get('resolved_ref'):
            print(f"  Resolved ref: {result['resolved_ref']}")
        elif result.get('resolved_refs'):
            print(f"  Resolved {len(result['resolved_refs'])} refs: {result['resolved_refs']}")

        print(f"  Updated link refs: {result['link'].get('refs')}")
        print(f"  Selected segments: {result['selected_segments']}")

        # Show detailed info for multiple resolutions
        if result.get("resolutions") and len(result["resolutions"]) > 1:
            print(f"\n  Detailed resolutions:")
            for i, res in enumerate(result["resolutions"], 1):
                print(f"    {i}. {res.get('original_ref')} → {res.get('resolved_ref')}")
                # Show citation text if available
                if res.get('citation_span') and res.get('citing_text'):
                    char_range = res['citation_span'].get('charRange', [])
                    if len(char_range) == 2:
                        start, end = char_range
                        citation = res['citing_text'][start:end] if 0 <= start < end <= len(res['citing_text']) else ''
                        if citation:
                            print(f"       Citation: {citation[:100]}{'...' if len(citation) > 100 else ''}")
                if res.get("reason"):
                    print(f"       Reason: {res['reason']}")
        elif result.get("reason"):
            print(f"  LLM reason: {result['reason']}")
            # Show citation text for single resolution
            if result.get('citation_span') and result.get('citing_text'):
                char_range = result['citation_span'].get('charRange', [])
                if len(char_range) == 2:
                    start, end = char_range
                    citation = result['citing_text'][start:end] if 0 <= start < end <= len(result['citing_text']) else ''
                    if citation:
                        print(f"  Citation text: {citation[:100]}{'...' if len(citation) > 100 else ''}")

    else:
        print(f"\n✗ NO RESOLUTION FOUND")


def get_user_feedback() -> str:
    """Get user feedback on whether to save this result as a test case."""
    while True:
        response = input("\nSave this result as a test case? (y/n/q) [y=save, n=skip, q=quit]: ").strip().lower()
        if response in ['y', 'n', 'q']:
            return response
        print("Invalid input. Please enter y, n, or q.")


def save_to_csv(results: list, filename: str):
    """Save test cases to a CSV file, overwriting duplicates."""
    fieldnames = [
        'timestamp',
        'citing_ref',
        'citing_text',
        'citing_language',
        'citing_version_title',
        'original_link_refs',
        'non_segment_ref',
        'citation_text',
        'citation_char_start',
        'citation_char_end',
        'expected_resolved_ref',
        'expected_segments',
        'resolution_reason'
    ]

    # Read existing test cases if file exists
    existing_cases = {}
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Use (citing_ref, non_segment_ref, char_start, char_end) as unique key
                # This ensures different citations from the same passage are distinguished
                key = (
                    row['citing_ref'],
                    row['non_segment_ref'],
                    row.get('citation_char_start', ''),
                    row.get('citation_char_end', '')
                )
                existing_cases[key] = row

    # Add/update with new results
    updated_count = 0
    new_count = 0
    for result_data in results:
        key = (
            result_data['citing_ref'],
            result_data['non_segment_ref'],
            result_data.get('citation_char_start', ''),
            result_data.get('citation_char_end', '')
        )
        if key in existing_cases:
            updated_count += 1
        else:
            new_count += 1
        existing_cases[key] = result_data

    # Write all cases back to file
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for case in existing_cases.values():
            writer.writerow(case)

    if updated_count > 0 and new_count > 0:
        print(f"\n✓ Saved {new_count} new test case(s) and updated {updated_count} existing test case(s) in {filename}")
    elif updated_count > 0:
        print(f"\n✓ Updated {updated_count} existing test case(s) in {filename}")
    else:
        print(f"\n✓ Saved {new_count} new test case(s) to {filename}")



def prepare_csv_row(item: dict, result: dict, non_segment_ref: str) -> dict:
    """Prepare a test case row for CSV export."""
    chunk = item["chunk"]
    link = item["link"]

    citing_text = chunk.get('he', chunk.get('en', ''))

    # Extract citation information from span
    citation_span = result.get('citation_span', {})
    char_range = citation_span.get('charRange', [None, None])
    citation_char_start = str(char_range[0]) if char_range[0] is not None else ''
    citation_char_end = str(char_range[1]) if char_range[1] is not None else ''

    # Extract the actual citation text from the citing text
    citation_text = ''
    if result.get('citing_text') and char_range[0] is not None and char_range[1] is not None:
        full_text = result['citing_text']
        start, end = char_range[0], char_range[1]
        if 0 <= start < end <= len(full_text):
            citation_text = full_text[start:end]

    return {
        'timestamp': datetime.now().isoformat(),
        'citing_ref': chunk.get('ref', ''),
        'citing_text': citing_text[:500] if citing_text else '',  # Truncate for CSV
        'citing_language': chunk.get('language', ''),
        'citing_version_title': chunk.get('versionTitle', ''),
        'original_link_refs': '|'.join(link.get('refs', [])),
        'non_segment_ref': non_segment_ref,
        'citation_text': citation_text[:200] if citation_text else '',  # The actual cited text
        'citation_char_start': citation_char_start,
        'citation_char_end': citation_char_end,
        'expected_resolved_ref': result['resolved_ref'] if result and result.get('resolved_ref') else '',
        'expected_segments': '|'.join(result['selected_segments']) if result and result.get('selected_segments') else '',
        'resolution_reason': result.get('reason', '') if result else ''
    }


def main():
    """Main interactive testing loop."""
    print("=" * 80)
    print("LLM SEGMENT RESOLVER - INTERACTIVE TESTER")
    print("=" * 80)

    # Configuration
    n_samples = input("\nHow many samples to test? (default: 5): ").strip()
    n_samples = int(n_samples) if n_samples else 5

    seed = input("Random seed? (default: 613): ").strip()
    seed = int(seed) if seed else 613

    csv_filename = input("CSV filename to save test cases? (default: test_cases.csv): ").strip()
    csv_filename = csv_filename if csv_filename else "test_cases.csv"

    print(f"\nLoading {n_samples} samples with seed {seed}...")

    # Initialize resolver and get samples
    resolver = LLMSegmentResolver()
    samples = get_random_non_segment_links_with_chunks(n=n_samples, use_remote=True, seed=seed, use_cache=True)

    saved_test_cases = []
    stats = {'saved': 0, 'skipped': 0}

    # Process each sample
    for i, item in enumerate(samples):
        link = item["link"]
        chunk = item["chunk"]

        # Resolve
        print(f"\n\nProcessing sample {i + 1}/{n_samples}...")
        result = resolver.resolve(link, chunk)

        # Display result
        print_result(i, item, result)

        # Get user feedback
        feedback = get_user_feedback()

        if feedback == 'q':
            print("\n\nQuitting early...")
            break
        elif feedback == 'n':
            stats['skipped'] += 1
            print("⊘ Skipped - not saved")
            continue
        elif feedback == 'y':
            # Handle multiple resolutions - save each one as a separate test case
            if result and result.get('resolutions'):
                # Multiple resolutions - save each separately
                for res in result['resolutions']:
                    csv_row = prepare_csv_row(item, res, res.get('original_ref', ''))
                    saved_test_cases.append(csv_row)
                stats['saved'] += len(result['resolutions'])
                print(f"✓ Saved {len(result['resolutions'])} test case(s)")
            elif result:
                # Single resolution (backward compatibility)
                non_segment_ref = ""
                for tref in link.get("refs", []):
                    try:
                        from sefaria.model.text import Ref
                        oref = Ref(tref)
                        if not oref.is_segment_level():
                            non_segment_ref = oref.normal()
                            break
                    except Exception:
                        continue

                csv_row = prepare_csv_row(item, result, non_segment_ref)
                saved_test_cases.append(csv_row)
                stats['saved'] += 1
                print("✓ Saved as test case")
            else:
                print("⚠ No result to save")


    # Save test cases
    if saved_test_cases:
        save_to_csv(saved_test_cases, csv_filename)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total processed: {i + 1}")
    print(f"Saved as test cases: {stats['saved']}")
    print(f"Skipped: {stats['skipped']}")

    print(f"\nTest cases saved to: {csv_filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()

