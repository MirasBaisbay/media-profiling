#!/usr/bin/env python3
"""
Media Type Analyzer Verification Script

This script validates the MediaTypeAnalyzer against a golden dataset
and tests both the lookup (deterministic) and LLM fallback paths.

Usage:
    python verify_media_type.py [--lookup-only] [--verbose] [--limit N]

Options:
    --lookup-only    Only test lookup path (no API key required)
    --verbose        Print detailed results for each domain
    --limit N        Limit number of domains to test
"""

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Golden Dataset
# =============================================================================

# Test dataset with expected media types
# Mix of domains that should be in lookup and those that need LLM
MEDIA_TYPE_GOLD_STANDARD = [
    # TV Networks (should be in lookup)
    ("cnn.com", "TV", True, "Major US cable news"),
    ("bbc.com", "TV", True, "British broadcaster"),
    ("foxnews.com", "TV", True, "US cable news"),
    ("msnbc.com", "TV", True, "US cable news"),
    ("aljazeera.com", "TV", True, "International broadcaster"),

    # Newspapers (should be in lookup)
    ("nytimes.com", "Newspaper", True, "Major US newspaper"),
    ("washingtonpost.com", "Newspaper", True, "Major US newspaper"),
    ("theguardian.com", "Newspaper", True, "UK newspaper"),
    ("wsj.com", "Newspaper", True, "Financial newspaper"),
    ("latimes.com", "Newspaper", True, "Regional US newspaper"),

    # Websites (should be in lookup)
    ("politico.com", "Website", True, "Digital political news"),
    ("axios.com", "Website", True, "Digital news"),
    ("vox.com", "Website", True, "Digital explanatory journalism"),
    ("huffpost.com", "Website", True, "Digital news aggregator"),
    ("dailywire.com", "Website", True, "Conservative digital news"),

    # Magazines (should be in lookup)
    ("time.com", "Magazine", True, "Weekly news magazine"),
    ("theatlantic.com", "Magazine", True, "Monthly magazine"),
    ("economist.com", "Magazine", True, "Weekly business magazine"),
    ("newyorker.com", "Magazine", True, "Weekly magazine"),
    ("slate.com", "Magazine", True, "Online magazine"),

    # News Agencies (should be in lookup)
    ("reuters.com", "News Agency", True, "Wire service"),
    ("apnews.com", "News Agency", True, "Wire service"),

    # Radio (should be in lookup)
    ("npr.org", "Radio", True, "Public radio"),

    # Domains NOT in lookup (should trigger LLM)
    ("currentaffairs.org", "Magazine", True, "Small progressive magazine - in lookup"),
    ("consortiumnews.com", "Website", False, "Independent news - NOT in lookup"),
    ("mintpressnews.com", "Website", False, "Independent news - NOT in lookup"),
    ("grayzone.com", "Website", False, "Independent investigative - NOT in lookup"),
    ("popularresistance.org", "Website", False, "Activist site - NOT in lookup"),
    ("wsws.org", "Website", False, "Socialist news - NOT in lookup"),
]


# =============================================================================
# Verification Logic
# =============================================================================


@dataclass
class MediaTypeTestResult:
    """Result of testing a single domain."""
    domain: str
    expected_type: str
    expected_in_lookup: bool
    actual_type: str
    actual_source: str
    confidence: float
    type_match: bool
    source_match: bool
    reasoning: str


def run_lookup_only_verification(limit: Optional[int] = None) -> dict:
    """
    Run lookup-only verification (no API key required).

    Tests the deterministic lookup path without making any LLM calls.
    """
    from refactored_analyzers import MediaTypeAnalyzer, KNOWN_MEDIA_TYPES_PATH
    from schemas import MediaType

    test_data = MEDIA_TYPE_GOLD_STANDARD[:limit] if limit else MEDIA_TYPE_GOLD_STANDARD

    print("\n" + "=" * 70)
    print("MEDIA TYPE ANALYZER - LOOKUP ONLY VERIFICATION")
    print("=" * 70)

    # Load lookup table directly
    known_types = {}
    if os.path.exists(KNOWN_MEDIA_TYPES_PATH):
        print(f"Loading lookup table from {KNOWN_MEDIA_TYPES_PATH}...")
        with open(KNOWN_MEDIA_TYPES_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].startswith("#"):
                    continue
                if len(row) >= 2:
                    domain = row[0].strip().lower()
                    media_type = row[1].strip()
                    known_types[domain] = media_type
        print(f"Loaded {len(known_types)} known media types")
    else:
        print(f"ERROR: Lookup file not found at {KNOWN_MEDIA_TYPES_PATH}")
        return {}

    # Test each domain
    results = []
    print(f"\nTesting {len(test_data)} domains...")
    print("-" * 70)

    for domain, expected_type, expected_in_lookup, notes in test_data:
        domain_lower = domain.lower()

        # Check lookup
        actual_type = known_types.get(domain_lower)
        in_lookup = actual_type is not None

        # Check matches
        type_match = actual_type and actual_type.lower() == expected_type.lower()
        source_match = expected_in_lookup == in_lookup

        results.append({
            "domain": domain,
            "expected_type": expected_type,
            "expected_in_lookup": expected_in_lookup,
            "actual_type": actual_type or "N/A",
            "in_lookup": in_lookup,
            "type_match": type_match,
            "source_match": source_match,
            "notes": notes,
        })

        # Print result
        type_status = "✓" if type_match else ("?" if not in_lookup else "✗")
        lookup_status = "✓" if source_match else "✗"
        actual_str = actual_type if actual_type else "NOT FOUND"
        print(f"  {domain:<30} Expected: {expected_type:<12} Actual: {actual_str:<12} "
              f"[Type:{type_status} Lookup:{lookup_status}]")

    # Calculate metrics
    total = len(results)
    in_lookup_results = [r for r in results if r["in_lookup"]]
    expected_in_lookup = [r for r in results if r["expected_in_lookup"]]

    lookup_detection_correct = sum(1 for r in results if r["source_match"])
    type_correct_when_found = sum(1 for r in in_lookup_results if r["type_match"])

    print("\n" + "=" * 70)
    print("LOOKUP-ONLY RESULTS")
    print("=" * 70)
    print(f"Total Domains Tested:      {total}")
    print(f"Found in Lookup:           {len(in_lookup_results)}/{len(expected_in_lookup)} expected")
    print(f"Lookup Detection Accuracy: {lookup_detection_correct}/{total} ({lookup_detection_correct/total:.1%})")
    if in_lookup_results:
        print(f"Type Accuracy (when found): {type_correct_when_found}/{len(in_lookup_results)} "
              f"({type_correct_when_found/len(in_lookup_results):.1%})")

    # Show mismatches
    type_mismatches = [r for r in in_lookup_results if not r["type_match"]]
    if type_mismatches:
        print(f"\nType Mismatches ({len(type_mismatches)}):")
        for r in type_mismatches:
            print(f"  {r['domain']}: expected {r['expected_type']}, got {r['actual_type']}")

    lookup_mismatches = [r for r in results if not r["source_match"]]
    if lookup_mismatches:
        print(f"\nLookup Detection Mismatches ({len(lookup_mismatches)}):")
        for r in lookup_mismatches:
            expected = "in lookup" if r["expected_in_lookup"] else "NOT in lookup"
            actual = "found" if r["in_lookup"] else "not found"
            print(f"  {r['domain']}: expected {expected}, actually {actual}")

    return {
        "total": total,
        "found_in_lookup": len(in_lookup_results),
        "lookup_detection_accuracy": lookup_detection_correct / total,
        "type_accuracy_when_found": type_correct_when_found / len(in_lookup_results) if in_lookup_results else 0,
        "results": results,
    }


def run_full_verification(
    limit: Optional[int] = None,
    verbose: bool = False
) -> dict:
    """
    Run full verification including LLM fallback (requires API key).
    """
    from refactored_analyzers import MediaTypeAnalyzer
    from schemas import MediaTypeSource

    test_data = MEDIA_TYPE_GOLD_STANDARD[:limit] if limit else MEDIA_TYPE_GOLD_STANDARD

    print("\n" + "=" * 70)
    print("MEDIA TYPE ANALYZER - FULL VERIFICATION")
    print("=" * 70)

    # Initialize analyzer
    print("Initializing MediaTypeAnalyzer...")
    analyzer = MediaTypeAnalyzer()

    # Show lookup stats
    stats = analyzer.get_lookup_stats()
    print(f"  Lookup Loaded: {stats['loaded']}")
    print(f"  Known Outlets: {stats['total_domains']}")

    # Test each domain
    results: list[MediaTypeTestResult] = []
    print(f"\nTesting {len(test_data)} domains...")
    print("-" * 70)

    for domain, expected_type, expected_in_lookup, notes in test_data:
        print(f"  Testing: {domain}...", end=" ", flush=True)

        try:
            result = analyzer.analyze(domain)

            # Determine if source matches expectation
            actual_in_lookup = result.source == MediaTypeSource.LOOKUP
            source_match = expected_in_lookup == actual_in_lookup

            # Check type match (case-insensitive)
            type_match = result.media_type.value.lower() == expected_type.lower()

            test_result = MediaTypeTestResult(
                domain=domain,
                expected_type=expected_type,
                expected_in_lookup=expected_in_lookup,
                actual_type=result.media_type.value,
                actual_source=result.source.value,
                confidence=result.confidence,
                type_match=type_match,
                source_match=source_match,
                reasoning=result.reasoning,
            )
            results.append(test_result)

            # Print status
            type_status = "✓" if type_match else "✗"
            source_status = "✓" if source_match else "✗"
            print(f"Type:{type_status} Source:{source_status} "
                  f"[{result.source.value}:{result.media_type.value}] "
                  f"(conf: {result.confidence:.2f})")

            if verbose and not type_match:
                print(f"      Expected: {expected_type}, Got: {result.media_type.value}")
                print(f"      Reasoning: {result.reasoning[:100]}...")

        except Exception as e:
            print(f"ERROR: {e}")
            results.append(MediaTypeTestResult(
                domain=domain,
                expected_type=expected_type,
                expected_in_lookup=expected_in_lookup,
                actual_type="ERROR",
                actual_source="ERROR",
                confidence=0.0,
                type_match=False,
                source_match=False,
                reasoning=str(e),
            ))

    # Calculate metrics
    total = len(results)
    type_correct = sum(1 for r in results if r.type_match)
    source_correct = sum(1 for r in results if r.source_match)

    lookup_results = [r for r in results if r.actual_source == "Lookup"]
    llm_results = [r for r in results if r.actual_source == "LLM"]

    print("\n" + "=" * 70)
    print("FULL VERIFICATION RESULTS")
    print("=" * 70)
    print(f"Total Domains Tested:      {total}")
    print(f"Type Accuracy:             {type_correct}/{total} ({type_correct/total:.1%})")
    print(f"Source Detection Accuracy: {source_correct}/{total} ({source_correct/total:.1%})")
    print(f"Lookup Hits:               {len(lookup_results)} ({len(lookup_results)/total:.1%})")
    print(f"LLM Fallbacks:             {len(llm_results)} ({len(llm_results)/total:.1%})")

    if lookup_results:
        lookup_type_correct = sum(1 for r in lookup_results if r.type_match)
        print(f"  Lookup Type Accuracy:    {lookup_type_correct}/{len(lookup_results)} "
              f"({lookup_type_correct/len(lookup_results):.1%})")

    if llm_results:
        llm_type_correct = sum(1 for r in llm_results if r.type_match)
        print(f"  LLM Type Accuracy:       {llm_type_correct}/{len(llm_results)} "
              f"({llm_type_correct/len(llm_results):.1%})")

    # Show mismatches
    type_mismatches = [r for r in results if not r.type_match]
    if type_mismatches:
        print(f"\nType Mismatches ({len(type_mismatches)}):")
        for r in type_mismatches:
            print(f"  {r.domain}:")
            print(f"    Expected: {r.expected_type}, Got: {r.actual_type}")
            print(f"    Source: {r.actual_source}, Confidence: {r.confidence:.2f}")
            print(f"    Reasoning: {r.reasoning[:80]}...")

    return {
        "total": total,
        "type_accuracy": type_correct / total,
        "source_accuracy": source_correct / total,
        "lookup_hits": len(lookup_results),
        "llm_fallbacks": len(llm_results),
        "results": results,
    }


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify MediaTypeAnalyzer against golden dataset"
    )
    parser.add_argument(
        "--lookup-only",
        action="store_true",
        help="Only test lookup path (no API key required)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each domain",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of domains to test",
    )
    args = parser.parse_args()

    if args.lookup_only:
        result = run_lookup_only_verification(limit=args.limit)
        if result:
            print("\n✓ Lookup-only verification completed")
            sys.exit(0)
        else:
            print("\n✗ Lookup-only verification failed")
            sys.exit(1)
    else:
        result = run_full_verification(limit=args.limit, verbose=args.verbose)
        if result and result["type_accuracy"] >= 0.7:
            print("\n✓ Full verification completed")
            sys.exit(0)
        else:
            print("\n⚠️  Type accuracy below 70% threshold")
            sys.exit(1)


if __name__ == "__main__":
    main()
