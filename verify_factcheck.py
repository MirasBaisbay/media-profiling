#!/usr/bin/env python3
"""
Verification script for FactCheckSearcher.

Tests the fact-check search functionality with known domains that should
have different fact-check profiles:
- NYT (New York Times): Should have few/no failed fact checks
- InfoWars: Should have many failed fact checks
- An obscure domain: Should have no results (fallback)

Usage:
    python verify_factcheck.py
"""

import logging
import sys

from refactored_analyzers import FactCheckSearcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run verification tests for FactCheckSearcher."""
    print("=" * 70)
    print("FACTCHECKSEARCHER VERIFICATION")
    print("=" * 70)

    # Initialize analyzer
    print("\nInitializing FactCheckSearcher...")
    analyzer = FactCheckSearcher()
    print(f"  Sites to search: {analyzer.sites}")

    # Test domains with expected outcomes
    test_cases = [
        {
            "domain": "nytimes.com",
            "name": "New York Times",
            "expected": "Few or no failed fact checks (reputable outlet)",
        },
        {
            "domain": "infowars.com",
            "name": "InfoWars",
            "expected": "Multiple failed fact checks (known misinformation)",
        },
        {
            "domain": "obscure-local-blog-12345.com",
            "name": None,  # Let it auto-generate
            "expected": "No results (unknown outlet, fallback score)",
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {test['domain']}")
        print(f"Expected: {test['expected']}")
        print("-" * 70)

        try:
            result = analyzer.analyze(test["domain"], test.get("name"))

            print(f"\nResults:")
            print(f"  Domain: {result.domain}")
            print(f"  Outlet Name: {result.outlet_name}")
            print(f"  Source: {result.source.value}")
            print(f"  Total Fact Checks Found: {result.total_checks_count}")
            print(f"  Failed Fact Checks: {result.failed_checks_count}")
            print(f"  Score: {result.score:.1f}/10 (0=excellent, 10=poor)")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reasoning: {result.reasoning}")

            if result.findings:
                print(f"\n  Findings ({len(result.findings)}):")
                for j, finding in enumerate(result.findings[:5], 1):  # Show max 5
                    print(f"    {j}. [{finding.source_site}] {finding.verdict.value}")
                    print(f"       Claim: {finding.claim_summary[:80]}...")
                if len(result.findings) > 5:
                    print(f"    ... and {len(result.findings) - 5} more")

            if result.search_snippets:
                print(f"\n  Search Snippets (truncated):")
                print(f"    {result.search_snippets[:300]}...")

            results.append({
                "domain": test["domain"],
                "success": True,
                "score": result.score,
                "failed_checks": result.failed_checks_count,
                "total_checks": result.total_checks_count,
            })

        except Exception as e:
            logger.error(f"Test failed for {test['domain']}: {e}")
            results.append({
                "domain": test["domain"],
                "success": False,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for result in results:
        if result["success"]:
            status = "PASS"
            details = f"Score: {result['score']:.1f}, Failed: {result['failed_checks']}/{result['total_checks']}"
        else:
            status = "FAIL"
            details = result.get("error", "Unknown error")
            all_passed = False
        print(f"  {result['domain']}: {status} - {details}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All tests completed successfully!")
        print("\nExpected behavior verification:")
        print("  - NYT should have a LOW score (0-3) with few/no failed checks")
        print("  - InfoWars should have a HIGH score (7-10) with multiple failed checks")
        print("  - Unknown domain should have score 5.0 (neutral fallback)")
    else:
        print("Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
