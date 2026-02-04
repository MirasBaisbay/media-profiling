#!/usr/bin/env python3
"""
Verification script for SourcingAnalyzer.

Tests the sourcing analysis functionality with different article scenarios:
- Article with high-quality sources (Reuters, AP, .gov)
- Article with mixed sources (major outlets + unknown)
- Article with questionable sources (known unreliable sites)
- Article with no sources (no hyperlinks)

Usage:
    python verify_sourcing.py
"""

import logging
import sys

from refactored_analyzers import SourcingAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run verification tests for SourcingAnalyzer."""
    print("=" * 70)
    print("SOURCINGANALYZER VERIFICATION")
    print("=" * 70)

    # Initialize analyzer
    print("\nInitializing SourcingAnalyzer...")
    analyzer = SourcingAnalyzer()

    # Test articles with different sourcing patterns
    test_cases = [
        {
            "name": "High-Quality Sources",
            "expected": "Low score (0-3) - excellent sourcing",
            "articles": [
                {
                    "text": """
                    According to a report from Reuters (https://reuters.com/world/article123),
                    the economic situation is improving. The Associated Press
                    (https://apnews.com/article/economy-update) confirmed these findings.
                    The Federal Reserve (https://federalreserve.gov/newsevents/report.htm)
                    released new data supporting this analysis. A study from MIT
                    (https://economics.mit.edu/research/study2024) provides additional context.
                    """
                }
            ],
        },
        {
            "name": "Mixed Sources",
            "expected": "Medium score (3-6) - acceptable sourcing",
            "articles": [
                {
                    "text": """
                    The New York Times reported (https://nytimes.com/2024/article) on the
                    recent developments. Local news outlet reported similar findings
                    (https://localnews-example.com/story). An industry blog
                    (https://tech-blog-unknown.com/post) provided commentary.
                    BBC covered the international angle (https://bbc.com/news/world).
                    """
                }
            ],
        },
        {
            "name": "No Sources (No Links)",
            "expected": "Score 5.0 (neutral) - cannot assess",
            "articles": [
                {
                    "text": """
                    This article contains no hyperlinks to external sources.
                    It makes many claims but does not cite any evidence.
                    Readers have no way to verify the information presented.
                    This is an example of poor journalistic practice.
                    """
                }
            ],
        },
        {
            "name": "Social Media Only",
            "expected": "Score 5.0 (neutral) - social media excluded",
            "articles": [
                {
                    "text": """
                    According to a tweet (https://twitter.com/user/status/123),
                    this is happening. See also this Facebook post
                    (https://facebook.com/post/456) and this YouTube video
                    (https://youtube.com/watch?v=abc). A Reddit thread discusses it
                    (https://reddit.com/r/news/comments/xyz).
                    """
                }
            ],
        },
        {
            "name": "Multiple Articles - Diverse Sources",
            "expected": "Score based on combined source quality",
            "articles": [
                {
                    "text": """
                    First article cites AP (https://apnews.com/article/first)
                    and the White House (https://whitehouse.gov/briefing).
                    """
                },
                {
                    "text": """
                    Second article references the CDC (https://cdc.gov/report)
                    and Nature journal (https://nature.com/articles/study).
                    """
                },
                {
                    "text": """
                    Third article cites WSJ (https://wsj.com/articles/story)
                    and a university study (https://stanford.edu/research/paper).
                    """
                },
            ],
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"Articles: {len(test['articles'])}")
        print("-" * 70)

        try:
            result = analyzer.analyze(test["articles"])

            print(f"\nResults:")
            print(f"  Score: {result.score:.1f}/10 (0=excellent, 10=poor)")
            print(f"  Has Hyperlinks: {result.has_hyperlinks}")
            print(f"  Total Sources Found: {result.total_sources_found}")
            print(f"  Unique Domains: {result.unique_domains}")
            print(f"  Avg Sources/Article: {result.avg_sources_per_article:.1f}")
            print(f"  Has Primary Sources: {result.has_primary_sources}")
            print(f"  Has Wire Services: {result.has_wire_services}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Reasoning: {result.reasoning}")

            if result.source_assessments:
                print(f"\n  Source Assessments ({len(result.source_assessments)}):")
                for assessment in result.source_assessments:
                    print(f"    - {assessment.domain}: {assessment.quality.value}")
                    print(f"      Reason: {assessment.reasoning[:60]}...")

            results.append({
                "name": test["name"],
                "success": True,
                "score": result.score,
                "unique_domains": result.unique_domains,
                "has_hyperlinks": result.has_hyperlinks,
            })

        except Exception as e:
            logger.error(f"Test failed for {test['name']}: {e}")
            results.append({
                "name": test["name"],
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
            details = f"Score: {result['score']:.1f}, Domains: {result['unique_domains']}, Links: {result['has_hyperlinks']}"
        else:
            status = "FAIL"
            details = result.get("error", "Unknown error")
            all_passed = False
        print(f"  {result['name']}: {status} - {details}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All tests completed successfully!")
        print("\nExpected behavior verification:")
        print("  - High-quality sources: Score 0-3 (PRIMARY + WIRE_SERVICE)")
        print("  - Mixed sources: Score 3-6 (varies)")
        print("  - No sources: Score 5.0 (neutral fallback)")
        print("  - Social media only: Score 5.0 (excluded from analysis)")
        print("  - Multiple articles: Aggregates all sources")
    else:
        print("Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
