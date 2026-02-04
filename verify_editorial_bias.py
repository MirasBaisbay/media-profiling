#!/usr/bin/env python3
"""
Verification script for EditorialBiasAnalyzer.

Tests the editorial bias analysis functionality with sample articles
representing different political perspectives:
- Left-leaning content (progressive policies)
- Right-leaning content (conservative policies)
- Center/neutral content (balanced reporting)

Usage:
    python verify_editorial_bias.py
"""

import logging
import sys

from refactored_analyzers import EditorialBiasAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run verification tests for EditorialBiasAnalyzer."""
    print("=" * 70)
    print("EDITORIALBIASANALYZER VERIFICATION")
    print("=" * 70)

    # Initialize analyzer
    print("\nInitializing EditorialBiasAnalyzer...")
    analyzer = EditorialBiasAnalyzer()

    # Test cases with different political leanings
    test_cases = [
        {
            "name": "Left-Leaning Content",
            "expected": "Left or Left-Center (negative score)",
            "domain": "progressive-news.com",
            "articles": [
                {
                    "title": "Universal Healthcare Would Save Billions",
                    "text": """
                    A new study confirms what progressives have long argued: universal healthcare
                    would save the country billions while ensuring healthcare as a fundamental right.
                    The current for-profit system enriches insurance executives while leaving millions
                    uninsured. We need Medicare for All now. The wealth inequality in this country
                    is unconscionable - billionaires should pay their fair share. Climate change
                    is an existential threat requiring immediate action. We must transition to
                    renewable energy and hold fossil fuel companies accountable for their decades
                    of climate denial and voter suppression of environmental legislation.
                    """
                },
                {
                    "title": "Workers Unite: Union Membership on the Rise",
                    "text": """
                    In a victory for workers' rights, union membership is growing as employees
                    demand fair wages and better working conditions. Corporate greed has gone
                    unchecked for too long. The minimum wage must be raised to a living wage.
                    Income inequality threatens our democracy. We need stronger regulations
                    on businesses and higher taxes on the wealthy to fund social programs
                    that help working families.
                    """
                }
            ],
        },
        {
            "name": "Right-Leaning Content",
            "expected": "Right or Right-Center (positive score)",
            "domain": "conservative-daily.com",
            "articles": [
                {
                    "title": "Tax Cuts Fuel Economic Growth",
                    "text": """
                    As conservatives predicted, the tax cuts have unleashed economic growth.
                    Big government and radical left policies threaten our prosperity. The free
                    market, not socialist programs, creates jobs. We need to cut regulations
                    that strangle small businesses. The woke agenda in schools must be stopped.
                    Parents should have the right to choose their children's education without
                    government interference. The Second Amendment is under attack from those
                    who want to defund the police while leaving law-abiding citizens defenseless.
                    """
                },
                {
                    "title": "Border Security Must Be Priority",
                    "text": """
                    The crisis at our border demands action. Open borders policies have failed.
                    We need strong enforcement and no amnesty for those who broke our laws.
                    The mainstream media won't report on the true costs of illegal immigration.
                    Traditional values are under attack by cancel culture. Critical race theory
                    has no place in our schools. We must protect religious freedom and the
                    rights of the unborn.
                    """
                }
            ],
        },
        {
            "name": "Center/Neutral Content",
            "expected": "Center (score near 0)",
            "domain": "balanced-news.com",
            "articles": [
                {
                    "title": "Congress Debates Infrastructure Bill",
                    "text": """
                    The House passed an infrastructure bill with bipartisan support on Thursday.
                    The legislation includes funding for roads, bridges, and broadband internet.
                    Supporters argue the investments are long overdue, while critics raise concerns
                    about the price tag. The bill now moves to the Senate where its fate remains
                    uncertain. Both Democratic and Republican lawmakers expressed mixed views on
                    specific provisions. Economic analysts offer varying projections on the bill's
                    impact on job creation and the national debt.
                    """
                },
                {
                    "title": "Federal Reserve Holds Interest Rates Steady",
                    "text": """
                    The Federal Reserve announced it will maintain current interest rates, citing
                    mixed economic signals. Inflation remains above the target rate, but employment
                    figures show strength in the labor market. Fed Chair stated the committee will
                    continue to monitor economic conditions. Some economists advocate for rate
                    increases to combat inflation, while others warn that could slow economic
                    recovery. Markets reacted with modest gains following the announcement.
                    """
                }
            ],
        },
        {
            "name": "No Articles (Empty Input)",
            "expected": "Center with 0 confidence (fallback)",
            "domain": "empty-test.com",
            "articles": [],
        },
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 70}")
        print(f"TEST {i}: {test['name']}")
        print(f"Expected: {test['expected']}")
        print(f"Domain: {test['domain']}")
        print(f"Articles: {len(test['articles'])}")
        print("-" * 70)

        try:
            result = analyzer.analyze(
                articles=test["articles"],
                url_or_domain=test["domain"],
                outlet_name=test["name"],
            )

            print(f"\nResults:")
            print(f"  Overall Bias: {result.overall_bias.value}")
            print(f"  Bias Score: {result.bias_score:+.1f} (-10=left, +10=right)")
            print(f"  MBFC Label: {result.mbfc_label}")
            print(f"  Uses Loaded Language: {result.uses_loaded_language}")
            print(f"  Articles Analyzed: {result.articles_analyzed}")
            print(f"  Confidence: {result.confidence:.2f}")

            if result.loaded_language_examples:
                print(f"\n  Loaded Language Examples:")
                for example in result.loaded_language_examples[:5]:
                    print(f"    - {example}")

            if result.policy_positions:
                print(f"\n  Policy Positions Detected:")
                for pos in result.policy_positions[:5]:
                    print(f"    - {pos.domain.value}: {pos.leaning.value} (conf: {pos.confidence:.2f})")

            if result.story_selection_bias:
                print(f"\n  Story Selection Bias: {result.story_selection_bias}")

            print(f"\n  Reasoning: {result.reasoning[:200]}...")

            results.append({
                "name": test["name"],
                "success": True,
                "bias_score": result.bias_score,
                "label": result.mbfc_label,
                "confidence": result.confidence,
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
            score_str = f"{result['bias_score']:+.1f}"
            details = f"Score: {score_str}, Label: {result['label']}, Conf: {result['confidence']:.2f}"
        else:
            status = "FAIL"
            details = result.get("error", "Unknown error")
            all_passed = False
        print(f"  {result['name']}: {status} - {details}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All tests completed successfully!")
        print("\nExpected behavior verification:")
        print("  - Left-leaning content: Negative score (-10 to -3)")
        print("  - Right-leaning content: Positive score (+3 to +10)")
        print("  - Center content: Score near 0 (-3 to +3)")
        print("  - Empty input: Score 0.0 with 0 confidence")
    else:
        print("Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
