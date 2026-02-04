#!/usr/bin/env python3
"""
Verification script for PseudoscienceAnalyzer.

Tests the pseudoscience detection functionality with sample articles:
- Pro-science content (respects scientific consensus)
- Pseudoscience-promoting content (anti-vax, climate denial, etc.)
- Mixed content (some good, some bad)

Usage:
    python verify_pseudoscience.py
"""

import logging
import sys

from refactored_analyzers import PseudoscienceAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run verification tests for PseudoscienceAnalyzer."""
    print("=" * 70)
    print("PSEUDOSCIENCEANALYZER VERIFICATION")
    print("=" * 70)

    # Initialize analyzer
    print("\nInitializing PseudoscienceAnalyzer...")
    analyzer = PseudoscienceAnalyzer()

    # Test cases with different science reporting quality
    test_cases = [
        {
            "name": "Pro-Science Content",
            "expected": "Low score (0-2), respects consensus",
            "domain": "science-daily.com",
            "articles": [
                {
                    "title": "New Study Confirms Vaccine Safety",
                    "text": """
                    A comprehensive study published in The Lancet has once again confirmed
                    the safety and efficacy of COVID-19 vaccines. The research, conducted
                    across 50 countries with over 10 million participants, found no link
                    between vaccines and adverse effects beyond those already known.

                    The scientific consensus remains clear: vaccines are one of the most
                    effective public health interventions available. The CDC and WHO
                    continue to recommend vaccination for eligible populations.

                    Climate scientists also released new data showing accelerating ice
                    melt in Antarctica, consistent with projections from climate models.
                    The findings underscore the urgency of reducing greenhouse gas emissions.
                    """
                },
                {
                    "title": "GMO Safety Confirmed by Decades of Research",
                    "text": """
                    A meta-analysis of over 900 studies spanning 20 years has found no
                    evidence that genetically modified organisms pose health risks to
                    consumers. The research, published in Nature, aligns with the
                    scientific consensus that approved GMOs are safe for consumption.

                    Major scientific organizations including the National Academy of
                    Sciences, the American Medical Association, and the World Health
                    Organization agree that GMO foods currently available are safe.
                    """
                }
            ],
        },
        {
            "name": "Pseudoscience-Promoting Content",
            "expected": "High score (7-10), promotes pseudoscience",
            "domain": "natural-truth.com",
            "articles": [
                {
                    "title": "Exposed: The Vaccine-Autism Link They Don't Want You to Know",
                    "text": """
                    Despite what the mainstream medical establishment claims, mounting
                    evidence suggests vaccines are not as safe as we're told. Parents
                    across the country report their children developed autism symptoms
                    shortly after vaccination.

                    Big Pharma doesn't want you to know the truth. Natural immunity is
                    far superior to artificial vaccines. Consider homeopathic alternatives
                    and essential oils for protection. The detoxification protocols we
                    recommend can help remove vaccine toxins from your body.

                    Don't be fooled by so-called "fact-checkers" funded by pharmaceutical
                    companies. Do your own research and question everything.
                    """
                },
                {
                    "title": "Climate Change: The Greatest Hoax of Our Time",
                    "text": """
                    The climate change narrative is falling apart. Scientists are
                    finally admitting their models were wrong. The Earth's climate
                    has always changed naturally - this is nothing new.

                    Follow the money: climate scientists get funding only if they
                    support the hoax. The globalist agenda behind climate alarmism
                    is about control, not science. 5G towers are also contributing
                    to environmental problems they blame on CO2.

                    Meanwhile, chemtrails continue to poison our skies as part of
                    secret geoengineering programs. Wake up and see the truth.
                    """
                }
            ],
        },
        {
            "name": "Mixed Content",
            "expected": "Medium score (4-6), inconsistent",
            "domain": "health-lifestyle.com",
            "articles": [
                {
                    "title": "Understanding Flu Vaccines",
                    "text": """
                    The flu vaccine remains an important tool for preventing seasonal
                    influenza. According to the CDC, vaccination reduces flu illness
                    and prevents thousands of hospitalizations each year. However,
                    some people prefer to boost their immune system naturally through
                    vitamin supplements and essential oils.

                    While science supports vaccination, there's also growing interest
                    in alternative approaches. Astrology enthusiasts note that certain
                    zodiac signs may be more susceptible to winter illnesses.
                    """
                },
                {
                    "title": "Cancer Prevention: What Science Says",
                    "text": """
                    Researchers have identified several evidence-based ways to reduce
                    cancer risk, including not smoking, maintaining healthy weight,
                    and limiting alcohol. Regular screenings can detect cancer early.

                    The American Cancer Society recommends following established
                    medical guidelines. While some promote alternative treatments,
                    there is no substitute for proven medical care when it comes
                    to cancer treatment.
                    """
                }
            ],
        },
        {
            "name": "No Articles (Empty Input)",
            "expected": "Neutral score (5.0) with 0 confidence",
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
            print(f"  Score: {result.score:.1f}/10 (0=pro-science, 10=promotes pseudoscience)")
            print(f"  Promotes Pseudoscience: {result.promotes_pseudoscience}")
            print(f"  Overall Severity: {result.overall_severity.value}")
            print(f"  Respects Scientific Consensus: {result.respects_scientific_consensus}")
            print(f"  Articles Analyzed: {result.articles_analyzed}")
            print(f"  Confidence: {result.confidence:.2f}")

            if result.categories_found:
                print(f"\n  Categories Found:")
                for cat in result.categories_found:
                    print(f"    - {cat.value}")

            if result.indicators:
                print(f"\n  Indicators ({len(result.indicators)}):")
                for indicator in result.indicators[:3]:
                    print(f"    - [{indicator.severity.value}] {indicator.category.value}")
                    print(f"      Evidence: {indicator.evidence[:80]}...")
                    print(f"      Consensus: {indicator.scientific_consensus[:60]}...")
                if len(result.indicators) > 3:
                    print(f"    ... and {len(result.indicators) - 3} more")

            print(f"\n  Reasoning: {result.reasoning[:200]}...")

            results.append({
                "name": test["name"],
                "success": True,
                "score": result.score,
                "promotes": result.promotes_pseudoscience,
                "severity": result.overall_severity.value,
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
            details = f"Score: {result['score']:.1f}, Promotes: {result['promotes']}, Severity: {result['severity']}"
        else:
            status = "FAIL"
            details = result.get("error", "Unknown error")
            all_passed = False
        print(f"  {result['name']}: {status} - {details}")

    print("\n" + "-" * 70)
    if all_passed:
        print("All tests completed successfully!")
        print("\nExpected behavior verification:")
        print("  - Pro-science content: Low score (0-2), NONE_DETECTED severity")
        print("  - Pseudoscience content: High score (7-10), PROMOTES severity")
        print("  - Mixed content: Medium score (4-6), MIXED or PRESENTS_UNCRITICALLY")
        print("  - Empty input: Score 5.0 with 0 confidence")
    else:
        print("Some tests failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
