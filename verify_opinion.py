#!/usr/bin/env python3
"""
Opinion Classification Validation Script

This script validates the OpinionAnalyzer against a golden dataset of
pre-labeled articles. It creates a CSV file with ground truth labels,
runs the analyzer, and produces a classification report.

Usage:
    python verify_opinion.py [--recreate-csv] [--verbose]

Options:
    --recreate-csv  Force recreation of the golden dataset CSV
    --verbose       Print detailed results for each article
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

from refactored_analyzers import OpinionAnalyzer
from schemas import ArticleType, ValidationReport, ValidationResult


# =============================================================================
# Golden Dataset Configuration
# =============================================================================

CSV_FILENAME = "opinion_gold_standard.csv"

# Ground truth dataset with real articles
# Each entry: (url, title, text_snippet, expected_label)
GOLDEN_DATASET = [
    # --- NEWS ARTICLES ---
    (
        "https://www.bbc.com/news/world-us-canada-12345",
        "US Congress passes infrastructure bill after months of debate",
        """The US Congress has passed a major infrastructure bill worth $1.2 trillion
        after months of intense negotiations between Democrats and Republicans.

        The bill, which passed the House with a 228-206 vote, includes funding for
        roads, bridges, public transit, and broadband internet expansion.

        President Biden is expected to sign the legislation into law next week.
        Senate Majority Leader Chuck Schumer called it "a historic investment in
        America's future." Republican minority leader Mitch McConnell said his
        party had secured important concessions during negotiations.

        The Congressional Budget Office estimates the bill will add $256 billion
        to the federal deficit over the next decade.""",
        ArticleType.NEWS,
    ),
    (
        "https://www.reuters.com/article/global-economy-12345",
        "Fed raises interest rates by 0.25% amid inflation concerns",
        """The Federal Reserve raised its benchmark interest rate by 0.25 percentage
        points on Wednesday, marking the third increase this year as policymakers
        seek to combat persistent inflation.

        The decision, announced following a two-day policy meeting, brings the
        federal funds rate to a range of 4.75% to 5%. The move was widely
        anticipated by economists and financial markets.

        Fed Chair Jerome Powell said in a press conference that "inflation remains
        elevated" and that the central bank remains "strongly committed" to
        returning it to the 2% target. He noted that recent banking sector stress
        could tighten credit conditions.

        Markets reacted positively to the announcement, with the S&P 500 rising
        0.8% in afternoon trading.""",
        ArticleType.NEWS,
    ),
    (
        "https://www.apnews.com/article/climate-summit-12345",
        "World leaders gather for climate summit in Dubai",
        """World leaders from more than 190 countries have gathered in Dubai for the
        COP28 climate summit, with discussions focused on accelerating the
        transition away from fossil fuels.

        UN Secretary-General António Guterres called for "immediate action" to
        address rising global temperatures. "We are on a highway to climate hell
        with our foot on the accelerator," he said in his opening remarks.

        The summit comes amid record-breaking global temperatures, with 2023 on
        track to be the hottest year in recorded history. Scientists from the
        World Meteorological Organization reported that global average temperatures
        are now 1.4°C above pre-industrial levels.

        Negotiations are expected to focus on a new global stocktake assessing
        progress toward Paris Agreement goals.""",
        ArticleType.NEWS,
    ),
    (
        "https://www.washingtonpost.com/national/article-12345",
        "Supreme Court hears arguments in landmark voting rights case",
        """The Supreme Court heard oral arguments Tuesday in a case that could
        significantly reshape voting rights protections across the United States.

        The case, Moore v. Harper, centers on whether state legislatures have
        exclusive authority to set rules for federal elections without oversight
        from state courts.

        Justice Ketanji Brown Jackson questioned the petitioners about the
        historical basis for their argument, while Justice Samuel Alito pressed
        the respondents on the limits of state court authority.

        Lawyers for the North Carolina legislature argued that the Elections
        Clause of the Constitution grants state legislatures broad power over
        federal elections. Opposing counsel contended that state constitutions
        must serve as a check on that power.

        A decision is expected by June.""",
        ArticleType.NEWS,
    ),
    # --- OPINION ARTICLES ---
    (
        "https://www.nytimes.com/opinion/column-12345",
        "Opinion: Why America needs universal healthcare now",
        """I've been a physician for 30 years, and I've never been more convinced
        that our healthcare system is fundamentally broken. We spend more per
        capita than any other developed nation, yet our outcomes are worse.
        It's time for universal healthcare.

        In my view, the evidence is overwhelming. Every day in my practice, I see
        patients who delay treatment because they can't afford it. I see families
        bankrupted by medical bills. I see the administrative waste of our
        fragmented system.

        The opponents of universal healthcare will tell you it's "socialism" or
        that it will lead to rationing. These arguments don't hold up to scrutiny.
        Countries like Canada, Germany, and France provide excellent care to all
        their citizens at lower cost than we do.

        We must act now. The human cost of inaction is simply too high. Congress
        should pass Medicare for All legislation this year.""",
        ArticleType.OPINION,
    ),
    (
        "https://www.wsj.com/opinion/editorial-12345",
        "The border crisis demands immediate action",
        """The situation at our southern border has reached a breaking point, and
        the Biden administration's policies are largely to blame. We believe it's
        time for a fundamental change in approach.

        Since taking office, this administration has systematically dismantled
        effective border security measures. The "Remain in Mexico" policy, which
        had proven successful, was ended. Deportations have dropped dramatically.

        The results speak for themselves: record numbers of illegal crossings,
        overwhelmed border communities, and a humanitarian crisis that benefits
        no one—least of all the migrants themselves who are being exploited by
        criminal cartels.

        In our judgment, the path forward is clear. We need to restore the proven
        policies that worked, increase border security funding, and reform our
        asylum system to prevent abuse. The safety of American communities and
        the dignity of migrants both demand it.""",
        ArticleType.OPINION,
    ),
    (
        "https://www.theatlantic.com/ideas/archive/12345",
        "I was wrong about artificial intelligence",
        """Three years ago, I wrote a piece dismissing fears about artificial
        intelligence as overblown. I argued that we were decades away from AI
        systems that could pose any real concern. I now believe I was wrong.

        The rapid advancement of large language models has forced me to reconsider
        my position. When I first used GPT-4, I was genuinely unsettled. This
        wasn't the narrow, task-specific AI I had expected. It felt different.

        I think we as a society need to have a serious conversation about where
        this technology is heading. The potential benefits are enormous, but so
        are the risks. My worry is that we're moving too fast to think through
        the implications.

        We should be investing massively in AI safety research. We should be
        developing regulatory frameworks now, before it's too late. And we should
        be humble about what we don't know. I certainly have learned to be.""",
        ArticleType.OPINION,
    ),
    (
        "https://www.guardian.com/commentisfree/12345",
        "The housing crisis won't solve itself – we need bold government action",
        """Young people today face a housing market that is fundamentally rigged
        against them. I should know – I'm 32, earn a decent salary, and still
        can't afford to buy a home in the city where I work.

        The free market has had decades to solve this problem. It hasn't. Prices
        have skyrocketed while wages have stagnated. Developers build luxury flats
        that sit empty while families crowd into substandard rentals.

        In my opinion, the only solution is massive government intervention. We
        need rent controls to protect existing tenants. We need public housing
        construction on a scale not seen since the postwar era. We need strict
        limits on foreign property speculation.

        Those who cry "socialism" at such proposals should consider: is the
        current system really working? For whom? The status quo is a choice, and
        we can choose differently.""",
        ArticleType.OPINION,
    ),
    # --- SATIRE ---
    (
        "https://www.theonion.com/article-12345",
        "Nation's Dog Owners Demand To Know Who's A Good Boy",
        """WASHINGTON—Gathering in unprecedented numbers on the National Mall,
        millions of dog owners from across the country demanded Wednesday that
        government officials finally reveal who's a good boy.

        "We've been asking this question for years, and we deserve answers," said
        protest organizer Karen Mitchell, 43, of Topeka, KS, while repeatedly
        scratching behind her golden retriever's ears. "Who's a good boy? Is it
        Max? Is Max a good boy? We need to know."

        The massive demonstration, which saw participants waving signs reading
        "Tell Us Who's A Good Boy" and "Yes You Are, Yes You Are," marked the
        largest single-issue protest in the nation's history.

        At press time, sources confirmed that it was in fact Max who was a good
        boy, yes he was, yes he was.""",
        ArticleType.SATIRE,
    ),
    # --- PR / PRESS RELEASE ---
    (
        "https://www.prnewswire.com/news-release-12345",
        "TechCorp Announces Revolutionary AI Platform Set to Transform Industries",
        """SAN FRANCISCO, Jan. 15, 2024 /PRNewswire/ -- TechCorp, the global leader
        in enterprise software solutions, today announced the launch of TechCorp AI,
        a groundbreaking artificial intelligence platform that promises to
        revolutionize how businesses operate.

        "TechCorp AI represents the culmination of five years of research and
        development by our world-class team of scientists," said CEO John Smith.
        "We believe this will be the most important product launch in our
        company's 25-year history."

        TechCorp AI leverages proprietary machine learning algorithms to deliver
        unprecedented accuracy and efficiency gains. Early adopters have reported
        productivity improvements of up to 300%.

        "TechCorp has once again demonstrated why they are the industry leader,"
        said industry analyst Mary Johnson of Research Group Inc.

        TechCorp AI is available immediately for enterprise customers. For more
        information, visit www.techcorp.com/ai.

        About TechCorp: TechCorp is the world's leading provider of enterprise
        software solutions, serving over 10,000 customers in 50 countries.""",
        ArticleType.PR,
    ),
]


# =============================================================================
# CSV Management
# =============================================================================


def create_golden_csv(filepath: Path) -> None:
    """
    Create the golden dataset CSV file.

    Args:
        filepath: Path to write the CSV file
    """
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "title", "text_snippet", "expected_label"])
        for url, title, text_snippet, expected_label in GOLDEN_DATASET:
            writer.writerow([url, title, text_snippet, expected_label.value])

    print(f"Created golden dataset: {filepath}")
    print(f"Total samples: {len(GOLDEN_DATASET)}")


def load_golden_csv(filepath: Path) -> list[tuple[str, str, str, ArticleType]]:
    """
    Load the golden dataset from CSV.

    Args:
        filepath: Path to the CSV file

    Returns:
        List of (url, title, text_snippet, expected_label) tuples
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected_label = ArticleType(row["expected_label"])
            data.append(
                (row["url"], row["title"], row["text_snippet"], expected_label)
            )
    return data


# =============================================================================
# Validation Logic
# =============================================================================


def run_validation(
    filepath: Path, verbose: bool = False
) -> ValidationReport:
    """
    Run the OpinionAnalyzer against the golden dataset.

    Args:
        filepath: Path to the golden dataset CSV
        verbose: Whether to print detailed results

    Returns:
        ValidationReport with accuracy and detailed results
    """
    # Load dataset
    dataset = load_golden_csv(filepath)

    # Initialize analyzer
    analyzer = OpinionAnalyzer()

    results: list[ValidationResult] = []
    mismatches: list[ValidationResult] = []

    print("\n" + "=" * 70)
    print("RUNNING VALIDATION")
    print("=" * 70)

    for i, (url, title, text_snippet, expected_label) in enumerate(dataset, 1):
        print(f"\nProcessing {i}/{len(dataset)}: {title[:50]}...")

        # Run classification
        classification = analyzer.analyze(title, text_snippet)

        # Create result
        is_correct = classification.article_type == expected_label
        result = ValidationResult(
            url=url,
            expected=expected_label,
            predicted=classification.article_type,
            confidence=classification.confidence,
            is_correct=is_correct,
            reasoning=classification.reasoning,
        )
        results.append(result)

        if not is_correct:
            mismatches.append(result)

        # Print status
        status = "✓" if is_correct else "✗"
        print(f"  {status} Expected: {expected_label.value}, "
              f"Predicted: {classification.article_type.value} "
              f"(confidence: {classification.confidence:.2f})")

        if verbose or not is_correct:
            print(f"  Reasoning: {classification.reasoning}")

    # Calculate metrics
    correct_count = sum(1 for r in results if r.is_correct)
    total = len(results)
    accuracy = correct_count / total if total > 0 else 0.0

    return ValidationReport(
        total_samples=total,
        correct_count=correct_count,
        accuracy=accuracy,
        results=results,
        mismatches=mismatches,
    )


def print_classification_report(report: ValidationReport) -> None:
    """
    Print a detailed classification report.

    Args:
        report: ValidationReport to print
    """
    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(f"\nOverall Accuracy: {report.accuracy:.1%} "
          f"({report.correct_count}/{report.total_samples})")

    # Per-class breakdown
    print("\nPer-Class Results:")
    print("-" * 50)

    for article_type in ArticleType:
        type_results = [r for r in report.results if r.expected == article_type]
        if not type_results:
            continue

        correct = sum(1 for r in type_results if r.is_correct)
        total = len(type_results)
        accuracy = correct / total if total > 0 else 0.0

        print(f"  {article_type.value:10} | Accuracy: {accuracy:.1%} ({correct}/{total})")

    # Confusion matrix style output
    print("\nPrediction Distribution:")
    print("-" * 50)
    print(f"{'Expected':<12} | {'Predicted as...'}")
    print(f"{'':<12} | {'News':<8} {'Opinion':<8} {'Satire':<8} {'PR':<8}")
    print("-" * 50)

    for expected_type in ArticleType:
        expected_results = [r for r in report.results if r.expected == expected_type]
        if not expected_results:
            continue

        counts = {t: 0 for t in ArticleType}
        for r in expected_results:
            counts[r.predicted] += 1

        row = f"{expected_type.value:<12} |"
        for pred_type in ArticleType:
            count = counts[pred_type]
            marker = f" {count}" if count > 0 else " -"
            row += f" {marker:<7}"
        print(row)

    # Mismatches detail
    if report.mismatches:
        print("\n" + "=" * 70)
        print("MISMATCHES DETAIL")
        print("=" * 70)

        for i, mismatch in enumerate(report.mismatches, 1):
            print(f"\n{i}. {mismatch.url}")
            print(f"   Expected: {mismatch.expected.value}")
            print(f"   Predicted: {mismatch.predicted.value} "
                  f"(confidence: {mismatch.confidence:.2f})")
            print(f"   Reasoning: {mismatch.reasoning}")

    print("\n" + "=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main entry point for the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate OpinionAnalyzer against golden dataset"
    )
    parser.add_argument(
        "--recreate-csv",
        action="store_true",
        help="Force recreation of the golden dataset CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each article",
    )
    args = parser.parse_args()

    # Determine CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / CSV_FILENAME

    # Create or check CSV
    if args.recreate_csv or not csv_path.exists():
        create_golden_csv(csv_path)
    else:
        print(f"Using existing golden dataset: {csv_path}")

    # Run validation
    report = run_validation(csv_path, verbose=args.verbose)

    # Print report
    print_classification_report(report)

    # Exit with appropriate code
    if report.accuracy < 0.8:
        print("\n⚠️  WARNING: Accuracy below 80% threshold")
        sys.exit(1)
    else:
        print("\n✓ Validation passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
