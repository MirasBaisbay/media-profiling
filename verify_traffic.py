#!/usr/bin/env python3
"""
Traffic Analysis Verification Script

This script validates the TrafficLongevityAnalyzer against a golden dataset
and compares the hybrid Tranco + LLM approach against LLM-only mode.

Features:
- Tests domains that should be in Tranco (deterministic path)
- Tests domains likely NOT in Tranco (LLM fallback path)
- Compares hybrid vs LLM-only approaches
- Generates comprehensive accuracy and timing reports

Usage:
    python verify_traffic.py [--recreate-csv] [--compare-llm] [--verbose] [--limit N]

Options:
    --recreate-csv   Force recreation of the golden dataset CSV
    --compare-llm    Also run LLM-only mode for comparison (slower)
    --verbose        Print detailed results for each domain
    --limit N        Only test first N domains (for quick testing)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import our analyzers
from refactored_analyzers import TrafficLongevityAnalyzer
from schemas import TrafficData, TrafficSource, TrafficTier


# =============================================================================
# Data Classes for Results
# =============================================================================


@dataclass
class DomainTestResult:
    """Result of testing a single domain."""
    domain: str
    expected_tier: str
    expected_in_tranco: bool

    # Hybrid approach results
    hybrid_tier: str
    hybrid_source: str
    hybrid_tranco_rank: Optional[int]
    hybrid_confidence: float
    hybrid_time_ms: float
    hybrid_tier_match: bool
    hybrid_tranco_match: bool

    # LLM-only results (optional)
    llm_only_tier: Optional[str] = None
    llm_only_confidence: Optional[float] = None
    llm_only_time_ms: Optional[float] = None
    llm_only_tier_match: Optional[bool] = None

    # WHOIS data
    whois_success: bool = False
    domain_age_years: Optional[float] = None
    whois_error: Optional[str] = None

    notes: str = ""


@dataclass
class VerificationReport:
    """Complete verification report."""
    timestamp: str
    total_domains: int

    # Hybrid approach metrics
    hybrid_tier_accuracy: float
    hybrid_tranco_detection_accuracy: float
    hybrid_avg_time_ms: float
    hybrid_tranco_hits: int
    hybrid_llm_fallbacks: int

    # LLM-only metrics (optional)
    llm_only_tier_accuracy: Optional[float] = None
    llm_only_avg_time_ms: Optional[float] = None

    # Breakdown by expected tier
    tier_breakdown: dict = field(default_factory=dict)

    # Tranco stats
    tranco_loaded: bool = False
    tranco_total_domains: int = 0

    # Individual results
    results: list = field(default_factory=list)

    # Errors
    errors: list = field(default_factory=list)


# =============================================================================
# Verification Logic
# =============================================================================


def load_gold_standard(filepath: Path) -> list[dict]:
    """Load the golden dataset from CSV."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "domain": row["domain"].strip(),
                "expected_tier": row["expected_tier"].strip(),
                "expected_in_tranco": row["expected_in_tranco"].strip().lower() == "true",
                "notes": row.get("notes", "").strip(),
            })
    return data


def tier_matches(expected: str, actual: str) -> bool:
    """Check if tiers match (case-insensitive)."""
    return expected.lower() == actual.lower()


def run_hybrid_test(
    analyzer: TrafficLongevityAnalyzer,
    domain: str
) -> tuple[TrafficData, float]:
    """
    Run hybrid (Tranco + LLM) analysis on a domain.

    Returns:
        Tuple of (TrafficData, time_in_ms)
    """
    start = time.perf_counter()
    result = analyzer.analyze(domain)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def run_llm_only_test(
    domain: str,
    model: str = "gpt-4o-mini"
) -> tuple[TrafficData, float]:
    """
    Run LLM-only analysis (no Tranco) on a domain.

    Returns:
        Tuple of (TrafficData, time_in_ms)
    """
    # Create analyzer with Tranco disabled
    analyzer = TrafficLongevityAnalyzer(
        model=model,
        auto_download_tranco=False,
        tranco_path="/nonexistent/path/to/disable/tranco.csv"
    )

    start = time.perf_counter()
    result = analyzer.analyze(domain)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def run_verification(
    gold_standard_path: Path,
    compare_llm: bool = False,
    verbose: bool = False,
    limit: Optional[int] = None
) -> VerificationReport:
    """
    Run the full verification suite.

    Args:
        gold_standard_path: Path to golden dataset CSV
        compare_llm: Whether to also run LLM-only comparison
        verbose: Print detailed results
        limit: Limit number of domains to test

    Returns:
        VerificationReport with all metrics
    """
    # Load gold standard
    gold_data = load_gold_standard(gold_standard_path)
    if limit:
        gold_data = gold_data[:limit]

    print("\n" + "=" * 70)
    print("TRAFFIC ANALYZER VERIFICATION")
    print("=" * 70)
    print(f"Gold Standard: {gold_standard_path}")
    print(f"Total Domains: {len(gold_data)}")
    print(f"Compare LLM-only: {compare_llm}")
    print("=" * 70)

    # Initialize hybrid analyzer
    print("\nInitializing hybrid analyzer (Tranco + LLM)...")
    hybrid_analyzer = TrafficLongevityAnalyzer(auto_download_tranco=True)
    tranco_stats = hybrid_analyzer.get_tranco_stats()

    print(f"  Tranco Loaded: {tranco_stats['loaded']}")
    print(f"  Tranco Domains: {tranco_stats['total_domains']:,}")

    results: list[DomainTestResult] = []
    errors: list[str] = []

    # Test each domain
    print(f"\nTesting {len(gold_data)} domains...")
    print("-" * 70)

    for i, entry in enumerate(gold_data, 1):
        domain = entry["domain"]
        expected_tier = entry["expected_tier"]
        expected_in_tranco = entry["expected_in_tranco"]
        notes = entry["notes"]

        print(f"[{i}/{len(gold_data)}] Testing: {domain}...", end=" ", flush=True)

        try:
            # Run hybrid test
            hybrid_result, hybrid_time = run_hybrid_test(hybrid_analyzer, domain)

            # Check matches
            hybrid_tier_match = tier_matches(expected_tier, hybrid_result.traffic_tier.value)
            hybrid_tranco_match = (
                (expected_in_tranco and hybrid_result.traffic_source == TrafficSource.TRANCO) or
                (not expected_in_tranco and hybrid_result.traffic_source != TrafficSource.TRANCO)
            )

            # Create result
            result = DomainTestResult(
                domain=domain,
                expected_tier=expected_tier,
                expected_in_tranco=expected_in_tranco,
                hybrid_tier=hybrid_result.traffic_tier.value,
                hybrid_source=hybrid_result.traffic_source.value,
                hybrid_tranco_rank=hybrid_result.tranco_rank,
                hybrid_confidence=hybrid_result.traffic_confidence,
                hybrid_time_ms=hybrid_time,
                hybrid_tier_match=hybrid_tier_match,
                hybrid_tranco_match=hybrid_tranco_match,
                whois_success=hybrid_result.whois_success,
                domain_age_years=hybrid_result.age_years,
                whois_error=hybrid_result.whois_error,
                notes=notes,
            )

            # Run LLM-only test if requested
            if compare_llm:
                print("(LLM)...", end=" ", flush=True)
                llm_result, llm_time = run_llm_only_test(domain)
                result.llm_only_tier = llm_result.traffic_tier.value
                result.llm_only_confidence = llm_result.traffic_confidence
                result.llm_only_time_ms = llm_time
                result.llm_only_tier_match = tier_matches(expected_tier, llm_result.traffic_tier.value)

            results.append(result)

            # Print status
            tier_status = "✓" if hybrid_tier_match else "✗"
            tranco_status = "✓" if hybrid_tranco_match else "✗"
            print(f"Tier:{tier_status} Tranco:{tranco_status} "
                  f"[{hybrid_result.traffic_source.value}:{hybrid_result.traffic_tier.value}] "
                  f"({hybrid_time:.0f}ms)")

            if verbose and not (hybrid_tier_match and hybrid_tranco_match):
                print(f"      Expected: tier={expected_tier}, in_tranco={expected_in_tranco}")
                print(f"      Got: tier={hybrid_result.traffic_tier.value}, "
                      f"source={hybrid_result.traffic_source.value}, "
                      f"rank={hybrid_result.tranco_rank}")

        except Exception as e:
            error_msg = f"Error testing {domain}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            print(f"ERROR: {e}")

    # Calculate metrics
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)

    total = len(results)

    # Hybrid metrics
    hybrid_tier_correct = sum(1 for r in results if r.hybrid_tier_match)
    hybrid_tranco_correct = sum(1 for r in results if r.hybrid_tranco_match)
    hybrid_tranco_hits = sum(1 for r in results if r.hybrid_source == "Tranco")
    hybrid_llm_fallbacks = sum(1 for r in results if r.hybrid_source == "LLM")
    hybrid_avg_time = sum(r.hybrid_time_ms for r in results) / total if total > 0 else 0

    # LLM-only metrics
    llm_tier_accuracy = None
    llm_avg_time = None
    if compare_llm:
        llm_tier_correct = sum(1 for r in results if r.llm_only_tier_match)
        llm_tier_accuracy = llm_tier_correct / total if total > 0 else 0
        llm_avg_time = sum(r.llm_only_time_ms for r in results if r.llm_only_time_ms) / total

    # Tier breakdown
    tier_breakdown = {}
    for tier in ["High", "Medium", "Low", "Minimal"]:
        tier_results = [r for r in results if r.expected_tier.lower() == tier.lower()]
        if tier_results:
            correct = sum(1 for r in tier_results if r.hybrid_tier_match)
            tier_breakdown[tier] = {
                "total": len(tier_results),
                "correct": correct,
                "accuracy": correct / len(tier_results),
                "tranco_hits": sum(1 for r in tier_results if r.hybrid_source == "Tranco"),
            }

    # Create report
    report = VerificationReport(
        timestamp=datetime.now().isoformat(),
        total_domains=total,
        hybrid_tier_accuracy=hybrid_tier_correct / total if total > 0 else 0,
        hybrid_tranco_detection_accuracy=hybrid_tranco_correct / total if total > 0 else 0,
        hybrid_avg_time_ms=hybrid_avg_time,
        hybrid_tranco_hits=hybrid_tranco_hits,
        hybrid_llm_fallbacks=hybrid_llm_fallbacks,
        llm_only_tier_accuracy=llm_tier_accuracy,
        llm_only_avg_time_ms=llm_avg_time,
        tier_breakdown=tier_breakdown,
        tranco_loaded=tranco_stats["loaded"],
        tranco_total_domains=tranco_stats["total_domains"],
        results=results,
        errors=errors,
    )

    return report


def print_report(report: VerificationReport) -> None:
    """Print a comprehensive verification report."""

    print("\n" + "=" * 70)
    print("VERIFICATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print(f"Total Domains Tested: {report.total_domains}")

    # Tranco stats
    print(f"\nTranco List:")
    print(f"  Loaded: {report.tranco_loaded}")
    print(f"  Total Domains: {report.tranco_total_domains:,}")

    # Hybrid approach results
    print(f"\n{'─' * 70}")
    print("HYBRID APPROACH (Tranco + LLM Fallback)")
    print(f"{'─' * 70}")
    print(f"  Tier Accuracy:           {report.hybrid_tier_accuracy:.1%} "
          f"({int(report.hybrid_tier_accuracy * report.total_domains)}/{report.total_domains})")
    print(f"  Tranco Detection Acc:    {report.hybrid_tranco_detection_accuracy:.1%}")
    print(f"  Tranco Hits:             {report.hybrid_tranco_hits} "
          f"({report.hybrid_tranco_hits/report.total_domains:.1%})")
    print(f"  LLM Fallbacks:           {report.hybrid_llm_fallbacks} "
          f"({report.hybrid_llm_fallbacks/report.total_domains:.1%})")
    print(f"  Average Time:            {report.hybrid_avg_time_ms:.0f}ms")

    # LLM-only results (if available)
    if report.llm_only_tier_accuracy is not None:
        print(f"\n{'─' * 70}")
        print("LLM-ONLY APPROACH (No Tranco)")
        print(f"{'─' * 70}")
        print(f"  Tier Accuracy:           {report.llm_only_tier_accuracy:.1%}")
        print(f"  Average Time:            {report.llm_only_avg_time_ms:.0f}ms")

        # Comparison
        print(f"\n{'─' * 70}")
        print("COMPARISON: Hybrid vs LLM-Only")
        print(f"{'─' * 70}")
        accuracy_diff = report.hybrid_tier_accuracy - report.llm_only_tier_accuracy
        time_diff = report.hybrid_avg_time_ms - report.llm_only_avg_time_ms
        print(f"  Accuracy Difference:     {accuracy_diff:+.1%} "
              f"({'Hybrid better' if accuracy_diff > 0 else 'LLM better' if accuracy_diff < 0 else 'Same'})")
        print(f"  Time Difference:         {time_diff:+.0f}ms "
              f"({'Hybrid faster' if time_diff < 0 else 'LLM faster' if time_diff > 0 else 'Same'})")

    # Tier breakdown
    print(f"\n{'─' * 70}")
    print("BREAKDOWN BY EXPECTED TIER")
    print(f"{'─' * 70}")
    print(f"{'Tier':<10} {'Total':>8} {'Correct':>8} {'Accuracy':>10} {'Tranco':>8}")
    print("-" * 50)
    for tier, data in report.tier_breakdown.items():
        print(f"{tier:<10} {data['total']:>8} {data['correct']:>8} "
              f"{data['accuracy']:>9.1%} {data['tranco_hits']:>8}")

    # Mismatches
    mismatches = [r for r in report.results if not r.hybrid_tier_match]
    if mismatches:
        print(f"\n{'─' * 70}")
        print(f"TIER MISMATCHES ({len(mismatches)} total)")
        print(f"{'─' * 70}")
        for r in mismatches:
            print(f"\n  {r.domain}")
            print(f"    Expected: {r.expected_tier}")
            print(f"    Got:      {r.hybrid_tier} (source: {r.hybrid_source}, "
                  f"confidence: {r.hybrid_confidence:.2f})")
            if r.hybrid_tranco_rank:
                print(f"    Tranco Rank: #{r.hybrid_tranco_rank:,}")
            if r.notes:
                print(f"    Notes: {r.notes}")

    # Tranco detection mismatches
    tranco_mismatches = [r for r in report.results if not r.hybrid_tranco_match]
    if tranco_mismatches:
        print(f"\n{'─' * 70}")
        print(f"TRANCO DETECTION MISMATCHES ({len(tranco_mismatches)} total)")
        print(f"{'─' * 70}")
        for r in tranco_mismatches:
            expected_str = "in Tranco" if r.expected_in_tranco else "NOT in Tranco"
            actual_str = f"source={r.hybrid_source}"
            print(f"  {r.domain}: expected {expected_str}, got {actual_str}")

    # WHOIS analysis
    whois_success = sum(1 for r in report.results if r.whois_success)
    print(f"\n{'─' * 70}")
    print("WHOIS ANALYSIS")
    print(f"{'─' * 70}")
    print(f"  Success Rate: {whois_success}/{report.total_domains} "
          f"({whois_success/report.total_domains:.1%})")

    whois_failures = [r for r in report.results if not r.whois_success]
    if whois_failures:
        print(f"  Failures:")
        for r in whois_failures[:5]:  # Show first 5
            error = r.whois_error or "Unknown error"
            print(f"    - {r.domain}: {error[:60]}...")
        if len(whois_failures) > 5:
            print(f"    ... and {len(whois_failures) - 5} more")

    # Errors
    if report.errors:
        print(f"\n{'─' * 70}")
        print(f"ERRORS ({len(report.errors)} total)")
        print(f"{'─' * 70}")
        for error in report.errors:
            print(f"  - {error}")

    print("\n" + "=" * 70)


def save_report_json(report: VerificationReport, filepath: Path) -> None:
    """Save report as JSON for further analysis."""
    # Convert to serializable dict
    report_dict = {
        "timestamp": report.timestamp,
        "total_domains": report.total_domains,
        "hybrid_tier_accuracy": report.hybrid_tier_accuracy,
        "hybrid_tranco_detection_accuracy": report.hybrid_tranco_detection_accuracy,
        "hybrid_avg_time_ms": report.hybrid_avg_time_ms,
        "hybrid_tranco_hits": report.hybrid_tranco_hits,
        "hybrid_llm_fallbacks": report.hybrid_llm_fallbacks,
        "llm_only_tier_accuracy": report.llm_only_tier_accuracy,
        "llm_only_avg_time_ms": report.llm_only_avg_time_ms,
        "tier_breakdown": report.tier_breakdown,
        "tranco_loaded": report.tranco_loaded,
        "tranco_total_domains": report.tranco_total_domains,
        "results": [
            {
                "domain": r.domain,
                "expected_tier": r.expected_tier,
                "expected_in_tranco": r.expected_in_tranco,
                "hybrid_tier": r.hybrid_tier,
                "hybrid_source": r.hybrid_source,
                "hybrid_tranco_rank": r.hybrid_tranco_rank,
                "hybrid_confidence": r.hybrid_confidence,
                "hybrid_time_ms": r.hybrid_time_ms,
                "hybrid_tier_match": r.hybrid_tier_match,
                "hybrid_tranco_match": r.hybrid_tranco_match,
                "llm_only_tier": r.llm_only_tier,
                "llm_only_confidence": r.llm_only_confidence,
                "llm_only_time_ms": r.llm_only_time_ms,
                "llm_only_tier_match": r.llm_only_tier_match,
                "whois_success": r.whois_success,
                "domain_age_years": r.domain_age_years,
                "notes": r.notes,
            }
            for r in report.results
        ],
        "errors": report.errors,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    print(f"\nReport saved to: {filepath}")


# =============================================================================
# Main Entry Point
# =============================================================================


def run_tranco_only_verification(gold_standard_path: Path, limit: Optional[int] = None) -> dict:
    """
    Run Tranco-only verification (no LLM calls needed).

    This tests the deterministic Tranco lookup without requiring an API key.
    """
    from refactored_analyzers import (
        TrafficLongevityAnalyzer,
        TRANCO_DEFAULT_PATH,
        DEFAULT_TRANCO_THRESHOLDS
    )
    import os

    gold_data = load_gold_standard(gold_standard_path)
    if limit:
        gold_data = gold_data[:limit]

    print("\n" + "=" * 70)
    print("TRANCO-ONLY VERIFICATION (No API Key Required)")
    print("=" * 70)

    # Create a minimal analyzer just to load Tranco
    # We'll manually check Tranco without LLM
    tranco_data = {}
    tranco_loaded = False

    if os.path.exists(TRANCO_DEFAULT_PATH):
        print(f"Loading Tranco from {TRANCO_DEFAULT_PATH}...")
        try:
            with open(TRANCO_DEFAULT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        rank_str, domain = parts
                        try:
                            rank = int(rank_str)
                            tranco_data[domain.lower()] = rank
                        except ValueError:
                            continue
            tranco_loaded = True
            print(f"Loaded {len(tranco_data):,} domains from Tranco")
        except Exception as e:
            print(f"Failed to load Tranco: {e}")
    else:
        print(f"Tranco file not found at {TRANCO_DEFAULT_PATH}")
        print("Downloading Tranco list...")

        import urllib.request
        try:
            url = "https://tranco-list.eu/download/Q4PJN/1000000"
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()

            # Check if zip
            if content[:2] == b"PK":
                import zipfile
                import io
                with zipfile.ZipFile(io.BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if name.endswith(".csv"):
                            with zf.open(name) as csv_file:
                                content = csv_file.read()
                            break

            with open(TRANCO_DEFAULT_PATH, "wb") as f:
                f.write(content)

            # Now load it
            with open(TRANCO_DEFAULT_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or "," not in line:
                        continue
                    parts = line.split(",", 1)
                    if len(parts) == 2:
                        rank_str, domain = parts
                        try:
                            rank = int(rank_str)
                            tranco_data[domain.lower()] = rank
                        except ValueError:
                            continue
            tranco_loaded = True
            print(f"Downloaded and loaded {len(tranco_data):,} domains")
        except Exception as e:
            print(f"Failed to download Tranco: {e}")

    if not tranco_loaded:
        print("Cannot proceed without Tranco data")
        return {}

    # Test each domain
    results = []
    thresholds = DEFAULT_TRANCO_THRESHOLDS

    print(f"\nTesting {len(gold_data)} domains against Tranco...")
    print(f"Thresholds: HIGH < {thresholds['HIGH']:,}, MEDIUM < {thresholds['MEDIUM']:,}, LOW < {thresholds['LOW']:,}")
    print("-" * 70)

    for entry in gold_data:
        domain = entry["domain"].lower()
        expected_tier = entry["expected_tier"]
        expected_in_tranco = entry["expected_in_tranco"]

        # Look up in Tranco
        rank = tranco_data.get(domain) or tranco_data.get(f"www.{domain}")
        in_tranco = rank is not None

        # Determine tier from rank
        if rank is not None:
            if rank < thresholds["HIGH"]:
                actual_tier = "High"
            elif rank < thresholds["MEDIUM"]:
                actual_tier = "Medium"
            elif rank < thresholds["LOW"]:
                actual_tier = "Low"
            else:
                actual_tier = "Minimal"
        else:
            actual_tier = "Unknown (not in Tranco)"

        tier_match = (rank is not None and tier_matches(expected_tier, actual_tier))
        tranco_match = (expected_in_tranco == in_tranco)

        results.append({
            "domain": domain,
            "expected_tier": expected_tier,
            "expected_in_tranco": expected_in_tranco,
            "tranco_rank": rank,
            "actual_tier": actual_tier,
            "in_tranco": in_tranco,
            "tier_match": tier_match,
            "tranco_match": tranco_match,
        })

        # Print result
        tier_status = "✓" if tier_match else ("?" if not in_tranco else "✗")
        tranco_status = "✓" if tranco_match else "✗"
        rank_str = f"#{rank:,}" if rank else "N/A"
        print(f"  {domain:<35} Rank: {rank_str:<12} Tier: {actual_tier:<10} "
              f"[Tier:{tier_status} Tranco:{tranco_status}]")

    # Calculate metrics
    total = len(results)
    in_tranco_results = [r for r in results if r["in_tranco"]]

    tranco_detection_correct = sum(1 for r in results if r["tranco_match"])
    tier_correct_when_found = sum(1 for r in in_tranco_results if r["tier_match"])

    print("\n" + "=" * 70)
    print("TRANCO-ONLY RESULTS")
    print("=" * 70)
    print(f"Total Domains Tested:     {total}")
    print(f"Found in Tranco:          {len(in_tranco_results)} ({len(in_tranco_results)/total:.1%})")
    print(f"Tranco Detection Accuracy: {tranco_detection_correct}/{total} ({tranco_detection_correct/total:.1%})")
    if in_tranco_results:
        print(f"Tier Accuracy (when found): {tier_correct_when_found}/{len(in_tranco_results)} "
              f"({tier_correct_when_found/len(in_tranco_results):.1%})")

    # Show mismatches
    tier_mismatches = [r for r in in_tranco_results if not r["tier_match"]]
    if tier_mismatches:
        print(f"\nTier Mismatches ({len(tier_mismatches)}):")
        for r in tier_mismatches:
            print(f"  {r['domain']}: expected {r['expected_tier']}, got {r['actual_tier']} (rank #{r['tranco_rank']:,})")

    tranco_mismatches = [r for r in results if not r["tranco_match"]]
    if tranco_mismatches:
        print(f"\nTranco Detection Mismatches ({len(tranco_mismatches)}):")
        for r in tranco_mismatches:
            expected = "in Tranco" if r["expected_in_tranco"] else "NOT in Tranco"
            actual = f"rank #{r['tranco_rank']:,}" if r["in_tranco"] else "not found"
            print(f"  {r['domain']}: expected {expected}, actually {actual}")

    return {
        "total": total,
        "found_in_tranco": len(in_tranco_results),
        "tranco_detection_accuracy": tranco_detection_correct / total,
        "tier_accuracy_when_found": tier_correct_when_found / len(in_tranco_results) if in_tranco_results else 0,
        "results": results,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify TrafficLongevityAnalyzer against golden dataset"
    )
    parser.add_argument(
        "--recreate-csv",
        action="store_true",
        help="Force recreation of the golden dataset CSV (not implemented - edit CSV manually)",
    )
    parser.add_argument(
        "--compare-llm",
        action="store_true",
        help="Also run LLM-only mode for comparison (slower)",
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
        help="Limit number of domains to test (for quick testing)",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save report as JSON to specified path",
    )
    parser.add_argument(
        "--tranco-only",
        action="store_true",
        help="Run Tranco-only verification (no API key required)",
    )
    args = parser.parse_args()

    # Determine CSV path
    script_dir = Path(__file__).parent
    csv_path = script_dir / "traffic_gold_standard.csv"

    if not csv_path.exists():
        print(f"ERROR: Golden dataset not found at {csv_path}")
        print("Please create the CSV file first.")
        sys.exit(1)

    # Run Tranco-only verification if requested (no API key needed)
    if args.tranco_only:
        result = run_tranco_only_verification(csv_path, limit=args.limit)
        if result:
            print("\n✓ Tranco-only verification completed")
            sys.exit(0)
        else:
            print("\n✗ Tranco-only verification failed")
            sys.exit(1)

    # Run full verification (requires API key)
    report = run_verification(
        gold_standard_path=csv_path,
        compare_llm=args.compare_llm,
        verbose=args.verbose,
        limit=args.limit,
    )

    # Print report
    print_report(report)

    # Save JSON if requested
    if args.save_json:
        save_report_json(report, Path(args.save_json))
    else:
        # Auto-save to default location
        json_path = script_dir / "traffic_verification_report.json"
        save_report_json(report, json_path)

    # Exit with appropriate code
    if report.hybrid_tier_accuracy < 0.6:
        print("\n⚠️  WARNING: Tier accuracy below 60% threshold")
        sys.exit(1)
    else:
        print("\n✓ Verification completed")
        sys.exit(0)


if __name__ == "__main__":
    main()
