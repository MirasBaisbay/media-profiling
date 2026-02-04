"""
Media Profiler - MBFC Methodology Compliant Orchestrator
Uses LangGraph for workflow orchestration with Human-in-the-Loop verification.

Two-Phase Architecture:
  PHASE 1: SCRAPE → ANALYZE → SCORE (Quick scoring)
  PHASE 2: RESEARCH → WRITE_REPORT (Comprehensive report with evidence)
"""
import sys
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from scraper import MediaScraper, Article
from analyzers import (
    # Bias Analyzers (4 components)
    EconomicAnalyzer,           # 35% of bias score
    SocialAnalyzer,             # 35% of bias score
    NewsReportingBalanceAnalyzer,  # 15% of bias score (NEW)
    EditorialBiasAnalyzer,      # 15% of bias score (NEW)
    # Factuality Analyzers (4 components)
    FactCheckSearcher,          # 40% of factuality score (NEW)
    SourcingAnalyzer,           # 25% of factuality score (NEW)
    TransparencyAnalyzer,       # 25% of factuality score
    PropagandaAnalyzer,         # 10% of factuality score
    # Supporting Analyzers
    MediaTypeAnalyzer,
    CountryFreedomAnalyzer,
    TrafficLongevityAnalyzer,
    # Scoring
    ScoringCalculator
)
from research import MediaResearcher
from report_writer import MBFCReportWriter, write_report_node
from evidence import AnalyzerOutput, create_analyzer_output

logging.basicConfig(level=logging.INFO, format='%(message)s')


class ProfilerState(TypedDict):
    target_url: str
    country_code: str
    use_local_model: bool  # Flag for local DeBERTa vs LLM propaganda detection
    full_report: bool      # Flag for comprehensive MBFC report generation
    articles: List[Article]
    metadata: Any

    # Bias Analysis Components (MBFC Methodology)
    economic_bias: float          # 35% weight - Economic position score
    social_bias: float            # 35% weight - Social position score
    news_reporting_bias: float    # 15% weight - Balance in news reporting
    editorial_bias: float         # 15% weight - Bias in opinion/editorial

    # Labels for each component
    economic_label: str
    social_label: str
    news_reporting_label: str
    editorial_label: str

    # Factuality Analysis Components (MBFC Methodology)
    fact_check_score: float       # 40% weight - Failed fact checks
    sourcing_score: float         # 25% weight - Quality of sources
    transparency_score: float     # 25% weight - Ownership/funding disclosure
    propaganda_score: float       # 10% weight - Propaganda techniques detected

    # Supporting Data
    media_type: str
    freedom_data: Dict
    traffic_data: Dict
    propaganda_analysis: Any  # Full analysis object for human review

    # Evidence for comprehensive reports
    economic_evidence: AnalyzerOutput
    social_evidence: AnalyzerOutput
    news_reporting_evidence: AnalyzerOutput
    editorial_evidence: AnalyzerOutput
    fact_check_evidence: AnalyzerOutput
    sourcing_evidence: AnalyzerOutput
    transparency_evidence: AnalyzerOutput
    propaganda_evidence: AnalyzerOutput

    # Calculated scores (from score_node)
    bias_score: float
    bias_label: str
    factuality_score: float
    factuality_label: str
    credibility_score: float
    credibility_label: str

    # Research results (from research_node)
    research_results: Dict

    final_report: str


def scrape_node(state: ProfilerState) -> Dict:
    """
    Scrapes target website for articles and metadata.
    Separates opinion/editorial articles from straight news for proper MBFC analysis.
    """
    scraper = MediaScraper(state["target_url"], max_articles=20)
    articles = scraper.scrape_feed()

    if not articles:
        articles = [Article(state["target_url"], "Error", "No content could be scraped")]

    # Log article types for debugging
    news_count = sum(1 for a in articles if not a.is_opinion)
    opinion_count = sum(1 for a in articles if a.is_opinion)
    logging.info(f"Scraped {len(articles)} articles: {news_count} news, {opinion_count} opinion/editorial")

    return {"articles": articles, "metadata": scraper.get_metadata()}


def analyze_node(state: ProfilerState) -> Dict:
    """
    Runs all MBFC-compliant analyzers.

    Bias Score = Economic (35%) + Social (35%) + News Reporting (15%) + Editorial (15%)
    Fact Score = Failed Fact Checks (40%) + Sourcing (25%) + Transparency (25%) + Propaganda (10%)
    """
    articles = state["articles"]
    meta = state["metadata"]
    url = state["target_url"]
    cc = state.get("country_code", "US")
    use_local = state.get("use_local_model", False)

    # Separate articles by type for proper MBFC analysis
    news_articles = [a for a in articles if not a.is_opinion]
    opinion_articles = [a for a in articles if a.is_opinion]

    logging.info(f"Analyzing {len(news_articles)} news articles and {len(opinion_articles)} opinion pieces...")

    # === BIAS ANALYSIS (4 Components) ===

    # 1. Economic Position (35%) - Analyzed from all articles
    economic_result = EconomicAnalyzer().analyze(articles)

    # 2. Social Position (35%) - Analyzed from all articles
    social_result = SocialAnalyzer().analyze(articles)

    # 3. News Reporting Balance (15%) - Only from straight news articles
    news_reporting_result = NewsReportingBalanceAnalyzer().analyze(news_articles if news_articles else articles)

    # 4. Editorial Bias (15%) - Only from opinion/editorial articles
    editorial_result = EditorialBiasAnalyzer().analyze(opinion_articles if opinion_articles else articles)

    # === FACTUALITY ANALYSIS (4 Components) ===

    # 1. Failed Fact Checks (40%) - Web search for fact check failures
    fact_check_result = FactCheckSearcher().analyze(url)

    # 2. Sourcing Quality (25%) - Analyzed from article source links
    sourcing_result = SourcingAnalyzer().analyze(articles)

    # 3. Transparency (25%) - From site metadata
    transparency_result = TransparencyAnalyzer().analyze(meta)

    # 4. Propaganda Detection (10%) - Using local DeBERTa or LLM
    propaganda_analyzer = PropagandaAnalyzer(use_local_model=use_local)
    propaganda_result = propaganda_analyzer.analyze(articles)

    # === SUPPORTING DATA ===
    media_type = MediaTypeAnalyzer().analyze(url, articles)
    freedom_data = CountryFreedomAnalyzer().analyze(cc)
    traffic_data = TrafficLongevityAnalyzer().analyze(url)

    # Build evidence objects for comprehensive reports
    economic_evidence = create_analyzer_output(
        "EconomicAnalyzer", economic_result["score"], economic_result["label"],
        methodology="LLM-based economic ideology classification"
    )
    social_evidence = create_analyzer_output(
        "SocialAnalyzer", social_result["score"], social_result["label"],
        methodology="LLM-based social values classification"
    )
    news_reporting_evidence = create_analyzer_output(
        "NewsReportingBalanceAnalyzer",
        news_reporting_result["score"],
        news_reporting_result["label"],
        methodology="3-component structured approach (selection + framing + sourcing)",
        **news_reporting_result.get("details", {})
    )
    editorial_evidence = create_analyzer_output(
        "EditorialBiasAnalyzer",
        editorial_result["score"],
        editorial_result["label"],
        methodology="Clickbait + loaded language + LLM verification",
        **editorial_result.get("details", {})
    )
    fact_check_evidence = create_analyzer_output(
        "FactCheckSearcher", fact_check_result, "",
        methodology="IFCN fact-checker search + LLM estimation"
    )
    sourcing_evidence_obj = create_analyzer_output(
        "SourcingAnalyzer", sourcing_result.score, "",
        methodology="Hyperlink analysis + credible source ratio",
        avg_sources=sourcing_result.avg_sources_per_article,
        credible_ratio=sourcing_result.credible_source_ratio
    )
    propaganda_evidence = create_analyzer_output(
        "PropagandaAnalyzer", propaganda_result.score, "",
        methodology="DeBERTa SI+TC pipeline" if use_local else "LLM-based detection"
    )

    return {
        # Bias components (raw scores on -10 to +10 scale)
        "economic_bias": economic_result["score"],
        "social_bias": social_result["score"],
        "news_reporting_bias": news_reporting_result["score"],
        "editorial_bias": editorial_result["score"],
        # Labels for each component
        "economic_label": economic_result["label"],
        "social_label": social_result["label"],
        "news_reporting_label": news_reporting_result["label"],
        "editorial_label": editorial_result["label"],
        # Factuality components (raw scores on 0-10 scale, lower is better)
        "fact_check_score": fact_check_result,
        "sourcing_score": sourcing_result.score,
        "transparency_score": transparency_result,
        "propaganda_score": propaganda_result.score,
        # Full propaganda analysis for human review
        "propaganda_analysis": propaganda_result,
        # Supporting data
        "media_type": media_type,
        "freedom_data": freedom_data,
        "traffic_data": traffic_data,
        # Evidence for comprehensive reports
        "economic_evidence": economic_evidence,
        "social_evidence": social_evidence,
        "news_reporting_evidence": news_reporting_evidence,
        "editorial_evidence": editorial_evidence,
        "fact_check_evidence": fact_check_evidence,
        "sourcing_evidence": sourcing_evidence_obj,
        "propaganda_evidence": propaganda_evidence
    }


def score_node(state: ProfilerState) -> Dict:
    """
    Calculates weighted scores from analyzer outputs.
    Extracts scoring logic into its own step for cleaner workflow.
    """
    # === Calculate Weighted Bias Score ===
    bias_analysis = ScoringCalculator.calculate_bias(
        economic_score=state["economic_bias"],
        social_score=state["social_bias"],
        news_reporting_score=state["news_reporting_bias"],
        editorial_score=state["editorial_bias"]
    )

    # === Calculate Weighted Factuality Score ===
    fact_analysis = ScoringCalculator.calculate_factuality(
        fact_check_score=state["fact_check_score"],
        sourcing_score=state["sourcing_score"],
        transparency_score=state["transparency_score"],
        propaganda_score=state["propaganda_score"]
    )

    # === Calculate Overall Credibility (0-10 scale) ===
    cred_score, cred_level, _, _ = ScoringCalculator.calculate_credibility(
        factuality_label=fact_analysis.final_label,
        bias_label=bias_analysis.final_label,
        traffic_points=state["traffic_data"]["points"],
        freedom_penalty=state["freedom_data"]["penalty"]
    )

    return {
        "bias_score": bias_analysis.weighted_total,
        "bias_label": bias_analysis.final_label,
        "factuality_score": fact_analysis.weighted_total,
        "factuality_label": fact_analysis.final_label,
        "credibility_score": cred_score,
        "credibility_label": cred_level
    }


def research_node(state: ProfilerState) -> Dict:
    """
    Gathers external context about the media outlet for comprehensive reports.
    Performs 2-3 targeted web searches for history, ownership, and criticism.
    """
    logging.info("Researching outlet for comprehensive report...")

    researcher = MediaResearcher()
    from dataclasses import asdict

    results = researcher.research_outlet(
        url=state["target_url"],
        outlet_name=None  # Will be extracted from URL
    )

    return {
        "research_results": asdict(results)
    }


def human_review_process(state: ProfilerState) -> ProfilerState:
    """
    Human-in-the-loop verification for propaganda detection.
    Allows human reviewer to verify, correct, or reject AI-detected propaganda instances.
    """
    analysis = state["propaganda_analysis"]
    instances = analysis.instances
    verified = []

    print("\n" + "="*60)
    print("  HUMAN VERIFICATION: PROPAGANDA DETECTION")
    print("="*60)

    if not instances:
        print("No propaganda instances detected by the model.")
        return state

    print(f"\nFound {len(instances)} potential propaganda instances to review.\n")

    for i, inst in enumerate(instances, 1):
        print(f"\n--- Finding #{i} of {len(instances)} ---")
        print(f"Context: \"{inst.context[:200]}...\"" if len(inst.context) > 200 else f"Context: \"{inst.context}\"")
        print(f"Snippet: \"{inst.text_snippet}\"")
        print(f"AI Classification: [{inst.technique}] (Confidence: {inst.confidence:.0%})")

        choice = input("\nIs this correct? (y=yes / n=no, correct it / s=skip / q=quit review): ").lower().strip()

        if choice == 'q':
            print("Exiting review early...")
            break
        elif choice == 'y':
            inst.verified = True
            verified.append(inst)
            print("  Verified.")
        elif choice == 'n':
            from config import PROPAGANDA_TECHNIQUES
            print("\nAvailable techniques:")
            for idx, tech in enumerate(PROPAGANDA_TECHNIQUES):
                print(f"  {idx}: {tech}")
            print("  x: Reject (not propaganda)")

            edit = input("Enter technique number or 'x' to reject: ").strip()
            if edit.isdigit() and int(edit) < len(PROPAGANDA_TECHNIQUES):
                inst.technique = PROPAGANDA_TECHNIQUES[int(edit)]
                inst.verified = True
                verified.append(inst)
                print(f"  Corrected to: {inst.technique}")
            elif edit.lower() == 'x':
                print("  Rejected.")
        else:
            print("  Skipped.")

    # Recalculate propaganda score based on verified instances
    new_score = min(10.0, len(verified) * 2.0)
    print(f"\n{'='*60}")
    print(f"Review Complete: {len(verified)} verified instances")
    print(f"Updated Propaganda Score: {new_score}/10")
    print(f"{'='*60}")

    analysis.instances = verified
    analysis.score = new_score
    state["propaganda_analysis"] = analysis
    state["propaganda_score"] = new_score

    return state


def report_node(state: ProfilerState) -> Dict:
    """
    Generates simplified MBFC report (quick mode).
    For comprehensive reports, use full_report_node instead.
    """
    from urllib.parse import urlparse
    from config import COUNTRY_NAMES

    # Get scores (already calculated in score_node)
    bias_score = state.get("bias_score", 0)
    bias_label = state.get("bias_label", "Least Biased")
    fact_score = state.get("factuality_score", 0)
    fact_label = state.get("factuality_label", "High")
    cred_level = state.get("credibility_label", "High Credibility")

    # === Extract domain for report ===
    domain = urlparse(state['target_url']).netloc.replace('www.', '')

    # Get country name
    country_name = COUNTRY_NAMES.get(state['country_code'], state['country_code'])

    # === Generate Simplified Report ===
    report = f"""
Detailed Report for {domain}
Bias Rating: {bias_label.upper()} ({bias_score:+.1f})
Factual Reporting: {fact_label.upper()} ({fact_score:.1f})
Country: {country_name}
MBFC's Country Freedom Rating: {state['freedom_data']['rating'].upper()}
Media Type: {state['media_type']}
Traffic/Popularity: {state['traffic_data']['traffic_label']} Traffic
MBFC Credibility Rating: {cred_level.upper()}
"""
    return {"final_report": report}


def full_report_node(state: ProfilerState) -> Dict:
    """
    Generates comprehensive MBFC-style prose report with evidence.
    Includes history, ownership, analysis with citations, and examples.
    """
    from urllib.parse import urlparse
    from config import COUNTRY_NAMES
    from evidence import ComprehensiveReportData

    logging.info("Generating comprehensive MBFC report...")

    # Build report data
    parsed = urlparse(state["target_url"])
    domain = parsed.netloc.replace('www.', '')

    report_data = ComprehensiveReportData(
        target_url=state["target_url"],
        target_domain=domain,
        country_code=state.get("country_code", ""),
        country_name=COUNTRY_NAMES.get(state.get("country_code", ""), state.get("country_code", "")),

        # Scores
        bias_score=state.get("bias_score", 0),
        bias_label=state.get("bias_label", "Least Biased"),
        factuality_score=state.get("factuality_score", 0),
        factuality_label=state.get("factuality_label", "High"),
        credibility_score=state.get("credibility_score", 0),
        credibility_label=state.get("credibility_label", "High Credibility"),

        # Component scores
        economic_score=state.get("economic_bias", 0),
        economic_label=state.get("economic_label", "Centrism"),
        social_score=state.get("social_bias", 0),
        social_label=state.get("social_label", "Balanced"),
        news_reporting_score=state.get("news_reporting_bias", 0),
        news_reporting_label=state.get("news_reporting_label", "Neutral/Balanced"),
        editorial_score=state.get("editorial_bias", 0),
        editorial_label=state.get("editorial_label", "Neutral/Balanced Editorial"),

        fact_check_score=state.get("fact_check_score", 0),
        sourcing_score=state.get("sourcing_score", 0),
        transparency_score=state.get("transparency_score", 0),
        propaganda_score=state.get("propaganda_score", 0),

        # Supporting data
        media_type=state.get("media_type", "News Website"),
        traffic_label=state.get("traffic_data", {}).get("traffic_label", "Medium"),
        freedom_rating=state.get("freedom_data", {}).get("rating", "Unknown"),
        freedom_score=state.get("freedom_data", {}).get("score", 0),

        # Evidence
        economic_evidence=state.get("economic_evidence"),
        social_evidence=state.get("social_evidence"),
        news_reporting_evidence=state.get("news_reporting_evidence"),
        editorial_evidence=state.get("editorial_evidence"),
        fact_check_evidence=state.get("fact_check_evidence"),
        sourcing_evidence=state.get("sourcing_evidence"),
        propaganda_evidence=state.get("propaganda_evidence"),

        # Research
        research=state.get("research_results"),

        # Metadata
        articles_analyzed=len(state.get("articles", [])),
        news_articles_count=len([a for a in state.get("articles", []) if not getattr(a, 'is_opinion', False)]),
        opinion_articles_count=len([a for a in state.get("articles", []) if getattr(a, 'is_opinion', False)])
    )

    # Generate report using MBFCReportWriter
    writer = MBFCReportWriter()
    report = writer.generate_report(report_data)

    return {"final_report": report}


# === Build LangGraph Workflows ===

def build_quick_workflow():
    """Build workflow for quick analysis (simplified report)."""
    workflow = StateGraph(ProfilerState)
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("score", score_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "analyze")
    workflow.add_edge("analyze", "score")
    workflow.add_edge("score", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


def build_full_workflow():
    """Build workflow for comprehensive analysis (full MBFC-style report)."""
    workflow = StateGraph(ProfilerState)
    workflow.add_node("scrape", scrape_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("score", score_node)
    workflow.add_node("research", research_node)
    workflow.add_node("full_report", full_report_node)

    workflow.set_entry_point("scrape")
    workflow.add_edge("scrape", "analyze")
    workflow.add_edge("analyze", "score")
    workflow.add_edge("score", "research")
    workflow.add_edge("research", "full_report")
    workflow.add_edge("full_report", END)

    return workflow.compile()


# Default to quick workflow for backwards compatibility
app = build_quick_workflow()
full_app = build_full_workflow()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Media Profiler - MBFC Methodology Compliant Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick analysis (simplified report)
  python profiler.py https://example.com US --model local

  # Comprehensive MBFC-style report with evidence
  python profiler.py https://news-site.com GB --model llm --full-report

  # Skip human review
  python profiler.py https://news-site.com US --no-review
        """
    )
    parser.add_argument("url", help="Target URL to analyze")
    parser.add_argument("country", help="Country Code (e.g., US, GB, KZ)")
    parser.add_argument(
        "--model",
        choices=["llm", "local"],
        default="llm",  # Default to LLM for broader compatibility
        help="Propaganda detection backend: 'local' (DeBERTa) or 'llm' (OpenAI)"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip human review of propaganda detection"
    )
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Generate comprehensive MBFC-style report with history, evidence, and citations"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MEDIA PROFILER - MBFC Methodology Analysis")
    print(f"{'='*60}")
    print(f"  Target: {args.url}")
    print(f"  Country: {args.country}")
    print(f"  Propaganda Model: {args.model.upper()}")
    print(f"  Human Review: {'Disabled' if args.no_review else 'Enabled'}")
    print(f"  Report Type: {'COMPREHENSIVE' if args.full_report else 'QUICK'}")
    print(f"{'='*60}\n")

    # Initialize state with all required fields
    initial_state = {
        "target_url": args.url,
        "country_code": args.country,
        "use_local_model": (args.model == "local"),
        "full_report": args.full_report
    }

    # Select workflow based on report type
    selected_app = full_app if args.full_report else app

    # Run the workflow
    current_state = initial_state.copy()

    for output in selected_app.stream(initial_state):
        for step_name, step_output in output.items():
            print(f"Completed step: {step_name}")
            current_state.update(step_output)

    # Human review (optional, only for quick mode)
    if not args.no_review and not args.full_report and current_state.get("propaganda_analysis"):
        current_state = human_review_process(current_state)
        # Regenerate report with updated propaganda score
        current_state = score_node(current_state)
        final_output = report_node(current_state)
        print(final_output["final_report"])
    else:
        print(current_state.get("final_report", "No report generated."))
