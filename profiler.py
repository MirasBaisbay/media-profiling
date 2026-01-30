"""
Media Profiler - MBFC Methodology Compliant Orchestrator
Uses LangGraph for workflow orchestration with Human-in-the-Loop verification.
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

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ProfilerState(TypedDict):
    target_url: str
    country_code: str
    use_local_model: bool  # Flag for local DeBERTa vs LLM propaganda detection
    articles: List[Article]
    metadata: Any

    # Bias Analysis Components (MBFC Methodology)
    economic_bias: float          # 35% weight - Economic position score
    social_bias: float            # 35% weight - Social position score
    news_reporting_bias: float    # 15% weight - Balance in news reporting
    editorial_bias: float         # 15% weight - Bias in opinion/editorial

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

    return {
        # Bias components (raw scores on -10 to +10 scale)
        "economic_bias": economic_result,
        "social_bias": social_result,
        "news_reporting_bias": news_reporting_result,
        "editorial_bias": editorial_result,
        # Factuality components (raw scores on 0-10 scale, lower is better)
        "fact_check_score": fact_check_result,
        "sourcing_score": sourcing_result,
        "transparency_score": transparency_result,
        "propaganda_score": propaganda_result.score,
        # Full propaganda analysis for human review
        "propaganda_analysis": propaganda_result,
        # Supporting data
        "media_type": media_type,
        "freedom_data": freedom_data,
        "traffic_data": traffic_data
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
    Generates final MBFC-compliant report with weighted scores.
    """
    # === Calculate Weighted Bias Score ===
    bias_score = ScoringCalculator.calculate_bias(
        economic_score=state["economic_bias"],
        social_score=state["social_bias"],
        reporting_score=state["news_reporting_bias"],
        editorial_score=state["editorial_bias"]
    )

    # === Calculate Weighted Factuality Score ===
    fact_score = ScoringCalculator.calculate_factuality(
        fact_check_score=state["fact_check_score"],
        sourcing_score=state["sourcing_score"],
        transparency_score=state["transparency_score"],
        propaganda_score=state["propaganda_score"]
    )

    # === Get MBFC Labels ===
    bias_label = ScoringCalculator.get_bias_label(bias_score)
    fact_label = ScoringCalculator.get_factuality_label(fact_score)

    # === Calculate Overall Credibility (0-10 scale) ===
    cred_score, cred_level, f_pts, b_pts = ScoringCalculator.calculate_credibility(
        fact_label=fact_label,
        bias_label=bias_label,
        traffic_points=state["traffic_data"]["points"],
        freedom_penalty=state["freedom_data"]["penalty"]
    )

    # === Generate Report ===
    report = f"""
================================================================================
                    MEDIA BIAS / FACT CHECK REPORT
                      (MBFC Methodology Compliant)
================================================================================

TARGET: {state['target_url']}
MEDIA TYPE: {state['media_type']}
COUNTRY: {state['country_code']} - Freedom Rating: {state['freedom_data']['rating']} ({state['freedom_data']['score']}/100)
TRAFFIC: {state['traffic_data']['details']}

--------------------------------------------------------------------------------
1. BIAS RATING
--------------------------------------------------------------------------------
   Component Scores (Scale: -10 Left to +10 Right):

   Economic Position (35%):      {state['economic_bias']:+.1f}
   Social Position (35%):        {state['social_bias']:+.1f}
   News Reporting Balance (15%): {state['news_reporting_bias']:+.1f}
   Editorial Bias (15%):         {state['editorial_bias']:+.1f}

   WEIGHTED BIAS SCORE: {bias_score:+.2f}
   BIAS LABEL: {bias_label.upper()}

--------------------------------------------------------------------------------
2. FACTUAL REPORTING
--------------------------------------------------------------------------------
   Component Scores (Scale: 0 Best to 10 Worst):

   Failed Fact Checks (40%):     {state['fact_check_score']:.1f}/10
   Sourcing Quality (25%):       {state['sourcing_score']:.1f}/10
   Transparency (25%):           {state['transparency_score']:.1f}/10
   Propaganda/Bias in News (10%): {state['propaganda_score']:.1f}/10

   WEIGHTED FACTUALITY SCORE: {fact_score:.2f} (Lower is Better)
   FACTUALITY LABEL: {fact_label.upper()}

--------------------------------------------------------------------------------
3. OVERALL CREDIBILITY SCORE (0-10)
--------------------------------------------------------------------------------
   Factual Reporting Points: +{f_pts} ({fact_label})
   Bias Rating Points:       +{b_pts} ({bias_label})
   Traffic Bonus:            +{state['traffic_data']['points']}
   Freedom Penalty:          {state['freedom_data']['penalty']}

   -------------------------------------
   TOTAL CREDIBILITY SCORE: {cred_score}/10
   VERDICT: {cred_level.upper()}
   -------------------------------------

================================================================================
                              END OF REPORT
================================================================================
"""
    return {"final_report": report}


# === Build LangGraph Workflow ===
workflow = StateGraph(ProfilerState)
workflow.add_node("scrape", scrape_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("scrape")
workflow.add_edge("scrape", "analyze")
workflow.add_edge("analyze", "report")
workflow.add_edge("report", END)

app = workflow.compile()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Media Profiler - MBFC Methodology Compliant Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profiler.py https://example.com US --model local
  python profiler.py https://news-site.com GB --model llm --no-review
        """
    )
    parser.add_argument("url", help="Target URL to analyze")
    parser.add_argument("country", help="Country Code (e.g., US, GB, KZ)")
    parser.add_argument(
        "--model",
        choices=["llm", "local"],
        default="local",  # Default to local DeBERTa model
        help="Propaganda detection backend: 'local' (DeBERTa) or 'llm' (OpenAI)"
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="Skip human review of propaganda detection"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MEDIA PROFILER - MBFC Methodology Analysis")
    print(f"{'='*60}")
    print(f"  Target: {args.url}")
    print(f"  Country: {args.country}")
    print(f"  Propaganda Model: {args.model.upper()}")
    print(f"  Human Review: {'Disabled' if args.no_review else 'Enabled'}")
    print(f"{'='*60}\n")

    # Initialize state with all required fields
    initial_state = {
        "target_url": args.url,
        "country_code": args.country,
        "use_local_model": (args.model == "local")
    }

    # Run the workflow
    current_state = initial_state.copy()

    for output in app.stream(initial_state):
        for step_name, step_output in output.items():
            print(f"Completed step: {step_name}")
            current_state.update(step_output)

    # Human review (optional)
    if not args.no_review and current_state.get("propaganda_analysis"):
        current_state = human_review_process(current_state)
        # Regenerate report with updated propaganda score
        final_output = report_node(current_state)
        print(final_output["final_report"])
    else:
        print(current_state.get("final_report", "No report generated."))
