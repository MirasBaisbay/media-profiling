"""
Media Profiler - Orchestrator with Human-in-the-Loop
"""
import sys
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

from scraper import MediaScraper, Article
from analyzers import (
    PropagandaAnalyzer, TransparencyAnalyzer, EconomicAnalyzer, 
    SocialAnalyzer, MediaTypeAnalyzer, CountryFreedomAnalyzer, TrafficLongevityAnalyzer,
    ScoringCalculator
)

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ProfilerState(TypedDict):
    target_url: str
    country_code: str
    articles: List[Article]
    metadata: Any
    
    # Analysis
    media_type: str
    freedom_data: Dict
    traffic_data: Dict
    propaganda_analysis: Any
    transparency_score: float
    economic_bias: str
    social_bias: str
    
    final_report: str

def scrape_node(state: ProfilerState):
    # INCREASED TO 20 ARTICLES FOR ACCURACY
    scraper = MediaScraper(state["target_url"], max_articles=20)
    articles = scraper.scrape_feed()
    if not articles: articles = [Article(state["target_url"], "Error", "No content")]
    return {"articles": articles, "metadata": scraper.get_metadata()}

def analyze_node(state: ProfilerState):
    articles = state["articles"]
    meta = state["metadata"]
    url = state["target_url"]
    cc = state.get("country_code", "US")
    
    return {
        "propaganda_analysis": PropagandaAnalyzer().analyze(articles),
        "transparency_score": TransparencyAnalyzer().analyze(meta),
        "economic_bias": EconomicAnalyzer().analyze(articles),
        "social_bias": SocialAnalyzer().analyze(articles),
        "media_type": MediaTypeAnalyzer().analyze(url, articles),
        "freedom_data": CountryFreedomAnalyzer().analyze(cc),
        "traffic_data": TrafficLongevityAnalyzer().analyze(url)
    }

def human_review_process(state: ProfilerState):
    analysis = state["propaganda_analysis"]
    instances = analysis.instances
    verified = []
    
    print("\n" + "="*60)
    print("🕵️  HUMAN VERIFICATION: PROPAGANDA")
    print("="*60)
    
    if not instances: print("No propaganda detected.")
    
    for i, inst in enumerate(instances, 1):
        print(f"\nFinding #{i}")
        print(f"📖 \"{inst.context}\"")
        print(f"🏷️  AI: [{inst.technique}]")
        choice = input("Correct? (y/n/skip): ").lower().strip()
        
        if choice == 'y':
            inst.verified = True
            verified.append(inst)
        elif choice == 'n':
            print("Select correct technique (0-17) or 'x' to reject:")
            from config import PROPAGANDA_TECHNIQUES
            edit = input("Input: ").strip()
            if edit.isdigit() and int(edit) < len(PROPAGANDA_TECHNIQUES):
                inst.technique = PROPAGANDA_TECHNIQUES[int(edit)]
                inst.verified = True
                verified.append(inst)
    
    new_score = min(10.0, len(verified) * 2.0)
    print(f"\n🔄 New Propaganda Score: {new_score}")
    analysis.instances = verified
    analysis.score = new_score
    state["propaganda_analysis"] = analysis
    return state

def report_node(state: ProfilerState):
    # 1. Calc Weighted Scores
    bias_score = ScoringCalculator.calculate_bias(state["economic_bias"], state["social_bias"])
    fact_score = ScoringCalculator.calculate_factuality(state["transparency_score"], state["propaganda_analysis"].score)
    
    # 2. Get Labels
    bias_label = ScoringCalculator.get_bias_label(bias_score)
    fact_label = ScoringCalculator.get_factuality_label(fact_score)
    
    # 3. Calc Credibility (10-point scale)
    cred_score, cred_level, f_pts, b_pts = ScoringCalculator.calculate_credibility(
        fact_label, 
        bias_label, 
        state["traffic_data"]["points"], 
        state["freedom_data"]["penalty"]
    )
    
    report = f"""
    ==================================================
    MEDIA BIAS / FACT CHECK REPORT (MBFC METHODOLOGY)
    ==================================================
    Target: {state['target_url']}
    Media Type: {state['media_type']}
    Traffic: {state['traffic_data']['details']}
    Country Freedom: {state['freedom_data']['rating']} ({state['freedom_data']['score']}/100)
    
    1. BIAS RATING
    --------------
    Economic: {state['economic_bias']}
    Social: {state['social_bias']}
    Weighted Score: {bias_score}
    LABEL: {bias_label.upper()}
    
    2. FACTUAL REPORTING
    --------------------
    Transparency: {state['transparency_score']}/10
    Propaganda: {state['propaganda_analysis'].score}/10
    Weighted Score: {fact_score} (Lower is Better)
    LABEL: {fact_label.upper()}
    
    3. CREDIBILITY SCORE (0-10)
    ---------------------------
    Factual Points: {f_pts} ({fact_label})
    Bias Points:    {b_pts} ({bias_label})
    Traffic Bonus:  {state['traffic_data']['points']}
    Freedom Penalty: {state['freedom_data']['penalty']}
    ---------------------------
    TOTAL: {cred_score}/10
    VERDICT: {cred_level.upper()}
    ==================================================
    """
    return {"final_report": report}

workflow = StateGraph(ProfilerState)
workflow.add_node("scrape", scrape_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("scrape")
workflow.add_edge("scrape", "analyze")

app = workflow.compile()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Media Profiler")
    parser.add_argument("url", help="Target URL")
    parser.add_argument("country", help="Country Code (e.g., US, KZ)")
    parser.add_argument("--model", choices=["llm", "local"], default="llm", help="Propaganda detection backend")
    
    args = parser.parse_args()
    
    print(f"Starting analysis of {args.url} ({args.country}) using {args.model.upper()} model...")
    
    # Pass this flag to your Analyze Node
    initial = {
        "target_url": args.url, 
        "country_code": args.country,
        "use_local_model": (args.model == "local") # Add this to state
    }
    curr = initial.copy()
    
    for output in app.stream(initial):
        for k, v in output.items():
            print(f"Finished step: {k}")
            curr.update(v)
            
    upd = human_review_process(curr)
    print(report_node(upd)["final_report"])