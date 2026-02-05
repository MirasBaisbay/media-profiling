"""
Research Module - Web Research and Comprehensive Profiling for MBFC Reports

This module gathers external context about media outlets and orchestrates
all analyzers to produce comprehensive MBFC-style reports.

Components:
1. MediaResearcher: Gathers history, ownership, and external analysis via web search
2. MediaProfiler: Orchestrates all analyzers to produce comprehensive reports

All LLM calls use LangChain's .with_structured_output() for type-safe responses.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI

from schemas import (
    ComprehensiveReportData,
    EditorialBiasResult,
    ExternalAnalysisItem,
    ExternalAnalysisLLMOutput,
    FactCheckAnalysisResult,
    HistoryLLMOutput,
    OwnershipLLMOutput,
    PseudoscienceAnalysisResult,
    SourcingAnalysisResult,
)

from refactored_analyzers import (
    EditorialBiasAnalyzer,
    FactCheckSearcher,
    MediaTypeAnalyzer,
    OpinionAnalyzer,
    PseudoscienceAnalyzer,
    SourcingAnalyzer,
    TrafficLongevityAnalyzer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Configuration
# =============================================================================


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """Get a configured LLM instance."""
    return ChatOpenAI(model=model, temperature=temperature)


# =============================================================================
# MediaResearcher - Web Research with Structured Output
# =============================================================================


class MediaResearcher:
    """
    Gathers external context for comprehensive MBFC-style reports.

    Uses DuckDuckGo search + LLM with structured output to gather:
    - History and founding information
    - Ownership and funding details
    - External criticism and analysis

    All LLM calls use .with_structured_output() for type-safe responses.
    """

    HISTORY_PROMPT = """You are extracting history information about a media outlet from search results.

Extract the following if available:
- Founding year
- Founder name(s)
- Original name (if different from current)
- Key events in the outlet's history (ownership changes, scandals, major milestones)

Be conservative - only extract information that is clearly stated in the search results.
If information is not found, leave fields as null."""

    OWNERSHIP_PROMPT = """You are extracting ownership and funding information about a media outlet.

Extract the following if available:
- Current owner (person or entity)
- Parent company (if applicable)
- Funding model (advertising, subscription, public funding, nonprofit, mixed)
- Headquarters location (city, country)

Be conservative - only extract information that is clearly stated in the search results.
If information is not found, leave fields as null."""

    EXTERNAL_ANALYSIS_PROMPT = """You are extracting external analyses and criticism about a media outlet.

Focus on:
- Media watchdog reviews (MBFC, Ad Fontes, NewsGuard, etc.)
- Academic studies
- Journalism reviews (CJR, Nieman Lab, etc.)
- Major controversies or notable praise

For each analysis found:
- Identify the source name
- Extract URL if available
- Summarize the key finding
- Categorize sentiment as: positive, negative, neutral, or mixed

Include up to 3-5 most relevant and credible analyses."""

    # Domains to exclude from search results
    SEARCH_BLACKLIST = {
        "facebook.com",
        "twitter.com",
        "x.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
        "linkedin.com",
        "reddit.com",
        "youtube.com",
    }

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the MediaResearcher.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
        """
        self.history_llm = get_llm(model, temperature).with_structured_output(
            HistoryLLMOutput
        )
        self.ownership_llm = get_llm(model, temperature).with_structured_output(
            OwnershipLLMOutput
        )
        self.analysis_llm = get_llm(model, temperature).with_structured_output(
            ExternalAnalysisLLMOutput
        )
        self.search = DDGS()

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _extract_outlet_name(self, url: str) -> str:
        """
        Extract a human-readable outlet name from URL.

        Args:
            url: The outlet's URL

        Returns:
            Human-readable name derived from domain
        """
        domain = self._extract_domain(url)
        # Generate name from domain (e.g., "nytimes.com" -> "Nytimes")
        name = domain.split(".")[0]
        return name.replace("-", " ").replace("_", " ").title()

    def _search(self, query: str, max_results: int = 5) -> str:
        """
        Perform a DuckDuckGo search and return combined snippets.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            Combined search snippets
        """
        try:
            # Request more results to account for blacklist filtering
            results = list(self.search.text(query, max_results=max_results + 5))
            if results:
                snippets = []
                for r in results:
                    url = r.get("href", "")
                    # Filter out blacklisted domains
                    result_domain = self._extract_domain(url)
                    if result_domain in self.SEARCH_BLACKLIST:
                        continue
                    title = r.get("title", "")
                    body = r.get("body", "")
                    snippets.append(f"{title}: {body} (URL: {url})")
                    if len(snippets) >= max_results:
                        break
                return "\n\n".join(snippets)
            return ""
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            return ""

    def research_history(self, outlet_name: str) -> HistoryLLMOutput:
        """
        Research outlet history and founding information.

        Args:
            outlet_name: Human-readable outlet name

        Returns:
            HistoryLLMOutput with extracted history
        """
        query = f'"{outlet_name}" founded history about us media news organization'
        snippets = self._search(query)

        if not snippets:
            return HistoryLLMOutput(
                summary="No history information found.",
                confidence=0.0
            )

        user_prompt = f"""Extract history information for "{outlet_name}" from these search results:

SEARCH RESULTS:
{snippets[:3000]}"""

        try:
            result: HistoryLLMOutput = self.history_llm.invoke(
                [
                    {"role": "system", "content": self.HISTORY_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"History research failed: {e}")
            return HistoryLLMOutput(
                summary=f"History research failed: {str(e)}",
                confidence=0.0
            )

    def research_ownership(self, outlet_name: str) -> OwnershipLLMOutput:
        """
        Research ownership and funding information.

        Args:
            outlet_name: Human-readable outlet name

        Returns:
            OwnershipLLMOutput with extracted ownership info
        """
        query = f'"{outlet_name}" ownership owner parent company funded by headquarters'
        snippets = self._search(query)

        if not snippets:
            return OwnershipLLMOutput(
                notes="No ownership information found.",
                confidence=0.0
            )

        user_prompt = f"""Extract ownership and funding information for "{outlet_name}" from these search results:

SEARCH RESULTS:
{snippets[:3000]}"""

        try:
            result: OwnershipLLMOutput = self.ownership_llm.invoke(
                [
                    {"role": "system", "content": self.OWNERSHIP_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"Ownership research failed: {e}")
            return OwnershipLLMOutput(
                notes=f"Ownership research failed: {str(e)}",
                confidence=0.0
            )

    def research_external_analysis(self, outlet_name: str) -> ExternalAnalysisLLMOutput:
        """
        Research external analyses and criticism.

        Args:
            outlet_name: Human-readable outlet name

        Returns:
            ExternalAnalysisLLMOutput with external analyses
        """
        query = f'"{outlet_name}" media bias analysis criticism review fact check rating'
        snippets = self._search(query, max_results=8)

        if not snippets:
            return ExternalAnalysisLLMOutput(
                analyses=[],
                confidence=0.0
            )

        user_prompt = f"""Extract external analyses and criticism for "{outlet_name}" from these search results:

SEARCH RESULTS:
{snippets[:4000]}"""

        try:
            result: ExternalAnalysisLLMOutput = self.analysis_llm.invoke(
                [
                    {"role": "system", "content": self.EXTERNAL_ANALYSIS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result
        except Exception as e:
            logger.error(f"External analysis research failed: {e}")
            return ExternalAnalysisLLMOutput(
                analyses=[],
                confidence=0.0
            )


# =============================================================================
# MediaProfiler - Comprehensive Analysis Orchestrator
# =============================================================================


class MediaProfiler:
    """
    Orchestrates all analyzers to produce comprehensive MBFC-style reports.

    Combines results from:
    - TrafficLongevityAnalyzer: Domain age and traffic tier
    - MediaTypeAnalyzer: Media type classification
    - OpinionAnalyzer: News vs Opinion classification
    - EditorialBiasAnalyzer: Political bias detection
    - FactCheckSearcher: Fact-checker search results
    - SourcingAnalyzer: Source quality assessment
    - PseudoscienceAnalyzer: Pseudoscience detection
    - MediaResearcher: History, ownership, external analysis

    Produces a ComprehensiveReportData object with all analysis results.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize all analyzers.

        Args:
            model: OpenAI model to use for all analyzers
            temperature: LLM temperature (0 for deterministic)
        """
        self.traffic_analyzer = TrafficLongevityAnalyzer(model=model, temperature=temperature)
        self.media_type_analyzer = MediaTypeAnalyzer(model=model, temperature=temperature)
        self.opinion_analyzer = OpinionAnalyzer(model=model, temperature=temperature)
        self.editorial_bias_analyzer = EditorialBiasAnalyzer(model=model, temperature=temperature)
        self.fact_check_searcher = FactCheckSearcher(model=model, temperature=temperature)
        self.sourcing_analyzer = SourcingAnalyzer(model=model, temperature=temperature)
        self.pseudoscience_analyzer = PseudoscienceAnalyzer(model=model, temperature=temperature)
        self.researcher = MediaResearcher(model=model, temperature=temperature)

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        domain = domain.split("/")[0]
        return domain.lower()

    def _score_to_factuality_label(self, score: float) -> str:
        """Convert factuality score to MBFC-style label."""
        if score <= 2.0:
            return "Very High"
        elif score <= 4.0:
            return "High"
        elif score <= 6.0:
            return "Mixed"
        elif score <= 8.0:
            return "Low"
        else:
            return "Very Low"

    def _calculate_credibility_score(
        self,
        fact_check_score: float,
        sourcing_score: float,
        pseudoscience_score: float,
    ) -> tuple[float, str]:
        """
        Calculate overall credibility score and label.

        Args:
            fact_check_score: Score from FactCheckSearcher (0-10)
            sourcing_score: Score from SourcingAnalyzer (0-10)
            pseudoscience_score: Score from PseudoscienceAnalyzer (0-10)

        Returns:
            Tuple of (credibility_score, credibility_label)
        """
        # Weighted average: fact checks 40%, sourcing 30%, pseudoscience 30%
        credibility_score = (
            fact_check_score * 0.4 +
            sourcing_score * 0.3 +
            pseudoscience_score * 0.3
        )

        if credibility_score <= 2.0:
            label = "Very High Credibility"
        elif credibility_score <= 4.0:
            label = "High Credibility"
        elif credibility_score <= 6.0:
            label = "Medium Credibility"
        elif credibility_score <= 8.0:
            label = "Low Credibility"
        else:
            label = "Very Low Credibility"

        return credibility_score, label

    def profile(
        self,
        url: str,
        articles: list[dict[str, str]],
        outlet_name: Optional[str] = None,
    ) -> ComprehensiveReportData:
        """
        Perform comprehensive profiling of a media outlet.

        Args:
            url: The outlet's URL
            articles: List of article dicts with 'title' and 'text' keys
            outlet_name: Optional human-readable name (auto-detected if not provided)

        Returns:
            ComprehensiveReportData with all analysis results
        """
        domain = self._extract_domain(url)
        outlet_name = outlet_name or self.researcher._extract_outlet_name(url)

        logger.info(f"Profiling: {outlet_name} ({domain})")

        # 1. Traffic and metadata analysis
        logger.info("  - Analyzing traffic and longevity...")
        traffic_data = self.traffic_analyzer.analyze(url)

        logger.info("  - Classifying media type...")
        media_type_result = self.media_type_analyzer.analyze(url)

        # 2. Content analysis (requires articles)
        editorial_bias_result: Optional[EditorialBiasResult] = None
        sourcing_result: Optional[SourcingAnalysisResult] = None
        pseudoscience_result: Optional[PseudoscienceAnalysisResult] = None

        if articles:
            logger.info(f"  - Analyzing {len(articles)} articles for bias...")
            editorial_bias_result = self.editorial_bias_analyzer.analyze(
                articles, url, outlet_name
            )

            logger.info("  - Analyzing sourcing quality...")
            sourcing_result = self.sourcing_analyzer.analyze(articles)

            logger.info("  - Checking for pseudoscience...")
            pseudoscience_result = self.pseudoscience_analyzer.analyze(
                articles, url, outlet_name
            )

        # 3. Fact check search
        logger.info("  - Searching fact-checkers...")
        fact_check_result = self.fact_check_searcher.analyze(url, outlet_name)

        # 4. External research
        logger.info("  - Researching history...")
        history = self.researcher.research_history(outlet_name)

        logger.info("  - Researching ownership...")
        ownership = self.researcher.research_ownership(outlet_name)

        logger.info("  - Gathering external analyses...")
        external_analyses = self.researcher.research_external_analysis(outlet_name)

        # 5. Calculate overall scores
        bias_score = editorial_bias_result.bias_score if editorial_bias_result else 0.0
        bias_label = editorial_bias_result.mbfc_label if editorial_bias_result else "Center"

        fact_check_score = fact_check_result.score
        sourcing_score = sourcing_result.score if sourcing_result else 5.0
        pseudoscience_score = pseudoscience_result.score if pseudoscience_result else 5.0

        # Factuality is average of fact check and sourcing
        factuality_score = (fact_check_score + sourcing_score) / 2
        factuality_label = self._score_to_factuality_label(factuality_score)

        # Credibility combines all factual indicators
        credibility_score, credibility_label = self._calculate_credibility_score(
            fact_check_score, sourcing_score, pseudoscience_score
        )

        # 6. Build comprehensive report
        report = ComprehensiveReportData(
            # Target info
            target_url=url,
            target_domain=domain,
            outlet_name=outlet_name,

            # Overall ratings
            bias_label=bias_label,
            bias_score=bias_score,
            factuality_label=factuality_label,
            factuality_score=factuality_score,
            credibility_label=credibility_label,
            credibility_score=credibility_score,

            # Traffic and metadata
            media_type=media_type_result.media_type.value,
            traffic_tier=traffic_data.traffic_tier.value,
            domain_age_years=traffic_data.age_years,

            # Component analysis results
            editorial_bias_result=editorial_bias_result,
            fact_check_result=fact_check_result,
            sourcing_result=sourcing_result,
            pseudoscience_result=pseudoscience_result,

            # Research results
            history_summary=history.summary,
            founding_year=history.founding_year,
            owner=ownership.owner or ownership.parent_company,
            funding_model=ownership.funding_model,
            headquarters=ownership.headquarters,

            # External analyses
            external_analyses=external_analyses.analyses,

            # Metadata
            analysis_date=datetime.now().strftime("%Y-%m-%d"),
            articles_analyzed=len(articles),
        )

        logger.info(f"  - Profiling complete for {outlet_name}")
        return report

    def generate_report_text(self, report: ComprehensiveReportData) -> str:
        """
        Generate a human-readable MBFC-style report.

        Args:
            report: ComprehensiveReportData from profile()

        Returns:
            Formatted text report
        """
        lines = []

        # Header
        lines.append("=" * 70)
        lines.append(f"MEDIA BIAS/FACT CHECK REPORT: {report.outlet_name.upper()}")
        lines.append("=" * 70)
        lines.append(f"URL: {report.target_url}")
        lines.append(f"Analysis Date: {report.analysis_date}")
        lines.append("")

        # Quick Summary
        lines.append("QUICK SUMMARY")
        lines.append("-" * 40)
        lines.append(f"  Bias Rating:        {report.bias_label}")
        lines.append(f"  Factuality Rating:  {report.factuality_label}")
        lines.append(f"  Credibility:        {report.credibility_label}")
        lines.append(f"  Media Type:         {report.media_type}")
        lines.append(f"  Traffic:            {report.traffic_tier}")
        if report.domain_age_years:
            lines.append(f"  Domain Age:         {report.domain_age_years:.1f} years")
        lines.append("")

        # History
        lines.append("HISTORY")
        lines.append("-" * 40)
        if report.founding_year:
            lines.append(f"  Founded: {report.founding_year}")
        if report.owner:
            lines.append(f"  Owner: {report.owner}")
        if report.funding_model:
            lines.append(f"  Funding: {report.funding_model}")
        if report.headquarters:
            lines.append(f"  Headquarters: {report.headquarters}")
        if report.history_summary:
            lines.append(f"\n  {report.history_summary}")
        lines.append("")

        # Bias Analysis
        lines.append("BIAS ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Overall Bias: {report.bias_label} (score: {report.bias_score:+.1f})")
        if report.editorial_bias_result:
            eb = report.editorial_bias_result
            if eb.uses_loaded_language:
                lines.append(f"  Uses Loaded Language: Yes")
                if eb.loaded_language_examples:
                    examples = ", ".join(eb.loaded_language_examples[:3])
                    lines.append(f"    Examples: {examples}")
            if eb.policy_positions:
                lines.append("  Policy Positions Detected:")
                for pos in eb.policy_positions[:3]:
                    lines.append(f"    - {pos.domain.value}: {pos.position}")
            lines.append(f"\n  Analysis: {eb.reasoning}")
        lines.append("")

        # Factuality Analysis
        lines.append("FACTUALITY ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"  Factuality Rating: {report.factuality_label} (score: {report.factuality_score:.1f}/10)")

        if report.fact_check_result:
            fc = report.fact_check_result
            lines.append(f"\n  Fact Check Search Results:")
            lines.append(f"    Total Fact Checks Found: {fc.total_checks_count}")
            lines.append(f"    Failed Fact Checks: {fc.failed_checks_count}")
            if fc.findings:
                lines.append("    Recent Findings:")
                for finding in fc.findings[:3]:
                    lines.append(f"      - [{finding.verdict.value}] {finding.claim[:60]}...")

        if report.sourcing_result:
            sr = report.sourcing_result
            lines.append(f"\n  Sourcing Quality:")
            lines.append(f"    Score: {sr.score:.1f}/10")
            lines.append(f"    Unique Sources: {sr.unique_domains}")
            lines.append(f"    Has Primary Sources: {'Yes' if sr.has_primary_sources else 'No'}")
            lines.append(f"    Has Wire Services: {'Yes' if sr.has_wire_services else 'No'}")
        lines.append("")

        # Pseudoscience
        if report.pseudoscience_result:
            lines.append("PSEUDOSCIENCE CHECK")
            lines.append("-" * 40)
            ps = report.pseudoscience_result
            lines.append(f"  Promotes Pseudoscience: {'Yes' if ps.promotes_pseudoscience else 'No'}")
            lines.append(f"  Respects Scientific Consensus: {'Yes' if ps.respects_scientific_consensus else 'No'}")
            if ps.categories_found:
                cats = ", ".join(c.value for c in ps.categories_found[:3])
                lines.append(f"  Categories Found: {cats}")
            lines.append(f"\n  Assessment: {ps.reasoning}")
            lines.append("")

        # External Analyses
        if report.external_analyses:
            lines.append("EXTERNAL ANALYSES")
            lines.append("-" * 40)
            for analysis in report.external_analyses[:3]:
                sentiment_emoji = {
                    "positive": "+",
                    "negative": "-",
                    "neutral": "~",
                    "mixed": "?"
                }.get(analysis.sentiment, "?")
                lines.append(f"  [{sentiment_emoji}] {analysis.source_name}")
                lines.append(f"      {analysis.summary}")
            lines.append("")

        # Footer
        lines.append("=" * 70)
        lines.append(f"Articles Analyzed: {report.articles_analyzed}")
        lines.append("Generated by Media Profiling System")
        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# Convenience Functions
# =============================================================================


def research_outlet(url: str, outlet_name: Optional[str] = None) -> dict:
    """
    Convenience function to research an outlet without full profiling.

    Args:
        url: The outlet's URL
        outlet_name: Optional human-readable name

    Returns:
        Dict with history, ownership, and external analyses
    """
    researcher = MediaResearcher()
    outlet_name = outlet_name or researcher._extract_outlet_name(url)

    history = researcher.research_history(outlet_name)
    ownership = researcher.research_ownership(outlet_name)
    external = researcher.research_external_analysis(outlet_name)

    return {
        "outlet_name": outlet_name,
        "history": history.model_dump(),
        "ownership": ownership.model_dump(),
        "external_analyses": external.model_dump(),
    }


def profile_outlet(
    url: str,
    articles: list[dict[str, str]],
    outlet_name: Optional[str] = None,
) -> ComprehensiveReportData:
    """
    Convenience function to profile a media outlet.

    Args:
        url: The outlet's URL
        articles: List of article dicts with 'title' and 'text' keys
        outlet_name: Optional human-readable name

    Returns:
        ComprehensiveReportData with all analysis results
    """
    profiler = MediaProfiler()
    return profiler.profile(url, articles, outlet_name)


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Configure logging for demo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("=" * 70)
    print("MEDIA PROFILER - DEMO")
    print("=" * 70)

    # Test with sample articles
    sample_articles = [
        {
            "title": "Climate Change Policy Faces Opposition",
            "text": """
            The administration's new climate policy has drawn sharp criticism from
            industry groups who claim it will devastate the economy. Environmental
            advocates, however, argue the measures don't go far enough to address
            the urgent threat of global warming. According to a new report from the
            IPCC, immediate action is needed to prevent catastrophic warming.
            Critics on the right have called the policy "radical" and "job-killing,"
            while progressive groups say it represents a step in the right direction.
            The EPA cited studies from nature.gov and the Department of Energy
            in defending the new regulations.
            """
        },
        {
            "title": "Healthcare Reform Debate Intensifies",
            "text": """
            As healthcare costs continue to rise, lawmakers are divided on solutions.
            Progressive members of Congress are pushing for expanded Medicare coverage,
            while conservatives argue for market-based reforms. A new study from the
            Kaiser Family Foundation found that healthcare spending now accounts for
            nearly 20% of GDP. The American Medical Association has expressed concerns
            about both approaches, citing potential impacts on physician autonomy.
            """
        },
    ]

    # Test with a domain
    test_url = "https://www.bbc.com"

    print(f"\nProfiling: {test_url}")
    print("-" * 50)

    profiler = MediaProfiler()
    report = profiler.profile(test_url, sample_articles)

    # Generate and print text report
    report_text = profiler.generate_report_text(report)
    print("\n" + report_text)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
