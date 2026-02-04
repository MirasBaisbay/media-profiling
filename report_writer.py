"""
Report Writer Module - MBFC-Style Prose Report Generator

This module synthesizes evidence from analyzers and external research
into comprehensive MBFC-style prose reports.

Report Structure:
1. Bias Category Description (standard MBFC text for category)
2. Overall Summary Paragraph (custom, evidence-based)
3. Detailed Scores (formatted)
4. History Section
5. Funded by / Ownership Section
6. Analysis Section (with citations)
7. Bias Examples Section
8. Failed Fact Checks Section
9. Final Summary with Date
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from evidence import (
    ComprehensiveReportData, AnalyzerOutput, ResearchResults,
    ArticleEvidence, FactCheckEvidence
)

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# =============================================================================
# MBFC CATEGORY DESCRIPTIONS
# =============================================================================

BIAS_CATEGORY_DESCRIPTIONS = {
    "Extreme Left": (
        "EXTREME LEFT\n"
        "These sources exclusively promote left-wing policies and rarely cite credible "
        "sources. They may use strong loaded language and appeal to emotion. Most fail "
        "fact checks and do not correct errors."
    ),
    "Left": (
        "LEFT BIAS\n"
        "These sources moderately to strongly favor liberal perspectives. They may "
        "utilize strong loaded words (wording that attempts to influence an audience "
        "by appeals to emotion or stereotypes), publish misleading reports, and omit "
        "information that may damage liberal causes. Some sources in this category "
        "may be untrustworthy."
    ),
    "Left-Center": (
        "LEFT-CENTER BIAS\n"
        "These sources have a slight to moderate liberal bias. They often publish "
        "factual information that utilizes loaded words (wording that attempts to "
        "influence an audience by appeals to emotion or stereotypes) to favor liberal "
        "causes. These sources are generally trustworthy for information but may "
        "require further investigation."
    ),
    "Least Biased": (
        "LEAST BIASED\n"
        "These sources have minimal bias and use very few loaded words (wording that "
        "attempts to influence an audience by appeals to emotion or stereotypes). "
        "The reporting is factual and usually sourced. These are the most credible "
        "media sources."
    ),
    "Right-Center": (
        "RIGHT-CENTER BIAS\n"
        "These sources have a slight to moderate conservative bias. They often publish "
        "factual information that utilizes loaded words (wording that attempts to "
        "influence an audience by appeals to emotion or stereotypes) to favor "
        "conservative causes. These sources are generally trustworthy for information "
        "but may require further investigation."
    ),
    "Right": (
        "RIGHT BIAS\n"
        "These sources moderately to strongly favor conservative perspectives. They may "
        "utilize strong loaded words (wording that attempts to influence an audience "
        "by appeals to emotion or stereotypes), publish misleading reports, and omit "
        "information that may damage conservative causes. Some sources in this category "
        "may be untrustworthy."
    ),
    "Extreme Right": (
        "EXTREME RIGHT\n"
        "These sources exclusively promote right-wing policies and rarely cite credible "
        "sources. They may use strong loaded language and appeal to emotion. Most fail "
        "fact checks and do not correct errors."
    )
}

FACTUALITY_DESCRIPTIONS = {
    "Very High": "Sources that always use credible sources, are well-sourced, and have a clean fact check record.",
    "High": "Sources that are generally reliable with minimal failed fact checks and good sourcing practices.",
    "Mostly Factual": "Sources that are generally reliable but may have occasional minor errors or unsourced claims.",
    "Mixed": "Sources that do not always use proper sourcing or have multiple failed fact checks.",
    "Low": "Sources that rarely use credible sources and have numerous failed fact checks.",
    "Very Low": "Sources that consistently fail fact checks and promote misinformation."
}


class MBFCReportWriter:
    """
    Generates comprehensive MBFC-style prose reports from evidence.

    Uses LLM to synthesize complex evidence into readable prose while
    maintaining factual accuracy and proper citations.
    """

    def __init__(self):
        self.llm = llm

    def generate_report(self, data: ComprehensiveReportData) -> str:
        """
        Generate a complete MBFC-style report.

        Args:
            data: ComprehensiveReportData with all scores and evidence

        Returns:
            Formatted MBFC-style report as string
        """
        logger.info(f"Generating report for {data.target_domain}")

        sections = []

        # 1. Bias Category Description
        sections.append(self._generate_category_description(data))

        # 2. Overall Summary
        sections.append(self._generate_summary(data))

        # 3. Detailed Scores
        sections.append(self._generate_detailed_scores(data))

        # 4. History Section
        if data.research:
            sections.append(self._generate_history_section(data))

        # 5. Ownership Section
        if data.research:
            sections.append(self._generate_ownership_section(data))

        # 6. Analysis Section
        sections.append(self._generate_analysis_section(data))

        # 7. Bias Examples Section
        sections.append(self._generate_bias_examples_section(data))

        # 8. Failed Fact Checks Section
        sections.append(self._generate_fact_checks_section(data))

        # 9. Final Summary
        sections.append(self._generate_final_summary(data))

        return "\n\n".join(filter(None, sections))

    def _generate_category_description(self, data: ComprehensiveReportData) -> str:
        """Generate the bias category description header."""
        category = data.bias_label
        description = BIAS_CATEGORY_DESCRIPTIONS.get(
            category,
            BIAS_CATEGORY_DESCRIPTIONS["Least Biased"]
        )
        return description

    def _generate_summary(self, data: ComprehensiveReportData) -> str:
        """Generate the overall summary paragraph using LLM."""
        # Build context for LLM
        context = {
            "outlet_name": data.target_domain,
            "bias_label": data.bias_label,
            "bias_score": data.bias_score,
            "factuality_label": data.factuality_label,
            "factuality_score": data.factuality_score,
            "articles_analyzed": data.articles_analyzed,
            "news_count": data.news_articles_count,
            "opinion_count": data.opinion_articles_count,
            "economic_label": data.economic_label,
            "social_label": data.social_label,
            "news_reporting_label": data.news_reporting_label,
            "editorial_label": data.editorial_label
        }

        # Add key evidence points
        evidence_points = []
        if data.editorial_evidence and data.editorial_evidence.raw_details:
            details = data.editorial_evidence.raw_details
            if "clickbait_score" in details:
                evidence_points.append(f"clickbait score of {details['clickbait_score']}/10")
            if "loaded_language_score" in details:
                evidence_points.append(f"loaded language score of {details['loaded_language_score']}/10")

        if data.news_reporting_evidence and data.news_reporting_evidence.raw_details:
            details = data.news_reporting_evidence.raw_details
            if "sourcing_diversity" in details:
                diversity = details["sourcing_diversity"]
                evidence_points.append(f"{diversity*100:.0f}% multi-sided sourcing")

        prompt = f"""
Write a 2-3 sentence summary paragraph for an MBFC-style media bias report.

OUTLET: {context['outlet_name']}
BIAS RATING: {context['bias_label']} ({context['bias_score']:+.1f})
FACTUALITY: {context['factuality_label']} ({context['factuality_score']:.1f})
ARTICLES ANALYZED: {context['articles_analyzed']} ({context['news_count']} news, {context['opinion_count']} opinion)

KEY FINDINGS:
- Economic stance: {context['economic_label']}
- Social stance: {context['social_label']}
- News reporting: {context['news_reporting_label']}
- Editorial bias: {context['editorial_label']}

EVIDENCE: {', '.join(evidence_points) if evidence_points else 'Standard analysis'}

Write in the style of MBFC: "Overall, we rate [outlet] [bias label] based on [key evidence]. We also rate them [factuality label] based on [factuality evidence]."

Return ONLY the summary paragraph, no other text.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            # Fallback to template
            return (
                f"Overall, we rate {data.target_domain} {data.bias_label} based on our "
                f"analysis of {data.articles_analyzed} articles. We also rate them "
                f"{data.factuality_label} for factual reporting."
            )

    def _generate_detailed_scores(self, data: ComprehensiveReportData) -> str:
        """Generate the detailed scores section."""
        return f"""Detailed Report
Bias Rating: {data.bias_label.upper()} ({data.bias_score:+.1f})
Factual Reporting: {data.factuality_label.upper()} ({data.factuality_score:.1f})
Country: {data.country_name}
MBFC's Country Freedom Rating: {data.freedom_rating.upper()}
Media Type: {data.media_type}
Traffic/Popularity: {data.traffic_label} Traffic
MBFC Credibility Rating: {data.credibility_label.upper()}"""

    def _generate_history_section(self, data: ComprehensiveReportData) -> str:
        """Generate the history section from research."""
        if not data.research:
            return ""

        history = data.research.get("history", {})
        if not history:
            return ""

        # Use LLM to write history narrative
        prompt = f"""
Write a 2-3 paragraph "History" section for a media bias report about {data.target_domain}.

RESEARCH DATA:
- Founded: {history.get('founding_year', 'Unknown')}
- Founder: {history.get('founder', 'Unknown')}
- Original name: {history.get('original_name', 'Same')}
- Key events: {history.get('key_events', [])}

Write in encyclopedia style, factual and neutral. Start with "History" as the section header.
If key information is missing, write what is known without inventing details.

Return ONLY the history section text.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"History generation failed: {e}")

            # Fallback to basic format
            if history.get("founding_year"):
                return f"History\nFounded in {history['founding_year']}."
            return ""

    def _generate_ownership_section(self, data: ComprehensiveReportData) -> str:
        """Generate the funded by / ownership section."""
        if not data.research:
            return ""

        ownership = data.research.get("ownership", {})
        if not ownership:
            return ""

        # Use LLM to write ownership narrative
        prompt = f"""
Write a 1-2 paragraph "Funded by / Ownership" section for a media bias report about {data.target_domain}.

RESEARCH DATA:
- Owner: {ownership.get('owner', 'Unknown')}
- Parent company: {ownership.get('parent_company', 'N/A')}
- Funding model: {ownership.get('funding_model', 'Unknown')}
- Headquarters: {ownership.get('headquarters', 'Unknown')}
- Notes: {ownership.get('transparency_notes', '')}

Write in factual, neutral style. Start with "Funded by / Ownership" as the section header.
Mention any transparency disclosures from the site itself.

Return ONLY the ownership section text.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Ownership generation failed: {e}")

            # Fallback
            parts = ["Funded by / Ownership"]
            if ownership.get("owner"):
                parts.append(f"Owned by {ownership['owner']}.")
            if ownership.get("funding_model"):
                parts.append(f"Funded through {ownership['funding_model']}.")
            return "\n".join(parts) if len(parts) > 1 else ""

    def _generate_analysis_section(self, data: ComprehensiveReportData) -> str:
        """Generate the analysis section with evidence."""
        sections = ["Analysis"]

        # Economic analysis
        if data.economic_evidence:
            sections.append(
                f"Economic Coverage: Our analysis found a {data.economic_label.lower()} "
                f"economic perspective (score: {data.economic_score:+.1f})."
            )

        # Social analysis
        if data.social_evidence:
            sections.append(
                f"Social Coverage: Articles showed a {data.social_label.lower()} "
                f"stance on social issues (score: {data.social_score:+.1f})."
            )

        # News reporting analysis
        if data.news_reporting_evidence:
            details = data.news_reporting_evidence.raw_details if data.news_reporting_evidence else {}
            sourcing = details.get("sourcing_diversity", 0) * 100
            sections.append(
                f"News Reporting: {data.news_reporting_label} with "
                f"{sourcing:.0f}% multi-sided sourcing (score: {data.news_reporting_score:+.1f})."
            )

        # Editorial analysis
        if data.editorial_evidence:
            details = data.editorial_evidence.raw_details if data.editorial_evidence else {}
            clickbait = details.get("clickbait_score", 0)
            loaded = details.get("loaded_language_score", 0)
            sections.append(
                f"Editorial Content: {data.editorial_label} with "
                f"clickbait score {clickbait:.1f}/10 and "
                f"loaded language {loaded:.1f}/10."
            )

        # External analysis citations
        if data.research:
            external = data.research.get("external_analyses", [])
            if external:
                sections.append("\nExternal Analysis:")
                for analysis in external[:3]:
                    if isinstance(analysis, dict):
                        sections.append(
                            f"- {analysis.get('source_name', 'Source')}: "
                            f"{analysis.get('summary', '')[:150]}..."
                        )

        return "\n".join(sections)

    def _generate_bias_examples_section(self, data: ComprehensiveReportData) -> str:
        """Generate the bias examples section with specific headlines."""
        sections = ["Bias"]

        # Collect headline examples
        headlines_analyzed = []

        if data.editorial_evidence and data.editorial_evidence.headline_evidence:
            for h in data.editorial_evidence.headline_evidence[:5]:
                if isinstance(h, dict):
                    headlines_analyzed.append(h)

        if data.news_reporting_evidence and data.news_reporting_evidence.article_evidence:
            for a in data.news_reporting_evidence.article_evidence[:5]:
                if isinstance(a, dict):
                    headlines_analyzed.append({
                        "headline": a.get("headline", ""),
                        "finding_type": a.get("finding_type", "neutral")
                    })

        if headlines_analyzed:
            sections.append("\nExample headlines analyzed:")
            for i, h in enumerate(headlines_analyzed[:5], 1):
                headline = h.get("headline", "Unknown")[:60]
                finding = h.get("finding_type", "")
                if headline:
                    sections.append(f"{i}. \"{headline}...\" - {finding}")
        else:
            sections.append(
                f"\nOur analysis of {data.articles_analyzed} articles found "
                f"{data.bias_label.lower()} bias patterns overall."
            )

        return "\n".join(sections)

    def _generate_fact_checks_section(self, data: ComprehensiveReportData) -> str:
        """Generate the failed fact checks section."""
        sections = ["Failed Fact Checks"]

        if data.fact_check_evidence and data.fact_check_evidence.fact_check_evidence:
            for fc in data.fact_check_evidence.fact_check_evidence:
                if isinstance(fc, dict):
                    claim = fc.get("claim", "")[:100]
                    verdict = fc.get("verdict", "Unknown")
                    source = fc.get("source", "")
                    sections.append(f"- {claim}... – {verdict}")
                    if source:
                        sections.append(f"  (Source: {source})")
        else:
            # Check raw details for fact check count
            if data.fact_check_evidence and data.fact_check_evidence.raw_details:
                count = data.fact_check_evidence.raw_details.get("failed_checks_count", 0)
                if count > 0:
                    sections.append(f"Found {count} failed fact check(s) via search.")
                else:
                    sections.append("No failed fact checks found in our search.")
            else:
                sections.append("No failed fact checks found in our search.")

        return "\n".join(sections)

    def _generate_final_summary(self, data: ComprehensiveReportData) -> str:
        """Generate the final summary with date."""
        date_str = data.analysis_date or datetime.now().strftime("%m/%d/%Y")
        return (
            f"Overall, we rate {data.target_domain} {data.bias_label} based on "
            f"our analysis. We also rate them {data.factuality_label} for factual "
            f"reporting. (Analysis Date: {date_str})\n\n"
            f"Source: {data.target_url}"
        )


def write_report_node(state: Dict) -> Dict:
    """
    LangGraph node for report writing phase.

    Generates comprehensive MBFC-style report from state data.
    """
    writer = MBFCReportWriter()

    # Build ComprehensiveReportData from state
    from urllib.parse import urlparse
    parsed = urlparse(state["target_url"])
    domain = parsed.netloc.replace('www.', '')

    # Country name mapping
    country_names = {
        "US": "United States", "GB": "United Kingdom", "CA": "Canada",
        "AU": "Australia", "DE": "Germany", "FR": "France",
        "JP": "Japan", "CN": "China", "IN": "India", "BR": "Brazil"
    }

    # Build report data
    report_data = ComprehensiveReportData(
        target_url=state["target_url"],
        target_domain=domain,
        country_code=state.get("country_code", ""),
        country_name=country_names.get(state.get("country_code", ""), state.get("country_code", "")),

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

        # Evidence (if collected)
        economic_evidence=state.get("economic_evidence"),
        social_evidence=state.get("social_evidence"),
        news_reporting_evidence=state.get("news_reporting_evidence"),
        editorial_evidence=state.get("editorial_evidence"),
        fact_check_evidence=state.get("fact_check_evidence"),
        sourcing_evidence=state.get("sourcing_evidence"),
        transparency_evidence=state.get("transparency_evidence"),
        propaganda_evidence=state.get("propaganda_evidence"),

        # Research
        research=state.get("research_results"),

        # Metadata
        articles_analyzed=len(state.get("articles", [])),
        news_articles_count=len([a for a in state.get("articles", []) if not getattr(a, 'is_opinion', False)]),
        opinion_articles_count=len([a for a in state.get("articles", []) if getattr(a, 'is_opinion', False)])
    )

    # Generate report
    report = writer.generate_report(report_data)

    return {"final_report": report}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with mock data
    test_data = ComprehensiveReportData(
        target_url="https://www.bbc.com",
        target_domain="bbc.com",
        country_code="GB",
        country_name="United Kingdom",

        bias_score=-1.5,
        bias_label="Least Biased",
        factuality_score=1.5,
        factuality_label="High",
        credibility_score=8,
        credibility_label="High Credibility",

        economic_score=-2.5,
        economic_label="Regulated Market Economy",
        social_score=-2.5,
        social_label="Mild Progressive",
        news_reporting_score=-1.0,
        news_reporting_label="Mild Left Reporting",
        editorial_score=-2.0,
        editorial_label="Mild Left Editorial Bias",

        fact_check_score=1.0,
        sourcing_score=1.5,
        transparency_score=0.0,
        propaganda_score=2.0,

        media_type="TV Station",
        traffic_label="High",
        freedom_rating="Mostly Free",
        freedom_score=87.18,

        articles_analyzed=20,
        news_articles_count=16,
        opinion_articles_count=4,

        research={
            "history": {
                "founding_year": 1922,
                "founder": "John Reith",
                "key_events": ["First TV broadcasts 1936", "Charter renewal 2017"]
            },
            "ownership": {
                "owner": "British Public",
                "funding_model": "License Fee",
                "headquarters": "London, UK"
            },
            "external_analyses": [
                {
                    "source_name": "Reuters Institute",
                    "summary": "BBC ranked #1 in trust among UK news sources"
                }
            ]
        }
    )

    writer = MBFCReportWriter()
    report = writer.generate_report(test_data)

    print("\n" + "=" * 80)
    print("GENERATED REPORT")
    print("=" * 80)
    print(report)
