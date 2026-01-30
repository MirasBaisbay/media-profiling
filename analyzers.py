"""
Analyzers Module - MBFC Methodology Compliant
Implements all scoring categories per Media Bias Fact Check methodology:

BIAS SCORING (4 categories, scale -10 to +10):
- Economic System (35%)
- Social Progressive vs Traditional Conservatism (35%)
- Straight News Reporting Balance (15%)
- Editorial Bias (15%)

FACTUAL SCORING (4 categories, scale 0-10 where lower is better):
- Failed Fact Checks (40%)
- Sourcing (25%)
- Transparency (25%)
- One-Sidedness/Propaganda (10%)
"""

import logging
import json
import csv
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

from config import (
    PROPAGANDA_TECHNIQUES, BiasWeights, FactualWeights,
    ECONOMIC_SCALE, SOCIAL_SCALE, NEWS_REPORTING_SCALE, EDITORIAL_BIAS_SCALE,
    ISO_MAPPING, FREEDOM_INDEX_FILE,
    FACTUALITY_RANGES, BIAS_RANGES, CREDIBILITY_POINTS,
    IFCN_FACT_CHECKERS, FACT_CHECK_SEARCH_TERMS
)
from scraper import Article, SiteMetadata
from local_detector import LocalPropagandaDetector

local_detector = None
logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = DuckDuckGoSearchRun()


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PropagandaInstance:
    text_snippet: str
    technique: str
    confidence: float
    context: str = ""
    verified: bool = False


@dataclass
class PropagandaAnalysis:
    score: float  # 0-10 scale
    instances: List[PropagandaInstance]
    verified_count: int = 0


@dataclass
class FactCheckResult:
    source: str
    claim: str
    verdict: str
    url: str


@dataclass
class FactCheckAnalysis:
    failed_checks_count: int
    score: float  # 0-10 scale
    results: List[FactCheckResult]
    details: str


@dataclass
class SourcingAnalysis:
    score: float  # 0-10 scale (lower is better)
    avg_sources_per_article: float
    has_hyperlinks: bool
    credible_source_ratio: float
    details: str


@dataclass
class BiasAnalysis:
    economic_label: str
    economic_score: float
    social_label: str
    social_score: float
    news_reporting_label: str
    news_reporting_score: float
    editorial_label: str
    editorial_score: float
    weighted_total: float
    final_label: str


@dataclass
class FactualAnalysis:
    fact_check_score: float  # 0-10
    sourcing_score: float  # 0-10
    transparency_score: float  # 0-10
    propaganda_score: float  # 0-10
    weighted_total: float
    final_label: str


# =============================================================================
# EXISTING ANALYZERS (Updated)
# =============================================================================

class CountryFreedomAnalyzer:
    """Calculates Freedom Score from RSF/Freedom House data."""

    def analyze(self, country_code: str) -> Dict[str, Any]:
        code = country_code.upper()
        iso3 = ISO_MAPPING.get(code, code)

        score = 0.0
        rating = "Unknown"
        found = False

        if os.path.exists(FREEDOM_INDEX_FILE):
            try:
                with open(FREEDOM_INDEX_FILE, mode='r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter=';')
                    for row in reader:
                        if row['ISO'] == iso3:
                            score_str = row.get('Score 2025', '0').replace(',', '.')
                            score = float(score_str)
                            found = True
                            break
            except Exception as e:
                logger.error(f"Error reading Freedom CSV: {e}")
        else:
            logger.warning(f"Freedom Index file '{FREEDOM_INDEX_FILE}' not found.")

        if not found:
            return {"rating": f"Unknown ({iso3})", "penalty": 0, "score": 0}

        # MBFC Freedom Rating Scale
        if score >= 90:
            rating = "Excellent Freedom"
        elif score >= 70:
            rating = "Mostly Free"
        elif score >= 50:
            rating = "Moderate Freedom"
        elif score >= 25:
            rating = "Limited Freedom"
        else:
            rating = "Total Oppression"

        # Credibility Penalty per MBFC
        penalty = 0
        if rating == "Limited Freedom":
            penalty = -1
        if rating == "Total Oppression":
            penalty = -2

        return {
            "rating": rating,
            "score": score,
            "penalty": penalty,
            "iso": iso3
        }


class TrafficLongevityAnalyzer:
    """Estimates traffic and age using LLM knowledge for credibility bonus."""

    def analyze(self, url: str) -> Dict[str, Any]:
        prompt = f"""
        Estimate the traffic and longevity for this media outlet: {url}

        1. Traffic Level:
           - High: Global reach or major national outlet (e.g., BBC, CNN, NYT)
           - Medium: Regional or specialized audience
           - Minimal: Local or very niche audience

        2. Age: Has this outlet existed for more than 10 years?

        Return ONLY valid JSON: {{"traffic": "High/Medium/Minimal", "older_than_10": true/false}}
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            content = res.content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            traffic_pts = CREDIBILITY_POINTS["traffic"].get(data["traffic"], 0)
            bonus = 1 if data.get("older_than_10", False) else 0

            return {
                "traffic_label": data["traffic"],
                "older_than_10": data.get("older_than_10", False),
                "points": traffic_pts + bonus,
                "details": f"{data['traffic']} Traffic + {'>10 Years' if bonus else '<10 Years'}"
            }
        except Exception as e:
            logger.error(f"Traffic analysis failed: {e}")
            return {"traffic_label": "Medium", "older_than_10": False, "points": 1, "details": "Estimation Failed"}


class MediaTypeAnalyzer:
    """Classifies media type based on content analysis."""

    def analyze(self, url: str, articles: List[Article]) -> str:
        headlines = "\n".join([a.title for a in articles[:5]])
        prompt = f"""
        Classify the media type for '{url}' based on its content.

        Choose ONE category:
        - TV Station
        - Newspaper
        - News Website
        - Magazine
        - Blog
        - State Media
        - Satire
        - Questionable Source

        Headlines sample: {headlines}

        Return ONLY the category name, nothing else.
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            return res.content.strip().replace(".", "")
        except Exception as e:
            logger.error(f"Media type analysis failed: {e}")
            return "News Website"


class TransparencyAnalyzer:
    """
    Assesses transparency based on 5 key elements (25% of Factuality Score):
    - About page
    - Ownership disclosure
    - Funding disclosure
    - Author information
    - Location disclosure

    Score 0-10 where 0 is fully transparent, 10 is no transparency.
    """

    def analyze(self, meta: SiteMetadata) -> float:
        score = 0.0
        details = []

        if not meta.has_about_page:
            score += 2.0
            details.append("Missing About page")
        if not meta.ownership_disclosed:
            score += 2.0
            details.append("Ownership not disclosed")
        if not meta.funding_disclosed:
            score += 2.0
            details.append("Funding not disclosed")
        if not meta.has_author_pages:
            score += 2.0
            details.append("No author information")
        if not meta.location_disclosed:
            score += 2.0
            details.append("Location not disclosed")

        return float(score)


# =============================================================================
# BIAS ANALYZERS (35% + 35% + 15% + 15% = 100%)
# =============================================================================

class EconomicAnalyzer:
    """
    Analyzes economic ideology (35% of Bias Score).
    Scale: -10 (Communism) to +10 (Radical Laissez-Faire)
    """

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        combined = "\n".join([f"- {a.title}: {a.text[:300]}" for a in articles[:10]])
        prompt = f"""
        Analyze the ECONOMIC ideology expressed in this media content.

        Economic Scale (choose ONE):
        - Communism (-10): Advocates full government ownership, no corporatism
        - Socialism (-7.5): Supports high regulation, significant government ownership
        - Democratic Socialism (-5): Endorses strongly regulated capitalism
        - Regulated Market Economy (-2.5): Promotes moderate corporatism with balanced regulations
        - Centrism (0): Balances regulation and corporate influence
        - Moderately Regulated Capitalism (+2.5): Leans toward corporatism with moderate intervention
        - Classical Liberalism (+5): Emphasizes moderate to high corporatism with lower regulations
        - Libertarianism (+7.5): Advocates low government intervention
        - Radical Laissez-Faire (+10): Minimal to no regulation, pure free market

        Content to analyze:
        {combined}

        Return ONLY the category name exactly as written above (e.g., "Regulated Market Economy").
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            label = res.content.strip().replace("'", "").replace('"', '')
            score = ECONOMIC_SCALE.get(label, 0.0)
            return {"label": label, "score": score}
        except Exception as e:
            logger.error(f"Economic analysis failed: {e}")
            return {"label": "Centrism", "score": 0.0}


class SocialAnalyzer:
    """
    Analyzes social values stance (35% of Bias Score).
    Scale: -10 (Strong Progressive) to +10 (Strong Traditional Conservative)
    """

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        combined = "\n".join([f"- {a.title}: {a.text[:300]}" for a in articles[:10]])
        prompt = f"""
        Analyze the SOCIAL VALUES stance expressed in this media content.
        Consider positions on: abortion, immigration, climate change, LGBTQ+ rights,
        social justice, equity, religious values, traditional family values.

        Social Scale (choose ONE):
        - Strong Progressive (-10): Highly progressive, focusing on equality, equity, inclusivity
        - Progressive (-7.5): Strongly supports liberal social policies
        - Moderate Progressive (-5): Leans toward liberal values with some moderation
        - Mild Progressive (-2.5): Slightly socially liberal
        - Balanced (0): Neutral stance incorporating multiple viewpoints
        - Mild Conservative (+2.5): Slightly favors traditional values
        - Moderate Conservative (+5): Advocates for conservative values with some flexibility
        - Traditional Conservative (+7.5): Strongly supports religious/traditional family values
        - Strong Traditional Conservative (+10): Exclusively promotes traditional/religious values

        Content to analyze:
        {combined}

        Return ONLY the category name exactly as written above (e.g., "Moderate Progressive").
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            label = res.content.strip().replace("'", "").replace('"', '')
            score = SOCIAL_SCALE.get(label, 0.0)
            return {"label": label, "score": score}
        except Exception as e:
            logger.error(f"Social analysis failed: {e}")
            return {"label": "Balanced", "score": 0.0}


class NewsReportingBalanceAnalyzer:
    """
    Analyzes straight news reporting balance (15% of Bias Score).
    Measures how well a source reports all sides in its NEWS stories.
    Scale: -10 (Extreme Left) to +10 (Extreme Right)
    """

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        # Filter to news articles only (not opinion)
        news_articles = [a for a in articles if not a.is_opinion][:10]
        if not news_articles:
            news_articles = articles[:10]

        combined = "\n".join([f"- {a.title}: {a.text[:300]}" for a in news_articles])
        prompt = f"""
        Analyze the STRAIGHT NEWS REPORTING BALANCE of this media outlet.
        Focus ONLY on news reporting, NOT opinion pieces.

        Consider:
        - Does the outlet report all sides of issues?
        - Is there story selection bias?
        - Are opposing viewpoints included in articles?
        - Is framing neutral or slanted?

        News Reporting Scale (choose ONE):
        - Extreme Left Reporting (-10): Exclusively promotes left-leaning perspectives
        - Strong Left Reporting (-7.5): Frequently promotes left perspectives, limited opposition
        - Moderate Left Reporting (-5): Often leans left but includes some counterpoints
        - Mild Left Reporting (-2.5): Slightly favors left framing
        - Neutral/Balanced (0): Equally represents all perspectives
        - Mild Right Reporting (+2.5): Slightly favors right framing
        - Moderate Right Reporting (+5): Often leans right but includes some counterpoints
        - Strong Right Reporting (+7.5): Frequently promotes right perspectives, limited opposition
        - Extreme Right Reporting (+10): Exclusively promotes right-leaning perspectives

        News content to analyze:
        {combined}

        Return ONLY the category name exactly as written above (e.g., "Neutral/Balanced").
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            label = res.content.strip().replace("'", "").replace('"', '')
            score = NEWS_REPORTING_SCALE.get(label, 0.0)
            return {"label": label, "score": score}
        except Exception as e:
            logger.error(f"News reporting analysis failed: {e}")
            return {"label": "Neutral/Balanced", "score": 0.0}


class EditorialBiasAnalyzer:
    """
    Analyzes editorial/opinion bias (15% of Bias Score).
    Evaluates bias in opinion pieces, editorials, and use of loaded emotional language.
    Scale: -10 (Extreme Left) to +10 (Extreme Right)
    """

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        # Prefer opinion articles if available
        opinion_articles = [a for a in articles if a.is_opinion][:5]
        if not opinion_articles:
            opinion_articles = articles[:5]

        combined = "\n".join([f"- {a.title}: {a.text[:400]}" for a in opinion_articles])
        prompt = f"""
        Analyze the EDITORIAL BIAS of this media outlet.
        Focus on opinion pieces, editorials, and use of emotional/loaded language.

        Consider:
        - Do editorials consistently favor one political side?
        - Is emotional or manipulative language used?
        - Are loaded terms used to describe political figures or policies?

        Editorial Bias Scale (choose ONE):
        - Extreme Left Editorial (-10): Exclusively promotes left views with highly emotional language
        - Strong Left Editorial (-7.5): Regularly supports left views with emotional language
        - Moderate Left Editorial (-5): Often leans left with some emotional framing
        - Mild Left Editorial (-2.5): Slightly favors left perspectives
        - Neutral/Balanced Editorial (0): Presents perspectives fairly, avoids loaded language
        - Mild Right Editorial (+2.5): Slightly favors right perspectives
        - Moderate Right Editorial (+5): Often leans right with some emotional framing
        - Strong Right Editorial (+7.5): Regularly supports right views with emotional language
        - Extreme Right Editorial (+10): Exclusively promotes right views with highly emotional language

        Content to analyze:
        {combined}

        Return ONLY the category name exactly as written above (e.g., "Neutral/Balanced Editorial").
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            label = res.content.strip().replace("'", "").replace('"', '')
            score = EDITORIAL_BIAS_SCALE.get(label, 0.0)
            return {"label": label, "score": score}
        except Exception as e:
            logger.error(f"Editorial bias analysis failed: {e}")
            return {"label": "Neutral/Balanced Editorial", "score": 0.0}


# =============================================================================
# FACTUALITY ANALYZERS (40% + 25% + 25% + 10% = 100%)
# =============================================================================

class FactCheckSearcher:
    """
    Searches for failed fact checks from IFCN-approved fact checkers (40% of Factuality Score).

    Score based on number of failed fact checks in past 5 years:
    0: No failed fact checks = Very High
    1: One failed = High
    2-3: A few failed = Mostly Factual
    4-7: Several failed = Mixed
    8-9: Frequent failures = Low
    10+: Many failures = Very Low
    """

    def __init__(self):
        self.search_tool = search_tool

    def search(self, domain: str, outlet_name: str = None) -> FactCheckAnalysis:
        """Search for fact checks about this media outlet."""
        logger.info(f"Searching for fact checks on: {domain}")

        results = []
        failed_count = 0

        # Extract domain name for search
        if outlet_name is None:
            outlet_name = domain.replace('www.', '').split('.')[0]

        # Search for fact checks from multiple sources
        search_queries = [
            f'"{domain}" fact check false',
            f'"{outlet_name}" misinformation debunked',
            f'site:politifact.com "{outlet_name}"',
            f'site:snopes.com "{outlet_name}"',
            f'site:factcheck.org "{outlet_name}"'
        ]

        seen_urls = set()

        for query in search_queries[:3]:  # Limit to 3 searches to avoid rate limits
            try:
                search_results = self.search_tool.run(query)

                # Parse results for fact check indicators
                if search_results:
                    # Check for negative verdicts
                    negative_indicators = ['false', 'pants on fire', 'misleading',
                                          'mostly false', 'incorrect', 'debunked',
                                          'misinformation', 'unproven', 'fake']

                    for indicator in negative_indicators:
                        if indicator.lower() in search_results.lower():
                            # Count unique fact checks (rough heuristic)
                            matches = re.findall(r'(https?://[^\s]+)', search_results)
                            for url in matches:
                                if url not in seen_urls:
                                    seen_urls.add(url)
                                    # Check if from credible fact checker
                                    for checker in IFCN_FACT_CHECKERS:
                                        if checker in url:
                                            failed_count += 1
                                            results.append(FactCheckResult(
                                                source=checker,
                                                claim="Found via search",
                                                verdict=indicator,
                                                url=url
                                            ))
                                            break
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                continue

        # Also use LLM to estimate based on its knowledge
        llm_estimate = self._llm_fact_check_estimate(domain, outlet_name)

        # Combine search results with LLM estimate (take higher)
        total_failed = max(failed_count, llm_estimate)

        # Cap at 10 for scoring
        score = min(10, total_failed)

        details = f"Found {failed_count} fact check failures via search, LLM estimated {llm_estimate}"

        return FactCheckAnalysis(
            failed_checks_count=total_failed,
            score=float(score),
            results=results,
            details=details
        )

    def analyze(self, url: str) -> float:
        """
        Analyze a URL for fact check failures.
        Returns the score (0-10) for use by profiler.
        """
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace('www.', '')
        result = self.search(domain)
        return result.score

    def _llm_fact_check_estimate(self, domain: str, outlet_name: str) -> int:
        """Use LLM knowledge to estimate fact check record."""
        prompt = f"""
        Based on your knowledge, estimate the number of FAILED fact checks for the media outlet: {outlet_name} ({domain})

        Consider:
        - Has this outlet been fact-checked by PolitiFact, Snopes, FactCheck.org, etc.?
        - Has it published known misinformation, conspiracy theories, or pseudoscience?
        - What is its general reputation for accuracy?

        Return ONLY a single number (0-10) representing estimated failed fact checks.
        0 = No known fact check failures, excellent record
        1-2 = Minor issues, generally reliable
        3-5 = Some fact check failures, mixed record
        6-8 = Frequent issues with accuracy
        9-10 = Known for spreading misinformation

        Return ONLY the number, nothing else.
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            estimate = int(res.content.strip())
            return min(10, max(0, estimate))
        except:
            return 0


class SourcingAnalyzer:
    """
    Analyzes source quality and citation practices (25% of Factuality Score).

    Score 0-10 where:
    0 = Perfect sourcing with credible references
    5 = Mixed sourcing (no hyperlinks automatically scores 5+)
    10 = No sourcing or reliance on discredited sources
    """

    def analyze(self, articles: List[Article]) -> SourcingAnalysis:
        if not articles:
            return SourcingAnalysis(
                score=5.0,
                avg_sources_per_article=0,
                has_hyperlinks=False,
                credible_source_ratio=0.0,
                details="No articles to analyze"
            )

        total_sources = 0
        articles_with_sources = 0
        credible_sources = 0
        total_external_sources = 0

        # Credible source domains
        credible_domains = [
            'reuters.com', 'apnews.com', 'bbc.com', 'nytimes.com',
            'washingtonpost.com', 'theguardian.com', 'npr.org',
            '.gov', '.edu', 'who.int', 'un.org', 'cdc.gov',
            'nature.com', 'sciencemag.org', 'pubmed.ncbi.nlm.nih.gov'
        ]

        for article in articles:
            if article.has_sources and article.source_links:
                articles_with_sources += 1
                total_sources += len(article.source_links)

                for link in article.source_links:
                    total_external_sources += 1
                    for credible in credible_domains:
                        if credible in link.lower():
                            credible_sources += 1
                            break

        avg_sources = total_sources / len(articles) if articles else 0
        has_hyperlinks = articles_with_sources > 0
        credible_ratio = credible_sources / total_external_sources if total_external_sources > 0 else 0

        # Calculate score
        score = 0.0

        # No hyperlinks = automatic 5+ score
        if not has_hyperlinks:
            score = 5.0
            details = "No hyperlinks or external citations found"
        else:
            # Base score on average sources per article
            if avg_sources >= 3:
                score = 0.0
            elif avg_sources >= 2:
                score = 1.0
            elif avg_sources >= 1:
                score = 2.0
            elif avg_sources >= 0.5:
                score = 3.0
            else:
                score = 4.0

            # Adjust for credibility of sources
            if credible_ratio < 0.3:
                score += 2.0
            elif credible_ratio < 0.5:
                score += 1.0

            details = f"Avg {avg_sources:.1f} sources/article, {credible_ratio*100:.0f}% credible"

        score = min(10.0, score)

        return SourcingAnalysis(
            score=score,
            avg_sources_per_article=avg_sources,
            has_hyperlinks=has_hyperlinks,
            credible_source_ratio=credible_ratio,
            details=details
        )


class PropagandaAnalyzer:
    """
    Analyzes propaganda/one-sidedness using DeBERTa model or LLM (10% of Factuality Score).

    Score 0-10 based on propaganda instances found:
    0 = No propaganda, perfectly balanced
    5 = Some bias and emotional language
    10 = Extreme propaganda
    """

    def __init__(self, use_local_model: bool = False):
        self.use_local_model = use_local_model

    def analyze(self, articles: List[Article]) -> PropagandaAnalysis:
        logger.info(f"Analyzing Propaganda (Mode: {'LOCAL DeBERTa' if self.use_local_model else 'LLM'})")

        combined_text = "\n".join([a.text[:1500] for a in articles[:5]])
        instances = []

        if self.use_local_model:
            # Use trained DeBERTa SI + TC models
            global local_detector
            if local_detector is None:
                local_detector = LocalPropagandaDetector()

            if local_detector.ready:
                raw_findings = local_detector.detect(combined_text)

                for item in raw_findings:
                    instances.append(PropagandaInstance(
                        text_snippet=item["text_snippet"],
                        technique=item["technique"],
                        confidence=item["confidence"],
                        context=item.get("context", "")
                    ))
            else:
                logger.warning("Local model not ready, falling back to LLM")
                instances = self._llm_analyze(combined_text)
        else:
            instances = self._llm_analyze(combined_text)

        # Calculate score based on findings (0-10 scale)
        # More instances = higher score (worse)
        score = min(10.0, len(instances) * 1.5)

        return PropagandaAnalysis(score=score, instances=instances)

    def _llm_analyze(self, text: str) -> List[PropagandaInstance]:
        """Fallback LLM-based propaganda detection."""
        prompt = f"""
        Identify propaganda techniques (SemEval-2020 Task 11 categories):
        {json.dumps(PROPAGANDA_TECHNIQUES, indent=2)}

        Analyze this text and find instances of these techniques.
        Return a JSON object with 'findings' array containing objects with:
        - "technique": exact technique name from list above
        - "text_snippet": the exact text containing propaganda
        - "context": the full sentence for context
        - "confidence": confidence score 0.0-1.0

        TEXT: {text[:3000]}

        Return ONLY valid JSON, no other text.
        """
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)
            return [PropagandaInstance(**item) for item in data.get("findings", [])]
        except Exception as e:
            logger.error(f"LLM Propaganda analysis failed: {e}")
            return []


# =============================================================================
# SCORING CALCULATOR (MBFC Methodology)
# =============================================================================

class ScoringCalculator:
    """
    Implements MBFC scoring methodology exactly as specified.

    BIAS: Economic (35%) + Social (35%) + News Reporting (15%) + Editorial (15%)
    FACTUALITY: Fact Checks (40%) + Sourcing (25%) + Transparency (25%) + Propaganda (10%)
    CREDIBILITY: Based on factuality + bias + traffic + freedom penalty
    """

    @staticmethod
    def get_bias_label(score: float) -> str:
        """Maps weighted bias score (-10 to +10) to MBFC label."""
        for low, high, label in BIAS_RANGES:
            if low <= score <= high:
                return label
        return "Least Biased"

    @staticmethod
    def get_factuality_label(score: float) -> str:
        """Maps weighted factuality score (0-10, lower is better) to MBFC label."""
        for low, high, label in FACTUALITY_RANGES:
            if low <= score <= high:
                return label
        return "Mixed"

    @staticmethod
    def calculate_bias(
        economic_score: float,
        social_score: float,
        news_reporting_score: float,
        editorial_score: float
    ) -> BiasAnalysis:
        """
        Calculate weighted bias score per MBFC methodology.

        Weights:
        - Economic System: 35%
        - Social Values: 35%
        - News Reporting Balance: 15%
        - Editorial Bias: 15%

        Returns BiasAnalysis with all component scores and final label.
        """
        w = BiasWeights()

        weighted_total = (
            (economic_score * w.economic) +
            (social_score * w.social) +
            (news_reporting_score * w.reporting) +
            (editorial_score * w.editorial)
        )

        weighted_total = round(weighted_total, 2)
        final_label = ScoringCalculator.get_bias_label(weighted_total)

        return BiasAnalysis(
            economic_label="",  # Filled by caller
            economic_score=economic_score,
            social_label="",
            social_score=social_score,
            news_reporting_label="",
            news_reporting_score=news_reporting_score,
            editorial_label="",
            editorial_score=editorial_score,
            weighted_total=weighted_total,
            final_label=final_label
        )

    @staticmethod
    def calculate_factuality(
        fact_check_score: float,
        sourcing_score: float,
        transparency_score: float,
        propaganda_score: float
    ) -> FactualAnalysis:
        """
        Calculate weighted factuality score per MBFC methodology.

        Weights:
        - Failed Fact Checks: 40%
        - Sourcing: 25%
        - Transparency: 25%
        - One-Sidedness/Propaganda: 10%

        Returns FactualAnalysis with all component scores and final label.
        """
        w = FactualWeights()

        weighted_total = (
            (fact_check_score * w.failed_fact_checks) +
            (sourcing_score * w.sourcing) +
            (transparency_score * w.transparency) +
            (propaganda_score * w.bias_propaganda)
        )

        weighted_total = round(weighted_total, 2)
        final_label = ScoringCalculator.get_factuality_label(weighted_total)

        return FactualAnalysis(
            fact_check_score=fact_check_score,
            sourcing_score=sourcing_score,
            transparency_score=transparency_score,
            propaganda_score=propaganda_score,
            weighted_total=weighted_total,
            final_label=final_label
        )

    @staticmethod
    def calculate_credibility(
        factuality_label: str,
        bias_label: str,
        traffic_points: int,
        freedom_penalty: int
    ) -> tuple:
        """
        Calculate MBFC credibility score (0-10 scale).

        Components:
        - Factual Reporting Points (0-4)
        - Bias Points (0-3)
        - Traffic/Longevity Points (0-3)
        - Freedom Penalty (-2 to 0)

        Returns: (total_score, credibility_level, fact_pts, bias_pts)
        """
        # Factual points
        fact_pts = CREDIBILITY_POINTS["factual"].get(factuality_label, 1)

        # Bias points
        bias_pts = 0
        for key, val in CREDIBILITY_POINTS["bias"].items():
            if key.lower() in bias_label.lower():
                bias_pts = val
                break

        # Calculate total
        total = fact_pts + bias_pts + traffic_points + freedom_penalty
        total = max(0, min(10, total))  # Clamp to 0-10

        # Determine credibility level
        if total >= 6:
            level = "High Credibility"
        elif total >= 3:
            level = "Medium Credibility"
        else:
            level = "Low Credibility"

        # MBFC special rule: Mostly Factual with 3.6-4.5 = Medium regardless
        if factuality_label == "Mostly Factual" and 3.6 <= total <= 4.5:
            level = "Medium Credibility"

        # Automatic Low for Questionable/Conspiracy/Pseudoscience
        questionable_indicators = ["Questionable", "Conspiracy", "Pseudoscience", "Very Low"]
        if any(ind in factuality_label for ind in questionable_indicators):
            level = "Low Credibility"

        return total, level, fact_pts, bias_pts
