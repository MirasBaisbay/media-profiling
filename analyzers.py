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
from editorial_bias_detection import (
    StructuredEditorialBiasAnalyzer as _StructuredEditorialBiasAnalyzer,
    score_to_editorial_label,
    EditorialBiasAnalysis
)

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
            # Try multiple encodings for CSV files with special characters
            for encoding in ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(FREEDOM_INDEX_FILE, mode='r', encoding=encoding) as f:
                        reader = csv.DictReader(f, delimiter=';')
                        for row in reader:
                            if row.get('ISO') == iso3:
                                score_str = row.get('Score 2025', '0').replace(',', '.')
                                score = float(score_str)
                                found = True
                                break
                    if found:
                        break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error reading Freedom CSV with {encoding}: {e}")
                    break
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

    Uses the structured editorial bias detection approach:
    1. Clickbait detection in headlines (pattern matching)
    2. Loaded language analysis (lexicon-based + LLM)
    3. Emotional manipulation scoring
    4. Political direction detection

    Based on:
    - Recasens et al. (2013) - Linguistic Models for Bias Detection
    - Chakraborty et al. (2016) - Clickbait Detection
    - QCRI Emotional Language Analysis
    """

    def __init__(self):
        self._structured_analyzer = _StructuredEditorialBiasAnalyzer()

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Analyze editorial bias using structured approach.

        Returns:
            dict with:
            - label: MBFC editorial bias label
            - score: -10 to +10 score
            - details: Additional analysis details
        """
        try:
            # Use structured analyzer (prefers opinion/editorial articles)
            result: EditorialBiasAnalysis = self._structured_analyzer.analyze(
                articles,
                prefer_opinion=True
            )

            return {
                "label": result.overall_label,
                "score": result.overall_score,
                "details": {
                    "direction": result.direction,
                    "clickbait_score": result.clickbait_score,
                    "loaded_language_score": result.loaded_language_score,
                    "emotional_manipulation_score": result.emotional_manipulation_score,
                    "methodology": result.methodology_notes
                }
            }
        except Exception as e:
            logger.error(f"Editorial bias analysis failed: {e}")
            return {"label": "Neutral/Balanced Editorial", "score": 0.0, "details": {}}


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


# =============================================================================
# IDEOLOGY DECISION TREE ANALYZER
# =============================================================================

@dataclass
class TopicAnalysisResult:
    """Result of analyzing a single topic."""
    topic_name: str
    is_relevant: bool  # From preliminary check
    stance: Optional[str]  # "left", "right", or None
    score: Optional[float]  # -10 to +10, or None if not relevant
    matched_question_id: Optional[str]
    evidence: str
    references: List[Dict[str, str]]


@dataclass
class DimensionAnalysisResult:
    """Result of analyzing a dimension (economic or social)."""
    dimension_name: str
    topic_results: List[TopicAnalysisResult]
    average_score: Optional[float]  # Average of relevant topics, None if no relevant topics
    relevant_topic_count: int


@dataclass
class IdeologyAnalysisResult:
    """Complete result of ideology analysis."""
    economic_analysis: DimensionAnalysisResult
    social_analysis: DimensionAnalysisResult
    combined_economic_score: Optional[float]
    combined_social_score: Optional[float]
    methodology_notes: str


class IdeologyDecisionTreeAnalyzer:
    """
    Implements a recursive decision-tree approach to ideology detection.

    Uses the ideology_question_bank.json to evaluate articles through:
    1. Topic Filter: Run preliminary_check for each topic
       - If NO: Score as None (exclude from average)
       - If YES: Proceed to stance detection

    2. Stance Detection (The Fork):
       - Ask LLM: "Does this article lean LEFT or RIGHT on this topic?"
       - If Left: Execute left_leaning branch starting from L1 (Extreme)
       - If Right: Execute right_leaning branch starting from R4 (Extreme)

    3. The "Stop" Condition (check extremes FIRST):
       - Left Branch: L1 (-10) -> L2 (-7.5) -> L3 (-5) -> L4 (-2.5) -> Centrism
       - Right Branch: R4 (+10) -> R3 (+7.5) -> R2 (+5) -> R1 (+2.5) -> Centrism

    WHY CHECK EXTREMES FIRST:
    If you ask the Moderate question first (e.g., "Do you support unions?"),
    a Communist would say "Yes." You would incorrectly score them as -5 instead of -10.
    """

    def __init__(self, question_bank_path: str = "ideology_question_bank.json"):
        self.question_bank = self._load_question_bank(question_bank_path)

    def _load_question_bank(self, path: str) -> Dict[str, Any]:
        """Load the ideology question bank from JSON file."""
        try:
            # Try relative path first
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Try path relative to this file
            module_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(module_dir, path)
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            logger.warning(f"Question bank not found at {path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load question bank: {e}")
            return {}

    def analyze(self, articles: List[Article]) -> IdeologyAnalysisResult:
        """
        Perform complete ideology analysis on articles.

        Returns IdeologyAnalysisResult with scores for economic and social dimensions.
        """
        logger.info(f"Starting ideology decision tree analysis on {len(articles)} articles")

        if not self.question_bank:
            return IdeologyAnalysisResult(
                economic_analysis=DimensionAnalysisResult("Economic System", [], None, 0),
                social_analysis=DimensionAnalysisResult("Social Values", [], None, 0),
                combined_economic_score=None,
                combined_social_score=None,
                methodology_notes="Question bank not loaded"
            )

        # Prepare article content for analysis
        combined_text = self._prepare_article_text(articles)

        # Analyze economic dimension
        economic_result = self._analyze_dimension(
            dimension_name="Economic System",
            dimension_key="economic_system",
            combined_text=combined_text,
            branch_left_key="left_leaning",
            branch_right_key="right_leaning"
        )

        # Analyze social dimension
        social_result = self._analyze_dimension(
            dimension_name="Social Values",
            dimension_key="social_values",
            combined_text=combined_text,
            branch_left_key="progressive_leaning",
            branch_right_key="conservative_leaning"
        )

        methodology_notes = self._build_methodology_notes(economic_result, social_result)

        return IdeologyAnalysisResult(
            economic_analysis=economic_result,
            social_analysis=social_result,
            combined_economic_score=economic_result.average_score,
            combined_social_score=social_result.average_score,
            methodology_notes=methodology_notes
        )

    def _prepare_article_text(self, articles: List[Article]) -> str:
        """Prepare combined article text for analysis."""
        texts = []
        for article in articles[:10]:  # Limit to 10 articles
            texts.append(f"HEADLINE: {article.title}\nCONTENT: {article.text[:800]}")
        return "\n\n---\n\n".join(texts)

    def _analyze_dimension(
        self,
        dimension_name: str,
        dimension_key: str,
        combined_text: str,
        branch_left_key: str,
        branch_right_key: str
    ) -> DimensionAnalysisResult:
        """Analyze a single dimension (economic or social)."""
        topic_results = []

        dimension_data = self.question_bank.get(dimension_key, {})
        topics = dimension_data.get("topics", {})

        for topic_name, topic_data in topics.items():
            result = self._analyze_topic(
                topic_name=topic_name,
                topic_data=topic_data,
                combined_text=combined_text,
                branch_left_key=branch_left_key,
                branch_right_key=branch_right_key
            )
            topic_results.append(result)

        # Calculate average score from relevant topics
        relevant_scores = [r.score for r in topic_results if r.is_relevant and r.score is not None]
        average_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else None

        return DimensionAnalysisResult(
            dimension_name=dimension_name,
            topic_results=topic_results,
            average_score=round(average_score, 2) if average_score is not None else None,
            relevant_topic_count=len(relevant_scores)
        )

    def _analyze_topic(
        self,
        topic_name: str,
        topic_data: Dict[str, Any],
        combined_text: str,
        branch_left_key: str,
        branch_right_key: str
    ) -> TopicAnalysisResult:
        """
        Analyze a single topic using the decision tree.

        Flow:
        1. Preliminary check (topic filter)
        2. Stance detection (left vs right)
        3. Execute appropriate branch checking extremes first
        """
        # Step 1: Preliminary Check (Topic Filter)
        preliminary = topic_data.get("preliminary_check", {})
        is_relevant = self._run_preliminary_check(preliminary, combined_text, topic_name)

        if not is_relevant:
            return TopicAnalysisResult(
                topic_name=topic_name,
                is_relevant=False,
                stance=None,
                score=None,
                matched_question_id=None,
                evidence="Topic not discussed in articles",
                references=[]
            )

        # Step 2: Stance Detection (The Fork)
        stance = self._detect_stance(combined_text, topic_name)

        # Step 3: Execute Branch (checking extremes FIRST)
        branch_questions = topic_data.get("branch_questions", {})
        centrism = branch_questions.get("centrism", {})

        if stance == "left":
            questions = branch_questions.get(branch_left_key, [])
            # Left branch: L1 (extreme -10) -> L2 -> L3 -> L4 (moderate -2.5)
            # Questions should already be ordered extreme to moderate in JSON
            score, question_id, evidence, refs = self._execute_branch(
                questions=questions,
                combined_text=combined_text,
                topic_name=topic_name,
                order="extreme_first"
            )
        elif stance == "right":
            questions = branch_questions.get(branch_right_key, [])
            # Right branch: R4 (extreme +10) -> R3 -> R2 -> R1 (moderate +2.5)
            # Need to reverse the order since JSON has R1 first
            score, question_id, evidence, refs = self._execute_branch(
                questions=questions,
                combined_text=combined_text,
                topic_name=topic_name,
                order="extreme_first_reverse"  # Start from R4, not R1
            )
        else:
            # Check centrism
            score, question_id, evidence, refs = self._check_centrism(
                centrism, combined_text, topic_name
            )

        # If no match in branch, check centrism
        if score is None and centrism:
            score, question_id, evidence, refs = self._check_centrism(
                centrism, combined_text, topic_name
            )

        return TopicAnalysisResult(
            topic_name=topic_name,
            is_relevant=True,
            stance=stance,
            score=score,
            matched_question_id=question_id,
            evidence=evidence,
            references=refs
        )

    def _run_preliminary_check(
        self,
        preliminary: Dict[str, Any],
        combined_text: str,
        topic_name: str
    ) -> bool:
        """
        Run the preliminary check to determine if topic is discussed.
        Returns True if topic is relevant, False otherwise.
        """
        if not preliminary:
            return False

        question = preliminary.get("question", "")
        if not question:
            return False

        prompt = f"""
Analyze the following articles and answer this question with ONLY "Yes" or "No":

QUESTION: {question}

ARTICLES:
{combined_text[:4000]}

Answer with ONLY "Yes" or "No". Nothing else.
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip().lower()
            return answer.startswith("yes")
        except Exception as e:
            logger.warning(f"Preliminary check failed for {topic_name}: {e}")
            return False

    def _detect_stance(self, combined_text: str, topic_name: str) -> str:
        """
        Detect whether articles lean LEFT or RIGHT on this topic.
        Returns: "left", "right", or "neutral"
        """
        prompt = f"""
Analyze the following articles regarding the topic: {topic_name}

Based on the perspective, framing, and arguments presented, does this content generally lean towards:
- LEFT/PROGRESSIVE perspectives (supporting government intervention, social equality, progressive values)
- RIGHT/CONSERVATIVE perspectives (supporting free markets, traditional values, limited government)
- NEUTRAL/BALANCED (no clear lean either direction)

ARTICLES:
{combined_text[:4000]}

Respond with ONLY one word: "left", "right", or "neutral"
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip().lower()
            if "left" in answer:
                return "left"
            elif "right" in answer:
                return "right"
            else:
                return "neutral"
        except Exception as e:
            logger.warning(f"Stance detection failed for {topic_name}: {e}")
            return "neutral"

    def _execute_branch(
        self,
        questions: List[Dict[str, Any]],
        combined_text: str,
        topic_name: str,
        order: str
    ) -> tuple:
        """
        Execute branch questions, checking extremes FIRST.

        order:
        - "extreme_first": Questions are already ordered extreme to moderate (left branch)
        - "extreme_first_reverse": Questions need to be reversed to go extreme to moderate (right branch)

        Returns: (score, question_id, evidence, references)
        """
        if not questions:
            return None, None, "No questions in branch", []

        # Order questions to check extremes first
        if order == "extreme_first_reverse":
            # Reverse the list so R4 (extreme +10) is checked before R1 (moderate +2.5)
            questions = list(reversed(questions))
        # For "extreme_first", questions are already in correct order (L1 extreme first)

        # Recursively check each question
        for question_data in questions:
            question_id = question_data.get("id", "")
            question = question_data.get("question", "")
            score_if_yes = question_data.get("score_if_yes")
            references = question_data.get("references", [])

            if not question:
                continue

            # Ask the LLM if this specific position is advocated
            answer, evidence = self._ask_question(question, combined_text, topic_name)

            if answer:  # If YES -> Score and STOP
                return score_if_yes, question_id, evidence, references

            # If NO -> Continue to next question (less extreme)

        # If all questions answered NO, return None to check centrism
        return None, None, "No match in branch", []

    def _ask_question(
        self,
        question: str,
        combined_text: str,
        topic_name: str
    ) -> tuple:
        """
        Ask a specific ideology question about the articles.
        Returns: (answer: bool, evidence: str)
        """
        prompt = f"""
Analyze the following articles regarding {topic_name}.

QUESTION: {question}

ARTICLES:
{combined_text[:4000]}

Based on the content, does this media source advocate for or support the position described in the question?

Respond in this exact JSON format:
{{"answer": "yes" or "no", "evidence": "Brief quote or summary supporting your answer"}}
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()

            data = json.loads(content)
            answer = data.get("answer", "no").lower() == "yes"
            evidence = data.get("evidence", "")

            return answer, evidence
        except Exception as e:
            logger.warning(f"Question evaluation failed: {e}")
            return False, f"Error: {str(e)}"

    def _check_centrism(
        self,
        centrism_data: Dict[str, Any],
        combined_text: str,
        topic_name: str
    ) -> tuple:
        """
        Check if content represents centrist/balanced position.
        Returns: (score, question_id, evidence, references)
        """
        if not centrism_data:
            return 0, None, "Default centrist score", []

        question = centrism_data.get("question", "")
        question_id = centrism_data.get("id", "")
        score_if_yes = centrism_data.get("score_if_yes", 0)
        references = centrism_data.get("references", [])

        if not question:
            return 0, question_id, "No centrism question defined", references

        answer, evidence = self._ask_question(question, combined_text, topic_name)

        if answer:
            return score_if_yes, question_id, evidence, references
        else:
            # If not centrist and no branch matched, return 0 as default
            return 0, question_id, "No clear position detected", references

    def _build_methodology_notes(
        self,
        economic: DimensionAnalysisResult,
        social: DimensionAnalysisResult
    ) -> str:
        """Build methodology notes for the analysis."""
        notes = []
        notes.append("Ideology Analysis Methodology:")
        notes.append("- Decision tree approach checking extremes FIRST")
        notes.append("- Topic relevance filtering via preliminary checks")
        notes.append("- Stance detection before branch execution")
        notes.append("")
        notes.append(f"Economic Dimension: {economic.relevant_topic_count} relevant topics analyzed")
        if economic.average_score is not None:
            notes.append(f"  Average Score: {economic.average_score:+.2f}")
        notes.append(f"Social Dimension: {social.relevant_topic_count} relevant topics analyzed")
        if social.average_score is not None:
            notes.append(f"  Average Score: {social.average_score:+.2f}")

        return "\n".join(notes)

    def get_economic_score(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Convenience method to get just economic score.
        Can be used as drop-in replacement for EconomicAnalyzer.
        """
        result = self.analyze(articles)
        score = result.combined_economic_score if result.combined_economic_score is not None else 0.0

        # Map score to label
        label = self._score_to_economic_label(score)

        return {"label": label, "score": score}

    def get_social_score(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Convenience method to get just social score.
        Can be used as drop-in replacement for SocialAnalyzer.
        """
        result = self.analyze(articles)
        score = result.combined_social_score if result.combined_social_score is not None else 0.0

        # Map score to label
        label = self._score_to_social_label(score)

        return {"label": label, "score": score}

    def _score_to_economic_label(self, score: float) -> str:
        """Map numeric score to economic ideology label."""
        if score <= -8:
            return "Communism"
        elif score <= -6:
            return "Socialism"
        elif score <= -4:
            return "Democratic Socialism"
        elif score <= -1.5:
            return "Regulated Market Economy"
        elif score <= 1.5:
            return "Centrism"
        elif score <= 4:
            return "Moderately Regulated Capitalism"
        elif score <= 6:
            return "Classical Liberalism"
        elif score <= 8:
            return "Libertarianism"
        else:
            return "Radical Laissez-Faire"

    def _score_to_social_label(self, score: float) -> str:
        """Map numeric score to social values label."""
        if score <= -8:
            return "Strong Progressive"
        elif score <= -6:
            return "Progressive"
        elif score <= -4:
            return "Moderate Progressive"
        elif score <= -1.5:
            return "Mild Progressive"
        elif score <= 1.5:
            return "Balanced"
        elif score <= 4:
            return "Mild Conservative"
        elif score <= 6:
            return "Moderate Conservative"
        elif score <= 8:
            return "Traditional Conservative"
        else:
            return "Strong Traditional Conservative"
