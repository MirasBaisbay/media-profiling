"""
Refactored analyzers using deterministic data retrieval and structured LLM outputs.

This module replaces heuristic-based methods with:
- LangChain's .with_structured_output() for type-safe LLM responses
- python-whois for deterministic domain age data
- DuckDuckGo search for external information gathering

Classes:
    OpinionAnalyzer: Classifies articles using LLM content analysis (no URL heuristics)
    TrafficLongevityAnalyzer: Gets domain age from WHOIS + traffic from search/LLM
    MediaTypeAnalyzer: Classifies media type using Wikipedia search + LLM
"""

import logging
import re
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

import whois
from duckduckgo_search import DDGS
from langchain_openai import ChatOpenAI

from schemas import (
    ArticleClassification,
    ArticleType,
    MediaType,
    MediaTypeClassification,
    TrafficData,
    TrafficEstimate,
    TrafficTier,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Configuration
# =============================================================================


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.0) -> ChatOpenAI:
    """
    Get a configured LLM instance.

    Args:
        model: The OpenAI model to use
        temperature: Temperature setting (0 for deterministic)

    Returns:
        Configured ChatOpenAI instance
    """
    return ChatOpenAI(model=model, temperature=temperature)


# =============================================================================
# OpinionAnalyzer
# =============================================================================


class OpinionAnalyzer:
    """
    Analyzes article content to classify it as News, Opinion, Satire, or PR.

    This analyzer uses pure content analysis via LLM - it does NOT rely on
    URL patterns or title heuristics. Classification is based on:
    - Writing style and tone
    - Use of first-person vs third-person
    - Presence of subjective language
    - Factual reporting vs commentary patterns

    Attributes:
        llm: The LangChain LLM with structured output binding
        max_text_chars: Maximum characters of text to analyze (default 1000)
    """

    SYSTEM_PROMPT = """You are an expert media analyst specializing in distinguishing
between different types of journalistic content. Your task is to classify articles
based on their actual content, writing style, and journalistic intent.

Classification Guidelines:

NEWS:
- Objective, fact-based reporting
- Third-person perspective
- Balanced presentation of multiple viewpoints
- Attribution to sources
- Inverted pyramid structure (most important facts first)
- Minimal adjectives and loaded language

OPINION:
- First-person perspective or clear editorial voice
- Subjective analysis and commentary
- Author's personal views and judgments
- Persuasive language and arguments
- May include "I think", "we should", "in my view"
- Editorial, op-ed, column, or analysis pieces

SATIRE:
- Exaggerated or absurd scenarios
- Ironic or sarcastic tone
- Humorous intent
- Clearly implausible claims played straight
- Parody of news formats

PR (Press Release / Promotional):
- Promotional language about a company/product/person
- One-sided positive framing
- Corporate speak and marketing language
- Quotes primarily from the subject being promoted
- Announcement-style structure

Analyze the CONTENT and STYLE, not the URL or publication name."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_text_chars: int = 1000,
    ):
        """
        Initialize the OpinionAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
            max_text_chars: Maximum characters of article text to analyze
        """
        self.llm = get_llm(model, temperature).with_structured_output(
            ArticleClassification
        )
        self.max_text_chars = max_text_chars

    def analyze(self, title: str, text: str) -> ArticleClassification:
        """
        Classify an article based on its title and text content.

        Args:
            title: The article headline/title
            text: The article body text

        Returns:
            ArticleClassification with type, confidence, and reasoning
        """
        # Truncate text to max_text_chars
        text_snippet = text[: self.max_text_chars] if text else ""

        user_prompt = f"""Classify the following article:

TITLE: {title}

TEXT (first {self.max_text_chars} characters):
{text_snippet}

Based on the writing style, tone, and content, classify this article."""

        try:
            result: ArticleClassification = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"OpinionAnalyzer failed: {e}")
            # Return a safe default with low confidence
            return ArticleClassification(
                article_type=ArticleType.NEWS,
                confidence=0.0,
                reasoning=f"Classification failed due to error: {str(e)}",
            )

    def analyze_batch(
        self, articles: list[dict[str, str]]
    ) -> list[ArticleClassification]:
        """
        Classify multiple articles.

        Args:
            articles: List of dicts with 'title' and 'text' keys

        Returns:
            List of ArticleClassification results
        """
        results = []
        for article in articles:
            result = self.analyze(
                title=article.get("title", ""),
                text=article.get("text", ""),
            )
            results.append(result)
        return results


# =============================================================================
# TrafficLongevityAnalyzer
# =============================================================================


class TrafficLongevityAnalyzer:
    """
    Analyzes domain traffic and longevity using deterministic and LLM methods.

    Domain age is retrieved deterministically via python-whois.
    Traffic estimation uses DuckDuckGo search + LLM parsing of results.

    Attributes:
        llm: LangChain LLM with structured output for traffic parsing
        search: DuckDuckGo search instance
    """

    TRAFFIC_PARSE_PROMPT = """You are analyzing search results to estimate website traffic.

Based on the search snippet provided, determine the traffic tier:

HIGH: Major websites with millions of monthly visits (>10M)
      - Nationally/globally recognized brands
      - Major news outlets, social media, e-commerce giants

MEDIUM: Established websites with hundreds of thousands to millions of visits (100K-10M)
        - Regional news outlets
        - Popular niche websites
        - Well-known blogs or specialty sites

LOW: Smaller websites with tens of thousands of visits (10K-100K)
     - Local news sites
     - Small business websites
     - Niche community sites

MINIMAL: Very small websites with under 10K visits
         - Personal blogs
         - New or obscure websites
         - Sites with little web presence

UNKNOWN: If the search results don't provide enough information

Look for indicators like:
- Explicit traffic numbers (e.g., "10M monthly visits")
- Alexa/Similarweb rankings
- Descriptions of reach/popularity
- Comparisons to known sites"""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the TrafficLongevityAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
        """
        self.llm = get_llm(model, temperature).with_structured_output(TrafficEstimate)
        self.search = DDGS()

    def _extract_domain(self, url: str) -> str:
        """Extract the root domain from a URL."""
        parsed = urlparse(url if url.startswith("http") else f"https://{url}")
        domain = parsed.netloc or parsed.path
        # Remove www. prefix
        domain = re.sub(r"^www\.", "", domain)
        return domain

    def _get_whois_data(self, domain: str) -> tuple[Optional[date], bool]:
        """
        Get domain creation date from WHOIS.

        Args:
            domain: The domain to look up

        Returns:
            Tuple of (creation_date, success_flag)
        """
        try:
            w = whois.whois(domain)
            creation_date = w.creation_date

            # Handle list of dates (some registrars return multiple)
            if isinstance(creation_date, list):
                creation_date = creation_date[0]

            # Convert datetime to date if needed
            if isinstance(creation_date, datetime):
                creation_date = creation_date.date()

            if creation_date:
                return creation_date, True

            return None, False

        except Exception as e:
            logger.warning(f"WHOIS lookup failed for {domain}: {e}")
            return None, False

    def _calculate_age_years(self, creation_date: Optional[date]) -> Optional[float]:
        """Calculate domain age in years from creation date."""
        if not creation_date:
            return None
        today = date.today()
        delta = today - creation_date
        return round(delta.days / 365.25, 2)

    def _search_traffic_info(self, domain: str) -> Optional[str]:
        """
        Search for traffic information about a domain.

        Args:
            domain: The domain to search for

        Returns:
            Search result snippet or None
        """
        query = f"{domain} monthly visits similarweb"
        try:
            results = list(self.search.text(query, max_results=3))
            if results:
                # Combine top results into a snippet
                snippets = []
                for r in results[:3]:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    snippets.append(f"{title}: {body}")
                return "\n".join(snippets)
            return None
        except Exception as e:
            logger.warning(f"Traffic search failed for {domain}: {e}")
            return None

    def _parse_traffic_with_llm(self, domain: str, snippet: str) -> TrafficEstimate:
        """
        Use LLM to parse traffic tier from search snippet.

        Args:
            domain: The domain being analyzed
            snippet: Search result snippet to parse

        Returns:
            TrafficEstimate with tier and reasoning
        """
        user_prompt = f"""Analyze the following search results for {domain} and estimate the traffic tier:

SEARCH RESULTS:
{snippet}

Determine the traffic tier based on any traffic data, rankings, or popularity indicators found."""

        try:
            result: TrafficEstimate = self.llm.invoke(
                [
                    {"role": "system", "content": self.TRAFFIC_PARSE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"Traffic LLM parsing failed: {e}")
            return TrafficEstimate(
                traffic_tier=TrafficTier.UNKNOWN,
                monthly_visits_estimate=None,
                confidence=0.0,
                reasoning=f"LLM parsing failed: {str(e)}",
            )

    def analyze(self, url_or_domain: str) -> TrafficData:
        """
        Analyze traffic and longevity for a domain.

        Args:
            url_or_domain: URL or domain to analyze

        Returns:
            TrafficData with WHOIS and traffic information
        """
        domain = self._extract_domain(url_or_domain)

        # Get deterministic WHOIS data
        creation_date, whois_success = self._get_whois_data(domain)
        age_years = self._calculate_age_years(creation_date)

        # Search for traffic information
        traffic_snippet = self._search_traffic_info(domain)

        # Parse traffic with LLM
        if traffic_snippet:
            traffic_estimate = self._parse_traffic_with_llm(domain, traffic_snippet)
        else:
            traffic_estimate = TrafficEstimate(
                traffic_tier=TrafficTier.UNKNOWN,
                monthly_visits_estimate=None,
                confidence=0.0,
                reasoning="No search results found for traffic estimation",
            )

        return TrafficData(
            domain=domain,
            creation_date=creation_date,
            age_years=age_years,
            traffic_tier=traffic_estimate.traffic_tier,
            monthly_visits_estimate=traffic_estimate.monthly_visits_estimate,
            traffic_confidence=traffic_estimate.confidence,
            whois_success=whois_success,
            traffic_search_snippet=traffic_snippet[:500] if traffic_snippet else None,
        )


# =============================================================================
# MediaTypeAnalyzer
# =============================================================================


class MediaTypeAnalyzer:
    """
    Classifies media outlet type using Wikipedia/web search and LLM.

    This analyzer does NOT use article content for classification.
    Instead, it searches for "{site_name} wikipedia type of media"
    and uses the LLM to interpret the results.

    Attributes:
        llm: LangChain LLM with structured output binding
        search: DuckDuckGo search instance
    """

    SYSTEM_PROMPT = """You are classifying the type of media outlet based on search results.

Media Type Definitions:

TV: Television broadcast networks and channels
    - Examples: CNN, BBC, Fox News, NBC, ABC

NEWSPAPER: Traditional print newspapers (may have online presence)
    - Examples: New York Times, Washington Post, The Guardian

WEBSITE: Online-only news or content sites (digital native)
    - Examples: Vox, BuzzFeed News, The Daily Wire, Axios

MAGAZINE: Periodical publications (weekly/monthly)
    - Examples: Time, The Economist, The Atlantic, Newsweek

RADIO: Radio broadcast networks
    - Examples: NPR, BBC Radio, iHeartRadio

NEWS_AGENCY: Wire services that provide content to other outlets
    - Examples: Reuters, Associated Press (AP), AFP

BLOG: Personal or small group blogs
    - Usually individual authors, informal style

PODCAST: Audio-first media
    - Examples: The Daily, Pod Save America

STREAMING: Streaming-first media services
    - Examples: Netflix, YouTube (when used as primary platform)

UNKNOWN: Cannot determine from available information

Classify based on the PRIMARY format of the outlet, not secondary formats.
For example, NYT is a NEWSPAPER even though they have a website and podcasts."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        """
        Initialize the MediaTypeAnalyzer.

        Args:
            model: OpenAI model to use
            temperature: LLM temperature (0 for deterministic)
        """
        self.llm = get_llm(model, temperature).with_structured_output(
            MediaTypeClassification
        )
        self.search = DDGS()

    def _extract_site_name(self, url_or_domain: str) -> str:
        """
        Extract a clean site name from URL or domain.

        Args:
            url_or_domain: URL or domain string

        Returns:
            Clean site name for searching
        """
        # Parse URL
        parsed = urlparse(
            url_or_domain if url_or_domain.startswith("http") else f"https://{url_or_domain}"
        )
        domain = parsed.netloc or parsed.path

        # Remove www. and TLD
        domain = re.sub(r"^www\.", "", domain)

        # Extract site name (remove TLD for cleaner search)
        site_name = domain.split(".")[0]

        # Title case for better search results
        return site_name.replace("-", " ").replace("_", " ").title()

    def _search_media_type(self, site_name: str, domain: str) -> Optional[str]:
        """
        Search for media type information.

        Args:
            site_name: Clean site name
            domain: Full domain for fallback search

        Returns:
            Search result snippet or None
        """
        # Primary query targeting Wikipedia
        query = f"{site_name} wikipedia type of media"

        try:
            results = list(self.search.text(query, max_results=3))

            if not results:
                # Fallback query
                query = f"{domain} news outlet type wikipedia"
                results = list(self.search.text(query, max_results=3))

            if results:
                snippets = []
                for r in results[:3]:
                    title = r.get("title", "")
                    body = r.get("body", "")
                    snippets.append(f"{title}: {body}")
                return "\n".join(snippets)

            return None

        except Exception as e:
            logger.warning(f"Media type search failed for {site_name}: {e}")
            return None

    def analyze(self, url_or_domain: str) -> MediaTypeClassification:
        """
        Classify the media type for a given URL or domain.

        Args:
            url_or_domain: URL or domain to classify

        Returns:
            MediaTypeClassification with type, confidence, and reasoning
        """
        # Extract site name and domain
        parsed = urlparse(
            url_or_domain if url_or_domain.startswith("http") else f"https://{url_or_domain}"
        )
        domain = parsed.netloc or parsed.path
        domain = re.sub(r"^www\.", "", domain)
        site_name = self._extract_site_name(url_or_domain)

        # Search for media type information
        search_snippet = self._search_media_type(site_name, domain)

        if not search_snippet:
            return MediaTypeClassification(
                media_type=MediaType.UNKNOWN,
                confidence=0.0,
                source_snippet="No search results found",
                reasoning="Could not find information about this media outlet",
            )

        # Use LLM to classify
        user_prompt = f"""Classify the media type for: {site_name} ({domain})

SEARCH RESULTS:
{search_snippet}

Based on these search results, what type of media outlet is this?"""

        try:
            result: MediaTypeClassification = self.llm.invoke(
                [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return result

        except Exception as e:
            logger.error(f"MediaTypeAnalyzer LLM call failed: {e}")
            return MediaTypeClassification(
                media_type=MediaType.UNKNOWN,
                confidence=0.0,
                source_snippet=search_snippet[:200] if search_snippet else "",
                reasoning=f"Classification failed: {str(e)}",
            )


# =============================================================================
# Convenience Functions
# =============================================================================


def classify_article(title: str, text: str) -> ArticleClassification:
    """
    Convenience function to classify a single article.

    Args:
        title: Article headline
        text: Article body text

    Returns:
        ArticleClassification result
    """
    analyzer = OpinionAnalyzer()
    return analyzer.analyze(title, text)


def get_traffic_data(url_or_domain: str) -> TrafficData:
    """
    Convenience function to get traffic and longevity data.

    Args:
        url_or_domain: URL or domain to analyze

    Returns:
        TrafficData result
    """
    analyzer = TrafficLongevityAnalyzer()
    return analyzer.analyze(url_or_domain)


def classify_media_type(url_or_domain: str) -> MediaTypeClassification:
    """
    Convenience function to classify media type.

    Args:
        url_or_domain: URL or domain to classify

    Returns:
        MediaTypeClassification result
    """
    analyzer = MediaTypeAnalyzer()
    return analyzer.analyze(url_or_domain)


# =============================================================================
# CLI / Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    # Example usage
    print("=" * 60)
    print("REFACTORED ANALYZERS - DEMO")
    print("=" * 60)

    # Test OpinionAnalyzer
    print("\n1. OpinionAnalyzer Test")
    print("-" * 40)
    sample_title = "The Economic Impact of Climate Change"
    sample_text = """
    Climate change is causing significant economic disruption across multiple sectors.
    According to a new report from the World Bank, global GDP could decline by 23%
    by 2100 if emissions continue at current levels. The study analyzed data from
    150 countries and found that agricultural yields have already decreased by 5%
    in affected regions. Economists warn that the poorest nations will be hit hardest.
    """
    opinion_analyzer = OpinionAnalyzer()
    result = opinion_analyzer.analyze(sample_title, sample_text)
    print(f"Title: {sample_title}")
    print(f"Classification: {result.article_type.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")

    # Test TrafficLongevityAnalyzer
    print("\n2. TrafficLongevityAnalyzer Test")
    print("-" * 40)
    test_domain = "bbc.com"
    traffic_analyzer = TrafficLongevityAnalyzer()
    traffic_data = traffic_analyzer.analyze(test_domain)
    print(f"Domain: {traffic_data.domain}")
    print(f"Creation Date: {traffic_data.creation_date}")
    print(f"Age (years): {traffic_data.age_years}")
    print(f"Traffic Tier: {traffic_data.traffic_tier.value}")
    print(f"WHOIS Success: {traffic_data.whois_success}")

    # Test MediaTypeAnalyzer
    print("\n3. MediaTypeAnalyzer Test")
    print("-" * 40)
    test_outlet = "nytimes.com"
    media_analyzer = MediaTypeAnalyzer()
    media_result = media_analyzer.analyze(test_outlet)
    print(f"Outlet: {test_outlet}")
    print(f"Media Type: {media_result.media_type.value}")
    print(f"Confidence: {media_result.confidence:.2f}")
    print(f"Reasoning: {media_result.reasoning}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
