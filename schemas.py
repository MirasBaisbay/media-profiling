"""
Pydantic v2 schemas for structured LLM outputs in media bias analysis.

These schemas are designed to work with LangChain's .with_structured_output()
method to ensure deterministic, type-safe responses from LLM calls.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# =============================================================================
# Article Classification Schemas
# =============================================================================


class ArticleType(str, Enum):
    """Classification of article type based on content analysis."""

    NEWS = "News"
    OPINION = "Opinion"
    SATIRE = "Satire"
    PR = "PR"  # Press Release / Promotional content


class ArticleClassification(BaseModel):
    """
    Structured output for article type classification.

    Used by OpinionAnalyzer to classify articles into distinct categories
    based on content analysis (not URL/title heuristics).
    """

    article_type: ArticleType = Field(
        description="The classified type of the article"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


# =============================================================================
# Media Type Classification Schemas
# =============================================================================


class MediaType(str, Enum):
    """Classification of media outlet type."""

    TV = "TV"
    NEWSPAPER = "Newspaper"
    WEBSITE = "Website"
    MAGAZINE = "Magazine"
    RADIO = "Radio"
    NEWS_AGENCY = "News Agency"
    BLOG = "Blog"
    PODCAST = "Podcast"
    STREAMING = "Streaming Service"
    UNKNOWN = "Unknown"


class MediaTypeSource(str, Enum):
    """Source of media type classification."""

    LOOKUP = "Lookup"  # From known_media_types.csv (deterministic)
    LLM = "LLM"  # From search + LLM parsing
    FALLBACK = "Fallback"  # Default when no data available


class MediaTypeLLMOutput(BaseModel):
    """
    Structured output for LLM media type parsing.

    This is the schema used by the LLM when parsing search results.
    It does not include 'source' since that's determined by the analyzer.
    """

    media_type: MediaType = Field(
        description="The type of media outlet"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of how the media type was determined from search results"
    )


class MediaTypeClassification(BaseModel):
    """
    Complete media type classification result.

    Used by MediaTypeAnalyzer to return classification results from
    either lookup table or web search + LLM.
    """

    media_type: MediaType = Field(
        description="The type of media outlet"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    source: MediaTypeSource = Field(
        default=MediaTypeSource.FALLBACK,
        description="Source of the classification (Lookup, LLM, or Fallback)"
    )
    source_snippet: Optional[str] = Field(
        default=None,
        description="The relevant snippet from search results (LLM method only)"
    )
    reasoning: str = Field(
        description="Brief explanation of how the media type was determined"
    )


# =============================================================================
# Traffic and Longevity Schemas
# =============================================================================


class TrafficTier(str, Enum):
    """Traffic level classification for websites."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"
    UNKNOWN = "Unknown"


class TrafficEstimate(BaseModel):
    """
    Structured output for traffic level estimation from search snippets.

    Used by TrafficLongevityAnalyzer to parse traffic information
    from DuckDuckGo search results.
    """

    traffic_tier: TrafficTier = Field(
        description="Estimated traffic tier based on search results"
    )
    monthly_visits_estimate: Optional[str] = Field(
        default=None,
        description="Estimated monthly visits if mentioned (e.g., '10M', '500K')"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Explanation of how traffic tier was determined from the snippet"
    )


class TrafficSource(str, Enum):
    """Source of traffic data."""

    TRANCO = "Tranco"  # Deterministic ranking from Tranco list
    LLM = "LLM"  # LLM-parsed from search results
    FALLBACK = "Fallback"  # Default when no data available


class TrafficData(BaseModel):
    """
    Complete traffic and longevity data for a domain.

    Combines deterministic data from multiple sources:
    - WHOIS for domain age
    - Tranco list for deterministic traffic ranking (when available)
    - LLM-parsed search results as fallback
    """

    domain: str = Field(
        description="The domain being analyzed"
    )
    creation_date: Optional[date] = Field(
        default=None,
        description="Domain creation date from WHOIS lookup"
    )
    age_years: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Age of the domain in years"
    )
    traffic_tier: TrafficTier = Field(
        description="Estimated traffic tier"
    )
    monthly_visits_estimate: Optional[str] = Field(
        default=None,
        description="Estimated monthly visits if available"
    )
    traffic_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in traffic tier estimate"
    )
    traffic_source: TrafficSource = Field(
        default=TrafficSource.FALLBACK,
        description="Source of the traffic data (Tranco, LLM, or Fallback)"
    )
    tranco_rank: Optional[int] = Field(
        default=None,
        description="Tranco list rank if found (1 = most popular)"
    )
    whois_success: bool = Field(
        description="Whether WHOIS lookup was successful"
    )
    whois_error: Optional[str] = Field(
        default=None,
        description="WHOIS error message if lookup failed"
    )
    traffic_search_snippet: Optional[str] = Field(
        default=None,
        description="The search snippet used for traffic estimation (LLM method only)"
    )


# =============================================================================
# Validation Dataset Schemas
# =============================================================================


class GoldenDatasetEntry(BaseModel):
    """
    Schema for entries in the opinion classification validation dataset.
    """

    url: str = Field(
        description="URL of the article"
    )
    title: str = Field(
        description="Title/headline of the article"
    )
    text_snippet: str = Field(
        description="First ~1000 characters of article text"
    )
    expected_label: ArticleType = Field(
        description="The expected/ground truth classification"
    )


class ValidationResult(BaseModel):
    """
    Result of validating a single article classification.
    """

    url: str = Field(
        description="URL of the article tested"
    )
    expected: ArticleType = Field(
        description="Expected classification"
    )
    predicted: ArticleType = Field(
        description="Predicted classification from analyzer"
    )
    confidence: float = Field(
        description="Confidence of the prediction"
    )
    is_correct: bool = Field(
        description="Whether prediction matched expected"
    )
    reasoning: str = Field(
        description="Model's reasoning for the classification"
    )


class ValidationReport(BaseModel):
    """
    Complete validation report for the golden dataset.
    """

    total_samples: int = Field(
        description="Total number of samples tested"
    )
    correct_count: int = Field(
        description="Number of correct predictions"
    )
    accuracy: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall accuracy (correct/total)"
    )
    results: list[ValidationResult] = Field(
        description="Individual results for each sample"
    )
    mismatches: list[ValidationResult] = Field(
        description="Only the incorrect predictions"
    )


# =============================================================================
# Fact Check Schemas
# =============================================================================


class FactCheckVerdict(str, Enum):
    """Verdict from a fact-checker."""

    TRUE = "True"
    MOSTLY_TRUE = "Mostly True"
    HALF_TRUE = "Half True"
    MIXED = "Mixed"
    MOSTLY_FALSE = "Mostly False"
    FALSE = "False"
    PANTS_ON_FIRE = "Pants on Fire"
    UNPROVEN = "Unproven"
    MISLEADING = "Misleading"
    NOT_RATED = "Not Rated"


class FactCheckSource(str, Enum):
    """Source of fact check data."""

    SEARCH = "Search"  # From direct fact-checker site search
    FALLBACK = "Fallback"  # No data found


class FactCheckFinding(BaseModel):
    """A single fact check finding parsed from search results."""

    source_site: str = Field(
        description="The fact-checking organization (e.g., 'PolitiFact', 'Snopes')"
    )
    claim_summary: str = Field(
        description="Brief summary of the claim that was fact-checked"
    )
    verdict: FactCheckVerdict = Field(
        description="The verdict given by the fact-checker"
    )
    url: Optional[str] = Field(
        default=None,
        description="URL to the fact-check article if available"
    )


class FactCheckLLMOutput(BaseModel):
    """Structured LLM output for parsing fact-check search results."""

    findings: list[FactCheckFinding] = Field(
        default_factory=list,
        description="List of fact check findings extracted from search results"
    )
    failed_count: int = Field(
        ge=0,
        description="Number of FALSE/MOSTLY_FALSE/PANTS_ON_FIRE/MISLEADING verdicts"
    )
    total_count: int = Field(
        ge=0,
        description="Total number of fact checks found"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the parsing accuracy"
    )
    reasoning: str = Field(
        description="Explanation of findings and any ambiguities"
    )


class FactCheckAnalysisResult(BaseModel):
    """Complete fact check analysis result."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name if known"
    )
    failed_checks_count: int = Field(
        ge=0,
        description="Number of failed fact checks found"
    )
    total_checks_count: int = Field(
        ge=0,
        description="Total number of fact checks found"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=excellent, 10=very poor)"
    )
    source: FactCheckSource = Field(
        default=FactCheckSource.FALLBACK,
        description="Source of the fact check data"
    )
    findings: list[FactCheckFinding] = Field(
        default_factory=list,
        description="Individual fact check findings"
    )
    search_snippets: Optional[str] = Field(
        default=None,
        description="Combined search snippets used for analysis"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in the analysis"
    )
    reasoning: str = Field(
        description="Explanation of the fact check analysis"
    )


# =============================================================================
# Sourcing Quality Schemas
# =============================================================================


class SourceQuality(str, Enum):
    """Quality tier of a source."""

    PRIMARY = "Primary"  # Original documents, official statements, studies
    WIRE_SERVICE = "Wire Service"  # Reuters, AP, AFP - highly credible
    MAJOR_OUTLET = "Major Outlet"  # NYT, BBC, WSJ - established outlets
    CREDIBLE = "Credible"  # Other established outlets with standards
    UNKNOWN = "Unknown"  # Cannot assess or unfamiliar
    QUESTIONABLE = "Questionable"  # Known unreliable sources


class SourceAssessment(BaseModel):
    """Assessment of a single source."""

    domain: str = Field(
        description="The domain of the source"
    )
    quality: SourceQuality = Field(
        description="Quality tier of this source"
    )
    reasoning: str = Field(
        description="Brief explanation of the quality assessment"
    )


class SourcingLLMOutput(BaseModel):
    """Structured LLM output for sourcing analysis."""

    sources_assessed: list[SourceAssessment] = Field(
        default_factory=list,
        description="Assessment of each unique source domain"
    )
    overall_quality_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Overall sourcing quality (0=excellent, 10=poor)"
    )
    has_primary_sources: bool = Field(
        description="Whether primary sources (official docs, studies) are cited"
    )
    has_wire_services: bool = Field(
        description="Whether wire services (Reuters, AP) are cited"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    overall_assessment: str = Field(
        description="Summary assessment of sourcing practices"
    )


class SourcingAnalysisResult(BaseModel):
    """Complete sourcing analysis result."""

    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=excellent, 10=poor)"
    )
    avg_sources_per_article: float = Field(
        ge=0.0,
        description="Average number of sources cited per article"
    )
    total_sources_found: int = Field(
        ge=0,
        description="Total number of source links found"
    )
    unique_domains: int = Field(
        ge=0,
        description="Number of unique source domains"
    )
    has_hyperlinks: bool = Field(
        description="Whether articles contain hyperlinks to sources"
    )
    source_assessments: list[SourceAssessment] = Field(
        default_factory=list,
        description="Individual source quality assessments"
    )
    has_primary_sources: bool = Field(
        default=False,
        description="Whether primary sources are cited"
    )
    has_wire_services: bool = Field(
        default=False,
        description="Whether wire services are cited"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the analysis"
    )
    reasoning: str = Field(
        description="Explanation of the sourcing analysis"
    )
