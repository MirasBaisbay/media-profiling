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
