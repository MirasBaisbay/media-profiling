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
        description="Assessment of each unique source domain OR named entity found in text"
    )
    
    # --- NEW FIELDS START ---
    vague_sourcing_detected: bool = Field(
        default=False,
        description="Whether articles rely on vague phrases like 'experts say', 'sources claim' without naming them"
    )
    vague_sourcing_examples: list[str] = Field(
        default_factory=list,
        description="Examples of vague attribution phrases found"
    )
    # --- NEW FIELDS END ---
    
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


# =============================================================================
# Editorial Bias Schemas
# =============================================================================


class BiasDirection(str, Enum):
    """Political bias direction on the left-right spectrum."""

    EXTREME_LEFT = "Extreme Left"
    LEFT = "Left"
    LEFT_CENTER = "Left-Center"
    CENTER = "Center"
    RIGHT_CENTER = "Right-Center"
    RIGHT = "Right"
    EXTREME_RIGHT = "Extreme Right"


class PolicyDomain(str, Enum):
    """Major policy domains for bias assessment."""

    ECONOMIC = "Economic Policy"
    SOCIAL = "Social Issues"
    ENVIRONMENTAL = "Environmental Policy"
    HEALTHCARE = "Healthcare"
    IMMIGRATION = "Immigration"
    FOREIGN_POLICY = "Foreign Policy"
    GUN_RIGHTS = "Gun Rights"
    EDUCATION = "Education"


class PolicyPosition(BaseModel):
    """Assessment of outlet's position on a specific policy domain."""

    domain: PolicyDomain = Field(
        description="The policy domain being assessed"
    )
    leaning: BiasDirection = Field(
        description="The detected leaning on this policy"
    )
    indicators: list[str] = Field(
        default_factory=list,
        description="Specific indicators or quotes showing this position"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in this assessment"
    )


class EditorialBiasLLMOutput(BaseModel):
    """Structured LLM output for editorial bias analysis."""

    overall_bias: BiasDirection = Field(
        description="Overall editorial bias direction"
    )
    bias_score: float = Field(
        ge=-10.0,
        le=10.0,
        description="Numeric score: -10 (far left) to +10 (far right), 0 = center"
    )
    policy_positions: list[PolicyPosition] = Field(
        default_factory=list,
        description="Positions on specific policy domains if detectable"
    )
    uses_loaded_language: bool = Field(
        description="Whether outlet uses politically loaded language"
    )
    loaded_language_examples: list[str] = Field(
        default_factory=list,
        description="Examples of loaded language found"
    )
    story_selection_bias: Optional[str] = Field(
        default=None,
        description="Notes on biased story selection patterns if detected"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall confidence in bias assessment"
    )
    reasoning: str = Field(
        description="Detailed explanation of bias assessment"
    )


class EditorialBiasResult(BaseModel):
    """Complete editorial bias analysis result."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name"
    )
    overall_bias: BiasDirection = Field(
        description="Overall editorial bias direction"
    )
    bias_score: float = Field(
        ge=-10.0,
        le=10.0,
        description="Numeric score: -10 (far left) to +10 (far right)"
    )
    mbfc_label: str = Field(
        description="MBFC-style label (Left, Left-Center, Center, etc.)"
    )
    policy_positions: list[PolicyPosition] = Field(
        default_factory=list,
        description="Positions on specific policy domains"
    )
    uses_loaded_language: bool = Field(
        default=False,
        description="Whether outlet uses loaded language"
    )
    loaded_language_examples: list[str] = Field(
        default_factory=list,
        description="Examples of loaded language"
    )
    story_selection_bias: Optional[str] = Field(
        default=None,
        description="Notes on story selection bias"
    )
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of bias assessment"
    )


# =============================================================================
# Pseudoscience Detection Schemas
# =============================================================================


class PseudoscienceCategory(str, Enum):
    """Categories of pseudoscientific content."""

    # Health-related pseudoscience
    ANTI_VACCINATION = "Anti-Vaccination"
    ALTERNATIVE_MEDICINE = "Alternative Medicine"
    CANCER_CURE_CLAIMS = "Alternative Cancer Treatments"
    AIDS_DENIALISM = "AIDS Denialism"
    COVID_MISINFORMATION = "COVID-19 Misinformation"
    MASK_MISINFORMATION = "Mask Misinformation"
    HOMEOPATHY = "Homeopathy"
    DETOX_CLAIMS = "Detoxification Claims"
    ESSENTIAL_OILS_CURE = "Essential Oils Cure Claims"

    # Climate/Environment
    CLIMATE_DENIAL = "Climate Change Denialism"
    GMO_DANGERS = "GMO Danger Claims"
    CHEMTRAILS = "Chemtrails Conspiracy"
    FIVE_G_HEALTH = "5G Health Conspiracy"

    # Paranormal/Supernatural
    ASTROLOGY = "Astrology"
    PSYCHIC_CLAIMS = "Psychic Claims"
    ANCIENT_ASTRONAUTS = "Ancient Astronauts"
    CRYSTAL_HEALING = "Crystal Healing"
    FAITH_HEALING = "Faith Healing"

    # Conspiracy theories
    FLAT_EARTH = "Flat Earth"
    MOON_LANDING_HOAX = "Moon Landing Conspiracy"
    DEEP_STATE = "Deep State Conspiracy"
    NEW_WORLD_ORDER = "New World Order"
    QAnon = "QAnon"

    # Other
    PSEUDOARCHAEOLOGY = "Pseudoarchaeology"
    CRYPTOZOOLOGY = "Cryptozoology"
    NUMEROLOGY = "Numerology"
    OTHER = "Other Pseudoscience"


class PseudoscienceSeverity(str, Enum):
    """Severity of pseudoscience promotion."""

    PROMOTES = "Promotes"  # Actively promotes pseudoscience as fact
    PRESENTS_UNCRITICALLY = "Presents Uncritically"  # Reports without debunking
    MIXED = "Mixed"  # Sometimes promotes, sometimes critical
    NONE_DETECTED = "None Detected"  # No pseudoscience found


class PseudoscienceIndicator(BaseModel):
    """A single instance of pseudoscience content detected."""

    category: PseudoscienceCategory = Field(
        description="Category of pseudoscience detected"
    )
    severity: PseudoscienceSeverity = Field(
        description="How the outlet treats this pseudoscience"
    )
    evidence: str = Field(
        description="Quote or description of the pseudoscience content"
    )
    scientific_consensus: str = Field(
        description="Brief statement of actual scientific consensus on this topic"
    )


class PseudoscienceLLMOutput(BaseModel):
    """Structured LLM output for pseudoscience detection."""

    indicators: list[PseudoscienceIndicator] = Field(
        default_factory=list,
        description="Pseudoscience indicators found in content"
    )
    promotes_pseudoscience: bool = Field(
        description="Whether the outlet actively promotes pseudoscience"
    )
    overall_severity: PseudoscienceSeverity = Field(
        description="Overall severity of pseudoscience content"
    )
    science_reporting_quality: float = Field(
        ge=0.0,
        le=10.0,
        description="Quality of science reporting (0=excellent, 10=promotes pseudoscience)"
    )
    respects_scientific_consensus: bool = Field(
        description="Whether outlet generally respects scientific consensus"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of pseudoscience assessment"
    )


class PseudoscienceAnalysisResult(BaseModel):
    """Complete pseudoscience analysis result."""

    domain: str = Field(
        description="The domain that was analyzed"
    )
    outlet_name: Optional[str] = Field(
        default=None,
        description="Human-readable outlet name"
    )
    score: float = Field(
        ge=0.0,
        le=10.0,
        description="MBFC-style score (0=pro-science, 10=promotes pseudoscience)"
    )
    promotes_pseudoscience: bool = Field(
        description="Whether outlet promotes pseudoscience"
    )
    overall_severity: PseudoscienceSeverity = Field(
        description="Overall severity classification"
    )
    categories_found: list[PseudoscienceCategory] = Field(
        default_factory=list,
        description="Categories of pseudoscience found"
    )
    indicators: list[PseudoscienceIndicator] = Field(
        default_factory=list,
        description="Detailed pseudoscience indicators"
    )
    respects_scientific_consensus: bool = Field(
        default=True,
        description="Whether outlet respects scientific consensus"
    )
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the assessment"
    )
    reasoning: str = Field(
        description="Explanation of pseudoscience assessment"
    )


# =============================================================================
# Research Module Schemas
# =============================================================================


class HistoryLLMOutput(BaseModel):
    """Structured LLM output for outlet history extraction."""

    official_name: Optional[str] = Field(
        default=None,
        description="The proper, official name of the organization (e.g., 'The Associated Press' instead of 'apnews', 'Wall Street Journal' instead of 'wsj')"
    )
    founding_year: Optional[int] = Field(
        default=None,
        description="Year the outlet was founded"
    )
    founder: Optional[str] = Field(
        default=None,
        description="Name of founder(s)"
    )
    original_name: Optional[str] = Field(
        default=None,
        description="Original name if different from current"
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Key events in the outlet's history"
    )
    summary: str = Field(
        description="2-3 sentence history summary"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


class OwnershipLLMOutput(BaseModel):
    """Structured LLM output for ownership/funding extraction."""

    owner: Optional[str] = Field(
        default=None,
        description="Current owner name"
    )
    parent_company: Optional[str] = Field(
        default=None,
        description="Parent company if applicable"
    )
    funding_model: Optional[str] = Field(
        default=None,
        description="Funding model: advertising, subscription, public, nonprofit, mixed"
    )
    headquarters: Optional[str] = Field(
        default=None,
        description="Headquarters location (city, country)"
    )
    notes: str = Field(
        default="",
        description="Additional notes about ownership/funding"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


class ExternalAnalysisItem(BaseModel):
    """A single external analysis/criticism of an outlet."""

    source_name: str = Field(
        description="Name of the source (e.g., 'Columbia Journalism Review')"
    )
    source_url: Optional[str] = Field(
        default=None,
        description="URL of the source if available"
    )
    summary: str = Field(
        description="Brief summary of the analysis/criticism"
    )
    sentiment: str = Field(
        description="Sentiment: positive, negative, neutral, or mixed"
    )


class ExternalAnalysisLLMOutput(BaseModel):
    """Structured LLM output for external analysis extraction."""

    analyses: list[ExternalAnalysisItem] = Field(
        default_factory=list,
        description="List of external analyses found"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in extracted information"
    )


class ComprehensiveReportData(BaseModel):
    """Complete data for generating an MBFC-style report."""

    # Target info
    target_url: str = Field(description="URL of the outlet")
    target_domain: str = Field(description="Domain name")
    outlet_name: str = Field(description="Human-readable outlet name")

    # Overall ratings
    bias_label: str = Field(description="MBFC-style bias label")
    bias_score: float = Field(
        ge=-10.0, le=10.0,
        description="Bias score (-10=far left, +10=far right)"
    )
    factuality_label: str = Field(description="Factuality rating label")
    factuality_score: float = Field(
        ge=0.0, le=10.0,
        description="Factuality score (0=excellent, 10=very poor)"
    )
    credibility_label: str = Field(description="Overall credibility label")
    credibility_score: float = Field(
        ge=0.0, le=10.0,
        description="Overall credibility score (0=excellent, 10=not credible)"
    )

    # Traffic and metadata
    media_type: str = Field(description="Type of media outlet")
    traffic_tier: str = Field(description="Traffic tier (HIGH/MEDIUM/LOW/MINIMAL)")
    domain_age_years: Optional[float] = Field(
        default=None,
        description="Age of domain in years"
    )

    # Component analysis results
    editorial_bias_result: Optional["EditorialBiasResult"] = Field(
        default=None,
        description="Editorial bias analysis result"
    )
    fact_check_result: Optional["FactCheckAnalysisResult"] = Field(
        default=None,
        description="Fact check search result"
    )
    sourcing_result: Optional["SourcingAnalysisResult"] = Field(
        default=None,
        description="Sourcing quality analysis result"
    )
    pseudoscience_result: Optional["PseudoscienceAnalysisResult"] = Field(
        default=None,
        description="Pseudoscience analysis result"
    )

    # Research results
    history_summary: Optional[str] = Field(
        default=None,
        description="Brief history of the outlet"
    )
    founding_year: Optional[int] = Field(
        default=None,
        description="Year founded"
    )
    founder: Optional[str] = Field(
        default=None,
        description="Founder(s) of the outlet"
    )
    original_name: Optional[str] = Field(
        default=None,
        description="Original name if different from current"
    )
    key_events: list[str] = Field(
        default_factory=list,
        description="Key events in the outlet's history"
    )
    owner: Optional[str] = Field(
        default=None,
        description="Owner/parent company"
    )
    parent_company: Optional[str] = Field(
        default=None,
        description="Parent company if applicable"
    )
    funding_model: Optional[str] = Field(
        default=None,
        description="Funding model"
    )
    headquarters: Optional[str] = Field(
        default=None,
        description="Headquarters location"
    )
    ownership_notes: Optional[str] = Field(
        default=None,
        description="Additional notes about ownership/funding"
    )

    # External analyses
    external_analyses: list[ExternalAnalysisItem] = Field(
        default_factory=list,
        description="External analyses from media watchdogs"
    )

    # Metadata
    analysis_date: str = Field(description="Date of analysis")
    articles_analyzed: int = Field(
        ge=0,
        description="Number of articles analyzed"
    )