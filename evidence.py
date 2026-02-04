"""
Evidence Module - Structured Evidence Collection for MBFC Reports

This module defines dataclasses for collecting and organizing evidence
from analyzers to support comprehensive MBFC-style prose reports.

Evidence flows through:
1. Analyzers collect ArticleEvidence during analysis
2. Research node gathers ExternalEvidence via web search
3. ReportWriter synthesizes all evidence into prose report
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime


# =============================================================================
# ARTICLE EVIDENCE (From Scraped Content)
# =============================================================================

@dataclass
class ArticleEvidence:
    """Evidence extracted from a specific article."""
    headline: str
    url: str
    quote: str                    # Specific text supporting the finding
    finding_type: str             # "economic_stance", "social_stance", "loaded_language", etc.
    direction: str                # "left", "right", "neutral"
    confidence: float = 0.0       # 0.0 to 1.0
    context: str = ""             # Additional context around the quote


@dataclass
class HeadlineEvidence:
    """Evidence from headline analysis (clickbait, framing)."""
    headline: str
    url: str
    finding_type: str             # "clickbait", "neutral", "sensational"
    patterns_matched: List[str]   # e.g., ["5w1h_start", "emotional_trigger"]
    severity: float               # 0.0 to 1.0


@dataclass
class LoadedLanguageEvidence:
    """Evidence of loaded/biased language in articles."""
    word_or_phrase: str
    category: str                 # "left_loaded", "right_loaded", "emotional", etc.
    article_headline: str
    article_url: str
    context: str                  # Surrounding text
    severity: float


@dataclass
class FactCheckEvidence:
    """Evidence of failed fact checks."""
    claim: str
    verdict: str                  # "False", "Mostly False", "Misleading", etc.
    source: str                   # Fact-checker name (PolitiFact, Snopes, etc.)
    source_url: str
    date: str


@dataclass
class PropagandaEvidence:
    """Evidence of propaganda techniques detected."""
    text_snippet: str
    technique: str                # One of 14 SemEval categories
    article_headline: str
    article_url: str
    confidence: float
    context: str


@dataclass
class SourcingEvidence:
    """Evidence about article sourcing quality."""
    article_headline: str
    article_url: str
    source_count: int
    credible_sources: List[str]   # URLs of credible sources cited
    non_credible_sources: List[str]
    has_hyperlinks: bool


# =============================================================================
# ANALYZER OUTPUT (Enhanced with Evidence)
# =============================================================================

@dataclass
class AnalyzerOutput:
    """
    Enhanced analyzer output that includes both score and supporting evidence.

    This replaces the simple dict returns from analyzers with structured data
    that can be used to generate prose reports.
    """
    analyzer_name: str
    score: float
    label: str

    # Evidence collections
    article_evidence: List[ArticleEvidence] = field(default_factory=list)
    headline_evidence: List[HeadlineEvidence] = field(default_factory=list)
    loaded_language_evidence: List[LoadedLanguageEvidence] = field(default_factory=list)
    fact_check_evidence: List[FactCheckEvidence] = field(default_factory=list)
    propaganda_evidence: List[PropagandaEvidence] = field(default_factory=list)
    sourcing_evidence: List[SourcingEvidence] = field(default_factory=list)

    # Methodology and details
    methodology_notes: str = ""
    raw_details: Dict[str, Any] = field(default_factory=dict)

    def get_top_evidence(self, n: int = 5) -> List[ArticleEvidence]:
        """Get top N evidence items by confidence."""
        sorted_evidence = sorted(
            self.article_evidence,
            key=lambda x: x.confidence,
            reverse=True
        )
        return sorted_evidence[:n]

    def get_evidence_summary(self) -> Dict[str, int]:
        """Get counts of evidence by type."""
        return {
            "article_evidence": len(self.article_evidence),
            "headline_evidence": len(self.headline_evidence),
            "loaded_language": len(self.loaded_language_evidence),
            "fact_checks": len(self.fact_check_evidence),
            "propaganda": len(self.propaganda_evidence),
            "sourcing": len(self.sourcing_evidence)
        }


# =============================================================================
# EXTERNAL EVIDENCE (From Web Research)
# =============================================================================

@dataclass
class ExternalEvidence:
    """Evidence gathered from external web sources."""
    source_name: str              # e.g., "Wikipedia", "Columbia Journalism Review"
    source_url: str
    finding: str                  # The relevant information found
    finding_type: str             # "history", "ownership", "criticism", "praise"
    date_accessed: str = ""
    reliability: str = "unknown"  # "high", "medium", "low", "unknown"


@dataclass
class HistoryInfo:
    """Information about the media outlet's history."""
    founding_year: Optional[int] = None
    founder: Optional[str] = None
    original_name: Optional[str] = None
    key_events: List[str] = field(default_factory=list)
    sources: List[ExternalEvidence] = field(default_factory=list)


@dataclass
class OwnershipInfo:
    """Information about ownership and funding."""
    owner: Optional[str] = None
    parent_company: Optional[str] = None
    funding_model: Optional[str] = None  # "advertising", "subscription", "public", etc.
    headquarters: Optional[str] = None
    transparency_notes: str = ""
    sources: List[ExternalEvidence] = field(default_factory=list)


@dataclass
class ExternalAnalysis:
    """External analysis/criticism of the outlet."""
    source_name: str
    source_url: str
    summary: str
    sentiment: str                # "positive", "negative", "neutral", "mixed"
    date: str = ""


# =============================================================================
# RESEARCH RESULTS
# =============================================================================

@dataclass
class ResearchResults:
    """Complete results from the research phase."""
    outlet_name: str
    outlet_url: str

    # Structured research findings
    history: HistoryInfo = field(default_factory=HistoryInfo)
    ownership: OwnershipInfo = field(default_factory=OwnershipInfo)
    external_analyses: List[ExternalAnalysis] = field(default_factory=list)

    # Raw external evidence
    all_evidence: List[ExternalEvidence] = field(default_factory=list)

    # Metadata
    research_date: str = ""
    search_queries_used: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.research_date:
            self.research_date = datetime.now().strftime("%Y-%m-%d")


# =============================================================================
# COMPREHENSIVE REPORT DATA
# =============================================================================

@dataclass
class ComprehensiveReportData:
    """
    All data needed to generate a comprehensive MBFC-style report.

    This aggregates:
    - Scores from all analyzers
    - Evidence from all analyzers
    - External research results
    - Site metadata
    """
    # Target info
    target_url: str
    target_domain: str
    country_code: str
    country_name: str

    # Scores
    bias_score: float
    bias_label: str
    factuality_score: float
    factuality_label: str
    credibility_score: float
    credibility_label: str

    # Component scores
    economic_score: float
    economic_label: str
    social_score: float
    social_label: str
    news_reporting_score: float
    news_reporting_label: str
    editorial_score: float
    editorial_label: str

    fact_check_score: float
    sourcing_score: float
    transparency_score: float
    propaganda_score: float

    # Supporting data
    media_type: str
    traffic_label: str
    freedom_rating: str
    freedom_score: float

    # Evidence from analyzers
    economic_evidence: AnalyzerOutput = None
    social_evidence: AnalyzerOutput = None
    news_reporting_evidence: AnalyzerOutput = None
    editorial_evidence: AnalyzerOutput = None
    fact_check_evidence: AnalyzerOutput = None
    sourcing_evidence: AnalyzerOutput = None
    transparency_evidence: AnalyzerOutput = None
    propaganda_evidence: AnalyzerOutput = None

    # External research
    research: ResearchResults = None

    # Metadata
    analysis_date: str = ""
    articles_analyzed: int = 0
    news_articles_count: int = 0
    opinion_articles_count: int = 0

    def __post_init__(self):
        if not self.analysis_date:
            self.analysis_date = datetime.now().strftime("%m/%d/%Y")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_article_evidence(
    headline: str,
    url: str,
    quote: str,
    finding_type: str,
    direction: str = "neutral",
    confidence: float = 0.5,
    context: str = ""
) -> ArticleEvidence:
    """Factory function to create ArticleEvidence."""
    return ArticleEvidence(
        headline=headline,
        url=url,
        quote=quote,
        finding_type=finding_type,
        direction=direction,
        confidence=confidence,
        context=context
    )


def create_analyzer_output(
    analyzer_name: str,
    score: float,
    label: str,
    methodology: str = "",
    **kwargs
) -> AnalyzerOutput:
    """Factory function to create AnalyzerOutput."""
    return AnalyzerOutput(
        analyzer_name=analyzer_name,
        score=score,
        label=label,
        methodology_notes=methodology,
        raw_details=kwargs
    )
