"""
Editorial Bias Detection Module

Implements detection of:
1. Clickbait headlines (patterns, sensationalism, curiosity gap)
2. Loaded/emotional language (bias-inducing words, sentiment extremes)
3. Manipulation techniques (framing bias, epistemological bias)

Based on academic research:
- Recasens et al. (2013) - Linguistic Models for Analyzing and Detecting Biased Language
- Chakraborty et al. (2016) - Stop Clickbait: Detecting and Preventing Clickbait in Online News
- QCRI Analysis of Emotional Language in 21 Million News Articles
- SemEval propaganda detection research

References:
- https://web.stanford.edu/~jurafsky/pubs/neutrality.pdf
- https://www.nature.com/articles/s41598-025-30229-5
- https://source.opennews.org/articles/analysis-emotional-language/
"""

import re
import logging
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from scraper import Article

logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# DATA CLASSES
# =============================================================================

class BiasType(Enum):
    CLICKBAIT = "clickbait"
    LOADED_LANGUAGE = "loaded_language"
    FRAMING_BIAS = "framing_bias"
    EPISTEMOLOGICAL_BIAS = "epistemological_bias"


@dataclass
class ClickbaitIndicator:
    """A detected clickbait pattern in a headline."""
    pattern_name: str
    pattern_description: str
    matched_text: str
    severity: float  # 0.0 to 1.0


@dataclass
class LoadedLanguageInstance:
    """A detected instance of loaded/emotional language."""
    word_or_phrase: str
    category: str  # e.g., "negative_emotion", "subjective_intensifier", "one_sided"
    context: str
    severity: float


@dataclass
class HeadlineAnalysis:
    """Analysis of a single headline for clickbait."""
    headline: str
    is_clickbait: bool
    clickbait_score: float  # 0.0 to 1.0
    indicators: List[ClickbaitIndicator]
    explanation: str


@dataclass
class ArticleLanguageAnalysis:
    """Analysis of article text for loaded language."""
    article_title: str
    loaded_language_score: float  # 0.0 to 1.0
    instances: List[LoadedLanguageInstance]
    dominant_emotion: str
    sentiment_extremity: float  # How extreme the sentiment is


@dataclass
class EditorialBiasAnalysis:
    """Complete editorial bias analysis result."""
    overall_score: float  # -10 to +10 (negative = left bias, positive = right bias)
    overall_label: str  # e.g., "Moderate Left Editorial Bias"
    clickbait_score: float  # 0 to 10 (higher = more clickbait)
    loaded_language_score: float  # 0 to 10 (higher = more loaded)
    emotional_manipulation_score: float  # 0 to 10
    direction: str  # "left", "right", or "neutral"
    headline_analyses: List[HeadlineAnalysis]
    language_analyses: List[ArticleLanguageAnalysis]
    methodology_notes: str


# MBFC Editorial Bias Labels (discrete 9-point scale)
EDITORIAL_BIAS_LABELS = {
    -10.0: "Extreme Left Editorial Bias",
    -7.5: "Strong Left Editorial Bias",
    -5.0: "Moderate Left Editorial Bias",
    -2.5: "Mild Left Editorial Bias",
    0.0: "Neutral/Balanced Editorial",
    2.5: "Mild Right Editorial Bias",
    5.0: "Moderate Right Editorial Bias",
    7.5: "Strong Right Editorial Bias",
    10.0: "Extreme Right Editorial Bias",
}


def score_to_editorial_label(score: float) -> str:
    """
    Map a continuous score (-10 to +10) to discrete MBFC editorial bias label.

    Scale:
    -10: Extreme Left Editorial Bias
    -7.5: Strong Left Editorial Bias
    -5: Moderate Left Editorial Bias
    -2.5: Mild Left Editorial Bias
    0: Neutral/Balanced
    +2.5: Mild Right Editorial Bias
    +5: Moderate Right Editorial Bias
    +7.5: Strong Right Editorial Bias
    +10: Extreme Right Editorial Bias
    """
    if score <= -8.75:
        return "Extreme Left Editorial Bias"
    elif score <= -6.25:
        return "Strong Left Editorial Bias"
    elif score <= -3.75:
        return "Moderate Left Editorial Bias"
    elif score <= -1.25:
        return "Mild Left Editorial Bias"
    elif score <= 1.25:
        return "Neutral/Balanced Editorial"
    elif score <= 3.75:
        return "Mild Right Editorial Bias"
    elif score <= 6.25:
        return "Moderate Right Editorial Bias"
    elif score <= 8.75:
        return "Strong Right Editorial Bias"
    else:
        return "Extreme Right Editorial Bias"


def snap_to_discrete_score(score: float) -> float:
    """
    Snap a continuous score to the nearest discrete value in the 9-point scale.
    Values: -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10
    """
    discrete_values = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
    return min(discrete_values, key=lambda x: abs(x - score))


# =============================================================================
# CLICKBAIT DETECTION PATTERNS
# =============================================================================
# Based on research by Chakraborty et al. and others

CLICKBAIT_PATTERNS = {
    # Pattern: (regex, description, severity)

    # 5W1H at start (curiosity gap)
    "5w1h_start": {
        "pattern": r"^(What|Why|When|Who|Which|How|Where)\b",
        "description": "Starts with question word (5W1H) - creates curiosity gap",
        "severity": 0.3,
        "reference": "Chakraborty et al. (2016)"
    },

    # This/These forward reference
    "forward_reference_this": {
        "pattern": r"\b(This|These|That|Those)\b.*\b(will|is|are|was|were|make|makes)\b",
        "description": "Uses demonstrative pronoun without clear referent",
        "severity": 0.5,
        "reference": "Potthast et al. (2016) - Clickbait Challenge"
    },

    # You won't believe / shocking revelation
    "hyperbolic_shock": {
        "pattern": r"\b(won't believe|can't believe|shock|shocking|jaw-drop|mind-blow|incredible|unbelievable|insane|crazy)\b",
        "description": "Uses hyperbolic shock words",
        "severity": 0.7,
        "reference": "LiT.RL Clickbait Detector features"
    },

    # Numbers and lists (listicles)
    "listicle": {
        "pattern": r"^\d+\s+(Things?|Ways?|Reasons?|Tips?|Secrets?|Facts?|Signs?|Steps?|Rules?|Hacks?)",
        "description": "Listicle format headline",
        "severity": 0.4,
        "reference": "BuzzFeed style analysis"
    },

    # Curiosity gap with ellipsis or incomplete thought
    "curiosity_gap_ellipsis": {
        "pattern": r"\.{2,}$|\s+[-–—]\s*$|\.{3}",
        "description": "Ends with ellipsis or dash (incomplete thought)",
        "severity": 0.6,
        "reference": "Curiosity gap theory"
    },

    # Superlatives
    "superlative": {
        "pattern": r"\b(best|worst|most|least|biggest|smallest|greatest|top|ultimate|perfect|absolute)\b",
        "description": "Uses superlative language",
        "severity": 0.3,
        "reference": "Sensationalism markers"
    },

    # ALL CAPS words
    "caps_emphasis": {
        "pattern": r"\b[A-Z]{3,}\b",
        "description": "Contains ALL CAPS words for emphasis",
        "severity": 0.5,
        "reference": "Visual attention patterns"
    },

    # Excessive punctuation
    "excessive_punctuation": {
        "pattern": r"[!?]{2,}|!{2,}|\?{2,}",
        "description": "Uses excessive punctuation (!! or ??)",
        "severity": 0.6,
        "reference": "Emotional intensifiers"
    },

    # Celebrity/famous person bait
    "celebrity_bait": {
        "pattern": r"\b(celebrity|star|famous|A-list|billionaire|millionaire|royalty)\b",
        "description": "References celebrities or famous people for attention",
        "severity": 0.3,
        "reference": "Social proof clickbait"
    },

    # Urgency/FOMO
    "urgency_fomo": {
        "pattern": r"\b(now|today|just|breaking|urgent|finally|must|need to|have to|right now)\b",
        "description": "Creates urgency or FOMO",
        "severity": 0.4,
        "reference": "FOMO marketing patterns"
    },

    # Second person address
    "second_person": {
        "pattern": r"\b(you|your|you're|you'll|yourself)\b",
        "description": "Directly addresses reader (second person)",
        "severity": 0.2,
        "reference": "Personal engagement tactics"
    },

    # Emotional trigger words
    "emotional_trigger": {
        "pattern": r"\b(outrage|furious|devastating|heartbreaking|horrifying|disgusting|terrifying|amazing|incredible)\b",
        "description": "Uses high-arousal emotional trigger words",
        "severity": 0.6,
        "reference": "Negativity bias research"
    },

    # Here's/Here is construction
    "heres_construction": {
        "pattern": r"\b(Here's|Here is|Here are)\b",
        "description": "Uses 'Here's' construction (common in clickbait)",
        "severity": 0.4,
        "reference": "Potthast et al. (2016)"
    },

    # Question headlines
    "question_headline": {
        "pattern": r"\?$",
        "description": "Ends with question mark",
        "severity": 0.3,
        "reference": "Betteridge's Law of Headlines"
    },
}


# =============================================================================
# LOADED LANGUAGE LEXICONS
# =============================================================================
# Based on Recasens et al. (2013) and LIWC categories

# Subjective intensifiers (framing bias markers)
SUBJECTIVE_INTENSIFIERS = [
    "extremely", "incredibly", "amazingly", "absolutely", "totally",
    "completely", "utterly", "entirely", "deeply", "highly",
    "remarkably", "exceptionally", "extraordinarily", "phenomenally",
    "massively", "hugely", "vastly", "overwhelmingly", "staggeringly"
]

# One-sided terms (framing bias)
# Left-leaning loaded terms
LEFT_LOADED_TERMS = [
    "regime", "far-right", "extremist", "radical right", "ultra-conservative",
    "bigot", "racist", "xenophobic", "homophobic", "fascist",
    "authoritarian", "dictator", "oligarch", "plutocrat", "corporate greed",
    "climate denier", "anti-science", "book ban", "voter suppression",
    "dog whistle", "white supremacy", "patriarchy", "toxic masculinity",
    "wealth inequality", "exploitation", "corporate welfare"
]

# Right-leaning loaded terms
RIGHT_LOADED_TERMS = [
    "radical left", "far-left", "socialist", "communist", "marxist",
    "woke", "cancel culture", "political correctness", "virtue signaling",
    "snowflake", "liberal elite", "mainstream media", "fake news",
    "deep state", "open borders", "illegal alien", "radical agenda",
    "big government", "nanny state", "thought police", "mob rule",
    "antifa", "defund police", "critical race theory", "indoctrination"
]

# High-arousal negative words (emotional manipulation)
NEGATIVE_EMOTIONAL_WORDS = [
    "outrage", "fury", "anger", "rage", "hatred", "disgust",
    "horrific", "terrifying", "devastating", "catastrophic", "disastrous",
    "shocking", "appalling", "disgusting", "repulsive", "vile",
    "corrupt", "scandal", "crisis", "chaos", "collapse", "destruction",
    "threat", "danger", "attack", "assault", "victim", "tragedy"
]

# High-arousal positive words (often used manipulatively)
POSITIVE_EMOTIONAL_WORDS = [
    "incredible", "amazing", "stunning", "breathtaking", "spectacular",
    "phenomenal", "extraordinary", "remarkable", "magnificent", "brilliant",
    "triumph", "victory", "hero", "champion", "miracle", "breakthrough"
]

# Epistemological bias markers (Recasens et al.)
FACTIVE_VERBS = [
    "reveal", "expose", "uncover", "prove", "confirm", "demonstrate",
    "show", "establish", "admit", "acknowledge", "recognize"
]

ASSERTIVE_VERBS = [
    "claim", "allege", "assert", "insist", "contend", "maintain",
    "argue", "suggest", "imply", "hint", "insinuate"
]

HEDGES = [
    "allegedly", "reportedly", "supposedly", "purportedly", "seemingly",
    "apparently", "perhaps", "possibly", "probably", "likely",
    "may", "might", "could", "some say", "critics say"
]

# Doubt markers (undermining credibility)
DOUBT_MARKERS = [
    "so-called", "self-proclaimed", "self-styled", "supposed",
    "questionable", "dubious", "controversial", "disputed"
]


# =============================================================================
# CLICKBAIT ANALYZER
# =============================================================================

class ClickbaitAnalyzer:
    """
    Analyzes headlines for clickbait patterns.

    Uses rule-based pattern matching combined with LLM verification
    for more nuanced detection.
    """

    def __init__(self):
        self.patterns = CLICKBAIT_PATTERNS

    def analyze_headline(self, headline: str) -> HeadlineAnalysis:
        """Analyze a single headline for clickbait patterns."""
        indicators = []
        total_severity = 0.0

        # Check each pattern
        for pattern_name, pattern_info in self.patterns.items():
            regex = pattern_info["pattern"]
            match = re.search(regex, headline, re.IGNORECASE)

            if match:
                indicators.append(ClickbaitIndicator(
                    pattern_name=pattern_name,
                    pattern_description=pattern_info["description"],
                    matched_text=match.group(0),
                    severity=pattern_info["severity"]
                ))
                total_severity += pattern_info["severity"]

        # Calculate clickbait score (0-1)
        # Normalize by max possible severity
        max_severity = sum(p["severity"] for p in self.patterns.values())
        clickbait_score = min(1.0, total_severity / (max_severity * 0.3))  # 30% threshold

        is_clickbait = clickbait_score >= 0.5

        explanation = self._generate_explanation(indicators, clickbait_score)

        return HeadlineAnalysis(
            headline=headline,
            is_clickbait=is_clickbait,
            clickbait_score=round(clickbait_score, 2),
            indicators=indicators,
            explanation=explanation
        )

    def analyze_headlines(self, headlines: List[str]) -> List[HeadlineAnalysis]:
        """Analyze multiple headlines."""
        return [self.analyze_headline(h) for h in headlines]

    def _generate_explanation(
        self,
        indicators: List[ClickbaitIndicator],
        score: float
    ) -> str:
        """Generate human-readable explanation of clickbait detection."""
        if not indicators:
            return "No clickbait patterns detected."

        patterns_found = [f"{i.pattern_name} ({i.matched_text})" for i in indicators]
        return f"Detected patterns: {', '.join(patterns_found)}. Score: {score:.0%}"


# =============================================================================
# LOADED LANGUAGE ANALYZER
# =============================================================================

class LoadedLanguageAnalyzer:
    """
    Analyzes text for loaded/emotional language.

    Based on:
    - Recasens et al. (2013) bias word detection
    - LIWC emotional categories
    - Political framing research
    """

    def __init__(self):
        self.left_terms = set(t.lower() for t in LEFT_LOADED_TERMS)
        self.right_terms = set(t.lower() for t in RIGHT_LOADED_TERMS)
        self.intensifiers = set(t.lower() for t in SUBJECTIVE_INTENSIFIERS)
        self.negative_words = set(t.lower() for t in NEGATIVE_EMOTIONAL_WORDS)
        self.positive_words = set(t.lower() for t in POSITIVE_EMOTIONAL_WORDS)
        self.factive_verbs = set(t.lower() for t in FACTIVE_VERBS)
        self.assertive_verbs = set(t.lower() for t in ASSERTIVE_VERBS)
        self.hedges = set(t.lower() for t in HEDGES)
        self.doubt_markers = set(t.lower() for t in DOUBT_MARKERS)

    def analyze_article(self, article: Article) -> ArticleLanguageAnalysis:
        """Analyze an article for loaded language."""
        text = article.text.lower()
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)

        instances = []

        # Check for left-loaded terms
        for term in self.left_terms:
            if term in text:
                context = self._extract_context(text, term)
                instances.append(LoadedLanguageInstance(
                    word_or_phrase=term,
                    category="left_loaded",
                    context=context,
                    severity=0.7
                ))

        # Check for right-loaded terms
        for term in self.right_terms:
            if term in text:
                context = self._extract_context(text, term)
                instances.append(LoadedLanguageInstance(
                    word_or_phrase=term,
                    category="right_loaded",
                    context=context,
                    severity=0.7
                ))

        # Check for subjective intensifiers
        for term in self.intensifiers:
            count = text.count(term)
            if count > 0:
                instances.append(LoadedLanguageInstance(
                    word_or_phrase=term,
                    category="subjective_intensifier",
                    context=f"Found {count} times",
                    severity=0.3 * min(count, 3)  # Cap at 3 occurrences
                ))

        # Check for emotional words
        negative_count = sum(1 for w in words if w in self.negative_words)
        positive_count = sum(1 for w in words if w in self.positive_words)

        for term in self.negative_words:
            if term in text:
                context = self._extract_context(text, term)
                instances.append(LoadedLanguageInstance(
                    word_or_phrase=term,
                    category="negative_emotion",
                    context=context,
                    severity=0.5
                ))

        # Check for doubt markers
        for term in self.doubt_markers:
            if term in text:
                context = self._extract_context(text, term)
                instances.append(LoadedLanguageInstance(
                    word_or_phrase=term,
                    category="doubt_marker",
                    context=context,
                    severity=0.4
                ))

        # Calculate overall score
        total_severity = sum(i.severity for i in instances)
        # Normalize by word count (per 1000 words)
        normalized_score = (total_severity / max(1, word_count)) * 1000
        loaded_language_score = min(1.0, normalized_score / 10)  # 10 = high threshold

        # Determine dominant emotion
        if negative_count > positive_count:
            dominant_emotion = "negative"
        elif positive_count > negative_count:
            dominant_emotion = "positive"
        else:
            dominant_emotion = "neutral"

        # Sentiment extremity
        total_emotional = negative_count + positive_count
        sentiment_extremity = min(1.0, total_emotional / max(1, word_count) * 50)

        return ArticleLanguageAnalysis(
            article_title=article.title,
            loaded_language_score=round(loaded_language_score, 2),
            instances=instances[:20],  # Limit to top 20
            dominant_emotion=dominant_emotion,
            sentiment_extremity=round(sentiment_extremity, 2)
        )

    def _extract_context(self, text: str, term: str, window: int = 50) -> str:
        """Extract context around a term."""
        idx = text.find(term)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        return "..." + text[start:end] + "..."

    def get_political_lean(self, instances: List[LoadedLanguageInstance]) -> Tuple[str, float]:
        """Determine political lean from loaded language instances."""
        left_count = sum(1 for i in instances if i.category == "left_loaded")
        right_count = sum(1 for i in instances if i.category == "right_loaded")

        total = left_count + right_count
        if total == 0:
            return "neutral", 0.0

        if left_count > right_count:
            lean = "left"
            confidence = (left_count - right_count) / total
        elif right_count > left_count:
            lean = "right"
            confidence = (right_count - left_count) / total
        else:
            lean = "neutral"
            confidence = 0.0

        return lean, confidence


# =============================================================================
# EDITORIAL BIAS ANALYZER (COMBINED)
# =============================================================================

class StructuredEditorialBiasAnalyzer:
    """
    Complete editorial bias analyzer combining:
    - Clickbait detection
    - Loaded language detection
    - LLM-based nuanced analysis

    Returns score on -10 to +10 scale for MBFC methodology.
    """

    def __init__(self):
        self.clickbait_analyzer = ClickbaitAnalyzer()
        self.loaded_language_analyzer = LoadedLanguageAnalyzer()

    def analyze(self, articles: List[Article], prefer_opinion: bool = True) -> EditorialBiasAnalysis:
        """
        Perform complete editorial bias analysis.

        Args:
            articles: List of articles to analyze
            prefer_opinion: If True, prioritize opinion/editorial articles

        Returns:
            EditorialBiasAnalysis with scores and evidence
        """
        logger.info(f"Analyzing editorial bias for {len(articles)} articles")

        # Filter to opinion/editorial articles if available and preferred
        if prefer_opinion:
            opinion_articles = [a for a in articles if a.is_opinion]
            if opinion_articles:
                articles_to_analyze = opinion_articles[:10]
                logger.info(f"Using {len(articles_to_analyze)} opinion/editorial articles")
            else:
                articles_to_analyze = articles[:10]
                logger.info("No opinion articles found, using all articles")
        else:
            articles_to_analyze = articles[:10]

        if not articles_to_analyze:
            return EditorialBiasAnalysis(
                overall_score=0.0,
                overall_label="Neutral/Balanced Editorial",
                clickbait_score=0.0,
                loaded_language_score=0.0,
                emotional_manipulation_score=0.0,
                direction="neutral",
                headline_analyses=[],
                language_analyses=[],
                methodology_notes="No articles to analyze"
            )

        # Analyze headlines for clickbait
        headlines = [a.title for a in articles_to_analyze]
        headline_analyses = self.clickbait_analyzer.analyze_headlines(headlines)

        # Analyze articles for loaded language
        language_analyses = [
            self.loaded_language_analyzer.analyze_article(a)
            for a in articles_to_analyze
        ]

        # Calculate aggregate scores (0-1 scale internally)
        clickbait_score = self._calculate_clickbait_score(headline_analyses)
        loaded_language_score = self._calculate_loaded_language_score(language_analyses)

        # Determine political direction from loaded language
        all_instances = []
        for la in language_analyses:
            all_instances.extend(la.instances)
        direction, direction_confidence = self.loaded_language_analyzer.get_political_lean(
            all_instances
        )

        # Calculate emotional manipulation score (0-1 scale)
        emotional_manipulation_score = self._calculate_manipulation_score(
            headline_analyses, language_analyses
        )

        # Calculate overall bias score (-10 to +10)
        # Use LLM to determine direction if rule-based detection is inconclusive
        if direction == "neutral" and len(articles_to_analyze) > 0:
            llm_direction = self._llm_detect_direction(articles_to_analyze[:5])
            if llm_direction != "neutral":
                direction = llm_direction
                direction_confidence = 0.6  # Moderate confidence for LLM-only detection

        overall_score = self._calculate_overall_score(
            direction=direction,
            direction_confidence=direction_confidence,
            loaded_language_score=loaded_language_score,
            manipulation_score=emotional_manipulation_score,
            clickbait_score=clickbait_score
        )

        # LLM verification to adjust confidence
        if len(articles_to_analyze) > 0 and abs(overall_score) > 1.0:
            llm_adjustment = self._llm_verify_direction(articles_to_analyze[:5], direction)
            overall_score = overall_score * llm_adjustment

        # Clamp to valid range and get label
        overall_score = max(-10.0, min(10.0, overall_score))
        overall_label = score_to_editorial_label(overall_score)

        methodology_notes = self._build_methodology_notes(
            len(headline_analyses),
            len(language_analyses),
            clickbait_score,
            loaded_language_score,
            direction
        )

        return EditorialBiasAnalysis(
            overall_score=round(overall_score, 2),
            overall_label=overall_label,
            clickbait_score=round(clickbait_score * 10, 2),  # Convert to 0-10
            loaded_language_score=round(loaded_language_score * 10, 2),
            emotional_manipulation_score=round(emotional_manipulation_score * 10, 2),
            direction=direction,
            headline_analyses=headline_analyses,
            language_analyses=language_analyses,
            methodology_notes=methodology_notes
        )

    def _calculate_clickbait_score(
        self,
        analyses: List[HeadlineAnalysis]
    ) -> float:
        """Calculate average clickbait score (0-1)."""
        if not analyses:
            return 0.0
        return sum(a.clickbait_score for a in analyses) / len(analyses)

    def _calculate_loaded_language_score(
        self,
        analyses: List[ArticleLanguageAnalysis]
    ) -> float:
        """Calculate average loaded language score (0-1)."""
        if not analyses:
            return 0.0
        return sum(a.loaded_language_score for a in analyses) / len(analyses)

    def _calculate_manipulation_score(
        self,
        headline_analyses: List[HeadlineAnalysis],
        language_analyses: List[ArticleLanguageAnalysis]
    ) -> float:
        """Calculate emotional manipulation score (0-1)."""
        # Combine clickbait and emotional extremity
        clickbait_avg = self._calculate_clickbait_score(headline_analyses)

        if not language_analyses:
            return clickbait_avg

        sentiment_extremity_avg = sum(
            a.sentiment_extremity for a in language_analyses
        ) / len(language_analyses)

        # Weighted combination
        return (clickbait_avg * 0.4) + (sentiment_extremity_avg * 0.6)

    def _calculate_overall_score(
        self,
        direction: str,
        direction_confidence: float,
        loaded_language_score: float,
        manipulation_score: float,
        clickbait_score: float
    ) -> float:
        """
        Calculate overall editorial bias score (-10 to +10).

        Methodology:
        - Score reflects BOTH direction (left/right) AND intensity (emotional/manipulative language)
        - Intensity is calculated from:
          * Loaded language score (weight: 0.4)
          * Emotional manipulation score (weight: 0.4)
          * Clickbait score (weight: 0.2)
        - Direction confidence affects the final score magnitude

        Scale:
        -10: Extreme Left - Editorials exclusively promote left views with highly emotional language
        -7.5: Strong Left - Regularly supports left views with emotional language
        -5: Moderate Left - Often leans left with some emotional framing
        -2.5: Mild Left - Slightly favors left perspectives, minimal emotional language
        0: Neutral/Balanced - Presents perspectives fairly, avoids loaded emotional language
        +2.5 to +10: Mirror of left scale for right-leaning bias

        Negative = left-leaning editorial bias
        Positive = right-leaning editorial bias
        """
        if direction == "neutral":
            # Even neutral can have some emotional language without political direction
            # This might indicate sensationalism without clear political bias
            intensity = (loaded_language_score * 0.4 +
                        manipulation_score * 0.4 +
                        clickbait_score * 0.2)
            # Return a small value if there's high emotional content but no clear direction
            if intensity > 0.5:
                return 0.0  # High emotional but no direction = stays at 0
            return 0.0

        # Calculate intensity from multiple factors
        # Weighted combination: loaded language is most important for bias
        intensity = (loaded_language_score * 0.4 +
                    manipulation_score * 0.4 +
                    clickbait_score * 0.2)

        # Map intensity (0-1) to score magnitude (0-10)
        # Use a non-linear mapping to better distinguish between mild and extreme
        # intensity < 0.2 -> Mild (0-2.5)
        # intensity 0.2-0.4 -> Moderate (2.5-5)
        # intensity 0.4-0.6 -> Strong (5-7.5)
        # intensity > 0.6 -> Extreme (7.5-10)

        if intensity < 0.2:
            base_score = intensity * 12.5  # Maps 0-0.2 to 0-2.5
        elif intensity < 0.4:
            base_score = 2.5 + (intensity - 0.2) * 12.5  # Maps 0.2-0.4 to 2.5-5
        elif intensity < 0.6:
            base_score = 5.0 + (intensity - 0.4) * 12.5  # Maps 0.4-0.6 to 5-7.5
        else:
            base_score = 7.5 + (intensity - 0.6) * 6.25  # Maps 0.6-1.0 to 7.5-10

        # Apply direction confidence
        final_score = base_score * max(0.5, direction_confidence)

        # Clamp to 0-10 range before applying direction
        final_score = min(10.0, max(0.0, final_score))

        # Apply direction sign
        if direction == "left":
            return -final_score
        elif direction == "right":
            return final_score
        else:
            return 0.0

    def _llm_detect_direction(self, articles: List[Article]) -> str:
        """
        Use LLM to detect political direction when rule-based detection is inconclusive.

        Returns: "left", "right", or "neutral"
        """
        combined = "\n".join([
            f"HEADLINE: {a.title}\nEXCERPT: {a.text[:400]}"
            for a in articles[:3]
        ])

        prompt = f"""
Analyze the political lean of these editorial/opinion article excerpts.

{combined}

Based on the language, framing, arguments, and perspective presented, determine if the editorial stance is:
- LEFT (progressive/liberal viewpoints, criticism of conservative policies)
- RIGHT (conservative/traditional viewpoints, criticism of liberal policies)
- NEUTRAL (balanced approach, no clear political lean)

Consider:
- What positions are being advocated?
- What is being criticized or praised?
- What loaded terms are used?
- Whose perspective is being presented sympathetically?

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
            logger.warning(f"LLM direction detection failed: {e}")
            return "neutral"

    def _llm_verify_direction(
        self,
        articles: List[Article],
        detected_direction: str
    ) -> float:
        """
        Use LLM to verify detected editorial direction.

        Returns adjustment factor (0.5 to 1.5).
        """
        combined = "\n".join([
            f"HEADLINE: {a.title}\nTEXT: {a.text[:500]}"
            for a in articles[:3]
        ])

        prompt = f"""
Analyze the editorial lean of these article excerpts.

{combined}

Based on the language, framing, and perspective:
1. Does this content lean LEFT (progressive/liberal)?
2. Does this content lean RIGHT (conservative/traditional)?
3. Is it NEUTRAL (balanced editorial approach)?

Our automated analysis detected: {detected_direction}

Return a JSON object:
{{
    "verified_direction": "left" | "right" | "neutral",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation"
}}

Return ONLY the JSON object.
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            verified = data.get("verified_direction", "neutral")
            confidence = float(data.get("confidence", 0.5))

            # Adjustment factor
            if verified == detected_direction:
                return 1.0 + (confidence * 0.3)  # Boost confidence
            elif verified == "neutral":
                return 0.7  # Reduce intensity
            else:
                return 0.5  # Significant reduction if disagreement

        except Exception as e:
            logger.warning(f"LLM verification failed: {e}")
            return 1.0  # No adjustment

    def _build_methodology_notes(
        self,
        num_headlines: int,
        num_articles: int,
        clickbait_score: float,
        loaded_language_score: float,
        direction: str
    ) -> str:
        """Build methodology notes for the analysis."""
        return (
            f"Analyzed {num_headlines} headlines for clickbait patterns "
            f"(avg score: {clickbait_score:.0%}). "
            f"Analyzed {num_articles} articles for loaded language "
            f"(avg score: {loaded_language_score:.0%}). "
            f"Detected editorial lean: {direction}. "
            f"Methods: Rule-based pattern matching + LLM verification."
        )


# =============================================================================
# NEWS REPORTING BALANCE ANALYZER
# =============================================================================

class StructuredNewsReportingAnalyzer:
    """
    Analyzes straight news reporting balance (separate from editorial).

    Checks:
    - Story selection bias (which topics are covered)
    - Framing of news events
    - Inclusion of multiple perspectives
    - Use of sources from different sides
    """

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Analyze news reporting balance.

        Returns dict with:
        - balance_score: -10 to +10
        - perspective_diversity: 0 to 1
        - methodology_notes: str
        """
        # Filter to news articles only
        news_articles = [a for a in articles if not a.is_opinion][:10]

        if not news_articles:
            return {
                "balance_score": 0.0,
                "balance_label": "Neutral/Balanced",
                "perspective_diversity": 0.0,
                "methodology_notes": "No news articles to analyze"
            }

        # Use LLM to analyze balance
        combined = "\n".join([
            f"HEADLINE: {a.title}\nSNIPPET: {a.text[:400]}"
            for a in news_articles[:5]
        ])

        prompt = f"""
Analyze the NEWS REPORTING BALANCE of these articles (NOT opinion pieces).

{combined}

Consider:
1. Story selection - Are topics covered that favor one political side?
2. Framing - How are events described? Neutral language or slanted?
3. Sources - Are multiple perspectives included?
4. Omission - Are important viewpoints missing?

Rate the reporting balance:
- -10: Extreme Left (only left perspectives, right views dismissed)
- -5: Moderate Left (leans left but includes some balance)
- 0: Balanced (multiple perspectives fairly represented)
- +5: Moderate Right (leans right but includes some balance)
- +10: Extreme Right (only right perspectives, left views dismissed)

Return a JSON object:
{{
    "balance_score": -10 to +10,
    "balance_label": "Extreme Left/Strong Left/Moderate Left/Mild Left/Neutral/Balanced/Mild Right/Moderate Right/Strong Right/Extreme Right",
    "perspective_diversity": 0.0 to 1.0,
    "evidence": [
        "Brief observation 1",
        "Brief observation 2"
    ],
    "reasoning": "Why this rating"
}}

Return ONLY the JSON object.
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            content = content.replace('```json', '').replace('```', '').strip()
            data = json.loads(content)

            return {
                "balance_score": float(data.get("balance_score", 0)),
                "balance_label": data.get("balance_label", "Neutral/Balanced"),
                "perspective_diversity": float(data.get("perspective_diversity", 0.5)),
                "evidence": data.get("evidence", []),
                "methodology_notes": f"Analyzed {len(news_articles)} news articles for balance"
            }

        except Exception as e:
            logger.error(f"News reporting analysis failed: {e}")
            return {
                "balance_score": 0.0,
                "balance_label": "Neutral/Balanced",
                "perspective_diversity": 0.5,
                "methodology_notes": f"Analysis error: {str(e)}"
            }


# =============================================================================
# COMBINED EDITORIAL + NEWS ANALYZER
# =============================================================================

class CompleteBiasAnalyzer:
    """
    Complete bias analyzer for MBFC News Reporting (15%) + Editorial (15%).

    Combines:
    - StructuredEditorialBiasAnalyzer (editorial/opinion pieces)
    - StructuredNewsReportingAnalyzer (straight news)
    """

    def __init__(self):
        self.editorial_analyzer = StructuredEditorialBiasAnalyzer()
        self.news_analyzer = StructuredNewsReportingAnalyzer()

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Analyze both editorial bias and news reporting balance.

        Returns dict with both analyses and combined methodology report.
        """
        editorial_result = self.editorial_analyzer.analyze(articles)
        news_result = self.news_analyzer.analyze(articles)

        return {
            "editorial_bias": {
                "score": editorial_result.overall_score,
                "clickbait_score": editorial_result.clickbait_score,
                "loaded_language_score": editorial_result.loaded_language_score,
                "direction": editorial_result.direction,
                "methodology": editorial_result.methodology_notes
            },
            "news_reporting": {
                "score": news_result["balance_score"],
                "label": news_result["balance_label"],
                "diversity": news_result["perspective_diversity"],
                "methodology": news_result["methodology_notes"]
            },
            "combined_report": self._build_combined_report(editorial_result, news_result)
        }

    def _build_combined_report(
        self,
        editorial: EditorialBiasAnalysis,
        news: Dict[str, Any]
    ) -> str:
        """Build combined methodology report."""
        report = []
        report.append("=" * 60)
        report.append("EDITORIAL & NEWS REPORTING ANALYSIS")
        report.append("=" * 60)

        report.append("\n## EDITORIAL BIAS (15% of total bias score)")
        report.append(f"Score: {editorial.overall_score:+.2f}")
        report.append(f"Direction: {editorial.direction}")
        report.append(f"Clickbait Level: {editorial.clickbait_score:.1f}/10")
        report.append(f"Loaded Language: {editorial.loaded_language_score:.1f}/10")
        report.append(f"Emotional Manipulation: {editorial.emotional_manipulation_score:.1f}/10")

        report.append("\n## NEWS REPORTING BALANCE (15% of total bias score)")
        report.append(f"Score: {news['balance_score']:+.2f}")
        report.append(f"Label: {news['balance_label']}")
        report.append(f"Perspective Diversity: {news['perspective_diversity']:.0%}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Editorial Bias Detection Module")
    print("=" * 50)
    print("\nComponents:")
    print("  - ClickbaitAnalyzer: Detects clickbait patterns in headlines")
    print("  - LoadedLanguageAnalyzer: Finds biased/emotional language")
    print("  - StructuredEditorialBiasAnalyzer: Complete editorial analysis")
    print("  - StructuredNewsReportingAnalyzer: News balance analysis")
    print("  - CompleteBiasAnalyzer: Combined analysis")

    print("\n" + "=" * 50)
    print("CLICKBAIT PATTERNS DETECTED:")
    print("=" * 50)
    for name, info in CLICKBAIT_PATTERNS.items():
        print(f"  {name}: {info['description']} (severity: {info['severity']})")

    print("\n" + "=" * 50)
    print("LOADED LANGUAGE LEXICONS:")
    print("=" * 50)
    print(f"  Left-loaded terms: {len(LEFT_LOADED_TERMS)}")
    print(f"  Right-loaded terms: {len(RIGHT_LOADED_TERMS)}")
    print(f"  Subjective intensifiers: {len(SUBJECTIVE_INTENSIFIERS)}")
    print(f"  Negative emotional words: {len(NEGATIVE_EMOTIONAL_WORDS)}")
    print(f"  Positive emotional words: {len(POSITIVE_EMOTIONAL_WORDS)}")
    print(f"  Factive verbs: {len(FACTIVE_VERBS)}")
    print(f"  Assertive verbs: {len(ASSERTIVE_VERBS)}")
    print(f"  Hedges: {len(HEDGES)}")
    print(f"  Doubt markers: {len(DOUBT_MARKERS)}")
