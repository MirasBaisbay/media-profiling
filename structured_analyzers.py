"""
Structured Analyzers for Media Bias Detection

This module implements a more rigorous methodology for analyzing media bias,
following the approach suggested by the research advisor:

1. Use structured questions with yes/no/irrelevant answers
2. Search articles using keywords related to ideologies
3. Aggregate answers to determine final scores
4. All questions backed by academic references

This replaces the simple "ask LLM to classify" approach with a more
methodologically sound multi-step analysis.
"""

import logging
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from scraper import Article
from ideology_questions import (
    ECONOMIC_QUESTIONS,
    SOCIAL_QUESTIONS,
    IdeologyQuestion,
    get_economic_keywords,
    get_social_keywords,
)
from config import (
    ECONOMIC_SCALE,
    SOCIAL_SCALE,
    BiasWeights,
)

logger = logging.getLogger(__name__)

# Use a more capable model for structured analysis
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class QuestionResult:
    """Result of evaluating a single question against an article."""
    question: str
    answer: str  # "yes", "no", "irrelevant"
    confidence: float  # 0.0 to 1.0
    evidence: str  # Quote or reasoning from article
    ideology_if_yes: str
    score_if_yes: float


@dataclass
class ArticleAnalysis:
    """Analysis results for a single article."""
    article_title: str
    article_text_preview: str
    matched_keywords: List[str]
    question_results: List[QuestionResult]
    relevant_ideologies: Dict[str, float]  # ideology -> confidence


@dataclass
class DimensionAnalysis:
    """Final analysis for an ideology dimension (economic or social)."""
    dimension: str  # "economic" or "social"
    final_score: float  # -10 to +10
    final_label: str
    confidence: float  # Overall confidence in the rating
    article_analyses: List[ArticleAnalysis]
    ideology_evidence: Dict[str, List[str]]  # ideology -> list of evidence quotes
    methodology_notes: str


# =============================================================================
# KEYWORD MATCHER
# =============================================================================

class KeywordMatcher:
    """Finds articles relevant to ideology questions using keyword matching."""

    def __init__(self):
        self.economic_keywords = get_economic_keywords()
        self.social_keywords = get_social_keywords()

    def find_matching_keywords(
        self,
        text: str,
        keywords: List[str]
    ) -> List[str]:
        """Find which keywords appear in the text."""
        text_lower = text.lower()
        matched = []
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        return matched

    def get_relevant_articles_economic(
        self,
        articles: List[Article],
        min_keywords: int = 1
    ) -> List[Tuple[Article, List[str]]]:
        """Filter articles relevant to economic ideology analysis."""
        results = []
        for article in articles:
            full_text = f"{article.title} {article.text}"
            matched = self.find_matching_keywords(full_text, self.economic_keywords)
            if len(matched) >= min_keywords:
                results.append((article, matched))
        return results

    def get_relevant_articles_social(
        self,
        articles: List[Article],
        min_keywords: int = 1
    ) -> List[Tuple[Article, List[str]]]:
        """Filter articles relevant to social values analysis."""
        results = []
        for article in articles:
            full_text = f"{article.title} {article.text}"
            matched = self.find_matching_keywords(full_text, self.social_keywords)
            if len(matched) >= min_keywords:
                results.append((article, matched))
        return results

    def get_questions_for_keywords(
        self,
        matched_keywords: List[str],
        questions: List[IdeologyQuestion]
    ) -> List[IdeologyQuestion]:
        """Find questions whose keywords match the article's keywords."""
        relevant_questions = []
        matched_set = set(kw.lower() for kw in matched_keywords)

        for question in questions:
            question_keywords = set(kw.lower() for kw in question.keywords)
            if matched_set & question_keywords:  # Intersection
                relevant_questions.append(question)

        return relevant_questions


# =============================================================================
# QUESTION EVALUATOR
# =============================================================================

class QuestionEvaluator:
    """Evaluates ideology questions against article content using LLM."""

    def __init__(self):
        self.llm = llm

    def evaluate_question(
        self,
        article: Article,
        question: IdeologyQuestion
    ) -> QuestionResult:
        """
        Evaluate a single yes/no question against an article.

        Returns:
            QuestionResult with answer, confidence, and evidence
        """
        prompt = f"""
You are analyzing a news article to answer a specific yes/no question about its ideological stance.

ARTICLE TITLE: {article.title}

ARTICLE TEXT (first 2000 chars):
{article.text[:2000]}

QUESTION: {question.question}

CONTEXT: This question is designed to detect "{question.yes_maps_to}" ideology (score: {question.yes_score:+.1f} on a -10 to +10 scale).

Keywords that might indicate relevance: {', '.join(question.keywords[:10])}

INSTRUCTIONS:
1. Carefully read the article
2. Determine if the article's content supports a "yes" answer to the question
3. Consider:
   - Does the article explicitly argue for or against this position?
   - Is there implicit support through framing, word choice, or story selection?
   - Is the topic even addressed in the article?

Return a JSON object with:
{{
    "answer": "yes" | "no" | "irrelevant",
    "confidence": 0.0 to 1.0,
    "evidence": "Direct quote or brief explanation (max 100 words)",
    "reasoning": "Why you chose this answer (max 50 words)"
}}

IMPORTANT:
- "irrelevant" means the article doesn't address this topic at all
- "no" means the article addresses the topic but takes the opposite stance
- "yes" means the article clearly supports the position in the question
- Be conservative - only say "yes" if there's clear evidence

Return ONLY the JSON object, no other text.
"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Clean JSON response
            content = content.replace('```json', '').replace('```', '').strip()

            data = json.loads(content)

            return QuestionResult(
                question=question.question,
                answer=data.get("answer", "irrelevant").lower(),
                confidence=float(data.get("confidence", 0.5)),
                evidence=data.get("evidence", ""),
                ideology_if_yes=question.yes_maps_to,
                score_if_yes=question.yes_score
            )

        except Exception as e:
            logger.error(f"Question evaluation failed: {e}")
            return QuestionResult(
                question=question.question,
                answer="irrelevant",
                confidence=0.0,
                evidence=f"Error: {str(e)}",
                ideology_if_yes=question.yes_maps_to,
                score_if_yes=question.yes_score
            )

    def evaluate_questions_batch(
        self,
        article: Article,
        questions: List[IdeologyQuestion]
    ) -> List[QuestionResult]:
        """Evaluate multiple questions against a single article."""
        results = []
        for question in questions:
            result = self.evaluate_question(article, question)
            results.append(result)
        return results


# =============================================================================
# STRUCTURED ECONOMIC ANALYZER
# =============================================================================

class StructuredEconomicAnalyzer:
    """
    Analyzes economic ideology using structured questions.

    Methodology:
    1. Find articles with economic keywords
    2. For each relevant article, ask relevant questions
    3. Aggregate yes/no answers to determine ideology score
    4. Weight by confidence and question importance
    """

    def __init__(self):
        self.keyword_matcher = KeywordMatcher()
        self.question_evaluator = QuestionEvaluator()

    def analyze(self, articles: List[Article]) -> DimensionAnalysis:
        """
        Perform structured economic analysis on articles.

        Returns:
            DimensionAnalysis with score, label, and evidence
        """
        logger.info(f"Starting structured economic analysis on {len(articles)} articles")

        # Step 1: Find articles with economic keywords
        relevant_articles = self.keyword_matcher.get_relevant_articles_economic(
            articles, min_keywords=1
        )
        logger.info(f"Found {len(relevant_articles)} articles with economic keywords")

        if not relevant_articles:
            return DimensionAnalysis(
                dimension="economic",
                final_score=0.0,
                final_label="Centrism",
                confidence=0.0,
                article_analyses=[],
                ideology_evidence={},
                methodology_notes="No articles with economic keywords found"
            )

        # Step 2: Analyze each relevant article
        article_analyses = []
        ideology_scores = defaultdict(list)  # ideology -> list of (score, confidence, weight)
        ideology_evidence = defaultdict(list)  # ideology -> list of evidence quotes

        for article, matched_keywords in relevant_articles[:10]:  # Limit to 10 articles
            # Find questions relevant to this article's keywords
            relevant_questions = self.keyword_matcher.get_questions_for_keywords(
                matched_keywords, ECONOMIC_QUESTIONS
            )

            if not relevant_questions:
                continue

            # Evaluate questions
            question_results = self.question_evaluator.evaluate_questions_batch(
                article, relevant_questions[:5]  # Limit questions per article
            )

            # Collect ideology signals
            relevant_ideologies = {}
            for result in question_results:
                if result.answer == "yes" and result.confidence > 0.5:
                    ideology = result.ideology_if_yes
                    score = result.score_if_yes

                    # Find question weight
                    question_obj = next(
                        (q for q in ECONOMIC_QUESTIONS if q.question == result.question),
                        None
                    )
                    weight = question_obj.weight if question_obj else 1.0

                    ideology_scores[ideology].append((score, result.confidence, weight))
                    ideology_evidence[ideology].append(result.evidence)
                    relevant_ideologies[ideology] = result.confidence

            article_analyses.append(ArticleAnalysis(
                article_title=article.title,
                article_text_preview=article.text[:200],
                matched_keywords=matched_keywords,
                question_results=question_results,
                relevant_ideologies=relevant_ideologies
            ))

        # Step 3: Calculate final score
        final_score, final_label, confidence = self._calculate_final_score(ideology_scores)

        return DimensionAnalysis(
            dimension="economic",
            final_score=final_score,
            final_label=final_label,
            confidence=confidence,
            article_analyses=article_analyses,
            ideology_evidence=dict(ideology_evidence),
            methodology_notes=f"Analyzed {len(article_analyses)} articles with {sum(len(a.question_results) for a in article_analyses)} question evaluations"
        )

    def _calculate_final_score(
        self,
        ideology_scores: Dict[str, List[Tuple[float, float, float]]]
    ) -> Tuple[float, str, float]:
        """
        Calculate final score from ideology evidence.

        Uses weighted average based on:
        - Number of supporting articles
        - Confidence of each answer
        - Weight of each question
        """
        if not ideology_scores:
            return 0.0, "Centrism", 0.0

        # Calculate weighted score for each ideology
        ideology_weights = {}
        for ideology, scores_list in ideology_scores.items():
            total_weight = 0
            for score, confidence, question_weight in scores_list:
                total_weight += confidence * question_weight
            ideology_weights[ideology] = total_weight

        # Calculate final weighted average score
        total_weight = sum(ideology_weights.values())
        if total_weight == 0:
            return 0.0, "Centrism", 0.0

        weighted_score = 0
        for ideology, weight in ideology_weights.items():
            # Get the score for this ideology
            score = ideology_scores[ideology][0][0]  # All entries have same score
            weighted_score += score * (weight / total_weight)

        # Map score to label
        final_label = self._score_to_label(weighted_score, ECONOMIC_SCALE)

        # Confidence based on amount of evidence
        confidence = min(1.0, total_weight / 5.0)  # Normalize

        return round(weighted_score, 2), final_label, round(confidence, 2)

    def _score_to_label(self, score: float, scale: Dict[str, float]) -> str:
        """Find the closest label for a score."""
        closest_label = "Centrism"
        closest_diff = float('inf')

        for label, label_score in scale.items():
            diff = abs(score - label_score)
            if diff < closest_diff:
                closest_diff = diff
                closest_label = label

        return closest_label


# =============================================================================
# STRUCTURED SOCIAL ANALYZER
# =============================================================================

class StructuredSocialAnalyzer:
    """
    Analyzes social values using structured questions.

    Same methodology as StructuredEconomicAnalyzer but for social dimension.
    """

    def __init__(self):
        self.keyword_matcher = KeywordMatcher()
        self.question_evaluator = QuestionEvaluator()

    def analyze(self, articles: List[Article]) -> DimensionAnalysis:
        """
        Perform structured social values analysis on articles.

        Returns:
            DimensionAnalysis with score, label, and evidence
        """
        logger.info(f"Starting structured social analysis on {len(articles)} articles")

        # Step 1: Find articles with social keywords
        relevant_articles = self.keyword_matcher.get_relevant_articles_social(
            articles, min_keywords=1
        )
        logger.info(f"Found {len(relevant_articles)} articles with social keywords")

        if not relevant_articles:
            return DimensionAnalysis(
                dimension="social",
                final_score=0.0,
                final_label="Balanced",
                confidence=0.0,
                article_analyses=[],
                ideology_evidence={},
                methodology_notes="No articles with social keywords found"
            )

        # Step 2: Analyze each relevant article
        article_analyses = []
        ideology_scores = defaultdict(list)
        ideology_evidence = defaultdict(list)

        for article, matched_keywords in relevant_articles[:10]:
            relevant_questions = self.keyword_matcher.get_questions_for_keywords(
                matched_keywords, SOCIAL_QUESTIONS
            )

            if not relevant_questions:
                continue

            question_results = self.question_evaluator.evaluate_questions_batch(
                article, relevant_questions[:5]
            )

            relevant_ideologies = {}
            for result in question_results:
                if result.answer == "yes" and result.confidence > 0.5:
                    ideology = result.ideology_if_yes
                    score = result.score_if_yes

                    question_obj = next(
                        (q for q in SOCIAL_QUESTIONS if q.question == result.question),
                        None
                    )
                    weight = question_obj.weight if question_obj else 1.0

                    ideology_scores[ideology].append((score, result.confidence, weight))
                    ideology_evidence[ideology].append(result.evidence)
                    relevant_ideologies[ideology] = result.confidence

            article_analyses.append(ArticleAnalysis(
                article_title=article.title,
                article_text_preview=article.text[:200],
                matched_keywords=matched_keywords,
                question_results=question_results,
                relevant_ideologies=relevant_ideologies
            ))

        # Step 3: Calculate final score
        final_score, final_label, confidence = self._calculate_final_score(ideology_scores)

        return DimensionAnalysis(
            dimension="social",
            final_score=final_score,
            final_label=final_label,
            confidence=confidence,
            article_analyses=article_analyses,
            ideology_evidence=dict(ideology_evidence),
            methodology_notes=f"Analyzed {len(article_analyses)} articles with {sum(len(a.question_results) for a in article_analyses)} question evaluations"
        )

    def _calculate_final_score(
        self,
        ideology_scores: Dict[str, List[Tuple[float, float, float]]]
    ) -> Tuple[float, str, float]:
        """Calculate final score from ideology evidence."""
        if not ideology_scores:
            return 0.0, "Balanced", 0.0

        ideology_weights = {}
        for ideology, scores_list in ideology_scores.items():
            total_weight = 0
            for score, confidence, question_weight in scores_list:
                total_weight += confidence * question_weight
            ideology_weights[ideology] = total_weight

        total_weight = sum(ideology_weights.values())
        if total_weight == 0:
            return 0.0, "Balanced", 0.0

        weighted_score = 0
        for ideology, weight in ideology_weights.items():
            score = ideology_scores[ideology][0][0]
            weighted_score += score * (weight / total_weight)

        final_label = self._score_to_label(weighted_score, SOCIAL_SCALE)
        confidence = min(1.0, total_weight / 5.0)

        return round(weighted_score, 2), final_label, round(confidence, 2)

    def _score_to_label(self, score: float, scale: Dict[str, float]) -> str:
        """Find the closest label for a score."""
        closest_label = "Balanced"
        closest_diff = float('inf')

        for label, label_score in scale.items():
            diff = abs(score - label_score)
            if diff < closest_diff:
                closest_diff = diff
                closest_label = label

        return closest_label


# =============================================================================
# COMBINED STRUCTURED BIAS ANALYZER
# =============================================================================

# Import editorial bias detection
from editorial_bias_detection import (
    StructuredEditorialBiasAnalyzer,
    StructuredNewsReportingAnalyzer,
    CompleteBiasAnalyzer as EditorialCompleteBiasAnalyzer,
)


class StructuredBiasAnalyzer:
    """
    Combined analyzer for full bias scoring using MBFC methodology.

    Combines:
    - Structured Economic Analysis (35%)
    - Structured Social Analysis (35%)
    - News Reporting Balance (15%)
    - Editorial Bias (15%)

    All components use structured, evidence-based analysis with academic backing.
    """

    def __init__(self):
        self.economic_analyzer = StructuredEconomicAnalyzer()
        self.social_analyzer = StructuredSocialAnalyzer()
        self.editorial_analyzer = StructuredEditorialBiasAnalyzer()
        self.news_reporting_analyzer = StructuredNewsReportingAnalyzer()

    def analyze(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Perform complete bias analysis using all 4 MBFC components.

        Returns dict with:
        - economic_analysis: DimensionAnalysis (35%)
        - social_analysis: DimensionAnalysis (35%)
        - news_reporting_analysis: dict (15%)
        - editorial_analysis: EditorialBiasAnalysis (15%)
        - final_score: float (-10 to +10)
        - final_label: str
        - methodology_report: str
        """
        logger.info("Starting complete structured bias analysis")

        # Run all four analyses
        economic = self.economic_analyzer.analyze(articles)
        social = self.social_analyzer.analyze(articles)
        news_reporting = self.news_reporting_analyzer.analyze(articles)
        editorial = self.editorial_analyzer.analyze(articles)

        # Calculate weighted final score using MBFC weights
        weights = BiasWeights()
        final_score = (
            economic.final_score * weights.economic +           # 35%
            social.final_score * weights.social +               # 35%
            news_reporting["balance_score"] * weights.reporting + # 15%
            editorial.overall_score * weights.editorial          # 15%
        )

        # Determine final label
        final_label = self._get_bias_label(final_score)

        # Build methodology report
        methodology_report = self._build_methodology_report(
            economic, social, news_reporting, editorial, final_score, final_label
        )

        return {
            "economic_analysis": economic,
            "social_analysis": social,
            "news_reporting_analysis": news_reporting,
            "editorial_analysis": editorial,
            "final_score": round(final_score, 2),
            "final_label": final_label,
            "methodology_report": methodology_report,
            # Component scores for easy access
            "component_scores": {
                "economic": economic.final_score,
                "social": social.final_score,
                "news_reporting": news_reporting["balance_score"],
                "editorial": editorial.overall_score
            }
        }

    def _get_bias_label(self, score: float) -> str:
        """Map score to MBFC bias label."""
        if score <= -8.0:
            return "Extreme Left"
        elif score <= -5.0:
            return "Left"
        elif score <= -2.0:
            return "Left-Center"
        elif score <= 1.9:
            return "Least Biased"
        elif score <= 4.9:
            return "Right-Center"
        elif score <= 7.9:
            return "Right"
        else:
            return "Extreme Right"

    def _build_methodology_report(
        self,
        economic: DimensionAnalysis,
        social: DimensionAnalysis,
        news_reporting: Dict[str, Any],
        editorial,
        final_score: float,
        final_label: str
    ) -> str:
        """Build a detailed methodology report."""
        report = []
        report.append("=" * 70)
        report.append("COMPLETE STRUCTURED BIAS ANALYSIS REPORT")
        report.append("(MBFC Methodology Compliant)")
        report.append("=" * 70)

        # Final Result
        report.append(f"\n### FINAL BIAS SCORE: {final_score:+.2f}")
        report.append(f"### FINAL LABEL: {final_label}")

        # Economic Analysis (35%)
        report.append("\n" + "-" * 70)
        report.append("## 1. ECONOMIC SYSTEM (35% weight)")
        report.append("-" * 70)
        report.append(f"Score: {economic.final_score:+.2f}")
        report.append(f"Label: {economic.final_label}")
        report.append(f"Confidence: {economic.confidence:.0%}")
        report.append(f"Method: {economic.methodology_notes}")

        if economic.ideology_evidence:
            report.append("\nEvidence found:")
            for ideology, evidence_list in list(economic.ideology_evidence.items())[:3]:
                report.append(f"  {ideology}:")
                for ev in evidence_list[:1]:
                    report.append(f"    \"{ev[:80]}...\"")

        # Social Analysis (35%)
        report.append("\n" + "-" * 70)
        report.append("## 2. SOCIAL VALUES (35% weight)")
        report.append("-" * 70)
        report.append(f"Score: {social.final_score:+.2f}")
        report.append(f"Label: {social.final_label}")
        report.append(f"Confidence: {social.confidence:.0%}")
        report.append(f"Method: {social.methodology_notes}")

        if social.ideology_evidence:
            report.append("\nEvidence found:")
            for ideology, evidence_list in list(social.ideology_evidence.items())[:3]:
                report.append(f"  {ideology}:")
                for ev in evidence_list[:1]:
                    report.append(f"    \"{ev[:80]}...\"")

        # News Reporting Balance (15%)
        report.append("\n" + "-" * 70)
        report.append("## 3. NEWS REPORTING BALANCE (15% weight)")
        report.append("-" * 70)
        report.append(f"Score: {news_reporting['balance_score']:+.2f}")
        report.append(f"Label: {news_reporting['balance_label']}")
        report.append(f"Perspective Diversity: {news_reporting['perspective_diversity']:.0%}")
        report.append(f"Method: {news_reporting['methodology_notes']}")

        if "evidence" in news_reporting and news_reporting["evidence"]:
            report.append("\nObservations:")
            for ev in news_reporting["evidence"][:2]:
                report.append(f"  - {ev}")

        # Editorial Bias (15%)
        report.append("\n" + "-" * 70)
        report.append("## 4. EDITORIAL BIAS (15% weight)")
        report.append("-" * 70)
        report.append(f"Score: {editorial.overall_score:+.2f}")
        report.append(f"Direction: {editorial.direction}")
        report.append(f"Clickbait Level: {editorial.clickbait_score:.1f}/10")
        report.append(f"Loaded Language: {editorial.loaded_language_score:.1f}/10")
        report.append(f"Emotional Manipulation: {editorial.emotional_manipulation_score:.1f}/10")
        report.append(f"Method: {editorial.methodology_notes}")

        # Clickbait headlines found
        clickbait_headlines = [h for h in editorial.headline_analyses if h.is_clickbait]
        if clickbait_headlines:
            report.append(f"\nClickbait headlines detected ({len(clickbait_headlines)}):")
            for h in clickbait_headlines[:3]:
                report.append(f"  - \"{h.headline[:60]}...\"")
                report.append(f"    Patterns: {h.explanation}")

        # Calculation Summary
        report.append("\n" + "-" * 70)
        report.append("## CALCULATION SUMMARY")
        report.append("-" * 70)
        report.append(f"Economic ({economic.final_score:+.2f}) × 0.35 = {economic.final_score * 0.35:+.2f}")
        report.append(f"Social ({social.final_score:+.2f}) × 0.35 = {social.final_score * 0.35:+.2f}")
        report.append(f"News Reporting ({news_reporting['balance_score']:+.2f}) × 0.15 = {news_reporting['balance_score'] * 0.15:+.2f}")
        report.append(f"Editorial ({editorial.overall_score:+.2f}) × 0.15 = {editorial.overall_score * 0.15:+.2f}")
        report.append(f"TOTAL: {final_score:+.2f} → {final_label}")

        report.append("\n" + "=" * 70)
        report.append("END OF REPORT")
        report.append("=" * 70)

        return "\n".join(report)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Structured Analyzers Module")
    print("=" * 50)
    print("\nThis module provides:")
    print("  - StructuredEconomicAnalyzer: Economic ideology analysis")
    print("  - StructuredSocialAnalyzer: Social values analysis")
    print("  - StructuredBiasAnalyzer: Combined bias analysis")
    print("\nMethodology:")
    print("  1. Find articles matching ideology keywords")
    print("  2. Ask structured yes/no questions")
    print("  3. Aggregate answers weighted by confidence")
    print("  4. All backed by academic references")
    print("\nUsage:")
    print("  from structured_analyzers import StructuredBiasAnalyzer")
    print("  analyzer = StructuredBiasAnalyzer()")
    print("  result = analyzer.analyze(articles)")
