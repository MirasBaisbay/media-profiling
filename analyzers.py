"""
Analyzers Module
Implements MBFC Math, SemEval Propaganda, and CSV-based Freedom Analysis
"""

import logging
import json
import csv
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

from config import (
    PROPAGANDA_TECHNIQUES, BiasWeights, FactualWeights, 
    ECONOMIC_SCALE, SOCIAL_SCALE, ISO_MAPPING, FREEDOM_INDEX_FILE,
    FACTUALITY_RANGES, BIAS_RANGES, CREDIBILITY_POINTS
)
from scraper import Article, SiteMetadata
from local_detector import LocalPropagandaDetector

local_detector = None
logger = logging.getLogger(__name__)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
search_tool = DuckDuckGoSearchRun()

# --- DATA CLASSES ---
@dataclass
class PropagandaInstance:
    text_snippet: str
    technique: str
    confidence: float
    context: str = ""
    verified: bool = False

@dataclass
class PropagandaAnalysis:
    score: float
    instances: List[PropagandaInstance]
    verified_count: int = 0

# --- ANALYZERS ---

class CountryFreedomAnalyzer:
    def analyze(self, country_code: str) -> Dict[str, Any]:
        """Calculates Freedom Score from 2025.csv file."""
        code = country_code.upper()
        # Convert 2-letter to 3-letter ISO if needed
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
                            # Parse Score (e.g., "87,18" -> 87.18)
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

        # Determine Rating Label (MBFC / RSF Scale)
        if score >= 90: rating = "Excellent Freedom"
        elif score >= 70: rating = "Mostly Free"
        elif score >= 50: rating = "Moderate Freedom"
        elif score >= 25: rating = "Limited Freedom"
        else: rating = "Total Oppression"
        
        # Determine Credibility Penalty
        penalty = 0
        if rating == "Limited Freedom": penalty = -1
        if rating == "Total Oppression": penalty = -2
        
        return {
            "rating": rating,
            "score": score,
            "penalty": penalty,
            "iso": iso3
        }

class TrafficLongevityAnalyzer:
    def analyze(self, url: str) -> Dict[str, Any]:
        """Estimates traffic and age using LLM knowledge."""
        prompt = f"""
        Estimate the traffic and longevity for: {url}
        1. Traffic: High (Global/Major National), Medium (Regional), or Minimal (Local).
        2. Age: Is it older than 10 years? (Yes/No)
        Return JSON: {{"traffic": "High/Medium/Minimal", "older_than_10": true/false}}
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            data = json.loads(res.content.replace('```json','').replace('```',''))
            
            traffic_pts = CREDIBILITY_POINTS["traffic"].get(data["traffic"], 0)
            bonus = 1 if data["older_than_10"] else 0
            
            return {
                "traffic_label": data["traffic"],
                "points": traffic_pts + bonus,
                "details": f"{data['traffic']} Traffic + {'>10 Years' if bonus else '<10 Years'}"
            }
        except:
            return {"traffic_label": "Medium", "points": 1, "details": "Estimation Failed"}

class MediaTypeAnalyzer:
    def analyze(self, url: str, articles: List[Article]) -> str:
        headlines = "\n".join([a.title for a in articles[:5]])
        prompt = f"""
        Classify Media Type for '{url}'.
        Choose ONE: [TV Station, Newspaper, Website, Magazine, Blog, State Media].
        Headlines: {headlines}
        Return ONLY category name.
        """
        try:
            res = llm.invoke([HumanMessage(content=prompt)])
            return res.content.strip().replace(".", "")
        except:
            return "Website"

class EconomicAnalyzer:
    def analyze(self, articles: List[Article]) -> str:
        combined = "\n".join([f"- {a.title}: {a.text[:200]}" for a in articles[:10]])
        prompt = f"""
        Analyze economic bias. Map to ONE: {list(ECONOMIC_SCALE.keys())}.
        Text: {combined}
        Return ONLY category name.
        """
        res = llm.invoke([HumanMessage(content=prompt)])
        return res.content.strip().replace("'", "").replace('"', '')

class SocialAnalyzer:
    def analyze(self, articles: List[Article]) -> str:
        combined = "\n".join([f"- {a.title}: {a.text[:200]}" for a in articles[:10]])
        prompt = f"""
        Analyze social bias. Map to ONE: {list(SOCIAL_SCALE.keys())}.
        Text: {combined}
        Return ONLY category name.
        """
        res = llm.invoke([HumanMessage(content=prompt)])
        return res.content.strip().replace("'", "").replace('"', '')

class PropagandaAnalyzer:
    def __init__(self, use_local_model: bool = False):
        self.use_local_model = use_local_model
        
    def analyze(self, articles: List[Article]) -> PropagandaAnalysis:
        logger.info(f"--- Analyzing Propaganda (Mode: {'LOCAL' if self.use_local_model else 'LLM'}) ---")
        
        combined_text = "\n".join([a.text[:1500] for a in articles[:3]])
        instances = []
        
        if self.use_local_model:
            # --- LOCAL PIPELINE ---
            global local_detector
            if local_detector is None:
                local_detector = LocalPropagandaDetector()
            
            raw_findings = local_detector.detect(combined_text)
            
            for item in raw_findings:
                instances.append(PropagandaInstance(
                    text_snippet=item["text_snippet"],
                    technique=item["technique"],
                    confidence=item["confidence"],
                    context=item["context"]
                ))
        else:
            # --- LLM PIPELINE (Your existing code) ---
            prompt = f"""
            Identify propaganda techniques (SemEval-2020 Task 11):
            {json.dumps(PROPAGANDA_TECHNIQUES, indent=2)}
            
            Return JSON list 'findings': {{"technique": "", "text_snippet": "", "context": "full sentence", "confidence": float}}
            TEXT: {combined_text}
            """
            try:
                response = llm.invoke([HumanMessage(content=prompt)])
                content = response.content.replace('```json', '').replace('```', '')
                data = json.loads(content)
                instances = [PropagandaInstance(**item) for item in data.get("findings", [])]
            except Exception as e:
                logger.error(f"LLM Propaganda analysis failed: {e}")

        # Common scoring logic
        score = min(10.0, len(instances) * 1.5)
        return PropagandaAnalysis(score=score, instances=instances)

class TransparencyAnalyzer:
    def analyze(self, meta: SiteMetadata) -> float:
        score = 0.0
        if not meta.has_about_page: score += 2.0
        if not meta.ownership_disclosed: score += 2.0
        if not meta.funding_disclosed: score += 2.0
        if not meta.has_author_pages: score += 2.0
        if not meta.location_disclosed: score += 2.0
        return float(score)

class ScoringCalculator:
    @staticmethod
    def get_bias_label(score: float) -> str:
        for low, high, label in BIAS_RANGES:
            if low <= score <= high: return label
        return "Least Biased"

    @staticmethod
    def get_factuality_label(score: float) -> str:
        for low, high, label in FACTUALITY_RANGES:
            if low <= score <= high: return label
        return "Mixed"

    @staticmethod
    def calculate_bias(econ_label, social_label):
        econ = ECONOMIC_SCALE.get(econ_label, 0.0)
        social = SOCIAL_SCALE.get(social_label, 0.0)
        w = BiasWeights()
        rep_val = -1.0 if econ < 0 else 1.0 # Heuristic alignment
        return round((econ * w.economic) + (social * w.social) + (rep_val * w.reporting * 2), 2)

    @staticmethod
    def calculate_factuality(transparency, propaganda):
        w = FactualWeights()
        # Assume 0 fact checks and 2.0 sourcing for now (Simulated)
        return round((0 * w.failed_fact_checks) + (2.0 * w.sourcing) + 
                     (transparency * w.transparency) + (propaganda * w.bias_propaganda), 2)
    
    @staticmethod
    def calculate_credibility(fact_label, bias_label, traffic_points, freedom_penalty):
        fact_pts = CREDIBILITY_POINTS["factual"].get(fact_label, 1)
        bias_pts = 0
        for key, val in CREDIBILITY_POINTS["bias"].items():
            if key in bias_label: bias_pts = val; break
        
        total = fact_pts + bias_pts + traffic_points + freedom_penalty
        
        if total >= 6: level = "High Credibility"
        elif total >= 3: level = "Medium Credibility"
        else: level = "Low Credibility"
        
        # Automatic downgrade rule
        if fact_label == "Mostly Factual" and level == "Low Credibility":
            level = "Medium Credibility"
            
        return total, level, fact_pts, bias_pts