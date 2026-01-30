"""
Configuration for Media Profiler
Aligned with MBFC Credibility & Freedom Methodology
"""
import os
from dataclasses import dataclass

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# =============================================================================
# FILE PATHS
# =============================================================================
FREEDOM_INDEX_FILE = "2025.csv"

# =============================================================================
# ISO MAPPING (2-Letter -> 3-Letter)
# =============================================================================
ISO_MAPPING = {
    "KZ": "KAZ", "RU": "RUS", "US": "USA", "GB": "GBR", "UK": "GBR",
    "NZ": "NZL", "CN": "CHN", "FR": "FRA", "DE": "DEU", "CA": "CAN",
    "AU": "AUS", "UA": "UKR", "IL": "ISR", "TR": "TUR", "BY": "BLR",
    "JP": "JPN", "KR": "KOR", "IN": "IND", "BR": "BRA", "ZA": "ZAF"
}

# =============================================================================
# SCORING CONSTANTS
# =============================================================================

# MBFC Credibility Points
CREDIBILITY_POINTS = {
    "factual": {
        "Very High": 4, "High": 3, "Mostly Factual": 2, 
        "Mixed": 1, "Low": 0, "Very Low": 0
    },
    "bias": {
        "Least Biased": 3, "Pro-Science": 3, 
        "Right-Center": 2, "Left-Center": 2, 
        "Right": 1, "Left": 1, 
        "Extreme": 0, "Questionable": 0
    },
    "traffic": {
        "High": 2, "Medium": 1, "Minimal": 0
    },
    "freedom_penalty": {
        "Limited Freedom": -1,
        "Total Oppression": -2
    }
}

# Mapping Weighted Fact Score (0-10, Lower is Better) to Labels
FACTUALITY_RANGES = [
    (0.0, 0.4, "Very High"),
    (0.5, 1.9, "High"),
    (2.0, 4.4, "Mostly Factual"),
    (4.5, 6.4, "Mixed"),
    (6.5, 8.4, "Low"),
    (8.5, 10.0, "Very Low")
]

# Mapping Weighted Bias Score (-10 to +10) to Labels
BIAS_RANGES = [
    (-10.0, -8.0, "Extreme Left"),
    (-7.9, -5.0, "Left"),
    (-4.9, -2.0, "Left-Center"),
    (-1.9, 1.9, "Least Biased"),
    (2.0, 4.9, "Right-Center"),
    (5.0, 7.9, "Right"),
    (8.0, 10.0, "Extreme Right")
]

# =============================================================================
# SEMEVAL & SCALES
# =============================================================================
PROPAGANDA_TECHNIQUES = [
    "Presenting Irrelevant Data (Red Herring)",
    "Misrepresentation of Someone's Position (Straw Man)",
    "Whataboutism",
    "Causal Oversimplification",
    "Obfuscation, Intentional Vagueness, Confusion",
    "Appeal to Authority",
    "Black-and-White Fallacy, Dictatorship",
    "Name Calling or Labeling",
    "Loaded Language",
    "Exaggeration or Minimization",
    "Flag-Waving",
    "Doubt",
    "Appeal to Fear/Prejudice",
    "Slogans",
    "Thought-terminating Cliche",
    "Bandwagon",
    "Reductio ad Hitlerum",
    "Repetition"
]

ECONOMIC_SCALE = {
    "Communism": -10.0, "Socialism": -7.5, "Democratic Socialism": -5.0,
    "Regulated Market Economy": -2.5, "Centrism": 0.0,
    "Moderately Regulated Capitalism": 2.5, "Classical Liberalism": 5.0,
    "Libertarianism": 7.5, "Radical Laissez-Faire": 10.0
}

SOCIAL_SCALE = {
    "Strong Progressive": -10.0, "Progressive": -7.5, "Moderate Progressive": -5.0,
    "Mild Progressive": -2.5, "Balanced": 0.0,
    "Mild Conservative": 2.5, "Moderate Conservative": 5.0,
    "Traditional Conservative": 7.5, "Strong Traditional Conservative": 10.0
}

@dataclass
class BiasWeights:
    economic: float = 0.35
    social: float = 0.35
    reporting: float = 0.15
    editorial: float = 0.15

@dataclass
class FactualWeights:
    failed_fact_checks: float = 0.40
    sourcing: float = 0.25
    transparency: float = 0.25
    bias_propaganda: float = 0.10