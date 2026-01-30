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

# Dataset paths for local SemEval 2020 Task 11 data
DATASET_DIR = "datasets"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
DEV_DIR = os.path.join(DATASET_DIR, "dev")
TEST_DIR = os.path.join(DATASET_DIR, "test")

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
# SEMEVAL 2020 TASK 11 - V2 PROPAGANDA TECHNIQUES (14 Classes)
# =============================================================================
# Version 2 merged similar techniques and excluded "Obfuscation,Intentional_Vagueness,Confusion"
#
# Merges applied:
#   - "Bandwagon" + "Reductio_ad_Hitlerum" -> "Bandwagon,Reductio_ad_Hitlerum"
#   - "Whataboutism" + "Straw_Men" + "Red_Herring" -> "Whataboutism,Straw_Men,Red_Herring"
#
# Excluded:
#   - "Obfuscation,Intentional_Vagueness,Confusion" (eliminated in V2)
# =============================================================================

PROPAGANDA_TECHNIQUES = [
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_Hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
]

# Label mappings for model training
LABEL2ID = {label: idx for idx, label in enumerate(PROPAGANDA_TECHNIQUES)}
ID2LABEL = {idx: label for idx, label in enumerate(PROPAGANDA_TECHNIQUES)}

# Span Identification BIO labels
SI_LABELS = ["O", "B-PROP", "I-PROP"]
SI_LABEL2ID = {label: idx for idx, label in enumerate(SI_LABELS)}
SI_ID2LABEL = {idx: label for idx, label in enumerate(SI_LABELS)}

# Legacy label mapping: maps V1 labels to V2 merged labels
LEGACY_LABEL_MAPPING = {
    # Direct mappings (unchanged labels)
    "Appeal_to_Authority": "Appeal_to_Authority",
    "Appeal_to_fear-prejudice": "Appeal_to_fear-prejudice",
    "Black-and-White_Fallacy": "Black-and-White_Fallacy",
    "Causal_Oversimplification": "Causal_Oversimplification",
    "Doubt": "Doubt",
    "Exaggeration,Minimisation": "Exaggeration,Minimisation",
    "Flag-Waving": "Flag-Waving",
    "Loaded_Language": "Loaded_Language",
    "Name_Calling,Labeling": "Name_Calling,Labeling",
    "Repetition": "Repetition",
    "Slogans": "Slogans",
    "Thought-terminating_Cliches": "Thought-terminating_Cliches",

    # Merged: Bandwagon + Reductio_ad_Hitlerum
    "Bandwagon": "Bandwagon,Reductio_ad_Hitlerum",
    "Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_Hitlerum",
    "Bandwagon,Reductio_ad_hitlerum": "Bandwagon,Reductio_ad_Hitlerum",

    # Merged: Whataboutism + Straw_Men + Red_Herring
    "Whataboutism": "Whataboutism,Straw_Men,Red_Herring"
}

# Labels to exclude (eliminated in V2)
EXCLUDED_LABELS = [
    "Obfuscation,Intentional_Vagueness,Confusion",
]

# =============================================================================
# ECONOMIC & SOCIAL SCALES
# =============================================================================
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

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
@dataclass
class ModelConfig:
    model_checkpoint: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    tc_max_length: int = 256
    context_window: int = 100  # chars before/after snippet for TC

@dataclass
class TrainingConfig:
    output_dir: str = "./propaganda_models"
    si_model_dir: str = "./propaganda_models/si_model"
    tc_model_dir: str = "./propaganda_models/tc_model"
    learning_rate_si: float = 2e-5
    learning_rate_tc: float = 3e-5
    batch_size_si: int = 8
    batch_size_tc: int = 16
    num_epochs_si: int = 8  # Increased from 1 - SI needs more epochs for imbalanced data
    num_epochs_tc: int = 5  # Increased from 1 - TC needs more epochs for 14 classes
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    use_class_weights: bool = True  # Enable class weighting for imbalanced data
    use_focal_loss: bool = True  # Use focal loss for better handling of hard examples
    focal_loss_gamma: float = 2.0  # Focal loss focusing parameter
    si_early_stopping_patience: int = 3  # Patience for early stopping
    tc_early_stopping_patience: int = 3  # Patience for early stopping
