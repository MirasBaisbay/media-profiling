# Media Profiler - MBFC Methodology Compliant Analysis System

A comprehensive media source analysis system that evaluates news outlets for **political bias**, **factual reliability**, and **overall credibility** using the [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com/methodology/) methodology.

The system leverages **DeBERTa-v3-Large** fine-tuned on the **SemEval 2020 Task 11** propaganda detection dataset for automated propaganda technique identification.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [MBFC Methodology Compliance](#mbfc-methodology-compliance)
- [DeBERTa Model & SemEval Dataset](#deberta-model--semeval-dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Scoring System](#scoring-system)

---

## Overview

This system performs automated media source evaluation by:

1. **Scraping** articles from a target news website
2. **Analyzing** content across 8 weighted dimensions (4 for bias, 4 for factuality)
3. **Detecting propaganda** using fine-tuned DeBERTa models
4. **Calculating** composite scores following MBFC methodology
5. **Generating** detailed credibility reports

### Key Features

- **MBFC-Compliant Scoring**: Implements the exact weighting system used by Media Bias/Fact Check
- **Two-Stage Propaganda Detection**: Uses DeBERTa-v3-Large for span identification + technique classification
- **Human-in-the-Loop Verification**: Optional manual review of AI-detected propaganda instances
- **Multi-Source Evaluation**: Combines web scraping, LLM analysis, fact-check searches, and metadata extraction
- **Country Freedom Adjustment**: Incorporates RSF/Freedom House press freedom ratings

---

## Architecture

### High-Level System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MEDIA PROFILER SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   SCRAPE     │────▶│   ANALYZE    │────▶│   REPORT     │                │
│  │    NODE      │     │    NODE      │     │    NODE      │                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│         │                    │                    │                         │
│         ▼                    ▼                    ▼                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │ MediaScraper │     │ 8 Analyzers  │     │ MBFC Report  │                │
│  │ - Articles   │     │ (Weighted)   │     │ - Bias Score │                │
│  │ - Metadata   │     │              │     │ - Fact Score │                │
│  │ - News/Op-Ed │     │              │     │ - Credibility│                │
│  └──────────────┘     └──────────────┘     └──────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ANALYZE NODE (8 Components)                        │
├───────────────────────────────────┬─────────────────────────────────────────┤
│         BIAS SCORING (±10)        │       FACTUALITY SCORING (0-10)         │
├───────────────────────────────────┼─────────────────────────────────────────┤
│                                   │                                         │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────┐       │
│  │ EconomicAnalyzer      (35%) │  │  │ FactCheckSearcher     (40%) │       │
│  │ -10 Communism → +10 Laissez │  │  │ IFCN fact-checker search    │       │
│  └─────────────────────────────┘  │  └─────────────────────────────┘       │
│                                   │                                         │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────┐       │
│  │ SocialAnalyzer        (35%) │  │  │ SourcingAnalyzer      (25%) │       │
│  │ -10 Progressive → +10 Trad  │  │  │ Hyperlink/citation quality  │       │
│  └─────────────────────────────┘  │  └─────────────────────────────┘       │
│                                   │                                         │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────┐       │
│  │ NewsReportingAnalyzer (15%) │  │  │ TransparencyAnalyzer  (25%) │       │
│  │ Balance in straight news    │  │  │ Ownership/funding disclosure│       │
│  └─────────────────────────────┘  │  └─────────────────────────────┘       │
│                                   │                                         │
│  ┌─────────────────────────────┐  │  ┌─────────────────────────────┐       │
│  │ EditorialBiasAnalyzer (15%) │  │  │ PropagandaAnalyzer    (10%) │       │
│  │ Opinion/editorial lean      │  │  │ DeBERTa SI+TC pipeline      │       │
│  └─────────────────────────────┘  │  └─────────────────────────────┘       │
│                                   │                                         │
├───────────────────────────────────┴─────────────────────────────────────────┤
│                              CREDIBILITY SCORE                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ = Factuality Points + Bias Points + Traffic Bonus + Freedom Penalty │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Propaganda Detection Pipeline (DeBERTa)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TWO-STAGE PROPAGANDA DETECTION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT TEXT: "The radical left wants to destroy our great nation..."       │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 1: SPAN IDENTIFICATION (SI)                                    │  │
│  │ Model: DeBERTa-v3-Large (Token Classification, BIO tagging)          │  │
│  │ Task: Find WHERE propaganda exists                                   │  │
│  │                                                                      │  │
│  │ Input:  ["The", "radical", "left", "wants", "to", "destroy", ...]    │  │
│  │ Output: [ O,    B-PROP,   I-PROP,  O,      O,   O,         ...]      │  │
│  │                  ^^^^^^^^^^^^^^                                      │  │
│  │                  Detected Span                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │ STAGE 2: TECHNIQUE CLASSIFICATION (TC)                               │  │
│  │ Model: DeBERTa-v3-Large (Sequence Classification, 14 classes)        │  │
│  │ Task: Identify WHICH technique is used                               │  │
│  │                                                                      │  │
│  │ Input:  "[Context] [SEP] radical left"                               │  │
│  │ Output: "Name_Calling,Labeling" (confidence: 0.87)                   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MBFC Methodology Compliance

This system implements the **Media Bias/Fact Check 2025 Methodology** with exact weight matching:

### Bias Scoring (Scale: -10 to +10)

| Component | Weight | Description |
|-----------|--------|-------------|
| **Economic System** | 35% | -10 (Communism) to +10 (Radical Laissez-Faire) |
| **Social Values** | 35% | -10 (Strong Progressive) to +10 (Strong Traditional Conservative) |
| **News Reporting Balance** | 15% | Balance in straight news coverage |
| **Editorial Bias** | 15% | Bias in opinion/editorial pieces |

**Bias Labels:**
- **-10 to -8.0**: Extreme Left
- **-7.9 to -5.0**: Left
- **-4.9 to -2.0**: Left-Center
- **-1.9 to +1.9**: Least Biased
- **+2.0 to +4.9**: Right-Center
- **+5.0 to +7.9**: Right
- **+8.0 to +10**: Extreme Right

### Factuality Scoring (Scale: 0 to 10, lower is better)

| Component | Weight | Description |
|-----------|--------|-------------|
| **Failed Fact Checks** | 40% | Count from IFCN-approved fact-checkers |
| **Sourcing Quality** | 25% | Hyperlinks, citations, credible references |
| **Transparency** | 25% | Ownership, funding, authorship disclosure |
| **Propaganda/One-Sidedness** | 10% | Detected propaganda techniques |

**Factuality Labels:**
- **0.0 - 0.4**: Very High
- **0.5 - 1.9**: High
- **2.0 - 4.4**: Mostly Factual
- **4.5 - 6.4**: Mixed
- **6.5 - 8.4**: Low
- **8.5 - 10.0**: Very Low

### Overall Credibility Score (0-10)

```
Credibility = Factuality Points + Bias Points + Traffic Bonus + Freedom Penalty

Where:
- Factuality Points: Very High=4, High=3, Mostly Factual=2, Mixed=1, Low/Very Low=0
- Bias Points: Least Biased=3, Center=2, Left/Right=1, Extreme=0
- Traffic Bonus: High=2, Medium=1, Minimal=0, +1 if >10 years old
- Freedom Penalty: Limited Freedom=-1, Total Oppression=-2
```

**Credibility Levels:**
- **6-10**: High Credibility
- **3-5**: Medium Credibility
- **0-2**: Low Credibility

---

## DeBERTa Model & SemEval Dataset

### Model: microsoft/deberta-v3-large

- **Parameters**: 304 million
- **Architecture**: DeBERTa v3 with disentangled attention
- **Max Sequence Length**: 512 tokens (SI), 384 tokens (TC)
- **Training**: Fine-tuned on SemEval 2020 Task 11

### Dataset: SemEval 2020 Task 11 - Propaganda Techniques Corpus (PTC)

The system uses the **V2** version of the dataset with 14 merged propaganda techniques:

| # | Technique | Description |
|---|-----------|-------------|
| 1 | Appeal_to_Authority | Citing authority figures to support claims |
| 2 | Appeal_to_fear-prejudice | Using fear or prejudice to influence |
| 3 | Bandwagon,Reductio_ad_Hitlerum | "Everyone does it" or Nazi comparisons |
| 4 | Black-and-White_Fallacy | Presenting only two choices |
| 5 | Causal_Oversimplification | Oversimplifying cause-effect relationships |
| 6 | Doubt | Questioning credibility without evidence |
| 7 | Exaggeration,Minimisation | Overstating or understating facts |
| 8 | Flag-Waving | Appealing to patriotism/nationalism |
| 9 | Loaded_Language | Using emotionally charged words |
| 10 | Name_Calling,Labeling | Using derogatory labels |
| 11 | Repetition | Repeating messages for emphasis |
| 12 | Slogans | Using catchy phrases |
| 13 | Thought-terminating_Cliches | Phrases that discourage critical thinking |
| 14 | Whataboutism,Straw_Men,Red_Herring | Deflection and misrepresentation tactics |

### Dataset Structure

```
datasets/
├── train/
│   ├── articles/
│   │   └── article*.txt              # Raw article text
│   └── labels/
│       └── article*.task-flc-tc.labels   # Annotations (TSV format)
├── dev/
│   ├── articles/
│   └── labels/
└── test/
    ├── articles/
    └── labels/
```

### Training Configuration

```python
ModelConfig:
    model_checkpoint = "microsoft/deberta-v3-large"
    max_length = 512          # SI model sequence length
    tc_max_length = 384       # TC model sequence length
    context_window = 150      # Context around propaganda spans

TrainingConfig:
    learning_rate_si = 1e-5   # Span Identification
    learning_rate_tc = 1.5e-5 # Technique Classification
    batch_size_si = 4         # With gradient accumulation: effective 32
    batch_size_tc = 8         # With gradient accumulation: effective 64
    num_epochs_si = 15
    num_epochs_tc = 10
    use_focal_loss = True     # Handle class imbalance
    focal_loss_gamma = 1.5
    early_stopping_patience = 4
```

---

## Project Structure

```
fine-tune-semeval/
│
├── config.py              # Configuration constants and scoring scales
│   ├── MBFC scoring constants (CREDIBILITY_POINTS, BIAS_RANGES, etc.)
│   ├── Propaganda techniques (14 classes)
│   ├── Economic/Social/Editorial scales
│   ├── ModelConfig (DeBERTa settings)
│   └── TrainingConfig (training hyperparameters)
│
├── train_pipeline.py      # DeBERTa fine-tuning pipeline
│   ├── Data loading (local SemEval dataset)
│   ├── SI dataset creation (BIO tagging)
│   ├── TC dataset creation (context + snippet)
│   ├── Class weight computation
│   ├── Focal loss implementation
│   └── Custom trainers with early stopping
│
├── local_detector.py      # Inference pipeline for trained models
│   └── LocalPropagandaDetector (SI → TC pipeline)
│
├── analyzers.py           # MBFC-compliant analysis components
│   ├── Bias Analyzers (Economic, Social, NewsReporting, Editorial)
│   ├── Factuality Analyzers (FactCheck, Sourcing, Transparency, Propaganda)
│   └── ScoringCalculator (weighted score computation)
│
├── scraper.py             # Web scraping for articles and metadata
│   ├── MediaScraper (threaded article extraction)
│   ├── Article dataclass
│   └── SiteMetadata dataclass
│
├── profiler.py            # LangGraph orchestration workflow
│   ├── ProfilerState (TypedDict)
│   ├── scrape_node → analyze_node → report_node
│   ├── Human review process
│   └── CLI interface
│
├── 2025.csv               # Freedom Index dataset (181 countries)
│
├── datasets/              # SemEval 2020 Task 11 data
│   ├── train/
│   ├── dev/
│   └── test/
│
└── propaganda_models/     # Trained model outputs
    ├── si_model/          # Span Identification model
    └── tc_model/          # Technique Classification model
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for training)
- 16GB+ RAM for inference, 24GB+ GPU VRAM for training

### Setup

```bash
# Clone repository
git clone https://github.com/MirasBaisbay/fine-tune-semeval.git
cd fine-tune-semeval

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch transformers datasets langgraph beautifulsoup4 requests scikit-learn duckduckgo-search openai

# Set API key (for LLM-based analyzers)
export OPENAI_API_KEY="your-api-key"
```

### Train Models (Optional)

If you need to train the propaganda detection models from scratch:

```bash
# Ensure datasets/ folder contains SemEval 2020 Task 11 data
python train_pipeline.py
```

This will create trained models in `propaganda_models/si_model/` and `propaganda_models/tc_model/`.

---

## Usage

### Basic Usage

```bash
# Analyze a news website (using local DeBERTa models)
python profiler.py https://example-news.com US --model local

# Analyze with LLM fallback (if local models unavailable)
python profiler.py https://example-news.com GB --model llm

# Skip human review of propaganda detections
python profiler.py https://example-news.com US --model local --no-review
```

### Command Line Options

```
usage: profiler.py [-h] [--model {llm,local}] [--no-review] url country

positional arguments:
  url                   Target URL to analyze
  country               Country Code (e.g., US, GB, KZ)

optional arguments:
  --model {llm,local}   Propaganda detection backend (default: local)
  --no-review           Skip human review of propaganda detection
```

### Sample Output

```
================================================================================
                    MEDIA BIAS / FACT CHECK REPORT
                      (MBFC Methodology Compliant)
================================================================================

TARGET: https://example-news.com
MEDIA TYPE: News/Media
COUNTRY: US - Freedom Rating: Excellent Freedom (88/100)
TRAFFIC: High Traffic (>1M monthly visits)

--------------------------------------------------------------------------------
1. BIAS RATING
--------------------------------------------------------------------------------
   Component Scores (Scale: -10 Left to +10 Right):

   Economic Position (35%):      -2.5
   Social Position (35%):        -5.0
   News Reporting Balance (15%): -2.5
   Editorial Bias (15%):         -5.0

   WEIGHTED BIAS SCORE: -3.63
   BIAS LABEL: LEFT-CENTER

--------------------------------------------------------------------------------
2. FACTUAL REPORTING
--------------------------------------------------------------------------------
   Component Scores (Scale: 0 Best to 10 Worst):

   Failed Fact Checks (40%):     1.0/10
   Sourcing Quality (25%):       2.0/10
   Transparency (25%):           0.0/10
   Propaganda/Bias in News (10%): 3.0/10

   WEIGHTED FACTUALITY SCORE: 1.20 (Lower is Better)
   FACTUALITY LABEL: HIGH

--------------------------------------------------------------------------------
3. OVERALL CREDIBILITY SCORE (0-10)
--------------------------------------------------------------------------------
   Factual Reporting Points: +3 (High)
   Bias Rating Points:       +2 (Left-Center)
   Traffic Bonus:            +2
   Freedom Penalty:          0

   -------------------------------------
   TOTAL CREDIBILITY SCORE: 7/10
   VERDICT: HIGH CREDIBILITY
   -------------------------------------

================================================================================
                              END OF REPORT
================================================================================
```

### Programmatic Usage

```python
from profiler import app

# Run analysis
result = app.invoke({
    "target_url": "https://example-news.com",
    "country_code": "US",
    "use_local_model": True  # Use DeBERTa models
})

print(result["final_report"])
```

---

## Scoring System

### Bias Score Calculation

```python
bias_score = (
    economic_score * 0.35 +    # Economic position
    social_score * 0.35 +      # Social values
    news_reporting * 0.15 +    # News balance
    editorial_bias * 0.15      # Editorial lean
)
```

### Factuality Score Calculation

```python
fact_score = (
    fact_checks * 0.40 +       # Failed fact checks
    sourcing * 0.25 +          # Source quality
    transparency * 0.25 +      # Disclosure level
    propaganda * 0.10          # Propaganda instances
)
```

### Credibility Score Calculation

```python
credibility = (
    factuality_points +        # 0-4 based on fact label
    bias_points +              # 0-3 based on bias label
    traffic_bonus +            # 0-3 based on traffic/age
    freedom_penalty            # -2 to 0 based on country
)
```

---

## License

MIT License

## References

- [Media Bias/Fact Check Methodology](https://mediabiasfactcheck.com/methodology/)
- [SemEval 2020 Task 11: Detection of Propaganda Techniques](https://propaganda.qcri.org/semeval2020-task11/)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)
- [Freedom House - Freedom in the World](https://freedomhouse.org/report/freedom-world)
- [Reporters Without Borders - Press Freedom Index](https://rsf.org/en/index)
