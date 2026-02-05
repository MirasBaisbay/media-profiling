# Media Profiler

A comprehensive media source analysis system that evaluates news outlets for **political bias**, **factual reliability**, and **overall credibility** using the [Media Bias/Fact Check (MBFC)](https://mediabiasfactcheck.com/methodology/) methodology.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
  - [Bias Scoring](#bias-scoring)
  - [Factuality Scoring](#factuality-scoring)
  - [Credibility Calculation](#credibility-calculation)
- [Core Modules](#core-modules)
- [Analyzer Flow Diagrams](#analyzer-flow-diagrams)
- [Propaganda Detection](#propaganda-detection)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [References](#references)

---

## Overview

Media Profiler automates the evaluation of news sources by:

1. **Scraping** articles from target news websites
2. **Analyzing** content using LLM-based analyzers with structured output
3. **Detecting** propaganda techniques using fine-tuned DeBERTa models
4. **Researching** outlet history, ownership, and external analyses via web search
5. **Calculating** composite scores following MBFC methodology
6. **Generating** detailed credibility reports

### Key Features

- **MBFC-Compliant Scoring** - Implements scoring aligned with Media Bias/Fact Check methodology
- **LLM-Based Analysis** - Uses structured LLM output for type-safe, reliable analysis
- **Two-Stage Propaganda Detection** - DeBERTa-v3-Large for span identification + technique classification
- **Hybrid Analyzers** - Combines deterministic lookups (Tranco, WHOIS) with LLM fallbacks
- **Comprehensive Research** - Gathers history, ownership, and external analyses via web search
- **Multi-Site Fact Checking** - Searches IFCN-approved fact-checkers for failed fact checks

---

## System Architecture

### Complete Analysis Pipeline

```
                         MEDIA PROFILER - COMPLETE PIPELINE
================================================================================

                                   INPUT
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  1. SCRAPE NODE (scraper.py)                                                │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  MediaScraper                                                   │     │
│     │  ├── Fetch homepage + sitemap                                   │     │
│     │  ├── Collect up to 20 articles                                  │     │
│     │  ├── Separate News vs Opinion articles                          │     │
│     │  │   ├── URL patterns (/opinion/, /editorial/)                  │     │
│     │  │   ├── Schema.org metadata                                    │     │
│     │  │   └── Title patterns ("Opinion:", "Editorial:")              │     │
│     │  └── Extract site metadata                                      │     │
│     │       ├── About page (ownership, funding)                       │     │
│     │       ├── Author information                                    │     │
│     │       └── Location disclosure                                   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  2. ANALYZE NODE (refactored_analyzers.py + research.py)                    │
│                                                                             │
│     ┌───────────────────────────┐   ┌───────────────────────────┐           │
│     │    CONTENT ANALYZERS      │   │   METADATA ANALYZERS      │           │
│     ├───────────────────────────┤   ├───────────────────────────┤           │
│     │ EditorialBiasAnalyzer     │   │ TrafficLongevityAnalyzer  │           │
│     │ PseudoscienceAnalyzer     │   │ MediaTypeAnalyzer         │           │
│     │ SourcingAnalyzer          │   │ OpinionAnalyzer           │           │
│     │ FactCheckSearcher         │   │                           │           │
│     └───────────────────────────┘   └───────────────────────────┘           │
│                                                                             │
│     ┌───────────────────────────────────────────────────────────────────┐   │
│     │  WEB RESEARCH (MediaResearcher)                                   │   │
│     │  ├── History research (founding, key events)                      │   │
│     │  ├── Ownership research (owner, funding, headquarters)            │   │
│     │  └── External analysis (MBFC, NewsGuard, academic reviews)        │   │
│     └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  3. REPORT NODE (research.py)                                               │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  MediaProfiler                                                  │     │
│     │  ├── Calculate bias score and label                             │     │
│     │  ├── Calculate factuality score and label                       │     │
│     │  ├── Calculate credibility score and label                      │     │
│     │  └── Generate comprehensive MBFC-style report                   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                                  OUTPUT
```

### Scoring Components

```
+-----------------------------------------------------------------------------+
|           BIAS SCORING (-10 to +10)    |    FACTUALITY SCORING (0-10)       |
|----------------------------------------|-------------------------------------|
|                                        |                                     |
|  EditorialBiasAnalyzer                 |  FactCheckSearcher          (40%)  |
|  LLM-based political bias detection    |  IFCN-approved fact-checker search |
|  -10 Left -> +10 Right                 |                                     |
|                                        |  SourcingAnalyzer           (30%)  |
|  Policy positions detected:            |  Link extraction + LLM quality     |
|  - Economic, Social, Environmental     |                                     |
|  - Healthcare, Immigration, etc.       |  PseudoscienceAnalyzer      (30%)  |
|  - Loaded language identification      |  Science misinformation detection  |
|                                        |                                     |
+-----------------------------------------------------------------------------+
|                                                                             |
|                           CREDIBILITY SCORE (0-10)                          |
|  = FactCheck (40%) + Sourcing (30%) + Pseudoscience (30%)                   |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## Methodology

### Bias Scoring

Scale: **-10 (Extreme Left) to +10 (Extreme Right)**

The `EditorialBiasAnalyzer` uses LLM analysis to evaluate:

| Indicator | Description |
|-----------|-------------|
| **Policy Positions** | Economic, Social, Environmental, Healthcare, Immigration, Gun Rights |
| **Loaded Language** | Left-loaded terms ("regime", "far-right") vs Right-loaded ("woke", "radical left") |
| **Story Selection** | Topics chosen and framing patterns |

**Bias Labels:**
| Score Range | Label |
|-------------|-------|
| -10 to -7.0 | Left |
| -7.0 to -3.0 | Left-Center |
| -3.0 to +3.0 | Center |
| +3.0 to +7.0 | Right-Center |
| +7.0 to +10 | Right |

### Factuality Scoring

Scale: **0 (Best) to 10 (Worst)**

| Component | Weight | Description |
|-----------|--------|-------------|
| **Failed Fact Checks** | 40% | Count from IFCN-approved fact-checkers |
| **Sourcing Quality** | 30% | Link extraction + LLM quality assessment |
| **Pseudoscience** | 30% | Detection of science misinformation |

**Factuality Labels:**
| Score Range | Label |
|-------------|-------|
| 0.0 - 2.0 | Very High |
| 2.0 - 4.0 | High |
| 4.0 - 6.0 | Mixed |
| 6.0 - 8.0 | Low |
| 8.0 - 10.0 | Very Low |

### Credibility Calculation

```
Credibility = FactCheck (40%) + Sourcing (30%) + Pseudoscience (30%)
```

**Credibility Levels:**
| Score Range | Label |
|-------------|-------|
| 0.0 - 2.0 | Very High Credibility |
| 2.0 - 4.0 | High Credibility |
| 4.0 - 6.0 | Medium Credibility |
| 6.0 - 8.0 | Low Credibility |
| 8.0 - 10.0 | Very Low Credibility |

---

## Core Modules

### research.py - Orchestration & Research

Main orchestrator combining web research with comprehensive profiling.

**Components:**
- `MediaResearcher` - Gathers history, ownership, and external analysis via web search
- `MediaProfiler` - Orchestrates all analyzers to produce MBFC-style reports

### refactored_analyzers.py - LLM-Based Analyzers

All analyzers use LangChain's structured output for type-safe LLM responses:

**Content Analyzers:**
- `OpinionAnalyzer` - Article type classification (News/Opinion/Satire/PR)
- `EditorialBiasAnalyzer` - LLM-based political bias detection
- `PseudoscienceAnalyzer` - Science misinformation detection
- `SourcingAnalyzer` - Link extraction + LLM quality assessment
- `FactCheckSearcher` - Multi-site fact-checker search + LLM parsing

**Metadata Analyzers:**
- `TrafficLongevityAnalyzer` - Hybrid Tranco + WHOIS + LLM
- `MediaTypeAnalyzer` - Hybrid lookup + LLM classification

### scraper.py - Web Scraping Engine

Brute-force article collection with:
- Browser-like headers to avoid blocking
- Rate limiting (0.5-1.5s delays)
- Threaded parallel scraping (5 workers)
- Opinion article detection (URL, title, meta tags, schema.org)
- Metadata extraction (about page, ownership, funding, authors)

### local_detector.py - DeBERTa Inference Pipeline

Two-stage propaganda detection using fine-tuned DeBERTa models:
- Stage 1: Span Identification (Token Classification)
- Stage 2: Technique Classification (Sequence Classification)

### parser.py - MBFC Website Parser

Specialized parser for scraping Media Bias/Fact Check website to collect source URLs.

---

## Analyzer Flow Diagrams

### TrafficLongevityAnalyzer

Hybrid deterministic + LLM approach for traffic and domain age analysis.

```
analyze(domain)
    │
    ├─► 1. WHOIS Lookup (always runs)
    │       └─► Extract creation_date → Calculate age_years
    │
    ├─► 2. Tranco Lookup (O(1) dict lookup)
    │       │   Source: https://tranco-list.eu/
    │       │   - Top 1M domains ranked by popularity
    │       │   - Auto-downloads if missing
    │       │
    │       ├─► Found?
    │       │       ├─► rank < 10,000    → HIGH traffic
    │       │       ├─► rank < 100,000   → MEDIUM traffic
    │       │       ├─► rank < 1,000,000 → LOW traffic
    │       │       └─► Return with confidence=1.0, source=TRANCO
    │       │
    │       └─► Not found? → Continue to step 3
    │
    └─► 3. LLM Fallback
            ├─► Search: "{domain} traffic stats similarweb hypestat semrush"
            ├─► Combine top 5 result snippets
            └─► Parse with structured LLM output → TrafficEstimate
                    ├─► traffic_tier: HIGH/MEDIUM/LOW/MINIMAL/UNKNOWN
                    ├─► monthly_visits_estimate (if found)
                    ├─► confidence: 0.0-1.0
                    └─► reasoning

OUTPUT: TrafficData
    ├── domain, creation_date, age_years
    ├── traffic_tier, traffic_confidence
    ├── traffic_source: TRANCO | LLM | FALLBACK
    ├── tranco_rank (if available)
    └── whois_success, whois_error
```

### MediaTypeAnalyzer

Hybrid lookup table + LLM approach for media type classification.

```
analyze(url_or_domain)
    │
    ├─► 1. Known Types Lookup (O(1) dict)
    │       │   Source: known_media_types.csv
    │       │   - Pre-classified major outlets
    │       │   - Maps domain → MediaType enum
    │       │
    │       ├─► Found?
    │       │       └─► Return with confidence=1.0, source=LOOKUP
    │       │
    │       └─► Not found? → Continue to step 2
    │
    └─► 2. LLM Classification
            ├─► Search: '"{domain}" type of media outlet newspaper television website magazine'
            ├─► Fallback: "{site_name} wikipedia media company"
            └─► Parse with structured LLM output → MediaTypeLLMOutput
                    ├─► media_type: TV/NEWSPAPER/WEBSITE/MAGAZINE/RADIO/NEWS_AGENCY/BLOG/PODCAST/STREAMING/UNKNOWN
                    ├─► confidence: 0.0-1.0
                    └─► reasoning

OUTPUT: MediaTypeClassification
    ├── media_type: MediaType enum
    ├── confidence: 0.0-1.0
    ├── source: LOOKUP | LLM | FALLBACK
    └── reasoning
```

### FactCheckSearcher (40% of Factuality Score)

Multi-site search + LLM parsing for fact-check findings.

```
analyze(url_or_domain, outlet_name?)
    │
    ├─► 1. Extract Domain & Outlet Name
    │       ├─► "nytimes.com" → "New York Times"
    │       └─► Uses known_names dict or generates from domain
    │
    ├─► 2. Search 5 Fact-Checker Sites
    │       │   Sites:
    │       │   ├── mediabiasfactcheck.com
    │       │   ├── politifact.com
    │       │   ├── snopes.com
    │       │   ├── factcheck.org
    │       │   └── fullfact.org
    │       │
    │       │   Query format:
    │       │   site:{site} "{domain}" OR "{outlet_name}"
    │       │
    │       └─► Collect up to 3 results per site → Combine snippets
    │
    ├─► 3. LLM Parsing
    │       └─► Parse snippets → FactCheckLLMOutput
    │               ├─► findings: List[FactCheckFinding]
    │               │       ├── source_site (PolitiFact, Snopes, etc.)
    │               │       ├── claim_summary
    │               │       ├── verdict: TRUE/MOSTLY_TRUE/HALF_TRUE/MIXED/
    │               │       │            MOSTLY_FALSE/FALSE/PANTS_ON_FIRE/
    │               │       │            MISLEADING/UNPROVEN/NOT_RATED
    │               │       └── url (if available)
    │               ├─► failed_count (FALSE, MOSTLY_FALSE, PANTS_ON_FIRE, MISLEADING)
    │               ├─► total_count
    │               └─► confidence, reasoning
    │
    └─► 4. Score Calculation
            ├─► 0 failed checks    → 0.0 (excellent)
            ├─► 1-2 failed checks  → 2.0-4.0
            ├─► 3-5 failed checks  → 5.0-7.0
            ├─► 6+ failed checks   → 8.0-10.0 (very poor)
            └─► No data found      → 5.0 (neutral)

OUTPUT: FactCheckAnalysisResult
    ├── domain, outlet_name
    ├── failed_checks_count, total_checks_count
    ├── score: 0.0-10.0
    ├── source: SEARCH | FALLBACK
    ├── findings: List[FactCheckFinding]
    └── confidence, reasoning
```

### SourcingAnalyzer (30% of Factuality Score)

Link extraction + LLM quality assessment for source evaluation.

```
analyze(articles: List[{text}])
    │
    ├─► 1. Extract Links from All Articles
    │       └─► Regex: https?://[^\s<>"')\]]+
    │
    ├─► 2. Extract Unique Domains
    │       │   Filter out social media:
    │       │   ├── twitter.com, x.com
    │       │   ├── facebook.com, instagram.com
    │       │   ├── youtube.com, tiktok.com
    │       │   ├── linkedin.com, reddit.com
    │       │   └── t.co (Twitter short links)
    │       │
    │       └─► No domains found? → Return score=5.0 (neutral)
    │
    └─► 3. LLM Quality Assessment
            └─► Assess each domain → SourcingLLMOutput
                    ├─► sources_assessed: List[SourceAssessment]
                    │       ├── domain
                    │       ├── quality: PRIMARY/WIRE_SERVICE/MAJOR_OUTLET/
                    │       │            CREDIBLE/UNKNOWN/QUESTIONABLE
                    │       └── reasoning
                    ├─► overall_quality_score: 0.0-10.0
                    ├─► has_primary_sources: bool
                    ├─► has_wire_services: bool
                    └─► overall_assessment

Quality Tiers:
    PRIMARY       → .gov, .edu, official sources, research papers
    WIRE_SERVICE  → Reuters, AP, AFP, UPI
    MAJOR_OUTLET  → NYT, BBC, WSJ, WaPo, Guardian, CNN
    CREDIBLE      → Regional papers, trade publications
    UNKNOWN       → Unfamiliar domains
    QUESTIONABLE  → Known unreliable sources

OUTPUT: SourcingAnalysisResult
    ├── score: 0.0-10.0 (0=excellent, 10=poor)
    ├── avg_sources_per_article
    ├── total_sources_found, unique_domains
    ├── has_hyperlinks, has_primary_sources, has_wire_services
    ├── source_assessments: List[SourceAssessment]
    └── confidence, reasoning
```

### EditorialBiasAnalyzer

LLM-based comprehensive political bias detection.

```
analyze(articles: List[{title, text}], url_or_domain?, outlet_name?)
    │
    ├─► 1. Format Articles for Analysis
    │       └─► Combine title + first 2000 chars of each article
    │
    └─► 2. LLM Analysis with MBFC Methodology
            │
            │   System Prompt encodes:
            │   ├── Bias Scale: -10 (far left) to +10 (far right)
            │   ├── Policy Domain Indicators:
            │   │       ├── Economic Policy (taxes, regulation, unions)
            │   │       ├── Social Issues (abortion, LGBTQ+, guns)
            │   │       ├── Environmental Policy (climate, regulations)
            │   │       ├── Healthcare (universal vs private)
            │   │       ├── Immigration (pathways vs enforcement)
            │   │       └── Gun Rights (control vs 2A)
            │   ├── Loaded Language Detection:
            │   │       ├── LEFT: "regime", "far-right", "fascist", "climate denier"
            │   │       └── RIGHT: "radical left", "woke", "cancel culture", "fake news"
            │   └── Story Selection Bias patterns
            │
            └─► Parse → EditorialBiasLLMOutput
                    ├─► overall_bias: EXTREME_LEFT/LEFT/LEFT_CENTER/CENTER/
                    │                 RIGHT_CENTER/RIGHT/EXTREME_RIGHT
                    ├─► bias_score: -10.0 to +10.0
                    ├─► policy_positions: List[PolicyPosition]
                    │       ├── domain: ECONOMIC/SOCIAL/ENVIRONMENTAL/HEALTHCARE/
                    │       │           IMMIGRATION/FOREIGN_POLICY/GUN_RIGHTS/EDUCATION
                    │       ├── leaning: BiasDirection
                    │       ├── indicators: List[str]
                    │       └── confidence
                    ├─► uses_loaded_language: bool
                    ├─► loaded_language_examples: List[str]
                    ├─► story_selection_bias: str (optional)
                    └─► confidence, reasoning

Score to Label Mapping:
    score <= -7  → "Left"
    -7 < score <= -3 → "Left-Center"
    -3 < score <= 3  → "Center"
    3 < score <= 7   → "Right-Center"
    score > 7    → "Right"

OUTPUT: EditorialBiasResult
    ├── domain, outlet_name
    ├── overall_bias: BiasDirection
    ├── bias_score: -10.0 to +10.0
    ├── mbfc_label: "Left"/"Left-Center"/"Center"/"Right-Center"/"Right"
    ├── policy_positions, loaded_language_examples
    ├── articles_analyzed
    └── confidence, reasoning
```

### PseudoscienceAnalyzer (30% of Factuality Score)

LLM-based detection of pseudoscience and conspiracy content.

```
analyze(articles: List[{title, text}], url_or_domain?, outlet_name?)
    │
    ├─► 1. Format Articles for Analysis
    │       └─► Combine title + first 2000 chars of each article
    │
    └─► 2. LLM Analysis with Scientific Consensus
            │
            │   System Prompt includes:
            │   ├── Pseudoscience Definition
            │   │       "Claims presented as scientific but incompatible
            │   │        with scientific method - unproven, untestable,
            │   │        or contradicting scientific consensus"
            │   │
            │   ├── Categories to Detect:
            │   │   HEALTH:
            │   │   ├── Anti-Vaccination (vaccines cause autism, etc.)
            │   │   ├── Alternative Medicine (homeopathy, crystal healing)
            │   │   ├── Alternative Cancer Treatments
            │   │   ├── COVID-19 Misinformation
            │   │   └── Detoxification Claims
            │   │
            │   │   CLIMATE/ENVIRONMENTAL:
            │   │   ├── Climate Change Denialism
            │   │   ├── 5G Health Conspiracy
            │   │   ├── Chemtrails
            │   │   └── GMO Danger Claims
            │   │
            │   │   PARANORMAL:
            │   │   ├── Astrology, Psychic Claims
            │   │   └── Faith Healing
            │   │
            │   │   CONSPIRACY:
            │   │   ├── Flat Earth, Moon Landing Hoax
            │   │   └── QAnon
            │   │
            │   └── Severity Assessment:
            │           PROMOTES → Actively promotes as fact
            │           PRESENTS_UNCRITICALLY → Reports without context
            │           MIXED → Inconsistent treatment
            │           NONE_DETECTED → Respects scientific consensus
            │
            └─► Parse → PseudoscienceLLMOutput
                    ├─► indicators: List[PseudoscienceIndicator]
                    │       ├── category: PseudoscienceCategory
                    │       ├── severity: PseudoscienceSeverity
                    │       ├── evidence: str
                    │       └── scientific_consensus: str
                    ├─► promotes_pseudoscience: bool
                    ├─► overall_severity: PseudoscienceSeverity
                    ├─► science_reporting_quality: 0.0-10.0
                    ├─► respects_scientific_consensus: bool
                    └─► confidence, reasoning

OUTPUT: PseudoscienceAnalysisResult
    ├── domain, outlet_name
    ├── score: 0.0-10.0 (0=pro-science, 10=promotes pseudoscience)
    ├── promotes_pseudoscience: bool
    ├── overall_severity: PROMOTES/PRESENTS_UNCRITICALLY/MIXED/NONE_DETECTED
    ├── categories_found: List[PseudoscienceCategory]
    ├── indicators: List[PseudoscienceIndicator]
    ├── respects_scientific_consensus: bool
    ├── articles_analyzed
    └── confidence, reasoning
```

---

## Propaganda Detection

### Two-Stage DeBERTa Pipeline

```
INPUT: "The radical left wants to destroy our great nation..."

+------------------------------------------------------------------+
|  STAGE 1: SPAN IDENTIFICATION (SI)                                |
|  Model: DeBERTa-v3-Large (Token Classification, BIO tagging)      |
|  Task: Find WHERE propaganda exists                               |
|                                                                   |
|  Input:  ["The", "radical", "left", "wants", "to", ...]          |
|  Output: [ O,    B-PROP,   I-PROP,  O,      O,  ...]              |
|                  ^^^^^^^^^^^^^^                                   |
|                  Detected Span                                    |
+------------------------------------------------------------------+
                              |
                              v
+------------------------------------------------------------------+
|  STAGE 2: TECHNIQUE CLASSIFICATION (TC)                           |
|  Model: DeBERTa-v3-Large (Sequence Classification, 14 classes)    |
|  Task: Identify WHICH technique is used                           |
|                                                                   |
|  Input:  "[Context] [SEP] radical left"                           |
|  Output: "Name_Calling,Labeling" (confidence: 0.87)               |
+------------------------------------------------------------------+
```

### Propaganda Techniques (14 Classes)

| # | Technique | Description |
|---|-----------|-------------|
| 1 | Appeal_to_Authority | Citing authority figures to support claims |
| 2 | Appeal_to_fear-prejudice | Using fear or prejudice to influence |
| 3 | Bandwagon,Reductio_ad_Hitlerum | "Everyone does it" or Nazi comparisons |
| 4 | Black-and-White_Fallacy | Presenting only two choices |
| 5 | Causal_Oversimplification | Oversimplifying cause-effect |
| 6 | Doubt | Questioning credibility without evidence |
| 7 | Exaggeration,Minimisation | Overstating or understating facts |
| 8 | Flag-Waving | Appealing to patriotism/nationalism |
| 9 | Loaded_Language | Using emotionally charged words |
| 10 | Name_Calling,Labeling | Using derogatory labels |
| 11 | Repetition | Repeating messages for emphasis |
| 12 | Slogans | Using catchy phrases |
| 13 | Thought-terminating_Cliches | Phrases discouraging critical thinking |
| 14 | Whataboutism,Straw_Men,Red_Herring | Deflection tactics |

### Training Configuration

```python
Model: microsoft/deberta-v3-large (304M parameters)

SI Model (Span Identification):
  max_length = 512
  learning_rate = 1e-5
  batch_size = 4 (effective 32 with accumulation)
  epochs = 15

TC Model (Technique Classification):
  max_length = 384
  learning_rate = 1.5e-5
  batch_size = 8 (effective 64 with accumulation)
  epochs = 10

Features:
  - Focal loss for class imbalance
  - Early stopping (patience=4)
  - SemEval 2020 Task 11 dataset
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for training)
- 16GB+ RAM for inference
- 24GB+ GPU VRAM for training

### Setup

```bash
# Clone repository
git clone https://github.com/MirasBaisbay/media-profiling.git
cd media-profiling

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install torch transformers datasets beautifulsoup4 \
            requests scikit-learn duckduckgo-search openai langchain-openai pydantic

# Set API key (for LLM-based analyzers)
export OPENAI_API_KEY="your-api-key"
```

### Train Models (Optional)

```bash
# Ensure datasets/ folder contains SemEval 2020 Task 11 data
python train_pipeline.py
```

---

## Usage

### Command Line

```bash
# Run the demo profiler
python research.py
```

### Programmatic

```python
from research import MediaProfiler

# Initialize profiler
profiler = MediaProfiler()

# Sample articles (in practice, use scraper.py to collect these)
articles = [
    {
        "title": "Climate Change Policy Faces Opposition",
        "text": "The administration's new climate policy has drawn criticism..."
    },
    {
        "title": "Healthcare Reform Debate Intensifies",
        "text": "As healthcare costs continue to rise, lawmakers are divided..."
    },
]

# Profile an outlet
report = profiler.profile("https://www.bbc.com", articles)

# Generate text report
report_text = profiler.generate_report_text(report)
print(report_text)
```

### Sample Output

```
======================================================================
MEDIA BIAS/FACT CHECK REPORT: BBC
======================================================================
URL: https://www.bbc.com
Analysis Date: 2025-01-15

QUICK SUMMARY
----------------------------------------
  Bias Rating:        Left-Center
  Factuality Rating:  High
  Credibility:        High Credibility
  Media Type:         TV
  Traffic:            HIGH
  Domain Age:         28.5 years

HISTORY
----------------------------------------
  Founded: 1922
  Owner: British Broadcasting Corporation
  Funding: Public funding
  Headquarters: London, United Kingdom

BIAS ANALYSIS
----------------------------------------
  Overall Bias: Left-Center (score: -2.5)
  Uses Loaded Language: No

FACTUALITY ANALYSIS
----------------------------------------
  Factuality Rating: High (score: 1.8/10)

  Fact Check Search Results:
    Total Fact Checks Found: 3
    Failed Fact Checks: 0

  Sourcing Quality:
    Score: 2.0/10
    Has Primary Sources: Yes
    Has Wire Services: Yes

PSEUDOSCIENCE CHECK
----------------------------------------
  Promotes Pseudoscience: No
  Respects Scientific Consensus: Yes

======================================================================
Articles Analyzed: 2
Generated by Media Profiling System
======================================================================
```

---

## Project Structure

```
media-profiling/
│
├── config.py                    # Configuration constants and scoring scales
│   ├── Propaganda techniques (14 classes)
│   └── Model and training configurations
│
├── schemas.py                   # Pydantic v2 schemas for structured LLM outputs
│   ├── Article Classification (ArticleType, ArticleClassification)
│   ├── Media Type (MediaType, MediaTypeClassification)
│   ├── Traffic/Longevity (TrafficTier, TrafficData)
│   ├── Fact Check (FactCheckVerdict, FactCheckFinding, FactCheckAnalysisResult)
│   ├── Sourcing (SourceQuality, SourceAssessment, SourcingAnalysisResult)
│   ├── Editorial Bias (BiasDirection, PolicyPosition, EditorialBiasResult)
│   └── Pseudoscience (PseudoscienceCategory, PseudoscienceIndicator, PseudoscienceAnalysisResult)
│
├── refactored_analyzers.py      # LLM-based analyzers with structured output
│   ├── OpinionAnalyzer (article type classification)
│   ├── TrafficLongevityAnalyzer (hybrid Tranco + WHOIS + LLM)
│   ├── MediaTypeAnalyzer (hybrid lookup + LLM)
│   ├── FactCheckSearcher (multi-site search + LLM parsing)
│   ├── SourcingAnalyzer (link extraction + LLM quality assessment)
│   ├── EditorialBiasAnalyzer (LLM-based political bias)
│   └── PseudoscienceAnalyzer (LLM-based pseudoscience detection)
│
├── research.py                  # Web research and profiling orchestrator
│   ├── MediaResearcher (history, ownership, external analysis)
│   ├── MediaProfiler (comprehensive analysis orchestrator)
│   └── Convenience functions (research_outlet, profile_outlet)
│
├── scraper.py                   # Web scraping for articles and metadata
│   ├── MediaScraper
│   ├── Article dataclass
│   └── SiteMetadata dataclass
│
├── local_detector.py            # DeBERTa inference pipeline
│   └── LocalPropagandaDetector
│
├── train_pipeline.py            # DeBERTa fine-tuning pipeline
│
├── parser.py                    # MBFC website parser
│
├── known_media_types.csv        # Pre-classified media outlet types
│
├── tranco_top1m.csv             # Tranco top 1M domains (auto-downloaded)
│
├── 2025.csv                     # Freedom Index dataset (181 countries)
│
├── verify_*.py                  # Verification scripts for analyzers
│   ├── verify_factcheck.py
│   ├── verify_sourcing.py
│   ├── verify_editorial_bias.py
│   ├── verify_pseudoscience.py
│   ├── verify_traffic.py
│   ├── verify_media_type.py
│   └── verify_opinion.py
│
├── datasets/                    # SemEval 2020 Task 11 data
│   ├── train/
│   ├── dev/
│   └── test/
│
└── propaganda_models/           # Trained model outputs
    ├── si_model/
    └── tc_model/
```

---

## References

### Methodology
- [Media Bias/Fact Check Methodology](https://mediabiasfactcheck.com/methodology/)
- [Freedom House - Freedom in the World](https://freedomhouse.org/report/freedom-world)
- [Reporters Without Borders - Press Freedom Index](https://rsf.org/en/index)

### Propaganda Detection
- [SemEval 2020 Task 11: Detection of Propaganda Techniques](https://propaganda.qcri.org/semeval2020-task11/)
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654)

---

## License

MIT License
