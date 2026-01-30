"""
local_detector.py
Inference pipeline for local DeBERTa models.
"""

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification,
    pipeline
)
from config import PROPAGANDA_TECHNIQUES

class LocalPropagandaDetector:
    def __init__(self, si_model_path="propaganda_models/si_model", tc_model_path="propaganda_models/tc_model"):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Loading Local Propaganda Models on device {self.device}...")
        
        try:
            # 1. Load Span Identification (SI)
            self.si_tokenizer = AutoTokenizer.from_pretrained(si_model_path)
            self.si_model = AutoModelForTokenClassification.from_pretrained(si_model_path)
            self.si_pipe = pipeline(
                "token-classification", 
                model=self.si_model, 
                tokenizer=self.si_tokenizer, 
                aggregation_strategy="simple",
                device=self.device
            )
            
            # 2. Load Technique Classification (TC)
            self.tc_tokenizer = AutoTokenizer.from_pretrained(tc_model_path)
            self.tc_model = AutoModelForSequenceClassification.from_pretrained(tc_model_path)
            self.tc_pipe = pipeline(
                "text-classification", 
                model=self.tc_model, 
                tokenizer=self.tc_tokenizer, 
                device=self.device,
                top_k=1
            )
            self.ready = True
        except Exception as e:
            print(f"❌ Could not load local models: {e}")
            print("Please run train_pipeline.py first.")
            self.ready = False

    def detect(self, text):
        if not self.ready:
            return []
            
        # Step 1: Identify Spans (Where is the propaganda?)
        # Returns list of dicts: {'entity_group': 'PROP', 'score': 0.9, 'word': 'fake news', 'start': 10, 'end': 19}
        si_results = self.si_pipe(text)
        
        findings = []
        
        # Step 2: Classify Techniques (What type is it?)
        for span in si_results:
            if span['entity_group'] == 'LABEL_0': # Ignore 'O' tag if mapped incorrectly
                continue
                
            snippet = text[span['start']:span['end']]
            
            # Get Context (Sentence + Surroundings)
            # Simple heuristic: grab 50 chars before and after
            start_ctx = max(0, span['start'] - 100)
            end_ctx = min(len(text), span['end'] + 100)
            context = text[start_ctx:end_ctx]
            
            # Run TC Model on the snippet + context
            # Input format: "[CLS] Context [SEP] Snippet [SEP]" works best for DeBERTa
            input_text = f"{context} [SEP] {snippet}"
            tc_result = self.tc_pipe(input_text)[0][0] # Get top result
            
            findings.append({
                "technique": tc_result['label'],
                "text_snippet": snippet,
                "context": context.strip(),
                "confidence": float(round(tc_result['score'], 2))
            })
            
        return findings