import json
import logging
from typing import Set, Dict
from rapidfuzz import fuzz, process
from quickstart.llm import LLMClient

logger = logging.getLogger(__name__)

class SemanticJudge:
    def __init__(self):
        self.llm = LLMClient()
        
    def check_equivalence(self, pred: str, gold: str) -> bool:
        """Uses an LLM to determine if two entities are semantically equivalent."""
        pred_clean = str(pred).strip().lower()
        gold_clean = str(gold).strip().lower()
        
        # Fast path: exact or highly fuzzy match
        if pred_clean == gold_clean:
            return True
        if fuzz.WRatio(pred_clean, gold_clean) > 85.0:
            return True
            
        prompt = f"""You are an objective evaluation judge.
Are the following two entities semantically equivalent or referring to the exact same concept/person/metric in a financial context?

Entity 1: "{pred}"
Entity 2: "{gold}"

Return ONLY a JSON object: {{"is_equivalent": true}} or {{"is_equivalent": false}}
"""
        try:
            res = self.llm.complete(prompt, temperature=0.0, response_mime_type="application/json")
            start_idx = res.find("{")
            end_idx = res.rfind("}") + 1
            data = json.loads(res[start_idx:end_idx])
            return data.get("is_equivalent", False)
        except Exception as e:
            logger.debug(f"Judge failed: {e}")
            return False

def compute_metrics(predicted: Set[str], gold: Set[str]) -> Dict[str, float]:
    """
    Computes set-based metrics using an LLM Judge to handle semantic equivalence.
    """
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
    judge = SemanticJudge()
    
    # Calculate True Positives semantically
    true_positives = 0
    matched_gold = set()
    
    for p in predicted:
        for g in gold:
            if g not in matched_gold and judge.check_equivalence(p, g):
                true_positives += 1
                matched_gold.add(g)
                break # Move to next prediction once matched
                
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(gold) if gold else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
