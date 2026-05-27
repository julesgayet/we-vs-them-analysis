import os
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from transformers import pipeline
from lime.lime_text import LimeTextExplainer


class ModelExplainer:
    """Computes token-level explanations for toxicity models using LIME or perturbation-based SHAP."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        self.device = self._determine_device(device)
        self._toxicity_pipeline: Optional[Any] = None

    def _determine_device(self, device: Optional[torch.device]) -> int:
        """Determines device index for HF pipelines (mps or cuda or -1 for cpu)."""
        if device is not None:
            if device.type == 'cuda':
                return 0
            if device.type == 'mps':
                return -1 # CPU is safer for pipeline explanations on Mac MPS
        if torch.backends.mps.is_available():
            return -1
        if torch.cuda.is_available():
            return 0
        return -1

    def initialize_pipeline(self) -> None:
        """Initializes the classification pipeline if not already loaded."""
        if self._toxicity_pipeline is not None:
            return

        print("Initializing classification pipeline for XAI...")
        model_name = "unitary/toxic-bert"
        self._toxicity_pipeline = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=self.device,
            return_all_scores=True
        )

    def _extract_toxic_score(self, pipeline_out: Any) -> float:
        """Helper to extract toxic label score from list of dicts or single dict."""
        if isinstance(pipeline_out, dict):
            label = pipeline_out.get('label', '')
            score = pipeline_out.get('score', 0.0)
            if label == 'toxic':
                return score
            return 1.0 - score if label in ['non-toxic', 'safe', 'neutral'] else 0.0

        if isinstance(pipeline_out, list):
            for item in pipeline_out:
                if isinstance(item, dict) and item.get('label') == 'toxic':
                    return item.get('score', 0.0)
            if pipeline_out and isinstance(pipeline_out[0], dict):
                return pipeline_out[0].get('score', 0.0)

        return 0.0

    def predict_probabilities(self, texts: List[str]) -> np.ndarray:
        """Predicts probabilities for a list of texts (returning [non-toxic, toxic] shape)."""
        self.initialize_pipeline()
        assert self._toxicity_pipeline is not None
        
        results = self._toxicity_pipeline(texts, truncation=True, max_length=128)
        
        from models.safety_guardrails import SafetyGuardrail
        guardrail = SafetyGuardrail()
        
        probs = []
        for text, out in zip(texts, results):
            toxic_score = self._extract_toxic_score(out)
            if guardrail.is_ultra_toxic(text):
                toxic_score = 1.0
            probs.append([1.0 - toxic_score, toxic_score])
        return np.array(probs)

    def explain_lime(self, text: str, num_features: int = 8) -> List[Tuple[str, float]]:
        """Computes word contributions using LIME."""
        if not text.strip():
            return []

        explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])
        exp = explainer.explain_instance(
            text, 
            self.predict_probabilities, 
            num_features=num_features,
            labels=(1,)
        )
        return exp.as_list(label=1)

    def explain_shap_perturbation(self, text: str) -> List[Tuple[str, float]]:
        """Computes word contributions using a fast perturbation-based SHAP (leave-one-out)."""
        if not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        # Base prediction
        base_prob = self.predict_probabilities([text])[0][1]

        contributions: List[Tuple[str, float]] = []
        for i in range(len(words)):
            perturbed_words = words[:i] + words[i+1:]
            perturbed_text = " ".join(perturbed_words)
            if not perturbed_text.strip():
                marginal_contribution = base_prob
            else:
                perturbed_prob = self.predict_probabilities([perturbed_text])[0][1]
                marginal_contribution = base_prob - perturbed_prob
            
            contributions.append((words[i], float(marginal_contribution)))

        return contributions

    def generate_heatmap_html(self, text: str, word_weights: List[Tuple[str, float]]) -> str:
        """Generates an HTML string with highlighted words based on contribution weights."""
        if not text.strip():
            return "<p>No text to explain.</p>"

        # Map weights for quick lookup
        weight_map = {str(word).lower().strip(",.!?\"'()"): weight for word, weight in word_weights}
        max_weight = max([abs(w) for w in weight_map.values()] + [1e-5])

        words = text.split()
        html_spans = []

        for word in words:
            clean_word = word.lower().strip(",.!?\"'() ")
            weight = weight_map.get(clean_word, 0.0)
            
            if abs(weight) > 1e-4:
                opacity = min(abs(weight) / max_weight, 1.0) * 0.8
                opacity = max(opacity, 0.15) # Keep a minimum visible opacity
                
                if weight > 0:
                    bg_color = f"rgba(239, 68, 68, {opacity:.2f})"
                    text_color = "#991b1b" if opacity > 0.3 else "#000000"
                else:
                    bg_color = f"rgba(59, 130, 246, {opacity:.2f})"
                    text_color = "#1e40af" if opacity > 0.3 else "#000000"
            else:
                bg_color = "transparent"
                text_color = "#000000"

            span = f'<span style="background-color: {bg_color}; color: {text_color}; padding: 2px 4px; margin: 2px; border-radius: 4px; display: inline-block;">{word}</span>'
            html_spans.append(span)

        legend_html = """
        <div style="margin-bottom: 15px; font-family: sans-serif; font-size: 13px; display: flex; gap: 15px;">
            <div><span style="background-color: rgba(239, 68, 68, 0.6); padding: 2px 6px; border-radius: 4px; color: white;">■</span> Toxic Trigger</div>
            <div><span style="background-color: rgba(59, 130, 246, 0.6); padding: 2px 6px; border-radius: 4px; color: white;">■</span> Mitigating/Safe word</div>
        </div>
        """

        container_style = "border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; background-color: #f8fafc; font-family: sans-serif; line-height: 1.6;"
        return f'<div style="{container_style}">{legend_html}{" ".join(html_spans)}</div>'
