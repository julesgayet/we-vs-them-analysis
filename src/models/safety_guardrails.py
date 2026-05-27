import re
from typing import List, Tuple


class SafetyGuardrail:
    """Detects ultra-toxic words and their obfuscated variants to override classifier output."""

    def __init__(self) -> None:
        # Regex patterns targeting variations of slurs to prevent bot bypass
        self.ultra_toxic_patterns = [
            # Variations of nigger/niggas/nighers
            re.compile(r'\bn[i1!l|*y][g98h*]{1,3}[e3aou*]r[sz5$]*\b', re.IGNORECASE),
            re.compile(r'\bn[i1!l|*y][g98h*]{1,3}[a4*]\b', re.IGNORECASE),
            re.compile(r'\bn[i1!l|*y]g[h]*[e3aou*]r[sz5$]*\b', re.IGNORECASE),
            # General obfuscation (e.g. n-i-g-g-e-r or n.i.g.g.e.r)
            re.compile(r'\bn[\s\-\.]*[i1!l|*][\s\-\.]*g[\s\-\.]*g[\s\-\.]*[e3a*][\s\-\.]*r[s_z]*\b', re.IGNORECASE)
        ]

    def is_ultra_toxic(self, text: str) -> bool:
        """Checks if the text contains any obfuscated ultra-toxic slur."""
        clean_text = str(text).strip()
        for pattern in self.ultra_toxic_patterns:
            if pattern.search(clean_text):
                return True
        return False

    def moderate_score(self, text: str, original_sentiment: str, original_toxicity: float) -> Tuple[str, float]:
        """Overrides predictions if an ultra-toxic slur is matched."""
        if self.is_ultra_toxic(text):
            return "negative", 1.0
        return original_sentiment, original_toxicity
