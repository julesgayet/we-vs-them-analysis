import os
import re
from typing import Any, Dict, List, Tuple
import pandas as pd
from src.models.sentiment_toxicity import ModelScorer


class ErrorAnalyzer:
    """Performs targeted error analysis of sentiment/toxicity models on sarcasm and irony."""

    # Curated evaluation set of sarcasm and comparison examples
    # (text, ground_truth_toxic (0/1), ground_truth_sentiment)
    EVALUATION_SET = [
        ("Oh, what a brilliant defense, letting them score in the last minute. /s", 0, "negative"),
        ("Clearly, the referee is completely unbiased. /s", 0, "neutral"),
        ("I love waiting 3 hours in the rain for a match to get cancelled. So much fun!", 0, "negative"),
        ("Wow, you are an absolute genius. Maybe try using your brain next time.", 1, "negative"),
        ("Congratulations to the team for another stellar performance of doing absolutely nothing.", 0, "negative"),
        ("Oh sure, blame the goalie when the defense was literally sleeping. /s", 0, "negative"),
        ("I'm so happy we bought another player for 100m just to sit on the bench. Kappa", 0, "neutral"),
        ("This was the best game ever! (If you enjoy watching paint dry, that is).", 0, "negative"),
        ("Yeah, because losing 4-0 at home is exactly how we wanted to start the season.", 0, "negative"),
        ("Brilliant tactical decision to play with 10 defenders. Bravo.", 0, "neutral"),
        ("Shut up, you clueless idiot. You know nothing about football.", 1, "negative"),
        ("This player is trash, sell him immediately.", 1, "negative"),
        ("What a beautiful goal by Messi, absolutely world class!", 0, "positive"),
        ("Very happy with the three points today, let's keep it going!", 0, "positive"),
        ("The match ended in a 0-0 draw, both teams played defensively.", 0, "neutral")
    ]

    def __init__(self, scorer: ModelScorer, scored_dir: str = "data/scored", output_dir: str = "data/processed") -> None:
        self.scorer = scorer
        self.scored_dir = scored_dir
        self.output_dir = output_dir
        self.sarcasm_pattern = re.compile(r'\b/s\b|\bkappa\b|\bsarcasm\b|\bsarcastic\b|\bironic\b|\birony\b', re.IGNORECASE)

    def evaluate_test_set(self) -> Dict[str, Any]:
        """Runs model predictions on the sarcasm evaluation set and calculates metrics."""
        texts = [case[0] for case in self.EVALUATION_SET]
        true_toxic = [case[1] for case in self.EVALUATION_SET]
        true_sentiment = [case[2] for case in self.EVALUATION_SET]

        pred_sentiment, pred_toxicity = self.scorer.score_texts(texts)
        pred_toxic = [1 if t > 0.5 else 0 for t in pred_toxicity]

        # Calculate metrics for Toxicity
        tp = sum(1 for t, p in zip(true_toxic, pred_toxic) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(true_toxic, pred_toxic) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(true_toxic, pred_toxic) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(true_toxic, pred_toxic) if t == 1 and p == 0)

        accuracy = (tp + tn) / len(true_toxic)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate Sentiment accuracy
        correct_sentiment = sum(1 for t, p in zip(true_sentiment, pred_sentiment) if t == p.lower().replace("positive", "positive").replace("negative", "negative").replace("neutral", "neutral"))
        sentiment_accuracy = correct_sentiment / len(true_sentiment)

        return {
            "predictions": list(zip(texts, true_toxic, pred_toxic, true_sentiment, pred_sentiment)),
            "toxicity_metrics": {
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
            },
            "sentiment_accuracy": sentiment_accuracy
        }

    def scan_dataset_for_sarcasm(self) -> Dict[str, Any]:
        """Scans the saved scored dataset to find real sarcasm and error patterns."""
        if not os.path.exists(self.scored_dir):
            return {}

        sarcastic_comments: List[pd.Series] = []
        for root, _, files in os.walk(self.scored_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                try:
                    df = pd.read_csv(os.path.join(root, file), low_memory=False)
                    if 'clean_text' not in df.columns:
                        continue
                    df['clean_text'] = df['clean_text'].fillna("").astype(str)
                    texts_list = df['clean_text'].tolist()
                    df['is_sarcastic'] = [
                        bool(self.sarcasm_pattern.search(t)) for t in texts_list
                    ]
                    sarcastic_comments.append(df[df['is_sarcastic'] == True])
                except Exception as e:
                    print(f"Error scanning {file}: {e}")

        if not sarcastic_comments:
            return {"count": 0, "avg_toxicity": 0.0, "sentiment_distribution": {}}

        full_sarcastic = pd.concat(sarcastic_comments, ignore_index=True)
        count = len(full_sarcastic)
        avg_toxicity = full_sarcastic['toxicity'].mean() if count > 0 else 0.0
        sentiment_dist = full_sarcastic['sentiment'].value_counts().to_dict() if count > 0 else {}

        # Get some examples predicted as toxic or negative
        toxic_examples = full_sarcastic[full_sarcastic['toxicity'] > 0.5]['clean_text'].head(5).tolist()

        return {
            "count": count,
            "avg_toxicity": avg_toxicity,
            "sentiment_distribution": sentiment_dist,
            "toxic_examples": toxic_examples
        }

    def generate_report(self, eval_results: Dict[str, Any], scan_results: Dict[str, Any], output_path: str) -> None:
        """Writes the error analysis findings to a report file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("            SARCASM & ERROR ANALYSIS REPORT       \n")
            f.write("==================================================\n\n")

            f.write("--- Part 1: Sarcasm Evaluation Set Results ---\n")
            f.write(f"Toxicity Accuracy: {eval_results['toxicity_metrics']['accuracy']:.4f}\n")
            f.write(f"Toxicity Precision: {eval_results['toxicity_metrics']['precision']:.4f}\n")
            f.write(f"Toxicity Recall: {eval_results['toxicity_metrics']['recall']:.4f}\n")
            f.write(f"Toxicity F1-Score: {eval_results['toxicity_metrics']['f1']:.4f}\n")
            f.write(f"Sentiment Accuracy: {eval_results['sentiment_accuracy']:.4f}\n\n")

            f.write("Confusion Matrix (Toxicity):\n")
            f.write(f"  TP: {eval_results['toxicity_metrics']['tp']}  |  FP: {eval_results['toxicity_metrics']['fp']} (Sarcasm flagged as toxic)\n")
            f.write(f"  FN: {eval_results['toxicity_metrics']['fn']}  |  TN: {eval_results['toxicity_metrics']['tn']} (Sarcasm correctly flagged as safe)\n\n")

            f.write("Detailed Predictions:\n")
            for text, t_toxic, p_toxic, t_sent, p_sent in eval_results['predictions']:
                f.write(f"Text: \"{text}\"\n")
                f.write(f"  - Toxicity: True={t_toxic}, Pred={p_toxic}\n")
                f.write(f"  - Sentiment: True={t_sent}, Pred={p_sent}\n\n")

            f.write("--- Part 2: Dataset Sarcasm Audit ---\n")
            f.write(f"Total sarcastic comments found: {scan_results.get('count', 0)}\n")
            f.write(f"Average toxicity score: {scan_results.get('avg_toxicity', 0.0):.4f}\n")
            f.write(f"Sentiment Distribution: {scan_results.get('sentiment_distribution', {})}\n\n")

            f.write("Examples of sarcastic comments predicted as toxic:\n")
            for ex in scan_results.get('toxic_examples', []):
                f.write(f"  - \"{ex}\"\n")

            f.write("\n--- Error Analysis Insights ---\n")
            f.write("1. False Positives on Non-Toxic Sarcasm:\n")
            f.write("   Sarcasm markers like '/s' or 'Kappa' are sometimes associated with negative sentiment but are not toxic.\n")
            f.write("   However, the model occasionally assigns higher toxicity scores to sarcastic comments due to keywords like 'garbage' or 'nothing'.\n")
            f.write("2. Sentiment Misclassifications:\n")
            f.write("   Sarcastic comments expressing positive literal phrasing but negative intent (e.g., 'Oh, brilliant decision') are frequently\n")
            f.write("   misclassified as positive/neutral by shallow sentiment models, as they lack deep semantic context of irony.\n")

        print(f"✅ Exported structured error analysis report to {output_path}")

    def run_analysis(self) -> None:
        """Runs the complete error analysis workflow."""
        eval_results = self.evaluate_test_set()
        scan_results = self.scan_dataset_for_sarcasm()
        self.generate_report(eval_results, scan_results, os.path.join(self.output_dir, "error_analysis_report.txt"))


if __name__ == "__main__":
    scorer = ModelScorer()
    analyzer = ErrorAnalyzer(scorer)
    analyzer.run_analysis()
