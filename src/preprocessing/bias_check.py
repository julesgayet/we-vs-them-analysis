import os
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd


class BiasAnalyzer:
    """Analyzes model predictions across platforms to calculate fairness metrics like FPR gaps and Equalized Odds."""

    def __init__(self, data_dir: str = "data/scored", output_dir: str = "data/processed") -> None:
        self.data_dir = data_dir
        self.output_dir = output_dir

    def load_scored_data(self) -> pd.DataFrame:
        """Loads and combines all scored datasets, appending platform column."""
        if not os.path.exists(self.data_dir):
            print(f"Directory not found: {self.data_dir}")
            return pd.DataFrame()

        dfs: List[pd.DataFrame] = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if not file.endswith(".csv"):
                    continue
                self._load_single_file(os.path.join(root, file), dfs)

        if not dfs:
            return pd.DataFrame()

        return pd.concat(dfs, ignore_index=True)

    def _load_single_file(self, filepath: str, dfs: List[pd.DataFrame]) -> None:
        """Helper to load csv and assign platform based on folder structure."""
        try:
            temp_df = pd.read_csv(filepath, low_memory=False)
            required_cols = {'clean_text', 'is_polarized', 'toxicity', 'sentiment'}
            if required_cols.issubset(temp_df.columns):
                rel_path = os.path.relpath(filepath, self.data_dir)
                path_parts = os.path.normpath(rel_path).split(os.sep)
                platform = path_parts[0].capitalize() if len(path_parts) > 1 else "General"
                temp_df['platform'] = platform
                dfs.append(temp_df)
        except Exception as e:
            print(f"Skipping file {filepath} due to error: {e}")

    def compute_fairness_metrics(self, df: pd.DataFrame, toxicity_threshold: float = 0.5) -> Dict[str, Any]:
        """Computes TPR, FPR, FPR gaps, and Equalized Odds (EO) disparities across platforms."""
        # Use polarization (is_polarized) as the target proxy, and toxicity > threshold as prediction
        df['y_true'] = df['is_polarized'].astype(int)
        df['y_pred'] = (df['toxicity'] > toxicity_threshold).astype(int)

        platforms = df['platform'].unique()
        platform_metrics = {}

        for plat in platforms:
            plat_df = df[df['platform'] == plat]
            tp = ((plat_df['y_true'] == 1) & (plat_df['y_pred'] == 1)).sum()
            fp = ((plat_df['y_true'] == 0) & (plat_df['y_pred'] == 1)).sum()
            tn = ((plat_df['y_true'] == 0) & (plat_df['y_pred'] == 0)).sum()
            fn = ((plat_df['y_true'] == 1) & (plat_df['y_pred'] == 0)).sum()

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            platform_metrics[plat] = {
                "tpr": tpr,
                "fpr": fpr,
                "total_samples": len(plat_df)
            }

        # Calculate gaps
        tprs = [m['tpr'] for m in platform_metrics.values()]
        fprs = [m['fpr'] for m in platform_metrics.values()]

        fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
        tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
        eo_disparity = fpr_gap + tpr_gap

        return {
            "platform_metrics": platform_metrics,
            "fpr_gap": fpr_gap,
            "tpr_gap": tpr_gap,
            "eo_disparity": eo_disparity
        }

    def save_visualization(
        self,
        platform_stats: pd.Series,
        polarized_stats: pd.Series,
        output_path: str
    ) -> None:
        """Generates and saves matplotlib plots for statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Distribution of Platform
        platform_stats.plot(kind='bar', color='skyblue', ax=ax1)
        ax1.set_title("Distribution of Data by Platform")
        ax1.set_ylabel("Percentage of Dataset (%)")
        ax1.tick_params(axis='x', rotation=45)

        # Plot 2: Polarization by Platform
        polarized_stats.plot(kind='bar', color='salmon', ax=ax2)
        ax2.set_title("Percentage of 'We vs Them' Content")
        ax2.set_ylabel("% Polarized")
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"\n✅ Saved new visualization to {output_path}")
        plt.close()

    def generate_fairness_report(self, df: pd.DataFrame, fairness: Dict[str, Any], output_path: str) -> None:
        """Generates and saves a structured fairness report file with mitigation suggestions."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write("            MODEL BIAS & FAIRNESS REPORT          \n")
            f.write("==================================================\n\n")
            f.write(f"Total scored comments analyzed: {len(df)}\n\n")
            
            f.write("--- Platform Fairness Metrics (Proxy: Polarization as Target) ---\n")
            for plat, metrics in fairness['platform_metrics'].items():
                f.write(f"Platform: {plat}\n")
                f.write(f"  - Total Samples: {metrics['total_samples']}\n")
                f.write(f"  - True Positive Rate (TPR): {metrics['tpr']:.4f}\n")
                f.write(f"  - False Positive Rate (FPR): {metrics['fpr']:.4f}\n\n")
            
            f.write("--- Disparities ---\n")
            f.write(f"  - False Positive Rate (FPR) Gap: {fairness['fpr_gap']:.4f}\n")
            f.write(f"  - True Positive Rate (TPR) Gap: {fairness['tpr_gap']:.4f}\n")
            f.write(f"  - Equalized Odds (EO) Disparity: {fairness['eo_disparity']:.4f}\n\n")
            
            f.write("--- Bias Mitigation Guidelines ---\n")
            f.write("1. Platform-Specific Classification Thresholds:\n")
            f.write("   Adjust toxicity classification thresholds per platform to calibrate and equalize FPRs.\n")
            f.write("2. Targeted Data Augmentation:\n")
            f.write("   Collect more training examples from under-represented platforms (e.g. TikTok) to align model representations.\n")
            f.write("3. Dialect/Slang Alignment:\n")
            f.write("   Fine-tune toxicity models on domain-specific social media text to decrease false positives caused by benign in-group slang.\n")
            f.write("4. Regular Audits:\n")
            f.write("   Run continuous fairness pipelines on newly collected samples to monitor drift in EO disparity.\n")

        print(f"✅ Exported structured fairness report to {output_path}")

    def run_bias_check(self) -> None:
        """Executes the complete bias analysis pipeline."""
        df = self.load_scored_data()
        if df.empty:
            print("Error: No scored data found. Make sure linguistic analysis and scoring are complete.")
            return

        # Simple distribution metrics
        platform_counts = df['platform'].value_counts(normalize=True) * 100
        polarized_stats = df.groupby('platform')['is_polarized'].mean() * 100

        # Calculate advanced fairness
        fairness = self.compute_fairness_metrics(df)

        print("\n--- Platform Distribution (%) ---")
        print(platform_counts.to_string())
        print("\n--- Polarization Rate (%) ---")
        print(polarized_stats.to_string())
        print(f"\n--- FPR Gap: {fairness['fpr_gap']:.4f} | Equalized Odds Disparity: {fairness['eo_disparity']:.4f} ---")

        # Save visualization and report
        self.save_visualization(platform_counts, polarized_stats, os.path.join(self.output_dir, "platform_distribution.png"))
        self.generate_fairness_report(df, fairness, os.path.join(self.output_dir, "fairness_report.txt"))


if __name__ == "__main__":
    analyzer = BiasAnalyzer()
    analyzer.run_bias_check()