"""
Evaluation Module
=================
Handles model evaluation with detailed metrics, confusion matrix,
and classification reports for the Question Quality Evaluator.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import os


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics."""

    def __init__(self, label_names=None):
        """
        Initialize evaluator.

        Args:
            label_names: Dictionary mapping label indices to names
        """
        self.label_names = label_names or {
            0: 'Low Quality',
            1: 'Medium Quality',
            2: 'High Quality'
        }
        self.results = {}

    def evaluate(self, y_true, y_pred, y_proba=None):
        """
        Compute all evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional, for AUC)

        Returns:
            dict with all metrics
        """
        self.results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_per_class': precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'recall_per_class': recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'f1_per_class': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(
                y_true, y_pred,
                target_names=list(self.label_names.values()),
                zero_division=0
            )
        }

        # ROC AUC (if probabilities available)
        if y_proba is not None:
            try:
                self.results['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                )
            except Exception as e:
                print(f"Could not compute ROC AUC: {e}")
                self.results['roc_auc_weighted'] = None

        return self.results

    def print_report(self):
        """Print a formatted evaluation report."""
        if not self.results:
            print("No results to display. Run evaluate() first.")
            return

        print("\n" + "=" * 60)
        print("          MODEL EVALUATION REPORT")
        print("=" * 60)

        print(f"\n  Accuracy:           {self.results['accuracy']:.4f}")
        print(f"  Precision (weighted): {self.results['precision_weighted']:.4f}")
        print(f"  Recall (weighted):    {self.results['recall_weighted']:.4f}")
        print(f"  F1 Score (weighted):  {self.results['f1_weighted']:.4f}")

        if self.results.get('roc_auc_weighted') is not None:
            print(f"  ROC AUC (weighted):   {self.results['roc_auc_weighted']:.4f}")

        print(f"\n{'─' * 60}")
        print("  Detailed Classification Report:")
        print(f"{'─' * 60}")
        print(self.results['classification_report'])

    def plot_confusion_matrix(self, save_path=None):
        """
        Plot and optionally save the confusion matrix.

        Args:
            save_path: File path to save the plot (optional)
        """
        if not self.results:
            print("No results to plot. Run evaluate() first.")
            return

        cm = np.array(self.results['confusion_matrix'])
        labels = list(self.label_names.values())

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)

        # Normalized (percentages)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=labels, yticklabels=labels, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_metrics_comparison(self, save_path=None):
        """
        Plot per-class metric comparison.

        Args:
            save_path: File path to save the plot (optional)
        """
        if not self.results:
            print("No results to plot. Run evaluate() first.")
            return

        labels = list(self.label_names.values())
        metrics = {
            'Precision': self.results['precision_per_class'],
            'Recall': self.results['recall_per_class'],
            'F1 Score': self.results['f1_per_class']
        }

        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (metric_name, values) in enumerate(metrics.items()):
            bars = ax.bar(x + i * width, values, width, label=metric_name)
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Quality Category', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Metrics comparison saved to {save_path}")

        plt.close()

    def save_report(self, save_path):
        """
        Save evaluation results as a text report.

        Args:
            save_path: File path for the report
        """
        if not self.results:
            print("No results to save. Run evaluate() first.")
            return

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("Question Quality Evaluator - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Accuracy:             {self.results['accuracy']:.4f}\n")
            f.write(f"Precision (weighted): {self.results['precision_weighted']:.4f}\n")
            f.write(f"Recall (weighted):    {self.results['recall_weighted']:.4f}\n")
            f.write(f"F1 Score (weighted):  {self.results['f1_weighted']:.4f}\n")
            if self.results.get('roc_auc_weighted') is not None:
                f.write(f"ROC AUC (weighted):   {self.results['roc_auc_weighted']:.4f}\n")
            f.write(f"\n{'─' * 50}\n")
            f.write("Classification Report:\n")
            f.write(f"{'─' * 50}\n")
            f.write(self.results['classification_report'])

        print(f"Report saved to {save_path}")


if __name__ == "__main__":
    # Quick test
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
    y_pred = np.array([0, 1, 2, 1, 1, 2, 0, 0, 2, 1])
    evaluator = ModelEvaluator()
    results = evaluator.evaluate(y_true, y_pred)
    evaluator.print_report()
