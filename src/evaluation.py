"""Model evaluation, comparison table, PR curves, and confusion matrices."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve,
    confusion_matrix, classification_report,
)
from src.utils import save_figure, setup_plotting


def compute_metrics(y_true, y_pred, y_prob):
    """Compute all required classification metrics.

    Returns dict with: accuracy, precision, recall, f1, auc_pr.
    L4: Use multiple metrics — accuracy alone is misleading on imbalanced data.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred),
        "AUC-PR": average_precision_score(y_true, y_prob),
    }


def build_comparison_table(y_test, gbdt_pred, gbdt_prob, mlp_pred, mlp_prob,
                           gbdt_time, mlp_time):
    """Build side-by-side comparison table for GBDT vs MLP.

    Same test set, same preprocessing — fair comparison (L4).
    """
    gbdt_metrics = compute_metrics(y_test, gbdt_pred, gbdt_prob)
    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_prob)

    gbdt_metrics["Training Time (s)"] = f"{gbdt_time:.2f}"
    mlp_metrics["Training Time (s)"] = f"{mlp_time:.2f}"

    table = pd.DataFrame({"GBDT (XGBoost)": gbdt_metrics, "MLP (sklearn)": mlp_metrics})

    # Format numeric values
    for col in table.columns:
        for idx in table.index:
            val = table.loc[idx, col]
            if isinstance(val, float):
                table.loc[idx, col] = f"{val:.4f}"

    print("\n=== GBDT vs MLP Comparison ===")
    print(table.to_string())
    return table


def plot_confusion_matrices(y_test, gbdt_pred, mlp_pred):
    """Plot side-by-side confusion matrices for both models."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, preds, title in [
        (axes[0], gbdt_pred, "GBDT (XGBoost)"),
        (axes[1], mlp_pred, "MLP (sklearn)"),
    ]:
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["No (0)", "Yes (1)"],
            yticklabels=["No (0)", "Yes (1)"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix: {title}")

    plt.tight_layout()
    save_figure(fig, "confusion_matrices.png")
    plt.show()
    return fig


def plot_pr_curves(y_test, gbdt_prob, mlp_prob):
    """Plot overlaid Precision-Recall curves for both models.

    AUC-PR is the right metric for imbalanced classification (L4).
    """
    setup_plotting()
    fig, ax = plt.subplots()

    for probs, label, color in [
        (gbdt_prob, "GBDT", "steelblue"),
        (mlp_prob, "MLP", "coral"),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, probs)
        auc_pr = average_precision_score(y_test, probs)
        ax.plot(recall, precision, label=f"{label} (AUC-PR={auc_pr:.3f})",
                linewidth=2, color=color)

    baseline = y_test.mean()
    ax.axhline(y=baseline, color="gray", linestyle="--", label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves: GBDT vs MLP")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save_figure(fig, "pr_curves.png")
    plt.show()
    return fig


def print_classification_reports(y_test, gbdt_pred, mlp_pred):
    """Print full classification reports for both models."""
    print("\n=== GBDT Classification Report ===")
    print(classification_report(y_test, gbdt_pred, target_names=["No", "Yes"]))
    print("\n=== MLP Classification Report ===")
    print(classification_report(y_test, mlp_pred, target_names=["No", "Yes"]))
