"""
Model Evaluation and Visualization Module
Generates performance metrics, confusion matrices, comparison charts, and category distributions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for server/headless generation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Professional visual styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = ["#2563EB", "#7C3AED", "#059669", "#D97706", "#DC2626", "#0891B2", "#4F46E5", "#DB2777", "#475569", "#0D9488"]

def evaluate_model(y_true, y_pred, model_name: str, categories: List[str]) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics for a classifier across specified categories."""
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, labels=categories, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(y_true, y_pred, labels=categories, average="weighted", zero_division=0)
    
    report_dict = classification_report(y_true, y_pred, labels=categories, target_names=categories, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, labels=categories, target_names=categories, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    
    return {
        "model_name": model_name,
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weight),
        "weighted_recall": float(r_weight),
        "weighted_f1": float(f1_weight),
        "classification_report": report_dict,
        "report_text": report_str,
        "confusion_matrix": cm.tolist()
    }

def plot_confusion_matrix(cm_data: np.ndarray, categories: List[str], model_name: str, save_path: Path):
    """Plots and saves a high-resolution confusion matrix heatmap for all categories."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=300)
    
    sns.heatmap(
        cm_data,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Number of Samples"},
        linewidths=0.5
    )
    plt.title(f"Confusion Matrix - {model_name}\n({len(categories)} News Categories)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Predicted Category", fontsize=12, labelpad=10, fontweight="bold")
    plt.ylabel("Actual Category", fontsize=12, labelpad=10, fontweight="bold")
    plt.xticks(rotation=40, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_model_comparison(comparison_df, save_path: Path):
    """
    Plots a multi-bar chart comparing Accuracy, Precision, Recall, F1-score, and Macro F1 across all evaluated models.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]
    
    plt.figure(figsize=(11, 6), dpi=300)
    x = np.arange(len(comparison_df))
    width = 0.16
    
    for i, metric in enumerate(metrics):
        if metric in comparison_df.columns:
            values = comparison_df[metric].values
            plt.bar(x + (i - 2.0) * width, values, width, label=metric, color=PALETTE[i % len(PALETTE)])
            for idx, v in enumerate(values):
                plt.text(x[idx] + (i - 2.0) * width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, rotation=0)

    plt.title("NLP Classifier Model Performance Comparison (10 Categories)", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("Machine Learning Model", fontsize=12, labelpad=10, fontweight="bold")
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12, labelpad=10, fontweight="bold")
    plt.xticks(x, comparison_df["Model"], fontsize=11, fontweight="bold")
    plt.ylim(0, 1.15)
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_category_distribution(category_counts: Dict[str, int], save_path: Path):
    """Plots and saves the category distribution bar chart for all 10 categories."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5.5), dpi=300)
    
    cats = list(category_counts.keys())
    counts = list(category_counts.values())
    
    bars = plt.bar(cats, counts, color="#2563EB", edgecolor="#1E40AF", width=0.55)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                 f"{int(height)}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
                 
    plt.title("Category Sample Distribution in Combined Marathi News Dataset", fontsize=14, pad=15, fontweight="bold")
    plt.xlabel("News Category", fontsize=12, labelpad=10, fontweight="bold")
    plt.ylabel("Number of Headlines", fontsize=12, labelpad=10, fontweight="bold")
    plt.xticks(rotation=35, ha="right", fontsize=10)
    plt.ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def save_metrics_json(metrics_data: Dict[str, Any], save_path: Path):
    """Saves metrics dictionary to a JSON file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4)
