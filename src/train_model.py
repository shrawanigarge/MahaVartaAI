"""
Model Training and Comparison Pipeline for Marathi News Classification (10 Categories)
Combines the old synthetic dataset (10 categories) and the new Marathi dataset (data/train.csv, data/valid.csv)
into a unified reproducible dataset: data/combined_train.csv.
Applies Devanagari NLP preprocessing, TF-IDF vectorization, trains 3 candidate models,
performs internal 10-category stratified evaluation and external validation on data/valid.csv,
and exports all artifacts and reports.
"""

import sys
from pathlib import Path

# Add project root to sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import shutil
import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.preprocessing import clean_marathi_text
from src.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_category_distribution,
    save_metrics_json
)

CATEGORIES = [
    "Business",
    "Education",
    "Entertainment",
    "Health",
    "India",
    "Maharashtra",
    "Politics",
    "Sports",
    "Technology",
    "World"
]

LABEL_MAPPING = {
    "state": "Maharashtra",
    "State": "Maharashtra",
    "maharashtra": "Maharashtra",
    "entertainment": "Entertainment",
    "Entertainment": "Entertainment",
    "sports": "Sports",
    "Sports": "Sports",
    "business": "Business",
    "Business": "Business",
    "education": "Education",
    "Education": "Education",
    "health": "Health",
    "Health": "Health",
    "india": "India",
    "India": "India",
    "politics": "Politics",
    "Politics": "Politics",
    "technology": "Technology",
    "Technology": "Technology",
    "world": "World",
    "World": "World"
}

def prepare_combined_dataset():
    """
    Loads the synthetic dataset and new train/valid datasets, standardizes labels,
    deduplicates rows and headlines, eliminates train-validation leakage,
    saves data/combined_train.csv and generates reports/dataset_report.txt.
    """
    data_dir = PROJECT_ROOT / "data"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    synth_path = data_dir / "marathi_news.csv"
    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    combined_train_path = data_dir / "combined_train.csv"
    
    print("=" * 70)
    print("STEP 1: DATASET PREPARATION & QUALITY CHECKS")
    print("=" * 70)
    
    if not synth_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {synth_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"New train dataset not found at {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"New valid dataset not found at {valid_path}")
        
    df_synth_raw = pd.read_csv(synth_path, encoding="utf-8")
    df_train_raw = pd.read_csv(train_path, encoding="utf-8")
    df_valid_raw = pd.read_csv(valid_path, encoding="utf-8")
    
    synth_raw_count = len(df_synth_raw)
    train_raw_count = len(df_train_raw)
    valid_raw_count = len(df_valid_raw)
    
    print(f"1. Old synthetic dataset rows (data/marathi_news.csv) : {synth_raw_count}")
    print(f"2. New train dataset rows (data/train.csv)             : {train_raw_count}")
    print(f"3. New valid dataset rows (data/valid.csv)             : {valid_raw_count}")
    
    # 1. Standardize column names
    df_synth = df_synth_raw.copy()
    if "category" in df_synth.columns and "label" not in df_synth.columns:
        df_synth = df_synth.rename(columns={"category": "label"})
        
    df_train = df_train_raw.copy()
    if "category" in df_train.columns and "label" not in df_train.columns:
        df_train = df_train.rename(columns={"category": "label"})
        
    df_valid = df_valid_raw.copy()
    if "category" in df_valid.columns and "label" not in df_valid.columns:
        df_valid = df_valid.rename(columns={"category": "label"})
        
    # 2. Standardize category names & map labels
    df_synth["label"] = df_synth["label"].astype(str).str.strip().map(LABEL_MAPPING).fillna(df_synth["label"])
    df_train["label"] = df_train["label"].astype(str).str.strip().map(LABEL_MAPPING).fillna(df_train["label"])
    df_valid["label"] = df_valid["label"].astype(str).str.strip().map(LABEL_MAPPING).fillna(df_valid["label"])
    
    df_synth["headline"] = df_synth["headline"].astype(str).str.strip()
    df_train["headline"] = df_train["headline"].astype(str).str.strip()
    df_valid["headline"] = df_valid["headline"].astype(str).str.strip()
    
    # 3. Clean and normalize text representations for duplicate & leakage detection
    df_synth["cleaned_text"] = df_synth["headline"].apply(clean_marathi_text)
    df_train["cleaned_text"] = df_train["headline"].apply(clean_marathi_text)
    df_valid["cleaned_text"] = df_valid["headline"].apply(clean_marathi_text)
    
    # 4. Process Validation Set (Unseen external evaluation dataset)
    valid_exact_dupes = df_valid.duplicated(subset=["headline", "label"]).sum()
    valid_headline_dupes = df_valid.duplicated(subset=["headline"]).sum()
    df_valid_clean = df_valid.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    df_valid_clean = df_valid_clean[df_valid_clean["cleaned_text"].str.len() > 1].reset_index(drop=True)
    
    val_raw_headlines = set(df_valid_clean["headline"].str.strip())
    val_cleaned_texts = set(df_valid_clean["cleaned_text"].str.strip())
    
    # 5. Combine Training Datasets (Synthetic + New train)
    df_synth_clean = df_synth.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    df_train_clean = df_train.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    
    df_combined_raw = pd.concat([df_synth_clean, df_train_clean], ignore_index=True)
    total_combined_before_dedup = len(df_combined_raw)
    
    # Deduplicate within combined training
    exact_dupes_train = df_combined_raw.duplicated(subset=["headline", "label"]).sum()
    headline_dupes_train = df_combined_raw.duplicated(subset=["headline"]).sum()
    df_combined_dedup = df_combined_raw.drop_duplicates(subset=["headline"]).reset_index(drop=True)
    
    # 6. Train-Validation Leakage Prevention
    leakage_mask = (
        df_combined_dedup["headline"].str.strip().isin(val_raw_headlines) |
        df_combined_dedup["cleaned_text"].str.strip().isin(val_cleaned_texts)
    )
    leakage_count = int(leakage_mask.sum())
    
    df_combined_final = df_combined_dedup[~leakage_mask].reset_index(drop=True)
    df_combined_final = df_combined_final[df_combined_final["cleaned_text"].str.len() > 1].reset_index(drop=True)
    
    # 7. Save reproducible combined training CSV (without altering original raw files)
    df_combined_export = df_combined_final[["headline", "label"]].copy()
    df_combined_export.to_csv(combined_train_path, index=False, encoding="utf-8")
    print(f"\n--> Successfully created reproducible dataset: {combined_train_path}")
    print(f"--> Total training rows: {len(df_combined_final)}")
    
    # 8. Category Distribution
    class_distribution = df_combined_final["label"].value_counts().to_dict()
    ordered_dist = {cat: class_distribution.get(cat, 0) for cat in CATEGORIES}
    
    print("\nClass Distribution in Final Training Dataset (10 Categories):")
    for cat, cnt in ordered_dist.items():
        pct = (cnt / len(df_combined_final)) * 100
        print(f"  - {cat:15}: {cnt:5d} ({pct:5.2f}%)")
        
    # 9. Generate dataset_report.txt
    report_path = reports_dir / "dataset_report.txt"
    report_lines = [
        "=" * 70,
        "MARATHI NEWS CLASSIFICATION - DATASET PREPARATION REPORT",
        f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "1. RAW SOURCE DATASETS SUMMARY:",
        f"   - Old Synthetic Dataset (data/marathi_news.csv) : {synth_raw_count:,} rows (10 categories)",
        f"   - New Training Dataset  (data/train.csv)        : {train_raw_count:,} rows (3 raw labels: state, entertainment, sports)",
        f"   - New Validation Dataset(data/valid.csv)        : {valid_raw_count:,} rows (3 raw labels: state, entertainment, sports)",
        f"   - Total Raw Records                             : {synth_raw_count + train_raw_count + valid_raw_count:,} rows",
        "",
        "2. LABEL MAPPING & STANDARDIZATION:",
        "   - state         -> Maharashtra",
        "   - entertainment -> Entertainment",
        "   - sports        -> Sports",
        "   - Synthetic set -> Preserved across all 10 target categories",
        "",
        "3. DATA CLEANING & DEDUPLICATION:",
        f"   - Synthetic rows after cleaning/dedup           : {len(df_synth_clean):,}",
        f"   - New train.csv rows after cleaning/dedup       : {len(df_train_clean):,}",
        f"   - Validation rows after cleaning/dedup          : {len(df_valid_clean):,}",
        f"   - Combined training rows before dedup           : {total_combined_before_dedup:,}",
        f"   - Exact duplicate rows removed                  : {exact_dupes_train:,}",
        f"   - Duplicate headlines removed across sources    : {headline_dupes_train:,}",
        f"   - Train/Validation overlapping samples removed  : {leakage_count:,}",
        f"   - Zero Data Leakage Guaranteed                  : YES (Strict disjoint separation)",
        "",
        "4. FINAL TRAINING DATASET (data/combined_train.csv):",
        f"   - Total Training Samples                        : {len(df_combined_final):,}",
        f"   - Total Categories Supported                    : {len(CATEGORIES)}",
        "   - Categories: " + ", ".join(CATEGORIES),
        "",
        "5. CLASS DISTRIBUTION (TRAINING SET):",
    ]
    for cat, cnt in ordered_dist.items():
        pct = (cnt / len(df_combined_final)) * 100
        report_lines.append(f"   - {cat:15}: {cnt:5d} ({pct:5.2f}%)")
        
    report_lines.extend([
        "",
        "6. EXTERNAL VALIDATION DATASET (data/valid.csv):",
        f"   - Total Validation Samples                      : {len(df_valid_clean):,}",
        "   - Note: External valid.csv strictly evaluates the 3 mapped categories (Maharashtra, Entertainment, Sports).",
        "   - Internal stratified validation split is used for full 10-category evaluation.",
        "=" * 70
    ])
    
    report_text = "\n".join(report_lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n--> Dataset quality report saved to: {report_path}")
    
    return df_combined_final, df_valid_clean, ordered_dist

def train_and_evaluate():
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    static_img_dir = PROJECT_ROOT / "static" / "images"
    
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    static_img_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Dataset Preparation
    train_df, valid_df, cat_counts = prepare_combined_dataset()
    
    # Plot Category Distribution
    cat_dist_path = results_dir / "category_distribution.png"
    plot_category_distribution(cat_counts, cat_dist_path)
    shutil.copy(cat_dist_path, static_img_dir / "category_distribution.png")
    
    # 2. Internal Stratified Train/Test Split (80% Train, 20% Test) across all 10 categories
    print("\n" + "=" * 70)
    print("STEP 2: STRATIFIED SPLIT FOR 10-CATEGORY INTERNAL EVALUATION")
    print("=" * 70)
    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        train_df["cleaned_text"],
        train_df["label"],
        test_size=0.20,
        random_state=42,
        stratify=train_df["label"]
    )
    print(f"Internal Training Split Samples (80%) : {len(X_train_text):,}")
    print(f"Internal Test Split Samples (20%)     : {len(X_test_text):,}")
    print(f"External Validation Samples           : {len(valid_df):,}")
    
    # 3. TF-IDF Feature Extraction
    print("\n" + "=" * 70)
    print("STEP 3: TF-IDF FEATURE EXTRACTION (INDIC NLP PIPELINE)")
    print("=" * 70)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    X_valid_tfidf = vectorizer.transform(valid_df["cleaned_text"])
    
    print(f"TF-IDF Vocabulary size: {len(vectorizer.vocabulary_):,} Devanagari n-gram features")
    
    # Save vectorizer
    tfidf_path = models_dir / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, tfidf_path)
    print(f"TF-IDF Vectorizer saved to: {tfidf_path}")
    
    # 4. Candidate Machine Learning Models
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING & COMPARING 3 CANDIDATE CLASSIFIERS")
    print("=" * 70)
    
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Linear SVM": CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", random_state=42, C=1.0),
            method="sigmoid",
            cv=3
        )
    }
    
    model_evaluations = {}
    comparison_rows = []
    
    for name, model in models.items():
        print(f"\n--> Training {name} on {len(X_train_text)} samples...")
        model.fit(X_train_tfidf, y_train)
        
        # Internal 10-Class Test Evaluation
        y_test_pred = model.predict(X_test_tfidf)
        eval_internal = evaluate_model(y_test, y_test_pred, name, CATEGORIES)
        
        # External valid.csv Evaluation (on available 3 categories)
        y_valid_pred = model.predict(X_valid_tfidf)
        valid_cats = sorted(valid_df["label"].unique())
        eval_external = evaluate_model(valid_df["label"], y_valid_pred, name, valid_cats)
        
        model_evaluations[name] = {
            "model_obj": model,
            "eval_internal": eval_internal,
            "eval_external": eval_external,
            "y_test_pred": y_test_pred,
            "y_valid_pred": y_valid_pred
        }
        
        comparison_rows.append({
            "Model": name,
            "Accuracy": eval_internal["accuracy"],
            "Precision": eval_internal["weighted_precision"],
            "Recall": eval_internal["weighted_recall"],
            "F1 Score": eval_internal["weighted_f1"],
            "Macro F1": eval_internal["macro_f1"],
            "External Accuracy (3-class)": eval_external["accuracy"],
            "External Macro F1 (3-class)": eval_external["macro_f1"]
        })
        
        print(f"    [Internal 10-Class Test] Accuracy: {eval_internal['accuracy']:.4f} | Weighted F1: {eval_internal['weighted_f1']:.4f} | Macro F1: {eval_internal['macro_f1']:.4f}")
        print(f"    [External valid.csv]    Accuracy: {eval_external['accuracy']:.4f} | Macro F1: {eval_external['macro_f1']:.4f}")
        
    comparison_df = pd.DataFrame(comparison_rows)
    print("\n" + "=" * 70)
    print("STEP 5: MODEL COMPARISON TABLE (INTERNAL 10-CATEGORY TEST SPLIT)")
    print("=" * 70)
    print(comparison_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Macro F1"]].to_string(index=False))
    
    # Plot Model Comparison Chart
    comp_chart_path = results_dir / "model_comparison.png"
    plot_model_comparison(comparison_df, comp_chart_path)
    shutil.copy(comp_chart_path, static_img_dir / "model_comparison.png")
    
    # 5. Automatically Select Best Model based on Macro F1 (Accounting for class imbalance)
    best_row = comparison_df.sort_values(by="Macro F1", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    best_model_data = model_evaluations[best_model_name]
    best_eval = best_model_data["eval_internal"]
    best_external_eval = best_model_data["eval_external"]
    
    print("\n" + "=" * 70)
    print(f"STEP 6: BEST MODEL SELECTED: {best_model_name}")
    print(f"Selected based on Macro F1 ({best_row['Macro F1']:.4f}) and Weighted F1 ({best_row['F1 Score']:.4f})")
    print("=" * 70)
    
    # Train the final selected classifier on the entire combined training dataset for deployment
    print(f"\n--> Training final production {best_model_name} on ALL {len(train_df)} combined training samples...")
    X_full_tfidf = vectorizer.fit_transform(train_df["cleaned_text"])
    joblib.dump(vectorizer, tfidf_path)  # re-dump with full vocabulary
    
    if best_model_name == "Multinomial Naive Bayes":
        final_model = MultinomialNB(alpha=0.1)
    elif best_model_name == "Logistic Regression":
        final_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    else:
        final_model = CalibratedClassifierCV(
            estimator=LinearSVC(dual="auto", random_state=42, C=1.0),
            method="sigmoid",
            cv=3
        )
        
    final_model.fit(X_full_tfidf, train_df["label"])
    
    # Save Best Model
    model_save_path = models_dir / "news_classifier.pkl"
    joblib.dump(final_model, model_save_path)
    print(f"Production model saved to: {model_save_path}")
    
    # Confusion Matrix for Best Model (on 10-Class Test Split)
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(np.array(best_eval["confusion_matrix"]), CATEGORIES, best_model_name, cm_path)
    shutil.copy(cm_path, static_img_dir / "confusion_matrix.png")
    
    # 6. Save Metrics JSON & Metadata
    all_metrics = {
        "dataset_total": len(train_df) + len(valid_df),
        "train_samples": len(train_df),
        "test_samples": len(X_test_text),
        "external_valid_samples": len(valid_df),
        "categories": CATEGORIES,
        "selected_model": best_model_name,
        "selected_model_accuracy": best_eval["accuracy"],
        "selected_model_f1": best_eval["weighted_f1"],
        "selected_model_macro_f1": best_eval["macro_f1"],
        "selected_model_precision": best_eval["weighted_precision"],
        "selected_model_recall": best_eval["weighted_recall"],
        "external_valid_accuracy": best_external_eval["accuracy"],
        "external_valid_macro_f1": best_external_eval["macro_f1"],
        "comparison": comparison_rows,
        "classification_report": best_eval["classification_report"],
        "confusion_matrix": best_eval["confusion_matrix"],
        "external_classification_report": best_external_eval["classification_report"]
    }
    
    metrics_path = results_dir / "metrics.json"
    save_metrics_json(all_metrics, metrics_path)
    
    metadata = {
        "model_name": best_model_name,
        "accuracy": best_eval["accuracy"],
        "weighted_f1": best_eval["weighted_f1"],
        "macro_f1": best_eval["macro_f1"],
        "categories": CATEGORIES,
        "total_categories": len(CATEGORIES),
        "train_samples": len(train_df),
        "internal_test_samples": len(X_test_text),
        "external_valid_samples": len(valid_df),
        "vectorizer_path": "models/tfidf_vectorizer.pkl",
        "model_path": "models/news_classifier.pkl",
        "confidence_method": "Calibrated Probability (predict_proba)",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Model metadata saved to: {meta_path}")
    print("\nDetailed Classification Report for Selected Model (10-Class Test Split):")
    print(best_eval["report_text"])
    print("\nTraining and Evaluation pipeline finished successfully!")

if __name__ == "__main__":
    train_and_evaluate()

