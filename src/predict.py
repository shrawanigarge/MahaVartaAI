"""
Prediction Engine and CLI Interface for Marathi News Classification
Loads trained TF-IDF vectorizer and classifier to predict categories for Marathi news headlines.
Returns dynamic intermediate NLP preprocessing stages and top TF-IDF features for live visualization.
"""

import sys
from pathlib import Path
import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.preprocessing import (
    clean_marathi_text,
    validate_marathi_input,
    normalize_unicode,
    remove_urls_and_markup,
    remove_special_characters,
    tokenize_marathi,
    MARATHI_STOPWORDS
)

class MarathiNewsPredictor:
    """Encapsulates model loading, text preprocessing, and inference."""
    
    def __init__(self, models_dir: Path = None):
        if models_dir is None:
            models_dir = PROJECT_ROOT / "models"
            
        vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
        model_path = models_dir / "news_classifier.pkl"
        
        if not vectorizer_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Model or vectorizer not found in {models_dir}. Please run 'src/train_model.py' first."
            )
            
        self.vectorizer = joblib.load(vectorizer_path)
        self.model = joblib.load(model_path)
        self.classes_ = list(self.model.classes_)
        self.feature_names_ = np.array(self.vectorizer.get_feature_names_out())
        
    def predict(self, raw_headline: str) -> dict:
        """
        Predicts category and confidence for a Marathi news headline.
        Computes exact intermediate NLP preprocessing stages for live transparency.
        
        Parameters:
            raw_headline (str): Input text in Marathi.
            
        Returns:
            dict containing:
                - original_text
                - normalized_text
                - cleaned_text
                - tokens
                - filtered_tokens
                - tfidf_features
                - category / predicted_category
                - confidence & confidence_percent
                - probabilities
        """
        is_valid, validation_msg = validate_marathi_input(raw_headline)
        if not is_valid:
            return {
                "success": False,
                "error": validation_msg,
                "headline": raw_headline,
                "original_text": raw_headline
            }
            
        # 1. Exact Preprocessing Steps
        # Step 01: Unicode Normalization (NFKC)
        normalized_text = normalize_unicode(raw_headline)
        
        # Step 02: Lowercasing + URL/markup stripping + special character / punctuation cleaning
        clean_pass1 = remove_urls_and_markup(normalized_text.lower())
        clean_pass2 = remove_special_characters(clean_pass1)
        cleaned_text = " ".join(clean_pass2.split()).strip()
        
        # Step 03: Tokenization
        tokens = tokenize_marathi(cleaned_text)
        
        # Step 04: Stopword Removal (and filtering single-character noise)
        filtered_tokens = [t for t in tokens if t not in MARATHI_STOPWORDS and len(t) > 1]
        final_cleaned_string = " ".join(filtered_tokens).strip()
        
        if not final_cleaned_string:
            return {
                "success": False,
                "error": "Headline contains no meaningful Marathi tokens after stopword removal.",
                "headline": raw_headline,
                "original_text": raw_headline,
                "normalized_text": normalized_text,
                "cleaned_text": cleaned_text,
                "tokens": tokens,
                "filtered_tokens": filtered_tokens
            }
            
        # Step 05: TF-IDF Feature Extraction
        tfidf_features_matrix = self.vectorizer.transform([final_cleaned_string])
        
        # Extract non-zero TF-IDF features with their term names and values
        coo = tfidf_features_matrix.tocoo()
        tfidf_features_list = []
        for col_idx, val in zip(coo.col, coo.data):
            term_name = str(self.feature_names_[col_idx])
            tfidf_features_list.append({
                "term": term_name,
                "value": round(float(val), 4)
            })
            
        # Sort descending by TF-IDF weight
        tfidf_features_list = sorted(tfidf_features_list, key=lambda item: item["value"], reverse=True)
        
        # Step 06: Model Prediction & Probability Calibration
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(tfidf_features_matrix)[0]
            pred_idx = int(np.argmax(probs))
            predicted_category = self.classes_[pred_idx]
            confidence_val = float(probs[pred_idx])
            confidence_type = "Calibrated Probability"
            category_probs = {cat: round(float(p), 4) for cat, p in zip(self.classes_, probs)}
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(tfidf_features_matrix)[0]
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / np.sum(exp_scores)
            pred_idx = int(np.argmax(probs))
            predicted_category = self.classes_[pred_idx]
            confidence_val = float(probs[pred_idx])
            confidence_type = "Estimated Confidence (Softmax Decision Score)"
            category_probs = {cat: round(float(p), 4) for cat, p in zip(self.classes_, probs)}
        else:
            predicted_category = self.model.predict(tfidf_features_matrix)[0]
            confidence_val = 1.0
            confidence_type = "Discrete Prediction"
            category_probs = {predicted_category: 1.0}
            
        # Sort probabilities descending
        sorted_probs = dict(sorted(category_probs.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "success": True,
            "headline": raw_headline,
            "original_text": raw_headline,
            "normalized_text": normalized_text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "filtered_tokens": filtered_tokens,
            "final_cleaned_string": final_cleaned_string,
            "tfidf_features": tfidf_features_list,
            "category": predicted_category,
            "predicted_category": predicted_category,
            "confidence": round(confidence_val, 4),
            "confidence_percent": f"{round(confidence_val * 100, 2)}%",
            "probabilities": sorted_probs,
            "confidence_type": confidence_type
        }

def run_cli():
    """Command Line Interface for real-time predictions."""
    try:
        predictor = MarathiNewsPredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        sys.exit(1)
        
    print("=" * 60)
    print("  Marathi News Classification - CLI Prediction Engine")
    print("=" * 60)
    
    # Check if headline was passed as command line argument
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        result = predictor.predict(input_text)
        if result["success"]:
            print(f"Input: {result['headline']}")
            print(f"Predicted Category: {result['category']}")
            print(f"Confidence: {result['confidence_percent']}")
            print("\nLive NLP Preprocessing Steps:")
            print(f"  1. Normalized Text : {result['normalized_text']}")
            print(f"  2. Cleaned Text    : {result['cleaned_text']}")
            print(f"  3. Tokens          : {result['tokens']}")
            print(f"  4. Filtered Tokens : {result['filtered_tokens']}")
            print(f"  5. TF-IDF Features : {result['tfidf_features']}")
        else:
            print(f"Error: {result['error']}")
        return

    # Interactive loop
    while True:
        try:
            headline = input("\nEnter Marathi news headline (or 'quit' to exit): ").strip()
            if headline.lower() in ("quit", "exit", "q"):
                print("Exiting CLI. Goodbye!")
                break
                
            if not headline:
                print("Please enter a Marathi news headline.")
                continue
                
            result = predictor.predict(headline)
            if result["success"]:
                print("-" * 50)
                print(f"Predicted Category: {result['category']}")
                print(f"Confidence: {result['confidence_percent']} ({result['confidence_type']})")
                print("-" * 50)
                print("NLP Preprocessing Pipeline:")
                print(f"  • Unicode Normalized : {result['normalized_text']}")
                print(f"  • Cleaned Text       : {result['cleaned_text']}")
                print(f"  • Tokens             : {result['tokens']}")
                print(f"  • Filtered Tokens    : {result['filtered_tokens']}")
                print(f"  • Top TF-IDF Features: {result['tfidf_features'][:5]}")
                print("-" * 50)
            else:
                print(f"Validation Warning: {result['error']}")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI. Goodbye!")
            break

if __name__ == "__main__":
    run_cli()
