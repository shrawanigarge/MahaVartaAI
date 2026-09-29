"""
Test suite for Marathi News Classification (3 Categories: State, Entertainment, Sports)
Tests prediction on target categories and edge cases.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.predict import MarathiNewsPredictor

def run_tests():
    predictor = MarathiNewsPredictor()
    
    test_cases = [
        ("Entertainment", "हृतिक करतोय बायोपिक 'सुपर 30'"),
        ("Entertainment", "बॉलीवूड अभिनेता शाहरुख खानच्या नवीन चित्रपटाचे जोरदार प्रमोशन सुरू"),
        ("Sports", "सचिन तेंडुलकर आणि विराट कोहलीने केली भारतीय क्रिकेट संघाची प्रशंसा, उत्कृष्ट फलंदाजी"),
        ("Sports", "रोहित शर्माच्या नेतृत्वाखाली भारतीय संघाने जिंकला टी-20 विश्वचषक"),
        ("State", "मुंबई पुणे महामार्गावर मुसळधार पावसामुळे दरड कोसळली, वाहतूक ठप्प"),
        ("State", "मुख्यमंत्री यांनी घेतली राज्य मंत्रिमंडळाची महत्त्वाची बैठक, विविध विकासकामांना मंजुरी"),
        ("State", "विधानसभा निवडणुकीसाठी विविध राजकीय पक्षांमध्ये जोरदार प्रचार सभा सुरु")
    ]
    
    print("=" * 75)
    print("EVALUATING MODEL ON 3 TARGET CATEGORIES (State, Entertainment, Sports)")
    print("=" * 75)
    
    passed = 0
    for expected, headline in test_cases:
        res = predictor.predict(headline)
        if not res["success"]:
            print(f"[FAIL] Expected: {expected} | Error: {res.get('error')}")
            continue
            
        predicted = res["category"]
        conf = res["confidence_percent"]
        probs = res["probabilities"]
        match = (predicted.lower() == expected.lower())
        status = "[PASS]" if match else "[FAIL]"
        if match:
            passed += 1
            
        print(f"{status} Expected: {expected:13} | Predicted: {predicted:13} | Conf: {conf:7}")
        print(f"       Probabilities: {probs}")
        print(f"       Input: {headline[:50]}...")

    print("=" * 75)
    print(f"Results: {passed} / {len(test_cases)} tests evaluated.")
    print("=" * 75)
    
    # Edge Cases Testing
    print("\nTESTING EDGE CASES & VALIDATION:")
    print("-" * 50)
    
    edge_cases = [
        ("Empty Input", ""),
        ("Whitespace Only", "   "),
        ("English Only", "Breaking news: Sensex crosses 80000 points today"),
        ("Too Short", "बात")
    ]
    
    for case_name, text in edge_cases:
        res = predictor.predict(text)
        print(f"Case: {case_name:16} | Success: {res['success']:5} | Response: {res.get('error')}")

if __name__ == "__main__":
    run_tests()
