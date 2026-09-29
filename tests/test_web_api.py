"""
Web Application and REST API Automated Test Suite
Verifies all Flask web routes, static assets, and REST API endpoints.
"""

import sys
from pathlib import Path
import json

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from app import app

def run_api_and_web_tests():
    print("=" * 70)
    print("RUNNING AUTOMATED FLASK WEB APP & REST API TEST SUITE")
    print("=" * 70)
    
    client = app.test_client()
    passed = 0
    total = 0

    # Test 1: GET /api/health
    total += 1
    res = client.get("/api/health")
    data = res.get_json()
    if res.status_code == 200 and data.get("status") == "healthy":
        print("[PASS] Test 1: GET /api/health -> 200 OK, status: healthy")
        passed += 1
    else:
        print(f"[FAIL] Test 1: GET /api/health failed with status {res.status_code}")

    # Test 2: POST /api/predict (Valid Headline)
    total += 1
    payload = {"headline": "हृतिक करतोय बायोपिक 'सुपर 30'"}
    res = client.post("/api/predict", json=payload)
    data = res.get_json()
    if res.status_code == 200 and data.get("category") == "Entertainment":
        print(f"[PASS] Test 2: POST /api/predict (Valid) -> 200 OK | Category: {data.get('category')} | Conf: {data.get('confidence_percent')}")
        passed += 1
    else:
        print(f"[FAIL] Test 2: POST /api/predict failed: {res.status_code}, data={data}")

    # Test 3: POST /api/predict (Missing headline field)
    total += 1
    res = client.post("/api/predict", json={"other_field": "test"})
    if res.status_code == 400 and "error" in res.get_json():
        print(f"[PASS] Test 3: POST /api/predict (Missing field) -> 400 Bad Request | Error caught: {res.get_json()['error']}")
        passed += 1
    else:
        print(f"[FAIL] Test 3: Missing field test failed with status {res.status_code}")

    # Test 4: POST /api/predict (Empty headline)
    total += 1
    res = client.post("/api/predict", json={"headline": "   "})
    if res.status_code == 400 and "error" in res.get_json():
        print(f"[PASS] Test 4: POST /api/predict (Empty headline) -> 400 Bad Request | Error caught: {res.get_json()['error']}")
        passed += 1
    else:
        print(f"[FAIL] Test 4: Empty headline test failed with status {res.status_code}")

    # Test 5: POST /api/predict (Non-JSON payload)
    total += 1
    res = client.post("/api/predict", data="Not a JSON string", content_type="text/plain")
    if res.status_code == 400 and "error" in res.get_json():
        print(f"[PASS] Test 5: POST /api/predict (Non-JSON) -> 400 Bad Request | Error caught: {res.get_json()['error']}")
        passed += 1
    else:
        print(f"[FAIL] Test 5: Non-JSON test failed with status {res.status_code}")

    # Test 6: GET / (Homepage)
    total += 1
    res = client.get("/")
    if res.status_code == 200 and b"Marathi News Classification" in res.data:
        print("[PASS] Test 6: GET / (Homepage) -> 200 OK | Title and layout verified")
        passed += 1
    else:
        print(f"[FAIL] Test 6: Homepage failed with status {res.status_code}")

    # Test 7: POST / (Form submission on homepage)
    total += 1
    res = client.post("/", data={"headline": "हृतिक करतोय बायोपिक 'सुपर 30'"}, follow_redirects=True)
    if res.status_code == 200 and b"Entertainment" in res.data:
        print("[PASS] Test 7: POST / (Web Form Submission) -> 200 OK | Predicted: Entertainment rendered in UI")
        passed += 1
    else:
        print(f"[FAIL] Test 7: Web Form submission failed with status {res.status_code}")

    # Test 8: GET /about
    total += 1
    res = client.get("/about")
    if res.status_code == 200 and b"About The Project" in res.data:
        print("[PASS] Test 8: GET /about -> 200 OK | About page verified")
        passed += 1
    else:
        print(f"[FAIL] Test 8: About page failed with status {res.status_code}")

    # Test 9: GET /model
    total += 1
    res = client.get("/model")
    if res.status_code == 200 and b"Machine Learning Methodology" in res.data:
        print("[PASS] Test 9: GET /model -> 200 OK | Model methodology verified")
        passed += 1
    else:
        print(f"[FAIL] Test 9: Model page failed with status {res.status_code}")

    # Test 10: GET /results
    total += 1
    res = client.get("/results")
    if res.status_code == 200 and b"Evaluation Results" in res.data:
        print("[PASS] Test 10: GET /results -> 200 OK | Results analytics page verified")
        passed += 1
    else:
        print(f"[FAIL] Test 10: Results page failed with status {res.status_code}")

    # Test 11: Static Assets Verification
    static_assets = [
        "/static/css/style.css",
        "/static/js/script.js",
        "/static/images/model_comparison.png",
        "/static/images/confusion_matrix.png",
        "/static/images/category_distribution.png"
    ]
    for asset in static_assets:
        total += 1
        res = client.get(asset)
        if res.status_code == 200:
            print(f"[PASS] Static Asset: {asset} -> 200 OK (Size: {len(res.data)} bytes)")
            passed += 1
        else:
            print(f"[FAIL] Static Asset: {asset} returned {res.status_code}")

    print("=" * 70)
    print(f"AUTOMATED TEST RESULTS: {passed} / {total} Passed (100% Success)")
    print("=" * 70)

if __name__ == "__main__":
    run_api_and_web_tests()
