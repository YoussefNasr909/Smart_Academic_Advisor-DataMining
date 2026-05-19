"""Small sanity checks for the Flask app. Run after training: python run_checks.py"""
from __future__ import annotations

from app import app

sample = {
    "algorithm": "random_forest",
    "gpa": "3.45",
    "interest": "AI/ML",
    "math_grade": "91",
    "programming_grade": "88",
    "data_structures_grade": "85",
    "algorithms_grade": "90",
    "databases_grade": "74",
    "networks_grade": "65",
    "ai_grade": "96",
    "web_grade": "70",
}

with app.test_client() as client:
    # Check that the main pages load without server errors.
    # بالمصري: بنتأكد إن الصفحات الأساسية بتفتح من غير مشاكل.
    for path in ["/", "/dashboard"]:
        response = client.get(path)
        assert response.status_code == 200, (path, response.status_code)

    # Check that all expected track labels appear on the dashboard.
    # بالمصري: بنتأكد إن أسماء التراكات ظاهرة في الداشبورد.
    dashboard_html = client.get("/dashboard").get_data(as_text=True)
    for short_label in ["AI/ML", "Web", "Networks", "Data", "Software"]:
        assert short_label in dashboard_html, f"Missing dashboard label: {short_label}"

    # Check prediction output for every valid model and one invalid fallback case.
    # بالمصري: بنجرب كل الموديلات، وكمان اختيار غلط عشان نتأكد إنه بيرجع لـ Random Forest.
    for algorithm in ["decision_tree", "random_forest", "knn", "bad_algorithm"]:
        payload = dict(sample)
        payload["algorithm"] = algorithm
        response = client.post("/predict", data=payload)
        assert response.status_code == 200, (algorithm, response.status_code)
        html = response.get_data(as_text=True)
        assert "Same student across all algorithms" in html
        assert "Print / save PDF report" in html
        assert "nan" not in html.lower()

print("All route and UI checks passed.")
